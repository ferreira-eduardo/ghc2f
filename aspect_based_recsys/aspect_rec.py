"""
Recomendador simples baseado em aspectos/sentimentos - versão 2.

Mudanças em relação à v1 (motivadas pelas estatísticas do dataset):
- Entrada em dois modos: termos normalizados (top-N de `aspect_n`) ou grupos (`aspect_grp`).
- Métricas de diversidade SEMPRE sobre os grupos (independe do modo de entrada).
- Leave-one-out: 1 avaliação de teste por usuário com >= 2 avaliações.
- Perfis de itens usam TODAS as avaliações de treino (inclusive usuários com 1 avaliação).
- Perfis guardados como somas esparsas por avaliação; no treino, a avaliação-alvo é
  subtraída dos perfis do usuário e do item (evita vazamento).
- Pré-treino do encoder como autoencoder sobre avaliações individuais.
- Resultados estratificados pelo nº de avaliações de treino do usuário.

Colunas esperadas: userId, itemId, aspect_n, aspect_grp, prob_positive, prob_negative
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Cfg:
    input_mode: str = "terms"       # "terms" | "groups"
    term_col: str = "aspect_n"      # coluna dos termos no modo "terms" (ex.: "aspect" = cru)
    n_terms: int = 1000
    channels: str = "all"           # "all" = [tf|pos|neg] ; "tf" = só frequência
    shrink: float = 3.0             # 0 = sem encolhimento do sentimento
    latent_dim: int = 64
    hidden: int = 256
    dropout: float = 0.2
    shared_encoder: bool = False
    interaction: str = "dot"        # "dot" | "mlp"
    loss: str = "bpr"               # "bpr" | "bce" | "infonce"
    pretrain_epochs: int = 3        # autoencoder por avaliação (0 = desliga)
    lambda_rec: float = 0.0         # reconstrução durante o ajuste supervisionado
    n_neg: int = 4
    temperature: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    batch_size: int = 1024
    k: int = 10
    item_top_aspects: int = 5       # grupos que um item "exibe" nas métricas
    metric_exclude_groups: tuple = (-1,)   # grupos fora das métricas de diversidade (ruído)
    alpha_clarke: float = 0.5
    ab_alpha: float = 0.05
    ab_beta: float = 0.5
    pool_size: int = 200
    n_eval_users: Optional[int] = None   # None = todos os usuários de teste
    diversity_metrics: bool = True  # False = só hit/ndcg/novidade/cobertura (bem mais rápido)
    split_seed: int = 42            # semente do split e da amostra de avaliação
    seed: int = 42                  # semente do treino


# ======================================================================================
# Dados
# ======================================================================================
def featurize(S, F, shrink, channels="all"):
    """Somas [contagem | soma prob_pos | soma prob_neg] -> [tf | pos | neg] (3F dims) ou só tf."""
    S = S.toarray() if sp.issparse(S) else np.asarray(S)
    S = np.maximum(S, 0)                         # ruído numérico após subtração
    C, P, N = S[:, :F], S[:, F:2 * F], S[:, 2 * F:]
    safe = np.maximum(C, 1e-9)
    tf = C / np.maximum(C.sum(1, keepdims=True), 1e-9)
    if channels == "tf":
        return tf.astype(np.float32)
    conf = C / np.maximum(C + shrink, 1e-9)
    return np.hstack([tf, P / safe * conf, N / safe * conf]).astype(np.float32)


def _review_sums(df, n_rev, n_feat, col):
    d = df[df[col] >= 0]
    g = (d.groupby(["rid", col])
         .agg(cnt=("rid", "size"), pos=("prob_positive", "sum"), neg=("prob_negative", "sum"))
         .reset_index())
    mk = lambda v: sp.csr_matrix((g[v].to_numpy(np.float32), (g["rid"], g[col])), shape=(n_rev, n_feat))
    return sp.hstack([mk("cnt"), mk("pos"), mk("neg")]).tocsr()


class Data:
    def __init__(self, df: pd.DataFrame, cfg: Cfg):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.split_seed)
        df = df.dropna(subset=["aspect_n", "aspect_grp"])
        df = df[df["aspect_n"] != ""].copy()
        df["uid"] = df["userId"].astype("category").cat.codes.astype(np.int64)
        df["iid"] = df["itemId"].astype("category").cat.codes.astype(np.int64)
        self.n_users, self.n_items = df["uid"].max() + 1, df["iid"].max() + 1

        if cfg.input_mode == "terms":
            vocab = df[cfg.term_col].value_counts().index[: cfg.n_terms]
            df["fid"] = pd.Categorical(df[cfg.term_col], categories=vocab).codes   # -1 = fora
        else:
            vocab = sorted(df["aspect_grp"].unique())
            df["fid"] = pd.Categorical(df["aspect_grp"], categories=vocab).codes
        groups = sorted(set(df["aspect_grp"].unique()) - set(cfg.metric_exclude_groups))
        df["gid"] = pd.Categorical(df["aspect_grp"], categories=groups).codes   # excluídos -> -1
        self.F, self.G = len(vocab), len(groups)

        # uma linha por avaliação (u, i)
        rv = df.groupby(["uid", "iid"])["prob_positive"].mean().rename("r").reset_index()
        rv["rid"] = np.arange(len(rv))
        df = df.merge(rv[["uid", "iid", "rid"]], on=["uid", "iid"])

        # leave-one-out
        n = rv.groupby("uid")["rid"].transform("size").to_numpy()
        rv["_rk"] = pd.Series(rng.random(len(rv))).groupby(rv["uid"]).rank(method="first").to_numpy()
        rv["test"] = (n >= 2) & (rv["_rk"] == 1)
        tr, te = rv[~rv["test"]], rv[rv["test"]]

        # somas por avaliação e agregação esparsa para usuários/itens (só treino)
        self.R = _review_sums(df, len(rv), self.F, "fid")
        RG = _review_sums(df, len(rv), self.G, "gid")
        ones = np.ones(len(tr), np.float32)
        Mu = sp.csr_matrix((ones, (tr["uid"], tr["rid"])), shape=(self.n_users, len(rv)))
        Mi = sp.csr_matrix((ones, (tr["iid"], tr["rid"])), shape=(self.n_items, len(rv)))
        self.U, self.I = (Mu @ self.R).tocsr(), (Mi @ self.R).tocsr()
        IG = (Mi @ RG).tocsr()[:, : self.G].toarray()                         # contagens por grupo

        # itens recomendáveis: >= 1 avaliação de treino (igual nos dois modos de entrada,
        # para que termos e grupos ranqueiem o mesmo conjunto de candidatos)
        self.item_ok = np.bincount(tr["iid"], minlength=self.n_items) > 0
        self.item_ok_idx = np.flatnonzero(self.item_ok)

        # pares de treino supervisionado: usuários com >= 2 avaliações de treino
        ntr = tr.groupby("uid")["rid"].transform("size")
        self.train_rev = tr.loc[ntr >= 2, ["rid", "uid", "iid"]].to_numpy()
        self.all_train_rid = tr["rid"].to_numpy()

        # estruturas de avaliação
        self.n_train_u = tr["uid"].value_counts()
        self.train_items = {u: g.to_numpy() for u, g in tr.groupby("uid")["iid"]}
        cold = ~self.item_ok[te["iid"].to_numpy()]      # item de teste sem nenhuma avaliação de treino
        self.test = te.loc[~cold, ["uid", "iid", "r"]].to_numpy()
        self.R_ui = sp.csr_matrix((tr["r"].to_numpy(), (tr["uid"], tr["iid"])),
                                  shape=(self.n_users, self.n_items))
        self.pop = np.bincount(tr["iid"], minlength=self.n_items)
        self.IA = self._item_aspects(IG, cfg.item_top_aspects)
        self.item_grp_n = IG / (np.linalg.norm(IG, axis=1, keepdims=True) + 1e-9)

        print(f"usuários={self.n_users:,} itens={self.n_items:,} (recomendáveis={self.item_ok.sum():,}) "
              f"avaliações={len(rv):,} features={self.F} grupos={self.G}")
        print(f"pares de treino supervisionado={len(self.train_rev):,}  usuários de teste={len(self.test):,} "
              f"(descartados por item frio: {cold.sum():,})")
        cov = (np.diff(self.R.indptr) > 0).mean()
        print(f"avaliações com ao menos 1 feature no vocabulário: {cov:.3f}")

    @staticmethod
    def _item_aspects(IG, m):
        IA = np.zeros_like(IG, dtype=np.float64)
        top = np.argsort(-IG, axis=1)[:, :m]
        rows = np.arange(len(IG))[:, None]
        IA[rows, top] = IG[rows, top] > 0
        return IA

    # features ------------------------------------------------------------------------
    @property
    def in_dim(self):
        return self.F * (1 if self.cfg.channels == "tf" else 3)

    def with_cfg(self, cfg):
        """Mesma base de dados (mesmo split), outra forma de montar as features."""
        import copy
        d = copy.copy(self)
        d.cfg = cfg
        return d

    def fx(self, S):
        return featurize(S, self.F, self.cfg.shrink, self.cfg.channels)

    def user_x(self, u, mask_rid=None):
        return self.fx(self.U[u] - self.R[mask_rid] if mask_rid is not None else self.U[u])

    def item_x(self, i, mask_rid=None):
        return self.fx(self.I[i] - self.R[mask_rid] if mask_rid is not None else self.I[i])


# ======================================================================================
# Modelo (PyTorch importado só aqui)
# ======================================================================================
def build_model(in_dim, cfg: Cfg):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def mlp(i, h, o, p):
        return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, o))

    class AspectTwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            d, h = cfg.latent_dim, cfg.hidden
            self.enc_u = mlp(in_dim, h, d, cfg.dropout)
            self.enc_i = self.enc_u if cfg.shared_encoder else mlp(in_dim, h, d, cfg.dropout)
            use_dec = cfg.pretrain_epochs > 0 or cfg.lambda_rec > 0
            self.dec = mlp(d, h, in_dim, 0.0) if use_dec else None
            if cfg.interaction == "mlp":
                self.inter = nn.Sequential(nn.Linear(4 * d, h), nn.ReLU(), nn.Linear(h, 1))

        def encode_u(self, x):
            return F.normalize(self.enc_u(x), dim=-1)

        def encode_i(self, x):
            return F.normalize(self.enc_i(x), dim=-1)

        def score(self, zu, zi):
            if cfg.interaction == "dot":
                return (zu * zi).sum(-1) / cfg.temperature
            return self.inter(torch.cat([zu, zi, zu * zi, (zu - zi).abs()], -1)).squeeze(-1)

    return AspectTwoTower()


def pretrain_autoencoder(model, data: Data, cfg: Cfg, device="cpu"):
    """Reconstrói o vetor de aspectos de cada avaliação de treino (inclui as ~94% de
    avaliações de usuários com uma só avaliação). Depois copia o encoder para as duas torres."""
    import torch
    import torch.nn.functional as F

    if cfg.pretrain_epochs <= 0:
        return model
    rng = np.random.default_rng(cfg.seed)
    params = list(model.enc_u.parameters()) + list(model.dec.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    rids = data.all_train_rid
    model.to(device).train()
    for ep in range(cfg.pretrain_epochs):
        perm, tot = rng.permutation(rids), 0.0
        for s in range(0, len(perm), cfg.batch_size):
            x = torch.tensor(data.fx(data.R[perm[s:s + cfg.batch_size]]), device=device)
            loss = F.mse_loss(model.dec(model.encode_u(x)), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        print(f"pré-treino {ep + 1:02d}  mse={tot / len(perm):.6f}")
    if not cfg.shared_encoder:
        model.enc_i.load_state_dict(model.enc_u.state_dict())
    return model


def train(model, data: Data, cfg: Cfg, device="cpu"):
    import torch
    import torch.nn.functional as F

    if cfg.loss == "infonce":
        assert cfg.interaction == "dot", "InfoNCE in-batch requer interaction='dot'"
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    T = lambda a: torch.tensor(a, device=device)

    for ep in range(cfg.epochs):
        model.train()
        perm, tot = rng.permutation(len(data.train_rev)), 0.0
        for s in range(0, len(perm), cfg.batch_size):
            r, u, i = data.train_rev[perm[s:s + cfg.batch_size]].T
            xu, xi = T(data.user_x(u, r)), T(data.item_x(i, r))   # avaliação-alvo mascarada
            zu, zi = model.encode_u(xu), model.encode_i(xi)

            if cfg.loss == "infonce":
                logits = zu @ zi.T / cfg.temperature
                it = torch.as_tensor(i, device=device)
                dup = (it[None] == it[:, None]) & ~torch.eye(len(it), dtype=torch.bool, device=device)
                loss = F.cross_entropy(logits.masked_fill(dup, -1e9), torch.arange(len(it), device=device))
            else:
                j = rng.choice(data.item_ok_idx, (len(i), cfg.n_neg))
                zj = model.encode_i(T(data.item_x(j.ravel()))).view(len(i), cfg.n_neg, -1)
                s_pos = model.score(zu, zi)
                s_neg = model.score(zu.unsqueeze(1).expand_as(zj), zj)
                if cfg.loss == "bpr":
                    loss = -F.logsigmoid(s_pos.unsqueeze(1) - s_neg).mean()
                else:
                    loss = (F.binary_cross_entropy_with_logits(s_pos, torch.ones_like(s_pos))
                            + F.binary_cross_entropy_with_logits(s_neg, torch.zeros_like(s_neg)))

            if cfg.lambda_rec > 0:
                loss = loss + cfg.lambda_rec * (F.mse_loss(model.dec(zu), xu) + F.mse_loss(model.dec(zi), xi))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(u)
        print(f"época {ep + 1:02d}  loss={tot / len(perm):.4f}")
    return model


def model_scorer(model, data: Data, cfg: Cfg, device="cpu"):
    import torch

    model.eval()
    with torch.no_grad():
        Zi = torch.cat([model.encode_i(torch.tensor(data.item_x(np.arange(s, min(s + 4096, data.n_items))),
                                                    device=device))
                        for s in range(0, data.n_items, 4096)])

    def f(users):
        with torch.no_grad():
            zu = model.encode_u(torch.tensor(data.user_x(users), device=device))
            if cfg.interaction == "dot":
                return (zu @ Zi.T / cfg.temperature).cpu().numpy()
            return np.stack([model.score(z.expand_as(Zi), Zi).cpu().numpy() for z in zu])
    return f


# ======================================================================================
# Baselines
# ======================================================================================
def popularity_scorer(data: Data):
    pop = data.pop.astype(np.float32)
    return lambda users: np.tile(pop, (len(users), 1))


def raw_cosine_scorer(data: Data):
    chunks = []
    for s in range(0, data.n_items, 8192):
        X = data.item_x(np.arange(s, min(s + 8192, data.n_items)))
        chunks.append(sp.csr_matrix(X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)))
    Xi = sp.vstack(chunks).tocsr()

    def f(users):
        Xu = data.user_x(users)
        Xu /= np.linalg.norm(Xu, axis=1, keepdims=True) + 1e-9
        return np.asarray((Xi @ Xu.T).T)
    return f


# ======================================================================================
# Métricas
# ======================================================================================
def dcg(g):
    return float(np.sum(np.asarray(g) / np.log2(np.arange(2, len(g) + 2))))


def alpha_ndcg(rec, rel, IA, alpha, k):
    """Clarke et al. (2008). Obs.: com 1 item de teste por usuário, coincide com nDCG."""
    def gains(seq):
        seen, g = np.zeros(IA.shape[1]), []
        for i in seq:
            g.append(float((IA[i] * (1 - alpha) ** seen).sum()) if i in rel else 0.0)
            if i in rel:
                seen += IA[i]
        return g
    cand, seen, ideal = list(rel), np.zeros(IA.shape[1]), []
    for _ in range(min(k, len(cand))):
        gs = [(IA[i] * (1 - alpha) ** seen).sum() for i in cand]
        b = int(np.argmax(gs))
        ideal.append(gs[b])
        seen += IA[cand[b]]
        cand.pop(b)
    idcg = dcg(ideal)
    return dcg(gains(rec[:k])) / idcg if idcg > 0 else 0.0


def ab_ndcg(rec, rel, IA, gamma_u, pool, alpha, beta, k):
    """Parapar & Radlinski (2021), Eq. 5-8 e 14-16, com r_max = 1."""
    p_item = lambda i: IA[i] * (beta * rel[i] if i in rel else alpha)

    interest, g = gamma_u.copy(), []
    for i in rec[:k]:
        p = p_item(i)
        g.append(1 - np.prod(1 - p * interest))
        interest *= 1 - p

    cand = np.array(sorted(set(rel) | set(pool)))
    P = np.stack([p_item(i) for i in cand])
    interest, ideal, alive = gamma_u.copy(), [], np.ones(len(cand), bool)
    for _ in range(min(k, len(cand))):
        gs = np.where(alive, 1 - np.prod(1 - P * interest, axis=1), -1)
        b = int(np.argmax(gs))
        ideal.append(gs[b])
        interest *= 1 - P[b]
        alive[b] = False
    idcg = dcg(ideal)
    return min(1.0, dcg(g) / idcg) if idcg > 0 else 0.0


def stratum(n):
    return "1" if n <= 1 else ("2-3" if n <= 3 else "4+")


def evaluate(name, scorer, data: Data, cfg: Cfg, return_users=False):
    rng = np.random.default_rng(cfg.split_seed)
    test = data.test
    if cfg.n_eval_users and len(test) > cfg.n_eval_users:
        test = test[rng.choice(len(test), cfg.n_eval_users, replace=False)]
    k, IA, out, recs = cfg.k, data.IA, [], set()
    self_info = -np.log2((data.pop + 1) / data.n_users)
    blocked = ~data.item_ok

    for s in range(0, len(test), 256):
        chunk = test[s:s + 256]
        users = chunk[:, 0].astype(np.int64)
        S = scorer(users).astype(np.float64)
        S[:, blocked] = -np.inf
        R_rows = data.R_ui[users]
        gam = np.asarray(R_rows @ IA)                                  # Eq. 9
        gam /= gam.sum(1, keepdims=True) + 1e-12
        aff = gam @ IA.T
        aff[:, blocked] = -np.inf
        for row, (u, i_test, r_test) in enumerate(chunk):
            u, i_test = int(u), int(i_test)
            seen = data.train_items.get(u, [])
            S[row, seen] = -np.inf
            aff[row, seen] = -np.inf
            rec = np.argpartition(-S[row], k)[:k]
            rec = rec[np.argsort(-S[row, rec])]
            rel = {i_test: float(r_test)}
            hit = np.flatnonzero(rec == i_test)
            m = dict(
                uid=u,
                strato=stratum(int(data.n_train_u.get(u, 0))),
                hit=float(len(hit) > 0),
                ndcg=1 / np.log2(hit[0] + 2) if len(hit) else 0.0,
                novelty=float(self_info[rec].mean()),
            )
            if cfg.diversity_metrics:
                pool = np.argpartition(-aff[row], cfg.pool_size)[:cfg.pool_size]
                V = data.item_grp_n[rec]
                ua = gam[row] > 0
                m.update(
                    alpha_ndcg=alpha_ndcg(list(rec), rel, IA, cfg.alpha_clarke, k),
                    ab_ndcg=ab_ndcg(list(rec), rel, IA, gam[row], pool, cfg.ab_alpha, cfg.ab_beta, k),
                    s_recall=(IA[rec].max(0)[ua] > 0).mean() if ua.any() else 0.0,
                    ild=float((1 - (V @ V.T)[np.triu_indices(k, 1)]).mean()),
                )
            out.append(m)
            recs.update(rec.tolist())

    per_user = pd.DataFrame(out)
    res = per_user.drop(columns="uid")
    tab = pd.concat([res.drop(columns="strato").mean().rename("todos").to_frame().T,
                     res.groupby("strato").mean()])
    tab["n_users"] = [len(res)] + res.groupby("strato").size().tolist()
    tab["coverage"] = len(recs) / data.item_ok.sum()      # cobertura só faz sentido no total
    tab.loc[tab.index != "todos", "coverage"] = np.nan
    tab.index = pd.MultiIndex.from_product([[name], tab.index], names=["modelo", "estrato"])
    return (tab, per_user) if return_users else tab


# ======================================================================================
# Execução
# ======================================================================================
def run(df, cfg: Cfg = Cfg(), device="cpu", train_neural=True, data: Data = None):
    data = data or Data(df, cfg)
    tabs = [evaluate("Popularidade", popularity_scorer(data), data, cfg),
            evaluate(f"Cosseno bruto ({cfg.input_mode})", raw_cosine_scorer(data), data, cfg)]
    if train_neural:
        model = build_model(data.in_dim, cfg)
        model = pretrain_autoencoder(model, data, cfg, device)
        model = train(model, data, cfg, device)
        tabs.append(evaluate(f"TwoTower-{cfg.loss} ({cfg.input_mode})",
                             model_scorer(model, data, cfg, device), data, cfg))
    table = pd.concat(tabs)
    print(table.round(4).to_string())
    return table


if __name__ == "__main__":
    # df_aspects precisa das colunas aspect_n e aspect_grp
    # run(df_aspects, Cfg(input_mode="terms", n_terms=1000, loss="bpr"))
    # run(df_aspects, Cfg(input_mode="groups", loss="infonce", lambda_rec=0.1))
    pass