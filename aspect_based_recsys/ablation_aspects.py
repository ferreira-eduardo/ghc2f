"""
Ablação do processamento de aspectos.

Pergunta: o processamento dos aspectos melhora a qualidade dos perfis?
Método: fixa o modelo (two-tower BPR) e o split, varia UMA escolha de processamento por vez
e compara cada variante com a referência, com várias sementes e IC por bootstrap pareado.

Variantes (cada uma difere da referência em um único fator):
- referência: termos normalizados (aspect_n), canais tf + sentimento, shrink = 3
- termos crus: coluna original `aspect` em vez de `aspect_n`
- grupos: aspect_grp em vez de termos
- só frequência: sem os canais de sentimento
- sem encolhimento: shrink = 0

Para cada variante roda também o cosseno bruto (sem treino), que mede a qualidade do perfil
sem nenhum aprendizado. Por fim, compara o two-tower de referência com popularidade e com
um híbrido sem treino "cosseno + λ·log(pop)", para separar o ganho da projeção aprendida
do ganho que viria só de incorporar popularidade.

Uso:
    from ablation_aspects import run_ablation
    resumo, vs_baselines, por_usuario = run_ablation(df_aspects)
"""
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from aspect_rec import (Cfg, Data, build_model, evaluate, model_scorer, popularity_scorer,
                              pretrain_autoencoder, raw_cosine_scorer, train)

# --------------------------------------------------------------------------------------
# Configuração do experimento
# --------------------------------------------------------------------------------------
DEVICE = "cuda"
SEEDS = [42, 43, 44]                 # sementes do TREINO (o split é sempre o mesmo)
RUN_NEURAL = True
HYBRID_LAMBDAS = [0.1, 0.3, 1.0, 3.0]
N_BOOT = 1000

BASE = Cfg(input_mode="terms", term_col="aspect_n", n_terms=1000, channels="all", shrink=3.0,
           loss="bpr", epochs=30, lambda_rec=0.0, pretrain_epochs=3,
           k=100, diversity_metrics=False, split_seed=42)

REF = "referência"
VARIANTS = {
    REF: {},
    "termos crus": dict(term_col="aspect"),
    "grupos": dict(input_mode="groups"),
    "só frequência": dict(channels="tf"),
    "sem encolhimento": dict(shrink=0.0),
}


# --------------------------------------------------------------------------------------
# Auxiliares
# --------------------------------------------------------------------------------------
def boot_ci(diff, n_boot=N_BOOT, seed=0):
    """IC 95% por bootstrap sobre usuários de uma diferença pareada (vetor por usuário)."""
    diff = np.asarray(diff, float)
    if len(diff) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.fromiter((diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)), float, n_boot)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return diff.mean(), lo, hi


def hybrid_scorer(data, lam):
    cos = raw_cosine_scorer(data)
    logpop = np.log1p(data.pop) / np.log1p(data.pop.max())
    return lambda users: cos(users) + lam * logpop[None, :]


def train_and_eval(name, data, cfg, device):
    import torch
    torch.manual_seed(cfg.seed)                      # inicialização dos pesos também semeada
    model = build_model(data.in_dim, cfg)
    model = pretrain_autoencoder(model, data, cfg, device)
    model = train(model, data, cfg, device)
    return evaluate(name, model_scorer(model, data, cfg, device), data, cfg, return_users=True)


def strata_iter(df):
    yield "todos", df
    for s in ["1", "2-3", "4+"]:
        yield s, df[df["strato"] == s]


# --------------------------------------------------------------------------------------
# Experimento
# --------------------------------------------------------------------------------------
def run_ablation(df, base=BASE, variants=VARIANTS, seeds=SEEDS, device=DEVICE, run_neural=RUN_NEURAL):
    t0 = time.time()
    data_cache, rows = {}, []          # rows: resultados por usuário de cada execução

    def get_data(cfg):
        key = (cfg.input_mode, cfg.term_col, cfg.n_terms)
        if key not in data_cache:
            print(f"\n=== montando dados: {key} ===")
            data_cache[key] = Data(df, cfg)
        return data_cache[key].with_cfg(cfg)   # canais/shrink mudam sem refazer o split

    for vname, kw in variants.items():
        cfg_v = replace(base, **kw)
        d = get_data(cfg_v)
        print(f"\n### variante: {vname}  (dim. de entrada = {d.in_dim})  [{time.time() - t0:.0f}s]")

        _, pu = evaluate("cos", raw_cosine_scorer(d), d, cfg_v, return_users=True)
        rows.append(pu.assign(variante=vname, modelo="Cosseno bruto", seed=-1))

        if run_neural:
            for sd in seeds:
                print(f"--- two-tower, semente {sd}")
                _, pu = train_and_eval("tt", d, replace(cfg_v, seed=sd), device)
                rows.append(pu.assign(variante=vname, modelo="Two-tower BPR", seed=sd))

    # baselines na base de referência
    d_ref = get_data(replace(base, **variants[REF]))
    _, pu = evaluate("pop", popularity_scorer(d_ref), d_ref, base, return_users=True)
    rows.append(pu.assign(variante="—", modelo="Popularidade", seed=-1))
    for lam in HYBRID_LAMBDAS:
        _, pu = evaluate("hyb", hybrid_scorer(d_ref, lam), d_ref, base, return_users=True)
        rows.append(pu.assign(variante="—", modelo=f"Cosseno+pop λ={lam}", seed=-1))

    por_usuario = pd.concat(rows, ignore_index=True)
    resumo = summarize_variants(por_usuario)
    vs_base = compare_with_baselines(por_usuario)

    pd.set_option("display.width", 200)
    print("\n\n===== ABLAÇÃO (Hit@k em %, diferença vs referência do mesmo modelo, em p.p.) =====")
    print(resumo.round(3).to_string())
    print("\n===== TWO-TOWER DE REFERÊNCIA vs BASELINES (Hit@k em %) =====")
    print(vs_base.round(3).to_string())
    print(f"\ntempo total: {(time.time() - t0) / 60:.1f} min")

    resumo.to_csv("ablacao_resumo.csv")
    vs_base.to_csv("ablacao_vs_baselines.csv")
    por_usuario.to_csv("ablacao_por_usuario.csv.gz", index=False)
    return resumo, vs_base, por_usuario


def summarize_variants(pu_all):
    """Para cada modelo e variante: hit médio, desvio entre sementes e diferença pareada vs
    a referência do mesmo modelo (IC 95% por bootstrap)."""
    out = []
    for modelo in ["Cosseno bruto", "Two-tower BPR"]:
        sub = pu_all[pu_all["modelo"] == modelo]
        if sub.empty:
            continue
        # média entre sementes por usuário
        per_user = sub.groupby(["variante", "uid", "strato"], as_index=False)["hit"].mean()
        ref = per_user[per_user["variante"] == REF][["uid", "hit"]].rename(columns={"hit": "hit_ref"})
        for vname in pu_all.loc[pu_all["modelo"] == modelo, "variante"].unique():
            v = per_user[per_user["variante"] == vname].merge(ref, on="uid")
            seeds_v = sub[sub["variante"] == vname]
            for s, part in strata_iter(v):
                sd_part = seeds_v if s == "todos" else seeds_v[seeds_v["strato"] == s]
                by_seed = sd_part.groupby("seed")["hit"].mean() * 100
                diff, lo, hi = boot_ci(part["hit"] - part["hit_ref"]) if vname != REF else (0, 0, 0)
                out.append(dict(modelo=modelo, variante=vname, estrato=s, n_users=len(part),
                                hit_pct=part["hit"].mean() * 100,
                                dp_sementes=by_seed.std() if len(by_seed) > 1 else np.nan,
                                dif_pp=diff * 100, ic95_inf=lo * 100, ic95_sup=hi * 100,
                                signif="*" if vname != REF and (lo > 0 or hi < 0) else ""))
    return pd.DataFrame(out).set_index(["modelo", "variante", "estrato"])


def compare_with_baselines(pu_all):
    """Two-tower de referência (média das sementes) contra popularidade e o melhor híbrido."""
    tt = pu_all[(pu_all["modelo"] == "Two-tower BPR") & (pu_all["variante"] == REF)]
    if tt.empty:
        return pd.DataFrame()
    tt = tt.groupby(["uid", "strato"], as_index=False)["hit"].mean().rename(columns={"hit": "hit_tt"})
    base = pu_all[pu_all["variante"] == "—"]
    hyb = base[base["modelo"].str.startswith("Cosseno+pop")]
    best_hyb = hyb.groupby("modelo")["hit"].mean().idxmax()   # escolhido no teste: otimista p/ o baseline
    out = []
    for bname in ["Popularidade", best_hyb]:
        b = base[base["modelo"] == bname][["uid", "hit"]].merge(tt, on="uid")
        for s, part in strata_iter(b):
            diff, lo, hi = boot_ci(part["hit_tt"] - part["hit"])
            out.append(dict(baseline=bname, estrato=s, n_users=len(part),
                            hit_baseline_pct=part["hit"].mean() * 100, hit_tt_pct=part["hit_tt"].mean() * 100,
                            dif_tt_menos_base_pp=diff * 100, ic95_inf=lo * 100, ic95_sup=hi * 100,
                            signif="*" if (lo > 0 or hi < 0) else ""))
    return pd.DataFrame(out).set_index(["baseline", "estrato"])


if __name__ == "__main__":
    df_aspects = pd.read_parquet("aspects.parquet")   # precisa de aspect, aspect_n, aspect_grp, prob_*
    res, vb, pu = run_ablation(df_aspects)
