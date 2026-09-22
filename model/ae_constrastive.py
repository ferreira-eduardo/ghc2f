import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.text_att_profile import TextProfile
from model.cf_autoencoder import CFAutoEncoder


class AE_Contrastive(CFAutoEncoder):
    """
    Contrastive-only text injection. NO gated fusion, NO hybrid dependency
    (does not inherit from GatedHybridCFAutoEncoder). The scoring/ranking
    pathway -- encode(ratings_in) -> decode(z_cf) -- is PURE AE-BPR,
    byte-identical to CFAutoEncoder's forward()/evaluate(), and is never
    touched by text at inference time. Text only enters through an
    auxiliary InfoNCE loss during training, pulling the collaborative
    embedding (or a projection of it -- see cl_projector_dim) toward a
    text-derived representation z_text.

    This removes the structural competition of the gated design: there is
    no dimension-by-dimension trade-off between "collaborative" and "text"
    content, because there is no fused vector at all -- z_cf is what it
    always was in plain AE-BPR, and text can at most nudge it during
    training via gradient, never replace or blend into it during a
    forward pass.

    evaluate() is inherited UNCHANGED from CFAutoEncoder -- confirms this
    class's ranking behavior is exactly AE-BPR's, with any improvement (or
    harm) attributable only to how the representation was shaped during
    training, not to any text dependency at scoring time.

    --------------------------------------------------------------------
    DESIGN OPTIONS (each exposed as a constructor flag -- see __init__):
    --------------------------------------------------------------------

    1. stop_grad_cf (bool, default False)
       Does the contrastive loss's gradient flow back into z_cf (and
       therefore into the shared encoder), or only into the text branch?
         - False (default): text can actually reshape the collaborative
           representation toward text-consistency. This is the only
           setting under which text can influence ranking quality at all
           -- required if the goal is "does text improve the model."
         - True: z_cf is detached before the contrastive comparison. The
           text encoder trains against a frozen target but gives NOTHING
           back to z_cf. Useful as a sanity/null baseline: if hit_rate
           under stop_grad_cf=True differs measurably from plain AE-BPR,
           something in the harness/training loop is injecting an effect
           that has nothing to do with text content (e.g. a stray shared
           parameter, or measurement noise) -- worth running once as a
           control before trusting the False-mode results.

    2. cl_projector_dim (int or None, default = code_dim, i.e. ON by
       default)
       This is the "unfair weight" problem's shadow: even with the gate
       removed, if the RAW z_cf (the exact vector fed to decode() for
       scoring) is what gets pulled toward text via InfoNCE, that single
       vector is still being asked to serve two masters -- rank items
       well AND look like the text profile. That's the same competition
       as before, just moved from the forward pass into loss space
       instead of being visible as an explicit gate.
         - int (default): a small SimCLR-style projection head maps z_cf
           into a separate space before the contrastive comparison; only
           the PROJECTED vector is pulled toward text, so z_cf itself
           stays free to specialize purely for ranking. Gradient still
           flows back through the projector into z_cf, but indirectly and
           without forcing z_cf itself to literally resemble a text
           embedding.
         - None: compare raw z_cf directly (closer to a literal reading
           of "align the collaborative embedding with text"), at the risk
           of recreating the original competition problem in a subtler
           form. Worth an explicit ablation against the projector-on
           default, not assumed inferior.

    3. reg_loss / include_item_reg (bool, default False)
       The old GHC2F's L2 shrinkage on item embeddings (from
       encoder[0].weight rows) was tied to a now-removed embedding-scoring
       pathway and was never clearly validated as beneficial on its own.
       Off by default here; exposed as an option since generic L2 reg on
       item rows is a legitimate independent regularizer worth an ablation
       if the base contrastive design shows promise.

    4. cl_weight / temperature
       Same tunable knobs as before, now acting on a much simpler signal
       path -- worth re-sweeping rather than reusing the old gated
       design's values, since the loss landscape this operates in has
       changed (no gate_entropy term, no competing fusion gradient).

    5. Not implemented here, worth considering as a future extension:
       an ITEM-level contrastive term (pulling the ranking-relevant item
       representation -- e.g. encoder[0].weight rows -- toward item text
       profiles), symmetric to the user-level one implemented below. Kept
       out for now to isolate one change at a time; the current class only
       aligns the user-level z_cf with a user+item-derived z_text, same
       asymmetric shape as the design being replaced.
    """

    def __init__(
        self,
        layer_sizes,
        num_users,
        num_items,
        text_dim=15,
        text_latent_dim=64,
        text_gamma=0.5,
        cl_weight=0.1,
        temperature=0.1,
        stop_grad_cf=True,
        cl_projector_dim="auto",  # "auto" -> same width as code_dim; int -> custom; None -> no projector (raw z_cf)
        include_item_reg=False,
        reg_weight=1e-5,
        **kwargs,
    ):
        super().__init__(layer_sizes, **kwargs)
        self.name = 'AE_contrastive'

        self.user_profiler = TextProfile(num_users, text_dim, text_latent_dim)
        self.item_profiler = TextProfile(num_items, text_dim, text_latent_dim)

        self.user_proj = nn.Linear(text_dim, self.code_dim)
        self.item_proj = nn.Linear(text_dim, self.code_dim)

        self.text_gamma = text_gamma
        self.cl_weight = cl_weight
        self.temperature = temperature
        self.stop_grad_cf = stop_grad_cf
        self.include_item_reg = include_item_reg
        self.reg_weight = reg_weight

        self.register_buffer("item_corpus_ids", None)
        self.register_buffer("item_corpus_text", None)
        self.register_buffer("item_corpus_mask", None)

        if cl_projector_dim is None:
            self.cf_projector = nn.Identity()
        else:
            proj_dim = self.code_dim if cl_projector_dim == "auto" else cl_projector_dim
            self.cf_projector = nn.Sequential(
                nn.Linear(self.code_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, self.code_dim),
            )

        # Optimizer-construction-order fix (same issue/pattern as the
        # earlier GatedHybridCFAutoEncoder fix): CFAutoEncoder.__init__
        # (via super().__init__() above) builds self.optimizer BEFORE any
        # of this class's own modules exist, so they'd be silently
        # excluded from optimization otherwise. Rebuilding here, now that
        # everything above exists, fixes it.
        learn_rate = self.optimizer.param_groups[0]['lr']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

    def set_item_corpus(self, item_ids: torch.Tensor, item_text: torch.Tensor, item_mask: torch.Tensor):
        """Call once after construction, same as GatedHybridCFAutoEncoder's version."""
        self.item_corpus_ids = item_ids.to(self.device)
        self.item_corpus_text = item_text.to(self.device)
        self.item_corpus_mask = item_mask.to(self.device)

    def compute_z_text(self, batch):
        """
        Same text-side computation as the gated design (user profile +
        item-history-aggregated profile, blended by text_gamma) -- the
        only thing that's changed is what happens to it downstream: it
        never gets fused into the encoder, only compared against z_cf via
        contrastive_loss.
        """
        u_ids = batch["user_ids"].to(self.device)
        u_text = batch["user_text"].to(self.device)
        u_mask = batch["user_mask"].to(self.device)

        z_user_text = self.user_profiler(u_ids, u_text, u_mask)
        text_user = self.user_proj(z_user_text)

        item_profiles = self.item_profiler(
            self.item_corpus_ids, self.item_corpus_text, self.item_corpus_mask
        )
        ratings_in = batch["ratings_in"].to(self.device)
        item_global = torch.zeros(
            (ratings_in.size(1), item_profiles.size(1)), device=self.device
        )
        item_global = item_global.index_copy(0, self.item_corpus_ids, item_profiles)

        hist_mask = (ratings_in != 0).float()
        counts = hist_mask.sum(dim=1, keepdim=True)
        text_item_global = (hist_mask @ item_global) / counts.clamp_min(1.0)
        text_item = self.item_proj(text_item_global)

        return self.text_gamma * text_user + (1 - self.text_gamma) * text_item

    def contrastive_loss(self, z_cf_proj, z_text):
        z_cf_proj = F.normalize(z_cf_proj, dim=-1)
        z_text = F.normalize(z_text, dim=-1)
        logits = torch.matmul(z_cf_proj, z_text.t()) / self.temperature
        labels = torch.arange(z_cf_proj.size(0), device=self.device)
        return F.cross_entropy(logits, labels)

    def calculate_loss(self, batch):
        ratings_in = batch["ratings_in"].to(self.device)
        out = self(batch)

        pos_scores, neg_scores = self._pos_neg_scores(out.recon, batch)
        loss_bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        z_text = self.compute_z_text(batch)
        z_cf = out.code.detach() if self.stop_grad_cf else out.code
        z_cf_proj = self.cf_projector(z_cf)
        loss_cl = self.contrastive_loss(z_cf_proj, z_text)

        loss_components = {"bpr": loss_bpr.item(), "cl": loss_cl.item()}
        total_loss = loss_bpr + self.cl_weight * loss_cl

        if self.include_item_reg:
            item_embeddings = self.encoder[0].weight.t()  # (num_items, hidden)
            pos_item_ids = batch["pos_item_id"].to(self.device)
            neg_item_ids = batch["neg_item_id"].to(self.device)
            reg_loss = (torch.norm(item_embeddings[pos_item_ids]) ** 2 +
                        torch.norm(item_embeddings[neg_item_ids]) ** 2)
            total_loss = total_loss + self.reg_weight * reg_loss
            loss_components["reg"] = reg_loss.item()

        self.last_loss_components = loss_components

        return total_loss, ratings_in.size(0)
