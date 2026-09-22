import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as weight_init
from dataclasses import dataclass
from typing import List, Optional

from model.cf_autoencoder import CFAutoEncoder, AEOutput, activation


@dataclass
class DisentangledAEOutput(AEOutput):
    """
    `recon`/`code` (inherited) are ALWAYS Pathway A (collaborative) only --
    this is what evaluate() and any downstream ranking code consumes, byte
    identical to plain CFAutoEncoder.

    z_text / recon_text are exposed purely so calculate_loss() (and any
    diagnostics) can get at Pathway B without a second forward pass.
    """
    z_text: torch.Tensor = None
    recon_text: torch.Tensor = None


class SemanticDisentangledAE(CFAutoEncoder):
    """
    Disentangled Dual-Pathway AE.

    Pathway A (collaborative): exactly CFAutoEncoder.encode/decode, on
    ratings_in only. Never sees text/aspect data, at train OR eval time.

    Pathway B (semantic): an independent nn.ModuleList encoder/decoder over
    a dense num_aspects-dim vector of per-aspect POLARITY scores in
    [-1, +1] (pos - neg; 0 == neutral -- see _to_polarity, which also
    accepts the raw 3*num_aspects [pos,neg,neu] format and collapses it).
    Its bottleneck targets align_dim (its own,
    independently-sized space), and Pathway A's code is mapped into that
    same space by a small align_projector before comparison -- so the two
    pathways are never forced to share code_dim itself. There is no
    weight sharing and no fusion between the two encoders/decoders.

    The two pathways only ever interact through the loss function
    (L_Alignment), never through the forward computation graph that
    produces `recon`. That's what makes this "disentangled" rather than a
    gated/fused hybrid like GatedHybridCFAutoEncoder/GHC2F: any change in
    ranking quality has to be explained by the shared collaborative
    encoder's weights being *regularized* toward a semantically-consistent
    geometry during training, not by text leaking into inference.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        num_aspects: int,
        sem_hidden_dims: Optional[List[int]] = None,
        alpha: float = 1.0,
        beta: float = 0.1,
        sem_nl_type: str = "relu",
        align_dim: Optional[int] = None,
        temperature: float = 0.1,
        detach_code_for_align: bool = False,
        **kwargs,
    ):
        # Pathway A is built (and self.optimizer is created, prematurely --
        # fixed at the bottom of this __init__) entirely by the base class.
        super().__init__(layer_sizes, **kwargs)
        self.name = "SemanticDisentangledAE"

        self.num_aspects = num_aspects
        # Collapsed to ONE scalar per aspect: polarity = pos - neg, in
        # [-1, +1], with 0 == neutral. Mathematically equivalent to the
        # ordinal expected value (pos + 0.5*neu) rescaled -- see
        # _to_polarity() -- but centered at zero, which matches tanh's
        # native range and is the standard sentiment-score convention.
        self.sem_input_dim = num_aspects
        self.alpha = alpha  # L_Recon weight
        # NOTE: squared error on a [-1,1] target has ~4x the range of a
        # [0,1] target, so L_Recon's natural magnitude is larger than it
        # was under the old 3*num_aspects/[0,1] formulation. Re-tune alpha
        # against loss_bpr/loss_align rather than reusing the old default.
        self.beta = beta    # L_Alignment (InfoNCE) weight
        self._sem_nl_type = sem_nl_type
        self.temperature = temperature
        # If True, out.code is detached before the alignment projector, so
        # L_Alignment trains the semantic branch + projector only and gives
        # NOTHING back to the shared CF encoder -- a null-baseline control,
        # same role as AE_Contrastive's stop_grad_cf.
        self.detach_code_for_align = detach_code_for_align

        # code_dim stays whatever's optimal for ranking; align_dim is a
        # SEPARATE width for the alignment comparison space, sized
        # independently so the two objectives don't have to fight over one
        # shared bottleneck (same "unfair weight" fix as AE_Contrastive's
        # cl_projector_dim). Defaults to code_dim if not given.
        self.align_dim = align_dim if align_dim is not None else self.code_dim

        # Projects Pathway A's code into the alignment space. Gradients
        # from L_Alignment flow back through this into the shared encoder
        # (unless detach_code_for_align=True) -- this is the ONLY path by
        # which text ever influences the collaborative weights; it never
        # touches the forward computation that produces `recon`.
        self.align_projector = nn.Sequential(
            nn.Linear(self.code_dim, self.align_dim),
            nn.ReLU(),
            nn.Linear(self.align_dim, self.align_dim),
        )

        # Semantic pathway's own bottleneck now targets align_dim directly
        # -- there's no reason to force it through code_dim first.
        if sem_hidden_dims is None:
            sem_hidden_dims = []
        sem_sizes = [self.sem_input_dim] + list(sem_hidden_dims) + [self.align_dim]

        ######## Pathway B: semantic encoder ########
        self.sem_encoder = nn.ModuleList()
        for in_dim, out_dim in zip(sem_sizes[:-1], sem_sizes[1:]):
            lin = nn.Linear(in_dim, out_dim)
            weight_init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
            self.sem_encoder.append(lin)

        ######## Pathway B: semantic decoder (mirror, untied) ########
        rev_sem_sizes = list(reversed(sem_sizes))
        self.sem_decoder = nn.ModuleList()
        for in_dim, out_dim in zip(rev_sem_sizes[:-1], rev_sem_sizes[1:]):
            lin = nn.Linear(in_dim, out_dim)
            weight_init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
            self.sem_decoder.append(lin)

        # --- Optimizer correction ---
        # CFAutoEncoder.__init__ (via super().__init__ above) built
        # self.optimizer over self.parameters() BEFORE sem_encoder/
        # sem_decoder existed, so those params would silently receive no
        # gradient updates otherwise. Rebuild now that everything exists.
        learn_rate = self.optimizer.param_groups[0]["lr"]
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)

    # ------------------------------------------------------------------
    # InfoNCE alignment: "given this user's (projected) collaborative
    # code, pick out their own semantic embedding from everyone else's in
    # the batch." See calculate_loss for the walkthrough.
    # ------------------------------------------------------------------
    def contrastive_alignment_loss(self, code_proj: torch.Tensor, z_text: torch.Tensor) -> torch.Tensor:
        code_n = F.normalize(code_proj, dim=-1)
        text_n = F.normalize(z_text, dim=-1)
        sim = torch.matmul(code_n, text_n.t()) / self.temperature  # (B, B)
        labels = torch.arange(code_n.size(0), device=self.device)
        # Symmetric InfoNCE: code->text AND text->code, so both directions
        # of the mapping are pulled into agreement, not just one.
        loss_c2t = F.cross_entropy(sim, labels)
        loss_t2c = F.cross_entropy(sim.t(), labels)
        return 0.5 * (loss_c2t + loss_t2c)

    # ------------------------------------------------------------------
    # Pathway B only
    # ------------------------------------------------------------------
    def encode_semantic(self, aspect_x: torch.Tensor) -> torch.Tensor:
        z = aspect_x
        last = len(self.sem_encoder) - 1
        for i, lin in enumerate(self.sem_encoder):
            z = lin(z)
            if i != last:
                z = activation(z, self._sem_nl_type)
        return z

    def decode_semantic(self, z: torch.Tensor) -> torch.Tensor:
        out = z
        last = len(self.sem_decoder) - 1
        for i, lin in enumerate(self.sem_decoder):
            out = lin(out)
            if i != last:
                out = activation(out, self._sem_nl_type)
        # Target is polarity in [-1, +1] -- tanh is the native match (it's
        # also zero-centered, unlike sigmoid, which matters for gradient
        # conditioning once the target itself is zero-centered).
        out = torch.tanh(out)
        return out

    @staticmethod
    def _dense_aspects(t: torch.Tensor) -> torch.Tensor:
        return t.to_dense() if t.is_sparse else t

    def _to_polarity(self, aspect_probs: torch.Tensor) -> torch.Tensor:
        """
        Collapse a (..., 3*num_aspects) [pos,neg,neu]-per-aspect tensor
        (aspect-major layout: a1_pos,a1_neg,a1_neu, a2_pos,...) into a
        (..., num_aspects) polarity tensor via pos - neg, in [-1, +1].
        0 == neutral, +1 == fully positive, -1 == fully negative.

        NOTE: conflated case -- an aspect with high pos AND high neg
        simultaneously (mixed/conflicted sentiment) collapses to the same
        ~0 value as a genuinely neutral aspect. That's an unavoidable
        consequence of going from 3 numbers to 1, not specific to this
        formula; check how common conflicted aspects are in the data
        before assuming this doesn't matter.

        If aspect_probs is already (..., num_aspects) -- e.g. the upstream
        pipeline was updated to emit polarity directly -- it's returned
        unchanged.
        """
        last_dim = aspect_probs.size(-1)
        if last_dim == self.num_aspects:
            return aspect_probs
        if last_dim == 3 * self.num_aspects:
            shape = aspect_probs.shape[:-1] + (self.num_aspects, 3)
            triplets = aspect_probs.view(*shape)
            pos, neg = triplets[..., 0], triplets[..., 1]
            return pos - neg
        raise ValueError(
            f"aspect_probs last dim {last_dim} matches neither num_aspects "
            f"({self.num_aspects}) nor 3*num_aspects ({3 * self.num_aspects})"
        )

    def _collapse_mask(self, aspect_mask: torch.Tensor) -> torch.Tensor:
        """
        Collapse a (..., 3*num_aspects) mask to (..., num_aspects) by
        treating an aspect as 'mentioned' if any of its 3 sentiment slots
        was mentioned. No-op if already (..., num_aspects).
        """
        last_dim = aspect_mask.size(-1)
        if last_dim == self.num_aspects:
            return aspect_mask
        if last_dim == 3 * self.num_aspects:
            shape = aspect_mask.shape[:-1] + (self.num_aspects, 3)
            return aspect_mask.view(*shape).amax(dim=-1)
        raise ValueError(
            f"aspect_mask last dim {last_dim} matches neither num_aspects "
            f"({self.num_aspects}) nor 3*num_aspects ({3 * self.num_aspects})"
        )

    # ------------------------------------------------------------------
    # Forward: Pathway A is untouched (inherited); Pathway B is computed
    # alongside it for convenience but never feeds into recon/code.
    # ------------------------------------------------------------------
    def forward(self, batch, item_indices=None) -> DisentangledAEOutput:
        # Pathway A -- identical to CFAutoEncoder.forward, byte for byte.
        base_out = super().forward(batch, item_indices=item_indices)

        z_text = None
        recon_text = None
        if "aspect_probs" in batch:
            raw = self._dense_aspects(batch["aspect_probs"]).to(self.device).float()
            aspect_x = self._to_polarity(raw)  # (B, num_aspects), in [-1, 1]
            z_text = self.encode_semantic(aspect_x)
            recon_text = self.decode_semantic(z_text)

        return DisentangledAEOutput(
            recon=base_out.recon,
            code=base_out.code,
            z_text=z_text,
            recon_text=recon_text,
        )

    # ------------------------------------------------------------------
    # Multi-task loss: L_total = L_BPR + alpha * L_Recon + beta * L_Align
    # ------------------------------------------------------------------
    def calculate_loss(self, batch):
        """
        L_total = L_BPR + alpha * L_Recon + beta * L_Alignment

        L_Alignment is InfoNCE (see contrastive_alignment_loss): for each
        user i in the batch, code_proj[i] must be more similar to
        z_text[i] (its own semantic embedding) than to z_text[j] for any
        other user j in the batch. This gives the alignment term real
        negatives -- unlike a plain MSE-of-normalized-vectors bridge,
        which has no mechanism to stop every user's code/text pair from
        collapsing toward the same average direction.
        """
        out = self(batch)

        # --- L_BPR : exactly the base class's mechanism, on Pathway A ---
        pos_scores, neg_scores = self._pos_neg_scores(out.recon, batch)
        loss_bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # --- L_Recon : masked MSE on Pathway B, target = polarity in [-1,1] ---
        raw = self._dense_aspects(batch["aspect_probs"]).to(self.device).float()
        aspect_x = self._to_polarity(raw)  # (B, num_aspects)
        raw_mask = self._dense_aspects(batch["aspect_mask"]).to(self.device).float()
        aspect_mask = self._collapse_mask(raw_mask)  # (B, num_aspects)

        sq_err = (out.recon_text - aspect_x) ** 2
        masked_sq_err = sq_err * aspect_mask
        denom = aspect_mask.sum().clamp_min(1.0)
        loss_recon = masked_sq_err.sum() / denom

        # --- L_Alignment : InfoNCE bridge, via the alignment projector ---
        code_for_align = out.code.detach() if self.detach_code_for_align else out.code
        code_proj = self.align_projector(code_for_align)
        loss_align = self.contrastive_alignment_loss(code_proj, out.z_text)

        total_loss = loss_bpr + self.alpha * loss_recon + self.beta * loss_align

        self.last_loss_components = {
            "bpr": loss_bpr.item(),
            "recon": loss_recon.item(),
            "align": loss_align.item(),
        }

        return total_loss, batch["ratings_in"].size(0)


