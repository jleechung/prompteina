"""Alignment prediction head for text-to-residue grounding supervision.

Auxiliary head (SteerViT-style): projects protein representations into text
token space and computes dot-product similarity, trained with BCE loss against
grounding masks.

Supports multiple source representations:
- block_hidden: Learned softmax-weighted mix of per-layer trunk hidden states
- block_hidden_last: Last transformer block output only (no layer mixing)
- cross_attn_delta: Learned softmax-weighted mix of per-layer cross-attn outputs
- output_bb_ca: C-alpha coordinates from ca_linear
- output_local_latents: Local latents from local_latents_linear
- output_both: Concat of bb_ca + local_latents
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AlignmentHead(nn.Module):
    """Dot-product alignment head predicting text-to-residue grounding.

    Args:
        d_model: Input dimension (depends on source_type).
        d_text: Text token dimension (projected text tokens).
        source_type: Which representation to supervise on.
        n_layers: Number of layers for multiscale mixing (block_hidden or
            cross_attn_delta sources).
    """

    MULTISCALE_SOURCES = {"block_hidden", "cross_attn_delta"}
    OUTPUT_SOURCES = {"output_bb_ca", "output_local_latents", "output_both"}
    ALL_SOURCES = MULTISCALE_SOURCES | {"block_hidden_last"} | OUTPUT_SOURCES

    def __init__(
        self,
        d_model: int,
        d_text: int,
        source_type: str = "block_hidden",
        n_layers: int | None = None,
        timestep_power: float = 0.0,
    ):
        super().__init__()
        if source_type not in self.ALL_SOURCES:
            raise ValueError(
                f"Unknown source_type={source_type!r}, expected one of {sorted(self.ALL_SOURCES)}"
            )
        self.source_type = source_type
        self.timestep_power = timestep_power

        self.protein_proj = nn.Linear(d_model, d_text, bias=False)
        nn.init.zeros_(self.protein_proj.weight)

        if source_type in self.MULTISCALE_SOURCES:
            if n_layers is None:
                raise ValueError(f"n_layers required for source_type={source_type!r}")
            self.layer_logits = nn.Parameter(torch.zeros(n_layers))

    @property
    def needs_block_hidden(self) -> bool:
        """Full per-layer collection (multiscale)."""
        return self.source_type == "block_hidden"

    @property
    def needs_block_hidden_last(self) -> bool:
        """Last-layer only (cheap, no per-layer stacking)."""
        return self.source_type == "block_hidden_last"

    @property
    def needs_cross_attn_deltas(self) -> bool:
        return self.source_type == "cross_attn_delta"

    @property
    def layer_weights(self) -> Tensor | None:
        if self.source_type not in self.MULTISCALE_SOURCES:
            return None
        return F.softmax(self.layer_logits, dim=0).detach()

    def _get_source(self, intermediates: dict[str, Tensor]) -> Tensor:
        """Extract and reduce the source representation.

        Args:
            intermediates: Dict with keys depending on source_type:
                - block_hidden: "block_hidden" [B, nlayers, N, d]
                - block_hidden_last: "block_hidden_last" [B, N, d]
                - cross_attn_delta: "cross_attn_deltas" [B, n_ca, N, d]
                - output_bb_ca: "bb_ca" [B, N, 3]
                - output_local_latents: "local_latents" [B, N, d_latent]
                - output_both: "bb_ca" [B, N, 3] and "local_latents" [B, N, d_latent]

        Returns:
            [B, N, d_model] tensor ready for projection.
        """
        if self.source_type == "block_hidden":
            all_hidden = intermediates["block_hidden"]  # [B, nlayers, N, d]
            weights = F.softmax(self.layer_logits, dim=0)
            return torch.einsum("l, blnd -> bnd", weights, all_hidden)

        if self.source_type == "block_hidden_last":
            return intermediates["block_hidden_last"]  # [B, N, d]

        if self.source_type == "cross_attn_delta":
            deltas = intermediates["cross_attn_deltas"]  # [B, n_ca, N, d]
            weights = F.softmax(self.layer_logits, dim=0)
            return torch.einsum("l, blnd -> bnd", weights, deltas)

        if self.source_type == "output_bb_ca":
            return intermediates["bb_ca"]  # [B, N, 3]

        if self.source_type == "output_local_latents":
            return intermediates["local_latents"]  # [B, N, d_latent]

        # output_both
        return torch.cat(
            [intermediates["bb_ca"], intermediates["local_latents"]], dim=-1
        )  # [B, N, 3 + d_latent]

    def forward(
        self,
        intermediates: dict[str, Tensor],
        text_tokens: Tensor,
        grounding_mask: Tensor,
        token_has_mask: Tensor,
        protein_mask: Tensor,
        residue_weight: Tensor | None = None,
        t: Tensor | None = None,
    ) -> Tensor:
        """Compute per-sample alignment BCE loss.

        Args:
            intermediates: Source representations (see _get_source).
            text_tokens: [B, L_txt, d_text] projected text tokens.
            grounding_mask: [B, L_txt, N_res] target (1.0 where grounded).
            token_has_mask: [B, L_txt] True for tokens with annotations.
            protein_mask: [B, N_res] True for valid residues.
            residue_weight: optional [B, N_res] per-residue loss weights.
            t: [B] diffusion timestep (0=noise, 1=clean). Used for
                timestep weighting when timestep_power > 0.

        Returns:
            [B] per-sample loss (zeros for samples with no valid positions).
        """
        B = text_tokens.shape[0]
        source = self._get_source(intermediates)  # [B, N, d_model]
        proj = self.protein_proj(source)  # [B, N, d_text]

        # Dot-product logits: each text token vs all residues
        logits = torch.einsum("bld, bnd -> bln", text_tokens, proj)  # [B, L_txt, N]

        # Truncate text dim if tokenization lengths differ
        min_txt = min(logits.shape[1], grounding_mask.shape[1], token_has_mask.shape[1])
        logits = logits[:, :min_txt, :]
        grounding_mask = grounding_mask[:, :min_txt, :]
        token_has_mask = token_has_mask[:, :min_txt]
        if residue_weight is not None:
            residue_weight = residue_weight[:, :logits.shape[2]]

        # Valid positions: annotated text tokens × valid residues
        valid = token_has_mask[..., None] & protein_mask[:, None, :]  # [B, L_txt, N]
        weight = None
        if residue_weight is not None:
            weight = residue_weight[:, None, :].expand_as(valid)

        # Per-sample loss with dynamic pos_weight to handle class imbalance.
        # Typical grounding mask density is ~3% positive, so without
        # reweighting the head converges to predicting all zeros.
        per_sample_loss = logits.new_zeros(B)
        for i in range(B):
            mask_i = valid[i]  # [L_txt, N]
            if not mask_i.any():
                continue
            targets_i = grounding_mask[i][mask_i]
            n_pos = targets_i.sum()
            n_total = targets_i.numel()
            if n_pos > 0:
                pw = (n_total - n_pos) / n_pos
            else:
                pw = targets_i.new_ones(())
            loss_i = F.binary_cross_entropy_with_logits(
                logits[i][mask_i], targets_i,
                pos_weight=pw, reduction="mean",
            )
            if weight is not None:
                losses_i = F.binary_cross_entropy_with_logits(
                    logits[i][mask_i], targets_i,
                    pos_weight=pw, reduction="none",
                )
                weights_i = weight[i][mask_i].to(dtype=losses_i.dtype)
                denom = weights_i.sum().clamp_min(1e-8)
                loss_i = (losses_i * weights_i).sum() / denom
            per_sample_loss[i] = loss_i

        # Timestep weighting: t^k suppresses loss at noisy timesteps
        if self.timestep_power > 0.0:
            if t is None:
                raise ValueError("timestep_power > 0 requires t to be provided")
            per_sample_loss = per_sample_loss * t.pow(self.timestep_power)

        return per_sample_loss
