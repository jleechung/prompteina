import torch

from proteinfoundation.nn.modules.adaptive_ln_scale import (
    AdaptiveLayerNorm,
    AdaptiveOutputScale,
    LinearCondGate,
)


class TextCrossAttentionADALN(torch.nn.Module):
    """
    Cross-attention with adaptive layer norm on input and adaptive scaling
    on output. Matches MultiHeadBiasedAttentionADALN_MM pattern:
    adaln -> attn -> scale, with NO internal residual connections
    (the trunk applies the single residual).

    Inputs:
      - x_seq:        [B, N_res, d_model]   (protein tokens)
      - text_tokens:  [B, N_txt, d_model]   (projected text tokens)
      - text_mask:    [B, N_txt] (bool)     (True = keep token)
      - cond:         [B, N_res, dim_cond]  (time conditioning)
      - mask:         [B, N_res] (bool)     (protein mask)

    Output:
      - x_seq': [B, N_res, d_model]
      - (optional) attn_weights: [B, N_res, N_txt] if return_attn_weights=True

    When side_compute_attn_weights=True, the main output uses FlashAttention
    (need_weights=False) while attention weights are computed separately via
    manual Q@K^T. This preserves the FlashAttention kernel for the forward
    pass while still providing attention maps for grounding loss.

    Output scaling modes (text_cross_attn_output_scale_mode):
      - "adaln_zero": sigmoid-bounded adaptive scale, bias=-2.0 (default, backward compat)
      - "adaln_zero_true": sigmoid-bounded, bias=-10.0 (true zero-init, sigmoid(-10)~0)
      - "linear_cond": unbounded Linear(cond) gate, zero-init W and b (SD3/Flux pattern)
      - "learnable": zero-init per-dimension learnable gate (unbounded, not conditioned)
      - "none": no output scaling
    """

    _VALID_SCALE_MODES = ("adaln_zero", "adaln_zero_true", "linear_cond", "learnable", "none")

    def __init__(self, d_model: int, nheads: int, dim_cond: int, **kwargs):
        super().__init__()
        self.output_scale_mode = kwargs.pop(
            "text_cross_attn_output_scale_mode", "adaln_zero",
        )
        if self.output_scale_mode not in self._VALID_SCALE_MODES:
            raise ValueError(
                f"Unknown output_scale_mode: {self.output_scale_mode!r}. "
                f"Valid: {self._VALID_SCALE_MODES}"
            )
        self.adaln = AdaptiveLayerNorm(dim=d_model, dim_cond=dim_cond)
        self.ln_kv = torch.nn.LayerNorm(d_model)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nheads, batch_first=True,
        )
        if self.output_scale_mode == "adaln_zero":
            self.scale_output = AdaptiveOutputScale(dim=d_model, dim_cond=dim_cond)
        elif self.output_scale_mode == "adaln_zero_true":
            self.scale_output = AdaptiveOutputScale(
                dim=d_model, dim_cond=dim_cond, adaln_zero_bias_init_value=-10.0,
            )
        elif self.output_scale_mode == "linear_cond":
            self.scale_output = LinearCondGate(dim=d_model, dim_cond=dim_cond)
        elif self.output_scale_mode == "learnable":
            self.output_gate = torch.nn.Parameter(torch.zeros(d_model))
        self.nheads = nheads
        self.side_compute_attn_weights = False

    def _compute_attn_weights_side(
        self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute attention weights via manual Q@K^T without changing the main
        attention kernel. Reuses self.attn's in_proj_weight so gradients from
        grounding loss flow through the projection layers."""
        E = self.attn.embed_dim
        w = self.attn.in_proj_weight
        b = self.attn.in_proj_bias
        q_proj = torch.nn.functional.linear(q, w[:E], b[:E] if b is not None else None)
        k_proj = torch.nn.functional.linear(kv, w[E:2*E], b[E:2*E] if b is not None else None)
        B, N_q, _ = q_proj.shape
        N_k = k_proj.shape[1]
        head_dim = E // self.nheads
        q_proj = q_proj.view(B, N_q, self.nheads, head_dim).transpose(1, 2)
        k_proj = k_proj.view(B, N_k, self.nheads, head_dim).transpose(1, 2)
        scale = head_dim ** -0.5
        attn_w = torch.matmul(q_proj * scale, k_proj.transpose(-2, -1))
        if key_padding_mask is not None:
            attn_w = attn_w.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn_w = torch.softmax(attn_w, dim=-1)
        return attn_w.mean(dim=1)  # [B, N_q, N_k] averaged over heads

    def forward(
        self,
        x_seq: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q = self.adaln(x_seq, cond, mask)
        kv = self.ln_kv(text_tokens)
        key_padding_mask = ~text_mask
        if return_attn_weights and self.side_compute_attn_weights:
            attn_out, _ = self.attn(
                q, kv, kv,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            attn_weights = self._compute_attn_weights_side(q, kv, key_padding_mask)
        else:
            attn_out, attn_weights = self.attn(
                q, kv, kv,
                key_padding_mask=key_padding_mask,
                need_weights=return_attn_weights,
                average_attn_weights=True,
            )
        if self.output_scale_mode in ("adaln_zero", "adaln_zero_true", "linear_cond"):
            x = self.scale_output(attn_out, cond, mask)
        elif self.output_scale_mode == "learnable":
            x = attn_out * self.output_gate
        else:
            x = attn_out
        x = x * mask[..., None]
        if return_attn_weights:
            return x, attn_weights  # attn_weights: [B, N_res, N_txt]
        return x


class TextReverseCrossAttention(torch.nn.Module):
    """
    Reverse cross-attention: text tokens attend to protein tokens.
    AdaLN on text queries, standard LN on protein KV, adaptive output scale.
    No internal residual (trunk applies residual).

    Inputs:
      - text_tokens:    [B, L, d_model]   (text token queries)
      - protein_tokens: [B, N, d_model]   (protein token keys/values)
      - protein_mask:   [B, N] (bool)     (True = keep)
      - cond:           [B, L, dim_cond]  (time conditioning)
      - text_mask:      [B, L] (bool)     (True = keep)

    Output:
      - [B, L, d_model]
    """

    def __init__(self, d_model: int, nheads: int, dim_cond: int, **kwargs):
        super().__init__()
        self.output_scale_mode = kwargs.get(
            "text_cross_attn_output_scale_mode", "adaln_zero",
        )
        self.adaln = AdaptiveLayerNorm(dim=d_model, dim_cond=dim_cond)
        self.ln_kv = torch.nn.LayerNorm(d_model)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nheads, batch_first=True,
        )
        if self.output_scale_mode == "adaln_zero":
            self.scale_output = AdaptiveOutputScale(dim=d_model, dim_cond=dim_cond)
        elif self.output_scale_mode == "adaln_zero_true":
            self.scale_output = AdaptiveOutputScale(
                dim=d_model, dim_cond=dim_cond, adaln_zero_bias_init_value=-10.0,
            )
        elif self.output_scale_mode == "linear_cond":
            self.scale_output = LinearCondGate(dim=d_model, dim_cond=dim_cond)
        elif self.output_scale_mode == "learnable":
            self.output_gate = torch.nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        text_tokens: torch.Tensor,
        protein_tokens: torch.Tensor,
        protein_mask: torch.Tensor,
        cond: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.adaln(text_tokens, cond, text_mask)
        kv = self.ln_kv(protein_tokens)
        key_padding_mask = ~protein_mask
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        if self.output_scale_mode in ("adaln_zero", "adaln_zero_true", "linear_cond"):
            x = self.scale_output(attn_out, cond, text_mask)
        elif self.output_scale_mode == "learnable":
            x = attn_out * self.output_gate
        else:
            x = attn_out
        return x * text_mask[..., None]


class TextSelfAttention(torch.nn.Module):
    """
    Self-attention among text tokens with AdaLN and adaptive output scale.
    No internal residual (trunk applies residual).

    Inputs:
      - text_tokens: [B, L, d_model]
      - text_mask:   [B, L] (bool, True = keep)
      - cond:        [B, L, dim_cond]

    Output:
      - [B, L, d_model]
    """

    def __init__(self, d_model: int, nheads: int, dim_cond: int, **kwargs):
        super().__init__()
        self.output_scale_mode = kwargs.get(
            "text_cross_attn_output_scale_mode", "adaln_zero",
        )
        self.adaln = AdaptiveLayerNorm(dim=d_model, dim_cond=dim_cond)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nheads, batch_first=True,
        )
        if self.output_scale_mode == "adaln_zero":
            self.scale_output = AdaptiveOutputScale(dim=d_model, dim_cond=dim_cond)
        elif self.output_scale_mode == "adaln_zero_true":
            self.scale_output = AdaptiveOutputScale(
                dim=d_model, dim_cond=dim_cond, adaln_zero_bias_init_value=-10.0,
            )
        elif self.output_scale_mode == "linear_cond":
            self.scale_output = LinearCondGate(dim=d_model, dim_cond=dim_cond)
        elif self.output_scale_mode == "learnable":
            self.output_gate = torch.nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x = self.adaln(text_tokens, cond, text_mask)
        key_padding_mask = ~text_mask
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        if self.output_scale_mode in ("adaln_zero", "adaln_zero_true", "linear_cond"):
            x = self.scale_output(attn_out, cond, text_mask)
        elif self.output_scale_mode == "learnable":
            x = attn_out * self.output_gate
        else:
            x = attn_out
        return x * text_mask[..., None]


class TextFFN(torch.nn.Module):
    """
    Feed-forward network for text tokens with AdaLN and adaptive output scale.
    No internal residual (trunk applies residual).

    Inputs:
      - text_tokens: [B, L, d_model]
      - text_mask:   [B, L] (bool, True = keep)
      - cond:        [B, L, dim_cond]

    Output:
      - [B, L, d_model]
    """

    def __init__(self, d_model: int, dim_cond: int, expansion_factor: int = 4, **kwargs):
        super().__init__()
        self.output_scale_mode = kwargs.get(
            "text_cross_attn_output_scale_mode", "adaln_zero",
        )
        self.adaln = AdaptiveLayerNorm(dim=d_model, dim_cond=dim_cond)
        self.linear1 = torch.nn.Linear(d_model, d_model * expansion_factor)
        self.act = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(d_model * expansion_factor, d_model)
        if self.output_scale_mode == "adaln_zero":
            self.scale_output = AdaptiveOutputScale(dim=d_model, dim_cond=dim_cond)
        elif self.output_scale_mode == "adaln_zero_true":
            self.scale_output = AdaptiveOutputScale(
                dim=d_model, dim_cond=dim_cond, adaln_zero_bias_init_value=-10.0,
            )
        elif self.output_scale_mode == "linear_cond":
            self.scale_output = LinearCondGate(dim=d_model, dim_cond=dim_cond)
        elif self.output_scale_mode == "learnable":
            self.output_gate = torch.nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x = self.adaln(text_tokens, cond, text_mask)
        x = self.linear2(self.act(self.linear1(x)))
        if self.output_scale_mode in ("adaln_zero", "adaln_zero_true", "linear_cond"):
            x = self.scale_output(x, cond, text_mask)
        elif self.output_scale_mode == "learnable":
            x = x * self.output_gate
        return x * text_mask[..., None]
