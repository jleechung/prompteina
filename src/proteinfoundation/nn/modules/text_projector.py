import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    """
    Simple MLP projector for text embeddings:
      d_in -> d_hidden -> d_out
    with GELU and LayerNorm in the middle.

    Preserves self.net = nn.Sequential(Linear, GELU, LN, Linear)
    for checkpoint compatibility with the original TextProjector.
    """

    def __init__(self, d_in: int, d_out: int, d_hidden: int | None = None):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_out

        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMLPProjector(nn.Module):
    """
    Deep MLP projector with residual blocks.

    Structure: input projection + (num_layers - 2) residual blocks + output projection.
    Each residual block: Linear -> GELU -> LayerNorm + skip connection.
    """

    def __init__(
        self, d_in: int, d_out: int, d_hidden: int | None = None, num_layers: int = 3
    ):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_out
        if num_layers < 2:
            raise ValueError("DeepMLPProjector requires num_layers >= 2")

        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
        )

        self.residual_blocks = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.residual_blocks.append(
                nn.Sequential(
                    nn.Linear(d_hidden, d_hidden),
                    nn.GELU(),
                    nn.LayerNorm(d_hidden),
                )
            )

        self.output_proj = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.residual_blocks:
            x = x + block(x)
        return self.output_proj(x)


class NormalizedMLPProjector(nn.Module):
    """
    MLP projector with L2-normalized output.

    Internal self.net structure is identical to MLPProjector, so existing
    checkpoint weights can load directly for fine-tuning experiments.
    """

    def __init__(self, d_in: int, d_out: int, d_hidden: int | None = None):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_out

        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class AttentionPoolProjector(nn.Module):
    """
    Learnable query tokens cross-attend to input, then mean-pool over queries.

    Suitable for the pooled path only — provides order-invariant aggregation.
    Handles both [B, d_in] and [B, L, d_in] inputs.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int | None = None,
        num_queries: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_out

        self.input_proj = nn.Linear(d_in, d_hidden)
        self.queries = nn.Parameter(torch.randn(num_queries, d_hidden) * 0.02)

        self.ln_q = nn.LayerNorm(d_hidden)
        self.ln_kv = nn.LayerNorm(d_hidden)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=num_heads,
            batch_first=True,
        )

        self.output_proj = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B, 1, d_in]

        x = self.input_proj(x)  # [B, L, d_hidden]
        B = x.shape[0]

        q = self.ln_q(self.queries.unsqueeze(0).expand(B, -1, -1))
        kv = self.ln_kv(x)

        attn_out, _ = self.attn(q, kv, kv, need_weights=False)  # [B, num_queries, d_hidden]
        pooled = attn_out.mean(dim=1)  # [B, d_hidden]
        return self.output_proj(pooled)


class SwiGLUMLPProjector(nn.Module):
    """
    MLP projector using SwiGLU activation, matching the transformer's Transition.
    Structure: Linear(d_in, d_hidden*2) -> SwiGLU -> LN -> Linear(d_hidden, d_out)
    """

    def __init__(self, d_in: int, d_out: int, d_hidden: int | None = None):
        super().__init__()
        from proteinfoundation.nn.modules.swiglu import SwiGLU

        if d_hidden is None:
            d_hidden = d_out

        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden * 2, bias=False),
            SwiGLU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_PROJECTOR_REGISTRY = {
    "mlp": MLPProjector,
    "deep_mlp": DeepMLPProjector,
    "normalized_mlp": NormalizedMLPProjector,
    "attention_pool": AttentionPoolProjector,
    "swiglu_mlp": SwiGLUMLPProjector,
}


def build_text_projector(
    arch: str,
    d_in: int,
    d_out: int,
    d_hidden: int | None = None,
    **kwargs,
) -> nn.Module:
    """Factory function that dispatches on architecture string.

    Args:
        arch: One of 'mlp', 'deep_mlp', 'normalized_mlp', 'attention_pool', or 'swiglu_mlp'.
        d_in: Input dimension.
        d_out: Output dimension.
        d_hidden: Hidden dimension (defaults to d_out inside each class).
        **kwargs: Architecture-specific arguments (num_layers, num_queries, num_heads).
    """
    if arch not in _PROJECTOR_REGISTRY:
        raise ValueError(
            f"Unknown projector arch '{arch}'. Choose from: {list(_PROJECTOR_REGISTRY.keys())}"
        )

    cls = _PROJECTOR_REGISTRY[arch]
    return cls(d_in=d_in, d_out=d_out, d_hidden=d_hidden, **kwargs)
