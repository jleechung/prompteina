"""The decoder's text path must be LIVE, not merely configured.

Every bug this file guards against had the same shape: the run completes, the
loss falls, checkpoints are written, and the text path learned nothing. None of
them raised. They are only catchable by asserting that text actually changes
the output and that gradient actually reaches the new parameters.

Caught in review, in order:
  1. training_step built input_decoder inline and never passed text at all.
  2. `self.hparams.get("decoder")` is ALWAYS {} -- save_hyperparameters() with
     no args stores one key per __init__ arg ({"cfg_ae", "store_dir"}), so
     every capability check read False and the whole path was inert.
  3. Zero-init of the pooled projection is a dead-gradient trap: the
     conditioning vector passes two SwiGLU Transitions and x=0 has an exactly
     zero Jacobian, so the pooled path could never leave zero.
"""
from __future__ import annotations

import pytest
import torch
import yaml
from omegaconf import OmegaConf

CFG_TEXT = "configs/nn_ae/nn_130m_text.yaml"
CFG_BASE = "configs/nn_ae/nn_130m.yaml"
POOLED_DIM = 256   # denoiser dim_cond == width of batch["text_cond"]


def test_initialization_does_not_create_validation_output_dir(tmp_path, monkeypatch):
    """Inference must not leave tmp_ae/val_stuff behind."""
    from proteinfoundation.partial_autoencoder import autoencoder as ae_mod

    class StubModule(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

    monkeypatch.setattr(ae_mod, "EncoderTransformer", StubModule)
    monkeypatch.setattr(ae_mod, "DecoderTransformer", StubModule)
    store_dir = tmp_path / "ae"
    cfg = OmegaConf.create({"nn_ae": {"decoder": {"type": "transformer"}, "latent_z_dim": 4}})

    ae_mod.AutoEncoder(cfg, store_dir=store_dir)

    assert not store_dir.exists()


def _cfg(path, nlayers=2):
    with open(path) as fh:
        c = yaml.safe_load(fh)
    c["decoder"]["nlayers"] = nlayers      # keep the test cheap; cond path unchanged
    return c


def _decoder(cfg):
    from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer
    return DecoderTransformer(**cfg)


def _batch(cfg, b=2, n=8, t=5, text=True):
    z = cfg["decoder"]["latent_z_dim"]
    out = {
        "z_latent": torch.randn(b, n, z),
        "ca_coors_nm": torch.randn(b, n, 3),
        "residue_mask": torch.ones(b, n, dtype=torch.bool),
        "mask": torch.ones(b, n, dtype=torch.bool),
    }
    if text:
        out["text_cond"] = torch.randn(b, POOLED_DIM)
        out["text_tokens"] = torch.randn(b, t, cfg["decoder"]["token_dim"])
        out["text_tokens_mask"] = torch.ones(b, t, dtype=torch.bool)
    return out


def test_stock_config_builds_no_text_path():
    """Backward compatibility: the stock AE must be untouched."""
    d = _decoder(_cfg(CFG_BASE))
    assert d.text_cross_attn_layers is None
    assert getattr(d.cond_factory, "ret_zero", False) is True


def test_text_config_builds_both_paths():
    cfg = _cfg(CFG_TEXT)
    d = _decoder(cfg)
    assert d.text_cross_attn_layers is not None and len(d.text_cross_attn_layers) == 2
    assert d.cond_factory.ret_zero is False
    # The pooled projection must be sized for the DENOISER's width (256), not
    # the decoder's dim_cond (128). Getting this wrong is a runtime shape error.
    assert tuple(d.cond_factory.linear_out.weight.shape) == (
        cfg["decoder"]["dim_cond"], POOLED_DIM)


def test_text_changes_the_sequence_but_never_the_backbone():
    """The whole premise: text may move identity/rotamers, never CA."""
    cfg = _cfg(CFG_TEXT)
    d = _decoder(cfg).eval()
    for layer in d.text_cross_attn_layers.values():
        torch.nn.init.normal_(layer.attn.out_proj.weight, std=0.05)
    torch.nn.init.normal_(d.cond_factory.linear_out.weight, std=0.5)

    base = _batch(cfg)
    other = dict(base)
    other["text_cond"] = torch.randn_like(base["text_cond"]) * 5
    other["text_tokens"] = torch.randn_like(base["text_tokens"]) * 5
    with torch.no_grad():
        a, c = d(dict(base)), d(other)
    assert (a["seq_logits"] - c["seq_logits"]).abs().max() > 1e-4, "text does nothing"
    # CA is overwritten with the decoder's INPUT (decoder.py), so it is
    # structurally impossible for text to move it. If this ever fails, the
    # frozen-denoiser design is void.
    assert torch.equal(a["coors_nm"][:, :, 1], c["coors_nm"][:, :, 1])


@pytest.mark.parametrize("param,std", [("pooled", 1e-4), ("xattn", 0.0)])
def test_gradient_reaches_both_text_paths(param, std):
    """A path that cannot receive gradient trains to nothing, silently.

    The pooled projection must NOT be zero-initialized: its output feeds two
    SwiGLU Transitions whose Jacobian at exactly 0 is 0, so a zero init is a
    permanent dead parameter. The cross-attention out_proj IS safe to zero --
    its gradient is input^T @ grad_out and the attention output is non-zero.
    """
    cfg = _cfg(CFG_TEXT)
    d = _decoder(cfg).train()
    torch.nn.init.normal_(d.cond_factory.linear_out.weight, std=1e-4)
    for layer in d.text_cross_attn_layers.values():
        torch.nn.init.zeros_(layer.attn.out_proj.weight)

    d(_batch(cfg))["seq_logits"].sum().backward()
    target = (d.cond_factory.linear_out.weight if param == "pooled"
              else d.text_cross_attn_layers["0"].attn.out_proj.weight)
    assert target.grad is not None, f"{param}: no grad at all"
    assert float(target.grad.abs().sum()) > 0.0, (
        f"{param} path receives exactly zero gradient -- it will train to "
        f"nothing while the loss still falls")


def test_capability_is_read_from_the_module_not_hparams():
    """Guards the bug that made everything inert.

    `save_hyperparameters()` with no args stores one key per __init__ argument,
    so self.hparams == {"cfg_ae", "store_dir"} and self.hparams["decoder"] does
    not exist. Any capability check routed through it silently reads False.
    """
    import inspect

    from proteinfoundation.partial_autoencoder import autoencoder as ae_mod

    # Strip comments: the fix's own explanatory comment quotes the broken
    # pattern, and matching that would fail forever on correct code.
    src = "\n".join(
        line for line in inspect.getsource(ae_mod).splitlines()
        if not line.strip().startswith("#")
    )
    for key in ("\"decoder\"", "\"text_cond\"", "\"latent_dropout_p\"",
                "\"text_p_uncond\"", "\"text_encoder_ckpt_path\""):
        assert f"self.hparams.get({key}" not in src and f"self.hparams[{key}]" not in src, (
            f"config read {key} routed through self.hparams; it is not the "
            f"experiment config and will silently return empty")


# ---------------------------------------------------------------------------
# Getting text INTO the decoder at generation time.
#
# The decoder can be perfectly text-conditioned and still be evaluated blind:
# every search strategy builds its own `extra_info` dict, and for a long time
# only proteina.py passed the text keys. All seven search paths -- including
# beam search, which the H-bond factorial runs on -- passed {"mask": ...} only,
# so `decode()` received text_cond=None and silently used its unconditional
# mode. That produces a clean, confident, meaningless null result.
# ---------------------------------------------------------------------------
import ast
import pathlib

SEARCH_DIR = pathlib.Path("src/proteinfoundation/search")


def _sample_formatting_calls(path):
    tree = ast.parse(path.read_text())
    return [c for c in ast.walk(tree)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            and c.func.id == "sample_formatting"]


@pytest.mark.parametrize("path", sorted(SEARCH_DIR.glob("*.py")), ids=lambda p: p.name)
def test_every_search_path_passes_text_to_the_decoder(path):
    """A decode that drops text evaluates the pretrained decoder, not ours."""
    for call in _sample_formatting_calls(path):
        extra = next((k.value for k in call.keywords if k.arg == "extra_info"), None)
        if extra is None:
            continue
        # The text keys arrive as **decoder_text_extra_info(...), which ast
        # records as a dict entry whose key is None.
        splatted = [
            v for k, v in zip(extra.keys, extra.values)
            if k is None and isinstance(v, ast.Call)
            and getattr(v.func, "id", None) == "decoder_text_extra_info"
        ]
        assert splatted, (
            f"{path.name}:{call.lineno} calls sample_formatting without "
            f"decoder_text_extra_info; the AE decoder will be handed "
            f"text_cond=None and silently decode unconditionally")


def test_text_rows_stay_aligned_with_their_samples():
    """Wrong expansion mode is worse than no text: it conditions on a
    DIFFERENT design and still looks like it works."""
    from proteinfoundation.utils.sample_utils import decoder_text_extra_info

    batch = {"text_cond": torch.tensor([[0.0], [1.0], [2.0]])}       # 3 samples
    interleaved = decoder_text_extra_info(batch, 6, "interleave")["text_cond"]
    tiled = decoder_text_extra_info(batch, 6, "tile")["text_cond"]
    assert interleaved.squeeze(-1).tolist() == [0, 0, 1, 1, 2, 2]
    assert tiled.squeeze(-1).tolist() == [0, 1, 2, 0, 1, 2]

    # Non-exact multiples are tiled up then truncated, matching how
    # generate_samples_to_completion trims its expanded mask.
    assert decoder_text_extra_info(batch, 4, "tile")["text_cond"].squeeze(-1).tolist() == [0, 1, 2, 0]

    # A text-blind run must stay clean rather than fabricating zeros.
    assert decoder_text_extra_info({}, 6, "tile") == {}
    with pytest.raises(ValueError):
        decoder_text_extra_info(batch, 6, "repeat_interleave")
