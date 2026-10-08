"""ESMFold2-Fast backend.

Folds a binder--target complex from sequence alone: no MSA, no template, no
structural input. This is the model the paper scores with, and it is fast
enough to use as a reward inside inference-time search.

Two operational notes, both load-bearing:

ESMFold2 and its ESM-C encoder must be loaded from compatible revisions. Both
are pinned below; changing either changes the reported scores. Export
HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 once the weights are cached.

The ESM-C encoder is roughly 24GB on its own. A 24GB card cannot hold it
alongside anything else, which rules out using this reward in-loop on one.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from proteinfoundation.metrics.folding.base import FoldingBackend, FoldResult, chain_labels

logger = logging.getLogger(__name__)

#: Repository and pinned revision per model. The revision is part of the
#: identity of a reported number; changing it changes the scores.
MODELS: dict[str, tuple[str, str]] = {
    "esmfold2-fast": ("biohub/ESMFold2-Fast", "c6c7958d63f5f2f1f0fed0bb9462316f8ccceea6"),
    "esmfold2": ("biohub/ESMFold2", "8fc3ff471022fdce52c77030685eb775de0c00a3"),
}

# The ESMFold2 releases above use the original Transformers 4.57 ESMC layout.
# Newer ESMC revisions use different parameter names and otherwise load almost
# the entire 6B encoder as random weights without raising an exception.
ESMC_MODEL = ("biohub/ESMC-6B", "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a")


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _harvest(result, n_res: int) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Pull scalars and per-residue or pairwise arrays off a fold result.

    The result object's attribute names are not part of a documented API, so
    this reads whatever is there and keeps what has the right shape, rather
    than naming fields that may move between releases.
    """
    scalars: dict[str, float] = {}
    arrays: dict[str, np.ndarray] = {}
    for attr in sorted(a for a in dir(result) if not a.startswith("_")):
        try:
            value = getattr(result, attr)
        except Exception:  # noqa: BLE001 - attribute access can raise anything
            continue
        if callable(value):
            continue
        try:
            arr = _as_numpy(value)
        except Exception:  # noqa: BLE001
            continue
        if arr.ndim == 0 and np.isfinite(arr):
            scalars[attr] = float(arr)
        elif arr.ndim == 1 and arr.shape[0] == n_res:
            arrays[attr] = arr
        elif arr.ndim == 2 and arr.shape[0] == arr.shape[1] == n_res:
            arrays[attr] = arr
    return scalars, arrays


class ESMFold2Backend(FoldingBackend):
    """ESMFold2 / ESMFold2-Fast, loaded once and reused.

    Args:
        model: Key into MODELS. "esmfold2-fast" is the default and is what the
            paper reports.
        num_loops: Recycling loops.
        num_sampling_steps: Diffusion steps.
        device: Torch device for the model.
    """

    #: Sequence is enough; the designed structure is not used.
    REQUIRES_TEMPLATE = False

    def __init__(
        self,
        model: str = "esmfold2-fast",
        num_loops: int = 3,
        num_sampling_steps: int = 50,
        device: str = "cuda",
    ) -> None:
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; options are {sorted(MODELS)}")
        self.name = model
        self.num_loops = num_loops
        self.num_sampling_steps = num_sampling_steps
        self.device = device
        self._model = None
        self._builder = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from esm.models.esmfold2 import ESMFold2InputBuilder
        from huggingface_hub import snapshot_download
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        repo_id, revision = MODELS[self.name]
        esmc_id, esmc_revision = ESMC_MODEL
        logger.info("loading %s @ %s with %s @ %s", repo_id, revision, esmc_id, esmc_revision)
        esmc_path = snapshot_download(esmc_id, revision=esmc_revision)
        self._model = ESMFold2Model.from_pretrained(
            repo_id, revision=revision, load_esmc=False
        ).to(self.device).eval()
        self._model.load_esmc(esmc_path)
        self._builder = ESMFold2InputBuilder()

    def fold(
        self,
        binder_seq: str,
        target_seqs: list[str],
        seed: int = 0,
        template_pdb: str | None = None,
    ) -> FoldResult:
        del template_pdb  # sequence-only model
        from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

        self._ensure_model()
        ids = chain_labels(1 + len(target_seqs))
        lengths = [len(binder_seq)] + [len(t) for t in target_seqs]
        n_res = sum(lengths)

        spi = StructurePredictionInput(
            sequences=[ProteinInput(id=ids[0], sequence=binder_seq)]
            + [ProteinInput(id=ids[k + 1], sequence=t) for k, t in enumerate(target_seqs)]
        )
        result = self._builder.fold(
            self._model,
            spi,
            num_loops=self.num_loops,
            num_sampling_steps=self.num_sampling_steps,
            num_diffusion_samples=1,
            seed=seed,
        )

        scalars, arrays = _harvest(result, n_res)
        pae = arrays.get("pae")
        if pae is None:
            two_d = {k: v.shape for k, v in arrays.items() if v.ndim == 2}
            raise RuntimeError(
                f"no ({n_res}, {n_res}) PAE on the fold result; 2-D arrays present: {two_d}"
            )
        return FoldResult(
            pae=pae,
            plddt=arrays.get("plddt"),
            cif=result.complex.to_mmcif(),
            chain_ids=ids,
            chain_lengths=lengths,
            scalars=scalars,
            backend=self.name,
        )

    def close(self) -> None:
        self._model = None
        self._builder = None
        # close() runs from __del__ paths too, where module globals may already
        # be torn down and `torch` is None. Freeing the cache is best effort.
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
