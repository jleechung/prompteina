"""ESMFold2-Fast reward: refold the complex, score it with ipSAE_min.

Written for inference-time search. `beam_search` rolls each candidate out to
t=1 and hands the reward a finished all-atom complex on disk, so the reward can
read both sequences off that file, refold them, and return a binding score.

What this is NOT: the AlphaFold2 reward that upstream Complexa uses. That path
goes through ColabDesign and scores with interface PAE, complex pLDDT and
binder RMSD. This one needs no MSA, no ColabDesign and no AlphaFold weights.

Reward convention: `total_reward` is ipSAE_min, and higher is better. The
search algorithms take `topk` over `total_reward`, so no sign flip is needed.

Cost. One fold per candidate, called serially by
`compute_reward_from_samples`, with the ESM-C encoder resident on the GPU
alongside the generative model. At `beam_width * n_branch * nsamples`
candidates per checkpoint this dominates wall clock, and it does not fit on a
24GB card. Budget for it before turning it on.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from proteinfoundation.metrics.binding_score import (
    DEFAULT_DIST_CUTOFF,
    DEFAULT_IMPLEMENTATION,
    DEFAULT_PAE_CUTOFF,
    ipsae_from_fold,
)
from proteinfoundation.metrics.folding.esmfold2 import ESMFold2Backend
from proteinfoundation.rewards.base_reward import BaseRewardModel, ensure_tensor, standardize_reward
from proteinfoundation.utils.pdb_utils import extract_seq_from_pdb, get_chain_ids_from_pdb

logger = logging.getLogger(__name__)


class ESMFold2Reward(BaseRewardModel):
    """Refolds a designed complex with ESMFold2-Fast and returns ipSAE_min.

    Args:
        model: "esmfold2-fast" (default) or "esmfold2".
        num_loops: ESMFold2 recycling loops.
        num_sampling_steps: ESMFold2 diffusion steps.
        seed: Fold seed. The paper's numbers are single-seed.
        pae_cutoff: PAE gate for ipSAE, in Angstrom.
        dist_cutoff: Distance gate, reference implementation only.
        implementation: "reference" for the vendored DunbrackLab script, which
            is what the paper reports, or "inprocess" for the torch version.
            The reference runs a subprocess per candidate; inside a search loop
            that is small next to the fold, but it is the knob to turn if the
            subprocess becomes the bottleneck.
        device: Torch device for the folding model.
        failure_reward: `total_reward` to return when a fold fails. Defaults to
            0.0, which is the floor of ipSAE and so ranks a failure last
            without making it look like a negative result.
    """

    IS_FOLDING_MODEL = True
    SUPPORTS_GRAD = False
    SUPPORTS_SAVE_PDB = True

    def __init__(
        self,
        model: str = "esmfold2-fast",
        num_loops: int = 3,
        num_sampling_steps: int = 50,
        seed: int = 0,
        pae_cutoff: float = DEFAULT_PAE_CUTOFF,
        dist_cutoff: float = DEFAULT_DIST_CUTOFF,
        implementation: str = DEFAULT_IMPLEMENTATION,
        device: str = "cuda",
        failure_reward: float = 0.0,
    ) -> None:
        self.backend = ESMFold2Backend(
            model=model,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            device=device,
        )
        self.seed = seed
        self.pae_cutoff = pae_cutoff
        self.dist_cutoff = dist_cutoff
        self.implementation = implementation
        self.failure_reward = failure_reward

    def score(
        self,
        pdb_path: str,
        requires_grad: bool = False,
        binder_chain: str | None = None,
        target_chain: str | None = None,
        save_pdb: str | bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fold the complex described by `pdb_path` and score the interface.

        Args:
            pdb_path: Designed complex. Only its sequences are used; its
                coordinates do not enter the fold.
            requires_grad: Must be False; this reward has no gradient.
            binder_chain: Chain ID of the binder. Supplied by
                `compute_reward_from_samples`. Falling back to the last chain
                is a last resort, not a default, because ipSAE is not
                symmetric in the two roles.
            target_chain: Comma-separated target chain IDs.
            save_pdb: Path to write the refolded complex to, as mmCIF.

        Returns:
            Standard reward dict. `total_reward` is ipSAE_min.
        """
        self._check_capabilities(requires_grad=requires_grad, save_pdb=bool(save_pdb))

        if binder_chain is None or target_chain is None:
            fallback_target, fallback_binder = get_chain_ids_from_pdb(pdb_path)
            if binder_chain is None:
                logger.warning(
                    "binder_chain not supplied for %s; falling back to the last chain %r. "
                    "ipSAE is asymmetric, so getting this wrong changes the score.",
                    pdb_path,
                    fallback_binder,
                )
                binder_chain = fallback_binder
            if target_chain is None:
                target_chain = fallback_target

        target_chains = [c for c in str(target_chain).split(",") if c]
        if not target_chains:
            raise ValueError(f"{pdb_path}: no target chains; got target_chain={target_chain!r}")

        binder_seq = extract_seq_from_pdb(pdb_path, chain_id=binder_chain)
        target_seqs = [extract_seq_from_pdb(pdb_path, chain_id=c) for c in target_chains]

        try:
            result = self.backend.fold(binder_seq, target_seqs, seed=self.seed)
            metrics = ipsae_from_fold(
                result,
                pae_cutoff=self.pae_cutoff,
                dist_cutoff=self.dist_cutoff,
                implementation=self.implementation,
            )
            if save_pdb:
                out = Path(save_pdb if isinstance(save_pdb, str) else f"{os.path.splitext(pdb_path)[0]}_refolded.cif")
                out.write_text(result.cif)
        except Exception as exc:  # noqa: BLE001 - a failed fold must not stop a search
            # CompositeRewardModel swallows child exceptions, so raising here
            # would silently contribute zero anyway while hiding the cause.
            logger.warning("ESMFold2 fold failed for %s: %s: %s", pdb_path, type(exc).__name__, exc)
            metrics = {"ipsae_min": self.failure_reward, "fold_failed": 1.0}

        reward = {f"{self.backend.name}_{k}".replace("-", "_"): ensure_tensor(v) for k, v in metrics.items()}
        return standardize_reward(
            reward=reward,
            grad={},
            total_reward=metrics["ipsae_min"],
            **{k: v for k, v in metrics.items() if k not in ("ipsae_min",)},
        )

    def cleanup(self) -> None:
        self.backend.close()
