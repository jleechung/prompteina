"""ipSAE over a predicted binder--target complex.

ipSAE is a TM-score-like transform of predicted aligned error restricted to
residue pairs that cross the interface, with the distance parameter scaled by
interface size. It is asymmetric, so it is computed in both directions and the
smaller of the two is taken; that is ipSAE_min, the quantity the paper reports.

Two implementations are available. "reference" is the DunbrackLab script
vendored at community_models/IPSAE/, which is what the paper's numbers were
produced with, and is the default. "inprocess" is metrics/ipsae.py, a torch
reimplementation that needs no subprocess; it is faster and differs in how it
treats residues with no confident partner, so it is a reasonable choice inside
a search loop but its numbers are not the reported ones.

The masks here are built from `FoldResult.chain_lengths`, which records the
order the chains were submitted in. They are not inferred from the structure
file. Folding models disagree about chain labeling and residue order, and
ipSAE is not invariant to swapping the binder and target roles, so a score
derived from file order is a different number than one derived from the
submitted order.

All target chains are merged into a single target side. ipSAE is defined over
one chain pair; a complex with several target chains has to be reduced to one
pair somehow, and merging keeps every cross-interface residue pair in the
calculation rather than averaging over per-chain scores.
"""

from __future__ import annotations

import numpy as np
import torch

from proteinfoundation.metrics.folding.base import FoldResult
from proteinfoundation.metrics.ipsae import ipSAE

#: PAE gate, in Angstrom. Residue pairs above this do not enter the score.
DEFAULT_PAE_CUTOFF = 10.0


def interface_masks(chain_lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean masks for the binder side and the merged target side.

    Args:
        chain_lengths: Residue count per chain in submitted order. The binder
            is first.

    Returns:
        (binder_mask, target_mask), each (L,) float tensors over the whole
        complex.
    """
    if len(chain_lengths) < 2:
        raise ValueError(f"need a binder and at least one target chain, got {chain_lengths}")
    total = sum(chain_lengths)
    binder = torch.zeros(total)
    target = torch.zeros(total)
    binder[: chain_lengths[0]] = 1.0
    target[chain_lengths[0] :] = 1.0
    return binder, target


#: Distance gate in Angstrom, used by the reference implementation only.
DEFAULT_DIST_CUTOFF = 15.0

#: Which ipSAE implementation to use by default.
DEFAULT_IMPLEMENTATION = "reference"


def ipsae_from_fold(
    result: FoldResult,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
    implementation: str = DEFAULT_IMPLEMENTATION,
) -> dict[str, float]:
    """Score one predicted complex.

    Args:
        result: Output of a FoldingBackend.
        pae_cutoff: PAE gate in Angstrom.
        dist_cutoff: Distance gate in Angstrom. Reference implementation only.
        implementation: "reference" for the vendored DunbrackLab script, which
            is what the paper reports, or "inprocess" for the torch version,
            which avoids a subprocess per design.

    Returns:
        Dict with ipsae_min, plus binder_plddt and any scalar the backend
        reported (ptm, iptm, ...), prefixed with the backend name so scores
        from different models never collide in one table. The reference
        implementation adds its full table: ipSAE_AB/_BA/_min/_max, pDockQ,
        pDockQ2, LIS and the ipTM variants.
    """
    if implementation == "reference":
        from proteinfoundation.metrics.ipsae_reference import score_fold

        out = score_fold(result, pae_cutoff=pae_cutoff, dist_cutoff=dist_cutoff)
    elif implementation == "inprocess":
        out = _ipsae_inprocess(result, pae_cutoff)
    else:
        raise ValueError(f"implementation must be 'reference' or 'inprocess', got {implementation!r}")

    if result.plddt is not None:
        out["binder_plddt"] = float(np.mean(result.plddt[: result.binder_length]))
        out["complex_plddt"] = float(np.mean(result.plddt))
    for key, value in result.scalars.items():
        out[f"{result.backend}_{key}" if result.backend else key] = float(value)
    return out


def _ipsae_inprocess(result: FoldResult, pae_cutoff: float) -> dict[str, float]:
    """metrics/ipsae.py, driven off the submitted chain order."""
    binder_mask, target_mask = interface_masks(result.chain_lengths)
    pae = torch.from_numpy(np.asarray(result.pae, dtype=np.float32))
    scores = ipSAE(pae, binder_mask, target_mask, cutoff=pae_cutoff)
    return {
        "ipsae_min": float(scores["min"]),
        "ipsae_max": float(scores["max"]),
        "ipsae_avg": float(scores["avg"]),
    }
