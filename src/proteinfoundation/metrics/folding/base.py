"""Folding backends for binder scoring.

A backend takes a binder sequence and one or more target sequences and returns
the predicted complex plus its predicted aligned error matrix. That is all the
binding score needs: `ipSAE` is a transform of PAE restricted to residue pairs
across the interface.

Chain order is fixed by this interface rather than inferred. The binder is
always chain A and submitted first; targets follow as B, C, ... in the order
given. `FoldResult.chain_lengths` records the submitted order, so the interface
masks are built from known lengths instead of being guessed from a file. Chain
labeling is not consistent between folding models, and a score computed with
the binder and target roles swapped is a different number.
"""

from __future__ import annotations

import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FoldResult:
    """One predicted complex.

    Attributes:
        pae: Predicted aligned error, (L, L), in Angstrom, where L is the total
            residue count across all chains in submitted order.
        plddt: Per-residue confidence, (L,), or None if the backend omits it.
        cif: mmCIF text of the predicted complex.
        chain_ids: Chain labels in submitted order; chain_ids[0] is the binder.
        chain_lengths: Residue count per chain, same order as chain_ids.
        scalars: Whatever scalar confidences the backend reported, e.g. ptm,
            iptm. Keys are backend-specific and are passed through unchanged.
        backend: Name of the backend that produced this, used to namespace
            those scalars so two models' columns never collide in one table.
    """

    pae: np.ndarray
    plddt: np.ndarray | None
    cif: str
    chain_ids: list[str]
    chain_lengths: list[int]
    scalars: dict[str, float] = field(default_factory=dict)
    backend: str = ""

    def __post_init__(self) -> None:
        total = sum(self.chain_lengths)
        if self.pae.shape != (total, total):
            raise ValueError(
                f"PAE is {self.pae.shape}, expected ({total}, {total}) from chain "
                f"lengths {self.chain_lengths}. A mismatch here means the PAE and "
                f"the structure disagree about residue order, and every interface "
                f"mask built from it would be wrong."
            )
        if len(self.chain_ids) != len(self.chain_lengths):
            raise ValueError("chain_ids and chain_lengths differ in length")

    @property
    def binder_length(self) -> int:
        return self.chain_lengths[0]

    @property
    def target_length(self) -> int:
        return sum(self.chain_lengths[1:])


def chain_labels(n_chains: int) -> list[str]:
    """Chain labels for a complex: binder first, then targets.

    Returns A, B, C, ... which is what both supported backends accept.
    """
    if n_chains > len(string.ascii_uppercase):
        raise ValueError(f"{n_chains} chains exceeds single-letter labeling")
    return list(string.ascii_uppercase[:n_chains])


class FoldingBackend(ABC):
    """Predicts a binder--target complex from sequence."""

    #: Short name used in config and in output column prefixes.
    name: str = "folding"

    #: True when the backend needs the designed structure, not just sequence.
    REQUIRES_TEMPLATE: bool = False

    @abstractmethod
    def fold(
        self,
        binder_seq: str,
        target_seqs: list[str],
        seed: int = 0,
        template_pdb: str | None = None,
    ) -> FoldResult:
        """Predict the complex.

        Args:
            binder_seq: Binder sequence, becomes chain A.
            target_seqs: One sequence per target chain, become B, C, ...
            seed: Backend random seed.
            template_pdb: Path to the designed structure. Backends with
                REQUIRES_TEMPLATE set need this; the others ignore it.
                AlphaFold3 run without MSAs and without a template returns an
                ipSAE of zero for every design, so for that backend the
                designed structure is not optional.

        Returns:
            FoldResult with the PAE and the predicted structure.
        """

    def close(self) -> None:
        """Release GPU memory. Safe to call more than once."""
