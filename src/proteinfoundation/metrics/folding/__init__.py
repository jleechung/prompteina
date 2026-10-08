"""Folding backends used for binder scoring.

Backends are imported lazily: ESMFold2 and AlphaFold3 pull in large, optional
dependency stacks, and importing this package must not require either.
"""

from proteinfoundation.metrics.folding.base import FoldingBackend, FoldResult, chain_labels

__all__ = ["FoldingBackend", "FoldResult", "chain_labels"]
