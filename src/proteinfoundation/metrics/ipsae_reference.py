"""ipSAE via the DunbrackLab reference implementation.

`community_models/IPSAE/ipsae.py` is vendored unmodified from
https://github.com/DunbrackLab/IPSAE. It is the implementation the paper's
numbers were produced with, so this is the path that reproduces them.

It is a command-line script, not a library, so this module drives it: write the
complex as a two-chain PDB and the predicted aligned error as an AlphaFold2
style JSON next to it, run the script, parse its table.

Two chains, always. ipSAE is defined over a chain pair. A complex with several
target chains has to be reduced to one pair, and this merges every target chain
into one rather than scoring the target chains separately and averaging, so no
cross-interface residue pair is dropped from the calculation.

Chain roles come from `FoldResult.chain_lengths`, the order the chains were
submitted in, never from the order a structure file lists them. ipSAE is
asymmetric in the two roles, so getting this backwards returns a different
number rather than an error.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from proteinfoundation.metrics.folding.base import FoldResult

logger = logging.getLogger(__name__)

def _find_ipsae_script() -> Path:
    """Locate the vendored ipsae.py.

    An editable install leaves this file under `src/`, so the repository root
    is four levels up. A plain `pip install .` puts it in site-packages, where
    that level is wrong, so the package directory is checked too. IPSAE_SCRIPT
    names the first candidate either way, which keeps the error message
    pointing somewhere a reader can act on.
    """
    env = os.environ.get("IPSAE_SCRIPT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "community_models" / "IPSAE" / "ipsae.py",  # repo root
        here.parents[2] / "community_models" / "IPSAE" / "ipsae.py",  # site-packages
        here.parents[2] / "IPSAE" / "ipsae.py",
    ]
    return next((c for c in candidates if c.exists()), candidates[0])


#: The vendored reference implementation. Override with IPSAE_SCRIPT.
IPSAE_SCRIPT = _find_ipsae_script()

#: Columns the script emits that are worth carrying forward. Each becomes
#: <name>_AB, <name>_BA, <name>_min and <name>_max.
NUMERIC_COLUMNS = (
    "ipSAE",
    "ipSAE_d0chn",
    "ipSAE_d0dom",
    "ipTM_af",
    "ipTM_d0chn",
    "pDockQ",
    "pDockQ2",
    "LIS",
)


def write_two_chain_pdb(result: FoldResult, path: Path) -> None:
    """Write the complex as chain A (binder) and chain B (all targets).

    Residue order follows the submitted chain order, so it matches the PAE
    matrix index for index.
    """
    import biotite.structure as bstruc
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure.io.pdbx as pdbx

    cif_path = path.with_suffix(".in.cif")
    cif_path.write_text(result.cif)
    block = pdbx.CIFFile.read(str(cif_path))
    arr = pdbx.get_structure(block, model=1)
    arr = arr[bstruc.filter_amino_acids(arr)]

    # Relabel: the first submitted chain is the binder, everything else merges.
    binder_id = result.chain_ids[0]
    new_chain = np.where(arr.chain_id == binder_id, "A", "B")
    arr.chain_id = new_chain

    # Renumber residues within each chain so the merged target is contiguous
    # and the script's residue walk does not skip.
    for chain in ("A", "B"):
        mask = arr.chain_id == chain
        if not mask.any():
            raise ValueError(f"chain {chain} is empty after merging; chains were {result.chain_ids}")
        res_ids = arr.res_id[mask]
        _, inverse = np.unique(res_ids, return_inverse=True)
        order = np.argsort(np.argsort(res_ids, kind="stable"), kind="stable")
        del order
        arr.res_id[mask] = inverse + 1

    out = pdb_io.PDBFile()
    out.set_structure(arr)
    out.write(str(path))
    cif_path.unlink(missing_ok=True)


def write_pae_json(path: Path, pae: np.ndarray, plddt: np.ndarray | None, scalars: dict) -> None:
    """AlphaFold2-style PAE JSON, which is what the script's af2 mode reads."""
    doc: dict = {"pae": np.asarray(pae, dtype=np.float32).tolist()}
    if plddt is not None:
        doc["plddt"] = np.asarray(plddt, dtype=np.float32).tolist()
    for key in ("ptm", "iptm"):
        if key in scalars:
            doc[key] = float(scalars[key])
    path.write_text(json.dumps(doc))


def parse_output(text: str) -> dict[str, float]:
    """Parse the score table into <column>_AB / _BA / _min / _max.

    The file opens with a blank line, so the header is found by content rather
    than by reading line zero -- doing the latter yields an empty result that
    looks like a legitimate zero score.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header_idx = next((i for i, ln in enumerate(lines) if ln.split()[:2] == ["Chn1", "Chn2"]), None)
    if header_idx is None:
        raise ValueError(f"no 'Chn1 Chn2' header in ipsae output:\n{text[:400]}")
    header = lines[header_idx].split()
    rows = [ln.split() for ln in lines[header_idx + 1 :]]
    asym = [r for r in rows if len(r) > 4 and r[4] == "asym"]
    if len(asym) != 2:
        raise ValueError(f"expected exactly 2 asym rows for a chain pair, got {len(asym)}")

    out: dict[str, float] = {}
    for column in NUMERIC_COLUMNS:
        if column not in header:
            continue
        col = header.index(column)
        try:
            ab, ba = float(asym[0][col]), float(asym[1][col])
        except (IndexError, ValueError):
            continue
        out[f"{column}_AB"] = ab
        out[f"{column}_BA"] = ba
        out[f"{column}_min"] = min(ab, ba)
        out[f"{column}_max"] = max(ab, ba)
    if "ipSAE_min" not in out:
        raise ValueError(f"no ipSAE column in ipsae output header: {header}")
    return out


def score_fold(
    result: FoldResult,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 15.0,
    python: str | None = None,
    keep_dir: Path | None = None,
) -> dict[str, float]:
    """Run the reference implementation on one predicted complex.

    Args:
        result: Output of a FoldingBackend.
        pae_cutoff: PAE gate in Angstrom.
        dist_cutoff: Distance gate in Angstrom.
        python: Interpreter to run the script with. Defaults to this one.
        keep_dir: Write the intermediates here instead of a temp directory
            that is deleted, for debugging.

    Returns:
        ipSAE_min and the rest of the table, plus ipsae_min as a lowercase
        alias so callers do not have to know which implementation produced it.
    """
    if not IPSAE_SCRIPT.exists():
        raise FileNotFoundError(
            f"vendored ipsae.py not found at {IPSAE_SCRIPT}. It ships in "
            f"community_models/IPSAE/; set IPSAE_SCRIPT to point at it, or "
            f"use --ipsae inprocess. See NOTICE for its license."
        )
    work = Path(keep_dir) if keep_dir else Path(tempfile.mkdtemp(prefix="ipsae_", dir=os.environ.get("TMPDIR")))
    work.mkdir(parents=True, exist_ok=True)
    try:
        pdb = work / "complex.pdb"
        pae_json = work / "complex_pae.json"
        write_two_chain_pdb(result, pdb)
        write_pae_json(pae_json, result.pae, result.plddt, result.scalars)

        # The script derives its output path from the structure path, the PAE
        # cutoff and the distance cutoff, with the cutoffs formatted as ints.
        stem = work / f"complex_{int(pae_cutoff)}_{int(dist_cutoff)}"
        proc = subprocess.run(
            [python or sys.executable, str(IPSAE_SCRIPT), str(pae_json), str(pdb), str(pae_cutoff), str(dist_cutoff)],
            capture_output=True,
            text=True,
            check=False,
        )
        out_txt = stem.with_suffix(".txt")
        if not out_txt.exists():
            raise RuntimeError(
                f"ipsae.py wrote no output.\nexit={proc.returncode}\n"
                f"stdout: {proc.stdout[-600:]}\nstderr: {proc.stderr[-600:]}"
            )
        scores = parse_output(out_txt.read_text())
        scores["ipsae_min"] = scores["ipSAE_min"]
        return scores
    finally:
        if keep_dir is None:
            shutil.rmtree(work, ignore_errors=True)
