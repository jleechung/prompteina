"""ProteinMPNN redesign of a binder, with and without a fixed interface.

The paper scores each design under three sequence conditions:

    self     the sequence the generative model co-designed
    MPNN     free ProteinMPNN redesign of the whole binder
    MPNN-FI  redesign with the generated interface residues held fixed

The gap between the last two separates an interface the model produced from one
ProteinMPNN rewrote.

ProteinMPNN ships as a command-line script, so this drives it as a subprocess:
`community_models/ProteinMPNN/protein_mpnn_run.py`, weights from
`./env/download_startup.sh --pmpnn`.

Only the binder chain is redesigned. Target chains are passed as context and
left alone, which `--pdb_path_chains` controls.
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

logger = logging.getLogger(__name__)

MPNN_DIR = Path(__file__).resolve().parents[3] / "community_models" / "ProteinMPNN"
MPNN_SCRIPT = MPNN_DIR / "protein_mpnn_run.py"

#: Heavy-atom distance defining an interface residue, in Angstrom.
INTERFACE_CUTOFF = 6.0

#: Paper settings: 8 sequences, temperature 0.1, the v_48_020 weights.
DEFAULT_NUM_SEQS = 8
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MODEL = "v_48_020"


def interface_residues(pdb_path: str, binder_chain: str, target_chains: list[str],
                       cutoff: float = INTERFACE_CUTOFF) -> list[int]:
    """Binder residues with any heavy atom within `cutoff` of any target heavy atom.

    Returns 1-based indices into the binder chain, which is the numbering
    ProteinMPNN's fixed-position dictionary expects. These are positions
    within the chain, not author residue numbers.
    """
    import biotite.structure as bstruc
    import biotite.structure.io.pdb as pdb_io

    arr = pdb_io.PDBFile.read(pdb_path).get_structure(model=1)
    arr = arr[bstruc.filter_amino_acids(arr)]
    arr = arr[arr.element != "H"]

    binder = arr[arr.chain_id == binder_chain]
    target = arr[np.isin(arr.chain_id, target_chains)]
    if len(binder) == 0 or len(target) == 0:
        raise ValueError(
            f"{pdb_path}: binder chain {binder_chain!r} has {len(binder)} atoms and "
            f"target chains {target_chains} have {len(target)}"
        )

    # Position within the chain, which is what ProteinMPNN indexes by. Using
    # author residue numbers here would silently pin the wrong residues on any
    # structure whose numbering does not start at 1.
    res_ids = np.unique(binder.res_id)
    position_of = {rid: i + 1 for i, rid in enumerate(res_ids)}

    cell = bstruc.CellList(target, cell_size=cutoff)
    contacts = cell.get_atoms(binder.coord, radius=cutoff)
    touching = (contacts != -1).any(axis=1)
    return sorted({position_of[rid] for rid in binder.res_id[touching]})


def _write_fixed_positions(path: Path, design_name: str, binder_chain: str,
                           target_chains: list[str], positions: list[int]) -> None:
    """ProteinMPNN's fixed-position JSONL.

    Target chains must appear with an empty list. Omitting them makes
    ProteinMPNN's featuriser raise a KeyError partway through the run.
    """
    entry = {design_name: {binder_chain: positions, **{c: [] for c in target_chains}}}
    path.write_text(json.dumps(entry) + "\n")


def _parse_fasta(path: Path) -> list[str]:
    """Sequences from a ProteinMPNN FASTA, dropping the input sequence.

    The first record is the structure's own sequence echoed back, not a
    redesign. Counting it as one of the eight would mix the `self` condition
    into the `MPNN` one.
    """
    seqs, current = [], []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current:
                seqs.append("".join(current))
                current = []
        elif line.strip():
            current.append(line.strip())
    if current:
        seqs.append("".join(current))
    return seqs[1:]


def redesign(
    pdb_path: str,
    binder_chain: str,
    target_chains: list[str],
    fix_interface: bool = False,
    num_seqs: int = DEFAULT_NUM_SEQS,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = 42,
    model_name: str = DEFAULT_MODEL,
    weights_dir: str | None = None,
) -> list[str]:
    """Redesign the binder chain and return the sampled sequences.

    Args:
        pdb_path: The binder--target complex.
        binder_chain: Chain to redesign.
        target_chains: Chains to keep as context.
        fix_interface: Hold the generated interface residues fixed (MPNN-FI).
        num_seqs: Sequences to sample.
        temperature: Sampling temperature.
        seed: ProteinMPNN seed. 0 means a random seed in its CLI, so avoid it.
        model_name: Weight set.
        weights_dir: Overrides the vendored weights location.

    Returns:
        `num_seqs` binder sequences, the input sequence excluded.
    """
    if not MPNN_SCRIPT.exists():
        raise FileNotFoundError(f"ProteinMPNN not found at {MPNN_SCRIPT}")
    weights = Path(weights_dir) if weights_dir else MPNN_DIR / "vanilla_model_weights"
    if not (weights / f"{model_name}.pt").exists():
        raise FileNotFoundError(
            f"ProteinMPNN weights missing at {weights / (model_name + '.pt')}. "
            f"Fetch them with ./env/download_startup.sh --pmpnn"
        )

    work = Path(tempfile.mkdtemp(prefix="mpnn_", dir=os.environ.get("TMPDIR")))
    try:
        design_name = Path(pdb_path).stem
        cmd = [
            sys.executable, str(MPNN_SCRIPT),
            "--pdb_path", str(pdb_path),
            "--pdb_path_chains", binder_chain,
            "--out_folder", str(work),
            "--num_seq_per_target", str(num_seqs),
            "--sampling_temp", str(temperature),
            "--seed", str(seed),
            "--model_name", model_name,
            "--path_to_model_weights", str(weights),
        ]
        if fix_interface:
            positions = interface_residues(pdb_path, binder_chain, target_chains)
            if not positions:
                logger.warning("%s: no interface residues within %.1f A; MPNN-FI is "
                               "equivalent to free redesign here", pdb_path, INTERFACE_CUTOFF)
            fixed = work / "fixed_positions.jsonl"
            _write_fixed_positions(fixed, design_name, binder_chain, target_chains, positions)
            cmd += ["--fixed_positions_jsonl", str(fixed)]

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        fasta = work / "seqs" / f"{design_name}.fa"
        if not fasta.exists():
            raise RuntimeError(
                f"ProteinMPNN wrote no sequences.\nexit={proc.returncode}\n"
                f"stdout: {proc.stdout[-600:]}\nstderr: {proc.stderr[-600:]}"
            )
        seqs = _parse_fasta(fasta)
        if not seqs:
            raise RuntimeError(f"ProteinMPNN FASTA held only the input sequence: {fasta}")
        return seqs
    finally:
        shutil.rmtree(work, ignore_errors=True)
