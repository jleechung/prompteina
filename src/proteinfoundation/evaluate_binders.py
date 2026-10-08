"""Score generated binders: refold each complex, report ipSAE_min.

Takes a directory of generated binder--target complexes, refolds each one with
ESMFold2-Fast or AlphaFold3, and writes one row per design with ipSAE_min and
the confidence terms the folding model reported.

    python -m proteinfoundation.evaluate_binders \\
        --designs ./inference/02_PDL1 \\
        --out ./scores/02_PDL1.csv \\
        --backend esmfold2-fast \\
        --sequences self mpnn mpnn-fi

Three sequence conditions, as in the paper:

    self     the sequence the generative model co-designed
    mpnn     free ProteinMPNN redesign, best of 8 by ipSAE_min
    mpnn-fi  redesign with the generated interface held fixed, best of 8

The gap between the last two separates an interface the model produced from one
ProteinMPNN rewrote. Each condition is a column; `ipsae_min` is the `self`
score so a run without redesign keeps the same column name.

ipSAE_min is a transform of predicted aligned error across the interface; it is
a proxy for binding, not a measurement of it.

ipSAE is computed by the DunbrackLab reference implementation, vendored at
`community_models/IPSAE/`, which is what the paper's numbers were produced
with. `--ipsae inprocess` switches to the torch reimplementation in
`metrics/ipsae.py`, which avoids a subprocess per design and differs in how it
treats residues with no confident partner.

AlphaFold3 note: run without MSAs and without a template, AF3 returns ipSAE 0
for every design. This script therefore passes the designed structure to AF3 as
a template. ESMFold2 takes sequence only and ignores it.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from proteinfoundation.metrics.binding_score import (
    DEFAULT_DIST_CUTOFF,
    DEFAULT_IMPLEMENTATION,
    DEFAULT_PAE_CUTOFF,
    ipsae_from_fold,
)

logger = logging.getLogger(__name__)

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
}


def _pdb_sequences(pdb_path: Path) -> dict[str, str]:
    """Read one sequence per chain without importing the training data stack."""
    residues: dict[str, list[str]] = {}
    seen: set[tuple[str, str, str]] = set()
    with pdb_path.open() as fh:
        for line in fh:
            record = line[:6].strip()
            if record == "ENDMDL":
                break
            if record not in {"ATOM", "HETATM"} or len(line) < 27:
                continue
            if line[12:16].strip() != "CA" or line[16] not in {" ", "A"}:
                continue
            chain = line[21].strip()
            if not chain:
                continue
            residue = (chain, line[22:26], line[26])
            if residue in seen:
                continue
            seen.add(residue)
            residues.setdefault(chain, []).append(_THREE_TO_ONE.get(line[17:20].strip(), "X"))
    return {chain: "".join(residues[chain]) for chain in sorted(residues)}


def build_backend(name: str, device: str, seed: int, **kwargs):
    """Construct a folding backend by name."""
    if name in ("esmfold2-fast", "esmfold2"):
        from proteinfoundation.metrics.folding.esmfold2 import ESMFold2Backend

        return ESMFold2Backend(model=name, device=device, **kwargs)
    if name == "alphafold3":
        from proteinfoundation.metrics.folding.alphafold3 import AlphaFold3Backend

        return AlphaFold3Backend(seed=seed, **kwargs)
    raise ValueError(f"unknown backend {name!r}")


def _fold_and_score(backend, binder_seq, target_seqs, pdb_path, seed, pae_cutoff,
                    dist_cutoff, implementation) -> dict:
    result = backend.fold(
        binder_seq,
        target_seqs,
        seed=seed,
        template_pdb=str(pdb_path) if backend.REQUIRES_TEMPLATE else None,
    )
    return ipsae_from_fold(
        result, pae_cutoff=pae_cutoff, dist_cutoff=dist_cutoff, implementation=implementation
    )


def score_design(backend, pdb_path: Path, seed: int, pae_cutoff: float,
                 dist_cutoff: float, implementation: str,
                 conditions: list[str], num_seqs: int) -> dict:
    """Refold one design under each sequence condition and score its interface."""
    sequences = _pdb_sequences(pdb_path)
    chain_ids = list(sequences)
    if len(chain_ids) < 2:
        raise ValueError(f"{pdb_path}: only one chain; a binder complex needs at least two")
    target_chains, binder_chain = chain_ids[:-1], chain_ids[-1]
    target_chain = ",".join(target_chains)

    binder_seq = sequences[binder_chain]
    target_seqs = [sequences[c] for c in target_chains]
    row = {
        "design": pdb_path.stem,
        "binder_chain": binder_chain,
        "target_chains": target_chain,
        "binder_len": len(binder_seq),
        "target_len": sum(len(t) for t in target_seqs),
        "binder_seq": binder_seq,
    }

    if "self" in conditions:
        scores = _fold_and_score(backend, binder_seq, target_seqs, pdb_path, seed,
                                 pae_cutoff, dist_cutoff, implementation)
        row.update(scores)

    for condition, fix in (("mpnn", False), ("mpnn-fi", True)):
        if condition not in conditions:
            continue
        from proteinfoundation.metrics.mpnn import redesign

        seqs = redesign(str(pdb_path), binder_chain, target_chains,
                        fix_interface=fix, num_seqs=num_seqs, seed=seed or 42)
        best, best_seq = None, None
        for seq in seqs:
            s = _fold_and_score(backend, seq, target_seqs, pdb_path, seed,
                                pae_cutoff, dist_cutoff, implementation)
            if best is None or s["ipsae_min"] > best["ipsae_min"]:
                best, best_seq = s, seq
        prefix = condition.replace("-", "_")
        row[f"{prefix}_ipsae_min"] = best["ipsae_min"]
        row[f"{prefix}_n_seqs"] = len(seqs)
        row[f"{prefix}_best_seq"] = best_seq
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", required=True, type=Path, help="directory of binder--target PDBs")
    ap.add_argument("--out", required=True, type=Path, help="output CSV")
    ap.add_argument(
        "--backend",
        default="esmfold2-fast",
        choices=["esmfold2-fast", "esmfold2", "alphafold3"],
        help="folding model; esmfold2-fast is what the paper reports",
    )
    ap.add_argument("--glob", default="*.pdb", help="pattern under --designs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pae-cutoff", type=float, default=DEFAULT_PAE_CUTOFF)
    ap.add_argument("--dist-cutoff", type=float, default=DEFAULT_DIST_CUTOFF,
                    help="distance gate; reference implementation only")
    ap.add_argument("--ipsae", default=DEFAULT_IMPLEMENTATION, choices=["reference", "inprocess"],
                    help="reference is the vendored DunbrackLab script, what the paper reports")
    ap.add_argument("--sequences", nargs="+", default=["self"],
                    choices=["self", "mpnn", "mpnn-fi"],
                    help="sequence conditions to score; mpnn and mpnn-fi each cost "
                         "one ProteinMPNN run plus --num-seqs folds per design")
    ap.add_argument("--num-seqs", type=int, default=8,
                    help="ProteinMPNN sequences per design, best taken by ipSAE_min")
    ap.add_argument("--limit", type=int, default=None, help="score at most this many designs")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    designs = sorted(args.designs.rglob(args.glob))
    if args.limit:
        designs = designs[: args.limit]
    if not designs:
        logger.error("no designs matched %s under %s", args.glob, args.designs)
        return 1
    logger.info("scoring %d designs with %s", len(designs), args.backend)

    backend = build_backend(args.backend, device=args.device, seed=args.seed)
    rows, failures = [], 0
    try:
        for i, pdb in enumerate(designs, 1):
            try:
                rows.append(
                    score_design(
                        backend, pdb, args.seed, args.pae_cutoff,
                        args.dist_cutoff, args.ipsae, args.sequences, args.num_seqs,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad design must not end the run
                failures += 1
                logger.warning("%s failed: %s: %s", pdb.name, type(exc).__name__, exc)
            if i % 25 == 0:
                logger.info("  %d/%d", i, len(designs))
    finally:
        backend.close()

    if not rows:
        logger.error("every design failed; nothing written")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    key = "ipsae_min" if "self" in args.sequences else f"{args.sequences[0].replace('-', '_')}_ipsae_min"
    scored = [r[key] for r in rows if key in r]
    scored.sort(reverse=True)
    logger.info(
        "wrote %d rows to %s (%d failed); %s best %.3f, top-8 mean %.3f, mean %.3f",
        len(rows),
        args.out,
        failures,
        key,
        scored[0],
        sum(scored[:8]) / min(8, len(scored)),
        sum(scored) / len(scored),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
