"""Agent-written monomer design prompts.

Two stages. A research stage searches the literature and writes a factual
dossier about the design objective. An expansion stage reads that dossier and
the case study's reference card, and writes one block per design: structural
bullets, a length band, and the caption that conditions the model.

    export ANTHROPIC_API_KEY=sk-ant-...
    python -m proteinfoundation.agent.monomer_agent \\
        --case-study nanopore --n-designs 24 --out ./specs

The caption is prose because the model was trained on prose, and the expansion
prompt carries the same instruction the training corpus was captioned with.
`--no-caption` conditions on the bullets instead, which is what the binder path
does.

There is no target here, so the research stage has web search and no structure
tools. For binder design see `design_agent.py`.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from proteinfoundation.agent.designs import (
    CAPTION_RULE,
    bullets_to_text,
    clean_caption,
    parse_designs,
    split_label,
    split_length,
)

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
CASE_STUDIES_DIR = HERE / "case_studies"

#: Design objectives with a reference card. `goal` is handed to the expansion
#: stage, `research` to the research stage.
CASE_STUDIES: dict[str, dict] = {
    "nanopore": {
        "card": "nanopore.md",
        "goal": (
            "a single-chain transmembrane beta-barrel nanopore: one protein "
            "chain that folds into a membrane-spanning beta barrel with a "
            "continuous open lumen wide enough to conduct ions and, at the "
            "larger diameters, to pass a single polymer strand"
        ),
        "research": (
            "Single-chain transmembrane beta-barrel proteins and de novo "
            "designed transmembrane beta-barrel nanopores. Report structural "
            "facts only: the relationship between strand number, barrel "
            "diameter and lumen diameter; typical residue counts per strand "
            "count; shear number; the hydrophobic belt and aromatic girdle and "
            "their axial extent; what distinguishes a conducting barrel from a "
            "soluble barrel with a filled interior; loop length at the two "
            "ends; and the measured lumen diameters and conductances reported "
            "for designed barrels."
        ),
    },
}

EXPAND_RULES = """
## Your task

Produce {n} design blocks for this objective:

{goal}

You have a research dossier below. Use it to choose values. The dossier
reports facts; the design decisions are yours.

Follow the output contract in the reference above exactly. Produce nothing but
the design blocks -- no preamble, no commentary, no summary after them.

Before the design blocks, think about the size-and-topology pairing for the
whole set, so that no single design pairs a fold with a residue count that
cannot accommodate it. Do not write that reasoning out; only the blocks.

Span the design space across the {n} designs rather than repeating one answer.
"""

def load_card(case_study: str) -> str:
    path = CASE_STUDIES_DIR / CASE_STUDIES[case_study]["card"]
    if not path.exists():
        raise FileNotFoundError(f"no reference card at {path}")
    return path.read_text()


def research(case_study: str, model_tag: str, max_searches: int = 40) -> str:
    """Stage one: a factual dossier on the design objective."""
    from proteinfoundation.agent.anthropic_api import _run

    system = (PROMPTS / "monomer_research_system.txt").read_text()
    text, _usage, _trace = _run(model_tag, system, CASE_STUDIES[case_study]["research"],
                                max_tokens=16000, use_web_search=True,
                                max_searches=max_searches, effort=None, local_tools=None)
    return text


def expand(dossier: str, case_study: str, n_designs: int, model_tag: str,
           want_caption: bool = True) -> str:
    """Stage two: the dossier and the card become design blocks."""
    from proteinfoundation.agent.anthropic_api import _run

    card = load_card(case_study)
    goal = CASE_STUDIES[case_study]["goal"]
    rules = EXPAND_RULES.format(n=n_designs, goal=goal)
    if want_caption:
        rules += CAPTION_RULE
    user = f"{card}\n\n{rules}\n\n## Research dossier\n\n{dossier}"
    text, _usage, _trace = _run(model_tag, "", user, max_tokens=32000,
                                use_web_search=False, max_searches=0,
                                effort=None, local_tools=None)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-study", default="nanopore", choices=sorted(CASE_STUDIES))
    ap.add_argument("--n-designs", type=int, default=24)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-tag", default="o46", help="key into anthropic_api.MODEL_MAP")
    ap.add_argument("--dossier", type=Path, default=None,
                    help="reuse this dossier instead of researching again")
    ap.add_argument("--no-caption", action="store_true",
                    help="condition on the bullets rather than a prose caption")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    dossier_path = args.dossier or (args.out / f"{args.case_study}.dossier.md")

    if dossier_path.exists():
        logger.info("reusing dossier %s", dossier_path)
        dossier = dossier_path.read_text()
    else:
        logger.info("researching %s", args.case_study)
        dossier = research(args.case_study, args.model_tag)
        dossier_path.write_text(dossier)
        logger.info("wrote %s (%d chars)", dossier_path, len(dossier))

    logger.info("expanding %d designs", args.n_designs)
    raw_path = args.out / f"{args.case_study}.expand.md"
    raw = expand(dossier, args.case_study, args.n_designs, args.model_tag,
                 want_caption=not args.no_caption)
    raw_path.write_text(raw)
    designs = parse_designs(raw)
    if not designs:
        logger.error("no design blocks parsed; see %s", raw_path)
        return 1
    logger.info("parsed %d designs", len(designs))

    rows, uncaptioned = [], 0
    for i, d in enumerate(designs, start=1):
        bullets, lo, hi = split_length(d["bullets"])
        bullets, written = split_label(bullets, "caption")
        if args.no_caption or not written:
            uncaptioned += not args.no_caption
            description = bullets_to_text(bullets)
        else:
            description = clean_caption(written)
        rows.append({"accession": f"{args.case_study}_{i:03d}",
                     "description": description, "nres_min": lo, "nres_max": hi})
    if uncaptioned:
        logger.warning("%d of %d designs carried no caption bullet and fall "
                       "back to their bullets; see %s",
                       uncaptioned, len(designs), raw_path)

    csv_path = args.out / f"{args.case_study}.prompts.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["accession", "description",
                                                "nres_min", "nres_max"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s; pass it as TEXTCOND_CSV to "
                "--config-name text_monomer_generate", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
