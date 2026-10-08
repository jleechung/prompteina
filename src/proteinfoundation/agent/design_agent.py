"""Agent-written binder design specifications.

Two stages, as in the paper. A research stage builds a factual dossier about
the target, using web and literature search plus structure-query tools over the
target itself. An expansion stage reads that dossier together with a method
reference and writes design specifications in whatever form that method
accepts.

    export ANTHROPIC_API_KEY=sk-ant-...
    python -m proteinfoundation.agent.design_agent \\
        --target 02_PDL1 --method prompteina --n-designs 200 --out ./specs

The dossier is written once per target and reused across methods, so every
method sees byte-identical research and the only input that differs between
them is the method reference, which lists the properties that method accepts.

Method references are in `method_refs/`. Three ship: `prompteina` (natural
language), `rfd3` (atom-level contigs) and `boltzgen` (residue-level
constraints). The latter two are here so the comparison in the paper can be
reproduced, not because this repository runs those models.

Each reference states its own output contract and the expansion stage follows
it. Prompteina's is one `### Design N` header per design followed by
`- Label: value` bullets, read by `designs.py`, which the monomer driver shares.
The other two ask for a JSON array, which is written out as given.

The agent decides in key-value bullets and then writes each block as a prose
caption, in the same call. The caption is the prompt the model receives, since
the model was trained on captions; `--no-caption` passes the bullets instead.

This needs no cluster and no scheduler. Its only external dependency is the
Anthropic API; the key comes from ANTHROPIC_API_KEY, or from the file named by
ANTHROPIC_KEY_FILE. PDB and UniProt lookups are cached under
TARGET_TOOLS_CACHE.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
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
METHOD_REFS = HERE / "method_refs"

#: Methods that have a reference document, i.e. what the agent can write for.
METHODS = sorted(p.name.split(".")[0] for p in METHOD_REFS.glob("*.SKILL.md"))

#: Specifications per expansion call. One call asked for 200 repeats itself.
BATCH = 10


def load_method_reference(method: str) -> str:
    path = METHOD_REFS / f"{method}.SKILL.md"
    if not path.exists():
        raise FileNotFoundError(f"no reference for {method!r}; have {METHODS}")
    return path.read_text()


def research(target_pdb: str, input_spec: str, model_tag: str,
             pdb_id: str | None = None, max_searches: int = 40) -> str:
    """Stage one: build the dossier.

    The agent gets web search plus the structure-query tools bound to this
    target. The tools return target properties rather than raw coordinates,
    which do not fit in a context window.
    """
    from proteinfoundation.agent.anthropic_api import _run
    from proteinfoundation.agent.target_tools import TargetTools

    tools = TargetTools(target_pdb, input_spec, pdb_id)
    system = (PROMPTS / "research_system.txt").read_text()
    user = (f"Target structure: {target_pdb}\nChains and ranges: {input_spec}\n"
            f"{'PDB ID: ' + pdb_id if pdb_id else ''}\n\n"
            "Build the dossier. Use the structure tools for anything about this "
            "structure, and search for anything about its biology.")
    text, _usage, _trace = _run(model_tag, system, user, max_tokens=16000,
                                use_web_search=True, max_searches=max_searches,
                                effort=None, local_tools=tools)
    return text


def _extract_json(text: str) -> list[dict]:
    """Pull a specification list out of a model response.

    This is the path for the references that ask for a JSON array, which are
    RFdiffusion3 and BoltzGen. The response may be a bare array, or an array
    inside a fenced block with prose around it. Anything else is an error
    rather than a silent empty batch, because an empty batch would loop
    forever.
    """
    for candidate in (text, *re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for value in parsed.values():
                if isinstance(value, list):
                    return value
            return [parsed]
    raise ValueError(f"no JSON array in the agent response:\n{text[:400]}")


def _expand_once(dossier: str, reference: str, want: int,
                 binder_length: tuple[int, int], model_tag: str,
                 want_caption: bool = True) -> str:
    """One expansion call. Returns the raw response."""
    from proteinfoundation.agent.anthropic_api import _run

    system = (PROMPTS / "design_system.txt").read_text()
    user = (
        f"--- RESEARCH DOSSIER ---\n{dossier}\n\n"
        f"--- METHOD REFERENCE ---\n{reference}\n\n"
        f"Produce {want} designs, following the output contract in the method "
        f"reference exactly. Write nothing but the designs -- no preamble, no "
        f"commentary, no summary after them.\n"
        f"Every design must have a binder length within "
        f"{binder_length[0]}-{binder_length[1]} residues.\n"
        f"The designs must be diverse: vary the scaffold, the binder size "
        f"within the band, and the epitope. Do not emit the same design twice.\n"
        f"Express every design only through channels the method reference "
        f"defines, using only values from its closed sets."
        + (CAPTION_RULE if want_caption else "")
    )
    text, _usage, _trace = _run(model_tag, system, user, max_tokens=16000,
                                use_web_search=False, max_searches=0,
                                effort=None, local_tools=None)
    return text


def expand(dossier: str, method: str, n_designs: int,
           binder_length: tuple[int, int], model_tag: str,
           batch: int = BATCH, want_caption: bool = True) -> list[dict]:
    """Stage two: turn the dossier into design specifications.

    Prompteina's reference asks for `### Design N` plus `- Label: value`
    bullets, which become one row each. The block's caption is the prompt;
    `nres_min`, `nres_max` and `hotspot_residues` come out as their own columns,
    because the generation config reads them as columns rather than as prose.

    The other references ask for JSON, which is passed through as parsed, and
    take no caption.
    """
    reference = load_method_reference(method)
    caption = want_caption and method == "prompteina"
    specs: list[dict] = []
    while len(specs) < n_designs:
        want = min(batch, n_designs - len(specs))
        text = _expand_once(dossier, reference, want, binder_length, model_tag,
                            want_caption=caption)
        if method == "prompteina":
            got = [_design_to_row(d, caption) for d in parse_designs(text)]
        else:
            got = _extract_json(text)
        if not got:
            raise RuntimeError(
                f"agent returned no designs for {method}; response began:\n"
                f"{text[:400]}"
            )
        specs.extend(got)
        logger.info("  %d/%d designs", len(specs), n_designs)
    return specs[:n_designs]


def _design_to_row(design: dict, want_caption: bool = True) -> dict:
    """One parsed design block becomes one CSV row.

    A block that carries no caption falls back to its bullets, so a dropped
    caption costs one design its prose rather than failing the batch.
    """
    bullets, lo, hi = split_length(design["bullets"])
    bullets, hotspots = split_label(bullets, "hotspot_residues")
    bullets, written = split_label(bullets, "caption")
    captioned = bool(want_caption and written)
    return {
        "description": clean_caption(written) if captioned else bullets_to_text(bullets),
        "nres_min": lo,
        "nres_max": hi,
        "hotspot_residues": hotspots,
        "captioned": captioned,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True,
                    help="target name in the targets config")
    ap.add_argument("--targets-config", type=Path,
                    default=Path("configs/targets/targets_dict.yaml"))
    ap.add_argument("--method", default="prompteina", choices=METHODS)
    ap.add_argument("--n-designs", type=int, default=200)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-tag", default="o46",
                    help="key into anthropic_api.MODEL_MAP")
    ap.add_argument("--dossier", type=Path, default=None,
                    help="reuse this dossier instead of researching again")
    ap.add_argument("--no-caption", action="store_true",
                    help="condition on the bullets rather than a prose caption")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import yaml

    cfg = yaml.safe_load(args.targets_config.read_text())
    cfg = cfg.get("target_dict_cfg", cfg)
    if args.target not in cfg:
        raise SystemExit(f"{args.target!r} is not in {args.targets_config}; "
                         f"have {sorted(cfg)}")
    entry = cfg[args.target]
    target_pdb = entry["target_path"]
    input_spec = entry.get("target_input", "A")
    pdb_id = entry.get("pdb_id")
    binder_length = tuple(entry.get("binder_length", [60, 150]))

    args.out.mkdir(parents=True, exist_ok=True)
    dossier_path = args.dossier or (args.out / f"{args.target}.dossier.md")

    if dossier_path.exists():
        logger.info("reusing dossier %s", dossier_path)
        dossier = dossier_path.read_text()
    else:
        logger.info("researching %s (%s, %s)", args.target, target_pdb, input_spec)
        dossier = research(target_pdb, input_spec, args.model_tag, pdb_id)
        dossier_path.write_text(dossier)
        logger.info("wrote %s (%d chars)", dossier_path, len(dossier))

    logger.info("expanding %d designs for %s", args.n_designs, args.method)
    specs = expand(dossier, args.method, args.n_designs, binder_length,
                   args.model_tag, want_caption=not args.no_caption)

    out = args.out / f"{args.target}.{args.method}.specs.json"
    out.write_text(json.dumps(specs, indent=1))
    logger.info("wrote %d designs to %s", len(specs), out)

    if args.method != "prompteina":
        # The other references define their own format for their own runner,
        # so the JSON is the deliverable and there is no CSV to write.
        return 0

    # Prompteina reads a CSV with a `description` column; `nres_min`,
    # `nres_max` and `hotspot_residues` are optional columns.
    csv_path = args.out / f"{args.target}.prompts.csv"
    fields = ["accession", "description", "nres_min", "nres_max", "hotspot_residues"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, spec in enumerate(specs, start=1):
            row = {k: v for k, v in spec.items() if k in fields}
            writer.writerow({"accession": f"{args.target}_{i:03d}", **row})

    missing = sum(1 for spec in specs if not spec.get("hotspot_residues"))
    if missing:
        logger.warning("%d of %d designs name no hotspot residues", missing, len(specs))
    uncaptioned = sum(1 for spec in specs if not spec.get("captioned"))
    if uncaptioned and not args.no_caption:
        logger.warning("%d of %d designs carried no caption bullet and fall back "
                       "to their bullets", uncaptioned, len(specs))
    logger.info("wrote %s; pass it as TEXTCOND_CSV to "
                "--config-name text_binder_generate", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
