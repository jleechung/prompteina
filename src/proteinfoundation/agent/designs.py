"""Parsing the design blocks an agent emits.

`method_refs/prompteina.SKILL.md` and `case_studies/nanopore.md` state the same
output contract: one `### Design N` header per design, followed by
`- Label: value` bullets. This module reads that format for both the binder and
the monomer driver.

Each block is then converted to a prose caption, in the same call that writes
the bullets, under CAPTION_RULE below. The caption is what reaches the model,
which was trained on captions. The bullets are the agent's working format.
"""

from __future__ import annotations

import re

DESIGN_RE = re.compile(r"^#{1,4}\s*Design\s+(\d+)", re.I)
BULLET_RE = re.compile(r"^\s*[-*]\s*([A-Za-z][A-Za-z0-9 _/'()→>-]*?)\s*:\s*(.+?)\s*$")

#: Bullet values meaning "absent". Dropped rather than passed on: the model
#: cannot condition on negation, as no training caption states that a property
#: is absent, so `- Strands: 0` is noise.
ABSENT = {"0", "none", "None", "N/A", "n/a"}

#: Labels that carry length rather than caption text.
LENGTH_LABELS = {"nres_min", "nres_max"}

#: Appended to an expansion prompt to get the prose caption in the same call.
#: The wording follows the instruction the training corpus was captioned with,
#: so the captions land on the distribution the model was trained on.
CAPTION_RULE = """
End each design block with one more bullet, `- Caption:`, on a single line. It
is a factual protein caption written from that block's own bullets. Only state
what those bullets say. Do not speculate, infer, or elaborate beyond them, and
do not name any protein. Write it as prose rather than as a list, and leave out
`nres_min`, `nres_max` and `hotspot_residues`, which are passed to the model
separately.
"""


def parse_designs(raw: str) -> list[dict]:
    """Split `### Design N` blocks into ordered (label, value) bullets.

    Falls back to blank-line-delimited blocks when the output carries no
    headers. That fallback is not hypothetical: a reference card without an
    output contract produces bare bullet blocks, and a header-only parser then
    finds nothing at all.
    """
    if not any(DESIGN_RE.match(ln) for ln in raw.splitlines()):
        return parse_headerless(raw)
    designs, cur = [], None
    for line in raw.splitlines():
        m = DESIGN_RE.match(line)
        if m:
            if cur is not None:
                designs.append(cur)
            cur = {"index": int(m.group(1)), "bullets": []}
            continue
        if cur is None:
            continue
        b = BULLET_RE.match(line)
        if not b:
            continue
        label, value = b.group(1).strip(), b.group(2).strip()
        if value in ABSENT:
            continue
        cur["bullets"].append((label, value))
    if cur is not None:
        designs.append(cur)
    return [d for d in designs if d["bullets"]]


def parse_headerless(raw: str) -> list[dict]:
    """Blank-line-delimited fallback, for output with no `### Design N` headers."""
    designs = []
    for block in re.split(r"\n\s*\n", raw):
        bullets = []
        for line in block.splitlines():
            b = BULLET_RE.match(line)
            if not b:
                continue
            label, value = b.group(1).strip(), b.group(2).strip()
            if value in ABSENT:
                continue
            bullets.append((label, value))
        if bullets:
            designs.append({"index": len(designs) + 1, "bullets": bullets})
    return designs


def split_label(bullets: list[tuple[str, str]], label: str) -> tuple[list, str | None]:
    """Remove one label from the bullets and return its value separately."""
    want = label.lower().replace(" ", "_")
    kept, found = [], None
    for k, v in bullets:
        if k.lower().replace(" ", "_") == want:
            found = v
        else:
            kept.append((k, v))
    return kept, found


def split_length(bullets: list[tuple[str, str]]) -> tuple[list, int | None, int | None]:
    """Pull `nres_min` and `nres_max` out; they are length, not caption text."""
    lo = hi = None
    kept = []
    for label, value in bullets:
        key = label.lower().replace(" ", "_")
        if key == "nres_min":
            lo = int(re.sub(r"\D", "", value) or 0) or None
        elif key == "nres_max":
            hi = int(re.sub(r"\D", "", value) or 0) or None
        else:
            kept.append((label, value))
    return kept, lo, hi


def bullets_to_text(bullets: list[tuple[str, str]]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in bullets)


def clean_caption(text: str) -> str:
    """Strip a stray code fence or `Caption:` label off a generated caption."""
    text = re.sub(r"^```[a-z]*\n|\n```$", "", text.strip()).strip()
    text = re.sub(r"^Caption:\s*", "", text).strip()
    return " ".join(text.split())
