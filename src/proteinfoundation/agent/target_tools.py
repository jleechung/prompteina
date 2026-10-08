#!/usr/bin/env python3
"""Structure-query tools handed to the expansion agent, one target per instance.

The expansion agent has to choose an epitope per design. Before this module it
could not look at the target at all -- `anthropic_api.py` ran `expand` with no
tools, so every arm's agent picked sites from the research dossier and its own
knowledge. `rfd3_agent` produced 91 distinct hotspot sets across 101 strategies
on `bp_ONCM` that way; our arm could not express a site at all.

**These tools compute; they do not dump coordinates.** An LLM reasons poorly over
raw ATOM records, and a 200-400 KB PDB does not belong in a context window. Every
return value is a small, already-derived answer -- exposure, adjacency,
contiguity -- so the agent spends its judgement on where to bind rather than on
arithmetic it would get wrong.

Every residue identifier in and out is `<chain><number>` in the numbering
`target_input` defines (e.g. `A17` for a target specified as `A1-169`). That is
the same numbering `hotspot_residues` is validated against downstream, so a site
the agent names here cannot fail to resolve later for a numbering reason.

Every method gets the same tools, so a comparison between methods differs only
in the method reference.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np

# Kyte-Doolittle > 1.8 is the conventional hydrophobic set; used only to label
# patches for the agent, never to filter anything downstream.
HYDROPHOBIC = set("AVILMFWCY")
CHARGED = set("DEKR")
POLAR = set("STNQHGP")

# Where PDB and UniProt lookups are cached between runs. Set TARGET_TOOLS_CACHE
# to move it; the default keeps it beside the working directory.
CACHE = Path(os.environ.get("TARGET_TOOLS_CACHE", ".target_tools_cache"))

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def parse_input_spec(spec: str) -> list[tuple[str, int, int]]:
    """'A1-169' or 'A1-232,B5-90' -> [(chain, first, last), ...]."""
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"([A-Za-z0-9])\s*(-?\d+)\s*-\s*(-?\d+)", tok)
        if not m:
            raise ValueError(f"cannot parse target_input segment {tok!r}")
        out.append((m.group(1), int(m.group(2)), int(m.group(3))))
    if not out:
        raise ValueError(f"empty target_input {spec!r}")
    return out


class TargetTools:
    """One instance per target. All methods return JSON-serialisable dicts."""

    def __init__(self, pdb_path: str, input_spec: str, pdb_id: str | None = None):
        import biotite.structure as bs
        import biotite.structure.io.pdb as pdb_io

        self._bs = bs
        self.pdb_path = str(pdb_path)
        self.input_spec = str(input_spec)
        self.pdb_id = (pdb_id or "").strip() or None
        self.segments = parse_input_spec(input_spec)

        arr = pdb_io.PDBFile.read(self.pdb_path).get_structure(model=1)
        arr = arr[bs.filter_amino_acids(arr)]
        # Restrict to what target_input names. A design cannot bind a residue the
        # generator was never given, so a tool must not offer one.
        keep = np.zeros(arr.array_length(), bool)
        for chain, lo, hi in self.segments:
            keep |= ((arr.chain_id == chain) & (arr.res_id >= lo) & (arr.res_id <= hi))
        self.arr = arr[keep]
        if self.arr.array_length() == 0:
            raise ValueError(f"target_input {input_spec!r} selects no atoms in "
                             f"{self.pdb_path}")

        ca = self.arr[self.arr.atom_name == "CA"]
        self.res_ids = [f"{c}{int(r)}" for c, r in zip(ca.chain_id, ca.res_id)]
        self.res_aa = [AA3.get(n, "X") for n in ca.res_name]
        self.ca_xyz = ca.coord
        self._sasa = None

    # -- helpers ---------------------------------------------------------

    def _per_residue_sasa(self) -> np.ndarray:
        if self._sasa is None:
            atom_sasa = self._bs.sasa(self.arr, vdw_radii="Single")
            atom_sasa = np.nan_to_num(atom_sasa)
            idx = {rid: k for k, rid in enumerate(self.res_ids)}
            acc = np.zeros(len(self.res_ids))
            for chain, rid, val in zip(self.arr.chain_id, self.arr.res_id, atom_sasa):
                k = idx.get(f"{chain}{int(rid)}")
                if k is not None:
                    acc[k] += val
            self._sasa = acc
        return self._sasa

    def _kind(self, aa: str) -> str:
        if aa in HYDROPHOBIC:
            return "hydrophobic"
        if aa in CHARGED:
            return "charged"
        return "polar" if aa in POLAR else "other"

    # -- tools -----------------------------------------------------------

    def target_overview(self) -> dict:
        """Chains, residue ranges, sequence and length in target_input numbering."""
        chains = {}
        for rid, aa in zip(self.res_ids, self.res_aa):
            chains.setdefault(rid[0], []).append((int(rid[1:]), aa))
        out = []
        for c, rows in sorted(chains.items()):
            rows.sort()
            out.append({
                "chain": c,
                "first_residue": f"{c}{rows[0][0]}",
                "last_residue": f"{c}{rows[-1][0]}",
                "n_residues": len(rows),
                "sequence": "".join(a for _, a in rows),
            })
        return {"pdb_id": self.pdb_id, "target_input": self.input_spec,
                "n_residues_total": len(self.res_ids), "chains": out,
                "note": "Every residue identifier is <chain><number>, e.g. "
                        f"{self.res_ids[0]}."}

    def secondary_structure(self) -> dict:
        """Per-chain SSE string: a = helix, b = strand, c = coil."""
        sse = self._bs.annotate_sse(self.arr)
        by_chain: dict[str, list[str]] = {}
        for rid, s in zip(self.res_ids, sse):
            by_chain.setdefault(rid[0], []).append(str(s))
        return {"legend": "a=helix b=strand c=coil",
                "chains": [{"chain": c, "first_residue": f"{c}"
                            + self.res_ids[0][1:], "sse": "".join(v)}
                           for c, v in sorted(by_chain.items())]}

    def surface_residues(self, min_sasa: float = 25.0, limit: int = 120) -> dict:
        """Solvent-exposed residues, most exposed first."""
        sasa = self._per_residue_sasa()
        order = np.argsort(-sasa)
        rows = [{"residue": self.res_ids[i], "aa": self.res_aa[i],
                 "sasa": round(float(sasa[i]), 1),
                 "kind": self._kind(self.res_aa[i])}
                for i in order if sasa[i] >= min_sasa][:limit]
        return {"min_sasa": min_sasa, "n_exposed": int((sasa >= min_sasa).sum()),
                "n_returned": len(rows), "residues": rows,
                "note": "Truncated to the most exposed; raise min_sasa or call "
                        "surface_patches for structure rather than a ranking."
                        if len(rows) == limit else ""}

    def neighbors(self, residue: str, radius: float = 10.0) -> dict:
        """Residues whose CA lies within `radius` angstrom of this one's CA."""
        try:
            k = self.res_ids.index(residue.strip())
        except ValueError:
            return {"error": f"{residue!r} is not in target_input "
                             f"{self.input_spec!r}"}
        d = np.linalg.norm(self.ca_xyz - self.ca_xyz[k], axis=1)
        sasa = self._per_residue_sasa()
        hits = [{"residue": self.res_ids[i], "aa": self.res_aa[i],
                 "distance": round(float(d[i]), 1),
                 "sasa": round(float(sasa[i]), 1)}
                for i in np.argsort(d) if i != k and d[i] <= radius]
        return {"center": residue, "radius": radius, "n": len(hits),
                "neighbors": hits[:60]}

    def surface_patches(self, kind: str = "hydrophobic", min_sasa: float = 25.0,
                        link_radius: float = 8.0, min_size: int = 3) -> dict:
        """Contiguous exposed patches of one character, largest first.

        Contiguity is spatial, not sequential: residues are linked when their CAs
        are within `link_radius`, then connected components are taken. A binder
        contacts a patch on the surface, which is rarely a run in the sequence.
        """
        sasa = self._per_residue_sasa()
        sel = [i for i in range(len(self.res_ids))
               if sasa[i] >= min_sasa
               and (kind == "any" or self._kind(self.res_aa[i]) == kind)]
        if not sel:
            return {"kind": kind, "n_patches": 0, "patches": []}
        pos = self.ca_xyz[sel]
        n = len(sel)
        adj = (np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
               <= link_radius)
        seen, comps = set(), []
        for s in range(n):
            if s in seen:
                continue
            stack, comp = [s], []
            seen.add(s)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in np.nonzero(adj[u])[0]:
                    if v not in seen:
                        seen.add(int(v))
                        stack.append(int(v))
            comps.append(comp)
        comps = [c for c in comps if len(c) >= min_size]
        comps.sort(key=len, reverse=True)
        patches = []
        for c in comps[:10]:
            members = [sel[u] for u in c]
            patches.append({
                "n_residues": len(members),
                "total_sasa": round(float(sum(sasa[i] for i in members)), 1),
                "residues": [self.res_ids[i] for i in members],
                "sequence": "".join(self.res_aa[i] for i in members),
            })
        return {"kind": kind, "min_sasa": min_sasa, "link_radius": link_radius,
                "n_patches": len(comps), "patches": patches}

    #: CATH v4.4.0 node names. `cath_topology` is NOT free text -- it is a
    #: T-level node in this file, 1,282 of them under the three classes a binder
    #: can be. Listing them in the method reference would cost ~20k tokens and
    #: bury everything else, so the agent queries the ones under the architecture
    #: it has already chosen.
    # cath-names.txt from https://www.cathdb.info/ ; optional. Without it
    # cath_topologies() returns codes but no names.
    CATH_NAMES = Path(os.environ.get("CATH_NAMES", "cath-names.txt"))

    def cath_topologies(self, architecture: str) -> dict:
        """Every CATH topology under one architecture, with its node code.

        `architecture` is a CATH architecture name exactly as the method
        reference lists it, e.g. "Up-down Bundle". Matching is case-insensitive
        and whitespace-tolerant, but the name must be real: a near-miss returns
        the available architectures rather than guessing, because a silently
        wrong architecture would return topologies the binder cannot adopt.
        """
        if not self.CATH_NAMES.is_file():
            return {"error": f"CATH names file not found at {self.CATH_NAMES}"}
        arch, topo = {}, {}
        for line in self.CATH_NAMES.read_text().splitlines():
            if line.startswith("#"):
                continue
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            node, label = parts[0], parts[2].lstrip(":").strip()
            depth = node.count(".")
            if depth == 1:
                arch[node] = label
            elif depth == 2:
                topo[node] = label
        want = " ".join(architecture.split()).casefold()
        hit = [n for n, l in arch.items()
               if " ".join(l.split()).casefold() == want and n.split(".")[0] in "123"]
        if not hit:
            return {"error": f"no CATH architecture named {architecture!r}",
                    "available": sorted(l for n, l in arch.items()
                                        if n.split(".")[0] in "123")}
        node = hit[0]
        out = [{"code": t, "name": topo[t]}
               for t in sorted(topo, key=lambda x: [int(i) for i in x.split(".")])
               if t.rsplit(".", 1)[0] == node]
        return {"architecture": arch[node], "architecture_code": node,
                "n_topologies": len(out), "topologies": out}

    def known_interfaces(self, contact_radius: float = 5.0) -> dict:
        """Target residues contacted by any other polymer chain in the deposit.

        Fetches the full deposited entry for `pdb_id`. The target file holds
        the isolated target, so the partner chains exist only upstream. Every
        method has the same access to this tool.
        """
        if not self.pdb_id:
            return {"available": False,
                    "reason": "no pdb_id in the target config"}
        try:
            path = _fetch_entry(self.pdb_id)
        except Exception as exc:                                  # noqa: BLE001
            return {"available": False,
                    "reason": f"could not fetch {self.pdb_id}: "
                              f"{type(exc).__name__}"}
        try:
            import biotite.structure as bs
            import biotite.structure.io.pdbx as pdbx
            f = pdbx.CIFFile.read(str(path))
            full = pdbx.get_structure(f, model=1)
            full = full[bs.filter_amino_acids(full)]
        except Exception as exc:                                  # noqa: BLE001
            return {"available": False,
                    "reason": f"could not parse {self.pdb_id}: "
                              f"{type(exc).__name__}"}

        our = set(self.res_aa)
        seq_by_chain = {}
        for cid in np.unique(full.chain_id):
            ca = full[(full.chain_id == cid) & (full.atom_name == "CA")]
            seq_by_chain[str(cid)] = "".join(AA3.get(n, "X") for n in ca.res_name)
        target_seq = self.target_overview()["chains"][0]["sequence"]
        # Identify which deposited chain IS our target, by sequence containment;
        # everything else is a potential partner.
        self_chains = {c for c, s in seq_by_chain.items()
                       if s and (s in target_seq or target_seq in s
                                 or _similar(s, target_seq))}
        partners = [c for c in seq_by_chain if c not in self_chains
                    and len(seq_by_chain[c]) >= 20]
        if not self_chains or not partners:
            return {"available": True, "pdb_id": self.pdb_id,
                    "n_partner_chains": len(partners),
                    "contacts": [],
                    "note": "the deposited entry has no second polymer chain in "
                            "contact with the target" if not partners else
                            "could not identify the target chain in the deposit"}

        tgt_ca = full[(np.isin(full.chain_id, list(self_chains)))
                      & (full.atom_name == "CA")]
        par = full[np.isin(full.chain_id, partners)]
        d = np.linalg.norm(tgt_ca.coord[:, None, :] - par.coord[None, :, :],
                           axis=-1)
        hit = d.min(axis=1) <= contact_radius + 4.0   # CA-to-any-atom
        # Map deposit numbering onto ours by residue number, which is the
        # numbering target_input is expressed in.
        contacts = []
        ours = set(self.res_ids)
        for cid, rid in zip(tgt_ca.chain_id[hit], tgt_ca.res_id[hit]):
            for chain, _, _ in self.segments:
                cand = f"{chain}{int(rid)}"
                if cand in ours and cand not in contacts:
                    contacts.append(cand)
        return {"available": True, "pdb_id": self.pdb_id,
                "partner_chains": partners,
                "n_contacts": len(contacts),
                "contact_residues": sorted(contacts, key=lambda r: int(r[1:])),
                "note": "Residues of the target contacted by another polymer "
                        "chain in the deposited entry, renumbered into "
                        "target_input numbering."}


def _similar(a: str, b: str, frac: float = 0.8) -> bool:
    if not a or not b:
        return False
    n = min(len(a), len(b))
    if n / max(len(a), len(b)) < frac:
        return False
    same = sum(1 for x, y in zip(a[:n], b[:n]) if x == y)
    return same / n >= frac


def _fetch_entry(pdb_id: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{pdb_id.lower()}.cif"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.cif"
    with urllib.request.urlopen(url, timeout=60) as fh:
        data = fh.read()
    dest.write_bytes(data)
    return dest


# ---------------------------------------------------------------------------
# Anthropic tool schemas + dispatch
# ---------------------------------------------------------------------------

SCHEMAS = [
    {"name": "target_overview",
     "description": "Chains, residue ranges, sequence and length of the target, "
                    "in the numbering target_input defines. Call this first to "
                    "fix the numbering every other tool and hotspot_residues use.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "secondary_structure",
     "description": "Per-chain secondary structure string (a=helix, b=strand, "
                    "c=coil), one character per residue in order.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "surface_residues",
     "description": "Solvent-exposed residues with per-residue SASA, most "
                    "exposed first. Use to see what is available to bind.",
     "input_schema": {"type": "object", "properties": {
         "min_sasa": {"type": "number",
                      "description": "minimum per-residue SASA, default 25"}},
         "required": []}},
    {"name": "neighbors",
     "description": "Residues whose CA is within a radius of the given "
                    "residue's CA, nearest first. Use to grow a contiguous site "
                    "around a chosen residue.",
     "input_schema": {"type": "object", "properties": {
         "residue": {"type": "string",
                     "description": "residue id, e.g. A17"},
         "radius": {"type": "number", "description": "angstrom, default 10"}},
         "required": ["residue"]}},
    {"name": "surface_patches",
     "description": "Contiguous exposed surface patches of one character, "
                    "largest first. Contiguity is spatial, not sequential.",
     "input_schema": {"type": "object", "properties": {
         "kind": {"type": "string",
                  "enum": ["hydrophobic", "charged", "polar", "any"],
                  "description": "default hydrophobic"},
         "min_sasa": {"type": "number", "description": "default 25"}},
         "required": []}},
    {"name": "known_interfaces",
     "description": "Target residues contacted by another polymer chain in the "
                    "deposited PDB entry -- i.e. where natural partners bind.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "cath_topologies",
     "description": "Every valid CATH topology name under one architecture. "
                    "`cath_topology` must be one of these -- it is a closed "
                    "vocabulary, not free text. Call this after choosing an "
                    "architecture.",
     "input_schema": {"type": "object", "properties": {
         "architecture": {"type": "string",
                          "description": "CATH architecture name, e.g. "
                                         "'Up-down Bundle' or 'Beta Barrel'"}},
         "required": ["architecture"]}},
]


def dispatch(tools: TargetTools, name: str, args: dict) -> str:
    fn = getattr(tools, name, None)
    if fn is None or name not in {s["name"] for s in SCHEMAS}:
        return json.dumps({"error": f"unknown tool {name!r}"})
    try:
        return json.dumps(fn(**(args or {})), separators=(",", ":"))
    except TypeError as exc:
        return json.dumps({"error": f"bad arguments for {name}: {exc}"})
    except Exception as exc:                                      # noqa: BLE001
        return json.dumps({"error": f"{name} failed: {type(exc).__name__}: {exc}"})
