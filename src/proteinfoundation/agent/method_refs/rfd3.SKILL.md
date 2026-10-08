# RFdiffusion3 — what the method can express

> **Sources.** RFdiffusion3 [input documentation](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md)
> and [README](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/README.md)
> (BSD 3-Clause); Butcher et al., *De novo Design of All-atom Biomolecular
> Interactions with RFdiffusion3*, bioRxiv 2025.09.18.676967.

## What the model is

A structure diffusion model that generates a protein backbone in the context of
a fixed input structure. For binder design the input structure is the target and
the generated chain is the binder.

It takes one `InputSpecification` per design — a JSON object of constraints
describing the binder you want and where on the target it should sit.

## What you can tweak — the complete set

One specification per design. Every channel below is set independently in each
one; only the target structure is fixed across a target's designs.

| channel | field | what it controls |
|---|---|---|
| binder length | `contig` | residue-count range of the generated chain |
| target epitope | `select_hotspots` | which target residues the binder should contact |
| placement | `infer_ori_strategy` | `hotspots` orients the binder on the chosen site; `com` on the target's center of mass |
| loop content | `is_non_loopy` | biases toward or against loop-rich backbones |
| burial | `select_buried`, `select_partially_buried`, `select_exposed` | how exposed each region of the binder should be |
| polar contacts | `select_hbond_donor`, `select_hbond_acceptor` | specific target atoms the binder should hydrogen-bond |

### Binder length

The first element of `contig`: `"<min>-<max>,/0,<target segments>"`. The
generated chain gets a length drawn uniformly from that range.

### Target epitope

`select_hotspots` maps target residues to atom names:

```json
"select_hotspots": {"A37": "CD1,CG1", "A39": "CZ,OH", "A49": "CD,OE1"}
```

Keys are `<chain><residue number>` in the input structure's own author numbering
— the numbering `target_overview()` reports. Typical sites are 2–6 residues.

From the developers: *"Hotspots will typically be at most 4.5 Å to any heavy atom
in the designed structure."* Residues that cannot all sit within that distance of
one binder ask for something the model cannot produce.

Set `infer_ori_strategy: "hotspots"` whenever you name hotspots. Use `"com"` only
when you name none, and then direct the binder with `select_buried` instead.

**Atom names must match the actual residue type at that position.** Look the
residue up in the target sequence and use its own atoms — ILE has `CD1,CG1`, TYR
has `CZ,OH`. Asking for TRP atoms at an ILE position selects nothing.

| Residue | atoms | Residue | atoms |
|---|---|---|---|
| ALA | CB,CA | LEU | CD1,CD2 |
| ARG | CZ,NH1 | LYS | NZ,CE |
| ASN | CG,OD1 | MET | SD,CE |
| ASP | CG,OD1 | PHE | CZ,CE1 |
| CYS | CB,SG | PRO | CG,CD |
| GLN | CD,OE1 | SER | CB,OG |
| GLU | CD,OE1 | THR | CB,OG1 |
| GLY | CA,C | TRP | CZ2,CH2 |
| HIS | CE1,NE2 | TYR | CZ,OH |
| ILE | CD1,CG1 | VAL | CG1,CG2 |

### Hydrogen-bond selections

Same dictionary form, but each atom must be a real donor or acceptor:

| Residue | donor | acceptor |
|---|---|---|
| ASN | ND2 | OD1 |
| ASP | — | OD1, OD2 |
| GLN | NE2 | OE1 |
| GLU | — | OE1, OE2 |
| HIS | ND1, NE2 | ND1, NE2 |
| LYS | NZ | — |
| ARG | NH1, NH2, NE | — |
| SER | OG | OG |
| THR | OG1 | OG1 |
| TYR | OH | OH |
| TRP | NE1 | — |
| backbone | N | O |

Use these sparingly — one to three contacts, only where there is real evidence
for a specific polar interaction. Over-constraining them reduces designability.

### Burial selections

`select_buried` / `select_partially_buried` / `select_exposed` take positions on
the **binder**, 1-indexed from its N-terminus, as a string like `"1-15,25-30"`.
The binder does not exist yet, so these express approximate intent — "the first
fifteen residues form the interface core". The three are mutually exclusive:
assigning one position to two of them is an error.

## The field set available here — use nothing outside this table

These are the fields you may set. **Do not invent keys**: RFD3's parser raises on
an unknown key rather than ignoring it, so a typo or a made-up field fails the
whole run. Hydrogen-bond donors and acceptors are separate fields:
`select_hbond_donor` and `select_hbond_acceptor`.

This table is a **subset of RFD3's full `InputSpecification`**, restricted to
what this harness supports. The omissions and why each is out of scope are listed
in *Fields deliberately not offered here* below. Nothing in that list would help
on this task; if you find yourself wanting one, say so in your reasoning rather
than writing it into a spec.

| Field | Type | What it does |
|---|---|---|
| `dialect` | `int` | Always `2`. |
| `input` | `str` | Path to the target structure. |
| `contig` | contig string | Binder length range and which target residues are kept. Grammar below. |
| `length` | `str` | Total design length, `"min-max"` or an int. Redundant with `contig` here. |
| `select_hotspots` | dict | Target residues the binder should contact. |
| `infer_ori_strategy` | `str` | `"hotspots"` or `"com"`. |
| `is_non_loopy` | `bool` | `true` = fewer loops, more regular secondary structure. |
| `select_buried` / `select_partially_buried` / `select_exposed` | str or dict | Burial of binder regions. Mutually exclusive. |
| `select_hbond_donor` / `select_hbond_acceptor` | dict | Target atoms to hydrogen-bond. Dictionary form only. |
| `select_fixed_atoms` | str or dict | Atoms whose coordinates are held fixed. |
| `select_unfixed_sequence` | str or dict | Where sequence may change. |
| `redesign_motif_sidechains` | `bool` | Fixed backbone, redesigned sidechains for input regions. |
| `plddt_enhanced` | `bool` | Default `true`. |
| `ligand` | `str` | Ligand by chemical component name or index. |

Dictionary-valued selections also accept shorthand atom sets: `ALL` (every atom),
`TIP` (the residue's tip atoms), `BKBN` (equivalent to `N,CA,C,O`).

## Fields deliberately not offered here

RFD3's `InputSpecification` has four more fields. Each is omitted for a reason,
not by oversight, and **writing any of them will fail the run**.

| field | why it is not offered |
|---|---|
| `partial_t` | Partial diffusion: re-noise an existing structure by N Å and re-denoise. It needs the structure being refined as `input`, i.e. a different file per design. The runner requires one shared target PDB across a spec file and aborts otherwise, and nothing carries a round's outputs forward as the next round's inputs. There is also no parent design to refine on the first round of any campaign. |
| `symmetry` | Symmetric assemblies. You are designing one monomeric binder against a fixed target; a symmetric binder is a different task, and the scoring pipeline treats the binder as a single chain. |
| `unindex` | Unindexed motif placement, for motif scaffolding. There is no motif here — the binder is generated de novo and only the target is fixed. |
| `ori_token` | Raw `[x,y,z]` for the ORI token. `infer_ori_strategy: hotspots` derives that point for you — the hotspot centroid pushed 10 Å along the core→hotspot vector — which is the same control expressed in the epitope terms you are working in. Setting it directly needs a coordinate in the target's frame, and your tools report residues, neighbors and SASA, never coordinates. |

The first three would each change what the task is rather than how well you do
it. If one of them looks genuinely necessary for a target, say so in your
reasoning — that is a signal worth having — but do not put it in a spec.

## Contig grammar

From the developers' documentation:

- Pieces are separated by commas.
- A residue range is written with a dash: `40-60`.
- `/0` is a chain break — no peptide bond across it.
- A range **with** a chain letter comes from the input structure.
- A range **without** a chain letter is generated, at a length drawn uniformly
  from that range.

Worked example — `"A40-60,70,A120-170,/0,B3-45"`:

- `A40-60` — take residues 40–60 of input chain A
- `70` — generate exactly 70 residues, bonded to A60
- `A120-170` — bond back to A120, then take A120–A170 from input
- `/0` — chain break
- `B3-45` — take B3–B45 from input

### One `/0` per chain boundary — the rule that decides comma vs `/0`

A comma is a **peptide bond**. `/0` is a **chain break**. So the separator is
chosen by what is on either side of it, never by position in the string:

- Between two segments of the **same** chain — comma. `A17-29,A41-131` fuses
  A29 to A41 and drops A30–A40, which is a deliberate truncation.
- Between segments of **different** chains — `/0`. A comma there **welds two
  separate chains into one**, and RFD3 will build it: no error, no warning, a
  structure that folds as a chimera.

**Every chain in the target needs its own `/0`.** A target with *n* chains has
*n* `/0`s in the contig — one before each — regardless of how many comma-joined
segments any single chain contributes.

Single-chain target `A1-232`, binder 90–120 residues:

```
"90-120,/0,A1-232"
```

Two-chain target — a homodimer whose chain A is kept as two segments:

```
"76-82,/0,A17-29,A41-131,/0,B19-127"
```

Read it as: generate 76–82 · break · chain A (two segments, comma-joined,
A30–A40 dropped) · **break** · chain B. Written `...A41-131,B19-127` instead,
A131 is peptide-bonded to B19 and the dimer becomes one 213-residue chain.

Three-chain target `A12-157`, `B12-157`, `C12-157`, binder 50–55:

```
"50-55,/0,A12-157,/0,B12-157,/0,C12-157"
```

Before emitting a contig, check it: **split on `/0`, and no piece may name more
than one chain letter.** Each piece is one physical chain, so two letters inside
a piece means those chains are fused.

## Output contract

Output one JSON array containing every design. Each element:

```json
{
    "strategy_name": "compact_three_helix_hydrophobic_core",
    "reasoning": "One or two sentences on why this approach suits this target.",
    "spec": {
        "dialect": 2,
        "input": "<the target path from the target config>",
        "contig": "90-120,/0,A1-232",
        "is_non_loopy": true,
        "infer_ori_strategy": "hotspots",
        "select_hotspots": {"A37": "CD1,CG1", "A39": "CZ,OH"}
    }
}
```

Rules:

- `strategy_name` is unique across the array and describes the approach.
- `spec` contains only fields from the closed table above.
- Omit optional fields you are not using. Do not emit empty values.
- The generated range in `contig` must match the length you stated for that
  design in the scaffold plan.
- The target segments in `contig` and the path in `input` come from the target
  config and are the same for every design.
- Every contig must satisfy `#distinct chain letters == #/0`. A contig with
  more chain letters than breaks fuses target chains and is invalid.
