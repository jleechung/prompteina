# BoltzGen — what the method can express

> **Sources.** [BoltzGen YAML specification](https://github.com/HannesStark/boltzgen/blob/main/example/README.md)
> (MIT), checked against `boltzgen 0.3.2`'s `boltzgen/data/parse/schema.py`.

## What the model is

A structure generator that designs a binder in the context of a fixed target,
conditioned by a YAML specification. The binder is declared as its own `protein`
entity; the target is supplied as a `file` entity pointing at a structure on
disk. One YAML document per design.

## What you can tweak — the complete set

One specification per design. Every channel below is set independently in each
one; only the target structure is fixed across a target's designs.

| channel | field | what it controls |
|---|---|---|
| binder length | `protein.sequence` | residue-count range of the binder |
| binder topology | `protein.secondary_structure` | which binder positions are helix, sheet or loop |
| target epitope | `file` → `binding_types` → `chain.binding` | which target residues the binder should contact |
| epitope exclusion | `file` → `binding_types` → `chain.not_binding` | which target residues it must avoid |
| residue identity | `protein.residue_constraints` | which amino acids are allowed at given binder positions |
| target neighborhood | `file.include_proximity` | pull in target residues within a radius of a target chain or region |

### Binder length

`protein.sequence` as a range: `70..73` designs between 70 and 73 residues. A
bare integer (`18`) fixes the length exactly. Letters (`AAAA`) fix the actual
amino acids, and forms may be mixed (`3..5C6C3`).

### Binder topology

`protein.secondary_structure` assigns regions over the binder's own 1-based
positions:

```yaml
secondary_structure:
  helix: 1..25,39..64
  loop: 26..38
```

### Target epitope

Expressed on the **file** entity, in the positional indexing described next:

```yaml
binding_types:
- chain: {id: A, binding: "37,39,49,98", not_binding: "2..6,79..83"}
```

Both the positive and the negative site live inside the same `chain` block:
`binding` for residues the binder should contact, `not_binding` for residues it
should avoid. That block, nested under `binding_types`, is where the whole
epitope is expressed.

## Residue indexing — read this before writing any index

From the developers' schema doc:

> **All residue indices are specified starting at 1**, using the canonical mmCIF
> `label_asym_id`, not the `auth_asym_id` author residue index.

This is a **1-based positional index into the chain**, not the author numbering
printed in a structure file. The two coincide only when the author numbering
starts at 1 with no gaps — often it does not. If `target_overview()` reports
chain A starting at author residue 282, then author 282 is position 1, author 283
is position 2, and so on.

Getting this wrong does not error. It silently aims the design at different
residues than you intended.

## Every index must lie inside the chain you name

A `binding`, `not_binding`, `res_index`, `design` or `secondary_structure` range
is resolved against the real chain length, and an over-run is a hard error:
`ValueError: Specified end 136..140 is higher than the length of the chain`.

Take chain lengths from `target_overview()` and stay inside them. The binder's own
ranges must likewise stay inside the length you declared in `protein.sequence`.

## THE TARGET IS FIXED — `include` must list EVERY target chain

**`include` selects which target chains enter the construct your binder is
designed against. It is not an epitope control.** Dropping a chain from
`include` means the binder is designed against a different molecule, and a
binder designed against one protomer of an oligomer is not a binder against the
target. Every design of a target must carry the same, complete construct.

For TNF-alpha that is **three** target chains (homotrimer); for IL-17A, **two**
(homodimer). List all of them in `include` on every design, always.

**To aim at one chain, restrict the EPITOPE, not the construct.** Put the
hotspots you want on that chain in `binding`, and leave `include` complete.
Optionally add the other chains to `not_binding`. That is a real design
strategy and it keeps the target intact.

```yaml
# WRONG -- designs against a monomer, and the design is discarded
include: [{chain: {id: A}}]
# RIGHT -- whole trimer present, contacts steered onto chain A
include: [{chain: {id: A}}, {chain: {id: B}}, {chain: {id: C}}]
binding_types:
  - chain: {id: A, binding: "31,32,35,65,67"}
  - chain: {id: B, not_binding: "all"}
```

Omitting a target chain does not raise. Generation exits clean and the design
is simply built against an incomplete target, so count the chains in the output
and reject any design that is short.

## Every chain id must exist in the target structure

Under the `file` entity, `include`, `exclude`, `binding_types` and
`include_proximity` are all resolved against the structure on disk before
generation starts. A chain id that is not in it is a hard error that voids the
whole specification: `ValueError: Specified chain id Z not in file`.

The designed `protein` entity carries its own id — `id: Z` is the usual choice —
and that id lives only inside your specification. Chain ids under the `file`
entity are a separate space: they name chains in the structure on disk, so take
them from `target_overview()`.

`include_proximity` widens the target selection around a target chain or region,
by radius. The epitope itself is already expressed by `binding_types`, so
`include_proximity` is optional.

## The key set is CLOSED — use nothing outside this list

These are the keys the parser accepts. It reads the whole file first and rejects
it outright if any other key appears, so every key you write should come from
this list.

Three keys BoltzGen accepts are **not** here, each out of scope for this task
rather than overlooked: `templates` and `affinity` are structure-prediction and
scoring inputs rather than controls on the binder you are designing, and
`modifications` declares non-standard residues, which the folding and scoring
pipeline downstream of you cannot handle.

```
entities  protein  dna  rna  id  sequence  ligand  ccd  secondary_structure
file  path  msa  include  chain  include_proximity  res_index  radius
binding_types  binding  not_binding  structure_groups  group  visibility
design  loop  helix  sheet  design_insertions  insertion  num_residues
fuse  exclude  smiles  cyclic  bonds  bond  atom1  atom2  bondtype
structure_group  constraints  total_len  min  max  reset_res_index
not_design  leaving_atoms  atom  use_assembly  symmetric_group
residue_constraints  position  allowed  disallowed
```

### Top-level structure

```yaml
entities: [...]     # molecular components
constraints: [...]  # optional structural constraints
```

### Protein entity

```yaml
- protein:
    id: <string>                  # chain identifier for the binder
    sequence: <sequence_spec>     # its length, as above
    secondary_structure: {...}    # optional
    residue_constraints: [...]    # optional
    cyclic: <bool>                # optional, default false
```

`residue_constraints` restricts amino acid identity at binder positions:

```yaml
residue_constraints:
- allowed: ADEFGHIKLMNQRSTVWY
  position: 26..38
```

### File entity

```yaml
- file:
    path: <string>              # the target structure
    include: [...]              # chains/regions to keep
    exclude: [...]              # chains/regions to drop
    include_proximity: [...]    # target residues within a radius; takes radius:
    binding_types: [...]        # the epitope, on target chains
    design: [...]               # designable regions
    reset_res_index: [...]
    fuse: <string>
    msa: <bool>
```

Nested `chain` blocks under `include` / `exclude` / `design` / `binding_types`:

```yaml
- chain:
    id: <string>
    res_index: <range_spec>       # "2..50,55.." or "..5,20..27"
    binding: <range_spec>
    not_binding: <range_spec>|"all"
```

### Constraints

```yaml
constraints:
- total_len: {min: <int>, max: <int>}   # combined polymer length of the complex
- bond:
    atom1: [<chain_id>, <res_num>, <atom_name>]
    atom2: [<chain_id>, <res_num>, <atom_name>]
```

## Output contract

Output every design in a single JSON array. Each element wraps one BoltzGen
specification:

```json
{
    "strategy_name": "three_helix_concave_cradle",
    "reasoning": "One or two sentences on why this approach suits this target.",
    "spec": {
        "entities": [
            {
                "protein": {
                    "id": "Z",
                    "sequence": "70..73",
                    "secondary_structure": {"helix": "1..20,27..46,53..70"}
                }
            },
            {
                "file": {
                    "path": "<the target path from the target config>",
                    "include": [{"chain": {"id": "A"}}],
                    "binding_types": [
                        {"chain": {"id": "A", "binding": "37,39,49,98",
                                   "not_binding": "2..6,79..83"}}
                    ]
                }
            }
        ]
    }
}
```

The JSON is the envelope; `spec` is the BoltzGen specification itself and follows
the schema above exactly.

Rules:

- `strategy_name` is unique across the array and describes the approach.
- Inside `spec`, use only keys from the closed list above.
- `binding` and `not_binding` are 1-based positional indices, not author
  numbering.
- Every `chain.id` under `file` exists in the target structure; the binder's own
  `id` never appears there.
- The binder length range in `protein.sequence` must match the length you stated
  for that design in the scaffold plan.
- `path` and the target chain ids come from the target config and are the same
  for every design.
