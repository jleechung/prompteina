# Prompteina — what the method can express

Everything the model can be conditioned on, every field you may set, and the
exact form your output must take.

## What the model is

A structure diffusion model that generates a **binder backbone** in the context
of a fixed target structure. Conditioning is by text: a set of structural facts
describing the binder you want, not the target you want it to bind.

The model outputs backbones. Sequences come from the model's own sequence head in
the self-sequence condition, or from ProteinMPNN in the redesign conditions.

## What you can tweak — the complete set

One specification per design. Every channel below is set independently in each
one; only the target structure is fixed across a target's designs.

| channel | granularity | field | what it controls |
|---|---|---|---|
| structural facts | per design | the category bullets, → `facts.csv` `description` | the binder's fold, topology and interface character |
| binder length | per design | `nres_min` / `nres_max`, → `facts.csv` `nres` | residue count of the binder |
| target epitope | per design | `hotspot_residues` | which residues on the target the binder should contact |
| target structure | per target | `target_path` + `target_input` in the target config | the fixed context the binder is generated against |

### Binder length

`nres_min` / `nres_max` are the range you reason about; the pipeline draws a
single `nres` per design from that range and writes it as an explicit column.

The draw happens once, when your specification is parsed. A design that carries
an explicit length is generated at exactly that length; one that carries only a
range is re-sampled per generation job, and the sampler collapses any range
narrower than about 32 residues to `nres_min + 6`. State the range you want and
the pipeline fixes a value inside it.

### Target epitope

`hotspot_residues` is a comma-separated list of residue identifiers on the
**target** — chain letter then residue number, in the numbering the target config
defines. For a target given as `A1-169`:

```
- hotspot_residues: A17, A22, A113, A120
```

It becomes a per-residue mask over the target, which the sampler reads while it
denoises the binder. Prefer solvent-exposed residues forming one contiguous patch
a single binder can engage.

Two model-side properties govern what makes a usable site:

- **The channel is optional by construction.** Training applied hotspot dropout,
  so the model has seen both conditioned and unconditioned inputs. An empty mask
  is in-distribution, not degenerate.
- **Training hotspots were true contact residues**, sampled from real complexes.
  A site that is not a plausible contact surface — buried, or on a face no
  partner approaches — is outside what the model was trained on and may be
  ignored.

Every identifier must resolve inside the target numbering; one that does not is
a hard error, never a silent drop.

## The bullet vocabulary is CLOSED — this is the complete list

Each bullet is `- Label: value`. **Use these labels exactly.** They are the
labels the model's training descriptions were written from; a label it has not
seen conditions it weakly or not at all, and an unrecognised one is dropped
without warning.

Each row gives the value the label expects and an example in the phrasing the
model was trained on.

**Length** — these two are read by the pipeline, not passed to the model as text.

| bullet | value | example |
|---|---|---|
| `nres_min` | integer, minimum residue count for this scaffold | `78` |
| `nres_max` | integer, maximum residue count for this scaffold | `95` |

**Fold**

| label | value | example |
|---|---|---|
| `Structural class` | one of the three classes below | `Mainly Alpha` |
| `Architecture` | architecture name from the closed list below | `Up-down Bundle` |
| `Topology` | free-form topology / fold-family name | `Globin-like` |

**Secondary structure**

| label | value | example |
|---|---|---|
| `Helices` | number of helices | `4` |
| `Strands` | number of beta strands | `7` |
| `Sheets` | number of sheets with strand composition | `2 (sheet 1 with 5 strands; sheet 2 with 4 strands)` |
| `Secondary structure` | overall SS composition | `42% helix, 18% strand, 40% coil` |
| `Beta topology` | one of `barrel`, `open sheet`, `simple sheet`, with strand count and orientation | `8-stranded parallel barrel` |

**Interface** — the binder's side of the interface it will form

| label | value | example |
|---|---|---|
| `Protein-protein interface` | number of interface residues | `24 interface residues` |
| `Buried surface area` | interface BSA in Å² | `1842 A^2` |
| `Interface H-bonds` | number of inter-chain hydrogen bonds | `12 total` |
| `Interface salt bridges` | number of inter-chain salt bridges | `4 total` |
| `Interface hydrophobic contacts` | number of inter-chain hydrophobic contacts | `18 total` |
| `Interface geometry` | shape, symmetry and character with residue count | `flat, isologous (symmetric), hydrophobic (28 interface residues)` |

**Chemistry and stability**

| label | value | example |
|---|---|---|
| `Disulfide bonds` | number, split intra/inter-chain | `2 intra-chain` |
| `Disulfide topology` | connectivity pattern | `single disulfide` |
| `Salt bridges` | number of intra-chain salt bridges | `8 total (mean distance 3.2 A)` |
| `Buried unsatisfied polar groups` | count and rate | `6 (2.7 per 100 residues)` |
| `Surface hydrophobicity` | one of `low`, `moderate`, `high` aggregation propensity, with patch count and largest area | `moderate aggregation propensity, 7 hydrophobic surface patches, largest 386 A^2` |
| `Charge character` | net charge / distribution in prose | `acidic surface, net charge -6` |
| `Hydrophobicity` | overall hydrophobic character | `moderately hydrophobic core` |
| `Turn composition` | beta-turn count and types | `9 turns (5 type I, 3 type II)` |
| `Cis-peptide bonds` | number of cis peptides | `1 cis-proline` |

**Global shape and dynamics**

| label | value | example |
|---|---|---|
| `Contact order` | one of `very low`, `low`, `moderate`, `high`, `very high` | `high (relative CO 0.2227)` |
| `Contact density` | packing density class | `densely packed` |
| `Compactness` | compactness class | `compact globular` |
| `Flexibility` | counts and sizes of rigid and flexible regions | `3 rigid regions (12 residues, 8 residues, 15 residues); 2 flexible regions (6 residues, 4 residues)` |
| `Surface geometry` | one of `no prominent flat face`, `small flat patch`, `has flat patch`, `large flat face`, with area | `large flat face (524 A^2)` |
| `Repeat structure` | number and size of repeats with one of `solenoid`, `propeller`, `barrel`, `mixed` | `4 repeats of 33 residues (solenoid, RMSD 0.8 A)` |
| `Symmetry` | internal symmetry within the chain | `asymmetric (no detectable internal repeats)` · `4-fold closed cyclic symmetry, 33-residue repeats` |
| `Coiled-coil` | coiled-coil register and length | `2-helix parallel coiled-coil, 28 residues` |

**Layout** — use only if the binder genuinely has more than one domain

| label | value | example |
|---|---|---|
| `Chain layout` | N-to-C walk with approximate region sizes | `N-terminal loop; 3-helix/2-strand region (~57 residues); loop (~15 residues)` |
| `Domain architecture` | domains with approximate sizes | `N-terminal Rossmann-like (~220 residues); C-terminal beta barrel (~155 residues)` |
| `Domain linkers` | linker lengths and character | `93-residue mixed linker` |
| `Domain interface` | one of `packed`, `moderate`, `loose`, `independent`, with contact count | `packed (70 inter-domain contacts)` |

**Behavior**

| label | value | example |
|---|---|---|
| `Behavior` | an observed structural or dynamic behavior, in prose | `The binder rigidifies upon target engagement` |

## Controlled vocabularies

Nine labels take a value from a closed set. A value outside the set is dropped
silently, exactly like an unknown label.

| label | allowed values |
|---|---|
| `Structural class` | Mainly Alpha · Mainly Beta · Alpha Beta |
| `Architecture` | see the table below |
| `Contact order` | very low · low · moderate · high · very high |
| `Surface hydrophobicity` | low · moderate · high |
| `Surface geometry` | no prominent flat face · small flat patch · has flat patch · large flat face |
| `Beta topology` | barrel · open sheet · simple sheet |
| `Repeat structure` | solenoid · propeller · barrel · mixed |
| `Domain interface` | packed · moderate · loose · independent |
| `Compactness` | very low · low · moderate · high · very high |

**Valid `Architecture` values, by structural class**

| class | architectures |
|---|---|
| Mainly Alpha | Orthogonal Bundle, Up-down Bundle, Alpha Horseshoe, Alpha solenoid, Alpha/alpha barrel |
| Mainly Beta | Ribbon, Single Sheet, Roll, Beta Barrel, Clam, Sandwich, Distorted Sandwich, Trefoil, Orthogonal Prism, Aligned Prism, 3-layer Sandwich, 3 Propeller, 4 Propeller, 5 Propeller, 6 Propeller, 7 Propeller, 8 Propeller, 2 Solenoid, 3 Solenoid, Beta Complex, Shell |
| Alpha Beta | Roll, Super Roll, Alpha-Beta Barrel, 2-Layer Sandwich, 3-Layer(aba) Sandwich, 3-Layer(bba) Sandwich, 3-Layer(bab) Sandwich, 4-Layer Sandwich, Alpha-beta prism, Box, 5-stranded Propeller, Alpha-Beta Horseshoe, Alpha-Beta Complex |

`Topology` takes a CATH T-level node name. There are 1,282 of them under the
three classes above, so rather than list them, call
`cath_topologies(architecture=...)` with the architecture you chose and pick one
of the names it returns. They look like this:

| under | code | name |
|---|---|---|
| Up-down Bundle | 1.20.120 | Four Helix Bundle (Hemerythrin (Met), subunit A) |
| Orthogonal Bundle | 1.10.287 | Helix Hairpins |
| Beta Barrel | 2.40.50 | OB fold (Dihydrolipoamide Acetyltransferase, E2P) |
| Sandwich | 2.60.40 | Immunoglobulin-like |
| Roll | 3.10.20 | Ubiquitin-like (UB roll) |
| 2-Layer Sandwich | 3.30.70 | Alpha-Beta Plaits |
| 3-Layer(aba) Sandwich | 3.40.50 | Rossmann fold |

Write the name, not the code. Names outside the returned list are dropped.

## Output contract

One design block per design, in order.

```
### Design 1
- nres_min: 78
- nres_max: 95
- hotspot_residues: A17, A22, A113, A120
- Structural class: Mainly Alpha
- Architecture: Up-down Bundle
- Helices: 4
- Protein-protein interface: 22 interface residues
- Buried surface area: 1650 A^2
- Interface geometry: concave, hydrophobic (22 interface residues)
```

- Each design: 5–10 structural bullets, plus `nres_min`, `nres_max` and
  `hotspot_residues`.
- Each bullet is `- Label: value`, using a label from the table above, with no
  explanation around it.
- State the optimal value for each property, not a range.
- Counts may be exact or qualitative; BSA may be numeric or qualitative.
- Omit zero-valued or absent properties — state only what IS present.
- **The structural bullets describe the BINDER only.** Do not name the target
  protein or its residues in any of them. `hotspot_residues` is the sole field
  that refers to the target, and it is required — a design that names no site is
  rejected, and every identifier is validated against the target numbering.
