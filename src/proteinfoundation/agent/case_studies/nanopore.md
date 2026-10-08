# Prompteina — designing a transmembrane barrel nanopore

Everything the model can be conditioned on for THIS objective, every field you
may set, and the exact form your output must take.

## What the model is

A structure diffusion model that generates a **single protein chain** from a
text description. There is no target, no partner, and no assembly — one chain,
at most 512 residues. Conditioning is by text: a set of structural facts
describing the chain you want.

The objective here is a **single-chain transmembrane beta-barrel with an open
lumen**. Every designed nanopore in the literature is single-chain, so this is
inside what the model can produce rather than a stretch of it.

## What carries the design, and what does not

Read this before writing anything. The lumen is produced by **barrel geometry**,
not by asking for a pore.

| channel | effect | why |
|---|---|---|
| `Architecture: Beta Barrel` + `Beta topology` | **This is the design.** | Barrel captions are common in training; the strand count and closure are what create a lumen |
| `Membrane protein` | weak, measured separately | Transmembrane-beta captions are rare, and rarer still alongside a barrel. Treat as a request that may not land |
| `Solvent tunnels` | **do not use** | It is a count with no radius or length, and it saturates. The annotation marks active-site tunnels in ordinary enzymes, so it means "this protein has cavities", not "this protein has a lumen". It spends a bullet and buys nothing |

There is no bullet that says "1.2 nm pore". Diameter is a **consequence** of
strand count and barrel radius, and it is measured afterwards, not requested.

## Size and topology are not independent

A strand count implies a residue count. Getting this wrong produces an
impossible specification — the failure that motivated the scaffold-plan rule.
The numbers below are measured from deposited structures, not estimated:

| strands | residues | evidence |
|---|---|---|
| 8 | 120–140 | 6X9Z 120, 6X1K 124 (designed, Vorobieva 2021); OmpA 1QJP 137 |
| 10–12 | 140–180 | 8UZL 141–143 (designed, Berhanu 2024) |
| 14 | 230–280 | OmpG 2IWV 277 |
| 16 | 290–340 | extrapolated at ~14–18 residues per strand-plus-turn unit |

Lumen diameter rises with strand count. Published ranges for designed barrels:
0.5–1.1 nm (Berhanu 2024), 0.7–1.5 nm at 10/12/14/16 strands (2026). Our own
measurement of 8UZL is 0.58 nm, inside its paper's range, so the instrument and
the literature agree on convention.

**Barrels below 8 strands do not close, and above 16 the model is being asked
for something with very few training examples. Stay in 8–16.**

## The bullet vocabulary is CLOSED — use these labels exactly

Each bullet is `- Label: value`. A label the model was not trained on is
dropped without warning. The labels below are the ones that apply to this
objective.

**Length** — not passed to the model as text, they set the residue count.

| bullet | value | example |
|---|---|---|
| `nres_min` | integer | `210` |
| `nres_max` | integer | `260` |

**Fold — the load-bearing bullets**

| label | value | example |
|---|---|---|
| `Structural class` | must be `Mainly Beta` for a barrel | `Mainly Beta` |
| `Architecture` | must be `Beta Barrel` | `Beta Barrel` |
| `Topology` | free-form CATH topology name | `Transmembrane beta-barrel` |
| `Beta topology` | `barrel` / `open sheet` / `simple sheet`, with strand count and orientation | `14-stranded antiparallel barrel` |
| `Strands` | number of beta strands — must agree with `Beta topology` | `14` |
| `Sheets` | sheet count and composition | `1 (sheet 1 with 14 strands)` |

**Membrane** — the measured arm. Include it or omit it as the run specifies;
never edit anything else between the two arms.

| label | value | example |
|---|---|---|
| `Membrane protein` | `alpha` or `beta`, with TM segment count and optional tilt | `beta, 14 TM segments` |

**Shape and surface**

| label | value | example |
|---|---|---|
| `Compactness` | class, shape, optional Rg and residue count | `moderate, elongated shape (aspect ratio 1.8)` |
| `Contact order` | `very low` · `low` · `moderate` · `high` · `very high` | `low` |
| `Surface hydrophobicity` | `low` · `moderate` · `high`, with patch count and largest area | `high aggregation propensity, 4 hydrophobic surface patches` |
| `Surface geometry` | `no prominent flat face` · `small flat patch` · `has flat patch` · `large flat face` | `no prominent flat face` |
| `Symmetry` | **internal** symmetry within the chain, NOT assembly symmetry | `asymmetric (no detectable internal repeats)` |

**Local features**

| label | value | example |
|---|---|---|
| `Turn composition` | beta-turn count and types | `13 beta-turns` |
| `Salt bridges` | intra-chain salt bridge count | `6 total` |
| `Buried unsatisfied polar groups` | count and rate | `8 (3.2 per 100 residues)` |

A transmembrane barrel is hydrophobic on its **outer** face and polar in its
**lumen**. The vocabulary has no bullet for that split — `Surface
hydrophobicity` is whole-surface. Ask for high surface hydrophobicity and
measure whether the banding appears; do not claim it was specified.

## Two traps specific to this objective

- **`Symmetry:` is internal, not quaternary.** Under v3.1 this label means
  CE-Symm repeats within the chain. It does not request an oligomer. There is no
  assembly to request here anyway — the model emits one chain.
- **Do not name a protein.** No `OmpG`, no `alpha-hemolysin`, no `MspA`. The
  bullets describe geometry. Naming a native protein is outside the caption
  distribution and is not how the fold is specified.

## Output contract

One design block per design, in order.

```
### Design 1
- nres_min: 230
- nres_max: 270
- Structural class: Mainly Beta
- Architecture: Beta Barrel
- Beta topology: 14-stranded antiparallel barrel
- Strands: 14
- Membrane protein: beta, 14 TM segments
- Surface hydrophobicity: high aggregation propensity, 4 hydrophobic surface patches
- Contact order: low
- Turn composition: 13 beta-turns
```

- Each design: 5–10 structural bullets, plus `nres_min` and `nres_max`.
- Each bullet is `- Label: value` on one line, with no headers beyond the
  design line.
- State one optimal value per property, never a range.
- Omit any property that is absent or zero — state only what IS present.
- `Strands` and `Beta topology` must agree. `nres_min`/`nres_max` must bracket
  the residue count the strand count implies, per the table above.
- Span the strand range across the design set: cover 8, 10, 12, 14 and 16
  strands rather than producing one size repeatedly.
