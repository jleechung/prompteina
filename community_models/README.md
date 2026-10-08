<!--
Modified by the Prompteina authors in 2026.
Changes are licensed under Apache 2.0; see LICENSE and NOTICE.
-->

# Vendored third-party code

Three upstream packages, copied in rather than pinned as dependencies so the
release is self-contained. Each keeps its own license; see `NOTICE` and
`licenses/license_third_party.txt` at the repository root.

| Package | What it is for | Source | License |
|---|---|---|---|
| **openfold** | Residue constants, rigid-body utilities and structure-module pieces the denoiser builds on | [aqlaboratory/openfold](https://github.com/aqlaboratory/openfold) | Apache 2.0 |
| **ProteinMPNN** | Binder redesign for the `MPNN` and `MPNN-FI` evaluation conditions | [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | MIT |
| **IPSAE** | Reference implementation of the ipSAE interface score | [DunbrackLab/IPSAE](https://github.com/DunbrackLab/IPSAE) | MIT |

`openfold` is exposed as a top-level import through the wheel-source mapping in
`pyproject.toml`. ProteinMPNN and IPSAE are scripts, not importable packages,
and are run as subprocesses by `metrics/mpnn.py` and
`metrics/ipsae_reference.py`.

## Weights

Only ProteinMPNN needs weights, and they are not vendored:

```bash
./env/download_startup.sh --pmpnn     # -> community_models/ProteinMPNN/vanilla_model_weights/
```

IPSAE has no weights. openfold is used for its utilities, not its model.
