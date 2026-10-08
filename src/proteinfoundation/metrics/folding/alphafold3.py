"""AlphaFold3 backend.

Wraps the AlphaFold3 runner that `proteinfoundation.rewards.alphafold3_reward`
already sets up — model load and JAX compilation are paid once and reused —
and exposes it through the `FoldingBackend` interface so the same scoring code
serves both folding models.

Two settings here are not defaults and are not interchangeable with them:

No MSA. `unpairedMsa` and `pairedMsa` are set to the empty string, which is
AlphaFold3's explicit single-sequence mode. Omitting the keys instead would
make it run its full data pipeline.

Templates on, from the designed structure. These are de novo designs with no
homologs, and AlphaFold3 given neither an MSA nor a template returns an ipSAE
of zero for every one of them. The designed complex is therefore written out as
one single-chain mmCIF per chain and supplied as a template, with query and
template indices mapped by residue number rather than by position, so that
domain-cropped targets do not shift the alignment.

AlphaFold3 needs its own source tree and weights, neither of which ships here.
Point `af3_src` at the checkout and `model_dir` at the directory holding
`af3.bin`, or set AF3_SRC and AF3_MODEL_DIR.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from proteinfoundation.metrics.folding.base import FoldingBackend, FoldResult
from proteinfoundation.rewards.alphafold3_reward import AF3RewardModel, _write_chain_templates

logger = logging.getLogger(__name__)


class AlphaFold3Backend(FoldingBackend):
    """AlphaFold3, single-sequence with the design as template.

    Args:
        model_dir: Directory holding af3.bin. Falls back to $AF3_MODEL_DIR.
        af3_src: AlphaFold3 source root, the one containing run_alphafold.py.
            Falls back to $AF3_SRC.
        num_recycles: 3, matching how the paper ran it, not AlphaFold3's own
            default of 10.
        seed: Fold seed.
        device_id: GPU index.
    """

    name = "alphafold3"
    #: AlphaFold3 without an MSA and without a template scores every design 0.
    REQUIRES_TEMPLATE = True

    def __init__(
        self,
        model_dir: str | None = None,
        af3_src: str | None = None,
        num_recycles: int = 3,
        seed: int = 42,
        device_id: int = 0,
        **kwargs,
    ) -> None:
        model_dir = model_dir or os.environ.get("AF3_MODEL_DIR")
        af3_src = af3_src or os.environ.get("AF3_SRC")
        if not model_dir or not af3_src:
            raise ValueError(
                "AlphaFold3 needs model_dir (holding af3.bin) and af3_src (the source "
                "root with run_alphafold.py). Pass them, or set AF3_MODEL_DIR and AF3_SRC."
            )
        self._reward = AF3RewardModel(
            model_dir=model_dir,
            af3_src=af3_src,
            num_recycles=num_recycles,
            seed=seed,
            device_id=device_id,
            use_templates=True,
            **kwargs,
        )
        self.seed = seed

    def fold(
        self,
        binder_seq: str,
        target_seqs: list[str],
        seed: int = 0,
        template_pdb: str | None = None,
    ) -> FoldResult:
        if template_pdb is None:
            raise ValueError(
                "AlphaFold3 needs the designed structure as a template. Without one, and "
                "with no MSA, it returns ipSAE 0 for every design. Pass template_pdb."
            )
        import jax
        import biotite.structure as bstruc
        import biotite.structure.io.pdb as pdb_io
        from alphafold3.common import folding_input
        from alphafold3.data import featurisation

        self._reward._ensure_runner()
        arr = pdb_io.PDBFile.read(template_pdb).get_structure(model=1)
        arr = arr[bstruc.filter_amino_acids(arr)]

        # The binder is chain A and the targets follow, matching FoldResult's
        # contract. The template chains are matched to them by sequence, not by
        # the order they appear in the file.
        seqs_by_chain = {
            c: str(bstruc.to_sequence(arr[arr.chain_id == c])[0][0]) for c in sorted(set(arr.chain_id))
        }
        wanted = [binder_seq] + list(target_seqs)
        chain_for: list[str] = []
        unused = dict(seqs_by_chain)
        for seq in wanted:
            match = next((c for c, s in unused.items() if s == seq), None)
            if match is None:
                raise ValueError(
                    f"{template_pdb}: no chain matches a requested sequence of length {len(seq)}. "
                    f"Template chains are {[(c, len(s)) for c, s in seqs_by_chain.items()]}."
                )
            chain_for.append(match)
            unused.pop(match)

        ids = ["A"] + [chr(ord("B") + k) for k in range(len(target_seqs))]
        lengths = [len(s) for s in wanted]
        name = Path(template_pdb).stem

        tmp = Path(tempfile.mkdtemp(prefix="af3fold_", dir=os.environ.get("TMPDIR", "/tmp")))
        try:
            tpl = _write_chain_templates(arr, tmp, name)
            entries = []
            for new_id, src_chain, seq in zip(ids, chain_for, wanted, strict=True):
                entries.append(
                    {
                        "protein": {
                            "id": new_id,
                            "sequence": seq,
                            "pairedMsa": "",
                            "unpairedMsa": "",
                            "templates": [
                                {
                                    "mmcifPath": str(tpl[src_chain]),
                                    "queryIndices": list(range(len(seq))),
                                    "templateIndices": list(range(len(seq))),
                                }
                            ],
                        }
                    }
                )
            doc = {
                "name": name,
                "sequences": entries,
                "modelSeeds": [seed or self.seed],
                "dialect": "alphafold3",
                "version": 2,
            }
            fold_input = folding_input.Input.from_json(json.dumps(doc))
            examples = featurisation.featurise_input(
                fold_input=fold_input, buckets=self._reward.buckets, ccd=self._reward._ccd, verbose=False
            )
            result = self._reward._runner.run_inference(examples[0], jax.random.PRNGKey(seed or self.seed))
            infer = self._reward._runner.extract_inference_results(
                batch=examples[0], result=result, target_name=name
            )[0]

            pae = np.asarray(infer.numerical_data["full_pae"], dtype=np.float32)
            scalars = {
                k: float(v)
                for k, v in (infer.metadata or {}).items()
                if isinstance(v, (int, float)) and k in ("ptm", "iptm", "ranking_score")
            }
            return FoldResult(
                pae=pae,
                plddt=None,
                cif=infer.predicted_structure.to_mmcif(),
                chain_ids=ids,
                chain_lengths=lengths,
                scalars=scalars,
                backend=self.name,
            )
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def close(self) -> None:
        self._reward.cleanup()
