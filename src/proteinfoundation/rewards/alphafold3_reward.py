"""AlphaFold 3 as a beam-search reward, in-process.

Drop-in replacement for :class:`AF2RewardModel` in the search loop. Emits the
same three quantities on the same thresholds, so ``CONJUNCTION_GATES`` values
and the ``min_margin`` formula carry over unchanged.

WHY IN-PROCESS. ``compute_reward_from_samples`` calls ``score()`` once per
structure, serially (reward_utils.py:135-160). Shelling out to
``run_alphafold.py`` would pay model load plus JAX compile on every one of the
~64 lookaheads in a beam search. ``ModelRunner``'s ``model_params`` and
``_model`` are ``cached_property`` (run_alphafold.py:451-481), so holding one
instance loads and JITs once per process.

WHY THE METRIC CODE IS DUPLICATED RATHER THAN IMPORTED. ``alphafold2_reward``
imports colabdesign at module scope, which pins jax 0.4.x; AF3 pins jax 0.10.2.
Importing it here would make the environment unbuildable. The three formulas
below must stay in step with ``AF2RewardModel.CONJUNCTION_GATES`` and
``_min_margin_reward``; both sides carry a comment pointing at the other.

METRIC PARITY, verified against ColabDesign's af/loss.py
(https://github.com/sokrypton/ColabDesign), which this release does not vendor:

  i_pae  colabdesign averages the SYMMETRISED pae over the binder x target block
         and divides by 31 (loss.py:59,268-276). Averaging the raw matrix over
         BOTH off-diagonal blocks is algebraically identical, since
         sum_{i in B, j in T} P_ji == sum_{i in T, j in B} P_ij by relabelling.
  rmsd   colabdesign aligns on the first L=len(target) positions and measures
         the remainder, CA only -- all_atom_positions[:,1] is CA
         (loss.py:518-530). Same here.
  plddt  NOT equivalent as AF3 ships it. colabdesign is 1 - pLDDT over binder
         RESIDUES (loss.py:54,264-266; predicted_lddt logits are per-residue);
         AF3's atom_b_factor is pLDDT 0-100 per ATOM over the whole complex.
         Converted below: mean within each residue first, then over binder
         residues -- a flat atom mean would overweight large residues (Trp has
         14 heavy atoms, Gly 4).

AF3 IS THE HARSHER CRITIC, and that is not a labeling nicety. AF2 runs with
``use_initial_guess=True``, which seeds the entire designed complex pose into
the recycling embedder (colabdesign/af/design.py:166-167). AF3 has no initial
guess and docks from noise. Under AF3 the scRMSD gate is a docking test; under
AF2 it is a pose-stability test.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from proteinfoundation.rewards.base_reward import BaseRewardModel, standardize_reward

logger = logging.getLogger(__name__)

PAE_SCALE = 31.0

# Mirrors AF2RewardModel.CONJUNCTION_GATES exactly. Expressed in loss units:
#   i_pae  iPAE < 7 A      -> loss = iPAE/31   -> 7/31
#   plddt  pLDDT > 0.90    -> loss = 1-pLDDT   -> 0.10   (= 1 - 0.90)
#   rmsd   scRMSD < 1.5 A  -> loss = A         -> 1.5
# If you change these, change alphafold2_reward.CONJUNCTION_GATES too.
CONJUNCTION_GATES = {"i_pae": 7.0 / PAE_SCALE, "plddt": 0.10, "rmsd": 1.5}


def _write_chain_templates(struct_arr, tpl_dir: Path, name: str) -> dict[str, Path]:
    """One single-chain mmCIF per chain -- AF3 requires it (folding_input.py:107).

    Occupancy and B-factor must be set explicitly: design PDBs carry neither and
    biotite then emits an empty occupancy column, which AF3's mmCIF parser
    rejects. AF3 also refuses any template with no release date
    (data/templates.py:855); these are de novo designs, so a fixed early date is
    supplied purely to satisfy the parser and keep --max_template_date from
    filtering them. Same construction the (purged) rescoring pipeline used, which the
    167,927-structure rescoring campaign used.
    """
    import biotite.structure as bstruc
    import biotite.structure.io.pdbx as pdbx

    out: dict[str, Path] = {}
    for chain in sorted(set(struct_arr.chain_id)):
        sub = struct_arr[struct_arr.chain_id == chain].copy()
        sub.set_annotation("occupancy", np.ones(sub.array_length(), dtype=float))
        sub.set_annotation("b_factor", np.zeros(sub.array_length(), dtype=float))
        cif_path = tpl_dir / f"{name}_{chain}.cif"
        block = pdbx.CIFFile()
        pdbx.set_structure(block, sub)
        block.block["pdbx_audit_revision_history"] = pdbx.CIFCategory({
            "ordinal": "1", "data_content_type": "Structure model",
            "major_revision": "1", "minor_revision": "0",
            "revision_date": "1989-11-17",
        })
        block.write(cif_path)
        out[chain] = cif_path
    return out


class AF3RewardModel(BaseRewardModel):
    """AlphaFold 3 folding reward for the binder protocol."""

    IS_FOLDING_MODEL = True
    SUPPORTS_GRAD = False
    SUPPORTS_SAVE_PDB = False

    def __init__(
        self,
        model_dir: str,
        af3_src: str = os.environ.get("AF3_SRC", "./alphafold3"),
        num_recycles: int = 3,
        num_diffusion_samples: int = 1,
        seed: int = 42,
        device_id: int = 0,
        reward_aggregation: str = "weighted_sum",
        reward_weights: dict[str, float] | None = None,
        conjunction_gates: dict[str, float] | None = None,
        flash_attention_implementation: str = "triton",
        buckets: tuple[int, ...] | None = None,
        target_msa_paths: dict[str, str] | None = None,
        use_templates: bool = True,
        emit_ipsae: bool = False,
    ):
        """
        Args:
            model_dir: directory holding af3.bin.
            af3_src: AlphaFold 3 source root. `run_alphafold` lives there as a
                script, not in the installed package, so it needs sys.path.
            num_recycles: 3 to match AF2RewardModel's default, not AF3's 10.
            num_diffusion_samples: 1 to match AF2's single deterministic model.
                AF3's default 5 is best-of-5, which makes the reward optimistic
                rather than expected-quality and costs 5x the diffusion.
            seed: fixed across calls, so the reward is reproducible and its error
                is correlated within a beam rather than resampled per candidate.
            reward_aggregation: "weighted_sum" or "min_margin".
            conjunction_gates: defaults to CONJUNCTION_GATES above.
            flash_attention_implementation: 'xla' is required on compute
                capability 7.x (run_alphafold.py:951-972).
        """
        super().__init__()
        if reward_aggregation not in ("weighted_sum", "min_margin"):
            raise ValueError(f"unknown reward_aggregation {reward_aggregation!r}")
        self.model_dir = model_dir
        self.af3_src = af3_src
        self.num_recycles = num_recycles
        self.num_diffusion_samples = num_diffusion_samples
        self.seed = seed
        self.device_id = device_id
        self.reward_aggregation = reward_aggregation
        self.reward_weights = dict(reward_weights or {"i_pae": -1.0})
        self.conjunction_gates = dict(conjunction_gates or CONJUNCTION_GATES)
        self.flash_attention_implementation = flash_attention_implementation
        self.buckets = buckets
        # Overath et al. -- the only AF3 setup with wet-lab validation behind it
        # -- run MSAs ON and templates OFF, the inverse of this campaign's
        # default. These knobs exist so the difference can be measured rather
        # than argued. Defaults preserve current campaign behavior.
        #
        # target_msa_paths maps CHAIN ID -> a3m and applies to TARGET chains
        # only. The binder never gets one: a de novo sequence has no natural
        # homologs, so its alignment is empty by construction. That is also what
        # makes MSAs affordable -- one per target, reused across every design,
        # not one per generated sequence.
        self.target_msa_paths = dict(target_msa_paths or {})
        self.use_templates = use_templates
        self.emit_ipsae = emit_ipsae
        self._runner = None
        self._ccd = None
        # Guards the one-compile-per-search invariant, which nothing else
        # enforces: TextPromptFeature samples a length per CSV row and
        # collate_fn pads mixed lengths happily, so a multi-row CSV would
        # silently recompile on every call instead of failing.
        self._seen_tokens: int | None = None

    # -- lazy init -----------------------------------------------------------
    def _ensure_runner(self):
        if self._runner is not None:
            return
        import jax
        from alphafold3.constants import chemical_components

        # `run_alphafold` is a script at the AF3 repo ROOT, not part of the
        # installed `alphafold3` package, so it is not importable without
        # putting that root on sys.path. ModelRunner and make_model_config live
        # there and are worth reusing rather than reimplementing -- they are the
        # exact code path the 167,927-structure campaign ran.
        import sys
        if self.af3_src not in sys.path:
            sys.path.insert(0, self.af3_src)
        # absl FLAGS must be marked parsed BEFORE anything reads a flag value.
        # Importing run_alphafold registers its flags; the first access to any
        # of them makes absl parse the real sys.argv, which under Hydra contains
        # --config-path and friends. absl then aborts with "Unknown command line
        # flag 'config-path'" -- and CompositeRewardModel swallows that into a
        # single warning, so every lookahead silently scores as nothing.
        # known_only=True makes absl ignore argv it does not recognize.
        from absl import flags as _absl_flags
        if not _absl_flags.FLAGS.is_parsed():
            _absl_flags.FLAGS(["af3_reward"], known_only=True)

        try:
            import run_alphafold
        except ImportError as e:  # pragma: no cover - environment problem
            raise ImportError(
                f"cannot import run_alphafold from {self.af3_src!r}. Set af3_src to the "
                "AlphaFold 3 source root, and ensure LD_LIBRARY_PATH includes "
                "/sw/gcc/13.2.0/lib64 (the C++ extension needs GLIBCXX_3.4.32)."
            ) from e

        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("AF3RewardModel requires a GPU; jax.devices('gpu') is empty")
        device = gpus[self.device_id]
        # Constructing ModelRunner directly skips run_alphafold.main()'s device
        # discovery and compute-capability validation (:937-974), so the
        # selection above is done explicitly rather than inherited.
        cfg = run_alphafold.make_model_config(
            flash_attention_implementation=self.flash_attention_implementation,
            num_diffusion_samples=self.num_diffusion_samples,
            num_recycles=self.num_recycles,
        )
        self._runner = run_alphafold.ModelRunner(
            config=cfg, device=device, model_dir=Path(self.model_dir)
        )
        self._ccd = chemical_components.Ccd()
        logger.info(
            "AF3RewardModel ready: recycles=%d samples=%d device=%s",
            self.num_recycles, self.num_diffusion_samples, device,
        )

    # -- metrics -------------------------------------------------------------
    @staticmethod
    def _ipae(pae: np.ndarray, token_chain_ids: list[str], binder_chain: str) -> float:
        is_b = np.asarray(token_chain_ids) == binder_chain
        if is_b.all() or (~is_b).all():
            # A per-arm chain-naming difference would silently delete an entire
            # arm, so this must never be treated as missing data.
            raise ValueError(f"binder chain {binder_chain!r} not separable in prediction")
        cross = np.concatenate([pae[np.ix_(is_b, ~is_b)].ravel(),
                                pae[np.ix_(~is_b, is_b)].ravel()])
        return float(cross.mean())

    @staticmethod
    def _binder_residue_plddt(struct, binder_chain: str) -> float:
        """Mean pLDDT (0-1) over binder RESIDUES, matching colabdesign's scope."""
        chain = np.asarray(struct.chain_id)
        res = np.asarray(struct.res_id)
        bf = np.asarray(struct.atom_b_factor, dtype=np.float64)
        sel = chain == binder_chain
        if not sel.any():
            raise ValueError(f"binder chain {binder_chain!r} absent from predicted structure")
        # Per-residue mean first, then mean over residues. A flat atom mean would
        # weight residues by heavy-atom count.
        r = res[sel]
        b = bf[sel]
        order = np.argsort(r, kind="stable")
        r, b = r[order], b[order]
        bounds = np.flatnonzero(np.diff(r)) + 1
        per_res = np.array([seg.mean() for seg in np.split(b, bounds)])
        return float(per_res.mean() / 100.0)

    @staticmethod
    def _scrmsd(pred_ca: dict[str, np.ndarray], des_ca: dict[str, np.ndarray],
                binder_chain: str) -> float:
        """Target-aligned binder CA displacement -- colabdesign's `rmsd` loss."""
        tgt = sorted(set(des_ca) - {binder_chain})
        pt = np.concatenate([pred_ca[c] for c in tgt])
        dt = np.concatenate([des_ca[c] for c in tgt])
        pb, db = pred_ca[binder_chain], des_ca[binder_chain]
        if len(pt) != len(dt) or len(pb) != len(db) or len(pt) == 0:
            raise ValueError(
                f"CA length mismatch pt{len(pt)}/dt{len(dt)}/pb{len(pb)}/db{len(db)}")
        # Kabsch on target only, then apply that transform to the binder.
        pc, dc = pt.mean(0), dt.mean(0)
        u, _, vt = np.linalg.svd((pt - pc).T @ (dt - dc))
        d = np.sign(np.linalg.det(u @ vt))
        rot = u @ np.diag([1.0, 1.0, d]) @ vt
        moved = (pb - pc) @ rot + dc
        return float(np.sqrt(((moved - db) ** 2).sum(axis=1).mean()))

    # -- main ----------------------------------------------------------------
    def score(self, pdb_path: str, requires_grad: bool = False, **kwargs) -> dict[str, Any]:
        self._check_capabilities(requires_grad=requires_grad, save_pdb=kwargs.get("save_pdb", False))
        self._ensure_runner()

        import jax
        import biotite.structure as bstruc
        import biotite.structure.io.pdb as pdb_io
        from alphafold3.common import folding_input
        from alphafold3.data import featurisation

        binder_chain = kwargs.get("binder_chain")
        arr = pdb_io.PDBFile.read(pdb_path).get_structure(model=1)
        arr = arr[bstruc.filter_amino_acids(arr)]
        chains = sorted(set(arr.chain_id))
        if len(chains) < 2:
            raise ValueError(f"{pdb_path}: expected >=2 chains, got {chains}")
        if binder_chain is None:
            # The caller normally supplies this; falling back to "last chain"
            # is the documented contract but is only correct because the PDBs
            # happen to be alphabetical, so it is a last resort, not a default.
            binder_chain = chains[-1]
            logger.warning("binder_chain not supplied; falling back to %r", binder_chain)

        seqs = {c: bstruc.to_sequence(arr[arr.chain_id == c])[0][0] for c in chains}
        seqs = {c: str(s) for c, s in seqs.items()}
        n_tokens = sum(len(s) for s in seqs.values())
        if self._seen_tokens is None:
            self._seen_tokens = n_tokens
        elif n_tokens != self._seen_tokens:
            # One bucket per search is what keeps this to a single JIT compile.
            raise ValueError(
                f"token count changed within one process ({self._seen_tokens} -> "
                f"{n_tokens}); every structure in a beam must share a length or "
                "AF3 recompiles per call. Check nsamples==1 and the prompt CSV.")

        tmp = Path(tempfile.mkdtemp(prefix="af3rw_", dir=os.environ.get("TMPDIR", "/tmp")))
        try:
            name = Path(pdb_path).stem
            tpl = _write_chain_templates(arr, tmp, name) if self.use_templates else {}
            seq_entries = []
            for c in chains:
                prot = {"id": c, "sequence": seqs[c], "pairedMsa": ""}
                a3m = self.target_msa_paths.get(c)
                if a3m and c != binder_chain:
                    # Path form, not inline: AF3 reads the a3m itself, and that
                    # is what Overath's pipeline does.
                    prot["unpairedMsaPath"] = a3m
                else:
                    prot["unpairedMsa"] = ""
                prot["templates"] = [{
                    "mmcifPath": str(tpl[c]),
                    "queryIndices": list(range(len(seqs[c]))),
                    "templateIndices": list(range(len(seqs[c]))),
                }] if self.use_templates else []
                seq_entries.append({"protein": prot})
            doc = {
                "name": name,
                "sequences": seq_entries,
                "modelSeeds": [self.seed],
                "dialect": "alphafold3", "version": 2,
            }
            import json
            fold_input = folding_input.Input.from_json(json.dumps(doc))
            examples = featurisation.featurise_input(
                fold_input=fold_input, buckets=self.buckets, ccd=self._ccd, verbose=False)
            result = self._runner.run_inference(examples[0], jax.random.PRNGKey(self.seed))
            infer = self._runner.extract_inference_results(
                batch=examples[0], result=result, target_name=name)[0]

            pae = np.asarray(infer.numerical_data["full_pae"], dtype=np.float32)
            token_chain_ids = [str(t) for t in infer.metadata["token_chain_ids"]]
            struct = infer.predicted_structure

            ipae_a = self._ipae(pae, token_chain_ids, binder_chain)
            plddt_frac = self._binder_residue_plddt(struct, binder_chain)

            def _ca(chain_ids, res_ids, names, coords):
                out: dict[str, np.ndarray] = {}
                for c in chains:
                    m = (chain_ids == c) & (names == "CA")
                    out[c] = coords[m]
                return out

            pred_ca = _ca(np.asarray(struct.chain_id), np.asarray(struct.res_id),
                          np.asarray(struct.atom_name), np.asarray(struct.coords))
            des_ca = {c: arr[(arr.chain_id == c) & (arr.atom_name == "CA")].coord
                      for c in chains}
            rmsd_a = self._scrmsd(pred_ca, des_ca, binder_chain)

            ipsae_min = float("nan")
            ipsae_min_10 = float("nan")
            if self.emit_ipsae:
                # Overath's best single predictor of wet-lab success. Off by
                # default so the campaign reward path is untouched; the settings
                # sweep turns it on. Computed HERE, inside the try, because the
                # cif it needs lives in `tmp`, which `finally` deletes. Failure
                # records NaN rather than raising -- ipSAE is a readout, not a
                # gate, and must never take down a reward call.
                # BOTH cutoffs, because AF2 emits both and they are different
                # metrics: colabdesign's headline `min_ipsae` uses pae_cutoff=15.0
                # ("standard for af2", ColabDesign af/loss.py:50)
                # and `min_ipsae_10` uses 10.0 (loss.py:51). complex_ipSAE defaults
                # to 10.0, so emitting one column silently produced AF2's _10
                # variant under a name that reads like the headline -- any
                # af3folding_ipsae_min vs af2folding_min_ipsae comparison would
                # have differed by cutoff alone.
                try:
                    from proteinfoundation.metrics.ipsae import complex_ipSAE
                    # AF3's Structure exposes to_mmcif() -> str (structure.py:2971).
                    # There is NO to_mmcif_file(); calling it raised AttributeError
                    # into the bare except below, so ipSAE was NaN in every cell.
                    cif = tmp / f"{name}_pred.cif"
                    cif.write_text(infer.predicted_structure.to_mmcif())
                    pae_t = torch.from_numpy(pae)
                    ipsae_min = float(complex_ipSAE(pae_t, str(cif), pae_cutoff=15.0)["min"])
                    ipsae_min_10 = float(complex_ipSAE(pae_t, str(cif), pae_cutoff=10.0)["min"])
                except Exception as e:
                    # Never raise -- ipSAE is a readout, not a gate. But never
                    # silent either: a systematic failure would otherwise write
                    # NaN into every row of the campaign and pass validation.
                    # warnings.warn, not logger: the stdlib logger produced no
                    # output at all in the campaign job logs, which is how the
                    # to_mmcif_file AttributeError stayed invisible.
                    import warnings as _w
                    _w.warn(f"ipSAE failed for {name}: {type(e).__name__}: {e}")
                    logger.warning("ipSAE failed for %s: %s: %s",
                                   name, type(e).__name__, e)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

        # colabdesign dialect, so CONJUNCTION_GATES applies unchanged.
        losses = {
            "i_pae": ipae_a / PAE_SCALE,
            "plddt": 1.0 - plddt_frac,
            "rmsd": rmsd_a,
        }
        components = {k: torch.tensor(v, dtype=torch.float32) for k, v in losses.items()}

        if self.reward_aggregation == "min_margin":
            total = self._min_margin(components)
        else:
            total = torch.tensor(0.0, dtype=torch.float32)
            for key, weight in self.reward_weights.items():
                if weight and key not in components:
                    # weighted_sum in AF2RewardModel silently treats a missing
                    # component as zero; refuse instead, since a non-zero weight
                    # on an unemitted term is always a config bug.
                    raise KeyError(f"weight on {key!r} but AF3 does not emit it")
                total = total + components.get(key, torch.tensor(0.0)) * weight

        # The gate reads a MIXED dialect: *_i_pae and *_rmsd off the losses, but
        # *_plddt_log off the log dict (aggregate_ablation_binding.py:260-273).
        #
        # These MUST go inside the reward dict, not alongside it.
        # `_extract_reward_components` (reward_utils.py:42-61) builds the CSV
        # columns from reward_dict[REWARD_KEY] ONLY and ignores every other
        # top-level key, so passing them as **extra to standardize_reward put
        # them in the returned dict and dropped them from the CSV -- which the
        # first successful smoke run caught: three loss columns present,
        # af3folding_plddt_log and af3folding_i_ptm_log absent.
        #
        # Safe to add as components: weighted_sum iterates reward_weights (which
        # does not name them) and min_margin iterates conjunction_gates (which
        # does not either), so neither treats them as an optimization term.
        components["plddt_log"] = torch.tensor(plddt_frac, dtype=torch.float32)
        components["i_ptm_log"] = torch.tensor(
            float(infer.metadata.get("iptm", float("nan"))), dtype=torch.float32)
        components["ipae_angstrom"] = torch.tensor(ipae_a, dtype=torch.float32)
        if self.emit_ipsae:
            components["ipsae_min"] = torch.tensor(ipsae_min, dtype=torch.float32)
            components["ipsae_min_10"] = torch.tensor(ipsae_min_10, dtype=torch.float32)
        return standardize_reward(reward=components, grad={}, total_reward=total)

    def _min_margin(self, components: dict[str, torch.Tensor]) -> torch.Tensor:
        """min over gates of (threshold - loss)/threshold.

        1 at zero loss, 0 exactly at the gate, negative when it fails, so the
        reward is positive iff the sample is a strict success. Mirrors
        AF2RewardModel._min_margin_reward -- keep the two in step.
        """
        margins = []
        for name, threshold in self.conjunction_gates.items():
            if name not in components:
                raise KeyError(f"conjunction gate {name!r} not emitted by AF3RewardModel")
            margins.append((threshold - components[name]) / threshold)
        return torch.stack(margins).min()

    def cleanup(self) -> None:
        self._runner = None
        self._ccd = None
