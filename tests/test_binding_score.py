"""The ipSAE scoring path, without loading a folding model.

Covers the parts that are easy to break silently: which residues end up on
which side of the interface, and whether a PAE that disagrees with the chain
layout is caught rather than scored.

A wrong interface mask does not raise. It returns a plausible number, which is
why this is a test and not a comment.
"""

import numpy as np
import pytest
import torch

from proteinfoundation.evaluate_binders import _pdb_sequences
from proteinfoundation.metrics.binding_score import interface_masks, ipsae_from_fold
from proteinfoundation.metrics.folding.base import FoldResult, chain_labels


def _atom_line(serial, atom, residue, chain, number, altloc=" "):
    return (
        f"ATOM  {serial:5d} {atom:>4s}{altloc}{residue:>3s} {chain}{number:4d}    "
        "   0.000   0.000   0.000  1.00  0.00           C\n"
    )


def test_lightweight_pdb_sequence_reader(tmp_path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(
        _atom_line(1, "CA", "ALA", "A", 1)
        + _atom_line(2, "CA", "GLY", "A", 2)
        + _atom_line(3, "CA", "MSE", "B", 1)
        + _atom_line(4, "CA", "TYR", "B", 2)
        + _atom_line(5, "CA", "TYR", "B", 2, altloc="A")
        + "END\n"
    )

    assert _pdb_sequences(pdb) == {"A": "AG", "B": "MY"}


def make_result(pae, chain_lengths, plddt=None, scalars=None, backend="esmfold2-fast"):
    total = sum(chain_lengths)
    return FoldResult(
        pae=np.asarray(pae, dtype=np.float32),
        plddt=plddt,
        cif="",
        chain_ids=chain_labels(len(chain_lengths)),
        chain_lengths=list(chain_lengths),
        scalars=scalars or {},
        backend=backend,
    )


def test_binder_is_the_first_chain_and_targets_merge():
    binder, target = interface_masks([10, 20, 5])
    assert binder.shape == target.shape == (35,)
    assert binder[:10].all() and not binder[10:].any()
    assert target[10:].all() and not target[:10].any()
    # every residue belongs to exactly one side
    assert torch.equal(binder + target, torch.ones(35))


def test_single_chain_is_rejected():
    with pytest.raises(ValueError, match="binder and at least one target"):
        interface_masks([10])


def test_pae_must_match_the_chain_layout():
    # A PAE of the wrong size means the structure and the error matrix disagree
    # about residue order. Every mask built from it would be wrong.
    with pytest.raises(ValueError, match="expected"):
        FoldResult(
            pae=np.zeros((5, 5), dtype=np.float32),
            plddt=None,
            cif="",
            chain_ids=["A", "B"],
            chain_lengths=[10, 20],
        )


def test_confident_interface_scores_above_unconfident():
    confident = make_result(np.full((30, 30), 1.0), [10, 20])
    unconfident = make_result(np.full((30, 30), 30.0), [10, 20])
    hi = ipsae_from_fold(confident, implementation="inprocess")["ipsae_min"]
    lo = ipsae_from_fold(unconfident, implementation="inprocess")["ipsae_min"]
    assert 0.0 < hi <= 1.0
    # 30 A is above the 10 A gate, so no pair enters the score
    assert lo == pytest.approx(0.0)
    assert hi > lo


def test_ipsae_min_is_never_above_ipsae_max():
    rng = np.random.default_rng(0)
    pae = rng.uniform(0.5, 20.0, size=(30, 30)).astype(np.float32)
    pae = (pae + pae.T) / 2
    scores = ipsae_from_fold(make_result(pae, [12, 18]), implementation="inprocess")
    assert scores["ipsae_min"] <= scores["ipsae_max"]
    assert scores["ipsae_min"] <= scores["ipsae_avg"] <= scores["ipsae_max"]


def test_swapping_binder_and_target_changes_the_score():
    # ipSAE is asymmetric. This is why chain roles come from the submitted
    # order rather than from whatever the structure file happens to list first.
    rng = np.random.default_rng(1)
    pae = rng.uniform(0.5, 12.0, size=(30, 30)).astype(np.float32)
    a = ipsae_from_fold(make_result(pae, [10, 20]), implementation="inprocess")
    b = ipsae_from_fold(make_result(pae, [20, 10]), implementation="inprocess")
    assert a["ipsae_min"] != pytest.approx(b["ipsae_min"])


def test_plddt_is_reported_for_the_binder_only():
    plddt = np.concatenate([np.full(10, 90.0), np.full(20, 10.0)]).astype(np.float32)
    scores = ipsae_from_fold(
        make_result(np.full((30, 30), 2.0), [10, 20], plddt=plddt), implementation="inprocess"
    )
    assert scores["binder_plddt"] == pytest.approx(90.0)
    assert scores["complex_plddt"] == pytest.approx(plddt.mean())


def test_backend_scalars_are_namespaced():
    scores = ipsae_from_fold(
        make_result(np.full((30, 30), 2.0), [10, 20], scalars={"ptm": 0.8}), implementation="inprocess"
    )
    assert "esmfold2-fast_ptm" in scores
    assert "ptm" not in scores


def test_reward_declares_its_capabilities_and_loads_lazily():
    from proteinfoundation.rewards.base_reward import BaseRewardModel
    from proteinfoundation.rewards.esmfold2_reward import ESMFold2Reward

    for attr in BaseRewardModel._REQUIRED_CAPABILITY_ATTRS:
        assert attr in ESMFold2Reward.__dict__, f"{attr} must be set on the class itself"
    reward = ESMFold2Reward(device="cpu")
    assert reward.backend._model is None, "constructing the reward must not load weights"
    reward.cleanup()
    reward.cleanup()  # idempotent


def test_unknown_implementation_is_rejected():
    with pytest.raises(ValueError, match="reference.*inprocess"):
        ipsae_from_fold(make_result(np.full((30, 30), 2.0), [10, 20]), implementation="nope")


def test_vendored_reference_script_is_present():
    # The default implementation shells out to this. If it is missing, every
    # score silently falls back to nothing -- so assert it ships.
    from proteinfoundation.metrics.ipsae_reference import IPSAE_SCRIPT

    assert IPSAE_SCRIPT.exists(), f"vendored ipsae.py missing at {IPSAE_SCRIPT}"
    assert (IPSAE_SCRIPT.parent / "LICENSE").exists(), "vendored IPSAE must ship its license"


def test_reference_output_parser_needs_a_real_header():
    from proteinfoundation.metrics.ipsae_reference import parse_output

    # The real file starts with a blank line; reading line 0 as the header
    # yields an empty result that looks like a legitimate zero score.
    good = (
        "\n"
        "Chn1 Chn2  PAE Dist  Type   ipSAE    pDockQ   Model\n"
        "A    B     10   15   asym  0.500000  0.40     m\n"
        "B    A     10   15   asym  0.300000  0.40     m\n"
        "A    B     10   15   max   0.500000  0.40     m\n"
    )
    scores = parse_output(good)
    assert scores["ipSAE_AB"] == pytest.approx(0.5)
    assert scores["ipSAE_BA"] == pytest.approx(0.3)
    assert scores["ipSAE_min"] == pytest.approx(0.3)
    assert scores["ipSAE_max"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="no 'Chn1 Chn2' header"):
        parse_output("something went wrong\n")
    with pytest.raises(ValueError, match="exactly 2 asym rows"):
        parse_output("\nChn1 Chn2  PAE Dist  Type   ipSAE  Model\nA B 10 15 asym 0.5 m\n")
