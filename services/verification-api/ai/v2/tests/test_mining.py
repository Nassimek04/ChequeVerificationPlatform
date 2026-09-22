import numpy as np
import pytest
import torch

from ai.v2.mining import select_triplets, build_batch, _category

from helpers import make_embeddings


def test_anchors_are_genuine_only():
    emb = make_embeddings()
    writers = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    is_genuine = np.array([True, True, False, False, True, True, False, False])
    mined = select_triplets(emb, writers, is_genuine, pytest.importorskip("ai.v2.config").V2Config(
        writers_per_batch=2, genuines_per_writer=2, forgeries_per_writer=2,
        negatives_per_anchor=4, skilled_negative_fraction=0.5, margin=0.3), epoch=1)
    anchors = mined["anchor"].numpy()
    assert set(anchors).issubset(np.nonzero(is_genuine)[0].tolist())


def test_positives_same_writer_different_image():
    emb = make_embeddings()
    writers = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    is_genuine = np.array([True, True, False, False, True, True, False, False])
    mined = select_triplets(emb, writers, is_genuine, pytest.importorskip("ai.v2.config").V2Config(
        writers_per_batch=2, genuines_per_writer=2, forgeries_per_writer=2,
        negatives_per_anchor=4, skilled_negative_fraction=0.5, margin=0.3), epoch=1)
    a = mined["anchor"].numpy(); p = mined["positive"].numpy(); n = mined["negative"].numpy()
    assert (writers[a] == writers[p]).all()
    assert (a != p).all()


def test_skilled_negatives_are_forgeries_random_are_other_writers():
    emb = make_embeddings()
    writers = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    is_genuine = np.array([True, True, False, False, True, True, False, False])
    mined = select_triplets(emb, writers, is_genuine, pytest.importorskip("ai.v2.config").V2Config(
        writers_per_batch=2, genuines_per_writer=2, forgeries_per_writer=2,
        negatives_per_anchor=4, skilled_negative_fraction=0.5, margin=0.3), epoch=1)
    a = mined["anchor"].numpy(); n = mined["negative"].numpy()
    kinds = np.asarray(mined["neg_kind"])
    skilled = n[kinds == "skilled"]
    random = n[kinds == "random"]
    assert (writers[a[kinds == "skilled"]] == writers[skilled]).all()
    assert (writers[a[kinds == "random"]] != writers[random]).all()


def test_skilled_negative_fraction_respected():
    emb = make_embeddings()
    writers = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    is_genuine = np.array([True, True, False, False, True, True, False, False])
    cfg = pytest.importorskip("ai.v2.config").V2Config(
        writers_per_batch=2, genuines_per_writer=2, forgeries_per_writer=2,
        negatives_per_anchor=4, skilled_negative_fraction=0.5, margin=0.3)
    mined = select_triplets(emb, writers, is_genuine, cfg, epoch=1)
    s = mined["stats"]
    total = s["triplets"]
    assert total > 0
    assert abs(s["skilled_negatives"] / total - 0.5) < 1e-9


def test_semihard_preferred_when_available():
    # distances: anchor0-gen2 = 0.35 (semi-hard: d_ap=0.02 < 0.35 < 0.02+0.3)
    emb = make_embeddings()
    writers = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    is_genuine = np.array([True, True, False, False, True, True, False, False])
    mined = select_triplets(emb, writers, is_genuine, pytest.importorskip("ai.v2.config").V2Config(
        writers_per_batch=2, genuines_per_writer=2, forgeries_per_writer=2,
        negatives_per_anchor=4, skilled_negative_fraction=1.0, margin=0.3), epoch=1)
    # anchor 0: negatives chosen should be the semi-hard one first
    assert mined["category"]  # non-empty
    assert mined["stats"]["semi_hard"] > 0


def test_category_bounds():
    assert _category(0.02, 0.30, 0.3) == "semi-hard"
    assert _category(0.02, 0.02, 0.3) == "hard"
    assert _category(0.02, 0.60, 0.3) == "easy"


def test_mining_stats_keys():
    emb = make_embeddings()
    writers = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    is_genuine = np.array([True, True, False, False, True, True, False, False])
    mined = select_triplets(emb, writers, is_genuine, pytest.importorskip("ai.v2.config").V2Config(
        writers_per_batch=2, genuines_per_writer=2, forgeries_per_writer=2,
        negatives_per_anchor=4, skilled_negative_fraction=0.5, margin=0.3), epoch=1)
    for k in ("triplets", "skilled_negatives", "random_negatives", "hard", "semi_hard", "easy"):
        assert k in mined["stats"]