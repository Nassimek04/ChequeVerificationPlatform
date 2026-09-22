from collections import Counter

from ai.config import ExperimentConfig
from ai.dataset import (
    PAIR_POSITIVE,
    PAIR_RANDOM_IMPOSTOR,
    PAIR_SKILLED_FORGERY,
    PairDataset,
    compute_writer_split,
    default_collate,
    generate_pairs,
    load_writer_index,
    sample_balanced_epoch,
    validate_no_identical_positive_pairs,
    validate_no_writer_leakage,
)


def test_dataset_scan_is_uniform(index):
    assert len(index) == 55
    for w, wi in index.items():
        assert wi.n_originals == 24
        assert wi.n_forgeries == 24


def test_writer_split_disjoint_and_covers_all(index):
    writers = sorted(index)
    tr, va, te = compute_writer_split(
        writers, ExperimentConfig().seed, 0.70, 0.15, guarantee_test_writer=7
    )
    assert set(tr).isdisjoint(set(va))
    assert set(tr).isdisjoint(set(te))
    assert set(va).isdisjoint(set(te))
    assert set(tr) | set(va) | set(te) == set(writers)
    assert 7 in te
    assert len(tr) == 38 and len(va) == 8 and len(te) == 9


def test_writer_split_deterministic(index):
    writers = sorted(index)
    a = compute_writer_split(writers, 42, 0.70, 0.15, 7)
    b = compute_writer_split(writers, 42, 0.70, 0.15, 7)
    assert a == b


def test_pair_labels_match_pair_types(index):
    cfg = ExperimentConfig()
    tr, va, te = compute_writer_split(sorted(index), cfg.seed, 0.70, 0.15, 7)
    pool = generate_pairs(index, tr, "train", cfg.seed, impostor_cap_per_writer=8, impostor_writer_pairs=4)
    assert pool.positive and pool.skilled_forgery and pool.random_impostor
    for p in pool.positive:
        assert p.label == 1 and p.pair_type == PAIR_POSITIVE
    for p in pool.skilled_forgery:
        assert p.label == 0 and p.pair_type == PAIR_SKILLED_FORGERY
    for p in pool.random_impostor:
        assert p.label == 0 and p.pair_type == PAIR_RANDOM_IMPOSTOR


def test_no_identical_image_positive_pair(index):
    cfg = ExperimentConfig()
    tr, _, _ = compute_writer_split(sorted(index), cfg.seed, 0.70, 0.15, 7)
    pool = generate_pairs(index, tr, "train", cfg.seed)
    validate_no_identical_positive_pairs(pool.positive)
    # Explicit sanity: every positive uses two distinct files of the same writer.
    for p in pool.positive:
        assert p.path_a != p.path_b
        assert p.writer == int(p.path_a.parent.name)


def test_positive_pairs_share_writer_and_are_genuine(index):
    cfg = ExperimentConfig()
    tr, _, _ = compute_writer_split(sorted(index), cfg.seed, 0.70, 0.15, 7)
    pool = generate_pairs(index, tr, "train", cfg.seed)
    for p in pool.positive:
        assert p.path_a.parent == p.path_b.parent
        assert "original_" in p.path_a.name and "original_" in p.path_b.name


def test_writer_split_leakage_none(index):
    cfg = ExperimentConfig()
    tr, va, te = compute_writer_split(sorted(index), cfg.seed, 0.70, 0.15, 7)
    train_pool = generate_pairs(index, tr, "train", cfg.seed)
    test_pool = generate_pairs(index, te, "test", cfg.seed)
    validate_no_writer_leakage(train_pool, test_pool)


def test_balanced_epoch_sample(index):
    cfg = ExperimentConfig()
    tr, _, _ = compute_writer_split(sorted(index), cfg.seed, 0.70, 0.15, 7)
    pool = generate_pairs(index, tr, "train", cfg.seed)
    pairs = sample_balanced_epoch(pool, 400, cfg.seed, epoch=3)
    c = Counter(p.pair_type for p in pairs)
    assert c[PAIR_POSITIVE] == 200
    assert c[PAIR_SKILLED_FORGERY] == 100
    assert c[PAIR_RANDOM_IMPOSTOR] == 100
    assert len(set((p.path_a, p.path_b) for p in pairs)) == 400  # no dup pairs


def test_skilled_forgery_pairs_same_writer_claim(index):
    cfg = ExperimentConfig()
    tr, _, _ = compute_writer_split(sorted(index), cfg.seed, 0.70, 0.15, 7)
    pool = generate_pairs(index, tr, "train", cfg.seed)
    for p in pool.skilled_forgery:
        assert p.path_a.parent == p.path_b.parent
        assert "forgeries_" in p.path_b.name


def test_impostor_pairs_different_writers(index):
    cfg = ExperimentConfig()
    tr, _, _ = compute_writer_split(sorted(index), cfg.seed, 0.70, 0.15, 7)
    pool = generate_pairs(index, tr, "train", cfg.seed)
    for p in pool.random_impostor:
        assert p.path_a.parent != p.path_b.parent
