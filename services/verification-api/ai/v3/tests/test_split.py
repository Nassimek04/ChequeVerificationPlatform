import hashlib

import pytest

from ai.v2.config import TRAIN_WRITERS, VAL_WRITERS, TEST_WRITERS
from ai.v3.dataset import assert_exact_v1_split, extra_split, build_extra_split_samples
from ai.v3.config import V3Config
from ai.v3.tests.helpers import load_v1_v2_hashes


def test_v3_split_matches_v1_v2(v3_cfg):
    assert sorted(v3_cfg.train_writers) == sorted(TRAIN_WRITERS)
    assert sorted(v3_cfg.val_writers) == sorted(VAL_WRITERS)
    assert sorted(v3_cfg.test_writers) == sorted(TEST_WRITERS)


def test_exact_split_assertion_passes(v3_cfg):
    assert_exact_v1_split(v3_cfg)  # raises on any mismatch


def test_split_is_disjoint_and_complete(v3_cfg):
    st, sv, ste = set(v3_cfg.train_writers), set(v3_cfg.val_writers), set(v3_cfg.test_writers)
    assert st.isdisjoint(sv) and st.isdisjoint(ste) and sv.isdisjoint(ste)
    assert st | sv | ste == set(range(1, 56))
    assert len(st) == 38 and len(sv) == 8 and len(ste) == 9


def test_writer7_is_test_only(v3_cfg):
    assert v3_cfg.guarantee_test_writer == 7
    assert 7 in v3_cfg.test_writers
    assert 7 not in v3_cfg.train_writers
    assert 7 not in v3_cfg.val_writers


def test_extra_splits_deterministic_and_valid(v3_cfg):
    for seed in v3_cfg.extra_split_seeds:
        tr, va, te = extra_split(v3_cfg, seed)
        assert sorted(tr) == sorted(set(tr)) and len(tr) == 38
        assert len(va) == 8 and len(te) == 9
        assert set(tr).isdisjoint(set(va)) and set(tr).isdisjoint(set(te)) and set(va).isdisjoint(set(te))
        assert set(tr) | set(va) | set(te) == set(range(1, 56))
        assert 7 in te
        # deterministic
        tr2, va2, te2 = extra_split(v3_cfg, seed)
        assert tr == tr2 and va == va2 and te == te2


def test_build_extra_split_samples_uses_only_split_writers(index, v3_cfg):
    seed = v3_cfg.extra_split_seeds[0]
    tr, va, te = extra_split(v3_cfg, seed)
    samples, va2, te2 = build_extra_split_samples(v3_cfg, index, seed)
    assert va2 == va and te2 == te
    assert set(samples.keys()) == set(tr)
    for w in tr:
        assert w in index
        assert len(samples[w].genuine_paths) == len(index[w].originals)
        assert len(samples[w].forgery_paths) == len(index[w].forgeries)


def test_v1_v2_checkpoints_preserved(v3_cfg):
    hashes = load_v1_v2_hashes()
    v1 = hashlib.sha256(v3_cfg.v1_checkpoint_path.read_bytes()).hexdigest()
    v2 = hashlib.sha256(v3_cfg.v2_checkpoint_path.read_bytes()).hexdigest()
    assert v1 == hashes["v1"], "V1 checkpoint modified by V3 work"
    assert v2 == hashes["v2"], "V2 checkpoint modified by V3 work"


def test_v1_v2_reports_present_and_unchanged(v3_cfg):
    from ai.config import ExperimentConfig
    from ai.v2.config import V2Config
    v1_rep = ExperimentConfig().reports_dir
    v2_rep = V2Config().reports_dir
    assert (v1_rep / "metrics.json").exists()
    assert (v2_rep / "metrics.json").exists()
    assert (v2_rep / "training_history.csv").exists()
    assert (v2_rep / "multi_reference_validation.csv").exists()