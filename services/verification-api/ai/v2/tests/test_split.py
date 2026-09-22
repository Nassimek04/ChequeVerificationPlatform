import pytest

from ai.v2.dataset import assert_exact_v1_split, WriterSplitMismatch
from ai.v2.config import V2Config, TRAIN_WRITERS, VAL_WRITERS, TEST_WRITERS


def test_default_split_is_v1():
    cfg = V2Config()
    assert_exact_v1_split(cfg)  # must not raise


def test_split_disjoint():
    cfg = V2Config()
    st, sv, ste = set(cfg.train_writers), set(cfg.val_writers), set(cfg.test_writers)
    assert st.isdisjoint(sv) and st.isdisjoint(ste) and sv.isdisjoint(ste)


def test_writer7_test_only():
    cfg = V2Config()
    assert 7 in cfg.test_writers and 7 not in cfg.train_writers and 7 not in cfg.val_writers


def test_split_covers_all_writers():
    cfg = V2Config()
    all_w = set(cfg.train_writers) | set(cfg.val_writers) | set(cfg.test_writers)
    assert all_w == set(range(1, 56))


def test_counts_match_v1():
    assert len(TRAIN_WRITERS) == 38
    assert len(VAL_WRITERS) == 8
    assert len(TEST_WRITERS) == 9


def test_tampered_split_raises():
    bad = V2Config(train_writers=tuple(sorted(set(TRAIN_WRITERS) - {1}) + [2]))
    with pytest.raises(WriterSplitMismatch):
        assert_exact_v1_split(bad)


def test_leakage_raises():
    bad = V2Config(test_writers=tuple(sorted(set(TEST_WRITERS) | {3})))
    with pytest.raises(WriterSplitMismatch):
        assert_exact_v1_split(bad)