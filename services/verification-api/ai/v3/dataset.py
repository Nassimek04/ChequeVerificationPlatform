"""V3 data utilities.

Reuses the exact V1/V2 writer split (hard-asserted) and V2 per-writer sample
inventories. Adds deterministic ADDITIONAL writer-independent splits used ONLY
for optional robustness checks (never for tuning the primary result).
"""

from __future__ import annotations

import random
from typing import Dict, Sequence, Tuple

from ai.dataset import WriterIndex, load_writer_index  # noqa: F401

from ai.v2.dataset import (  # noqa: F401  (reused utilities)
    WriterSplitMismatch,
    WriterSamples,
    assert_exact_v1_split,
    build_samples,
    rng_for,
    split_enrollment_queries,
)

from .config import V3Config


def extra_split(cfg: V3Config, seed: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Deterministic writer-independent split (38 train / 8 val / 9 test).

    - seeded RNG over writer ids 1..55
    - writer 7 is always forced into the test partition
    - partitions are disjoint and cover all 55 writers
    - size balance matches the primary split
    """
    rng = random.Random(seed)
    others = [w for w in range(1, 56) if w != cfg.guarantee_test_writer]
    rng.shuffle(others)
    tr = tuple(sorted(others[:38]))
    va = tuple(sorted(others[38:46]))
    te = tuple(sorted(others[46:] + [cfg.guarantee_test_writer]))
    assert set(tr).isdisjoint(set(va)) and set(tr).isdisjoint(set(te)) and set(va).isdisjoint(set(te))
    assert set(tr) | set(va) | set(te) == set(range(1, 56))
    assert cfg.guarantee_test_writer in te
    assert len(tr) == 38 and len(va) == 8 and len(te) == 9
    return tr, va, te


def build_extra_split_samples(cfg: V3Config, index, seed: int):
    """Return (train_samples, val_writers, test_writers) for an extra split."""
    tr, va, te = extra_split(cfg, seed)
    return build_samples(index, tr), va, te