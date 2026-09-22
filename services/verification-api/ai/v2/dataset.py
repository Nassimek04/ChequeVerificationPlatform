"""V2 data utilities.

Reuses the V1 CEDAR scanner (ai.dataset.load_writer_index) and enforces the
EXACT V1 writer split with hard assertions. Provides per-writer sample
inventories (needed for writer-grouped batches) and deterministic
enrollment/query splits for the multi-reference evaluation.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ai.dataset import WriterIndex, load_writer_index

from .config import V2Config


class WriterSplitMismatch(AssertionError):
    pass


def assert_exact_v1_split(cfg: V2Config) -> None:
    """Hard-assert the split is byte-for-byte the V1 writer split."""
    ref = __import__("ai.v2.config", fromlist=["TRAIN_WRITERS", "VAL_WRITERS", "TEST_WRITERS"])
    if sorted(cfg.train_writers) != sorted(ref.TRAIN_WRITERS):
        raise WriterSplitMismatch("train writers differ from V1")
    if sorted(cfg.val_writers) != sorted(ref.VAL_WRITERS):
        raise WriterSplitMismatch("val writers differ from V1")
    if sorted(cfg.test_writers) != sorted(ref.TEST_WRITERS):
        raise WriterSplitMismatch("test writers differ from V1")
    st, sv, ste = set(cfg.train_writers), set(cfg.val_writers), set(cfg.test_writers)
    assert st.isdisjoint(sv), "train ∩ val != ∅"
    assert st.isdisjoint(ste), "train ∩ test != ∅"
    assert sv.isdisjoint(ste), "val ∩ test != ∅"
    assert cfg.guarantee_test_writer in ste, "writer 7 must be test-only"
    assert st | sv | ste == set(range(1, 56)), "split must cover writers 1..55"


@dataclass
class WriterSamples:
    """Per-writer sample inventory for V2 batch construction."""

    writer_id: int
    genuine_paths: List[Path]
    forgery_paths: List[Path]

    @property
    def n_genuine(self) -> int:
        return len(self.genuine_paths)

    @property
    def n_forgery(self) -> int:
        return len(self.forgery_paths)


def build_samples(index: Dict[int, WriterIndex], writers: Sequence[int]) -> Dict[int, WriterSamples]:
    return {
        w: WriterSamples(w, list(index[w].originals), list(index[w].forgeries))
        for w in writers
    }


# ---------------------------------------------------------------------------
# Deterministic enrollment / query splits (multi-reference evaluation)
# ---------------------------------------------------------------------------
def split_enrollment_queries(
    samples: WriterSamples,
    k: int,
    seed: int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Return (reference_genuines, query_genuines, query_forgeries).

    References = the first `k` genuine signatures in deterministic filename
    order (numbering = n in original_w_n.png). Queries = remaining genuines +
    ALL forgeries. No test labels are used to pick references.
    """
    genuines = sorted(
        samples.genuine_paths,
        key=lambda p: int(re.search(r"_(\d+)\.png$", p.name).group(1)),
    )
    forgeries = sorted(
        samples.forgery_paths,
        key=lambda p: int(re.search(r"_(\d+)\.png$", p.name).group(1)),
    )
    assert k <= len(genuines), f"k={k} > genuines available for writer {samples.writer_id}"
    refs = genuines[:k]
    queries_gen = genuines[k:]
    return refs, queries_gen, forgeries


def rng_for(seed: int, *parts) -> random.Random:
    return random.Random(":".join(str(p) for p in (seed, *parts)))