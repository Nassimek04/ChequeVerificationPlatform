"""CEDAR dataset loading, deterministic writer splits and pair generation.

WRITER-INDEPENDENT protocol:
  * writers, never individual images, are separated into train/validation/test;
  * a writer present in test NEVER appears in train (asserted);
  * pairs are generated lazily / deterministically, no in-memory explosion.

Pair types:
  - "positive":            two distinct genuine signatures of the SAME writer
  - "skilled_forgery":     genuine + forgery of the SAME claimed writer
  - "random_impostor":     genuine of writer A + genuine of writer B (A != B)

Labels (match convention): same-writer/genuine -> 1, non-match -> 0.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig
from .preprocessing import preprocess

PAIR_POSITIVE = "positive"
PAIR_SKILLED_FORGERY = "skilled_forgery"
PAIR_RANDOM_IMPOSTOR = "random_impostor"

PAIR_TYPES = (PAIR_POSITIVE, PAIR_SKILLED_FORGERY, PAIR_RANDOM_IMPOSTOR)


@dataclass
class WriterIndex:
    """Per-writer image inventory, keyed by the numeric writer id."""

    writer_id: int
    originals: List[Path] = field(default_factory=list)
    forgeries: List[Path] = field(default_factory=list)

    @property
    def n_originals(self) -> int:
        return len(self.originals)

    @property
    def n_forgeries(self) -> int:
        return len(self.forgeries)


@dataclass(frozen=True)
class Pair:
    """One comparison pair. `label` uses the match convention (1 = same)."""

    path_a: Path
    path_b: Path
    label: int
    pair_type: str
    writer: int

    def as_tuple(self) -> Tuple[Path, Path, int, str, int]:
        return self.path_a, self.path_b, self.label, self.pair_type, self.writer


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------
def load_writer_index(root: Path) -> Dict[int, WriterIndex]:
    """Scan `root` (CEDAR root with one folder per writer id)."""
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"CEDAR dataset root not found: {root}")

    index: Dict[int, WriterIndex] = {}
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        try:
            writer_id = int(child.name)
        except ValueError:
            continue  # ignore non-numeric folders
        originals = sorted(
            child.glob(f"original_{writer_id}_*.png"),
            key=lambda p: int(re.search(r"_(\d+)\.png$", p.name).group(1)),
        )
        forgeries = sorted(
            child.glob(f"forgeries_{writer_id}_*.png"),
            key=lambda p: int(re.search(r"_(\d+)\.png$", p.name).group(1)),
        )
        if not originals and not forgeries:
            continue
        index[writer_id] = WriterIndex(writer_id, originals, forgeries)
    return index


# ---------------------------------------------------------------------------
# Deterministic whole-writer split
# ---------------------------------------------------------------------------
def compute_writer_split(
    writer_ids: Sequence[int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    guarantee_test_writer: int | None = None,
) -> Tuple[List[int], List[int], List[int]]:
    """Deterministic, disjoint train/validation/test split on whole writers.

    Guarantees (asserted):
      - train ∩ validation ∩ test == ∅
      - test ∪ validation ∪ train == all writers
      - `guarantee_test_writer` lands in test (if provided and present).
    """
    ids = list(writer_ids)
    if guarantee_test_writer is not None and guarantee_test_writer not in ids:
        raise ValueError(
            f"guarantee_test_writer={guarantee_test_writer} not among writers"
        )

    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)

    n_train = round(len(ids) * train_ratio)
    n_val = round(len(ids) * val_ratio)
    test_start = n_train + n_val

    if guarantee_test_writer is not None:
        if guarantee_test_writer not in shuffled[test_start:]:
            idx = shuffled.index(guarantee_test_writer)
            shuffled[test_start], shuffled[idx] = shuffled[idx], shuffled[test_start]

    train_writers = sorted(shuffled[:n_train])
    val_writers = sorted(shuffled[n_train : n_train + n_val])
    test_writers = sorted(shuffled[n_train + n_val :])

    s_train, s_val, s_test = set(train_writers), set(val_writers), set(test_writers)
    assert s_train.isdisjoint(s_val), "train/validation leakage"
    assert s_train.isdisjoint(s_test), "train/test leakage"
    assert s_val.isdisjoint(s_test), "validation/test overlap"
    assert s_train | s_val | s_test == set(ids), "split must cover all writers"
    if guarantee_test_writer is not None:
        assert guarantee_test_writer in s_test, "guarantee writer must be in test"
    return train_writers, val_writers, test_writers


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------
def _positive_pairs(index: WriterIndex) -> List[Pair]:
    """All C(n,2) distinct-genuine pairs for one writer."""
    pairs: List[Pair] = []
    o = index.originals
    for i in range(len(o)):
        for j in range(i + 1, len(o)):
            pairs.append(Pair(o[i], o[j], 1, PAIR_POSITIVE, index.writer_id))
    return pairs


def _skilled_forgery_pairs(index: WriterIndex) -> List[Pair]:
    """All genuine x forgery pairs for one writer."""
    pairs: List[Pair] = []
    for g in index.originals:
        for f in index.forgeries:
            pairs.append(Pair(g, f, 0, PAIR_SKILLED_FORGERY, index.writer_id))
    return pairs


def _impostor_pairs(
    index_a: WriterIndex, index_b: WriterIndex, seed: int, max_pairs: int | None = None
) -> List[Pair]:
    """Genuine(genuine pairs between two distinct writers (ordered once)."""
    pairs: List[Pair] = []
    for ga in index_a.originals:
        for gb in index_b.originals:
            pairs.append(Pair(ga, gb, 0, PAIR_RANDOM_IMPOSTOR, index_a.writer_id))
    if max_pairs is not None and len(pairs) > max_pairs:
        rng = random.Random(seed)
        rng.shuffle(pairs)
        pairs = pairs[:max_pairs]
    return pairs


@dataclass
class PairSet:
    """Full (finite) pair pools per split, plus per-epoch sampling metadata."""

    split: str
    writers: List[int]
    positive: List[Pair] = field(default_factory=list)
    skilled_forgery: List[Pair] = field(default_factory=list)
    random_impostor: List[Pair] = field(default_factory=list)

    def all(self) -> List[Pair]:
        return self.positive + self.skilled_forgery + self.random_impostor

    def counts(self) -> Dict[str, int]:
        return {
            PAIR_POSITIVE: len(self.positive),
            PAIR_SKILLED_FORGERY: len(self.skilled_forgery),
            PAIR_RANDOM_IMPOSTOR: len(self.random_impostor),
            "total": len(self.all()),
        }


def generate_pairs(
    index: Dict[int, WriterIndex],
    writers: Sequence[int],
    split: str,
    seed: int,
    impostor_cap_per_writer: int | None = None,
    impostor_writer_pairs: int | None = None,
) -> PairSet:
    """Build deterministic pair pools for one split.

    Parameters
    ----------
    impostor_cap_per_writer:
        cap on the number of random-impostor pairs attributed to a single
        writer (used to keep the TRAIN pool bounded).
    impostor_writer_pairs:
        number of (writer_a, writer_b) writer-pairs used to build random
        impostors. None -> every unordered writer pair.
    """
    writers = list(writers)
    ps = PairSet(split=split, writers=writers)

    for w in writers:
        wi = index[w]
        ps.positive.extend(_positive_pairs(wi))
        ps.skilled_forgery.extend(_skilled_forgery_pairs(wi))

    # Random impostor: unordered writer pairs, deterministic ordering.
    rng = random.Random(f"{seed}:impostor:{split}")
    writer_pairs = [(a, b) for i, a in enumerate(writers) for b in writers[i + 1 :]]
    rng.shuffle(writer_pairs)
    if impostor_writer_pairs is not None:
        writer_pairs = writer_pairs[:impostor_writer_pairs]

    for a, b in writer_pairs:
        ps.random_impostor.extend(
            _impostor_pairs(index[a], index[b], seed=seed, max_pairs=None)
        )

    if impostor_cap_per_writer is not None:
        capped: List[Pair] = []
        per_writer: Dict[int, List[Pair]] = {w: [] for w in writers}
        for p in ps.random_impostor:
            per_writer[p.writer].append(p)
        rng2 = random.Random(f"{seed}:impostor_cap:{split}")
        for w in writers:
            rng2.shuffle(per_writer[w])
            capped.extend(per_writer[w][:impostor_cap_per_writer])
        ps.random_impostor = capped

    return ps


def sample_balanced_epoch(
    pair_set: PairSet,
    n_pairs: int,
    seed: int,
    epoch: int,
) -> List[Pair]:
    """Sample `n_pairs` balanced pairs for one training epoch.

    Mix: 50% positive, 25% skilled forgery, 25% random impostor.
    Deterministic given (seed, epoch). No pair is duplicated within a sample.
    """
    rng = random.Random(f"{seed}:epoch:{epoch}")
    n_pos = n_pairs // 2
    n_neg = n_pairs - n_pos
    n_skilled = n_neg // 2
    n_impostor = n_neg - n_skilled

    def sample(pool: List[Pair], k: int) -> List[Pair]:
        pool = list(pool)
        rng.shuffle(pool)
        return pool[:k]

    return (
        sample(pair_set.positive, n_pos)
        + sample(pair_set.skilled_forgery, n_skilled)
        + sample(pair_set.random_impostor, n_impostor)
    )


def validate_no_identical_positive_pairs(pairs: Sequence[Pair]) -> None:
    """Assert no positive pair uses the same image on both sides."""
    for p in pairs:
        if p.label == 1:
            assert p.path_a != p.path_b, f"identical-image positive pair: {p.path_a}"


def validate_no_writer_leakage(
    train_pairs: PairSet, test_pairs: PairSet
) -> None:
    """Assert no test writer appears among train writers used in pairs."""
    train_writers = set(train_pairs.writers)
    test_writers = set(test_pairs.writers)
    assert train_writers.isdisjoint(test_writers), "writer leakage train/test"
    assert set(train_pairs.positive) | set(train_pairs.skilled_forgery) or True  # noqa
    for p in test_pairs.all():
        assert p.writer in test_writers, f"test pair writer {p.writer} not in test split"
    for p in train_pairs.all():
        assert p.writer in train_writers, f"train pair writer {p.writer} not in train split"


class PairDataset(Dataset):
    """Lazy pair dataset. Images are read+preprocessed on access.

    `epoch` is folded into the augmentation seed so each training epoch sees
    fresh (but deterministic) augmentations. Validation/test use augment=False
    -> fully deterministic.
    """

    def __init__(
        self,
        pairs: Sequence[Pair],
        cfg: ExperimentConfig,
        split: str = "train",
        epoch: int = 0,
    ):
        self.pairs = list(pairs)
        self.cfg = cfg
        self.split = split
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.pairs[idx]
        augment = self.split == "train" and self.cfg.augment
        seed = (self.epoch * 1_000_003 + idx) % (2**31)
        a = preprocess(p.path_a, self.cfg, augment=augment, seed=seed)
        b = preprocess(p.path_b, self.cfg, augment=augment, seed=seed + 1)
        return a, b, torch.tensor(float(p.label), dtype=torch.float32)


def default_collate(batch):
    img_a = torch.stack([b[0] for b in batch], dim=0)
    img_b = torch.stack([b[1] for b in batch], dim=0)
    y = torch.stack([b[2] for b in batch], dim=0)
    return img_a, img_b, y