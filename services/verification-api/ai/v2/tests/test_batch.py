import numpy as np
import pytest
import torch

from ai.v2.mining import build_batch
from ai.v2.dataset import WriterSamples, build_samples, split_enrollment_queries, rng_for
from ai.v2.config import V2Config


@pytest.fixture
def tiny_index():
    """Synthetic 4-writer index: 3 genuines + 3 forgeries each."""
    from ai.dataset import WriterIndex
    from pathlib import Path
    idx = {}
    for w in range(1, 5):
        g = [Path(f"original_{w}_{i}.png") for i in range(1, 4)]
        f = [Path(f"forgeries_{w}_{i}.png") for i in range(1, 4)]
        idx[w] = WriterIndex(w, originals=g, forgeries=f)
    return idx


@pytest.fixture
def tiny_cfg(tmp_path):
    return V2Config(
        canvas_width=64,
        canvas_height=32,
        embedding_dim=32,
        augment=False,
        seed=42,
        train_writers=(1, 2, 3, 4),
        writers_per_batch=2,
        genuines_per_writer=2,
        forgeries_per_writer=2,
        batches_per_epoch=2,
    )


@pytest.fixture
def tiny_samples(tiny_cfg, tmp_path):
    import cv2
    rng = np.random.default_rng(0)
    sm = {}
    for w in range(1, 5):
        d = tmp_path / str(w)
        d.mkdir()
        g = [d / f"original_{w}_{i}.png" for i in range(1, 4)]
        f = [d / f"forgeries_{w}_{i}.png" for i in range(1, 4)]
        for p in g + f:
            img = np.full((64, 128), 255, dtype=np.uint8)
            y = rng.uniform(20, 45)
            for x in range(0, 128, 5):
                y = int(np.clip(y + rng.normal(0, 4), 10, 54))
                cv2.line(img, (x, y), (x + 4, y), 0, 3)
            cv2.imwrite(str(p), img)
        sm[w] = WriterSamples(w, g, f)
    return sm


def test_build_samples_counts(tiny_index):
    sm = build_samples(tiny_index, [1, 2, 3])
    assert set(sm) == {1, 2, 3}
    assert sm[1].n_genuine == 3
    assert sm[1].n_forgery == 3


def test_build_batch_shapes_and_composition(tiny_cfg, tiny_samples):
    images, writers, is_genuine = build_batch(tiny_samples, tiny_cfg, epoch=1, batch_idx=0)
    n_w = tiny_cfg.writers_per_batch
    assert images.shape == (n_w * (tiny_cfg.genuines_per_writer + tiny_cfg.forgeries_per_writer),
                            3, tiny_cfg.canvas_height, tiny_cfg.canvas_width)
    assert len(writers) == images.shape[0]
    assert len(is_genuine) == images.shape[0]
    uniq, counts = np.unique(writers, return_counts=True)
    assert set(uniq.tolist()).issubset({1, 2, 3, 4})
    assert (counts == tiny_cfg.genuines_per_writer + tiny_cfg.forgeries_per_writer).all()
    for w in uniq:
        mask = writers == w
        assert int(is_genuine[mask].sum()) == tiny_cfg.genuines_per_writer


def test_build_batch_deterministic_per_seed(tiny_cfg, tiny_samples):
    im1, w1, g1 = build_batch(tiny_samples, tiny_cfg, epoch=1, batch_idx=0)
    im2, w2, g2 = build_batch(tiny_samples, tiny_cfg, epoch=1, batch_idx=0)
    assert (w1 == w2).all()
    assert (g1 == g2).all()
    # augment enabled -> different epochs give different pixels
    aug_cfg = V2Config(
        seed=42, train_writers=(1, 2, 3, 4), writers_per_batch=2,
        genuines_per_writer=2, forgeries_per_writer=2, batches_per_epoch=2,
        canvas_width=64, canvas_height=32, augment=True)
    im3, _, _ = build_batch(tiny_samples, aug_cfg, epoch=2, batch_idx=0)
    assert not torch.equal(im1, im3)


def test_rng_for_deterministic():
    r1 = rng_for(42, "batch", 1, 0)
    r2 = rng_for(42, "batch", 1, 0)
    assert r1.sample(range(100), 10) == r2.sample(range(100), 10)
    r3 = rng_for(42, "batch", 1, 1)
    assert r1.sample(range(100), 10) != r3.sample(range(100), 10)


def test_split_enrollment_queries():
    from pathlib import Path
    sm = WriterSamples(7, [Path(f"original_7_{i}.png") for i in range(1, 25)],
                       [Path(f"forgeries_7_{i}.png") for i in range(1, 25)])
    refs, qg, qf = split_enrollment_queries(sm, 5, 42)
    assert len(refs) == 5 and refs[0].name == "original_7_1.png"
    assert len(qg) == 19
    assert len(qf) == 24