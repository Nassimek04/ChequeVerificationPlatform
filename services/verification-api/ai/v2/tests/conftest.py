import os
import sys

SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

import numpy as np
import pytest
import torch

from ai.v2.config import V2Config
from ai.v2.dataset import build_samples, split_enrollment_queries
from ai.dataset import load_writer_index


def make_embeddings(device=torch.device("cpu")):
    """Craft distinct cluster embeddings for a fake batch.

    Batch layout: 2 writers x (2 genuine + 2 forged) = 8 embeddings.
    Writer 1 at angle 0, writer 2 at angle pi/2. Same-writer positives are
    close; same-writer forgeries offset slightly; other-writer genuines far.
    """
    d = torch.as_tensor([0.0, 0.02, 0.35, 0.4, 1.5, 1.52, 1.6, 1.62], device=device)
    e = torch.stack([torch.cos(d), torch.sin(d)], dim=1)
    return torch.nn.functional.normalize(e, p=2, dim=1)


@pytest.fixture(scope="session")
def v2_cfg():
    return V2Config()


@pytest.fixture(scope="session")
def index():
    cfg = V2Config()
    try:
        return load_writer_index(cfg.dataset_root)
    except FileNotFoundError:
        pytest.skip("CEDAR dataset not available")


@pytest.fixture(scope="session")
def tiny_v2_cfg():
    return V2Config(
        canvas_width=64,
        canvas_height=32,
        embedding_dim=32,
        writers_per_batch=2,
        genuines_per_writer=2,
        forgeries_per_writer=2,
        augment=False,
    )