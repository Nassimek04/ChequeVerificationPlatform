import os
import sys

SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

import pytest
import torch

from ai.v3.config import V3Config
from ai.dataset import load_writer_index


@pytest.fixture(scope="session")
def v3_cfg():
    return V3Config()


@pytest.fixture(scope="session")
def index():
    cfg = V3Config()
    try:
        return load_writer_index(cfg.dataset_root)
    except FileNotFoundError:
        pytest.skip("CEDAR dataset not available")


@pytest.fixture(scope="session")
def tiny_v3_cfg():
    return V3Config(
        canvas_width=64,
        canvas_height=32,
        embedding_dim=32,
        writers_per_batch=2,
        genuines_per_writer=4,
        forgeries_per_writer=4,
        negatives_per_anchor=6,
        augment=False,
        epochs=20,
    )


@pytest.fixture(scope="session")
def cpu():
    return torch.device("cpu")