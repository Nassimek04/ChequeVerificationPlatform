"""V3 model — backbone intentionally unchanged from V1/V2.

ResNet18, ImageNet pretrained, 128-D L2-normalized embeddings, 256x128
preprocessing. Any improvement must come from mining + calibration.
"""

from __future__ import annotations

from ai.model import SiameseResNet18  # noqa: F401
from ai.model import build_model, cosine_similarity, euclidean_distance  # noqa: F401