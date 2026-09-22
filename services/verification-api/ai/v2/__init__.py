"""AI signature-verification experiment V2 (metric learning).

Replaces the V1 contrastive objective with a Triplet loss + hard/semi-hard
negative mining targeting SKILLED FORGERIES, keeping the ResNet18 backbone and
the exact V1 writer split unchanged. V1 is intentionally left untouched
(checkpoint, reports, source modules).

EXPERIMENTAL ONLY. Nothing here is imported by the FastAPI service.
"""

__version__ = "0.2.0"