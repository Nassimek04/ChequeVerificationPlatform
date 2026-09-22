"""AI signature-verification experiment V3 (multi-reference enrollment).

V3 keeps the V1/V2 ResNet18/ImageNet/128-D backbone and exact writer split, and
attacks the two V2 limitations with:
  - improved hard-negative mining (candidate bank + scheduled phases),
  - enrollment-only calibration (z / MAD / prototype-relative / centroid),
  - multi-reference enrollment (K=5 primary) as the PRIMARY protocol,
  - security-oriented threshold policies selected on validation only.

EXPERIMENTAL ONLY. Nothing here is imported by the FastAPI service.
"""

__version__ = "0.3.0"