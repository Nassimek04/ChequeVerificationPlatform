"""Technical image processing service.

This module decodes cheque images in memory and computes basic technical
information. It deliberately does NOT perform signature detection, cropping,
OCR or any business scoring.
"""

import cv2
import numpy as np


class ImageProcessingError(Exception):
    """Raised when an image cannot be decoded or processed."""


class ImageAnalysisResult:
    """Result of a technical image analysis."""

    def __init__(
        self,
        width: int,
        height: int,
        channels: int,
        grayscale_ready: bool,
    ) -> None:
        self.width = width
        self.height = height
        self.channels = channels
        self.grayscale_ready = grayscale_ready

    def to_dict(self) -> dict:
        return {
            "decoded": True,
            "grayscale_ready": self.grayscale_ready,
        }


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into an OpenCV BGR image.

    Raises ImageProcessingError if the bytes cannot be decoded.
    """
    if not image_bytes:
        raise ImageProcessingError("Le fichier est vide.")

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise ImageProcessingError("L'image ne peut pas être décodée.")

    return image


def analyze_image(image_bytes: bytes) -> ImageAnalysisResult:
    """Analyze a cheque image and return technical information.

    Performs: decode -> dimensions/channels -> grayscale conversion check.
    """
    image = decode_image(image_bytes)

    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1

    grayscale_ready = _can_convert_grayscale(image, channels)

    return ImageAnalysisResult(
        width=width,
        height=height,
        channels=channels,
        grayscale_ready=grayscale_ready,
    )


def _can_convert_grayscale(image: np.ndarray, channels: int) -> bool:
    try:
        if channels == 1:
            return True
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray is not None
    except cv2.error:
        return False