"""Experimental signature comparison service (V1, OpenCV baseline).

This module compares a *query* signature (typically the crop extracted from a
cheque) with a *reference* signature (a customer's reference). It produces a
normalized technical similarity score in [0, 1].

IMPORTANT:
- The score is an experimental measure of VISUAL similarity.
- It is NOT a probability of authenticity, a probability of fraud, a banking
  decision or an automatic conformity verdict.
- No ML model, no PyTorch/TensorFlow, no OCR is used: everything is a
  deterministic OpenCV / NumPy baseline.
- No threshold is applied here; conformity is out of scope.
"""

import cv2
import numpy as np

METHOD = "opencv_baseline"
VERSION = "v1"

# Shift range (in pixels) explored when looking for the best ink-mask overlap.
# Tolerates small translations between the two normalized signatures.
_BEST_ALIGNMENT_SHIFT = 4

# Weights of the similarity metrics. Justification in _compare_masks.
_WEIGHT_OVERLAP = 0.50
_WEIGHT_CORRELATION = 0.35
_WEIGHT_DENSITY = 0.15

# If more than this fraction of the binarized image is foreground, the input is
# considered blank (Otsu cannot separate ink on a uniformly bright/dark image).
_BLANK_FOREGROUND_RATIO = 0.90


class SignatureComparisonError(Exception):
    """Raised when a signature cannot be compared."""


class SignatureDecodeError(SignatureComparisonError):
    """Raised when the uploaded bytes are not a decodable image."""


class SignatureComparisonResult:
    """Result of a comparison between two signatures."""

    def __init__(
        self,
        similarity_score: float,
        mask_overlap: float,
        normalized_correlation: float,
        density_similarity: float,
    ) -> None:
        self.similarity_score = similarity_score
        self.mask_overlap = mask_overlap
        self.normalized_correlation = normalized_correlation
        self.density_similarity = density_similarity


def _decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise SignatureComparisonError("Le fichier est vide.")

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise SignatureDecodeError("L'image ne peut pas être décodée.")

    return image


def _binarize_ink(image: np.ndarray) -> np.ndarray:
    """Grayscale + Otsu + THRESH_BINARY_INV => ink (dark) is foreground (255)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    height, width = binary.shape
    foreground_ratio = int(np.count_nonzero(binary)) / (height * width)

    if foreground_ratio > _BLANK_FOREGROUND_RATIO:
        raise SignatureComparisonError(
            "Aucune signature détectée (image blanche ou sans encre exploitable)."
        )

    return binary


def _trim_to_ink(binary: np.ndarray) -> np.ndarray:
    """Crop the binary image to its ink bounding box (removes margins)."""
    ys, xs = np.where(binary > 0)
    if ys.size == 0:
        raise SignatureComparisonError("Aucune encre détectée dans la signature.")

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return binary[y0:y1, x0:x1]


def _fit_to_canvas(ink: np.ndarray, canvas_width: int, canvas_height: int) -> np.ndarray:
    """Resize preserving the aspect ratio, then center on a fixed black canvas.

    The signature is NOT distorted: only the largest dimension is fitted and the
    other axis is computed from the original aspect ratio. Padding is added
    around the ink so both signatures share the same coordinate frame.
    """
    height, width = ink.shape
    scale = min(canvas_width / width, canvas_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(ink, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # cv2.resize with INTER_AREA can introduce intermediate grayscale values
    # when downscaling: re-threshold to keep a strict binary mask.
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    x_offset = (canvas_width - new_width) // 2
    y_offset = (canvas_height - new_height) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return canvas


def normalize_signature(
    image_bytes: bytes, canvas_width: int, canvas_height: int
) -> np.ndarray:
    """Apply the exact same preprocessing to any signature image.

    Pipeline (identical for the query and the reference):
    1. decode (OpenCV) ;
    2. grayscale + light blur ;
    3. Otsu + THRESH_BINARY_INV binarization (ink = foreground) ;
    4. reject blank images (foreground ratio too high => Otsu failed) ;
    5. trim to the ink bounding box (remove unnecessary margins) ;
    6. resize preserving the aspect ratio ;
    7. center on a fixed-size canvas.

    Returns a binary NumPy array (0 = background, 255 = ink).
    """
    image = _decode_image(image_bytes)
    binary = _binarize_ink(image)
    ink = _trim_to_ink(binary)
    return _fit_to_canvas(ink, canvas_width, canvas_height)


def _compare_masks(
    mask_a: np.ndarray, mask_b: np.ndarray
) -> tuple[float, float, float, float]:
    """Compute the documented similarity metrics between two normalized masks.

    Metrics (all normalized in [0, 1]):

    - ``mask_overlap``: IoU of the ink masks, maximized over small translations
      of ``mask_b`` (shift range ``_BEST_ALIGNMENT_SHIFT`` px). Tolerates small
      translations and captures shape overlap.
    - ``normalized_correlation``: mean-subtracted normalized cross-correlation
      (``cv2.TM_CCOEFF_NORMED``) at zero shift. Measures consistent stroke
      structure; clamped to [0, 1].
    - ``density_similarity``: ``1 - |density_a - density_b|`` where density is
      the fraction of ink pixels. Robust to moderate stroke-thickness and scale
      differences.

    Final score (documented weights):
        score = 0.50 * overlap + 0.35 * correlation + 0.15 * density

    - overlap is the primary metric (shape + alignment) ;
    - correlation is secondary (complementary structural signal) ;
    - density has the smallest weight: ink quantity should not dominate.

    The combination is deliberately conservative: a single metric can never
    drive the score above 0.85 on its own.
    """
    a = mask_a > 0
    b = mask_b > 0

    density_a = float(np.mean(a))
    density_b = float(np.mean(b))
    density_similarity = max(0.0, min(1.0, 1.0 - abs(density_a - density_b)))

    # Best-alignment IoU over small translations.
    best_iou = 0.0
    height, width = mask_a.shape
    for dy in range(-_BEST_ALIGNMENT_SHIFT, _BEST_ALIGNMENT_SHIFT + 1):
        for dx in range(-_BEST_ALIGNMENT_SHIFT, _BEST_ALIGNMENT_SHIFT + 1):
            shifted = np.zeros_like(b)
            src_y0 = max(0, -dy)
            dst_y0 = max(0, dy)
            src_x0 = max(0, -dx)
            dst_x0 = max(0, dx)
            h = height - abs(dy)
            w = width - abs(dx)
            if h <= 0 or w <= 0:
                continue
            shifted[dst_y0 : dst_y0 + h, dst_x0 : dst_x0 + w] = b[
                src_y0 : src_y0 + h, src_x0 : src_x0 + w
            ]

            inter = int(np.count_nonzero(a & shifted))
            union = int(np.count_nonzero(a | shifted))
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou

    correlation = float(cv2.matchTemplate(mask_a, mask_b, cv2.TM_CCOEFF_NORMED).max())
    correlation = max(0.0, min(1.0, correlation))

    score = (
        _WEIGHT_OVERLAP * best_iou
        + _WEIGHT_CORRELATION * correlation
        + _WEIGHT_DENSITY * density_similarity
    )
    score = round(max(0.0, min(1.0, score)), 6)

    return score, best_iou, correlation, density_similarity


def compare_signatures(
    extracted_bytes: bytes,
    reference_bytes: bytes,
    canvas_width: int,
    canvas_height: int,
) -> SignatureComparisonResult:
    """Compare an extracted signature with a reference signature.

    Both images go through the exact same normalization before being compared.
    Raises SignatureComparisonError on invalid / blank input.
    """
    mask_extracted = normalize_signature(extracted_bytes, canvas_width, canvas_height)
    mask_reference = normalize_signature(reference_bytes, canvas_width, canvas_height)

    score, overlap, correlation, density = _compare_masks(mask_extracted, mask_reference)

    return SignatureComparisonResult(
        similarity_score=score,
        mask_overlap=round(overlap, 6),
        normalized_correlation=round(correlation, 6),
        density_similarity=round(density, 6),
    )