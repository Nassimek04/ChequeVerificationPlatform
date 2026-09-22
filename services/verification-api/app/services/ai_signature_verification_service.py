"""AI V2 signature verification service (production inference).

Loads the trained AI V2 metric model (ResNet18 backbone, 128-D L2-normalized
embeddings, triplet-loss trained, writer-independent CEDAR protocol) and
exposes:

- ``encode_signature(image_bytes)`` -> 128-D L2-normalized embedding ;
- ``compare_signatures_ai(extracted_bytes, reference_bytes)`` -> raw cosine
  similarity between the two embeddings (no threshold, no probability) ;
- ``compare_against_references_ai(extracted_bytes, reference_images)`` ->
  multi-reference aggregation prepared for future use (max / mean / median /
  top2_mean / prototype), with optional enrollment z-normalization only when
  enough references are available.

Design rules
------------
- The model is loaded ONCE (lazy singleton) and reused across requests.
- Inference reuses the exact deterministic preprocessing used by AI V2
  evaluation (grayscale -> Gaussian blur 3x3 -> Otsu THRESH_BINARY_INV ->
  ink bbox crop -> aspect-preserving fit on a 256x128 canvas -> 3-channel
  broadcast -> ImageNet normalization). NO training augmentation.
- The checkpoint is validated (model version, architecture, embedding
  dimension, canvas) and a mismatched checkpoint fails explicitly.
- This module is production code: it never imports training-only logic.
- The returned score is a similarity measure, not an authenticity verdict.
"""

from __future__ import annotations

import functools
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.core.config import get_settings, Settings

logger = logging.getLogger(__name__)

try:  # torch is optional at import time: required only when the AI is enabled.
    import torch
except ImportError:  # pragma: no cover - exercised only on torch-less deployments
    torch = None


def _inference_mode(method):
    """torch.inference_mode() when torch is available, identity otherwise."""
    if torch is None:
        return method
    return torch.inference_mode()(method)

# ---------------------------------------------------------------------------
# Stable identifiers — V5-A frozen prototype
# ---------------------------------------------------------------------------
METHOD = "ai_metric"
VERSION = "v5a-phase7"
MODEL_NAME = "sig-verif-ai-v5a"
EMBEDDING_DIMENSION = 128
CANVAS_WIDTH = 256
CANVAS_HEIGHT = 128

# Expected checkpoint metadata (validated at load time) — V5-A
REQUIRED_MODEL_VERSION = "ai_metric_v5a"
REQUIRED_BACKBONE = "resnet18"
REQUIRED_CANVAS = (CANVAS_WIDTH, CANVAS_HEIGHT)

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class AIError(Exception):
    """Base error for the AI verification service."""


class AIDisabledError(AIError):
    """Raised when the AI model is disabled by configuration."""


class AICheckpointError(AIError):
    """Raised when the checkpoint is missing, corrupt or mismatched."""


class AIImageError(AIError):
    """Raised when an image cannot be decoded / contains no usable ink."""


# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_service: Optional["AISignatureVerificationService"] = None
_load_error: Optional[BaseException] = None


def _resolve_checkpoint(checkpoint: str) -> Path:
    """Resolve the checkpoint path (absolute or relative to the service root)."""
    path = Path(checkpoint)
    if not path.is_absolute():
        # app/core/config.py -> parents[2] == service root.
        root = Path(__file__).resolve().parents[2]
        path = root / path
    return path


def _resolve_device(device: str) -> Tuple[torch.device, str]:
    """Resolve the configured device; CUDA is never required."""
    if torch is None:
        raise AIError(
            "PyTorch n'est pas installé : le service IA ne peut pas être activé."
        )

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return dev, dev.type
    if device == "cuda":
        if not torch.cuda.is_available():
            raise AIError(
                "SIGNATURE_AI_DEVICE=cuda mais CUDA n'est pas disponible sur cette machine."
            )
        return torch.device("cuda"), "cuda"
    if device == "cpu":
        return torch.device("cpu"), "cpu"
    raise AIError(f"Périphérique AI inconnu : {device!r} (attendu : auto|cuda|cpu).")


def _validate_checkpoint(payload: dict) -> None:
    """Validate checkpoint metadata; fail explicitly on mismatch (V5-A)."""
    version = payload.get("model_version")
    if version != REQUIRED_MODEL_VERSION:
        raise AICheckpointError(
            f"Checkpoint incompatible : model_version attendu 'ai_metric_v5a', "
            f"reçu {version!r}."
        )

    arch = payload.get("architecture") or {}
    if arch.get("backbone") != REQUIRED_BACKBONE:
        raise AICheckpointError(
            f"Checkpoint incompatible : backbone attendu 'resnet18', reçu "
            f"{arch.get('backbone')!r}."
        )
    try:
        if int(arch.get("embedding_dim", -1)) != EMBEDDING_DIMENSION:
            raise AICheckpointError(
                f"Checkpoint incompatible : embedding_dim attendu "
                f"{EMBEDDING_DIMENSION}, reçu {arch.get('embedding_dim')!r}."
            )
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise AICheckpointError(
            f"Checkpoint incompatible : embedding_dim invalide "
            f"{arch.get('embedding_dim')!r}."
        ) from exc

    canvas = payload.get("canvas_size") or {}
    # V5-A checkpoint stores canvas in config (canvas_width/height) rather than canvas_size
    if canvas and "width" in canvas and "height" in canvas:
        try:
            canvas_size = (int(canvas.get("width", -1)), int(canvas.get("height", -1)))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise AICheckpointError(
                f"Checkpoint incompatible : canvas_size invalide {canvas!r}."
            ) from exc
    else:
        cfg = payload.get("config") or {}
        cw = cfg.get("canvas_width")
        ch = cfg.get("canvas_height")
        if cw is not None and ch is not None:
            try:
                canvas_size = (int(cw), int(ch))
            except (TypeError, ValueError) as exc:
                raise AICheckpointError(
                    f"Checkpoint incompatible : canvas invalide config {cfg!r}."
                ) from exc
        else:
            # No canvas metadata — assume required (256x128) for backward compat
            canvas_size = REQUIRED_CANVAS
    if canvas_size != REQUIRED_CANVAS:
        raise AICheckpointError(
            f"Checkpoint incompatible : canvas attendu 256x128 (WxH), reçu "
            f"{canvas_size}."
        )

    state = payload.get("model_state")
    if not isinstance(state, dict) or len(state) == 0:
        raise AICheckpointError(
            "Checkpoint invalide : 'model_state' doit être un dictionnaire non vide."
        )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class AISignatureVerificationService:
    """Lazy, cached, thread-safe wrapper around the AI V2 model."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device, self.device_str = _resolve_device(settings.signature_ai_device)
        self.checkpoint_path = _resolve_checkpoint(settings.signature_ai_checkpoint)
        self.model = None
        self.model_name = MODEL_NAME
        self.version = VERSION
        self.embedding_dim = EMBEDDING_DIMENSION

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> "AISignatureVerificationService":
        if self.model is not None:
            return self
        import torch

        if not self.checkpoint_path.exists():
            raise AICheckpointError(
                f"Checkpoint AI introuvable : {self.checkpoint_path}"
            )
        try:
            payload = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
        except Exception as exc:
            raise AICheckpointError(
                f"Impossible de charger le checkpoint AI {self.checkpoint_path} : {exc}"
            ) from exc

        _validate_checkpoint(payload)

        # Architecture-only build (pretrained=False avoids downloading ImageNet
        # weights: the trained state dict fully overwrites the initialization).
        cfg = _ModelConfig(embedding_dim=self.embedding_dim)
        model = _build_model(cfg)
        model.load_state_dict(payload["model_state"])
        model.eval()
        model.to(self.device)

        self.model = model
        logger.info(
            "AI V5-A model loaded: %s | device=%s | embedding_dim=%d | epoch=%s | sha=%s",
            self.checkpoint_path,
            self.device_str,
            self.embedding_dim,
            payload.get("training_epoch"),
            payload.get("model_version"),
        )
        return self

    # ------------------------------------------------------------------
    # Preprocessing (exact AI V2 evaluation parity, no augmentation)
    # ------------------------------------------------------------------
    def _prepare_image(self, image_bytes: bytes) -> Tuple[np.ndarray, dict]:
        """Decode + preprocess to a [1,3,256,128] ImageNet-normalized tensor.

        Returns (tensor, ink_bbox_in_canvas) mirroring ai.preprocessing:
        grayscale -> GaussianBlur(3x3) -> Otsu THRESH_BINARY (then invert, i.e.
        THRESH_BINARY_INV) -> ink bbox crop -> aspect-preserving fit on the
        256x128 canvas (never stretched, never upscaled) -> 3-channel broadcast
        -> ImageNet normalization. No augmentation in inference.
        """
        import cv2

        from ai.preprocessing import binarize_ink, crop_to_ink, fit_to_canvas

        if not image_bytes:
            raise AIImageError("Fichier image vide.")

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        gray = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise AIImageError("L'image ne peut pas être décodée.")

        binary = binarize_ink(gray)
        try:
            ink = crop_to_ink(binary)
        except ValueError as exc:
            raise AIImageError(
                "Aucune encre exploitable dans l'image (image blanche ou sans signature)."
            ) from exc

        canvas = fit_to_canvas(ink, CANVAS_WIDTH, CANVAS_HEIGHT)

        h, w = ink.shape
        scale = min(CANVAS_WIDTH / w, CANVAS_HEIGHT / h, 1.0)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        x0 = (CANVAS_WIDTH - new_w) // 2
        y0 = (CANVAS_HEIGHT - new_h) // 2
        ink_bbox = {"x": x0, "y": y0, "width": new_w, "height": new_h}

        mean = torch.tensor((0.485, 0.456, 0.406)).view(-1, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(-1, 1, 1)
        canvas = np.clip(canvas, 0.0, 1.0)
        gray_t = torch.from_numpy(canvas).float().unsqueeze(0)  # [1,H,W]
        rgb = gray_t.repeat(3, 1, 1)                            # [3,H,W]
        tensor = ((rgb - mean) / std).unsqueeze(0)              # [1,3,H,W]
        return tensor, ink_bbox

    @_inference_mode
    def encode_signature(self, image_bytes: bytes) -> np.ndarray:
        """Return the L2-normalized 128-D embedding of one signature image."""
        self._require_loaded()
        tensor, _ = self._prepare_image(image_bytes)
        embedding = self.model.encode(tensor.to(self.device))
        return embedding[0].cpu().numpy()

    def _require_loaded(self) -> None:
        if self.model is None:
            raise AIDisabledError("Modèle AI non chargé.")

    # ------------------------------------------------------------------
    # Pair comparison
    # ------------------------------------------------------------------
    def compare_signatures_ai(
        self, extracted_bytes: bytes, reference_bytes: bytes
    ) -> Dict:
        """Raw cosine similarity between two signatures (no threshold)."""
        emb_extracted = self.encode_signature(extracted_bytes)
        emb_reference = self.encode_signature(reference_bytes)
        similarity = float(np.dot(emb_extracted, emb_reference))
        similarity = float(np.clip(similarity, -1.0, 1.0))
        return {
            "similarity_score": round(similarity, 6),
            "method": METHOD,
            "version": VERSION,
            "model": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device_str,
        }

    # ------------------------------------------------------------------
    # Multi-reference support (prepared for future use)
    # ------------------------------------------------------------------
    @staticmethod
    def aggregate(similarities: Sequence[float], strategy: str) -> float:
        """Aggregate per-reference similarities with a documented strategy."""
        if not similarities:
            raise AIError("Aucun score de référence à agréger.")
        sims = list(similarities)
        if strategy == "max":
            return float(np.max(sims))
        if strategy == "mean":
            return float(np.mean(sims))
        if strategy == "median":
            return float(np.median(sims))
        if strategy == "top2_mean":
            return float(np.mean(sorted(sims, reverse=True)[:2]))
        if strategy == "prototype":
            # Prototype = mean of the reference embeddings; the query similarity
            # to the prototype is computed on the fly by the caller.
            raise AIError("'prototype' nécessite les embeddings de référence.")
        raise AIError(f"Stratégie d'agrégation inconnue : {strategy!r}")

    def _z_normalize(self, similarities: Sequence[float], refs: Sequence[np.ndarray]):
        """Enrollment z-normalization (AI V2 protocol).

        Uses the pairwise similarity statistics of the reference embeddings
        only (never query labels). Requires K >= 3 and a positive spread;
        otherwise returns the raw scores and marks calibration unavailable.
        """
        k = len(similarities)
        if k < 3:
            return list(similarities), "raw", False
        pairwise = []
        for i in range(k):
            for j in range(i + 1, k):
                pairwise.append(float(np.dot(refs[i], refs[j])))
        mean = float(np.mean(pairwise))
        std = float(np.std(pairwise))
        if std <= _EPS:
            return list(similarities), "raw", False
        normalized = [(s - mean) / std for s in similarities]
        return normalized, "z", True

    def compare_against_references_ai(
        self, extracted_bytes: bytes, reference_images: Sequence[bytes]
    ) -> Dict:
        """Compare one extracted signature against multiple references.

        Returns one raw cosine similarity per reference plus aggregation
        candidates (max/mean/median/top2_mean). No final decision.
        """
        emb_extracted = self.encode_signature(extracted_bytes)
        ref_embs = [self.encode_signature(b) for b in reference_images]
        similarities = [
            float(np.clip(float(np.dot(emb_extracted, r)), -1.0, 1.0))
            for r in ref_embs
        ]

        scores, normalization, cal_available = self._z_normalize(
            similarities, ref_embs
        )
        aggregations = {}
        for strategy in ("max", "mean", "median", "top2_mean"):
            aggregations[strategy] = round(self.aggregate(scores, strategy), 6)

        return {
            "K": len(similarities),
            "similarities": [round(s, 6) for s in similarities],
            "aggregations": aggregations,
            "normalization": normalization,
            "calibration_available": cal_available,
            "method": METHOD,
            "version": VERSION,
            "model": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device_str,
        }


# ---------------------------------------------------------------------------
# Minimal model config (production-safe, no training settings)
# ---------------------------------------------------------------------------
class _ModelConfig:
    """Minimal dataclass-like config accepted by the shared model builder."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIMENSION) -> None:
        self.pretrained = False  # architecture only; state dict is loaded after
        self.embedding_dim = embedding_dim
        self.canvas_width = CANVAS_WIDTH
        self.canvas_height = CANVAS_HEIGHT


def _build_model(cfg):
    """Build the exact ResNet18 metric model (shared stable code)."""
    from ai.model import SiameseResNet18

    return SiameseResNet18(cfg)


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------
def get_ai_service(settings: Optional[Settings] = None) -> AISignatureVerificationService:
    """Return the loaded AI service (cached); raise a controlled AIError."""
    global _service, _load_error

    if _service is not None:
        return _service
    if _load_error is not None:
        raise AIDisabledError(
            f"Service IA indisponible (échec de chargement précédent) : {_load_error}"
        ) from _load_error

    settings = settings or get_settings()
    if not settings.signature_ai_enabled:
        _load_error = AIDisabledError(
            "Le modèle IA est désactivé (SIGNATURE_AI_ENABLED=false)."
        )
        raise AIDisabledError(
            "Le modèle IA est désactivé (SIGNATURE_AI_ENABLED=false)."
        )

    with _lock:
        if _service is not None:
            return _service
        if _load_error is not None:
            raise AIDisabledError(
                f"Service IA indisponible (échec de chargement précédent) : {_load_error}"
            ) from _load_error
        try:
            service = AISignatureVerificationService(settings)
            service.load()
            _service = service
            _load_error = None
            return service
        except AIError as exc:
            logger.error("AI model load failed (controlled): %s", exc)
            _load_error = exc
            raise exc
        except Exception as exc:  # unexpected -> controlled failure, OpenCV stays up
            logger.exception("Unexpected AI model load failure (controlled)")
            _load_error = exc
            raise AIDisabledError(
                f"Service IA indisponible : {exc}"
            ) from exc


def reset_ai_service() -> None:
    """Testing helper: drop the cached service so it can be reloaded."""
    global _service, _load_error
    _service = None
    _load_error = None