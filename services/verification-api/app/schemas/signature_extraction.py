from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int = Field(ge=0)
    height: int = Field(ge=0)


class SignatureExtractionResponse(BaseModel):
    success: bool
    original_width: int
    original_height: int
    candidate_roi: BoundingBox
    signature_bbox: BoundingBox
    extraction_quality: float = Field(ge=0.0, le=1.0)
    image_format: str
    signature_image_base64: str
    # Hybrid localization: "roi" (expected cheque ROI) or "global_fallback"
    # (full-document fallback). signature_bbox is expressed in candidate_roi
    # coordinates (ROI-local, or absolute when the ROI is the full document).
    localization_mode: str = "roi"
    # Full-document fallback evidence (0 / -1 / 0.0 / "" when the ROI won).
    fallback_candidate_count: int = 0
    fallback_selected_index: int = -1
    fallback_selected_score: float = 0.0
    fallback_selection_reason: str = ""
    final_crop_width: int = 0
    final_crop_height: int = 0