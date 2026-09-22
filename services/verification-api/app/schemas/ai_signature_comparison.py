"""AI V2 signature-verification response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class InkBBox(BaseModel):
    """Normalized-canvas ink bounding box (diagnostic, safe information)."""

    x: int
    y: int
    width: int
    height: int


class AiSignatureComparisonResponse(BaseModel):
    """Response of the AI pair comparison endpoint.

    The score is a raw cosine similarity between L2-normalized embeddings. It
    is a technical measure of similarity, NOT a probability of authenticity, a
    probability of fraud, a banking decision or a conformity verdict. No
    decision threshold is applied here.
    """

    success: bool
    similarity_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    method: str
    version: str
    model: str
    embedding_dimension: int
    device: str
    message: str
    extracted_ink_bbox: Optional[InkBBox] = None
    reference_ink_bbox: Optional[InkBBox] = None


class AiStatusResponse(BaseModel):
    """Runtime status of the AI V2 model (no filesystem paths)."""

    enabled: bool
    loaded: bool
    model: str
    version: str
    device: str