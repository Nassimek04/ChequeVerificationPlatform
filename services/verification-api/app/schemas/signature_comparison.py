from pydantic import BaseModel, Field


class SignatureComparisonMetrics(BaseModel):
    """Detailed technical metrics behind the V1 similarity score."""

    mask_overlap: float = Field(ge=0.0, le=1.0)
    normalized_correlation: float = Field(ge=0.0, le=1.0)
    density_similarity: float = Field(ge=0.0, le=1.0)


class SignatureComparisonResponse(BaseModel):
    success: bool
    similarity_score: float = Field(ge=0.0, le=1.0)
    method: str
    version: str
    metrics: SignatureComparisonMetrics