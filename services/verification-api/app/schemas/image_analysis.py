from pydantic import BaseModel


class ImageProcessingInfo(BaseModel):
    decoded: bool
    grayscale_ready: bool


class ImageAnalysisResponse(BaseModel):
    success: bool
    width: int
    height: int
    channels: int
    content_type: str
    processing: ImageProcessingInfo