from typing import List, Optional

from pydantic import BaseModel, Field


class OcrLine(BaseModel):
    text: str = Field(description="Recognized text for this line")
    confidence: float = Field(ge=0.0, le=1.0, description="OCR confidence [0,1]")
    box: List[List[float]] = Field(default_factory=list, description="4-point polygon [[x,y],...]")


class OcrFields(BaseModel):
    cheque_number: Optional[str] = None
    date: Optional[str] = None
    amount_text: Optional[str] = None
    amount_numeric: Optional[str] = None
    account_number: Optional[str] = None
    # CMC7 constrained parse (additive; None when unavailable or invalid).
    cmc7_raw: Optional[str] = None
    cmc7_cheque_number: Optional[str] = None
    cmc7_account_number: Optional[str] = None
    cmc7_valid: bool = False
    cmc7_error: Optional[str] = None
    cmc7_trailing_noise: bool = False
    printed_cheque_number: Optional[str] = None
    printed_account_number: Optional[str] = None
    cmc7_cross_check: Optional[str] = None
    is_probable_verso: bool = False


class ChequeOcrResponse(BaseModel):
    success: bool
    message: str
    full_text: str = Field(description="Concatenated OCR text")
    lines: List[OcrLine] = Field(default_factory=list)
    fields: OcrFields = Field(default_factory=OcrFields)
    processing_ms: int = Field(description="Processing time in ms")
    lang: str = Field(description="Language used, e.g. 'fr'")
    device: str = Field(description="Device, e.g. 'cpu'")
