from typing import Optional
from pydantic import BaseModel, Field


class DocumentExtractionResult(BaseModel):
    """
    Standardized result model for document text extraction across all supported formats (TXT, PDF, DOCX, CSV).
    """
    filename: str = Field(..., description="Name of the uploaded document file")
    file_type: str = Field(..., description="Normalized document type/format (txt, pdf, docx, csv)")
    file_size: int = Field(..., ge=0, description="Size of the document in bytes")
    extracted_text: str = Field(..., description="Raw text extracted from the document")
    character_count: int = Field(..., ge=0, description="Total number of characters extracted")
    page_count: Optional[int] = Field(None, description="Total number of pages if applicable (e.g., PDF)")
    extraction_success: bool = Field(True, description="Whether text extraction completed successfully")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Extraction processing time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error diagnostics if extraction failed")
