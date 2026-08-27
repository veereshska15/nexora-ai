import time
from abc import ABC, abstractmethod
from typing import Optional
from documents.models.document_result import DocumentExtractionResult


class BaseDocumentParser(ABC):
    """
    Abstract Base Class for format-specific document parsers in NEXORA AI.
    Provides uniform error handling, execution timing, and validation.
    """

    def __init__(self, file_type: str):
        self.file_type = file_type

    def validate_non_empty(self, file_bytes: bytes) -> None:
        """Validates that document binary payload is not empty."""
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError("Uploaded document file is empty (0 bytes).")

    @abstractmethod
    def _parse_content(self, file_bytes: bytes, filename: str) -> tuple[str, Optional[int]]:
        """
        Subclasses must implement actual format extraction.
        Returns: (extracted_text: str, page_count: Optional[int])
        """
        pass

    def extract(self, file_bytes: bytes, filename: str) -> DocumentExtractionResult:
        """
        Executes document text extraction with safety guards, timing, and error handling.
        """
        start_time = time.perf_counter()
        file_size = len(file_bytes) if file_bytes else 0

        try:
            self.validate_non_empty(file_bytes)
            extracted_text, page_count = self._parse_content(file_bytes, filename)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)

            return DocumentExtractionResult(
                filename=filename,
                file_type=self.file_type,
                file_size=file_size,
                extracted_text=extracted_text,
                character_count=len(extracted_text),
                page_count=page_count,
                extraction_success=True,
                processing_time_ms=elapsed_ms,
                error_message=None,
            )
        except Exception as err:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)
            return DocumentExtractionResult(
                filename=filename,
                file_type=self.file_type,
                file_size=file_size,
                extracted_text="",
                character_count=0,
                page_count=None,
                extraction_success=False,
                processing_time_ms=elapsed_ms,
                error_message=str(err),
            )
