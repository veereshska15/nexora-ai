import os
import time
from typing import Dict, Optional
from core.config import settings
from documents.models.document_result import DocumentExtractionResult
from documents.parsers.base_parser import BaseDocumentParser
from documents.parsers.txt_parser import TxtParser
from documents.parsers.pdf_parser import PdfParser
from documents.parsers.docx_parser import DocxParser
from documents.parsers.csv_parser import CsvParser


class DocumentIngestionService:
    """
    Centralized Multilingual Document Ingestion Service for NEXORA AI.
    Handles file validation, size limits, format dispatching, and text extraction.
    """

    def __init__(self):
        self._parsers: Dict[str, BaseDocumentParser] = {
            ".txt": TxtParser(),
            ".pdf": PdfParser(),
            ".docx": DocxParser(),
            ".csv": CsvParser(),
        }

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitizes filename by stripping directory paths and null bytes."""
        clean_name = os.path.basename(filename or "document")
        return clean_name.replace("\x00", "").strip()

    def validate_file(self, file_bytes: bytes, filename: str) -> str:
        """
        Validates file size, non-emptiness, and allowed file extension.
        Returns the sanitized normalized extension.
        """
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError("Uploaded document file is empty (0 bytes).")

        if len(file_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
            max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
            raise ValueError(
                f"File size exceeds maximum allowed upload limit of {max_mb:.1f} MB (received {len(file_bytes)} bytes)."
            )

        sanitized_name = self._sanitize_filename(filename)
        _, ext = os.path.splitext(sanitized_name)
        norm_ext = ext.lower().strip()

        if not norm_ext or norm_ext not in settings.ALLOWED_DOCUMENT_EXTENSIONS:
            allowed = ", ".join(settings.ALLOWED_DOCUMENT_EXTENSIONS)
            raise ValueError(
                f"Unsupported file format '{norm_ext}'. Allowed formats are: {allowed}"
            )

        return norm_ext

    def extract_from_bytes(self, file_bytes: bytes, filename: str) -> DocumentExtractionResult:
        """
        Validates and extracts raw text from document bytes.
        """
        start_time = time.perf_counter()
        clean_filename = self._sanitize_filename(filename)

        try:
            norm_ext = self.validate_file(file_bytes, clean_filename)
            parser = self._parsers[norm_ext]
            return parser.extract(file_bytes, clean_filename)
        except Exception as err:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)
            _, ext = os.path.splitext(clean_filename)
            file_type = ext.replace(".", "").lower() if ext else "unknown"

            return DocumentExtractionResult(
                filename=clean_filename,
                file_type=file_type,
                file_size=len(file_bytes) if file_bytes else 0,
                extracted_text="",
                character_count=0,
                page_count=None,
                extraction_success=False,
                processing_time_ms=elapsed_ms,
                error_message=str(err),
            )


document_service = DocumentIngestionService()
