import csv
import io
from typing import Optional
from documents.parsers.base_parser import BaseDocumentParser


class CsvParser(BaseDocumentParser):
    """
    Parser for Comma-Separated Values documents (.csv).
    Decodes rows into structured tabular text format preserving Indic text.
    """

    def __init__(self):
        super().__init__(file_type="csv")

    def _parse_content(self, file_bytes: bytes, filename: str) -> tuple[str, Optional[int]]:
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        text_content = None

        for enc in encodings:
            try:
                text_content = file_bytes.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if text_content is None:
            text_content = file_bytes.decode("utf-8", errors="replace")

        # Parse CSV rows
        f = io.StringIO(text_content)
        reader = csv.reader(f)
        
        rows = []
        for row in reader:
            if any(field.strip() for field in row):
                rows.append(" | ".join(field.strip() for field in row))

        extracted = "\n".join(rows).strip()
        return extracted, None
