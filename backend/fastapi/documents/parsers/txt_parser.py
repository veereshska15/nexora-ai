from typing import Optional
from documents.parsers.base_parser import BaseDocumentParser


class TxtParser(BaseDocumentParser):
    """
    Parser for plain text documents (.txt).
    Robustly decodes UTF-8, UTF-16, and Latin-1 encodings preserving Indic & Kannada Unicode.
    """

    def __init__(self):
        super().__init__(file_type="txt")

    def _parse_content(self, file_bytes: bytes, filename: str) -> tuple[str, Optional[int]]:
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
        decoded_text = None

        for enc in encodings:
            try:
                decoded_text = file_bytes.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if decoded_text is None:
            # Fallback with replacement of un-decodable bytes
            decoded_text = file_bytes.decode("utf-8", errors="replace")

        return decoded_text.strip(), None
