import io
from typing import Optional
import pypdf
from documents.parsers.base_parser import BaseDocumentParser


class PdfParser(BaseDocumentParser):
    """
    Parser for PDF documents (.pdf) using lightweight pure-Python pypdf.
    Extracts text page-by-page and reports total page count.
    """

    def __init__(self):
        super().__init__(file_type="pdf")

    def _parse_content(self, file_bytes: bytes, filename: str) -> tuple[str, Optional[int]]:
        pdf_stream = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_stream)
        
        pages_text = []
        page_count = len(reader.pages)

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages_text.append(page_text.strip())

        extracted = "\n\n".join(pages_text).strip()
        return extracted, page_count
