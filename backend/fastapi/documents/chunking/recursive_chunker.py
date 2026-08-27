from typing import Any, Dict, List, Optional
from documents.chunking.base_chunker import BaseChunker
from documents.chunking.models.chunk_result import Chunk


class CharacterChunker(BaseChunker):
    """
    Fixed-window character chunker with overlap and Unicode grapheme cluster boundary safety.
    """

    def __init__(self):
        super().__init__(strategy_name="character")

    def split(
        self,
        text: str,
        document_id: str,
        document_name: str,
        chunk_size: int,
        chunk_overlap: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        self.validate_params(text, chunk_size, chunk_overlap)
        base_meta = metadata or {}
        lang = base_meta.get("language", "en")

        chunks: List[Chunk] = []
        total_len = len(text)
        step = max(1, chunk_size - chunk_overlap)
        start = 0
        chunk_idx = 0

        while start < total_len:
            raw_end = min(start + chunk_size, total_len)
            # Adjust end boundary so we don't break Indic combining marks or Ottaksharas
            end = self.adjust_for_unicode_graphemes(text, raw_end)
            if end <= start:
                end = min(start + chunk_size, total_len)

            chunk_text = text[start:end]
            if chunk_text.strip():
                c_id = f"{document_id}_chunk_{chunk_idx}"
                chunks.append(
                    Chunk(
                        chunk_id=c_id,
                        document_id=document_id,
                        document_name=document_name,
                        chunk_index=chunk_idx,
                        content=chunk_text,
                        character_count=len(chunk_text),
                        token_count=self.calculate_tokens(chunk_text, lang),
                        start_offset=start,
                        end_offset=end,
                        metadata=base_meta.copy(),
                    )
                )
                chunk_idx += 1

            if end >= total_len:
                break

            start = start + step
            # Ensure forward progress
            if chunks and start <= chunks[-1].start_offset:
                start = chunks[-1].start_offset + 1

        return chunks


class RecursiveChunker(BaseChunker):
    """
    Semantic Recursive Text Splitter for Indic and Latin multilingual text.
    Recursively breaks down text by hierarchy of separators:
    1. Paragraphs ("\n\n")
    2. Lines ("\n")
    3. Indic Sentence Dandas ("।", "॥")
    4. Western Sentence Endings (". ", "! ", "? ")
    5. Clauses ("; ", ", ")
    6. Words (" ")
    7. Characters ("")
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", "।", "॥", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(self, separators: Optional[List[str]] = None):
        super().__init__(strategy_name="recursive")
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _merge_pieces(self, pieces: List[str], separator: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Combines smaller pieces into chunks up to chunk_size with overlap."""
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for piece in pieces:
            piece_clean = piece.strip()
            if not piece_clean:
                continue

            piece_len = len(piece_clean)
            sep_len = len(separator) if current_chunk else 0

            if current_len + sep_len + piece_len > chunk_size and current_chunk:
                merged = separator.join(current_chunk).strip()
                if merged:
                    chunks.append(merged)
                # Keep trailing pieces for overlap
                while current_len > chunk_overlap and len(current_chunk) > 1:
                    popped = current_chunk.pop(0)
                    current_len -= len(popped) + len(separator)

                if current_chunk and (current_len + len(separator) + piece_len > chunk_size):
                    current_chunk = []
                    current_len = 0

            current_chunk.append(piece_clean)
            current_len += piece_len + (len(separator) if len(current_chunk) > 1 else 0)

        if current_chunk:
            merged = separator.join(current_chunk).strip()
            if merged:
                chunks.append(merged)

        return chunks

    def _split_text_recursive(self, text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        final_chunks: List[str] = []

        # Find the first matching separator
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        _separator = " " if separator == "" else separator

        for s in splits:
            s_clean = s.strip()
            if not s_clean:
                continue

            if len(s_clean) <= chunk_size:
                good_splits.append(s_clean)
            else:
                if good_splits:
                    merged = self._merge_pieces(good_splits, _separator, chunk_size, chunk_overlap)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s_clean)
                else:
                    sub_chunks = self._split_text_recursive(s_clean, new_separators, chunk_size, chunk_overlap)
                    final_chunks.extend(sub_chunks)

        if good_splits:
            merged = self._merge_pieces(good_splits, _separator, chunk_size, chunk_overlap)
            final_chunks.extend(merged)

        return final_chunks

    def split(
        self,
        text: str,
        document_id: str,
        document_name: str,
        chunk_size: int,
        chunk_overlap: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        self.validate_params(text, chunk_size, chunk_overlap)
        base_meta = metadata or {}
        lang = base_meta.get("language", "en")

        chunk_texts = self._split_text_recursive(text, self.separators, chunk_size, chunk_overlap)

        chunks: List[Chunk] = []
        search_offset = 0

        for idx, c_text in enumerate(chunk_texts):
            c_text_clean = c_text.strip()
            if not c_text_clean:
                continue

            # Find matching start offset
            prefix = c_text_clean[:min(25, len(c_text_clean))]
            pos = text.find(prefix, search_offset)
            if pos == -1:
                pos = text.find(prefix)
            if pos == -1:
                pos = search_offset

            end_pos = pos + len(c_text_clean)
            search_offset = max(pos + 1, end_pos - chunk_overlap)

            c_id = f"{document_id}_chunk_{idx}"
            chunks.append(
                Chunk(
                    chunk_id=c_id,
                    document_id=document_id,
                    document_name=document_name,
                    chunk_index=idx,
                    content=c_text_clean,
                    character_count=len(c_text_clean),
                    token_count=self.calculate_tokens(c_text_clean, lang),
                    start_offset=pos,
                    end_offset=end_pos,
                    metadata=base_meta.copy(),
                )
            )

        return chunks
