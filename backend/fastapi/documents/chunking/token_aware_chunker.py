from typing import Any, Dict, List, Optional
from nlp.tokenizers.tokenizer_factory import tokenizer_factory
from documents.chunking.base_chunker import BaseChunker
from documents.chunking.models.chunk_result import Chunk


class TokenAwareChunker(BaseChunker):
    """
    Subword and Token-Aware Chunker.
    Splits text strictly along subword token boundaries.
    """

    def __init__(self):
        super().__init__(strategy_name="token")

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

        tokenizer = tokenizer_factory.get(lang)
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)

        total_tokens = len(tokens)
        if total_tokens == 0:
            return []

        chunks: List[Chunk] = []
        step = max(1, chunk_size - chunk_overlap)
        start_token_idx = 0
        chunk_idx = 0
        search_offset = 0

        while start_token_idx < total_tokens:
            end_token_idx = min(start_token_idx + chunk_size, total_tokens)
            selected_ids = token_ids[start_token_idx:end_token_idx]

            # Decode subword token window back to text
            chunk_content = tokenizer.decode(selected_ids).strip()
            if not chunk_content:
                chunk_content = " ".join(tokens[start_token_idx:end_token_idx]).strip()

            if chunk_content:
                prefix = chunk_content[:min(20, len(chunk_content))]
                pos = text.find(prefix, search_offset)
                if pos == -1:
                    pos = text.find(prefix)
                if pos == -1:
                    pos = search_offset
                end_pos = min(len(text), pos + len(chunk_content))
                search_offset = max(pos + 1, end_pos - (chunk_overlap * 4))

                c_id = f"{document_id}_chunk_{chunk_idx}"
                chunks.append(
                    Chunk(
                        chunk_id=c_id,
                        document_id=document_id,
                        document_name=document_name,
                        chunk_index=chunk_idx,
                        content=chunk_content,
                        character_count=len(chunk_content),
                        token_count=len(selected_ids),
                        start_offset=pos,
                        end_offset=end_pos,
                        metadata=base_meta.copy(),
                    )
                )
                chunk_idx += 1

            if end_token_idx >= total_tokens:
                break

            start_token_idx += step

        return chunks
