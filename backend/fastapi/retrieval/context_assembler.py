from typing import Any, Dict, List, Optional
from retrieval.models.retrieval_result import AssembledContext, RerankedChunk

DEFAULT_MAX_CONTEXT_CHARS = 4000
DEFAULT_MAX_CONTEXT_CHUNKS = 5


class ContextAssembler:
    """
    Assembles prioritized and reranked document chunks into a clean,
    structured markdown context suitable for LLM RAG prompt augmentation.
    Enforces strict character and chunk budget constraints.
    """

    def __init__(
        self,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_context_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS,
    ):
        self.max_context_chars = max_context_chars
        self.max_context_chunks = max_context_chunks

    def assemble_context(
        self,
        chunks: List[RerankedChunk],
        max_chars: Optional[int] = None,
        max_chunks: Optional[int] = None,
    ) -> AssembledContext:
        """
        Formats and bundles top chunks into a cohesive context block.
        """
        char_limit = max_chars or self.max_context_chars
        chunk_limit = max_chunks or self.max_context_chunks

        if not chunks:
            return AssembledContext(
                context_text="",
                total_chunks=0,
                total_characters=0,
                sources=[],
                truncated=False,
            )

        context_blocks: List[str] = []
        sources: List[Dict[str, Any]] = []
        current_chars = 0
        truncated = False
        included_count = 0

        for chunk in chunks:
            if included_count >= chunk_limit:
                truncated = True
                break

            header = (
                f"### [Source: {chunk.document_name} | ID: {chunk.document_id} | "
                f"Chunk: {chunk.chunk_index} | Lang: {chunk.language} | Relevance: {chunk.rerank_score:.2f}]\n"
            )
            block = f"{header}{chunk.content.strip()}\n\n"
            block_len = len(block)

            # Check character budget
            if current_chars + block_len > char_limit:
                # If no chunk has been added yet, truncate content to fit strict budget
                if included_count == 0:
                    remaining_budget = max(0, char_limit - len(header) - 5)
                    if remaining_budget > 0:
                        truncated_content = chunk.content[:remaining_budget] + "..."
                        block = f"{header}{truncated_content}\n\n"
                    else:
                        block = header[:char_limit]
                    context_blocks.append(block)
                    current_chars += len(block)
                    included_count += 1
                    sources.append({
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name,
                        "chunk_index": chunk.chunk_index,
                        "rerank_score": chunk.rerank_score,
                    })
                truncated = True
                break

            context_blocks.append(block)
            current_chars += block_len
            included_count += 1
            sources.append({
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "chunk_index": chunk.chunk_index,
                "rerank_score": chunk.rerank_score,
                "language": chunk.language,
                "script": chunk.script,
            })

        assembled_str = "".join(context_blocks).strip()
        if len(assembled_str) > char_limit:
            assembled_str = assembled_str[:char_limit]
            truncated = True

        return AssembledContext(
            context_text=assembled_str,
            total_chunks=included_count,
            total_characters=len(assembled_str),
            sources=sources,
            truncated=truncated,
        )


context_assembler = ContextAssembler()
