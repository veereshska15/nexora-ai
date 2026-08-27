import time
import uuid
from typing import Any, Dict, List, Optional
from nlp.detectors.script_identifier import unicode_script_identifier
from nlp.disambiguation.language_disambiguator import language_disambiguator
from documents.chunking.base_chunker import BaseChunker
from documents.chunking.models.chunk_result import Chunk, ChunkingResult
from documents.chunking.recursive_chunker import CharacterChunker, RecursiveChunker
from documents.chunking.token_aware_chunker import TokenAwareChunker


class ChunkingService:
    """
    Centralized Multilingual Document Chunking Service for NEXORA AI.
    Routes text splitting requests to character, recursive, or token-aware chunkers
    with automatic Indic language detection and metadata enrichment.
    """

    def __init__(self):
        self._chunkers: Dict[str, BaseChunker] = {
            "character": CharacterChunker(),
            "recursive": RecursiveChunker(),
            "token": TokenAwareChunker(),
        }

    def chunk_text(
        self,
        text: str,
        document_name: str = "document.txt",
        document_id: Optional[str] = None,
        strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChunkingResult:
        """
        Executes document text chunking with validation, telemetry, and metadata enrichment.
        """
        start_time = time.perf_counter()
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
        strat_key = (strategy or "recursive").lower().strip()

        if strat_key not in self._chunkers:
            allowed = ", ".join(self._chunkers.keys())
            raise ValueError(f"Unsupported chunking strategy '{strategy}'. Allowed strategies: {allowed}")

        # Enrich metadata with language and script detection if not provided
        chunk_meta = (metadata or {}).copy()
        if "language" not in chunk_meta and text and text.strip():
            disamb_res = language_disambiguator.disambiguate(text)
            chunk_meta["language"] = disamb_res.language
            chunk_meta["script"] = disamb_res.script
            chunk_meta["is_indic"] = disamb_res.language in ["kn", "hi", "ta", "te", "ml", "mr", "bn"]

        chunker = self._chunkers[strat_key]
        chunks = chunker.split(
            text=text,
            document_id=doc_id,
            document_name=document_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=chunk_meta,
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)

        return ChunkingResult(
            document_id=doc_id,
            document_name=document_name,
            strategy=strat_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            total_chunks=len(chunks),
            chunks=chunks,
            processing_time_ms=elapsed_ms,
        )


chunking_service = ChunkingService()
