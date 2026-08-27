import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from documents.chunking.chunking_service import chunking_service
from embeddings.embedding_service import embedding_service
from nlp.disambiguation.language_disambiguator import language_disambiguator


class EnrichedChunk(BaseModel):
    """
    Complete enriched chunk containing text, positional offsets, NLP metadata, and vector embeddings.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    document_name: str = Field(..., description="Parent document filename/title")
    chunk_index: int = Field(..., ge=0, description="Chunk sequence index")
    content: str = Field(..., min_length=1, description="Chunk text content")
    character_count: int = Field(..., ge=1, description="Character count")
    token_count: int = Field(..., ge=0, description="Subword token count")
    start_offset: int = Field(..., ge=0, description="Start character offset")
    end_offset: int = Field(..., ge=0, description="End character offset")
    language: str = Field(..., description="Detected ISO language code (e.g., 'kn', 'en', 'hi')")
    script: str = Field(..., description="Detected primary script")
    embedding: List[float] = Field(..., description="Dense 1536d normalized vector embedding")
    embedding_dimension: int = Field(1536, description="Embedding dimensionality")
    embedding_provider: str = Field(..., description="Embedding provider name")
    embedding_model: str = Field(..., description="Embedding model name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Preserved and enriched metadata")


class EnrichedDocumentResult(BaseModel):
    """
    Complete response container for the hybrid document ingestion and enrichment pipeline.
    """
    document_id: str = Field(..., description="Document identifier")
    document_name: str = Field(..., description="Document title/filename")
    strategy: str = Field(..., description="Applied chunking strategy")
    total_chunks: int = Field(..., ge=0, description="Total number of enriched chunks")
    embedding_dimension: int = Field(1536, description="Vector dimension")
    embedding_provider: str = Field(..., description="Embedding provider name")
    chunks: List[EnrichedChunk] = Field(default_factory=list, description="List of enriched vector chunks")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Total pipeline latency in milliseconds")


class HybridIngestionService:
    """
    Hybrid Multilingual Document Ingestion and Vector Enrichment Engine.
    Coordinates document extraction, semantic chunking, Indic NLP profiling,
    and normalized vector embedding generation without external network calls.
    """

    def __init__(self):
        self._chunking_service = chunking_service
        self._embedding_service = embedding_service

    def enrich_document(
        self,
        text: str,
        document_name: str = "document.txt",
        document_id: Optional[str] = None,
        strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        embedding_provider: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EnrichedDocumentResult:
        """
        Executes the full in-memory document enrichment pipeline:
        Extraction -> Semantic Chunking -> NLP Language/Script Profiling -> Vector Embedding.
        """
        start_time = time.perf_counter()
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"

        # 1. Chunk the document text
        chunk_res = self._chunking_service.chunk_text(
            text=text,
            document_name=document_name,
            document_id=doc_id,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=metadata,
        )

        if not chunk_res.chunks:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)
            return EnrichedDocumentResult(
                document_id=doc_id,
                document_name=document_name,
                strategy=strategy,
                total_chunks=0,
                embedding_dimension=1536,
                embedding_provider=embedding_provider or "development_deterministic",
                chunks=[],
                processing_time_ms=elapsed_ms,
            )

        # 2. Extract texts for batch embedding
        chunk_texts = [c.content for c in chunk_res.chunks]
        emb_results = self._embedding_service.embed_batch(chunk_texts, provider=embedding_provider)

        # 3. Assemble enriched chunks with full metadata and vectors
        enriched_chunks: List[EnrichedChunk] = []
        for c, emb in zip(chunk_res.chunks, emb_results):
            # Resolve language and script from chunk metadata or fallback detection
            lang = c.metadata.get("language")
            script = c.metadata.get("script")
            if not lang or not script:
                disamb = language_disambiguator.disambiguate(c.content)
                lang = disamb.language
                script = disamb.script

            enriched_chunks.append(
                EnrichedChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    document_name=c.document_name,
                    chunk_index=c.chunk_index,
                    content=c.content,
                    character_count=c.character_count,
                    token_count=c.token_count,
                    start_offset=c.start_offset,
                    end_offset=c.end_offset,
                    language=lang,
                    script=script,
                    embedding=emb.vector,
                    embedding_dimension=emb.dimension,
                    embedding_provider=emb.provider,
                    embedding_model=emb.model,
                    metadata=c.metadata.copy(),
                )
            )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)

        return EnrichedDocumentResult(
            document_id=doc_id,
            document_name=document_name,
            strategy=strategy,
            total_chunks=len(enriched_chunks),
            embedding_dimension=emb_results[0].dimension if emb_results else 1536,
            embedding_provider=emb_results[0].provider if emb_results else "development_deterministic",
            chunks=enriched_chunks,
            processing_time_ms=elapsed_ms,
        )


hybrid_ingestion_service = HybridIngestionService()
