# NEXORA AI — Phase 08: Multilingual RAG & Grounded Generation Architecture

## 1. Overview & Vision

Phase 08 delivers the enterprise **Multilingual Document Ingestion, Semantic Chunking, Dense Vector Embeddings, Hybrid Vector Persistence, Multi-Signal Reranking, Grounded Generation & Citation Engine** for NEXORA AI.

```
                           ┌────────────────────────────────────────────────────────┐
                           │            NEXORA MULTILINGUAL RAG ECOSYSTEM           │
                           │   - Multi-Format Document Ingestion Engine             │
                           │   - TXT, PDF, DOCX, CSV In-Memory Stream Parsers       │
                           │   - Unicode & Indic (Kannada) Character Preservation   │
                           │   - Defensive Validation & Zero-Disk-Persistence       │
                           │   - Semantic & Token-Aware Chunking Engine             │
                           │   - 1536d Deterministic Normalized Vector Embeddings   │
                           │   - Dual-Store PostgreSQL (pgvector) & Qdrant Engine   │
                           │   - Multi-Store Hybrid Candidate Retrieval & Fusion    │
                           │   - Deterministic Multi-Signal Reranking Engine        │
                           │   - Structured Markdown RAG Context Assembler          │
                           │   - Grounded LLM Provider Abstraction                  │
                           │   - Verified Source Citation Extraction & Mapping      │
                           │   - Hallucination Mitigation & Grounding Guardrails    │
                           └───────────────────────────┬────────────────────────────┘
                                                       │
         ┌─────────────────────────┬───────────────────┴───────────────────┬─────────────────────────┐
         ▼                         ▼                                       ▼                         ▼
     TXT Parser               PDF Parser                              DOCX Parser               CSV Parser
  UTF-8/16/Latin-1          pypdf Stream                            python-docx Stream        csv Stream IO
  Kannada & Indic           Page Count & Text                       Paragraphs & Tables       Tabular Text Rows
```

---

## 2. Complete Phase 08 Directory Structure

```
backend/fastapi/
├── documents/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── document_result.py            # Standardized DocumentExtractionResult
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── base_parser.py                # BaseDocumentParser abstract base class
│   │   ├── txt_parser.py                 # TxtParser with multi-encoding fallbacks
│   │   ├── pdf_parser.py                 # PdfParser using pypdf
│   │   ├── docx_parser.py                # DocxParser using python-docx
│   │   └── csv_parser.py                 # CsvParser with delimiter & table formatting
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── chunk_result.py           # Standardized Chunk & ChunkingResult models
│   │   ├── base_chunker.py               # BaseChunker abstract base class with Unicode safety
│   │   ├── recursive_chunker.py          # CharacterChunker & RecursiveChunker
│   │   ├── token_aware_chunker.py        # TokenAwareChunker with SubwordTokenizer integration
│   │   └── chunking_service.py           # Centralized ChunkingService routing & metadata enrichment
│   ├── persistence/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── persistence_result.py     # ChunkPersistenceRecord, DocumentPersistenceResult, Status
│   │   ├── postgres_vector_repository.py # PostgresVectorRepository (pgvector HNSW Cosine Index)
│   │   ├── qdrant_vector_repository.py   # QdrantVectorRepository (1536d Cosine Collections)
│   │   └── vector_persistence_service.py # VectorPersistenceService dual-store orchestrator
│   ├── document_service.py               # DocumentIngestionService singleton & validator
│   └── hybrid_ingestion_service.py       # HybridIngestionService coordinating end-to-end enrichment
├── embeddings/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── embedding_result.py           # EmbeddingResult standardized model
│   ├── base_embedding.py                 # BaseEmbeddingProvider abstract interface
│   ├── providers/
│   │   ├── __init__.py
│   │   └── development_embedding.py      # DevelopmentEmbeddingProvider (1536d, deterministic, offline)
│   ├── embedding_factory.py              # EmbeddingFactory provider registry
│   └── embedding_service.py              # EmbeddingService single & batch vector generator
├── retrieval/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── retrieval_result.py           # CandidateChunk, RerankedChunk, AssembledContext, RetrievalResult
│   ├── hybrid_retriever.py               # HybridRetriever combining Qdrant + PostgreSQL with fusion scoring
│   ├── reranker.py                       # DevelopmentReranker with vector + lexical + language + script signals
│   ├── context_assembler.py              # ContextAssembler enforcing character and chunk limits
│   └── retrieval_service.py              # End-to-end retrieval orchestration service
└── rag/
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   ├── citation.py                   # Citation model with marker, doc_id, snippet, relevance
    │   └── rag_result.py                 # RAGResponse model with answer, citations, confidence, warnings
    ├── providers/
    │   ├── __init__.py
    │   ├── base_llm.py                   # BaseLLMProvider abstract interface
    │   └── development_llm.py            # DevelopmentLLMProvider (deterministic offline grounded synthesis)
    ├── prompt/
    │   ├── __init__.py
    │   └── grounded_prompt.py            # GroundedPromptBuilder with strict multilingual directives
    ├── citation/
    │   ├── __init__.py
    │   └── citation_service.py           # CitationService extracting and mapping [1], [2] markers
    ├── guardrails/
    │   ├── __init__.py
    │   └── grounding_guard.py            # GroundingGuard hallucination check & confidence evaluation
    └── rag_service.py                    # RAGService end-to-end orchestration gateway
```

---

## 3. End-to-End RAG Query Orchestration Pipeline

```
Query -> NLP Analysis -> 1536d Vector -> Hybrid Retrieval -> Reranker -> Context -> Grounded Prompt -> LLM Provider -> Citation Mapping -> Grounding Guard -> RAGResponse
```

---

## 4. REST API Endpoints

### 1. Synchronous RAG Query: `POST /api/v1/rag/query`
```json
// Request
{
  "query": "ಕನ್ನಡ ಭಾಷೆಯ ಬಗ್ಗೆ ವಿವರಿಸಿ",
  "top_k": 3
}

// Response (HTTP 200 OK)
{
  "query": "ಕನ್ನಡ ಭಾಷೆಯ ಬಗ್ಗೆ ವಿವರಿಸಿ",
  "answer": "Based on the retrieved documents, here is the verified information regarding ಕನ್ನಡ ಭಾಷೆಯ ಬಗ್ಗೆ ವಿವರಿಸಿ:\n\nಕನ್ನಡ ಭಾಷೆಯು ಭಾರತದ ಅತ್ಯಂತ ಪ್ರಾಚೀನ ಶಾಸ್ತ್ರೀಯ ಭಾಷೆಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ [1].",
  "language": "kn",
  "script": "Kannada",
  "grounded": true,
  "grounding_confidence": 0.91,
  "citations": [
    {
      "citation_id": 1,
      "marker": "[1]",
      "document_id": "doc_a91b4f2c",
      "document_name": "kannada_ai.txt",
      "chunk_id": "doc_a91b4f2c_chunk_0",
      "chunk_index": 0,
      "content_snippet": "ಕನ್ನಡ ಭಾಷೆಯು ಭಾರತದ ಅತ್ಯಂತ ಪ್ರಾಚೀನ ಶಾಸ್ತ್ರೀಯ ಭಾಷೆಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ...",
      "language": "kn",
      "script": "Kannada",
      "relevance_score": 0.91
    }
  ],
  "retrieved_chunks": 1,
  "provider": "development",
  "model": "development-grounded",
  "processing_time_ms": 3.2,
  "warnings": []
}
```

### 2. Streaming RAG Query: `POST /api/v1/rag/query/stream`
Returns streamed text chunks with `media_type="text/plain"`.
