from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db_session
from schemas.documents import (
    DocumentExtractionResponse,
    PersistDocumentRequest,
    PersistDocumentResponse,
    DocumentStatusResponse,
    DocumentSearchQueryRequest,
    DocumentSearchQueryResponse,
)
from schemas.chunking import ChunkTextRequest, ChunkTextResponse, ChunkSchema
from schemas.embeddings import EnrichDocumentRequest, EnrichDocumentResponse, EnrichedChunkSchema
from documents.document_service import document_service
from documents.chunking.chunking_service import chunking_service
from documents.hybrid_ingestion_service import hybrid_ingestion_service, EnrichedChunk
from documents.persistence.vector_persistence_service import vector_persistence_service

router = APIRouter(prefix="/documents", tags=["Document Ingestion & Persistence"])


@router.post(
    "/extract",
    response_model=DocumentExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract text from uploaded document (TXT, PDF, DOCX, CSV)",
)
async def extract_document_text(
    file: UploadFile = File(..., description="Document file to extract text from")
) -> DocumentExtractionResponse:
    """
    Accepts multipart file upload (TXT, PDF, DOCX, CSV) and extracts plain text
    with metadata (file size, character count, page count, processing latency).
    Enforces file size limits and security controls without writing files to disk.
    """
    try:
        filename = file.filename or "uploaded_document"
        file_bytes = await file.read()

        result = document_service.extract_from_bytes(file_bytes, filename)

        if not result.extraction_success:
            # If failure was due to validation, raise 400 Bad Request
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message or "Failed to extract text from uploaded document.",
            )

        return DocumentExtractionResponse(**result.model_dump())

    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal document extraction error: {str(err)}",
        )


@router.post(
    "/chunk",
    response_model=ChunkTextResponse,
    status_code=status.HTTP_200_OK,
    summary="Split document text into semantic or token-aware chunks",
)
async def chunk_document_text(request: ChunkTextRequest) -> ChunkTextResponse:
    """
    Splits text using configurable strategies ('character', 'recursive', or 'token')
    with Indic Unicode conjunct protection and multilingual token count metadata.
    """
    try:
        result = chunking_service.chunk_text(
            text=request.text,
            document_name=request.document_name,
            document_id=request.document_id,
            strategy=request.strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            metadata=request.metadata,
        )

        return ChunkTextResponse(
            document_id=result.document_id,
            document_name=result.document_name,
            strategy=result.strategy,
            total_chunks=result.total_chunks,
            chunks=[ChunkSchema(**c.model_dump()) for c in result.chunks],
            processing_time_ms=result.processing_time_ms,
        )

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal chunking error: {str(err)}",
        )


@router.post(
    "/enrich",
    response_model=EnrichDocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Enrich document text with chunking, multilingual NLP metadata, and dense vector embeddings",
)
async def enrich_document_text(request: EnrichDocumentRequest) -> EnrichDocumentResponse:
    """
    Executes complete hybrid document enrichment pipeline:
    Text -> Chunking -> Language / Script Detection -> 1536d Vector Embeddings.
    """
    try:
        result = hybrid_ingestion_service.enrich_document(
            text=request.text,
            document_name=request.document_name,
            document_id=request.document_id,
            strategy=request.strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            embedding_provider=request.embedding_provider,
            metadata=request.metadata,
        )

        return EnrichDocumentResponse(
            document_id=result.document_id,
            document_name=result.document_name,
            strategy=result.strategy,
            total_chunks=result.total_chunks,
            embedding_dimension=result.embedding_dimension,
            embedding_provider=result.embedding_provider,
            chunks=[EnrichedChunkSchema(**c.model_dump()) for c in result.chunks],
            processing_time_ms=result.processing_time_ms,
        )

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal document enrichment error: {str(err)}",
        )


@router.post(
    "/persist",
    response_model=PersistDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Persist enriched document chunks into PostgreSQL pgvector and Qdrant collections",
)
async def persist_document_chunks(
    request: PersistDocumentRequest,
    db: AsyncSession = Depends(get_db_session),
) -> PersistDocumentResponse:
    """
    Persists enriched document chunks into PostgreSQL pgvector and Qdrant.
    Guarantees user scoping and idempotent upserts.
    """
    try:
        enriched_chunks = [EnrichedChunk(**c.model_dump()) for c in request.chunks]
        result = await vector_persistence_service.persist_chunks(
            session=db,
            document_id=request.document_id,
            document_name=request.document_name,
            chunks=enriched_chunks,
            user_id=request.user_id or "00000000-0000-0000-0000-000000000001",
        )

        return PersistDocumentResponse(
            document_id=result.document_id,
            document_name=result.document_name,
            user_id=result.user_id,
            total_chunks=result.total_chunks,
            postgres_success_count=result.postgres_success_count,
            qdrant_success_count=result.qdrant_success_count,
            status=result.status,
            chunks=result.chunks,
            processing_time_ms=result.processing_time_ms,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector persistence error: {str(err)}",
        )


@router.delete(
    "/{document_id}",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Delete document chunks from both PostgreSQL and Qdrant",
)
async def delete_document_vectors(
    document_id: str,
    user_id: str = "00000000-0000-0000-0000-000000000001",
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """
    Deletes all vector chunks belonging to the specified document for the authenticated user.
    """
    try:
        deleted = await vector_persistence_service.delete_document(
            session=db,
            document_id=document_id,
            user_id=user_id,
        )
        return {
            "message": f"Document '{document_id}' vector chunks deleted successfully",
            "document_id": document_id,
            "deleted": deleted,
        }
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document vector deletion error: {str(err)}",
        )


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document persistence status across vector stores",
)
async def get_document_persistence_status(
    document_id: str,
    user_id: str = "00000000-0000-0000-0000-000000000001",
    db: AsyncSession = Depends(get_db_session),
) -> DocumentStatusResponse:
    """
    Returns chunk counts and synchronization state for a document across PostgreSQL and Qdrant.
    """
    try:
        status_res = await vector_persistence_service.get_document_status(
            session=db,
            document_id=document_id,
            user_id=user_id,
        )
        return DocumentStatusResponse(**status_res.model_dump())
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving document persistence status: {str(err)}",
        )


@router.post(
    "/search",
    response_model=DocumentSearchQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic vector similarity search across Qdrant and pgvector",
)
async def search_documents_semantic(
    request: DocumentSearchQueryRequest,
    db: AsyncSession = Depends(get_db_session),
) -> DocumentSearchQueryResponse:
    """
    Embeds user search query into 1536d vector and retrieves nearest neighbor chunks
    filtered strictly by authenticated user scope.
    """
    try:
        result = await vector_persistence_service.search_documents(
            session=db,
            query=request.query,
            user_id=request.user_id or "00000000-0000-0000-0000-000000000001",
            top_k=request.top_k,
            document_id=request.document_id,
        )

        return DocumentSearchQueryResponse(
            query=result.query,
            total_results=result.total_results,
            results=result.results,
            processing_time_ms=result.processing_time_ms,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic document search error: {str(err)}",
        )
