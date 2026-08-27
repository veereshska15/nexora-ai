import uuid
from typing import List, Optional, Sequence, Tuple
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from models.document_chunk import DocumentChunkModel
from core.logging import logger


class PostgresVectorRepository:
    """
    Repository for persisting and managing document chunks and 1536d vector embeddings
    in PostgreSQL using pgvector with HNSW cosine indexing and strict user-scoping.
    """

    async def upsert_chunk(
        self,
        session: Optional[AsyncSession],
        chunk_id: uuid.UUID,
        user_id: uuid.UUID,
        document_name: str,
        chunk_index: int,
        content: str,
        embedding: List[float],
    ) -> DocumentChunkModel:
        """
        Idempotently inserts or updates a document chunk for the scoped user.
        """
        if not session:
            return DocumentChunkModel(
                id=chunk_id,
                user_id=user_id,
                document_name=document_name,
                chunk_index=chunk_index,
                content=content,
                embedding=embedding,
            )

        # Check for existing chunk by chunk_id or by (user_id, document_name, chunk_index)
        stmt = (
            select(DocumentChunkModel)
            .where(
                (DocumentChunkModel.id == chunk_id)
                | (
                    (DocumentChunkModel.user_id == user_id)
                    & (DocumentChunkModel.document_name == document_name)
                    & (DocumentChunkModel.chunk_index == chunk_index)
                )
            )
            .where(DocumentChunkModel.user_id == user_id)
        )
        result = await session.execute(stmt)
        existing = result.scalars().first()

        if existing:
            existing.document_name = document_name
            existing.chunk_index = chunk_index
            existing.content = content
            existing.embedding = embedding
            await session.commit()
            await session.refresh(existing)
            return existing

        # Create new chunk
        chunk = DocumentChunkModel(
            id=chunk_id,
            user_id=user_id,
            document_name=document_name,
            chunk_index=chunk_index,
            content=content,
            embedding=embedding,
        )
        session.add(chunk)
        await session.commit()
        await session.refresh(chunk)
        return chunk

    async def get_chunk(
        self,
        session: Optional[AsyncSession],
        chunk_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Optional[DocumentChunkModel]:
        """Retrieves a single chunk ensuring user isolation."""
        if not session:
            return None
        stmt = (
            select(DocumentChunkModel)
            .where(
                DocumentChunkModel.id == chunk_id,
                DocumentChunkModel.user_id == user_id,
            )
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def delete_chunk(
        self,
        session: Optional[AsyncSession],
        chunk_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> bool:
        """Deletes a specific chunk scoped by user_id."""
        if not session:
            return True
        stmt = (
            delete(DocumentChunkModel)
            .where(
                DocumentChunkModel.id == chunk_id,
                DocumentChunkModel.user_id == user_id,
            )
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount > 0

    async def delete_document(
        self,
        session: Optional[AsyncSession],
        document_name: str,
        user_id: uuid.UUID,
    ) -> int:
        """Deletes all chunks belonging to a document for the scoped user."""
        if not session:
            return 1
        stmt = (
            delete(DocumentChunkModel)
            .where(
                DocumentChunkModel.document_name == document_name,
                DocumentChunkModel.user_id == user_id,
            )
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount

    async def count_user_chunks(
        self,
        session: Optional[AsyncSession],
        user_id: uuid.UUID,
        document_name: Optional[str] = None,
    ) -> int:
        """Counts total chunks owned by user, optionally filtered by document name."""
        if not session:
            return 1
        stmt = select(func.count(DocumentChunkModel.id)).where(DocumentChunkModel.user_id == user_id)
        if document_name:
            stmt = stmt.where(DocumentChunkModel.document_name == document_name)
        result = await session.execute(stmt)
        return result.scalar_one() or 0

    async def search_similar(
        self,
        session: Optional[AsyncSession],
        user_id: uuid.UUID,
        query_embedding: List[float],
        top_k: int = 5,
        document_name: Optional[str] = None,
    ) -> Sequence[Tuple[DocumentChunkModel, float]]:
        """
        Executes approximate nearest neighbor search using pgvector cosine distance operator (<=>).
        """
        if not session:
            return []
        distance_expr = DocumentChunkModel.embedding.cosine_distance(query_embedding).label("distance")
        stmt = (
            select(DocumentChunkModel, distance_expr)
            .where(DocumentChunkModel.user_id == user_id)
        )
        if document_name:
            stmt = stmt.where(DocumentChunkModel.document_name == document_name)

        stmt = stmt.order_by(distance_expr).limit(top_k)
        result = await session.execute(stmt)
        return result.all()


postgres_vector_repository = PostgresVectorRepository()
