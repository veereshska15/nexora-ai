import uuid
from typing import Sequence
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from models.document_chunk import DocumentChunkModel

class VectorRepository:

    async def insert_chunk(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        document_name: str,
        chunk_index: int,
        content: str,
        embedding: list[float],
    ) -> DocumentChunkModel:
        chunk = DocumentChunkModel(
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

    async def similarity_search(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int = 5,
        document_name: str | None = None,
    ) -> Sequence[tuple[DocumentChunkModel, float]]:
        # Calculate cosine distance using pgvector operator <=>
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

    async def get_by_id(
        self,
        session: AsyncSession,
        chunk_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> DocumentChunkModel | None:
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
        session: AsyncSession,
        chunk_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> bool:
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

vector_repository = VectorRepository()
