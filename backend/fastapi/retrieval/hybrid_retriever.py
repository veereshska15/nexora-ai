import uuid
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import logger
from documents.persistence.postgres_vector_repository import postgres_vector_repository, PostgresVectorRepository
from documents.persistence.qdrant_vector_repository import qdrant_vector_repository, QdrantVectorRepository
from retrieval.models.retrieval_result import CandidateChunk

DEV_DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"


class HybridRetriever:
    """
    Hybrid Vector Retriever combining dense vector search results from
    Qdrant vector database and PostgreSQL pgvector (HNSW cosine index)
    with strict multi-tenant isolation and deterministic result fusion.
    """

    def __init__(
        self,
        pg_repo: Optional[PostgresVectorRepository] = None,
        qdrant_repo: Optional[QdrantVectorRepository] = None,
    ):
        self.pg_repo = pg_repo or postgres_vector_repository
        self.qdrant_repo = qdrant_repo or qdrant_vector_repository

    def _to_uuid(self, val: str) -> uuid.UUID:
        try:
            return uuid.UUID(val)
        except (ValueError, AttributeError):
            return uuid.uuid5(uuid.NAMESPACE_DNS, val or str(uuid.uuid4()))

    async def retrieve_candidates(
        self,
        query_vector: List[float],
        user_id: str,
        session: Optional[AsyncSession] = None,
        candidate_k: int = 20,
        minimum_similarity: float = 0.0,
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
        language: Optional[str] = None,
        script: Optional[str] = None,
    ) -> List[CandidateChunk]:
        """
        Executes parallel or combined candidate retrieval across Qdrant and PostgreSQL pgvector.
        Deduplicates chunks and calculates fused similarity scores.
        """
        effective_user_id = user_id or DEV_DEFAULT_USER_ID
        user_uuid = self._to_uuid(effective_user_id)

        # Dictionary tracking unique candidate keys: (document_id/name, chunk_index) -> (CandidateChunk, qdrant_sim, pg_sim)
        candidates_map: Dict[str, Tuple[CandidateChunk, Optional[float], Optional[float]]] = {}

        # 1. Retrieve from Qdrant
        try:
            qdrant_hits = await self.qdrant_repo.search_similar(
                query_vector=query_vector,
                user_id=effective_user_id,
                top_k=candidate_k,
                document_id=document_id,
            )
            for hit in qdrant_hits:
                p = hit.payload or {}
                doc_id = p.get("document_id", "") or str(hit.id)
                doc_name = p.get("document_name", "") or doc_id
                chunk_idx = p.get("chunk_index", 0)
                sim = max(0.0, min(1.0, round(float(hit.score), 4)))

                key = f"{doc_name}_{chunk_idx}"
                candidate = CandidateChunk(
                    chunk_id=str(hit.id),
                    document_id=doc_id,
                    document_name=doc_name,
                    chunk_index=chunk_idx,
                    content=p.get("content", ""),
                    similarity=sim,
                    language=p.get("language", "en"),
                    script=p.get("script", "Latin"),
                    source="qdrant",
                    metadata=p.get("metadata", {}),
                )
                candidates_map[key] = (candidate, sim, None)
        except Exception as e:
            logger.warning("Qdrant candidate retrieval fallback", error=str(e))

        # 2. Retrieve from PostgreSQL pgvector
        if session:
            try:
                pg_hits = await self.pg_repo.search_similar(
                    session=session,
                    user_id=user_uuid,
                    query_embedding=query_vector,
                    top_k=candidate_k,
                    document_name=document_name or document_id,
                )
                for chunk_model, distance in pg_hits:
                    dist_val = round(float(distance), 4)
                    sim = max(0.0, min(1.0, round(1.0 - dist_val, 4)))

                    key = f"{chunk_model.document_name}_{chunk_model.chunk_index}"
                    if key in candidates_map:
                        existing_chunk, q_sim, _ = candidates_map[key]
                        candidates_map[key] = (existing_chunk, q_sim, sim)
                    else:
                        candidate = CandidateChunk(
                            chunk_id=str(chunk_model.id),
                            document_id=chunk_model.document_name,
                            document_name=chunk_model.document_name,
                            chunk_index=chunk_model.chunk_index,
                            content=chunk_model.content,
                            similarity=sim,
                            language="en",
                            script="Latin",
                            source="postgres",
                            metadata={},
                        )
                        candidates_map[key] = (candidate, None, sim)
            except Exception as e:
                logger.warning("PostgreSQL pgvector candidate retrieval fallback", error=str(e))

        # 3. Fuse scores and apply filters
        fused_candidates: List[CandidateChunk] = []

        for key, (chunk, q_sim, pg_sim) in candidates_map.items():
            # Deterministic Fusion Formula:
            # If in both stores: S_fusion = min(1.0, max(q_sim, pg_sim) + 0.05 * min(q_sim, pg_sim))
            if q_sim is not None and pg_sim is not None:
                fused_score = min(1.0, round(max(q_sim, pg_sim) + (0.05 * min(q_sim, pg_sim)), 4))
                source = "hybrid"
            elif q_sim is not None:
                fused_score = q_sim
                source = "qdrant"
            elif pg_sim is not None:
                fused_score = pg_sim
                source = "postgres"
            else:
                fused_score = chunk.similarity
                source = chunk.source

            # Apply similarity threshold
            if fused_score < minimum_similarity:
                continue

            # Apply metadata filters
            if language and chunk.language.lower() != language.lower():
                continue
            if script and chunk.script.lower() != script.lower():
                continue
            if document_name and chunk.document_name != document_name:
                continue
            if document_id and chunk.document_id != document_id:
                continue

            chunk.similarity = fused_score
            chunk.source = source
            fused_candidates.append(chunk)

        # Sort by similarity descending and limit to candidate_k
        fused_candidates.sort(key=lambda c: c.similarity, reverse=True)
        return fused_candidates[:candidate_k]


hybrid_retriever = HybridRetriever()
