import re
from abc import ABC, abstractmethod
from typing import List, Optional, Set
from retrieval.models.retrieval_result import CandidateChunk, RerankedChunk


class BaseReranker(ABC):
    """
    Abstract Base Class for Document Chunk Rerankers.
    Allows seamless plug-and-play replacement with Cross-Encoder or Neural Rerankers.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[CandidateChunk],
        query_language: str = "en",
        query_script: str = "Latin",
        top_k: int = 5,
    ) -> List[RerankedChunk]:
        pass


class DevelopmentReranker(BaseReranker):
    """
    Deterministic Multi-Signal Development Reranker.
    Evaluates vector similarity, lexical token overlap, language alignment,
    and script consistency without requiring external LLM or GPU dependencies.
    """

    WEIGHT_VECTOR = 0.50
    WEIGHT_LEXICAL = 0.30
    WEIGHT_LANGUAGE = 0.10
    WEIGHT_SCRIPT = 0.10

    def _tokenize(self, text: str) -> Set[str]:
        """Simple multilingual regex word tokenizer for lexical overlap calculation."""
        tokens = re.findall(r"[\w\u0900-\u0DFF]+", text.lower())
        return set(tokens)

    def _calculate_lexical_overlap(self, query: str, content: str) -> float:
        """
        Calculates Jaccard token overlap similarity between query and document chunk.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        content_tokens = self._tokenize(content)
        if not content_tokens:
            return 0.0

        intersection = query_tokens.intersection(content_tokens)
        if not intersection:
            return 0.0

        # Overlap ratio relative to query length
        ratio = len(intersection) / len(query_tokens)
        return min(1.0, round(ratio, 4))

    def rerank(
        self,
        query: str,
        candidates: List[CandidateChunk],
        query_language: str = "en",
        query_script: str = "Latin",
        top_k: int = 5,
    ) -> List[RerankedChunk]:
        """
        Reranks candidate chunks using weighted multi-signal scoring.
        """
        if not candidates:
            return []

        reranked: List[RerankedChunk] = []

        for c in candidates:
            # 1. Vector similarity signal
            s_vector = max(0.0, min(1.0, c.similarity))

            # 2. Lexical token overlap signal
            s_lexical = self._calculate_lexical_overlap(query, c.content)

            # 3. Language match signal
            lang_match = (c.language.lower() == query_language.lower()) if query_language else False
            s_language = 1.0 if lang_match else 0.0

            # 4. Script match signal
            script_match = (c.script.lower() == query_script.lower()) if query_script else False
            s_script = 1.0 if script_match else 0.0

            # Composite weighted rerank score
            rerank_score = round(
                (self.WEIGHT_VECTOR * s_vector)
                + (self.WEIGHT_LEXICAL * s_lexical)
                + (self.WEIGHT_LANGUAGE * s_language)
                + (self.WEIGHT_SCRIPT * s_script),
                4,
            )

            # Build descriptive matched signal list
            signals: List[str] = []
            signals.append(f"vector_sim_{s_vector:.2f}")
            if s_lexical > 0.0:
                signals.append(f"lexical_overlap_{s_lexical:.2f}")
            if lang_match:
                signals.append(f"lang_match_{c.language}")
            if script_match:
                signals.append(f"script_match_{c.script}")
            if c.source == "hybrid":
                signals.append("dual_store_hybrid")

            reranked.append(
                RerankedChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    document_name=c.document_name,
                    chunk_index=c.chunk_index,
                    content=c.content,
                    similarity=s_vector,
                    rerank_score=rerank_score,
                    lexical_score=s_lexical,
                    language_match=lang_match,
                    script_match=script_match,
                    language=c.language,
                    script=c.script,
                    matched_signals=signals,
                    metadata=c.metadata,
                )
            )

        # Sort descending by composite rerank score, with vector similarity as tiebreaker
        reranked.sort(key=lambda x: (x.rerank_score, x.similarity), reverse=True)
        return reranked[:top_k]


development_reranker = DevelopmentReranker()
