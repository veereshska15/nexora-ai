import re
from typing import List, Optional, Set
from rag.models.citation import Citation
from retrieval.models.retrieval_result import RerankedChunk


class CitationService:
    """
    Service for extracting, indexing, and validating source citations from retrieved chunks.
    Maps inline answer citation markers ([1], [2], etc.) back to verified source document chunks.
    """

    def build_citations(self, chunks: List[RerankedChunk]) -> List[Citation]:
        """
        Constructs an indexed list of Citation models from retrieved reranked chunks.
        """
        citations: List[Citation] = []
        for i, c in enumerate(chunks, 1):
            snippet = c.content.strip()
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."

            citations.append(
                Citation(
                    citation_id=i,
                    marker=f"[{i}]",
                    document_id=c.document_id,
                    document_name=c.document_name,
                    chunk_id=c.chunk_id,
                    chunk_index=c.chunk_index,
                    content_snippet=snippet,
                    language=c.language,
                    script=c.script,
                    relevance_score=c.rerank_score,
                )
            )
        return citations

    def extract_referenced_citations(
        self,
        answer: str,
        all_citations: List[Citation],
    ) -> List[Citation]:
        """
        Extracts only citations explicitly referenced by markers in the generated answer text.
        If no markers are found in the text, returns all retrieved citations.
        """
        if not all_citations:
            return []

        # Find markers like [1], [2], [1, 2]
        found_ids: Set[int] = set()
        raw_matches = re.findall(r"\[(\d+)\]", answer)
        for m in raw_matches:
            try:
                found_ids.add(int(m))
            except ValueError:
                continue

        if not found_ids:
            # Fallback to returning all retrieved citations
            return all_citations

        # Filter strictly to existing citations
        referenced = [c for c in all_citations if c.citation_id in found_ids]
        return referenced or all_citations


citation_service = CitationService()
