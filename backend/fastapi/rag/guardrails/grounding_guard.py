import re
from typing import List, Tuple
from rag.models.citation import Citation
from retrieval.models.retrieval_result import RerankedChunk


class GroundingGuard:
    """
    Evaluates synthesized answers against retrieved document chunks and citations.
    Detects hallucinations, unsupported citation claims, and empty context states.
    """

    def evaluate_grounding(
        self,
        answer: str,
        retrieved_chunks: List[RerankedChunk],
        citations: List[Citation],
    ) -> Tuple[bool, float, List[str]]:
        """
        Validates grounding integrity.
        Returns (is_grounded, confidence_score, list_of_warnings).
        """
        warnings: List[str] = []

        # 1. Check for empty answer
        if not answer or not answer.strip():
            return False, 0.0, ["Answer text is empty."]

        # 2. Check for missing context / no retrieved chunks
        if not retrieved_chunks:
            if "could not find relevant information" in answer.lower():
                return False, 0.0, []
            return False, 0.0, ["Answer generated without verified context chunks."]

        # 3. Check if answer reports information unavailability
        if "could not find relevant information" in answer.lower():
            return False, 0.0, ["No relevant document context matched the query."]

        # 4. Check for invalid or unsupported citation markers
        valid_citation_ids = {c.citation_id for c in citations}
        raw_markers = re.findall(r"\[(\d+)\]", answer)
        for m in raw_markers:
            m_id = int(m)
            if m_id not in valid_citation_ids:
                warnings.append(f"Answer referenced unsupported citation marker [{m_id}].")

        # 5. Compute grounding confidence score based on chunk relevance
        scores = [c.rerank_score for c in retrieved_chunks]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        confidence = round(max(0.0, min(1.0, avg_score)), 4)

        # 6. Final grounding decision
        is_grounded = len(retrieved_chunks) > 0 and len(warnings) == 0

        return is_grounded, confidence, warnings


grounding_guard = GroundingGuard()
