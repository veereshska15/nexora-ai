import time
from typing import List, Optional
from embeddings.embedding_factory import embedding_factory
from embeddings.models.embedding_result import EmbeddingResult

MAX_TEXT_LENGTH = 50000
MAX_BATCH_SIZE = 100


class EmbeddingService:
    """
    Centralized Vector Embedding Service for NEXORA AI.
    Executes single and batch text embedding with validation, provider dispatch, and telemetry.
    """

    def __init__(self):
        self._factory = embedding_factory

    def validate_text(self, text: str) -> None:
        """Validates a single text input string."""
        if text is None or not text.strip():
            raise ValueError("Text input cannot be empty or whitespace-only.")
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text length ({len(text)}) exceeds maximum allowed limit of {MAX_TEXT_LENGTH} characters.")

    def validate_batch(self, texts: List[str]) -> None:
        """Validates a batch list of text inputs."""
        if not texts or len(texts) == 0:
            raise ValueError("Batch input list cannot be empty.")
        if len(texts) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size ({len(texts)}) exceeds maximum allowed limit of {MAX_BATCH_SIZE} texts.")
        for idx, t in enumerate(texts):
            if t is None or not str(t).strip():
                raise ValueError(f"Batch element at index {idx} cannot be empty.")
            if len(str(t)) > MAX_TEXT_LENGTH:
                raise ValueError(f"Batch element at index {idx} exceeds maximum allowed limit of {MAX_TEXT_LENGTH} characters.")

    def embed_single(self, text: str, provider: Optional[str] = None) -> EmbeddingResult:
        """
        Generates an embedding vector for a single text input.
        """
        self.validate_text(text)
        start_time = time.perf_counter()

        prov = self._factory.get_provider(provider)
        vector = prov.embed_text(text)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)

        return EmbeddingResult(
            text=text,
            vector=vector,
            dimension=prov.dimension,
            provider=prov.provider_name,
            model=prov.model_name,
            processing_time_ms=elapsed_ms,
        )

    def embed_batch(self, texts: List[str], provider: Optional[str] = None) -> List[EmbeddingResult]:
        """
        Generates embedding vectors for a batch list of text inputs.
        """
        self.validate_batch(texts)
        start_time = time.perf_counter()

        prov = self._factory.get_provider(provider)
        vectors = prov.embed_texts(texts)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)
        per_item_ms = round(elapsed_ms / len(texts), 3) if texts else 0.0

        results = []
        for t, v in zip(texts, vectors):
            results.append(
                EmbeddingResult(
                    text=t,
                    vector=v,
                    dimension=prov.dimension,
                    provider=prov.provider_name,
                    model=prov.model_name,
                    processing_time_ms=per_item_ms,
                )
            )
        return results


embedding_service = EmbeddingService()
