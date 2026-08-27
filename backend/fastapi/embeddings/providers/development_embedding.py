import hashlib
from typing import List
import numpy as np
from embeddings.base_embedding import BaseEmbeddingProvider


class DevelopmentEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic Development Embedding Provider for NEXORA AI.
    Generates exact 1536-dimensional normalized unit vectors offline without API keys.
    Guarantees deterministic, identical outputs for identical text inputs.
    """

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension
        self._provider_name = "development_deterministic"
        self._model_name = f"development-{dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def _generate_vector(self, text: str) -> List[float]:
        """
        Generates a 1536-dimensional normalized vector from text using SHA-256 seeded pseudo-random distribution.
        """
        if not text:
            return [0.0] * self._dimension

        # Compute deterministic seed from SHA-256 hash
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(hash_bytes[:8], byteorder="big") % (2**32 - 1)

        # Generate base random vector from deterministic seed
        rng = np.random.RandomState(seed)
        raw_vec = rng.randn(self._dimension).astype(np.float32)

        # Add lexical signal from character code points into the first 128 dimensions
        for i, char in enumerate(text[:128]):
            raw_vec[i % self._dimension] += (ord(char) % 100) / 50.0

        # L2-normalize vector to unit length for cosine similarity
        norm = np.linalg.norm(raw_vec)
        if norm > 0:
            norm_vec = raw_vec / norm
        else:
            norm_vec = raw_vec

        return [round(float(x), 6) for x in norm_vec]

    def embed_text(self, text: str) -> List[float]:
        """Generates a 1536d normalized vector for a single text input."""
        return self._generate_vector(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates 1536d normalized vectors for a batch of text inputs."""
        return [self._generate_vector(t) for t in texts]
