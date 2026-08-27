from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """
    Abstract Base Class for vector embedding providers in NEXORA AI.
    Standardizes vector generation across development, OpenAI, Gemini, and local embeddings.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the generated vector embeddings (e.g. 1536)."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the embedding provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generates an embedding vector for a single text input."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a batch of text inputs."""
        pass
