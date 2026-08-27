from typing import Dict, List, Optional
from embeddings.base_embedding import BaseEmbeddingProvider
from embeddings.providers.development_embedding import DevelopmentEmbeddingProvider


class EmbeddingFactory:
    """
    Factory for instantiating and managing embedding providers in NEXORA AI.
    Supports development offline deterministic embeddings and future external providers.
    """

    def __init__(self):
        self._providers: Dict[str, BaseEmbeddingProvider] = {}
        self._initialize_default_providers()

    def _initialize_default_providers(self) -> None:
        """Registers the primary offline development embedding provider."""
        dev_provider = DevelopmentEmbeddingProvider(dimension=1536)
        self._providers["development"] = dev_provider
        self._providers["development_deterministic"] = dev_provider
        self._providers["default"] = dev_provider

    def register_provider(self, name: str, provider: BaseEmbeddingProvider) -> None:
        """Registers a new or custom embedding provider."""
        if not name or not isinstance(provider, BaseEmbeddingProvider):
            raise ValueError("Valid provider name and BaseEmbeddingProvider instance required.")
        self._providers[name.lower().strip()] = provider

    def get_provider(self, name: Optional[str] = None) -> BaseEmbeddingProvider:
        """
        Retrieves the requested embedding provider.
        Falls back to the default development provider if name is None or unrecognized.
        """
        if not name or not isinstance(name, str):
            return self._providers["development"]

        key = name.lower().strip()
        if key in self._providers:
            return self._providers[key]

        return self._providers["development"]

    def list_providers(self) -> List[str]:
        """Returns the list of registered embedding provider names."""
        return sorted(list(set(self._providers.keys())))


embedding_factory = EmbeddingFactory()
