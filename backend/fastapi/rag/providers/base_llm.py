from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Optional


class BaseLLMProvider(ABC):
    """
    Abstract Base Class for Large Language Model generation providers in NEXORA AI.
    Standardizes synchronous/asynchronous complete text generation and streaming tokens.
    Designed for seamless future plug-ins: Gemini, OpenAI, Claude, local Ollama, Hugging Face.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider service (e.g. 'development', 'gemini', 'openai')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the specific model (e.g. 'development-grounded', 'gemini-1.5-pro')."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> str:
        """
        Generates a complete grounded response string.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Streams generated response tokens/chunks asynchronously.
        """
        pass
