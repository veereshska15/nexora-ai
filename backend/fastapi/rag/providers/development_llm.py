import asyncio
import re
from typing import Any, AsyncGenerator, Dict, List, Optional
from rag.providers.base_llm import BaseLLMProvider


class DevelopmentLLMProvider(BaseLLMProvider):
    """
    Deterministic Development LLM Provider for NEXORA AI.
    Synthesizes grounded answers strictly from supplied context without external
    network requests, API keys, or GPU dependencies.
    """

    @property
    def provider_name(self) -> str:
        return "development"

    @property
    def model_name(self) -> str:
        return "development-grounded"

    def _extract_context_sources(self, prompt: str) -> List[Dict[str, str]]:
        """
        Parses structured context blocks from the assembled prompt.
        """
        # Look for headers: ### [Source: {doc} | ID: {id} | Chunk: {idx} | Lang: {lang} | Relevance: {score}]
        pattern = r"### \[Source: ([^\|]+) \| ID: ([^\|]+) \| Chunk: (\d+) \| Lang: ([^\|]+) \| Relevance: ([^\]]+)\]\s*\n(.*?)(?=\n### \[Source:|\n\nUSER QUESTION:|\Z)"
        matches = re.findall(pattern, prompt, re.DOTALL)
        sources = []
        for i, m in enumerate(matches, 1):
            sources.append({
                "citation_id": str(i),
                "marker": f"[{i}]",
                "document_name": m[0].strip(),
                "document_id": m[1].strip(),
                "chunk_index": m[2].strip(),
                "language": m[3].strip(),
                "score": m[4].strip(),
                "content": m[5].strip(),
            })
        return sources

    def _synthesize_answer(self, prompt: str) -> str:
        """
        Synthesizes a clean grounded answer referencing the supplied sources.
        """
        sources = self._extract_context_sources(prompt)

        # If no context found in prompt or empty
        if not sources:
            return "I could not find relevant information in the documents available to me."

        # Extract user question from prompt
        question_match = re.search(r"USER QUESTION:\s*(.*?)(?=\n\n|\Z)", prompt, re.DOTALL)
        question = question_match.group(1).strip() if question_match else "your question"

        # Build grounded response text
        points = []
        for s in sources:
            snippet = s["content"]
            # Clean snippet for summary line
            first_sentence = re.split(r"[\n.।]", snippet)[0].strip()
            if first_sentence:
                points.append(f"{first_sentence} {s['marker']}")

        joined_facts = "\n\n".join(points)
        answer = (
            f"Based on the retrieved documents, here is the verified information regarding {question}:\n\n"
            f"{joined_facts}"
        )
        return answer

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> str:
        """Generates complete grounded answer."""
        await asyncio.sleep(0.01)  # Yield to event loop
        return self._synthesize_answer(prompt)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Streams generated response tokens."""
        full_answer = self._synthesize_answer(prompt)
        words = full_answer.split(" ")

        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield chunk
            await asyncio.sleep(0.005)


development_llm = DevelopmentLLMProvider()
