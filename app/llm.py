"""
LLM client for interacting with Ollama.

Handles:
- Connection to the Ollama HTTP API
- Prompt formatting
- Answer synthesis from SQL / RAG results
"""

import logging
from typing import Any

import requests

from app.config import settings
from app.prompts.synthesis import get_synthesis_prompt

logger = logging.getLogger(__name__)

# Keep the synthesis context small enough for a 7B model's context window.
MAX_SQL_ROWS_IN_CONTEXT = 50
MAX_NARRATIVE_CHARS = 1000


class OllamaClient:
    """Client for interacting with an Ollama LLM."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
    ):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.timeout = timeout or settings.llm_timeout_seconds

    def _options(self, temperature: float | None, max_tokens: int | None) -> dict[str, Any]:
        # `is None` checks so that an explicit temperature of 0.0 is respected.
        return {
            "temperature": self.temperature if temperature is None else temperature,
            "num_predict": self.max_tokens if max_tokens is None else max_tokens,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a single completion for `prompt`."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self._options(temperature, max_tokens),
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.RequestException as e:
            logger.error("Ollama generation failed: %s", e)
            raise

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Chat completion over a list of {"role": ..., "content": ...} messages."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._options(temperature, max_tokens),
        }

        try:
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.RequestException as e:
            logger.error("Ollama chat failed: %s", e)
            raise

    def is_available(self) -> bool:
        """Return True if the Ollama server responds."""
        try:
            return requests.get(f"{self.host}/api/tags", timeout=3).ok
        except requests.RequestException:
            return False


_llm_client: OllamaClient | None = None


def get_llm_client() -> OllamaClient:
    """Get or create the shared LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = OllamaClient()
    return _llm_client


def build_context(
    sql_result: dict[str, Any] | None = None,
    rag_result: dict[str, Any] | None = None,
) -> str:
    """Format SQL rows and RAG narratives into a plain-text context block."""
    parts: list[str] = []

    if sql_result:
        if sql_result.get("error"):
            parts.append(f"SQL query failed: {sql_result['error']}")
        else:
            rows = sql_result.get("results", [])
            parts.append(f"SQL query: {sql_result.get('sql', '')}")
            parts.append(f"SQL results ({len(rows)} rows):")
            parts.extend(str(row) for row in rows[:MAX_SQL_ROWS_IN_CONTEXT])

    if rag_result:
        documents = rag_result.get("documents", [])
        parts.append(f"\nRelevant complaint narratives ({len(documents)}):")
        for doc in documents:
            narrative = (doc.get("narrative") or "")[:MAX_NARRATIVE_CHARS]
            parts.append(
                f"- [Complaint {doc.get('complaint_id')}] {doc.get('company')} / "
                f"{doc.get('issue')}: {narrative}"
            )

    return "\n".join(parts) if parts else "No data was retrieved."


def synthesize_answer(
    question: str,
    sql_result: dict[str, Any] | None = None,
    rag_result: dict[str, Any] | None = None,
    route: str = "unknown",
) -> str:
    """Synthesize a natural-language answer from SQL and/or RAG results."""
    context = build_context(sql_result, rag_result)
    prompt = get_synthesis_prompt(question, context, route)
    return get_llm_client().generate(prompt)
