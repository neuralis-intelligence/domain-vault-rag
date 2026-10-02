"""
LLM client for interacting with Ollama.

Handles:
- Connection to Ollama API
- Prompt formatting
- Response parsing
- Fallback to Groq API if configured
"""

import os
import requests
from typing import Optional, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama LLM."""

    def __init__(
        self,
        host: str = None,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response from Ollama.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated text response
        """
        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Chat completion with conversation history.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated text response
        """
        url = f"{self.host}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")

        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise


# Singleton client
_llm_client = None


def get_llm_client() -> OllamaClient:
    """Get or create the LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = OllamaClient()
    return _llm_client


def synthesize_answer(
    question: str,
    sql_result: Optional[Dict[str, Any]] = None,
    rag_result: Optional[Dict[str, Any]] = None,
    route: str = "unknown"
) -> str:
    """
    Synthesize a final answer from SQL and/or RAG results.

    Args:
        question: Original user question
        sql_result: Results from BigQuery (if sql/hybrid route)
        rag_result: Results from Qdrant (if rag/hybrid route)
        route: Which route was taken

    Returns:
        Natural language answer
    """
    client = get_llm_client()

    # Build context from results
    context_parts = []

    if sql_result:
        context_parts.append("SQL Query Results:")
        context_parts.append(str(sql_result))

    if rag_result:
        context_parts.append("\nRelevant Complaint Narratives:")
        for doc in rag_result.get("documents", []):
            context_parts.append(f"- {doc.get('narrative', '')}")

    context = "\n".join(context_parts)

    # Load synthesis prompt
    from app.prompts.synthesis import get_synthesis_prompt

    prompt = get_synthesis_prompt(question, context, route)

    # Generate answer
    answer = client.generate(prompt)

    return answer
