"""
LLM client for interacting with Ollama.

Handles:
- Connection to the Ollama HTTP API, one cached client per model
- Listing the installed models the user can choose from
- Prompt formatting
- Answer synthesis from SQL / RAG results
"""

import logging
from typing import Any

import requests

from app.config import settings
from app.lob import LineOfBusiness, get_lob
from app.prompts.synthesis import get_synthesis_prompt

logger = logging.getLogger(__name__)

# Model families that only produce embeddings and can't answer questions.
_EMBEDDING_FAMILIES = {"bert", "nomic-bert"}


class ModelNotAvailableError(ValueError):
    """Raised when a request names a model that isn't installed or allowed."""


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


_llm_clients: dict[str, OllamaClient] = {}


def get_llm_client(model: str | None = None) -> OllamaClient:
    """Get or create the shared client for `model` (default: OLLAMA_MODEL)."""
    model = model or settings.ollama_model
    if model not in _llm_clients:
        _llm_clients[model] = OllamaClient(model=model)
    return _llm_clients[model]


def list_models() -> list[dict[str, Any]]:
    """
    Installed Ollama chat models the user may pick, smallest first.

    Embedding-only models are left out. If ALLOWED_MODELS is set only those are offered;
    otherwise every model up to MAX_MODEL_SIZE_GB on disk.

    Raises:
        requests.RequestException: if Ollama is unreachable.
    """
    response = requests.get(f"{settings.ollama_host.rstrip('/')}/api/tags", timeout=5)
    response.raise_for_status()

    models = []
    for m in response.json().get("models", []):
        details = m.get("details") or {}
        families = set(details.get("families") or [details.get("family")])
        size_gb = round(m.get("size", 0) / 1e9, 1)
        if families & _EMBEDDING_FAMILIES or "embed" in m["name"]:
            continue
        if settings.allowed_models:
            if m["name"] not in settings.allowed_models:
                continue
        elif size_gb > settings.max_model_size_gb:
            continue
        models.append(
            {
                "name": m["name"],
                "size_gb": size_gb,
                "parameter_size": details.get("parameter_size"),
                "family": details.get("family"),
                "quantization": details.get("quantization_level"),
            }
        )
    return sorted(models, key=lambda m: m["size_gb"])


def resolve_model(model: str | None) -> str:
    """Return `model` (or the default) if it can be used, else raise ModelNotAvailableError."""
    model = model or settings.ollama_model
    available = [m["name"] for m in list_models()]
    if model not in available:
        raise ModelNotAvailableError(
            f"Model {model!r} is not available. Installed: {', '.join(available) or 'none'}. "
            f"Install it with `ollama pull {model}`."
        )
    return model


def build_context(
    sql_result: dict[str, Any] | None = None,
    rag_result: dict[str, Any] | None = None,
    doc_label: str = "Record",
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
        parts.append(f"\nRelevant {doc_label.lower()} texts ({len(documents)}):")
        for doc in documents:
            text = (doc.get("text") or "")[:MAX_NARRATIVE_CHARS]
            fields = " / ".join(
                str(v) for k, v in (doc.get("fields") or {}).items() if v and k != "date_received"
            )
            parts.append(f"- [{doc_label} {doc.get('id')}] {fields}: {text}")

    return "\n".join(parts) if parts else "No data was retrieved."


def synthesize_answer(
    question: str,
    sql_result: dict[str, Any] | None = None,
    rag_result: dict[str, Any] | None = None,
    route: str = "unknown",
    lob: LineOfBusiness | None = None,
    model: str | None = None,
) -> str:
    """Synthesize a natural-language answer from SQL and/or RAG results."""
    lob = lob or get_lob()
    doc_label = lob.rag.doc_label if lob.rag else "Record"
    context = build_context(sql_result, rag_result, doc_label=doc_label)
    prompt = get_synthesis_prompt(
        question,
        context,
        route,
        dataset=lob.name,
        doc_label=doc_label,
        has_text=lob.rag is not None,
    )
    return get_llm_client(model).generate(prompt)
