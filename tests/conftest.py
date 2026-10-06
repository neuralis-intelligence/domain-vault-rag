"""Shared fixtures."""

import pytest


class FakeLLM:
    """Stands in for OllamaClient; returns queued responses and records prompts."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.responses.pop(0) if self.responses else ""

    def is_available(self) -> bool:
        return True


@pytest.fixture
def fake_llm(monkeypatch):
    """Patch every module that calls get_llm_client() to use a FakeLLM."""
    llm = FakeLLM()
    for module in ("app.llm", "app.router", "app.sql_tool"):
        monkeypatch.setattr(f"{module}.get_llm_client", lambda: llm)
    return llm
