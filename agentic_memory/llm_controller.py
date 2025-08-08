from typing import Optional, Literal, List
import os
from abc import ABC, abstractmethod
from .llm.client import LLMClient
from .llm.types import LLMRequest


class BaseLLMController(ABC):
    """Abstract base for LLM controllers used in tests and implementations."""

    @abstractmethod
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.3) -> str:
        """Return a completion string for the given prompt."""
        raise NotImplementedError

    def get_embedding(self, text: str) -> List[float]:
        """Optional: return an embedding for the given text. Default is a zero vector for tests."""
        return [0.0] * 384


class LLMController:
    """LLM-based controller for memory metadata generation

    Public API remains stable for callers: `get_completion(prompt, response_format, temperature)`
    Internally delegates to a unified `LLMClient` which routes to:
    - OpenAI Responses API via the official OpenAI SDK for provider "openai"
    - LiteLLM for all other providers ("openrouter", "anthropic", "ollama")
    """

    def __init__(self,
                 backend: Literal["openai", "openrouter", "anthropic", "ollama"] = "openai",
                 model: str = "gpt-4.1-mini",
                 api_key: Optional[str] = None) -> None:
        self.provider = backend
        # If a non-OpenAI provider is selected but the model is the OpenAI default alias,
        # defer model selection to the registry by setting it to None.
        if backend != "openai" and model in {"gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"}:
            self.model = None  # registry will provide provider-specific default
        else:
            self.model = model
        # Best-effort: if api_key passed explicitly, set env for the chosen provider
        if api_key:
            if backend == "openai":
                os.environ.setdefault("OPENAI_API_KEY", api_key)
            elif backend == "anthropic":
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            elif backend == "openrouter":
                os.environ.setdefault("OPENROUTER_API_KEY", api_key)
        self.client = LLMClient()

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.3) -> str:
        messages = [
            {"role": "system", "content": "You must respond with a JSON object."},
            {"role": "user", "content": prompt},
        ]
        req = LLMRequest(
            provider=self.provider,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=1000,
            response_format=response_format,
        )
        result = self.client.generate(req)
        return result.text
