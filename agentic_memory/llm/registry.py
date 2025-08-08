from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from .types import Provider

@dataclass(frozen=True)
class ModelConfig:
    alias: str
    provider: Provider
    model_id: str  # id used by the underlying SDK (OpenAI or LiteLLM)
    reasoning: bool = False  # whether model uses reasoning effort instead of temperature
    # Optional future: defaults like temperature, max_tokens, etc.

class ModelRegistry:
    """
    Simple in-memory registry to map model aliases to provider + model ids.
    Clients can register custom models to avoid repeating provider/base ids.
    """
    def __init__(self) -> None:
        self._models: Dict[str, ModelConfig] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        # OpenAI common models (Responses API)
        self.register(ModelConfig(
            alias="gpt-4.1-mini", provider="openai", model_id="gpt-4.1-mini"
        ))
        self.register(ModelConfig(
            alias="gpt-4o-mini", provider="openai", model_id="gpt-4o-mini"
        ))
        self.register(ModelConfig(
            alias="gpt-4.1", provider="openai", model_id="gpt-4.1"
        ))
        # Add gpt-5-mini alias (OpenAI Responses-capable model)
        self.register(ModelConfig(
            alias="gpt-5-mini", provider="openai", model_id="gpt-5-mini", reasoning=True
        ))
        # Anthropic via LiteLLM
        self.register(ModelConfig(
            alias="claude-3-5-sonnet-20240620", provider="anthropic", model_id="anthropic/claude-3-5-sonnet-20240620"
        ))
        # Ollama via LiteLLM (local)
        self.register(ModelConfig(
            alias="ollama/llama3.1", provider="ollama", model_id="ollama/llama3.1"
        ))

    def register(self, cfg: ModelConfig) -> None:
        self._models[cfg.alias] = cfg

    def get(self, alias_or_id: str) -> Optional[ModelConfig]:
        return self._models.get(alias_or_id)

    def _is_reasoning_model(self, provider: Provider, model_id: str) -> bool:
        """Heuristic: OpenAI o1/o3/o4 and gpt-5* models use reasoning effort."""
        if provider != "openai":
            return False
        prefixes = ("o1", "o3", "o4", "gpt-5")
        return any(model_id.startswith(p) for p in prefixes)

    def resolve(self, provider: Optional[Provider], model: Optional[str]) -> ModelConfig:
        """
        Resolve a provider + model to a ModelConfig. If the provided model
        matches a registered alias, use its provider/model_id. Otherwise,
        assume the provided provider and model are the final identifiers.
        """
        if model:
            found = self.get(model)
            if found:
                return found
        # Fall back to raw config
        if not provider:
            provider = "openai"  # default
        if not model:
            # default model per provider
            if provider == "openai":
                model = "gpt-4.1-mini"
            elif provider == "anthropic":
                model = "anthropic/claude-3-5-sonnet-20240620"
            elif provider == "ollama":
                model = "ollama/llama3.1"
            elif provider == "openrouter":
                # OpenRouter requires an explicit model id; no safe default
                raise ValueError("For provider 'openrouter', you must specify a concrete model id (e.g., 'openrouter/<vendor>/<model>'). Consider registering an alias in the registry.")
            else:
                model = ""
        # If model is an alias, map it
        found2 = self.get(model)
        if found2:
            return found2
        return ModelConfig(alias=model, provider=provider, model_id=model, reasoning=self._is_reasoning_model(provider, model))
