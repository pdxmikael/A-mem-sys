from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Provider = Literal["openai", "openrouter", "anthropic", "ollama"]

@dataclass
class LLMTool:
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None  # JSON schema

@dataclass
class ResponseFormat:
    # Either {"type": "json_schema", "json_schema": {...}} or {"type": "json_object"}
    type: str
    json_schema: Optional[Dict[str, Any]] = None

@dataclass
class LLMRequest:
    provider: Optional[Provider] = None
    model: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: Optional[int] = 1000
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[LLMTool]] = None
    stream: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMResult:
    text: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Any] = None
