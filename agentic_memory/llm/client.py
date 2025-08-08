from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional

from .types import LLMRequest, LLMResult
from .registry import ModelRegistry, ModelConfig


def _messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    # Simple linearized chat -> prompt string for Responses API when not using structured inputs
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _generate_empty_value(schema_type: str, schema_items: Optional[Dict[str, Any]] = None) -> Any:
    if schema_type == "array":
        return []
    if schema_type == "string":
        return ""
    if schema_type == "object":
        return {}
    if schema_type == "number":
        return 0
    if schema_type == "integer":
        return 0
    if schema_type == "boolean":
        return False
    return None


def _generate_empty_response_from_schema(response_format: Dict[str, Any]) -> Dict[str, Any]:
    if not response_format or "json_schema" not in response_format:
        return {}
    schema = response_format["json_schema"].get("schema", {})
    result: Dict[str, Any] = {}
    if "properties" in schema and isinstance(schema["properties"], dict):
        for prop_name, prop_schema in schema["properties"].items():
            result[prop_name] = _generate_empty_value(prop_schema.get("type"), prop_schema.get("items"))
    return result


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    # Find first {...} block and try to parse
    try:
        # Simple heuristic that tolerates leading text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
    except Exception:
        return None
    return None


class LLMClient:
    def __init__(self, registry: Optional[ModelRegistry] = None) -> None:
        self.registry = registry or ModelRegistry()

    def generate(self, req: LLMRequest) -> LLMResult:
        cfg: ModelConfig = self.registry.resolve(req.provider, req.model)

        if cfg.provider == "openai":
            return self._generate_openai(cfg, req)
        else:
            return self._generate_litellm(cfg, req)

    def _postprocess_json(self, text: str, response_format: Optional[Dict[str, Any]]) -> str:
        if not response_format:
            return text
        rtype = response_format.get("type")
        if rtype == "json_object":
            try:
                json.loads(text)
                return text
            except Exception:
                obj = _extract_first_json(text) or {}
                return json.dumps(obj)
        if rtype == "json_schema":
            try:
                json.loads(text)
                return text
            except Exception:
                empty_obj = _generate_empty_response_from_schema(response_format)
                return json.dumps(empty_obj)
        return text

    def _generate_openai(self, cfg: ModelConfig, req: LLMRequest) -> LLMResult:
        # Use OpenAI Responses API, with fallback to Chat Completions for robustness
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        temperature = req.temperature
        max_out = req.max_tokens
        response_format = req.response_format

        # Try Responses API
        text: Optional[str] = None
        raw: Any = None
        try:
            prompt = _messages_to_prompt(req.messages)
            resp = client.responses.create(
                model=cfg.model_id,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_out,
                response_format=response_format,
            )
            raw = resp
            # The SDK provides output_text convenience; fall back to stringify
            text = getattr(resp, "output_text", None)
            if text is None:
                text = str(resp)
        except Exception:
            # Fallback to Chat Completions if Responses path fails
            cc = client.chat.completions.create(
                model=cfg.model_id,
                messages=[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in req.messages],
                temperature=temperature,
                max_tokens=max_out,
                response_format=response_format,
            )
            raw = cc
            text = cc.choices[0].message.content or ""

        text = self._postprocess_json(text, response_format)
        return LLMResult(text=text, tool_calls=[], usage=None, raw=raw)

    def _generate_litellm(self, cfg: ModelConfig, req: LLMRequest) -> LLMResult:
        from litellm import completion
        try:
            resp = completion(
                model=cfg.model_id,
                messages=[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in req.messages],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                response_format=req.response_format,
                **(req.extra_params or {}),
            )
            text = resp.choices[0].message.content or ""
        except Exception:
            # As a last resort, produce empty JSON for schema requests
            if req.response_format:
                text = json.dumps(_generate_empty_response_from_schema(req.response_format))
                resp = None
            else:
                text = ""
                resp = None
        text = self._postprocess_json(text, req.response_format)
        return LLMResult(text=text, tool_calls=[], usage=None, raw=resp)
