from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional

from .types import LLMRequest, LLMResult
from .registry import ModelRegistry, ModelConfig

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


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
    

    def _enforce_required_all(self, schema):
        if not isinstance(schema, dict):
            return
        t = schema.get("type")
        if t == "object":
            props = schema.get("properties", {})
            if isinstance(props, dict) and props:
                # ensure required is present and includes all property keys
                required = schema.get("required")
                prop_keys = list(props.keys())
                if not isinstance(required, list):
                    schema["required"] = prop_keys
                else:
                    # add any missing keys
                    for k in prop_keys:
                        if k not in required:
                            required.append(k)
                # recurse into each property schema
                for v in props.values():
                    self._enforce_required_all(v)
        elif t == "array":
            items = schema.get("items")
            if items:
                self._enforce_required_all(items)


    def _enforce_no_additional_properties(self, schema):
        if not isinstance(schema, dict):
            return
        t = schema.get("type")
        if t == "object":
            # root and all nested objects must explicitly set this when strict=True
            schema.setdefault("additionalProperties", False)
            props = schema.get("properties", {})
            if isinstance(props, dict):
                for _, v in props.items():
                    self._enforce_no_additional_properties(v)
        elif t == "array":
            items = schema.get("items")
            if items:
                self._enforce_no_additional_properties(items)

    def _rf_to_text_format(self, response_format: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Convert legacy response_format to Responses API 1.99.x text.format shape.
        Supports:
        - {"type":"json_object"}
        - {"type":"json_schema","json_schema":{"name":..., "schema":..., "strict":...}}  (your older shape)
        - {"type":"json_schema","name":..., "schema":..., "strict":...}                  (newer shape)
        """
        if not response_format:
            return None
        rtype = response_format.get("type")
        if rtype == "json_object":
            return {"format": {"type": "json_object"}}
    
        if rtype == "json_schema":
            name = response_format.get("name") or response_format.get("json_schema", {}).get("name") or "Result"
            schema = response_format.get("schema") or response_format.get("json_schema", {}).get("schema")
            strict = response_format.get("strict")
            if strict is None:
                strict = response_format.get("json_schema", {}).get("strict", True)  # you currently default to True
            if not schema:
                return None
        
            # Deep copy to avoid mutating caller dicts
            import copy
            schema_copy = copy.deepcopy(schema)

            # If strict, enforce "additionalProperties": false recursively
            if strict:
                self._enforce_no_additional_properties(schema_copy)
                self._enforce_required_all(schema_copy)

            return {"format": {"type": "json_schema", "name": name, "schema": schema_copy, "strict": bool(strict)}}
        
        return None
    
    def _extract_text_or_parsed(self, resp) -> str:
        parsed = getattr(resp, "output_parsed", None)
        if parsed is not None:
            # Make sure to stringify (your LLMResult.text is a string)
            return json.dumps(parsed) if not isinstance(parsed, str) else parsed

        # Fallback: concatenate any output_text from message items
        chunks = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) == "output_text":
                        chunks.append(part.text)
        return "".join(chunks)

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
            # Build parameters conditionally to avoid unsupported args
            params: Dict[str, Any] = {
                "model": cfg.model_id,
                "input": _messages_to_prompt(req.messages),   # your simple linearized prompt
                "max_output_tokens": max_out,
            }

            # Map response_format -> text.format for 1.99.x
            text_cfg = self._rf_to_text_format(response_format)
            if text_cfg:
                params["text"] = text_cfg

            # Reasoning vs temperature
            if getattr(cfg, "reasoning", False):
                # For gpt-5 / o4* use reasoning.effort only
                if req.temperature is not None:
                    def _temp_to_effort(t: float) -> str:
                        if t <= 0.3: return "minimal"
                        if t < 0.7:  return "low"
                        if t < 1.0:  return "medium"
                        return "high"
                    params["reasoning"] = {"effort": _temp_to_effort(float(req.temperature))}
            else:
                if req.temperature is not None:
                    params["temperature"] = req.temperature

            resp = client.responses.create(**params)
            raw = resp
            text = self._extract_text_or_parsed(resp)
        except Exception:
            # If feature flag is set and model is Responses-capable, fail fast
            disable_fallback = os.getenv("OPENAI_DISABLE_CHAT_FALLBACK", "").lower() in ("1", "true", "yes", "on")
            responses_capable_models = {"gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-5-mini"}
            if disable_fallback and cfg.model_id in responses_capable_models:
                raise
            # Fallback to Chat Completions if Responses path fails
            cc_params: Dict[str, Any] = {
                "model": cfg.model_id,
                "messages": [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in req.messages],
                "max_tokens": max_out,
            }

            # Only json_object is supported on Chat Completions (not json_schema)
            if response_format and response_format.get("type") == "json_object":
                cc_params["response_format"] = {"type": "json_object"}

            # Avoid sending temperature for reasoning models
            if not getattr(cfg, "reasoning", False) and temperature is not None:
                cc_params["temperature"] = temperature

            cc = client.chat.completions.create(**cc_params)
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
