"""
OpenTelemetry utilities for the agentic-memory library.

Library-friendly behavior:
- Does NOT install or override a global TracerProvider/exporters by default.
- Uses the application's tracer/provider so traces flow into the same session.
- Optional fallback provider setup is available if AGENTIC_MEMORY_INIT_PROVIDER=true.
- Can be disabled with TELEMETRY_ENABLED=false.

Env vars:
- TELEMETRY_ENABLED: "true" (default) or "false"
- AGENTIC_MEMORY_INIT_PROVIDER: "false" (default). When true, sets up OTLP exporter.
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint, default "http://localhost:4317"
- TELEMETRY_CONSOLE_EXPORTER: "false" (default). When true, adds console exporter for debugging.
"""
from __future__ import annotations
import asyncio
import json
import os
import logging
from functools import wraps
from typing import Any, Callable, Optional

# OpenTelemetry API imports only (safe for library usage)
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

logger = logging.getLogger(__name__)

# Global variables for telemetry components
_tracer = None
_telemetry_enabled = False


def initialize_telemetry(service_name: str = "agentic-memory") -> None:
    """
    Initialize library telemetry. By default, do not install a provider/exporters
    so we join the host application's tracing session. Optionally, when
    AGENTIC_MEMORY_INIT_PROVIDER=true, install a basic OTLP exporter.
    """
    global _tracer, _telemetry_enabled

    # Respect explicit disable
    if os.getenv("TELEMETRY_ENABLED", "true").lower() == "false":
        logger.info("Telemetry disabled via TELEMETRY_ENABLED env var")
        _telemetry_enabled = False
        _tracer = None
        return

    try:
        # Optional provider/exporter initialization for standalone debugging
        init_provider = os.getenv("AGENTIC_MEMORY_INIT_PROVIDER", "false").lower() == "true"
        if init_provider:
            try:
                from opentelemetry.sdk.trace import TracerProvider  # type: ignore
                from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter  # type: ignore
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore
                from opentelemetry.sdk.resources import Resource  # type: ignore
                from opentelemetry.semconv.resource import ResourceAttributes  # type: ignore

                resource = Resource(attributes={
                    ResourceAttributes.SERVICE_NAME: service_name,
                })
                provider = TracerProvider(resource=resource)

                otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))

                if os.getenv("TELEMETRY_CONSOLE_EXPORTER", "false").lower() == "true":
                    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

                trace.set_tracer_provider(provider)
                logger.info(f"agentic-memory telemetry initialized with OTLP endpoint: {otlp_endpoint}")
            except Exception as e:
                # If provider setup fails, continue with API-only mode
                logger.warning(f"Optional provider initialization failed, continuing with API-only telemetry: {e}")

        # Always acquire a tracer from the current global provider (app-provided or optional above)
        _tracer = trace.get_tracer("agentic_memory")
        _telemetry_enabled = True
    except Exception as e:
        logger.error(f"Failed to initialize telemetry: {e}")
        _telemetry_enabled = False
        _tracer = None


def get_tracer():
    """Return the tracer if telemetry is enabled, else None."""
    return _tracer if _telemetry_enabled else None


def is_telemetry_enabled() -> bool:
    return _telemetry_enabled


def trace_function(span_name: Optional[str] = None, span_kind: SpanKind = SpanKind.INTERNAL):
    """
    Decorator to trace function execution, capturing limited args/kwargs and the result.
    - Includes `session.id` if the first arg (self) has a `session_id` attribute.
    - Truncates large values to keep spans reasonable.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not _telemetry_enabled or _tracer is None:
                return await func(*args, **kwargs)

            name = span_name or f"{func.__module__}.{func.__name__}"
            with _tracer.start_as_current_span(name, kind=span_kind) as span:
                try:
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)

                    # session id if available on self
                    if args and hasattr(args[0], "session_id"):
                        try:
                            span.set_attribute("session.id", getattr(args[0], "session_id"))
                        except Exception:
                            pass

                    # Limited args/kwargs
                    for i, arg in enumerate(args[:3]):
                        try:
                            if isinstance(arg, (str, int, float, bool)):
                                span.set_attribute(f"arg.{i}", str(arg)[:1000])
                        except Exception:
                            pass
                    # If method and first non-self arg looks like a note with id/content/tags, capture brief input
                    try:
                        start_index = 1 if args and hasattr(args[0], "__dict__") else 0
                        if len(args) > start_index:
                            candidate = args[start_index]
                            if hasattr(candidate, "id"):
                                span.set_attribute("input.note.id", getattr(candidate, "id"))
                            if hasattr(candidate, "content"):
                                span.set_attribute("input.note.content", str(getattr(candidate, "content"))[:512])
                            if hasattr(candidate, "tags"):
                                span.set_attribute("input.note.tags", str(getattr(candidate, "tags"))[:512])
                    except Exception:
                        pass
                    for key, value in list(kwargs.items())[:5]:
                        try:
                            if isinstance(value, (str, int, float, bool)) and len(key) < 50:
                                span.set_attribute(f"kwarg.{key}", str(value)[:1000])
                        except Exception:
                            pass

                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))

                    # Store structured result when possible
                    try:
                        # Special case: (should_evolve: bool, note: MemoryNote)
                        if isinstance(result, tuple) and len(result) == 2:
                            try:
                                span.set_attribute("result.should_evolve", bool(result[0]))
                            except Exception:
                                pass
                            note_obj = result[1]
                            if hasattr(note_obj, "id"):
                                try:
                                    span.set_attribute("result.note.id", getattr(note_obj, "id"))
                                except Exception:
                                    pass
                            if hasattr(note_obj, "content"):
                                try:
                                    span.set_attribute("result.note.content", str(getattr(note_obj, "content"))[:512])
                                except Exception:
                                    pass
                            if hasattr(note_obj, "tags"):
                                try:
                                    span.set_attribute("result.note.tags", str(getattr(note_obj, "tags"))[:512])
                                except Exception:
                                    pass
                        else:
                            span.set_attribute("result", str(result)[:2000])
                    except Exception:
                        pass
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not _telemetry_enabled or _tracer is None:
                return func(*args, **kwargs)

            name = span_name or f"{func.__module__}.{func.__name__}"
            with _tracer.start_as_current_span(name, kind=span_kind) as span:
                try:
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)

                    if args and hasattr(args[0], "session_id"):
                        try:
                            span.set_attribute("session.id", getattr(args[0], "session_id"))
                        except Exception:
                            pass

                    for i, arg in enumerate(args[:3]):
                        try:
                            if isinstance(arg, (str, int, float, bool)):
                                span.set_attribute(f"arg.{i}", str(arg)[:1000])
                        except Exception:
                            pass
                    for key, value in list(kwargs.items())[:5]:
                        try:
                            if isinstance(value, (str, int, float, bool)) and len(key) < 50:
                                span.set_attribute(f"kwarg.{key}", str(value)[:1000])
                        except Exception:
                            pass

                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    try:
                        if isinstance(result, tuple) and len(result) == 2:
                            try:
                                span.set_attribute("result.should_evolve", bool(result[0]))
                            except Exception:
                                pass
                            note_obj = result[1]
                            if hasattr(note_obj, "id"):
                                try:
                                    span.set_attribute("result.note.id", getattr(note_obj, "id"))
                                except Exception:
                                    pass
                            if hasattr(note_obj, "content"):
                                try:
                                    span.set_attribute("result.note.content", str(getattr(note_obj, "content"))[:512])
                                except Exception:
                                    pass
                            if hasattr(note_obj, "tags"):
                                try:
                                    span.set_attribute("result.note.tags", str(getattr(note_obj, "tags"))[:512])
                                except Exception:
                                    pass
                        else:
                            span.set_attribute("result", str(result)[:2000])
                    except Exception:
                        pass
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def trace_llm_call(span_name: str = "llm.call"):
    """Decorator specifically for tracing LLM calls with full context."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not _telemetry_enabled or _tracer is None:
                return await func(*args, **kwargs)

            with _tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                try:
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)
                    span.set_attribute("llm.system", "agentic-memory")

                    # Provider/model if present on self
                    if args and hasattr(args[0], "provider"):
                        try:
                            span.set_attribute("llm.provider", getattr(args[0], "provider"))
                        except Exception:
                            pass
                    if args and hasattr(args[0], "model"):
                        try:
                            span.set_attribute("llm.model", getattr(args[0], "model"))
                        except Exception:
                            pass

                    # Try to extract prompt from kwargs/args
                    prompt = kwargs.get("prompt") if isinstance(kwargs, dict) else None
                    if prompt is None:
                        start_index = 1 if args and hasattr(args[0], "__dict__") else 0
                        if len(args) > start_index:
                            prompt = args[start_index]
                    if prompt is None and len(args) > 1:
                        prompt = args[1]
                    if prompt is not None:
                        span.set_attribute("llm.prompt", str(prompt)[:4096])

                    # Include simple, small kwargs as llm.* params
                    for key, value in kwargs.items():
                        if key in ("prompt", "system_prompt"):
                            continue
                        if isinstance(value, (str, int, float, bool)) and len(key) < 50:
                            span.set_attribute(f"llm.{key}", value)

                    result = await func(*args, **kwargs)

                    # Capture response (string/dict)
                    if result is not None:
                        try:
                            if isinstance(result, dict):
                                span.set_attribute("llm.response", json.dumps(result)[:4096])
                            else:
                                span.set_attribute("llm.response", str(result)[:4096])
                        except Exception:
                            pass

                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not _telemetry_enabled or _tracer is None:
                return func(*args, **kwargs)

            with _tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                try:
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)
                    span.set_attribute("llm.system", "agentic-memory")

                    if args and hasattr(args[0], "provider"):
                        try:
                            span.set_attribute("llm.provider", getattr(args[0], "provider"))
                        except Exception:
                            pass
                    if args and hasattr(args[0], "model"):
                        try:
                            span.set_attribute("llm.model", getattr(args[0], "model"))
                        except Exception:
                            pass

                    prompt = kwargs.get("prompt") if isinstance(kwargs, dict) else None
                    if prompt is None:
                        start_index = 1 if args and hasattr(args[0], "__dict__") else 0
                        if len(args) > start_index:
                            prompt = args[start_index]
                    if prompt is None and len(args) > 1:
                        prompt = args[1]
                    if prompt is not None:
                        span.set_attribute("llm.prompt", str(prompt)[:4096])

                    for key, value in kwargs.items():
                        if key in ("prompt", "system_prompt"):
                            continue
                        if isinstance(value, (str, int, float, bool)) and len(key) < 50:
                            span.set_attribute(f"llm.{key}", value)

                    result = func(*args, **kwargs)

                    if result is not None:
                        try:
                            if isinstance(result, dict):
                                span.set_attribute("llm.response", json.dumps(result)[:4096])
                            else:
                                span.set_attribute("llm.response", str(result)[:4096])
                        except Exception:
                            pass

                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def add_event_to_current_span(name: str, attributes: Optional[dict] = None) -> None:
    if not _telemetry_enabled or _tracer is None:
        return
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.add_event(name, attributes or {})


def set_span_attribute(key: str, value: Any) -> None:
    if not _telemetry_enabled or _tracer is None:
        return
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.set_attribute(key, value)


# Initialize on import in API-only mode (no provider installed unless opted-in)
initialize_telemetry()
