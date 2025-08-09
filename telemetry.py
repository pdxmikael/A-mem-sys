"""
OpenTelemetry setup and instrumentation for the Antho-RPG application.
"""
import asyncio
import json
import os
import logging
from functools import wraps
from typing import Any, Callable, Optional

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode

# Initialize logger
logger = logging.getLogger(__name__)

# Global variables for telemetry components
_tracer = None
_telemetry_enabled = False

def initialize_telemetry(service_name: str = "antho-rpg"):
    """
    Initialize OpenTelemetry tracing.
    
    Args:
        service_name: Name of the service for telemetry identification
    """
    global _tracer, _telemetry_enabled
    
    # Check if telemetry is explicitly disabled
    if os.getenv("TELEMETRY_ENABLED", "true").lower() == "false":
        logger.info("Telemetry is disabled via TELEMETRY_ENABLED environment variable")
        _telemetry_enabled = False
        return
    
    try:
        # Create resource with service name
        resource = Resource(attributes={
            ResourceAttributes.SERVICE_NAME: service_name,
            ResourceAttributes.SERVICE_VERSION: "0.4.0",
        })
        
        # Create tracer provider
        provider = TracerProvider(resource=resource)
        
        # Set up OTLP exporter (default to localhost:4317)
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        
        # Add span processor
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        
        # Add console exporter only if explicitly enabled for debugging
        if os.getenv("TELEMETRY_CONSOLE_EXPORTER", "false").lower() == "true":
            console_exporter = ConsoleSpanExporter()
            console_processor = BatchSpanProcessor(console_exporter)
            provider.add_span_processor(console_processor)
            logger.info("Console exporter enabled for debugging")
        
        # Set global tracer provider
        trace.set_tracer_provider(provider)
        
        # Get tracer
        _tracer = trace.get_tracer(__name__)
        
        _telemetry_enabled = True
        logger.info(f"Telemetry initialized successfully for service: {service_name}")
        logger.info(f"OTLP endpoint: {otlp_endpoint}")
        
    except Exception as e:
        logger.error(f"Failed to initialize telemetry: {e}")
        _telemetry_enabled = False

def get_tracer():
    """
    Get the global tracer instance.
    
    Returns:
        Tracer instance or None if telemetry is disabled
    """
    return _tracer if _telemetry_enabled else None

def is_telemetry_enabled():
    """
    Check if telemetry is enabled.
    
    Returns:
        bool: True if telemetry is enabled, False otherwise
    """
    return _telemetry_enabled

def trace_function(span_name: Optional[str] = None, span_kind: SpanKind = SpanKind.INTERNAL):
    """
    Decorator to trace function execution.
    
    Args:
        span_name: Name of the span (defaults to function name)
        span_kind: Kind of the span
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not _telemetry_enabled or _tracer is None:
                return await func(*args, **kwargs)
            
            name = span_name or f"{func.__module__}.{func.__name__}"
            with _tracer.start_as_current_span(name, kind=span_kind) as span:
                try:
                    # Add function details as attributes
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)
                    
                    # Add arguments as attributes (be careful with sensitive data)
                    # Only add simple types to avoid large payloads
                    for i, arg in enumerate(args[:3]):  # Limit to first 3 args
                        if isinstance(arg, (str, int, float, bool)):
                            span.set_attribute(f"arg.{i}", str(arg)[:1000])  # Limit size
                    
                    for key, value in list(kwargs.items())[:5]:  # Limit to first 5 kwargs
                        if isinstance(value, (str, int, float, bool)) and len(key) < 50:
                            span.set_attribute(f"kwarg.{key}", str(value)[:1000])  # Limit size
                    
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    
                    # Add result as attribute if it's a simple type
                    if isinstance(result, (str, int, float, bool)):
                        span.set_attribute("result", str(result)[:1000])  # Limit size
                    
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
                    # Add function details as attributes
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)
                    
                    # Add arguments as attributes (be careful with sensitive data)
                    # Only add simple types to avoid large payloads
                    for i, arg in enumerate(args[:3]):  # Limit to first 3 args
                        if isinstance(arg, (str, int, float, bool)):
                            span.set_attribute(f"arg.{i}", str(arg)[:1000])  # Limit size
                    
                    for key, value in list(kwargs.items())[:5]:  # Limit to first 5 kwargs
                        if isinstance(value, (str, int, float, bool)) and len(key) < 50:
                            span.set_attribute(f"kwarg.{key}", str(value)[:1000])  # Limit size
                    
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    
                    # Add result as attribute if it's a simple type
                    if isinstance(result, (str, int, float, bool)):
                        span.set_attribute("result", str(result)[:1000])  # Limit size
                    
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        # Return appropriate wrapper based on function type
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def trace_llm_call(span_name: str = "llm.call"):
    """
    Decorator specifically for tracing LLM calls with full context.
    
    Args:
        span_name: Name of the span
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not _telemetry_enabled or _tracer is None:
                return await func(*args, **kwargs)
            
            with _tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                try:
                    # Add LLM-specific attributes
                    span.set_attribute("llm.system", "custom")
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)
                    
                    # Extract prompt information more intelligently
                    # For methods, args[0] is self, args[1] is usually the prompt
                    # For functions, args[0] is usually the prompt
                    prompt = None
                    system_prompt = None
                    
                    # Check kwargs first
                    if "prompt" in kwargs:
                        prompt = kwargs["prompt"]
                    if "system_prompt" in kwargs:
                        system_prompt = kwargs["system_prompt"]
                    
                    # If not in kwargs, check args
                    if prompt is None:
                        # Skip self/cls argument for methods
                        start_index = 1 if args and hasattr(args[0], '__dict__') else 0
                        if len(args) > start_index:
                            prompt = args[start_index]
                    
                    # If we still don't have prompt from args, check if it's the second arg (for methods)
                    if prompt is None and len(args) > 1:
                        prompt = args[1]
                    
                    # Add prompts to span if found
                    if prompt is not None:
                        span.set_attribute("llm.prompt", str(prompt)[:4096])  # Limit size
                    
                    if system_prompt is not None:
                        span.set_attribute("llm.system_prompt", str(system_prompt)[:4096])  # Limit size
                    
                    # Add all kwargs as attributes (LLM parameters)
                    for key, value in kwargs.items():
                        # Skip prompt and system_prompt as they're already handled
                        if key not in ["prompt", "system_prompt"]:
                            # Only add simple types to avoid large payloads
                            if isinstance(value, (str, int, float, bool)) and len(key) < 50:
                                span.set_attribute(f"llm.{key}", value)
                    
                    # Execute the function
                    result = await func(*args, **kwargs)
                    
                    # Add response to span
                    if result is not None:
                        # For structured responses, serialize to JSON
                        if isinstance(result, dict):
                            try:
                                response_str = json.dumps(result)
                                span.set_attribute("llm.response", response_str[:4096])  # Limit size
                            except (TypeError, ValueError):
                                # Fallback if JSON serialization fails
                                span.set_attribute("llm.response", str(result)[:4096])  # Limit size
                        else:
                            span.set_attribute("llm.response", str(result)[:4096])  # Limit size
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        # Store reference to original function for manual attribute setting
        async_wrapper._original_func = func
        return async_wrapper
    
    return decorator

def add_event_to_current_span(name: str, attributes: Optional[dict] = None):
    """
    Add an event to the current active span.
    
    Args:
        name: Name of the event
        attributes: Optional attributes to add to the event
    """
    if not _telemetry_enabled or _tracer is None:
        return
    
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.add_event(name, attributes or {})

def set_span_attribute(key: str, value: Any):
    """
    Set an attribute on the current active span.
    
    Args:
        key: Attribute key
        value: Attribute value
    """
    if not _telemetry_enabled or _tracer is None:
        return
    
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.set_attribute(key, value)

# Initialize telemetry on module import
initialize_telemetry()
