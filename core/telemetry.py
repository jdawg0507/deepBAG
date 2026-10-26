"""
OpenTelemetry (OTel) + Langfuse Telemetry & Distributed Tracing for deepBAG.
Wraps route resolution, cache hits/misses, and API latency into distributed spans.
Provides graceful no-op / local fallback when observability credentials are absent.
"""
import os
import time
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger("deepbag.telemetry")

# ---------------------------------------------------------
# OpenTelemetry Setup
# ---------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

    # Check if a tracer provider is already registered
    current_provider = trace.get_tracer_provider()
    if not isinstance(current_provider, TracerProvider):
        resource = Resource.create({"service.name": "deepbag-gateway", "environment": os.environ.get("ENV", "production")})
        provider = TracerProvider(resource=resource)
        # In test/dev mode without OTLP collector, avoid noisy console dumps unless DEBUG_OTEL is set
        if os.environ.get("DEBUG_OTEL") == "true":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("deepbag.gateway")
except Exception as e:
    logger.warning(f"Failed to initialize OpenTelemetry: {e}")
    tracer = None

# ---------------------------------------------------------
# Langfuse Setup
# ---------------------------------------------------------
try:
    from langfuse import Langfuse
    langfuse_pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    langfuse_sk = os.environ.get("LANGFUSE_SECRET_KEY")
    langfuse_host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    if langfuse_pk and langfuse_sk:
        langfuse_client = Langfuse(public_key=langfuse_pk, secret_key=langfuse_sk, host=langfuse_host)
        logger.info("Langfuse collector connected successfully.")
    else:
        langfuse_client = None
        logger.info("Langfuse credentials absent; operating in no-op telemetry mode.")
except Exception as e:
    logger.warning(f"Langfuse initialization notice: {e}")
    langfuse_client = None


class TelemetryManager:
    """
    Centralized tracing coordinator combining OpenTelemetry distributed spans
    with Langfuse agentic generation observability.
    """
    @staticmethod
    @contextmanager
    def start_span(name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Context manager for OpenTelemetry distributed tracing spans.
        Records execution duration, status codes, and custom attributes.
        """
        start_time = time.time()
        span = None
        if tracer:
            span = tracer.start_span(name)
            if attributes:
                for k, v in attributes.items():
                    if isinstance(v, (str, bool, int, float)):
                        span.set_attribute(k, v)
                    else:
                        span.set_attribute(k, str(v))

        try:
            yield span
            if span:
                span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            if span:
                span.set_status(Status(StatusCode.ERROR, description=str(exc)))
                span.record_exception(exc)
            raise
        finally:
            duration = time.time() - start_time
            if span:
                span.set_attribute("duration_seconds", duration)
                span.end()

    @staticmethod
    def record_cache_event(hit: bool, query: str):
        """Records cache lookup telemetry in active OTel span and logs."""
        current_span = trace.get_current_span() if trace else None
        if current_span and current_span.is_recording():
            current_span.set_attribute("cache.hit", hit)
            current_span.set_attribute("cache.query_sample", query[:60])

    @staticmethod
    def record_api_call(endpoint: str, status_code: int, duration: float, error: Optional[str] = None):
        """Records NBA API network execution latency, status, and errors."""
        current_span = trace.get_current_span() if trace else None
        if current_span and current_span.is_recording():
            current_span.set_attribute("nba_api.endpoint", endpoint)
            current_span.set_attribute("nba_api.status_code", status_code)
            current_span.set_attribute("nba_api.duration_seconds", duration)
            if error:
                current_span.set_attribute("nba_api.error", error)

    @staticmethod
    def record_llm_generation(
        name: str,
        model: str,
        prompt: str,
        completion: str,
        tokens_used: Optional[int] = None,
        duration: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Sends LLM generation traces and token metrics to Langfuse if configured."""
        if langfuse_client:
            try:
                langfuse_client.trace(
                    name=name,
                    metadata=metadata or {},
                ).generation(
                    name=name,
                    model=model,
                    input=prompt,
                    output=completion,
                    usage={"total": tokens_used} if tokens_used else None,
                )
            except Exception as e:
                logger.warning(f"Error sending trace to Langfuse: {e}")
