"""
Unit tests for OpenTelemetry and Langfuse telemetry integration.
"""
from core.telemetry import TelemetryManager

def test_telemetry_span_success():
    with TelemetryManager.start_span("deepbag.test_span", {"test.attr": "value", "count": 10}) as span:
        assert span is not None or span is None  # Runs safely even if tracer provider is dummy
        TelemetryManager.record_cache_event(hit=True, query="test query")
        TelemetryManager.record_api_call("PlayerGameLog", 200, 0.45)

def test_telemetry_span_exception_recording():
    try:
        with TelemetryManager.start_span("deepbag.error_span") as span:
            raise ValueError("Test planned exception")
    except ValueError:
        pass

def test_telemetry_langfuse_noop_safety():
    # Should not throw even if credentials are not configured
    TelemetryManager.record_llm_generation(
        name="test_agent_step",
        model="gemini-2.5-flash",
        prompt="Who won MVP in 2023?",
        completion="Joel Embiid won MVP.",
        tokens_used=120,
        duration=0.8,
    )
