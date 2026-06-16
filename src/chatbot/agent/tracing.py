"""OpenTelemetry observability for the OpenAI Agents SDK runner.

Bridges Agents SDK traces/spans to an OTel ``TracerProvider`` with a console
exporter and a simple redaction layer (emails / phone numbers).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from agents import set_trace_processors, set_tracing_disabled
from agents.tracing import Span, Trace, TracingProcessor
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from chatbot.settings.app_config import TracingConfig, load_config

logger = logging.getLogger(__name__)

_REDACTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (
        re.compile(r"\b\+?\d{1,3}[\s\-()]?\d{2,4}[\s\-()]?\d{3,4}[\s\-()]?\d{3,4}\b"),
        "[REDACTED_PHONE]",
    ),
]
_GUARDRAIL_ERROR_RE = re.compile(r"GUARDRAIL\[(?P<stage>[^:]+):(?P<code>[^\]]+)\]")

def _redact(text: str) -> str:
    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

def setup_opentelemetry(cfg: Optional[TracingConfig] = None) -> TracerProvider:
    cfg = cfg or load_config().tracing
    provider = TracerProvider()
    for exporter in cfg.exporters:
        if exporter == "console":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        else:
            logger.warning("Unsupported tracing exporter %r - ignoring", exporter)
    otel_trace.set_tracer_provider(provider)
    return provider

class OtelTracingProcessor(TracingProcessor):
    def __init__(self, cfg: Optional[TracingConfig] = None) -> None:
        self._cfg = cfg or load_config().tracing
        self._tracer = otel_trace.get_tracer("ai-final-chatbot")
        self._otel_spans: Dict[str, otel_trace.Span] = {}

    def _sanitize_output(self, output: Any) -> str:
        if not isinstance(output, str):
            try:
                output = json.dumps(output, default=str)
            except (TypeError, ValueError):
                output = str(output)
        if self._cfg.redact_outputs:
            output = _redact(output)
        return output[: self._cfg.max_output_chars]

    def on_trace_start(self, trace: Trace) -> None:
        if not self._cfg.enabled:
            return
        otel_span = self._tracer.start_span(
            name=f"trace:{trace.name}",
            attributes={"trace.id": trace.trace_id},
        )
        self._otel_spans[trace.trace_id] = otel_span

    def on_trace_end(self, trace: Trace) -> None:
        if not self._cfg.enabled:
            return
        otel_span = self._otel_spans.pop(trace.trace_id, None)
        if otel_span:
            otel_span.end()

    def on_span_start(self, span: Span[Any]) -> None:
        if not self._cfg.enabled:
            return
        span_data = span.span_data
        span_type = type(span_data).__name__
        attributes: Dict[str, str] = {"span.type": span_type, "span.id": span.span_id}
        span_name = span_type
        if hasattr(span_data, "name") and span_data.name:
            span_name = f"{span_type}:{span_data.name}"
            attributes["name"] = span_data.name
        otel_span = self._tracer.start_span(name=span_name, attributes=attributes)
        self._otel_spans[span.span_id] = otel_span

    def on_span_end(self, span: Span[Any]) -> None:
        if not self._cfg.enabled:
            return
        span_data = span.span_data
        otel_span = self._otel_spans.pop(span.span_id, None)
        if otel_span:
            if span.error:
                otel_span.set_attribute("error", True)
                message = str(span.error.message)
                otel_span.set_attribute("error.message", message)
                match = _GUARDRAIL_ERROR_RE.search(message)
                if match:
                    otel_span.set_attribute("guardrail.blocked", True)
                    otel_span.set_attribute("guardrail.stage", match.group("stage"))
                    otel_span.set_attribute("guardrail.code", match.group("code"))
            if hasattr(span_data, "output") and span_data.output:
                otel_span.set_attribute("output", self._sanitize_output(span_data.output))
            otel_span.end()

    def shutdown(self) -> None:
        for otel_span in self._otel_spans.values():
            otel_span.end()
        self._otel_spans.clear()

    def force_flush(self) -> None:
        pass

def install_tracing(cfg: Optional[TracingConfig] = None) -> Optional[OtelTracingProcessor]:
    cfg = cfg or load_config().tracing
    if not cfg.enabled:
        set_tracing_disabled(True)
        return None
    setup_opentelemetry(cfg)
    processor = OtelTracingProcessor(cfg)
    set_trace_processors([processor])
    return processor
