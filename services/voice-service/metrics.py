"""Prometheus metrics for the voice service.

All metric names are prefixed with `voice_` for namespace clarity.
Metrics are exposed at GET /metrics in Prometheus text format.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Dedicated registry to avoid default process/platform collectors
# that clutter dashboards. Add PROCESS_COLLECTOR back if needed.
registry = CollectorRegistry()

# ---------------------------------------------------------------------------
# STT metrics
# ---------------------------------------------------------------------------

stt_requests_total = Counter(
    "voice_stt_requests_total",
    "Total number of STT transcription requests",
    labelnames=["language", "status"],
    registry=registry,
)

stt_duration_seconds = Histogram(
    "voice_stt_duration_seconds",
    "Time spent processing STT requests (wall clock)",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=registry,
)

stt_audio_duration_seconds = Histogram(
    "voice_stt_audio_duration_seconds",
    "Duration of incoming audio files in seconds",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
    registry=registry,
)

stt_errors_total = Counter(
    "voice_stt_errors_total",
    "Total number of STT errors",
    labelnames=["error_type"],
    registry=registry,
)

# ---------------------------------------------------------------------------
# TTS metrics
# ---------------------------------------------------------------------------

tts_requests_total = Counter(
    "voice_tts_requests_total",
    "Total number of TTS synthesis requests",
    labelnames=["engine", "voice", "status"],
    registry=registry,
)

tts_duration_seconds = Histogram(
    "voice_tts_duration_seconds",
    "Time spent processing TTS requests (wall clock)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=registry,
)

tts_characters_total = Counter(
    "voice_tts_characters_total",
    "Total number of characters synthesized",
    registry=registry,
)

tts_audio_duration_seconds = Histogram(
    "voice_tts_audio_duration_seconds",
    "Duration of generated audio in seconds",
    buckets=(0.5, 1, 2, 5, 10, 30, 60),
    registry=registry,
)

tts_errors_total = Counter(
    "voice_tts_errors_total",
    "Total number of TTS errors",
    labelnames=["engine", "error_type"],
    registry=registry,
)

tts_voice_clone_requests_total = Counter(
    "voice_tts_voice_clone_requests_total",
    "Total number of voice clone preview requests",
    labelnames=["status"],
    registry=registry,
)

# ---------------------------------------------------------------------------
# System metrics
# ---------------------------------------------------------------------------

model_loaded = Gauge(
    "voice_model_loaded",
    "Whether a model is loaded and ready (1=yes, 0=no)",
    labelnames=["model_type", "model_name"],
    registry=registry,
)

inference_queue_depth = Gauge(
    "voice_inference_queue_depth",
    "Number of requests waiting for inference",
    labelnames=["model_type"],
    registry=registry,
)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    """Expose Prometheus metrics in text format."""
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
    )
