"""Voice Pipeline Service — Real-time conversational AI (PRD-74 Phase 3)

Provides:
  WS   /ws/voice        — Bidirectional voice streaming (Pipecat pipeline)
  GET  /health           — Health check
  GET  /metrics          — Prometheus metrics
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pythonjsonlogger import jsonlogger

from .config import config
from .stt import VoiceServiceSTT
from .tts import VoiceServiceTTS
from .orchestrator_processor import OrchestratorProcessor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
handler = logging.StreamHandler()
handler.setFormatter(
    jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "name": "component"},
    )
)
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)

for noisy in ("httpx", "httpcore", "websockets"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("voice-pipeline")

# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------
voice_sessions_active = Gauge(
    "voice_pipeline_sessions_active",
    "Number of active voice sessions",
)
voice_sessions_total = Counter(
    "voice_pipeline_sessions_total",
    "Total voice sessions started",
    ["status"],
)
voice_session_duration = Histogram(
    "voice_pipeline_session_duration_seconds",
    "Voice session duration in seconds",
    buckets=[10, 30, 60, 120, 300, 600],
)
voice_turns_total = Counter(
    "voice_pipeline_turns_total",
    "Total voice conversation turns",
)

# ---------------------------------------------------------------------------
# Session tracking
# ---------------------------------------------------------------------------
_active_sessions: dict[str, float] = {}

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(
        "voice_pipeline_starting",
        extra={
            "voice_service_url": config.voice_service_url,
            "orchestrator_url": config.orchestrator_url,
            "port": config.port,
            "max_sessions": config.max_concurrent_sessions,
        },
    )
    yield
    logger.info("voice_pipeline_shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Automatos Voice Pipeline",
    description="Real-time conversational AI via Pipecat (PRD-74 Phase 3)",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "healthy",
        "service": "voice-pipeline",
        "active_sessions": len(_active_sessions),
        "max_sessions": config.max_concurrent_sessions,
        "voice_service_url": config.voice_service_url,
        "orchestrator_url": config.orchestrator_url,
    })


if config.metrics_enabled:
    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )


@app.websocket("/ws/voice")
async def voice_websocket(
    websocket: WebSocket,
    workspace_id: str = Query(...),
    agent_id: int = Query(None),
    token: str = Query(None),
    conversation_id: str = Query(None),
):
    """Real-time voice conversation via Pipecat pipeline.

    Connect with: ws://host:8301/ws/voice?workspace_id=xxx&token=jwt&agent_id=123

    Audio format:
      - Input:  16-bit PCM, 16kHz, mono (wrapped in protobuf frames)
      - Output: 16-bit PCM, 24kHz, mono (wrapped in protobuf frames)

    Use @pipecat-ai/client-js with WebSocketTransport on the frontend.
    """
    # Enforce session limit
    if len(_active_sessions) >= config.max_concurrent_sessions:
        await websocket.close(code=1013, reason="Too many active voice sessions")
        voice_sessions_total.labels(status="rejected").inc()
        return

    await websocket.accept()
    session_id = conversation_id or f"vs_{id(websocket)}"
    session_start = time.monotonic()
    _active_sessions[session_id] = session_start
    voice_sessions_active.inc()
    voice_sessions_total.labels(status="started").inc()

    logger.info(
        "voice_session_started",
        extra={
            "session_id": session_id,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "active_sessions": len(_active_sessions),
        },
    )

    try:
        await _run_pipeline(
            websocket=websocket,
            workspace_id=workspace_id,
            agent_id=agent_id,
            auth_token=token,
            session_id=session_id,
        )
    except WebSocketDisconnect:
        logger.info("voice_session_disconnected", extra={"session_id": session_id})
    except Exception as e:
        logger.error(
            "voice_session_error",
            extra={"session_id": session_id, "error": str(e)},
            exc_info=True,
        )
        voice_sessions_total.labels(status="error").inc()
    finally:
        duration = time.monotonic() - session_start
        _active_sessions.pop(session_id, None)
        voice_sessions_active.dec()
        voice_session_duration.observe(duration)
        logger.info(
            "voice_session_ended",
            extra={
                "session_id": session_id,
                "duration_s": round(duration, 1),
                "active_sessions": len(_active_sessions),
            },
        )


async def _run_pipeline(
    websocket: WebSocket,
    workspace_id: str,
    agent_id: int | None,
    auth_token: str | None,
    session_id: str,
) -> None:
    """Build and run a Pipecat pipeline for this WebSocket connection."""
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask, PipelineParams
    from pipecat.serializers.protobuf import ProtobufFrameSerializer
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams

    # 1. Transport — bidirectional audio over WebSocket
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=config.vad_confidence,
                    start_secs=config.vad_start_secs,
                    stop_secs=config.vad_stop_secs,
                    min_volume=config.vad_min_volume,
                )
            ),
            serializer=ProtobufFrameSerializer(),
        ),
    )

    # 2. STT — proxy to voice-service
    stt = VoiceServiceSTT(
        api_url=config.voice_service_url,
        model=config.stt_model,
        language=config.stt_language,
        sample_rate=config.stt_sample_rate,
    )

    # 3. Orchestrator bridge — sends transcript, gets AI response
    orchestrator = OrchestratorProcessor(
        orchestrator_url=config.orchestrator_url,
        workspace_id=workspace_id,
        agent_id=agent_id,
        auth_token=auth_token,
    )
    orchestrator.set_conversation_id(session_id)

    # 4. TTS — proxy to voice-service
    tts = VoiceServiceTTS(
        api_url=config.voice_service_url,
        model=config.tts_model,
        voice=config.tts_voice,
        sample_rate=config.tts_sample_rate,
    )

    # 5. Wire the pipeline:
    #    mic audio → VAD → STT → Orchestrator → TTS → speaker audio
    pipeline = Pipeline([
        transport.input(),
        stt,
        orchestrator,
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=config.metrics_enabled,
        ),
    )

    runner = PipelineRunner()
    await runner.run(task)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "services.voice-pipeline.main:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )
