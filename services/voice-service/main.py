"""Automatos Voice Service — OpenAI-compatible STT and TTS API.

Provides:
  POST /v1/audio/transcriptions  (STT via faster-whisper)
  POST /v1/audio/speech           (TTS via Kokoro)
  GET  /health
  GET  /metrics
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pythonjsonlogger import jsonlogger

from .config import config
from .health import router as health_router, set_model_status
from .metrics import (
    inference_queue_depth,
    model_loaded,
    router as metrics_router,
    stt_audio_duration_seconds,
    stt_duration_seconds,
    stt_errors_total,
    stt_requests_total,
    tts_audio_duration_seconds,
    tts_characters_total,
    tts_duration_seconds,
    tts_errors_total,
    tts_requests_total,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_handler = logging.StreamHandler()
_log_handler.setFormatter(
    jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "component"},
    )
)

logger = logging.getLogger("voice-service")
logger.handlers = [_log_handler]
logger.setLevel(config.log_level.upper())

# Suppress noisy library loggers
for noisy in ("faster_whisper", "ctranslate2", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Global model holders (set during lifespan)
# ---------------------------------------------------------------------------

_whisper_model: Any = None
_kokoro_pipeline: Any = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _resolve_device() -> str:
    """Resolve 'auto' device to cpu or cuda."""
    device = config.stt.device
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def _load_whisper() -> Any:
    """Load the faster-whisper model. Blocking — run in executor."""
    from faster_whisper import WhisperModel

    device = _resolve_device()
    compute_type = config.stt.compute_type
    # CPU doesn't support float16
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    logger.info(
        "Loading Whisper model",
        extra={
            "model": config.stt.model,
            "device": device,
            "compute_type": compute_type,
        },
    )

    model = WhisperModel(
        config.stt.model,
        device=device,
        compute_type=compute_type,
        download_root=config.model_cache_dir,
    )

    logger.info("Whisper model loaded successfully")
    return model


def _load_kokoro() -> Any:
    """Load the Kokoro TTS pipeline. Blocking — run in executor."""
    try:
        import kokoro

        logger.info(
            "Loading Kokoro TTS pipeline",
            extra={"voice": config.tts.default_voice},
        )

        pipeline = kokoro.KPipeline(lang_code="a")

        logger.info("Kokoro TTS pipeline loaded successfully")
        return pipeline
    except Exception:
        logger.exception("Failed to load Kokoro TTS pipeline")
        raise


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Load models in background on startup, clean up on shutdown."""
    global _whisper_model, _kokoro_pipeline

    loop = asyncio.get_running_loop()

    # Load STT
    set_model_status("stt", "loading")
    try:
        _whisper_model = await loop.run_in_executor(None, _load_whisper)
        set_model_status("stt", "loaded")
        model_loaded.labels(model_type="stt", model_name=config.stt.model).set(1)
    except Exception:
        logger.exception("Failed to load Whisper model")
        set_model_status("stt", "error")
        model_loaded.labels(model_type="stt", model_name=config.stt.model).set(0)

    # Load TTS
    set_model_status("tts", "loading")
    try:
        _kokoro_pipeline = await loop.run_in_executor(None, _load_kokoro)
        set_model_status("tts", "loaded")
        model_loaded.labels(model_type="tts", model_name=config.tts.default_model).set(1)
    except Exception:
        logger.exception("Failed to load Kokoro TTS pipeline")
        set_model_status("tts", "error")
        model_loaded.labels(model_type="tts", model_name=config.tts.default_model).set(0)

    yield

    # Cleanup
    _whisper_model = None
    _kokoro_pipeline = None
    logger.info("Voice service shut down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Automatos Voice Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
if config.metrics_enabled:
    app.include_router(metrics_router)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request body."""

    input: str = Field(..., min_length=1, max_length=4096, description="Text to synthesize")
    voice: str = Field(default=config.tts.default_voice, description="Voice ID")
    model: str = Field(default=config.tts.default_model, description="TTS model name")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed multiplier")
    response_format: str = Field(default="mp3", description="Output audio format")


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response."""

    text: str


# ---------------------------------------------------------------------------
# WAV encoding helper
# ---------------------------------------------------------------------------


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 numpy array to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _encode_mp3(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 numpy array to MP3 bytes via a temp WAV + ffmpeg fallback.

    If pydub is available, use it. Otherwise fall back to raw WAV.
    """
    try:
        from pydub import AudioSegment

        wav_bytes = _encode_wav(audio, sample_rate)
        segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        mp3_buf = io.BytesIO()
        segment.export(mp3_buf, format="mp3")
        mp3_buf.seek(0)
        return mp3_buf.read()
    except ImportError:
        # Fallback: return WAV if pydub not installed
        logger.warning("pydub not installed — returning WAV instead of MP3")
        return _encode_wav(audio, sample_rate)


# ---------------------------------------------------------------------------
# STT endpoint
# ---------------------------------------------------------------------------


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str | None = Form(default=None, description="ISO 639-1 language code"),
    model: str = Form(default="whisper-1", description="Model name (ignored, uses configured model)"),
) -> TranscriptionResponse:
    """Transcribe an audio file to text (OpenAI-compatible)."""
    if _whisper_model is None:
        raise HTTPException(status_code=503, detail="STT model is not loaded")

    inference_queue_depth.labels(model_type="stt").inc()
    start_time = time.monotonic()
    lang_label = language or "auto"

    try:
        # Read uploaded file to a temp file (faster-whisper needs a file path)
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(content)
            tmp.flush()
            tmp.close()

            # Get audio duration for metrics
            try:
                info = sf.info(tmp.name)
                stt_audio_duration_seconds.observe(info.duration)
            except Exception:
                pass  # non-critical — don't fail on duration extraction

            # Run inference in thread pool
            loop = asyncio.get_running_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: _whisper_model.transcribe(
                    tmp.name,
                    language=language,
                    beam_size=config.stt.beam_size,
                    vad_filter=config.stt.vad_filter,
                ),
            )

            # Collect all segment texts
            text_parts = await loop.run_in_executor(
                None,
                lambda: [segment.text for segment in segments],
            )
            full_text = " ".join(text_parts).strip()

        finally:
            Path(tmp.name).unlink(missing_ok=True)

        elapsed = time.monotonic() - start_time
        stt_duration_seconds.observe(elapsed)
        stt_requests_total.labels(language=lang_label, status="success").inc()

        logger.info(
            "Transcription complete",
            extra={
                "language": lang_label,
                "duration_s": round(elapsed, 3),
                "text_length": len(full_text),
            },
        )

        return TranscriptionResponse(text=full_text)

    except HTTPException:
        raise
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        stt_duration_seconds.observe(elapsed)
        stt_requests_total.labels(language=lang_label, status="error").inc()
        stt_errors_total.labels(error_type=type(exc).__name__).inc()
        logger.exception("STT transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        inference_queue_depth.labels(model_type="stt").dec()


# ---------------------------------------------------------------------------
# TTS endpoint
# ---------------------------------------------------------------------------


@app.post("/v1/audio/speech")
async def synthesize(request: TTSRequest) -> Response:
    """Synthesize speech from text (OpenAI-compatible)."""
    if _kokoro_pipeline is None:
        raise HTTPException(status_code=503, detail="TTS model is not loaded")

    inference_queue_depth.labels(model_type="tts").inc()
    start_time = time.monotonic()
    voice = request.voice

    try:
        loop = asyncio.get_running_loop()

        # Run Kokoro synthesis in thread pool
        def _synthesize() -> tuple[np.ndarray, int]:
            """Generate audio samples from text."""
            samples_list: list[np.ndarray] = []
            sample_rate = config.tts.sample_rate

            for _gs, _ps, audio in _kokoro_pipeline(
                request.input,
                voice=request.voice,
                speed=request.speed,
            ):
                if audio is not None:
                    samples_list.append(audio)

            if not samples_list:
                raise ValueError("Kokoro produced no audio output")

            combined = np.concatenate(samples_list)
            return combined, sample_rate

        audio_array, sample_rate = await loop.run_in_executor(None, _synthesize)

        # Encode to requested format
        if request.response_format == "wav":
            audio_bytes = _encode_wav(audio_array, sample_rate)
            media_type = "audio/wav"
        else:
            audio_bytes = _encode_mp3(audio_array, sample_rate)
            media_type = "audio/mpeg"

        elapsed = time.monotonic() - start_time
        audio_duration = len(audio_array) / sample_rate

        tts_duration_seconds.observe(elapsed)
        tts_characters_total.inc(len(request.input))
        tts_audio_duration_seconds.observe(audio_duration)
        tts_requests_total.labels(voice=voice, status="success").inc()

        logger.info(
            "TTS synthesis complete",
            extra={
                "voice": voice,
                "chars": len(request.input),
                "duration_s": round(elapsed, 3),
                "audio_duration_s": round(audio_duration, 3),
                "format": request.response_format,
            },
        )

        return Response(content=audio_bytes, media_type=media_type)

    except HTTPException:
        raise
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        tts_duration_seconds.observe(elapsed)
        tts_requests_total.labels(voice=voice, status="error").inc()
        tts_errors_total.labels(error_type=type(exc).__name__).inc()
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from exc
    finally:
        inference_queue_depth.labels(model_type="tts").dec()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level,
    )
