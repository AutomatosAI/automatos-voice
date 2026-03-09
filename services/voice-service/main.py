"""Automatos Voice Service — OpenAI-compatible STT and TTS API.

Provides:
  POST /v1/audio/transcriptions  (STT via faster-whisper)
  POST /v1/audio/speech           (TTS via Kokoro or Chatterbox)
  POST /v1/audio/clone-preview    (Voice cloning preview via Chatterbox)
  GET  /health
  GET  /metrics
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
    tts_voice_clone_requests_total,
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
# Global model holders (lazy-loaded on first request)
# ---------------------------------------------------------------------------

_whisper_model: Any = None
_kokoro_pipeline: Any = None
_chatterbox_model: Any = None
_loading_locks: dict[str, asyncio.Lock] = {}


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


def _resolve_chatterbox_device() -> str:
    """Resolve device for Chatterbox model."""
    device = config.tts.chatterbox_device
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def _load_chatterbox() -> Any:
    """Load the Chatterbox TTS model. Blocking — run in executor."""
    try:
        from chatterbox.tts import ChatterboxTTS

        device = _resolve_chatterbox_device()
        logger.info(
            "Loading Chatterbox TTS model",
            extra={"device": device},
        )

        model = ChatterboxTTS.from_pretrained(device=device)

        logger.info("Chatterbox TTS model loaded successfully")
        return model
    except Exception:
        logger.exception("Failed to load Chatterbox TTS model")
        raise


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


async def _get_lock(name: str) -> asyncio.Lock:
    """Get or create a named lock for lazy model loading."""
    if name not in _loading_locks:
        _loading_locks[name] = asyncio.Lock()
    return _loading_locks[name]


async def _ensure_whisper() -> Any:
    """Lazy-load Whisper model on first STT request."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    lock = await _get_lock("whisper")
    async with lock:
        if _whisper_model is not None:
            return _whisper_model
        set_model_status("stt", "loading")
        try:
            loop = asyncio.get_running_loop()
            _whisper_model = await loop.run_in_executor(None, _load_whisper)
            set_model_status("stt", "loaded")
            model_loaded.labels(model_type="stt", model_name=config.stt.model).set(1)
            return _whisper_model
        except Exception:
            logger.exception("Failed to load Whisper model")
            set_model_status("stt", "error")
            model_loaded.labels(model_type="stt", model_name=config.stt.model).set(0)
            raise


async def _ensure_kokoro() -> Any:
    """Lazy-load Kokoro TTS on first request."""
    global _kokoro_pipeline
    if _kokoro_pipeline is not None:
        return _kokoro_pipeline

    lock = await _get_lock("kokoro")
    async with lock:
        if _kokoro_pipeline is not None:
            return _kokoro_pipeline
        set_model_status("tts", "loading")
        try:
            loop = asyncio.get_running_loop()
            _kokoro_pipeline = await loop.run_in_executor(None, _load_kokoro)
            set_model_status("tts", "loaded")
            model_loaded.labels(model_type="tts", model_name="kokoro").set(1)
            return _kokoro_pipeline
        except Exception:
            logger.exception("Failed to load Kokoro TTS pipeline")
            set_model_status("tts", "error")
            model_loaded.labels(model_type="tts", model_name="kokoro").set(0)
            raise


async def _ensure_chatterbox() -> Any:
    """Lazy-load Chatterbox TTS on first voice-clone request."""
    global _chatterbox_model
    if _chatterbox_model is not None:
        return _chatterbox_model

    lock = await _get_lock("chatterbox")
    async with lock:
        if _chatterbox_model is not None:
            return _chatterbox_model
        set_model_status("tts_chatterbox", "loading")
        try:
            loop = asyncio.get_running_loop()
            _chatterbox_model = await loop.run_in_executor(None, _load_chatterbox)
            set_model_status("tts_chatterbox", "loaded")
            model_loaded.labels(model_type="tts", model_name="chatterbox").set(1)
            return _chatterbox_model
        except Exception:
            logger.exception("Failed to load Chatterbox TTS model")
            set_model_status("tts_chatterbox", "error")
            model_loaded.labels(model_type="tts", model_name="chatterbox").set(0)
            raise


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Lightweight startup — models are lazy-loaded on first request."""
    logger.info(
        "Voice service starting (lazy model loading)",
        extra={"engine": config.tts.engine, "stt_model": config.stt.model},
    )
    yield
    # Cleanup
    _whisper_model_ref = _whisper_model
    _kokoro_ref = _kokoro_pipeline
    _chatterbox_ref = _chatterbox_model
    logger.info(
        "Voice service shut down",
        extra={
            "whisper_was_loaded": _whisper_model_ref is not None,
            "kokoro_was_loaded": _kokoro_ref is not None,
            "chatterbox_was_loaded": _chatterbox_ref is not None,
        },
    )


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
    """OpenAI-compatible TTS request body with Chatterbox extensions."""

    input: str = Field(..., min_length=1, max_length=4096, description="Text to synthesize")
    voice: str = Field(default=config.tts.default_voice, description="Voice ID")
    model: str = Field(default=config.tts.default_model, description="TTS model name")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed multiplier")
    response_format: str = Field(default="mp3", description="Output audio format")

    # Chatterbox-specific fields
    reference_audio: str | None = Field(
        default=None,
        description="Base64-encoded reference audio for voice cloning (Chatterbox only)",
    )
    exaggeration: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Emotion exaggeration level 0.0-1.0 (Chatterbox only)",
    )


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
    try:
        await _ensure_whisper()
    except Exception:
        raise HTTPException(status_code=503, detail="STT model failed to load")

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
# Reference audio helpers
# ---------------------------------------------------------------------------


def _decode_reference_audio(b64_audio: str) -> str:
    """Decode base64 reference audio to a temp file path.

    Returns the path to a temporary WAV file. Caller must clean up.
    """
    audio_bytes = base64.b64decode(b64_audio)
    if len(audio_bytes) == 0:
        raise ValueError("Reference audio is empty")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def _chatterbox_synthesize(
    text: str,
    reference_audio_path: str | None,
    exaggeration: float,
) -> tuple[np.ndarray, int]:
    """Run Chatterbox synthesis. Blocking — run in executor."""
    import torch

    kwargs: dict[str, Any] = {"exaggeration": exaggeration}
    if reference_audio_path is not None:
        kwargs["audio_prompt"] = reference_audio_path

    wav_tensor = _chatterbox_model.generate(text, **kwargs)

    # Convert torch tensor to numpy float32 array
    if isinstance(wav_tensor, torch.Tensor):
        audio_array = wav_tensor.squeeze().cpu().numpy().astype(np.float32)
    else:
        audio_array = np.asarray(wav_tensor, dtype=np.float32)

    return audio_array, config.tts.chatterbox_sample_rate


# ---------------------------------------------------------------------------
# TTS endpoint
# ---------------------------------------------------------------------------


@app.post("/v1/audio/speech")
async def synthesize(request: TTSRequest) -> Response:
    """Synthesize speech from text (OpenAI-compatible).

    Routes to Kokoro or Chatterbox based on ``request.model``.
    Falls back to the configured default engine.
    """
    # Determine which engine to use
    engine = request.model if request.model in ("kokoro", "chatterbox") else config.tts.engine
    if engine == "both":
        engine = "kokoro"  # default to kokoro when engine="both"

    try:
        if engine == "chatterbox":
            await _ensure_chatterbox()
        else:
            await _ensure_kokoro()
    except Exception:
        raise HTTPException(status_code=503, detail=f"{engine} TTS model failed to load")

    inference_queue_depth.labels(model_type="tts").inc()
    start_time = time.monotonic()
    voice = request.voice
    ref_audio_path: str | None = None

    try:
        loop = asyncio.get_running_loop()

        if engine == "chatterbox":
            # Decode reference audio if provided
            if request.reference_audio:
                ref_audio_path = await loop.run_in_executor(
                    None, _decode_reference_audio, request.reference_audio
                )

            exaggeration = (
                request.exaggeration
                if request.exaggeration is not None
                else config.tts.chatterbox_exaggeration
            )

            audio_array, sample_rate = await loop.run_in_executor(
                None,
                _chatterbox_synthesize,
                request.input,
                ref_audio_path,
                exaggeration,
            )
        else:
            # Kokoro synthesis
            def _synthesize_kokoro() -> tuple[np.ndarray, int]:
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

            audio_array, sample_rate = await loop.run_in_executor(None, _synthesize_kokoro)

        # Encode to requested format
        if request.response_format == "pcm":
            # Raw 16-bit signed PCM for streaming pipelines
            pcm_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
            audio_bytes = pcm_int16.tobytes()
            media_type = "audio/pcm"
        elif request.response_format == "wav":
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
        tts_requests_total.labels(engine=engine, voice=voice, status="success").inc()

        logger.info(
            "TTS synthesis complete",
            extra={
                "engine": engine,
                "voice": voice,
                "chars": len(request.input),
                "duration_s": round(elapsed, 3),
                "audio_duration_s": round(audio_duration, 3),
                "format": request.response_format,
                "voice_cloned": request.reference_audio is not None,
            },
        )

        return Response(content=audio_bytes, media_type=media_type)

    except HTTPException:
        raise
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        tts_duration_seconds.observe(elapsed)
        tts_requests_total.labels(engine=engine, voice=voice, status="error").inc()
        tts_errors_total.labels(engine=engine, error_type=type(exc).__name__).inc()
        logger.exception("TTS synthesis failed", extra={"engine": engine})
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from exc
    finally:
        inference_queue_depth.labels(model_type="tts").dec()
        # Clean up temp reference audio file
        if ref_audio_path is not None:
            Path(ref_audio_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Voice clone preview endpoint
# ---------------------------------------------------------------------------


@app.post("/v1/audio/clone-preview")
async def clone_preview(
    reference_audio: UploadFile = File(..., description="Reference audio file for voice cloning"),
    text: str = Form(
        default="Hello, this is a preview of my cloned voice.",
        description="Text to synthesize with the cloned voice",
        min_length=1,
        max_length=4096,
    ),
    exaggeration: float = Form(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Emotion exaggeration level",
    ),
    response_format: str = Form(default="mp3", description="Output audio format"),
) -> Response:
    """Preview a cloned voice using Chatterbox TTS.

    Upload a reference audio file and receive synthesized speech
    that mimics the reference voice.
    """
    try:
        await _ensure_chatterbox()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Chatterbox TTS model failed to load. Set TTS_ENGINE=chatterbox or TTS_ENGINE=both.",
        )

    start_time = time.monotonic()
    ref_audio_path: str | None = None

    try:
        # Read uploaded reference audio to temp file
        content = await reference_audio.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty reference audio file")

        suffix = Path(reference_audio.filename or "ref.wav").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(content)
        tmp.flush()
        tmp.close()
        ref_audio_path = tmp.name

        loop = asyncio.get_running_loop()
        audio_array, sample_rate = await loop.run_in_executor(
            None,
            _chatterbox_synthesize,
            text,
            ref_audio_path,
            exaggeration,
        )

        # Encode to requested format
        if response_format == "wav":
            audio_bytes = _encode_wav(audio_array, sample_rate)
            media_type = "audio/wav"
        else:
            audio_bytes = _encode_mp3(audio_array, sample_rate)
            media_type = "audio/mpeg"

        elapsed = time.monotonic() - start_time
        audio_duration = len(audio_array) / sample_rate

        tts_duration_seconds.observe(elapsed)
        tts_audio_duration_seconds.observe(audio_duration)
        tts_voice_clone_requests_total.labels(status="success").inc()

        logger.info(
            "Voice clone preview complete",
            extra={
                "duration_s": round(elapsed, 3),
                "audio_duration_s": round(audio_duration, 3),
                "text_chars": len(text),
                "exaggeration": exaggeration,
                "format": response_format,
            },
        )

        return Response(content=audio_bytes, media_type=media_type)

    except HTTPException:
        raise
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        tts_duration_seconds.observe(elapsed)
        tts_voice_clone_requests_total.labels(status="error").inc()
        tts_errors_total.labels(engine="chatterbox", error_type=type(exc).__name__).inc()
        logger.exception("Voice clone preview failed")
        raise HTTPException(status_code=500, detail=f"Voice clone failed: {exc}") from exc
    finally:
        if ref_audio_path is not None:
            Path(ref_audio_path).unlink(missing_ok=True)


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
