"""Voice Pipeline Configuration (PRD-74 Phase 3)"""

import os
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration for the real-time voice pipeline."""

    # Server
    host: str = os.getenv("PIPELINE_HOST", "0.0.0.0")
    port: int = int(os.getenv("PIPELINE_PORT", "8301"))

    # Voice Service (STT/TTS engines from Phase 1/2)
    voice_service_url: str = os.getenv(
        "VOICE_SERVICE_URL", "http://voice-service.railway.internal:8300"
    )

    # Orchestrator Backend
    orchestrator_url: str = os.getenv(
        "ORCHESTRATOR_URL", "http://automatos-ai-api.railway.internal:8200"
    )

    # STT
    stt_model: str = os.getenv("STT_MODEL", "whisper-1")
    stt_language: str = os.getenv("STT_LANGUAGE", "en")
    stt_sample_rate: int = int(os.getenv("STT_SAMPLE_RATE", "16000"))

    # TTS
    tts_model: str = os.getenv("TTS_MODEL", "kokoro")
    tts_voice: str = os.getenv("TTS_VOICE", "af_heart")
    tts_sample_rate: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))

    # VAD
    vad_confidence: float = float(os.getenv("VAD_CONFIDENCE", "0.7"))
    vad_start_secs: float = float(os.getenv("VAD_START_SECS", "0.2"))
    vad_stop_secs: float = float(os.getenv("VAD_STOP_SECS", "0.3"))
    vad_min_volume: float = float(os.getenv("VAD_MIN_VOLUME", "0.6"))

    # Session
    session_timeout: int = int(os.getenv("SESSION_TIMEOUT", "300"))
    max_concurrent_sessions: int = int(os.getenv("MAX_CONCURRENT_SESSIONS", "10"))

    # Metrics
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"

    # Auth
    auth_enabled: bool = os.getenv("AUTH_ENABLED", "true").lower() == "true"


config = PipelineConfig()
