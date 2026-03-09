"""Centralized configuration for the voice service.

All settings are read from environment variables with sensible defaults.
No os.getenv() calls should exist outside this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class STTConfig:
    """Speech-to-text configuration."""

    model: str = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-large-v3")
    device: str = os.getenv("WHISPER_DEVICE", "auto")
    compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    beam_size: int = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
    vad_filter: bool = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"


@dataclass(frozen=True)
class TTSConfig:
    """Text-to-speech configuration."""

    default_model: str = os.getenv("TTS_DEFAULT_MODEL", "kokoro")
    default_voice: str = os.getenv("TTS_DEFAULT_VOICE", "af_heart")
    sample_rate: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))


@dataclass(frozen=True)
class ServiceConfig:
    """Top-level service configuration."""

    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "info")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8300"))
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "/app/models")

    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


# Singleton instance — import this everywhere
config = ServiceConfig()
