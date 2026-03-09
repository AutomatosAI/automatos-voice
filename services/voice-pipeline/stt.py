"""Custom STT service — proxies to voice-service /v1/audio/transcriptions (PRD-74 Phase 3)"""

from __future__ import annotations

import logging
import time
from typing import AsyncGenerator

import aiohttp

from pipecat.frames.frames import Frame, TranscriptionFrame, ErrorFrame
from pipecat.services.ai_services import SegmentedSTTService

logger = logging.getLogger(__name__)


class VoiceServiceSTT(SegmentedSTTService):
    """Sends audio segments to our voice-service for transcription.

    SegmentedSTTService accumulates audio during VAD speech events,
    then calls run_stt() with the complete WAV-wrapped audio when
    the user stops speaking.
    """

    def __init__(
        self,
        *,
        api_url: str,
        model: str = "whisper-1",
        language: str = "en",
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._api_url = f"{api_url}/v1/audio/transcriptions"
        self._model = model
        self._language = language

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """POST audio to voice-service, yield TranscriptionFrame."""
        start = time.monotonic()

        form = aiohttp.FormData()
        form.add_field(
            "file",
            audio,
            filename="audio.wav",
            content_type="audio/wav",
        )
        form.add_field("model", self._model)
        if self._language != "auto":
            form.add_field("language", self._language)
        form.add_field("response_format", "json")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self._api_url, data=form) as resp:
                    elapsed_ms = (time.monotonic() - start) * 1000

                    if resp.status == 200:
                        result = await resp.json()
                        text = result.get("text", "").strip()
                        if text:
                            logger.info(
                                "stt_complete",
                                extra={
                                    "text_length": len(text),
                                    "audio_bytes": len(audio),
                                    "latency_ms": round(elapsed_ms, 1),
                                },
                            )
                            yield TranscriptionFrame(
                                text=text,
                                user_id="",
                                timestamp="",
                            )
                        else:
                            logger.debug("stt_empty_result")
                    else:
                        body = await resp.text()
                        logger.error(
                            "stt_error",
                            extra={"status": resp.status, "body": body[:200]},
                        )
                        yield ErrorFrame(f"STT failed: {resp.status}")
        except Exception as e:
            logger.error("stt_exception", extra={"error": str(e)}, exc_info=True)
            yield ErrorFrame(f"STT exception: {e}")
