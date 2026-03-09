"""Custom TTS service — proxies to voice-service /v1/audio/speech (PRD-74 Phase 3)"""

from __future__ import annotations

import logging
import time
from typing import AsyncGenerator

import aiohttp

from pipecat.frames.frames import Frame, TTSAudioRawFrame, ErrorFrame
from pipecat.services.tts_service import TTSService

logger = logging.getLogger(__name__)


class VoiceServiceTTS(TTSService):
    """Sends text to our voice-service TTS endpoint, streams raw PCM audio back.

    Requests response_format=pcm to get raw 16-bit mono PCM,
    which Pipecat's transport can play directly to the browser.
    """

    def __init__(
        self,
        *,
        api_url: str,
        model: str = "kokoro",
        voice: str = "af_heart",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._api_url = f"{api_url}/v1/audio/speech"
        self._model = model
        self._voice = voice
        self._sample_rate = sample_rate

    def set_voice(self, voice: str) -> None:
        """Allow dynamic voice switching per session."""
        self._voice = voice

    def set_model(self, model: str) -> None:
        """Allow dynamic model switching per session."""
        self._model = model

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """POST text to voice-service, yield audio frames as they stream back."""
        start = time.monotonic()
        total_bytes = 0

        payload = {
            "model": self._model,
            "voice": self._voice,
            "input": text,
            "response_format": "pcm",
            "speed": 1.0,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status == 200:
                        async for chunk in resp.content.iter_chunked(4096):
                            if chunk:
                                total_bytes += len(chunk)
                                yield TTSAudioRawFrame(
                                    audio=chunk,
                                    sample_rate=self._sample_rate,
                                    num_channels=1,
                                )

                        elapsed_ms = (time.monotonic() - start) * 1000
                        logger.info(
                            "tts_complete",
                            extra={
                                "text_length": len(text),
                                "audio_bytes": total_bytes,
                                "latency_ms": round(elapsed_ms, 1),
                                "voice": self._voice,
                                "model": self._model,
                            },
                        )
                    else:
                        body = await resp.text()
                        logger.error(
                            "tts_error",
                            extra={"status": resp.status, "body": body[:200]},
                        )
                        yield ErrorFrame(f"TTS failed: {resp.status}")
        except Exception as e:
            logger.error("tts_exception", extra={"error": str(e)}, exc_info=True)
            yield ErrorFrame(f"TTS exception: {e}")
