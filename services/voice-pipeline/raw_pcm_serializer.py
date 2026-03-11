"""Raw PCM Frame Serializer for browser WebSocket clients.

Browsers send raw Int16 PCM audio as ArrayBuffer over WebSocket.
This serializer converts those binary messages into Pipecat InputAudioRawFrame
objects, and serializes OutputAudioRawFrame back to raw bytes for playback.

No protobuf, no framing — just raw PCM audio in both directions.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

logger = logging.getLogger(__name__)


class RawPCMSerializer(FrameSerializer):
    """Serialize/deserialize raw PCM audio for browser WebSocket clients.

    Input (browser → server):  raw bytes = Int16 PCM, 16kHz mono
    Output (server → browser): raw bytes = Int16 PCM, 24kHz mono
    """

    def __init__(
        self,
        *,
        sample_rate_in: int = 16000,
        sample_rate_out: int = 24000,
        num_channels: int = 1,
    ):
        super().__init__()
        self._sample_rate_in = sample_rate_in
        self._sample_rate_out = sample_rate_out
        self._num_channels = num_channels

    def serialize(self, frame: Frame) -> Optional[bytes]:
        """Serialize outgoing frames to raw bytes for the browser."""
        if isinstance(frame, (OutputAudioRawFrame, TTSAudioRawFrame)):
            return frame.audio
        return None

    def deserialize(self, data: bytes | str) -> Optional[Frame]:
        """Deserialize incoming raw PCM bytes from the browser."""
        if isinstance(data, bytes) and len(data) > 0:
            return InputAudioRawFrame(
                audio=data,
                sample_rate=self._sample_rate_in,
                num_channels=self._num_channels,
            )
        if isinstance(data, str):
            # Browser might send JSON control messages — ignore for now
            try:
                msg = json.loads(data)
                logger.debug("Received JSON control message: %s", msg.get("type"))
            except (json.JSONDecodeError, ValueError):
                pass
        return None
