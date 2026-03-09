"""Orchestrator Bridge — sends transcripts to Automatos chat pipeline (PRD-74 Phase 3)

This processor sits between STT and TTS in the Pipecat pipeline.
It receives TranscriptionFrames, posts them to the orchestrator's
streaming chat endpoint, and emits LLM text frames for TTS.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

import aiohttp

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    TextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    ErrorFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


class OrchestratorProcessor(FrameProcessor):
    """Bridge between Pipecat pipeline and the Automatos orchestrator.

    On receiving a TranscriptionFrame:
    1. POST transcript to orchestrator streaming chat endpoint
    2. Parse AI SDK streaming format (0:"text chunks")
    3. Emit TextFrames for TTS to synthesize

    Maintains conversation_id for multi-turn context.
    """

    def __init__(
        self,
        *,
        orchestrator_url: str,
        workspace_id: str,
        agent_id: Optional[int] = None,
        auth_token: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._orchestrator_url = orchestrator_url.rstrip("/")
        self._workspace_id = workspace_id
        self._agent_id = agent_id
        self._auth_token = auth_token
        self._conversation_id = str(uuid.uuid4())

    def set_auth_token(self, token: str) -> None:
        self._auth_token = token

    def set_agent_id(self, agent_id: int) -> None:
        self._agent_id = agent_id

    def set_conversation_id(self, conversation_id: str) -> None:
        self._conversation_id = conversation_id

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                return

            logger.info(
                "orchestrator_request",
                extra={
                    "transcript": text[:100],
                    "conversation_id": self._conversation_id,
                    "agent_id": self._agent_id,
                },
            )

            await self.push_frame(LLMFullResponseStartFrame())

            try:
                response_text = await self._stream_chat(text)
                if response_text:
                    # Emit as a single TextFrame for TTS
                    await self.push_frame(TextFrame(text=response_text))
                else:
                    await self.push_frame(
                        TextFrame(text="I didn't catch that. Could you try again?")
                    )
            except Exception as e:
                logger.error(
                    "orchestrator_error",
                    extra={"error": str(e)},
                    exc_info=True,
                )
                await self.push_frame(
                    TextFrame(text="Sorry, I had trouble processing that.")
                )

            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)

    async def _stream_chat(self, transcript: str) -> str:
        """Post transcript to orchestrator streaming endpoint, collect response."""
        headers = {
            "Content-Type": "application/json",
            "X-Workspace-ID": self._workspace_id,
        }
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        payload = {
            "id": self._conversation_id,
            "messages": [
                {
                    "role": "user",
                    "content": transcript,
                    "parts": [{"type": "text", "text": transcript}],
                }
            ],
        }
        if self._agent_id:
            payload["agent_id"] = self._agent_id

        url = f"{self._orchestrator_url}/api/chat"
        collected = []

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        "orchestrator_http_error",
                        extra={"status": resp.status, "body": body[:300]},
                    )
                    return ""

                # Parse AI SDK streaming format
                async for line in resp.content:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if not decoded:
                        continue

                    # AI SDK format: 0:"text chunk"
                    if decoded.startswith("0:"):
                        try:
                            text_part = json.loads(decoded[2:])
                            if isinstance(text_part, str):
                                collected.append(text_part)
                        except (json.JSONDecodeError, ValueError):
                            pass

        response = "".join(collected)
        logger.info(
            "orchestrator_response",
            extra={
                "response_length": len(response),
                "conversation_id": self._conversation_id,
            },
        )
        return response
