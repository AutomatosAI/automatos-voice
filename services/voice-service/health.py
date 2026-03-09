"""Health check router.

Reports service health and model load status for STT and TTS engines.
Used by Railway healthcheck and load balancers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter()

# Model status is updated by the model manager on load/error.
# Possible values: "not_loaded", "loading", "loaded", "error"
_model_status: dict[str, str] = {
    "stt": "not_loaded",
    "tts": "not_loaded",
}


def set_model_status(model_type: str, status: str) -> None:
    """Update the status of a model type (stt or tts)."""
    if model_type not in _model_status:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'stt' or 'tts'.")
    valid_statuses = ("not_loaded", "loading", "loaded", "error")
    if status not in valid_statuses:
        raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}.")
    _model_status[model_type] = status


def get_model_status(model_type: str) -> str:
    """Get the current status of a model type."""
    return _model_status.get(model_type, "not_loaded")


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint.

    Returns 200 even if models are still loading — Railway needs the
    HTTP server to be up. The 'models' field tells callers whether
    inference is actually ready.
    """
    all_loaded = all(s == "loaded" for s in _model_status.values())
    any_error = any(s == "error" for s in _model_status.values())

    if any_error:
        status = "degraded"
    elif all_loaded:
        status = "healthy"
    else:
        status = "starting"

    return {
        "status": status,
        "models": {
            "stt": _model_status["stt"],
            "tts": _model_status["tts"],
        },
    }
