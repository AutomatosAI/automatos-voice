# Automatos Voice Service

Voice service for the Automatos AI Platform. Provides STT (speech-to-text) and TTS (text-to-speech) as an OpenAI-compatible REST API.

## Architecture

Custom FastAPI service using:
- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2-based Whisper inference)
- **TTS**: [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) (fast neural TTS)
- **API contract**: OpenAI Audio API compatible (`/v1/audio/transcriptions`, `/v1/audio/speech`)
- **Monitoring**: Prometheus metrics + structured JSON logging

Deployed as an independent Railway service. No knowledge of orchestrator, agents, or workspaces — pure voice I/O.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/audio/transcriptions` | Speech-to-text. Multipart form with `file` field, optional `language`. Returns `{"text": "..."}` |
| POST | `/v1/audio/speech` | Text-to-speech. JSON body `{"input": "text", "voice": "af_heart", "model": "kokoro", "speed": 1.0}`. Returns `audio/mpeg` bytes |
| GET | `/health` | Health check with model load status |
| GET | `/metrics` | Prometheus metrics endpoint |

## Local Development

```bash
# 1. Copy environment config
cp .env.example .env

# 2. Run with Docker Compose
docker-compose up --build

# Or use the setup script
./scripts/setup-local.sh
```

The service starts on `http://localhost:8300`.

### Quick test

```bash
# STT
curl -X POST http://localhost:8300/v1/audio/transcriptions \
  -F file=@sample.wav

# TTS
curl -X POST http://localhost:8300/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "af_heart"}' \
  --output output.mp3
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Runtime environment |
| `LOG_LEVEL` | `info` | Logging level |
| `WHISPER_MODEL` | `Systran/faster-whisper-large-v3` | Whisper model to load |
| `WHISPER_DEVICE` | `auto` | Device for inference (`cpu`, `cuda`, `auto`) |
| `WHISPER_COMPUTE_TYPE` | `float16` | Compute precision |
| `TTS_DEFAULT_MODEL` | `kokoro` | Default TTS model |
| `TTS_DEFAULT_VOICE` | `af_heart` | Default TTS voice |
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8300` | Bind port |
| `METRICS_ENABLED` | `true` | Enable Prometheus metrics |

## Deployment

Deployed on Railway using `railway/voice-service.toml`. The Dockerfile handles model downloads on first boot with a persistent volume for caching.
