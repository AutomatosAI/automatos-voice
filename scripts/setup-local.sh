#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

echo "=== Automatos Voice Service — Local Setup ==="

# Check dependencies
for cmd in docker docker-compose; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd is not installed. Please install it first."
        exit 1
    fi
done

# Create .env from example if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "  -> .env created. Edit it if you need to change defaults."
else
    echo "  -> .env already exists, skipping."
fi

# Create model cache directory
mkdir -p models
echo "  -> models/ directory ready (used for caching downloaded models)."

# Build and start
echo ""
echo "Building and starting voice service..."
docker-compose up --build -d

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Voice service is starting on http://localhost:8300"
echo "  Health:  http://localhost:8300/health"
echo "  Metrics: http://localhost:8300/metrics"
echo ""
echo "View logs:  docker-compose logs -f voice-service"
echo "Stop:       docker-compose down"
