FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/voice-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/voice-service/ ./src/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_CACHE_DIR=/app/models

RUN mkdir -p /app/models

EXPOSE 8300

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8300", "--log-level", "info"]
