# ── Stage 1: build deps ───────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed by some Python packages (e.g. faiss-cpu, torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-prod.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source (excludes everything in .dockerignore)
COPY api/       api/
COPY rag/       rag/
COPY services/  services/
COPY config/    config/
COPY utils/     utils/
COPY main.py    .

# Non-root user for least-privilege execution
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

# 4 Uvicorn workers via Gunicorn.
# Override WORKERS at runtime: docker run -e WORKERS=2 ...
ENV WORKERS=4

CMD ["sh", "-c", \
     "gunicorn api.app:create_app \
      --factory \
      --worker-class uvicorn.workers.UvicornWorker \
      --workers ${WORKERS} \
      --bind 0.0.0.0:8000 \
      --timeout 120 \
      --access-logfile - \
      --error-logfile -"]
