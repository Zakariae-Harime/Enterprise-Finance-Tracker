 # Dockerfile

  # ==========================================
  # Stage 1: BUILDER - Install dependencies
  # ==========================================
  FROM python:3.13-slim AS builder

  WORKDIR /app

  # Install build tools 
  RUN apt-get update && apt-get install -y \
      build-essential \
      libpq-dev \
      && rm -rf /var/lib/apt/lists/*

  # Install Python packages to user directory
  COPY requirements.txt .
  RUN pip install --user --no-cache-dir -r requirements.txt

  # ==========================================
  # Stage 2: PRODUCTION - Minimal final image
  # ==========================================
  FROM python:3.13-slim AS production

  WORKDIR /app

  # Only runtime dependencies 
    RUN apt-get update && apt-get install -y \
      libpq5 \
      curl \
      && rm -rf /var/lib/apt/lists/* \
      && useradd --create-home appuser

  # CP installed packages from builder
  COPY --from=builder /root/.local /home/appuser/.local
  ENV PATH=/home/appuser/.local/bin:$PATH

  # CP application code
  COPY --chown=appuser:appuser src/ ./src/

  # Run as non-root user (security best practice)
  USER appuser

  # Health check for container orchestration
  HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
      CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

  CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]