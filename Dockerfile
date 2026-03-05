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
COPY requirements-prod.txt .
RUN pip install --user --no-cache-dir -r requirements-prod.txt

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

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY --chown=appuser:appuser src/ ./src/

# Copy only inference models (TF-IDF + int8 ONNX) — final_model/ PyTorch weights not needed at runtime
COPY --chown=appuser:appuser src/ml/models/tfidf_vectorizer.joblib ./src/ml/models/
COPY --chown=appuser:appuser src/ml/models/tfidf_classifier.joblib ./src/ml/models/
COPY --chown=appuser:appuser src/ml/models/categorizer_int8.onnx ./src/ml/models/

# Run as non-root user (security best practice)
USER appuser

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
