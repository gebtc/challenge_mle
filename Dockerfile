# =============================================================================
# Dockerfile — Flight Delay Prediction API
#
# Multi-stage build:
#   1. `builder`   — Install production Python dependencies.
#   2. `runtime`   — Minimal image with only what is needed at runtime.
#
# Usage:
#   docker build -t flight-delay-api:latest .
#   docker run -p 8000:8000 flight-delay-api:latest
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — Builder
# ---------------------------------------------------------------------------
FROM python:3.9-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered logging.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies (if any) into a temporary layer.
# Using --no-install-recommends keeps the layer small.
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        gcc \
        libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment so that we can copy it wholesale into the
# runtime image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install production dependencies only.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — Runtime
# ---------------------------------------------------------------------------
FROM python:3.9-slim AS runtime

# Create a non-root user and group for security.
#  - UID/GID 1000 is the conventional first non-system user.
#  - The home directory is set to /app because that is where the code lives.
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home \
        --home-dir /app appuser

# Copy the pre-built virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set the working directory.
WORKDIR /app

# Copy application source code and data.
# The data.csv file is required at startup to train the model.
COPY challenge/ ./challenge/
COPY data/ ./data/

# Ensure the non-root user owns everything under /app.
RUN chown -R appuser:appuser /app

# Switch to the non-root user.
USER appuser

# --- Health check ----------------------------------------------------------
# Cloud Run and orchestrators rely on this to determine container readiness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# --- Port ------------------------------------------------------------------
EXPOSE 8000

# --- Command ---------------------------------------------------------------
# Run the FastAPI application via uvicorn.
# The `application` object is exported from challenge/__init__.py.
# Single worker is intentional — Cloud Run scales horizontally.
CMD ["uvicorn", "challenge:application", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
