# ─── Stage 1: dependency builder ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System packages needed to compile psycopg2 and other C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies into an isolated prefix
COPY agt_ss_api/requirements.txt requirements.txt

# Strip test-only packages from the production install
RUN grep -v -E "^(pytest|pytest-asyncio)" requirements.txt > prod-requirements.txt && \
    pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r prod-requirements.txt


# ─── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Runtime system packages only (libpq for psycopg2 at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
# agt_ss     — agent runtime (LangGraph orchestrator + sub-agents)
# agt_ss_api — FastAPI service layer
COPY agt_ss/     ./agt_ss/
COPY agt_ss_api/ ./agt_ss_api/

# Ensure both packages are importable
RUN pip install --no-cache-dir -e ./agt_ss/ 2>/dev/null || true

# Temp directory for checkpoint dev fallback (used when DB_HOST is unset)
RUN mkdir -p /tmp/agt_ss_checkpoints && chown appuser:appuser /tmp/agt_ss_checkpoints

# Drop to non-root
USER appuser

# Expose the uvicorn port
EXPOSE 8000

# Health check — mirrors /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint — single worker per container (ECS scales horizontally)
# LangGraph workflow runs are CPU-bound and use internal thread pools;
# multiple uvicorn workers would fight over those pools.
CMD ["python", "-m", "uvicorn", "agt_ss_api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]
