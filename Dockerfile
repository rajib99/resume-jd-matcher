# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12
ARG VERSION=0.0.0

# ── builder ───────────────────────────────────────────────────────────────────
# Installs dependencies into an isolated venv.
# Nothing from this stage except /venv lands in the final image.
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build

# uv: significantly faster than pip for dependency resolution + installation
RUN pip install --no-cache-dir uv

COPY requirements.txt .

RUN uv venv /venv && \
    uv pip install --python /venv/bin/python --no-cache -r requirements.txt

# ── runtime ───────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG VERSION
ARG PYTHON_VERSION

LABEL org.opencontainers.image.title="Resume JD Matcher" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.description="AI-powered resume / job-description matching" \
      org.opencontainers.image.base.name="python:${PYTHON_VERSION}-slim"

# curl for health check probe; clean up apt lists to keep image small
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with explicit UID/GID for reproducible file ownership
RUN groupadd --system --gid 1001 appuser && \
    useradd  --system --uid 1001 --gid 1001 \
             --no-create-home --shell /sbin/nologin appuser

WORKDIR /app

# Bring only the pre-built venv — no build tools, no uv, no pip in final image
COPY --from=builder /venv /venv

# Application source only; everything else is excluded via .dockerignore
COPY --chown=appuser:appuser app/ ./app/

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/venv/bin:$PATH" \
    APP_VERSION="${VERSION}"

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/health || exit 1

# Exec form with full venv path — no shell interpolation, no PATH lookup
CMD ["/venv/bin/uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
