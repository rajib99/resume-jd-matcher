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

# Probes the /health endpoint; python -c raises an exception (exit 1) on
# network error or non-200 response, which marks the container unhealthy.
HEALTHCHECK --interval=10s --timeout=10s --start-period=15s --retries=3 \
  CMD /venv/bin/python -c \
      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# Exec form with full venv path — no shell interpolation, no PATH lookup
CMD ["/venv/bin/uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
