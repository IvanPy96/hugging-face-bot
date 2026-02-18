FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

# Create non-root user
RUN adduser --disabled-password --gecos "" --uid 1000 appuser

# Install dependencies (pyproject.toml is the single source of truth)
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
RUN pip install --no-cache-dir .

# Create data directory with proper permissions
RUN mkdir -p /app/data && chown appuser:appuser /app/data

USER appuser

STOPSIGNAL SIGTERM

CMD ["python", "-m", "hf_bot"]
