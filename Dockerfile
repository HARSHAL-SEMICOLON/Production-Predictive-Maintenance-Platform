# Multi-stage production Dockerfile for Sentinel AI API
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime image
FROM python:3.12-slim

WORKDIR /app

# Create non-root user for security
RUN groupadd -r sentinel && useradd -r -g sentinel sentinel

# Copy installed python packages from builder
COPY --from=builder /root/.local /home/sentinel/.local
ENV PATH=/home/sentinel/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy source code and artifacts
COPY . .

# Set permissions
RUN chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
