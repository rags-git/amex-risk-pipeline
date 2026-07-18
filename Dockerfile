# Stage 1: Build & Dependency Resolution Layer
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Minimal Production Runtime Layer
FROM python:3.10-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages dependencies from the builder stage
COPY --from=builder /root/.local /root/.local
COPY config.yaml .
COPY setup.py .
COPY src/ ./src/

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.data_pipeline:app", "--host", "0.0.0.0", "--port", "8000"]
