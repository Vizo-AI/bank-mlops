# ----------- Stage 1: Build layer with dependencies ----------- 
FROM python:3.11-slim AS build

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements (improves caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ----------- Stage 2: Runtime layer (lighter) ----------- 
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy installed Python packages from build stage
COPY --from=build /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=build /usr/local/bin /usr/local/bin

# Copy your application code (no models baked into image)
COPY src/ ./src/
COPY scripts/start_mlflow.sh ./scripts/start_mlflow.sh

# Make script executable
RUN chmod +x ./scripts/start_mlflow.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080

# Expose ports (FastAPI: 8080, MLflow optional: 5000)
EXPOSE 8080 5000

# Use Cloud Run provided PORT dynamically
CMD ["/bin/sh", "-c", "exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
