# ---------- base image ----------
FROM python:3.11-slim

# ---------- set work directory ----------
WORKDIR /app

# ---------- copy dependency list ----------
COPY requirements.txt .

# ---------- install dependencies ----------
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip config set global.disable-pip-version-check true

# ---------- copy project code ----------
COPY . /app

# ---------- entrypoint helper ----------
COPY scripts/start_mlflow.sh /app/start_mlflow.sh
RUN chmod +x /app/start_mlflow.sh   

# ---------- documentation ----------
# 8080 for FastAPI, 5000 for MLflow UI
EXPOSE 8080 5000        

# ---------- set environment variables ----------
ENV PYTHONPATH=/app

# ---------- run server ----------
# CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "${PORT:-8080}"]
CMD ["/bin/sh", "-c", "exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
