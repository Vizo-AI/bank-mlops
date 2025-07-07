# ---------- base image ----------
FROM python:3.11-slim

# ---------- set work directory ----------
WORKDIR /app

# ---------- copy dependency list ----------
COPY requirements.txt .

# ---------- install dependencies ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- copy project code ----------
COPY . .

# ---------- run server ----------
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
