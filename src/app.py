import os
from fastapi import FastAPI, HTTPException
import tensorflow as tf, joblib, numpy as np, pandas as pd, logging, traceback
from pydantic import BaseModel
from src.monitoring import log_prediction
import requests


# ── top of file ───────────────────────────────────────────
import logging, time
logging.basicConfig(level=logging.INFO)     # ensure INFO logs appear
# ---------------------------------------------------------

app = FastAPI()
# print(os.getcwd())  # For debugging purposes, to check the current working directory
# print(os.listdir('data/processed'))
MODEL_PATH = "models/best_tf_model.keras"
PREP_PATH = "data/processed/prep.joblib"


def download_file(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logging.info("Downloading %s -> %s", url, dest)
    resp = requests.get(url)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)


model_url = os.getenv("MODEL_URL")
prep_url = os.getenv("PREP_URL")

if not os.path.exists(PREP_PATH):
    if not prep_url:
        raise RuntimeError("PREP_URL environment variable not set and prep file missing")
    download_file(prep_url, PREP_PATH)

if not os.path.exists(MODEL_PATH):
    if not model_url:
        raise RuntimeError("MODEL_URL environment variable not set and model file missing")
    download_file(model_url, MODEL_PATH)

pre = joblib.load(PREP_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

class Client(BaseModel):
    LIMIT_BAL: int
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: int
    BILL_AMT2: int
    BILL_AMT3: int
    BILL_AMT4: int
    BILL_AMT5: int
    BILL_AMT6: int
    PAY_AMT1: int
    PAY_AMT2: int
    PAY_AMT3: int
    PAY_AMT4: int
    PAY_AMT5: int
    PAY_AMT6: int

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(client: Client):
    try:
        start = time.time()                     # ⏱️ start timer
        # Build a 1-row DataFrame; columns stay in correct order
        X_df = pd.DataFrame([client.dict()])        # shape (1, 23)

        # Run it through your saved preprocessing pipeline
        X_processed = pre.transform(X_df)           # now 2-D array

        # Predict default probability
        prob = float(model.predict(X_processed)[0][0])
        log_prediction(client.dict(), prob, os.getenv("MODEL_VERSION","1.16"))

        elapsed = time.time() - start
        logging.info(f"scored in {elapsed:.3f}s prob={prob:.4f}")

        return {"default_probability": prob}

    except Exception as e:
        # Print the full stack trace to your uvicorn console for debugging
        logging.error("Prediction failed\n%s", traceback.format_exc())
        # Send a clean 500 back to the client
        raise HTTPException(status_code=500, detail=str(e))

