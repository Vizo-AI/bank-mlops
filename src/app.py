import os
from fastapi import FastAPI, HTTPException
import tensorflow as tf, joblib, numpy as np, pandas as pd, logging, traceback
from pydantic import BaseModel
from src.monitoring import log_prediction


# ── top of file ───────────────────────────────────────────
import logging, time
logging.basicConfig(level=logging.INFO)     # ensure INFO logs appear
# ---------------------------------------------------------

app = FastAPI()
print(os.getcwd())  # For debugging purposes, to check the current working directory
print(os.listdir('data/processed'))
pre = joblib.load('data/processed/prep.joblib')
model = tf.keras.models.load_model('models/best_tf_model.keras')

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

