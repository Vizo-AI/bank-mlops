# src/monitoring.py  (new file)
import json, logging, uuid, time, os

logger = logging.getLogger("uvicorn.access")  # ends up in Cloud Logging

def log_prediction(features: dict, prob: float, model: str):
    payload = {
        "event_type":       "prediction",
        "request_id":       str(uuid.uuid4()),
        "timestamp":        int(time.time()),
        "model_version":    model,
        "prob_default":     round(prob, 6),
        "features":         features,         # or a hash for privacy
    }
    logger.info(json.dumps(payload))
