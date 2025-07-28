import importlib
import os
from fastapi.testclient import TestClient
import pytest
import joblib
import tensorflow as tf


class DummyPreprocessor:
    def transform(self, df):
        # return numpy array with same shape
        return df.to_numpy()

class DummyModel:
    def predict(self, X):
        # return constant probability
        return [[0.42]]

sample_input = {
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 3913,
    "BILL_AMT2": 3102,
    "BILL_AMT3": 689,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": 0,
    "PAY_AMT2": 689,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0,
}


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr(joblib, "load", lambda path: DummyPreprocessor())
    monkeypatch.setattr(tf.keras.models, "load_model", lambda path: DummyModel())
    monkeypatch.setattr(os, "listdir", lambda path: [])

    import src.app
    importlib.reload(src.app)
    client = TestClient(src.app.app)

    resp = client.post("/predict", json=sample_input)
    assert resp.status_code == 200
    body = resp.json()
    assert "default_probability" in body
    assert isinstance(body["default_probability"], float)