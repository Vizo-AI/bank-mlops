import mlflow, mlflow.tensorflow

# Set up MLflow tracking
# Make sure to use the same tracking URI as in preprocess.py
# (i.e., the SQLite database)
# This is necessary to log the model and metrics in the same experiment
# as the preprocessing step.
# If you run this script separately, it will create a new experiment
# with the same name, but it will not overwrite the existing one.
# If you run this script multiple times, it will create new runs
# under the same experiment, allowing you to compare different runs.
# If you want to overwrite the existing experiment, you can delete the
# existing experiment in the MLflow UI or use the `mlflow.delete_experiment` function
# to delete it programmatically.
# Make sure to run this script in the same environment where you ran preprocess.py
# to ensure that the tracking URI points to the same SQLite database.
mlflow.set_tracking_uri("sqlite:///mlflow.db")      # ① point to the same DB
mlflow.set_experiment("credit-default")             # ② auto-creates if needed

import numpy as np, tensorflow as tf, joblib, os
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score

PROC = Path('data/processed')
X_train = np.load(PROC/'X_train.npy')
y_train = np.load(PROC/'y_train.npy')
X_test  = np.load(PROC/'X_test.npy')
y_test  = np.load(PROC/'y_test.npy')

with mlflow.start_run():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20,
                        batch_size=256, validation_split=0.2,
                        verbose=0)

    # log model & metrics
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    auc = roc_auc_score(y_test, y_pred_prob)
    acc = accuracy_score(y_test, (y_pred_prob>0.5).astype(int))
    mlflow.log_metric("test_auc", auc)
    mlflow.log_metric("test_acc", acc)
    mlflow.tensorflow.log_model(model, artifact_path="model")
