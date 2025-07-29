# User Guide – How to Retrain the Model

1. **Activate the Virtual Environment:**
    ```bash
    source .venv/bin/activate
    ```

2. **Run the Training Script:**
    ```bash
    python train.py
    ```

3. **Check Artifacts:**
    Ensure that the following files are created:
    - `models/best_tf_model.keras`
    - `data/processed/prep.joblib`

4. **Upload Artifacts to Google Cloud Storage:**
    ```bash
    gsutil cp models/best_tf_model.keras gs://bank-mlops-models/
    gsutil cp data/processed/prep.joblib gs://bank-mlops-models/
    ```

5. **Push Code to Repository:**
    ```bash
    git add .
    git commit -m "Updated model"
    git push origin main
    ```

6. **Automatic Deployment:**
    The pipeline will deploy the updated model automatically after the code is pushed.

