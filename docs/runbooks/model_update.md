# Runbook – Model Update

## Steps to Update Model

### 1. Retrain Locally
```bash
python train.py


2. Upload Artifacts
gsutil cp models/best_tf_model.keras gs://bank-mlops-models/
gsutil cp data/processed/prep.joblib gs://bank-mlops-models/


3. Trigger Deployment
Commit code changes (if any) & push to main.
GitHub Actions triggers build & deploy automatically.

4. Verify Deployment
curl https://<CLOUD_RUN_URL>/predict -d '{"sample": "payload"}'

5. Monitor Logs
gcloud run services logs read $SERVICE --region=$REGION