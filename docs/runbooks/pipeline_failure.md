## docs/runbooks/pipeline_failure.md
```markdown
# Runbook – Pipeline Failure

## Common Issues & Fixes

### 1. Build Failure (Docker)
**Symptoms:** GitHub Actions shows `docker build` error.
**Fix:**
- Check `Dockerfile` path and Python version.
- Ensure all required dependencies exist in `requirements.txt`.
- Test locally: `docker build -t test-api .`

### 2. Test Failures
**Symptoms:** `pytest` fails.
**Fix:**
- Run tests locally: `pytest tests_api.py test_smoke.py`.
- Check for missing artifacts (model or preprocessor).

### 3. Cloud Deploy Failure
**Symptoms:** `gcloud run deploy` step fails.
**Fix:**
- Check if `GCP_SA_KEY`, `PROJECT_ID`, `REGION`, `SERVICE`, `REPO` secrets are set.
- Verify IAM roles: Service account must have `roles/run.admin` and `roles/artifactregistry.admin`.

### 4. Model Download Errors
**Symptoms:** API logs show `403 Forbidden` when loading model from GCS.
**Fix:**
- Make bucket object public: `gsutil iam ch allUsers:objectViewer gs://bank-mlops-models`.
- Verify correct URLs in environment variables.

### Escalation
- If issue persists > 30 minutes → escalate to MLOps lead.