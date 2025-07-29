# User Guide – Debugging Pipeline Failures

## 1. Check GitHub Actions Logs
- Navigate to your repository on GitHub.
- Go to the **Actions** tab.
- Select the failed workflow run.
- Review the logs for error details.

## 2. Common Failure Points & Solutions
- **Dependency errors:**  
    - Review and update `requirements.txt`.
    - Run `pip install -r requirements.txt` locally to verify.
- **Test failures:**  
    - Run `pytest` locally to reproduce and debug.
- **Docker build failures:**  
    - Build the Docker image locally using `docker build .` and check for errors.
- **GCP deployment failures:**  
    - Verify service account permissions and roles.
    - Check GCP console for deployment error messages.

## 3. Accessing Cloud Run Logs
```bash
gcloud run services logs read $SERVICE --region=$REGION
```
- Replace `$SERVICE` and `$REGION` with your service name and region.

## 4. Escalation Path
- If unresolved after 30 minutes, escalate by contacting the MLOps lead.
- Provide relevant error logs and steps already taken.
