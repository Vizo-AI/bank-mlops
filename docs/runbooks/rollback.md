# Runbook – Cloud Run Rollback

## Purpose

To quickly revert a Cloud Run service to a previous stable revision if issues occur after a deployment.

## Steps

1. **List all revisions:**
    ```bash
    gcloud run revisions list --service=$SERVICE --region=$REGION
    ```

2. **Identify the previous revision:**  
    Note the revision name that was serving traffic before the latest deployment.

3. **Redirect traffic to the previous revision:**
    ```bash
    gcloud run services update-traffic $SERVICE \
      --to-revisions <REVISION_NAME>=100 --region=$REGION
    ```

4. **Verify traffic assignment:**
    ```bash
    gcloud run services describe $SERVICE --region=$REGION | grep traffic
    ```

5. **Communicate:**  
    Notify stakeholders (e.g., via Slack or email) that the rollback is complete.