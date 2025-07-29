# Bank MLOps Project

# Bank Customer Default-Risk API 🚀

Predict whether a credit-card customer will default next month.  
This project demonstrates a **full MLOps pipeline** on **Google Cloud Run** with CI/CD.

---

## 1 • Problem Statement  
Banks lose millions to defaulted credit-card payments.  
Goal → serve an API returning **probability of default** for a given customer.

---

## 2 • Tech Stack 💼

| Phase               | Tooling                               |
|---------------------|---------------------------------------|
| Model Training      | Python 3.10, TensorFlow-CPU, scikit-learn, joblib |
| Experiment Tracking | MLflow (lightweight / skinny build)   |
| Version Control     | Git + GitHub Actions                  |
| Containerization    | Docker (multi-stage, <1 GB image)     |
| Deployment          | Cloud Run (fully managed, serverless) |
| Monitoring          | Cloud Logging & custom latency logs   |

---

## 3 • Architecture 🖼️

```text
┌─────────┐    git push     ┌────────────────────┐
│Developer│ ──────────────► │ GitHub Actions CI  │
└─────────┘                 │  • pytest          │
                            │  • model hash check│
                            │  • docker build    │
                            └─────────┬──────────┘
                                      │ image
                    gcloud builds     ▼
                            ┌────────────────────┐
                            │  Artifact Registry │
                            └─────────┬──────────┘
                                      │ deploy
                                      ▼
                            ┌────────────────────┐
                            │ Cloud Run service  │
                            │ FastAPI + TF model │
                            └─────────┬──────────┘
                                      ▼
                          Cloud Logging / Monitoring
```

## 4 • Setup 🛠️
Works on macOS & Linux.
Windows: For best results, run commands in Git Bash or WSL.  
Note: Activating the virtual environment differs by shell:  
- **Git Bash/WSL:** `source .venv/bin/activate`  
- **CMD:** `.venv\Scripts\activate.bat`  
- **PowerShell:** `.venv\Scripts\Activate.ps1`

# ① Clone code
git clone https://github.com/<your-user>/bank-mlops.git
source .venv/bin/activate   # Windows: .venv\Scripts\activate (CMD) or .venv\Scripts\Activate.ps1 (PowerShell)

# ② Create & activate Python 3.10 env
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# ③ Install deps
pip install --upgrade pip
pip install -r requirements.txt

# ④ Train & track an experiment
python src/train.py              # logs run in ./mlruns

# ⑤  Docker build & run
docker build -t bank-default-api:local .
docker run -p 8080:8080 bank-default-api:local
# The Docker container exposes the API on port 8080.
# Access the docs at http://127.0.0.1:8080/docs

# ⑥ Smoke-test API locally (for development, not in Docker)
uvicorn src.app:app --reload
## 4.1 • Models & Preprocessor

Models and preprocessor are **not bundled** in the Docker image; they are loaded dynamically at runtime:

- `PREP_URL` → GCS link to `prep.joblib`
- `MODEL_URL` → GCS link to `best_tf_model.keras`

> **Warning:**  
> The Docker container will **fail to start** if the `MODEL_URL` and `PREP_URL` environment variables are not set, or if the referenced files are not accessible (public or via service account).

**Make these files public** (or provide a service account with access) using:
- `MODEL_URL` → GCS link to `best_tf_model.keras`

> ⚠️ **Security Warning:**  
> Making files public allows anyone on the internet to download your model and preprocessor artefacts.  
> **For sensitive or production models, prefer granting access via a service account** instead of making files public.

**To make files public (not recommended for sensitive data),** use:

```bash
gsutil iam ch allUsers:objectViewer gs://bank-mlops-models
```

## 5 • CI/CD – One-Click Build & Deploy 🚦

On every push to `main`, the pipeline:

1. **Runs tests:**  
   - `pytest tests_api.py test_smoke.py`
2. **Checks model hash:**  
   - Only uploads a new model if it has changed.
3. **Builds & pushes Docker image:**  
   - Uses Cloud Build to build and push to Artifact Registry (e.g. `REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$SERVICE:$GITHUB_SHA`)
4. **Deploys to Cloud Run:**  
   - Deploys the new image to the `$SERVICE` on Cloud Run.

This ensures every commit is tested, versioned, and automatically deployed if successful.

### Required GitHub Secrets

- `GCP_SA_KEY` – JSON service account key with Cloud Run & Artifact Registry permissions.
- `PROJECT_ID` – GCP project ID
- `REGION` – GCP region for Cloud Run (e.g. `us-central1`).
- `SERVICE` – Cloud Run service name
- `REPO` – Artifact Registry repo name

### Deploy Command Example

```bash
gcloud run deploy $SERVICE \
  --image=$IMAGE \
  --region=$REGION \
  --update-env-vars MODEL_URL=$MODEL_URL,PREP_URL=$PREP_URL
```

5. **View Logs**

```bash
gcloud run services logs read $SERVICE
```


Example log extract from a successful run:

```text
Deploying container to Cloud Run service [bank-api] in project [mlops-demo] region [us-central1]
✔ Deploying... done
└─ https://bank-api-xyz-uc.a.run.app
```



## 6 • Live Demo 🌐
A **fully deployed endpoint** is live on Google Cloud Run.

- **Swagger UI:**  
  [https://bank-api-<hash>.a.run.app/docs](https://bank-api-<hash>.a.run.app/docs)  
- **Try it:**  
  Hit `/predict` using the sample payload provided in Swagger.

## 7 • Monitoring

- All container stdout/stderr logs stream automatically to **Cloud Logging**.
- Each API prediction logs latency and output:
  ```python
  logging.info("scored in %.3fs prob=%.4f", time.time()-start, prob)

- Log-based metric prediction_latency_seconds feeds a dashboard & alert.

## 8 • Future Work 🛣️

1. Feature Store – Vertex AI Feature Store for online features
2. Data-drift alerts – BigQuery scheduled query + Cloud Scheduler
3. Canary releases – traffic-split revisions in Cloud Run
4. Prometheus / Grafana – scrape custom /metrics endpoint on GKE
5. CI security – Trivy scan in GitHub Actions

## 9 • License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.


---

# 🗒️ Part 2 – Step-by-Step Notebook (Interview Cheat-Sheet)

> **How to read:** Each step shows the exact command, common issues, and the solution/fix in one column.

---

### 0 • Environment Setup

| Step | Command(s) | Issue / Solution |
|------|------------|------------------|
| 0-1  | `python3.10 -m venv .venv`<br>`source .venv/bin/activate` | Always create and activate the virtual environment before installing packages. |
| 0-2  | `pip install --upgrade pip`<br>`pip install -r requirements.txt` | Installs all dependencies inside the venv; prevents system package conflicts. |
| 0-3  | (Optional) `pip install xlrd` | Only if Excel import is needed. |

---

### 1 • Git & GitHub

| 1-2  | `.gitignore` adjustments | For local runs, ensure `models/` and `prep.joblib` are **not** ignored in `.gitignore` (remove them from `.gitignore` if present); keep large or sensitive files excluded as appropriate. |
|------|------------|------------------|
| 1-1  | `git clone https://github.com/<your-user>/bank-mlops.git`<br>`cd bank-mlops` | Clone the repo and enter project directory. |
| 1-2  | `.gitignore` adjustments | Ensure models and preprocessor artefacts are not ignored if needed for local runs. |
| 1-3  | `git pull --rebase origin main`<br>`git push` | Use rebase to resolve non-fast-forward errors before pushing. |

---

### 2 • Model Training & MLflow

| Step | Command(s) | Issue / Solution |
|------|------------|------------------|
| 2-1  | `python src/train.py` | Trains model and logs to MLflow. If experiment not found, add `mlflow.set_experiment("bank-default")` in script. |
| 2-2  | (macOS only) `brew install xz`<br>`pip install backports.lzma` | Fixes `_lzma` module error if encountered. |
| 2-3  | Model saving: `model.save("models/best_tf_model.keras")` | Use `.keras` format for Keras 3 compatibility ([see migration guide](https://keras.io/guides/migrating_to_keras_3/)). |

---

### 3 • FastAPI App

| Step | Command(s) | Issue / Solution |
|------|------------|------------------|
| 3-1  | `uvicorn src.app:app --reload` | Runs API locally. Ensure `models/best_tf_model.keras` and `prep.joblib` exist in the project root, or set `MODEL_URL` and `PREP_URL` environment variables to valid GCS URLs (e.g., `gs://bucket/path/to/model.keras`). |
| 3-2  | `/predict` request | If input shape error, wrap input as 2D array: `np.array([...]).reshape(1, -1)`. |

---

### 4 • Docker Build & Local Test

| Step | Command(s) | Issue / Solution |
|------|------------|------------------|
| 4-1  | `docker build -t bank-default-api:local .` | Builds Docker image locally. |
| 4-2  | `docker run -p 8080:8080 bank-default-api:local` | Runs container; test API at `http://127.0.0.1:8080/docs`. |
| 4-3  | `.dockerignore` review | Ensure required artefacts (if any) are not excluded from build context; avoid accidentally excluding `requirements.txt` or source code (e.g., `src/`). |

---

### 5 • CI/CD Pipeline

| Step | Command(s) / Action | Issue / Solution |
|------|---------------------|------------------|
| 5-1  | Push to `main` branch | Triggers GitHub Actions: runs tests, checks model hash, builds & pushes Docker image, deploys to Cloud Run. |
| 5-2  | Set GitHub secrets: `GCP_SA_KEY`, `PROJECT_ID`, `REGION`, `SERVICE`, `REPO` | Required for deployment. |
| 5-3  | Deploy: see example below | Use `gcloud run deploy $SERVICE --image=$IMAGE --region=$REGION --update-env-vars MODEL_URL=$MODEL_URL,PREP_URL=$PREP_URL` |
| 5-4  | Logs: `gcloud run services logs read $SERVICE` | Check deployment and API logs for troubleshooting. |
| 5-5  | Monitoring | Cloud Logging auto-streams logs; latency metrics logged in API code. |

---

**Tip:**  
Models and preprocessor are loaded at runtime from GCS using `MODEL_URL` and `PREP_URL` env vars. Make sure these files are public or accessible by the service account.


---

## **Part 3 – Documentation & Knowledge Sharing**

This project includes additional documentation to support operations, onboarding, and collaboration. All referenced docs are in the `docs/` directory.

### **A. Runbooks (`docs/runbooks/`)**
- **rollback.md**: Steps to revert Cloud Run to a previous revision.
- **pipeline_failure.md**: Troubleshooting CI/CD and deployment issues.
- **model_update.md**: How to retrain, upload, and redeploy models.

### **B. Architecture Diagram (`docs/architecture/`)**
- Visual diagrams (PNG and editable Draw.io) illustrating the end-to-end MLOps pipeline.

### **C. User Guides (`docs/user_guides/`)**
- **retraining.md**: Step-by-step model retraining and upload.
- **debugging_pipeline.md**: CI/CD pipeline debugging and log access.
- **feature_requests.md**: Submitting feature requests and workflow.

### **D. Internal Demo Script (`docs/demo_script.md`)**
- Live walkthrough for onboarding and stakeholder demos, including API usage, log inspection, and CI/CD review.

---

> **Note:** For detailed procedures, troubleshooting, and onboarding, refer to the relevant files in the `docs/` folder.
