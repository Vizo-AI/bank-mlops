# Bank MLOps Project

# Bank Customer Default-Risk API 🚀  

Predict whether a credit-card customer will default next month.  
The project demonstrates a **full MLOps pipeline** on Google Cloud Run.

---

## 1 • Problem Statement  
Banks lose millions to defaulted credit-card payments.  
Goal → ship an API that returns the **probability of default** for a given
customer so analysts can act proactively.

---

## 2 • Tech Stack 💼  

| Phase | Tooling | Why it matches job specs |
|-------|---------|--------------------------|
| Model | **Python 3.11**, TensorFlow /Keras, scikit-learn | Nearly every FinTech posting asks for Py / TF |
| Experiment Tracking | **MLflow 2** | “Must understand experiment management” |
| Version Control | **Git + GitHub Actions** | CI pipeline builds & tests container |
| Packaging | **Docker (amd64 + arm64)** | “Containerise ML services” |
| Serving | **FastAPI + Uvicorn** on **Cloud Run** | “Deploy scalable REST services on GCP” |
| Monitoring | Cloud Logging & Monitoring + custom latency log lines | “Operational monitoring experience” |

---

## 3 • Architecture 🖼️  

```text
┌─────────┐    git push     ┌────────────────────┐
│Developer│ ──────────────► │ GitHub Actions CI  │
└─────────┘                 │  • pytest          │
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
Windows: run the commands in Git Bash or WSL.

# ① Clone code
git clone https://github.com/<your-user>/bank-mlops.git
cd bank-mlops

# ② Create & activate Python 3.11 env
python3.11 -m venv bankenv
source bankenv/bin/activate      # Windows: bankenv\Scripts\activate

# ③ Install deps
pip install --upgrade pip
pip install -r requirements.txt

# ④ Train & track an experiment
python src/train.py              # logs run in ./mlruns

# ⑤ Smoke-test API locally
uvicorn src.app:app --reload
# open http://127.0.0.1:8000/docs

# ⑥ Docker build & run
docker build -t bank-default-api:local .
docker run -p 8080:8080 bank-default-api:local


## 5 • CI/CD – one-click build
Every push to main triggers GitHub Actions:

1. Run pytest
2. Build Docker image
3. Publish to GCP Artifact Registry (via Cloud Build)

See .github/workflows/ci.yml for the job definition.


## 6 • Live Demo 🌐
Swagger UI → https://bank-api-xyz-uc.a.run.app/docs
(hit /predict with the sample payload in the doc)

## 7 • Monitoring

Cloud Run streams container stdout/stderr to Cloud Logging.
Each prediction logs latency:

python
>>> logging.info("scored in %.3fs prob=%.4f", time.time()-start, prob)

Log-based metric prediction_latency_seconds feeds a dashboard &
alert.

## 8 • Future Work 🛣️

1. Feature Store – Vertex AI Feature Store for online features
2. Data-drift alerts – BigQuery scheduled query + Cloud Scheduler
3. Canary releases – traffic-split revisions in Cloud Run
4. Prometheus / Grafana – scrape custom /metrics endpoint on GKE
5. CI security – Trivy scan in GitHub Actions

## 9 • License


---

# 🗒️ Part 2 – Your *personal* step-by-step notebook (interview cheat-sheet)  

> **How to read:** numbered like a diary; each step shows *your exact
> command*, the **problem you hit**, and **the fix**.

---

### 0 • Environment bootstrap  

| When | Command(s) | Why / fix |
|------|------------|-----------|
| 00-01 | `brew install python@3.11` | Install Python 3.11 on mac (Homebrew) |
|      | `python3.11 -m venv bankenv`<br>`source bankenv/bin/activate` | Create virtual-env |
| 00-02 | **Mistake**: installed packages *outside* venv → huge global install | **Fix**: activated env, then `pip install --upgrade pip` & all deps |
| 00-03 | Missing Excel reader | `pip install xlrd` *(fix “xlrd optional dependency”)* |

---

### 1 • Git & GitHub  

| When | Command | Issue | Fix |
|------|---------|-------|-----|
| 01-01 | `git init` → `git remote add origin …` | `git push -u origin main` → *“src refspec main does not match”* | `git branch -M main` then push |
| 01-02 | `.gitignore` blocked `data/` and `models/`; `git add` gave “ignored by .gitignore” | Added *whitelist* lines to re-add processed artefacts |
| 01-03 | Push rejected (*non-fast-forward*) | `git pull --rebase origin main` then push |

---

### 2 • Model training & MLflow  

| Cmd | Issue | Fix |
|-----|-------|-----|
| `python src/train.py` | `mlflow.exceptions.MlflowException: Could not find experiment with ID 0` | Added `mlflow.set_experiment("bank-default")` at top of train script |
|  | `ModuleNotFoundError: _lzma` (macOS system Python) | `brew install xz && pip install backports.lzma` |
|  | Keras 3 error: *load_model only supports .keras / .h5* | Saved model as `.keras`: `model.save("models/best_tf_model.keras")` |

---

### 3 • FastAPI app  

| Cmd | Issue | Fix |
|-----|-------|-----|
| `uvicorn src.app:app` | `FileNotFoundError prep.joblib` | Added correct relative path, committed artefact |
| `/predict` request → 500 | `ValueError: Expected 2D array` | Wrapped input as list of dict values: `np.array([…]).reshape(1,-1)` |

---

### 4 • Docker build (local)  

| Cmd | Issue | Fix |
|-----|-------|-----|
| `docker build -t bank-default-api:0.1 .` | SUCCESS |
| `docker run -p 8080:8080 …` | FastAPI worked |
| Move to Cloud Build | **Artefacts missing** because `.dockerignore` stripped them |

---

### 5 • Ignore-file saga (24 h blocker)  

1. **Initial state** – both `.gcloudignore` and `.dockerignore` had  
   `data/**` and `models/**` but **no allow-rules** → artefacts missing.
2. Added allow-rules but directory line was missing (`!data/processed/`) →
   Docker still pruned file.
3. Final working tail (must be *last lines* in both files):

   ```text
   !models/
   !models/best_tf_model.keras
   !data/processed/
   !data/processed/prep.joblib
```
