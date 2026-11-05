# Render Deployment Guide: Emergency Department Patient Triage AI

This repository is fully configured and ready for 1-click or manual deployment on **Render** (https://render.com) as a live, production-grade cloud application.

---

## 🛠️ Deployment Configuration Included

| File | Purpose |
|---|---|
| `render.yaml` | Render Blueprint specification (auto-configures service, runtime, commands) |
| `Procfile` | Cloud process runner (`web: python web_app.py`) |
| `.python-version` | Pins runtime to Python `3.10.11` for exact model compatibility |
| `requirements.txt` | Streamlined runtime ML dependencies for ultra-fast builds (< 45s) |
| `web_app.py` | Multi-threaded server (`ThreadingHTTPServer`) listening on `0.0.0.0:$PORT` |
| `models/*.joblib` | Pre-trained models (`co_training`, `random_forest`, `logistic_regression`, etc.) |
| `outputs/` | Research artifacts, performance metrics, and all 12 publication charts |

---

## 🚀 Step 1: Push Code to GitHub

1. Open [GitHub](https://github.com/new) in your browser and create a new repository (e.g. `patient-triage-ai`). Keep it **Public** or **Private**.
2. In your terminal at `e:\company\Python\Patient Triage`, run:

```bash
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/patient-triage-ai.git
git branch -M main
git push -u origin main
```

*(Note: Replace `<YOUR_GITHUB_USERNAME>` with your actual GitHub username, e.g. `vaxanth01`).*

---

## 🌐 Step 2: Deploy on Render

### Option A: 1-Click Blueprint (Recommended)
1. Go to your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** (top right) and select **Blueprint**.
3. Connect your GitHub account and select the `patient-triage-ai` repository.
4. Render automatically reads `render.yaml` and sets:
   - **Service Type:** Web Service
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python web_app.py`
   - **Plan:** Free
5. Click **Apply**. Render will build and deploy the application automatically.

---

### Option B: Standard Web Service
If you prefer setting up manually without Blueprints:
1. Go to [Render Dashboard](https://dashboard.render.com/) -> **New +** -> **Web Service**.
2. Connect your GitHub repository `patient-triage-ai`.
3. Fill in the details:
   - **Name:** `patient-triage-ai`
   - **Region:** Any (e.g., Frankfurt or Oregon)
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python web_app.py`
   - **Instance Type:** `Free`
4. Under **Advanced**, add Environment Variable:
   - Key: `PYTHON_VERSION`, Value: `3.10.11`
5. Click **Create Web Service**.

---

## 🔍 Step 3: Verify the Live Application

Once deployment completes (~1 to 2 minutes), Render assigns a public HTTPS URL:
`https://<your-service-name>.onrender.com`

### Available Live Endpoints:
- **Interactive Web Application:** `https://<your-service-name>.onrender.com/`
  - Real-time clinical patient triage form (SpO2, Blood Pressure, Heart Rate, Symptoms, History).
  - Multi-model selector: Co-Training, Random Forest, Logistic Regression.
  - Interactive prediction badge with class probabilities, severity gauges, and red-flag clinical alerts.
  - Built-in publication analytics viewer with all 12 research charts and metrics table.
- **REST API Inference:** `POST https://<your-service-name>.onrender.com/api/predict`
  - Send patient JSON payload to receive immediate triage classification, class probabilities, and clinical red flags.
- **Metrics API:** `GET https://<your-service-name>.onrender.com/api/metrics`
- **Charts:** `GET https://<your-service-name>.onrender.com/figures/<filename>.png`

