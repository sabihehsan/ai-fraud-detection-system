# AI Fraud Detection System
## Overview
A real-time fraud detection dashboard built with Python, using unsupervised machine learning algorithms — **Isolation Forest** and **K-Means Clustering** — to detect anomalous bank transactions without requiring labeled fraud data.

## Features
- **Isolation Forest** — detects anomalies by isolating outliers in feature space
- **K-Means Clustering** — groups transactions into behavioral cohorts; outlier distances flag fraud
- **Ensemble Risk Score** — combines both models (65% IF + 35% KMeans)
- **5-Tab Dashboard**: Overview, Alert Queue, Model Analysis, Risk Trends, Transaction Lookup
- **Real-Time Classification** — enter any transaction and get instant AI risk scoring
- **Risk Matrix**: HIGH / MEDIUM / LOW with automated action recommendations

## Tech Stack
- Python 3.10+
- Streamlit (dashboard UI)
- Scikit-Learn (ML models)
- Pandas & NumPy (data processing)
- Plotly (interactive charts)

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open in browser
The app will open automatically at `http://localhost:8501`

---

## How It Works

### Pipeline
1. **Data Ingestion** — Synthetic transaction records (ID, amount, category, location, timestamp, frequency)
2. **Preprocessing** — Normalization via `StandardScaler`, label encoding for categoricals
3. **Model Training** — Isolation Forest (200 estimators) + K-Means (6 clusters) trained unsupervised
4. **Risk Scoring** — Each transaction gets a 0–1 ensemble risk score
5. **Classification** — HIGH (≥0.85), MEDIUM (0.50–0.84), LOW (<0.50)
6. **Dashboard** — Real-time visualizations and alert queue

### Risk Levels & Actions
| Level | Score | Action |
|-------|-------|--------|
| 🔴 HIGH | 0.85–1.00 | Card freeze + SMS alert |
| 🟡 MEDIUM | 0.50–0.84 | Analyst review + MFA check |
| 🟢 LOW | 0.00–0.49 | Auto-cleared, no interference |
