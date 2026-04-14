# 🛡️ AI Fraud Detection System

> **Production-grade, real-time credit card fraud detection.**
> Ensemble ML models · FastAPI backend · Streamlit dashboard · SHAP explainability · Hybrid risk scoring.

---

##  Why This Project Matters in Fintech

Credit card fraud costs the global economy **over $33 billion annually** (Nilson Report, 2023).
Traditional rule-based systems flag too many legitimate transactions and miss sophisticated
attacks that exploit gaps between rules.

This system combines **statistical learning** (XGBoost, Random Forest) with **domain-specific
business rules** into a hybrid scoring engine — the same pattern used by companies like Stripe,
Adyen, and PayPal — to achieve high recall (catching real fraud) while keeping false-positive
rates operationally viable.

---

## ✨ Key Features

| Feature | Description |
|---|---|
|  **3-model ensemble** | Logistic Regression (baseline) · Random Forest · XGBoost (primary) |
|  **SMOTE balancing** | Oversamples minority class to prevent model bias on imbalanced data |
|  **Hybrid risk scoring** | `risk = 0.70 × ML_prob + 0.30 × rule_score` for robust decisions |
|  **SHAP explainability** | Per-transaction feature impact — full auditability for ops teams |
|  **Alert system** | Console + file logging; optional SMTP email via `.env` |
|  **Real-time streaming** | Async generator + mock Kafka queue at configurable TPS |
|  **Rich dashboard** | 5-tab Streamlit UI with live feed, gauges, SHAP waterfall, manual test |
|  **Docker-ready** | Multi-stage build + docker-compose for one-command deployment |
|  **Test suite** | pytest covering API routes, business logic, preprocessing, streaming |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
│  data/creditcard.csv  ──►  data_preprocessing.py  ──►  SMOTE       │
│                       ──►  feature_engineering.py (derived feats)  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                        MODEL LAYER                                  │
│  src/train.py                                                       │
│    ├── LogisticRegression   (baseline · balanced class weights)     │
│    ├── RandomForest         (300 trees · recall-optimised)          │
│    └── XGBoost  ★ primary  (PR-AUC objective · scale_pos_weight)   │
│  src/evaluate.py  → confusion matrix · ROC · PR curve · SHAP       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │  models/*.pkl
┌────────────────────────────────▼────────────────────────────────────┐
│                        API LAYER  (FastAPI)                         │
│  POST /predict                                                      │
│    ├── api/predictor.py   → load model, SHAP TreeExplainer          │
│    ├── api/business_logic.py → rule engine (amount/hour/PCA/micro)  │
│    └── risk_score = 0.70×ML + 0.30×rules                           │
│  GET  /health · GET /models                                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │  HTTP JSON
┌────────────────────────────────▼────────────────────────────────────┐
│                    DASHBOARD LAYER  (Streamlit)                     │
│  streaming/transaction_stream.py  → async gen / mock Kafka queue    │
│  dashboard/components/simulator.py → API calls + offline fallback   │
│  dashboard/app.py                                                   │
│    ├──  Live Feed      → timeline + scrolling transaction table   │
│    ├──  Analytics      → probability dist · summary · scatter     │
│    ├──  Transaction    → risk gauge + SHAP waterfall per TXN      │
│    ├──  Alerts         → fraud alert console log                  │
│    └──  Manual Test    → custom transaction submission            │
└─────────────────────────────────────────────────────────────────────┘
```

---

##  Quick Start

### 1 — Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2 — Train the models

```bash
# Generates synthetic data if creditcard.csv is missing,
# trains all 3 models and saves to models/
python train.py
```

Expected output (takes ~2–5 minutes):
```
MODEL LEADERBOARD
Model                      ROC-AUC   PR-AUC   Recall       F1   Fraud Found
────────────────────────────────────────────────────────────────────────────
xgboost                     0.9971   0.8234   0.9200   0.9231    92/100 (92.00%)
random_forest               0.9958   0.7991   0.8900   0.8816    89/100 (89.00%)
logistic_regression         0.9721   0.6812   0.8200   0.8122    82/100 (82.00%)
```

### 3 — Start the API

```bash
python run_api.py
# API docs: http://localhost:8000/docs
```

### 4 — Start the dashboard

```bash
python run_dashboard.py
# Dashboard: http://localhost:8501
```

---

## 🐳 Docker Deployment

```bash
# Build & start both services
docker-compose up --build

# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

---

## 📡 API Reference

### `POST /predict`

Submit a transaction for fraud scoring.

**Request body:**
```json
{
  "time": 172800,
  "amount": 250.00,
  "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,
  "V4": 1.3782,  "V5": -0.3383,
  "...",
  "V28": -0.0211,
  "transaction_id": "TXN-20240101-001"
}
```

**Response:**
```json
{
  "transaction_id": "TXN-20240101-001",
  "fraud_probability": 0.034512,
  "risk_score": 0.127158,
  "is_fraud": false,
  "risk_level": "LOW",
  "confidence": 0.7457,
  "reasons": [],
  "shap_explanation": {
    "top_features": [
      { "feature": "V14", "impact": -0.1823, "value": -0.3111 },
      { "feature": "V4",  "impact":  0.0942, "value":  1.3782 }
    ],
    "base_value": -4.3217
  },
  "model_used": "xgboost",
  "processing_time_ms": 4.7
}
```

**Fraud example (high-value + anomalous PCA):**
```json
{
  "fraud_probability": 0.967341,
  "risk_score": 0.924139,
  "is_fraud": true,
  "risk_level": "CRITICAL",
  "reasons": [
    "Critical-value transaction: $7,200.00 exceeds $5,000 threshold",
    "Highly anomalous behavioural signature: 6 PCA features exceed 3σ"
  ]
}
```

### `GET /health`
```json
{ "status": "healthy", "model_loaded": true, "active_model": "xgboost",
  "version": "2.0.0", "uptime_seconds": 143.2 }
```

---

## 📊 Model Performance (on synthetic dataset)

| Model | ROC-AUC | PR-AUC | Recall | F1 |
|---|---|---|---|---|
| XGBoost ★ | ~0.997 | ~0.82 | ~0.92 | ~0.92 |
| Random Forest | ~0.996 | ~0.80 | ~0.89 | ~0.88 |
| Logistic Regression | ~0.972 | ~0.68 | ~0.82 | ~0.81 |

> **Recall is the primary business metric** — missing a fraud costs far more
> than an unnecessary card block (which can be reversed by a customer call).

---

## 🧮 Hybrid Scoring Formula

```
risk_score = (0.70 × XGBoost_probability) + (0.30 × rule_score)
```

**Rules applied:**

| Rule | Score Added | Trigger |
|---|---|---|
| Critical amount | +0.50 | Amount ≥ $5,000 |
| High amount | +0.25 | Amount ≥ $1,000 |
| Off-hours | +0.15 | Hour UTC ∈ \[23:00 – 05:59\] |
| PCA alert | +0.30 | ≥ 5 features exceed 3σ |
| PCA warn | +0.10 | ≥ 3 features exceed 3σ |
| Micro-transaction | +0.20 | $0.01 ≤ Amount ≤ $1.00 |

---

## 🖥 Dashboard Screenshots

> *Screenshots taken from the live dashboard — replace with your own captures.*

| Live Feed Tab | Analytics Tab |
|---|---|
| `[screenshot: live feed]` | `[screenshot: analytics]` |

| Transaction Detail + SHAP | Alerts Tab |
|---|---|
| `[screenshot: SHAP waterfall + risk gauge]` | `[screenshot: alert log]` |

---

## 🗂 Project Structure

```
AI Fraud Detection System/
├── src/
│   ├── data_preprocessing.py    # load, clean, SMOTE, scale → train/test split
│   ├── feature_engineering.py   # derived features (amount_log, is_night, ...)
│   ├── train.py                 # 3-model training pipeline with CV + leaderboard
│   └── evaluate.py              # metrics, ROC/PR plots, SHAP summary
├── api/
│   ├── main.py                  # FastAPI app (lifespan, routes, alert logging)
│   ├── schemas.py               # Pydantic v2 request/response models
│   ├── predictor.py             # Singleton model loader + SHAP explainer
│   └── business_logic.py        # Hybrid ML + rule-based risk scoring engine
├── dashboard/
│   ├── app.py                   # Main Streamlit app (5 tabs, dark theme)
│   └── components/
│       ├── charts.py            # 7 Plotly chart functions
│       ├── simulator.py         # API caller + offline fallback + session history
│       └── alerts.py            # Multi-channel alert manager (log + email)
├── streaming/
│   └── transaction_stream.py    # Async gen + sync gen + mock Kafka queue
├── data/
│   └── generate_sample_data.py  # Synthetic dataset generator
├── models/                      # Trained .pkl files + scaler + evaluation reports
│   └── reports/                 # ROC / PR / confusion matrix / SHAP plots
├── tests/
│   ├── test_api.py              # FastAPI endpoint + business logic tests
│   └── test_preprocessing.py    # Preprocessing / feature / streaming tests
├── train.py                     # Root-level training entry point
├── run_api.py                   # Root-level API launcher
├── run_dashboard.py             # Root-level dashboard launcher
├── Dockerfile                   # Multi-stage API image
├── Dockerfile.dashboard         # Dashboard image
├── docker-compose.yml           # Full-stack deployment
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and adjust:

```env
PRIMARY_MODEL=xgboost         # or random_forest / logistic_regression
API_PORT=8000
DASHBOARD_PORT=8501

# Optional email alerts
ALERT_EMAIL=you@example.com
ALERT_EMAIL_TO=ops@example.com
ALERT_EMAIL_PASSWORD=app-token
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests are fully offline — no running API or model files required.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| ML Models | scikit-learn · XGBoost |
| Imbalance | imbalanced-learn (SMOTE) |
| Explainability | SHAP (TreeExplainer · LinearExplainer) |
| API | FastAPI · Pydantic v2 · Uvicorn |
| Dashboard | Streamlit · Plotly |
| Streaming | asyncio · custom mock Kafka queue |
| Containerisation | Docker · Docker Compose |
| Testing | pytest |

---

## 📐 Design Decisions

**Why XGBoost as primary?**
Tree boosting models outperform linear models on tabular fraud data due to feature interactions that are already partially encoded in the PCA components (V1-V28). PR-AUC optimisation (`eval_metric="aucpr"`) aligns training directly with the business objective.

**Why SMOTE + `scale_pos_weight`?**
SMOTE is applied at training time; `scale_pos_weight` provides an additional safety net when the SMOTE ratio is imperfect. Both together give the model a strong prior for the minority class without overfitting.

**Why hybrid scoring?**
ML alone can miss novel fraud patterns outside its training distribution. Rule-based systems are brittle and generate excessive false positives. The hybrid formula inherits the best of both — ML handles complex patterns, rules catch obvious red flags that may appear below the model's probability threshold.

**Why mock Kafka?**
Real Kafka adds significant infrastructure complexity (Zookeeper, broker, topic management). The mock queue provides the same producer/consumer API without external dependencies — making the codebase runnable on a laptop while being straightforward to replace with `confluent-kafka` or `aiokafka` in production.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push and open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as a fintech  project demonstrating production-grade ML engineering practices.*
