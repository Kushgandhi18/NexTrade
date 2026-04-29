# StockIQ — AI Stock Prediction System

> End-to-end ML system for stock price prediction using LSTM, GRU, Transformer, ARIMA-SVM, and Ensemble models.

---

## Architecture

```
Frontend (Dashboard) → FastAPI Gateway → [ Data Service | Model Service | Prediction API ]
                                                    ↓               ↓              ↓
                                               PostgreSQL      S3/MLflow         Redis
```

## Features

| Feature | Description |
|---|---|
| **5 Models** | LSTM, GRU, Transformer, ARIMA-SVM, Ensemble |
| **Regime Detection** | Auto-selects best model (trending/cyclical/volatile/stable) |
| **Walk-Forward Validation** | Time-correct split — no data leakage |
| **Backtesting** | Simulated trading with Sharpe ratio + max drawdown |
| **Drift Detection** | Monitors RMSE, auto-triggers retraining |
| **Redis Caching** | 1-hour prediction cache |
| **MLflow** | Experiment tracking + model versioning |
| **Airflow** | Daily retraining DAG (weekdays 2 AM) |
| **Async Training** | Non-blocking via FastAPI BackgroundTasks |

### Fault Tolerance & Reliability
- **Event-Loop Mitigation**: The `OrderMatchingEngine` offloads inherently blocking I/O calls and third-party API polling across `yf.Ticker` structures to background threads utilizing `asyncio.to_thread`, preserving the FastAPI web-handler bandwidth globally.
- **Race Condition Immunity**: Implemented strict row-level execution locks via SQLAlchemy's `with_for_update(skip_locked=True)`, protecting the database ledgers against concurrent worker race-conditions.
- **Graceful Task Termination**: The backend startup sequence explicitly awaits and swallows `asyncio.CancelledError` on teardown logic, guaranteeing DB sessions disconnect gracefully with no broken pipes during ECS cluster scale-downs.
- **Offline Client Hydration**: The Javascript frontend intercepts failed network responses transparently and switches to utilizing `localStorage` caching mechanics to orchestrate trades entirely offline if the AWS cluster is unreachable.
- **Upstream Rate-Limit Resilience**: The Yahoo Finance ETL integration intercepts HTTP 429 Too Many Requests errors and falls back to proxy metadata/placeholders, allowing admin stock ingestion to proceed. The background worker seamlessly hydrates the exact data on subsequent loops once rate limits naturally expire.
- **Dynamic Asynchronous Boot**: The vanilla JS frontend defers the splash screen teardown until all mission-critical telemetry and portfolio metadata is resolved, explicitly eliminating transient blank UI states on cold boots.

---

## Quick Start

### 1. Start all services
```bash
docker-compose up -d
```

### 2. Train a model
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"stock": "AAPL", "model": "auto"}'
```

### 3. Get a prediction
```bash
curl "http://localhost:8000/predict?stock=AAPL&model=gru"
```

### 4. Open the dashboard
Open `frontend/index.html` in your browser.

### 5. MLflow UI
Visit [http://localhost:5000](http://localhost:5000)

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/predict?stock=AAPL&model=gru` | Next-day prediction |
| `GET`  | `/predict/backtest?stock=AAPL&model=gru` | Backtest simulation |
| `POST` | `/train` | Start training (async) |
| `GET`  | `/train/status/{job_id}` | Poll training status |
| `GET`  | `/metrics?stock=AAPL` | Model evaluation metrics |
| `GET`  | `/metrics/compare?stock=AAPL` | Side-by-side model ranking |
| `GET`  | `/docs` | Interactive Swagger UI |

### Train Request Body
```json
{
  "stock": "AAPL",
  "model": "auto",
  "sequence_length": 60,
  "hyperparams": { "epochs": 150, "learning_rate": 0.0005 }
}
```

---

## Project Structure

```
stock_prediction/
├── backend/
│   ├── main.py                        # FastAPI entry
│   ├── models/
│   │   ├── base_model.py              # Abstract interface
│   │   ├── model_factory.py           # Factory + regime routing
│   │   ├── lstm_model.py
│   │   ├── gru_model.py
│   │   ├── transformer_model.py
│   │   ├── arima_svm.py
│   │   └── ensemble_model.py
│   ├── services/
│   │   ├── data_service.py            # yfinance + cleaning
│   │   ├── feature_service.py         # Indicators + sequences
│   │   ├── training_pipeline.py       # Full train flow
│   │   ├── inference_pipeline.py      # Prediction flow
│   │   └── prediction_service.py      # Top-level orchestrator
│   ├── utils/
│   │   ├── regime_detector.py         # ADF + Hurst exponent
│   │   ├── backtesting.py             # Sharpe + drawdown
│   │   ├── drift_detector.py          # RMSE monitoring
│   │   └── cache.py                   # Redis wrapper
│   ├── routers/
│   │   ├── predict.py                 # GET /predict, /backtest
│   │   ├── train.py                   # POST /train
│   │   └── metrics.py                 # GET /metrics
│   ├── db/
│   │   └── postgres.py                # SQLAlchemy models
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── index.html                     # Dashboard (no build step)
├── airflow/
│   └── dags/retrain_dag.py            # Daily retraining
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## Evaluation Metrics

- **MAE** — Mean Absolute Error ($)  
- **RMSE** — Root Mean Square Error ($)  
- **R²** — Coefficient of determination  
- **Direction Accuracy** — % of correctly predicted up/down days  
- **Sharpe Ratio** — Risk-adjusted return (annualized)  
- **Max Drawdown** — Worst peak-to-trough loss  

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | FastAPI, Python 3.11 |
| ML | TensorFlow 2.16, scikit-learn, statsmodels |
| Data | yfinance, pandas, numpy |
| MLOps | MLflow, Airflow |
| DB | PostgreSQL (SQLAlchemy) |
| Cache | Redis |
| DevOps | Docker, GitHub Actions |
| Frontend | Vanilla HTML/JS, CSS, Chart.js |

---

## Deployment

| Component | Option A (Simple) | Option B (Production) |
|---|---|---|
| Backend | Railway / Render | AWS EC2 |
| DB | Supabase | AWS RDS |
| Models | Local store | AWS S3 |
| Frontend | Vercel / GitHub Pages | AWS CloudFront |
