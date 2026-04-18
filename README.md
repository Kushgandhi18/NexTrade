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
| Frontend | Vanilla HTML/JS, Chart.js |

---

## Deployment

| Component | Option A (Simple) | Option B (Production) |
|---|---|---|
| Backend | Railway / Render | AWS EC2 |
| DB | Supabase | AWS RDS |
| Models | Local store | AWS S3 |
| Frontend | Vercel / GitHub Pages | AWS CloudFront |
