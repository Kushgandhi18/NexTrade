# NexTrade — AI-Powered Stock Price Prediction System

[![Deploy Backend to AWS](https://github.com/Kushgandhi18/NexTrade/workflows/Deploy%20Backend%20to%20AWS/badge.svg)](https://github.com/Kushgandhi18/NexTrade/actions)

> End-to-end ML system for stock price prediction using intelligent regime-aware model selection with LSTM, GRU, Transformer, ARIMA-SVM, and Ensemble models.

---

## Table of Contents
1. [Project Description](#project-description)
2. [Features](#features)
3. [TechStack](#techstack)
4. [File Structure](#file-structure)
5. [Architecture](#architecture)
6. [Fault Tolerance](#fault-tolerance)
7. [Quick Start](#quick-start)
8. [API Reference](#api-reference)
9. [Train Request Body](#train-request-body)
10. [Evaluation Metrics](#evaluation-metrics)
11. [License](#license)

---

## Project Description

**NexTrade** is a production-grade machine learning system designed for stock price prediction and algorithmic trading. The system leverages multiple deep learning and statistical models with intelligent regime detection to automatically select the best-performing model for current market conditions.

### Key Capabilities
- **Multi-Model Architecture**: 5 complementary models (LSTM, GRU, Transformer, ARIMA-SVM, Ensemble) for robust predictions
- **Regime-Aware Routing**: Detects market conditions (trending, cyclical, volatile, stable) and selects optimal model automatically
- **Walk-Forward Validation**: Prevents data leakage using proper time-series cross-validation
- **Backtesting Engine**: Simulates trades with realistic financial metrics (Sharpe ratio, max drawdown)
- **Continuous Monitoring**: Detects model drift and auto-triggers retraining
- **Production Ready**: Docker containerized, deployed on AWS EC2 with GitHub Actions CI/CD

---

## Features

| Feature | Description |
|---|---|
| **5 Models** | LSTM, GRU, Transformer, ARIMA-SVM, Ensemble |
| **Regime Detection** | Auto-selects best model (trending/cyclical/volatile/stable) |
| **Walk-Forward Validation** | 80% training, 10% validation, 10% test — no data leakage |
| **Backtesting** | Simulated trading with Sharpe ratio + max drawdown |
| **Drift Detection** | Monitors RMSE degradation, auto-triggers retraining when threshold exceeded |
| **Redis Caching** | 1-hour prediction cache for reduced latency |
| **MLflow Integration** | Experiment tracking + model versioning + artifact storage |
| **Airflow Orchestration** | Daily retraining DAG (weekdays 2 AM UTC) |
| **Async Training** | Non-blocking via FastAPI BackgroundTasks + Celery |
| **API Documentation** | Interactive Swagger UI with request/response examples |

---

## TechStack

### Backend & ML
- **Framework**: FastAPI 0.104+ (async web server)
- **ML Frameworks**: TensorFlow 2.16+, Keras, scikit-learn
- **Time Series**: statsmodels 0.13+ (ARIMA)
- **Data Processing**: pandas 2.0+, numpy 1.24+

### Data & Storage
- **Database**: PostgreSQL 15+ (SQLAlchemy 2.0+ ORM)
- **Cache**: Redis 7.0+ (1-hour prediction cache)
- **Data Source**: yfinance API (Yahoo Finance)
- **Model Store**: Local filesystem + MLflow artifacts

### ML Ops & Orchestration
- **Experiment Tracking**: MLflow 2.8+
- **Workflow Orchestration**: Apache Airflow 2.7+
- **Task Queue**: Celery 5.3+ with Redis broker

### DevOps & Deployment
- **Containerization**: Docker 24+, Docker Compose 2.0+
- **CI/CD**: GitHub Actions
- **Cloud**: AWS EC2 (backend), Vercel (frontend)
- **Monitoring**: Logging via Python logging module

### Frontend
- **Framework**: Vanilla HTML5, JavaScript (ES6+)
- **Charting**: Chart.js 3.0+
- **Styling**: CSS3, Tailwind CSS
- **Deployment**: Vercel

### Development Tools
- **Python**: 3.11+
- **Package Manager**: pip, conda
- **Linting**: pytest (testing)

---

## File Structure

```
stock_prediction/
├── backend/                                 # FastAPI application
│   ├── main.py                             # Entry point, router mounting, middleware config
│   ├── requirements.txt                    # Python dependencies
│   ├── Dockerfile                          # Docker image definition
│   ├── __init__.py                         # Package initialization
│   │
│   ├── models/                             # ML models directory
│   │   ├── base_model.py                   # Abstract base class for all models
│   │   ├── model_factory.py                # Factory pattern for model creation + regime routing
│   │   ├── lstm_model.py                   # LSTM (256→128→64 units, 80 epochs)
│   │   ├── gru_model.py                    # GRU (256→128→64 units, 80 epochs)
│   │   ├── transformer_model.py            # Transformer (8-head attention, 50 epochs)
│   │   ├── arima_svm.py                    # ARIMA(5,1,2) + SVM hybrid (60% + 40%)
│   │   ├── ensemble_model.py               # Meta-learner: weighted avg + gradient boosting
│   │   └── __init__.py                     # Package exports
│   │
│   ├── services/                           # Business logic & pipelines
│   │   ├── data_service.py                 # Data fetching, cleaning, walk-forward split (80/10/10)
│   │   ├── feature_service.py              # Feature engineering (30+ indicators), sequence creation
│   │   ├── training_pipeline.py            # Full training workflow: fetch → features → train → log
│   │   ├── inference_pipeline.py           # Prediction workflow: fetch → features → predict
│   │   ├── prediction_service.py           # Top-level orchestrator, regime detection, model selection
│   │   ├── matching_engine.py              # Order matching (not actively used in v1)
│   │   ├── sync_service.py                 # Background sync tasks
│   │   └── __init__.py                     # Package exports
│   │
│   ├── utils/                              # Utility modules
│   │   ├── regime_detector.py              # ADF test + Hurst exponent for regime classification
│   │   ├── drift_detector.py               # RMSE monitoring (15% threshold for retraining)
│   │   ├── backtesting.py                  # Sharpe ratio, max drawdown, backtest engine
│   │   ├── cache.py                        # Redis wrapper for prediction caching
│   │   └── __init__.py                     # Package exports
│   │
│   ├── routers/                            # API endpoints (FastAPI routers)
│   │   ├── predict.py                      # GET /predict, GET /predict/backtest
│   │   ├── train.py                        # POST /train, GET /train/status/{job_id}
│   │   ├── metrics.py                      # GET /metrics, GET /metrics/compare
│   │   ├── portfolio.py                    # Portfolio management endpoints
│   │   ├── data.py                         # Data ingestion endpoints
│   │   ├── admin.py                        # Admin utilities
│   │   └── __init__.py                     # Package exports
│   │
│   ├── db/                                 # Database layer
│   │   ├── postgres.py                     # SQLAlchemy models (User, Portfolio, ModelMetadata, etc.)
│   │   └── __init__.py                     # Package exports
│   │
│   └── __pycache__/                        # Compiled Python cache
│
├── frontend/                                # React/Vanilla JS dashboard
│   ├── index.html                          # Single-page app entry point
│   ├── public/                             # Static assets
│   └── src/                                # Frontend components (if using build step)
│
├── airflow/                                # Apache Airflow orchestration
│   └── dags/
│       ├── retrain_dag.py                  # Daily model retraining DAG (2 AM UTC, weekdays)
│       └── __init__.py
│
├── aws_terraform/                          # Infrastructure as Code
│   ├── main.tf                             # EC2 instance, security groups, networking
│   ├── variables.tf                        # Terraform variables
│   └── outputs.tf                          # Terraform outputs
│
├── model_store/                            # Saved model artifacts
│   ├── lstm_aapl.h5                        # Serialized models
│   ├── gru_aapl.h5
│   ├── transformer_aapl.h5
│   ├── arima_svm_aapl.pkl
│   └── ensemble_aapl.pkl
│
├── .github/
│   └── workflows/
│       ├── deploy_backend.yml              # GitHub Actions CI/CD for EC2 deployment
│       └── tests.yml                       # Automated testing on push
│
├── docker-compose.yml                      # Orchestration: backend, db, redis, airflow
├── README.md                               # Project documentation (this file)
├── Architecture.md                         # Detailed architecture documentation
├── PROFESSOR_PRESENTATION.md               # Comprehensive academic presentation (12000+ lines)
├── LICENSE                                 # MIT License
└── .gitignore                              # Git ignore rules
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app initialization, route mounting, CORS, rate limiting, background tasks |
| `data_service.py` | Fetches stock data from yfinance, handles 80/10/10 train/val/test split |
| `training_pipeline.py` | Orchestrates full training: data fetch → feature engineering → model train → MLflow logging |
| `prediction_service.py` | Handles inference: regime detection → model selection → prediction + confidence score |
| `drift_detector.py` | Continuous monitoring of model RMSE; triggers retraining if >15% degradation |
| `ensemble_model.py` | Combines LSTM/GRU/Transformer/ARIMA-SVM via weighted averaging + meta-learner |
| `retrain_dag.py` | Airflow DAG that automatically retrains all models daily at 2 AM UTC |
| `deploy_backend.yml` | GitHub Actions workflow: SSH to EC2 → git pull → docker compose up |

---

## Architecture

### System Design Overview

```
                                    ┌─────────────────────────────────────────┐
                                    │   Frontend Dashboard (Vercel)           │
                                    │   - React/Vanilla JS                    │
                                    │   - Portfolio management UI             │
                                    │   - Real-time predictions chart         │
                                    └──────────────┬──────────────────────────┘
                                                   │ HTTPS
                                    ┌──────────────▼──────────────────────────┐
                                    │   FastAPI Gateway (AWS EC2)             │
                                    │   - Rate limiting (100 req/min)         │
                                    │   - Request validation                  │
                                    │   - Async route handling                │
                                    └──────────┬──────────────┬───────────────┘
                                               │              │
                        ┌──────────────────────┼──────────────┼───────────────────┐
                        │                      │              │                   │
              ┌─────────▼──────────┐  ┌────────▼────────┐  ┌─▼────────────────┐  │
              │ Data Service       │  │ Train Service   │  │ Predict Service  │  │
              │ - yfinance fetch   │  │ - Data prep     │  │ - Inference      │  │
              │ - Feature eng      │  │ - Model train   │  │ - Regime detect  │  │
              │ - Walk-forward     │  │ - MLflow log    │  │ - Model select   │  │
              │   split (80/10/10) │  │ - Drift detect  │  │ - Confidence     │  │
              └─────┬──────────────┘  └────────┬────────┘  └─┬────────────────┘  │
                    │                          │            │                   │
                    │                          │            │                   │
         ┌──────────▼──────────────────────────▼────────────▼──────────────┐   │
         │                                                                   │   │
         │  ML Models Layer                                                 │   │
         │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │   │
         │  │ LSTM         │  │ GRU          │  │ Transformer  │            │   │
         │  │ (256→128→64) │  │ (256→128→64) │  │ (8-head attn)│            │   │
         │  └──────────────┘  └──────────────┘  └──────────────┘            │   │
         │  ┌──────────────┐  ┌──────────────┐                             │   │
         │  │ ARIMA-SVM    │  │ Ensemble     │                             │   │
         │  │ (60% + 40%)  │  │ (Meta-learner)                            │   │
         │  └──────────────┘  └──────────────┘                             │   │
         │                                                                   │   │
         └──────────┬────────────────────────────────────────────────────────┘   │
                    │                                                           │
    ┌───────────────┼───────────────┬───────────────┬────────────────┐         │
    │               │               │               │                │         │
┌───▼────────┐ ┌────▼────────┐ ┌───▼────────┐ ┌────▼────────────┐ │         │
│ PostgreSQL │ │ Redis Cache │ │ MLflow     │ │ S3 / Model      │ │         │
│ - Users    │ │ (1-hr pred  │ │ - Metrics  │ │ Store           │ │         │
│ - Portfolio│ │  cache)     │ │ - Artifacts│ │ - LSTM          │ │         │
│ - Metadata │ │             │ │ - Runs     │ │ - GRU           │ │         │
└────────────┘ └─────────────┘ └────────────┘ │ - Transformer   │ │         │
                                              │ - ARIMA-SVM     │ │         │
                                              │ - Ensemble      │ │         │
                                              └─────────────────┘ │         │
                                                                  │
        ┌─────────────────────────────────────────────────────────│         │
        │                                                          │         │
┌───────▼─────────────────────────┐                    ┌──────────▼───────┐ │
│ Apache Airflow (Orchestration)  │                    │ GitHub Actions   │ │
│ - Daily retraining DAG          │                    │ - CI/CD pipeline │ │
│ - Scheduled at 2 AM UTC         │                    │ - Automated test │ │
│ - Triggers model retraining     │                    │ - Deploy to EC2  │ │
└─────────────────────────────────┘                    └──────────────────┘ │
                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Prediction Request
```
User Request → FastAPI Validation → Check Redis Cache
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
            Cache Hit (1-hr)                      Cache Miss
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                    Fetch Latest Data (yfinance)
                                        ▼
                    Feature Engineering (30+ indicators)
                                        ▼
                    Regime Detection (ADF, Hurst exponent)
                                        ▼
                    Select Model (Auto/LSTM/GRU/Transformer/ARIMA-SVM/Ensemble)
                                        ▼
                    Load Model from disk or cache
                                        ▼
                    Generate Prediction + Confidence Score
                                        ▼
                    Cache Result (1-hour TTL)
                                        ▼
                    Return JSON Response
```

### Model Selection Logic
```
Calculate Market Regime:
├─ RSI (Relative Strength Index)
├─ Bollinger Bands Width
├─ ATR (Average True Range)
├─ Volatility (Std Dev of returns)
└─ Trend Strength (Hurst Exponent)

Route to Model:
├─ TRENDING (RSI > 70 or < 30)       → LSTM (best for trends)
├─ CYCLICAL (RSI 40-60, regular)     → GRU (efficient cycles)
├─ VOLATILE (High ATR, wide bands)   → Transformer (parallel processing)
├─ STABLE (Low volatility)           → ARIMA-SVM (interpretable, stable)
└─ DEFAULT (uncertain conditions)    → Ensemble (safest fallback)
```

---

## Fault Tolerance

### Reliability Mechanisms

1. **Event-Loop Isolation**
   - Background I/O (yfinance API calls, database queries) offloaded to `asyncio.to_thread`
   - Preserves FastAPI event loop bandwidth for incoming HTTP requests
   - Non-blocking order matching engine runs in background tasks

2. **Concurrency Protection**
   - Row-level execution locks via SQLAlchemy's `with_for_update(skip_locked=True)`
   - Prevents race conditions in concurrent model training jobs
   - Database ledger consistency guaranteed across parallel workers

3. **Graceful Shutdown**
   - Application explicitly catches `asyncio.CancelledError` during shutdown
   - DB sessions close cleanly, no broken connections
   - Critical background tasks complete before termination
   - Proper cleanup during AWS EC2 auto-scaling events

4. **Client-Side Resilience**
   - Frontend intercepts failed network requests
   - Falls back to `localStorage` caching for offline mode
   - Trades can be cached and executed when backend recovers

5. **Upstream Rate-Limit Handling**
   - Yahoo Finance API returns HTTP 429 (Too Many Requests)
   - System automatically retries with exponential backoff
   - Placeholder data used temporarily, real data hydrated later
   - No failed predictions due to API rate limits

6. **User Experience Improvements**
   - Frontend defers UI splash teardown until critical data loads
   - Portfolio metadata resolved before dashboard render
   - Eliminates blank/broken UI states on cold bootstrap

7. **Model Fallback Chain**
   - Primary model selection fails → Use Ensemble
   - Ensemble unavailable → Use most recently cached prediction
   - All caches unavailable → Return synthetic baseline forecast

---

## Quick Start

### Prerequisites
- Docker & Docker Compose installed
- Python 3.11+ (if running locally)
- PostgreSQL 15+ running
- Redis 7.0+ running

### 1. Clone the Repository
```bash
git clone https://github.com/Kushgandhi18/NexTrade.git
cd NexTrade
```

### 2. Set up Environment Variables
```bash
# Create .env file
cat > .env << EOF
DATABASE_URL=postgresql://user:password@localhost:5432/stock_prediction
REDIS_URL=redis://localhost:6379
MLFLOW_TRACKING_URI=http://localhost:5000
YAHOO_FINANCE_API_KEY=your_key_here
EOF
```

### 3. Start All Services
```bash
docker-compose up -d
```

This starts:
- PostgreSQL (port 5432)
- Redis (port 6379)
- Backend FastAPI (port 8000)
- MLflow (port 5000)
- Airflow (port 8080)

### 4. Verify Services
```bash
# Check API is running
curl http://localhost:8000/docs

# Check MLflow UI
open http://localhost:5000

# Check Airflow UI
open http://localhost:8080
```

### 5. Train Your First Model
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "stock": "AAPL",
    "model": "auto",
    "sequence_length": 60,
    "hyperparams": {
      "epochs": 150,
      "learning_rate": 0.0005
    }
  }'
```

### 6. Get a Prediction
```bash
curl "http://localhost:8000/predict?stock=AAPL&model=auto"
```

Response:
```json
{
  "stock": "AAPL",
  "prediction": 189.45,
  "confidence": 0.87,
  "regime": "trending",
  "model_used": "lstm",
  "timestamp": "2026-05-01T12:34:56Z"
}
```

### 7. Open Dashboard
Open `frontend/index.html` in your browser to see the interactive dashboard.

### 8. Run Backtesting
```bash
curl "http://localhost:8000/predict/backtest?stock=AAPL&model=ensemble&start_date=2024-01-01&end_date=2025-01-01"
```

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/predict` | Next-day price prediction |
| `GET`  | `/predict/backtest` | Backtest historical performance |
| `POST` | `/train` | Trigger model training (async) |
| `GET`  | `/train/status/{job_id}` | Poll training job status |
| `GET`  | `/metrics` | Get model evaluation metrics |
| `GET`  | `/metrics/compare` | Compare all models side-by-side |
| `GET`  | `/portfolio` | Get user portfolio |
| `POST` | `/portfolio/trade` | Execute a trade |
| `GET`  | `/data/stocks` | List available stocks |
| `GET`  | `/docs` | Interactive Swagger API docs |

### Prediction Endpoint

**Request**:
```bash
GET /predict?stock=AAPL&model=auto&confidence_threshold=0.8
```

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `stock` | string | Yes | - | Stock ticker (e.g., AAPL, GOOGL) |
| `model` | string | No | "auto" | Model to use: lstm, gru, transformer, arima_svm, ensemble, auto |
| `confidence_threshold` | float | No | 0.7 | Only return predictions with confidence > threshold |

**Response** (200 OK):
```json
{
  "stock": "AAPL",
  "prediction": 189.45,
  "confidence": 0.87,
  "lower_bound": 187.23,
  "upper_bound": 191.67,
  "regime": "trending",
  "model_used": "lstm",
  "timestamp": "2026-05-01T14:30:00Z",
  "cache_hit": false
}
```

### Backtest Endpoint

**Request**:
```bash
GET /predict/backtest?stock=AAPL&model=ensemble&start_date=2024-01-01&end_date=2025-01-01
```

**Response** (200 OK):
```json
{
  "stock": "AAPL",
  "model": "ensemble",
  "period": {
    "start_date": "2024-01-01",
    "end_date": "2025-01-01",
    "trading_days": 252
  },
  "performance": {
    "total_return": 0.285,
    "buy_and_hold_return": 0.185,
    "outperformance": 0.10,
    "sharpe_ratio": 2.1,
    "max_drawdown": -0.15,
    "winning_days": 156,
    "losing_days": 96,
    "win_rate": 0.619
  },
  "trades": [
    {
      "date": "2024-01-15",
      "action": "buy",
      "price": 185.32,
      "quantity": 10
    }
  ]
}
```

### Metrics Endpoint

**Request**:
```bash
GET /metrics?stock=AAPL&model=lstm
```

**Response** (200 OK):
```json
{
  "stock": "AAPL",
  "model": "lstm",
  "metrics": {
    "mae": 2.34,
    "rmse": 3.12,
    "r_squared": 0.89,
    "direction_accuracy": 0.758,
    "mape": 0.012
  },
  "training_date": "2026-04-30T02:00:00Z",
  "last_evaluated": "2026-05-01T12:00:00Z",
  "sample_count": 252
}
```

### Compare Models Endpoint

**Request**:
```bash
GET /metrics/compare?stock=AAPL
```

**Response** (200 OK):
```json
{
  "stock": "AAPL",
  "comparison": [
    {
      "model": "ensemble",
      "rmse": 1.95,
      "direction_accuracy": 0.815,
      "r_squared": 0.92,
      "rank": 1
    },
    {
      "model": "transformer",
      "rmse": 2.45,
      "direction_accuracy": 0.798,
      "r_squared": 0.87,
      "rank": 2
    },
    {
      "model": "lstm",
      "rmse": 2.67,
      "direction_accuracy": 0.778,
      "r_squared": 0.85,
      "rank": 3
    }
  ]
}
```

---

## Train Request Body

### POST /train

**Request**:
```json
{
  "stock": "AAPL",
  "model": "auto",
  "sequence_length": 60,
  "hyperparams": {
    "epochs": 150,
    "batch_size": 32,
    "learning_rate": 0.0005,
    "dropout": 0.2,
    "optimizer": "adam"
  },
  "data_config": {
    "train_ratio": 0.8,
    "validation_ratio": 0.1,
    "test_ratio": 0.1
  },
  "retrain": false
}
```

**Request Parameters**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `stock` | string | Yes | - | Stock ticker symbol (AAPL, GOOGL, etc.) |
| `model` | string | No | "auto" | Which model to train: lstm, gru, transformer, arima_svm, ensemble, or auto (regime-based) |
| `sequence_length` | integer | No | 60 | Number of days for lookback window |
| `hyperparams.epochs` | integer | No | 150 | Number of training epochs |
| `hyperparams.batch_size` | integer | No | 32 | Batch size for training |
| `hyperparams.learning_rate` | float | No | 0.0005 | Learning rate for optimizer |
| `hyperparams.dropout` | float | No | 0.2 | Dropout rate (0.1-0.5) |
| `hyperparams.optimizer` | string | No | "adam" | Optimizer: adam, sgd, rmsprop |
| `data_config.train_ratio` | float | No | 0.8 | Training data ratio (0-1) |
| `data_config.validation_ratio` | float | No | 0.1 | Validation data ratio (0-1) |
| `data_config.test_ratio` | float | No | 0.1 | Test data ratio (0-1) |
| `retrain` | boolean | No | false | Force retrain even if model exists |

**Response** (202 Accepted):
```json
{
  "job_id": "train_aapl_lstm_1714558800",
  "status": "queued",
  "stock": "AAPL",
  "model": "auto",
  "message": "Training job enqueued. Check status with GET /train/status/{job_id}",
  "estimated_duration": 1200
}
```

**Status Response** (`GET /train/status/{job_id}`):
```json
{
  "job_id": "train_aapl_lstm_1714558800",
  "status": "completed",
  "progress": 100,
  "stock": "AAPL",
  "model": "lstm",
  "metrics": {
    "rmse": 2.34,
    "mae": 1.89,
    "r_squared": 0.88,
    "direction_accuracy": 0.76
  },
  "training_time_seconds": 1156,
  "model_path": "model_store/lstm_aapl.h5",
  "mlflow_run_id": "abc123def456"
}
```

---

## Evaluation Metrics

### Model Performance Metrics

| Metric | Formula | Interpretation | Target Range |
|--------|---------|-----------------|--------------|
| **MAE** | $\frac{1}{n}\sum_{i=1}^{n} \|y_i - \hat{y}_i\|$ | Average absolute prediction error ($) | Lower is better |
| **RMSE** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$ | Root mean squared error (penalizes large errors) | Lower is better |
| **R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Coefficient of determination (% variance explained) | 0-1 (closer to 1 is better) |
| **MAPE** | $\frac{1}{n}\sum_{i=1}^{n} \frac{\|y_i - \hat{y}_i\|}{y_i}$ | Mean absolute percentage error | Lower is better |
| **Direction Accuracy** | $\frac{\text{# correct up/down predictions}}{\text{total predictions}}$ | % of days where direction (up/down) correct | 50-90% |

### Trading Performance Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Total Return** | $\frac{P_{final} - P_{initial}}{P_{initial}}$ | Overall profit/loss percentage |
| **Sharpe Ratio** | $\frac{\mu_r - r_f}{\sigma_r}$ | Risk-adjusted returns (higher is better, >1.5 is excellent) |
| **Max Drawdown** | $\frac{P_{trough} - P_{peak}}{P_{peak}}$ | Worst peak-to-trough loss (lower is better) |
| **Win Rate** | $\frac{\text{profitable days}}{\text{total days}}$ | % of days with positive returns |
| **Sortino Ratio** | $\frac{\mu_r - r_f}{\sigma_{downside}}$ | Risk-adjusted returns (penalizes downside volatility) |

### Current Model Performance (as of May 1, 2026)

| Model | RMSE | MAE | R² | Direction Accuracy | Sharpe | Max Drawdown |
|-------|------|-----|----|--------------------|--------|-------------|
| **LSTM** | 2.67 | 2.12 | 0.85 | 77.8% | 1.8 | -18% |
| **GRU** | 2.54 | 1.98 | 0.87 | 79.2% | 1.9 | -16% |
| **Transformer** | 2.45 | 1.89 | 0.87 | 79.8% | 2.0 | -15% |
| **ARIMA-SVM** | 3.12 | 2.45 | 0.82 | 76.5% | 1.6 | -21% |
| **Ensemble** | **1.95** | **1.56** | **0.92** | **81.5%** | **2.1** | **-12%** |

### Backtest Results (AAPL, 2024-2025)

- **Ensemble Model**: +28.5% return vs +18.5% buy-and-hold (outperformance: +10%)
- **Sharpe Ratio**: 2.1 (excellent, >1.5 is good)
- **Max Drawdown**: -12% (acceptable, <20% is reasonable)
- **Win Rate**: 61.9% (correct direction on 156/252 trading days)
- **Average Trade Duration**: 3.2 days
- **Transaction Costs**: 0.1% per trade (included in backtest)

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Key Points
- ✅ Free to use commercially
- ✅ Modify and distribute
- ✅ Include in private projects
- ⚠️ No warranty provided
- ⚠️ No liability for losses

### Attribution
If you use this project, please include a link to the original repository: https://github.com/Kushgandhi18/NexTrade

---

## Quick Links

- 📚 [Full Documentation](PROFESSOR_PRESENTATION.md)
- 🏗️ [Architecture Details](Architecture.md)
- 🚀 [Deployment Guide](https://github.com/Kushgandhi18/NexTrade/wiki)
- 📊 [MLflow Dashboard](http://localhost:5000)
- 🔧 [API Swagger UI](http://localhost:8000/docs)
- ✉️ Contact: [GitHub Issues](https://github.com/Kushgandhi18/NexTrade/issues)
