"""
main.py
FastAPI application entry point.
Mounts all routers, configures CORS, rate limiting, and Swagger docs.
"""

import logging
import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.routers import admin, data, metrics, portfolio, predict, train
from backend.db.postgres import init_db
from backend.services.matching_engine import OrderMatchingEngine

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Rate Limiter
# ------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


# ------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Stock Prediction API...")

    # Initialize database tables
    try:
        init_db()
        logger.info("Database initialized.")
    except Exception as e:
        logger.warning(f"Database init failed (running without DB): {e}")

    # Launch background tick worker for Limit/SL orders
    engine_task = asyncio.create_task(OrderMatchingEngine.run_loop())

    yield
    logger.info("Shutting down Stock Prediction API.")
    engine_task.cancel()
    try:
        await engine_task
    except asyncio.CancelledError:
        pass


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
app = FastAPI(
    title="Stock Prediction API",
    description=(
        "End-to-end stock price prediction system using LSTM, GRU, "
        "Transformer, ARIMA-SVM, and Ensemble models. "
        "Includes regime detection, backtesting, and drift monitoring."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow all local origins in dev
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:5500,http://127.0.0.1:5500,http://127.0.0.1:8080,http://localhost:8080",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",  # catch any local port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------
app.include_router(predict.router)
app.include_router(train.router)
app.include_router(metrics.router)
app.include_router(portfolio.router)
app.include_router(data.router)
app.include_router(admin.router)


# ------------------------------------------------------------------
# Root endpoints
# ------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Stock Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
