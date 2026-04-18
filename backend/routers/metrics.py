"""
metrics.py
GET /metrics?stock=AAPL — returns stored metrics for all trained models.
GET /metrics/compare?stock=AAPL — side-by-side model comparison.
"""

import logging
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from backend.db.postgres import get_db, ModelMetrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("")
async def get_metrics(
    stock: str = Query(..., description="Ticker symbol e.g. AAPL"),
    model: str = Query(None, description="Filter by model name"),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Return stored evaluation metrics for a stock.
    Optionally filter by model name.
    """
    stock = stock.upper().strip()
    query = db.query(ModelMetrics).filter(ModelMetrics.symbol == stock)
    if model:
        query = query.filter(ModelMetrics.model == model.lower())
    results = query.order_by(ModelMetrics.trained_at.desc()).limit(limit).all()

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for {stock}. Run POST /train first.",
        )

    return [
        {
            "id": r.id,
            "symbol": r.symbol,
            "model": r.model,
            "version": r.version,
            "regime": r.regime,
            "mae": r.mae,
            "rmse": r.rmse,
            "r2": r.r2,
            "direction_accuracy": r.direction_accuracy,
            "sharpe_ratio": r.sharpe_ratio,
            "max_drawdown": r.max_drawdown,
            "train_size": r.train_size,
            "test_size": r.test_size,
            "trained_at": str(r.trained_at),
        }
        for r in results
    ]


@router.get("/compare")
async def compare_models(
    stock: str = Query(..., description="Ticker symbol e.g. AAPL"),
    db: Session = Depends(get_db),
):
    """
    Side-by-side comparison of the latest run for each model.
    Returns ranked by RMSE (ascending = better).
    """
    stock = stock.upper().strip()
    models = ["lstm", "gru", "arima_svm", "transformer", "ensemble"]
    comparison = []

    for model_name in models:
        latest = (
            db.query(ModelMetrics)
            .filter(ModelMetrics.symbol == stock, ModelMetrics.model == model_name)
            .order_by(ModelMetrics.trained_at.desc())
            .first()
        )
        if latest:
            comparison.append({
                "model": model_name,
                "rmse": latest.rmse,
                "mae": latest.mae,
                "r2": latest.r2,
                "direction_accuracy": latest.direction_accuracy,
                "regime": latest.regime,
                "trained_at": str(latest.trained_at),
            })

    if not comparison:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics for {stock}. Train models first.",
        )

    # Rank by RMSE
    comparison.sort(key=lambda x: x["rmse"] or float("inf"))
    return {"stock": stock, "ranking": comparison}
