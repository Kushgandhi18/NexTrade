"""
predict.py
GET /predict?stock=AAPL&model=gru
GET /backtest?stock=AAPL&model=gru
"""

import logging
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from backend.services.prediction_service import PredictionService
from backend.utils.backtesting import BacktestingService
from backend.services.data_service import DataService
from backend.services.feature_service import FeatureService
from backend.db.postgres import get_db, Prediction
from backend.models.model_factory import SUPPORTED_MODELS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])

prediction_service = PredictionService()
backtest_service = BacktestingService()
data_service = DataService()
feature_service = FeatureService()


@router.get("")
async def predict(
    stock: str = Query(..., description="Ticker symbol e.g. AAPL"),
    model: str = Query("gru", description=f"Model: {SUPPORTED_MODELS}"),
    db: Session = Depends(get_db),
):
    """
    Get next-day price prediction for a stock using the specified model.
    Results are cached in Redis for 1 hour.
    """
    stock = stock.upper().strip()
    model = model.lower().strip()

    if model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Supported: {SUPPORTED_MODELS}",
        )

    try:
        result = prediction_service.predict(stock=stock, model=model)

        # Persist prediction to DB
        db_pred = Prediction(
            symbol=stock,
            model=model,
            predicted_price=result["predicted_price"],
            last_known_price=result["last_price"],
            direction=result["direction"],
            change_pct=result["change_pct"],
        )
        db.add(db_pred)
        db.commit()

        return result

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for {stock}/{model}. POST /train first.",
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest")
async def backtest(
    stock: str = Query(..., description="Ticker symbol e.g. AAPL"),
    model: str = Query("gru", description=f"Model: {SUPPORTED_MODELS}"),
    period: str = Query("2y", description="Backtest period e.g. 1y, 2y, 5y"),
):
    """
    Run backtest simulation using predictions vs actual prices.
    Returns profit, Sharpe ratio, max drawdown, and portfolio history.
    """
    stock = stock.upper().strip()
    model = model.lower().strip()

    try:
        # Fetch historical data
        df = data_service.fetch_stock_data(stock, period=period)
        df_feat = feature_service.add_indicators(df)
        X, y, _ = feature_service.create_sequences(df_feat, fit_scaler=True)

        # Get model and generate predictions
        from backend.services.inference_pipeline import InferencePipeline
        pipeline = InferencePipeline()
        loaded_model = pipeline._get_or_load_model(stock, model)
        predictions = loaded_model.predict(X)

        # Run backtest
        result = backtest_service.simulate(predictions=predictions, actual_prices=y)
        result["stock"] = stock
        result["model"] = model
        result["period"] = period
        return result

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for {stock}/{model}. POST /train first.",
        )
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
