"""
prediction_service.py
Thin top-level orchestrator over InferencePipeline.
Adds fallback logic for API/data failures.
"""

import logging
from backend.services.inference_pipeline import InferencePipeline
from backend.utils.cache import CacheService

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Orchestrates predictions.
    Falls back to last-known prediction on data fetch failure.
    """

    def __init__(self):
        self.pipeline = InferencePipeline()
        self.cache = CacheService()

    def predict(self, stock: str, model: str) -> dict:
        """
        Main entrypoint for GET /predict.
        Returns prediction dict or last-known cached result on failure.
        """
        try:
            return self.pipeline.run(stock=stock, model_name=model)
        except FileNotFoundError as e:
            logger.error(f"Model not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Prediction failed for {stock}/{model}: {e}")

            # Fallback: return last known result from cache
            fallback_key = f"predict:{stock}:{model}"
            cached = self.cache.get(fallback_key)
            if cached:
                cached["warning"] = "Live prediction failed — returning cached result"
                logger.warning(f"Returning stale cache for {fallback_key}")
                return cached

            raise RuntimeError(
                f"Prediction failed and no cache available for {stock}/{model}: {e}"
            ) from e
