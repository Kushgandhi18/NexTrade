"""
inference_pipeline.py
Handles real-time prediction:
  Fetch latest data → Feature engineering → Load model → Predict → Cache
"""

import logging
import os
import numpy as np

from backend.services.data_service import DataService
from backend.services.feature_service import FeatureService
from backend.models.model_factory import ModelFactory
from backend.utils.cache import CacheService

logger = logging.getLogger(__name__)

MODEL_STORE_DIR = os.environ.get("MODEL_STORE_DIR", "model_store")
SEQUENCE_LENGTH = 60


class InferencePipeline:
    """Handles a single prediction request end-to-end."""

    def __init__(self):
        self.data_service = DataService()
        self.feature_service = FeatureService()
        self.model_factory = ModelFactory()
        self.cache = CacheService()

        # In-process model cache (avoids reload on every request)
        self._loaded_models: dict = {}

    def run(self, stock: str, model_name: str) -> dict:
        """
        Execute inference for a given stock and model.

        Returns:
            dict with prediction, confidence interval, and last known price
        """
        cache_key = f"predict:{stock}:{model_name}"

        # 1. Check Redis cache first
        cached = self.cache.get(cache_key)
        if cached:
            logger.info(f"Cache HIT for {cache_key}")
            return cached

        logger.info(f"Cache MISS — running inference for {stock} | model={model_name}")

        # 2. Fetch latest 18 months of data (enough for 60-day window + indicators)
        df_raw = self.data_service.fetch_stock_data(stock, period="18mo")
        last_price = float(df_raw["Close"].iloc[-1])
        last_date = str(df_raw.index[-1].date())

        # 3. Feature engineering (do NOT fit scaler — use pre-fitted)
        df_features = self.feature_service.add_indicators(df_raw)

        # 4. Create sequences using pre-fitted scaler
        #    Use fit_scaler=True here since inference pipeline manages its own scaler
        X, _, scaler = self.feature_service.create_sequences(
            df_features,
            sequence_length=SEQUENCE_LENGTH,
            fit_scaler=True,
        )

        # Use only the last window for next-step prediction
        X_last = X[-1:].reshape(1, SEQUENCE_LENGTH, -1)

        # 5. Load model
        model = self._get_or_load_model(stock, model_name)

        # 6. Predict
        pred_scaled = model.predict(X_last)
        # Inverse transform
        n_features = X_last.shape[2]
        pred_price = self.feature_service.inverse_transform_predictions(
            pred_scaled, n_features
        )[0]

        # 7. Simple confidence interval (±1 rolling std)
        recent_std = float(df_raw["Close"].tail(20).std())
        confidence = {
            "lower": round(pred_price - recent_std, 2),
            "upper": round(pred_price + recent_std, 2),
        }

        result = {
            "stock": stock,
            "model": model_name,
            "last_price": round(last_price, 2),
            "predicted_price": round(float(pred_price), 2),
            "confidence_interval": confidence,
            "direction": "UP" if pred_price > last_price else "DOWN",
            "change_pct": round((pred_price - last_price) / last_price * 100, 2),
            "as_of_date": last_date,
        }

        # 8. Cache result for 1 hour (3600s)
        self.cache.set(cache_key, result, ttl=3600)
        return result

    def _pull_model_from_s3(self, stock: str, model_name: str, dest_dir: str) -> bool:
        """Attempt to pull model artifacts from S3."""
        bucket_name = os.environ.get("S3_MODEL_BUCKET")
        if not bucket_name:
            return False
            
        try:
            import boto3
            s3 = boto3.client("s3")
            prefix = f"models/{stock}/{model_name}/"
            
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            if "Contents" not in response:
                return False
                
            os.makedirs(dest_dir, exist_ok=True)
            for obj in response["Contents"]:
                key = obj["Key"]
                if key.endswith('/'):
                    continue
                rel_path = key[len(prefix):]
                local_file_path = os.path.join(dest_dir, rel_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                logger.info(f"Downloading s3://{bucket_name}/{key} to {local_file_path}")
                s3.download_file(bucket_name, key, local_file_path)
            return True
        except Exception as e:
            logger.warning(f"Failed to pull model {stock}/{model_name} from S3: {e}")
            return False

    def _get_or_load_model(self, stock: str, model_name: str):
        """Load model from disk or S3, with in-process caching."""
        key = f"{stock}_{model_name}"
        if key in self._loaded_models:
            logger.info(f"Model '{key}' found in process cache")
            return self._loaded_models[key]

        # Find latest version in model store
        model_dir = os.path.join(MODEL_STORE_DIR, stock, model_name)
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            pulled = self._pull_model_from_s3(stock, model_name, model_dir)
            if not pulled:
                raise FileNotFoundError(
                    f"No trained model found locally or in S3 for {stock}/{model_name}. "
                    "Please run training pipeline and upload to S3."
                )

        versions = sorted(os.listdir(model_dir), reverse=True)
        if not versions:
            raise FileNotFoundError(f"No model versions in {model_dir}")

        latest = os.path.join(model_dir, versions[0])
        logger.info(f"Loading model from {latest}")

        model = self.model_factory.get_model(model_name)
        model.load(latest)
        self._loaded_models[key] = model
        return model

    def invalidate_cache(self, stock: str, model_name: str):
        """Force-expire the Redis cache for this stock/model pair."""
        cache_key = f"predict:{stock}:{model_name}"
        self.cache.delete(cache_key)
        # Also remove from process cache so model reloads
        self._loaded_models.pop(f"{stock}_{model_name}", None)
