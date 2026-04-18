"""
model_factory.py
Returns the correct model instance by name.
Integrates with RegimeDetector for smart model selection.
"""

import logging
from typing import Optional

from backend.models.base_model import BaseModel

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ["lstm", "gru", "arima_svm", "transformer", "ensemble"]


class ModelFactory:
    """Factory that returns model instances by name."""

    def get_model(self, name: str, **kwargs) -> BaseModel:
        """
        Return a fresh (untrained) instance of the requested model.
        
        Args:
            name: one of 'lstm', 'gru', 'arima_svm', 'transformer', 'ensemble'
            **kwargs: passed to the model constructor for hyperparameter override
        """
        name = name.lower().strip()

        if name == "lstm":
            from backend.models.lstm_model import LSTMModel
            return LSTMModel(**kwargs)

        elif name == "gru":
            from backend.models.gru_model import GRUModel
            return GRUModel(**kwargs)

        elif name in ("arima_svm", "arima-svm", "arima"):
            from backend.models.arima_svm import ArimaSVMModel
            return ArimaSVMModel(**kwargs)

        elif name == "transformer":
            from backend.models.transformer_model import TransformerModel
            return TransformerModel(**kwargs)

        elif name == "ensemble":
            from backend.models.ensemble_model import EnsembleModel
            from backend.models.gru_model import GRUModel
            from backend.models.lstm_model import LSTMModel
            from backend.models.arima_svm import ArimaSVMModel
            from backend.models.transformer_model import TransformerModel

            sub_models = {
                "gru": GRUModel(),
                "lstm": LSTMModel(),
                "arima_svm": ArimaSVMModel(),
                "transformer": TransformerModel(),
            }
            return EnsembleModel(models=sub_models, **kwargs)

        else:
            raise ValueError(
                f"Unknown model: '{name}'. Supported: {SUPPORTED_MODELS}"
            )

    def select_by_regime(self, regime: str) -> str:
        """
        Map stock regime to best model name (from research paper insights).
        
        Regimes:
          'trending'  → GRU performs best
          'cyclical'  → LSTM performs best
          'volatile'  → Ensemble (hedges risk)
          'stable'    → ARIMA-SVM (linear trend sufficient)
        """
        mapping = {
            "trending": "gru",
            "cyclical": "lstm",
            "volatile": "ensemble",
            "stable": "arima_svm",
        }
        model_name = mapping.get(regime, "ensemble")
        logger.info(f"Regime '{regime}' → selected model: {model_name}")
        return model_name
