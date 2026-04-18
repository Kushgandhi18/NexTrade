"""
train.py
POST /train — triggers model training in the background.
GET  /train/status — returns training job status.
"""

import logging
import uuid
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from backend.services.training_pipeline import TrainingPipeline
from backend.models.model_factory import SUPPORTED_MODELS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/train", tags=["Training"])

# In-memory job status store (replace with Redis/DB for production)
_jobs: dict = {}


class TrainRequest(BaseModel):
    stock: str
    model: str = "auto"
    sequence_length: int = 60
    hyperparams: Optional[dict] = None


def _run_training(job_id: str, request: TrainRequest):
    """Background training task."""
    _jobs[job_id]["status"] = "running"
    try:
        pipeline = TrainingPipeline()
        result = pipeline.run(
            stock=request.stock.upper(),
            model_name=request.model,
            sequence_length=request.sequence_length,
            hyperparams=request.hyperparams,
        )
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = result
        logger.info(f"Training job {job_id} completed for {request.stock}")
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
        logger.error(f"Training job {job_id} failed: {e}")


@router.post("")
async def trigger_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger model training in the background.
    Returns a job_id to poll for status.
    
    model: 'lstm', 'gru', 'arima_svm', 'transformer', 'ensemble', or 'auto'
    'auto' uses RegimeDetector to select the best model automatically.
    """
    if request.model not in SUPPORTED_MODELS + ["auto"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Supported: {SUPPORTED_MODELS + ['auto']}",
        )

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "stock": request.stock.upper(), "model": request.model}

    background_tasks.add_task(_run_training, job_id, request)

    return {
        "job_id": job_id,
        "message": f"Training started for {request.stock.upper()} using model '{request.model}'",
        "poll_url": f"/train/status/{job_id}",
    }


@router.get("/status/{job_id}")
async def get_training_status(job_id: str):
    """Poll the status of a training job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _jobs[job_id]


@router.get("/jobs")
async def list_jobs():
    """List all training jobs and their statuses."""
    return _jobs
