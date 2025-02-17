"""
Utility functions for path resolution and management
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
from src.config.paths import *


def get_data_path(
    name: str,
    category: str = "raw",
    date: Optional[datetime] = None,
    create: bool = True,
) -> Path:
    """Resolve data file path based on category and date."""
    if date is None:
        date = datetime.now()

    date_str = date.strftime("%Y%m%d")

    if category == "raw":
        base_dir = RAW_DAILY
    elif category == "processed":
        base_dir = PROCESSED_DAILY
    elif category == "features":
        base_dir = PROCESSED_FEATURES
    else:
        raise ValueError(f"Invalid category: {category}")

    path = base_dir / f"{name}_{date_str}.parquet"

    if create:
        path.parent.mkdir(parents=True, exist_ok=True)

    return path


def get_model_path(
    model_type: str, model_name: str, version: Optional[str] = None
) -> Path:
    """Resolve model file path."""
    if model_type == "lstm":
        base_dir = LSTM_MODELS
    elif model_type == "hmm":
        base_dir = ML_MODELS / "hmm"
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")

    return base_dir / f"{model_name}_{version}"
