"""
Data validation utilities for market and volatility data.
"""

import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from src.config.settings import MIN_HISTORY_DAYS, MAX_MISSING_PCTS


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class DataValidationError(Exception):
    pass


def validate_market_data(data: pd.DataFrame) -> ValidationResult:
    """Validate market data quality and completeness."""
    errors = []
    warnings = []

    # Check history length
    if len(data) < MIN_HISTORY_DAYS:
        errors.append(f"Insufficient history: {len(data)} days < {MIN_HISTORY_DAYS}")

    # Check for missing values
    missing_pcts = data.isnull().mean()
    bad_cols = missing_pcts[missing_pcts > MAX_MISSING_PCTS].index.tolist()
    if bad_cols:
        errors.append(f"Excessive missing data in columns: {bad_cols}")

    # Check for price anomalies
    if "close" in data.columns:
        price_changes = data["close"].pct_change().abs()
        extreme_changes = price_changes[price_changes > 0.2]
        if not extreme_changes.empty:
            warnings.append(
                f"Large price changes detected on dates: {extreme_changes.index.tolist()}"
            )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_volatility_features(data: pd.DataFrame) -> ValidationResult:
    """Validate volatility features for LSTM input."""
    required_columns = [
        "realized_volatility",
        "implied_volatility",
        "vstoxx",
        "sentiment_score",
        "bond_spread",
    ]

    errors = []
    warnings = []

    # Check required columns
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check for sufficient non-null values
    if not errors:
        null_pcts = data[required_columns].isnull().mean()
        bad_cols = null_pcts[null_pcts > 0.1].index.tolist()
        if bad_cols:
            warnings.append(f"High null percentage in columns: {bad_cols}")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
