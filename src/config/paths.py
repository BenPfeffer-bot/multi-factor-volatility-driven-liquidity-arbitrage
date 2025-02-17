"""
Path configuration for data and outputs.

This module defines all paths used in the project for data storage and outputs.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Data directories
DATA_DIR = PROJECT_ROOT / "db"

# Raw data
RAW_DATA = DATA_DIR / "raw"
RAW_DAILY = RAW_DATA / "daily"
RAW_INTRADAY = RAW_DATA / "intraday"
RAW_OPTIONS = RAW_DATA / "options"

# Processed data
PROCESSED_DATA = DATA_DIR / "processed"
PROCESSED_DAILY = PROCESSED_DATA / "daily"
PROCESSED_INTRADAY = PROCESSED_DATA / "intraday"
PROCESSED_FEATURES = PROCESSED_DATA / "features"
PROCESSED_OPTIONS = PROCESSED_DATA / "options"

# Cache
CACHE_DIR = DATA_DIR / "cache"
FEDERAL_FUNDS_CACHE = CACHE_DIR / "federal_funds"
MARKET_DATA_CACHE = CACHE_DIR / "market_data"
NEWS_CACHE = CACHE_DIR / "news_sentiment"
TREASURY_YIELDS_CACHE = CACHE_DIR / "treasury_yields"
FEATURES_CACHE = CACHE_DIR / "features"
MODELS_CACHE = CACHE_DIR / "models"
CACHE_OPTIONS = CACHE_DIR / "options"


# History
HISTORY_DIR = DATA_DIR / "history"


# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "output"

# Models
MODELS_DIR = OUTPUTS_DIR / "models"
ML_MODELS = MODELS_DIR / "ml"
RL_MODELS = MODELS_DIR / "rl"

# Analysis
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
VOLATILITY_STRATEGY_ANALYSIS = ANALYSIS_DIR / "volatility_strategy"
SIGNALS_DIR = ANALYSIS_DIR / "signals"
PLOTS_DIR_ANALYSIS = ANALYSIS_DIR / "plots"

# Backtest
BACKTEST_DIR = OUTPUTS_DIR / "backtest"
BACKTEST_RESULTS = BACKTEST_DIR / "results"
PERFORMANCE_DIR = BACKTEST_DIR / "performance"
TRADES_DIR = BACKTEST_DIR / "trades"
PLOTS_DIR = BACKTEST_DIR / "plots"

# Reports
REPORTS_DIR = OUTPUTS_DIR / "reports"
DAILY_REPORTS = REPORTS_DIR / "daily"
WEEKLY_REPORTS = REPORTS_DIR / "weekly"
MONTHLY_REPORTS = REPORTS_DIR / "monthly"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"

# Version control
VERSIONS_DIR = DATA_DIR / "versions"
METADATA_DIR = DATA_DIR / "metadata"

# Temporary storage
TEMP_DIR = DATA_DIR / "temp"

# Add these paths after the ML_MODELS definition
LSTM_MODELS = ML_MODELS / "lstm"
LSTM_CHECKPOINTS = LSTM_MODELS / "checkpoints"
LSTM_PREDICTIONS = ANALYSIS_DIR / "lstm_predictions"

# Create all directories
DIRS = [
    RAW_DAILY,
    RAW_INTRADAY,
    PROCESSED_DATA,
    PROCESSED_DAILY,
    PROCESSED_INTRADAY,
    PROCESSED_FEATURES,
    MARKET_DATA_CACHE,
    FEATURES_CACHE,
    MODELS_CACHE,
    HISTORY_DIR,
    ML_MODELS,
    RL_MODELS,
    VOLATILITY_STRATEGY_ANALYSIS,
    SIGNALS_DIR,
    PERFORMANCE_DIR,
    TRADES_DIR,
    PLOTS_DIR,
    DAILY_REPORTS,
    WEEKLY_REPORTS,
    MONTHLY_REPORTS,
    LOGS_DIR,
    PLOTS_DIR_ANALYSIS,
    VERSIONS_DIR,
    METADATA_DIR,
    TEMP_DIR,
    FEDERAL_FUNDS_CACHE,
    LSTM_MODELS,
    LSTM_CHECKPOINTS,
    LSTM_PREDICTIONS,
    NEWS_CACHE,
    TREASURY_YIELDS_CACHE,
    RAW_OPTIONS,
    BACKTEST_RESULTS,
    CACHE_OPTIONS,
]


def create_directories():
    """Create all required directories if they don't exist."""
    for directory in DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# Create directories when module is imported
create_directories()
