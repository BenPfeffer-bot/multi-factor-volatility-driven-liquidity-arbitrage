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
RAW_INTRADAY = RAW_DATA / "intraday"
RAW_OPTIONS = RAW_DATA / "options"
RAW_MACRO = RAW_DATA / "macro"
RAW_NEWS = RAW_DATA / "news"

# Processed data
PROCESSED_DATA = DATA_DIR / "processed"
PROCESSED_INTRADAY = PROCESSED_DATA / "intraday"
PROCESSED_OPTIONS = PROCESSED_DATA / "options"
PROCESSED_MACRO = PROCESSED_DATA / "macro"
PROCESSED_NEWS = PROCESSED_DATA / "news"

# Cache
CACHE_DIR = DATA_DIR / "cache"
FEATURES_CACHE = CACHE_DIR / "features"
MODELS_CACHE = CACHE_DIR / "models"

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


# Create all directories
DIRS = [
    RAW_DATA,
    RAW_INTRADAY,
    RAW_OPTIONS,
    RAW_MACRO,
    RAW_NEWS,
    PROCESSED_DATA,
    PROCESSED_INTRADAY,
    PROCESSED_OPTIONS,
    PROCESSED_MACRO,
    PROCESSED_NEWS,
    CACHE_DIR,
    FEATURES_CACHE,
    MODELS_CACHE,
    HISTORY_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
    ML_MODELS,
    RL_MODELS,
    ANALYSIS_DIR,
    VOLATILITY_STRATEGY_ANALYSIS,
    SIGNALS_DIR,
    PLOTS_DIR_ANALYSIS,
    BACKTEST_DIR,
    BACKTEST_RESULTS,
    PERFORMANCE_DIR,
    TRADES_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    DAILY_REPORTS,
    WEEKLY_REPORTS,
    MONTHLY_REPORTS,
    LOGS_DIR,
    VERSIONS_DIR,
    METADATA_DIR,
    TEMP_DIR,
]


def create_directories():
    """Create all required directories if they don't exist."""
    for directory in DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# Create directories when module is imported
create_directories()
