"""
Configuration Module

This module handles project-wide configuration settings.
"""

import pandas as pd
from .paths import *

# # Market data parameters
# MIN_HISTORY_DAYS = 252  # Minimum days of history required
# MAX_MISSING_PCTS = 0.1  # Maximum allowed percentage of missing data
# PRICE_DECIMAL_PLACES = 4  # Number of decimal places for price data

# Feature generation parameters
VOLATILITY_WINDOW = 20
CORRELATION_WINDOW = 50
MOMENTUM_WINDOWS = [5, 10, 20]
RSI_WINDOW = 14
TRANSFER_ENTROPY_WINDOW = 50

# Model parameters
CV_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Trading parameters
POSITION_SIZE = 0.1  # Maximum position size as fraction of portfolio
STOP_LOSS = 0.02  # Stop loss threshold
TAKE_PROFIT = 0.05  # Take profit threshold

# Dow Jones Global Titans 50 Index
DJ_TITANS_50_TICKER = [
    "MMM",  # 3M
    "ABBV",  # AbbVie
    # "ALIZY",  # Allianz
    "GOOG",  # Alphabet
    "AMZN",  # Amazon
    "AMGN",  # Amgen
    "BUD",  # Anheuser-Busch InBev
    "AAPL",  # Apple
    "BHP",  # BHP
    "BA",  # Boeing
    "BP",  # BP
    "BTI",  # British American Tobacco
    "CVX",  # Chevron
    "CSCO",  # Cisco
    "C",  # Citigroup
    "KO",  # Coca-Cola
    "DD",  # DuPont
    "XOM",  # ExxonMobil
    "META",  # Meta
    "GE",  # General Electric
    "GSK",  # GlaxoSmithKline
    "HSBC",  # HSBC
    "INTC",  # Intel
    "IBM",  # IBM
    "JNJ",  # Johnson & Johnson
    "JPM",  # JPMorgan Chase
    "MA",  # Mastercard
    "MCD",  # McDonald's
    "MRK",  # Merck
    "MSFT",  # Microsoft
    # "NSRGY",  # Nestlé
    "NVS",  # Novartis
    "NVDA",  # Nvidia
    "ORCL",  # Oracle
    "PEP",  # PepsiCo
    "PFE",  # Pfizer
    "PM",  # Philip Morris
    "PG",  # Procter & Gamble
    # "RHHBY",  # Roche
    "RY",  # Royal Bank of Canada
    "SHEL",  # Shell
    # "SSNLF",  # Samsung
    "SOFR",  # Amplify Samsung SOFR ETF
    "SNY",  # Sanofi
    # "SIEGY",  # Siemens
    "TSM",  # TSMC
    "TTE",  # TotalEnergies
    "V",  # Visa
    "TM",  # Toyota
    "WMT",  # Walmart
    "DIS",  # Disney
    "VIXM",  # VIX Index
    "VIXY",  # VIX Index
]



# Error handling
class DataError(Exception):
    """Base class for data-related errors."""

    pass


class InsufficientDataError(DataError):
    """Raised when there is insufficient historical data."""

    pass


class DataQualityError(DataError):
    """Raised when data quality does not meet requirements."""

    pass


# def validate_market_data(data: pd.DataFrame) -> None:
#     """
#     Validate market data meets requirements.

#     Args:
#         data: DataFrame of market data

#     Raises:
#         InsufficientDataError: If insufficient history
#         DataQualityError: If data quality issues detected
#     """
#     if len(data) < MIN_HISTORY_DAYS:
#         raise InsufficientDataError(
#             f"Insufficient history: {len(data)} days < {MIN_HISTORY_DAYS} required"
#         )

#     missing_pcts = data.isnull().mean()
#     if (missing_pcts > MAX_MISSING_PCTS).any():
#         bad_cols = missing_pcts[missing_pcts > MAX_MISSING_PCTS].index.tolist()
#         raise DataQualityError(f"Excessive missing data in columns: {bad_cols}")


# # Intraday Data Settings
# INTRADAY_SETTINGS = {
#     "default_interval": "5min",
#     "cache_expiry_days": 1,  # How long to keep cached data
#     "batch_size": 5,  # Number of parallel requests
#     "retries": 3,  # Number of retry attempts
#     "retry_delay": 5,  # Seconds between retries
# }

# Data Management Settings
DATA_MANAGEMENT = {
    "cleanup_threshold_days": 30,
    "version_retention": 3,
    "compression": "snappy",
    "cache_expiry_hours": 24,
}

# Logging Settings
LOGGING = {
    "console_level": "INFO",
    "file_level": "DEBUG",
    "rotation_size_mb": 10,
    "backup_count": 5,
}

# Add data validation parameters (uncomment and update existing ones)
MIN_HISTORY_DAYS = 252
MAX_MISSING_PCTS = 0.1
PRICE_DECIMAL_PLACES = 4
