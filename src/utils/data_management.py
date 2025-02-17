"""
Data management utilities for handling data storage, versioning, and cleanup.
"""

import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, Any, Optional
import hashlib
import numpy as np

from src.config.paths import *
from src.utils.log_utils import setup_logging, log_execution_time
from src.config.validation import validate_market_data, validate_volatility_features
from src.config.validation import DataValidationError

logger = setup_logging(__name__)


class DataManager:
    def __init__(self):
        self.metadata_file = DATA_DIR / "metadata.json"
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load or create metadata tracking file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"datasets": {}, "last_cleanup": None}
            self._save_metadata()

    def _save_metadata(self) -> None:
        """Save metadata to file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=4)

    @log_execution_time
    def save_dataset(
        self,
        data: pd.DataFrame,
        name: str,
        category: str = "raw",
        version: Optional[str] = None,
    ) -> Path:
        """
        Save dataset with versioning and metadata tracking.

        Args:
            data: DataFrame to save
            name: Dataset name
            category: Data category (raw/processed)
            version: Optional version string
        """
        # Generate version if not provided
        if not version:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Calculate data hash
        data_hash = hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()

        # Determine save path
        base_dir = RAW_DATA if category == "raw" else PROCESSED_DATA
        save_path = base_dir / f"{name}_{version}.parquet"

        # Save data
        data.to_parquet(save_path)

        # Update metadata
        self.metadata["datasets"][str(save_path)] = {
            "name": name,
            "category": category,
            "version": version,
            "created": datetime.now().isoformat(),
            "hash": data_hash,
            "rows": len(data),
            "columns": list(data.columns),
        }
        self._save_metadata()

        logger.info(f"Saved dataset {name} version {version} to {save_path}")
        return save_path

    @log_execution_time
    def cleanup_old_data(self, days_threshold: int = 30) -> None:
        """Remove old data files based on threshold."""
        current_time = datetime.now()
        deleted_count = 0

        for filepath, meta in list(self.metadata["datasets"].items()):
            created = datetime.fromisoformat(meta["created"])
            if (current_time - created).days > days_threshold:
                Path(filepath).unlink(missing_ok=True)
                del self.metadata["datasets"][filepath]
                deleted_count += 1

        self.metadata["last_cleanup"] = current_time.isoformat()
        self._save_metadata()

        logger.info(f"Cleaned up {deleted_count} old data files")

    def prepare_lstm_features(
        self, ticker: str, start_date: str, end_date: str, sequence_length: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """Prepare features for LSTM volatility prediction."""

        # Load and merge all required data
        market_data = self._load_dataset(f"{ticker}_market", "processed")
        options_data = self._load_dataset(f"{ticker}_options", "processed")
        sentiment_data = self._load_dataset("market_sentiment", "processed")
        macro_data = self._load_dataset("macro_indicators", "processed")

        # Validate data quality
        validation = validate_market_data(market_data)
        if not validation.is_valid:
            raise DataValidationError(
                f"Market data validation failed: {validation.errors}"
            )

        # Create feature matrix
        features = pd.DataFrame(index=market_data.index)

        # Historical volatility features
        for window in [5, 10, 20, 30]:
            features[f"rv_{window}d"] = (
                market_data["realized_volatility"].rolling(window).mean()
            )

        # Options market features
        features["implied_vol"] = options_data["implied_volatility"]
        features["vol_spread"] = (
            options_data["implied_volatility"] - market_data["realized_volatility"]
        )
        features["put_call_ratio"] = (
            options_data["put_volume"] / options_data["call_volume"]
        )

        # Sentiment and macro features
        features["sentiment"] = sentiment_data["sentiment_ma5"]
        features["vstoxx"] = macro_data["vstoxx"]
        features["bond_spread"] = macro_data["spread_10y"]

        # Validate final feature set
        validation = validate_volatility_features(features)
        if not validation.is_valid:
            raise DataValidationError(f"Feature validation failed: {validation.errors}")

        # Create sequences for LSTM
        X, y = self._create_sequences(features, sequence_length)

        return {
            "features": features,
            "X": X,
            "y": y,
            "validation_warnings": validation.warnings,
        }

    def _create_sequences(
        self, data: pd.DataFrame, sequence_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []

        for i in range(len(data) - sequence_length):
            X.append(data.iloc[i : i + sequence_length].values)
            y.append(data.iloc[i + sequence_length]["realized_volatility"])

        return np.array(X), np.array(y)

    def load_market_data(
        self, ticker: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Load market data for a given ticker and date range.

        Args:
            ticker: The ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with market data or None if no data is available
        """
        try:
            # Check both possible file patterns
            patterns = [
                f"{ticker}_1m_*.csv",  # Pattern 1: TICKER_1m_DATE.csv
                f"intraday_{ticker}_*.csv",  # Pattern 2: intraday_TICKER_DATE.csv
            ]

            files = []
            for pattern in patterns:
                files.extend(list(PROCESSED_INTRADAY.glob(pattern)))

            if not files:
                logger.warning(
                    f"No data files found for {ticker} in {PROCESSED_INTRADAY}"
                    f"\nTried patterns: {patterns}"
                )
                return None

            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading data from {latest_file}")

            # Load the data
            data = pd.read_csv(latest_file)

            # Verify required columns
            required_columns = ["date", "close", "high", "low", "volume", "log_returns"]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Convert datetime column
            data["datetime"] = pd.to_datetime(data["date"])

            # Convert to UTC if not already
            if data["datetime"].dt.tz is not None:
                data["datetime"] = data["datetime"].dt.tz_convert("UTC")
            else:
                data["datetime"] = data["datetime"].dt.tz_localize("UTC")

            # Set datetime as index
            data.set_index("datetime", inplace=True)

            # Calculate returns if not present
            if "returns" not in data.columns:
                data["returns"] = data["close"].pct_change()

            # Map volatility columns to standard names
            vol_mapping = {
                "returns_vol": "realized_volatility",
                "parkinson_vol": "parkinson_volatility",
                "garman_klass_vol": "garman_klass_volatility",
                "yang_zhang_vol": "yang_zhang_volatility",
            }

            for old_name, new_name in vol_mapping.items():
                if old_name in data.columns:
                    data[new_name] = data[old_name]

            # Set main realized_volatility if not present
            if (
                "realized_volatility" not in data.columns
                and "returns_vol" in data.columns
            ):
                data["realized_volatility"] = data["returns_vol"]

            # Calculate rolling volatilities if not present
            if "realized_volatility" not in data.columns:
                data["realized_volatility"] = data["returns"].rolling(
                    window=30, min_periods=15
                ).std() * np.sqrt(252)

            # Convert start and end dates to UTC timestamps
            start_ts = pd.Timestamp(start_date).tz_localize("UTC")
            end_ts = pd.Timestamp(end_date).tz_localize("UTC")

            # Filter date range
            mask = (data.index >= start_ts) & (data.index <= end_ts)
            data = data[mask].copy()

            if len(data) == 0:
                logger.warning(
                    f"No data available for {ticker} between {start_date} and {end_date}"
                )
                return None

            # Fill NaN values
            data = data.fillna(method="ffill").fillna(method="bfill")

            return data

        except Exception as e:
            logger.error(f"Error loading market data for {ticker}: {str(e)}")
            return None

    def load_iv_data(
        self, ticker: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Load implied volatility data for a given ticker and date range.

        Args:
            ticker: The ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with IV data or None if no data is available
        """
        try:
            # Find the most recent IV data file
            files = list(Path("db/raw/iv").glob(f"{ticker}_iv_*.csv"))
            if not files:
                logger.warning(f"No IV data files found for {ticker}")
                return None
            latest_file = max(files)

            # Load the data
            data = pd.read_csv(latest_file)

            # Convert datetime with proper timezone handling
            data["datetime"] = pd.to_datetime(data["datetime"], utc=True)

            # Get the actual date range in the data
            min_date = data["datetime"].min()
            max_date = data["datetime"].max()
            logger.info(f"IV data for {ticker} ranges from {min_date} to {max_date}")

            # Convert start and end dates to UTC timestamps
            start_ts = pd.Timestamp(start_date).tz_localize("UTC")
            end_ts = pd.Timestamp(end_date).tz_localize("UTC")

            # If the requested date range is completely outside the available data, return None
            if pd.Timestamp(end_date).tz_localize("UTC") < min_date.tz_localize(
                "UTC"
            ) or pd.Timestamp(start_date).tz_localize("UTC") > max_date.tz_localize(
                "UTC"
            ):
                logger.warning(
                    f"Requested date range ({start_date} to {end_date}) is outside available IV data range for {ticker}"
                )
                return None

            # Convert all timestamps to UTC for consistent comparison
            data["datetime"] = data["datetime"].dt.tz_convert("UTC")
            start_ts = pd.Timestamp(start_date).tz_localize("UTC")
            end_ts = pd.Timestamp(end_date).tz_localize("UTC")

            # Filter the data to the requested date range
            mask = (data["datetime"] >= start_ts) & (data["datetime"] <= end_ts)
            data = data[mask].copy()

            if len(data) == 0:
                logger.warning(
                    f"No IV data available for {ticker} between {start_date} and {end_date}"
                )
                return None

            # Set datetime as index
            data.set_index("datetime", inplace=True)

            return data

        except Exception as e:
            logger.error(f"Error loading IV data for {ticker}: {str(e)}")
            return None

    def load_options_data(
        self, ticker: str, start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Load options data for a given ticker and date range.

        Args:
            ticker: The ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dictionary containing options data components or None if no data available
        """
        options_data = {}

        # Load options chain
        chain_pattern = f"{ticker}_options_chain_latest_*.parquet"
        chain_files = list(RAW_OPTIONS.glob(chain_pattern))

        if not chain_files:
            logger.warning(f"No options chain files found for {ticker}")
            return None

        try:
            latest_chain = max(chain_files, key=lambda x: x.stat().st_mtime)
            chain = pd.read_parquet(latest_chain)

            # Convert datetime columns to UTC
            for col in ["expiration", "lastTradeDate", "timestamp", "quoteDate"]:
                if col in chain.columns:
                    chain[col] = pd.to_datetime(chain[col])
                    if chain[col].dt.tz is not None:
                        chain[col] = chain[col].dt.tz_convert("UTC")
                    else:
                        chain[col] = chain[col].dt.tz_localize("UTC")

            # Filter by date range
            if "lastTradeDate" in chain.columns:
                start_ts = pd.Timestamp(start_date).tz_localize("UTC")
                end_ts = pd.Timestamp(end_date).tz_localize("UTC")
                chain = chain[
                    (chain["lastTradeDate"] >= start_ts)
                    & (chain["lastTradeDate"] <= end_ts)
                ]

            if not chain.empty:
                options_data["chain"] = chain

            # Load IV surface
            surface_pattern = f"{ticker}_iv_surface_latest_*.parquet"
            surface_files = list(PROCESSED_OPTIONS.glob(surface_pattern))

            if surface_files:
                latest_surface = max(surface_files, key=lambda x: x.stat().st_mtime)
                surface = pd.read_parquet(latest_surface)

                # Convert timestamp to UTC if present
                if "timestamp" in surface.columns:
                    surface["timestamp"] = pd.to_datetime(surface["timestamp"])
                    if surface["timestamp"].dt.tz is not None:
                        surface["timestamp"] = surface["timestamp"].dt.tz_convert("UTC")
                    else:
                        surface["timestamp"] = surface["timestamp"].dt.tz_localize(
                            "UTC"
                        )

                if not surface.empty:
                    options_data["surface"] = surface

            # Load options metrics
            metrics_pattern = f"{ticker}_options_metrics_latest_*.parquet"
            metrics_files = list(PROCESSED_OPTIONS.glob(metrics_pattern))

            if metrics_files:
                latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)
                metrics = pd.read_parquet(latest_metrics)

                # Convert timestamp to UTC if present
                if "timestamp" in metrics.columns:
                    metrics["timestamp"] = pd.to_datetime(metrics["timestamp"])
                    if metrics["timestamp"].dt.tz is not None:
                        metrics["timestamp"] = metrics["timestamp"].dt.tz_convert("UTC")
                    else:
                        metrics["timestamp"] = metrics["timestamp"].dt.tz_localize(
                            "UTC"
                        )

                if not metrics.empty:
                    options_data["metrics"] = metrics

            return options_data if options_data else None

        except Exception as e:
            logger.error(f"Error loading options data for {ticker}: {str(e)}")
            return None

    def load_sentiment_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load market sentiment data for a given date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with sentiment data
        """
        # Find latest sentiment file
        sentiment_pattern = "market_sentiment_*.csv"
        sentiment_files = list(RAW_DATA.glob(sentiment_pattern))

        if not sentiment_files:
            logger.warning("No sentiment data files found")
            return pd.DataFrame()

        latest_file = max(sentiment_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading sentiment data from {latest_file}")

        try:
            # Read and process data
            data = pd.read_csv(latest_file)

            # Convert timestamp to UTC timezone
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            if data["timestamp"].dt.tz is not None:
                data["timestamp"] = data["timestamp"].dt.tz_convert("UTC")
            else:
                data["timestamp"] = data["timestamp"].dt.tz_localize("UTC")

            data = data.set_index("timestamp").sort_index()

            # Convert start and end dates to UTC timestamps
            start_ts = pd.Timestamp(start_date).tz_localize("UTC")
            end_ts = pd.Timestamp(end_date).tz_localize("UTC")

            # Filter date range
            mask = (data.index >= start_ts) & (data.index <= end_ts)
            data = data[mask].copy()

            # Ensure required columns exist and fill missing values
            required_columns = ["sentiment_score", "sentiment_ma5", "sentiment_std5"]
            for col in required_columns:
                if col not in data.columns:
                    data[col] = 0.0
                else:
                    # Fill missing values with forward fill and backward fill
                    data[col] = data[col].fillna(method="ffill").fillna(method="bfill")

            return data

        except Exception as e:
            logger.error(f"Error loading sentiment data: {str(e)}")
            return pd.DataFrame()

    def load_macro_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load macro economic data for a given date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with macro data
        """
        macro_data = pd.DataFrame()

        # Load VSTOXX data
        vstoxx_pattern = "vstoxx_*.csv"
        vstoxx_files = list(RAW_DATA.glob(vstoxx_pattern))

        if vstoxx_files:
            latest_file = max(vstoxx_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading VSTOXX data from {latest_file}")

            try:
                vstoxx = pd.read_csv(latest_file)
                date_col = "Date" if "Date" in vstoxx.columns else "date"

                if date_col in vstoxx.columns:
                    # Convert to datetime with UTC timezone
                    vstoxx[date_col] = pd.to_datetime(vstoxx[date_col])
                    if vstoxx[date_col].dt.tz is not None:
                        vstoxx[date_col] = vstoxx[date_col].dt.tz_convert("UTC")
                    else:
                        vstoxx[date_col] = vstoxx[date_col].dt.tz_localize("UTC")

                    vstoxx = vstoxx.set_index(date_col).sort_index()

                    # Convert start and end dates to UTC timestamps
                    start_ts = pd.Timestamp(start_date).tz_localize("UTC")
                    end_ts = pd.Timestamp(end_date).tz_localize("UTC")

                    # Filter date range
                    mask = (vstoxx.index >= start_ts) & (vstoxx.index <= end_ts)
                    vstoxx = vstoxx[mask].copy()

                    # Add to macro_data
                    macro_data = vstoxx

            except Exception as e:
                logger.error(f"Error loading VSTOXX data: {str(e)}")

        # Load bond spreads (if needed)
        spreads_pattern = "bond_spreads_*.csv"
        spreads_files = list(RAW_DATA.glob(spreads_pattern))

        if spreads_files:
            latest_file = max(spreads_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading bond spreads from {latest_file}")

            try:
                spreads = pd.read_csv(latest_file)
                date_col = "Date" if "Date" in spreads.columns else "date"

                if date_col in spreads.columns:
                    # Convert to datetime with UTC timezone
                    spreads[date_col] = pd.to_datetime(spreads[date_col])
                    if spreads[date_col].dt.tz is not None:
                        spreads[date_col] = spreads[date_col].dt.tz_convert("UTC")
                    else:
                        spreads[date_col] = spreads[date_col].dt.tz_localize("UTC")

                    spreads = spreads.set_index(date_col).sort_index()

                    # Convert start and end dates to UTC timestamps
                    start_ts = pd.Timestamp(start_date).tz_localize("UTC")
                    end_ts = pd.Timestamp(end_date).tz_localize("UTC")

                    # Filter date range
                    mask = (spreads.index >= start_ts) & (spreads.index <= end_ts)
                    spreads = spreads[mask].copy()

                    # Add bond spread to macro_data
                    if not macro_data.empty:
                        macro_data["bond_spread"] = spreads["spread_10y"]
                    else:
                        macro_data = spreads

            except Exception as e:
                logger.error(f"Error loading bond spreads: {str(e)}")

        return macro_data

    def load_intraday_data(
        self, ticker: str, start_date: str, end_date: str, interval: str = "1min"
    ) -> Optional[pd.DataFrame]:
        """Load intraday market data for a given ticker and date range.

        Args:
            ticker: The ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (e.g., 1min, 5min, 15min)

        Returns:
            DataFrame with intraday market data or None if no data is available
        """
        try:
            # Find the most recent 1-minute data file
            files = list(Path("db/processed/intraday").glob(f"intraday_{ticker}_*.csv"))
            if not files:
                logger.warning(f"No intraday data files found for {ticker}")
                return None
            latest_file = max(files)

            # Load the data
            data = pd.read_csv(latest_file)

            # Check if the column "datetime" exists
            # otherwise we can add a step to check if 'date' exists and then convert to datetime
            if "datetime" not in data.columns:
                if "date" in data.columns:
                    data["datetime"] = pd.to_datetime(data["date"])
                else:
                    logger.warning(f"No datetime or date column found in {latest_file}")
                    return None
            # Convert datetime with proper timezone handling
            data["datetime"] = pd.to_datetime(data["datetime"], utc=True)

            # Get the actual date range in the data
            min_date = data["datetime"].min()
            max_date = data["datetime"].max()
            logger.info(
                f"Intraday data for {ticker} ranges from {min_date} to {max_date}"
            )

            # Convert start and end dates to UTC timestamps
            start_ts = pd.Timestamp(start_date).tz_localize("UTC")
            end_ts = pd.Timestamp(end_date).tz_localize("UTC")

            # Calculate basic features
            data["returns"] = data["close"].pct_change()
            data["log_returns"] = np.log(data["close"] / data["close"].shift(1))

            # Calculate realized volatility using different windows
            for window in [5, 10, 20, 30]:
                # Standard realized volatility
                data[f"realized_volatility_{window}"] = data["returns"].rolling(
                    window=window, min_periods=window // 2
                ).std() * np.sqrt(252)

                # Log-return based volatility
                data[f"log_volatility_{window}"] = data["log_returns"].rolling(
                    window=window, min_periods=window // 2
                ).std() * np.sqrt(252)

                # Parkinson volatility (using high-low range)
                data[f"parkinson_volatility_{window}"] = (
                    np.sqrt(1 / (4 * np.log(2)))
                    * (np.log(data["high"] / data["low"]))
                    .rolling(window=window, min_periods=window // 2)
                    .std()
                    * np.sqrt(252)
                )

                # Garman-Klass volatility
                data[f"garman_klass_volatility_{window}"] = np.sqrt(
                    0.5 * np.log(data["high"] / data["low"]) ** 2
                    - (2 * np.log(2) - 1) * np.log(data["close"] / data["open"]) ** 2
                ).rolling(window=window, min_periods=window // 2).mean() * np.sqrt(252)

            # Set the main realized_volatility column (30-day window)
            data["realized_volatility"] = data["realized_volatility_30"]

            # Volume-based features
            data["volume_ma5"] = data["volume"].rolling(window=5).mean()
            data["volume_ma20"] = data["volume"].rolling(window=20).mean()
            data["relative_volume"] = data["volume"] / data["volume_ma20"]

            # Price-based features
            data["price_ma5"] = data["close"].rolling(window=5).mean()
            data["price_ma20"] = data["close"].rolling(window=20).mean()
            data["price_ma50"] = data["close"].rolling(window=50).mean()

            # Technical indicators
            data["rsi"] = self._calculate_rsi(data["close"])
            data["atr"] = self._calculate_atr(data[["high", "low", "close"]])

            # Momentum features
            for window in [5, 10, 20]:
                data[f"momentum_{window}"] = data["close"].pct_change(window)

            # Volatility of volatility
            data["vol_of_vol"] = data["realized_volatility"].rolling(window=20).std()

            # Statistical moments
            data["returns_skew"] = data["returns"].rolling(window=20).skew()
            data["returns_kurt"] = data["returns"].rolling(window=20).kurt()

            # If the requested date range is completely outside the available data, return None
            if end_ts < min_date or start_ts > max_date:
                logger.warning(
                    f"Requested date range ({start_date} to {end_date}) is outside available data range for {ticker}"
                )
                return None

            # Filter the data to the requested date range
            mask = (data["datetime"] >= start_ts) & (data["datetime"] <= end_ts)
            data = data[mask].copy()

            if len(data) == 0:
                logger.warning(
                    f"No data available for {ticker} between {start_date} and {end_date}"
                )
                return None

            # Set datetime as index
            data.set_index("datetime", inplace=True)

            # Resample to requested interval if different from 1min
            if interval != "1min":
                # Define aggregation rules for each column
                agg_dict = {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "returns": "sum",
                    "log_returns": "sum",
                    "realized_volatility": "last",
                    "yang_zhang_vol": "last",
                    "rsi": "last",
                    "volume_ma5": "last",
                    "volume_ma20": "last",
                    "relative_volume": "last",
                    "momentum_5": "last",
                    "momentum_10": "last",
                    "momentum_20": "last",
                }

                # Resample with specified aggregation rules
                data = data.resample(interval).agg(agg_dict).dropna()

            # Calculate additional intraday features
            data = self._calculate_intraday_features(data, interval)

            return data

        except Exception as e:
            logger.error(f"Error loading intraday data for {ticker}: {str(e)}")
            return None

    def _calculate_intraday_features(
        self, data: pd.DataFrame, interval: str
    ) -> pd.DataFrame:
        """Calculate additional features specific to intraday data.

        Args:
            data: DataFrame with raw intraday data
            interval: Data interval

        Returns:
            DataFrame with additional features
        """
        try:
            # Calculate returns if not already present
            if "returns" not in data.columns:
                data["returns"] = data["close"].pct_change()
                data["log_returns"] = np.log(data["close"] / data["close"].shift(1))

            # Calculate intraday volatility
            interval_minutes = int(interval.replace("min", ""))
            periods_per_day = 390 // interval_minutes  # 6.5 hours trading day

            # Realized volatility (rolling window)
            for window in [5, 10, 20, 30]:
                # Standard realized volatility
                data[f"realized_volatility_{window}"] = data["returns"].rolling(
                    window=window * periods_per_day,
                    min_periods=window * periods_per_day // 2,
                ).std() * np.sqrt(252)

                # Log-return based volatility
                data[f"log_volatility_{window}"] = data["log_returns"].rolling(
                    window=window * periods_per_day,
                    min_periods=window * periods_per_day // 2,
                ).std() * np.sqrt(252)

                # Parkinson volatility
                data[f"parkinson_volatility_{window}"] = (
                    np.sqrt(1 / (4 * np.log(2)))
                    * (np.log(data["high"] / data["low"]))
                    .rolling(
                        window=window * periods_per_day,
                        min_periods=window * periods_per_day // 2,
                    )
                    .std()
                    * np.sqrt(252)
                )

                # Garman-Klass volatility
                data[f"garman_klass_volatility_{window}"] = np.sqrt(
                    0.5 * np.log(data["high"] / data["low"]) ** 2
                    - (2 * np.log(2) - 1) * np.log(data["close"] / data["open"]) ** 2
                ).rolling(
                    window=window * periods_per_day,
                    min_periods=window * periods_per_day // 2,
                ).mean() * np.sqrt(252)

            # Set main realized_volatility column
            data["realized_volatility"] = data["realized_volatility_30"]

            # Volume profile
            data["volume_ma5"] = data["volume"].rolling(5 * periods_per_day).mean()
            data["volume_ma20"] = data["volume"].rolling(20 * periods_per_day).mean()
            data["relative_volume"] = data["volume"] / data["volume_ma20"]

            # Price momentum and moving averages
            for window in [5, 10, 20]:
                data[f"momentum_{window}"] = data["close"].pct_change(
                    window * periods_per_day
                )
                data[f"price_ma_{window}"] = (
                    data["close"].rolling(window * periods_per_day).mean()
                )

            # Technical indicators
            data["rsi"] = self._calculate_rsi(data["close"])
            data["atr"] = self._calculate_atr(data[["high", "low", "close"]])

            # Volatility of volatility
            data["vol_of_vol"] = data["realized_volatility"].rolling(
                20 * periods_per_day
            ).std() * np.sqrt(252)

            # Statistical moments
            data["returns_skew"] = data["returns"].rolling(20 * periods_per_day).skew()
            data["returns_kurt"] = data["returns"].rolling(20 * periods_per_day).kurt()

            # Microstructure features
            if all(col in data.columns for col in ["bid", "ask"]):
                data["bid_ask_spread"] = (data["ask"] - data["bid"]) / (
                    (data["ask"] + data["bid"]) / 2
                )
                data["mid_price"] = (data["bid"] + data["ask"]) / 2
                data["price_impact"] = abs(data["returns"]) / (data["volume"] + 1e-8)
                data["effective_spread"] = (
                    2 * abs(data["close"] - data["mid_price"]) / data["mid_price"]
                )

            # Intraday seasonality
            data["hour"] = data.index.hour
            data["minute"] = data.index.minute
            data["time_from_open"] = (
                data.index.hour * 60 + data.index.minute - 540
            )  # Minutes from 9:00
            data["time_to_close"] = 1020 - (
                data.index.hour * 60 + data.index.minute
            )  # Minutes to 17:00

            return data

        except Exception as e:
            logger.error(f"Error calculating intraday features: {str(e)}")
            return data  # Return original data if feature calculation fails

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index) for a given series of prices."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Neutral RSI for missing values

    def _calculate_atr(
        self, high_low_close: pd.DataFrame, period: int = 14
    ) -> pd.Series:
        """Calculate ATR (Average True Range) for a given DataFrame of high, low, and close prices."""
        high = high_low_close["high"]
        low = high_low_close["low"]
        close = high_low_close["close"]

        # Calculate True Range
        tr1 = high - low  # Current high - current low
        tr2 = abs(high - close.shift())  # Current high - previous close
        tr3 = abs(low - close.shift())  # Current low - previous close
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        return atr.fillna(method="bfill")  # Backfill missing values
