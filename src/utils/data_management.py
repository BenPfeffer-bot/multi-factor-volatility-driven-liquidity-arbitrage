"""
Data management utilities for handling market data, sentiment, and insider transactions.
Provides versioning, validation, and cleanup functionality.
"""

import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, Any, Optional, Union, List, Tuple
import hashlib
import numpy as np

from src.config.paths import *
from src.utils.log_utils import setup_logging, log_execution_time
from src.config.validation import validate_market_data, validate_volatility_features
from src.config.validation import DataValidationError

logger = setup_logging(__name__)

class MarketDataManager:
    """Manages market data, sentiment data, and insider transactions."""

    def __init__(self):
        self.metadata_file = DATA_DIR / "metadata.json"
        self.data_types = {
            'intraday': RAW_INTRADAY,  # intraday data -> {ticker}_1min.csv
            'options': RAW_OPTIONS,  # options data -> {ticker}_options.csv
            'treasury_yields': RAW_MACRO / "treasury_yields",  # treasury yields data for different maturities
            'federal_funds': RAW_MACRO / "federal_funds",  # federal funds data (daily & monthly)
            'bond_spreads': RAW_MACRO,  # bond spreads data -> bond_spreads_{date}.csv
            'ecb_rates': RAW_MACRO,  # ECB rates data -> ecb_rates_{date}.csv
            'cpi': RAW_MACRO / "cpi",  # Consumer Price Index data
            'inflation': RAW_MACRO / "inflation",  # Inflation rate data
            'unemployment': RAW_MACRO / "unemployment",  # Unemployment rate data
            'sentiment': RAW_NEWS / "sentiment",  # sentiment data -> news_{ticker}_{date}.csv
            'insider': RAW_NEWS / "insider"  # insider data -> {ticker}_insider_transactions_{date}.csv
        }
        
        # Define timeframes for different data types
        self.data_timeframes = {
            'intraday': {'period': '1Y', 'interval': '1min'},
            'options': {'period': '1Y', 'interval': '1D'},
            'treasury_yields': {'period': '5Y', 'interval': '1D'},
            'federal_funds': {'period': '5Y', 'interval': '1D'},
            'bond_spreads': {'period': '30D', 'interval': '1D'},
            'ecb_rates': {'period': '2Y', 'interval': '1D'},
            'cpi': {'period': '5Y', 'interval': '1M'},
            'inflation': {'period': '5Y', 'interval': '1M'},
            'unemployment': {'period': '5Y', 'interval': '1M'}
        }
        
        # Define available treasury yield maturities
        self.treasury_maturities = ['3m', '2y', '5y', '7y', '10y', '30y']
        
        # Define bond spread metrics
        self.bond_spread_metrics = [
            'DE10', 'DE02', 'FR10', 'FR02', 'IT10', 'IT02', 'ES10', 'ES02', 'PT10', 'GR10',
            'IT10_spread', 'IT_CDS', 'ES10_spread', 'ES_CDS', 'FR10_spread', 'FR_CDS',
            'PT10_spread', 'PT_CDS', 'GR10_spread', 'GR_CDS', 'IT02_spread', 'ES02_spread',
            'FR02_spread', 'DE_term_spread', 'IT_term_spread', 'ES_term_spread', 'FR_term_spread',
            'peripheral_spread', 'peripheral_cds', 'spread_volatility'
        ]
        
        # Add data quality thresholds
        self.quality_thresholds = {
            'intraday': {
                'min_rows': 100,  # Minimum rows for valid intraday data
                'max_gap_minutes': 30,  # Maximum allowed gap in minutes
                'min_volume': 1000  # Minimum average daily volume
            },
            'options': {
                'min_strikes': 10,  # Minimum number of strikes per expiration
                'min_expirations': 4  # Minimum number of expiration dates
            },
            'macro': {
                'max_missing_pct': 0.1,  # Maximum allowed percentage of missing values
                'max_gap_days': 5  # Maximum allowed gap in days
            }
        }
        
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load or create metadata tracking file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "datasets": {},
                "last_cleanup": None,
                "data_versions": {
                    data_type: {} for data_type in self.data_types.keys()
                }
            }
            self._save_metadata()

    def _save_metadata(self) -> None:
        """Save metadata to file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=4)

    def _generate_version(self, data_type: str, symbol: str) -> str:
        """Generate version string for a dataset."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{data_type}_{symbol}_{timestamp}"

    @log_execution_time
    def save_market_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        data_type: str,
        interval: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Path:
        """
        Save market data with versioning and metadata tracking.

        Args:
            data: DataFrame to save
            symbol: Stock symbol
            data_type: Type of data (intraday/options/macro/sentiment/insider)
            interval: Data interval (for intraday/macro data)
            version: Optional version string
        """
        if data_type not in self.data_types:
            raise ValueError(f"Invalid data type. Must be one of {list(self.data_types.keys())}")

        # Generate version if not provided
        if not version:
            version = self._generate_version(data_type, symbol)

        # Calculate data hash
        data_hash = hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()

        # Determine save path
        base_dir = self.data_types[data_type]
        filename = f"{symbol}_{version}.parquet"
        if interval:
            filename = f"{symbol}_{interval}_{version}.parquet"
        save_path = base_dir / filename

        # Save data
        data.to_parquet(save_path)

        # Update metadata
        metadata_entry = {
            "symbol": symbol,
            "data_type": data_type,
            "interval": interval,
            "version": version,
            "created": datetime.now().isoformat(),
            "hash": data_hash,
            "rows": len(data),
            "columns": list(data.columns),
            "file_path": str(save_path)
        }

        self.metadata["datasets"][str(save_path)] = metadata_entry
        self.metadata["data_versions"][data_type][symbol] = version
        self._save_metadata()

        logger.info(f"Saved {data_type} data for {symbol} version {version} to {save_path}")
        return save_path

    def get_latest_version(self, data_type: str, symbol: str) -> Optional[str]:
        """Get the latest version for a symbol's data type."""
        return self.metadata["data_versions"][data_type].get(symbol)

    def load_latest_data(
        self,
        symbol: str,
        data_type: str,
        interval: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Load the latest version of data for a symbol."""
        version = self.get_latest_version(data_type, symbol)
        if not version:
            logger.warning(f"No data found for {symbol} {data_type}")
            return None

        base_dir = self.data_types[data_type]
        pattern = f"{symbol}_{interval}_{version}.parquet" if interval else f"{symbol}_{version}.parquet"
        file_path = base_dir / pattern

        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}")
            return None

        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            return None

    def get_data_info(self, symbol: str, data_type: str) -> List[Dict[str, Any]]:
        """Get information about all versions of a symbol's data."""
        info = []
        for metadata in self.metadata["datasets"].values():
            if metadata["symbol"] == symbol and metadata["data_type"] == data_type:
                info.append(metadata)
        return info

    @log_execution_time
    def cleanup_old_data(
        self,
        days_threshold: int = 30,
        keep_latest: bool = True
    ) -> None:
        """
        Remove old data files based on threshold.
        
        Args:
            days_threshold: Number of days after which to remove data
            keep_latest: Whether to keep the latest version regardless of age
        """
        current_time = datetime.now()
        deleted_count = 0
        latest_versions = {
            data_type: {
                symbol: version
                for symbol, version in versions.items()
            }
            for data_type, versions in self.metadata["data_versions"].items()
        }

        for filepath, meta in list(self.metadata["datasets"].items()):
            created = datetime.fromisoformat(meta["created"])
            is_latest = (
                meta["version"] == 
                latest_versions.get(meta["data_type"], {}).get(meta["symbol"])
            )

            if (current_time - created).days > days_threshold and not (keep_latest and is_latest):
                Path(filepath).unlink(missing_ok=True)
                del self.metadata["datasets"][filepath]
                deleted_count += 1

        self.metadata["last_cleanup"] = current_time.isoformat()
        self._save_metadata()

        logger.info(f"Cleaned up {deleted_count} old data files")

    def validate_data(
        self,
        data: pd.DataFrame,
        data_type: str,
        **kwargs
    ) -> bool:
        """
        Enhanced data validation for all data types.
        
        Args:
            data: DataFrame to validate
            data_type: Type of data
            **kwargs: Additional validation parameters
            
        Returns:
            bool: Whether the data is valid
        """
        try:
            if data_type == 'treasury_yields':
                required_columns = ['date', 'value']
                if not all(col in data.columns for col in required_columns):
                    logger.error(f"Missing required columns for {data_type}: {required_columns}")
                    return False
                
                # Check for missing values
                if data[required_columns].isnull().any().any():
                    logger.error(f"Found missing values in {data_type} data")
                    return False
                
                return True
                
            elif data_type == 'intraday':
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in data.columns for col in required_columns):
                    logger.error(f"Missing required columns for {data_type}: {required_columns}")
                    return False
                
                # Check for missing values
                if data[required_columns].isnull().any().any():
                    logger.error(f"Found missing values in {data_type} data")
                    return False
                
                # Check for negative prices
                if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
                    logger.error(f"Found negative or zero prices in {data_type} data")
                    return False
                
                # Check OHLC relationship
                if not (
                    (data['high'] >= data['low']).all() and
                    (data['high'] >= data['open']).all() and
                    (data['high'] >= data['close']).all() and
                    (data['low'] <= data['open']).all() and
                    (data['low'] <= data['close']).all()
                ):
                    logger.error("OHLC relationship violated")
                    return False
                
                return True
                
            elif data_type == 'options':
                required_columns = [
                    'date', 'expiry_date', 'strike', 'type',
                    'implied_volatility', 'underlying_price'
                ]
                if not all(col in data.columns for col in required_columns):
                    logger.error(f"Missing required columns for {data_type}: {required_columns}")
                    return False
                
                # Check for missing values
                if data[required_columns].isnull().any().any():
                    logger.error(f"Found missing values in {data_type} data")
                    return False
                
                # Check option types
                if not data['type'].isin(['call', 'put']).all():
                    logger.error("Invalid option types found")
                    return False
                
                return True
                
            elif data_type == 'sentiment':
                required_columns = ['sentiment_score', 'mention_count']
                if not all(col in data.columns for col in required_columns):
                    logger.error(f"Missing required columns for {data_type}: {required_columns}")
                    return False
                
                # Check for missing values
                if data[required_columns].isnull().any().any():
                    logger.error(f"Found missing values in {data_type} data")
                    return False
                
                # Check sentiment score range
                if not (data['sentiment_score'].between(-1, 1).all()):
                    logger.error("Sentiment scores outside valid range [-1, 1]")
                    return False
                
                return True
            
            else:
                logger.error(f"Unknown data type: {data_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error validating {data_type} data: {str(e)}")
            return False

    def get_data_timeframe(self, data_type: str) -> Dict[str, str]:
        """
        Get the default timeframe for a data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            Dict with period and interval
        """
        return self.data_timeframes.get(data_type, {'period': '1Y', 'interval': '1D'})

    def load_treasury_yields(
        self,
        maturity: str,
        start_date: str,
        end_date: str,
        interval: str = 'daily'
    ) -> Optional[pd.DataFrame]:
        """
        Load treasury yield data for a specific maturity.
        
        Args:
            maturity: Yield maturity (3m, 2y, 5y, 7y, 10y, 30y)
            start_date: Start date
            end_date: End date
            interval: Data interval (daily, weekly, monthly)
            
        Returns:
            DataFrame with treasury yield data
        """
        if maturity not in self.treasury_maturities:
            logger.error(f"Invalid treasury maturity: {maturity}")
            return None
            
        try:
            file_path = self.data_types['treasury_yields'] / f"treasury_yield_{maturity}_{interval}.csv"
            if not file_path.exists():
                logger.warning(f"Treasury yield file not found: {file_path}")
                return None
                
            data = pd.read_csv(file_path)
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            
            # Apply date filtering
            data = data[
                (data.index >= pd.to_datetime(start_date)) &
                (data.index <= pd.to_datetime(end_date))
            ]
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading treasury yield data for {maturity}: {str(e)}")
            return None

    def load_bond_spreads(
        self,
        start_date: str,
        end_date: str,
        metrics: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load bond spread data.
        
        Args:
            start_date: Start date
            end_date: End date
            metrics: List of specific spread metrics to load
            
        Returns:
            DataFrame with bond spread data
        """
        try:
            # Get latest bond spreads file
            bond_files = list(self.data_types['bond_spreads'].glob("bond_spreads_*.csv"))
            if not bond_files:
                logger.warning("No bond spread files found")
                return None
                
            latest_file = max(bond_files, key=lambda x: x.stem.split('_')[-1])
            
            data = pd.read_csv(latest_file)
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            
            # Filter by date
            data = data[
                (data.index >= pd.to_datetime(start_date)) &
                (data.index <= pd.to_datetime(end_date))
            ]
            
            # Filter specific metrics if requested
            if metrics:
                valid_metrics = [m for m in metrics if m in self.bond_spread_metrics]
                if not valid_metrics:
                    logger.warning("No valid metrics specified")
                    return None
                data = data[valid_metrics]
                
            return data
            
        except Exception as e:
            logger.error(f"Error loading bond spread data: {str(e)}")
            return None

    def load_ecb_rates(
        self,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Load ECB rates data.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with ECB rates
        """
        try:
            # Get latest ECB rates file
            ecb_files = list(self.data_types['ecb_rates'].glob("ecb_rates_*.csv"))
            if not ecb_files:
                logger.warning("No ECB rates files found")
                return None
                
            latest_file = max(ecb_files, key=lambda x: x.stem.split('_')[-1])
            
            data = pd.read_csv(latest_file)
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            
            # Filter by date
            data = data[
                (data.index >= pd.to_datetime(start_date)) &
                (data.index <= pd.to_datetime(end_date))
            ]
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading ECB rates data: {str(e)}")
            return None

    def validate_data_quality(
        self,
        data: pd.DataFrame,
        data_type: str,
        **kwargs
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate data quality beyond basic structure checks.
        
        Args:
            data: DataFrame to validate
            data_type: Type of data
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (is_valid, quality_metrics)
        """
        quality_metrics = {}
        
        try:
            if data_type == 'intraday':
                # Check number of rows
                quality_metrics['row_count'] = len(data)
                if quality_metrics['row_count'] < self.quality_thresholds['intraday']['min_rows']:
                    logger.warning(f"Insufficient data rows: {quality_metrics['row_count']}")
                    return False, quality_metrics
                
                # Check for gaps
                data.index = pd.to_datetime(data.index)
                gaps = data.index.to_series().diff().dt.total_seconds() / 60
                max_gap = gaps.max()
                quality_metrics['max_gap_minutes'] = max_gap
                if max_gap > self.quality_thresholds['intraday']['max_gap_minutes']:
                    logger.warning(f"Large time gap detected: {max_gap} minutes")
                    return False, quality_metrics
                
                # Check volume
                avg_volume = data['volume'].mean()
                quality_metrics['avg_volume'] = avg_volume
                if avg_volume < self.quality_thresholds['intraday']['min_volume']:
                    logger.warning(f"Low average volume: {avg_volume}")
                    return False, quality_metrics
                
            elif data_type == 'options':
                # Check strike coverage
                strikes_per_exp = data.groupby('expiration')['strike'].nunique()
                min_strikes = strikes_per_exp.min()
                quality_metrics['min_strikes_per_exp'] = min_strikes
                if min_strikes < self.quality_thresholds['options']['min_strikes']:
                    logger.warning(f"Insufficient strike coverage: {min_strikes} strikes")
                    return False, quality_metrics
                
                # Check expiration dates
                quality_metrics['expiration_count'] = data['expiration'].nunique()
                if quality_metrics['expiration_count'] < self.quality_thresholds['options']['min_expirations']:
                    logger.warning(f"Insufficient expiration dates: {quality_metrics['expiration_count']}")
                    return False, quality_metrics
                
            elif data_type in ['treasury_yields', 'federal_funds', 'bond_spreads', 'ecb_rates', 'cpi', 'inflation', 'unemployment']:
                # Check for missing values
                missing_pct = data.isnull().mean()
                quality_metrics['missing_pct'] = missing_pct.max()
                if quality_metrics['missing_pct'] > self.quality_thresholds['macro']['max_missing_pct']:
                    logger.warning(f"High missing value percentage: {quality_metrics['missing_pct']}")
                    return False, quality_metrics
                
                # Check for gaps in time series
                if 'date' in data.columns:
                    data.index = pd.to_datetime(data['date'])
                gaps = data.index.to_series().diff().dt.days
                max_gap = gaps.max()
                quality_metrics['max_gap_days'] = max_gap
                if max_gap > self.quality_thresholds['macro']['max_gap_days']:
                    logger.warning(f"Large time gap detected: {max_gap} days")
                    return False, quality_metrics
            
            return True, quality_metrics
            
        except Exception as e:
            logger.error(f"Error in data quality validation: {str(e)}")
            return False, quality_metrics

    def load_macro_data(
        self,
        data_type: str,
        interval: str = 'daily',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Enhanced macro data loader supporting all macro data types.
        
        Args:
            data_type: Type of macro data
            interval: Data interval
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments for specific data types
            
        Returns:
            DataFrame with macro data
        """
        try:
            if data_type == 'treasury_yields':
                maturity = kwargs.get('maturity', '10y')
                return self.load_treasury_yields(maturity, start_date, end_date, interval)
                
            elif data_type == 'bond_spreads':
                metrics = kwargs.get('metrics')
                return self.load_bond_spreads(start_date, end_date, metrics)
                
            elif data_type == 'ecb_rates':
                return self.load_ecb_rates(start_date, end_date)
                
            elif data_type in ['federal_funds', 'cpi', 'inflation', 'unemployment']:
                file_path = self.data_types[data_type] / f"{data_type}_{interval}.csv"
                
            else:
                logger.error(f"Invalid macro data type: {data_type}")
                return None
                
            if not file_path.exists():
                logger.warning(f"Data file not found: {file_path}")
                return None
                
            data = pd.read_csv(file_path)
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            
            # Apply date filtering
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]
                
            # Validate data quality
            is_valid, quality_metrics = self.validate_data_quality(data, data_type)
            if not is_valid:
                logger.warning(f"Data quality validation failed for {data_type}: {quality_metrics}")
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"Error loading {data_type} data: {str(e)}")
            return None

    def load_intraday_data(
        self,
        symbol: str,
        interval: str = "1min",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load intraday market data for a symbol with date filtering.
        
        Args:
            symbol: Stock symbol
            interval: Time interval (1min, 5min, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data if available
        """
        try:
            data = self.load_latest_data(symbol, "intraday", interval)
            if data is not None:
                # Ensure timestamp is datetime
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
                
                # Apply date filtering if specified
                if start_date:
                    data = data[data.index >= pd.to_datetime(start_date)]
                if end_date:
                    data = data[data.index <= pd.to_datetime(end_date)]
                
                # Convert numeric columns
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                
                return data.sort_index()
            return None
        except Exception as e:
            logger.error(f"Error loading intraday data for {symbol}: {str(e)}")
            return None

    def load_options_data(
        self,
        symbol: str,
        expiration_date: Optional[str] = None,
        option_type: Optional[str] = None,
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load options data with filtering capabilities.
        
        Args:
            symbol: Stock symbol
            expiration_date: Specific expiration date (YYYY-MM-DD)
            option_type: 'call' or 'put'
            min_strike: Minimum strike price
            max_strike: Maximum strike price
            
        Returns:
            Filtered options DataFrame
        """
        try:
            data = self.load_latest_data(symbol, "options")
            if data is not None:
                # Convert dates to datetime
                data['expiration'] = pd.to_datetime(data['expiration'])
                data['date'] = pd.to_datetime(data['date'])
                
                # Convert numeric columns
                numeric_cols = [
                    'strike', 'last', 'mark', 'bid', 'ask', 'volume',
                    'open_interest', 'implied_volatility', 'delta', 'gamma',
                    'theta', 'vega', 'rho'
                ]
                for col in numeric_cols:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Apply filters
                if expiration_date:
                    data = data[data['expiration'] == pd.to_datetime(expiration_date)]
                if option_type:
                    data = data[data['type'].str.lower() == option_type.lower()]
                if min_strike is not None:
                    data = data[data['strike'] >= min_strike]
                if max_strike is not None:
                    data = data[data['strike'] <= max_strike]
                
                return data.sort_values(['expiration', 'strike'])
            return None
        except Exception as e:
            logger.error(f"Error loading options data for {symbol}: {str(e)}")
            return None

    def load_sentiment_data(
        self,
        symbol: str,
        min_sentiment_score: Optional[float] = None,
        min_relevance_score: Optional[float] = None,
        sentiment_label: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load sentiment data with enhanced filtering.
        
        Args:
            symbol: Stock symbol
            min_sentiment_score: Minimum sentiment score
            min_relevance_score: Minimum relevance score
            sentiment_label: Specific sentiment label to filter for
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Filtered sentiment DataFrame
        """
        try:
            data = self.load_latest_data(symbol, "sentiment")
            if data is not None:
                # Convert time_published to datetime
                data['time_published'] = pd.to_datetime(data['time_published'], format='%Y%m%dT%H%M%S')
                
                # Convert numeric columns
                numeric_cols = ['overall_sentiment_score', 'relevance_score']
                ticker_cols = [f"{symbol}_relevance_score", f"{symbol}_sentiment_score"]
                
                for col in numeric_cols + ticker_cols:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Apply filters
                if start_date:
                    data = data[data['time_published'] >= pd.to_datetime(start_date)]
                if end_date:
                    data = data[data['time_published'] <= pd.to_datetime(end_date)]
                if min_sentiment_score is not None:
                    score_col = f"{symbol}_sentiment_score" if f"{symbol}_sentiment_score" in data.columns else 'overall_sentiment_score'
                    data = data[data[score_col] >= min_sentiment_score]
                if min_relevance_score is not None:
                    rel_col = f"{symbol}_relevance_score" if f"{symbol}_relevance_score" in data.columns else 'relevance_score'
                    data = data[data[rel_col] >= min_relevance_score]
                if sentiment_label:
                    label_col = f"{symbol}_sentiment_label" if f"{symbol}_sentiment_label" in data.columns else 'overall_sentiment_label'
                    data = data[data[label_col] == sentiment_label]
                
                return data.sort_values('time_published', ascending=False)
            return None
        except Exception as e:
            logger.error(f"Error loading sentiment data for {symbol}: {str(e)}")
            return None

    def load_insider_data(
        self,
        symbol: str,
        transaction_type: Optional[str] = None,
        min_value: Optional[float] = None,
        insider_roles: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_shares: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load and filter insider transaction data with enhanced capabilities.
        
        Args:
            symbol: Stock symbol
            transaction_type: Type of transaction (A for acquisition, D for disposal)
            min_value: Minimum transaction value in dollars
            insider_roles: List of insider roles to filter for (e.g., ['CEO', 'CFO', 'Director'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_shares: Minimum number of shares
            
        Returns:
            Filtered insider transactions DataFrame
        """
        try:
            data = self.load_latest_data(symbol, "insider")
            if data is not None:
                # Convert dates to datetime
                for date_col in ['transaction_date', 'filing_date']:
                    if date_col in data.columns:
                        data[date_col] = pd.to_datetime(data[date_col])
                
                # Convert numeric columns
                data['shares'] = pd.to_numeric(data['shares'], errors='coerce')
                data['share_price'] = pd.to_numeric(data['share_price'], errors='coerce')
                data['transaction_value'] = data['shares'] * data['share_price']
                
                # Apply filters
                if start_date:
                    data = data[data['transaction_date'] >= pd.to_datetime(start_date)]
                if end_date:
                    data = data[data['transaction_date'] <= pd.to_datetime(end_date)]
                if transaction_type:
                    data = data[data['acquisition_or_disposal'] == transaction_type]
                if min_shares is not None:
                    data = data[data['shares'] >= min_shares]
                if min_value is not None:
                    data = data[data['transaction_value'] >= min_value]
                if insider_roles:
                    role_pattern = '|'.join(insider_roles)
                    data = data[data['executive_title'].str.contains(role_pattern, case=False, na=False)]
                
                # Add derived features
                data['days_since_filing'] = (pd.Timestamp.now() - data['filing_date']).dt.days
                data['filing_delay'] = (data['filing_date'] - data['transaction_date']).dt.days
                
                # Sort by transaction date and value
                return data.sort_values(['transaction_date', 'transaction_value'], ascending=[False, False])
            return None
        except Exception as e:
            logger.error(f"Error loading insider data for {symbol}: {str(e)}")
            return None

    def sync_intraday_with_options(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        option_moneyness_range: Tuple[float, float] = (0.8, 1.2)
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Synchronize intraday price data with options data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            option_moneyness_range: Range of strike/spot to include (min, max)
            
        Returns:
            Dictionary containing synchronized intraday and options DataFrames
        """
        try:
            # Load intraday data
            intraday = self.load_intraday_data(symbol, start_date, end_date)
            if intraday is None:
                return None
            
            # Load options data
            options = self.load_options_data(symbol)
            if options is None:
                return None
            
            # Ensure datetime format
            intraday.index = pd.to_datetime(intraday.index)
            options['quote_datetime'] = pd.to_datetime(options['quote_datetime'])
            
            # Calculate moneyness for options
            spot_price = intraday['close'].resample('1min').ffill()
            options['moneyness'] = options['strike'] / spot_price
            
            # Filter options by moneyness
            min_moneyness, max_moneyness = option_moneyness_range
            options = options[
                (options['moneyness'] >= min_moneyness) & 
                (options['moneyness'] <= max_moneyness)
            ]
            
            # Align timestamps
            options.set_index('quote_datetime', inplace=True)
            options = options.sort_index()
            
            return {
                'intraday': intraday,
                'options': options
            }
        except Exception as e:
            logger.error(f"Error synchronizing intraday and options data for {symbol}: {str(e)}")
            return None

    def align_sentiment_with_price(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sentiment_window: str = '1H'
    ) -> Optional[pd.DataFrame]:
        """
        Align sentiment data with price movements.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            sentiment_window: Resampling window for sentiment aggregation
            
        Returns:
            DataFrame with aligned price and sentiment data
        """
        try:
            # Load price data
            price_data = self.load_intraday_data(symbol, start_date, end_date)
            if price_data is None:
                return None
            
            # Load sentiment data
            sentiment_data = self.load_latest_data(symbol, "sentiment")
            if sentiment_data is None:
                return None
            
            # Ensure datetime format
            sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])
            sentiment_data.set_index('timestamp', inplace=True)
            
            # Aggregate sentiment metrics
            sentiment_agg = sentiment_data.resample(sentiment_window).agg({
                'sentiment_score': 'mean',
                'positive_mentions': 'sum',
                'negative_mentions': 'sum',
                'neutral_mentions': 'sum'
            }).fillna(method='ffill')
            
            # Calculate rolling sentiment metrics
            sentiment_agg['sentiment_ma'] = sentiment_agg['sentiment_score'].rolling(window=3).mean()
            sentiment_agg['sentiment_std'] = sentiment_agg['sentiment_score'].rolling(window=3).std()
            
            # Merge with price data
            aligned_data = pd.merge_asof(
                price_data,
                sentiment_agg,
                left_index=True,
                right_index=True,
                direction='backward'
            )
            
            return aligned_data
        except Exception as e:
            logger.error(f"Error aligning sentiment with price data for {symbol}: {str(e)}")
            return None

    def correlate_insider_with_market(
        self,
        symbol: str,
        lookback_days: int = 30,
        lookforward_days: int = 10,
        min_transaction_value: float = 100000
    ) -> Optional[pd.DataFrame]:
        """
        Correlate insider transactions with price/volume data.
        
        Args:
            symbol: Stock symbol
            lookback_days: Days to look back before transaction
            lookforward_days: Days to look forward after transaction
            min_transaction_value: Minimum transaction value to consider
            
        Returns:
            DataFrame with insider transactions and corresponding market data
        """
        try:
            # Load insider transactions
            insider_data = self.load_insider_data(
                symbol,
                min_value=min_transaction_value
            )
            if insider_data is None:
                return None
            
            # Load price/volume data
            price_data = self.load_intraday_data(symbol)
            if price_data is None:
                return None
            
            results = []
            for _, transaction in insider_data.iterrows():
                trans_date = pd.to_datetime(transaction['transaction_date'])
                
                # Get price data around transaction
                start_date = trans_date - pd.Timedelta(days=lookback_days)
                end_date = trans_date + pd.Timedelta(days=lookforward_days)
                
                period_data = price_data[
                    (price_data.index >= start_date) &
                    (price_data.index <= end_date)
                ]
                
                if not period_data.empty:
                    # Calculate market metrics
                    pre_price = period_data[:trans_date]['close'].mean()
                    post_price = period_data[trans_date:]['close'].mean()
                    price_change = (post_price - pre_price) / pre_price
                    
                    pre_volume = period_data[:trans_date]['volume'].mean()
                    post_volume = period_data[trans_date:]['volume'].mean()
                    volume_change = (post_volume - pre_volume) / pre_volume
                    
                    results.append({
                        'transaction_date': trans_date,
                        'insider': transaction['insider_name'],
                        'role': transaction['executive_title'],
                        'transaction_type': transaction['acquisition_or_disposal'],
                        'transaction_value': transaction['transaction_value'],
                        'price_change_pct': price_change * 100,
                        'volume_change_pct': volume_change * 100,
                        'pre_price': pre_price,
                        'post_price': post_price,
                        'pre_volume': pre_volume,
                        'post_volume': post_volume
                    })
            
            return pd.DataFrame(results)
        except Exception as e:
            logger.error(f"Error correlating insider with market data for {symbol}: {str(e)}")
            return None

    def validate_data_completeness(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate data completeness across all data sources.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary containing validation results for each data source
        """
        validation_results = {}
        
        try:
            # Validate intraday data
            intraday = self.load_intraday_data(symbol, start_date, end_date)
            if intraday is not None:
                missing_intervals = self._check_time_gaps(intraday, '1min')
                validation_results['intraday'] = {
                    'status': 'available' if not intraday.empty else 'empty',
                    'rows': len(intraday),
                    'missing_intervals': missing_intervals,
                    'data_quality': self._check_data_quality(intraday)
                }
            
            # Validate options data
            options = self.load_options_data(symbol)
            if options is not None:
                validation_results['options'] = {
                    'status': 'available' if not options.empty else 'empty',
                    'rows': len(options),
                    'unique_strikes': options['strike'].nunique() if 'strike' in options.columns else 0,
                    'data_quality': self._check_data_quality(options)
                }
            
            # Validate sentiment data
            sentiment = self.load_latest_data(symbol, "sentiment")
            if sentiment is not None:
                validation_results['sentiment'] = {
                    'status': 'available' if not sentiment.empty else 'empty',
                    'rows': len(sentiment),
                    'data_quality': self._check_data_quality(sentiment)
                }
            
            # Validate insider data
            insider = self.load_latest_data(symbol, "insider")
            if insider is not None:
                validation_results['insider'] = {
                    'status': 'available' if not insider.empty else 'empty',
                    'rows': len(insider),
                    'data_quality': self._check_data_quality(insider)
                }
            
            return validation_results
        except Exception as e:
            logger.error(f"Error validating data completeness for {symbol}: {str(e)}")
            return {'error': str(e)}

    def _check_time_gaps(
        self,
        data: pd.DataFrame,
        expected_freq: str
    ) -> List[Dict[str, Any]]:
        """
        Check for gaps in time series data.
        
        Args:
            data: DataFrame with datetime index
            expected_freq: Expected frequency of data
            
        Returns:
            List of dictionaries containing gap information
        """
        if data.empty:
            return []
        
        # Ensure index is datetime
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        
        # Create expected time range
        full_range = pd.date_range(
            start=data.index.min(),
            end=data.index.max(),
            freq=expected_freq
        )
        
        # Find missing times
        missing_times = full_range.difference(data.index)
        
        # Group consecutive missing times
        gaps = []
        if len(missing_times) > 0:
            gap_start = missing_times[0]
            prev_time = missing_times[0]
            
            for time in missing_times[1:]:
                if time != prev_time + pd.Timedelta(expected_freq):
                    gaps.append({
                        'start': gap_start,
                        'end': prev_time,
                        'duration': (prev_time - gap_start).total_seconds() / 60  # minutes
                    })
                    gap_start = time
                prev_time = time
            
            # Add last gap
            gaps.append({
                'start': gap_start,
                'end': prev_time,
                'duration': (prev_time - gap_start).total_seconds() / 60
            })
        
        return gaps

    def _check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform quality checks on DataFrame.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Dictionary containing quality metrics
        """
        quality_metrics = {
            'null_counts': data.isnull().sum().to_dict(),
            'null_percentage': (data.isnull().sum() / len(data) * 100).to_dict()
        }
        
        # Check for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Calculate basic statistics
            quality_metrics['numeric_stats'] = {
                col: {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'zeros_count': (data[col] == 0).sum(),
                    'zeros_percentage': (data[col] == 0).sum() / len(data) * 100
                }
                for col in numeric_cols
            }
            
            # Check for anomalies (values outside 3 standard deviations)
            for col in numeric_cols:
                mean = data[col].mean()
                std = data[col].std()
                anomalies = data[col][(data[col] - mean).abs() > 3 * std]
                quality_metrics['numeric_stats'][col]['anomalies_count'] = len(anomalies)
                quality_metrics['numeric_stats'][col]['anomalies_percentage'] = len(anomalies) / len(data) * 100
        
        # Check for duplicate rows
        duplicates = data.duplicated()
        quality_metrics['duplicates'] = {
            'count': duplicates.sum(),
            'percentage': duplicates.sum() / len(data) * 100
        }
        
        return quality_metrics

    def detect_anomalies(
        self,
        symbol: str,
        lookback_window: int = 30,
        std_threshold: float = 3.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect anomalies in price and volume data.
        
        Args:
            symbol: Stock symbol
            lookback_window: Window size for rolling calculations
            std_threshold: Number of standard deviations for anomaly detection
            
        Returns:
            Dictionary containing detected anomalies
        """
        try:
            price_data = self.load_intraday_data(symbol)
            if price_data is None:
                return {'error': 'No price data available'}
            
            anomalies = {
                'price': [],
                'volume': [],
                'volatility': []
            }
            
            # Price anomalies
            price_data['returns'] = price_data['close'].pct_change()
            price_data['rolling_mean'] = price_data['returns'].rolling(window=lookback_window).mean()
            price_data['rolling_std'] = price_data['returns'].rolling(window=lookback_window).std()
            
            price_anomalies = price_data[
                abs(price_data['returns'] - price_data['rolling_mean']) > 
                std_threshold * price_data['rolling_std']
            ]
            
            for idx, row in price_anomalies.iterrows():
                anomalies['price'].append({
                    'timestamp': idx,
                    'price': row['close'],
                    'return': row['returns'],
                    'std_dev': abs(row['returns'] - row['rolling_mean']) / row['rolling_std']
                })
            
            # Volume anomalies
            price_data['volume_ma'] = price_data['volume'].rolling(window=lookback_window).mean()
            price_data['volume_std'] = price_data['volume'].rolling(window=lookback_window).std()
            
            volume_anomalies = price_data[
                abs(price_data['volume'] - price_data['volume_ma']) > 
                std_threshold * price_data['volume_std']
            ]
            
            for idx, row in volume_anomalies.iterrows():
                anomalies['volume'].append({
                    'timestamp': idx,
                    'volume': row['volume'],
                    'std_dev': abs(row['volume'] - row['volume_ma']) / row['volume_std']
                })
            
            # Volatility anomalies
            price_data['volatility'] = price_data['returns'].rolling(window=lookback_window).std()
            price_data['volatility_ma'] = price_data['volatility'].rolling(window=lookback_window).mean()
            price_data['volatility_std'] = price_data['volatility'].rolling(window=lookback_window).std()
            
            volatility_anomalies = price_data[
                abs(price_data['volatility'] - price_data['volatility_ma']) > 
                std_threshold * price_data['volatility_std']
            ]
            
            for idx, row in volatility_anomalies.iterrows():
                anomalies['volatility'].append({
                    'timestamp': idx,
                    'volatility': row['volatility'],
                    'std_dev': abs(row['volatility'] - row['volatility_ma']) / row['volatility_std']
                })
            
            return anomalies
        except Exception as e:
            logger.error(f"Error detecting anomalies for {symbol}: {str(e)}")
            return {'error': str(e)}

# Initialize global data manager instance
market_data_manager = MarketDataManager()
    
