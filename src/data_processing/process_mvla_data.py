"""
Data processing script for the Multi-Factor Volatility-Driven Liquidity Arbitrage (MVLA) strategy.

This script processes raw data from multiple sources and calculates all required metrics:
1. Intraday volatility metrics
2. Liquidity metrics
3. Options market metrics
4. Macro indicators
5. Sentiment metrics

The processed data is saved in the processed folder with appropriate versioning.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import statsmodels.api as sm
from arch import arch_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.data_management import market_data_manager
from src.strategies.volatility_metrics import (
    calculate_multi_timeframe_rv,
    calculate_amihud_illiquidity,
    calculate_order_flow_toxicity,
    calculate_options_skew_features,
    calculate_yield_curve_features,
    calculate_sentiment_impact_features
)
from src.config.paths import (
    PROCESSED_INTRADAY,
    RAW_INTRADAY,
    RAW_OPTIONS,
    RAW_MACRO,
    RAW_NEWS,
    PROCESSED_OPTIONS,
    PROCESSED_MACRO,
    PROCESSED_NEWS
)
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)

class MVLADataProcessor:
    """Processes and prepares data for the MVLA strategy."""
    
    def __init__(self):
        # Set date range for market data and macro data (2024)
        self.market_start_date = '2024-02-01'
        self.market_end_date = '2024-02-16'
        
        # Set date for sentiment data (2025)
        self.sentiment_date = '20250217'
        
        # Set date for bond spreads and ECB rates (2025)
        self.macro_date = '20250216'
        
        # Initialize data directories
        self.RAW_INTRADAY = Path('db/raw/intraday')
        self.RAW_OPTIONS = Path('db/raw/options')
        self.RAW_NEWS = Path('db/raw/news')
        self.RAW_MACRO = Path('db/raw/macro')
        self.PROCESSED_INTRADAY = Path('db/processed/intraday')
        self.PROCESSED_OPTIONS = Path('db/processed/options')
        self.PROCESSED_NEWS = Path('db/processed/news')
        self.PROCESSED_MACRO = Path('db/processed/macro')
        
        # Initialize symbols list
        self.symbols = ['MMM', 'ABBV', 'ALIZY', 'AAPL', 'BHP', 'BA', 'BP', 'BTI', 'CVX', 'CSCO', 'C', 'KO', 
                        'DD', 'XOM', 'META', 'GE', 'GSK', 'HSBC', 'INTC', 'IBM', 'JNJ', 'MA', 'MCD', 'MSFT', 
                        'NSRGY', 'NVS', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PM', 'PG', 'RY', 'SHEL', 'SSNLF', 'SNY', 
                        'SIEGY', 'TSM', 'TTE', 'V', 'TM', 'WMT', 'DIS']
    
    def process_intraday_data(self, symbol):
        try:
            # Check if intraday data file exists
            intraday_file = self.RAW_INTRADAY / f"{symbol}_1min.csv"
            if not intraday_file.exists():
                logger.warning(f"No intraday data found for {symbol}")
                return None
            
            # Read and process intraday data
            df = pd.read_csv(intraday_file)
            
            # Check for required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns for {symbol}")
                return None
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by date range
            df = df[(df['timestamp'].dt.date >= pd.to_datetime(self.market_start_date).date()) & 
                    (df['timestamp'].dt.date <= pd.to_datetime(self.market_end_date).date())]
            
            if df.empty:
                logger.warning(f"No data available in date range for {symbol}")
                return None
            
            # Convert price and volume columns to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Save processed data
            output_path = self.PROCESSED_INTRADAY / f"intraday_{symbol}_processed.parquet"
            os.makedirs(self.PROCESSED_INTRADAY, exist_ok=True)
            df.to_parquet(output_path)
            logger.info(f"Successfully processed and saved intraday data for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing intraday data for {symbol}: {str(e)}")
            return None
    
    def process_options_data(self, symbol):
        try:
            # Check if options data file exists
            options_file = self.RAW_OPTIONS / f"{symbol}_options.csv"
            if not options_file.exists():
                logger.warning(f"No options data available for {symbol}")
                return None
            
            # Read and process options data
            df = pd.read_csv(options_file)
            
            # Convert date columns
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'].dt.date >= pd.to_datetime(self.market_start_date).date()) & 
                        (df['date'].dt.date <= pd.to_datetime(self.market_end_date).date())]
            
            if df.empty:
                logger.warning(f"No options data available in date range for {symbol}")
                return None
            
            # Convert numeric columns
            numeric_columns = ['strike', 'last', 'mark', 'bid', 'bid_size', 'ask', 'ask_size', 'volume', 
                              'open_interest', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Save processed data
            output_path = self.PROCESSED_OPTIONS / f"options_{symbol}_processed.parquet"
            os.makedirs(self.PROCESSED_OPTIONS, exist_ok=True)
            df.to_parquet(output_path)
            logger.info(f"Successfully processed and saved options data for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing options data for {symbol}: {str(e)}")
            return None
    
    def process_sentiment_data(self, symbol):
        try:
            sentiment_file = self.RAW_NEWS / "sentiment" / f"news_{symbol}_{self.sentiment_date}.csv"
            if not sentiment_file.exists():
                logger.warning(f"No sentiment data found for {symbol}")
                return None
            
            df = pd.read_csv(sentiment_file)
            if df.empty:
                logger.warning(f"No sentiment data available for {symbol}")
                return None
            
            # Convert time_published to datetime
            df['time_published'] = pd.to_datetime(df['time_published'].apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}T{x[9:11]}:{x[11:13]}:{x[13:15]}"))
            
            # Filter by date range
            df = df[(df['time_published'].dt.date >= pd.to_datetime(self.market_start_date).date()) & 
                    (df['time_published'].dt.date <= pd.to_datetime(self.market_end_date).date())]
            
            if df.empty:
                logger.warning(f"No sentiment data available in date range for {symbol}")
                return None
            
            # Convert sentiment scores to numeric
            score_columns = ['overall_sentiment_score', 'relevance_score', f'{symbol}_relevance_score', f'{symbol}_sentiment_score']
            for col in score_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Save processed sentiment data
            output_path = self.PROCESSED_NEWS / f"sentiment_{symbol}_processed.parquet"
            os.makedirs(self.PROCESSED_NEWS, exist_ok=True)
            df.to_parquet(output_path)
            logger.info(f"Successfully processed and saved sentiment data for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing sentiment data for {symbol}: {str(e)}")
            return None
    
    def process_macro_data(self):
        try:
            # Process bond spreads
            bond_spreads_file = self.RAW_MACRO / f"bond_spreads_{self.macro_date}.csv"
            bond_spreads = pd.read_csv(bond_spreads_file)
            
            # Process ECB rates
            ecb_rates_file = self.RAW_MACRO / f"ecb_rates_{self.macro_date}.csv"
            ecb_rates = pd.read_csv(ecb_rates_file)
            
            # Process other macro data with 2024 dates
            cpi = pd.read_csv(self.RAW_MACRO / "cpi/cpi_monthly.csv")
            federal_funds = pd.read_csv(self.RAW_MACRO / "federal_funds/federal_funds_daily.csv")
            inflation = pd.read_csv(self.RAW_MACRO / "inflation/inflation_monthly.csv")
            unemployment = pd.read_csv(self.RAW_MACRO / "unemployment/unemployment_monthly.csv")
            
            # Process treasury yields
            treasury_yields = {}
            for duration in ['3month', '2year', '5year', '7year', '10year', '30year']:
                file_path = self.RAW_MACRO / "treasury_yields" / f"treasury_yield_{duration}_daily.csv"
                treasury_yields[duration] = pd.read_csv(file_path)
            
            # Filter macro data by date range
            for df in [cpi, federal_funds, inflation, unemployment] + list(treasury_yields.values()):
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[(df['date'] >= self.market_start_date) & (df['date'] <= self.market_end_date)]
            
            # Combine all macro data
            macro_data = pd.DataFrame(index=pd.date_range(start=self.market_start_date, end=self.market_end_date))
            
            # Add bond spreads data
            if not bond_spreads.empty:
                bond_spreads['date'] = pd.to_datetime(bond_spreads['date'])
                bond_spreads.set_index('date', inplace=True)
                spread_columns = ['IT10_spread', 'ES10_spread', 'FR10_spread', 'PT10_spread', 'GR10_spread',
                                'peripheral_spread', 'peripheral_cds', 'spread_volatility']
                for col in spread_columns:
                    if col in bond_spreads.columns:
                        macro_data[col] = bond_spreads[col]
            
            # Add ECB rates data
            if not ecb_rates.empty:
                ecb_rates['date'] = pd.to_datetime(ecb_rates['date'])
                ecb_rates.set_index('date', inplace=True)
                rate_columns = ['deposit_rate', 'refi_rate', 'corridor_width']
                for col in rate_columns:
                    if col in ecb_rates.columns:
                        macro_data[col] = ecb_rates[col]
            
            # Add time series data
            for df, col_name in [(cpi, 'cpi'), (federal_funds, 'federal_funds'), 
                                (inflation, 'inflation'), (unemployment, 'unemployment')]:
                if not df.empty and 'value' in df.columns:
                    df.set_index('date', inplace=True)
                    macro_data[col_name] = df['value']
            
            # Add treasury yields
            for duration, df in treasury_yields.items():
                if not df.empty and 'value' in df.columns:
                    df.set_index('date', inplace=True)
                    macro_data[f'treasury_{duration}'] = df['value']
            
            # Forward fill any missing values
            macro_data.ffill(inplace=True)
            
            # Save processed macro data
            os.makedirs(self.PROCESSED_MACRO, exist_ok=True)
            macro_data.to_parquet(self.PROCESSED_MACRO / "macro_processed.parquet")
            logger.info("Successfully processed and saved macro data")
            
        except Exception as e:
            logger.error(f"Error processing macro data: {str(e)}")
            raise
    
    def process_all_data(self):
        logger.info("Starting full data processing pipeline")
        
        # Process macro data first
        logger.info("Processing macro data")
        macro_data = self.process_macro_data()
        
        # Process data for each symbol
        for symbol in self.symbols:
            logger.info(f"Processing all data for {symbol}")
            
            # Process intraday data
            logger.info(f"Processing intraday data for {symbol}")
            intraday_data = self.process_intraday_data(symbol)
            
            # Process options data
            logger.info(f"Processing options data for {symbol}")
            options_data = self.process_options_data(symbol)
            
            # Process sentiment data
            logger.info(f"Processing sentiment data for {symbol}")
            sentiment_data = self.process_sentiment_data(symbol)
            
            if any(data is None for data in [intraday_data, options_data, sentiment_data]):
                logger.warning(f"Some data processing failed for {symbol}")
        
        logger.info("Completed full data processing pipeline")

if __name__ == "__main__":
    # Define symbols
    symbols = [
        "MMM", "ABBV", "ALIZY", "GOOG", "AMZN", "AMGN", "BUD", "AAPL", "BHP", "BA",
        "BP", "BTI", "CVX", "CSCO", "C", "KO", "DD", "XOM", "META", "GE", "GSK",
        "HSBC", "INTC", "IBM", "JNJ", "JPM", "MA", "MCD", "MRK", "MSFT", "NSRGY",
        "NVS", "NVDA", "ORCL", "PEP", "PFE", "PM", "PG", "RHHBY", "RY", "SHEL",
        "SSNLF", "SNY", "SIEGY", "TSM", "TTE", "V", "TM", "WMT", "DIS", "VIXM", "VIXY"
    ]
    
    # Create processor instance
    processor = MVLADataProcessor()
    
    # Process all data
    processor.process_all_data() 