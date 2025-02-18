"""
Market Data Fetching Module

This module handles fetching intraday market data from Alpha Vantage API.
It supports different intervals and handles rate limiting appropriately.
"""

import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import sys, os
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.paths import RAW_INTRADAY
from src.config.settings import DJ_TITANS_50_TICKER
from src.utils.log_utils import setup_logging

load_dotenv()

# Setup logging
logger = setup_logging(__name__)

class AlphaVantageError(Exception):
    """Base exception for Alpha Vantage API errors."""
    pass

def fetch_market_intraday(
    ticker: str,
    interval: str = "1min",
    output_dir: Path = RAW_INTRADAY,
    api_key: Optional[str] = None,
    datatype: str = "csv",
    adjusted: bool = True,
    extended_hours: bool = True,
    outputsize: str = "full"
) -> Path:
    """
    Fetch intraday market data for a given ticker from Alpha Vantage.
    For 1min interval, fetches data month by month for the past year.
    For other intervals, fetches the most recent 30 days of data.

    Args:
        ticker (str): The ticker symbol of the stock to fetch data for.
        interval (str): Time interval between data points. One of: 1min, 5min, 15min, 30min, 60min
        output_dir (Path): The directory to save the fetched data.
        api_key (str): Alpha Vantage API key. If None, will look for ALPHA_VANTAGE_API_KEY environment variable
        datatype (str): Return data format - 'json' or 'csv'
        adjusted (bool): Whether to adjust for splits/dividends
        extended_hours (bool): Whether to include extended trading hours
        outputsize (str): Amount of data to return - 'compact' or 'full'

    Returns:
        Path: Path to the saved data file

    Raises:
        AlphaVantageError: If API request fails or returns invalid data
        ValueError: If invalid parameters are provided
    """
    if api_key is None:
        import os
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("No API key provided. Set ALPHA_VANTAGE_API_KEY environment variable or pass api_key parameter")

    # Validate parameters
    valid_intervals = ["1min", "5min", "15min", "30min", "60min"]
    if interval not in valid_intervals:
        raise ValueError(f"Invalid interval. Must be one of {valid_intervals}")

    logger.info(f"Starting data fetch for {ticker} with {interval} interval")
    data_file = output_dir / f"{ticker}_{interval}.{datatype}"
    
    try:
        # For 1min interval, fetch month by month for past year
        if interval == "1min":
            logger.info("Using 1min interval - fetching data month by month for past year")
            months = []
            current_date = datetime.now()
            for i in range(12):
                first_of_month = current_date.replace(day=1)
                months.append(first_of_month.strftime("%Y-%m"))
                current_date = first_of_month - timedelta(days=1)
            
            logger.info(f"Will fetch data for months: {months}")
            
            all_data = []
            header = None
            for month in months:
                logger.info(f"Fetching data for {ticker} - {month}")
                params = {
                    'function': 'TIME_SERIES_INTRADAY',
                    'symbol': ticker,
                    'interval': interval,
                    'apikey': api_key,
                    'datatype': datatype,
                    'adjusted': str(adjusted).lower(),
                    'extended_hours': str(extended_hours).lower(),
                    'outputsize': outputsize,
                    'month': month
                }
                
                response = requests.get('https://www.alphavantage.co/query', params=params)
                response.raise_for_status()
                
                # Parse CSV data
                lines = response.text.split('\n')
                if not header:
                    header = lines[0]
                    all_data.append(header)
                data_lines = lines[1:]
                all_data.extend(data_lines)
                
                logger.info(f"Retrieved {len(data_lines)} data points for {month}")
                
                # Rate limiting - 5 API calls per minute
                time.sleep(0.8)  # Wait 12 seconds between requests
                
        else:
            logger.info(f"Using {interval} interval - fetching most recent data")
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': ticker,
                'interval': interval,
                'apikey': api_key,
                'datatype': datatype,
                'adjusted': str(adjusted).lower(),
                'extended_hours': str(extended_hours).lower(),
                'outputsize': outputsize
            }
            
            response = requests.get('https://www.alphavantage.co/query', params=params)
            response.raise_for_status()
            all_data = response.text.split('\n')
            logger.info(f"Retrieved {len(all_data)-1} data points")

        # Write combined data to file
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing data to {data_file}")
        with open(data_file, "w") as f:
            f.write('\n'.join(filter(None, all_data)))  # Filter out empty lines

        logger.info("Data fetch completed successfully")
        return data_file

    except requests.exceptions.RequestException as e:
        raise AlphaVantageError(f"Failed to fetch data from Alpha Vantage: {str(e)}")
    except Exception as e:
        raise AlphaVantageError(f"Unexpected error while fetching data: {str(e)}")

def fetch_all_tickers(
    tickers: list = DJ_TITANS_50_TICKER,
    interval: str = "1min",
    api_key: Optional[str] = None
) -> None:
    """
    Fetch market data for multiple tickers.

    Args:
        tickers (list): List of ticker symbols to fetch
        interval (str): Time interval for the data
        api_key (str, optional): Alpha Vantage API key
    """
    logger.info(f"Starting batch fetch for {len(tickers)} tickers")
    
    for ticker in tickers:
        try:
            fetch_market_intraday(ticker, interval=interval, api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
            continue

if __name__ == "__main__":
    # Example usage
    fetch_all_tickers()
