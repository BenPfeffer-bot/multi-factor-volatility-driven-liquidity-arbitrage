"""
Options Data Fetching Module

This module handles fetching historical options data from Alpha Vantage API.
It supports fetching data for multiple symbols with proper rate limiting and error handling.
"""

import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys, os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.paths import RAW_OPTIONS
from src.config.settings import DJ_TITANS_50_TICKER
from src.utils.log_utils import setup_logging

load_dotenv()

# Setup logging
logger = setup_logging(__name__)

class OptionsDataError(Exception):
    """Base exception for options data fetching errors."""
    pass

def fetch_historical_options(
    symbol: str,
    api_key: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Path = RAW_OPTIONS
) -> List[Dict[str, Any]]:
    """
    Fetch historical options data for a given symbol and compile into a single file.
    
    Args:
        symbol (str): Stock symbol to fetch options data for
        api_key (str, optional): Alpha Vantage API key. If None, will look for ALPHA_VANTAGE_API_KEY environment variable
        start_date (str, optional): Start date in YYYY-MM-DD format. Defaults to 1 year ago
        end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today
        output_dir (Path): Directory to save the options data
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing historical options data
        
    Raises:
        OptionsDataError: If there's an error fetching or processing the options data
        ValueError: If invalid parameters are provided
    """
    if api_key is None:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("No API key provided. Set ALPHA_VANTAGE_API_KEY environment variable or pass api_key parameter")

    base_url = "https://www.alphavantage.co/query"
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
    # Validate dates
    try:
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Ensure start date is not before 2008-01-01 (API limitation)
        min_date = datetime(2008, 1, 1)
        if start_date_dt < min_date:
            logger.warning(f"Start date {start_date} before 2008-01-01, adjusting to 2008-01-01")
            start_date_dt = min_date
            start_date = min_date.strftime('%Y-%m-%d')
            
        if end_date_dt < start_date_dt:
            raise ValueError("End date must be after start date")
            
    except ValueError as e:
        raise ValueError(f"Invalid date format. Dates must be in YYYY-MM-DD format: {str(e)}")
        
    logger.info(f"Fetching options data for {symbol} from {start_date} to {end_date}")
    
    all_data = []
    current_date = end_date_dt
    failed_dates = []
    
    while current_date >= start_date_dt:
        date_str = current_date.strftime('%Y-%m-%d')
        
        params = {
            "function": "HISTORICAL_OPTIONS",
            "symbol": symbol,
            "date": date_str,
            "apikey": api_key
        }
        
        try:
            # Add rate limiting delay - 5 API calls per minute
            time.sleep(0.8)  # 800ms delay between requests
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Validate response data
            if not data or not isinstance(data, dict):
                raise OptionsDataError(f"Invalid response format for {symbol} on {date_str}")
                
            if 'data' in data and isinstance(data['data'], list):
                all_data.extend(data['data'])
                logger.info(f"Retrieved {len(data['data'])} options contracts for {symbol} on {date_str}")
            else:
                logger.warning(f"No options data available for {symbol} on {date_str}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {symbol} on {date_str}: {str(e)}")
            failed_dates.append(date_str)
        except Exception as e:
            logger.error(f"Error processing data for {symbol} on {date_str}: {str(e)}")
            failed_dates.append(date_str)
            
        # Move to previous day
        current_date -= timedelta(days=1)
    
    # Save data if any was collected
    if all_data:
        try:
            df = pd.DataFrame(all_data)
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            output_file = output_dir / f"{symbol}_options.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(all_data)} options contracts for {symbol} to {output_file}")
            
            # Log any failed dates
            if failed_dates:
                logger.warning(f"Failed to fetch data for {len(failed_dates)} dates for {symbol}: {failed_dates}")
                
        except Exception as e:
            raise OptionsDataError(f"Error saving options data for {symbol}: {str(e)}")
    else:
        logger.warning(f"No options data collected for {symbol} in the specified date range")
    
    return all_data

def fetch_all_options(
    tickers: List[str] = DJ_TITANS_50_TICKER,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None
) -> None:
    """
    Fetch historical options data for multiple symbols.
    
    Args:
        tickers (List[str]): List of stock symbols to fetch options data for
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        api_key (str, optional): Alpha Vantage API key
    """
    logger.info(f"Starting batch options data fetch for {len(tickers)} symbols")
    
    failed_symbols = []
    for symbol in tickers:
        try:
            data = fetch_historical_options(
                symbol=symbol,
                api_key=api_key,
                start_date=start_date,
                end_date=end_date
            )
            if data:
                logger.info(f"Successfully fetched {len(data)} options contracts for {symbol}")
            else:
                logger.warning(f"No options data available for {symbol}")
                
        except Exception as e:
            logger.error(f"Failed to fetch options data for {symbol}: {str(e)}")
            failed_symbols.append(symbol)
            continue
    
    if failed_symbols:
        logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")
    
    logger.info("Batch options data fetch completed")

if __name__ == "__main__":
    # Example usage
    fetch_historical_options("AAPL", start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    # fetch_all_options(
    #     start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    # )
