"""
Macro Economic Data Fetching Module

This module handles fetching various macro economic indicators from Alpha Vantage API:
- Treasury Yields (multiple maturities)
- Federal Funds Rate
- Inflation Rate
- Consumer Price Index (CPI)
- Unemployment Rate

It supports different intervals and handles rate limiting appropriately.
"""

import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import sys, os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.paths import RAW_MACRO
from src.utils.log_utils import setup_logging

load_dotenv()

# Setup logging
logger = setup_logging(__name__)

class MacroDataError(Exception):
    """Base exception for macro data fetching errors."""
    pass

def fetch_treasury_yields(
    api_key: Optional[str] = None,
    interval: str = 'daily',
    years: int = 5,
    output_dir: Path = RAW_MACRO / "treasury_yields"
) -> Dict[str, Any]:
    """
    Fetch treasury yield data for multiple maturities from Alpha Vantage.
    
    Args:
        api_key (str, optional): Alpha Vantage API key
        interval (str): Data interval - 'daily', 'weekly', or 'monthly'
        years (int): Number of years of historical data to fetch
        output_dir (Path): Directory to save output files
        
    Returns:
        Dict[str, Any]: Dictionary containing treasury yield data for each maturity
    """
    if api_key is None:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("No API key provided. Set ALPHA_VANTAGE_API_KEY environment variable or pass api_key parameter")

    base_url = "https://www.alphavantage.co/query"
    maturities = ['3month', '2year', '5year', '7year', '10year', '30year']
    
    # Calculate date range
    n_years_ago = datetime.now() - timedelta(days=365*years)
    time_from = n_years_ago.strftime("%Y-%m-%d")
    
    all_data = {}
    failed_maturities = []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize DataFrames for each interval
    daily_data = []
    weekly_data = []
    monthly_data = []
    
    for maturity in maturities:
        params = {
            "function": "TREASURY_YIELD",
            "maturity": maturity,
            "interval": interval,
            "apikey": api_key
        }
        
        try:
            time.sleep(0.8)  # Rate limiting
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'data' not in data:
                raise MacroDataError(f"Invalid response format for {maturity} treasury yield")
            
            # Filter and process data
            filtered_data = [d for d in data['data'] if d['date'] >= time_from]
            
            # Add maturity column to data
            for record in filtered_data:
                record['maturity'] = maturity
            
            # Add to appropriate interval list
            if interval == 'daily':
                daily_data.extend(filtered_data)
            elif interval == 'weekly':
                weekly_data.extend(filtered_data)
            else:  # monthly
                monthly_data.extend(filtered_data)
            
            logger.info(f"Processed {len(filtered_data)} records for {maturity} treasury yield")
            all_data[maturity] = data
            
        except Exception as e:
            logger.error(f"Error fetching treasury yield data for {maturity}: {str(e)}")
            failed_maturities.append(maturity)
            continue
    
    # Save compiled data for each interval
    if daily_data:
        df_daily = pd.DataFrame(daily_data)
        output_file = output_dir / "treasury_yields_daily.csv"
        df_daily.to_csv(output_file, index=False)
        logger.info(f"Saved {len(daily_data)} daily records to {output_file}")
        
    if weekly_data:
        df_weekly = pd.DataFrame(weekly_data)
        output_file = output_dir / "treasury_yields_weekly.csv"
        df_weekly.to_csv(output_file, index=False)
        logger.info(f"Saved {len(weekly_data)} weekly records to {output_file}")
        
    if monthly_data:
        df_monthly = pd.DataFrame(monthly_data)
        output_file = output_dir / "treasury_yields_monthly.csv"
        df_monthly.to_csv(output_file, index=False)
        logger.info(f"Saved {len(monthly_data)} monthly records to {output_file}")
    
    if failed_maturities:
        logger.warning(f"Failed to fetch data for maturities: {failed_maturities}")
    return all_data

def fetch_federal_funds_rate(
    api_key: Optional[str] = None,
    interval: str = 'daily',
    years: int = 5,
    output_dir: Path = RAW_MACRO / "federal_funds"
) -> Dict[str, Any]:
    """
    Fetch Federal Funds Rate data from Alpha Vantage.
    
    Args:
        api_key (str, optional): Alpha Vantage API key
        interval (str): Data interval - 'daily', 'weekly', or 'monthly'
        years (int): Number of years of historical data to fetch
        output_dir (Path): Directory to save output files
        
    Returns:
        Dict[str, Any]: Federal Funds Rate data
    """
    if api_key is None:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("No API key provided")

    base_url = "https://www.alphavantage.co/query"
    n_years_ago = datetime.now() - timedelta(days=365*years)
    time_from = n_years_ago.strftime("%Y-%m-%d")
    
    params = {
        "function": "FEDERAL_FUNDS_RATE",
        "interval": interval,
        "apikey": api_key
    }
    
    try:
        time.sleep(0.8)  # Rate limiting
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data or 'data' not in data:
            raise MacroDataError("Invalid response format for federal funds rate")
        
        # Filter and process data
        filtered_data = [d for d in data['data'] if d['date'] >= time_from]
        data['data'] = filtered_data
        
        # Save to CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(filtered_data)
        output_file = output_dir / f"federal_funds_{interval}.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(filtered_data)} records for federal funds rate")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching federal funds rate data: {str(e)}")
        raise MacroDataError(f"Failed to fetch federal funds rate data: {str(e)}")

def fetch_inflation_rate(
    api_key: Optional[str] = None,
    interval: str = 'monthly',
    years: int = 50,
    output_dir: Path = RAW_MACRO / "inflation"
) -> Dict[str, Any]:
    """
    Fetch Inflation Rate data from Alpha Vantage.
    
    Args:
        api_key (str, optional): Alpha Vantage API key
        interval (str): Data interval - 'monthly' or 'annual'
        years (int): Number of years of historical data to fetch
        output_dir (Path): Directory to save output files
        
    Returns:
        Dict[str, Any]: Inflation Rate data
    """
    if api_key is None:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("No API key provided")

    base_url = "https://www.alphavantage.co/query"
    n_years_ago = datetime.now() - timedelta(days=365*years)
    time_from = n_years_ago.strftime("%Y-%m-%d")
    
    params = {
        "function": "INFLATION",
        "apikey": api_key
    }
    
    try:
        time.sleep(0.8)  # Rate limiting
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data or 'data' not in data:
            raise MacroDataError("Invalid response format for inflation rate")
        
        # Filter and process data
        filtered_data = [d for d in data['data'] if d['date'] >= time_from]
        data['data'] = filtered_data
        
        # Save to CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(filtered_data)
        output_file = output_dir / f"inflation_{interval}.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(filtered_data)} records for inflation rate")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching inflation rate data: {str(e)}")
        raise MacroDataError(f"Failed to fetch inflation rate data: {str(e)}")

def fetch_cpi(
    api_key: Optional[str] = None,
    interval: str = 'monthly',
    years: int = 50,
    output_dir: Path = RAW_MACRO / "cpi"
) -> Dict[str, Any]:
    """
    Fetch Consumer Price Index (CPI) data from Alpha Vantage.
    
    Args:
        api_key (str, optional): Alpha Vantage API key
        interval (str): Data interval - 'monthly' or 'semiannual'
        years (int): Number of years of historical data to fetch
        output_dir (Path): Directory to save output files
        
    Returns:
        Dict[str, Any]: CPI data
    """
    if api_key is None:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("No API key provided")

    base_url = "https://www.alphavantage.co/query"
    n_years_ago = datetime.now() - timedelta(days=365*years)
    time_from = n_years_ago.strftime("%Y-%m-%d")
    
    params = {
        "function": "CPI",
        "interval": interval,
        "apikey": api_key
    }
    
    try:
        time.sleep(0.8)  # Rate limiting
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data or 'data' not in data:
            raise MacroDataError("Invalid response format for CPI")
        
        # Filter and process data
        filtered_data = [d for d in data['data'] if d['date'] >= time_from]
        data['data'] = filtered_data
        
        # Save to CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(filtered_data)
        output_file = output_dir / f"cpi_{interval}.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(filtered_data)} records for CPI")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching CPI data: {str(e)}")
        raise MacroDataError(f"Failed to fetch CPI data: {str(e)}")

def fetch_unemployment_rate(
    api_key: Optional[str] = None,
    interval: str = 'monthly',
    years: int = 50,
    output_dir: Path = RAW_MACRO / "unemployment"
) -> Dict[str, Any]:
    """
    Fetch Unemployment Rate data from Alpha Vantage.
    
    Args:
        api_key (str, optional): Alpha Vantage API key
        interval (str): Data interval - 'monthly' or 'annual'
        years (int): Number of years of historical data to fetch
        output_dir (Path): Directory to save output files
        
    Returns:
        Dict[str, Any]: Unemployment Rate data
    """
    if api_key is None:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("No API key provided")

    base_url = "https://www.alphavantage.co/query"
    n_years_ago = datetime.now() - timedelta(days=365*years)
    time_from = n_years_ago.strftime("%Y-%m-%d")
    
    params = {
        "function": "UNEMPLOYMENT",
        "apikey": api_key
    }
    
    try:
        time.sleep(0.8)  # Rate limiting
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data or 'data' not in data:
            raise MacroDataError("Invalid response format for unemployment rate")
        
        # Filter and process data
        filtered_data = [d for d in data['data'] if d['date'] >= time_from]
        data['data'] = filtered_data
        
        # Save to CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(filtered_data)
        output_file = output_dir / f"unemployment_{interval}.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(filtered_data)} records for unemployment rate")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching unemployment rate data: {str(e)}")
        raise MacroDataError(f"Failed to fetch unemployment rate data: {str(e)}")

def fetch_all_macro_data(
    api_key: Optional[str] = None,
    years: int = 5,
    include_daily: bool = True
) -> None:
    """
    Fetch all macro economic indicators.
    
    Args:
        api_key (str, optional): Alpha Vantage API key
        years (int): Number of years of historical data to fetch
        include_daily (bool): Whether to fetch daily data where available
    """
    logger.info("Starting batch macro data fetch")
    
    try:
        # Treasury Yields (daily, weekly, monthly available)
        intervals = ['daily', 'weekly', 'monthly'] if include_daily else ['weekly', 'monthly']
        for interval in intervals:
            fetch_treasury_yields(api_key=api_key, interval=interval, years=years)
        
        # Federal Funds Rate (daily available)
        if include_daily:
            fetch_federal_funds_rate(api_key=api_key, interval='daily', years=years)
        fetch_federal_funds_rate(api_key=api_key, interval='monthly', years=years)
        
        # Monthly indicators
        fetch_inflation_rate(api_key=api_key, years=years)
        fetch_cpi(api_key=api_key, years=years)
        fetch_unemployment_rate(api_key=api_key, years=years)
        
        logger.info("Batch macro data fetch completed successfully")
        
    except Exception as e:
        logger.error(f"Error in batch macro data fetch: {str(e)}")
        raise MacroDataError(f"Failed to complete batch macro data fetch: {str(e)}")

if __name__ == "__main__":
    # Example usage
    fetch_all_macro_data(years=5, include_daily=True)
