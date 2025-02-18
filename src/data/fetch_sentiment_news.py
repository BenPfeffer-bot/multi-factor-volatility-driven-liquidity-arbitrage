"""
Market Sentiment and Insider Transactions Module

This module handles fetching market news, sentiment data, and insider transactions
from Alpha Vantage API. It supports filtering by tickers, topics, and date ranges.
"""

import time
import logging
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import sys, os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.paths import RAW_NEWS
from src.config.settings import DJ_TITANS_50_TICKER
from src.utils.log_utils import setup_logging

load_dotenv()

# Setup logging
logger = setup_logging(__name__)

class SentimentDataError(Exception):
    """Base exception for sentiment data fetching errors."""
    pass

def fetch_market_sentiment(
    tickers: Optional[Union[str, List[str]]] = None,
    topics: Optional[Union[str, List[str]]] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    sort: str = "LATEST",
    limit: int = 1000,
    api_key: Optional[str] = None,
    output_dir: Path = RAW_NEWS / "sentiment"
) -> Dict[str, Any]:
    """
    Fetch market news and sentiment data from Alpha Vantage.
    """
    if api_key is None:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("No API key provided")

    base_url = "https://www.alphavantage.co/query"
    
    # Convert list parameters to comma-separated strings
    if isinstance(tickers, list):
        tickers = ','.join(tickers)
    if isinstance(topics, list):
        topics = ','.join(topics)
    
    # Build minimal params first
    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": api_key,
    }
    
    # Only add optional parameters if they have values
    if tickers:
        params["tickers"] = tickers
    if topics:
        params["topics"] = topics
    if sort:
        params["sort"] = sort
    if limit:
        params["limit"] = limit
    if time_from:
        # Ensure proper format YYYYMMDDTHHMM
        try:
            dt = datetime.strptime(time_from, "%Y%m%dT%H%M")
            params["time_from"] = dt.strftime("%Y%m%dT%H%M")
        except ValueError:
            try:
                dt = datetime.strptime(time_from, "%Y-%m-%dT%H:%M")
                params["time_from"] = dt.strftime("%Y%m%dT%H%M")
            except ValueError:
                logger.warning(f"Invalid time_from format: {time_from}. Skipping.")
    
    if time_to:
        # Ensure proper format YYYYMMDDTHHMM
        try:
            dt = datetime.strptime(time_to, "%Y%m%dT%H%M")
            params["time_to"] = dt.strftime("%Y%m%dT%H%M")
        except ValueError:
            try:
                dt = datetime.strptime(time_to, "%Y-%m-%dT%H:%M")
                params["time_to"] = dt.strftime("%Y%m%dT%H%M")
            except ValueError:
                logger.warning(f"Invalid time_to format: {time_to}. Skipping.")
        
    try:
        # Log the request parameters (excluding API key)
        log_params = {k: v for k, v in params.items() if k != 'apikey'}
        logger.info(f"Fetching sentiment data with params: {log_params}")
        
        response = requests.get(base_url, params=params)
        
        # Log the response status
        logger.info(f"Response status: {response.status_code}")
        
        response.raise_for_status()
        data = response.json()
        
        # Log the response structure
        logger.info(f"Response keys: {data.keys()}")
        
        if 'Note' in data:
            logger.warning(f"API rate limit message: {data['Note']}")
            time.sleep(0.8)  # Wait longer if we hit rate limit
            return None
            
        if 'Information' in data:
            logger.warning(f"API information: {data['Information']}")
            return None
            
        if not data or 'feed' not in data:
            logger.warning(f"No sentiment data found. Response: {data}")
            return None
        
        feed_length = len(data['feed'])
        if feed_length == 0:
            logger.warning(f"Empty sentiment feed")
            return None
            
        logger.info(f"Retrieved {feed_length} news items")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate shorter filename
        date_str = datetime.now().strftime("%Y%m%d")
        ticker_str = tickers.split(',')[0] if tickers else 'ALL'
        
        # Save sentiment metrics to CSV
        sentiment_data = []
        for article in data['feed']:
            # Extract ticker-specific sentiment
            ticker_sentiments = article.get('ticker_sentiment', [])
            ticker_data = {ts['ticker']: ts for ts in ticker_sentiments}
            
            base_data = {
                'time_published': article.get('time_published'),
                'title': article.get('title'),
                'summary': article.get('summary', ''),
                'source': article.get('source', ''),
                'overall_sentiment_score': article.get('overall_sentiment_score'),
                'overall_sentiment_label': article.get('overall_sentiment_label'),
                'relevance_score': article.get('relevance_score', 0),
                'url': article.get('url')
            }
            
            # Add ticker-specific sentiment if available
            if tickers:
                for ticker in tickers.split(','):
                    if ticker in ticker_data:
                        ts = ticker_data[ticker]
                        base_data.update({
                            f'{ticker}_relevance_score': ts.get('relevance_score'),
                            f'{ticker}_sentiment_score': ts.get('sentiment_score'),
                            f'{ticker}_sentiment_label': ts.get('sentiment_label')
                        })
            
            sentiment_data.append(base_data)
        
        if sentiment_data:
            df = pd.DataFrame(sentiment_data)
            csv_file = output_dir / f"news_{ticker_str}_{date_str}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved {feed_length} news records to CSV for {ticker_str}")
        
        return data
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Response text: {e.response.text}")
        raise SentimentDataError(f"Failed to fetch sentiment data: {str(e)}")
    except Exception as e:
        logger.error(f"Error fetching sentiment data: {str(e)}")
        raise SentimentDataError(f"Failed to fetch sentiment data: {str(e)}")

def fetch_insider_transactions(
    symbol: str,
    api_key: Optional[str] = None,
    output_dir: Path = RAW_NEWS / "insider"
) -> Dict[str, Any]:
    """
    Fetch insider transactions data from Alpha Vantage.
    
    This function retrieves the latest and historical insider transactions made by key stakeholders
    (e.g., founders, executives, board members, etc.) of a specific company.
    
    Args:
        symbol (str): The stock symbol to fetch insider transactions for (e.g., 'IBM')
        api_key (str, optional): Alpha Vantage API key. If None, will look for ALPHA_VANTAGE_API_KEY environment variable
        output_dir (Path): Directory to save the insider transactions data
        
    Returns:
        Dict[str, Any]: Dictionary containing insider transactions data if successful, None otherwise
        
    Raises:
        ValueError: If no API key is provided
        SentimentDataError: If there's an error fetching the data
    """
    if api_key is None:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("No API key provided. Set ALPHA_VANTAGE_API_KEY environment variable or pass api_key parameter")

    base_url = "https://www.alphavantage.co/query"
    
    params = {
        "function": "INSIDER_TRANSACTIONS",
        "symbol": symbol,
        "apikey": api_key
    }
    
    try:
        logger.info(f"Fetching insider transactions for {symbol}")
        
        # Add rate limiting delay
        time.sleep(0.8)  # 800ms delay between requests
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Log raw response for debugging
        logger.debug(f"Raw response for {symbol}: {data}")
        
        # Check for API errors or rate limiting
        if 'Note' in data:
            logger.warning(f"API rate limit message: {data['Note']}")
            return None
            
        if 'Information' in data:
            logger.warning(f"API information message: {data['Information']}")
            return None
        
        # Validate the response format
        if not isinstance(data, dict):
            logger.error(f"Invalid response type for {symbol}: {type(data)}")
            return None
            
        # Log all keys in response for debugging
        logger.info(f"Response keys for {symbol}: {data.keys()}")
        
        # The API returns data with 'data' key containing the transactions
        if 'data' not in data:
            logger.warning(f"No data key found in response for {symbol}")
            return None
            
        transactions = data['data']
        if not transactions:
            logger.warning(f"Empty transactions list for {symbol}")
            return None
            
        if not isinstance(transactions, list):
            logger.error(f"Invalid transactions format for {symbol}: {type(transactions)}")
            return None
        
        # Log first transaction for debugging
        if transactions:
            logger.debug(f"First transaction for {symbol}: {transactions[0]}")
        
        # Log summary of transactions
        logger.info(f"Retrieved {len(transactions)} insider transactions for {symbol}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with symbol and date
        date_str = datetime.now().strftime("%Y%m%d")
        output_base = output_dir / f"{symbol}_insider_transactions_{date_str}"
        
        # Save as JSON for raw data preservation
        # with open(f"{output_base}.json", 'w') as f:
        #     json.dump(data, f, indent=2)
        # logger.info(f"Saved raw insider transactions data to JSON for {symbol}")
        
        # Convert transactions to DataFrame and clean up
        df = pd.DataFrame(transactions)
        
        # Log DataFrame columns for debugging
        logger.debug(f"DataFrame columns for {symbol}: {df.columns.tolist()}")
        
        # Map API column names to our desired names
        column_mapping = {
            'filing_date': 'filing_date',
            'transaction_date': 'transaction_date',
            'transaction_type': 'transaction_type',
            'shares': 'shares',
            'transaction_price': 'price',
            'name': 'insider_name',
            'title': 'insider_title'
        }
        
        # Rename columns to our standard format
        df = df.rename(columns=column_mapping)
        
        # Save as CSV for easier analysis
        df.to_csv(f"{output_base}.csv", index=False)
        logger.info(f"Saved insider transactions to CSV for {symbol}")
        
        # Log some basic statistics
        if not df.empty:
            latest_date = df['filing_date'].max()
            earliest_date = df['filing_date'].min()
            logger.info(f"Transactions date range for {symbol}: {earliest_date} to {latest_date}")
            
            # Count unique transaction types
            transaction_types = df['transaction_type'].unique()
            logger.info(f"Transaction types found for {symbol}: {', '.join(filter(None, transaction_types))}")
            
            # Log summary of transaction counts by type
            type_counts = df['transaction_type'].value_counts()
            logger.info(f"Transaction type counts for {symbol}:")
            for t_type, count in type_counts.items():
                logger.info(f"  {t_type}: {count}")
        
        return data
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {symbol}: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logger.error(f"Response text: {e.response.text}")
        raise SentimentDataError(f"Failed to fetch insider transactions: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing insider transactions for {symbol}: {str(e)}")
        raise SentimentDataError(f"Failed to process insider transactions: {str(e)}")

def fetch_all_sentiment(
    tickers: List[str] = DJ_TITANS_50_TICKER,
    topics: Optional[List[str]] = [
        "technology", "earnings", "financial_markets", "economy_macro", 
        "finance", "economy_fiscal", "economy_monetary", "mergers_and_acquisitions",
        "ipo", "retail_wholesale"
    ],
    api_key: Optional[str] = None,
    batch_size: int = 1  # Process one ticker at a time to ensure success
) -> None:
    """
    Fetch sentiment data and insider transactions for multiple tickers.
    """
    logger.info(f"Starting batch sentiment and insider fetch for {len(tickers)} symbols")
    
    # Process tickers one at a time
    for ticker in tickers:
        try:
            # Fetch sentiment data
            # Try first without topics to get ticker-specific news
            sentiment_data = fetch_market_sentiment(
                tickers=ticker,
                api_key=api_key,
                limit=1000
            )
            
            if not sentiment_data:
                # If no data found, try with topics
                sentiment_data = fetch_market_sentiment(
                    tickers=ticker,
                    topics=topics,
                    api_key=api_key,
                    limit=1000
                )
            
            if sentiment_data:
                logger.info(f"Successfully fetched sentiment data for {ticker}")
            else:
                logger.warning(f"No sentiment data available for {ticker}")

            # Fetch insider transactions
            insider_data = fetch_insider_transactions(
                symbol=ticker,
                api_key=api_key
            )

            if insider_data:
                logger.info(f"Successfully fetched insider transactions for {ticker}")
            else:
                logger.warning(f"No insider transactions available for {ticker}")
            
            time.sleep(0.8)  # Respect rate limit (5 calls per minute)
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
            time.sleep(0.8)
            continue
    
    logger.info("Batch sentiment and insider fetch completed")

if __name__ == "__main__":
    # Example usage
    fetch_all_sentiment()
