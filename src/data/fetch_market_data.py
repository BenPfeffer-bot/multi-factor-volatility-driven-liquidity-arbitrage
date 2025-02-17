"""
1. Retrieve Historical Price Data for all 50 stocks in the EUROSTOXX 50 and the EUROSTOXX 50 Index.
Data Source: Bloomberg, Yahoo Finance, or Quandl.
Timeframe: 5-minute, 1-hour, daily for precision in volatility calculations.

Features:
Open, High, Low, Close, Volume (OHLCV)
Adjusted Close (for corporate actions)
Bid-Ask Spread (for liquidity consideration)

"""

from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import sys
from dotenv import load_dotenv
from typing import List, Optional
import logging
import os
import yfinance as yf
import time


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.config.paths import (
    RAW_DAILY,
    RAW_INTRADAY,
    PROCESSED_DAILY,
    PROCESSED_INTRADAY,
    LOGS_DIR,
)
from src.config.settings import TICKERS
from src.utils.log_utils import setup_logging, log_execution_time
from src.utils.path_utils import get_data_path

logger = setup_logging(__name__)


class DataFetcher:
    def __init__(
        self,
        output_dir_daily: Path = RAW_DAILY,
        output_dir_intraday: Path = RAW_INTRADAY,
        output_dir_processed_daily: Path = PROCESSED_DAILY,
        output_dir_processed_intraday: Path = PROCESSED_INTRADAY,
        log_dir: Path = LOGS_DIR,
    ):
        """
        Initialize the DataFetcher with the output directories for the daily and intraday data.
        """
        self.output_dir_daily = output_dir_daily
        self.output_dir_intraday = output_dir_intraday
        self.output_dir_processed_daily = output_dir_processed_daily
        self.output_dir_processed_intraday = output_dir_processed_intraday
        self.log_dir = log_dir
        load_dotenv()
        self.api_key = os.getenv("TWELVE_DATA_API_KEY")
        if not self.api_key:
            raise ValueError("TWELVE_DATA_API_KEY not found in environment variables")

        # Define interval mappings
        self.interval_mappings = {
            # Internal format to TwelveData format
            "1d": "1day",
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1hour",
            # TwelveData format to YFinance format
            "1day": "1d",
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "1hour": "1h",
        }

        # Rate limiting parameters
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum time between requests in seconds

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()

    def get_stock_data(
        self,
        symbol: str,
        api_key: str,
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        outputsize: int = None,
    ) -> pd.DataFrame:
        """
        Retrieve stock data with rate limiting and proper interval handling
        """
        try:
            self._wait_for_rate_limit()

            # Convert interval to TwelveData format
            twelve_data_interval = self.interval_mappings.get(interval, interval)

            params = {
                "symbol": symbol,
                "interval": twelve_data_interval,
                "apikey": api_key,
            }

            if start_date and end_date:
                params["start_date"] = start_date
                params["end_date"] = end_date
            elif outputsize:
                params["outputsize"] = outputsize

            response = requests.get(
                "https://api.twelvedata.com/time_series", params=params
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                raise Exception(f"API Error: {data.get('message', 'Unknown error')}")

            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            numeric_columns = ["open", "high", "low", "close", "volume"]
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

            return df.sort_values("datetime").reset_index(drop=True)

        except Exception as e:
            logger.warning(
                f"Twelve Data API failed for {symbol}: {str(e)}. Falling back to yfinance..."
            )
            try:
                self._wait_for_rate_limit()

                # Convert interval to yfinance format
                yf_interval = self.interval_mappings.get(twelve_data_interval, interval)

                ticker = yf.Ticker(symbol)
                if start_date and end_date:
                    df = ticker.history(
                        start=start_date, end=end_date, interval=yf_interval
                    )
                else:
                    df = ticker.history(period="1y", interval=yf_interval)

                # Standardize column names
                df = df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )

                df = df.reset_index()
                df = df.rename(columns={"Date": "datetime", "Datetime": "datetime"})

                return df.sort_values("datetime").reset_index(drop=True)

            except Exception as yf_error:
                logger.error(
                    f"Failed to fetch data from both APIs for {symbol}: {str(yf_error)}"
                )
                return pd.DataFrame()

    def save_stock_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        output_dir: Path = RAW_DAILY,
    ) -> Path:
        """
        Save stock data to CSV file

        Parameters:
        -----------
        df : pd.DataFrame
            Stock data to save
        symbol : str
            Stock symbol
        interval : str
            Time interval of the data
        output_dir : Path
            Directory to save the data (default: RAW_DAILY)

        Returns:
        --------
        Path
            Path to the saved file

        Raises:
        -------
        Exception
            If saving fails
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{interval}_{timestamp}.csv"
            filepath = output_dir / filename

            df.to_csv(filepath, index=False)
            print(f"Data saved successfully to {filepath}")
            return filepath

        except Exception as e:
            raise Exception(f"Failed to save data: {str(e)}")

    def get_eurostoxx_data(self, interval: str = "1d", years: int = 3) -> pd.DataFrame:
        """
        Fetch EUROSTOXX 50 Index data. Falls back to yfinance if primary API fails.
        """
        try:
            logger.info(
                f"Fetching EUROSTOXX 50 data with ticker {self.EUROSTOXX50_TICKER}"
            )
            df = self.get_stock_data(
                self.EUROSTOXX50_TICKER, self.api_key, interval=interval, years=years
            )
            df["index_returns"] = df["close"].pct_change()
            return df

        except Exception as e:
            logger.warning(
                f"Failed to fetch EUROSTOXX data from primary API: {str(e)}. Trying yfinance..."
            )
            try:
                # Use the correct Yahoo Finance ticker for EURO STOXX 50
                ticker = yf.Ticker("^STOXX50E")

                # Ensure interval is in yfinance format
                yf_interval = interval

                logger.info(
                    f"Fetching {years} years of EURO STOXX 50 data with interval {yf_interval}"
                )
                df = ticker.history(period=f"{years}y", interval=yf_interval)

                if df.empty:
                    raise Exception("Received empty dataframe from yfinance")

                # Rename columns to match our format
                df = df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )

                df["index_returns"] = df["close"].pct_change()
                df = df.reset_index()
                df = df.rename(columns={"Date": "datetime", "Datetime": "datetime"})

                logger.info(
                    f"Successfully fetched {len(df)} rows of EURO STOXX 50 data"
                )
                return df

            except Exception as yf_error:
                logger.error(f"Failed to fetch data from both APIs: {str(yf_error)}")
                raise Exception("Failed to fetch EUROSTOXX data from all sources")

    def get_intraday_data(
        self, symbol: str, interval: str = "1m", days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch intraday data with bid-ask spreads. Falls back to yfinance if primary API fails.
        For 1-minute data, fetches data in 7-day chunks due to YFinance limitations.
        """
        if interval not in self.intraday_intervals:
            raise ValueError(
                f"Invalid interval. Must be one of {self.intraday_intervals}"
            )

        # Calculate proper outputsize based on interval
        intervals_per_day = {
            "1m": 1440,  # 24 hours * 60 minutes
            "5m": 288,  # 24 hours * 12 intervals
            "15m": 96,  # 24 hours * 4 intervals
            "30m": 48,  # 24 hours * 2 intervals
            "1h": 24,  # 24 hours
        }

        # Adjust for trading hours (approximately 8 hours per day)
        trading_hours_factor = 8 / 24
        outputsize = int(days * intervals_per_day[interval] * trading_hours_factor)

        try:
            logger.info(
                f"Fetching {days} days of intraday data for {symbol} with interval {interval}"
            )
            df = self.get_stock_data(
                symbol=symbol,
                api_key=self.api_key,
                interval=interval,
                years=None,
                outputsize=outputsize,
            )

            if df.empty:
                raise Exception("Received empty dataframe from primary API")

            # Add bid-ask spread calculation if available
            if "bid" in df.columns and "ask" in df.columns:
                df["spread"] = df["ask"] - df["bid"]
                df["spread_pct"] = df["spread"] / df["mid_price"]

            return df

        except Exception as e:
            logger.warning(
                f"Failed to fetch intraday data from primary API for {symbol}: {str(e)}. Trying yfinance..."
            )
            try:
                ticker = yf.Ticker(symbol)
                all_data = []

                # For 1-minute data, we need to fetch in chunks due to YFinance limitations
                if interval == "1m":
                    chunk_size = 7  # YFinance allows up to 7 days of 1-min data
                    for i in range(0, days, chunk_size):
                        chunk_days = min(chunk_size, days - i)
                        end_date = datetime.now() - pd.Timedelta(days=i)
                        start_date = end_date - pd.Timedelta(days=chunk_days)

                        chunk_df = ticker.history(
                            start=start_date, end=end_date, interval=interval
                        )
                        if not chunk_df.empty:
                            all_data.append(chunk_df)

                        # Sleep briefly to avoid rate limiting
                        time.sleep(0.5)

                    if all_data:
                        df = pd.concat(all_data)
                    else:
                        raise Exception("No data received from YFinance")
                else:
                    # For other intervals, we can fetch the entire period
                    df = ticker.history(period=f"{days}d", interval=interval)

                if df.empty:
                    raise Exception("Received empty dataframe from yfinance")

                # Standardize column names
                df = df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )

                # Drop any unused columns
                columns_to_keep = ["open", "high", "low", "close", "volume"]
                df = df[columns_to_keep]

                df = df.reset_index()
                df = df.rename(columns={"Date": "datetime", "Datetime": "datetime"})

                # Sort by datetime and remove duplicates
                df = df.sort_values("datetime").drop_duplicates(subset=["datetime"])

                logger.info(
                    f"Successfully fetched {len(df)} rows of intraday data for {symbol}"
                )
                return df

            except Exception as yf_error:
                logger.error(
                    f"Failed to fetch intraday data for {symbol} from both APIs: {str(yf_error)}"
                )
                raise Exception(
                    f"Failed to fetch intraday data for {symbol} from all sources"
                )

    # def calculate_realized_volatility(
    #     self, df: pd.DataFrame, window: int = 5
    # ) -> pd.Series:
    #     """Calculate realized volatility using 5-minute returns"""
    #     # Resample to 5-minute if needed
    #     df_5min = df.resample("5T").last()

    #     # Calculate log returns
    #     returns = np.log(df_5min["close"]).diff()

    #     # Calculate realized volatility
    #     rv = np.sqrt(np.sum(returns**2) * 252 * 78)  # Annualized
    #     return rv

    # def calculate_cross_sectional_volatility(
    #     self, symbols: List[str], interval: str = "1day"
    # ) -> pd.DataFrame:
    #     """
    #     Calculate cross-sectional volatility across index components
    #     """
    #     all_returns = []

    #     for symbol in symbols:
    #         try:
    #             df = self.get_stock_data(symbol, self.api_key, interval=interval)
    #             returns = df["close"].pct_change()
    #             all_returns.append(returns)
    #         except Exception as e:
    #             logger.warning(f"Failed to fetch data for {symbol}: {e}")

    #     returns_df = pd.concat(all_returns, axis=1)
    #     returns_df.columns = symbols

    #     # Calculate cross-sectional stats
    #     csv = pd.DataFrame(
    #         {
    #             "cross_sectional_vol": returns_df.std(axis=1),
    #             "cross_sectional_skew": returns_df.skew(axis=1),
    #             "cross_sectional_kurt": returns_df.kurtosis(axis=1),
    #         }
    #     )

    #     return csv


def main():
    data_fetcher = DataFetcher()
    interval = "1d"

    try:
        # Fetch data for individual tickers
        for symbol in TICKERS:
            try:
                logger.info(f"Starting market data fetch for {symbol}")

                # Get stock data
                df = data_fetcher.get_stock_data(
                    symbol, data_fetcher.api_key, interval=interval
                )
                data_fetcher.save_stock_data(df, symbol, interval)

                # Get intraday data if needed
                intraday_df = data_fetcher.get_intraday_data(symbol)

                # Verify intraday data before saving
                if intraday_df is not None and not intraday_df.empty:
                    saved_path = data_fetcher.save_stock_data(
                        intraday_df,
                        symbol,
                        "1m",
                        output_dir=data_fetcher.output_dir_intraday,
                    )
                    logger.info(
                        f"Saved {len(intraday_df)} rows of intraday data for {symbol} to {saved_path}"
                    )
                else:
                    logger.error(f"No intraday data available for {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                continue

        # Get EUROSTOXX data
        try:
            logger.info("Attempting to fetch EURO STOXX 50 data...")
            eurostoxx_df = data_fetcher.get_eurostoxx_data(interval=interval)

            if eurostoxx_df is not None and not eurostoxx_df.empty:
                saved_path = data_fetcher.save_stock_data(
                    eurostoxx_df, "^STOXX50E", interval
                )
                logger.info(
                    f"Successfully saved {len(eurostoxx_df)} rows of EURO STOXX 50 data to {saved_path}"
                )
            else:
                logger.error("No EURO STOXX 50 data available")
        except Exception as e:
            logger.error(f"Failed to fetch EUROSTOXX data: {str(e)}")

        logger.info("Market data fetch and save completed")

    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
