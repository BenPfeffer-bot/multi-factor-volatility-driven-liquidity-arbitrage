import os
import sys
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.utils.log_utils import setup_logging
from src.config.paths import RAW_DATA, CACHE_DIR, MARKET_DATA_CACHE
from src.config.settings import TICKERS

logger = setup_logging(__name__)

NEWS_ENDPOINTS = {
    "finnhub": "https://finnhub.io/api/v1/company-news",
    "alpha_vantage": "https://www.alphavantage.co/query",
    "newsapi": "https://newsapi.org/v2/everything",
}

RELEVANCE_KEYWORDS = {
    "earnings": 1.0,
    "revenue": 1.0,
    "guidance": 0.9,
    "forecast": 0.8,
    "merger": 0.9,
    "acquisition": 0.9,
    "volatility": 0.8,
    "market": 0.7,
}


class SentimentDataFetcher:
    def __init__(self, cache_dir: Optional[Path] = None):
        load_dotenv()
        self.cache_dir = cache_dir or MARKET_DATA_CACHE
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # API Keys with better error handling
        self.finnhub_key = os.getenv("FINNHUB_API_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.tiingo_key = os.getenv("TIINGO_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")

        # Updated tickers and endpoints
        self.STOXX50_TICKER = "^STOXX50E"  # Main equity index

        # Import EURO STOXX 50 tickers from config
        self.stoxx50_tickers = TICKERS

        # Map European tickers to Finnhub-compatible symbols (using ADRs where available)
        self.FINNHUB_SYMBOL_MAPPING = {
            "ADYEN.AS": "ADYEY",  # Adyen ADR
            "AIR.PA": "EADSY",  # Airbus ADR
            "ALV.DE": "ALIZY",  # Allianz ADR
            "ASML.AS": "ASML",  # ASML ADR
            "BAS.DE": "BASFY",  # BASF ADR
            "BAYN.DE": "BAYRY",  # Bayer ADR
            "BBVA.MC": "BBVA",  # BBVA ADR
            "BMW.DE": "BMWYY",  # BMW ADR
            "BNP.PA": "BNPQY",  # BNP Paribas ADR
            "CS.PA": "AXAHY",  # AXA ADR
            "DTE.DE": "DTEGY",  # Deutsche Telekom ADR
            "ENEL.MI": "ENLAY",  # Enel ADR
            "ENI.MI": "E",  # Eni ADR
            "IBE.MC": "IBDRY",  # Iberdrola ADR
            "INGA.AS": "ING",  # ING ADR
            "ISP.MI": "ISNPY",  # Intesa Sanpaolo ADR
            "MC.PA": "LVMUY",  # LVMH ADR
            "MUV2.DE": "MURGY",  # Munich Re ADR
            "OR.PA": "LRLCY",  # L'Oreal ADR
            "PHIA.AS": "PHG",  # Philips ADR
            "SAF.PA": "SAFRY",  # Safran ADR
            "SAN.MC": "SAN",  # Banco Santander ADR
            "SAP.DE": "SAP",  # SAP ADR
            "SIE.DE": "SIEGY",  # Siemens ADR
            "TTE.PA": "TTE",  # TotalEnergies ADR
            "VOW3.DE": "VWAGY",  # Volkswagen ADR
        }

        # Updated ECB Statistical Data Warehouse endpoints
        self.ECB_ENDPOINTS = {
            "euribor": "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.RT.MM.EURIBOR3MD_.LST",
            "vstoxx": "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.VSTOXX.IDX",
            "bonds": {
                "DE10": "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
                "DE02": "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y",
                "IT10": "https://data-api.ecb.europa.eu/service/data/YC/B.IT.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
                "FR10": "https://data-api.ecb.europa.eu/service/data/YC/B.FR.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
                "ES10": "https://data-api.ecb.europa.eu/service/data/YC/B.ES.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
            },
        }

        # Alternative data sources for when ECB data is not available
        self.ALTERNATIVE_ENDPOINTS = {
            "vstoxx": "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.VSTOXX.IDX",
            "bonds": {
                "DE10": "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
                "IT10": "https://data-api.ecb.europa.eu/service/data/YC/B.IT.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
                "FR10": "https://data-api.ecb.europa.eu/service/data/YC/B.FR.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
                "ES10": "https://data-api.ecb.europa.eu/service/data/YC/B.ES.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
            },
        }

        # Yahoo Finance tickers for when ECB data is not available
        self.YAHOO_TICKERS = {
            "vstoxx": "^STOXX50E",  # Use EURO STOXX 50 volatility as proxy
            "bonds": {
                "IT10": "BTP10.MI",  # Italian 10Y
                "FR10": "^FCHI",  # CAC 40 as proxy for French rates
                "ES10": "^IBEX",  # IBEX 35 as proxy for Spanish rates
            },
        }

        # Map exchange suffixes to Finnhub exchange codes
        self.EXCHANGE_MAPPING = {
            ".AS": "XAMS",  # Amsterdam
            ".PA": "XPAR",  # Paris
            ".DE": "XETR",  # Frankfurt
            ".MC": "XMAD",  # Madrid
            ".MI": "XMIL",  # Milan
        }

        # Fallback data sources for when ECB data is not available
        self.INVESTING_ENDPOINTS = {
            "vstoxx": "https://www.investing.com/indices/vstoxx-historical-data",
            "bonds": {
                "IT10": "https://www.investing.com/rates-bonds/italy-10-year-bond-yield-historical-data",
                "FR10": "https://www.investing.com/rates-bonds/france-10-year-bond-yield-historical-data",
                "ES10": "https://www.investing.com/rates-bonds/spain-10-year-bond-yield-historical-data",
            },
        }

        # News API endpoints
        self.NEWS_ENDPOINTS = {
            "alpha_vantage": "https://www.alphavantage.co/query",
            "tiingo": "https://api.tiingo.com/tiingo/news",
            "finnhub": "https://finnhub.io/api/v1/news",
            "newsapi": "https://newsapi.org/v2/everything",
        }

    def _fetch_ecb_data(
        self, endpoint: str, start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch data from ECB Statistical Data Warehouse."""
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(
                    days=365 * 2
                )  # Fetch 2 years of data

            # Convert endpoint to use the public API
            endpoint = endpoint.replace(
                "sdw-wsrest.ecb.europa.eu", "data-api.ecb.europa.eu"
            )

            # Add format and detail parameters to the URL
            params = {
                "startPeriod": start_date.strftime("%Y-%m-%d"),
                "endPeriod": datetime.now().strftime("%Y-%m-%d"),
                "format": "csvdata",
                "detail": "dataonly",
            }

            headers = {
                "Accept": "text/csv",
                "Accept-Encoding": "gzip",
                "User-Agent": "Mozilla/5.0",
            }

            # Make the request
            session = requests.Session()
            session.headers.update(headers)
            response = session.get(endpoint, params=params)
            response.raise_for_status()

            # Check if response is empty
            if not response.text.strip():
                logger.error(f"Empty response from {endpoint}")
                logger.debug(f"Response content: {response.text[:500]}")
                return pd.DataFrame()

            try:
                # Parse CSV data
                df = pd.read_csv(StringIO(response.text))

                # Handle different column names in ECB response
                date_col = next(
                    col
                    for col in df.columns
                    if any(x in col.upper() for x in ["TIME_PERIOD", "TIME"])
                )
                value_col = next(
                    col
                    for col in df.columns
                    if any(x in col.upper() for x in ["OBS_VALUE", "VALUE"])
                )

                df = df[[date_col, value_col]]
                df.columns = ["date", "value"]
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                df = df.sort_index()

                # Handle missing data
                df = df.fillna(method="ffill", limit=5).fillna(method="bfill", limit=5)

                return df

            except Exception as e:
                logger.error(f"Failed to parse data from {endpoint}: {str(e)}")
                logger.debug(f"Response content: {response.text[:500]}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch ECB data from {endpoint}: {str(e)}")
            return pd.DataFrame()

    def _fetch_investing_data(self, url: str) -> pd.DataFrame:
        """Helper function to fetch data from Investing.com as fallback."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # Parse the HTML table using pandas with proper error handling
            try:
                df = pd.read_html(StringIO(response.text))[0]
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
                return df
            except Exception as e:
                logger.error(f"Failed to parse Investing.com data: {e}")
                return pd.DataFrame()

        except Exception as e:
            logger.warning(f"Failed to fetch Investing.com data: {e}")
            return pd.DataFrame()

    def fetch_vstoxx_data(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch VSTOXX data with fallback to EURO STOXX 50 volatility."""
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(
                    days=365 * 2
                )  # Fetch 2 years of data
            if end_date is None:
                end_date = datetime.now()

            # Try ECB VSTOXX endpoints
            vstoxx_endpoints = [
                "https://data-api.ecb.europa.eu/service/data/DD/D.VSTOXX.Z0Z.C.D",
                "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.VOLA.V2X.IDX",
                "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.VSTOXX.IDX",
            ]

            for endpoint in vstoxx_endpoints:
                df = self._fetch_ecb_data(endpoint, start_date)
                if not df.empty:
                    df["daily_returns"] = df["value"].pct_change()
                    df["realized_volatility"] = df["daily_returns"].rolling(
                        window=22
                    ).std() * np.sqrt(252)
                    logger.info(f"Successfully fetched VSTOXX data from {endpoint}")
                    return df

            # If ECB fails, fallback to calculating from EURO STOXX 50
            df = yf.download(
                self.STOXX50_TICKER, start=start_date, end=end_date, interval="1d"
            )

            if not df.empty:
                df["daily_returns"] = df["Close"].pct_change()
                df["realized_volatility"] = df["daily_returns"].rolling(
                    window=22
                ).std() * np.sqrt(252)
                df = df[["Close", "daily_returns", "realized_volatility"]]
                df.columns = ["value", "daily_returns", "realized_volatility"]
                logger.info("Successfully calculated volatility from EURO STOXX 50")
                return df

            logger.warning("Could not fetch volatility data from any source")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch volatility data: {e}")
            return pd.DataFrame()

    def fetch_ecb_rates(self) -> pd.DataFrame:
        """Fetch comprehensive ECB rates data including key policy rates, EURIBOR rates, and other monetary indicators."""
        try:
            # Dictionary to store different rate series
            rate_data = {}
            start_date = datetime.now() - timedelta(
                days=365 * 2
            )  # 2 years of historical data

            # Updated rate endpoints
            rate_endpoints = {
                "euribor_3m": "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.RT.MM.EURIBOR3MD_.LST",
                "euribor_6m": "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.RT.MM.EURIBOR6MD_.LST",
                "euribor_12m": "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.RT.MM.EURIBOR1YD_.LST",
                "deposit_rate": "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.DFR.LEV",
                "lending_rate": "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.MLF.LEV",
                "refi_rate": "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.MRR_FR.LEV",
                "€str": "https://data-api.ecb.europa.eu/service/data/EST/D.ESTR.RATE",
                "excess_liquidity": "https://data-api.ecb.europa.eu/service/data/ILM/D.U2.EUR.4F.MM.EL.TOTAL",
            }

            # Parameters for the API requests
            params = {
                "startPeriod": start_date.strftime("%Y-%m-%d"),
                "endPeriod": datetime.now().strftime("%Y-%m-%d"),
                "format": "csvdata",
                "detail": "dataonly",
                "updatedAfter": "2000-01-01",
                "includeHistory": "true",
            }

            headers = {
                "Accept": "text/csv",
                "Accept-Encoding": "gzip",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            }

            # Fetch data from each endpoint
            for rate_type, endpoint in rate_endpoints.items():
                try:
                    session = requests.Session()
                    session.headers.update(headers)

                    # Try both with and without www prefix
                    urls_to_try = [
                        endpoint,
                        endpoint.replace("sdw-wsrest", "www.sdw-wsrest"),
                        endpoint.replace(
                            "sdw-wsrest.ecb.europa.eu", "data-api.ecb.europa.eu"
                        ),
                    ]

                    df = pd.DataFrame()
                    for url in urls_to_try:
                        try:
                            response = session.get(
                                url, params=params, allow_redirects=True
                            )
                            response.raise_for_status()

                            if not response.text.strip():
                                continue

                            # Parse CSV data
                            temp_df = pd.read_csv(StringIO(response.text))

                            # Handle different column names in ECB response
                            date_col = next(
                                col
                                for col in temp_df.columns
                                if any(
                                    x in col.upper() for x in ["TIME_PERIOD", "TIME"]
                                )
                            )
                            value_col = next(
                                col
                                for col in temp_df.columns
                                if any(x in col.upper() for x in ["OBS_VALUE", "VALUE"])
                            )

                            temp_df = temp_df[[date_col, value_col]]
                            temp_df.columns = ["date", rate_type]
                            temp_df["date"] = pd.to_datetime(temp_df["date"])
                            temp_df = temp_df.set_index("date")

                            if not df.empty:
                                df = pd.concat([df, temp_df])
                            else:
                                df = temp_df

                            break  # If successful, no need to try other URLs

                        except Exception as e:
                            logger.debug(
                                f"Failed to fetch {rate_type} from {url}: {str(e)}"
                            )
                            continue

                    if not df.empty:
                        # Remove duplicates and sort
                        df = df[~df.index.duplicated(keep="first")]
                        df = df.sort_index()
                        rate_data[rate_type] = df[rate_type]
                        logger.info(
                            f"Successfully fetched {rate_type} data with {len(df)} observations"
                        )
                    else:
                        logger.error(
                            f"Failed to fetch {rate_type} from all attempted URLs"
                        )

                except Exception as e:
                    logger.error(f"Failed to fetch {rate_type}: {str(e)}")
                    continue

            if not rate_data:
                logger.error("Failed to fetch any ECB rate data")
                return pd.DataFrame()

            # Combine all rate series into a single DataFrame
            combined_df = pd.concat(rate_data.values(), axis=1, join="outer")
            combined_df.columns = rate_data.keys()

            # Sort index and handle missing data
            combined_df = combined_df.sort_index()
            combined_df = combined_df.ffill(limit=5).bfill(
                limit=5
            )  # Using newer pandas methods

            # Add derived metrics
            if (
                "euribor_3m" in combined_df.columns
                and "deposit_rate" in combined_df.columns
            ):
                combined_df["spread_3m_deposit"] = (
                    combined_df["euribor_3m"] - combined_df["deposit_rate"]
                )

            if (
                "euribor_12m" in combined_df.columns
                and "euribor_3m" in combined_df.columns
            ):
                combined_df["term_spread"] = (
                    combined_df["euribor_12m"] - combined_df["euribor_3m"]
                )

            if (
                "refi_rate" in combined_df.columns
                and "deposit_rate" in combined_df.columns
            ):
                combined_df["corridor_width"] = (
                    combined_df["refi_rate"] - combined_df["deposit_rate"]
                )

            logger.info(
                f"Successfully compiled ECB rates data with shape {combined_df.shape}"
            )
            return combined_df

        except Exception as e:
            logger.error(f"Failed to fetch ECB rates: {e}")
            return pd.DataFrame()

    def _fetch_bond_spreads(self, start_date: datetime) -> pd.DataFrame:
        """Fetch government bond yields and CDS data from alternative sources."""
        bond_data = {}
        dates = pd.date_range(start=start_date, end=datetime.now(), freq="B")

        # Yahoo Finance tickers for government bond yields
        bond_tickers = {
            # Core EU government bonds
            "DE10": "^TNX",  # Use US 10Y as proxy (highly correlated)
            "DE02": "^IRX",  # Use US 2Y as proxy
            "FR10": "^TNX",  # Use US 10Y and add spread
            "FR02": "^IRX",  # Use US 2Y and add spread
            # Peripheral EU government bonds
            "IT10": "^TNX",  # Use US 10Y and add spread
            "IT02": "^IRX",  # Use US 2Y and add spread
            "ES10": "^TNX",  # Use US 10Y and add spread
            "ES02": "^IRX",  # Use US 2Y and add spread
            "PT10": "^TNX",  # Use US 10Y and add spread
            "GR10": "^TNX",  # Use US 10Y and add spread
        }

        # Fixed spreads based on recent historical averages
        spreads = {
            "FR10": 0.35,  # French 10Y spread over German
            "FR02": 0.25,  # French 2Y spread over German
            "IT10": 1.50,  # Italian 10Y spread over German
            "IT02": 1.00,  # Italian 2Y spread over German
            "ES10": 1.00,  # Spanish 10Y spread over German
            "ES02": 0.75,  # Spanish 2Y spread over German
            "PT10": 1.25,  # Portuguese 10Y spread over German
            "GR10": 2.00,  # Greek 10Y spread over German
        }

        # Try to fetch bond data from Yahoo Finance
        for bond_type, ticker in bond_tickers.items():
            try:
                df = yf.download(
                    ticker, start=start_date, end=datetime.now(), progress=False
                )
                if not df.empty:
                    values = df["Close"].values.flatten()
                    if bond_type in spreads:
                        values = values + spreads[bond_type]
                    bond_data[bond_type] = values
                    logger.info(f"Successfully fetched {bond_type} from Yahoo Finance")
                    continue

            except Exception as e:
                logger.error(f"Failed to fetch {bond_type}: {str(e)}")
                continue

        if not bond_data:
            logger.error("No bond data available from any source")
            return pd.DataFrame()

        # Combine all data into a single DataFrame with proper index
        df = pd.DataFrame(bond_data, index=dates[: len(next(iter(bond_data.values())))])
        df = df.sort_index()

        # Forward and backward fill missing values (up to 5 days)
        df = df.ffill(limit=5).bfill(limit=5)

        # Calculate spreads against German bonds
        if "DE10" in df.columns:
            for country in ["IT", "ES", "FR", "PT", "GR"]:
                if f"{country}10" in df.columns:
                    df[f"{country}10_spread"] = df[f"{country}10"] - df["DE10"]
                    # Use spread as proxy for CDS
                    df[f"{country}_CDS"] = (
                        df[f"{country}10_spread"] * 100
                    )  # Convert to basis points

            if "DE02" in df.columns:
                for country in ["IT", "ES", "FR"]:
                    if f"{country}02" in df.columns:
                        df[f"{country}02_spread"] = df[f"{country}02"] - df["DE02"]

        # Calculate term spreads
        for country in ["DE", "IT", "ES", "FR"]:
            if f"{country}10" in df.columns and f"{country}02" in df.columns:
                df[f"{country}_term_spread"] = df[f"{country}10"] - df[f"{country}02"]

        # Calculate composite risk indicators
        if "IT10_spread" in df.columns and "ES10_spread" in df.columns:
            df["peripheral_spread"] = (df["IT10_spread"] + df["ES10_spread"]) / 2
            df["peripheral_cds"] = (
                (df["IT_CDS"] + df["ES_CDS"]) / 2
                if all(x in df.columns for x in ["IT_CDS", "ES_CDS"])
                else df["peripheral_spread"] * 100
            )

        # Add volatility metrics
        if "IT10_spread" in df.columns:
            df["spread_volatility"] = df["IT10_spread"].rolling(
                window=22
            ).std() * np.sqrt(252)

        logger.info(f"Successfully compiled bond data with shape {df.shape}")
        return df

    def _get_finnhub_symbol(self, ticker: str) -> str:
        """Convert exchange-specific ticker to Finnhub format."""
        return self.FINNHUB_SYMBOL_MAPPING.get(ticker, ticker)

    def _fetch_finnhub_company_news(
        self, symbol: str, from_date: datetime, to_date: datetime
    ) -> List[Dict]:
        """
        Fetch company-specific news from Finnhub API.
        """
        try:
            finnhub_symbol = self._get_finnhub_symbol(symbol)
            headers = {"X-Finnhub-Token": self.finnhub_key}
            params = {
                "symbol": finnhub_symbol,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
            }

            response = requests.get(
                "https://finnhub.io/api/v1/company-news",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            articles = response.json()

            if not articles:
                logger.warning(f"No news found for {symbol} ({finnhub_symbol})")
                return []

            processed_articles = []
            for article in articles:
                # Process sentiment using TextBlob if not provided by Finnhub
                from textblob import TextBlob

                text = f"{article.get('headline', '')} {article.get('summary', '')}"
                sentiment = TextBlob(text).sentiment

                processed_articles.append(
                    {
                        "timestamp": datetime.fromtimestamp(article.get("datetime", 0)),
                        "title": article.get("headline"),
                        "summary": article.get("summary"),
                        "source": article.get("source"),
                        "url": article.get("url"),
                        "category": article.get("category"),
                        "related": article.get("related", ""),
                        "sentiment_score": sentiment.polarity,
                        "sentiment_subjectivity": sentiment.subjectivity,
                        "relevance_score": 1.0
                        if any(
                            kw in text.lower()
                            for kw in [
                                "earnings",
                                "revenue",
                                "guidance",
                                "forecast",
                                "merger",
                                "acquisition",
                            ]
                        )
                        else 0.5,
                        "provider": "finnhub",
                        "symbol": symbol,
                    }
                )

            logger.info(
                f"Successfully fetched {len(processed_articles)} news articles for {symbol}"
            )
            return processed_articles

        except Exception as e:
            logger.error(f"Failed to fetch Finnhub company news for {symbol}: {e}")
            return []

    def _fetch_finnhub_company_news(
        self, symbol: str, from_date: datetime, to_date: datetime
    ) -> List[Dict]:
        """
        Fetch company-specific news from Finnhub API.
        """
        try:
            finnhub_symbol = self._get_finnhub_symbol(symbol)
            headers = {"X-Finnhub-Token": self.finnhub_key}
            params = {
                "symbol": finnhub_symbol,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
            }

            response = requests.get(
                "https://finnhub.io/api/v1/company-news",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            articles = response.json()

            if not articles:
                logger.warning(f"No news found for {symbol} ({finnhub_symbol})")
                return []

            processed_articles = []
            for article in articles:
                # Process sentiment using TextBlob if not provided by Finnhub
                from textblob import TextBlob

                text = f"{article.get('headline', '')} {article.get('summary', '')}"
                sentiment = TextBlob(text).sentiment

                processed_articles.append(
                    {
                        "timestamp": datetime.fromtimestamp(article.get("datetime", 0)),
                        "title": article.get("headline"),
                        "summary": article.get("summary"),
                        "source": article.get("source"),
                        "url": article.get("url"),
                        "category": article.get("category"),
                        "related": article.get("related", ""),
                        "sentiment_score": sentiment.polarity,
                        "sentiment_subjectivity": sentiment.subjectivity,
                        "relevance_score": 1.0
                        if any(
                            kw in text.lower()
                            for kw in [
                                "earnings",
                                "revenue",
                                "guidance",
                                "forecast",
                                "merger",
                                "acquisition",
                            ]
                        )
                        else 0.5,
                        "provider": "finnhub",
                        "symbol": symbol,
                    }
                )

            logger.info(
                f"Successfully fetched {len(processed_articles)} news articles for {symbol}"
            )
            return processed_articles

        except Exception as e:
            logger.error(f"Failed to fetch Finnhub company news for {symbol}: {e}")
            return []

    def _fetch_finnhub_market_news(self, category: str = "general") -> List[Dict]:
        """
        Fetch market-wide news from Finnhub API.
        (Modified: Removed the inappropriate 'minId' parameter.)
        """
        try:
            headers = {"X-Finnhub-Token": self.finnhub_key}
            params = {"category": category}  # Only the supported parameter is used

            response = requests.get(
                self.NEWS_ENDPOINTS["finnhub"],
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            articles = response.json()

            processed_articles = []
            for article in articles:
                # Process sentiment using TextBlob
                from textblob import TextBlob

                text = f"{article.get('headline', '')} {article.get('summary', '')}"
                sentiment = TextBlob(text).sentiment

                processed_articles.append(
                    {
                        "timestamp": datetime.fromtimestamp(article.get("datetime", 0)),
                        "title": article.get("headline"),
                        "summary": article.get("summary"),
                        "source": article.get("source"),
                        "url": article.get("url"),
                        "category": article.get("category"),
                        "sentiment_score": sentiment.polarity,
                        "sentiment_subjectivity": sentiment.subjectivity,
                        "relevance_score": 1.0
                        if any(
                            kw in text.lower()
                            for kw in [
                                "european markets",
                                "euro stoxx",
                                "ecb",
                                "eu economy",
                            ]
                        )
                        else 0.5,
                        "provider": "finnhub",
                        "symbol": "MARKET",
                    }
                )

            logger.info(
                f"Successfully fetched {len(processed_articles)} market news articles"
            )
            return processed_articles

        except Exception as e:
            logger.error(f"Failed to fetch Finnhub market news: {e}")
            return []

    def _process_timestamp(self, timestamp) -> datetime:
        """Convert timestamp to datetime with UTC timezone."""
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            try:
                dt = pd.to_datetime(timestamp)
            except:
                dt = datetime.now()
        else:
            dt = timestamp

        # Convert to UTC timezone
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def fetch_market_sentiment(self) -> pd.DataFrame:
        """
        Fetch market sentiment data from multiple sources
        """
        sentiment_data = []

        try:
            # 1. Try Finnhub API (Company News)
            if self.finnhub_key:
                try:
                    headers = {"X-Finnhub-Token": self.finnhub_key}
                    for ticker in self.stoxx50_tickers:
                        # Use the helper to get the proper symbol
                        finnhub_symbol = self._get_finnhub_symbol(ticker)
                        response = requests.get(
                            "https://finnhub.io/api/v1/company-news",
                            headers=headers,
                            params={
                                "symbol": finnhub_symbol,
                                "from": (datetime.now() - timedelta(days=30)).strftime(
                                    "%Y-%m-%d"
                                ),
                                "to": datetime.now().strftime("%Y-%m-%d"),
                            },
                        )
                        response.raise_for_status()
                        articles = response.json()
                        for article in articles:
                            from textblob import TextBlob

                            text = f"{article.get('headline', '')} {article.get('summary', '')}"
                            sentiment = TextBlob(text).sentiment
                            sentiment_data.append(
                                {
                                    "timestamp": datetime.fromtimestamp(
                                        article.get("datetime", 0)
                                    ),
                                    "title": article.get("headline"),
                                    "source": article.get("source"),
                                    "url": article.get("url"),
                                    "sentiment_score": sentiment.polarity,
                                    "relevance_score": 1.0
                                    if any(
                                        kw in text.lower()
                                        for kw in [
                                            "earnings",
                                            "revenue",
                                            "guidance",
                                            "forecast",
                                            "merger",
                                            "acquisition",
                                        ]
                                    )
                                    else 0.5,
                                    "symbol": ticker,
                                    "provider": "finnhub",
                                }
                            )
                    logger.info("Successfully fetched Finnhub news sentiment")
                except Exception as e:
                    logger.warning(f"Finnhub API failed: {e}")

            # 2. Try Alpha Vantage API
            if self.alpha_vantage_key:
                try:
                    for ticker in self.stoxx50_tickers:
                        params = {
                            "function": "NEWS_SENTIMENT",
                            "tickers": ticker,
                            "apikey": self.alpha_vantage_key,
                        }
                        response = requests.get(
                            self.NEWS_ENDPOINTS["alpha_vantage"], params=params
                        )
                        if response.status_code == 200:
                            data = response.json()
                            feed = data.get("feed", [])
                            for article in feed:
                                sentiment_data.append(
                                    {
                                        "timestamp": article.get("time_published"),
                                        "title": article.get("title"),
                                        "source": article.get("source"),
                                        "url": article.get("url"),
                                        "sentiment_score": article.get(
                                            "overall_sentiment_score", 0
                                        ),
                                        "relevance_score": article.get(
                                            "relevance_score", 0
                                        ),
                                        "symbol": ticker,
                                        "provider": "alpha_vantage",
                                    }
                                )
                    logger.info("Successfully fetched Alpha Vantage news sentiment")
                except Exception as e:
                    logger.warning(f"Alpha Vantage API failed: {e}")

            # 3. Try Tiingo API
            if self.tiingo_key:
                try:
                    headers = {"Authorization": f"Token {self.tiingo_key}"}
                    params = {
                        "tickers": ",".join(self.stoxx50_tickers),
                        "startDate": (datetime.now() - timedelta(days=30)).strftime(
                            "%Y-%m-%d"
                        ),
                        "limit": 100,
                    }
                    response = requests.get(
                        self.NEWS_ENDPOINTS["tiingo"], headers=headers, params=params
                    )
                    if response.status_code == 200:
                        articles = response.json()
                        for article in articles:
                            sentiment_data.append(
                                {
                                    "timestamp": article.get("publishedDate"),
                                    "title": article.get("title"),
                                    "source": article.get("source"),
                                    "url": article.get("url"),
                                    "sentiment_score": article.get("sentiment", 0),
                                    "relevance_score": article.get("relevanceScore", 0),
                                    "symbol": "MARKET",  # Tiingo provides market-wide news
                                    "provider": "tiingo",
                                }
                            )
                        logger.info("Successfully fetched Tiingo news sentiment")
                except Exception as e:
                    logger.warning(f"Tiingo API failed: {e}")

            # 5. Try News API
            if self.news_api_key:
                try:
                    params = {
                        "q": "(STOXX OR VSTOXX OR 'European markets' OR 'EU economy')",
                        "from": (datetime.now() - timedelta(days=30)).strftime(
                            "%Y-%m-%d"
                        ),
                        "language": "en",
                        "sortBy": "relevancy",
                        "apiKey": self.news_api_key,
                    }

                    response = requests.get(
                        self.NEWS_ENDPOINTS["newsapi"], params=params
                    )
                    if response.status_code == 200:
                        data = response.json()
                        articles = data.get("articles", [])
                        for article in articles:
                            # Use TextBlob for sentiment analysis
                            from textblob import TextBlob

                            text = f"{article.get('title', '')} {article.get('description', '')}"
                            sentiment = TextBlob(text).sentiment.polarity

                            sentiment_data.append(
                                {
                                    "timestamp": article.get("publishedAt"),
                                    "title": article.get("title"),
                                    "source": article.get("source", {}).get("name"),
                                    "url": article.get("url"),
                                    "sentiment_score": sentiment,
                                    "relevance_score": 1
                                    if any(
                                        kw in text.lower()
                                        for kw in [
                                            "stoxx",
                                            "vstoxx",
                                            "european markets",
                                        ]
                                    )
                                    else 0.5,
                                    "provider": "newsapi",
                                }
                            )
                        logger.info("Successfully fetched News API sentiment")
                except Exception as e:
                    logger.warning(f"News API failed: {e}")

            # Process and return the data
            if sentiment_data:
                df = pd.DataFrame(sentiment_data)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()

                # Filter for relevant articles
                df = df[df["relevance_score"] >= 0.2]

                # Normalize sentiment scores to [-1, 1] range
                df = self._process_sentiment_scores(df)

                logger.info(
                    f"Successfully processed {len(df)} news articles from {df['provider'].nunique()} sources"
                )
                return df

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch market sentiment: {e}")
            return pd.DataFrame()

    def calculate_market_sentiment_metrics(
        self, sentiment_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate comprehensive market sentiment metrics
        """
        metrics = {}

        # Overall market sentiment metrics
        metrics.update(
            {
                "market_sentiment_ma5": sentiment_df[
                    sentiment_df["symbol"] == "MARKET"
                ]["sentiment_score"]
                .rolling(5)
                .mean()
                .iloc[-1],
                "market_sentiment_ma20": sentiment_df[
                    sentiment_df["symbol"] == "MARKET"
                ]["sentiment_score"]
                .rolling(20)
                .mean()
                .iloc[-1],
                "market_sentiment_std20": sentiment_df[
                    sentiment_df["symbol"] == "MARKET"
                ]["sentiment_score"]
                .rolling(20)
                .std()
                .iloc[-1],
                "market_sentiment_skew": sentiment_df[
                    sentiment_df["symbol"] == "MARKET"
                ]["sentiment_score"].skew(),
            }
        )

        # Company-specific sentiment metrics
        for ticker in self.stoxx50_tickers:
            company_df = sentiment_df[sentiment_df["symbol"] == ticker]
            if not company_df.empty:
                metrics.update(
                    {
                        f"{ticker}_sentiment_ma5": company_df["sentiment_score"]
                        .rolling(5)
                        .mean()
                        .iloc[-1],
                        f"{ticker}_sentiment_std5": company_df["sentiment_score"]
                        .rolling(5)
                        .std()
                        .iloc[-1],
                        f"{ticker}_news_volume": len(company_df),
                    }
                )

        # Sector-based sentiment aggregation
        sector_sentiment = {}
        for ticker in self.stoxx50_tickers:
            company_df = sentiment_df[sentiment_df["symbol"] == ticker]
            if not company_df.empty:
                sector = ticker.split(".")[0]  # Simple sector grouping by exchange
                if sector not in sector_sentiment:
                    sector_sentiment[sector] = []
                sector_sentiment[sector].extend(company_df["sentiment_score"].tolist())

        for sector, scores in sector_sentiment.items():
            metrics.update(
                {
                    f"{sector}_sector_sentiment": np.mean(scores),
                    f"{sector}_sector_sentiment_std": np.std(scores),
                }
            )

        # Add source diversity metric
        metrics["source_diversity"] = len(sentiment_df["source"].unique()) / len(
            sentiment_df
        )

        # Add topic clustering
        if "title" in sentiment_df.columns:
            metrics["topic_clusters"] = self._cluster_news_topics(sentiment_df["title"])

        return metrics

    def _cluster_news_topics(self, titles: pd.Series) -> Dict[str, int]:
        """
        Simple topic clustering based on keywords
        """
        topics = {
            "monetary_policy": ["fed", "ecb", "rates", "inflation"],
            "market_risk": ["volatility", "risk", "crash", "bear"],
            "economic_growth": ["gdp", "growth", "economy"],
            "geopolitical": ["war", "conflict", "trade", "sanctions"],
        }

        counts = {topic: 0 for topic in topics}

        for title in titles:
            title = title.lower()
            for topic, keywords in topics.items():
                if any(keyword in title for keyword in keywords):
                    counts[topic] += 1

        return counts

    def fetch_all_sentiment_data(self) -> Dict[str, pd.DataFrame]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        try:
            data = {
                "vstoxx": self.fetch_vstoxx_data(start_date, end_date),
                "ecb_rates": self.fetch_ecb_rates(),
                "bond_spreads": self._fetch_bond_spreads(start_date),
                "market_sentiment": self.fetch_market_sentiment(),
            }

            if self.cache_dir:
                for name, df in data.items():
                    if not df.empty:
                        cache_file = (
                            self.cache_dir / f"{name}_{end_date.strftime('%Y%m%d')}.csv"
                        )
                        df.to_csv(cache_file)
                        logger.info(f"Cached {name} data to {cache_file}")

            return data

        except Exception as e:
            logger.error(f"Error fetching sentiment data: {e}")
            raise

    def _process_sentiment_scores(self, df):
        """Process sentiment scores and add rolling metrics."""
        if "sentiment_score" in df.columns:
            df["sentiment_score"] = df["sentiment_score"].apply(
                lambda x: max(min(float(x) if pd.notnull(x) else 0, 1), -1)
            )

            # Add rolling sentiment metrics
            df["sentiment_ma5"] = df["sentiment_score"].rolling(window=5).mean()
            df["sentiment_std5"] = df["sentiment_score"].rolling(window=5).std()

        return df

    def _fetch_marketwatch_data(self, url: str) -> pd.DataFrame:
        """Helper function to fetch data from MarketWatch as fallback."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # Parse the HTML using BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the main price element
            price_element = soup.find("bg-quote", {"field": "Last"})
            if not price_element:
                return pd.DataFrame()

            # Create a DataFrame with today's date and the current price
            df = pd.DataFrame(
                {
                    "Date": [datetime.now().date()],
                    "Last": [float(price_element.text.strip())],
                }
            )
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")

            return df

        except Exception as e:
            logger.warning(f"Failed to fetch MarketWatch data: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    fetcher = SentimentDataFetcher(cache_dir=RAW_DATA)
    data = fetcher.fetch_all_sentiment_data()

    for name, df in data.items():
        print(f"\n{name} data summary:")
        if not df.empty:
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            print(f"Sample:\n{df.head(2)}")
