"""
Module for fetching news data from Alpha Vantage API.
Specialized for EURO STOXX 50 tickers.
"""

import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import requests
from dotenv import load_dotenv
import time

from src.utils.log_utils import setup_logging
from src.config.paths import RAW_DATA, CACHE_DIR
from src.config.settings import TICKERS

logger = setup_logging(__name__)


class AlphaVantageNewsFetcher:
    """Class for fetching and processing news data from Alpha Vantage."""

    # ADR Mapping for EURO STOXX 50 companies
    ADR_MAPPING = {
        "LVMH": "LVMUY",
        "L'Oréal": "LRLCY",
        "TotalEnergies": "TTE",
        "Sanofi": "SNY",
        "Airbus": "EADSY",
        "Schneider Electric": "SBGSY",
        "Air Liquide": "AIQUY",
        "Hermès International": "HESAY",
        "Safran": "SAFRY",
        "EssilorLuxottica": "ESLOY",
        "BNP Paribas": "BNPQY",
        "AXA": "AXAHY",
        "Vinci": "VCISY",
        "Saint-Gobain": "CODYY",
        "Danone": "DANOY",
        "Pernod Ricard": "PDRDY",
        "Kering": "PPRUY",
        "SAP": "SAP",
        "Siemens": "SIEGY",
        "Deutsche Telekom": "DTEGY",
        "Allianz": "ALIZY",
        "Munich Re": "MURGY",
        "Deutsche Börse": "DBOEY",
        "Infineon Technologies": "IFNNY",
        "Adidas": "ADDYY",
        "BASF": "BASFY",
        "Mercedes-Benz Group": "MBGAF",
        "Deutsche Post": "DPSGY",
        "BMW": "BMWYY",
        "Bayer": "BAYRY",
        "Volkswagen": "VWAGY",
        "ASML Holding": "ASML",
        "Prosus": "PROSY",
        "ING Group": "ING",
        "Adyen": "ADYEY",
        "Wolters Kluwer": "WTKWY",
        "Ahold Delhaize": "ADRNY",
        "Iberdrola": "IBDRY",
        "Banco Santander": "SAN",
        "Inditex": "IDEXY",
        "BBVA": "BBVA",
        "Intesa Sanpaolo": "ISNPY",
        "UniCredit": "UNCRY",
        "Enel": "ENLAY",
        "Ferrari": "RACE",
        "Eni": "E",
        "Stellantis": "STLA",
        "Anheuser-Busch InBev": "BUD",
        "Nordea Bank": "NRDBY",
        "Nokia": "NOK",
    }

    # Available topics for Alpha Vantage News API
    AVAILABLE_TOPICS = [
        "blockchain",
        "earnings",
        "ipo",
        "mergers_and_acquisitions",
        "financial_markets",
        "economy_fiscal",
        "economy_monetary",
        "economy_macro",
        "energy_transportation",
        "finance",
        "life_sciences",
        "manufacturing",
        "real_estate",
        "retail_wholesale",
        "technology",
    ]

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the fetcher with API key and cache directory."""
        load_dotenv()
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set")

        self.cache_dir = cache_dir or CACHE_DIR / "alpha_vantage_news"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.base_url = "https://www.alphavantage.co/query"

        # Load EUROSTOXX list
        self.companies = self._load_eurostoxx_list()
        self.company_symbols = {}  # Will store company name to symbol mappings

    def _load_eurostoxx_list(self) -> List[Dict]:
        """Load company information from EUROSTOXX_list.json."""
        try:
            with open("EUROSTOX_list.json", "r") as f:
                data = json.load(f)
                companies = []
                for country, company_list in data.items():
                    for company in company_list:
                        company["country"] = country
                        companies.append(company)
                logger.info(f"Loaded {len(companies)} companies from EUROSTOXX list")
                return companies
        except Exception as e:
            logger.error(f"Error loading EUROSTOXX list: {str(e)}")
            return []

    def _search_company_symbol(self, company_name: str) -> Optional[str]:
        """
        Get the ADR symbol for a company.

        Args:
            company_name: Full company name to search for

        Returns:
            ADR symbol or None if not found
        """
        # First try direct mapping
        if company_name in self.ADR_MAPPING:
            symbol = self.ADR_MAPPING[company_name]
            logger.info(f"Found ADR symbol {symbol} for {company_name}")
            return symbol

        # If not found in mapping, try Alpha Vantage search
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": company_name,
            "apikey": self.api_key,
        }

        try:
            logger.info(f"Searching for symbol for company: {company_name}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if "bestMatches" in data and data["bestMatches"]:
                # Sort matches by match score
                matches = sorted(
                    data["bestMatches"],
                    key=lambda x: float(x["9. matchScore"]),
                    reverse=True,
                )

                # Get the best match
                best_match = matches[0]
                symbol = best_match["1. symbol"]
                logger.info(
                    f"Found symbol {symbol} for {company_name} (score: {best_match['9. matchScore']})"
                )
                return symbol

            logger.warning(f"No matches found for {company_name}")
            return None

        except Exception as e:
            logger.error(f"Error searching for company {company_name}: {str(e)}")
            return None

    def initialize_symbols(self):
        """Initialize symbol mappings for all companies."""
        logger.info("Initializing company symbols...")

        for company in self.companies:
            company_name = company["company"]
            if company_name not in self.company_symbols:
                symbol = self._search_company_symbol(company_name)
                if symbol:
                    self.company_symbols[company_name] = symbol
                    logger.info(f"Mapped {company_name} to symbol {symbol}")
                    # Wait to avoid hitting rate limits
                    time.sleep(12)  # 12 second delay between searches
                else:
                    logger.warning(f"Could not find symbol for {company_name}")

        logger.info(f"Initialized {len(self.company_symbols)} company symbols")
        return self.company_symbols

    def fetch_news_sentiment(
        self,
        companies: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        time_from: Optional[str] = None,
        limit: int = 1000,
        sort: str = "LATEST",
    ) -> pd.DataFrame:
        """
        Fetch news and sentiment data for specified companies.

        Args:
            companies: List of company names to fetch news for. Defaults to all EUROSTOXX companies.
            topics: List of topics to filter news by.
            time_from: Start time for news articles (format: YYYYMMDDTHHMM).
            limit: Maximum number of news items to return (default: 1000).
            sort: Sort order for news items ('LATEST', 'EARLIEST', 'RELEVANCE').

        Returns:
            DataFrame containing news and sentiment data.
        """
        # Get symbols for requested companies
        if companies:
            symbols = [self.company_symbols.get(company) for company in companies]
            symbols = [s for s in symbols if s]  # Remove None values
        else:
            symbols = list(self.company_symbols.values())

        if not symbols:
            logger.warning("No valid symbols found for the requested companies")
            return pd.DataFrame()

        # Validate topics
        if topics:
            invalid_topics = [t for t in topics if t not in self.AVAILABLE_TOPICS]
            if invalid_topics:
                logger.warning(f"Invalid topics: {invalid_topics}. Will be ignored.")
                topics = [t for t in topics if t in self.AVAILABLE_TOPICS]

        # Use default topics if none specified
        if not topics:
            topics = [
                "financial_markets",
                "economy_monetary",
                "economy_macro",
                "mergers_and_acquisitions",
                "earnings",
            ]

        # Prepare parameters
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "limit": limit,
            "sort": sort,
        }

        # Add optional parameters
        if symbols:
            params["tickers"] = ",".join(symbols)
        if topics:
            params["topics"] = ",".join(topics)
        if time_from:
            params["time_from"] = time_from

        try:
            # Make API request
            logger.info(f"Making Alpha Vantage API request with parameters: {params}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Log the full API response for debugging
            logger.debug(f"Alpha Vantage API Response: {data}")

            # Check for API error messages
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API Error: {data['Error Message']}")
                return pd.DataFrame()

            if "Information" in data:
                logger.warning(f"Alpha Vantage API Information: {data['Information']}")

            # Extract feed and metadata
            feed = data.get("feed", [])

            if not feed:
                logger.warning("No news articles found")
                logger.debug(f"API Response: {data}")
                return pd.DataFrame()

            # Process articles into a list of dictionaries
            processed_articles = []
            for article in feed:
                # Process ticker sentiment
                ticker_sentiments = {}
                for ticker_sent in article.get("ticker_sentiment", []):
                    ticker = ticker_sent.get("ticker")
                    # Find company name for this ticker
                    company_name = next(
                        (
                            name
                            for name, symbol in self.company_symbols.items()
                            if symbol == ticker
                        ),
                        ticker,
                    )
                    ticker_sentiments[company_name] = {
                        "relevance_score": float(ticker_sent.get("relevance_score", 0)),
                        "ticker_sentiment_score": float(
                            ticker_sent.get("ticker_sentiment_score", 0)
                        ),
                        "ticker_sentiment_label": ticker_sent.get(
                            "ticker_sentiment_label", ""
                        ),
                    }

                # Create article entry
                processed_article = {
                    "timestamp": article.get("time_published", ""),
                    "title": article.get("title", ""),
                    "summary": article.get("summary", ""),
                    "source": article.get("source", ""),
                    "url": article.get("url", ""),
                    "authors": article.get("authors", []),
                    "overall_sentiment_score": float(
                        article.get("overall_sentiment_score", 0)
                    ),
                    "overall_sentiment_label": article.get(
                        "overall_sentiment_label", ""
                    ),
                    "topics": article.get("topics", []),
                    "ticker_sentiments": ticker_sentiments,
                }
                processed_articles.append(processed_article)

            # Convert to DataFrame
            df = pd.DataFrame(processed_articles)

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()

            # Cache the results
            self._cache_results(df)

            logger.info(f"Successfully fetched {len(df)} news articles")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch news from Alpha Vantage: {str(e)}")
            return pd.DataFrame()

    def _cache_results(self, df: pd.DataFrame) -> None:
        """Cache the fetched results."""
        if not df.empty:
            cache_file = (
                self.cache_dir
                / f"news_data_{datetime.now().strftime('%Y%m%d_%H%M')}.parquet"
            )
            df.to_parquet(cache_file)
            logger.info(f"Cached news data to {cache_file}")

            # Clean up old cache files
            cache_files = list(self.cache_dir.glob("news_data_*.parquet"))
            if len(cache_files) > 10:  # Keep only last 10 cache files
                oldest_files = sorted(cache_files, key=lambda x: x.stat().st_mtime)[
                    :-10
                ]
                for file in oldest_files:
                    file.unlink()
                    logger.info(f"Removed old cache file: {file}")


if __name__ == "__main__":
    # Example usage
    fetcher = AlphaVantageNewsFetcher()

    # Initialize symbols for all companies
    company_symbols = fetcher.initialize_symbols()

    # Print found symbols
    print("\nFound symbols for companies:")
    for company, symbol in company_symbols.items():
        print(f"{company}: {symbol}")

    # Test with a single company
    test_company = "LVMH"  # Test with LVMH
    if test_company in company_symbols:
        logger.info(f"Testing news fetch with {test_company}")

        # Try fetching news
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)

        news_df = fetcher.fetch_news_sentiment(
            companies=[test_company],
            topics=[
                "financial_markets",
                "mergers_and_acquisitions",
                "earnings",
                "economy_macro",
                "finance",
            ],
            time_from=start_date.strftime("%Y%m%dT%H%M"),
            limit=50,
        )

        if not news_df.empty:
            print(f"\nFetched {len(news_df)} news articles for {test_company}")
            print("\nSample of news data:")
            print(news_df.head(2))

            # Save to CSV for inspection
            output_file = (
                RAW_DATA / f"market_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            news_df.to_csv(output_file)
            print(f"\nSaved results to {output_file}")
        else:
            print(f"\nNo articles found for {test_company}")
    else:
        print(f"\nNo symbol found for {test_company}")
