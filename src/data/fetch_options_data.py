"""
Retrieve Historical Options Data for EUROSTOXX 50 stocks and index.

This module fetches historical options data including:
- Strike prices and expiration dates
- Implied volatility (IV) surface
- Greeks (delta, gamma, theta, vega, rho)
- Bid-Ask spreads for liquidity modeling
- Option chains sorted by expiration dates and strike prices

Data Sources:
1. Eurex Exchange GraphQL API (Primary for EUROSTOXX options)
2. Yahoo Finance (ADRs and US-listed options)
3. EOD Historical Data (Coming soon)
4. Alpha Vantage (Coming soon)
5. Twelve Data (Coming soon)
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import requests
import json
import time

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

from src.config.paths import RAW_OPTIONS, PROCESSED_OPTIONS
from src.config.settings import TICKERS
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)

# Update data source endpoints
EUREX_API_URL = (
    "https://api.developer.deutsche-boerse.com/prod/accesstot7data-1-0/1.0.0/graphql"
)
EOD_API_URL = "https://eodhistoricaldata.com/api"  # EOD Historical Data API endpoint
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"  # Alpha Vantage API endpoint
TWELVE_DATA_URL = "https://api.twelvedata.com"  # Twelve Data API endpoint

# Update ADR mappings for Yahoo Finance with more accurate symbols
ADR_MAPPINGS = {
    # Major ADRs with confirmed options availability
    "ASML.AS": "ASML",  # ASML Holding N.V.
    "SAP.DE": "SAP",  # SAP SE
    "TTE.PA": "TTE",  # TotalEnergies SE
    "STLA.PA": "STLA",  # Stellantis N.V.
    "DTEGY": "DTEGY",  # Deutsche Telekom AG
    "DB.DE": "DB",  # Deutsche Bank AG
    "CS.PA": "CS",  # AXA SA
    "UL": "UL",  # Unilever PLC
    # Technology & Semiconductors
    "STM.PA": "STM",  # STMicroelectronics N.V.
    "NOKIA.HE": "NOK",  # Nokia Corporation
    # Financial Services
    "SAN.MC": "SAN",  # Banco Santander
    "BBVA.MC": "BBVA",  # BBVA
    "ING.AS": "ING",  # ING Groep
    # Energy & Utilities
    "ENI.MI": "E",  # Eni SpA
    "ENEL.MI": "ENEL",  # Enel SpA
    "IBE.MC": "IBE",  # Iberdrola
    # Industrial & Materials
    "SIE.DE": "SIEGY",  # Siemens AG
    "ALV.DE": "ALVG",  # Allianz SE
    "BAS.DE": "BASFY",  # BASF SE
    # Consumer & Retail
    "BMW.DE": "BMWYY",  # BMW AG
    "VOW3.DE": "VWAGY",  # Volkswagen AG
    "MC.PA": "LVMH",  # LVMH Moët Hennessy
    # Healthcare & Pharma
    "SAN.PA": "SNY",  # Sanofi
    "BAYN.DE": "BAYRY",  # Bayer AG
}

# EURO STOXX 50 Index Options Configuration
STOXX50_CONFIG = {
    "symbol": "OESX",  # Eurex product ID for EURO STOXX 50 Index Options
    "product_isin": "DE0009652396",  # Product ISIN
    "underlying_isin": "EU0009658145",  # Underlying ISIN
    "currency": "EUR",
    "bloomberg_ticker": "SX5E Index OMON",
    "refinitiv_ric": "<0#STXE*.EX>",
}

# Update Eurex API endpoint and product IDs
EUREX_PRODUCT_QUERY = """
query GetProducts($productId: String, $isin: String) {
  products(filter: { productId: $productId, isin: $isin }) {
    data {
      productId
      productType
      isin
      symbol
      name
      currency
      productAssignmentGroup
      underlyingType
      underlyingSymbol
    }
  }
}
"""

EUREX_CONTRACT_QUERY = """
query GetContracts($productId: String!, $isin: String) {
  contracts(productId: $productId, filter: { isin: $isin }) {
    data {
      contractId
      isin
      expirationDate
      strikePrice
      callPut
      contractSize
      currency
      exerciseStyle
      exercisePrice
      settlementType
      lastTradingDate
      firstTradingDate
      contractCycle
    }
  }
}
"""

# EOD Historical Data symbol mappings
EOD_SYMBOL_MAPPINGS = {
    "^STOXX50E": "ESTX50.INDX",  # EURO STOXX 50 index
    "ADYEN.AS": "ADYEN.AS",  # Adyen
    "AIR.PA": "AIR.PA",  # Airbus
    "ALV.DE": "ALV.DE",  # Allianz
    "ASML.AS": "ASML.AS",  # ASML
    "BAS.DE": "BAS.DE",  # BASF
    "BAYN.DE": "BAYN.DE",  # Bayer
    "BMW.DE": "BMW.DE",  # BMW
    "BNP.PA": "BNP.PA",  # BNP Paribas
    "CS.PA": "ACA.PA",  # Credit Agricole
    "DTE.DE": "DTE.DE",  # Deutsche Telekom
    "ENEL.MI": "ENEL.MI",  # Enel
    "ENI.MI": "ENI.MI",  # Eni
    "IBE.MC": "IBE.MC",  # Iberdrola
    "INGA.AS": "INGA.AS",  # ING Group
    "ISP.MI": "ISP.MI",  # Intesa Sanpaolo
    "MC.PA": "MC.PA",  # LVMH
    "MUV2.DE": "MUV2.DE",  # Munich Re
    "OR.PA": "ORA.PA",  # Orange
    "PHIA.AS": "PHIA.AS",  # Philips
    "SAF.PA": "SAF.PA",  # Safran
    "SAN.MC": "SAN.MC",  # Banco Santander
    "SAP.DE": "SAP.DE",  # SAP
    "SIE.DE": "SIE.DE",  # Siemens
    "TTE.PA": "TTE.PA",  # TotalEnergies
    "VOW3.DE": "VOW3.DE",  # Volkswagen
    # Additional components
    "ABI.BR": "ABI.BR",  # AB InBev
    "AD.AS": "AD.AS",  # Ahold Delhaize
    "AI.PA": "AI.PA",  # Air Liquide
    "CRH.IR": "CRH.IR",  # CRH
    "DB1.DE": "DB1.DE",  # Deutsche Boerse
    "DPW.DE": "DPW.DE",  # Deutsche Post
    "EL.PA": "EL.PA",  # EssilorLuxottica
    "FLTR.IR": "FLTR.IR",  # Flutter Entertainment
    "HEI.DE": "HEI.DE",  # HeidelbergCement
    "HEIA.AS": "HEIA.AS",  # Heineken
    "ITX.MC": "ITX.MC",  # Inditex
    "KER.PA": "KER.PA",  # Kering
    "LIN.DE": "LIN.DE",  # Linde
    "MBG.DE": "MBG.DE",  # Mercedes-Benz
    "NOKIA.HE": "NOKIA.HE",  # Nokia
    "PUB.PA": "PUB.PA",  # Publicis
    "RMS.PA": "RMS.PA",  # Hermes
    "RWE.DE": "RWE.DE",  # RWE
    "SU.PA": "SU.PA",  # Schneider Electric
    "STM.PA": "STM.PA",  # STMicroelectronics
    "TEF.MC": "TEF.MC",  # Telefonica
    "VNA.DE": "VNA.DE",  # Vonovia
    "VWS.CO": "VWS.CO",  # Vestas Wind Systems
}

# Alpha Vantage symbol mappings (using US options where available)
ALPHA_VANTAGE_MAPPINGS = {
    "^STOXX50E": "ESTX50",  # EURO STOXX 50 index
    # Technology
    "ASML.AS": "ASML",  # ASML ADR
    "SAP.DE": "SAP",  # SAP ADR
    "STM.PA": "STM",  # STMicroelectronics ADR
    "NOKIA.HE": "NOK",  # Nokia ADR
    # Energy & Utilities
    "TTE.PA": "TTE",  # TotalEnergies ADR
    "ENEL.MI": "ENLAY",  # Enel ADR
    "ENI.MI": "E",  # ENI ADR
    "IBE.MC": "IBDRY",  # Iberdrola ADR
    # Financial Services
    "SAN.MC": "SAN",  # Santander ADR
    "BBVA.MC": "BBVA",  # BBVA ADR
    "ING.AS": "ING",  # ING Group ADR
    "ACA.PA": "CRARY",  # Credit Agricole ADR
    "DB1.DE": "DBOEY",  # Deutsche Boerse ADR
    # Telecommunications
    "TEF.MC": "TEF",  # Telefonica ADR
    "DTE.DE": "DTEGY",  # Deutsche Telekom ADR
    "ORA.PA": "ORAN",  # Orange ADR
    # Industrial & Materials
    "SIE.DE": "SIEGY",  # Siemens ADR
    "ALV.DE": "ALIZY",  # Allianz ADR
    "BAS.DE": "BASFY",  # BASF ADR
    "AI.PA": "AIQUY",  # Air Liquide ADR
    # Consumer & Retail
    "OR.PA": "LRLCY",  # L'Oreal ADR
    "MC.PA": "LVMUY",  # LVMH ADR
    "AD.AS": "ADRNY",  # Ahold Delhaize ADR
    "BMW.DE": "BMWYY",  # BMW ADR
    "VOW3.DE": "VWAGY",  # Volkswagen ADR
    # Healthcare & Pharma
    "SAN.PA": "SNY",  # Sanofi ADR
    "BAYN.DE": "BAYRY",  # Bayer ADR
    # Additional Components
    "ABI.BR": "BUD",  # AB InBev (Direct US Listing)
    "FLTR.IR": "PDYPY",  # Flutter Entertainment ADR
    "RMS.PA": "HESAY",  # Hermes ADR
    "KER.PA": "PPRUY",  # Kering ADR
    "ITX.MC": "IDEXY",  # Inditex ADR
    "MBG.DE": "MBGYY",  # Mercedes-Benz ADR
    "VNA.DE": "VONOY",  # Vonovia ADR
}

# Twelve Data symbol mappings (using exchange suffixes)
TWELVE_DATA_MAPPINGS = {
    "^STOXX50E": "ESTX50.EUR",  # EURO STOXX 50 index
    # Technology
    "ASML.AS": "ASML.AMS",  # ASML
    "SAP.DE": "SAP.XETRA",  # SAP
    "STM.PA": "STM.PAR",  # STMicroelectronics
    "NOKIA.HE": "NOKIA.HEL",  # Nokia
    # Energy & Utilities
    "TTE.PA": "TTE.PAR",  # TotalEnergies
    "ENEL.MI": "ENEL.MI",  # Enel
    "ENI.MI": "ENI.MI",  # ENI
    "IBE.MC": "IBE.MC",  # Iberdrola
    # Financial Services
    "SAN.MC": "SAN.MC",  # Santander
    "BBVA.MC": "BBVA.MC",  # BBVA
    "ING.AS": "INGA.AMS",  # ING Group
    "ACA.PA": "ACA.PAR",  # Credit Agricole
    "DB1.DE": "DB1.XETRA",  # Deutsche Boerse
    # Telecommunications
    "TEF.MC": "TEF.MC",  # Telefonica
    "DTE.DE": "DTE.XETRA",  # Deutsche Telekom
    "ORA.PA": "ORA.PAR",  # Orange
    # Industrial & Materials
    "SIE.DE": "SIE.XETRA",  # Siemens
    "ALV.DE": "ALV.XETRA",  # Allianz
    "BAS.DE": "BAS.XETRA",  # BASF
    "AI.PA": "AI.PAR",  # Air Liquide
    # Consumer & Retail
    "OR.PA": "OR.PAR",  # L'Oreal
    "MC.PA": "MC.PAR",  # LVMH
    "AD.AS": "AD.AMS",  # Ahold Delhaize
    "BMW.DE": "BMW.XETRA",  # BMW
    "VOW3.DE": "VOW3.XETRA",  # Volkswagen
    # Healthcare & Pharma
    "SAN.PA": "SAN.PAR",  # Sanofi
    "BAYN.DE": "BAYN.XETRA",  # Bayer
    # Additional Components
    "ABI.BR": "ABI.BR",  # AB InBev
    "FLTR.IR": "FLTR.IR",  # Flutter Entertainment
    "RMS.PA": "RMS.PAR",  # Hermes
    "KER.PA": "KER.PAR",  # Kering
    "ITX.MC": "ITX.MC",  # Inditex
    "MBG.DE": "MBG.XETRA",  # Mercedes-Benz
    "VNA.DE": "VNA.XETRA",  # Vonovia
}

# Update symbol mappings for better coverage
SYMBOL_MAPPINGS = {
    # German stocks with ADR symbols
    "SIE.DE": "SIEGY",
    "ALV.DE": "ALIZY",
    "MUV2.DE": "MURGY",
    "DB1.DE": "DBOEY",
    "IFX.DE": "IFNNY",
    "ADS.DE": "ADDYY",
    "BAS.DE": "BASFY",
    "MBG.DE": "MBGYY",
    "DHL.DE": "DPSGY",
    "BMW.DE": "BMWYY",
    "BAYN.DE": "BAYRY",
    "VOW3.DE": "VWAGY",
    # French stocks with better mappings
    "MC.PA": "LVMUY",
    "RMS.PA": "HESAY",
    "SAF.PA": "SAFRY",
    "SGO.PA": "SGBLY",
    "BNP.PA": "BNPQY",
    "CS.PA": "AXAHY",
    "KER.PA": "PPRUY",
    # Keep existing mappings that work
    "OR.PA": "OR",
    "TTE.PA": "TTE",
    "SAN.PA": "SNY",
    "AIR.PA": "AIR",
    "SU.PA": "SU",
    "AI.PA": "AI",
    "BN.PA": "BN",
    "DG.PA": "DG",
    "EL.PA": "EL",
    "SAP.DE": "SAP",
    "DTE.DE": "DTE",
}


class OptionsDataFetcher:
    def __init__(self):
        """Initialize the options data fetcher with multiple data sources."""
        self.output_dir_raw = RAW_OPTIONS
        self.output_dir_processed = PROCESSED_OPTIONS

        # Load environment variables
        load_dotenv()
        self.eurex_api_key = os.getenv("EUREX_API_KEY")
        self.eod_api_key = os.getenv("EOD_API_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.twelve_data_key = os.getenv("TWELVE_DATA_API_KEY")

        # Set up clients
        self.eurex_client = self._setup_eurex_client()
        self.session = requests.Session()

        # Create output directories
        self.output_dir_raw.mkdir(parents=True, exist_ok=True)
        self.output_dir_processed.mkdir(parents=True, exist_ok=True)

    def _setup_eurex_client(self) -> Optional[Client]:
        """Set up the Eurex GraphQL client."""
        if not self.eurex_api_key:
            logger.warning("No Eurex API key found in environment variables")
            return None

        try:
            transport = RequestsHTTPTransport(
                url=EUREX_API_URL,
                headers={
                    "X-DBP-APIKEY": self.eurex_api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "Python/3.12",
                    "X-DBP-CONSUMER-ID": "eurex_options_fetcher",
                    "X-DBP-CONSUMER-VERSION": "1.0.0",
                    "Authorization": f"Bearer {self.eurex_api_key}",
                },
                verify=True,
                retries=3,
            )
            client = Client(transport=transport, fetch_schema_from_transport=True)
            logger.info("Successfully set up Eurex client")
            return client
        except Exception as e:
            logger.error(f"Failed to set up Eurex client: {e}")
            return None

    def _get_yahoo_symbol(self, symbol: str) -> str:
        """Get the Yahoo Finance symbol for a given exchange symbol."""
        # For STOXX50E index, use the correct Yahoo symbol
        if symbol == "^STOXX50E":
            return "^STOXX50E"  # Yahoo Finance uses this exact symbol

        # First check if we have a direct mapping
        if symbol in SYMBOL_MAPPINGS:
            return SYMBOL_MAPPINGS[symbol]

        # For other symbols, try to use the base symbol without exchange suffix
        if "." in symbol:
            base_symbol = symbol.split(".")[0]
            # Check if we have a mapping for the base symbol
            if base_symbol in SYMBOL_MAPPINGS:
                return SYMBOL_MAPPINGS[base_symbol]
            return base_symbol

        return symbol

    def _interpolate_series(self, series: pd.Series) -> pd.Series:
        """Safely interpolate a series without warnings."""
        return series.interpolate(method="linear")

    def _fetch_yfinance_options(
        self, symbol: str, max_retries: int = 3, retry_delay: int = 2
    ) -> pd.DataFrame:
        """Fetch options data from Yahoo Finance with improved error handling."""
        yahoo_symbol = self._get_yahoo_symbol(symbol)
        logger.info(f"Fetching options for {symbol} using Yahoo symbol: {yahoo_symbol}")

        attempts = 0
        last_error = None

        while attempts < max_retries:
            try:
                # Get the stock info
                stock = yf.Ticker(yahoo_symbol)

                # Special handling for EURO STOXX 50 index
                if symbol == "^STOXX50E":
                    try:
                        # Get index value
                        index_value = stock.history(period="1d")["Close"].iloc[-1]
                        logger.info(f"Current EURO STOXX 50 index value: {index_value}")
                    except Exception as e:
                        logger.warning(
                            f"Could not fetch EURO STOXX 50 index value: {str(e)}"
                        )
                        index_value = None

                # Get current price with retry and fallback
                try:
                    current_price = stock.info.get("regularMarketPrice")
                    if not current_price:
                        current_price = stock.history(period="1d")["Close"].iloc[-1]
                    if current_price:
                        logger.info(
                            f"Current price for {yahoo_symbol}: {current_price}"
                        )
                    else:
                        raise ValueError("Could not get current price")
                except Exception as e:
                    logger.warning(
                        f"Could not fetch current price for {yahoo_symbol}: {str(e)}"
                    )
                    current_price = None

                # Get expiration dates
                exp_dates = stock.options

                if not exp_dates:
                    logger.warning(f"No expiration dates found for {yahoo_symbol}")
                    attempts += 1
                    if attempts < max_retries:
                        time.sleep(min(retry_delay * (1.5**attempts), 10))
                        continue
                    else:
                        break

                logger.info(
                    f"Found {len(exp_dates)} expiration dates for {yahoo_symbol}"
                )

                # Initialize list to store all options
                all_options = []
                successful_dates = 0

                # Process each expiration date
                for exp_date in exp_dates:
                    try:
                        # Get options chain for this expiration
                        opt = stock.option_chain(exp_date)

                        # Process calls
                        calls = opt.calls.copy()
                        calls["option_type"] = "call"
                        calls["expiration"] = pd.to_datetime(exp_date)

                        # Process puts
                        puts = opt.puts.copy()
                        puts["option_type"] = "put"
                        puts["expiration"] = pd.to_datetime(exp_date)

                        # Add moneyness if we have current price
                        if current_price is not None:
                            calls["moneyness"] = calls["strike"] / current_price - 1
                            puts["moneyness"] = puts["strike"] / current_price - 1

                        # Add days to expiry
                        for chain in [calls, puts]:
                            chain["days_to_expiry"] = (
                                pd.to_datetime(exp_date) - pd.Timestamp.now()
                            ).days

                            # Fill missing implied volatilities
                            if "impliedVolatility" in chain.columns:
                                chain["impliedVolatility"] = self._interpolate_series(
                                    chain["impliedVolatility"]
                                )

                            # Add bid-ask spread metrics
                            if all(col in chain.columns for col in ["bid", "ask"]):
                                chain["bid_ask_spread"] = chain["ask"] - chain["bid"]
                                chain["bid_ask_spread_pct"] = (
                                    chain["ask"] - chain["bid"]
                                ) / ((chain["bid"] + chain["ask"]) / 2)

                        # Combine calls and puts
                        options = pd.concat([calls, puts])

                        # Add EURO STOXX 50 specific information if applicable
                        if symbol == "^STOXX50E":
                            options["index_value"] = index_value
                            options["underlying_isin"] = STOXX50_CONFIG[
                                "underlying_isin"
                            ]
                            options["currency"] = STOXX50_CONFIG["currency"]

                        all_options.append(options)
                        successful_dates += 1

                    except Exception as e:
                        logger.warning(
                            f"Error processing expiration date {exp_date} for {yahoo_symbol}: {str(e)}"
                        )
                        continue

                logger.info(
                    f"Successfully processed {successful_dates}/{len(exp_dates)} expiration dates for {yahoo_symbol}"
                )

                if not all_options:
                    raise ValueError("No options data collected")

                # Combine all options into a single DataFrame
                df = pd.concat(all_options, ignore_index=True)

                # Add metadata
                df["symbol"] = symbol
                df["yahoo_symbol"] = yahoo_symbol
                df["data_source"] = "yahoo"
                df["timestamp"] = pd.Timestamp.now()

                # Add EURO STOXX 50 specific metadata
                if symbol == "^STOXX50E":
                    df["bloomberg_ticker"] = STOXX50_CONFIG["bloomberg_ticker"]
                    df["refinitiv_ric"] = STOXX50_CONFIG["refinitiv_ric"]
                    df["product_isin"] = STOXX50_CONFIG["product_isin"]

                # Validate the data
                is_valid, error_msg = self._validate_options_data(df)
                if not is_valid:
                    logger.warning(
                        f"Data validation failed for {yahoo_symbol}: {error_msg}"
                    )
                    attempts += 1
                    if attempts < max_retries:
                        time.sleep(min(retry_delay * (1.5**attempts), 10))
                        continue
                    else:
                        break

                logger.info(
                    f"Successfully fetched {len(df)} options contracts for {yahoo_symbol}"
                )
                return df

            except Exception as e:
                last_error = e
                attempts += 1
                if attempts < max_retries:
                    time.sleep(min(retry_delay * (1.5**attempts), 10))
                    continue
                else:
                    break

        # If we get here, all retries failed
        raise ValueError(
            f"Failed to fetch options data after {max_retries} attempts. Last error: {str(last_error)}"
        )

    def _fetch_eurex_options(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch options data from Eurex with improved error handling and validation."""
        if not self.eurex_api_key:
            logger.debug(f"Skipping Eurex fetch for {symbol} - no API key available")
            return None

        # Special handling for EURO STOXX 50 index
        if symbol == "^STOXX50E":
            eurex_symbol = STOXX50_CONFIG["symbol"]
            product_isin = STOXX50_CONFIG["product_isin"]
        else:
            # Convert symbol to Eurex format
            eurex_symbol = symbol
            if symbol.endswith(".DE"):
                eurex_symbol = "O" + symbol[:-3]  # German stock options
            elif symbol.endswith(".AS"):
                eurex_symbol = "O" + symbol[:-3]  # Dutch stock options
            elif symbol.endswith(".PA"):
                eurex_symbol = "O" + symbol[:-3]  # French stock options
            elif symbol.endswith(".MC"):
                eurex_symbol = "O" + symbol[:-3]  # Spanish stock options
            elif symbol.endswith(".MI"):
                eurex_symbol = "O" + symbol[:-3]  # Italian stock options
            product_isin = None

        logger.debug(f"Converting {symbol} to Eurex symbol: {eurex_symbol}")

        try:
            # Set up headers with API key
            headers = {
                "X-DBP-APIKEY": self.eurex_api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Python/3.12",
                "X-DBP-CONSUMER-ID": "eurex_options_fetcher",
                "X-DBP-CONSUMER-VERSION": "1.0.0",
            }

            # First get products with both productId and ISIN
            products_query = {
                "query": EUREX_PRODUCT_QUERY,
                "variables": {"productId": eurex_symbol, "isin": product_isin},
                "operationName": "GetProducts",
            }

            response = requests.post(
                EUREX_API_URL, json=products_query, headers=headers
            )
            response.raise_for_status()
            products_result = response.json()

            logger.debug(f"Products response: {json.dumps(products_result, indent=2)}")

            if "errors" in products_result:
                logger.warning(
                    f"GraphQL errors in products query: {json.dumps(products_result['errors'], indent=2)}"
                )
                return None

            # Find matching product
            matching_product = None
            for product in (
                products_result.get("data", {}).get("products", {}).get("data", [])
            ):
                if (
                    product.get("productId") == eurex_symbol
                    or product.get("isin") == product_isin
                ):
                    matching_product = product
                    break

            if not matching_product:
                logger.debug(f"No matching Eurex product found for {eurex_symbol}")
                return None

            # Get contracts for the product with ISIN if available
            contracts_query = {
                "query": EUREX_CONTRACT_QUERY,
                "variables": {
                    "productId": matching_product["productId"],
                    "isin": product_isin,
                },
                "operationName": "GetContracts",
            }

            response = requests.post(
                EUREX_API_URL, json=contracts_query, headers=headers
            )
            response.raise_for_status()
            contracts_result = response.json()

            logger.debug(
                f"Contracts response: {json.dumps(contracts_result, indent=2)}"
            )

            if "errors" in contracts_result:
                logger.warning(
                    f"GraphQL errors in contracts query: {json.dumps(contracts_result['errors'], indent=2)}"
                )
                return None

            # Process contracts into DataFrame with enhanced information
            options_list = []
            for contract in (
                contracts_result.get("data", {}).get("contracts", {}).get("data", [])
            ):
                try:
                    contract_data = {
                        "symbol": symbol,
                        "strike": float(contract.get("strikePrice", 0)),
                        "expiration": pd.to_datetime(contract["expirationDate"]),
                        "contract_name": contract.get("contractId", ""),
                        "isin": contract.get("isin", ""),
                        "option_type": contract.get("callPut", "").lower(),
                        "contract_size": float(contract.get("contractSize", 0)),
                        "currency": contract.get("currency", ""),
                        "exercise_style": contract.get("exerciseStyle", ""),
                        "exercise_price": float(contract.get("exercisePrice", 0)),
                        "settlement_type": contract.get("settlementType", ""),
                        "last_trading_date": pd.to_datetime(
                            contract.get("lastTradingDate")
                        ),
                        "first_trading_date": pd.to_datetime(
                            contract.get("firstTradingDate")
                        ),
                        "contract_cycle": contract.get("contractCycle", ""),
                        "timestamp": datetime.now(),
                    }
                    options_list.append(contract_data)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error processing contract data: {e}")
                    continue

            if not options_list:
                logger.warning(f"No valid options contracts found for {symbol}")
                return None

            df = pd.DataFrame(options_list)

            # Add metadata
            df["source"] = "eurex"
            df["fetch_timestamp"] = datetime.now()
            if symbol == "^STOXX50E":
                df["underlying_isin"] = STOXX50_CONFIG["underlying_isin"]
                df["product_isin"] = STOXX50_CONFIG["product_isin"]
                df["bloomberg_ticker"] = STOXX50_CONFIG["bloomberg_ticker"]
                df["refinitiv_ric"] = STOXX50_CONFIG["refinitiv_ric"]

            logger.info(
                f"Successfully fetched {len(df)} options contracts from Eurex for {symbol}"
            )
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Eurex options for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing Eurex options for {symbol}: {e}")
            return None

    # def _fetch_eod_options(self, symbol: str) -> Optional[pd.DataFrame]:
    #     """Fetch options data from EOD Historical Data.

    #     Args:
    #         symbol: Stock symbol

    #     Returns:
    #         DataFrame with options data or None if fetch fails
    #     """
    #     if not self.eod_api_key:
    #         logger.warning("No EOD Historical Data API key found")
    #         return None

    #     if symbol not in EOD_SYMBOL_MAPPINGS:
    #         logger.debug(f"No EOD mapping found for symbol: {symbol}")
    #         return None

    #     try:
    #         eod_symbol = EOD_SYMBOL_MAPPINGS[symbol]
    #         params = {"api_token": self.eod_api_key, "fmt": "json"}

    #         # Fetch options chain
    #         response = self.session.get(
    #             f"{EOD_API_URL}/options/{eod_symbol}", params=params
    #         )
    #         response.raise_for_status()

    #         data = response.json()
    #         if not data or "data" not in data:
    #             logger.warning(f"No options data returned from EOD for {symbol}")
    #             return None

    #         # Process options data into DataFrame
    #         options_list = []
    #         for contract in data["data"]:
    #             contract_data = {
    #                 "symbol": symbol,
    #                 "strike": float(contract["strike"]),
    #                 "expiration": pd.to_datetime(contract["expiration"]),
    #                 "option_type": contract["type"].lower(),
    #                 "impliedVolatility": float(contract["impliedVolatility"]),
    #                 "volume": int(contract["volume"]),
    #                 "openInterest": int(contract["openInterest"]),
    #                 "bid": float(contract["bid"]),
    #                 "ask": float(contract["ask"]),
    #                 "last": float(contract["last"]),
    #                 "delta": float(contract.get("delta", 0)),
    #                 "gamma": float(contract.get("gamma", 0)),
    #                 "theta": float(contract.get("theta", 0)),
    #                 "vega": float(contract.get("vega", 0)),
    #                 "timestamp": datetime.now(),
    #             }
    #             options_list.append(contract_data)

    #         if not options_list:
    #             logger.warning(f"No valid options contracts found for {symbol}")
    #             return None

    #         df = pd.DataFrame(options_list)

    #         # Validate the data
    #         is_valid, error_msg = self._validate_options_data(df)
    #         if not is_valid:
    #             logger.warning(f"Invalid options data for {symbol}: {error_msg}")
    #             return None

    #         logger.info(
    #             f"Successfully fetched {len(df)} options contracts from EOD for {symbol}"
    #         )
    #         return df

    #     except Exception as e:
    #         logger.error(f"Error fetching EOD options for {symbol}: {str(e)}")
    #         return None

    # def _fetch_alpha_vantage_options(self, symbol: str) -> Optional[pd.DataFrame]:
    #     """Fetch options data from Alpha Vantage.

    #     Args:
    #         symbol: Stock symbol

    #     Returns:
    #         DataFrame with options data or None if fetch fails
    #     """
    #     if not self.alpha_vantage_key:
    #         logger.warning("No Alpha Vantage API key found")
    #         return None

    #     if symbol not in ALPHA_VANTAGE_MAPPINGS:
    #         logger.debug(f"No Alpha Vantage mapping found for symbol: {symbol}")
    #         return None

    #     try:
    #         av_symbol = ALPHA_VANTAGE_MAPPINGS[symbol]
    #         params = {
    #             "function": "OPTIONS_CHAIN",
    #             "symbol": av_symbol,
    #             "apikey": self.alpha_vantage_key,
    #             "datatype": "json",
    #         }

    #         # Fetch options chain
    #         response = self.session.get(ALPHA_VANTAGE_URL, params=params)
    #         response.raise_for_status()

    #         data = response.json()
    #         if not data or "options" not in data:
    #             logger.warning(
    #                 f"No options data returned from Alpha Vantage for {symbol}"
    #             )
    #             return None

    #         # Process options data into DataFrame
    #         options_list = []
    #         for contract in data["options"]:
    #             contract_data = {
    #                 "symbol": symbol,
    #                 "strike": float(contract["strikePrice"]),
    #                 "expiration": pd.to_datetime(contract["expirationDate"]),
    #                 "option_type": contract["optionType"].lower(),
    #                 "impliedVolatility": float(contract["impliedVolatility"]),
    #                 "volume": int(contract.get("volume", 0)),
    #                 "openInterest": int(contract.get("openInterest", 0)),
    #                 "bid": float(contract.get("bid", 0)),
    #                 "ask": float(contract.get("ask", 0)),
    #                 "last": float(contract.get("lastPrice", 0)),
    #                 "delta": float(contract.get("delta", 0)),
    #                 "gamma": float(contract.get("gamma", 0)),
    #                 "theta": float(contract.get("theta", 0)),
    #                 "vega": float(contract.get("vega", 0)),
    #                 "timestamp": datetime.now(),
    #             }
    #             options_list.append(contract_data)

    #         if not options_list:
    #             logger.warning(f"No valid options contracts found for {symbol}")
    #             return None

    #         df = pd.DataFrame(options_list)

    #         # Validate the data
    #         is_valid, error_msg = self._validate_options_data(df)
    #         if not is_valid:
    #             logger.warning(f"Invalid options data for {symbol}: {error_msg}")
    #             return None

    #         logger.info(
    #             f"Successfully fetched {len(df)} options contracts from Alpha Vantage for {symbol}"
    #         )
    #         return df

    #     except Exception as e:
    #         logger.error(f"Error fetching Alpha Vantage options for {symbol}: {str(e)}")
    #         return None

    # def _fetch_twelve_data_options(self, symbol: str) -> Optional[pd.DataFrame]:
    #     """Fetch options data from Twelve Data.

    #     Args:
    #         symbol: Stock symbol

    #     Returns:
    #         DataFrame with options data or None if fetch fails
    #     """
    #     if not self.twelve_data_key:
    #         logger.warning("No Twelve Data API key found")
    #         return None

    #     if symbol not in TWELVE_DATA_MAPPINGS:
    #         logger.debug(f"No Twelve Data mapping found for symbol: {symbol}")
    #         return None

    #     try:
    #         td_symbol = TWELVE_DATA_MAPPINGS[symbol]
    #         params = {
    #             "symbol": td_symbol,
    #             "apikey": self.twelve_data_key,
    #             "show_all": "true",  # Get full options chain
    #             "timezone": "UTC",
    #         }

    #         # Fetch options chain
    #         response = self.session.get(f"{TWELVE_DATA_URL}/options", params=params)
    #         response.raise_for_status()

    #         data = response.json()
    #         if not data or "options" not in data:
    #             logger.warning(
    #                 f"No options data returned from Twelve Data for {symbol}"
    #             )
    #             return None

    #         # Process options data into DataFrame
    #         options_list = []
    #         for contract in data["options"]:
    #             try:
    #                 contract_data = {
    #                     "symbol": symbol,
    #                     "strike": float(contract["strike"]),
    #                     "expiration": pd.to_datetime(contract["expiration"]),
    #                     "option_type": contract["type"].lower(),
    #                     "impliedVolatility": float(contract["implied_volatility"]),
    #                     "volume": int(contract.get("volume", 0)),
    #                     "openInterest": int(contract.get("open_interest", 0)),
    #                     "bid": float(contract.get("bid", 0)),
    #                     "ask": float(contract.get("ask", 0)),
    #                     "last": float(contract.get("last", 0)),
    #                     "delta": float(contract.get("delta", 0)),
    #                     "gamma": float(contract.get("gamma", 0)),
    #                     "theta": float(contract.get("theta", 0)),
    #                     "vega": float(contract.get("vega", 0)),
    #                     "timestamp": datetime.now(),
    #                 }
    #                 options_list.append(contract_data)
    #             except (ValueError, KeyError) as e:
    #                 logger.warning(f"Error processing contract data: {e}")
    #                 continue

    #         if not options_list:
    #             logger.warning(f"No valid options contracts found for {symbol}")
    #             return None

    #         df = pd.DataFrame(options_list)

    #         # Validate the data
    #         is_valid, error_msg = self._validate_options_data(df)
    #         if not is_valid:
    #             logger.warning(f"Invalid options data for {symbol}: {error_msg}")
    #             return None

    #         logger.info(
    #             f"Successfully fetched {len(df)} options contracts from Twelve Data for {symbol}"
    #         )
    #         return df

    #     except Exception as e:
    #         logger.error(f"Error fetching Twelve Data options for {symbol}: {str(e)}")
    #         return None

    def _validate_options_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate options data quality with enhanced checks."""
        if df is None or df.empty:
            return False, "Empty DataFrame"

        # Required columns for analysis
        required_cols = {
            "strike",
            "expiration",
            "option_type",
            "bid",
            "ask",
            "volume",
            "openInterest",
        }

        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"

        # Check for minimum number of strikes and expirations
        n_strikes = df["strike"].nunique()
        n_expirations = df["expiration"].nunique()

        if n_strikes < 3:  # Reduced from 5 to 3 for better coverage
            return False, f"Insufficient strike prices: {n_strikes} < 3"
        if n_expirations < 1:  # Reduced from 2 to 1 for better coverage
            return False, f"Insufficient expiration dates: {n_expirations} < 1"

        # Check for reasonable implied volatility values if available
        if "impliedVolatility" in df.columns:
            invalid_iv = (
                df["impliedVolatility"].isna()
                | (df["impliedVolatility"] <= 0)
                | (df["impliedVolatility"] > 5)
            )
            if invalid_iv.any():
                pct_invalid = (invalid_iv.sum() / len(df)) * 100
                if pct_invalid > 50:  # Increased from 20% to 50% for better coverage
                    return (
                        False,
                        f"Too many invalid implied volatilities: {pct_invalid:.1f}%",
                    )

        # Check for reasonable bid-ask spreads
        if all(col in df.columns for col in ["bid", "ask"]):
            invalid_spread = (df["ask"] < df["bid"]) | (df["bid"] < 0) | (df["ask"] < 0)
            if invalid_spread.any():
                pct_invalid = (invalid_spread.sum() / len(df)) * 100
                if pct_invalid > 10:
                    return (
                        False,
                        f"Too many invalid bid-ask spreads: {pct_invalid:.1f}%",
                    )

        return True, ""

    def _determine_data_source(self, symbol: str) -> str:
        """Determine the data source based on symbol."""
        # Try Eurex first for European options
        try:
            if self.eurex_client:
                product_result = self.eurex_client.execute(gql(EUREX_PRODUCT_QUERY))
                if product_result and "products" in product_result:
                    for prod in product_result["products"]["data"]:
                        if prod["productId"] == symbol:
                            return "eurex"
        except Exception:
            pass

        # Try other sources
        if symbol in EOD_SYMBOL_MAPPINGS:
            return "eod"
        elif symbol in ALPHA_VANTAGE_MAPPINGS:
            return "alpha_vantage"
        elif symbol in TWELVE_DATA_MAPPINGS:
            return "twelve_data"
        else:
            return "yahoo"

    def fetch_options_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch options data from multiple sources with fallback."""
        logger.info(f"Processing options for {symbol}...")

        df = None

        # 1. Try Eurex first for European options
        try:
            if self.eurex_client:
                df = self._fetch_eurex_options(symbol)
                if df is not None and not df.empty:
                    logger.info(f"Successfully fetched Eurex options data for {symbol}")
                    return df
        except Exception as e:
            logger.warning(f"Error checking Eurex availability for {symbol}: {e}")

        # 2. Try Yahoo Finance as fallback
        if df is None:
            df = self._fetch_yfinance_options(symbol)
            if df is not None and not df.empty:
                logger.info(
                    f"Successfully fetched Yahoo Finance options data for {symbol}"
                )
                return df

        logger.warning(f"Failed to fetch options data for {symbol} from all sources")
        return None

    def save_options_data(self, df: pd.DataFrame, symbol: str) -> None:
        """Save raw options data to parquet file."""
        if df is None or df.empty:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_options_chain_latest_{timestamp}.parquet"
            filepath = self.output_dir_raw / filename

            # Convert datetime columns to proper format
            date_columns = ["expiration", "lastTradeDate", "timestamp", "quoteDate"]
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            # Add metadata
            df["fetch_timestamp"] = datetime.now()
            df["data_source"] = self._determine_data_source(symbol)

            # Save to parquet with compression
            df.to_parquet(filepath, compression="snappy")
            logger.info(f"Saved raw options data to {filepath}")

            # Clean up old files
            self._cleanup_old_files(symbol, "options_chain", keep_days=30)

        except Exception as e:
            logger.error(f"Error saving options data for {symbol}: {str(e)}")

    def _cleanup_old_files(
        self, symbol: str, file_type: str, keep_days: int = 30
    ) -> None:
        """Clean up old data files.

        Args:
            symbol: Ticker symbol
            file_type: Type of file (e.g., 'options_chain', 'iv_surface')
            keep_days: Number of days to keep files
        """
        try:
            pattern = f"{symbol}_{file_type}_latest_*.parquet"
            files = list(self.output_dir_raw.glob(pattern))

            if len(files) <= 1:
                return

            # Sort files by modification time
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Keep the most recent file and delete others older than keep_days
            cutoff_time = datetime.now() - timedelta(days=keep_days)

            for file in files[1:]:  # Skip the most recent file
                if datetime.fromtimestamp(file.stat().st_mtime) < cutoff_time:
                    file.unlink()
                    logger.debug(f"Deleted old file: {file}")

        except Exception as e:
            logger.error(f"Error cleaning up old files for {symbol}: {str(e)}")

    def process_options_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Process raw options data into features needed for volatility analysis."""
        try:
            if raw_df is None or raw_df.empty:
                return pd.DataFrame()

            # Calculate days to expiration
            raw_df["expiration"] = pd.to_datetime(raw_df["expiration"])
            raw_df["daysToExpiration"] = (
                raw_df["expiration"] - pd.Timestamp.now()
            ).dt.days

            # 1. Calculate implied volatility surface
            processed = pd.DataFrame(index=raw_df.index)

            # ATM volatility
            if "inTheMoney" in raw_df.columns:
                atm_options = raw_df[
                    (~raw_df["inTheMoney"]) & (raw_df["daysToExpiration"] <= 30)
                ]
            else:
                # If inTheMoney is not available, use strike price relative to current price
                current_price = raw_df["strike"].median()  # Approximate current price
                atm_options = raw_df[
                    (
                        raw_df["strike"].between(
                            current_price * 0.95, current_price * 1.05
                        )
                    )
                    & (raw_df["daysToExpiration"] <= 30)
                ]

            if not atm_options.empty:
                processed["atm_vol"] = atm_options.groupby("expiration")[
                    "impliedVolatility"
                ].mean()

            # Volatility skew
            if "strike" in raw_df.columns and "impliedVolatility" in raw_df.columns:
                calls = raw_df[raw_df["option_type"] == "call"]
                puts = raw_df[raw_df["option_type"] == "put"]

                # Put skew (OTM puts)
                put_skew = puts[puts["strike"] < puts["strike"].mean()]
                if not put_skew.empty:
                    processed["put_skew"] = put_skew.groupby("expiration")[
                        "impliedVolatility"
                    ].mean()

                # Call skew (OTM calls)
                call_skew = calls[calls["strike"] > calls["strike"].mean()]
                if not call_skew.empty:
                    processed["call_skew"] = call_skew.groupby("expiration")[
                        "impliedVolatility"
                    ].mean()

            # 2. Calculate Greeks if available
            for greek in ["delta", "gamma", "theta", "vega", "rho"]:
                if greek in raw_df.columns:
                    processed[f"avg_{greek}"] = raw_df.groupby("expiration")[
                        greek
                    ].mean()

            # 3. Liquidity metrics
            if "volume" in raw_df.columns and "openInterest" in raw_df.columns:
                processed["volume"] = raw_df.groupby("expiration")["volume"].sum()
                processed["open_interest"] = raw_df.groupby("expiration")[
                    "openInterest"
                ].sum()

                # Put-Call ratio
                put_volume = raw_df[raw_df["option_type"] == "put"]["volume"].sum()
                call_volume = raw_df[raw_df["option_type"] == "call"]["volume"].sum()
                processed["put_call_ratio"] = put_volume / (
                    call_volume + 1e-8
                )  # Avoid division by zero

            # 4. Term structure
            if "impliedVolatility" in raw_df.columns:
                term_structure = raw_df.groupby("daysToExpiration")[
                    "impliedVolatility"
                ].mean()
                processed["vol_term_structure"] = term_structure

            # Fill missing values using modern methods
            processed = processed.ffill().bfill().fillna(0)

            # Add metadata
            processed["timestamp"] = datetime.now()
            if "symbol" in raw_df.columns:
                processed["symbol"] = raw_df["symbol"].iloc[0]

            logger.info(
                f"Successfully processed options data with shape {processed.shape}"
            )
            return processed

        except Exception as e:
            logger.error(f"Error processing options data: {str(e)}")
            return pd.DataFrame()

    def save_processed_data(self, df: pd.DataFrame, symbol: str) -> None:
        """Save processed options data to parquet file."""
        if df is None or df.empty:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save IV surface
        surface_filename = f"{symbol}_iv_surface_latest_{timestamp}.parquet"
        surface_filepath = self.output_dir_processed / surface_filename
        surface_cols = [col for col in df.columns if "vol" in col or "skew" in col]
        if surface_cols:
            df[surface_cols].to_parquet(surface_filepath)
            logger.info(f"Saved IV surface to {surface_filepath}")

        # Save Greeks
        greeks_filename = f"{symbol}_greeks_latest_{timestamp}.parquet"
        greeks_filepath = self.output_dir_processed / greeks_filename
        greek_cols = [
            col
            for col in df.columns
            if any(g in col for g in ["delta", "gamma", "theta", "vega", "rho"])
        ]
        if greek_cols:
            df[greek_cols].to_parquet(greeks_filepath)
            logger.info(f"Saved Greeks to {greeks_filepath}")

        # Save metrics (put-call ratio, volume, etc.)
        metrics_filename = f"{symbol}_options_metrics_latest_{timestamp}.parquet"
        metrics_filepath = self.output_dir_processed / metrics_filename
        metric_cols = [
            "put_call_ratio",
            "volume",
            "open_interest",
            "timestamp",
            "symbol",
        ]
        if any(col in df.columns for col in metric_cols):
            df[[col for col in metric_cols if col in df.columns]].to_parquet(
                metrics_filepath
            )
            logger.info(f"Saved options metrics to {metrics_filepath}")

    def process_options(self) -> None:
        """Process options data for all symbols."""
        # First try to fetch EURO STOXX 50 index options
        try:
            logger.info("Fetching EURO STOXX 50 index options...")
            self.fetch_options_data("^STOXX50E")
        except Exception as e:
            logger.error(f"Failed to process index options: {str(e)}")

        for symbol in TICKERS:
            try:
                self.fetch_options_data(symbol)
            except Exception as e:
                logger.error(f"Failed to process options for {symbol}: {str(e)}")

        # Process EURO STOXX 50 index options again at the end
        try:
            self.fetch_options_data("^STOXX50E")
        except Exception as e:
            logger.error(f"Failed to process options for ^STOXX50E: {str(e)}")

        logger.info("Options data processing completed")

    def fetch_and_process_options(self, symbol: str) -> None:
        """Fetch and process options data for a symbol."""
        try:
            # Fetch raw data
            raw_df = self.fetch_options_data(symbol)
            if raw_df is not None and not raw_df.empty:
                # Save raw data
                self.save_options_data(raw_df, symbol)

                # Process data
                processed_df = self.process_options_data(raw_df)
                if not processed_df.empty:
                    # Save processed data
                    self.save_processed_data(processed_df, symbol)
                    logger.info(f"Successfully processed options data for {symbol}")
                else:
                    logger.warning(f"No processed data generated for {symbol}")
            else:
                logger.warning(f"No raw data fetched for {symbol}")

        except Exception as e:
            logger.error(f"Error in fetch_and_process_options for {symbol}: {str(e)}")


def main():
    """Main function to fetch and process options data."""
    options_fetcher = OptionsDataFetcher()

    # Process index first
    try:
        logger.info("Processing EURO STOXX 50 index options...")
        options_fetcher.fetch_and_process_options("^STOXX50E")
    except Exception as e:
        logger.error(f"Failed to process index options: {str(e)}")

    # Process individual stocks
    for ticker in TICKERS:
        try:
            logger.info(f"Processing options for {ticker}...")
            options_fetcher.fetch_and_process_options(ticker)
        except Exception as e:
            logger.error(f"Failed to process options for {ticker}: {str(e)}")
            continue

    logger.info("Options data processing completed")


if __name__ == "__main__":
    main()
