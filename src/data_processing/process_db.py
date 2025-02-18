import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from src.utils.log_utils import setup_logging
from src.utils.data_management import market_data_manager
from src.config.settings import DJ_TITANS_50_TICKER
from src.config.paths import *

logger = setup_logging(__name__)

class ProcessDB:
    """"""

    def __init__(self):
        """Initialize the ProcessDB class."""

        #############################################
        # PATHS DATASETS
        #############################################

        # raw data options & market intraday
        self.raw_intraday_dir: Path = RAW_INTRADAY # 1min/1year
        self.raw_options_dir: Path = RAW_OPTIONS # 1day/1year
        
        # insider transactions & news sentiments
        self.raw_news_dir: Path = RAW_NEWS
        self.insider_transactions_dir: Path = RAW_NEWS / "insider"
        self.news_sentiments_dir: Path = RAW_NEWS / "sentiments"

        # macro data
        self.raw_macro_dir: Path = RAW_MACRO
        self.cpi_dir: Path = RAW_MACRO / "cpi"
        self.gdp_dir: Path = RAW_MACRO / "gdp"
        self.unemployment_dir: Path = RAW_MACRO / "unemployment"
        self.treasury_yields_dir: Path = RAW_MACRO / "treasury_yields"
        self.bond_spreads_dir: Path = RAW_MACRO 
        self.ecb_rates_dir: Path = RAW_MACRO 


        
