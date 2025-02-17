import glob
import os
import re
from typing import Optional, Dict
import pandas as pd
from log import logger


class DataLoader:
    def load_options_data(self, ticker: str) -> Optional[Dict]:
        """Load options data for a given ticker."""
        options_data = {}
        chain_files = glob.glob(
            os.path.join(self.options_dir, f"{ticker}_chain_*.parquet")
        )

        if not chain_files:
            return None

        try:
            for file in chain_files:
                date_str = re.search(r"_chain_(\d{8})\.parquet", file).group(1)
                surface_file = os.path.join(
                    self.options_dir, f"{ticker}_surface_{date_str}.parquet"
                )
                metrics_file = os.path.join(
                    self.options_dir, f"{ticker}_metrics_{date_str}.parquet"
                )

                # Check if all required files exist
                if not (os.path.exists(file) and os.path.exists(surface_file)):
                    continue

                # Read the files
                chain = pd.read_parquet(file)
                surface = pd.read_parquet(surface_file)

                if chain.shape[0] > 0 and surface.shape[0] > 0:
                    if date_str not in options_data:
                        options_data[date_str] = {}
                    options_data[date_str]["chain"] = chain
                    options_data[date_str]["surface"] = surface

                    # Only add metrics if the file exists and has data
                    if os.path.exists(metrics_file):
                        metrics = pd.read_parquet(metrics_file)
                        if metrics.shape[0] > 0:
                            options_data[date_str]["metrics"] = metrics

            return options_data if options_data else None

        except Exception as e:
            logger.warning(f"Error loading options data for {ticker}: {str(e)}")
            return None
