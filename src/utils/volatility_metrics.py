import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple, Dict
from pathlib import Path

from src.config.paths import (
    PROCESSED_DAILY,
    PROCESSED_INTRADAY,
    RAW_DAILY,
    RAW_INTRADAY,
)
from src.utils.log_utils import setup_logging, log_execution_time

logger = setup_logging(__name__)


class VolatilityEstimator:
    """
    Advanced volatility estimation framework incorporating multiple estimators
    and statistical validation techniques.
    """

    def __init__(
        self,
        window_size: int = 30,
        annualization_factor: float = 252,
        estimation_method: str = "all",
    ):
        self.window_size = window_size
        self.annualization_factor = annualization_factor
        self.estimation_method = estimation_method

    def _parkinson_estimator(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Parkinson volatility estimator (1980) - uses high-low range.
        More efficient than close-to-close estimator.

        σ_p = √(1/(4ln(2)) * Σ(ln(H_i/L_i))²)
        """
        return np.sqrt(1.0 / (4.0 * np.log(2.0)) * (np.log(high / low) ** 2))

    def _garman_klass_estimator(
        self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """
        Garman-Klass volatility estimator (1980) - incorporates opening and closing prices.
        More efficient than Parkinson estimator.

        σ_gk = √(0.5 * (ln(H/L))² - (2ln(2)-1) * (ln(C/O))²)
        """
        return np.sqrt(
            0.5 * np.log(high / low) ** 2
            - (2 * np.log(2) - 1) * np.log(close / open_) ** 2
        )

    def _rogers_satchell_estimator(
        self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """
        Rogers-Satchell volatility estimator (1991) - accounts for drift in returns.

        σ_rs = √(ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O))
        """
        return np.sqrt(
            np.log(high / close) * np.log(high / open_)
            + np.log(low / close) * np.log(low / open_)
        )

    def _yang_zhang_estimator(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        alpha: float = 0.34,
    ) -> pd.Series:
        """
        Yang-Zhang volatility estimator (2000) - combines overnight and trading volatility.
        Minimum variance and drift-independent.

        σ²_yz = σ²_o + k * σ²_c + (1-k) * σ²_rs
        where k = α / (1 + α)
        """
        # Overnight volatility
        overnight_vol = np.log(open_ / close.shift(1)) ** 2

        # Open-to-close volatility
        open_close_vol = np.log(close / open_) ** 2

        # Rogers-Satchell volatility
        rs_vol = self._rogers_satchell_estimator(open_, high, low, close)

        k = alpha / (1 + alpha)
        return np.sqrt(overnight_vol + k * open_close_vol + (1 - k) * rs_vol**2)

    def estimate_all_volatilities(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all volatility estimators and their statistical properties.
        """
        result = data.copy()

        # Ensure data is sorted by datetime
        if not isinstance(result.index, pd.DatetimeIndex):
            if "datetime" in result.columns:
                result["datetime"] = pd.to_datetime(result["datetime"])
                result.set_index("datetime", inplace=True)
            elif "date" in result.columns:
                result["datetime"] = pd.to_datetime(result["date"])
                result.set_index("datetime", inplace=True)
        result = result.sort_index()

        # Standard log returns volatility
        result["log_returns"] = np.log(result["close"] / result["close"].shift(1))
        result["returns_vol"] = result["log_returns"].rolling(
            window=self.window_size, min_periods=self.window_size // 2
        ).std() * np.sqrt(self.annualization_factor)

        # Advanced estimators with proper window handling
        result["parkinson_vol"] = self._parkinson_estimator(
            result["high"], result["low"]
        ).rolling(
            window=self.window_size, min_periods=self.window_size // 2
        ).mean() * np.sqrt(self.annualization_factor)

        result["garman_klass_vol"] = self._garman_klass_estimator(
            result["open"], result["high"], result["low"], result["close"]
        ).rolling(
            window=self.window_size, min_periods=self.window_size // 2
        ).mean() * np.sqrt(self.annualization_factor)

        result["rogers_satchell_vol"] = self._rogers_satchell_estimator(
            result["open"], result["high"], result["low"], result["close"]
        ).rolling(
            window=self.window_size, min_periods=self.window_size // 2
        ).mean() * np.sqrt(self.annualization_factor)

        result["yang_zhang_vol"] = self._yang_zhang_estimator(
            result["open"], result["high"], result["low"], result["close"]
        ).rolling(
            window=self.window_size, min_periods=self.window_size // 2
        ).mean() * np.sqrt(self.annualization_factor)

        # Statistical moments with proper window handling
        for vol_type in [
            "returns_vol",
            "parkinson_vol",
            "garman_klass_vol",
            "rogers_satchell_vol",
            "yang_zhang_vol",
        ]:
            result[f"{vol_type}_skew"] = (
                result[vol_type]
                .rolling(window=self.window_size, min_periods=self.window_size // 2)
                .skew()
            )
            result[f"{vol_type}_kurt"] = (
                result[vol_type]
                .rolling(window=self.window_size, min_periods=self.window_size // 2)
                .kurt()
            )

        # Forward fill any remaining NaN values at the beginning
        vol_cols = [col for col in result.columns if "vol" in col.lower()]
        result[vol_cols] = result[vol_cols].ffill()

        # Drop any remaining rows with NaN values
        result = result.dropna(subset=vol_cols)

        return result

    def compute_iv_rv_spread(
        self, implied_vol: pd.Series, realized_vol: pd.Series
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Compute and analyze the IV-RV spread with statistical properties.
        """
        spread = implied_vol - realized_vol

        stats_dict = {
            "mean_spread": spread.mean(),
            "std_spread": spread.std(),
            "skew_spread": stats.skew(spread.dropna()),
            "kurt_spread": stats.kurtosis(spread.dropna()),
            "jarque_bera": stats.jarque_bera(spread.dropna())[0],
            "jb_pvalue": stats.jarque_bera(spread.dropna())[1],
        }

        return spread, stats_dict

    def save_processed_data(
        self,
        data: pd.DataFrame,
        file_path: str = PROCESSED_DAILY,
        include_timestamp: bool = True,
        ticker: str = None,
    ) -> None:
        """
        Save processed volatility data to a CSV file with configurable options.

        Args:
            data: DataFrame containing processed volatility metrics
            file_path: Path to save the data file. Defaults to PROCESSED_DAILY
            include_timestamp: Whether to include timestamp in filename
            ticker: Stock ticker symbol to include in filename

        The data is saved with proper index handling.
        Creates parent directories if they don't exist.
        """
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Add timestamp and ticker to filename if requested
        stem = file_path.stem
        suffix = ".csv"  # Force CSV extension
        filename_parts = [stem]

        if ticker:
            filename_parts.append(ticker)

        if include_timestamp:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename_parts.append(timestamp)

        file_path = file_path / f"{'_'.join(filename_parts)}{suffix}"

        # Save with proper index handling
        data.to_csv(
            file_path,
            index=True,  # Preserve time series index
            index_label="date",
            float_format="%.6f",  # Control precision of float values
        )

    def compute_index_specific_metrics(
        self,
        index_data: pd.DataFrame,
        options_data: pd.DataFrame,
        vstoxx_data: pd.DataFrame,
    ) -> Dict[str, pd.Series]:
        """
        Compute EUROSTOXX 50 specific volatility metrics
        """
        metrics = {}

        # High-frequency realized volatility
        metrics["high_freq_rv"] = self._compute_high_freq_rv(index_data)

        # Term structure metrics
        term_structure = self._compute_term_structure(options_data)
        metrics.update(term_structure)

        # VSTOXX-based metrics
        if not vstoxx_data.empty:
            metrics["vstoxx_level"] = vstoxx_data["vstoxx"]
            metrics["vstoxx_term_spread"] = (
                vstoxx_data["vstoxx_3m"] - vstoxx_data["vstoxx"]
            )

        return metrics

    def _compute_high_freq_rv(self, data: pd.DataFrame) -> pd.Series:
        """Compute high-frequency realized volatility."""
        return data["returns"].rolling(window=self.window_size).std() * np.sqrt(
            self.annualization_factor
        )

    def _compute_term_structure(self, options_data: pd.DataFrame) -> Dict[str, float]:
        """Compute volatility term structure metrics."""
        term_structure = {}
        if not options_data.empty:
            # Group by expiration and compute metrics
            grouped = options_data.groupby("expiration")
            term_structure["term_slope"] = grouped["implied_volatility"].mean().diff()
            term_structure["term_curvature"] = (
                grouped["implied_volatility"].mean().diff(2)
            )
        return term_structure


def process_volatility_data(
    data: pd.DataFrame,
    ticker: str = None,
    save_results: bool = True,
    is_intraday: bool = False,
) -> pd.DataFrame:
    """
    Process market data to compute various volatility metrics.

    Args:
        data: Input market data DataFrame
        ticker: Stock ticker symbol
        save_results: Whether to save processed results
        is_intraday: Whether the data is intraday

    Returns:
        DataFrame with computed volatility metrics
    """
    # Initialize volatility estimator
    estimator = VolatilityEstimator(
        window_size=30
        if not is_intraday
        else 78,  # 30 days or 6.5 hours * 12 5-min periods
        annualization_factor=252 if not is_intraday else 252 * 78,
    )

    # Compute all volatility metrics
    result = estimator.estimate_all_volatilities(data)

    if save_results:
        # Determine save path based on data type
        save_path = PROCESSED_INTRADAY if is_intraday else PROCESSED_DAILY
        estimator.save_processed_data(result, save_path, ticker=ticker)

    return result


if __name__ == "__main__":
    # Process all stocks in both daily and intraday frequencies
    try:
        # Process daily data
        daily_dir = RAW_DAILY
        for file in daily_dir.glob("*_1d_*.csv"):
            try:
                ticker = file.stem.split("_")[0]  # Extract ticker from filename
                logger.info(f"Processing daily data for {ticker}")

                data = pd.read_csv(file)
                vol_data = process_volatility_data(
                    data, ticker=ticker, save_results=True, is_intraday=False
                )
                logger.info(f"Successfully processed daily data for {ticker}")

            except Exception as e:
                logger.error(f"Failed to process daily data for {ticker}: {str(e)}")
                continue

        # Process intraday data
        intraday_dir = RAW_INTRADAY
        for file in intraday_dir.glob("*_1m_*.csv"):
            try:
                ticker = file.stem.split("_")[0]  # Extract ticker from filename
                logger.info(f"Processing intraday data for {ticker}")

                data = pd.read_csv(file)
                vol_data = process_volatility_data(
                    data, ticker=ticker, save_results=True, is_intraday=True
                )
                logger.info(f"Successfully processed intraday data for {ticker}")

            except Exception as e:
                logger.error(f"Failed to process intraday data for {ticker}: {str(e)}")
                continue

        logger.info("Completed processing all stock data")

    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        raise
