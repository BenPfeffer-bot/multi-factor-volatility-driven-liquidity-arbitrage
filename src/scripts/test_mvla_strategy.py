"""
Test script for the MVLA strategy implementation.

This script validates the signal generation and combination logic
by running tests on historical data and producing diagnostic plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys, os
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.strategies.mvla_strategy import MVLAStrategy
from src.utils.data_management import market_data_manager
from src.utils.log_utils import setup_logging
from src.config.settings import DJ_TITANS_50_TICKER
from src.config.paths import *

logger = setup_logging(__name__)

def initialize_test_data(test_symbols: List[str], start_date: str, end_date: str) -> None:
    """Initialize test data for all required symbols."""
    logger.info(f"Initializing test data for {len(test_symbols)} symbols")
    
    # Create all required directories
    for path in [
        'db/raw/intraday',
        'db/raw/options',
        'db/raw/macro/treasury_yields',
        'db/raw/news/sentiment'
    ]:
        os.makedirs(path, exist_ok=True)
    
    # Convert dates to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    for symbol in test_symbols:
        # Generate base price series with realistic properties
        dates = pd.date_range(start_dt, end_dt, freq='1min')
        n_minutes = len(dates)
        
        # Generate realistic price series with mean reversion and volatility clustering
        np.random.seed(42)
        base_price = 150.0
        returns = np.random.normal(0, 0.0002, n_minutes)  # Small random returns
        vol_cluster = np.abs(np.random.normal(0, 0.0001, n_minutes))
        returns = returns * (1 + vol_cluster)  # Add volatility clustering
        
        # Ensure no extreme values
        returns = np.clip(returns, -0.01, 0.01)
        
        # Calculate prices ensuring they're always positive
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        intraday_data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'close': prices * (1 + np.random.normal(0, 0.0001, n_minutes)),
            'volume': np.abs(np.random.lognormal(10, 1, n_minutes))
        })
        
        # Ensure high/low prices maintain OHLC relationship
        intraday_data['high'] = np.maximum(
            intraday_data['open'],
            intraday_data['close']
        ) * (1 + np.abs(np.random.normal(0, 0.0002, n_minutes)))
        
        intraday_data['low'] = np.minimum(
            intraday_data['open'],
            intraday_data['close']
        ) * (1 - np.abs(np.random.normal(0, 0.0002, n_minutes)))
        
        # Calculate clean returns and realized volatility
        intraday_data['returns'] = intraday_data['close'].pct_change().fillna(0)
        intraday_data['realized_volatility'] = intraday_data['returns'].rolling(30).std().fillna(0)
        
        # Set date as index after calculations
        intraday_data.set_index('date', inplace=True)
        
        # Generate options data
        expiry_dates = pd.date_range(end_dt + pd.Timedelta(days=1), periods=12, freq='ME')
        strikes = np.linspace(0.8, 1.2, 21) * base_price
        
        options_data = []
        current_price = prices[-1]
        
        for expiry in expiry_dates:
            days_to_expiry = (expiry - end_dt).days
            for strike in strikes:
                moneyness = strike / current_price
                
                # Calculate implied volatility with smile effect
                base_vol = 0.2
                smile_effect = 0.1 * (moneyness - 1) ** 2
                time_effect = np.sqrt(max(1, days_to_expiry) / 365)
                implied_vol = (base_vol + smile_effect) * time_effect
                
                for option_type in ['call', 'put']:
                    options_data.append({
                        'date': end_dt,
                        'expiry_date': expiry,
                        'strike': strike,
                        'type': option_type,
                        'implied_volatility': implied_vol,
                        'underlying_price': current_price,
                        'days_to_expiry': days_to_expiry,
                        'volume': np.random.lognormal(5, 1),
                        'open_interest': np.random.lognormal(6, 1),
                        'bid': current_price * 0.99,
                        'ask': current_price * 1.01,
                        'delta': np.random.uniform(0.2, 0.8),
                        'gamma': np.random.uniform(0.01, 0.1),
                        'vega': np.random.uniform(0.1, 0.5),
                        'theta': np.random.uniform(-0.1, -0.01)
                    })
        
        options_df = pd.DataFrame(options_data)
        
        # Generate sentiment data
        sentiment_data = pd.DataFrame({
            'date': dates,
            'sentiment_score': np.random.normal(0, 0.2, n_minutes),
            'mention_count': np.random.poisson(10, n_minutes),
            'returns': intraday_data['returns']
        })
        sentiment_data.set_index('date', inplace=True)
        
        # Save data to appropriate locations
        intraday_data.to_csv(f'db/raw/intraday/{symbol}_1min.csv')
        options_df.to_csv(f'db/raw/options/{symbol}_options.csv', index=False)
        sentiment_data.to_csv(f'db/raw/news/sentiment/{symbol}_sentiment.csv')
    
    # Generate treasury yields data with realistic term structure
    base_yields = {
        '2Y': 0.025,
        '5Y': 0.028,
        '10Y': 0.032,
        '30Y': 0.035
    }
    
    daily_dates = pd.date_range(start_dt, end_dt, freq='D')
    yields_data = pd.DataFrame(index=daily_dates)
    yields_data.index.name = 'date'
    
    for tenor, base_yield in base_yields.items():
        yields_data[tenor] = base_yield + np.random.normal(0, 0.0002, len(daily_dates))
    
    yields_data.to_csv('db/raw/macro/treasury_yields/treasury_yield_10y_daily.csv')
    
    logger.info("Successfully initialized test data")

class StrategyTester:
    def __init__(self, test_symbols: List[str], start_date: str, end_date: str):
        self.test_symbols = test_symbols
        self.start_date = start_date
        self.end_date = end_date
        
        # Create output directories
        self.results_dir = ANALYSIS_DIR / "strategy_tests" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize test data
        initialize_test_data(test_symbols, start_date, end_date)
        
        # Initialize strategy with lower thresholds for testing
        self.strategy = MVLAStrategy(
            symbols=test_symbols,
            lookback_days=30,
            vol_window=20,
            min_liquidity_threshold=10000,  # Lower threshold for testing
            sentiment_impact_threshold=0.3,  # Lower threshold for testing
            iv_rank_threshold=0.5,  # Lower threshold for testing
            vol_signal_threshold=1.5,  # Lower threshold for testing
            liquidity_signal_threshold=1.5,  # Lower threshold for testing
            macro_impact_threshold=1.0  # Lower threshold for testing
        )
        
        # Load data
        self.strategy.load_data(start_date=start_date, end_date=end_date)
    
    def save_plot(self, fig: plt.Figure, name: str) -> None:
        """Save plot to the results directory."""
        plot_path = self.results_dir / f"{name}.png"
        fig.savefig(plot_path)
        logger.info(f"Saved plot to {plot_path}")
    
    def save_signals(self, signals: pd.DataFrame, symbol: str) -> None:
        """Save signal data to the results directory."""
        signal_path = self.results_dir / f"{symbol}_signals.csv"
        signals.to_csv(signal_path)
        logger.info(f"Saved signals for {symbol} to {signal_path}")
    
    def plot_signal_components(self, signals: pd.DataFrame, symbol: str) -> None:
        """Plot individual signal components and their combinations."""
        fig = plt.figure(figsize=(15, 10))
        
        # Plot individual signals
        plt.subplot(3, 1, 1)
        for col in ['vol_signal', 'liquidity_signal', 'options_signal', 'macro_signal', 'sentiment_signal']:
            if col in signals.columns:
                plt.plot(signals.index, signals[col], label=col)
        plt.title(f'Individual Signals - {symbol}')
        plt.legend()
        plt.grid(True)
        
        # Plot combined signal and strength
        plt.subplot(3, 1, 2)
        plt.plot(signals.index, signals['combined_signal'], label='Combined Signal')
        plt.plot(signals.index, signals['signal_strength'], label='Signal Strength')
        plt.title('Combined Signal and Strength')
        plt.legend()
        plt.grid(True)
        
        # Plot conviction and agreement
        plt.subplot(3, 1, 3)
        plt.plot(signals.index, signals['combined_conviction'], label='Conviction')
        plt.plot(signals.index, signals['signal_agreement'], label='Agreement')
        plt.title('Signal Conviction and Agreement')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        self.save_plot(fig, f"{symbol}_signal_components")
    
    def analyze_signal_correlations(self, signals: pd.DataFrame, symbol: str) -> None:
        """Analyze correlations between different signal components."""
        signal_cols = [
            'vol_signal', 'liquidity_signal', 'options_signal',
            'macro_signal', 'sentiment_signal'
        ]
        
        corr_matrix = signals[signal_cols].corr()
        
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(f'Signal Component Correlations - {symbol}')
        self.save_plot(fig, f"{symbol}_correlations")
    
    def validate_signal_generation(self, symbol: str) -> None:
        """Validate signal generation for a single symbol."""
        logger.info(f"Validating signals for {symbol}")
        
        try:
            # Generate signals
            signals = self.strategy.generate_signals()
            
            if signals.empty:
                logger.warning(f"No signals generated for {symbol}")
                return
            
            # Filter signals for the current symbol
            symbol_signals = signals[signals['symbol'] == symbol] if 'symbol' in signals.columns else signals
            
            if symbol_signals.empty:
                logger.warning(f"No signals found for {symbol}")
                return
            
            # Save signals
            self.save_signals(symbol_signals, symbol)
            
            # Basic validation checks
            validation_results = {
                'non_zero_signals': (symbol_signals['combined_signal'] != 0).sum(),
                'signal_distribution': symbol_signals['combined_signal'].value_counts(),
                'avg_conviction': symbol_signals['combined_conviction'].mean(),
                'avg_agreement': symbol_signals['signal_agreement'].mean()
            }
            
            # Save validation results
            validation_path = self.results_dir / f"{symbol}_validation.txt"
            with open(validation_path, 'w') as f:
                for key, value in validation_results.items():
                    f.write(f"{key}:\n{value}\n\n")
            
            logger.info(f"Saved validation results to {validation_path}")
            
            # Plot diagnostics
            self.plot_signal_components(symbol_signals, symbol)
            self.analyze_signal_correlations(symbol_signals, symbol)
            
        except Exception as e:
            logger.error(f"Error validating signals for {symbol}: {str(e)}")
    
    def run_tests(self) -> None:
        """Run all strategy tests."""
        if not self.strategy.symbols:
            logger.warning("No valid symbols available for testing")
            return
        
        for symbol in self.test_symbols:
            self.validate_signal_generation(symbol)
        
        logger.info(f"All test results saved to {self.results_dir}")

def main():
    """Main test function."""
    # Define test parameters
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2025-01-18'
    end_date = '2025-02-17'
    
    # Create tester instance
    tester = StrategyTester(test_symbols, start_date, end_date)
    
    # Run tests
    tester.run_tests()

if __name__ == "__main__":
    main() 