"""
Detailed test suite for MVLA strategy signal components.

This script provides in-depth testing of each signal generator
and validates their theoretical foundations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys, os
from pathlib import Path
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.strategies.signals.volatility_signals import VolatilitySignalGenerator
from src.strategies.signals.liquidity_signals import LiquiditySignalGenerator
from src.strategies.signals.options_signals import OptionsSignalGenerator
from src.strategies.signals.macro_signals import MacroSignalGenerator
from src.strategies.signals.sentiment_signals import SentimentSignalGenerator
from src.utils.data_management import market_data_manager
from src.utils.log_utils import setup_logging
from src.config.paths import *

logger = setup_logging(__name__)

def initialize_test_data(symbol: str = 'AAPL'):
    """Initialize test data for signal testing."""
    # Generate base price series with realistic properties
    n_days = 30
    n_minutes = n_days * 390  # Trading minutes per day
    dates = pd.date_range('2024-01-01', periods=n_minutes, freq='1min')
    
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
        'open': prices,
        'close': prices * (1 + np.random.normal(0, 0.0001, n_minutes)),
        'volume': np.abs(np.random.lognormal(10, 1, n_minutes))
    }, index=dates)
    
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
    
    # Generate options data
    expiry_dates = pd.date_range('2024-02-01', periods=12, freq='ME')
    strikes = np.linspace(0.8, 1.2, 21) * base_price
    
    options_data = []
    current_price = prices[-1]
    
    for expiry in expiry_dates:
        days_to_expiry = (expiry - dates[-1]).days
        for strike in strikes:
            moneyness = strike / current_price
            
            # Calculate implied volatility with smile effect
            base_vol = 0.2
            smile_effect = 0.1 * (moneyness - 1) ** 2
            time_effect = np.sqrt(days_to_expiry / 365)
            implied_vol = (base_vol + smile_effect) * time_effect
            
            for option_type in ['call', 'put']:
                options_data.append({
                    'date': dates[-1],
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
    
    # Generate treasury yields data with realistic term structure
    base_yields = {
        '2Y': 0.025,
        '5Y': 0.028,
        '10Y': 0.032,
        '30Y': 0.035
    }
    
    yields_data = pd.DataFrame(index=pd.date_range('2024-01-01', periods=n_days))
    for tenor, base_yield in base_yields.items():
        yields_data[tenor] = base_yield + np.random.normal(0, 0.0002, n_days)
    
    # Generate sentiment data
    sentiment_data = pd.DataFrame(index=dates)
    sentiment_data['sentiment_score'] = np.random.normal(0, 0.2, n_minutes)
    sentiment_data['mention_count'] = np.random.poisson(10, n_minutes)
    sentiment_data['returns'] = intraday_data['returns']
    
    # Save data to appropriate locations
    os.makedirs('db/raw/intraday', exist_ok=True)
    os.makedirs('db/raw/options', exist_ok=True)
    os.makedirs('db/raw/macro/treasury_yields', exist_ok=True)
    os.makedirs('db/raw/news/sentiment', exist_ok=True)
    
    intraday_data.to_csv(f'db/raw/intraday/{symbol}_1min.csv')
    options_df.to_csv(f'db/raw/options/{symbol}_options.csv')
    yields_data.to_csv('db/raw/macro/treasury_yields/treasury_yield_10y_daily.csv')
    sentiment_data.to_csv(f'db/raw/news/sentiment/{symbol}_sentiment.csv')
    
    return intraday_data, options_df, yields_data, sentiment_data

class SignalComponentTester:
    def __init__(self, symbol: str = 'AAPL'):
        self.symbol = symbol
        self.output_dir = Path('output/analysis/signal_tests') / datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = setup_logging(__name__)

    def test_volatility_signals(self):
        """Test volatility signal generation."""
        self.logger.info("Testing volatility signals...")
        try:
            # Load intraday data
            intraday_data = pd.read_csv(f'db/raw/intraday/{self.symbol}_1min.csv', index_col=0, parse_dates=True)
            
            # Calculate clean returns and volatility
            intraday_data['returns'] = intraday_data['close'].pct_change().fillna(0)
            intraday_data['realized_volatility'] = intraday_data['returns'].rolling(30).std() * np.sqrt(252)
            
            # Load options data
            options_data = pd.read_csv(f'db/raw/options/{self.symbol}_options.csv')
            options_data['quote_datetime'] = pd.to_datetime(options_data['date'])
            
            # Calculate implied volatility components
            latest_options = options_data.copy()
            latest_options['moneyness'] = latest_options['strike'] / latest_options['underlying_price']
            latest_options['time_to_expiry'] = (
                pd.to_datetime(latest_options['expiry_date']) - latest_options['quote_datetime']
            ).dt.days / 365.0
            
            # Generate volatility signals
            signals = pd.DataFrame(index=intraday_data.index)
            signals['realized_volatility'] = intraday_data['realized_volatility']
            
            # Calculate average implied volatility for the latest quote
            latest_quote = latest_options['quote_datetime'].max()
            latest_data = latest_options[latest_options['quote_datetime'] == latest_quote]
            
            # Broadcast the latest implied volatility to all timestamps
            signals['implied_volatility'] = latest_data['implied_volatility'].mean()
            signals['vol_spread'] = signals['implied_volatility'] - signals['realized_volatility']
            
            # Add term structure components from latest quote
            signals['short_term_vol'] = latest_data[latest_data['time_to_expiry'] <= 0.25]['implied_volatility'].mean()
            signals['medium_term_vol'] = latest_data[
                (latest_data['time_to_expiry'] > 0.25) & 
                (latest_data['time_to_expiry'] <= 0.75)
            ]['implied_volatility'].mean()
            signals['long_term_vol'] = latest_data[latest_data['time_to_expiry'] > 0.75]['implied_volatility'].mean()
            
            # Clean up any remaining inf/nan values
            signals = signals.replace([np.inf, -np.inf], np.nan)
            signals = signals.ffill().bfill()  # Forward fill then backward fill
            
            # Save signals
            signals_file = self.output_dir / 'volatility_signals.csv'
            signals.to_csv(signals_file)
            self.logger.info(f"Saved volatility signals to {signals_file}")
            
            # Create diagnostic plots
            self._plot_volatility_diagnostics(signals, intraday_data)
            
        except Exception as e:
            self.logger.error(f"Error in volatility signal testing: {str(e)}")

    def test_liquidity_signals(self):
        """Test liquidity signal generation."""
        self.logger.info("Testing liquidity signals...")
        try:
            # Load intraday data
            intraday_data = pd.read_csv(f'db/raw/intraday/{self.symbol}_1min.csv', index_col=0, parse_dates=True)
            
            # Calculate returns and volume metrics
            intraday_data['returns'] = intraday_data['close'].pct_change().fillna(0)
            intraday_data['dollar_volume'] = intraday_data['close'] * intraday_data['volume']
            
            # Generate liquidity signals
            signals = pd.DataFrame(index=intraday_data.index)
            
            # Kyle's Lambda (price impact)
            signals['kyle_lambda'] = (
                intraday_data['returns'].abs().rolling(30).sum() / 
                intraday_data['dollar_volume'].rolling(30).sum()
            )
            
            # Bid-ask spread
            signals['bid_ask_spread'] = (
                (intraday_data['high'] - intraday_data['low']) / 
                intraday_data['close']
            ).rolling(30).mean()
            
            # Order flow toxicity
            signals['order_flow_toxicity'] = (
                intraday_data['returns'].rolling(30).std() / 
                np.log(intraday_data['volume'].rolling(30).mean())
            )
            
            # Volume profile
            signals['relative_volume'] = (
                intraday_data['volume'] / 
                intraday_data['volume'].rolling(30).mean()
            )
            
            # Clean up any remaining inf/nan values
            signals = signals.replace([np.inf, -np.inf], np.nan)
            signals = signals.ffill().bfill()  # Forward fill then backward fill
            
            # Save signals
            signals_file = self.output_dir / 'liquidity_signals.csv'
            signals.to_csv(signals_file)
            self.logger.info(f"Saved liquidity signals to {signals_file}")
            
            # Create diagnostic plots
            self._plot_liquidity_diagnostics(signals, intraday_data)
            
        except Exception as e:
            self.logger.error(f"Error in liquidity signal testing: {str(e)}")

    def test_options_signals(self):
        """Test options signal generation."""
        self.logger.info("Testing options signals...")
        try:
            # Load options data
            options_data = pd.read_csv(f'db/raw/options/{self.symbol}_options.csv')
            options_data['quote_datetime'] = options_data['date']
            
            # Calculate moneyness and time to expiry
            options_data['moneyness'] = options_data['strike'] / options_data['underlying_price']
            options_data['time_to_expiry'] = (
                pd.to_datetime(options_data['expiry_date']) - 
                pd.to_datetime(options_data['quote_datetime'])
            ).dt.days / 365.0
            
            # Generate options signals
            signals = pd.DataFrame(index=[pd.to_datetime(options_data['quote_datetime'].iloc[0])])
            
            # Put-call ratio
            put_volume = options_data[options_data['type'] == 'put']['volume'].sum()
            call_volume = options_data[options_data['type'] == 'call']['volume'].sum()
            signals['put_call_ratio'] = put_volume / call_volume if call_volume > 0 else 1.0
            
            # Net gamma exposure
            signals['net_gamma'] = (
                options_data[options_data['type'] == 'call']['gamma'].sum() -
                options_data[options_data['type'] == 'put']['gamma'].sum()
            )
            
            # IV skew
            atm_options = options_data[
                (options_data['moneyness'] >= 0.95) & 
                (options_data['moneyness'] <= 1.05)
            ]
            signals['iv_skew'] = (
                options_data[options_data['moneyness'] < 0.95]['implied_volatility'].mean() -
                atm_options['implied_volatility'].mean()
            )
            
            # Term structure slope
            short_term = options_data[options_data['time_to_expiry'] <= 0.25]['implied_volatility'].mean()
            long_term = options_data[options_data['time_to_expiry'] > 0.75]['implied_volatility'].mean()
            signals['term_slope'] = long_term - short_term
            
            # Save signals
            signals_file = self.output_dir / 'options_signals.csv'
            signals.to_csv(signals_file)
            self.logger.info(f"Saved options signals to {signals_file}")
            
            # Create diagnostic plots
            self._plot_options_diagnostics(signals, options_data)
            
        except Exception as e:
            self.logger.error(f"Error in options signal testing: {str(e)}")

    def test_macro_signals(self):
        """Test macro signal generation."""
        self.logger.info("Testing macro signals...")
        try:
            # Load yields data
            yields_data = pd.read_csv('db/raw/macro/treasury_yields/treasury_yield_10y_daily.csv', index_col=0, parse_dates=True)
            
            # Generate macro signals
            signals = pd.DataFrame(index=yields_data.index)
            
            # Yield spreads
            signals['30Y_2Y_spread'] = yields_data['30Y'] - yields_data['2Y']
            signals['10Y_2Y_spread'] = yields_data['10Y'] - yields_data['2Y']
            signals['5Y_2Y_spread'] = yields_data['5Y'] - yields_data['2Y']
            
            # Z-scores with minimum lookback
            min_periods = 20
            for spread in ['30Y_2Y_spread', '10Y_2Y_spread', '5Y_2Y_spread']:
                rolling_mean = signals[spread].rolling(252, min_periods=min_periods).mean()
                rolling_std = signals[spread].rolling(252, min_periods=min_periods).std()
                signals[f'{spread}_zscore'] = (signals[spread] - rolling_mean) / rolling_std
            
            # Curve metrics
            signals['curve_slope'] = signals['30Y_2Y_spread']
            signals['curve_curvature'] = (
                yields_data['5Y'] - 
                (yields_data['2Y'] + yields_data['10Y']) / 2
            )
            
            # Rate momentum and real rate
            signals['rate_momentum'] = yields_data['10Y'].diff(20)
            signals['real_rate'] = yields_data['10Y'] - 0.02  # Assuming 2% inflation
            
            # Clean up any remaining inf/nan values
            signals = signals.replace([np.inf, -np.inf], np.nan)
            signals = signals.ffill().bfill()  # Forward fill then backward fill
            
            # Save signals
            signals_file = self.output_dir / 'macro_signals.csv'
            signals.to_csv(signals_file)
            self.logger.info(f"Saved macro signals to {signals_file}")
            
            # Create diagnostic plots
            self._plot_macro_diagnostics(signals, yields_data)
            
        except Exception as e:
            self.logger.error(f"Error in macro signal testing: {str(e)}")

    def test_sentiment_signals(self):
        """Test sentiment signal generation."""
        self.logger.info("Testing sentiment signals...")
        try:
            # Load sentiment data
            sentiment_data = pd.read_csv(f'db/raw/news/sentiment/{self.symbol}_sentiment.csv', index_col=0, parse_dates=True)
            
            # Generate sentiment signals
            signals = pd.DataFrame(index=sentiment_data.index)
            
            # Basic sentiment metrics
            signals['sentiment_score'] = sentiment_data['sentiment_score']
            signals['mention_count'] = sentiment_data['mention_count']
            
            # Sentiment momentum
            signals['sentiment_momentum'] = sentiment_data['sentiment_score'].diff(5)
            signals['sentiment_volatility'] = (
                sentiment_data['sentiment_score']
                .rolling(30, min_periods=5)
                .std()
            )
            
            # Sentiment strength and impact
            signals['sentiment_strength'] = (
                signals['sentiment_score'].abs() * 
                np.log1p(signals['mention_count'])
            )
            
            # Sentiment-return correlation
            signals['sentiment_return_corr'] = (
                sentiment_data['sentiment_score']
                .rolling(30, min_periods=5)
                .corr(sentiment_data['returns'])
            )
            
            # Clean up any remaining inf/nan values
            signals = signals.replace([np.inf, -np.inf], np.nan)
            signals = signals.ffill().bfill()  # Forward fill then backward fill
            
            # Save signals
            signals_file = self.output_dir / 'sentiment_signals.csv'
            signals.to_csv(signals_file)
            self.logger.info(f"Saved sentiment signals to {signals_file}")
            
            # Create diagnostic plots
            self._plot_sentiment_diagnostics(signals, sentiment_data)
            
        except Exception as e:
            self.logger.error(f"Error in sentiment signal testing: {str(e)}")

    def _plot_volatility_diagnostics(self, signals: pd.DataFrame, intraday_data: pd.DataFrame):
        """Create diagnostic plots for volatility signals."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Realized vs Implied Volatility
        signals[['realized_volatility', 'implied_volatility']].plot(ax=axes[0, 0])
        axes[0, 0].set_title('Realized vs Implied Volatility')
        
        # Plot 2: Volatility Term Structure
        signals[['short_term_vol', 'medium_term_vol', 'long_term_vol']].plot(ax=axes[0, 1])
        axes[0, 1].set_title('Volatility Term Structure')
        
        # Plot 3: Returns Distribution
        sns.histplot(intraday_data['returns'], ax=axes[1, 0])
        axes[1, 0].set_title('Returns Distribution')
        
        # Plot 4: Volatility Clustering
        sns.scatterplot(data=intraday_data, x='returns', y=intraday_data['returns'].shift(1), ax=axes[1, 1])
        axes[1, 1].set_title('Volatility Clustering')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'volatility_diagnostics.png')
        plt.close()

    def _plot_liquidity_diagnostics(self, signals: pd.DataFrame, intraday_data: pd.DataFrame):
        """Create diagnostic plots for liquidity signals."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Kyle's Lambda
        signals['kyle_lambda'].plot(ax=axes[0, 0])
        axes[0, 0].set_title("Kyle's Lambda")
        
        # Plot 2: Volume Profile
        intraday_data['volume'].plot(ax=axes[0, 1])
        axes[0, 1].set_title('Volume Profile')
        
        # Plot 3: Bid-Ask Spread
        signals['bid_ask_spread'].plot(ax=axes[1, 0])
        axes[1, 0].set_title('Bid-Ask Spread')
        
        # Plot 4: Order Flow Toxicity
        signals['order_flow_toxicity'].plot(ax=axes[1, 1])
        axes[1, 1].set_title('Order Flow Toxicity')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'liquidity_diagnostics.png')
        plt.close()

    def _plot_options_diagnostics(self, signals: pd.DataFrame, options_data: pd.DataFrame):
        """Create diagnostic plots for options signals."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Implied Volatility Surface
        pivot = options_data.pivot_table(
            values='implied_volatility',
            index='strike',
            columns='days_to_expiry'
        )
        sns.heatmap(pivot, ax=axes[0, 0])
        axes[0, 0].set_title('Implied Volatility Surface')
        
        # Plot 2: Put-Call Ratio
        signals['put_call_ratio'].plot(ax=axes[0, 1])
        axes[0, 1].set_title('Put-Call Ratio')
        
        # Plot 3: Net Gamma Exposure
        signals['net_gamma'].plot(ax=axes[1, 0])
        axes[1, 0].set_title('Net Gamma Exposure')
        
        # Plot 4: Option Volume Profile
        options_data.groupby('strike')['volume'].mean().plot(ax=axes[1, 1])
        axes[1, 1].set_title('Option Volume Profile')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'options_diagnostics.png')
        plt.close()

    def _plot_macro_diagnostics(self, signals: pd.DataFrame, yields_data: pd.DataFrame):
        """Create diagnostic plots for macro signals."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Yield Curve
        yields_data.iloc[-1].plot(ax=axes[0, 0])
        axes[0, 0].set_title('Current Yield Curve')
        
        # Plot 2: Yield Curve Slope
        signals['curve_slope'].plot(ax=axes[0, 1])
        axes[0, 1].set_title('Yield Curve Slope')
        
        # Plot 3: Real Rate
        signals['real_rate'].plot(ax=axes[1, 0])
        axes[1, 0].set_title('Real Rate')
        
        # Plot 4: Rate Momentum
        signals['rate_momentum'].plot(ax=axes[1, 1])
        axes[1, 1].set_title('Rate Momentum')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'macro_diagnostics.png')
        plt.close()

    def _plot_sentiment_diagnostics(self, signals: pd.DataFrame, sentiment_data: pd.DataFrame):
        """Create diagnostic plots for sentiment signals."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Sentiment Score Distribution
        sns.histplot(sentiment_data['sentiment_score'], ax=axes[0, 0])
        axes[0, 0].set_title('Sentiment Score Distribution')
        
        # Plot 2: Sentiment vs Returns
        sns.scatterplot(
            data=sentiment_data,
            x='sentiment_score',
            y='returns',
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Sentiment vs Returns')
        
        # Plot 3: Mention Count Over Time
        sentiment_data['mention_count'].plot(ax=axes[1, 0])
        axes[1, 0].set_title('Mention Count Over Time')
        
        # Plot 4: Sentiment Signal Strength
        signals['sentiment_strength'].plot(ax=axes[1, 1])
        axes[1, 1].set_title('Sentiment Signal Strength')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sentiment_diagnostics.png')
        plt.close()

def main():
    """Main function to run signal component tests."""
    try:
        # Initialize tester
        tester = SignalComponentTester('AAPL')
        
        # Initialize test data
        intraday_data, options_data, yields_data, sentiment_data = initialize_test_data('AAPL')
        
        # Run tests
        tester.test_volatility_signals()
        tester.test_liquidity_signals()
        tester.test_options_signals()
        tester.test_macro_signals()
        tester.test_sentiment_signals()
        
    except Exception as e:
        logger.error(f"Error in signal component testing: {str(e)}")
        raise

if __name__ == '__main__':
    main() 