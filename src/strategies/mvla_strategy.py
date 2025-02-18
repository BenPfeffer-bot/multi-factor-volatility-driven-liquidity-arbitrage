"""
Multi-Factor Volatility-Driven Liquidity Arbitrage (MVLA) Strategy.

This module implements a sophisticated arbitrage strategy that leverages:
1. Intraday volatility clustering
2. Options market dislocations
3. Liquidity shock analysis
4. Macro-driven volatility premia
5. News sentiment impact
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import statsmodels.api as sm
from arch import arch_model

from src.utils.data_management import market_data_manager
from src.config.settings import DJ_TITANS_50_TICKER, VOLATILITY_WINDOW
from src.config.paths import *
from src.strategies.volatility_metrics import (
    calculate_multi_timeframe_rv,
    calculate_amihud_illiquidity,
    calculate_order_flow_toxicity,
    calculate_options_skew_features,
    calculate_yield_curve_features,
    calculate_sentiment_impact_features
)
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)

class MVLAStrategy:
    """Multi-Factor Volatility-Driven Liquidity Arbitrage Strategy."""
    
    def __init__(
        self,
        symbols: List[str] = DJ_TITANS_50_TICKER,
        lookback_days: int = 20,
        vol_window: int = VOLATILITY_WINDOW,
        min_liquidity_threshold: float = 100000,
        sentiment_impact_threshold: float = 0.5,
        iv_rank_threshold: float = 0.7,
        vol_signal_threshold: float = 2.0,
        liquidity_signal_threshold: float = 2.5,
        macro_impact_threshold: float = 1.5
    ):
        """
        Initialize the MVLA strategy.
        
        Args:
            symbols: List of symbols to trade
            lookback_days: Historical lookback period for feature calculation
            vol_window: Window for volatility calculations
            min_liquidity_threshold: Minimum volume for liquidity consideration
            sentiment_impact_threshold: Threshold for significant sentiment impact
            iv_rank_threshold: Threshold for IV rank signals
            vol_signal_threshold: Z-score threshold for volatility signals
            liquidity_signal_threshold: Z-score threshold for liquidity signals
            macro_impact_threshold: Z-score threshold for macro impact
        """
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.vol_window = vol_window
        self.min_liquidity_threshold = min_liquidity_threshold
        self.sentiment_impact_threshold = sentiment_impact_threshold
        self.iv_rank_threshold = iv_rank_threshold
        self.vol_signal_threshold = vol_signal_threshold
        self.liquidity_signal_threshold = liquidity_signal_threshold
        self.macro_impact_threshold = macro_impact_threshold
        
        # Initialize data manager reference
        self.market_data_manager = market_data_manager
        
        # Initialize data containers
        self.intraday_data: Dict[str, pd.DataFrame] = {}
        self.options_data: Dict[str, pd.DataFrame] = {}
        self.macro_data: Dict[str, pd.DataFrame] = {}
        self.sentiment_data: Dict[str, pd.DataFrame] = {}
        
        # Feature containers
        self.volatility_features: Dict[str, pd.DataFrame] = {}
        self.liquidity_features: Dict[str, pd.DataFrame] = {}
        self.options_features: Dict[str, pd.DataFrame] = {}
        self.macro_features: Dict[str, pd.DataFrame] = {}
        self.sentiment_features: Dict[str, pd.DataFrame] = {}
        
    def load_data(self, start_date: str, end_date: str) -> None:
        """
        Load all required data for the strategy with validation and synchronization checks.
        
        Args:
            start_date: Start date for data loading
            end_date: End date for data loading
        """
        logger.info(f"Loading data for {len(self.symbols)} symbols from {start_date} to {end_date}")
        
        # Load and validate data for each symbol
        for symbol in self.symbols:
            logger.info(f"Loading data for {symbol}")
            
            # Validate data completeness
            validation_results = self.market_data_manager.validate_data_completeness(
                symbol, start_date, end_date
            )
            
            if 'error' in validation_results:
                logger.error(f"Data validation failed for {symbol}: {validation_results['error']}")
                continue
                
            # Load intraday data
            self.intraday_data[symbol] = self.market_data_manager.load_intraday_data(
                symbol=symbol,
                interval="1min",
                start_date=start_date,
                end_date=end_date
            )
            
            if self.intraday_data[symbol] is None:
                logger.warning(f"No intraday data available for {symbol}")
                continue
            
            # Load options data with synchronization
            synced_data = self.market_data_manager.sync_intraday_with_options(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if synced_data:
                self.options_data[symbol] = synced_data['options']
            else:
                logger.warning(f"No synchronized options data available for {symbol}")
            
            # Load sentiment data with price alignment
            aligned_data = self.market_data_manager.align_sentiment_with_price(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if aligned_data is not None:
                self.sentiment_data[symbol] = aligned_data
            else:
                logger.warning(f"No aligned sentiment data available for {symbol}")
        
        # Load macro data
        logger.info("Loading macro data")
        
        # Load treasury yields
        self.macro_data['treasury_yields'] = self.market_data_manager.load_macro_data(
            data_type='treasury_yields',
            interval='daily',
            start_date=start_date,
            end_date=end_date
        )
        
        if self.macro_data['treasury_yields'] is None:
            logger.warning("No treasury yields data available")
        
        # Load federal funds data
        self.macro_data['federal_funds'] = self.market_data_manager.load_macro_data(
            data_type='federal_funds',
            interval='daily',
            start_date=start_date,
            end_date=end_date
        )
        
        if self.macro_data['federal_funds'] is None:
            logger.warning("No federal funds data available")
        
        # Validate loaded data
        self._validate_loaded_data()
        
    def _validate_loaded_data(self) -> None:
        """Validate completeness and quality of loaded data."""
        valid_symbols = []
        
        for symbol in self.symbols:
            # Check if all required data is available
            has_intraday = symbol in self.intraday_data and not self.intraday_data[symbol].empty
            has_options = symbol in self.options_data and not self.options_data[symbol].empty
            has_sentiment = symbol in self.sentiment_data and not self.sentiment_data[symbol].empty
            
            if has_intraday and has_options and has_sentiment:
                # Perform quality checks
                intraday_quality = self.market_data_manager._check_data_quality(
                    self.intraday_data[symbol]
                )
                
                # Check for sufficient liquidity
                avg_volume = self.intraday_data[symbol]['volume'].mean()
                if avg_volume >= self.min_liquidity_threshold:
                    valid_symbols.append(symbol)
                else:
                    logger.warning(
                        f"Insufficient liquidity for {symbol}: "
                        f"average volume {avg_volume} < threshold {self.min_liquidity_threshold}"
                    )
            else:
                logger.warning(
                    f"Incomplete data for {symbol} - "
                    f"intraday: {has_intraday}, options: {has_options}, "
                    f"sentiment: {has_sentiment}"
                )
        
        # Update symbols list to include only valid symbols
        self.symbols = valid_symbols
        logger.info(f"Valid symbols after data validation: {len(valid_symbols)}")
        
    def calculate_intraday_volatility_features(self, symbol: str) -> pd.DataFrame:
        """
        Calculate enhanced intraday volatility features.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with volatility features
        """
        data = self.intraday_data[symbol].copy()
        
        # Calculate multi-timeframe realized volatility
        rv_features = calculate_multi_timeframe_rv(
            data,
            windows=[5, 15, 30, 60],
            annualization=252 * 390
        )
        
        # Calculate volatility regime features
        for col in rv_features.columns:
            rv_features[f'{col}_zscore'] = (
                rv_features[col] - 
                rv_features[col].rolling(window=self.vol_window).mean()
            ) / rv_features[col].rolling(window=self.vol_window).std()
            
            rv_features[f'{col}_regime'] = np.where(
                rv_features[f'{col}_zscore'] > self.vol_signal_threshold,
                1,  # High volatility regime
                np.where(
                    rv_features[f'{col}_zscore'] < -self.vol_signal_threshold,
                    -1,  # Low volatility regime
                    0  # Normal regime
                )
            )
        
        return rv_features
    
    def calculate_liquidity_features(self, symbol: str) -> pd.DataFrame:
        """
        Calculate enhanced liquidity features.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with liquidity features
        """
        data = self.intraday_data[symbol].copy()
        
        # Calculate Amihud illiquidity
        illiquidity = calculate_amihud_illiquidity(data, window=self.vol_window)
        
        # Calculate order flow toxicity
        flow_metrics = calculate_order_flow_toxicity(data, window=self.vol_window)
        
        # Enhanced liquidity metrics
        liq_features = pd.DataFrame(index=data.index)
        liq_features['illiquidity'] = illiquidity
        
        # Calculate relative spread
        liq_features['relative_spread'] = (data['high'] - data['low']) / data['close']
        
        # Calculate volume-weighted effective spread
        vwap = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        liq_features['effective_spread'] = 2 * abs(data['close'] - vwap) / vwap
        
        # Calculate Hasbrouck's lambda (price impact coefficient)
        returns = np.log(data['close'] / data['close'].shift(1))
        signed_volume = data['volume'] * np.sign(returns)
        
        def calculate_hasbrouck_lambda(x, y, window):
            if len(x) < window:
                return np.nan
            model = sm.OLS(y, sm.add_constant(x)).fit()
            return model.params[1]
        
        liq_features['price_impact'] = pd.Series(
            [calculate_hasbrouck_lambda(
                signed_volume[max(0, i-self.vol_window):i],
                returns[max(0, i-self.vol_window):i],
                self.vol_window
            ) for i in range(len(returns))],
            index=data.index
        )
        
        # Calculate volume imbalance
        buy_volume = data['volume'] * (data['close'] > data['open']).astype(int)
        sell_volume = data['volume'] * (data['close'] <= data['open']).astype(int)
        liq_features['volume_imbalance'] = (
            (buy_volume - sell_volume) / (buy_volume + sell_volume)
        ).rolling(window=self.vol_window).mean()
        
        # Calculate turnover ratio
        liq_features['turnover'] = (
            data['volume'] / data['volume'].rolling(window=self.vol_window).mean()
        )
        
        # Add flow metrics
        liq_features = pd.concat([liq_features, flow_metrics], axis=1)
        
        # Calculate liquidity timing indicator
        def calculate_liquidity_timing(data, window):
            vol = data['volume'].rolling(window=window).std()
            spread = data['relative_spread'].rolling(window=window).mean()
            return (vol * spread).rank(pct=True)
        
        liq_features['liquidity_timing'] = calculate_liquidity_timing(
            data, self.vol_window
        )
        
        # Calculate z-scores for all features
        for col in liq_features.columns:
            liq_features[f'{col}_zscore'] = (
                liq_features[col] - 
                liq_features[col].rolling(window=self.vol_window).mean()
            ) / liq_features[col].rolling(window=self.vol_window).std()
        
        # Identify liquidity regimes with enhanced logic
        liq_features['liquidity_regime'] = 0
        
        # Highly illiquid regime (multiple confirmations required)
        illiquid_conditions = (
            (liq_features['illiquidity_zscore'] > self.liquidity_signal_threshold) &
            (liq_features['effective_spread_zscore'] > self.liquidity_signal_threshold) &
            (liq_features['price_impact_zscore'] > self.liquidity_signal_threshold) &
            (abs(liq_features['volume_imbalance_zscore']) > self.liquidity_signal_threshold) &
            (liq_features['turnover_zscore'] < -self.liquidity_signal_threshold)
        )
        
        # Highly liquid regime
        liquid_conditions = (
            (liq_features['illiquidity_zscore'] < -self.liquidity_signal_threshold) &
            (liq_features['effective_spread_zscore'] < -self.liquidity_signal_threshold) &
            (liq_features['price_impact_zscore'] < -self.liquidity_signal_threshold) &
            (abs(liq_features['volume_imbalance_zscore']) < self.liquidity_signal_threshold) &
            (liq_features['turnover_zscore'] > self.liquidity_signal_threshold)
        )
        
        liq_features.loc[illiquid_conditions, 'liquidity_regime'] = -1
        liq_features.loc[liquid_conditions, 'liquidity_regime'] = 1
        
        # Calculate regime conviction scores
        liq_features['regime_conviction'] = (
            abs(liq_features['illiquidity_zscore']) +
            abs(liq_features['effective_spread_zscore']) +
            abs(liq_features['price_impact_zscore']) +
            abs(liq_features['volume_imbalance_zscore']) +
            abs(liq_features['turnover_zscore'])
        ) / 5
        
        return liq_features
    
    def calculate_options_features(self, symbol: str) -> pd.DataFrame:
        """
        Calculate enhanced options market features.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with options features
        """
        options = self.options_data[symbol].copy()
        
        # Calculate advanced skew features
        skew_features = calculate_options_skew_features(
            options,
            window=self.vol_window
        )
        
        # Add volatility surface analysis
        def calculate_surface_metrics(data):
            # Calculate wing steepness (10-delta vs 25-delta)
            put_wing = (
                data[data['delta'].between(-0.1, -0.09)]['implied_volatility'].mean() -
                data[data['delta'].between(-0.25, -0.24)]['implied_volatility'].mean()
            )
            call_wing = (
                data[data['delta'].between(0.09, 0.1)]['implied_volatility'].mean() -
                data[data['delta'].between(0.24, 0.25)]['implied_volatility'].mean()
            )
            
            # Calculate surface curvature
            atm_iv = data[data['delta'].between(-0.5, 0.5)]['implied_volatility'].mean()
            wing_iv = (
                data[data['delta'].abs().between(0.1, 0.25)]['implied_volatility'].mean()
            )
            curvature = wing_iv - atm_iv
            
            return pd.Series({
                'put_wing_steepness': put_wing,
                'call_wing_steepness': call_wing,
                'surface_curvature': curvature
            })
        
        surface_features = options.groupby(level=0).apply(calculate_surface_metrics)
        
        # Add volatility risk premium features
        realized_vol = self.volatility_features[symbol]['rv_5min'].resample('D').last()
        vrp_features = pd.DataFrame(index=options.index)
        
        vrp_features['vol_risk_premium'] = (
            options.groupby('date')['implied_volatility'].mean() - 
            realized_vol
        )
        
        vrp_features['vrp_zscore'] = (
            vrp_features['vol_risk_premium'] -
            vrp_features['vol_risk_premium'].rolling(window=self.vol_window).mean()
        ) / vrp_features['vol_risk_premium'].rolling(window=self.vol_window).std()
        
        # Add volatility cone analysis
        def calculate_vol_cone(data):
            iv_percentiles = data.groupby('days_to_expiry')['implied_volatility'].agg([
                ('iv_min', 'min'),
                ('iv_25', lambda x: x.quantile(0.25)),
                ('iv_median', 'median'),
                ('iv_75', lambda x: x.quantile(0.75)),
                ('iv_max', 'max')
            ])
            return iv_percentiles
        
        vol_cone = calculate_vol_cone(options)
        
        # Calculate vol cone position scores
        current_iv = options.groupby('days_to_expiry')['implied_volatility'].mean()
        vol_cone_scores = pd.DataFrame(index=current_iv.index)
        
        vol_cone_scores['vol_richness'] = (
            (current_iv - vol_cone['iv_median']) /
            (vol_cone['iv_75'] - vol_cone['iv_25'])
        )
        
        # Combine all options features
        opt_features = pd.concat([
            skew_features,
            surface_features,
            vrp_features,
            vol_cone_scores
        ], axis=1)
        
        return opt_features
    
    def calculate_macro_features(self) -> pd.DataFrame:
        """
        Calculate enhanced macro-driven volatility features.
        
        Returns:
            DataFrame with macro features
        """
        # Calculate yield curve features
        yields = self.macro_data['treasury_yields'].copy()
        curve_features = calculate_yield_curve_features(
            yields,
            window=self.vol_window
        )
        
        # Calculate federal funds features
        fed_funds = self.macro_data['federal_funds'].copy()
        fed_features = pd.DataFrame(index=fed_funds.index)
        
        fed_features['rate_change'] = fed_funds['value'].diff()
        fed_features['rate_vol'] = fed_funds['value'].rolling(
            window=self.vol_window
        ).std()
        
        # Combine features
        macro_features = pd.concat([curve_features, fed_features], axis=1)
        
        # Calculate macro impact score
        macro_features['macro_impact'] = (
            abs(macro_features['yield_change']) * 
            macro_features['yield_vol'] * 
            abs(macro_features['rate_change'])
        )
        
        macro_features['macro_impact_zscore'] = (
            macro_features['macro_impact'] -
            macro_features['macro_impact'].rolling(window=self.vol_window).mean()
        ) / macro_features['macro_impact'].rolling(window=self.vol_window).std()
        
        return macro_features
    
    def calculate_sentiment_features(self, symbol: str) -> pd.DataFrame:
        """
        Calculate enhanced sentiment-based features.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with sentiment features
        """
        sentiment = self.sentiment_data[symbol].copy()
        price_data = self.intraday_data[symbol].copy()
        
        # Calculate sentiment impact features
        impact_features = calculate_sentiment_impact_features(
            sentiment,
            price_data,
            window=self.vol_window
        )
        
        return impact_features
    
    def generate_volatility_signals(self, symbol: str) -> pd.DataFrame:
        """
        Generate volatility-based arbitrage signals by analyzing volatility clusters and dislocations.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with volatility signals
        """
        try:
            # Get synchronized intraday and options data
            synced_data = self.market_data_manager.sync_intraday_with_options(symbol)
            if not synced_data:
                return pd.DataFrame()
                
            intraday = synced_data['intraday']
            options = synced_data['options']
            
            # 1. Calculate multi-timeframe realized volatility using Parkinson estimator
            rv_features = calculate_multi_timeframe_rv(
                intraday,
                windows=[5, 15, 30, 60],  # minutes
                annualization=252 * 390
            )
            
            # 2. Calculate implied volatility surface metrics
            atm_options = options[
                (options['moneyness'] >= 0.95) &
                (options['moneyness'] <= 1.05)
            ]
            
            # Group by expiry for term structure
            iv_term = atm_options.groupby(['expiry_date'])['implied_volatility'].mean()
            front_month_iv = iv_term.iloc[0] if not iv_term.empty else None
            back_month_iv = iv_term.iloc[-1] if len(iv_term) > 1 else None
            
            # 3. Compute enhanced volatility signals
            signals = pd.DataFrame(index=intraday.index)
            
            # IV-RV spread signals for multiple timeframes
            for window in [5, 15, 30, 60]:
                rv_col = f'rv_{window}min'
                hl_rv_col = f'hl_rv_{window}min'  # Parkinson estimator
                gk_rv_col = f'gk_rv_{window}min'  # Garman-Klass estimator
                
                # Calculate spreads using different estimators
                signals[f'vol_spread_{window}min'] = front_month_iv - rv_features[rv_col]
                signals[f'hl_spread_{window}min'] = front_month_iv - rv_features[hl_rv_col]
                signals[f'gk_spread_{window}min'] = front_month_iv - rv_features[gk_rv_col]
                
                # Z-scores for each spread
                for spread_type in ['vol', 'hl', 'gk']:
                    col = f'{spread_type}_spread_{window}min'
                    signals[f'{col}_zscore'] = (
                        signals[col] -
                        signals[col].rolling(window=self.vol_window).mean()
                    ) / signals[col].rolling(window=self.vol_window).std()
            
            # 4. Detect volatility clusters using GARCH(1,1)
            returns = np.log(intraday['close'] / intraday['close'].shift(1))
            garch_model = arch_model(returns, vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            signals['conditional_vol'] = np.sqrt(garch_fit.conditional_volatility)
            
            # 5. Calculate volatility risk premium
            if front_month_iv is not None and back_month_iv is not None:
                signals['term_structure'] = back_month_iv - front_month_iv
                signals['term_structure_zscore'] = (
                    signals['term_structure'] -
                    signals['term_structure'].rolling(window=self.vol_window).mean()
                ) / signals['term_structure'].rolling(window=self.vol_window).std()
            
            # 6. Generate trading signals with enhanced logic
            signals['vol_signal'] = 0
            
            # Short volatility conditions (multiple confirmations required)
            short_vol_conditions = (
                (signals['vol_spread_5min_zscore'] > self.vol_signal_threshold) &
                (signals['hl_spread_5min_zscore'] > self.vol_signal_threshold) &
                (signals['gk_spread_5min_zscore'] > self.vol_signal_threshold) &
                (signals['conditional_vol'] < signals['conditional_vol'].rolling(window=self.vol_window).mean()) &
                (signals.get('term_structure_zscore', 0) > 0)  # Backwardation
            )
            
            # Long volatility conditions
            long_vol_conditions = (
                (signals['vol_spread_5min_zscore'] < -self.vol_signal_threshold) &
                (signals['hl_spread_5min_zscore'] < -self.vol_signal_threshold) &
                (signals['gk_spread_5min_zscore'] < -self.vol_signal_threshold) &
                (signals['conditional_vol'] > signals['conditional_vol'].rolling(window=self.vol_window).mean()) &
                (signals.get('term_structure_zscore', 0) < 0)  # Contango
            )
            
            # Apply signals with position sizing based on conviction
            signals.loc[short_vol_conditions, 'vol_signal'] = -1
            signals.loc[long_vol_conditions, 'vol_signal'] = 1
            
            # Add signal conviction metrics
            signals['vol_conviction'] = 0
            
            # Add conviction for short signals
            signals.loc[signals['vol_signal'] == -1, 'vol_conviction'] = (
                abs(signals['vol_spread_5min_zscore']) +
                abs(signals['hl_spread_5min_zscore']) +
                abs(signals['gk_spread_5min_zscore'])
            ) / 3
            
            # Add conviction for long signals
            signals.loc[signals['vol_signal'] == 1, 'vol_conviction'] = (
                abs(signals['vol_spread_5min_zscore']) +
                abs(signals['hl_spread_5min_zscore']) +
                abs(signals['gk_spread_5min_zscore'])
            ) / 3
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating volatility signals for {symbol}: {str(e)}")
            return pd.DataFrame()

    def generate_liquidity_signals(self, symbol: str) -> pd.DataFrame:
        """
        Generate liquidity-based arbitrage signals using order book imbalance detection.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with liquidity signals
        """
        try:
            # Get intraday data
            intraday = self.market_data_manager.load_intraday_data(symbol)
            if intraday is None:
                return pd.DataFrame()
            
            # 1. Calculate Kyle's Lambda (price impact)
            returns = np.log(intraday['close'] / intraday['close'].shift(1))
            dollar_volume = intraday['close'] * intraday['volume']
            signed_volume = dollar_volume * np.sign(returns)
            
            def calculate_kyle_lambda(data, window):
                model = sm.OLS(
                    data['returns'],
                    sm.add_constant(data['signed_volume'])
                ).fit()
                return model.params[1]  # Kyle's Lambda coefficient
            
            # Calculate rolling Kyle's Lambda
            signals = pd.DataFrame(index=intraday.index)
            signals['kyle_lambda'] = returns.rolling(
                window=self.vol_window
            ).apply(
                lambda x: calculate_kyle_lambda(
                    pd.DataFrame({
                        'returns': x,
                        'signed_volume': signed_volume[x.index]
                    }),
                    self.vol_window
                )
            )
            
            # 2. Calculate enhanced Amihud illiquidity
            illiquidity = calculate_amihud_illiquidity(intraday, window=self.vol_window)
            signals['illiquidity'] = illiquidity
            
            # 3. Calculate advanced order flow toxicity metrics
            flow_metrics = calculate_order_flow_toxicity(intraday, window=self.vol_window)
            signals = pd.concat([signals, flow_metrics], axis=1)
            
            # 4. Calculate additional microstructure metrics
            # Bid-ask spread estimation using Roll's model
            signals['roll_spread'] = 2 * np.sqrt(abs(
                returns.rolling(window=self.vol_window).autocorr()
            )) * intraday['close']
            
            # Volume-weighted average price deviation
            vwap = (intraday['close'] * intraday['volume']).cumsum() / intraday['volume'].cumsum()
            signals['vwap_deviation'] = abs(intraday['close'] - vwap) / vwap
            
            # Order imbalance metrics
            signals['order_imbalance'] = (
                (intraday['close'] - intraday['open']) /
                (intraday['high'] - intraday['low']).replace(0, np.nan)
            )
            
            # 5. Calculate z-scores for all metrics
            for col in [
                'kyle_lambda', 'illiquidity', 'vpin', 'flow_imbalance',
                'effective_spread', 'roll_spread', 'vwap_deviation',
                'order_imbalance'
            ]:
                signals[f'{col}_zscore'] = (
                    signals[col] -
                    signals[col].rolling(window=self.vol_window).mean()
                ) / signals[col].rolling(window=self.vol_window).std()
            
            # 6. Generate trading signals with sophisticated logic
            signals['liquidity_signal'] = 0
            
            # Extreme illiquidity conditions (multiple confirmations required)
            illiquid_conditions = (
                (signals['kyle_lambda_zscore'] > self.liquidity_signal_threshold) &
                (signals['illiquidity_zscore'] > self.liquidity_signal_threshold) &
                (signals['vpin_zscore'] > self.liquidity_signal_threshold) &
                (signals['roll_spread_zscore'] > self.liquidity_signal_threshold) &
                (signals['vwap_deviation_zscore'] > self.liquidity_signal_threshold)
            )
            
            # High liquidity conditions
            liquid_conditions = (
                (signals['kyle_lambda_zscore'] < -self.liquidity_signal_threshold) &
                (signals['illiquidity_zscore'] < -self.liquidity_signal_threshold) &
                (signals['vpin_zscore'] < -self.liquidity_signal_threshold) &
                (signals['roll_spread_zscore'] < -self.liquidity_signal_threshold) &
                (signals['vwap_deviation_zscore'] < -self.liquidity_signal_threshold)
            )
            
            # Apply signals with conviction scores
            signals.loc[illiquid_conditions, 'liquidity_signal'] = -1  # Sell in illiquid conditions
            signals.loc[liquid_conditions, 'liquidity_signal'] = 1     # Buy in liquid conditions
            
            # Calculate signal conviction
            signals['liquidity_conviction'] = 0
            
            # Conviction for illiquid signals
            signals.loc[signals['liquidity_signal'] == -1, 'liquidity_conviction'] = (
                abs(signals['kyle_lambda_zscore']) +
                abs(signals['illiquidity_zscore']) +
                abs(signals['vpin_zscore']) +
                abs(signals['roll_spread_zscore']) +
                abs(signals['vwap_deviation_zscore'])
            ) / 5
            
            # Conviction for liquid signals
            signals.loc[signals['liquidity_signal'] == 1, 'liquidity_conviction'] = (
                abs(signals['kyle_lambda_zscore']) +
                abs(signals['illiquidity_zscore']) +
                abs(signals['vpin_zscore']) +
                abs(signals['roll_spread_zscore']) +
                abs(signals['vwap_deviation_zscore'])
            ) / 5
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating liquidity signals for {symbol}: {str(e)}")
            return pd.DataFrame()

    def generate_options_signals(self, symbol: str) -> pd.DataFrame:
        """
        Generate options-based arbitrage signals using skew and term structure analysis.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with options signals
        """
        try:
            # Get options data
            options = self.market_data_manager.load_options_data(symbol)
            if options is None:
                return pd.DataFrame()
            
            # 1. Calculate enhanced skew features
            skew_features = calculate_options_skew_features(options, window=self.vol_window)
            
            # Initialize signals DataFrame
            signals = pd.DataFrame(index=options.index)
            
            # 2. Calculate put-call skew ratios at different deltas
            def calculate_skew_ratio(data, delta_range):
                puts = data[
                    (data['type'] == 'put') &
                    (data['delta'].abs() >= delta_range[0]) &
                    (data['delta'].abs() <= delta_range[1])
                ]
                calls = data[
                    (data['type'] == 'call') &
                    (data['delta'] >= delta_range[0]) &
                    (data['delta'] <= delta_range[1])
                ]
                put_iv = puts.groupby(level=0)['implied_volatility'].mean()
                call_iv = calls.groupby(level=0)['implied_volatility'].mean()
                return put_iv / call_iv
            
            # Calculate skew ratios for different delta ranges
            delta_ranges = [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
            for low, high in delta_ranges:
                signals[f'skew_ratio_{int(low*100)}_{int(high*100)}'] = calculate_skew_ratio(
                    options, (low, high)
                )
            
            # 3. Calculate term structure features
            def calculate_term_structure(data):
                near_term = data[data['days_to_expiry'] <= 30]['implied_volatility'].mean()
                mid_term = data[
                    (data['days_to_expiry'] > 30) &
                    (data['days_to_expiry'] <= 90)
                ]['implied_volatility'].mean()
                far_term = data[data['days_to_expiry'] > 90]['implied_volatility'].mean()
                
                return pd.Series({
                    'near_mid_spread': mid_term - near_term,
                    'mid_far_spread': far_term - mid_term,
                    'term_slope': (far_term - near_term) / (
                        data[data['days_to_expiry'] > 90]['days_to_expiry'].mean() -
                        data[data['days_to_expiry'] <= 30]['days_to_expiry'].mean()
                    )
                })
            
            term_features = options.groupby(level=0).apply(calculate_term_structure)
            signals = pd.concat([signals, term_features], axis=1)
            
            # 4. Calculate volatility risk premium features
            atm_options = options[
                (options['moneyness'] >= 0.95) &
                (options['moneyness'] <= 1.05)
            ]
            
            signals['atm_iv'] = atm_options.groupby(level=0)['implied_volatility'].mean()
            signals['vrp'] = signals['atm_iv'] - options.groupby(level=0)['realized_vol'].first()
            
            # 5. Calculate gamma exposure
            def calculate_gamma_exposure(data):
                return (data['gamma'] * data['open_interest'] * 100 * data['underlying_price']**2 * 0.01).sum()
            
            signals['gamma_exposure'] = options.groupby(level=0).apply(calculate_gamma_exposure)
            
            # 6. Calculate z-scores for all metrics
            for col in signals.columns:
                if col not in ['options_signal', 'options_conviction']:
                    signals[f'{col}_zscore'] = (
                        signals[col] -
                        signals[col].rolling(window=self.vol_window).mean()
                    ) / signals[col].rolling(window=self.vol_window).std()
            
            # 7. Generate sophisticated trading signals
            signals['options_signal'] = 0
            
            # Overpriced puts signal (high skew + term structure confirmation)
            high_skew_conditions = (
                (signals['skew_ratio_20_30_zscore'] > self.iv_rank_threshold) &
                (signals['skew_ratio_30_40_zscore'] > self.iv_rank_threshold) &
                (signals['near_mid_spread_zscore'] > 0) &
                (signals['vrp_zscore'] > self.iv_rank_threshold) &
                (signals['gamma_exposure_zscore'] < 0)  # Negative gamma environment
            )
            
            # Overpriced calls signal (low skew + term structure confirmation)
            low_skew_conditions = (
                (signals['skew_ratio_20_30_zscore'] < -self.iv_rank_threshold) &
                (signals['skew_ratio_30_40_zscore'] < -self.iv_rank_threshold) &
                (signals['near_mid_spread_zscore'] < 0) &
                (signals['vrp_zscore'] < -self.iv_rank_threshold) &
                (signals['gamma_exposure_zscore'] > 0)  # Positive gamma environment
            )
            
            # Apply signals with conviction scores
            signals.loc[high_skew_conditions, 'options_signal'] = 1  # Sell puts, buy calls
            signals.loc[low_skew_conditions, 'options_signal'] = -1  # Buy puts, sell calls
            
            # Calculate signal conviction
            signals['options_conviction'] = 0
            
            # Conviction for high skew signals
            signals.loc[signals['options_signal'] == 1, 'options_conviction'] = (
                abs(signals['skew_ratio_20_30_zscore']) +
                abs(signals['skew_ratio_30_40_zscore']) +
                abs(signals['near_mid_spread_zscore']) +
                abs(signals['vrp_zscore']) +
                abs(signals['gamma_exposure_zscore'])
            ) / 5
            
            # Conviction for low skew signals
            signals.loc[signals['options_signal'] == -1, 'options_conviction'] = (
                abs(signals['skew_ratio_20_30_zscore']) +
                abs(signals['skew_ratio_30_40_zscore']) +
                abs(signals['near_mid_spread_zscore']) +
                abs(signals['vrp_zscore']) +
                abs(signals['gamma_exposure_zscore'])
            ) / 5
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating options signals for {symbol}: {str(e)}")
            return pd.DataFrame()

    def generate_macro_signals(self) -> pd.DataFrame:
        """
        Generate macro-driven volatility signals based on yield curve analysis.
        
        Returns:
            DataFrame with macro signals
        """
        try:
            # Load yield curve data
            yields = self.market_data_manager.load_macro_data('treasury_yields')
            if yields is None:
                return pd.DataFrame()
            
            # 1. Calculate enhanced yield curve features
            curve_features = calculate_yield_curve_features(yields, window=self.vol_window)
            
            # 2. Calculate yield spreads across multiple tenors
            def calculate_yield_spreads(data):
                spreads = pd.Series()
                tenors = ['2Y', '5Y', '10Y', '30Y']
                
                for i in range(len(tenors)-1):
                    for j in range(i+1, len(tenors)):
                        spread_name = f'{tenors[j]}_{tenors[i]}_spread'
                        spreads[spread_name] = data[tenors[j]] - data[tenors[i]]
                
                # Calculate butterfly spreads
                spreads['5Y_fly'] = data['5Y'] - (data['2Y'] + data['10Y'])/2
                spreads['10Y_fly'] = data['10Y'] - (data['5Y'] + data['30Y'])/2
                
                return spreads
            
            # Calculate daily yield spreads
            spread_features = yields.groupby(level=0).apply(calculate_yield_spreads)
            
            # 3. Add Federal Funds Rate impact
            fed_funds = self.market_data_manager.load_macro_data('federal_funds')
            if fed_funds is not None:
                # Calculate Fed policy features
                fed_features = pd.DataFrame(index=fed_funds.index)
                fed_features['fed_change'] = fed_funds['value'].diff()
                fed_features['fed_momentum'] = fed_funds['value'].diff(5)  # 5-day momentum
                
                # Calculate real rate (Fed Funds - CPI)
                cpi_data = self.market_data_manager.load_macro_data('cpi')
                if cpi_data is not None:
                    fed_features['real_rate'] = fed_funds['value'] - cpi_data['value']
                
                # Combine with spread features
                spread_features = pd.concat([spread_features, fed_features], axis=1)
            
            # 4. Calculate volatility regime features
            signals = pd.DataFrame(index=spread_features.index)
            
            # Calculate rolling volatilities for each spread
            for col in spread_features.columns:
                signals[f'{col}_vol'] = spread_features[col].rolling(
                    window=self.vol_window
                ).std()
            
            # 5. Calculate z-scores for all metrics
            for col in spread_features.columns:
                signals[f'{col}_zscore'] = (
                    spread_features[col] -
                    spread_features[col].rolling(window=self.vol_window).mean()
                ) / spread_features[col].rolling(window=self.vol_window).std()
                
                signals[f'{col}_vol_zscore'] = (
                    signals[f'{col}_vol'] -
                    signals[f'{col}_vol'].rolling(window=self.vol_window).mean()
                ) / signals[f'{col}_vol'].rolling(window=self.vol_window).std()
            
            # 6. Generate sophisticated macro signals
            signals['macro_signal'] = 0
            
            # Yield curve inversion signals (risk-off environment)
            inversion_conditions = (
                (signals['30Y_2Y_spread_zscore'] < -self.macro_impact_threshold) &
                (signals['10Y_2Y_spread_zscore'] < -self.macro_impact_threshold) &
                (signals['5Y_fly_zscore'] > self.macro_impact_threshold) &  # Curve humping
                (signals.get('real_rate_zscore', 0) < -self.macro_impact_threshold)  # Negative real rates
            )
            
            # Yield curve steepening signals (risk-on environment)
            steepening_conditions = (
                (signals['30Y_2Y_spread_zscore'] > self.macro_impact_threshold) &
                (signals['10Y_2Y_spread_zscore'] > self.macro_impact_threshold) &
                (signals['5Y_fly_zscore'] < -self.macro_impact_threshold) &  # Curve flattening in belly
                (signals.get('real_rate_zscore', 0) > self.macro_impact_threshold)  # Positive real rates
            )
            
            # Volatility regime conditions
            high_vol_regime = (
                (signals['30Y_2Y_spread_vol_zscore'] > self.macro_impact_threshold) |
                (signals['10Y_2Y_spread_vol_zscore'] > self.macro_impact_threshold)
            )
            
            low_vol_regime = (
                (signals['30Y_2Y_spread_vol_zscore'] < -self.macro_impact_threshold) &
                (signals['10Y_2Y_spread_vol_zscore'] < -self.macro_impact_threshold)
            )
            
            # Apply signals with regime overlay
            signals.loc[
                inversion_conditions & high_vol_regime,
                'macro_signal'
            ] = -1  # Risk-off signal stronger in high vol regime
            
            signals.loc[
                steepening_conditions & low_vol_regime,
                'macro_signal'
            ] = 1   # Risk-on signal stronger in low vol regime
            
            # Calculate signal conviction
            signals['macro_conviction'] = 0
            
            # Conviction for risk-off signals
            signals.loc[signals['macro_signal'] == -1, 'macro_conviction'] = (
                abs(signals['30Y_2Y_spread_zscore']) +
                abs(signals['10Y_2Y_spread_zscore']) +
                abs(signals['5Y_fly_zscore']) +
                abs(signals.get('real_rate_zscore', 0)) +
                abs(signals['30Y_2Y_spread_vol_zscore'])
            ) / 5
            
            # Conviction for risk-on signals
            signals.loc[signals['macro_signal'] == 1, 'macro_conviction'] = (
                abs(signals['30Y_2Y_spread_zscore']) +
                abs(signals['10Y_2Y_spread_zscore']) +
                abs(signals['5Y_fly_zscore']) +
                abs(signals.get('real_rate_zscore', 0)) +
                abs(signals['30Y_2Y_spread_vol_zscore'])
            ) / 5
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating macro signals: {str(e)}")
            return pd.DataFrame()

    def generate_sentiment_signals(self, symbol: str) -> pd.DataFrame:
        """
        Generate sentiment-driven volatility signals using NLP analysis.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with sentiment signals
        """
        try:
            # Get aligned sentiment and price data
            aligned_data = self.market_data_manager.align_sentiment_with_price(symbol)
            if aligned_data is None:
                return pd.DataFrame()
            
            # Initialize signals DataFrame
            signals = pd.DataFrame(index=aligned_data.index)
            
            # 1. Calculate multi-timeframe sentiment momentum
            for window in [5, 10, 20, 60]:
                signals[f'sent_momentum_{window}'] = (
                    aligned_data['sentiment_score'].diff(window) /
                    window
                )
                
                # Exponentially weighted sentiment
                signals[f'sent_ema_{window}'] = aligned_data['sentiment_score'].ewm(
                    span=window
                ).mean()
            
            # 2. Calculate sentiment volatility
            signals['sent_vol'] = aligned_data['sentiment_score'].rolling(
                window=self.vol_window
            ).std()
            
            # 3. Calculate price-sentiment correlation features
            def rolling_correlation(x, y, window):
                return pd.Series(x).rolling(window).corr(pd.Series(y))
            
            # Multiple timeframe correlations
            for window in [5, 10, 20]:
                signals[f'sent_price_corr_{window}'] = rolling_correlation(
                    aligned_data['sentiment_score'],
                    aligned_data['returns'],
                    window
                )
                
                signals[f'sent_vol_corr_{window}'] = rolling_correlation(
                    aligned_data['sentiment_score'],
                    aligned_data['realized_volatility'],
                    window
                )
            
            # 4. Calculate sentiment regime features
            signals['sentiment_ma'] = aligned_data['sentiment_score'].rolling(
                window=self.vol_window
            ).mean()
            
            # Regime detection with multiple confirmations
            signals['sentiment_regime'] = 0
            
            # Extreme bearish sentiment regime
            bearish_conditions = (
                (signals['sent_momentum_5'] < -self.sentiment_impact_threshold) &
                (signals['sent_momentum_20'] < -self.sentiment_impact_threshold) &
                (signals['sent_price_corr_5'] < -0.5) &
                (signals['sent_vol'] > signals['sent_vol'].rolling(window=self.vol_window).mean())
            )
            
            # Extreme bullish sentiment regime
            bullish_conditions = (
                (signals['sent_momentum_5'] > self.sentiment_impact_threshold) &
                (signals['sent_momentum_20'] > self.sentiment_impact_threshold) &
                (signals['sent_price_corr_5'] < -0.5) &
                (signals['sent_vol'] > signals['sent_vol'].rolling(window=self.vol_window).mean())
            )
            
            signals.loc[bearish_conditions, 'sentiment_regime'] = -1
            signals.loc[bullish_conditions, 'sentiment_regime'] = 1
            
            # 5. Calculate sentiment impact scores
            # Short-term impact
            signals['short_term_impact'] = (
                abs(signals['sent_momentum_5']) *
                abs(signals['sent_price_corr_5']) *
                signals['sent_vol']
            )
            
            # Medium-term impact
            signals['medium_term_impact'] = (
                abs(signals['sent_momentum_20']) *
                abs(signals['sent_price_corr_20']) *
                signals['sent_vol']
            )
            
            # Calculate z-scores for impact metrics
            for col in ['short_term_impact', 'medium_term_impact']:
                signals[f'{col}_zscore'] = (
                    signals[col] -
                    signals[col].rolling(window=self.vol_window).mean()
                ) / signals[col].rolling(window=self.vol_window).std()
            
            # 6. Generate sophisticated sentiment signals
            signals['sentiment_signal'] = 0
            
            # Mean reversion signals for extreme sentiment
            extreme_bearish_conditions = (
                (signals['sentiment_regime'] == -1) &
                (signals['short_term_impact_zscore'] > self.sentiment_impact_threshold) &
                (signals['sent_vol_corr_5'] > 0.5) &  # High correlation with volatility
                (signals['sent_momentum_5'].rolling(window=5).mean() < 
                 signals['sent_momentum_5'].rolling(window=20).mean())  # Momentum divergence
            )
            
            extreme_bullish_conditions = (
                (signals['sentiment_regime'] == 1) &
                (signals['short_term_impact_zscore'] > self.sentiment_impact_threshold) &
                (signals['sent_vol_corr_5'] > 0.5) &  # High correlation with volatility
                (signals['sent_momentum_5'].rolling(window=5).mean() > 
                 signals['sent_momentum_5'].rolling(window=20).mean())  # Momentum divergence
            )
            
            # Apply signals with conviction scores
            signals.loc[extreme_bearish_conditions, 'sentiment_signal'] = 1  # Expect positive reversion
            signals.loc[extreme_bullish_conditions, 'sentiment_signal'] = -1  # Expect negative reversion
            
            # Calculate signal conviction
            signals['sentiment_conviction'] = 0
            
            # Conviction for mean reversion signals
            signals.loc[signals['sentiment_signal'] != 0, 'sentiment_conviction'] = (
                abs(signals['short_term_impact_zscore']) +
                abs(signals['medium_term_impact_zscore']) +
                abs(signals['sent_vol_corr_5']) +
                abs(signals['sent_momentum_5']) +
                signals['sent_vol']
            ) / 5
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating sentiment signals for {symbol}: {str(e)}")
            return pd.DataFrame()

    def combine_signals(
        self,
        symbol: str,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Combine all signal components with optional weighting.
        
        Args:
            symbol: Stock symbol
            weights: Optional dictionary of signal weights
            
        Returns:
            DataFrame with combined signals
        """
        if weights is None:
            weights = {
                'volatility': 1.5,
                'liquidity': 1.2,
                'options': 1.0,
                'macro': 0.8,
                'sentiment': 0.5
            }
        
        try:
            # Generate all signals
            vol_signals = self.generate_volatility_signals(symbol)
            liq_signals = self.generate_liquidity_signals(symbol)
            opt_signals = self.generate_options_signals(symbol)
            macro_signals = self.generate_macro_signals()
            sent_signals = self.generate_sentiment_signals(symbol)
            
            # Combine signals on a common index
            signals = pd.DataFrame()
            
            if not vol_signals.empty:
                signals['vol_signal'] = vol_signals['vol_signal']
            if not liq_signals.empty:
                signals['liquidity_signal'] = liq_signals['liquidity_signal']
            if not opt_signals.empty:
                signals['options_signal'] = opt_signals['options_signal']
            if not macro_signals.empty:
                signals['macro_signal'] = macro_signals['macro_signal']
            if not sent_signals.empty:
                signals['sentiment_signal'] = sent_signals['sentiment_signal']
            
            # Calculate weighted combined signal
            signals['combined_signal'] = (
                weights['volatility'] * signals.get('vol_signal', 0) +
                weights['liquidity'] * signals.get('liquidity_signal', 0) +
                weights['options'] * signals.get('options_signal', 0) +
                weights['macro'] * signals.get('macro_signal', 0) +
                weights['sentiment'] * signals.get('sentiment_signal', 0)
            )
            
            # Add signal strength and conviction metrics
            signals['signal_strength'] = abs(signals['combined_signal'])
            signals['signal_conviction'] = (
                (signals['vol_signal'] != 0).astype(int) +
                (signals['liquidity_signal'] != 0).astype(int) +
                (signals['options_signal'] != 0).astype(int) +
                (signals['macro_signal'] != 0).astype(int) +
                (signals['sentiment_signal'] != 0).astype(int)
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"Error combining signals for {symbol}: {str(e)}")
            return pd.DataFrame()

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate enhanced trading signals based on all calculated features.
        
        Returns:
            DataFrame with trading signals for each symbol
        """
        signals = pd.DataFrame()
        
        for symbol in self.symbols:
            # Calculate all features
            vol_features = self.calculate_intraday_volatility_features(symbol)
            liq_features = self.calculate_liquidity_features(symbol)
            opt_features = self.calculate_options_features(symbol)
            sent_features = self.calculate_sentiment_features(symbol)
            macro_features = self.calculate_macro_features()
            
            # Combine features
            symbol_features = pd.concat([
                vol_features,
                liq_features,
                opt_features,
                sent_features,
                macro_features
            ], axis=1)
            
            # Generate signals based on feature combinations
            symbol_signals = pd.DataFrame(index=symbol_features.index)
            
            # 1. Volatility arbitrage signals
            symbol_signals['vol_arb_signal'] = (
                (symbol_features['rv_5min_regime'] == 1) &  # High short-term volatility
                (symbol_features['vrp_zscore'] < -self.vol_signal_threshold) &  # Overpriced options
                (abs(symbol_features['sentiment_impact_score']) < self.sentiment_impact_threshold)  # No significant sentiment impact
            ).astype(int)
            
            # 2. Liquidity arbitrage signals
            symbol_signals['liq_arb_signal'] = (
                (symbol_features['liquidity_regime'] == -1) &  # Illiquid regime
                (symbol_features['vpin'] > self.liquidity_signal_threshold) &  # High informed trading
                (abs(symbol_features['macro_impact_zscore']) < self.macro_impact_threshold)  # No significant macro impact
            ).astype(int)
            
            # 3. Options mispricing signals
            symbol_signals['opt_arb_signal'] = (
                (symbol_features['forward_skew_zscore'] > self.iv_rank_threshold) &  # High skew
                (symbol_features['vol_risk_premium'] < -0.05) &  # Overpriced volatility
                (symbol_features['yield_regime'] == 0)  # Stable yield environment
            ).astype(int)
            
            # 4. Macro volatility signals
            symbol_signals['macro_vol_signal'] = (
                (symbol_features['macro_impact_zscore'] > self.macro_impact_threshold) &
                (symbol_features['yield_vol'] > symbol_features['yield_vol'].rolling(
                    window=self.vol_window
                ).mean() + self.macro_impact_threshold * symbol_features['yield_vol'].rolling(
                    window=self.vol_window
                ).std())
            ).astype(int)
            
            # 5. Sentiment-driven volatility signals
            symbol_signals['sent_vol_signal'] = (
                (abs(symbol_features['sentiment_momentum']) > self.sentiment_impact_threshold) &
                (symbol_features['sent_price_corr'] < -0.5) &  # Negative correlation with price
                (symbol_features['rv_30min_regime'] == 0)  # Normal volatility regime
            ).astype(int)
            
            # Combine signals with weights
            symbol_signals['combined_signal'] = (
                1.5 * symbol_signals['vol_arb_signal'] +
                1.2 * symbol_signals['liq_arb_signal'] +
                1.0 * symbol_signals['opt_arb_signal'] +
                0.8 * symbol_signals['macro_vol_signal'] +
                0.5 * symbol_signals['sent_vol_signal']
            )
            
            # Add signal strength
            symbol_signals['signal_strength'] = abs(symbol_signals['combined_signal'])
            
            # Add symbol column
            symbol_signals['symbol'] = symbol
            signals = pd.concat([signals, symbol_signals])
            
        return signals.sort_values('signal_strength', ascending=False) 