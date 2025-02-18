"""
Advanced volatility and liquidity metrics for the MVLA strategy.

This module provides sophisticated implementations of:
1. Multi-timeframe realized volatility
2. Amihud illiquidity measure
3. Advanced options analytics
4. Yield curve analysis
5. Enhanced sentiment metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from arch import arch_model

def calculate_multi_timeframe_rv(
    data: pd.DataFrame,
    windows: List[int] = [5, 30, 60],
    annualization: float = 252 * 390  # 252 trading days * 390 minutes per day
) -> pd.DataFrame:
    """
    Calculate realized volatility over multiple timeframes.
    
    Args:
        data: DataFrame with 'close' prices
        windows: List of minute windows for RV calculation
        annualization: Annualization factor
        
    Returns:
        DataFrame with RV for each timeframe
    """
    rv_data = pd.DataFrame(index=data.index)
    returns = np.log(data['close'] / data['close'].shift(1))
    
    for window in windows:
        # Standard RV
        rv = returns.rolling(window=window).std() * np.sqrt(annualization)
        rv_data[f'rv_{window}min'] = rv
        
        # Parkinson High-Low Range
        high_low_rv = np.sqrt(
            (1.0 / (4.0 * np.log(2.0))) * 
            np.log(data['high'] / data['low'])**2
        ).rolling(window=window).mean() * np.sqrt(annualization)
        rv_data[f'hl_rv_{window}min'] = high_low_rv
        
        # Garman-Klass Volatility
        log_hl = np.log(data['high'] / data['low'])**2
        log_co = np.log(data['close'] / data['open'])**2
        gk_rv = np.sqrt(
            0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        ).rolling(window=window).mean() * np.sqrt(annualization)
        rv_data[f'gk_rv_{window}min'] = gk_rv
    
    return rv_data

def calculate_amihud_illiquidity(
    data: pd.DataFrame,
    window: int = 30
) -> pd.Series:
    """
    Calculate Amihud's illiquidity ratio.
    
    Args:
        data: DataFrame with price and volume data
        window: Rolling window size
        
    Returns:
        Series with illiquidity ratios
    """
    # Calculate absolute returns
    returns = np.log(data['close'] / data['close'].shift(1)).abs()
    
    # Calculate dollar volume
    dollar_volume = data['close'] * data['volume']
    
    # Amihud ratio: |return| / dollar_volume
    illiquidity = returns / dollar_volume
    
    # Rolling average of illiquidity
    return illiquidity.rolling(window=window).mean()

def calculate_order_flow_toxicity(
    data: pd.DataFrame,
    window: int = 30
) -> pd.DataFrame:
    """
    Calculate order flow toxicity metrics.
    
    Args:
        data: DataFrame with OHLCV data
        window: Rolling window size
        
    Returns:
        DataFrame with toxicity metrics
    """
    flow_metrics = pd.DataFrame(index=data.index)
    
    # Volume-synchronized probability of informed trading (VPIN)
    volume_buckets = pd.qcut(data['volume'], q=50, labels=False)
    buy_volume = data['volume'] * (data['close'] > data['open']).astype(int)
    sell_volume = data['volume'] * (data['close'] <= data['open']).astype(int)
    
    flow_metrics['vpin'] = abs(
        buy_volume - sell_volume
    ).rolling(window=window).sum() / data['volume'].rolling(window=window).sum()
    
    # Order flow imbalance
    flow_metrics['flow_imbalance'] = (
        (data['close'] - data['open']) / 
        (data['high'] - data['low']).replace(0, np.nan)
    ).rolling(window=window).mean()
    
    # Bid-ask spread estimation
    flow_metrics['effective_spread'] = 2 * np.sqrt(abs(
        np.log(data['high'] / data['low'])
    )).rolling(window=window).mean()
    
    return flow_metrics

def calculate_options_skew_features(
    options_data: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    Calculate advanced options skew metrics.
    
    Args:
        options_data: DataFrame with options data
        window: Rolling window size
        
    Returns:
        DataFrame with skew metrics
    """
    skew_features = pd.DataFrame()
    
    # Group by date and option type
    daily_options = options_data.groupby(['date', 'type'])
    
    # Calculate forward skew (25-delta call vs put)
    def calculate_forward_skew(group):
        calls = group[group['type'] == 'call']
        puts = group[group['type'] == 'put']
        
        call_25d = calls[calls['delta'].between(0.20, 0.30)]['implied_volatility'].mean()
        put_25d = puts[puts['delta'].between(-0.30, -0.20)]['implied_volatility'].mean()
        
        return put_25d - call_25d
    
    skew_features['forward_skew'] = daily_options.apply(calculate_forward_skew)
    
    # Calculate term structure
    def calculate_term_structure(group):
        near_term = group[group['days_to_expiry'] <= 30]['implied_volatility'].mean()
        far_term = group[group['days_to_expiry'] > 30]['implied_volatility'].mean()
        return far_term - near_term
    
    skew_features['term_structure'] = daily_options.apply(calculate_term_structure)
    
    # Calculate volatility risk premium
    skew_features['vol_risk_premium'] = (
        options_data.groupby('date')['implied_volatility'].mean() -
        options_data.groupby('date')['realized_vol'].first()
    )
    
    # Rolling metrics
    for col in skew_features.columns:
        skew_features[f'{col}_zscore'] = (
            skew_features[col] - 
            skew_features[col].rolling(window=window).mean()
        ) / skew_features[col].rolling(window=window).std()
    
    return skew_features

def calculate_yield_curve_features(
    yields_data: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    Calculate yield curve based features.
    
    Args:
        yields_data: DataFrame with yield data
        window: Rolling window size
        
    Returns:
        DataFrame with yield curve features
    """
    curve_features = pd.DataFrame(index=yields_data.index)
    
    # Calculate yield changes
    curve_features['yield_change'] = yields_data['value'].diff()
    
    # Yield volatility
    curve_features['yield_vol'] = yields_data['value'].rolling(window=window).std()
    
    # Yield momentum
    curve_features['yield_momentum'] = yields_data['value'].diff(window)
    
    # Yield curve regime
    curve_features['yield_regime'] = np.where(
        curve_features['yield_change'].rolling(window=5).sum() > 0,
        1,  # Rising yield regime
        np.where(
            curve_features['yield_change'].rolling(window=5).sum() < 0,
            -1,  # Falling yield regime
            0  # Stable regime
        )
    )
    
    # Yield volatility regime
    curve_features['vol_regime'] = np.where(
        curve_features['yield_vol'] > curve_features['yield_vol'].rolling(window=window).mean() +
        curve_features['yield_vol'].rolling(window=window).std(),
        1,  # High volatility regime
        np.where(
            curve_features['yield_vol'] < curve_features['yield_vol'].rolling(window=window).mean() -
            curve_features['yield_vol'].rolling(window=window).std(),
            -1,  # Low volatility regime
            0  # Normal regime
        )
    )
    
    return curve_features

def calculate_sentiment_impact_features(
    sentiment_data: pd.DataFrame,
    price_data: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    Calculate advanced sentiment impact features.
    
    Args:
        sentiment_data: DataFrame with sentiment scores
        price_data: DataFrame with price data
        window: Rolling window size
        
    Returns:
        DataFrame with sentiment impact features
    """
    impact_features = pd.DataFrame(index=sentiment_data.index)
    
    # Calculate sentiment volatility
    impact_features['sentiment_vol'] = sentiment_data['sentiment_score'].rolling(
        window=window
    ).std()
    
    # Sentiment momentum
    impact_features['sentiment_momentum'] = sentiment_data['sentiment_score'].diff(window)
    
    # Sentiment-price correlation
    def rolling_correlation(x, y, window):
        return pd.Series(x).rolling(window).corr(pd.Series(y))
    
    impact_features['sent_price_corr'] = rolling_correlation(
        sentiment_data['sentiment_score'],
        price_data['returns'],
        window
    )
    
    # Sentiment regime
    impact_features['sentiment_regime'] = np.where(
        impact_features['sentiment_momentum'] > impact_features['sentiment_momentum'].rolling(
            window=window
        ).mean() + impact_features['sentiment_momentum'].rolling(window=window).std(),
        1,  # Bullish sentiment regime
        np.where(
            impact_features['sentiment_momentum'] < impact_features['sentiment_momentum'].rolling(
                window=window
            ).mean() - impact_features['sentiment_momentum'].rolling(window=window).std(),
            -1,  # Bearish sentiment regime
            0  # Neutral regime
        )
    )
    
    # Sentiment impact score
    impact_features['impact_score'] = (
        impact_features['sentiment_vol'] * 
        abs(impact_features['sent_price_corr']) * 
        abs(impact_features['sentiment_momentum'])
    )
    
    return impact_features 