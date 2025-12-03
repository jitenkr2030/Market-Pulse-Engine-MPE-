"""
Market Regime Forecaster Module

This module analyzes market conditions to identify current market regimes
and forecast future regime transitions. Market regimes represent distinct
market states characterized by specific patterns in volatility, returns,
correlations, and trading behavior.

Author: MiniMax Agent
Date: December 2025
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeIndicators:
    """Calculate various regime indicators from market data."""
    
    @staticmethod
    def volatility_regime(returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate volatility regime indicator."""
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        vol_percentile = rolling_vol.rolling(window=window).rank(pct=True)
        
        # Define volatility regimes
        regimes = pd.Series(index=returns.index, dtype='object')
        regimes[vol_percentile <= 0.25] = 'Low Vol'
        regimes[(vol_percentile > 0.25) & (vol_percentile <= 0.75)] = 'Medium Vol'
        regimes[vol_percentile > 0.75] = 'High Vol'
        
        return regimes.ffill()
    
    @staticmethod
    def trend_regime(returns: pd.Series, window: int = 50) -> pd.Series:
        """Calculate trend regime indicator."""
        # Calculate rolling returns and trend strength
        rolling_returns = returns.rolling(window=window).sum()
        trend_strength = rolling_returns.rolling(window=window).std()
        
        # Define trend regimes based on return patterns
        regimes = pd.Series(index=returns.index, dtype='object')
        high_threshold = rolling_returns.quantile(0.75)
        low_threshold = rolling_returns.quantile(0.25)
        
        regimes[rolling_returns >= high_threshold] = 'Strong Bull'
        regimes[(rolling_returns < high_threshold) & (rolling_returns > 0)] = 'Bull'
        regimes[(rolling_returns <= 0) & (rolling_returns > low_threshold)] = 'Bear'
        regimes[rolling_returns <= low_threshold] = 'Strong Bear'
        
        return regimes.ffill()
    
    @staticmethod
    def correlation_regime(returns: pd.DataFrame, window: int = 60) -> pd.Series:
        """Calculate correlation regime indicator."""
        # Handle single asset case
        if len(returns.columns) < 2:
            # Return default correlation regime for single asset
            return pd.Series(index=returns.index, data='Medium Correlation', dtype='object')
        
        rolling_corr = returns.rolling(window=window).corr()
        
        # Calculate average correlation across all pairs
        n_assets = len(returns.columns)
        avg_corr = pd.Series(index=returns.index, dtype='float64')
        
        for i in range(len(returns)):
            if i >= window - 1 and i < len(avg_corr):
                corr_matrix = rolling_corr.iloc[i]
                # Extract upper triangle correlations
                correlations = []
                for j in range(n_assets):
                    for k in range(j + 1, n_assets):
                        if j in corr_matrix and k in corr_matrix[j]:
                            correlations.append(corr_matrix[j][k])
                
                if correlations:
                    avg_corr.at[returns.index[i]] = np.mean(correlations)
        
        avg_corr = avg_corr.ffill()
        
        # Define correlation regimes
        regimes = pd.Series(index=returns.index, dtype='object')
        high_threshold = avg_corr.quantile(0.75)
        low_threshold = avg_corr.quantile(0.25)
        
        regimes[avg_corr >= high_threshold] = 'High Correlation'
        regimes[(avg_corr < high_threshold) & (avg_corr > low_threshold)] = 'Medium Correlation'
        regimes[avg_corr <= low_threshold] = 'Low Correlation'
        
        return regimes.ffill()
    
    @staticmethod
    def liquidity_regime(returns: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """Calculate liquidity regime indicator."""
        try:
            # Ensure volume is numeric and clean
            volume = pd.to_numeric(volume, errors='coerce').fillna(1)
            volume = volume.replace(0, 1)  # Avoid division by zero
            
            # Calculate volume-adjusted returns and spread proxy
            volume_mean = volume.rolling(window=window).mean()
            volume_mean = volume_mean.replace(0, 1)  # Avoid division by zero
            volume_normalized = volume / volume_mean
            price_impact = np.abs(returns) / volume_normalized
            
            # Calculate rolling percentiles
            impact_percentile = price_impact.rolling(window=window).rank(pct=True)
            
            # Define liquidity regimes
            regimes = pd.Series(index=returns.index, dtype='object')
            impact_percentile_clean = impact_percentile.dropna()
            
            if len(impact_percentile_clean) > 0:
                regimes[impact_percentile <= 0.25] = 'High Liquidity'
                regimes[(impact_percentile > 0.25) & (impact_percentile <= 0.75)] = 'Medium Liquidity'
                regimes[impact_percentile > 0.75] = 'Low Liquidity'
            
            return regimes.fillna('Medium Liquidity')
            
        except Exception as e:
            logger.warning(f"Error in liquidity_regime: {str(e)}")
            # Return a default regime
            return pd.Series('Medium Liquidity', index=returns.index)
    
    @staticmethod
    def market_microstructure_regime(returns: pd.Series, volume: pd.Series, 
                                   window: int = 20) -> pd.Series:
        """Calculate market microstructure regime indicator."""
        try:
            # Ensure volume is numeric and clean
            volume = pd.to_numeric(volume, errors='coerce').fillna(1)
            volume = volume.replace(0, 1)  # Avoid division by zero
            
            # Calculate various microstructure indicators
            volume_mean = volume.rolling(window=window).mean().fillna(1)
            turnover = volume / volume_mean.replace(0, 1)
            volatility = returns.rolling(window=window).std()
            
            # Calculate Amihud illiquidity ratio with safe division
            time_diff = returns.index.to_series().diff().dt.days.fillna(1)
            # Ensure proper alignment and type conversion
            time_diff_aligned = time_diff.reindex(volume.index).fillna(1)
            
            # DEBUG: Log data types and shapes before division
            logger.debug(f"DEBUG market_microstructure_regime:")
            logger.debug(f"  returns type: {type(returns)}, dtype: {returns.dtype}, shape: {returns.shape}")
            logger.debug(f"  volume type: {type(volume)}, dtype: {volume.dtype}, shape: {volume.shape}")
            logger.debug(f"  time_diff_aligned type: {type(time_diff_aligned)}, dtype: {time_diff_aligned.dtype}, shape: {time_diff_aligned.shape}")
            logger.debug(f"  np.abs(returns) type: {type(np.abs(returns))}, dtype: {np.abs(returns).dtype}")
            
            try:
                illiquidity = np.abs(returns) / (volume * time_diff_aligned + 1)
            except Exception as div_error:
                logger.error(f"Division error details: {div_error}")
                logger.error(f"  volume * time_diff_aligned type: {type(volume * time_diff_aligned)}, dtype: {(volume * time_diff_aligned).dtype}")
                logger.error(f"  (volume * time_diff_aligned + 1) type: {type(volume * time_diff_aligned + 1)}, dtype: {(volume * time_diff_aligned + 1).dtype}")
                raise div_error
            illiquidity = illiquidity.rolling(window=window).mean()
            
            # Handle division by zero in rolling statistics
            vol_std = volatility.rolling(window=window).std()
            turnover_std = turnover.rolling(window=window).std()
            illiquidity_std = illiquidity.rolling(window=window).std()
            
            # Replace zeros and NaNs with 1 to avoid division issues
            vol_std = vol_std.replace(0, 1).fillna(1)
            turnover_std = turnover_std.replace(0, 1).fillna(1)
            illiquidity_std = illiquidity_std.replace(0, 1).fillna(1)
            
            # Combine indicators into composite score
            microstructure_score = (
                (volatility / vol_std) * 0.3 +
                (turnover / turnover_std) * 0.3 +
                (illiquidity / illiquidity_std) * 0.4
            )
            
            # Define microstructure regimes
            regimes = pd.Series(index=returns.index, dtype='object')
            valid_scores = microstructure_score.dropna()
            
            if len(valid_scores) > 0:
                high_threshold = valid_scores.quantile(0.75)
                low_threshold = valid_scores.quantile(0.25)
                
                regimes[microstructure_score >= high_threshold] = 'Stressed'
                regimes[(microstructure_score < high_threshold) & (microstructure_score > low_threshold)] = 'Normal'
                regimes[microstructure_score <= low_threshold] = 'Calm'
            
            return regimes.fillna('Normal')
            
        except Exception as e:
            logger.warning(f"Error in market_microstructure_regime: {str(e)}")
            # Return a default regime
            return pd.Series('Normal', index=returns.index)


class RegimeTransitionDetector:
    """Detect regime transitions using various statistical methods."""
    
    @staticmethod
    def change_point_detection(series: pd.Series, method: str = 'cumulative_sum') -> List[datetime]:
        """Detect change points in time series."""
        if method == 'cumulative_sum':
            return RegimeTransitionDetector._cusum_detection(series)
        elif method == 'variance':
            return RegimeTransitionDetector._variance_detection(series)
        elif method == 'bayesian':
            return RegimeTransitionDetector._bayesian_detection(series)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def _cusum_detection(series: pd.Series) -> List[datetime]:
        """CUSUM-based change point detection."""
        try:
            # Convert string regime values to numeric for CUSUM analysis
            if series.dtype == 'object' or series.dtype.name.startswith('string'):
                # Map regime strings to numeric values
                regime_mapping = {
                    'Calm': 0, 'Normal': 1, 'Stressed': 2, 'Volatile': 3,
                    'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3,
                    'Bull': 1, 'Bear': -1, 'Neutral': 0,
                    'Stable': 0, 'Unstable': 1
                }
                
                # Convert to numeric, defaulting to 1 for unknown regimes
                numeric_series = series.map(lambda x: regime_mapping.get(x, 1))
                values = numeric_series.dropna().values
            else:
                values = series.dropna().values
            
            change_points = []
            
            # Calculate cumulative sum
            mean_val = np.mean(values)
            cusum = np.cumsum(values - mean_val)
            
            # Detect significant deviations
            threshold = 2 * np.std(values)
            for i in range(1, len(cusum)):
                if abs(cusum[i] - cusum[i-1]) > threshold:
                    change_points.append(series.index[min(i, len(series)-1)])
            
            return change_points
            
        except Exception as e:
            logger.warning(f"CUSUM detection failed, returning empty list: {str(e)}")
            return []
    
    @staticmethod
    def _variance_detection(series: pd.Series, window: int = 30) -> List[datetime]:
        """Variance-based change point detection."""
        try:
            # Convert string regime values to numeric for variance analysis
            if series.dtype == 'object' or series.dtype.name.startswith('string'):
                # Map regime strings to numeric values
                regime_mapping = {
                    'Calm': 0, 'Normal': 1, 'Stressed': 2, 'Volatile': 3,
                    'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3,
                    'Bull': 1, 'Bear': -1, 'Neutral': 0,
                    'Stable': 0, 'Unstable': 1
                }
                
                # Convert to numeric, defaulting to 1 for unknown regimes
                numeric_series = series.map(lambda x: regime_mapping.get(x, 1))
            else:
                numeric_series = series
            
            change_points = []
            
            for i in range(window, len(numeric_series) - window):
                before_var = numeric_series.iloc[i-window:i].var()
                after_var = numeric_series.iloc[i:i+window].var()
                
                # Test for significant variance change
                f_stat = max(before_var, after_var) / min(before_var, after_var)
                if f_stat > 3.0:  # F-test threshold
                    change_points.append(series.index[i])
            
            return change_points
            
        except Exception as e:
            logger.warning(f"Variance detection failed, returning empty list: {str(e)}")
            return []
    
    @staticmethod
    def _bayesian_detection(series: pd.Series) -> List[datetime]:
        """Bayesian change point detection."""
        try:
            # Convert string regime values to numeric for Bayesian analysis
            if series.dtype == 'object' or series.dtype.name.startswith('string'):
                # Map regime strings to numeric values
                regime_mapping = {
                    'Calm': 0, 'Normal': 1, 'Stressed': 2, 'Volatile': 3,
                    'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3,
                    'Bull': 1, 'Bear': -1, 'Neutral': 0,
                    'Stable': 0, 'Unstable': 1
                }
                
                # Convert to numeric, defaulting to 1 for unknown regimes
                numeric_series = series.map(lambda x: regime_mapping.get(x, 1))
                values = numeric_series.dropna().astype(float).values
            else:
                values = series.dropna().astype(float).values
            
            if len(values) < 20:
                return []
            
            change_points = []
            
            # Calculate posterior probabilities for change points
            n = len(values)
            for i in range(10, n-10):
                before = values[:i]
                after = values[i:]
                
                # Ensure we have valid data
                if len(before) == 0 or len(after) == 0:
                    continue
                    
                try:
                    # Calculate Bayes factor with proper type conversion
                    before_mean = float(np.mean(before))
                    after_mean = float(np.mean(after))
                    
                    # Ensure positive variance
                    all_values = np.concatenate([before, after])
                    pooled_var = max(np.var(all_values), 1e-8)  # Minimum variance to avoid issues
                    
                    # Calculate likelihoods with proper type handling
                    before_std = np.sqrt(pooled_var)
                    after_std = np.sqrt(pooled_var)
                    combined_std = np.sqrt(pooled_var)
                    
                    # Ensure valid probability density inputs
                    if before_std <= 0 or after_std <= 0 or combined_std <= 0:
                        continue
                    
                    likelihood_before = np.prod(stats.norm.pdf(before, before_mean, before_std))
                    likelihood_after = np.prod(stats.norm.pdf(after, after_mean, after_std))
                    likelihood_combined = np.prod(stats.norm.pdf(values, np.mean(values), combined_std))
                    
                    # Avoid division by zero
                    if likelihood_combined <= 0:
                        continue
                    
                    bayes_factor = (likelihood_before * likelihood_after) / likelihood_combined
                    
                    if bayes_factor > 10:  # Strong evidence threshold
                        change_points.append(series.index[i])
                        
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    # Skip problematic calculations
                    continue
            
            return change_points
            
        except Exception as e:
            logger.warning(f"Error in Bayesian change point detection: {str(e)}")
            return []
    
    @staticmethod
    def regime_stability_score(regime_series: pd.Series, window: int = 30) -> pd.Series:
        """Calculate regime stability score."""
        stability_scores = pd.Series(index=regime_series.index, dtype='float64')
        
        for i in range(len(regime_series)):
            if i >= window - 1 and i < len(stability_scores):
                window_data = regime_series.iloc[i-window+1:i+1]
                # Calculate mode frequency in window
                mode_count = window_data.value_counts().max()
                stability_scores.at[regime_series.index[i]] = mode_count / window
        
        return stability_scores.fillna(1.0)


class RegimeForecaster:
    """Machine learning-based regime forecasting."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
    
    def prepare_features(self, data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Prepare features for regime prediction."""
        features = pd.DataFrame()
        
        # Price-based features
        for symbol, returns in data.items():
            features[f'{symbol}_return'] = returns
            features[f'{symbol}_volatility'] = returns.rolling(20).std()
            features[f'{symbol}_skewness'] = returns.rolling(20).skew()
            features[f'{symbol}_kurtosis'] = returns.rolling(20).kurt()
        
        # Cross-sectional features
        if len(data) > 1:
            returns_df = pd.DataFrame(data)
            features['cross_sectional_mean'] = returns_df.mean(axis=1)
            features['cross_sectional_std'] = returns_df.std(axis=1)
            features['cross_sectional_skew'] = returns_df.skew(axis=1)
            features['cross_sectional_kurt'] = returns_df.kurtosis(axis=1)
            features['correlation'] = returns_df.rolling(60).corr().mean(axis=1)
        
        # Momentum features
        for symbol, returns in data.items():
            features[f'{symbol}_momentum_1m'] = returns.rolling(20).sum()
            features[f'{symbol}_momentum_3m'] = returns.rolling(60).sum()
            features[f'{symbol}_reversal_1w'] = returns.rolling(5).sum()
        
        return features.dropna()
    
    def train_regime_classifier(self, features: pd.DataFrame, regimes: pd.Series) -> Dict[str, Any]:
        """Train regime classification model."""
        # Align features and targets
        common_index = features.index.intersection(regimes.index)
        X = features.loc[common_index]
        y = regimes.loc[common_index]
        
        if len(X) < 100:
            return {'error': 'Insufficient data for training'}
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_scaled, y)
        
        # Calculate feature importance
        feature_importance = pd.Series(
            rf_model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        return {
            'model': rf_model,
            'scaler': scaler,
            'feature_importance': feature_importance,
            'training_accuracy': rf_model.score(X_scaled, y)
        }
    
    def predict_regime_probability(self, features: pd.DataFrame, model_info: Dict[str, Any]) -> pd.DataFrame:
        """Predict regime probabilities."""
        if 'error' in model_info:
            return pd.DataFrame()
        
        # Scale features
        X_scaled = model_info['scaler'].transform(features)
        
        # Predict probabilities
        probabilities = model_info['model'].predict_proba(X_scaled)
        classes = model_info['model'].classes_
        
        prob_df = pd.DataFrame(probabilities, columns=classes, index=features.index)
        return prob_df


class MarketRegimeForecaster:
    """
    Market Regime Forecaster - Analyzes market conditions to identify 
    current regimes and forecast regime transitions.
    """
    
    def __init__(self):
        self.indicators = RegimeIndicators()
        self.transition_detector = RegimeTransitionDetector()
        self.forecaster = RegimeForecaster()
        self.logger = logging.getLogger(__name__)
    
    async def analyze(self, symbols: tuple, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Analyze market regimes for given symbols.
        
        Args:
            symbols: Tuple of stock symbols
            start_date: Start date for analysis
            end_date: End date for analysis
        
        Returns:
            Dictionary containing regime analysis results
        """
        try:
            # Download data
            data = await self._download_data(symbols, start_date, end_date)
            if not data:
                return {'error': 'Failed to download data'}
            
            # Calculate returns
            returns_data = self._calculate_returns(data)
            
            # Analyze different regime dimensions
            regime_analysis = await self._analyze_regime_dimensions(returns_data)
            
            # Detect regime transitions
            transition_analysis = await self._detect_regime_transitions(returns_data)
            
            # Forecast future regimes
            forecast_analysis = await self._forecast_regimes(returns_data)
            
            # Generate signals
            signals = await self._generate_regime_signals(regime_analysis, forecast_analysis)
            
            return {
                'symbols': symbols,
                'analysis_period': {'start': start_date, 'end': end_date},
                'regime_dimensions': regime_analysis,
                'transition_analysis': transition_analysis,
                'forecast': forecast_analysis,
                'signals': signals,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in market regime analysis: {str(e)}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    async def _download_data(self, symbols: tuple, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download market data for symbols."""
        data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty and len(hist) > 50:
                    data[symbol] = hist
                else:
                    self.logger.warning(f"Insufficient data for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Failed to download data for {symbol}: {str(e)}")
                continue
        
        return data
    
    def _calculate_returns(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Calculate returns for all symbols."""
        returns_data = {}
        
        for symbol, df in data.items():
            try:
                # Handle multi-level columns (when downloading multiple symbols)
                if isinstance(df.columns, pd.MultiIndex):
                    # Find the Close price column for this symbol
                    close_col = None
                    for col in df.columns:
                        if col[0] == 'Close' and col[1] == symbol:
                            close_col = col
                            break
                    
                    if close_col is not None:
                        prices = df[close_col]
                    else:
                        # Fallback: use the first Close column
                        close_cols = [col for col in df.columns if col[0] == 'Close']
                        if close_cols:
                            prices = df[close_cols[0]]
                        else:
                            self.logger.warning(f"No Close price found for {symbol}")
                            continue
                else:
                    # Single-level columns
                    if 'Close' in df.columns:
                        prices = df['Close']
                    elif 'Adj Close' in df.columns:
                        prices = df['Adj Close']
                    else:
                        self.logger.warning(f"No price data found for {symbol}")
                        continue
                
                returns = prices.pct_change().dropna()
                if len(returns) > 0:
                    returns_data[symbol] = returns
                else:
                    self.logger.warning(f"No valid returns calculated for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error calculating returns for {symbol}: {str(e)}")
                continue
        
        return returns_data
    
    async def _analyze_regime_dimensions(self, returns_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze different regime dimensions."""
        analysis = {}
        
        # Combine returns for cross-sectional analysis
        if len(returns_data) > 1:
            returns_df = pd.DataFrame(returns_data)
        else:
            returns_df = pd.DataFrame(list(returns_data.values()))
            returns_df.columns = list(returns_data.keys())
        
        # Volatility regimes
        vol_regimes = {}
        for symbol, returns in returns_data.items():
            vol_regimes[symbol] = self.indicators.volatility_regime(returns).to_dict()
        analysis['volatility_regimes'] = vol_regimes
        
        # Trend regimes
        trend_regimes = {}
        for symbol, returns in returns_data.items():
            trend_regimes[symbol] = self.indicators.trend_regime(returns).to_dict()
        analysis['trend_regimes'] = trend_regimes
        
        # Correlation regimes
        if len(returns_data) > 1:
            correlation_regime = self.indicators.correlation_regime(returns_df)
            analysis['correlation_regime'] = correlation_regime.to_dict()
        else:
            analysis['correlation_regime'] = {}
        
        # Liquidity regimes
        liquidity_regimes = {}
        for symbol in returns_data.keys():
            try:
                # Get volume data (simplified - would need actual volume data)
                volume = pd.Series(np.random.lognormal(10, 1, len(returns_data[symbol])))
                volume.index = returns_data[symbol].index
                liquidity_regimes[symbol] = self.indicators.liquidity_regime(
                    returns_data[symbol], volume
                ).to_dict()
            except Exception:
                liquidity_regimes[symbol] = {}
        analysis['liquidity_regimes'] = liquidity_regimes
        
        # Market microstructure regimes
        microstructure_regimes = {}
        for symbol in returns_data.keys():
            try:
                volume = pd.Series(np.random.lognormal(10, 1, len(returns_data[symbol])))
                volume.index = returns_data[symbol].index
                microstructure_regimes[symbol] = self.indicators.market_microstructure_regime(
                    returns_data[symbol], volume
                ).to_dict()
            except Exception:
                microstructure_regimes[symbol] = {}
        analysis['microstructure_regimes'] = microstructure_regimes
        
        # Current regime summary
        current_regimes = self._get_current_regime_summary(analysis)
        analysis['current_regime_summary'] = current_regimes
        
        return analysis
    
    def _get_current_regime_summary(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Get current regime summary for all dimensions."""
        summary = {}
        
        # Get latest regimes for each dimension
        for dimension, regimes in analysis.items():
            if dimension.endswith('_regimes') and isinstance(regimes, dict):
                latest_regimes = {}
                for symbol, regime_data in regimes.items():
                    if regime_data:
                        latest_date = max(regime_data.keys(), key=lambda x: pd.to_datetime(x))
                        latest_regimes[symbol] = regime_data[latest_date]
                
                if latest_regimes:
                    # Get most common regime
                    regime_counts = pd.Series(list(latest_regimes.values())).value_counts()
                    summary[dimension.replace('_regimes', '')] = regime_counts.index[0]
        
        return summary
    
    async def _detect_regime_transitions(self, returns_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Detect regime transitions."""
        transitions = {}
        
        for symbol, returns in returns_data.items():
            symbol_transitions = {}
            
            # Detect volatility regime changes
            vol_regime = self.indicators.volatility_regime(returns)
            vol_changes = self.transition_detector.change_point_detection(vol_regime)
            symbol_transitions['volatility'] = [str(date) for date in vol_changes]
            
            # Detect trend regime changes
            trend_regime = self.indicators.trend_regime(returns)
            trend_changes = self.transition_detector.change_point_detection(trend_regime)
            symbol_transitions['trend'] = [str(date) for date in trend_changes]
            
            # Calculate regime stability
            vol_stability = self.transition_detector.regime_stability_score(vol_regime)
            trend_stability = self.transition_detector.regime_stability_score(trend_regime)
            
            symbol_transitions['stability'] = {
                'volatility': vol_stability.to_dict(),
                'trend': trend_stability.to_dict()
            }
            
            transitions[symbol] = symbol_transitions
        
        return transitions
    
    async def _forecast_regimes(self, returns_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Forecast future market regimes."""
        forecast = {}
        
        try:
            # Prepare features for forecasting
            features = self.forecaster.prepare_features(returns_data)
            
            if features.empty:
                return {'error': 'Insufficient data for forecasting'}
            
            # Train models for each symbol
            for symbol, returns in returns_data.items():
                symbol_forecast = {}
                
                try:
                    # Create regime labels (simplified)
                    vol_regime = self.indicators.volatility_regime(returns)
                    trend_regime = self.indicators.trend_regime(returns)
                    
                    # Combine into composite regime
                    composite_regime = vol_regime.combine(trend_regime, 
                                                        lambda x, y: f"{x}_{y}")
                    
                    # Train model
                    model_info = self.forecaster.train_regime_classifier(features, composite_regime)
                    
                    if 'error' not in model_info:
                        # Predict future regimes
                        future_features = features.tail(10)  # Last 10 observations
                        probabilities = self.forecaster.predict_regime_probability(
                            future_features, model_info
                        )
                        
                        symbol_forecast = {
                            'model_accuracy': model_info['training_accuracy'],
                            'feature_importance': model_info['feature_importance'].to_dict(),
                            'predicted_regimes': probabilities.to_dict() if not probabilities.empty else {},
                            'most_likely_regime': probabilities.idxmax(axis=1).to_dict() if not probabilities.empty else {}
                        }
                    
                except Exception as e:
                    self.logger.error(f"Forecasting error for {symbol}: {str(e)}")
                    symbol_forecast = {'error': str(e)}
                
                forecast[symbol] = symbol_forecast
        
        except Exception as e:
            forecast = {'error': f'Forecasting failed: {str(e)}'}
        
        return forecast
    
    async def _generate_regime_signals(self, regime_analysis: Dict[str, Any], 
                                     forecast_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on regime analysis."""
        signals = {}
        
        try:
            # Analyze current market state
            current_regimes = regime_analysis.get('current_regime_summary', {})
            
            # Generate volatility-based signals
            volatility_signal = self._generate_volatility_signal(regime_analysis)
            signals['volatility_signal'] = volatility_signal
            
            # Generate trend-based signals
            trend_signal = self._generate_trend_signal(regime_analysis)
            signals['trend_signal'] = trend_signal
            
            # Generate transition signals
            transition_signal = self._generate_transition_signal(regime_analysis)
            signals['transition_signal'] = transition_signal
            
            # Generate forecast-based signals
            forecast_signal = self._generate_forecast_signal(forecast_analysis)
            signals['forecast_signal'] = forecast_signal
            
            # Composite signal
            composite_signal = self._calculate_composite_signal(signals)
            signals['composite_signal'] = composite_signal
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            signals = {'error': str(e)}
        
        return signals
    
    def _generate_volatility_signal(self, regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate volatility-based signals."""
        vol_regimes = regime_analysis.get('volatility_regimes', {})
        
        if not vol_regimes:
            return {'signal': 'neutral', 'confidence': 0.0}
        
        # Count current volatility regimes
        current_vols = []
        for symbol_regimes in vol_regimes.values():
            if symbol_regimes:
                latest_date = max(symbol_regimes.keys(), key=lambda x: pd.to_datetime(x))
                current_vols.append(symbol_regimes[latest_date])
        
        if not current_vols:
            return {'signal': 'neutral', 'confidence': 0.0}
        
        vol_counts = pd.Series(current_vols).value_counts()
        dominant_vol = vol_counts.index[0]
        confidence = vol_counts.iloc[0] / len(current_vols)
        
        # Generate signals based on volatility regime
        if 'High Vol' in dominant_vol:
            signal = 'risk_off'
        elif 'Low Vol' in dominant_vol:
            signal = 'risk_on'
        else:
            signal = 'neutral'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'dominant_regime': dominant_vol,
            'regime_distribution': vol_counts.to_dict()
        }
    
    def _generate_trend_signal(self, regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trend-based signals."""
        trend_regimes = regime_analysis.get('trend_regimes', {})
        
        if not trend_regimes:
            return {'signal': 'neutral', 'confidence': 0.0}
        
        # Count current trend regimes
        current_trends = []
        for symbol_regimes in trend_regimes.values():
            if symbol_regimes:
                latest_date = max(symbol_regimes.keys(), key=lambda x: pd.to_datetime(x))
                current_trends.append(symbol_regimes[latest_date])
        
        if not current_trends:
            return {'signal': 'neutral', 'confidence': 0.0}
        
        trend_counts = pd.Series(current_trends).value_counts()
        dominant_trend = trend_counts.index[0]
        confidence = trend_counts.iloc[0] / len(current_trends)
        
        # Generate signals based on trend regime
        if 'Bull' in dominant_trend:
            signal = 'bullish'
        elif 'Bear' in dominant_trend:
            signal = 'bearish'
        else:
            signal = 'neutral'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'dominant_regime': dominant_trend,
            'regime_distribution': trend_counts.to_dict()
        }
    
    def _generate_transition_signal(self, regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate regime transition signals."""
        # Analyze recent transitions
        recent_transitions = 0
        transition_symbols = []
        
        # Count transitions in recent period (simplified)
        for symbol, transitions in regime_analysis.get('transition_analysis', {}).items():
            vol_transitions = transitions.get('volatility', [])
            trend_transitions = transitions.get('trend', [])
            
            # Count recent transitions (last 30 days)
            recent_count = len([t for t in vol_transitions + trend_transitions 
                              if pd.to_datetime(t) > pd.Timestamp.now() - pd.Timedelta(days=30)])
            
            if recent_count > 0:
                recent_transitions += recent_count
                transition_symbols.append(symbol)
        
        # Generate signal based on transition frequency
        if recent_transitions >= 5:
            signal = 'high_transition'
            confidence = min(recent_transitions / 10.0, 1.0)
        elif recent_transitions >= 2:
            signal = 'moderate_transition'
            confidence = recent_transitions / 10.0
        else:
            signal = 'stable'
            confidence = 1.0 - (recent_transitions / 10.0)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'transition_count': recent_transitions,
            'active_symbols': transition_symbols
        }
    
    def _generate_forecast_signal(self, forecast_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate forecast-based signals."""
        if 'error' in forecast_analysis:
            return {'signal': 'neutral', 'confidence': 0.0, 'error': forecast_analysis['error']}
        
        # Aggregate predictions across symbols
        all_predictions = []
        confidences = []
        
        for symbol, forecast in forecast_analysis.items():
            if isinstance(forecast, dict) and 'most_likely_regime' in forecast:
                for date, regime in forecast['most_likely_regime'].items():
                    all_predictions.append(regime)
        
        if not all_predictions:
            return {'signal': 'neutral', 'confidence': 0.0}
        
        # Analyze prediction consensus
        pred_counts = pd.Series(all_predictions).value_counts()
        dominant_prediction = pred_counts.index[0]
        confidence = pred_counts.iloc[0] / len(all_predictions)
        
        # Generate signal based on predicted regimes
        if 'Bull' in dominant_prediction:
            signal = 'forecast_bullish'
        elif 'Bear' in dominant_prediction:
            signal = 'forecast_bearish'
        else:
            signal = 'forecast_neutral'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'predicted_regime': dominant_prediction,
            'regime_distribution': pred_counts.to_dict()
        }
    
    def _calculate_composite_signal(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite signal from all individual signals."""
        signal_weights = {
            'volatility_signal': 0.3,
            'trend_signal': 0.4,
            'transition_signal': 0.2,
            'forecast_signal': 0.1
        }
        
        # Score each signal type
        signal_scores = {}
        total_weighted_score = 0
        total_weight = 0
        
        for signal_name, signal_data in signals.items():
            if isinstance(signal_data, dict) and 'signal' in signal_data:
                weight = signal_weights.get(signal_name, 0.1)
                
                # Convert signal to numeric score
                signal_value = signal_data['signal']
                confidence = signal_data.get('confidence', 0.0)
                
                # Signal scoring (bullish = 1, neutral = 0, bearish = -1)
                if 'bullish' in signal_value.lower() or 'risk_on' in signal_value.lower():
                    score = confidence
                elif 'bearish' in signal_value.lower() or 'risk_off' in signal_value.lower():
                    score = -confidence
                else:
                    score = 0.0
                
                signal_scores[signal_name] = {
                    'score': score,
                    'weight': weight,
                    'contribution': score * weight
                }
                
                total_weighted_score += score * weight
                total_weight += weight
        
        # Calculate composite score
        composite_score = total_weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine composite signal
        if composite_score > 0.3:
            composite_signal = 'strong_bullish'
        elif composite_score > 0.1:
            composite_signal = 'bullish'
        elif composite_score < -0.3:
            composite_signal = 'strong_bearish'
        elif composite_score < -0.1:
            composite_signal = 'bearish'
        else:
            composite_signal = 'neutral'
        
        return {
            'composite_signal': composite_signal,
            'composite_score': composite_score,
            'signal_breakdown': signal_scores,
            'confidence': min(abs(composite_score), 1.0)
        }


# Example usage and testing
async def main():
    """Example usage of MarketRegimeForecaster."""
    forecaster = MarketRegimeForecaster()
    
    # Analyze market regimes for major indices
    symbols = ('SPY', 'QQQ', 'IWM')
    start_date = '2023-01-01'
    end_date = '2024-12-01'
    
    result = await forecaster.analyze(symbols, start_date, end_date)
    
    if 'error' not in result:
        print("Market Regime Analysis Complete")
        print(f"Current Regime Summary: {result['regime_dimensions']['current_regime_summary']}")
        print(f"Composite Signal: {result['signals']['composite_signal']['composite_signal']}")
    else:
        print(f"Analysis failed: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())