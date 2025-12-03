"""
Liquidity Prediction Engine Module

This module predicts future liquidity conditions across markets using
machine learning models, statistical analysis, and market microstructure
indicators. Liquidity prediction is crucial for optimal execution and
risk management.

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
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiquidityIndicators:
    """Calculate various liquidity indicators from market data."""
    
    @staticmethod
    def amihud_illiquidity(returns: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Amihud illiquidity ratio."""
        # Handle zero or negative volume
        volume_safe = volume.replace(0, np.nan)
        illiquidity = np.abs(returns) / volume_safe
        return illiquidity.rolling(window=20).mean()
    
    @staticmethod
    def roll_effective_spread(returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Roll's effective spread estimator."""
        returns_lag = returns.shift(1)
        
        # Ensure both series have the same index before calculating covariance
        common_index = list(returns.index.intersection(returns_lag.index))
        returns_aligned = returns.loc[common_index]
        returns_lag_aligned = returns_lag.loc[common_index]
        
        # Remove any remaining NaN values
        valid_data = ~(returns_aligned.isna() | returns_lag_aligned.isna())
        returns_clean = returns_aligned[valid_data]
        returns_lag_clean = returns_lag_aligned[valid_data]
        
        if len(returns_clean) < 2:
            roll_spread = np.nan
        else:
            try:
                cov = np.cov(returns_clean, returns_lag_clean)
                if cov.shape == (2, 2) and cov[0, 1] < 0:
                    roll_spread = 2 * np.sqrt(-cov[0, 1])
                else:
                    roll_spread = np.nan
            except Exception as e:
                logger.warning(f"Error calculating Roll effective spread: {e}")
                roll_spread = np.nan
        
        return pd.Series([roll_spread] * len(returns), index=returns.index)
    
    @staticmethod
    def market_impact(returns: pd.Series, volume: pd.Series, 
                     market_volume: pd.Series, window: int = 20) -> pd.Series:
        """Calculate market impact measure."""
        # Calculate relative volume
        rel_volume = volume / market_volume.rolling(window=20).mean()
        
        # Calculate market impact (simplified)
        impact = np.abs(returns) / rel_volume
        return impact.rolling(window=window).mean()
    
    @staticmethod
    def bid_ask_spread_proxy(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Estimate bid-ask spread using price range."""
        # High-low spread proxy
        spread_proxy = (high - low) / close
        return spread_proxy.rolling(window=20).mean()
    
    @staticmethod
    def liquidity_ratio(high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate liquidity ratio based on price range and volume."""
        price_range = (high - low) / low
        volume_normalized = volume / volume.rolling(window=20).mean()
        liquidity_ratio = 1 / (price_range * volume_normalized)
        return liquidity_ratio.rolling(window=20).mean()
    
    @staticmethod
    def turnover_liquidity(volume: pd.Series, window: int = 20) -> pd.Series:
        """Calculate turnover-based liquidity measure."""
        # Standardized turnover
        turnover = volume.rolling(window=window).mean()
        turnover_vol = volume.rolling(window=window).std()
        standardized_turnover = (volume - turnover) / turnover_vol
        return standardized_turnover.rolling(window=window).mean()
    
    @staticmethod
    def price_efficiency(returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate price efficiency measure."""
        # Calculate price reversal components
        returns_squared = returns ** 2
        abs_returns = np.abs(returns)
        
        efficiency = returns_squared.rolling(window=window).sum() / abs_returns.rolling(window=window).sum()
        return efficiency.fillna(1.0)
    
    @staticmethod
    def volume_volatility_ratio(volume: pd.Series, returns: pd.Series, 
                               window: int = 20) -> pd.Series:
        """Calculate volume-to-volatility ratio."""
        vol_ratio = volume.rolling(window=window).std() / returns.rolling(window=window).std()
        return vol_ratio.fillna(method='ffill')


class LiquidityFactorModel:
    """Model liquidity using multiple factors."""
    
    def __init__(self):
        self.factors = {}
        self.weights = {}
        self.factor_loadings = {}
    
    def extract_factors(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract liquidity factors from market data."""
        factors = pd.DataFrame()
        
        for symbol, df in data.items():
            returns = df['Close'].pct_change().dropna()
            volume = df['Volume'].fillna(method='ffill')
            high = df['High']
            low = df['Low']
            
            # Calculate various liquidity measures
            factors[f'{symbol}_amihud'] = LiquidityIndicators.amihud_illiquidity(returns, volume)
            factors[f'{symbol}_roll_spread'] = LiquidityIndicators.roll_effective_spread(returns)
            factors[f'{symbol}_bid_ask'] = LiquidityIndicators.bid_ask_spread_proxy(high, low, df['Close'])
            factors[f'{symbol}_liquidity_ratio'] = LiquidityIndicators.liquidity_ratio(high, low, volume)
            factors[f'{symbol}_turnover'] = LiquidityIndicators.turnover_liquidity(volume)
            factors[f'{symbol}_efficiency'] = LiquidityIndicators.price_efficiency(returns)
            factors[f'{symbol}_vol_vol_ratio'] = LiquidityIndicators.volume_volatility_ratio(volume, returns)
            
            # Market microstructure factors
            factors[f'{symbol}_volume_trend'] = volume.pct_change().rolling(10).mean()
            factors[f'{symbol}_volume_volatility'] = volume.rolling(20).std()
            factors[f'{symbol}_price_impact'] = np.abs(returns).rolling(20).mean()
            factors[f'{symbol}_volatility'] = returns.rolling(20).std()
            factors[f'{symbol}_skewness'] = returns.rolling(20).skew()
            factors[f'{symbol}_kurtosis'] = returns.rolling(20).kurt()
        
        # Cross-sectional liquidity factors
        if len(data) > 1:
            # Average liquidity across market
            liquidity_cols = [col for col in factors.columns if 'liquidity_ratio' in col]
            if liquidity_cols:
                factors['market_avg_liquidity'] = factors[liquidity_cols].mean(axis=1)
                factors['market_liquidity_dispersion'] = factors[liquidity_cols].std(axis=1)
            
            # Market-wide volume factor
            volume_cols = [col for col in factors.columns if 'volume_trend' in col]
            if volume_cols:
                factors['market_volume_trend'] = factors[volume_cols].mean(axis=1)
        
        return factors.fillna(method='ffill').fillna(0)
    
    def calculate_liquidity_score(self, factors: pd.DataFrame) -> pd.Series:
        """Calculate composite liquidity score from factors."""
        # Define factor weights (can be optimized)
        factor_weights = {
            'amihud': -1.0,  # Negative correlation with liquidity
            'roll_spread': -1.0,  # Negative correlation with liquidity
            'bid_ask': -1.0,  # Negative correlation with liquidity
            'liquidity_ratio': 1.0,  # Positive correlation with liquidity
            'turnover': 1.0,  # Positive correlation with liquidity
            'efficiency': 1.0,  # Positive correlation with liquidity
            'vol_vol_ratio': 1.0,  # Positive correlation with liquidity
        }
        
        liquidity_score = pd.Series(0.0, index=factors.index)
        
        # Calculate weighted score
        for symbol in set([col.split('_')[0] for col in factors.columns if '_' in col]):
            for factor_name, weight in factor_weights.items():
                col_name = f'{symbol}_{factor_name}'
                if col_name in factors.columns:
                    # Normalize factor
                    factor_normalized = (factors[col_name] - factors[col_name].rolling(60).mean()) / factors[col_name].rolling(60).std()
                    liquidity_score += weight * factor_normalized.fillna(0)
        
        # Average across symbols if multiple
        if len(set([col.split('_')[0] for col in factors.columns if '_' in col])) > 1:
            liquidity_score = liquidity_score / len(set([col.split('_')[0] for col in factors.columns if '_' in col]))
        
        return liquidity_score


class LiquidityRegimeModel:
    """Model different liquidity regimes."""
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.regime_model = None
        self.regime_thresholds = {}
        
    def identify_regimes(self, liquidity_score: pd.Series) -> pd.Series:
        """Identify liquidity regimes using clustering."""
        # Prepare data for clustering
        data = liquidity_score.dropna().values.reshape(-1, 1)
        
        if len(data) < 50:
            # Insufficient data, use simple thresholds
            regimes = self._simple_regime_classification(liquidity_score)
            return regimes
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Apply K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        regime_labels = kmeans.fit_predict(data_scaled)
        
        # Convert back to series
        regimes = pd.Series(index=liquidity_score.index, dtype='object')
        regimes.loc[liquidity_score.dropna().index] = [['High', 'Medium', 'Low'][i] for i in regime_labels]
        
        # Sort regimes by liquidity level
        regime_means = {regime: liquidity_score[regimes == regime].mean() 
                       for regime in ['High', 'Medium', 'Low'] if (regimes == regime).any()}
        
        if regime_means:
            sorted_regimes = sorted(regime_means.keys(), key=lambda x: regime_means[x], reverse=True)
            regime_mapping = {sorted_regimes[0]: 'High Liquidity', 
                            sorted_regimes[1]: 'Medium Liquidity',
                            sorted_regimes[2]: 'Low Liquidity'}
            
            for old_regime, new_regime in regime_mapping.items():
                regimes[regimes == old_regime] = new_regime
        
        return regimes.fillna('Medium Liquidity')
    
    def _simple_regime_classification(self, liquidity_score: pd.Series) -> pd.Series:
        """Simple regime classification using percentiles."""
        regimes = pd.Series(index=liquidity_score.index, dtype='object')
        
        high_threshold = liquidity_score.quantile(0.75)
        low_threshold = liquidity_score.quantile(0.25)
        
        regimes[liquidity_score >= high_threshold] = 'High Liquidity'
        regimes[liquidity_score <= low_threshold] = 'Low Liquidity'
        regimes[(liquidity_score > low_threshold) & (liquidity_score < high_threshold)] = 'Medium Liquidity'
        
        return regimes.fillna('Medium Liquidity')
    
    def calculate_regime_transition_probs(self, regimes: pd.Series) -> pd.DataFrame:
        """Calculate regime transition probabilities."""
        # Create transition matrix
        regime_states = ['High Liquidity', 'Medium Liquidity', 'Low Liquidity']
        transition_matrix = pd.DataFrame(0.0, index=regime_states, columns=regime_states)
        
        # Count transitions
        for i in range(1, len(regimes)):
            current_regime = regimes.iloc[i]
            previous_regime = regimes.iloc[i-1]
            
            if current_regime in regime_states and previous_regime in regime_states:
                transition_matrix.loc[previous_regime, current_regime] += 1
        
        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        for state in regime_states:
            if row_sums[state] > 0:
                transition_matrix.loc[state] = transition_matrix.loc[state] / row_sums[state]
        
        return transition_matrix.fillna(0)


class LiquidityPredictor:
    """Machine learning-based liquidity prediction."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_horizons = [1, 5, 10, 20]  # days
        
    def prepare_features(self, factors: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for liquidity prediction."""
        try:
            # DEBUG: Log input data details
            logger.debug(f"DEBUG prepare_features input:")
            logger.debug(f"  factors type: {type(factors)}, shape: {factors.shape}, dtypes: {factors.dtypes.tolist()}")
            logger.debug(f"  target type: {type(target)}, shape: {target.shape}, dtype: {target.dtype}")
            logger.debug(f"  factors index: {factors.index.tolist()}")
            logger.debug(f"  target index: {target.index.tolist()}")
            
            # Align data
            common_index = factors.index.intersection(target.index)
            X = factors.loc[common_index]
            y = target.loc[common_index]
            
            if X.empty or y.empty:
                return X, y
            
            # Create lag features
            lag_features = pd.DataFrame(index=X.index)
            
            for col in X.columns:
                try:
                    col_data = X[col].astype(float)
                    for lag in [1, 2, 3, 5, 10]:
                        lag_features[f'{col}_lag_{lag}'] = col_data.shift(lag)
                except Exception as e:
                    logger.warning(f"Error creating lag features for {col}: {str(e)}")
                    continue
            
            # Create rolling statistics
            for col in X.columns:
                try:
                    col_data = X[col].astype(float)
                    lag_features[f'{col}_ma_5'] = col_data.rolling(5).mean()
                    lag_features[f'{col}_ma_10'] = col_data.rolling(10).mean()
                    lag_features[f'{col}_std_5'] = col_data.rolling(5).std()
                    lag_features[f'{col}_vol_5'] = col_data.rolling(5).std()
                except Exception as e:
                    logger.warning(f"Error creating rolling features for {col}: {str(e)}")
                    continue
            
            # Combine features, ensuring proper alignment
            try:
                # Ensure both DataFrames have the same index and handle NaN values
                X_clean = X.fillna(0)  # Fill original features first
                lag_clean = lag_features.fillna(0)  # Fill lag features first
                
                # Ensure same index by taking intersection
                common_index = X_clean.index.intersection(lag_clean.index)
                X_aligned = X_clean.loc[common_index]
                lag_aligned = lag_clean.loc[common_index]
                
                # Additional safety check: ensure same length
                if len(X_aligned) != len(lag_aligned):
                    # Find the common length and truncate to the smaller one
                    min_len = min(len(X_aligned), len(lag_aligned))
                    X_aligned = X_aligned.iloc[:min_len]
                    lag_aligned = lag_aligned.iloc[:min_len]
                    logger.warning(f"Truncated features to common length: {min_len}")
                
                # DEBUG: Log detailed info before concatenation
                logger.debug(f"DEBUG prepare_features concatenation:")
                logger.debug(f"  X_aligned shape: {X_aligned.shape}, lag_aligned shape: {lag_aligned.shape}")
                logger.debug(f"  X_aligned columns count: {len(X_aligned.columns)}, lag_aligned columns count: {len(lag_aligned.columns)}")
                logger.debug(f"  X_aligned index: {X_aligned.index.tolist()}")
                logger.debug(f"  lag_aligned index: {lag_aligned.index.tolist()}")
                
                try:
                    X_combined = pd.concat([X_aligned, lag_aligned], axis=1, join='inner', sort=False)
                except Exception as concat_error:
                    logger.error(f"Concatenation error: {concat_error}")
                    logger.error(f"  X_aligned dtypes: {X_aligned.dtypes.tolist()}")
                    logger.error(f"  lag_aligned dtypes: {lag_aligned.dtypes.tolist()}")
                    raise concat_error
                
                # Align y with X_combined
                y = y.reindex(X_combined.index, method='ffill').fillna(0)
                
                return X_combined, y
                
            except Exception as e:
                logger.warning(f"Error combining features: {str(e)}")
                return X.fillna(0), y
                
        except Exception as e:
            logger.warning(f"Error in prepare_features: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train prediction models for different horizons."""
        models = {}
        
        for horizon in self.prediction_horizons:
            try:
                # Create target for specific horizon
                y_horizon = y.shift(-horizon)
                
                # Align X and y
                valid_index = X.index.intersection(y_horizon.dropna().index)
                X_valid = X.loc[valid_index]
                y_valid = y_horizon.loc[valid_index]
                
                if len(X_valid) < 100:
                    continue
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_valid)
                
                # Train ensemble model
                models_dict = {
                    'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                    'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
                    'ridge': Ridge(alpha=1.0)
                }
                
                horizon_models = {}
                scores = {}
                
                for name, model in models_dict.items():
                    try:
                        model.fit(X_scaled, y_valid)
                        predictions = model.predict(X_scaled)
                        
                        if hasattr(model, 'feature_importances_'):
                            importance = pd.Series(model.feature_importances_, index=X_valid.columns)
                        else:
                            importance = pd.Series(np.abs(model.coef_), index=X_valid.columns)
                        
                        horizon_models[name] = model
                        scores[name] = {
                            'mse': mean_squared_error(y_valid, predictions),
                            'mae': mean_absolute_error(y_valid, predictions)
                        }
                        
                    except Exception as e:
                        logger.error(f"Error training {name} model for horizon {horizon}: {str(e)}")
                        continue
                
                models[horizon] = {
                    'models': horizon_models,
                    'scaler': scaler,
                    'scores': scores,
                    'feature_importance': importance if 'importance' in locals() else pd.Series()
                }
                
            except Exception as e:
                logger.error(f"Error preparing data for horizon {horizon}: {str(e)}")
                continue
        
        return models
    
    def predict_liquidity(self, X: pd.DataFrame, models: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
        """Predict liquidity for different horizons."""
        predictions = {}
        
        for horizon, model_info in models.items():
            try:
                # Scale features
                X_scaled = model_info['scaler'].transform(X)
                
                horizon_predictions = {}
                
                # Ensemble predictions
                for name, model in model_info['models'].items():
                    pred = model.predict(X_scaled)
                    horizon_predictions[name] = pred[-1] if len(pred) > 0 else 0  # Latest prediction
                
                # Average ensemble predictions
                if horizon_predictions:
                    predictions[horizon] = {
                        'prediction': np.mean(list(horizon_predictions.values())),
                        'ensemble': horizon_predictions,
                        'confidence': 1.0 / (1.0 + np.std(list(horizon_predictions.values())))
                    }
                else:
                    predictions[horizon] = {'prediction': 0.0, 'ensemble': {}, 'confidence': 0.0}
                    
            except Exception as e:
                logger.error(f"Error predicting for horizon {horizon}: {str(e)}")
                predictions[horizon] = {'prediction': 0.0, 'ensemble': {}, 'confidence': 0.0}
        
        return predictions
    
    def calculate_prediction_intervals(self, X: pd.DataFrame, models: Dict[str, Any], 
                                     confidence: float = 0.95) -> Dict[int, Dict[str, float]]:
        """Calculate prediction intervals using ensemble variance."""
        intervals = {}
        
        for horizon, model_info in models.items():
            try:
                # Get predictions from all models
                X_scaled = model_info['scaler'].transform(X)
                predictions = []
                
                for name, model in model_info['models'].items():
                    pred = model.predict(X_scaled)
                    predictions.append(pred[-1] if len(pred) > 0 else 0)
                
                if len(predictions) > 1:
                    pred_mean = np.mean(predictions)
                    pred_std = np.std(predictions)
                    
                    # Calculate confidence interval
                    z_score = stats.norm.ppf((1 + confidence) / 2)
                    margin = z_score * pred_std
                    
                    intervals[horizon] = {
                        'lower': pred_mean - margin,
                        'upper': pred_mean + margin,
                        'mean': pred_mean,
                        'std': pred_std
                    }
                else:
                    intervals[horizon] = {
                        'lower': np.nan,
                        'upper': np.nan,
                        'mean': np.nan,
                        'std': np.nan
                    }
                    
            except Exception as e:
                logger.error(f"Error calculating intervals for horizon {horizon}: {str(e)}")
                intervals[horizon] = {
                    'lower': np.nan,
                    'upper': np.nan,
                    'mean': np.nan,
                    'std': np.nan
                }
        
        return intervals


class LiquidityRiskModel:
    """Model liquidity risk and stress scenarios."""
    
    def __init__(self):
        self.stress_scenarios = {
            'market_crash': {'liquidity_multiplier': 0.3, 'spread_multiplier': 2.5},
            'volatility_spike': {'liquidity_multiplier': 0.6, 'spread_multiplier': 1.8},
            'credit_squeeze': {'liquidity_multiplier': 0.4, 'spread_multiplier': 2.0},
            'systemic_stress': {'liquidity_multiplier': 0.2, 'spread_multiplier': 3.0}
        }
    
    def simulate_stress_scenarios(self, baseline_liquidity: float) -> Dict[str, Dict[str, float]]:
        """Simulate liquidity under different stress scenarios."""
        stress_results = {}
        
        for scenario, multipliers in self.stress_scenarios.items():
            stressed_liquidity = baseline_liquidity * multipliers['liquidity_multiplier']
            
            stress_results[scenario] = {
                'liquidity_level': stressed_liquidity,
                'liquidity_ratio': multipliers['liquidity_multiplier'],
                'expected_spread_widening': multipliers['spread_multiplier'],
                'risk_level': self._calculate_risk_level(stressed_liquidity, baseline_liquidity)
            }
        
        return stress_results
    
    def _calculate_risk_level(self, stressed_liquidity: float, baseline_liquidity: float) -> str:
        """Calculate liquidity risk level."""
        if baseline_liquidity <= 0:
            return 'Unknown'
        
        liquidity_ratio = stressed_liquidity / baseline_liquidity
        
        if liquidity_ratio <= 0.3:
            return 'Critical'
        elif liquidity_ratio <= 0.5:
            return 'High'
        elif liquidity_ratio <= 0.7:
            return 'Medium'
        else:
            return 'Low'


class LiquidityPredictionEngine:
    """
    Liquidity Prediction Engine - Predicts future liquidity conditions
    across markets using machine learning and statistical models.
    """
    
    def __init__(self):
        self.factor_model = LiquidityFactorModel()
        self.regime_model = LiquidityRegimeModel()
        self.predictor = LiquidityPredictor()
        self.risk_model = LiquidityRiskModel()
        self.logger = logging.getLogger(__name__)
    
    async def analyze(self, symbols: tuple, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Analyze and predict liquidity for given symbols.
        
        Args:
            symbols: Tuple of stock symbols
            start_date: Start date for analysis
            end_date: End date for analysis
        
        Returns:
            Dictionary containing liquidity analysis and predictions
        """
        try:
            # Download data
            data = await self._download_data(symbols, start_date, end_date)
            if not data:
                return {'error': 'Failed to download data'}
            
            # Extract liquidity factors
            factors = self.factor_model.extract_factors(data)
            
            # Calculate composite liquidity score
            liquidity_score = self.factor_model.calculate_liquidity_score(factors)
            
            # Identify liquidity regimes
            regimes = self.regime_model.identify_regimes(liquidity_score)
            regime_transitions = self.regime_model.calculate_regime_transition_probs(regimes)
            
            # Prepare training data
            X, y = self.predictor.prepare_features(factors, liquidity_score)
            
            # Train prediction models
            models = self.predictor.train_models(X, y)
            
            # Make predictions
            latest_data = X.tail(10)
            predictions = self.predictor.predict_liquidity(latest_data, models)
            prediction_intervals = self.predictor.calculate_prediction_intervals(latest_data, models)
            
            # Calculate risk metrics
            latest_liquidity = liquidity_score.iloc[-1] if not liquidity_score.empty else 0
            stress_scenarios = self.risk_model.simulate_stress_scenarios(latest_liquidity)
            
            # Generate signals
            signals = await self._generate_liquidity_signals(
                liquidity_score, regimes, predictions, stress_scenarios
            )
            
            return {
                'symbols': symbols,
                'analysis_period': {'start': start_date, 'end': end_date},
                'current_liquidity': {
                    'score': float(latest_liquidity) if not pd.isna(latest_liquidity) else 0,
                    'regime': regimes.iloc[-1] if not regimes.empty else 'Unknown',
                    'timestamp': regimes.index[-1] if not regimes.empty else None
                },
                'factors': {
                    'extracted_factors': factors.to_dict(),
                    'factor_importance': self._get_factor_importance(models)
                },
                'regime_analysis': {
                    'current_regime': regimes.iloc[-1] if not regimes.empty else 'Unknown',
                    'regime_history': regimes.to_dict(),
                    'transition_probabilities': regime_transitions.to_dict() if not regime_transitions.empty else {}
                },
                'predictions': {
                    'horizon_predictions': predictions,
                    'prediction_intervals': prediction_intervals,
                    'confidence_scores': {h: pred.get('confidence', 0) for h, pred in predictions.items()}
                },
                'risk_assessment': {
                    'current_risk_level': self._assess_current_risk(latest_liquidity),
                    'stress_scenarios': stress_scenarios,
                    'stress_indicators': self._calculate_stress_indicators(liquidity_score, regimes)
                },
                'signals': signals,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in liquidity prediction analysis: {str(e)}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    async def _download_data(self, symbols: tuple, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download market data for symbols."""
        data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty and len(hist) > 50:
                    # Handle multi-level columns (when downloading multiple symbols)
                    if isinstance(hist.columns, pd.MultiIndex):
                        # Create single-level columns for this symbol
                        symbol_data = pd.DataFrame(index=hist.index)
                        
                        for col in hist.columns:
                            if col[1] == symbol:  # Only include columns for this symbol
                                symbol_data[col[0]] = hist[col]
                        
                        # Check if we have the required columns
                        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        if all(col in symbol_data.columns for col in required_cols):
                            data[symbol] = symbol_data
                        else:
                            self.logger.warning(f"Missing required columns for {symbol}")
                    else:
                        # Single-level columns
                        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        if all(col in hist.columns for col in required_cols):
                            data[symbol] = hist
                        else:
                            self.logger.warning(f"Missing required columns for {symbol}")
                else:
                    self.logger.warning(f"Insufficient data for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Failed to download data for {symbol}: {str(e)}")
                continue
        
        return data
    
    def _get_factor_importance(self, models: Dict[str, Any]) -> Dict[str, float]:
        """Get average feature importance across models."""
        if not models:
            return {}
        
        # Get feature importance from first available model
        for horizon, model_info in models.items():
            if 'feature_importance' in model_info and not model_info['feature_importance'].empty:
                return model_info['feature_importance'].to_dict()
        
        return {}
    
    def _assess_current_risk(self, liquidity_score: float) -> str:
        """Assess current liquidity risk level."""
        if pd.isna(liquidity_score):
            return 'Unknown'
        
        # Use percentile-based thresholds
        if liquidity_score < -1.0:
            return 'High Risk'
        elif liquidity_score < -0.5:
            return 'Medium Risk'
        elif liquidity_score > 1.0:
            return 'Low Risk'
        else:
            return 'Moderate Risk'
    
    def _calculate_stress_indicators(self, liquidity_score: pd.Series, regimes: pd.Series) -> Dict[str, Any]:
        """Calculate liquidity stress indicators."""
        indicators = {}
        
        if liquidity_score.empty:
            return {'error': 'No liquidity data available'}
        
        # Current level vs historical
        current_level = liquidity_score.iloc[-1]
        percentile_rank = (liquidity_score <= current_level).mean()
        
        indicators['current_percentile'] = float(percentile_rank)
        
        # Volatility of liquidity
        liquidity_volatility = liquidity_score.rolling(30).std().iloc[-1]
        indicators['liquidity_volatility'] = float(liquidity_volatility) if not pd.isna(liquidity_volatility) else 0
        
        # Trend in liquidity
        recent_trend = liquidity_score.tail(10).mean() - liquidity_score.tail(30).head(20).mean()
        indicators['liquidity_trend'] = float(recent_trend) if not pd.isna(recent_trend) else 0
        
        # Regime persistence
        if not regimes.empty:
            recent_regimes = regimes.tail(5)
            regime_persistence = recent_regimes.value_counts().max() / len(recent_regimes)
            indicators['regime_persistence'] = float(regime_persistence)
        
        return indicators
    
    async def _generate_liquidity_signals(self, liquidity_score: pd.Series, regimes: pd.Series,
                                         predictions: Dict[int, Dict[str, float]],
                                         stress_scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate liquidity-based trading signals."""
        signals = {}
        
        try:
            # Current liquidity signal
            current_signal = self._generate_current_liquidity_signal(liquidity_score, regimes)
            signals['current_signal'] = current_signal
            
            # Prediction-based signals
            prediction_signals = self._generate_prediction_signals(predictions)
            signals['prediction_signals'] = prediction_signals
            
            # Risk-based signals
            risk_signals = self._generate_risk_signals(stress_scenarios)
            signals['risk_signals'] = risk_signals
            
            # Composite signal
            composite_signal = self._calculate_composite_liquidity_signal(signals)
            signals['composite_signal'] = composite_signal
            
        except Exception as e:
            self.logger.error(f"Error generating liquidity signals: {str(e)}")
            signals = {'error': str(e)}
        
        return signals
    
    def _generate_current_liquidity_signal(self, liquidity_score: pd.Series, regimes: pd.Series) -> Dict[str, Any]:
        """Generate signal based on current liquidity conditions."""
        if liquidity_score.empty:
            return {'signal': 'neutral', 'confidence': 0.0}
        
        current_liquidity = liquidity_score.iloc[-1]
        
        # Determine signal based on liquidity level
        if current_liquidity > 1.0:
            signal = 'high_liquidity_favorable'
            confidence = min(current_liquidity / 2.0, 1.0)
        elif current_liquidity < -1.0:
            signal = 'low_liquidity_caution'
            confidence = min(abs(current_liquidity) / 2.0, 1.0)
        else:
            signal = 'normal_liquidity'
            confidence = 0.5
        
        # Adjust for current regime
        if not regimes.empty:
            current_regime = regimes.iloc[-1]
            if current_regime == 'Low Liquidity':
                signal = 'stress_liquidity'
                confidence = min(confidence + 0.2, 1.0)
            elif current_regime == 'High Liquidity':
                signal = 'abundant_liquidity'
                confidence = min(confidence + 0.1, 1.0)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'liquidity_score': float(current_liquidity) if not pd.isna(current_liquidity) else 0
        }
    
    def _generate_prediction_signals(self, predictions: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """Generate signals based on liquidity predictions."""
        if not predictions:
            return {'signal': 'no_prediction', 'confidence': 0.0}
        
        # Aggregate predictions across horizons
        pred_values = [pred.get('prediction', 0) for pred in predictions.values() if 'prediction' in pred]
        
        if not pred_values:
            return {'signal': 'invalid_predictions', 'confidence': 0.0}
        
        avg_prediction = np.mean(pred_values)
        prediction_confidence = np.mean([pred.get('confidence', 0) for pred in predictions.values()])
        
        # Generate signal based on predicted liquidity trend
        if avg_prediction > 1.0:
            signal = 'liquidity_improving'
        elif avg_prediction < -1.0:
            signal = 'liquidity_deteriorating'
        else:
            signal = 'liquidity_stable'
        
        return {
            'signal': signal,
            'confidence': prediction_confidence,
            'predicted_liquidity': float(avg_prediction),
            'horizon_details': {str(h): pred for h, pred in predictions.items()}
        }
    
    def _generate_risk_signals(self, stress_scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate signals based on stress scenario analysis."""
        if not stress_scenarios:
            return {'signal': 'no_stress_data', 'confidence': 0.0}
        
        # Analyze worst-case scenario
        worst_case_liquidity = min([scenario.get('liquidity_level', 0) 
                                  for scenario in stress_scenarios.values()])
        
        # Assess overall stress level
        critical_scenarios = sum(1 for scenario in stress_scenarios.values() 
                               if scenario.get('risk_level') == 'Critical')
        
        if critical_scenarios >= 2:
            signal = 'extreme_liquidity_stress'
            confidence = min(critical_scenarios / 4.0, 1.0)
        elif worst_case_liquidity < 0.1:
            signal = 'significant_liquidity_risk'
            confidence = 0.8
        else:
            signal = 'manageable_liquidity_risk'
            confidence = 0.5
        
        return {
            'signal': signal,
            'confidence': confidence,
            'worst_case_liquidity': float(worst_case_liquidity),
            'stress_scenario_summary': {
                scenario: {
                    'liquidity_level': data.get('liquidity_level', 0),
                    'risk_level': data.get('risk_level', 'Unknown')
                }
                for scenario, data in stress_scenarios.items()
            }
        }
    
    def _calculate_composite_liquidity_signal(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite liquidity signal."""
        signal_weights = {
            'current_signal': 0.4,
            'prediction_signals': 0.3,
            'risk_signals': 0.3
        }
        
        # Score each signal component
        total_score = 0
        total_weight = 0
        
        for signal_name, weight in signal_weights.items():
            if signal_name in signals and isinstance(signals[signal_name], dict):
                signal_data = signals[signal_name]
                signal_value = signal_data.get('signal', 'neutral')
                confidence = signal_data.get('confidence', 0.0)
                
                # Convert signal to numeric score
                if 'favorable' in signal_value.lower() or 'improving' in signal_value.lower():
                    score = confidence
                elif 'caution' in signal_value.lower() or 'deteriorating' in signal_value.lower() or 'stress' in signal_value.lower():
                    score = -confidence
                else:
                    score = 0.0
                
                total_score += score * weight
                total_weight += weight
        
        # Calculate composite score
        composite_score = total_score / total_weight if total_weight > 0 else 0
        
        # Determine composite signal
        if composite_score > 0.3:
            composite_signal = 'strong_liquidity_favorable'
        elif composite_score > 0.1:
            composite_signal = 'liquidity_favorable'
        elif composite_score < -0.3:
            composite_signal = 'strong_liquidity_stress'
        elif composite_score < -0.1:
            composite_signal = 'liquidity_concern'
        else:
            composite_signal = 'liquidity_neutral'
        
        return {
            'composite_signal': composite_signal,
            'composite_score': composite_score,
            'confidence': min(abs(composite_score), 1.0),
            'recommendation': self._get_liquidity_recommendation(composite_score)
        }
    
    def _get_liquidity_recommendation(self, composite_score: float) -> str:
        """Get trading recommendation based on liquidity score."""
        if composite_score > 0.3:
            return "High liquidity conditions favor active trading and larger position sizes"
        elif composite_score > 0.1:
            return "Moderate liquidity allows for normal trading activity"
        elif composite_score < -0.3:
            return "Low liquidity conditions require caution, reduce position sizes and execution costs"
        elif composite_score < -0.1:
            return "Liquidity concerns suggest cautious approach and tighter risk controls"
        else:
            return "Neutral liquidity conditions, maintain standard trading practices"


# Example usage and testing
async def main():
    """Example usage of LiquidityPredictionEngine."""
    engine = LiquidityPredictionEngine()
    
    # Analyze liquidity for major indices
    symbols = ('SPY', 'QQQ', 'IWM')
    start_date = '2023-01-01'
    end_date = '2024-12-01'
    
    result = await engine.analyze(symbols, start_date, end_date)
    
    if 'error' not in result:
        print("Liquidity Prediction Analysis Complete")
        print(f"Current Liquidity Score: {result['current_liquidity']['score']:.3f}")
        print(f"Current Regime: {result['current_liquidity']['regime']}")
        print(f"Composite Signal: {result['signals']['composite_signal']['composite_signal']}")
        print(f"Recommendation: {result['signals']['composite_signal']['recommendation']}")
    else:
        print(f"Analysis failed: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())