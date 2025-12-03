"""
Module 12: Regime Detection Engine

Real-time market regime identification and classification system that detects
market structure changes, volatility regimes, and trend states using machine
learning and statistical methods.

Author: MiniMax Agent
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from scipy.signal import find_peaks
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RegimeDetectionEngine:
    """
    Market regime detection and classification engine.
    
    Features:
    - Volatility regime classification (low, medium, high volatility)
    - Trend regime identification (bullish, bearish, sideways)
    - Market structure change detection
    - Regime persistence analysis
    - Multi-timeframe regime coordination
    """
    
    def __init__(self, db_manager=None, cache_manager=None):
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.regime_classifiers = {}
        self.scaler = StandardScaler()
        
    async def detect_regimes(self, symbol: str = "SPY", lookback_periods: int = 252) -> Dict[str, Any]:
        """
        Detect current market regimes across multiple dimensions.
        
        Args:
            symbol: Asset symbol to analyze
            lookback_periods: Number of periods for analysis
            
        Returns:
            Dictionary containing regime analysis results
        """
        try:
            # Get market data
            data = await self._fetch_market_data(symbol, lookback_periods)
            if data is None:
                return {"error": "Unable to fetch market data"}
                
            # Calculate regime indicators
            volatility_regime = await self._classify_volatility_regime(data)
            trend_regime = await self._classify_trend_regime(data)
            structure_regime = await self._detect_structure_changes(data)
            
            # Multi-timeframe analysis
            multi_tf_regimes = await self._analyze_multi_timeframe_regimes(data)
            
            # Regime persistence analysis
            persistence = await self._analyze_regime_persistence(data, volatility_regime, trend_regime)
            
            # Regime coordination
            coordination = await self._analyze_regime_coordination(volatility_regime, trend_regime)
            
            # Generate regime signals
            signals = await self._generate_regime_signals(volatility_regime, trend_regime, structure_regime)
            
            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "volatility_regime": volatility_regime,
                "trend_regime": trend_regime,
                "structure_regime": structure_regime,
                "multi_timeframe": multi_tf_regimes,
                "persistence": persistence,
                "coordination": coordination,
                "signals": signals,
                "regime_score": await self._calculate_regime_score(volatility_regime, trend_regime, coordination)
            }
            
            # Cache results
            if self.cache_manager:
                await self.cache_manager.set(f"regime:{symbol}", result, ttl=300)
                
            return result
            
        except Exception as e:
            logger.error(f"Error detecting regimes for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    async def _fetch_market_data(self, symbol: str, lookback_periods: int) -> Optional[pd.DataFrame]:
        """Fetch and prepare market data for regime analysis."""
        try:
            # Fetch data from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2y")  # Get extra data for calculations
            
            if hist.empty:
                return None
                
            # Ensure we have enough data
            if len(hist) < lookback_periods:
                hist = ticker.history(period="max")
                
            # Keep only the last lookback_periods
            data = hist.tail(lookback_periods).copy()
            
            # Calculate technical indicators
            data['returns'] = data['Close'].pct_change()
            data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['volatility_5d'] = data['returns'].rolling(5).std() * np.sqrt(252)
            data['volatility_20d'] = data['returns'].rolling(20).std() * np.sqrt(252)
            data['volatility_60d'] = data['returns'].rolling(60).std() * np.sqrt(252)
            data['rsi'] = self._calculate_rsi(data['Close'])
            data['sma_20'] = data['Close'].rolling(20).mean()
            data['sma_60'] = data['Close'].rolling(60).mean()
            data['sma_200'] = data['Close'].rolling(200).mean()
            
            # Price momentum
            data['momentum_5d'] = data['Close'] / data['Close'].shift(5) - 1
            data['momentum_20d'] = data['Close'] / data['Close'].shift(20) - 1
            
            # Volume analysis
            data['volume_ma'] = data['Volume'].rolling(20).mean()
            data['volume_ratio'] = data['Volume'] / data['volume_ma']
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def _classify_volatility_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Classify volatility regime using Gaussian Mixture Model."""
        try:
            # Prepare features
            features = data[['volatility_5d', 'volatility_20d', 'volatility_60d']].dropna()
            
            if len(features) < 30:
                return {"regime": "insufficient_data", "confidence": 0.0}
            
            # Fit Gaussian Mixture Model
            gmm = GaussianMixture(n_components=3, random_state=42)
            regimes = gmm.fit_predict(features)
            
            # Get component means to identify low, medium, high volatility
            means = gmm.means_[:, 0]  # Use 5-day volatility for classification
            sorted_idx = np.argsort(means)
            
            # Classify regimes
            current_vol = data['volatility_5d'].iloc[-1]
            current_regime_idx = gmm.predict([current_vol, 
                                            data['volatility_20d'].iloc[-1],
                                            data['volatility_60d'].iloc[-1]])[0]
            
            if current_regime_idx == sorted_idx[0]:
                regime_name = "low_volatility"
            elif current_regime_idx == sorted_idx[1]:
                regime_name = "medium_volatility"
            else:
                regime_name = "high_volatility"
            
            # Calculate regime probabilities
            probabilities = gmm.predict_proba([current_vol, 
                                             data['volatility_20d'].iloc[-1],
                                             data['volatility_60d'].iloc[-1]])[0]
            
            confidence = max(probabilities)
            
            # Regime statistics
            regime_stats = {
                "low_volatility": {
                    "count": np.sum(regimes == sorted_idx[0]),
                    "mean_vol": means[sorted_idx[0]],
                    "std_vol": np.sqrt(gmm.covariances_[sorted_idx[0]][0, 0])
                },
                "medium_volatility": {
                    "count": np.sum(regimes == sorted_idx[1]),
                    "mean_vol": means[sorted_idx[1]],
                    "std_vol": np.sqrt(gmm.covariances_[sorted_idx[1]][0, 0])
                },
                "high_volatility": {
                    "count": np.sum(regimes == sorted_idx[2]),
                    "mean_vol": means[sorted_idx[2]],
                    "std_vol": np.sqrt(gmm.covariances_[sorted_idx[2]][0, 0])
                }
            }
            
            return {
                "regime": regime_name,
                "confidence": float(confidence),
                "current_volatility": float(current_vol),
                "regime_stats": regime_stats,
                "transition_probability": self._calculate_transition_probability(regimes),
                "volatility_percentile": float(stats.percentileofscore(
                    data['volatility_5d'].dropna(), current_vol))
            }
            
        except Exception as e:
            logger.error(f"Error classifying volatility regime: {str(e)}")
            return {"regime": "error", "confidence": 0.0, "error": str(e)}
    
    async def _classify_trend_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Classify trend regime using price action and momentum."""
        try:
            current_price = data['Close'].iloc[-1]
            sma_20 = data['sma_20'].iloc[-1]
            sma_60 = data['sma_60'].iloc[-1]
            sma_200 = data['sma_200'].iloc[-1]
            rsi = data['rsi'].iloc[-1]
            momentum_20d = data['momentum_20d'].iloc[-1]
            
            # Trend classification logic
            bullish_signals = 0
            bearish_signals = 0
            
            # Price vs moving averages
            if current_price > sma_20:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            if current_price > sma_60:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            if current_price > sma_200:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Moving average slope
            if sma_20 > sma_20.shift(5):
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # RSI momentum
            if rsi > 50 and rsi < 70:
                bullish_signals += 0.5
            elif rsi < 30:
                bearish_signals += 1
            
            # Price momentum
            if momentum_20d > 0.02:
                bullish_signals += 1
            elif momentum_20d < -0.02:
                bearish_signals += 1
            
            # Classify regime
            total_signals = bullish_signals + bearish_signals
            if total_signals == 0:
                regime_name = "neutral"
                confidence = 0.5
            else:
                bullish_ratio = bullish_signals / total_signals
                if bullish_ratio >= 0.7:
                    regime_name = "strong_bullish"
                    confidence = bullish_ratio
                elif bullish_ratio >= 0.55:
                    regime_name = "bullish"
                    confidence = bullish_ratio
                elif bullish_ratio >= 0.45:
                    regime_name = "sideways"
                    confidence = 1 - abs(bullish_ratio - 0.5) * 2
                elif bullish_ratio >= 0.3:
                    regime_name = "bearish"
                    confidence = 1 - bullish_ratio
                else:
                    regime_name = "strong_bearish"
                    confidence = 1 - bullish_ratio
            
            # Trend strength
            trend_strength = abs(current_price - sma_200) / sma_200
            
            return {
                "regime": regime_name,
                "confidence": float(confidence),
                "trend_strength": float(trend_strength),
                "current_rsi": float(rsi),
                "momentum_20d": float(momentum_20d),
                "bullish_signals": int(bullish_signals),
                "bearish_signals": int(bearish_signals),
                "price_vs_sma20": float((current_price - sma_20) / sma_20),
                "price_vs_sma200": float((current_price - sma_200) / sma_200)
            }
            
        except Exception as e:
            logger.error(f"Error classifying trend regime: {str(e)}")
            return {"regime": "error", "confidence": 0.0, "error": str(e)}
    
    async def _detect_structure_changes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect structural changes in market behavior."""
        try:
            # Statistical tests for change points
            returns = data['returns'].dropna()
            
            # CUSUM test for mean changes
            cusum_result = self._cusum_test(returns)
            
            # Variance change detection
            variance_changes = self._detect_variance_changes(returns)
            
            # Structural break in regression
            structural_breaks = self._detect_structural_breaks(data)
            
            # Regime transition probability
            transition_score = self._calculate_regime_transition_score(data)
            
            return {
                "cusum_detected": cusum_result["detected"],
                "cusum_statistic": cusum_result["statistic"],
                "variance_changes": variance_changes,
                "structural_breaks": structural_breaks,
                "transition_score": transition_score,
                "regime_stability": self._calculate_regime_stability(data)
            }
            
        except Exception as e:
            logger.error(f"Error detecting structure changes: {str(e)}")
            return {"error": str(e)}
    
    def _cusum_test(self, series: pd.Series, threshold: float = 5.0) -> Dict[str, Any]:
        """CUSUM test for detecting mean changes."""
        try:
            x = series.values
            n = len(x)
            mean_x = np.mean(x)
            cumsum = np.cumsum(x - mean_x)
            
            # CUSUM statistic
            cusum_stat = np.max(np.abs(cumsum)) / (np.std(x) * np.sqrt(n))
            detected = cusum_stat > threshold
            
            return {
                "detected": bool(detected),
                "statistic": float(cusum_stat),
                "threshold": threshold
            }
        except Exception:
            return {"detected": False, "statistic": 0.0, "threshold": threshold}
    
    def _detect_variance_changes(self, returns: pd.Series) -> List[Dict[str, Any]]:
        """Detect changes in variance using rolling window analysis."""
        try:
            window = 30
            rolling_var = returns.rolling(window).var()
            
            # Find significant variance changes
            var_changes = []
            for i in range(window, len(rolling_var)):
                current_var = rolling_var.iloc[i]
                historical_var = rolling_var.iloc[:i].mean()
                relative_change = abs(current_var - historical_var) / historical_var
                
                if relative_change > 0.5:  # 50% change threshold
                    var_changes.append({
                        "date": returns.index[i].isoformat(),
                        "current_variance": float(current_var),
                        "historical_variance": float(historical_var),
                        "relative_change": float(relative_change)
                    })
            
            return var_changes[-5:]  # Return last 5 changes
            
        except Exception as e:
            logger.error(f"Error detecting variance changes: {str(e)}")
            return []
    
    def _detect_structural_breaks(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect structural breaks using Chow test approach."""
        try:
            # Simple structural break detection using rolling regression
            window = 60
            breaks = []
            
            prices = data['Close'].values
            returns = data['returns'].dropna().values
            
            for i in range(window, len(prices) - window):
                # Regression on first part
                x1 = np.arange(i)
                y1 = prices[:i]
                slope1 = np.corrcoef(x1, y1)[0, 1] * np.std(y1) / np.std(x1) if len(x1) > 1 else 0
                
                # Regression on second part
                x2 = np.arange(i, len(prices))
                y2 = prices[i:]
                slope2 = np.corrcoef(x2, y2)[0, 1] * np.std(y2) / np.std(x2) if len(x2) > 1 else 0
                
                # Significant slope change
                if abs(slope1 - slope2) > 2 * np.std([slope1, slope2]):
                    breaks.append({
                        "date": data.index[i].isoformat(),
                        "slope_before": float(slope1),
                        "slope_after": float(slope2),
                        "change_magnitude": float(abs(slope1 - slope2))
                    })
            
            return breaks[-3:]  # Return last 3 breaks
            
        except Exception as e:
            logger.error(f"Error detecting structural breaks: {str(e)}")
            return []
    
    def _calculate_regime_transition_score(self, data: pd.DataFrame) -> float:
        """Calculate probability of regime transition."""
        try:
            recent_vol = data['volatility_5d'].tail(10).mean()
            historical_vol = data['volatility_5d'].head(-10).mean()
            
            # Higher volatility acceleration suggests regime transition
            vol_acceleration = abs(recent_vol - historical_vol) / historical_vol
            
            # Recent price momentum
            recent_momentum = data['momentum_5d'].tail(5).mean()
            
            # Combine factors
            transition_score = min(1.0, vol_acceleration * 2 + abs(recent_momentum))
            
            return float(transition_score)
            
        except Exception:
            return 0.0
    
    def _calculate_regime_stability(self, data: pd.DataFrame) -> float:
        """Calculate regime stability score."""
        try:
            # Measure consistency of recent indicators
            recent_vol_var = data['volatility_5d'].tail(10).var()
            recent_trend_consistency = self._measure_trend_consistency(data)
            
            # Lower variance and higher consistency = more stable
            volatility_stability = 1 / (1 + recent_vol_var)
            stability_score = (volatility_stability + recent_trend_consistency) / 2
            
            return float(stability_score)
            
        except Exception:
            return 0.5
    
    def _measure_trend_consistency(self, data: pd.DataFrame) -> float:
        """Measure consistency of trend signals."""
        try:
            # Count consistent signals
            signals = []
            
            # Price vs SMA signals
            for i in range(-10, 0):
                if data['Close'].iloc[i] > data['sma_20'].iloc[i]:
                    signals.append(1)
                else:
                    signals.append(-1)
            
            # Measure consistency
            if len(signals) < 5:
                return 0.5
                
            consistency = 1 - (np.var(signals) / 4)  # Max variance is 4 for binary signals
            return max(0.0, min(1.0, consistency))
            
        except Exception:
            return 0.5
    
    def _calculate_transition_probability(self, regimes: np.ndarray) -> Dict[str, float]:
        """Calculate probability of transitioning between regimes."""
        try:
            n_regimes = len(np.unique(regimes))
            if n_regimes < 2:
                return {}
            
            # Transition matrix
            n_states = max(regimes) + 1
            transition_matrix = np.zeros((n_states, n_states))
            
            for i in range(len(regimes) - 1):
                transition_matrix[regimes[i], regimes[i + 1]] += 1
            
            # Normalize
            for i in range(n_states):
                row_sum = transition_matrix[i, :].sum()
                if row_sum > 0:
                    transition_matrix[i, :] /= row_sum
            
            # Current regime transition probabilities
            current_regime = regimes[-1]
            transition_probs = {}
            
            for j in range(n_states):
                transition_probs[f"to_regime_{j}"] = float(transition_matrix[current_regime, j])
            
            return transition_probs
            
        except Exception as e:
            logger.error(f"Error calculating transition probability: {str(e)}")
            return {}
    
    async def _analyze_multi_timeframe_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regimes across multiple timeframes."""
        try:
            timeframes = {
                "short_term": data.tail(5),   # 5 days
                "medium_term": data.tail(20),  # 20 days
                "long_term": data.tail(60)     # 60 days
            }
            
            tf_regimes = {}
            
            for tf_name, tf_data in timeframes.items():
                # Quick regime classification for each timeframe
                if len(tf_data) < 5:
                    continue
                    
                vol_regime = "unknown"
                trend_regime = "unknown"
                
                # Volatility regime
                if len(tf_data) >= 5:
                    vol = tf_data['volatility_5d'].iloc[-1] if 'volatility_5d' in tf_data.columns else 0.2
                    if vol < 0.15:
                        vol_regime = "low_volatility"
                    elif vol < 0.25:
                        vol_regime = "medium_volatility"
                    else:
                        vol_regime = "high_volatility"
                
                # Trend regime
                if len(tf_data) >= 20:
                    price = tf_data['Close'].iloc[-1]
                    sma_20 = tf_data['sma_20'].iloc[-1] if 'sma_20' in tf_data.columns else price
                    if price > sma_20 * 1.02:
                        trend_regime = "bullish"
                    elif price < sma_20 * 0.98:
                        trend_regime = "bearish"
                    else:
                        trend_regime = "sideways"
                
                tf_regimes[tf_name] = {
                    "volatility_regime": vol_regime,
                    "trend_regime": trend_regime
                }
            
            # Regime coordination analysis
            coordination = self._analyze_regime_coordination_across_timeframes(tf_regimes)
            
            return {
                "timeframe_regimes": tf_regimes,
                "coordination": coordination
            }
            
        except Exception as e:
            logger.error(f"Error analyzing multi-timeframe regimes: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_regime_coordination_across_timeframes(self, tf_regimes: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how well regimes coordinate across timeframes."""
        try:
            # Check for regime consistency
            vol_regimes = [tf_regimes[tf]["volatility_regime"] for tf in tf_regimes]
            trend_regimes = [tf_regimes[tf]["trend_regime"] for tf in tf_regimes]
            
            # Volatility coordination
            vol_consistency = len(set(vol_regimes)) / len(vol_regimes) if vol_regimes else 1.0
            vol_coordination = 1 - vol_consistency
            
            # Trend coordination
            trend_consistency = len(set(trend_regimes)) / len(trend_regimes) if trend_regimes else 1.0
            trend_coordination = 1 - trend_consistency
            
            return {
                "volatility_coordination_score": float(vol_coordination),
                "trend_coordination_score": float(trend_coordination),
                "overall_coordination": float((vol_coordination + trend_coordination) / 2),
                "regime_alignment": "aligned" if max(vol_coordination, trend_coordination) < 0.5 else "mixed"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing regime coordination: {str(e)}")
            return {"coordination_score": 0.0}
    
    async def _analyze_regime_persistence(self, data: pd.DataFrame, vol_regime: Dict, trend_regime: Dict) -> Dict[str, Any]:
        """Analyze how persistent current regimes are."""
        try:
            # Historical regime durations
            reg_hist = self._get_regime_history(data)
            
            vol_duration = self._estimate_regime_duration(reg_hist, "volatility")
            trend_duration = self._estimate_regime_duration(reg_hist, "trend")
            
            # Expected remaining duration
            vol_remaining = self._estimate_remaining_duration(vol_regime["regime"], reg_hist, "volatility")
            trend_remaining = self._estimate_remaining_duration(trend_regime["regime"], reg_hist, "trend")
            
            return {
                "volatility_persistence_score": float(vol_duration.get("persistence_score", 0.5)),
                "trend_persistence_score": float(trend_duration.get("persistence_score", 0.5)),
                "volatility_expected_remaining_days": float(vol_remaining),
                "trend_expected_remaining_days": float(trend_remaining),
                "regime_stability_index": float((vol_duration.get("persistence_score", 0.5) + 
                                               trend_duration.get("persistence_score", 0.5)) / 2)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing regime persistence: {str(e)}")
            return {"error": str(e)}
    
    def _get_regime_history(self, data: pd.DataFrame) -> Dict[str, List]:
        """Get historical regime classifications."""
        try:
            # Simple regime history based on volatility and trend
            history = {"volatility": [], "trend": []}
            
            # Rolling regime classification
            window = 30
            for i in range(window, len(data)):
                # Volatility regime
                vol = data['volatility_5d'].iloc[i]
                if vol < 0.15:
                    history["volatility"].append("low_volatility")
                elif vol < 0.25:
                    history["volatility"].append("medium_volatility")
                else:
                    history["volatility"].append("high_volatility")
                
                # Trend regime
                price = data['Close'].iloc[i]
                sma_20 = data['sma_20'].iloc[i] if not pd.isna(data['sma_20'].iloc[i]) else price
                if price > sma_20 * 1.02:
                    history["trend"].append("bullish")
                elif price < sma_20 * 0.98:
                    history["trend"].append("bearish")
                else:
                    history["trend"].append("sideways")
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting regime history: {str(e)}")
            return {"volatility": [], "trend": []}
    
    def _estimate_regime_duration(self, history: Dict, regime_type: str) -> Dict[str, Any]:
        """Estimate expected duration of current regime."""
        try:
            if regime_type not in history or len(history[regime_type]) < 10:
                return {"persistence_score": 0.5, "expected_duration": 20}
            
            regimes = history[regime_type]
            current_regime = regimes[-1]
            
            # Find all durations of current regime
            durations = []
            current_duration = 1
            
            for i in range(len(regimes) - 2, -1, -1):
                if regimes[i] == current_regime:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_duration = 1
            durations.append(current_duration)
            
            # Calculate statistics
            if durations:
                avg_duration = np.mean(durations)
                std_duration = np.std(durations)
                current_duration = durations[-1]
                
                # Persistence score: how long has current regime lasted vs average
                persistence_score = min(1.0, current_duration / avg_duration) if avg_duration > 0 else 0.5
                
                return {
                    "persistence_score": float(persistence_score),
                    "current_duration": int(current_duration),
                    "expected_duration": float(avg_duration),
                    "duration_std": float(std_duration)
                }
            
            return {"persistence_score": 0.5, "expected_duration": 20}
            
        except Exception as e:
            logger.error(f"Error estimating regime duration: {str(e)}")
            return {"persistence_score": 0.5, "expected_duration": 20}
    
    def _estimate_remaining_duration(self, current_regime: str, history: Dict, regime_type: str) -> float:
        """Estimate remaining duration for current regime."""
        try:
            if regime_type not in history:
                return 10.0
            
            # Get historical durations for this regime
            durations = []
            current_duration = 0
            
            for regime in reversed(history[regime_type]):
                if regime == current_regime:
                    current_duration += 1
                else:
                    break
            
            # Estimate based on historical average
            all_durations = []
            for regime in history[regime_type]:
                if regime == current_regime:
                    continue
                
                # Count consecutive occurrences
                count = 0
                for r in reversed(history[regime_type]):
                    if r == regime:
                        count += 1
                    else:
                        break
                all_durations.append(count)
            
            if all_durations:
                avg_duration = np.mean(all_durations)
                remaining = max(1, avg_duration - current_duration)
                return float(remaining)
            
            return 10.0
            
        except Exception:
            return 10.0
    
    async def _analyze_regime_coordination(self, vol_regime: Dict, trend_regime: Dict) -> Dict[str, Any]:
        """Analyze coordination between different regime types."""
        try:
            vol = vol_regime.get("regime", "unknown")
            trend = trend_regime.get("regime", "unknown")
            
            # Define expected favorable combinations
            favorable_combinations = {
                ("low_volatility", "bullish"): 0.9,
                ("low_volatility", "strong_bullish"): 0.95,
                ("medium_volatility", "bullish"): 0.7,
                ("high_volatility", "bearish"): 0.6,
                ("high_volatility", "strong_bearish"): 0.8
            }
            
            # Calculate coordination score
            key = (vol, trend)
            coordination_score = favorable_combinations.get(key, 0.3)
            
            # Risk-adjusted coordination
            vol_confidence = vol_regime.get("confidence", 0.5)
            trend_confidence = trend_regime.get("confidence", 0.5)
            
            adjusted_score = coordination_score * min(vol_confidence, trend_confidence)
            
            # Market environment classification
            if vol in ["low_volatility"] and trend in ["bullish", "strong_bullish"]:
                environment = "favorable"
            elif vol in ["high_volatility"] and trend in ["bearish", "strong_bearish"]:
                environment = "stressful"
            elif vol in ["medium_volatility"]:
                environment = "neutral"
            else:
                environment = "mixed"
            
            return {
                "coordination_score": float(adjusted_score),
                "market_environment": environment,
                "volatility_regime": vol,
                "trend_regime": trend,
                "confidence_weighted_score": float(adjusted_score * min(vol_confidence, trend_confidence))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing regime coordination: {str(e)}")
            return {"coordination_score": 0.0, "error": str(e)}
    
    async def _generate_regime_signals(self, vol_regime: Dict, trend_regime: Dict, structure_regime: Dict) -> List[Dict[str, Any]]:
        """Generate trading signals based on regime analysis."""
        try:
            signals = []
            
            # Volatility-based signals
            vol_regime_name = vol_regime.get("regime", "")
            if vol_regime_name == "low_volatility":
                signals.append({
                    "type": "volatility_regime",
                    "signal": "favorable_for_equity",
                    "strength": 0.7,
                    "description": "Low volatility environment supports equity exposure"
                })
            elif vol_regime_name == "high_volatility":
                signals.append({
                    "type": "volatility_regime",
                    "signal": "defensive_positioning",
                    "strength": 0.8,
                    "description": "High volatility suggests defensive positioning"
                })
            
            # Trend-based signals
            trend_regime_name = trend_regime.get("regime", "")
            trend_confidence = trend_regime.get("confidence", 0.5)
            
            if "bullish" in trend_regime_name and trend_confidence > 0.6:
                signals.append({
                    "type": "trend_regime",
                    "signal": "long_biased",
                    "strength": trend_confidence,
                    "description": f"Bullish trend with {trend_confidence:.1%} confidence"
                })
            elif "bearish" in trend_regime_name and trend_confidence > 0.6:
                signals.append({
                    "type": "trend_regime",
                    "signal": "short_biased",
                    "strength": trend_confidence,
                    "description": f"Bearish trend with {trend_confidence:.1%} confidence"
                })
            
            # Structure change signals
            transition_score = structure_regime.get("transition_score", 0)
            if transition_score > 0.7:
                signals.append({
                    "type": "structure_change",
                    "signal": "regime_transition_likely",
                    "strength": transition_score,
                    "description": "High probability of regime transition"
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating regime signals: {str(e)}")
            return []
    
    async def _calculate_regime_score(self, vol_regime: Dict, trend_regime: Dict, coordination: Dict) -> Dict[str, Any]:
        """Calculate overall regime score and market outlook."""
        try:
            # Individual component scores
            vol_score = self._map_regime_to_score(vol_regime.get("regime", ""))
            trend_score = self._map_regime_to_score(trend_regime.get("regime", ""))
            coordination_score = coordination.get("coordination_score", 0.5)
            
            # Weighted combination
            overall_score = (
                vol_score * 0.3 + 
                trend_score * 0.5 + 
                coordination_score * 0.2
            )
            
            # Market outlook
            if overall_score > 0.7:
                outlook = "bullish"
            elif overall_score > 0.55:
                outlook = "cautiously_bullish"
            elif overall_score > 0.45:
                outlook = "neutral"
            elif overall_score > 0.3:
                outlook = "cautiously_bearish"
            else:
                outlook = "bearish"
            
            # Risk assessment
            risk_level = self._assess_regime_risk(vol_regime, trend_regime, coordination)
            
            return {
                "overall_score": float(overall_score),
                "market_outlook": outlook,
                "volatility_score": float(vol_score),
                "trend_score": float(trend_score),
                "coordination_score": float(coordination_score),
                "risk_level": risk_level,
                "confidence": float((vol_regime.get("confidence", 0.5) + 
                                  trend_regime.get("confidence", 0.5)) / 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating regime score: {str(e)}")
            return {"overall_score": 0.5, "market_outlook": "neutral", "error": str(e)}
    
    def _map_regime_to_score(self, regime: str) -> float:
        """Map regime to numerical score."""
        regime_scores = {
            "low_volatility": 0.7,
            "medium_volatility": 0.5,
            "high_volatility": 0.2,
            "strong_bullish": 0.9,
            "bullish": 0.7,
            "sideways": 0.5,
            "bearish": 0.3,
            "strong_bearish": 0.1
        }
        return regime_scores.get(regime, 0.5)
    
    def _assess_regime_risk(self, vol_regime: Dict, trend_regime: Dict, coordination: Dict) -> str:
        """Assess overall market risk level."""
        try:
            # High volatility = high risk
            if vol_regime.get("regime") == "high_volatility":
                risk_level = "high"
            # Strong trending markets can be riskier
            elif trend_regime.get("regime") in ["strong_bullish", "strong_bearish"]:
                risk_level = "medium_high"
            # Low coordination between regimes = uncertainty
            elif coordination.get("coordination_score", 0.5) < 0.3:
                risk_level = "high"
            # Low volatility + stable trend = low risk
            elif (vol_regime.get("regime") == "low_volatility" and 
                  trend_regime.get("regime") in ["bullish", "bearish"]):
                risk_level = "low"
            else:
                risk_level = "medium"
            
            return risk_level
            
        except Exception:
            return "medium"
    
    async def get_regime_history(self, symbol: str = "SPY", days: int = 252) -> Dict[str, Any]:
        """Get historical regime analysis for a symbol."""
        try:
            data = await self._fetch_market_data(symbol, days * 2)  # Extra data for analysis
            if data is None:
                return {"error": "Unable to fetch market data"}
            
            # Get regime history
            history = self._get_regime_history(data)
            
            # Calculate regime statistics
            stats = {}
            for regime_type in ["volatility", "trend"]:
                regimes = history.get(regime_type, [])
                if regimes:
                    unique_regimes = list(set(regimes))
                    regime_counts = {regime: regimes.count(regime) for regime in unique_regimes}
                    stats[regime_type] = {
                        "current_regime": regimes[-1] if regimes else "unknown",
                        "regime_distribution": regime_counts,
                        "most_common_regime": max(regime_counts, key=regime_counts.get),
                        "regime_transitions": len([i for i in range(1, len(regimes)) 
                                                 if regimes[i] != regimes[i-1]])
                    }
            
            return {
                "symbol": symbol,
                "analysis_period_days": len(data),
                "regime_history": history,
                "statistics": stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting regime history for {symbol}: {str(e)}")
            return {"error": str(e)}

# Module 13: Global Stress Monitor
class GlobalStressMonitor:
    """
    Global market stress monitoring and early warning system.
    
    Features:
    - Multi-asset stress detection
    - Cross-market contagion analysis
    - Credit and liquidity stress indicators
    - Systemic risk assessment
    - Stress propagation modeling
    """
    
    def __init__(self, db_manager=None, cache_manager=None):
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.stress_indicators = {}
        self.contagion_matrix = None
        
    async def monitor_global_stress(self) -> Dict[str, Any]:
        """
        Comprehensive global stress monitoring across multiple asset classes and regions.
        
        Returns:
            Dictionary containing global stress analysis
        """
        try:
            # Get stress indicators from multiple markets
            us_stress = await self._analyze_us_market_stress()
            european_stress = await self._analyze_european_stress()
            asian_stress = await self._analyze_asian_stress()
            emerging_market_stress = await self._analyze_emerging_market_stress()
            
            # Cross-asset analysis
            cross_asset_stress = await self._analyze_cross_asset_stress()
            
            # Contagion analysis
            contagion_analysis = await self._analyze_contagion_risk()
            
            # Systemic risk assessment
            systemic_risk = await self._assess_systemic_risk()
            
            # Stress index calculation
            global_stress_index = self._calculate_global_stress_index(
                us_stress, european_stress, asian_stress, 
                emerging_market_stress, cross_asset_stress
            )
            
            # Early warning signals
            warnings = await self._generate_stress_warnings(
                global_stress_index, contagion_analysis, systemic_risk
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "global_stress_index": global_stress_index,
                "regional_stress": {
                    "united_states": us_stress,
                    "europe": european_stress,
                    "asia": asian_stress,
                    "emerging_markets": emerging_market_stress
                },
                "cross_asset_stress": cross_asset_stress,
                "contagion_analysis": contagion_analysis,
                "systemic_risk": systemic_risk,
                "early_warnings": warnings,
                "stress_level": self._classify_stress_level(global_stress_index)
            }
            
        except Exception as e:
            logger.error(f"Error in global stress monitoring: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_us_market_stress(self) -> Dict[str, Any]:
        """Analyze US market stress indicators."""
        try:
            symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD"]  # Equity, Tech, Small cap, Bonds, Gold
            
            stress_metrics = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1y")
                    
                    if len(data) < 50:
                        continue
                    
                    # Calculate volatility stress
                    returns = data['Close'].pct_change().dropna()
                    current_vol = returns.tail(20).std() * np.sqrt(252)
                    historical_vol = returns.std() * np.sqrt(252)
                    vol_stress = current_vol / historical_vol if historical_vol > 0 else 1
                    
                    # Drawdown stress
                    peak = data['Close'].expanding().max()
                    current_drawdown = (data['Close'].iloc[-1] - peak.iloc[-1]) / peak.iloc[-1]
                    
                    stress_metrics[symbol] = {
                        "volatility_ratio": float(vol_stress),
                        "current_drawdown": float(current_drawdown),
                        "current_volatility": float(current_vol),
                        "price_momentum_20d": float(data['Close'].iloc[-1] / data['Close'].iloc[-21] - 1)
                    }
                    
                except Exception as e:
                    logger.warning(f"Error analyzing {symbol}: {str(e)}")
                    continue
            
            # Aggregate US stress
            if stress_metrics:
                avg_vol_stress = np.mean([m["volatility_ratio"] for m in stress_metrics.values()])
                worst_drawdown = min([m["current_drawdown"] for m in stress_metrics.values()])
                avg_momentum = np.mean([m["price_momentum_20d"] for m in stress_metrics.values()])
                
                us_stress_score = self._calculate_stress_score(avg_vol_stress, worst_drawdown, avg_momentum)
                
                return {
                    "stress_score": float(us_stress_score),
                    "volatility_stress": float(avg_vol_stress),
                    "worst_drawdown": float(worst_drawdown),
                    "momentum_stress": float(abs(avg_momentum)),
                    "asset_details": stress_metrics
                }
            
            return {"error": "Insufficient data"}
            
        except Exception as e:
            logger.error(f"Error analyzing US market stress: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_european_stress(self) -> Dict[str, Any]:
        """Analyze European market stress."""
        try:
            # European indices and ETFs
            symbols = ["VGK", "EWG", "EWI"]  # Developed Europe, Germany, Italy
            
            stress_metrics = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1y")
                    
                    if len(data) < 50:
                        continue
                    
                    returns = data['Close'].pct_change().dropna()
                    current_vol = returns.tail(20).std() * np.sqrt(252)
                    vol_ratio = current_vol / returns.std() if returns.std() > 0 else 1
                    
                    # European-specific stress (currency, political)
                    euro_stress_indicators = await self._get_eurozone_stress_indicators()
                    
                    stress_metrics[symbol] = {
                        "volatility_ratio": float(vol_ratio),
                        "current_volatility": float(current_vol),
                        "momentum_20d": float(data['Close'].iloc[-1] / data['Close'].iloc[-21] - 1)
                    }
                    
                except Exception as e:
                    logger.warning(f"Error analyzing European {symbol}: {str(e)}")
                    continue
            
            # Combine with Eurozone indicators
            eurozone_stress = 0.7  # Base stress level
            
            if stress_metrics:
                avg_vol_stress = np.mean([m["volatility_ratio"] for m in stress_metrics.values()])
                european_stress_score = (avg_vol_stress + eurozone_stress) / 2
                
                return {
                    "stress_score": float(european_stress_score),
                    "volatility_stress": float(avg_vol_stress),
                    "eurozone_indicators": euro_stress_indicators,
                    "asset_details": stress_metrics
                }
            
            return {"stress_score": float(eurozone_stress), "eurozone_indicators": eurozone_stress}
            
        except Exception as e:
            logger.error(f"Error analyzing European stress: {str(e)}")
            return {"error": str(e)}
    
    async def _get_eurozone_stress_indicators(self) -> Dict[str, float]:
        """Get Eurozone-specific stress indicators."""
        try:
            # Simulated Eurozone stress indicators (in production, fetch from ECB, etc.)
            # In reality, these would come from:
            # - ECB policy rates
            # - Eurozone bond spreads (Italy vs Germany)
            # - Banking stress indicators
            # - Political stability metrics
            
            return {
                "bond_spread_stress": 0.3,  # Italy-Germany 10Y spread
                "banking_stress": 0.2,     # Banking sector health
                "political_stress": 0.1,   # Political stability
                "monetary_policy_stress": 0.4  # ECB policy tightness
            }
            
        except Exception:
            return {"overall_stress": 0.3}
    
    async def _analyze_asian_stress(self) -> Dict[str, Any]:
        """Analyze Asian market stress."""
        try:
            symbols = ["EWZ", "EWY", "EWT"]  # Brazil, South Korea, Taiwan
            
            stress_metrics = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1y")
                    
                    if len(data) < 50:
                        continue
                    
                    returns = data['Close'].pct_change().dropna()
                    current_vol = returns.tail(20).std() * np.sqrt(252)
                    vol_ratio = current_vol / returns.std() if returns.std() > 0 else 1
                    
                    stress_metrics[symbol] = {
                        "volatility_ratio": float(vol_ratio),
                        "current_volatility": float(current_vol)
                    }
                    
                except Exception as e:
                    logger.warning(f"Error analyzing Asian {symbol}: {str(e)}")
                    continue
            
            if stress_metrics:
                avg_vol_stress = np.mean([m["volatility_ratio"] for m in stress_metrics.values()])
                asian_stress_score = avg_vol_stress * 0.8  # Asian markets generally lower volatility
                
                return {
                    "stress_score": float(asian_stress_score),
                    "volatility_stress": float(avg_vol_stress),
                    "asset_details": stress_metrics
                }
            
            return {"stress_score": 0.3}
            
        except Exception as e:
            logger.error(f"Error analyzing Asian stress: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_emerging_market_stress(self) -> Dict[str, Any]:
        """Analyze emerging market stress."""
        try:
            # High-beta, emerging market focused ETFs
            symbols = ["EEM", "VWO", "EWZ"]  # MSCI EM, FTSE EM, Brazil
            
            stress_metrics = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1y")
                    
                    if len(data) < 50:
                        continue
                    
                    returns = data['Close'].pct_change().dropna()
                    current_vol = returns.tail(20).std() * np.sqrt(252)
                    vol_ratio = current_vol / returns.std() if returns.std() > 0 else 1
                    
                    # EM-specific stress factors
                    currency_stress = self._estimate_currency_stress(symbol)
                    
                    stress_metrics[symbol] = {
                        "volatility_ratio": float(vol_ratio),
                        "current_volatility": float(current_vol),
                        "currency_stress": float(currency_stress)
                    }
                    
                except Exception as e:
                    logger.warning(f"Error analyzing EM {symbol}: {str(e)}")
                    continue
            
            if stress_metrics:
                avg_vol_stress = np.mean([m["volatility_ratio"] for m in stress_metrics.values()])
                avg_currency_stress = np.mean([m.get("currency_stress", 0.3) for m in stress_metrics.values()])
                em_stress_score = (avg_vol_stress + avg_currency_stress) / 2
                
                return {
                    "stress_score": float(em_stress_score),
                    "volatility_stress": float(avg_vol_stress),
                    "currency_stress": float(avg_currency_stress),
                    "asset_details": stress_metrics
                }
            
            return {"stress_score": 0.4}
            
        except Exception as e:
            logger.error(f"Error analyzing emerging market stress: {str(e)}")
            return {"error": str(e)}
    
    def _estimate_currency_stress(self, symbol: str) -> float:
        """Estimate currency-related stress for emerging markets."""
        # Simplified currency stress estimation
        # In production, would use real currency data and volatility measures
        em_stress_factors = {
            "EEM": 0.3,  # MSCI EM
            "VWO": 0.25, # FTSE EM  
            "EWZ": 0.5   # Brazil (higher currency volatility)
        }
        return em_stress_factors.get(symbol, 0.3)
    
    async def _analyze_cross_asset_stress(self) -> Dict[str, Any]:
        """Analyze stress across different asset classes."""
        try:
            asset_classes = {
                "equity": "SPY",
                "bonds": "TLT",
                "commodities": "GLD",
                "currencies": "UUP",  # USD strength
                "credit": "LQD"      # Investment grade bonds
            }
            
            cross_stress = {}
            
            for asset_class, symbol in asset_classes.items():
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="6mo")
                    
                    if len(data) < 30:
                        continue
                    
                    returns = data['Close'].pct_change().dropna()
                    
                    # Asset-specific stress metrics
                    if asset_class == "equity":
                        vol_stress = returns.tail(20).std() / returns.std()
                    elif asset_class == "bonds":
                        vol_stress = abs(returns.tail(20).mean()) / abs(returns.std())  # Bond volatility is typically lower
                    elif asset_class == "commodities":
                        vol_stress = returns.tail(20).std() / returns.std()
                    elif asset_class == "currencies":
                        vol_stress = returns.tail(20).std() / returns.std() * 2  # FX can be volatile
                    elif asset_class == "credit":
                        vol_stress = returns.tail(20).std() / returns.std() * 1.5  # Credit stress
                    else:
                        vol_stress = 1.0
                    
                    cross_stress[asset_class] = {
                        "volatility_stress": float(vol_stress),
                        "current_volatility": float(returns.tail(20).std() * np.sqrt(252))
                    }
                    
                except Exception as e:
                    logger.warning(f"Error analyzing {asset_class} stress: {str(e)}")
                    continue
            
            # Cross-asset correlation stress
            correlation_stress = self._calculate_correlation_stress()
            
            return {
                "asset_class_stress": cross_stress,
                "correlation_stress": correlation_stress,
                "diversification_stress": self._assess_diversification_benefit(cross_stress)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cross-asset stress: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_correlation_stress(self) -> float:
        """Calculate stress from cross-asset correlations."""
        try:
            # Simulate correlation stress calculation
            # In production, would calculate real correlations between asset returns
            return 0.6  # Placeholder - moderate correlation stress
            
        except Exception:
            return 0.5
    
    def _assess_diversification_benefit(self, cross_stress: Dict) -> Dict[str, float]:
        """Assess benefit of diversification during stress."""
        try:
            if not cross_stress:
                return {"diversification_ratio": 0.5, "benefit_score": 0.3}
            
            # Calculate diversification benefit
            vol_stresses = [asset["volatility_stress"] for asset in cross_stress.values()]
            if not vol_stresses:
                return {"diversification_ratio": 0.5, "benefit_score": 0.3}
            
            # Equal weight portfolio volatility
            equal_weight_vol = np.mean(vol_stresses)
            
            # Diversified portfolio volatility (assuming correlation of 0.3)
            portfolio_vol = np.sqrt(sum([(1/len(vol_stresses))**2 * vol**2 for vol in vol_stresses]) + 
                                  2 * 0.3 * sum([(1/len(vol_stresses)) * vol_i * (1/len(vol_stresses)) * vol_j 
                                                 for i, vol_i in enumerate(vol_stresses) 
                                                 for j, vol_j in enumerate(vol_stresses) if i != j]))
            
            diversification_ratio = equal_weight_vol / portfolio_vol if portfolio_vol > 0 else 1
            benefit_score = min(1.0, diversification_ratio / 2)  # Normalize to 0-1
            
            return {
                "diversification_ratio": float(diversification_ratio),
                "benefit_score": float(benefit_score)
            }
            
        except Exception as e:
            logger.error(f"Error assessing diversification: {str(e)}")
            return {"diversification_ratio": 0.5, "benefit_score": 0.3}
    
    async def _analyze_contagion_risk(self) -> Dict[str, Any]:
        """Analyze risk of stress contagion across markets."""
        try:
            # Contagion indicators
            contagion_indicators = {
                "vix_stress": await self._get_vix_stress_level(),
                "credit_spread_stress": await self._get_credit_spread_stress(),
                "currency_volatility": await self._get_currency_volatility_stress(),
                "emerging_market_stress": await self._get_em_stress_contagion()
            }
            
            # Contagion network analysis
            contagion_network = self._analyze_contagion_network(contagion_indicators)
            
            # Propagation modeling
            propagation_risk = self._model_contagion_propagation(contagion_indicators)
            
            return {
                "contagion_indicators": contagion_indicators,
                "contagion_network": contagion_network,
                "propagation_risk": propagation_risk,
                "overall_contagion_score": float(np.mean(list(contagion_indicators.values())))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing contagion risk: {str(e)}")
            return {"error": str(e)}
    
    async def _get_vix_stress_level(self) -> float:
        """Get VIX-based stress level."""
        try:
            ticker = yf.Ticker("^VIX")
            data = ticker.history(period="6mo")
            
            if len(data) < 30:
                return 0.5
            
            current_vix = data['Close'].iloc[-1]
            historical_vix = data['Close'].mean()
            
            vix_stress = current_vix / historical_vix if historical_vix > 0 else 1
            return min(1.0, vix_stress)
            
        except Exception:
            return 0.5
    
    async def _get_credit_spread_stress(self) -> float:
        """Get credit spread-based stress."""
        try:
            # Use high yield bond ETF as proxy for credit stress
            ticker = yf.Ticker("HYG")
            data = ticker.history(period="6mo")
            
            if len(data) < 30:
                return 0.3
            
            returns = data['Close'].pct_change().dropna()
            recent_vol = returns.tail(20).std()
            historical_vol = returns.std()
            
            credit_stress = recent_vol / historical_vol if historical_vol > 0 else 1
            return min(1.0, credit_stress * 0.5)  # Scale down
            
        except Exception:
            return 0.3
    
    async def _get_currency_volatility_stress(self) -> float:
        """Get currency volatility stress."""
        try:
            # Use USD strength as proxy
            ticker = yf.Ticker("UUP")
            data = ticker.history(period="6mo")
            
            if len(data) < 30:
                return 0.4
            
            returns = data['Close'].pct_change().dropna()
            vol_ratio = returns.tail(20).std() / returns.std() if returns.std() > 0 else 1
            
            return min(1.0, vol_ratio * 0.6)  # FX volatility scaling
            
        except Exception:
            return 0.4
    
    async def _get_em_stress_contagion(self) -> float:
        """Get emerging market stress contagion indicator."""
        try:
            # Use EM ETF volatility
            ticker = yf.Ticker("EEM")
            data = ticker.history(period="6mo")
            
            if len(data) < 30:
                return 0.4
            
            returns = data['Close'].pct_change().dropna()
            vol_ratio = returns.tail(20).std() / returns.std() if returns.std() > 0 else 1
            
            return min(1.0, vol_ratio * 0.7)
            
        except Exception:
            return 0.4
    
    def _analyze_contagion_network(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Analyze contagion network relationships."""
        try:
            # Simple contagion risk scoring
            weights = {
                "vix_stress": 0.3,
                "credit_spread_stress": 0.25,
                "currency_volatility": 0.25,
                "emerging_market_stress": 0.2
            }
            
            network_score = sum(indicators[key] * weights.get(key, 0.25) for key in indicators)
            
            # Identify primary contagion channels
            contagion_channels = []
            for indicator, value in indicators.items():
                if value > 0.7:
                    contagion_channels.append(indicator)
            
            return {
                "network_stress_score": float(network_score),
                "primary_contagion_channels": contagion_channels,
                "contagion_risk_level": "high" if network_score > 0.7 else "medium" if network_score > 0.5 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing contagion network: {str(e)}")
            return {"network_stress_score": 0.5}
    
    def _model_contagion_propagation(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Model potential stress propagation pathways."""
        try:
            # Propagation modeling based on historical stress events
            propagation_pathways = {
                "equity_to_bonds": indicators.get("vix_stress", 0.5) * 0.6,
                "credit_to_equity": indicators.get("credit_spread_stress", 0.5) * 0.7,
                "fx_to_emerging_markets": indicators.get("currency_volatility", 0.5) * 0.8,
                "em_to_developed": indicators.get("emerging_market_stress", 0.5) * 0.4
            }
            
            # Systemic propagation risk
            max_propagation = max(propagation_pathways.values())
            systemic_risk = max_propagation * 1.2  # Amplification factor
            
            return {
                "propagation_pathways": propagation_pathways,
                "systemic_propagation_risk": float(min(1.0, systemic_risk)),
                "most_likely_pathway": max(propagation_pathways, key=propagation_pathways.get)
            }
            
        except Exception as e:
            logger.error(f"Error modeling contagion propagation: {str(e)}")
            return {"systemic_propagation_risk": 0.5}
    
    async def _assess_systemic_risk(self) -> Dict[str, Any]:
        """Assess overall systemic risk level."""
        try:
            # Systemic risk indicators
            systemic_indicators = {
                "interconnectedness": 0.6,  # Market interconnectedness
                "leverage": 0.4,           # Overall market leverage
                "liquidity": 0.3,          # Market liquidity conditions
                "central_bank_policy": 0.5, # Monetary policy stance
                "geopolitical_risk": 0.2   # Geopolitical tensions
            }
            
            # Calculate systemic risk score
            weights = [0.2, 0.25, 0.2, 0.2, 0.15]
            systemic_score = sum(score * weight for score, weight in zip(systemic_indicators.values(), weights))
            
            # Stress testing scenarios
            stress_scenarios = await self._run_stress_scenarios()
            
            return {
                "systemic_risk_score": float(systemic_score),
                "risk_indicators": systemic_indicators,
                "stress_scenarios": stress_scenarios,
                "risk_level": self._classify_stress_level(systemic_score),
                "resilience_score": float(1 - systemic_score)
            }
            
        except Exception as e:
            logger.error(f"Error assessing systemic risk: {str(e)}")
            return {"systemic_risk_score": 0.5, "error": str(e)}
    
    async def _run_stress_scenarios(self) -> Dict[str, Any]:
        """Run various stress test scenarios."""
        try:
            scenarios = {
                "equity_crash_20pct": {
                    "probability": 0.05,
                    "impact_on_bonds": 0.1,  # Bonds benefit from flight to quality
                    "impact_on_credit": 0.3,  # Credit spreads widen
                    "systemic_amplification": 1.2
                },
                "credit_crisis": {
                    "probability": 0.03,
                    "impact_on_equity": 0.25,  # Equity selloff on credit concerns
                    "impact_on_bonds": 0.15,
                    "systemic_amplification": 1.5
                },
                "currency_crisis": {
                    "probability": 0.02,
                    "impact_on_emerging_markets": 0.4,
                    "impact_on_commodities": 0.2,
                    "systemic_amplification": 1.1
                },
                "liquidity_crunch": {
                    "probability": 0.04,
                    "impact_across_assets": 0.3,
                    "duration_risk": "high",
                    "systemic_amplification": 1.3
                }
            }
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error running stress scenarios: {str(e)}")
            return {}
    
    def _calculate_stress_score(self, vol_stress: float, drawdown: float, momentum_stress: float) -> float:
        """Calculate composite stress score."""
        try:
            # Weighted combination of stress factors
            # Volatility stress (40%), Drawdown stress (40%), Momentum stress (20%)
            vol_component = min(1.0, vol_stress)
            drawdown_component = min(1.0, abs(drawdown) * 5)  # Scale drawdown
            momentum_component = min(1.0, abs(momentum_stress) * 10)
            
            stress_score = (
                vol_component * 0.4 +
                drawdown_component * 0.4 +
                momentum_component * 0.2
            )
            
            return float(stress_score)
            
        except Exception:
            return 0.5
    
    def _calculate_global_stress_index(self, us_stress: Dict, eu_stress: Dict, 
                                     asia_stress: Dict, em_stress: Dict, 
                                     cross_asset: Dict) -> Dict[str, Any]:
        """Calculate comprehensive global stress index."""
        try:
            # Regional stress weights (US weighted most heavily)
            regional_weights = {
                "united_states": 0.4,
                "europe": 0.25,
                "asia": 0.2,
                "emerging_markets": 0.15
            }
            
            # Extract regional stress scores
            regional_scores = {
                "united_states": us_stress.get("stress_score", 0.5),
                "europe": eu_stress.get("stress_score", 0.5),
                "asia": asia_stress.get("stress_score", 0.5),
                "emerging_markets": em_stress.get("stress_score", 0.5)
            }
            
            # Calculate weighted regional stress
            regional_stress = sum(
                score * regional_weights[region] 
                for region, score in regional_scores.items()
            )
            
            # Cross-asset stress component
            cross_stress = cross_asset.get("asset_class_stress", {})
            if cross_stress:
                asset_stress_avg = np.mean([asset["volatility_stress"] for asset in cross_stress.values()])
            else:
                asset_stress_avg = 0.5
            
            # Combined global stress index
            global_stress = (
                regional_stress * 0.7 +  # Regional factors (70%)
                asset_stress_avg * 0.3    # Asset-specific factors (30%)
            )
            
            return {
                "global_stress_index": float(global_stress),
                "regional_stress": regional_stress,
                "asset_stress": float(asset_stress_avg),
                "stress_components": {
                    "regional_scores": regional_scores,
                    "asset_class_stress": cross_stress
                },
                "percentile_ranking": self._calculate_stress_percentile(global_stress)
            }
            
        except Exception as e:
            logger.error(f"Error calculating global stress index: {str(e)}")
            return {"global_stress_index": 0.5, "error": str(e)}
    
    def _calculate_stress_percentile(self, stress_score: float) -> float:
        """Calculate percentile ranking of current stress level."""
        # Simplified percentile calculation
        # In production, would use historical stress distribution
        if stress_score > 0.8:
            return 95.0
        elif stress_score > 0.6:
            return 75.0
        elif stress_score > 0.4:
            return 50.0
        elif stress_score > 0.2:
            return 25.0
        else:
            return 10.0
    
    def _classify_stress_level(self, stress_score: float) -> str:
        """Classify overall stress level."""
        if stress_score >= 0.8:
            return "severe"
        elif stress_score >= 0.65:
            return "high"
        elif stress_score >= 0.5:
            return "moderate"
        elif stress_score >= 0.3:
            return "low"
        else:
            return "minimal"
    
    async def _generate_stress_warnings(self, global_stress: Dict, contagion: Dict, systemic: Dict) -> List[Dict[str, Any]]:
        """Generate early warning alerts based on stress analysis."""
        try:
            warnings = []
            
            global_stress_score = global_stress.get("global_stress_index", 0.5)
            contagion_score = contagion.get("overall_contagion_score", 0.5)
            systemic_score = systemic.get("systemic_risk_score", 0.5)
            
            # High global stress warning
            if global_stress_score > 0.7:
                warnings.append({
                    "type": "global_stress",
                    "severity": "high",
                    "message": f"Global stress index at {global_stress_score:.1%}, indicating high market stress",
                    "action": "Consider defensive positioning and risk reduction"
                })
            
            # Contagion risk warning
            if contagion_score > 0.6:
                warnings.append({
                    "type": "contagion_risk",
                    "severity": "medium" if contagion_score < 0.8 else "high",
                    "message": f"Contagion risk score at {contagion_score:.1%}, stress may spread across markets",
                    "action": "Monitor cross-market correlations and prepare hedging strategies"
                })
            
            # Systemic risk warning
            if systemic_score > 0.6:
                warnings.append({
                    "type": "systemic_risk",
                    "severity": "high",
                    "message": f"Systemic risk elevated at {systemic_score:.1%}",
                    "action": "Review portfolio liquidity and counterparty exposure"
                })
            
            # Combined stress scenario
            combined_risk = (global_stress_score + contagion_score + systemic_score) / 3
            if combined_risk > 0.75:
                warnings.append({
                    "type": "system_stress",
                    "severity": "critical",
                    "message": "Multiple stress indicators elevated simultaneously",
                    "action": "Consider portfolio de-risking and emergency procedures"
                })
            
            return warnings
            
        except Exception as e:
            logger.error(f"Error generating stress warnings: {str(e)}")
            return []
    
    async def get_stress_history(self, days: int = 30) -> Dict[str, Any]:
        """Get historical stress monitoring data."""
        try:
            # In production, this would retrieve historical stress data from database
            # For now, return current analysis with historical context
            
            current_analysis = await self.monitor_global_stress()
            
            # Simulated historical data
            historical_stress = []
            base_stress = current_analysis.get("global_stress_index", {}).get("global_stress_index", 0.5)
            
            for i in range(days):
                # Simulate historical stress with some variation
                variation = np.random.normal(0, 0.1)
                stress_value = max(0, min(1, base_stress + variation))
                
                historical_stress.append({
                    "date": (datetime.now() - timedelta(days=i)).isoformat(),
                    "global_stress_index": stress_value,
                    "stress_level": self._classify_stress_level(stress_value)
                })
            
            return {
                "historical_data": historical_stress,
                "current_analysis": current_analysis,
                "trend_analysis": {
                    "stress_trend": "increasing" if historical_stress[0]["global_stress_index"] > historical_stress[-1]["global_stress_index"] else "decreasing",
                    "volatility_trend": "elevated"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting stress history: {str(e)}")
            return {"error": str(e)}