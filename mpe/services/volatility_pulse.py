"""
Volatility Pulse Engine - Market Energy Build-Up Detection
Forecasts short-term volatility pressure and measures compression/expansion cycles
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import yfinance as yf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import talib
from database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class VolatilityPulseEngine:
    """Volatility Pressure Meter - Predicting energy build-up in markets"""
    
    def __init__(self):
        self.name = "Volatility Pulse Engine"
        self.version = "1.0.0"
        self.status = "initialized"
        self.last_update = None
        self.volatility_cache = {}
        self.models = {}
        self.scaler = StandardScaler()
        
        # Volatility models configuration
        self.volatility_models = {
            "garch": self._garch_volatility,
            "ewma": self._ewma_volatility,
            "realized": self._calculate_realized_volatility,
            "implied": self._implied_volatility,
            "ensemble": self._ensemble_volatility
        }
        
        # Prediction horizons (minutes)
        self.prediction_horizons = [5, 15, 30, 60, 120, 240]
        
        # Volatility percentiles for reference
        self.percentiles = [5, 10, 25, 50, 75, 90, 95]
        
        # Compression/Expansion thresholds
        self.compression_threshold = 0.8
        self.expansion_threshold = 1.2
        
        # Real-time volatility indicators
        self.indicators = {
            "atr": self._calculate_atr,
            "bollinger_width": self._calculate_bollinger_width,
            "rvol": self._calculate_rvol,
            "volatility_skew": self._calculate_volatility_skew,
            "volatility_term_structure": self._calculate_vol_term_structure
        }
        
        # Volatility regime classifications
        self.regimes = {
            "low_vol": {"threshold": 0.15, "description": "Low volatility environment"},
            "normal_vol": {"threshold": 0.25, "description": "Normal volatility range"},
            "high_vol": {"threshold": 0.35, "description": "High volatility environment"},
            "extreme_vol": {"threshold": 0.50, "description": "Extreme volatility period"}
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "models": list(self.volatility_models.keys()),
            "horizons": self.prediction_horizons,
            "cache_size": len(self.volatility_cache)
        }
    
    async def get_pulse_data(self, assets: List[str] = None) -> Dict[str, Any]:
        """Get comprehensive volatility pulse data"""
        if not assets:
            assets = ["SPY", "QQQ", "IWM", "DXY", "TLT", "GLD", "BTCUSD"]
        
        try:
            # Collect current volatility data
            volatility_data = await self._collect_volatility_data(assets)
            
            # Calculate volatility pulse
            pulse_data = await self._calculate_volatility_pulse(volatility_data)
            
            # Generate volatility forecasts
            forecast_data = await self._generate_volatility_forecasts(assets, pulse_data)
            
            # Update cache
            self.volatility_cache = {
                "pulse_data": pulse_data,
                "forecast_data": forecast_data,
                "timestamp": datetime.utcnow()
            }
            self.last_update = datetime.utcnow()
            
            return {
                "timestamp": self.last_update.isoformat(),
                "engine": self.name,
                "assets": assets,
                "volatility_pulse": pulse_data,
                "forecast_data": forecast_data,
                "market_volatility": self._get_market_volatility_summary(pulse_data),
                "volatility_alerts": await self._get_volatility_alerts()
            }
            
        except Exception as e:
            logger.error(f"Error getting volatility pulse data: {e}")
            return {"error": str(e)}
    
    async def _collect_volatility_data(self, assets: List[str]) -> Dict[str, Dict[str, float]]:
        """Collect volatility data from multiple models and sources"""
        volatility_data = {}
        
        for asset in assets:
            volatility_data[asset] = {}
            
            try:
                # Get price data
                ticker = yf.Ticker(asset)
                
                # Different time periods for different analysis
                data_1d = ticker.history(period="2d", interval="1m")  # 1-day minute data
                data_5d = ticker.history(period="5d", interval="5m")  # 5-day 5-minute data
                data_30d = ticker.history(period="30d", interval="1d")  # 30-day daily data
                
                # Calculate various volatility measures
                volatility_measures = {}
                
                if len(data_5d) > 1:
                    volatility_measures["current_5min"] = self._calculate_realized_volatility(data_5d['Close'])
                
                if len(data_30d) > 1:
                    volatility_measures["current_daily"] = self._calculate_realized_volatility(data_30d['Close'])
                    volatility_measures["atr"] = self._calculate_atr(data_30d)
                    volatility_measures["bollinger_width"] = self._calculate_bollinger_width(data_30d['Close'])
                    volatility_measures["rvol"] = self._calculate_rvol(data_30d)
                
                # Calculate model-based volatilities
                for model_name, model_func in self.volatility_models.items():
                    try:
                        vol = model_func(data_30d['Close'] if len(data_30d) > 1 else pd.Series())
                        volatility_measures[model_name] = vol
                    except Exception as e:
                        logger.warning(f"Error calculating {model_name} volatility for {asset}: {e}")
                        volatility_measures[model_name] = 0.0
                
                # Add percentile information
                if len(data_30d) > 1:
                    percentiles = self._calculate_volatility_percentiles(data_30d['Close'])
                    volatility_measures.update(percentiles)
                
                volatility_data[asset] = volatility_measures
                
            except Exception as e:
                logger.error(f"Error collecting volatility data for {asset}: {e}")
                volatility_data[asset] = {"error": str(e)}
        
        return volatility_data
    
    def _calculate_realized_volatility(self, price_series: pd.Series) -> float:
        """Calculate realized volatility"""
        try:
            returns = price_series.pct_change().dropna()
            if len(returns) > 1:
                volatility = returns.std() * np.sqrt(252 * 24 * 12)  # Annualized (assuming 5-min data)
                return volatility
            return 0.0
        except:
            return 0.0
    
    def _garch_volatility(self, price_series: pd.Series) -> float:
        """Simple GARCH-like volatility estimation"""
        try:
            if len(price_series) < 2:
                return 0.0
            
            returns = price_series.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            # Simple GARCH(1,1) approximation
            alpha = 0.1
            beta = 0.85
            omega = 0.01
            
            # Initialize
            var_prev = returns.var()
            
            # GARCH recursion
            for ret in returns[-20:]:  # Use last 20 observations
                var_new = omega + alpha * (ret ** 2) + beta * var_prev
                var_prev = var_new
            
            volatility = np.sqrt(var_prev * 252)  # Annualized
            return volatility
            
        except Exception as e:
            logger.warning(f"GARCH volatility calculation error: {e}")
            return 0.0
    
    def _ewma_volatility(self, price_series: pd.Series) -> float:
        """Exponentially Weighted Moving Average volatility"""
        try:
            if len(price_series) < 2:
                return 0.0
            
            returns = price_series.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            # EWMA with lambda = 0.94
            lambda_wt = 0.94
            ewma_var = returns.ewm(span=int(2 / (1 - lambda_wt) - 1)).var().iloc[-1]
            
            volatility = np.sqrt(ewma_var * 252)  # Annualized
            return volatility
            
        except Exception as e:
            logger.warning(f"EWMA volatility calculation error: {e}")
            return 0.0
    
    def _implied_volatility(self, price_series: pd.Series) -> float:
        """Calculate implied volatility proxy"""
        try:
            if len(price_series) < 2:
                return 0.0
            
            # Simple implied vol approximation using price momentum
            returns = price_series.pct_change().dropna()
            if len(returns) < 5:
                return 0.0
            
            # Recent momentum and volatility
            recent_vol = returns.tail(5).std() * np.sqrt(252)
            momentum = returns.tail(5).mean() * 252
            
            # Approximate implied vol (higher for higher momentum)
            implied_vol = recent_vol * (1 + abs(momentum))
            
            return implied_vol
            
        except Exception as e:
            logger.warning(f"Implied volatility calculation error: {e}")
            return 0.0
    
    def _ensemble_volatility(self, price_series: pd.Series) -> float:
        """Ensemble of multiple volatility models"""
        try:
            if len(price_series) < 2:
                return 0.0
            
            # Calculate different volatility estimates
            estimates = []
            
            # Realized volatility
            realized_vol = self._calculate_realized_volatility(price_series)
            if realized_vol > 0:
                estimates.append(realized_vol)
            
            # GARCH estimate
            garch_vol = self._garch_volatility(price_series)
            if garch_vol > 0:
                estimates.append(garch_vol)
            
            # EWMA estimate
            ewma_vol = self._ewma_volatility(price_series)
            if ewma_vol > 0:
                estimates.append(ewma_vol)
            
            # Ensemble average (weighted)
            if len(estimates) > 0:
                ensemble_vol = np.mean(estimates)
                return ensemble_vol
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Ensemble volatility calculation error: {e}")
            return 0.0
    
    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        try:
            if len(data) < 2:
                return 0.0
            
            # Use talib if available
            try:
                atr_values = talib.ATR(
                    data['High'].values,
                    data['Low'].values,
                    data['Close'].values,
                    timeperiod=14
                )
                atr = atr_values[-1] if len(atr_values) > 0 else 0.0
                
                # Normalize by price
                current_price = data['Close'].iloc[-1]
                normalized_atr = atr / current_price if current_price > 0 else 0.0
                
                return normalized_atr * np.sqrt(252)  # Annualized
                
            except:
                # Manual ATR calculation
                high_low = data['High'] - data['Low']
                high_close = np.abs(data['High'] - data['Close'].shift(1))
                low_close = np.abs(data['Low'] - data['Close'].shift(1))
                
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=14).mean().iloc[-1]
                
                current_price = data['Close'].iloc[-1]
                normalized_atr = atr / current_price if current_price > 0 else 0.0
                
                return normalized_atr * np.sqrt(252)
            
        except Exception as e:
            logger.warning(f"ATR calculation error: {e}")
            return 0.0
    
    def _calculate_bollinger_width(self, price_series: pd.Series) -> float:
        """Calculate Bollinger Band width"""
        try:
            if len(price_series) < 20:
                return 0.0
            
            # Calculate Bollinger Bands
            sma = price_series.rolling(window=20).mean()
            std = price_series.rolling(window=20).std()
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            # Bollinger width as percentage of price
            current_price = price_series.iloc[-1]
            current_sma = sma.iloc[-1]
            current_std = std.iloc[-1]
            
            if current_sma > 0:
                bollinger_width = (upper_band.iloc[-1] - lower_band.iloc[-1]) / current_sma
                return bollinger_width
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Bollinger width calculation error: {e}")
            return 0.0
    
    def _calculate_rvol(self, data: pd.DataFrame) -> float:
        """Calculate Relative Volatility"""
        try:
            if len(data) < 2:
                return 0.0
            
            # Current volatility vs historical average
            current_vol = data['Close'].pct_change().tail(5).std()
            historical_vol = data['Close'].pct_change().rolling(window=20).std().mean()
            
            if historical_vol > 0:
                rvol = current_vol / historical_vol
                return rvol
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Relative volatility calculation error: {e}")
            return 0.0
    
    def _calculate_volatility_skew(self, data: pd.Series) -> float:
        """Calculate volatility skew"""
        try:
            if len(data) < 30:
                return 0.0
            
            returns = data.pct_change().dropna()
            if len(returns) < 10:
                return 0.0
            
            # Calculate skewness of returns
            skew = stats.skew(returns.tail(20))
            return skew
            
        except Exception as e:
            logger.warning(f"Volatility skew calculation error: {e}")
            return 0.0
    
    def _calculate_vol_term_structure(self, price_series: pd.Series) -> Dict[str, float]:
        """Calculate volatility term structure"""
        try:
            if len(price_series) < 10:
                return {}
            
            # Calculate volatilities for different time periods
            returns = price_series.pct_change().dropna()
            
            if len(returns) >= 5:
                short_term_vol = returns.tail(5).std() * np.sqrt(252)
            else:
                short_term_vol = 0.0
            
            if len(returns) >= 20:
                medium_term_vol = returns.tail(20).std() * np.sqrt(252)
            else:
                medium_term_vol = 0.0
            
            if len(returns) >= 60:
                long_term_vol = returns.tail(60).std() * np.sqrt(252)
            else:
                long_term_vol = 0.0
            
            return {
                "short_term": short_term_vol,
                "medium_term": medium_term_vol,
                "long_term": long_term_vol,
                "term_structure_ratio": short_term_vol / long_term_vol if long_term_vol > 0 else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Volatility term structure calculation error: {e}")
            return {}
    
    def _calculate_volatility_percentiles(self, price_series: pd.Series) -> Dict[str, float]:
        """Calculate volatility percentiles"""
        try:
            if len(price_series) < 10:
                return {}
            
            returns = price_series.pct_change().dropna()
            if len(returns) < 5:
                return {}
            
            # Calculate rolling volatilities
            rolling_vols = returns.rolling(window=5).std() * np.sqrt(252)
            rolling_vols = rolling_vols.dropna()
            
            if len(rolling_vols) == 0:
                return {}
            
            # Calculate percentiles
            percentiles = {}
            for p in self.percentiles:
                percentiles[f"vol_percentile_{p}"] = np.percentile(rolling_vols, p)
            
            return percentiles
            
        except Exception as e:
            logger.warning(f"Volatility percentile calculation error: {e}")
            return {}
    
    async def _calculate_volatility_pulse(self, volatility_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate overall volatility pulse for each asset"""
        pulse_data = {}
        
        for asset, vol_data in volatility_data.items():
            if "error" in vol_data:
                continue
            
            # Calculate composite volatility score
            volatility_scores = []
            weights = []
            
            # Collect different volatility measures
            measures_to_use = ["realized", "garch", "ewma", "atr", "current_daily"]
            
            for measure in measures_to_use:
                if measure in vol_data and vol_data[measure] > 0:
                    volatility_scores.append(vol_data[measure])
                    weights.append(1.0)
            
            # Calculate weighted average
            if volatility_scores:
                composite_vol = np.average(volatility_scores, weights=weights)
            else:
                composite_vol = 0.0
            
            # Determine volatility regime
            regime = self._classify_volatility_regime(composite_vol)
            
            # Calculate compression/expansion signals
            compression_signal = self._calculate_compression_signal(vol_data)
            expansion_signal = self._calculate_expansion_signal(vol_data)
            
            # Calculate volatility momentum
            momentum = self._calculate_volatility_momentum(asset, vol_data)
            
            # Determine volatility clustering
            clustering = self._calculate_volatility_clustering(vol_data)
            
            pulse_data[asset] = {
                "composite_volatility": composite_vol,
                "volatility_regime": regime,
                "compression_signal": compression_signal,
                "expansion_signal": expansion_signal,
                "volatility_momentum": momentum,
                "volatility_clustering": clustering,
                "volatility_measures": vol_data,
                "pressure_index": self._calculate_volatility_pressure_index(vol_data),
                "volatility_term_structure": vol_data.get("volatility_term_structure", {})
            }
        
        return pulse_data
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility regime"""
        if volatility <= self.regimes["low_vol"]["threshold"]:
            return "low_volatility"
        elif volatility <= self.regimes["normal_vol"]["threshold"]:
            return "normal_volatility"
        elif volatility <= self.regimes["high_vol"]["threshold"]:
            return "high_volatility"
        else:
            return "extreme_volatility"
    
    def _calculate_compression_signal(self, vol_data: Dict[str, float]) -> float:
        """Calculate volatility compression signal"""
        try:
            if "current_5min" in vol_data and "current_daily" in vol_data:
                # Compare short-term to long-term volatility
                short_vol = vol_data["current_5min"]
                long_vol = vol_data["current_daily"]
                
                if long_vol > 0:
                    ratio = short_vol / long_vol
                    
                    # Compression signal (lower ratio = more compression)
                    compression = 1 - min(ratio, 1.0)
                    return compression
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Compression signal calculation error: {e}")
            return 0.0
    
    def _calculate_expansion_signal(self, vol_data: Dict[str, float]) -> float:
        """Calculate volatility expansion signal"""
        try:
            if "vol_percentile_95" in vol_data and "current_daily" in vol_data:
                # Check if current volatility is in upper percentiles
                current_vol = vol_data["current_daily"]
                vol_95 = vol_data["vol_percentile_95"]
                
                if vol_95 > 0:
                    expansion = min(current_vol / vol_95, 2.0)  # Cap at 2x
                    return expansion
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Expansion signal calculation error: {e}")
            return 0.0
    
    def _calculate_volatility_momentum(self, asset: str, vol_data: Dict[str, float]) -> float:
        """Calculate volatility momentum"""
        # Simplified momentum calculation
        # In production, this would track volatility changes over time
        
        if asset in self.volatility_cache.get("pulse_data", {}):
            prev_vol = self.volatility_cache["pulse_data"][asset].get("composite_volatility", 0)
            current_vol = vol_data.get("current_daily", 0)
            
            if prev_vol > 0:
                momentum = (current_vol - prev_vol) / prev_vol
                return momentum
        
        return 0.0
    
    def _calculate_volatility_clustering(self, vol_data: Dict[str, float]) -> float:
        """Calculate volatility clustering effect"""
        try:
            # Check for volatility clustering (high vol followed by high vol)
            clustering_score = 0.0
            
            # Multiple measures suggest clustering
            indicators = ["rvol", "volatility_momentum"]
            
            clustering_signals = []
            for indicator in indicators:
                if indicator in vol_data:
                    signal = vol_data[indicator]
                    clustering_signals.append(signal)
            
            if clustering_signals:
                # High values suggest clustering
                clustering_score = np.mean([max(0, signal) for signal in clustering_signals])
            
            return clustering_score
            
        except Exception as e:
            logger.warning(f"Volatility clustering calculation error: {e}")
            return 0.0
    
    def _calculate_volatility_pressure_index(self, vol_data: Dict[str, float]) -> float:
        """Calculate overall volatility pressure index"""
        try:
            pressure_components = []
            
            # Current volatility level
            if "current_daily" in vol_data:
                current_vol = vol_data["current_daily"]
                pressure_components.append(min(current_vol / 0.25, 1.0))  # Normalized to 25% vol
            
            # Compression/expansion
            if "compression_signal" in vol_data:
                pressure_components.append(vol_data["compression_signal"])
            
            if "expansion_signal" in vol_data:
                pressure_components.append(vol_data["expansion_signal"])
            
            # Clustering
            if "volatility_clustering" in vol_data:
                pressure_components.append(vol_data["volatility_clustering"])
            
            # Aggregate pressure index
            if pressure_components:
                pressure_index = np.mean(pressure_components)
                return min(pressure_index, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Volatility pressure index calculation error: {e}")
            return 0.0
    
    async def _generate_volatility_forecasts(self, assets: List[str], pulse_data: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Generate volatility forecasts for different horizons"""
        forecast_data = {}
        
        for asset in assets:
            if asset not in pulse_data:
                continue
                
            asset_forecasts = {}
            current_vol = pulse_data[asset]["composite_volatility"]
            
            for horizon in self.prediction_horizons:
                # Simple volatility forecasting model
                # In production, this would use more sophisticated models
                
                # Forecast based on current regime and momentum
                regime_factor = self._get_regime_forecast_factor(pulse_data[asset]["volatility_regime"])
                momentum_factor = 1 + (pulse_data[asset]["volatility_momentum"] * horizon / 60)  # Scale momentum
                
                forecast_vol = current_vol * regime_factor * momentum_factor
                
                # Add confidence intervals
                confidence_range = 0.1 * horizon / 60  # Wider range for longer horizons
                
                asset_forecasts[f"{horizon}min"] = {
                    "forecast_volatility": forecast_vol,
                    "confidence_lower": forecast_vol * (1 - confidence_range),
                    "confidence_upper": forecast_vol * (1 + confidence_range),
                    "confidence_level": max(0.1, 1 - horizon / 480)  # Lower confidence for longer horizons
                }
            
            forecast_data[asset] = asset_forecasts
        
        return forecast_data
    
    def _get_regime_forecast_factor(self, regime: str) -> float:
        """Get forecast factor based on volatility regime"""
        regime_factors = {
            "low_volatility": 1.2,    # Tendency to increase
            "normal_volatility": 1.0, # Stable
            "high_volatility": 0.8,   # Tendency to decrease
            "extreme_volatility": 0.6  # Strong tendency to mean revert
        }
        
        return regime_factors.get(regime, 1.0)
    
    def _get_market_volatility_summary(self, pulse_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get overall market volatility summary"""
        if not pulse_data:
            return {"overall_volatility": 0.0, "regime": "unknown"}
        
        # Calculate overall market volatility
        volatilities = [data["composite_volatility"] for data in pulse_data.values()]
        overall_volatility = np.mean(volatilities)
        
        # Determine dominant regime
        regimes = [data["volatility_regime"] for data in pulse_data.values()]
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        dominant_regime = max(regime_counts.keys(), key=regime_counts.get)
        
        # Volatility stress indicators
        high_vol_assets = [asset for asset, data in pulse_data.items() 
                          if data["volatility_regime"] in ["high_volatility", "extreme_volatility"]]
        
        compression_alerts = [asset for asset, data in pulse_data.items() 
                             if data["compression_signal"] > 0.7]
        
        return {
            "overall_volatility": overall_volatility,
            "overall_regime": dominant_regime,
            "regime_distribution": regime_counts,
            "high_volatility_assets": high_vol_assets,
            "compression_alerts": compression_alerts,
            "average_pressure_index": np.mean([data["pressure_index"] for data in pulse_data.values()]),
            "most_volatile_asset": max(pulse_data.keys(), key=lambda x: pulse_data[x]["composite_volatility"]),
            "least_volatile_asset": min(pulse_data.keys(), key=lambda x: pulse_data[x]["composite_volatility"])
        }
    
    async def _get_volatility_alerts(self) -> List[Dict[str, Any]]:
        """Generate volatility-based alerts"""
        alerts = []
        
        if "pulse_data" not in self.volatility_cache:
            return alerts
        
        pulse_data = self.volatility_cache["pulse_data"]
        
        for asset, data in pulse_data.items():
            # Extreme volatility alerts
            if data["volatility_regime"] == "extreme_volatility":
                alerts.append({
                    "type": "extreme_volatility",
                    "asset": asset,
                    "severity": "high",
                    "message": f"Extreme volatility detected in {asset} ({data['composite_volatility']:.3f})",
                    "data": data
                })
            
            # Compression alerts (calm before the storm)
            if data["compression_signal"] > 0.7:
                alerts.append({
                    "type": "volatility_compression",
                    "asset": asset,
                    "severity": "medium",
                    "message": f"Low volatility compression detected in {asset} - potential breakout risk",
                    "data": data
                })
            
            # Expansion alerts
            if data["expansion_signal"] > 1.2:
                alerts.append({
                    "type": "volatility_expansion",
                    "asset": asset,
                    "severity": "high",
                    "message": f"Volatility expansion detected in {asset}",
                    "data": data
                })
            
            # High momentum alerts
            if abs(data["volatility_momentum"]) > 0.3:
                direction = "increasing" if data["volatility_momentum"] > 0 else "decreasing"
                alerts.append({
                    "type": "volatility_momentum",
                    "asset": asset,
                    "severity": "medium",
                    "message": f"Volatility momentum {direction} in {asset}",
                    "data": data
                })
        
        return alerts
    
    async def cleanup(self):
        """Cleanup resources"""
        self.volatility_cache.clear()
        self.models.clear()
        logger.info("Volatility Pulse Engine cleaned up")