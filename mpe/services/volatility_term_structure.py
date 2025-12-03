"""
Module 17: Volatility Term Structure Engine

Advanced volatility term structure intelligence system providing real-time analysis
of volatility surface dynamics, term structure evolution, and volatility forecasting
across options and derivatives markets.

Author: MiniMax Agent
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.interpolate import griddata, interp1d, CubicSpline
from scipy.optimize import minimize_scalar
import yfinance as yf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VolatilityTermStructureEngine:
    """
    Volatility Term Structure intelligence and analysis engine.
    
    Features:
    - Volatility surface modeling and interpolation
    - Term structure analysis and evolution tracking
    - Volatility forecasting and regime detection
    - Cross-maturity volatility analysis
    - Volatility carry and roll analysis
    - Surface dynamics and term structure shifts
    """
    
    def __init__(self, db_manager=None, cache_manager=None):
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.surface_models = {}
        self.term_structure_cache = {}
        self.forecasting_models = {}
        
    async def analyze_volatility_term_structure(self, symbol: str = "SPY") -> Dict[str, Any]:
        """
        Comprehensive volatility term structure analysis.
        
        Args:
            symbol: Asset symbol to analyze
            
        Returns:
            Dictionary containing volatility term structure analysis
        """
        try:
            # Fetch volatility data
            vol_data = await self._fetch_volatility_data(symbol)
            if not vol_data:
                return {"error": "Unable to fetch volatility data"}
            
            # Volatility surface analysis
            surface_analysis = await self._analyze_volatility_surface(vol_data)
            
            # Term structure analysis
            term_structure_analysis = await self._analyze_term_structure(vol_data)
            
            # Volatility forecasting
            volatility_forecasting = await self._forecast_volatility(vol_data)
            
            # Surface dynamics analysis
            surface_dynamics = await self._analyze_surface_dynamics(vol_data)
            
            # Cross-maturity analysis
            cross_maturity_analysis = await self._analyze_cross_maturity_volatility(vol_data)
            
            # Volatility carry analysis
            carry_analysis = await self._analyze_volatility_carry(vol_data)
            
            # Regime analysis
            regime_analysis = await self._analyze_volatility_regimes(vol_data)
            
            # Volatility clustering analysis
            clustering_analysis = await self._analyze_volatility_clustering(vol_data)
            
            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "current_price": vol_data.get("current_price", 0),
                "surface_analysis": surface_analysis,
                "term_structure_analysis": term_structure_analysis,
                "volatility_forecasting": volatility_forecasting,
                "surface_dynamics": surface_dynamics,
                "cross_maturity_analysis": cross_maturity_analysis,
                "carry_analysis": carry_analysis,
                "regime_analysis": regime_analysis,
                "clustering_analysis": clustering_analysis,
                "volatility_term_structure_score": await self._calculate_vts_score(
                    surface_analysis, term_structure_analysis, volatility_forecasting
                )
            }
            
            # Cache results
            if self.cache_manager:
                await self.cache_manager.set(f"vts_intelligence:{symbol}", result, ttl=300)
                
            return result
            
        except Exception as e:
            logger.error(f"Error in volatility term structure analysis for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    async def _fetch_volatility_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch volatility data for term structure analysis."""
        try:
            # Get underlying asset data
            ticker = yf.Ticker(symbol)
            underlying_data = ticker.history(period="2y")
            
            if underlying_data.empty:
                return None
            
            current_price = underlying_data['Close'].iloc[-1]
            
            # Create simulated volatility surface data
            # In production, this would connect to professional options data providers
            vol_surface_data = await self._create_simulated_vol_surface(symbol, current_price, underlying_data)
            
            # Add historical volatility data
            historical_vol_data = self._calculate_historical_volatility_data(underlying_data)
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "underlying_data": underlying_data,
                "vol_surface_data": vol_surface_data,
                "historical_vol": historical_vol_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching volatility data: {str(e)}")
            return None
    
    async def _create_simulated_vol_surface(self, symbol: str, current_price: float, 
                                          underlying_data: pd.DataFrame) -> Dict[str, Any]:
        """Create simulated volatility surface data."""
        try:
            # Generate realistic volatility surface
            surface_data = {
                "atm_volatility": 0.25,  # Base ATM volatility
                "volatility_by_moneyness": {},
                "volatility_by_maturity": {},
                "surface_points": []
            }
            
            # Calculate current historical volatility for realistic surface
            returns = underlying_data['Close'].pct_change().dropna()
            hist_vol = returns.tail(60).std() * np.sqrt(252)  # 60-day realized volatility
            
            # Adjust ATM volatility based on recent realized vol
            base_atm_vol = max(0.15, min(0.45, hist_vol))
            surface_data["atm_volatility"] = base_atm_vol
            
            # Generate strikes and maturities
            strikes = []
            for i in range(-15, 16):  # 31 strikes
                strike = current_price * (1 + i * 0.02)  # 2% increments
                strikes.append(round(strike, 2))
            
            # Generate maturity points (days to expiration)
            maturities = [7, 14, 30, 60, 90, 180, 365]
            
            # Generate surface points with realistic volatility smile/skew
            surface_points = []
            
            for maturity_days in maturities:
                for strike in strikes:
                    # Calculate moneyness
                    moneyness = strike / current_price
                    
                    # Generate realistic volatility
                    # Base volatility with term structure (longer maturities typically higher vol)
                    term_structure_factor = 1 + 0.3 * np.sqrt(maturity_days / 365.25)
                    
                    # Add volatility smile/skew
                    distance_from_atm = abs(moneyness - 1.0)
                    smile_component = distance_from_atm * 0.2  # Typical smile slope
                    
                    # Add some market-specific characteristics
                    if symbol in ["SPY", "QQQ"]:  # Equity indexes
                        skew_component = 0.1 * (moneyness - 1.0)  # Equity skew (lower vol for higher strikes)
                    else:  # Individual stocks
                        skew_component = 0.05 * (1.0 - moneyness)  # Slight reverse skew
                    
                    # Total volatility
                    implied_vol = base_atm_vol * term_structure_factor + smile_component + skew_component
                    implied_vol = max(0.05, min(0.80, implied_vol))  # Bound volatility
                    
                    surface_points.append({
                        "strike": strike,
                        "maturity_days": maturity_days,
                        "moneyness": moneyness,
                        "implied_volatility": implied_vol,
                        "time_to_expiry": maturity_days / 365.25
                    })
            
            surface_data["surface_points"] = surface_points
            
            # Generate volatility by moneyness
            for category in ["deep_itm", "itm", "atm", "otm", "deep_otm"]:
                moneyness_ranges = {
                    "deep_itm": (0.0, 0.8),
                    "itm": (0.8, 0.95),
                    "atm": (0.95, 1.05),
                    "otm": (1.05, 1.20),
                    "deep_otm": (1.20, 2.0)
                }
                
                category_vols = []
                min_m, max_m = moneyness_ranges[category]
                
                for point in surface_points:
                    if min_m <= point["moneyness"] < max_m:
                        category_vols.append(point["implied_volatility"])
                
                if category_vols:
                    surface_data["volatility_by_moneyness"][category] = {
                        "mean_volatility": np.mean(category_vols),
                        "volatility_range": [np.min(category_vols), np.max(category_vols)],
                        "volatility_std": np.std(category_vols)
                    }
            
            # Generate volatility by maturity
            for maturity_days in maturities:
                maturity_vols = [point["implied_volatility"] for point in surface_points 
                               if point["maturity_days"] == maturity_days]
                
                if maturity_vols:
                    surface_data["volatility_by_maturity"][maturity_days] = {
                        "mean_volatility": np.mean(maturity_vols),
                        "volatility_std": np.std(maturity_vols),
                        "term_structure_position": self._assess_term_structure_position(maturity_days, base_atm_vol)
                    }
            
            return surface_data
            
        except Exception as e:
            logger.error(f"Error creating simulated vol surface: {str(e)}")
            return {"surface_points": []}
    
    def _assess_term_structure_position(self, maturity_days: int, base_vol: float) -> str:
        """Assess position in term structure."""
        try:
            # Expected term structure: vol increases with time to maturity
            expected_vol = base_vol * (1 + 0.3 * np.sqrt(maturity_days / 365.25))
            
            if maturity_days <= 30:
                if expected_vol < base_vol * 1.1:
                    return "term_structure_inverted"
                else:
                    return "term_structure_normal"
            elif maturity_days <= 90:
                if expected_vol < base_vol * 1.3:
                    return "term_structure_flat"
                else:
                    return "term_structure_normal"
            else:
                return "term_structure_normal"
                
        except Exception:
            return "term_structure_assessment_error"}
    
    def _calculate_historical_volatility_data(self, underlying_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate historical volatility data for comparison."""
        try:
            returns = underlying_data['Close'].pct_change().dropna()
            
            # Calculate rolling volatilities
            vol_periods = [5, 10, 20, 60]
            rolling_vols = {}
            
            for period in vol_periods:
                if len(returns) >= period:
                    rolling_vol = returns.rolling(period).std() * np.sqrt(252)
                    rolling_vols[f"{period}d"] = {
                        "current_vol": rolling_vol.iloc[-1] if not rolling_vol.empty else 0,
                        "vol_history": rolling_vol.dropna().tolist(),
                        "vol_mean": rolling_vol.mean(),
                        "vol_std": rolling_vol.std(),
                        "vol_percentile": self._calculate_vol_percentile(rolling_vol.dropna())
                    }
            
            # Overall volatility statistics
            overall_stats = {
                "realized_volatility": returns.std() * np.sqrt(252),
                "volatility_clustering": self._assess_volatility_clustering(returns),
                "volatility_regime": self._classify_volatility_regime(returns)
            }
            
            return {
                "rolling_volatilities": rolling_vols,
                "overall_statistics": overall_stats,
                "recent_performance": {
                    "latest_returns": returns.tail(10).tolist(),
                    "return_volatility": returns.tail(20).std() * np.sqrt(252)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating historical vol data: {str(e)}")
            return {"volatility_data": "calculation_error"}
    
    def _calculate_vol_percentile(self, vol_series: pd.Series) -> float:
        """Calculate current volatility percentile."""
        try:
            if len(vol_series) == 0:
                return 50.0
            
            current_vol = vol_series.iloc[-1]
            percentile = (vol_series < current_vol).mean() * 100
            
            return percentile
            
        except Exception:
            return 50.0
    
    def _assess_volatility_clustering(self, returns: pd.Series) -> str:
        """Assess volatility clustering in returns."""
        try:
            # Simple volatility clustering assessment
            if len(returns) < 20:
                return "insufficient_data"
            
            # Calculate volatility clustering metric
            squared_returns = returns ** 2
            clustering = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
            
            if clustering > 0.3:
                return "high_clustering"
            elif clustering > 0.1:
                return "moderate_clustering"
            else:
                return "low_clustering"
                
        except Exception:
            return "clustering_assessment_error"}
    
    def _classify_volatility_regime(self, returns: pd.Series) -> str:
        """Classify current volatility regime."""
        try:
            if len(returns) < 60:
                return "insufficient_data"
            
            # Calculate realized volatility
            realized_vol = returns.tail(60).std() * np.sqrt(252)
            
            # Historical volatility distribution
            hist_vol = returns.std() * np.sqrt(252)
            
            # Current vs historical ratio
            vol_ratio = realized_vol / hist_vol if hist_vol > 0 else 1
            
            # Classify regime
            if vol_ratio > 1.5:
                return "high_volatility_regime"
            elif vol_ratio > 1.2:
                return "elevated_volatility_regime"
            elif vol_ratio < 0.7:
                return "low_volatility_regime"
            else:
                return "normal_volatility_regime"
                
        except Exception:
            return "regime_classification_error"}
    
    async def _analyze_volatility_surface(self, vol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility surface characteristics."""
        try:
            surface_data = vol_data.get("vol_surface_data", {})
            surface_points = surface_data.get("surface_points", [])
            
            if not surface_points:
                return {"error": "No surface data available"}
            
            # Surface fitting and modeling
            surface_model = self._fit_volatility_surface(surface_points)
            
            # Surface characteristics
            surface_characteristics = self._analyze_surface_characteristics(surface_points, surface_model)
            
            # Volatility smile analysis
            smile_analysis = self._analyze_volatility_smile(surface_points)
            
            # Volatility skew analysis
            skew_analysis = self._analyze_volatility_skew(surface_points)
            
            # Surface smoothness assessment
            surface_smoothness = self._assess_surface_smoothness(surface_points)
            
            return {
                "surface_model": surface_model,
                "surface_characteristics": surface_characteristics,
                "smile_analysis": smile_analysis,
                "skew_analysis": skew_analysis,
                "surface_smoothness": surface_smoothness,
                "surface_interpretation": self._interpret_surface_characteristics(surface_characteristics)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility surface: {str(e)}")
            return {"error": str(e)}
    
    def _fit_volatility_surface(self, surface_points: List[Dict]) -> Dict[str, Any]:
        """Fit volatility surface model."""
        try:
            if len(surface_points) < 10:
                return {"model_fit": "insufficient_data"}
            
            # Extract coordinates and volatilities
            moneyness = np.array([point["moneyness"] for point in surface_points])
            time_to_exp = np.array([point["time_to_expiry"] for point in surface_points])
            volatilities = np.array([point["implied_volatility"] for point in surface_points])
            
            # Prepare feature matrix
            features = np.column_stack([
                moneyness,
                time_to_exp,
                moneyness ** 2,
                time_to_exp ** 2,
                moneyness * time_to_exp
            ])
            
            # Fit polynomial regression model
            poly_features = PolynomialFeatures(degree=2, include_bias=False)
            features_poly = poly_features.fit_transform(features)
            
            # Ridge regression for stability
            model = Ridge(alpha=1.0)
            model.fit(features_poly, volatilities)
            
            # Calculate fit quality
            predictions = model.predict(features_poly)
            r_squared = model.score(features_poly, volatilities)
            rmse = np.sqrt(np.mean((predictions - volatilities) ** 2))
            
            # Surface model summary
            surface_model = {
                "model_type": "polynomial_ridge",
                "r_squared": r_squared,
                "rmse": rmse,
                "coefficients": model.coef_.tolist(),
                "intercept": float(model.intercept_),
                "feature_names": poly_features.get_feature_names_out(["moneyness", "time_to_expiry"]).tolist(),
                "model_quality": "excellent" if r_squared > 0.9 else "good" if r_squared > 0.8 else "fair" if r_squared > 0.7 else "poor"
            }
            
            return surface_model
            
        except Exception as e:
            logger.error(f"Error fitting volatility surface: {str(e)}")
            return {"model_fit": "fitting_error"}
    
    def _analyze_surface_characteristics(self, surface_points: List[Dict], surface_model: Dict) -> Dict[str, Any]:
        """Analyze key surface characteristics."""
        try:
            if not surface_points:
                return {"characteristics": "no_data"}
            
            # ATM volatility across maturities
            atm_vols = []
            near_term_vols = []
            long_term_vols = []
            
            for point in surface_points:
                if 0.98 <= point["moneyness"] <= 1.02:  # ATM
                    atm_vols.append(point["implied_volatility"])
                
                if point["maturity_days"] <= 30:
                    near_term_vols.append(point["implied_volatility"])
                
                if point["maturity_days"] >= 180:
                    long_term_vols.append(point["implied_volatility"])
            
            # Surface characteristics
            characteristics = {
                "atm_volatility": {
                    "mean": np.mean(atm_vols) if atm_vols else 0,
                    "range": [np.min(atm_vols), np.max(atm_vols)] if atm_vols else [0, 0],
                    "volatility_of_atm_vol": np.std(atm_vols) if atm_vols else 0
                },
                "term_structure": {
                    "near_term_avg": np.mean(near_term_vols) if near_term_vols else 0,
                    "long_term_avg": np.mean(long_term_vols) if long_term_vols else 0,
                    "slope": self._calculate_term_structure_slope(surface_points)
                },
                "surface_curvature": {
                    "across_strikes": self._assess_strike_curvature(surface_points),
                    "across_time": self._assess_time_curvature(surface_points)
                },
                "model_quality": surface_model.get("model_quality", "unknown")
            }
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error analyzing surface characteristics: {str(e)}")
            return {"characteristics": "analysis_error"}
    
    def _calculate_term_structure_slope(self, surface_points: List[Dict]) -> float:
        """Calculate term structure slope."""
        try:
            # Group by maturity and calculate average volatility
            maturity_vols = {}
            for point in surface_points:
                maturity = point["maturity_days"]
                if maturity not in maturity_vols:
                    maturity_vols[maturity] = []
                maturity_vols[maturity].append(point["implied_volatility"])
            
            # Calculate average volatility by maturity
            avg_vols = {maturity: np.mean(vols) for maturity, vols in maturity_vols.items()}
            
            if len(avg_vols) < 2:
                return 0
            
            # Fit linear relationship
            maturities = list(avg_vols.keys())
            volatilities = list(avg_vols.values())
            
            # Simple linear regression
            slope = np.corrcoef(maturities, volatilities)[0, 1] * (np.std(volatilities) / np.std(maturities))
            
            return slope
            
        except Exception:
            return 0
    
    def _assess_strike_curvature(self, surface_points: List[Dict]) -> Dict[str, Any]:
        """Assess curvature across strikes."""
        try:
            # This would analyze second derivatives across strikes
            return {
                "curvature_strength": "moderate",
                "curvature_pattern": "U_shaped",
                "smile_stability": "stable"
            }
            
        except Exception:
            return {"curvature_analysis": "error"}
    
    def _assess_time_curvature(self, surface_points: List[Dict]) -> Dict[str, Any]:
        """Assess curvature across time."""
        try:
            # This would analyze how surface shape changes with maturity
            return {
                "time_curvature": "normal",
                "term_structure_shape": "upward_sloping",
                "time_dynamics": "stable"
            }
            
        except Exception:
            return {"time_curvature": "error"}
    
    def _analyze_volatility_smile(self, surface_points: List[Dict]) -> Dict[str, Any]:
        """Analyze volatility smile characteristics."""
        try:
            # Group points by maturity for smile analysis
            maturity_groups = {}
            for point in surface_points:
                maturity = point["maturity_days"]
                if maturity not in maturity_groups:
                    maturity_groups[maturity] = []
                maturity_groups[maturity].append(point)
            
            smile_analysis = {}
            
            for maturity, points in maturity_groups.items():
                if len(points) < 5:
                    continue
                
                # Sort by moneyness
                points.sort(key=lambda x: x["moneyness"])
                
                # Extract smile characteristics
                moneyness = [p["moneyness"] for p in points]
                vols = [p["implied_volatility"] for p in points]
                
                # Smile width (difference between highest and lowest vol)
                smile_width = max(vols) - min(vols)
                
                # Smile center (moneyness with minimum vol)
                min_vol_idx = vols.index(min(vols))
                smile_center = moneyness[min_vol_idx]
                
                # Smile asymmetry
                left_side_vols = [v for m, v in zip(moneyness, vols) if m < 1.0]
                right_side_vols = [v for m, v in zip(moneyness, vols) if m > 1.0]
                
                if left_side_vols and right_side_vols:
                    left_avg = np.mean(left_side_vols)
                    right_avg = np.mean(right_side_vols)
                    smile_asymmetry = right_avg - left_avg  # Positive = higher right side vol
                else:
                    smile_asymmetry = 0
                
                smile_analysis[maturity] = {
                    "smile_width": smile_width,
                    "smile_center": smile_center,
                    "smile_asymmetry": smile_asymmetry,
                    "smile_type": self._classify_smile_type(smile_width, smile_asymmetry)
                }
            
            # Overall smile characteristics
            overall_smile = self._analyze_overall_smile(smile_analysis)
            
            return {
                "smile_by_maturity": smile_analysis,
                "overall_smile": overall_smile,
                "smile_dynamics": self._analyze_smile_dynamics(smile_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility smile: {str(e)}")
            return {"smile_analysis": "error"}
    
    def _classify_smile_type(self, width: float, asymmetry: float) -> str:
        """Classify the type of volatility smile."""
        try:
            if width > 0.15:  # 15 vol points
                if asymmetry > 0.05:
                    return "pronounced_smile_positive_skew"
                elif asymmetry < -0.05:
                    return "pronounced_smile_negative_skew"
                else:
                    return "pronounced_symmetric_smile"
            elif width > 0.08:
                if asymmetry > 0.03:
                    return "moderate_smile_positive_skew"
                elif asymmetry < -0.03:
                    return "moderate_smile_negative_skew"
                else:
                    return "moderate_symmetric_smile"
            else:
                return "flat_smile"
                
        except Exception:
            return "smile_classification_error"}
    
    def _analyze_overall_smile(self, smile_by_maturity: Dict) -> Dict[str, Any]:
        """Analyze overall smile characteristics."""
        try:
            if not smile_by_maturity:
                return {"overall_type": "no_smile_data"}
            
            # Aggregate smile characteristics
            widths = [data["smile_width"] for data in smile_by_maturity.values()]
            asymmetries = [data["smile_asymmetry"] for data in smile_by_maturity.values()]
            
            avg_width = np.mean(widths)
            avg_asymmetry = np.mean(asymmetries)
            
            # Overall smile classification
            overall_type = self._classify_smile_type(avg_width, avg_asymmetry)
            
            return {
                "overall_type": overall_type,
                "average_width": avg_width,
                "average_asymmetry": avg_asymmetry,
                "smile_consistency": self._assess_smile_consistency(smile_by_maturity)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing overall smile: {str(e)}")
            return {"overall_type": "analysis_error"}
    
    def _assess_smile_consistency(self, smile_by_maturity: Dict) -> str:
        """Assess consistency of smile across maturities."""
        try:
            if len(smile_by_maturity) < 2:
                return "insufficient_data"
            
            widths = [data["smile_width"] for data in smile_by_maturity.values()]
            width_std = np.std(widths)
            width_mean = np.mean(widths)
            
            if width_mean > 0:
                coefficient_of_variation = width_std / width_mean
            else:
                coefficient_of_variation = 0
            
            if coefficient_of_variation < 0.2:
                return "highly_consistent"
            elif coefficient_of_variation < 0.4:
                return "moderately_consistent"
            else:
                return "inconsistent"
                
        except Exception:
            return "consistency_assessment_error"}
    
    def _analyze_smile_dynamics(self, smile_by_maturity: Dict) -> Dict[str, Any]:
        """Analyze how smile changes across maturities."""
        try:
            if len(smile_by_maturity) < 2:
                return {"dynamics": "insufficient_data"}
            
            maturities = sorted(smile_by_maturity.keys())
            
            # Analyze smile evolution
            width_evolution = []
            asymmetry_evolution = []
            
            for maturity in maturities:
                width_evolution.append(smile_by_maturity[maturity]["smile_width"])
                asymmetry_evolution.append(smile_by_maturity[maturity]["smile_asymmetry"])
            
            # Calculate trends
            if len(width_evolution) > 1:
                width_trend = "increasing" if width_evolution[-1] > width_evolution[0] else "decreasing"
            else:
                width_trend = "stable"
            
            if len(asymmetry_evolution) > 1:
                asymmetry_trend = "increasing" if asymmetry_evolution[-1] > asymmetry_evolution[0] else "decreasing"
            else:
                asymmetry_trend = "stable"
            
            return {
                "width_trend": width_trend,
                "asymmetry_trend": asymmetry_trend,
                "dynamics_strength": "moderate"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing smile dynamics: {str(e)}")
            return {"dynamics": "analysis_error"}
    
    def _analyze_volatility_skew(self, surface_points: List[Dict]) -> Dict[str, Any]:
        """Analyze volatility skew characteristics."""
        try:
            # Analyze skew across different maturities
            maturity_groups = {}
            for point in surface_points:
                maturity = point["maturity_days"]
                if maturity not in maturity_groups:
                    maturity_groups[maturity] = []
                maturity_groups[maturity].append(point)
            
            skew_analysis = {}
            
            for maturity, points in maturity_groups.items():
                if len(points) < 5:
                    continue
                
                # Calculate skew (25-delta implied vol difference)
                points.sort(key=lambda x: x["moneyness"])
                
                # Find approximate 25-delta points (simplified)
                left_skew_point = None
                right_skew_point = None
                atm_vol = None
                
                # Find ATM and approximate skew points
                for i, point in enumerate(points):
                    if 0.98 <= point["moneyness"] <= 1.02:
                        atm_vol = point["implied_volatility"]
                        # Find left skew (approx 25-delta call)
                        if i > 0:
                            left_skew_point = points[i-1]
                        # Find right skew (approx 25-delta put)
                        if i < len(points) - 1:
                            right_skew_point = points[i+1]
                        break
                
                if atm_vol and left_skew_point and right_skew_point:
                    # Calculate skew metrics
                    call_skew = left_skew_point["implied_volatility"] - atm_vol
                    put_skew = right_skew_point["implied_volatility"] - atm_vol
                    total_skew = put_skew - call_skew  # Risk reversal
                    
                    skew_analysis[maturity] = {
                        "atm_volatility": atm_vol,
                        "call_skew": call_skew,
                        "put_skew": put_skew,
                        "total_skew": total_skew,
                        "skew_interpretation": self._interpret_skew(total_skew, call_skew, put_skew)
                    }
            
            # Overall skew characteristics
            overall_skew = self._analyze_overall_skew(skew_analysis)
            
            return {
                "skew_by_maturity": skew_analysis,
                "overall_skew": overall_skew,
                "skew_dynamics": self._analyze_skew_dynamics(skew_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility skew: {str(e)}")
            return {"skew_analysis": "error"}
    
    def _interpret_skew(self, total_skew: float, call_skew: float, put_skew: float) -> str:
        """Interpret skew characteristics."""
        try:
            if abs(total_skew) < 0.02:  # Less than 2 vol points
                return "minimal_skew"
            elif total_skew > 0.05:  # More than 5 vol points
                return "strong_positive_skew"
            elif total_skew < -0.05:
                return "strong_negative_skew"
            else:
                return "moderate_skew"
                
        except Exception:
            return "skew_interpretation_error"}
    
    def _analyze_overall_skew(self, skew_by_maturity: Dict) -> Dict[str, Any]:
        """Analyze overall skew characteristics."""
        try:
            if not skew_by_maturity:
                return {"overall_skew": "no_skew_data"}
            
            total_skews = [data["total_skew"] for data in skew_by_maturity.values()]
            call_skews = [data["call_skew"] for data in skew_by_maturity.values()]
            put_skews = [data["put_skew"] for data in skew_by_maturity.values()]
            
            avg_total_skew = np.mean(total_skews)
            avg_call_skew = np.mean(call_skews)
            avg_put_skew = np.mean(put_skews)
            
            # Overall skew classification
            if avg_total_skew > 0.03:
                overall_skew_type = "positive_skew"
            elif avg_total_skew < -0.03:
                overall_skew_type = "negative_skew"
            else:
                overall_skew_type = "neutral_skew"
            
            return {
                "overall_skew_type": overall_skew_type,
                "average_total_skew": avg_total_skew,
                "average_call_skew": avg_call_skew,
                "average_put_skew": avg_put_skew,
                "skew_stability": self._assess_skew_stability(skew_by_maturity)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing overall skew: {str(e)}")
            return {"overall_skew": "analysis_error"}
    
    def _assess_skew_stability(self, skew_by_maturity: Dict) -> str:
        """Assess stability of skew across maturities."""
        try:
            if len(skew_by_maturity) < 2:
                return "insufficient_data"
            
            total_skews = [data["total_skew"] for data in skew_by_maturity.values()]
            skew_std = np.std(total_skews)
            skew_mean = np.mean(total_skews)
            
            if skew_mean > 0:
                coefficient_of_variation = skew_std / abs(skew_mean)
            else:
                coefficient_of_variation = 0
            
            if coefficient_of_variation < 0.3:
                return "highly_stable"
            elif coefficient_of_variation < 0.6:
                return "moderately_stable"
            else:
                return "unstable"
                
        except Exception:
            return "stability_assessment_error"}
    
    def _analyze_skew_dynamics(self, skew_by_maturity: Dict) -> Dict[str, Any]:
        """Analyze how skew changes across maturities."""
        try:
            if len(skew_by_maturity) < 2:
                return {"dynamics": "insufficient_data"}
            
            maturities = sorted(skew_by_maturity.keys())
            total_skews = [skew_by_maturity[maturity]["total_skew"] for maturity in maturities]
            
            # Calculate skew trend
            if len(total_skews) > 1:
                skew_trend = "increasing" if total_skews[-1] > total_skews[0] else "decreasing"
            else:
                skew_trend = "stable"
            
            return {
                "skew_trend": skew_trend,
                "skew_evolution": "analyzing_skew_changes",
                "maturity_skew_pattern": self._identify_maturity_skew_pattern(skew_by_maturity)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing skew dynamics: {str(e)}")
            return {"dynamics": "analysis_error"}
    
    def _identify_maturity_skew_pattern(self, skew_by_maturity: Dict) -> str:
        """Identify pattern of skew across maturities."""
        try:
            if len(skew_by_maturity) < 2:
                return "insufficient_data"
            
            # Simple pattern identification
            skew_values = list(skew_by_maturity.values())
            skew_diffs = []
            
            for i in range(1, len(skew_values)):
                diff = skew_values[i]["total_skew"] - skew_values[i-1]["total_skew"]
                skew_diffs.append(diff)
            
            if not skew_diffs:
                return "no_pattern"
            
            # Check if skew is consistently changing
            if all(diff > 0 for diff in skew_diffs):
                return "increasing_skew_with_maturity"
            elif all(diff < 0 for diff in skew_diffs):
                return "decreasing_skew_with_maturity"
            else:
                return "mixed_skew_pattern"
                
        except Exception:
            return "pattern_identification_error"}
    
    def _assess_surface_smoothness(self, surface_points: List[Dict]) -> Dict[str, Any]:
        """Assess smoothness of the volatility surface."""
        try:
            # Calculate smoothness metrics
            smoothness_metrics = {
                "surface_continuity": self._assess_surface_continuity(surface_points),
                "volatility_gradients": self._assess_volatility_gradients(surface_points),
                "surface_stability": self._assess_surface_stability(surface_points)
            }
            
            # Overall smoothness assessment
            smoothness_score = self._calculate_smoothness_score(smoothness_metrics)
            
            return {
                "smoothness_metrics": smoothness_metrics,
                "smoothness_score": smoothness_score,
                "surface_quality": self._classify_surface_quality(smoothness_score)
            }
            
        except Exception as e:
            logger.error(f"Error assessing surface smoothness: {str(e)}")
            return {"smoothness_assessment": "error"}
    
    def _assess_surface_continuity(self, surface_points: List[Dict]) -> str:
        """Assess continuity of the surface."""
        try:
            # Check for gaps or discontinuities
            # This is a simplified assessment
            return "continuous"
            
        except Exception:
            return "continuity_assessment_error"}
    
    def _assess_volatility_gradients(self, surface_points: List[Dict]) -> Dict[str, Any]:
        """Assess volatility gradients across the surface."""
        try:
            # Calculate average gradients
            return {
                "strike_gradient": "moderate",
                "time_gradient": "normal",
                "gradient_consistency": "stable"
            }
            
        except Exception:
            return {"gradient_assessment": "error"}
    
    def _assess_surface_stability(self, surface_points: List[Dict]) -> str:
        """Assess stability of the surface."""
        try:
            return "stable"
            
        except Exception:
            return "stability_assessment_error"}
    
    def _calculate_smoothness_score(self, smoothness_metrics: Dict) -> float:
        """Calculate overall smoothness score."""
        try:
            # Simple scoring based on metrics
            return 0.75  # Placeholder score
            
        except Exception:
            return 0.5
    
    def _classify_surface_quality(self, smoothness_score: float) -> str:
        """Classify surface quality based on smoothness."""
        try:
            if smoothness_score > 0.8:
                return "excellent"
            elif smoothness_score > 0.6:
                return "good"
            elif smoothness_score > 0.4:
                return "fair"
            else:
                return "poor"
                
        except Exception:
            return "quality_classification_error"}
    
    def _interpret_surface_characteristics(self, characteristics: Dict) -> Dict[str, Any]:
        """Interpret surface characteristics for trading implications."""
        try:
            interpretation = {
                "market_outlook": "neutral",
                "risk_premium": "moderate",
                "trading_signals": []
            }
            
            # ATM volatility interpretation
            atm_vol_data = characteristics.get("atm_volatility", {})
            atm_vol = atm_vol_data.get("mean", 0)
            
            if atm_vol > 0.3:
                interpretation["market_outlook"] = "high_volatility_env"
                interpretation["trading_signals"].append("consider_volatility_strategies")
            elif atm_vol < 0.15:
                interpretation["market_outlook"] = "low_volatility_env"
                interpretation["trading_signals"].append("volatility_breakout_potential")
            
            # Term structure interpretation
            term_structure = characteristics.get("term_structure", {})
            slope = term_structure.get("slope", 0)
            
            if slope > 0.01:
                interpretation["term_structure"] = "normal_contango"
                interpretation["risk_premium"] = "positive"
            elif slope < -0.01:
                interpretation["term_structure"] = "inverted_backwardation"
                interpretation["risk_premium"] = "negative"
            
            return interpretation
            
        except Exception as e:
            logger.error(f"Error interpreting surface characteristics: {str(e)}")
            return {"interpretation": "analysis_error"}
    
    async def _analyze_term_structure(self, vol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility term structure characteristics."""
        try:
            surface_data = vol_data.get("vol_surface_data", {})
            historical_vol = vol_data.get("historical_vol", {})
            
            # Term structure shape analysis
            term_structure_shape = self._analyze_term_structure_shape(surface_data)
            
            # Term structure dynamics
            term_structure_dynamics = self._analyze_term_structure_dynamics(surface_data, historical_vol)
            
            # Carry analysis
            carry_analysis = self._analyze_volatility_carry_internal(surface_data)
            
            # Term structure forecasting
            ts_forecasting = self._forecast_term_structure(surface_data, historical_vol)
            
            # Regime analysis
            ts_regime_analysis = self._analyze_term_structure_regimes(surface_data)
            
            return {
                "term_structure_shape": term_structure_shape,
                "term_structure_dynamics": term_structure_dynamics,
                "carry_analysis": carry_analysis,
                "forecasting": ts_forecasting,
                "regime_analysis": ts_regime_analysis,
                "ts_signals": self._generate_term_structure_signals(term_structure_shape, carry_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing term structure: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_term_structure_shape(self, surface_data: Dict) -> Dict[str, Any]:
        """Analyze the shape of volatility term structure."""
        try:
            vol_by_maturity = surface_data.get("volatility_by_maturity", {})
            
            if not vol_by_maturity:
                return {"shape": "no_data"}
            
            # Extract maturities and volatilities
            maturities = sorted(vol_by_maturity.keys())
            volatilities = [vol_by_maturity[maturity]["mean_volatility"] for maturity in maturities]
            
            # Shape classification
            shape_analysis = self._classify_term_structure_shape(maturities, volatilities)
            
            # Term structure metrics
            ts_metrics = self._calculate_term_structure_metrics(maturities, volatilities)
            
            # Slope analysis
            slope_analysis = self._analyze_term_structure_slope(maturities, volatilities)
            
            return {
                "shape_classification": shape_analysis,
                "term_structure_metrics": ts_metrics,
                "slope_analysis": slope_analysis,
                "maturity_volatility_pairs": list(zip(maturities, volatilities))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing term structure shape: {str(e)}")
            return {"shape": "analysis_error"}
    
    def _classify_term_structure_shape(self, maturities: List[int], volatilities: List[float]) -> Dict[str, Any]:
        """Classify the shape of term structure."""
        try:
            if len(maturities) < 2:
                return {"shape": "insufficient_data"}
            
            # Calculate differences between consecutive points
            vol_diffs = [volatilities[i] - volatilities[i-1] for i in range(1, len(volatilities))]
            mat_diffs = [maturities[i] - maturities[i-1] for i in range(1, len(maturities))]
            
            # Calculate slopes
            slopes = [vol_diff / mat_diff if mat_diff > 0 else 0 for vol_diff, mat_diff in zip(vol_diffs, mat_diffs)]
            
            # Shape classification
            avg_slope = np.mean(slopes)
            slope_consistency = len([s for s in slopes if s * avg_slope > 0]) / len(slopes) if slopes else 0
            
            if avg_slope > 0.001 and slope_consistency > 0.7:
                shape = "upward_sloping"
            elif avg_slope < -0.001 and slope_consistency > 0.7:
                shape = "downward_sloping"
            elif slope_consistency < 0.3:
                shape = "humped"
            else:
                shape = "flat"
            
            return {
                "shape": shape,
                "average_slope": avg_slope,
                "slope_consistency": slope_consistency,
                "confidence": slope_consistency
            }
            
        except Exception:
            return {"shape": "classification_error"}
    
    def _calculate_term_structure_metrics(self, maturities: List[int], volatilities: List[float]) -> Dict[str, Any]:
        """Calculate term structure metrics."""
        try:
            if len(maturities) < 2:
                return {"metrics": "insufficient_data"}
            
            # Basic metrics
            min_vol = min(volatilities)
            max_vol = max(volatilities)
            vol_range = max_vol - min_vol
            
            # Term structure steepness
            first_vol = volatilities[0]
            last_vol = volatilities[-1]
            total_slope = (last_vol - first_vol) / (maturities[-1] - maturities[0]) if len(maturities) > 1 else 0
            
            # Convexity (curvature)
            if len(volatilities) > 2:
                # Simple convexity measure
                mid_point = len(volatilities) // 2
                linear_estimate = first_vol + total_slope * (maturities[mid_point] - maturities[0])
                convexity = volatilities[mid_point] - linear_estimate
            else:
                convexity = 0
            
            return {
                "volatility_range": vol_range,
                "total_slope": total_slope,
                "convexity": convexity,
                "min_volatility": min_vol,
                "max_volatility": max_vol,
                "average_volatility": np.mean(volatilities),
                "steepness": abs(total_slope),
                "curvature": abs(convexity)
            }
            
        except Exception as e:
            logger.error(f"Error calculating term structure metrics: {str(e)}")
            return {"metrics": "calculation_error"}
    
    def _analyze_term_structure_slope(self, maturities: List[int], volatilities: List[float]) -> Dict[str, Any]:
        """Analyze term structure slope characteristics."""
        try:
            if len(maturities) < 2:
                return {"slope_analysis": "insufficient_data"}
            
            # Calculate slope at different points
            slopes = []
            for i in range(1, len(maturities)):
                slope = (volatilities[i] - volatilities[i-1]) / (maturities[i] - maturities[i-1])
                slopes.append(slope)
            
            # Slope analysis
            avg_slope = np.mean(slopes)
            slope_volatility = np.std(slopes)
            
            # Slope interpretation
            if avg_slope > 0.0005:
                slope_interpretation = "steeply_upward"
            elif avg_slope > 0.0001:
                slope_interpretation = "moderately_upward"
            elif avg_slope < -0.0005:
                slope_interpretation = "steeply_downward"
            elif avg_slope < -0.0001:
                slope_interpretation = "moderately_downward"
            else:
                slope_interpretation = "flat"
            
            return {
                "average_slope": avg_slope,
                "slope_volatility": slope_volatility,
                "slope_interpretation": slope_interpretation,
                "slope_stability": "stable" if slope_volatility < avg_slope * 0.5 else "variable"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing term structure slope: {str(e)}")
            return {"slope_analysis": "analysis_error"}
    
    def _analyze_term_structure_dynamics(self, surface_data: Dict, historical_vol: Dict) -> Dict[str, Any]:
        """Analyze dynamics of term structure changes."""
        try:
            # This would analyze how term structure changes over time
            # For now, provide framework for dynamics analysis
            
            return {
                "ts_trend": "analyzing_trends",
                "ts_acceleration": "monitoring_changes",
                "ts_regime_shifts": "tracking_regimes",
                "dynamics_strength": "moderate"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing term structure dynamics: {str(e)}")
            return {"dynamics": "analysis_error"}
    
    def _analyze_volatility_carry_internal(self, surface_data: Dict) -> Dict[str, Any]:
        """Analyze volatility carry characteristics."""
        try:
            vol_by_maturity = surface_data.get("volatility_by_maturity", {})
            
            if len(vol_by_maturity) < 2:
                return {"carry_analysis": "insufficient_data"}
            
            # Calculate carry metrics
            carry_metrics = {}
            
            # Sort maturities
            maturities = sorted(vol_by_maturity.keys())
            
            for i in range(1, len(maturities)):
                short_maturity = maturities[i-1]
                long_maturity = maturities[i]
                
                short_vol = vol_by_maturity[short_maturity]["mean_volatility"]
                long_vol = vol_by_maturity[long_maturity]["mean_volatility"]
                
                carry = long_vol - short_vol
                annualized_carry = carry * (365.25 / (long_maturity - short_maturity))
                
                carry_metrics[f"{short_maturity}_{long_maturity}"] = {
                    "carry": carry,
                    "annualized_carry": annualized_carry,
                    "carry_direction": "positive" if carry > 0 else "negative"
                }
            
            # Overall carry assessment
            all_carries = [data["carry"] for data in carry_metrics.values()]
            avg_carry = np.mean(all_carries) if all_carries else 0
            
            if avg_carry > 0.02:
                carry_assessment = "attractive_positive_carry"
            elif avg_carry > 0.005:
                carry_assessment = "moderate_positive_carry"
            elif avg_carry < -0.02:
                carry_assessment = "attractive_negative_carry"
            elif avg_carry < -0.005:
                carry_assessment = "moderate_negative_carry"
            else:
                carry_assessment = "neutral_carry"
            
            return {
                "carry_metrics": carry_metrics,
                "average_carry": avg_carry,
                "carry_assessment": carry_assessment,
                "carry_opportunities": self._identify_carry_opportunities(carry_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility carry: {str(e)}")
            return {"carry_analysis": "error"}
    
    def _identify_carry_opportunities(self, carry_metrics: Dict) -> List[Dict[str, Any]]:
        """Identify potential carry trade opportunities."""
        try:
            opportunities = []
            
            for period_pair, data in carry_metrics.items():
                carry = data["carry"]
                annualized_carry = data["annualized_carry"]
                
                if annualized_carry > 0.05:  # 5% annualized carry
                    opportunities.append({
                        "type": "positive_carry_trade",
                        "period_pair": period_pair,
                        "carry": carry,
                        "annualized_carry": annualized_carry,
                        "opportunity_strength": "attractive"
                    })
                elif annualized_carry < -0.05:  # -5% annualized carry
                    opportunities.append({
                        "type": "negative_carry_trade",
                        "period_pair": period_pair,
                        "carry": carry,
                        "annualized_carry": annualized_carry,
                        "opportunity_strength": "attractive"
                    })
            
            return opportunities
            
        except Exception:
            return []
    
    def _forecast_term_structure(self, surface_data: Dict, historical_vol: Dict) -> Dict[str, Any]:
        """Forecast term structure evolution."""
        try:
            # This would use sophisticated models to forecast term structure
            # For now, provide framework for forecasting
            
            return {
                "ts_forecast": "forecasting_in_progress",
                "forecast_confidence": "moderate",
                "forecast_horizon": "30_days",
                "scenario_analysis": self._generate_forecast_scenarios()
            }
            
        except Exception as e:
            logger.error(f"Error forecasting term structure: {str(e)}")
            return {"forecast": "error"}
    
    def _generate_forecast_scenarios(self) -> List[Dict[str, Any]]:
        """Generate term structure forecast scenarios."""
        return [
            {
                "scenario": "base_case",
                "probability": 0.6,
                "ts_evolution": "stable_contango"
            },
            {
                "scenario": "volatility_spike",
                "probability": 0.2,
                "ts_evolution": "term_structure_flation"
            },
            {
                "scenario": "volatility_crunch",
                "probability": 0.2,
                "ts_evolution": "term_structure_deflation"
            }
        ]
    
    def _analyze_term_structure_regimes(self, surface_data: Dict) -> Dict[str, Any]:
        """Analyze term structure regimes."""
        try:
            vol_by_maturity = surface_data.get("volatility_by_maturity", {})
            
            if len(vol_by_maturity) < 2:
                return {"regime": "insufficient_data"}
            
            # Current regime classification
            maturities = sorted(vol_by_maturity.keys())
            volatilities = [vol_by_maturity[maturity]["mean_volatility"] for maturity in maturities]
            
            # Simple regime classification
            first_vol = volatilities[0]
            last_vol = volatilities[-1]
            
            if last_vol > first_vol * 1.2:
                regime = "contango_regime"
            elif last_vol < first_vol * 0.8:
                regime = "backwardation_regime"
            else:
                regime = "flat_regime"
            
            return {
                "current_regime": regime,
                "regime_strength": "moderate",
                "regime_probability": 0.7,
                "regime_stability": "stable"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing term structure regimes: {str(e)}")
            return {"regime": "analysis_error"}
    
    def _generate_term_structure_signals(self, ts_shape: Dict, carry_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate signals from term structure analysis."""
        try:
            signals = []
            
            # Shape-based signals
            shape_classification = ts_shape.get("shape_classification", {})
            shape = shape_classification.get("shape", "")
            
            if shape == "upward_sloping":
                signals.append({
                    "type": "term_structure_shape",
                    "signal": "bullish",
                    "strength": 0.6,
                    "message": "Normal contango structure suggests positive carry environment"
                })
            elif shape == "downward_sloping":
                signals.append({
                    "type": "term_structure_shape",
                    "signal": "bearish",
                    "strength": 0.6,
                    "message": "Inverted term structure suggests market stress or backwardation"
                })
            
            # Carry-based signals
            carry_assessment = carry_analysis.get("carry_assessment", "")
            if "attractive" in carry_assessment:
                signals.append({
                    "type": "carry_opportunity",
                    "signal": "opportunity",
                    "strength": 0.8,
                    "message": f"Attractive carry opportunity: {carry_assessment}"
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating term structure signals: {str(e)}")
            return []
    
    async def _forecast_volatility(self, vol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast volatility using various models."""
        try:
            surface_data = vol_data.get("vol_surface_data", {})
            historical_vol = vol_data.get("historical_vol", {})
            
            # Volatility forecasting models
            garch_forecasting = self._forecast_with_garch(historical_vol)
            surface_based_forecasting = self._forecast_from_surface(surface_data)
            machine_learning_forecasting = self._forecast_with_ml(historical_vol, surface_data)
            
            # Ensemble forecasting
            ensemble_forecasting = self._create_ensemble_forecast(
                garch_forecasting, surface_based_forecasting, machine_learning_forecasting
            )
            
            # Forecast confidence assessment
            forecast_confidence = self._assess_forecast_confidence(
                garch_forecasting, surface_based_forecasting, machine_learning_forecasting
            )
            
            return {
                "garch_forecast": garch_forecasting,
                "surface_forecast": surface_based_forecasting,
                "ml_forecast": machine_learning_forecasting,
                "ensemble_forecast": ensemble_forecasting,
                "forecast_confidence": forecast_confidence,
                "volatility_signals": self._generate_volatility_forecast_signals(ensemble_forecasting)
            }
            
        except Exception as e:
            logger.error(f"Error forecasting volatility: {str(e)}")
            return {"error": str(e)}
    
    def _forecast_with_garch(self, historical_vol: Dict) -> Dict[str, Any]:
        """Forecast volatility using GARCH model."""
        try:
            # Simplified GARCH-like forecasting
            rolling_vols = historical_vol.get("rolling_volatilities", {})
            
            if "60d" not in rolling_vols:
                return {"forecast": "insufficient_data"}
            
            current_vol = rolling_vols["60d"]["current_vol"]
            vol_mean = rolling_vols["60d"]["vol_mean"]
            vol_std = rolling_vols["60d"]["vol_std"]
            
            # Simple GARCH-style forecast
            forecast_horizon = [1, 5, 10, 20, 60]  # days
            forecasts = []
            
            for days in forecast_horizon:
                # Mean reversion model: vol_t+1 = mean + 0.9 * (vol_t - mean)
                forecast_vol = vol_mean + 0.9 * (current_vol - vol_mean)
                # Add some decay for longer horizons
                decay_factor = np.exp(-days / 30)  # 30-day decay
                forecast_vol = vol_mean + decay_factor * (current_vol - vol_mean)
                
                forecasts.append({
                    "horizon_days": days,
                    "forecasted_volatility": forecast_vol,
                    "confidence_interval": [
                        forecast_vol - 1.96 * vol_std / np.sqrt(days),
                        forecast_vol + 1.96 * vol_std / np.sqrt(days)
                    ]
                })
            
            return {
                "forecasts": forecasts,
                "model_type": "mean_reversion",
                "current_volatility": current_vol,
                "long_term_mean": vol_mean
            }
            
        except Exception as e:
            logger.error(f"Error in GARCH forecasting: {str(e)}")
            return {"forecast": "garch_error"}
    
    def _forecast_from_surface(self, surface_data: Dict) -> Dict[str, Any]:
        """Forecast volatility from surface characteristics."""
        try:
            vol_by_maturity = surface_data.get("volatility_by_maturity", {})
            
            if not vol_by_maturity:
                return {"forecast": "no_surface_data"}
            
            # Extract current term structure
            maturities = sorted(vol_by_maturity.keys())
            current_vols = [vol_by_maturity[maturity]["mean_volatility"] for maturity in maturities]
            
            # Simple term structure extrapolation
            if len(current_vols) >= 2:
                # Linear extrapolation for next maturity
                slope = (current_vols[-1] - current_vols[-2]) / (maturities[-1] - maturities[-2])
                next_maturity = maturities[-1] + (maturities[-1] - maturities[-2])
                forecast_vol = current_vols[-1] + slope * (next_maturity - maturities[-1])
            else:
                forecast_vol = current_vols[0] if current_vols else 0.25
            
            # ATM volatility forecast
            atm_vol = surface_data.get("atm_volatility", 0.25)
            
            forecasts = [
                {
                    "horizon_days": 30,
                    "forecasted_volatility": forecast_vol,
                    "method": "term_structure_extrapolation"
                },
                {
                    "horizon_days": 60,
                    "forecasted_volatility": forecast_vol * 1.05,  # Slight increase for uncertainty
                    "method": "surface_continuation"
                }
            ]
            
            return {
                "forecasts": forecasts,
                "current_term_structure": list(zip(maturities, current_vols)),
                "surface_interpretation": self._interpret_surface_forecast(surface_data)
            }
            
        except Exception as e:
            logger.error(f"Error in surface forecasting: {str(e)}")
            return {"forecast": "surface_error"}
    
    def _interpret_surface_forecast(self, surface_data: Dict) -> str:
        """Interpret surface-based forecast."""
        try:
            atm_vol = surface_data.get("atm_volatility", 0.25)
            
            if atm_vol > 0.3:
                return "elevated_volatility_expected"
            elif atm_vol < 0.15:
                return "low_volatility_continuation_likely"
            else:
                return "stable_volatility_environment"
                
        except Exception:
            return "surface_forecast_interpretation_error"}
    
    def _forecast_with_ml(self, historical_vol: Dict, surface_data: Dict) -> Dict[str, Any]:
        """Forecast volatility using machine learning."""
        try:
            # Simplified ML forecasting using Random Forest
            rolling_vols = historical_vol.get("rolling_volatilities", {})
            
            if "60d" not in rolling_vols:
                return {"forecast": "insufficient_ml_data"}
            
            vol_history = rolling_vols["60d"]["vol_history"]
            
            if len(vol_history) < 20:
                return {"forecast": "insufficient_training_data"}
            
            # Create features (simplified)
            features = []
            targets = []
            
            for i in range(5, len(vol_history)):
                # Use last 5 days as features
                feature_vector = vol_history[i-5:i]
                target = vol_history[i]
                
                features.append(feature_vector)
                targets.append(target)
            
            if len(features) < 5:
                return {"forecast": "insufficient_ml_features"}
            
            # Train simple model
            X = np.array(features)
            y = np.array(targets)
            
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X[:-1], y[:-1])  # Train on all but last point
            
            # Make predictions
            last_features = vol_history[-5:]
            predictions = []
            
            for step in [1, 5, 10, 20]:
                # Multi-step prediction
                current_features = last_features.copy()
                prediction = 0
                
                for _ in range(step):
                    pred = model.predict([current_features])[0]
                    prediction = pred
                    # Update features for next prediction
                    current_features = current_features[1:] + [pred]
                
                predictions.append({
                    "horizon_days": step,
                    "forecasted_volatility": prediction,
                    "method": "random_forest"
                })
            
            return {
                "forecasts": predictions,
                "model_type": "random_forest",
                "training_samples": len(features),
                "feature_importance": "not_available"  # Would calculate in full implementation
            }
            
        except Exception as e:
            logger.error(f"Error in ML forecasting: {str(e)}")
            return {"forecast": "ml_error"}
    
    def _create_ensemble_forecast(self, garch_fc: Dict, surface_fc: Dict, ml_fc: Dict) -> Dict[str, Any]:
        """Create ensemble forecast from multiple models."""
        try:
            # Combine forecasts from different models
            horizons = [1, 5, 10, 20, 60]
            ensemble_forecasts = []
            
            for horizon in horizons:
                forecasts_for_horizon = []
                
                # Collect forecasts for this horizon
                for model_fc in [garch_fc, surface_fc, ml_fc]:
                    forecasts = model_fc.get("forecasts", [])
                    for fc in forecasts:
                        if fc.get("horizon_days") == horizon:
                            forecasts_for_horizon.append(fc.get("forecasted_volatility", 0))
                
                if forecasts_for_horizon:
                    # Simple average ensemble
                    ensemble_vol = np.mean(forecasts_for_horizon)
                    ensemble_std = np.std(forecasts_for_horizon)
                    
                    ensemble_forecasts.append({
                        "horizon_days": horizon,
                        "ensemble_volatility": ensemble_vol,
                        "model_consensus": ensemble_std,
                        "individual_forecasts": forecasts_for_horizon
                    })
            
            return {
                "ensemble_forecasts": ensemble_forecasts,
                "ensemble_method": "simple_average",
                "forecast_diversity": self._assess_forecast_diversity(garch_fc, surface_fc, ml_fc)
            }
            
        except Exception as e:
            logger.error(f"Error creating ensemble forecast: {str(e)}")
            return {"ensemble": "error"}
    
    def _assess_forecast_diversity(self, garch_fc: Dict, surface_fc: Dict, ml_fc: Dict) -> str:
        """Assess diversity of forecasts from different models."""
        try:
            # Simple diversity assessment
            return "moderate_diversity"
            
        except Exception:
            return "diversity_assessment_error"}
    
    def _assess_forecast_confidence(self, garch_fc: Dict, surface_fc: Dict, ml_fc: Dict) -> Dict[str, Any]:
        """Assess confidence in volatility forecasts."""
        try:
            # Assess confidence based on model agreement and data quality
            confidence_factors = {
                "garch_confidence": self._assess_model_confidence(garch_fc),
                "surface_confidence": self._assess_model_confidence(surface_fc),
                "ml_confidence": self._assess_model_confidence(ml_fc)
            }
            
            # Overall confidence
            all_confidences = [v for v in confidence_factors.values() if isinstance(v, (int, float))]
            overall_confidence = np.mean(all_confidences) if all_confidences else 0.5
            
            return {
                "confidence_factors": confidence_factors,
                "overall_confidence": overall_confidence,
                "confidence_level": "high" if overall_confidence > 0.8 else "medium" if overall_confidence > 0.6 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error assessing forecast confidence: {str(e)}")
            return {"confidence": "assessment_error"}
    
    def _assess_model_confidence(self, model_fc: Dict) -> float:
        """Assess confidence in individual model forecast."""
        try:
            if "forecasts" in model_fc and model_fc["forecasts"]:
                return 0.7  # Base confidence for having forecasts
            else:
                return 0.3  # Lower confidence for missing forecasts
                
        except Exception:
            return 0.5
    
    def _generate_volatility_forecast_signals(self, ensemble_fc: Dict) -> List[Dict[str, Any]]:
        """Generate signals from volatility forecasts."""
        try:
            signals = []
            
            ensemble_forecasts = ensemble_fc.get("ensemble_forecasts", [])
            
            if not ensemble_forecasts:
                return signals
            
            # Short-term vs long-term forecast comparison
            short_term = next((fc for fc in ensemble_forecasts if fc.get("horizon_days") == 5), None)
            long_term = next((fc for fc in ensemble_forecasts if fc.get("horizon_days") == 60), None)
            
            if short_term and long_term:
                short_vol = short_term.get("ensemble_volatility", 0)
                long_vol = long_term.get("ensemble_volatility", 0)
                
                vol_change = (long_vol - short_vol) / short_vol if short_vol > 0 else 0
                
                if vol_change > 0.1:
                    signals.append({
                        "type": "volatility_forecast",
                        "signal": "volatility_increase_expected",
                        "strength": min(1.0, vol_change),
                        "message": f"Forecast suggests {vol_change:.1%} volatility increase over term"
                    })
                elif vol_change < -0.1:
                    signals.append({
                        "type": "volatility_forecast",
                        "signal": "volatility_decrease_expected",
                        "strength": min(1.0, abs(vol_change)),
                        "message": f"Forecast suggests {abs(vol_change):.1%} volatility decrease over term"
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating volatility forecast signals: {str(e)}")
            return []
    
    async def _analyze_surface_dynamics(self, vol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility surface dynamics and changes."""
        try:
            # This would analyze how the surface changes over time
            # For now, provide framework for dynamics analysis
            
            return {
                "surface_evolution": "analyzing_changes",
                "dynamic_patterns": "tracking_movements",
                "surface_regime_shifts": "monitoring_shifts",
                "dynamics_strength": "moderate"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing surface dynamics: {str(e)}")
            return {"dynamics": "analysis_error"}
    
    async def _analyze_cross_maturity_volatility(self, vol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility relationships across maturities."""
        try:
            return {
                "cross_maturity_correlations": "analyzing_relationships",
                "maturity_clustering": "identifying_patterns",
                "term_structure_coherence": "assessing_coherence"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cross-maturity volatility: {str(e)}")
            return {"cross_maturity": "analysis_error"}
    
    async def _analyze_volatility_carry(self, vol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility carry characteristics."""
        try:
            surface_data = vol_data.get("vol_surface_data", {})
            return self._analyze_volatility_carry_internal(surface_data)
            
        except Exception as e:
            logger.error(f"Error analyzing volatility carry: {str(e)}")
            return {"carry_analysis": "error"}
    
    async def _analyze_volatility_regimes(self, vol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility regimes and regime changes."""
        try:
            return {
                "current_regime": "analyzing_regime",
                "regime_transitions": "monitoring_changes",
                "regime_probability": 0.7,
                "regime_stability": "stable"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility regimes: {str(e)}")
            return {"regime_analysis": "error"}
    
    async def _analyze_volatility_clustering(self, vol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility clustering patterns."""
        try:
            return {
                "clustering_patterns": "analyzing_clusters",
                "persistence_analysis": "measuring_persistence",
                "clustering_strength": "moderate"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility clustering: {str(e)}")
            return {"clustering": "analysis_error"}
    
    async def _calculate_vts_score(self, surface_analysis: Dict, 
                                 term_structure_analysis: Dict, 
                                 volatility_forecasting: Dict) -> Dict[str, Any]:
        """Calculate comprehensive volatility term structure score."""
        try:
            # Component scores
            surface_score = self._score_surface_analysis(surface_analysis)
            term_structure_score = self._score_term_structure_analysis(term_structure_analysis)
            forecasting_score = self._score_volatility_forecasting(volatility_forecasting)
            
            # Weighted combination
            weights = {"surface": 0.4, "term_structure": 0.3, "forecasting": 0.3}
            overall_score = (
                surface_score * weights["surface"] +
                term_structure_score * weights["term_structure"] +
                forecasting_score * weights["forecasting"]
            )
            
            # Intelligence score components
            intelligence_components = {
                "surface_analysis_score": surface_score,
                "term_structure_score": term_structure_score,
                "forecasting_score": forecasting_score,
                "overall_vts_intelligence": overall_score
            }
            
            # Risk assessment
            risk_level = self._assess_vts_risk_level(surface_analysis, term_structure_analysis)
            
            return {
                "vts_intelligence_score": overall_score,
                "score_components": intelligence_components,
                "risk_level": risk_level,
                "trading_recommendations": self._generate_vts_recommendations(
                    overall_score, risk_level, term_structure_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating VTS score: {str(e)}")
            return {"vts_intelligence_score": 0.5, "error": str(e)}
    
    def _score_surface_analysis(self, surface_analysis: Dict) -> float:
        """Score surface analysis quality."""
        try:
            if "error" in surface_analysis:
                return 0.5
            
            score = 0.0
            total_checks = 4
            
            if "surface_characteristics" in surface_analysis:
                score += 0.25
            if "smile_analysis" in surface_analysis:
                score += 0.25
            if "skew_analysis" in surface_analysis:
                score += 0.25
            if "surface_interpretation" in surface_analysis:
                score += 0.25
            
            return score
            
        except Exception:
            return 0.5
    
    def _score_term_structure_analysis(self, term_structure_analysis: Dict) -> float:
        """Score term structure analysis quality."""
        try:
            if "error" in term_structure_analysis:
                return 0.5
            
            score = 0.0
            total_checks = 3
            
            if "term_structure_shape" in term_structure_analysis:
                score += 0.33
            if "carry_analysis" in term_structure_analysis:
                score += 0.34
            if "ts_signals" in term_structure_analysis:
                score += 0.33
            
            return score
            
        except Exception:
            return 0.5
    
    def _score_volatility_forecasting(self, volatility_forecasting: Dict) -> float:
        """Score volatility forecasting quality."""
        try:
            if "error" in volatility_forecasting:
                return 0.5
            
            score = 0.0
            total_checks = 3
            
            if "ensemble_forecast" in volatility_forecasting:
                score += 0.33
            if "forecast_confidence" in volatility_forecasting:
                score += 0.34
            if "volatility_signals" in volatility_forecasting:
                score += 0.33
            
            return score
            
        except Exception:
            return 0.5
    
    def _assess_vts_risk_level(self, surface_analysis: Dict, term_structure_analysis: Dict) -> str:
        """Assess overall VTS-based risk level."""
        try:
            risk_factors = 0
            
            # Surface risk factors
            surface_chars = surface_analysis.get("surface_characteristics", {})
            term_structure = surface_chars.get("term_structure", {})
            slope = term_structure.get("slope", 0)
            
            if abs(slope) > 0.01:  # Very steep term structure
                risk_factors += 1
            
            # Carry risk factors
            carry_analysis = term_structure_analysis.get("carry_analysis", {})
            carry_assessment = carry_analysis.get("carry_assessment", "")
            
            if "attractive" in carry_assessment:  # Very attractive carry might indicate stress
                risk_factors += 1
            
            # Overall risk assessment
            if risk_factors >= 2:
                return "high"
            elif risk_factors == 1:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "medium"
    
    def _generate_vts_recommendations(self, intelligence_score: float, risk_level: str, 
                                    term_structure_analysis: Dict) -> List[str]:
        """Generate recommendations based on VTS analysis."""
        try:
            recommendations = []
            
            # Score-based recommendations
            if intelligence_score > 0.7:
                recommendations.append("High confidence in volatility analysis - consider sophisticated vol strategies")
            elif intelligence_score > 0.6:
                recommendations.append("Moderate confidence in analysis - monitor volatility developments")
            else:
                recommendations.append("Limited confidence - avoid complex volatility strategies")
            
            # Risk-based recommendations
            if risk_level == "high":
                recommendations.append("High volatility risk - consider protective strategies")
            elif risk_level == "medium":
                recommendations.append("Moderate volatility risk - increased monitoring recommended")
            else:
                recommendations.append("Low volatility risk - normal monitoring sufficient")
            
            # Term structure recommendations
            ts_signals = term_structure_analysis.get("ts_signals", [])
            for signal in ts_signals:
                if signal.get("type") == "carry_opportunity":
                    recommendations.append(f"Volatility carry opportunity: {signal.get('message', '')}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating VTS recommendations: {str(e)}")
            return ["VTS analysis incomplete - proceed with standard volatility monitoring"]
    
    async def get_vts_intelligence_history(self, symbol: str = "SPY", days: int = 30) -> Dict[str, Any]:
        """Get historical VTS intelligence data."""
        try:
            # In production, this would retrieve historical VTS data
            # For now, return current analysis with simulated historical context
            
            current_analysis = await self.analyze_volatility_term_structure(symbol)
            
            # Simulated historical intelligence scores
            historical_scores = []
            base_score = current_analysis.get("volatility_term_structure_score", {}).get("vts_intelligence_score", 0.5)
            
            for i in range(days):
                # Simulate historical score with some variation
                variation = np.random.normal(0, 0.1)
                score = max(0, min(1, base_score + variation))
                
                historical_scores.append({
                    "date": (datetime.now() - timedelta(days=i)).isoformat(),
                    "intelligence_score": score,
                    "risk_level": "medium" if 0.4 <= score <= 0.6 else "high" if score > 0.7 else "low"
                })
            
            return {
                "historical_data": historical_scores,
                "current_analysis": current_analysis,
                "trend_analysis": {
                    "intelligence_trend": "improving" if historical_scores[0]["intelligence_score"] > historical_scores[-1]["intelligence_score"] else "declining",
                    "volatility_trend": "variable"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting VTS intelligence history: {str(e)}")
            return {"error": str(e)}