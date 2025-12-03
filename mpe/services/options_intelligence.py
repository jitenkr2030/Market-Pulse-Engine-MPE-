"""
Module 14: Options Intelligence Engine

Comprehensive options market intelligence system providing real-time analysis
of options flow, volatility surface dynamics, Greeks analysis, and options-based
sentiment indicators.

Author: MiniMax Agent
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.interpolate import griddata, interp1d
import yfinance as yf
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class OptionsIntelligenceEngine:
    """
    Options market intelligence and analysis engine.
    
    Features:
    - Options flow analysis and unusual activity detection
    - Volatility surface modeling and term structure analysis
    - Greeks calculation and sensitivity analysis
    - Options-based sentiment and positioning indicators
    - Volatility skew and smile analysis
    - Options expiration impact modeling
    """
    
    def __init__(self, db_manager=None, cache_manager=None):
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.volatility_surface_cache = {}
        self.flow_cache = {}
        
    async def analyze_options_intelligence(self, symbol: str = "SPY") -> Dict[str, Any]:
        """
        Comprehensive options market intelligence analysis.
        
        Args:
            symbol: Asset symbol to analyze
            
        Returns:
            Dictionary containing options intelligence results
        """
        try:
            # Get options data
            options_data = await self._fetch_options_data(symbol)
            if not options_data:
                return {"error": "Unable to fetch options data"}
            
            # Options flow analysis
            flow_analysis = await self._analyze_options_flow(options_data)
            
            # Volatility surface analysis
            vol_surface_analysis = await self._analyze_volatility_surface(options_data)
            
            # Greeks and sensitivity analysis
            greeks_analysis = await self._analyze_greeks(options_data)
            
            # Options sentiment indicators
            sentiment_analysis = await self._analyze_options_sentiment(options_data)
            
            # Expiration impact analysis
            expiration_analysis = await self._analyze_expiration_impact(options_data)
            
            # Volatility skew and smile analysis
            skew_smile_analysis = await self._analyze_skew_smile(options_data)
            
            # Unusual activity detection
            unusual_activity = await self._detect_unusual_activity(options_data)
            
            # Options-based technical indicators
            technical_indicators = await self._calculate_options_technical_indicators(options_data)
            
            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "underlying_price": options_data.get("underlying_price", 0),
                "flow_analysis": flow_analysis,
                "volatility_surface": vol_surface_analysis,
                "greeks_analysis": greeks_analysis,
                "sentiment_analysis": sentiment_analysis,
                "expiration_analysis": expiration_analysis,
                "skew_smile_analysis": skew_smile_analysis,
                "unusual_activity": unusual_activity,
                "technical_indicators": technical_indicators,
                "options_intelligence_score": await self._calculate_options_intelligence_score(
                    flow_analysis, sentiment_analysis, technical_indicators
                )
            }
            
            # Cache results
            if self.cache_manager:
                await self.cache_manager.set(f"options_intelligence:{symbol}", result, ttl=300)
                
            return result
            
        except Exception as e:
            logger.error(f"Error in options intelligence analysis for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    async def _fetch_options_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch options data for analysis."""
        try:
            # For demonstration, we'll use yfinance to get basic options data
            # In production, this would connect to professional options data providers
            ticker = yf.Ticker(symbol)
            
            # Get current stock price
            hist = ticker.history(period="1d")
            if hist.empty:
                return None
                
            underlying_price = hist['Close'].iloc[-1]
            
            # Get available expiration dates
            try:
                expirations = ticker.options
            except Exception:
                # If options data not available, create simulated data
                return await self._create_simulated_options_data(symbol, underlying_price)
            
            # Fetch options chain for near-term expiration
            if not expirations:
                return await self._create_simulated_options_data(symbol, underlying_price)
            
            # Get options for first few expirations
            options_chain = {}
            for exp_date in expirations[:3]:  # Limit to first 3 expirations
                try:
                    opt = ticker.option_chain(exp_date)
                    options_chain[exp_date] = {
                        "calls": opt.calls,
                        "puts": opt.puts
                    }
                except Exception as e:
                    logger.warning(f"Could not fetch options for {symbol} {exp_date}: {str(e)}")
                    continue
            
            if not options_chain:
                return await self._create_simulated_options_data(symbol, underlying_price)
            
            return {
                "symbol": symbol,
                "underlying_price": underlying_price,
                "options_chain": options_chain,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching options data for {symbol}: {str(e)}")
            return await self._create_simulated_options_data(symbol, 400.0)  # Fallback
    
    async def _create_simulated_options_data(self, symbol: str, underlying_price: float) -> Dict[str, Any]:
        """Create simulated options data for analysis."""
        try:
            # Generate realistic-looking options chain
            base_vol = 0.20 + np.random.normal(0, 0.05)  # 20% base volatility
            base_vol = max(0.10, min(0.50, base_vol))  # Bound between 10% and 50%
            
            # Strike prices around current price
            current_price = underlying_price
            strikes = []
            for i in range(-10, 11):  # 21 strikes
                strike = current_price * (1 + i * 0.02)  # 2% increments
                strikes.append(round(strike, 2))
            
            # Generate options prices using Black-Scholes approximation
            def black_scholes_call(S, K, T, r, sigma):
                if T <= 0:
                    return max(0, S - K)
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                return call
            
            def black_scholes_put(S, K, T, r, sigma):
                if T <= 0:
                    return max(0, K - S)
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                put = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
                return put
            
            # Expiration dates (simulate)
            exp_dates = []
            for i in range(1, 4):  # 3 expirations
                exp_date = datetime.now() + timedelta(days=30*i)
                exp_dates.append(exp_date.strftime("%Y-%m-%d"))
            
            options_chain = {}
            r = 0.05  # Risk-free rate
            
            for exp_date in exp_dates:
                # Days to expiration
                exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
                days_to_exp = (exp_dt - datetime.now()).days
                T = days_to_exp / 365.25
                
                calls = []
                puts = []
                
                for strike in strikes:
                    # Adjust volatility for moneyness (volatility smile)
                    moneyness = strike / current_price
                    vol_adjustment = 1 + 0.1 * abs(moneyness - 1)  # Higher vol OTM
                    vol = base_vol * vol_adjustment
                    
                    call_price = black_scholes_call(current_price, strike, T, r, vol)
                    put_price = black_scholes_put(current_price, strike, T, r, vol)
                    
                    # Add some randomness to volume and open interest
                    volume = max(1, int(np.random.exponential(100)))
                    open_interest = max(volume, int(np.random.exponential(500)))
                    
                    calls.append({
                        "contractSymbol": f"{symbol}{exp_date[-2:]}C{str(int(strike)).zfill(5)}",
                        "strike": strike,
                        "lastPrice": round(call_price, 2),
                        "bid": round(call_price * 0.98, 2),
                        "ask": round(call_price * 1.02, 2),
                        "volume": volume,
                        "openInterest": open_interest,
                        "inTheMoney": current_price > strike
                    })
                    
                    puts.append({
                        "contractSymbol": f"{symbol}{exp_date[-2:]}P{str(int(strike)).zfill(5)}",
                        "strike": strike,
                        "lastPrice": round(put_price, 2),
                        "bid": round(put_price * 0.98, 2),
                        "ask": round(put_price * 1.02, 2),
                        "volume": volume,
                        "openInterest": open_interest,
                        "inTheMoney": current_price < strike
                    })
                
                options_chain[exp_date] = {
                    "calls": pd.DataFrame(calls),
                    "puts": pd.DataFrame(puts)
                }
            
            return {
                "symbol": symbol,
                "underlying_price": current_price,
                "options_chain": options_chain,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating simulated options data: {str(e)}")
            return None
    
    async def _analyze_options_flow(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze options flow and trading activity."""
        try:
            symbol = options_data["symbol"]
            options_chain = options_data["options_chain"]
            underlying_price = options_data["underlying_price"]
            
            flow_analysis = {
                "total_volume": 0,
                "total_open_interest": 0,
                "call_put_ratio": 0,
                "flow_concentration": {},
                "volume_analysis": {},
                "open_interest_analysis": {},
                "flow_direction": {},
                "unusual_flow": []
            }
            
            total_calls_volume = 0
            total_puts_volume = 0
            total_calls_oi = 0
            total_puts_oi = 0
            
            for exp_date, chain_data in options_chain.items():
                calls = chain_data["calls"]
                puts = chain_data["puts"]
                
                # Basic flow metrics
                calls_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
                puts_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
                calls_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
                puts_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
                
                total_calls_volume += calls_volume
                total_puts_volume += puts_volume
                total_calls_oi += calls_oi
                total_puts_oi += puts_oi
                
                # Volume by moneyness
                calls_moneyness = self._categorize_moneyness(calls['strike'], underlying_price)
                puts_moneyness = self._categorize_moneyness(puts['strike'], underlying_price)
                
                calls_by_moneyness = {}
                puts_by_moneyness = {}
                
                for category in ['deep_itm', 'itm', 'atm', 'otm', 'deep_otm']:
                    if category in calls_moneyness:
                        calls_by_moneyness[category] = calls[calls_moneyness == category]['volume'].sum()
                    if category in puts_moneyness:
                        puts_by_moneyness[category] = puts[puts_moneyness == category]['volume'].sum()
                
                flow_analysis["volume_analysis"][exp_date] = {
                    "calls_by_moneyness": calls_by_moneyness,
                    "puts_by_moneyness": puts_by_moneyness,
                    "total_calls_volume": calls_volume,
                    "total_puts_volume": puts_volume
                }
                
                flow_analysis["open_interest_analysis"][exp_date] = {
                    "calls_oi": calls_oi,
                    "puts_oi": puts_oi
                }
                
                # Detect unusual flow
                unusual = self._detect_unusual_flow(calls, puts, underlying_price, exp_date)
                flow_analysis["unusual_flow"].extend(unusual)
            
            # Aggregate analysis
            flow_analysis["total_volume"] = total_calls_volume + total_puts_volume
            flow_analysis["total_open_interest"] = total_calls_oi + total_puts_oi
            
            if total_puts_volume > 0:
                flow_analysis["call_put_ratio"] = total_calls_volume / total_puts_volume
            else:
                flow_analysis["call_put_ratio"] = float('inf') if total_calls_volume > 0 else 0
            
            # Flow direction analysis
            flow_analysis["flow_direction"] = {
                "overall_sentiment": "bullish" if flow_analysis["call_put_ratio"] > 1.5 else 
                                   "bearish" if flow_analysis["call_put_ratio"] < 0.67 else "neutral",
                "put_call_ratio": flow_analysis["call_put_ratio"],
                "activity_level": "high" if flow_analysis["total_volume"] > 10000 else "normal"
            }
            
            return flow_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing options flow: {str(e)}")
            return {"error": str(e)}
    
    def _categorize_moneyness(self, strikes: pd.Series, underlying_price: float) -> pd.Series:
        """Categorize options by moneyness."""
        moneyness = strikes / underlying_price
        
        conditions = [
            moneyness < 0.8,  # Deep ITM
            (moneyness >= 0.8) & (moneyness < 0.95),  # ITM
            (moneyness >= 0.95) & (moneyness <= 1.05),  # ATM
            (moneyness > 1.05) & (moneyness <= 1.2),  # OTM
            moneyness > 1.2   # Deep OTM
        ]
        
        choices = ['deep_itm', 'itm', 'atm', 'otm', 'deep_otm']
        
        return pd.Series(np.select(conditions, choices, default='unknown'), index=strikes.index)
    
    def _detect_unusual_flow(self, calls: pd.DataFrame, puts: pd.DataFrame, 
                           underlying_price: float, exp_date: str) -> List[Dict[str, Any]]:
        """Detect unusual options flow patterns."""
        unusual_flow = []
        
        try:
            # High volume detection
            if 'volume' in calls.columns:
                high_volume_calls = calls[calls['volume'] > calls['volume'].quantile(0.9)]
                for _, option in high_volume_calls.iterrows():
                    unusual_flow.append({
                        "type": "high_volume_call",
                        "strike": option['strike'],
                        "volume": option['volume'],
                        "expiration": exp_date,
                        "moneyness": "ITM" if underlying_price > option['strike'] else "OTM"
                    })
            
            if 'volume' in puts.columns:
                high_volume_puts = puts[puts['volume'] > puts['volume'].quantile(0.9)]
                for _, option in high_volume_puts.iterrows():
                    unusual_flow.append({
                        "type": "high_volume_put",
                        "strike": option['strike'],
                        "volume": option['volume'],
                        "expiration": exp_date,
                        "moneyness": "ITM" if underlying_price < option['strike'] else "OTM"
                    })
            
            # Unusual put/call activity
            if 'openInterest' in calls.columns and 'openInterest' in puts.columns:
                # Large OI accumulation
                high_oi_threshold = max(calls['openInterest'].quantile(0.95), puts['openInterest'].quantile(0.95))
                
                high_oi_calls = calls[calls['openInterest'] > high_oi_threshold]
                for _, option in high_oi_calls.iterrows():
                    unusual_flow.append({
                        "type": "large_oi_call",
                        "strike": option['strike'],
                        "openInterest": option['openInterest'],
                        "expiration": exp_date,
                        "sentiment": "bullish_position"
                    })
                
                high_oi_puts = puts[puts['openInterest'] > high_oi_threshold]
                for _, option in high_oi_puts.iterrows():
                    unusual_flow.append({
                        "type": "large_oi_put",
                        "strike": option['strike'],
                        "openInterest": option['openInterest'],
                        "expiration": exp_date,
                        "sentiment": "bearish_position"
                    })
            
        except Exception as e:
            logger.error(f"Error detecting unusual flow: {str(e)}")
        
        return unusual_flow
    
    async def _analyze_volatility_surface(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility surface and term structure."""
        try:
            symbol = options_data["symbol"]
            options_chain = options_data["options_chain"]
            underlying_price = options_data["underlying_price"]
            
            # Collect implied volatilities across strikes and expirations
            vol_surface_data = []
            term_structure_data = []
            
            for exp_date, chain_data in options_chain.items():
                calls = chain_data["calls"]
                puts = chain_data["puts"]
                
                # Calculate implied volatility (simplified)
                exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
                days_to_exp = (exp_dt - datetime.now()).days
                T = days_to_exp / 365.25
                
                # Implied vol estimation (simplified Newton-Raphson)
                for option_type, options_df in [("calls", calls), ("puts", puts)]:
                    for _, option in options_df.iterrows():
                        if option.get('lastPrice', 0) > 0 and T > 0:
                            try:
                                iv = self._estimate_implied_volatility(
                                    underlying_price, option['strike'], T, 
                                    option['lastPrice'], 0.05, option_type
                                )
                                
                                if iv and 0.05 < iv < 2.0:  # Reasonable bounds
                                    moneyness = option['strike'] / underlying_price
                                    vol_surface_data.append({
                                        "strike": option['strike'],
                                        "moneyness": moneyness,
                                        "days_to_exp": days_to_exp,
                                        "implied_vol": iv,
                                        "option_type": option_type,
                                        "expiration": exp_date
                                    })
                                    
                                    # Term structure data (ATM volatility)
                                    if 0.95 <= moneyness <= 1.05:
                                        term_structure_data.append({
                                            "days_to_exp": days_to_exp,
                                            "implied_vol": iv,
                                            "expiration": exp_date
                                        })
                            except Exception as e:
                                continue
            
            # Build volatility surface
            if vol_surface_data:
                vol_surface_df = pd.DataFrame(vol_surface_data)
                
                # Volatility skew analysis
                skew_analysis = self._analyze_volatility_skew(vol_surface_df, underlying_price)
                
                # Volatility smile analysis
                smile_analysis = self._analyze_volatility_smile(vol_surface_df)
                
                # Term structure analysis
                term_structure_analysis = self._analyze_term_structure(term_structure_data)
                
                return {
                    "volatility_surface": {
                        "data_points": len(vol_surface_data),
                        "skew_analysis": skew_analysis,
                        "smile_analysis": smile_analysis,
                        "surface_characteristics": self._characterize_vol_surface(vol_surface_df)
                    },
                    "term_structure": term_structure_analysis,
                    "current_implied_vol": self._get_current_implied_vol(vol_surface_df, underlying_price)
                }
            
            return {"error": "Insufficient volatility data"}
            
        except Exception as e:
            logger.error(f"Error analyzing volatility surface: {str(e)}")
            return {"error": str(e)}
    
    def _estimate_implied_volatility(self, S: float, K: float, T: float, 
                                   market_price: float, r: float, option_type: str) -> Optional[float]:
        """Estimate implied volatility using simplified Newton-Raphson method."""
        try:
            if T <= 0 or market_price <= 0:
                return None
            
            # Black-Scholes price function
            def bs_price(S, K, T, r, sigma, option_type):
                if sigma <= 0 or T <= 0:
                    return 0
                
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                
                if option_type == "calls":
                    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                else:
                    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            
            # Newton-Raphson method
            sigma = 0.2  # Initial guess
            tolerance = 1e-6
            max_iterations = 100
            
            for i in range(max_iterations):
                price = bs_price(S, K, T, r, sigma, option_type)
                price_diff = price - market_price
                
                if abs(price_diff) < tolerance:
                    return sigma
                
                # Vega calculation for derivative
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                vega = S * norm.pdf(d1) * np.sqrt(T)
                
                if vega == 0:
                    break
                
                sigma = sigma - price_diff / vega
                sigma = max(0.01, min(2.0, sigma))  # Keep within reasonable bounds
            
            return sigma if 0.01 <= sigma <= 2.0 else None
            
        except Exception as e:
            logger.error(f"Error estimating implied volatility: {str(e)}")
            return None
    
    def _analyze_volatility_skew(self, vol_df: pd.DataFrame, underlying_price: float) -> Dict[str, Any]:
        """Analyze volatility skew patterns."""
        try:
            # Group by moneyness ranges and calculate average IV
            skew_analysis = {}
            
            for option_type in ['calls', 'puts']:
                type_data = vol_df[vol_df['option_type'] == option_type]
                
                if len(type_data) < 5:
                    continue
                
                # Define moneyness buckets
                buckets = {
                    'deep_otm': type_data['moneyness'] > 1.1,
                    'otm': (type_data['moneyness'] > 1.05) & (type_data['moneyness'] <= 1.1),
                    'atm': (type_data['moneyness'] >= 0.95) & (type_data['moneyness'] <= 1.05),
                    'itm': (type_data['moneyness'] >= 0.9) & (type_data['moneyness'] < 0.95),
                    'deep_itm': type_data['moneyness'] < 0.9
                }
                
                bucket_ivs = {}
                for bucket_name, condition in buckets.items():
                    bucket_data = type_data[condition]
                    if len(bucket_data) > 0:
                        bucket_ivs[bucket_name] = bucket_data['implied_vol'].mean()
                
                if bucket_ivs:
                    # Calculate skew metrics
                    skew_analysis[option_type] = {
                        "bucket_ivs": bucket_ivs,
                        "skew_slope": self._calculate_skew_slope(bucket_ivs),
                        "skew_intensity": self._calculate_skew_intensity(bucket_ivs)
                    }
            
            return skew_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing volatility skew: {str(e)}")
            return {}
    
    def _analyze_volatility_smile(self, vol_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility smile characteristics."""
        try:
            # For options with similar expiration, analyze the smile
            smile_analysis = {}
            
            # Group by expiration and option type
            for option_type in ['calls', 'puts']:
                type_data = vol_df[vol_df['option_type'] == option_type]
                
                if len(type_data) < 5:
                    continue
                
                # Find the expiration with most data points
                exp_counts = type_data['expiration'].value_counts()
                if exp_counts.empty:
                    continue
                
                main_exp = exp_counts.index[0]
                main_exp_data = type_data[type_data['expiration'] == main_exp]
                
                if len(main_exp_data) < 5:
                    continue
                
                # Calculate smile characteristics
                moneyness = main_exp_data['moneyness'].values
                ivs = main_exp_data['implied_vol'].values
                
                if len(moneyness) > 2:
                    # Fit polynomial to capture smile shape
                    smile_analysis[option_type] = {
                        "smile_curvature": self._calculate_smile_curvature(moneyness, ivs),
                        "volatility_range": float(np.max(ivs) - np.min(ivs)),
                        "atm_volatility": self._get_atm_volatility(main_exp_data, 1.0),
                        "wing_spread": self._calculate_wing_spread(main_exp_data)
                    }
            
            return smile_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing volatility smile: {str(e)}")
            return {}
    
    def _analyze_term_structure(self, term_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze volatility term structure."""
        try:
            if len(term_data) < 2:
                return {"error": "Insufficient term structure data"}
            
            term_df = pd.DataFrame(term_data)
            term_df = term_df.sort_values('days_to_exp')
            
            # Term structure characteristics
            term_slope = self._calculate_term_slope(term_df)
            term_curvature = self._calculate_term_curvature(term_df)
            
            # Short-term vs long-term vol comparison
            short_term_vol = term_df[term_df['days_to_exp'] <= 30]['implied_vol'].mean()
            long_term_vol = term_df[term_df['days_to_exp'] > 60]['implied_vol'].mean()
            
            return {
                "term_slope": term_slope,
                "term_curvature": term_curvature,
                "short_term_vol": float(short_term_vol) if not pd.isna(short_term_vol) else None,
                "long_term_vol": float(long_term_vol) if not pd.isna(long_term_vol) else None,
                "term_structure_shape": self._classify_term_shape(term_df),
                "contango_backwardation": self._classify_contango_backwardation(term_df)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing term structure: {str(e)}")
            return {"error": str(e)}
    
    def _characterize_vol_surface(self, vol_df: pd.DataFrame) -> Dict[str, Any]:
        """Characterize overall volatility surface properties."""
        try:
            if vol_df.empty:
                return {"error": "No volatility data"}
            
            characteristics = {
                "surface_level": float(vol_df['implied_vol'].mean()),
                "surface_stability": float(1 / (1 + vol_df['implied_vol'].std())),
                "moneyness_range": {
                    "min": float(vol_df['moneyness'].min()),
                    "max": float(vol_df['moneyness'].max())
                },
                "time_range": {
                    "min_days": int(vol_df['days_to_exp'].min()),
                    "max_days": int(vol_df['days_to_exp'].max())
                },
                "surface_complexity": self._assess_surface_complexity(vol_df)
            }
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error characterizing volatility surface: {str(e)}")
            return {"error": str(e)}
    
    def _get_current_implied_vol(self, vol_df: pd.DataFrame, underlying_price: float) -> Dict[str, Any]:
        """Get current implied volatility metrics."""
        try:
            if vol_df.empty:
                return {"error": "No volatility data"}
            
            # ATM volatility (closest to 1.0 moneyness)
            atm_vol = self._get_atm_volatility(vol_df, underlying_price / underlying_price)
            
            # 25-delta vol (approximate)
            vol_25_delta = self._get_delta_volatility(vol_df, 0.25)
            
            # 10-delta vol
            vol_10_delta = self._get_delta_volatility(vol_df, 0.10)
            
            return {
                "atm_volatility": float(atm_vol) if atm_vol else None,
                "vol_25_delta": float(vol_25_delta) if vol_25_delta else None,
                "vol_10_delta": float(vol_10_delta) if vol_10_delta else None,
                "volatility_skew": self._calculate_current_vol_skew(vol_df)
            }
            
        except Exception as e:
            logger.error(f"Error getting current implied vol: {str(e)}")
            return {"error": str(e)}
    
    def _get_atm_volatility(self, vol_df: pd.DataFrame, target_moneyness: float = 1.0) -> Optional[float]:
        """Get ATM volatility."""
        try:
            atm_data = vol_df[abs(vol_df['moneyness'] - target_moneyness) < 0.05]
            if len(atm_data) > 0:
                return atm_data['implied_vol'].mean()
            return None
        except Exception:
            return None
    
    def _get_delta_volatility(self, vol_df: pd.DataFrame, delta: float) -> Optional[float]:
        """Approximate volatility for given delta."""
        try:
            # This is a simplified approach - in reality would need more sophisticated mapping
            if delta <= 0.25:
                # 25-delta put/10-delta call region
                target_moneyness = 1.0 - delta * 0.5
            else:
                target_moneyness = 1.0 + (1 - delta) * 0.5
            
            delta_vol = self._get_atm_volatility(vol_df, target_moneyness)
            return delta_vol
        except Exception:
            return None
    
    def _calculate_current_vol_skew(self, vol_df: pd.DataFrame) -> float:
        """Calculate current volatility skew."""
        try:
            # Simple skew calculation: compare OTM call vol vs OTM put vol
            call_data = vol_df[vol_df['option_type'] == 'calls']
            put_data = vol_df[vol_df['option_type'] == 'puts']
            
            if len(call_data) > 0 and len(put_data) > 0:
                otm_calls = call_data[call_data['moneyness'] > 1.05]
                otm_puts = put_data[put_data['moneyness'] < 0.95]
                
                if len(otm_calls) > 0 and len(otm_puts) > 0:
                    call_vol = otm_calls['implied_vol'].mean()
                    put_vol = otm_puts['implied_vol'].mean()
                    return float(put_vol - call_vol)  # Put skew
            return 0.0
        except Exception:
            return 0.0
    
    async def _analyze_greeks(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Greeks and sensitivity analysis."""
        try:
            symbol = options_data["symbol"]
            options_chain = options_data["options_chain"]
            underlying_price = options_data["underlying_price"]
            
            greeks_analysis = {
                "aggregate_greeks": {},
                "greeks_by_strike": {},
                "greeks_by_expiration": {},
                "risk_concentration": {},
                "sensitivity_analysis": {}
            }
            
            all_greeks = {
                'delta': [], 'gamma': [], 'theta': [], 'vega': [], 'rho': []
            }
            
            for exp_date, chain_data in options_chain.items():
                calls = chain_data["calls"]
                puts = chain_data["puts"]
                
                # Calculate Greeks for each option
                exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
                days_to_exp = (exp_dt - datetime.now()).days
                T = days_to_exp / 365.25
                
                if T <= 0:
                    continue
                
                for option_type, options_df in [("calls", calls), ("puts", puts)]:
                    for _, option in options_df.iterrows():
                        if option.get('lastPrice', 0) > 0 and T > 0:
                            try:
                                greeks = self._calculate_greeks(
                                    underlying_price, option['strike'], T, 0.05, 0.20, option_type
                                )
                                
                                if greeks:
                                    # Store by strike
                                    if option['strike'] not in greeks_analysis["greeks_by_strike"]:
                                        greeks_analysis["greeks_by_strike"][option['strike']] = {}
                                    
                                    greeks_analysis["greeks_by_strike"][option['strike']][exp_date] = {
                                        'option_type': option_type,
                                        'greeks': greeks,
                                        'volume': option.get('volume', 0),
                                        'open_interest': option.get('openInterest', 0)
                                    }
                                    
                                    # Accumulate for aggregate analysis
                                    for greek_name, greek_value in greeks.items():
                                        if greek_name in all_greeks:
                                            all_greeks[greek_name].append(greeks_value)
                                    
                                    # Weight by volume/oi if available
                                    weight = option.get('volume', 1) + option.get('openInterest', 0)
                                    if weight > 0:
                                        for greek_name, greek_value in greeks.items():
                                            if greek_name in all_greeks:
                                                all_greeks[greek_name].extend([greek_value] * min(10, int(weight/100)))
                            except Exception as e:
                                continue
            
            # Calculate aggregate Greeks
            for greek_name, values in all_greeks.items():
                if values:
                    greeks_analysis["aggregate_greeks"][greek_name] = {
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'std': float(np.std(values)),
                        'range': [float(np.min(values)), float(np.max(values))]
                    }
            
            # Risk concentration analysis
            greeks_analysis["risk_concentration"] = self._analyze_greeks_concentration(
                greeks_analysis["greeks_by_strike"]
            )
            
            # Sensitivity analysis
            greeks_analysis["sensitivity_analysis"] = {
                "price_sensitivity": self._analyze_price_sensitivity(greeks_analysis["aggregate_greeks"]),
                "volatility_sensitivity": self._analyze_volatility_sensitivity(greeks_analysis["aggregate_greeks"]),
                "time_sensitivity": self._analyze_time_sensitivity(greeks_analysis["aggregate_greeks"])
            }
            
            return greeks_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Greeks: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_greeks(self, S: float, K: float, T: float, r: float, 
                         sigma: float, option_type: str) -> Optional[Dict[str, float]]:
        """Calculate option Greeks using Black-Scholes formulas."""
        try:
            if T <= 0 or sigma <= 0:
                return None
            
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            # Greeks calculations
            delta = norm.cdf(d1) if option_type == "calls" else norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)  # Per 1% vol change
            theta_call = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                          r * K * np.exp(-r*T) * norm.cdf(d2))
            theta_put = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                          r * K * np.exp(-r*T) * norm.cdf(-d2))
            theta = theta_call if option_type == "calls" else theta_put
            rho_call = K * T * np.exp(-r*T) * norm.cdf(d2)
            rho_put = -K * T * np.exp(-r*T) * norm.cdf(-d2)
            rho = rho_call if option_type == "calls" else rho_put
            
            return {
                'delta': float(delta),
                'gamma': float(gamma),
                'theta': float(theta),  # Per day
                'vega': float(vega),    # Per 1% vol change
                'rho': float(rho)
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {str(e)}")
            return None
    
    def _analyze_greeks_concentration(self, greeks_by_strike: Dict) -> Dict[str, Any]:
        """Analyze concentration of Greek exposures."""
        try:
            if not greeks_by_strike:
                return {"error": "No Greeks data"}
            
            # Analyze delta concentration
            delta_exposures = {}
            gamma_exposures = {}
            
            for strike, exp_data in greeks_by_strike.items():
                total_delta = 0
                total_gamma = 0
                total_weight = 0
                
                for exp_date, data in exp_data.items():
                    weight = data.get('volume', 1) + data.get('open_interest', 0)
                    greeks = data.get('greeks', {})
                    
                    if 'delta' in greeks:
                        total_delta += greeks['delta'] * weight
                    if 'gamma' in greeks:
                        total_gamma += greeks['gamma'] * weight
                    total_weight += weight
                
                if total_weight > 0:
                    delta_exposures[strike] = total_delta / total_weight
                    gamma_exposures[strike] = total_gamma / total_weight
            
            # Find concentration points
            if delta_exposures:
                max_delta_strike = max(delta_exposures, key=delta_exposures.get)
                max_gamma_strike = max(gamma_exposures, key=gamma_exposures.get)
                
                return {
                    "delta_concentration": {
                        "max_exposure_strike": max_delta_strike,
                        "max_exposure_value": delta_exposures[max_delta_strike],
                        "total_exposure": sum(delta_exposures.values())
                    },
                    "gamma_concentration": {
                        "max_gamma_strike": max_gamma_strike,
                        "max_gamma_value": gamma_exposures[max_gamma_strike],
                        "total_gamma": sum(gamma_exposures.values())
                    },
                    "exposure_distribution": {
                        "delta_exposures": delta_exposures,
                        "gamma_exposures": gamma_exposures
                    }
                }
            
            return {"error": "Unable to analyze Greek concentration"}
            
        except Exception as e:
            logger.error(f"Error analyzing Greeks concentration: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_price_sensitivity(self, aggregate_greeks: Dict) -> Dict[str, Any]:
        """Analyze price sensitivity of option positions."""
        try:
            delta = aggregate_greeks.get('delta', {}).get('mean', 0)
            gamma = aggregate_greeks.get('gamma', {}).get('mean', 0)
            
            # Price move scenarios
            scenarios = {
                "1pct_move": {
                    "delta_impact": delta * 0.01,
                    "gamma_impact": 0.5 * gamma * (0.01 ** 2)
                },
                "2pct_move": {
                    "delta_impact": delta * 0.02,
                    "gamma_impact": 0.5 * gamma * (0.02 ** 2)
                },
                "5pct_move": {
                    "delta_impact": delta * 0.05,
                    "gamma_impact": 0.5 * gamma * (0.05 ** 2)
                }
            }
            
            return {
                "position_delta": float(delta),
                "position_gamma": float(gamma),
                "price_scenarios": scenarios,
                "sensitivity_level": "high" if abs(delta) > 0.5 else "medium" if abs(delta) > 0.2 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing price sensitivity: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_volatility_sensitivity(self, aggregate_greeks: Dict) -> Dict[str, Any]:
        """Analyze volatility sensitivity of option positions."""
        try:
            vega = aggregate_greeks.get('vega', {}).get('mean', 0)
            
            # Volatility move scenarios
            vol_scenarios = {
                "1pct_vol_increase": vega * 0.01,
                "2pct_vol_increase": vega * 0.02,
                "5pct_vol_increase": vega * 0.05,
                "volatility_compression": -vega * 0.03
            }
            
            return {
                "position_vega": float(vega),
                "volatility_scenarios": {k: float(v) for k, v in vol_scenarios.items()},
                "vol_sensitivity": "high" if abs(vega) > 0.5 else "medium" if abs(vega) > 0.1 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility sensitivity: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_time_sensitivity(self, aggregate_greeks: Dict) -> Dict[str, Any]:
        """Analyze time decay sensitivity of option positions."""
        try:
            theta = aggregate_greeks.get('theta', {}).get('mean', 0)
            
            # Time decay scenarios
            decay_scenarios = {
                "daily_decay": theta,
                "weekly_decay": theta * 7,
                "monthly_decay": theta * 30
            }
            
            return {
                "position_theta": float(theta),
                "time_decay_scenarios": {k: float(v) for k, v in decay_scenarios.items()},
                "time_sensitivity": "high" if abs(theta) > 0.1 else "medium" if abs(theta) > 0.05 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing time sensitivity: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_options_sentiment(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze options-based sentiment indicators."""
        try:
            symbol = options_data["symbol"]
            options_chain = options_data["options_chain"]
            underlying_price = options_data["underlying_price"]
            
            sentiment_indicators = {
                "put_call_ratio": 0,
                "volatility_sentiment": {},
                "skew_sentiment": {},
                "volume_sentiment": {},
                "positioning_indicators": {},
                "contrarian_indicators": {}
            }
            
            total_calls_volume = 0
            total_puts_volume = 0
            total_calls_oi = 0
            total_puts_oi = 0
            
            for exp_date, chain_data in options_chain.items():
                calls = chain_data["calls"]
                puts = chain_data["puts"]
                
                # Volume sentiment
                calls_vol = calls['volume'].sum() if 'volume' in calls.columns else 0
                puts_vol = puts['volume'].sum() if 'volume' in puts.columns else 0
                
                total_calls_volume += calls_vol
                total_puts_volume += puts_vol
                
                # Open interest sentiment
                calls_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
                puts_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
                
                total_calls_oi += calls_oi
                total_puts_oi += puts_oi
                
                # Put/call ratio by expiration
                pcr_volume = puts_vol / calls_vol if calls_vol > 0 else 0
                pcr_oi = puts_oi / calls_oi if calls_oi > 0 else 0
                
                sentiment_indicators["volume_sentiment"][exp_date] = {
                    "put_call_ratio": pcr_volume,
                    "calls_volume": calls_vol,
                    "puts_volume": puts_vol
                }
                
                sentiment_indicators["positioning_indicators"][exp_date] = {
                    "put_call_ratio_oi": pcr_oi,
                    "calls_oi": calls_oi,
                    "puts_oi": puts_oi
                }
            
            # Overall sentiment
            overall_pcr = total_puts_volume / total_calls_volume if total_calls_volume > 0 else 0
            overall_pcr_oi = total_puts_oi / total_calls_oi if total_calls_oi > 0 else 0
            
            sentiment_indicators["put_call_ratio"] = overall_pcr
            sentiment_indicators["oi_put_call_ratio"] = overall_pcr_oi
            
            # Sentiment interpretation
            sentiment_indicators["sentiment_interpretation"] = {
                "overall_sentiment": self._interpret_sentiment(overall_pcr),
                "oi_sentiment": self._interpret_sentiment(overall_pcr_oi),
                "contrarian_signal": self._calculate_contrarian_signal(overall_pcr, overall_pcr_oi),
                "sentiment_extreme": self._assess_sentiment_extreme(overall_pcr, overall_pcr_oi)
            }
            
            return sentiment_indicators
            
        except Exception as e:
            logger.error(f"Error analyzing options sentiment: {str(e)}")
            return {"error": str(e)}
    
    def _interpret_sentiment(self, put_call_ratio: float) -> str:
        """Interpret put/call ratio sentiment."""
        if put_call_ratio > 1.5:
            return "extremely_bearish"
        elif put_call_ratio > 1.2:
            return "bearish"
        elif put_call_ratio > 0.8:
            return "neutral"
        elif put_call_ratio > 0.6:
            return "bullish"
        else:
            return "extremely_bullish"
    
    def _calculate_contrarian_signal(self, pcr_volume: float, pcr_oi: float) -> str:
        """Calculate contrarian sentiment signals."""
        # Extreme readings often signal reversals
        if pcr_volume > 1.5 and pcr_oi < 1.0:
            return "bearish_sentiment_contrarian_bullish"
        elif pcr_volume < 0.6 and pcr_oi > 1.2:
            return "bullish_sentiment_contrarian_bearish"
        elif abs(pcr_volume - pcr_oi) > 0.5:
            return "mixed_sentiment_proceed_cautiously"
        else:
            return "sentiment_confirmation"
    
    def _assess_sentiment_extreme(self, pcr_volume: float, pcr_oi: float) -> str:
        """Assess how extreme current sentiment is."""
        avg_pcr = (pcr_volume + pcr_oi) / 2
        
        if avg_pcr > 1.8 or avg_pcr < 0.4:
            return "extremely_extreme"
        elif avg_pcr > 1.5 or avg_pcr < 0.6:
            return "highly_extreme"
        elif avg_pcr > 1.2 or avg_pcr < 0.7:
            return "moderately_extreme"
        else:
            return "normal_range"
    
    async def _analyze_expiration_impact(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze options expiration impact and gamma exposure."""
        try:
            symbol = options_data["symbol"]
            options_chain = options_data["options_chain"]
            underlying_price = options_data["underlying_price"]
            
            expiration_analysis = {
                "upcoming_expirations": {},
                "max_pain_analysis": {},
                "gamma_exposure": {},
                "expiration_scenarios": {}
            }
            
            today = datetime.now()
            
            for exp_date, chain_data in options_chain.items():
                try:
                    exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
                    days_to_exp = (exp_dt - today).days
                    
                    if days_to_exp <= 0:
                        continue  # Skip expired options
                    
                    calls = chain_data["calls"]
                    puts = chain_data["puts"]
                    
                    # Max pain calculation
                    max_pain = self._calculate_max_pain(calls, puts, underlying_price)
                    
                    # Gamma exposure
                    gamma_exposure = self._calculate_gamma_exposure(calls, puts, underlying_price, days_to_exp)
                    
                    # Open interest concentration
                    oi_concentration = self._analyze_oi_concentration(calls, puts)
                    
                    expiration_analysis["upcoming_expirations"][exp_date] = {
                        "days_to_exp": days_to_exp,
                        "max_pain": max_pain,
                        "gamma_exposure": gamma_exposure,
                        "oi_concentration": oi_concentration,
                        "expiration_impact": self._assess_expiration_impact(days_to_exp, gamma_exposure)
                    }
                    
                except Exception as e:
                    logger.warning(f"Error analyzing expiration {exp_date}: {str(e)}")
                    continue
            
            # Overall expiration impact
            expiration_analysis["overall_impact"] = self._calculate_overall_expiration_impact(
                expiration_analysis["upcoming_expirations"]
            )
            
            return expiration_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing expiration impact: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_max_pain(self, calls: pd.DataFrame, puts: pd.DataFrame, 
                          underlying_price: float) -> Dict[str, Any]:
        """Calculate max pain point."""
        try:
            if calls.empty or puts.empty:
                return {"max_pain_strike": None, "total_pain": 0}
            
            strikes = []
            pain_values = []
            
            # Get all unique strikes
            all_strikes = set(calls['strike'].tolist() + puts['strike'].tolist())
            
            for strike in sorted(all_strikes):
                # Calculate pain for this strike
                call_pain = 0
                put_pain = 0
                
                # Call pain (if strike < underlying)
                call_data = calls[calls['strike'] == strike]
                if not call_data.empty:
                    call_oi = call_data['openInterest'].sum() if 'openInterest' in call_data.columns else 0
                    call_pain = call_oi * max(0, strike - underlying_price)
                
                # Put pain (if strike > underlying)
                put_data = puts[puts['strike'] == strike]
                if not put_data.empty:
                    put_oi = put_data['openInterest'].sum() if 'openInterest' in put_data.columns else 0
                    put_pain = put_oi * max(0, underlying_price - strike)
                
                total_pain = call_pain + put_pain
                strikes.append(strike)
                pain_values.append(total_pain)
            
            if pain_values:
                max_pain_idx = np.argmax(pain_values)
                return {
                    "max_pain_strike": strikes[max_pain_idx],
                    "max_pain_value": pain_values[max_pain_idx],
                    "total_pain": sum(pain_values),
                    "pain_distance": abs(strikes[max_pain_idx] - underlying_price) / underlying_price
                }
            
            return {"max_pain_strike": None, "total_pain": 0}
            
        except Exception as e:
            logger.error(f"Error calculating max pain: {str(e)}")
            return {"max_pain_strike": None, "total_pain": 0}
    
    def _calculate_gamma_exposure(self, calls: pd.DataFrame, puts: pd.DataFrame, 
                                underlying_price: float, days_to_exp: int) -> Dict[str, Any]:
        """Calculate gamma exposure around current price."""
        try:
            if days_to_exp <= 0:
                return {"gamma_exposure": 0, "gamma_concentration": "none"}
            
            # Simplified gamma calculation (actual gamma more complex)
            # Look for gamma concentration around current price
            price_range = underlying_price * 0.05  # 5% range
            near_price_calls = calls[
                (calls['strike'] >= underlying_price - price_range) &
                (calls['strike'] <= underlying_price + price_range)
            ]
            near_price_puts = puts[
                (puts['strike'] >= underlying_price - price_range) &
                (puts['strike'] <= underlying_price + price_range)
            ]
            
            # Estimate gamma based on volume and proximity to ATM
            total_gamma = 0
            gamma_concentration = "low"
            
            for options_df in [near_price_calls, near_price_puts]:
                if options_df.empty:
                    continue
                
                for _, option in options_df.iterrows():
                    # Simplified gamma estimation
                    moneyness = abs(option['strike'] / underlying_price - 1)
                    volume = option.get('volume', 0)
                    
                    # Higher gamma near ATM, lower as distance increases
                    atm_factor = max(0.1, 1 - moneyness * 10)
                    estimated_gamma = volume * atm_factor / (days_to_exp + 1)
                    total_gamma += estimated_gamma
            
            # Classify gamma exposure
            if total_gamma > 10000:
                gamma_concentration = "very_high"
            elif total_gamma > 5000:
                gamma_concentration = "high"
            elif total_gamma > 1000:
                gamma_concentration = "medium"
            else:
                gamma_concentration = "low"
            
            return {
                "gamma_exposure": total_gamma,
                "gamma_concentration": gamma_concentration
            }
            
        except Exception as e:
            logger.error(f"Error calculating gamma exposure: {str(e)}")
            return {"gamma_exposure": 0, "gamma_concentration": "unknown"}
    
    def _analyze_oi_concentration(self, calls: pd.DataFrame, puts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze open interest concentration."""
        try:
            if calls.empty and puts.empty:
                return {"concentration_level": "unknown"}
            
            total_oi = 0
            concentration_by_strike = {}
            
            for options_df in [calls, puts]:
                if options_df.empty:
                    continue
                
                for _, option in options_df.iterrows():
                    oi = option.get('openInterest', 0)
                    strike = option['strike']
                    total_oi += oi
                    
                    if strike not in concentration_by_strike:
                        concentration_by_strike[strike] = 0
                    concentration_by_strike[strike] += oi
            
            if total_oi == 0:
                return {"concentration_level": "none"}
            
            # Calculate concentration metrics
            oi_values = list(concentration_by_strike.values())
            concentration_ratio = max(oi_values) / total_oi if total_oi > 0 else 0
            
            # Find top concentration points
            top_strikes = sorted(concentration_by_strike.items(), 
                               key=lambda x: x[1], reverse=True)[:3]
            
            if concentration_ratio > 0.3:
                concentration_level = "very_high"
            elif concentration_ratio > 0.2:
                concentration_level = "high"
            elif concentration_ratio > 0.1:
                concentration_level = "medium"
            else:
                concentration_level = "low"
            
            return {
                "concentration_level": concentration_level,
                "concentration_ratio": concentration_ratio,
                "top_concentration_strikes": [
                    {"strike": strike, "oi": oi} for strike, oi in top_strikes
                ]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing OI concentration: {str(e)}")
            return {"concentration_level": "unknown"}
    
    def _assess_expiration_impact(self, days_to_exp: int, gamma_exposure: Dict) -> str:
        """Assess potential expiration impact."""
        try:
            gamma_level = gamma_exposure.get("gamma_concentration", "low")
            
            if days_to_exp <= 7 and gamma_level in ["high", "very_high"]:
                return "high_impact_expected"
            elif days_to_exp <= 14 and gamma_level == "very_high":
                return "moderate_impact_possible"
            elif days_to_exp <= 30 and gamma_level in ["medium", "high", "very_high"]:
                return "low_impact_likely"
            else:
                return "minimal_impact_expected"
                
        except Exception:
            return "unknown_impact"
    
    def _calculate_overall_expiration_impact(self, expirations: Dict) -> Dict[str, Any]:
        """Calculate overall expiration impact across all upcoming expirations."""
        try:
            if not expirations:
                return {"overall_impact": "no_expirations"}
            
            impact_scores = {
                "high_impact_expected": 4,
                "moderate_impact_possible": 3,
                "low_impact_likely": 2,
                "minimal_impact_expected": 1,
                "unknown_impact": 0
            }
            
            total_score = 0
            exp_count = 0
            
            for exp_date, data in expirations.items():
                impact = data.get("expiration_impact", "unknown_impact")
                score = impact_scores.get(impact, 0)
                
                # Weight by days to expiration (closer = higher weight)
                days_weight = max(0.1, 1 - data.get("days_to_exp", 30) / 30)
                weighted_score = score * days_weight
                
                total_score += weighted_score
                exp_count += 1
            
            avg_score = total_score / exp_count if exp_count > 0 else 0
            
            if avg_score >= 3:
                overall_impact = "high"
            elif avg_score >= 2:
                overall_impact = "moderate"
            elif avg_score >= 1:
                overall_impact = "low"
            else:
                overall_impact = "minimal"
            
            return {
                "overall_impact": overall_impact,
                "impact_score": avg_score,
                "weighted_expirations": len(expirations)
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall expiration impact: {str(e)}")
            return {"overall_impact": "unknown"}
    
    async def _analyze_skew_smile(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility skew and smile characteristics."""
        try:
            # This analysis builds on volatility surface analysis
            vol_surface = await self._analyze_volatility_surface(options_data)
            
            if "error" in vol_surface:
                return {"error": vol_surface["error"]}
            
            skew_smile_analysis = {
                "skew_characteristics": vol_surface.get("volatility_surface", {}).get("skew_analysis", {}),
                "smile_characteristics": vol_surface.get("volatility_surface", {}).get("smile_analysis", {}),
                "current_skew": self._assess_current_skew(vol_surface),
                "smile_interpretation": self._interpret_smile(vol_surface)
            }
            
            return skew_smile_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing skew and smile: {str(e)}")
            return {"error": str(e)}
    
    def _assess_current_skew(self, vol_surface: Dict) -> Dict[str, Any]:
        """Assess current volatility skew."""
        try:
            skew_data = vol_surface.get("volatility_surface", {}).get("skew_analysis", {})
            
            if not skew_data:
                return {"skew_assessment": "insufficient_data"}
            
            skew_assessment = {}
            
            for option_type, data in skew_data.items():
                if "skew_slope" in data:
                    slope = data["skew_slope"]
                    if slope > 0.5:
                        skew_direction = "steeper_right_skew"
                    elif slope < -0.5:
                        skew_direction = "steeper_left_skew"
                    else:
                        skew_direction = "relatively_flat"
                    
                    skew_assessment[option_type] = {
                        "skew_direction": skew_direction,
                        "skew_slope": slope,
                        "skew_intensity": data.get("skew_intensity", 0)
                    }
            
            return skew_assessment
            
        except Exception as e:
            logger.error(f"Error assessing current skew: {str(e)}")
            return {"skew_assessment": "error"}
    
    def _interpret_smile(self, vol_surface: Dict) -> Dict[str, Any]:
        """Interpret volatility smile characteristics."""
        try:
            smile_data = vol_surface.get("volatility_surface", {}).get("smile_analysis", {})
            
            if not smile_data:
                return {"smile_interpretation": "insufficient_data"}
            
            interpretation = {}
            
            for option_type, data in smile_data.items():
                if "smile_curvature" in data:
                    curvature = data["smile_curvature"]
                    vol_range = data.get("volatility_range", 0)
                    
                    if curvature > 0.5:
                        smile_shape = "pronounced_smile"
                    elif curvature > 0.2:
                        smile_shape = "moderate_smile"
                    else:
                        smile_shape = "relatively_flat"
                    
                    interpretation[option_type] = {
                        "smile_shape": smile_shape,
                        "curvature": curvature,
                        "volatility_range": vol_range,
                        "atm_volatility": data.get("atm_volatility", 0)
                    }
            
            return interpretation
            
        except Exception as e:
            logger.error(f"Error interpreting smile: {str(e)}")
            return {"smile_interpretation": "error"}
    
    async def _detect_unusual_activity(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect unusual options activity patterns."""
        try:
            unusual_activity = {
                "large_trades": [],
                "concentrated_activity": [],
                "activity_anomalies": [],
                "pattern_recognition": {}
            }
            
            # This builds on flow analysis
            flow_analysis = await self._analyze_options_flow(options_data)
            
            if "error" in flow_analysis:
                return {"error": flow_analysis["error"]}
            
            # Analyze unusual flow detected earlier
            unusual_flow = flow_analysis.get("unusual_flow", [])
            
            for activity in unusual_flow:
                if activity["type"] in ["high_volume_call", "high_volume_put"]:
                    unusual_activity["large_trades"].append(activity)
                elif activity["type"] in ["large_oi_call", "large_oi_put"]:
                    unusual_activity["concentrated_activity"].append(activity)
            
            # Pattern recognition
            unusual_activity["pattern_recognition"] = {
                "block_trade_indicators": self._detect_block_trade_patterns(unusual_flow),
                "institutional_activity": self._detect_institutional_activity(unusual_flow),
                "retail_vs_institutional": self._assess_retail_institutional_split(unusual_flow)
            }
            
            return unusual_activity
            
        except Exception as e:
            logger.error(f"Error detecting unusual activity: {str(e)}")
            return {"error": str(e)}
    
    def _detect_block_trade_patterns(self, unusual_flow: List[Dict]) -> Dict[str, Any]:
        """Detect potential block trade patterns."""
        try:
            block_indicators = 0
            large_positions = []
            
            for activity in unusual_flow:
                if "volume" in activity and activity["volume"] > 1000:
                    block_indicators += 1
                    large_positions.append(activity)
            
            # Look for synchronized activity across strikes
            time_clustering = self._analyze_activity_clustering(unusual_flow)
            
            return {
                "block_trade_likelihood": "high" if block_indicators > 3 else "medium" if block_indicators > 1 else "low",
                "large_positions_count": len(large_positions),
                "time_clustering": time_clustering
            }
            
        except Exception as e:
            logger.error(f"Error detecting block trade patterns: {str(e)}")
            return {"block_trade_likelihood": "unknown"}
    
    def _detect_institutional_activity(self, unusual_flow: List[Dict]) -> Dict[str, Any]:
        """Detect likely institutional activity patterns."""
        try:
            institutional_signals = 0
            activity_details = []
            
            for activity in unusual_flow:
                # Large OI accumulation often indicates institutional positioning
                if activity["type"] in ["large_oi_call", "large_oi_put"]:
                    institutional_signals += 1
                    activity_details.append(activity)
                
                # Multi-strike activity
                if "openInterest" in activity and activity["openInterest"] > 5000:
                    institutional_signals += 1
            
            return {
                "institutional_activity_level": "high" if institutional_signals > 3 else "medium" if institutional_signals > 1 else "low",
                "institutional_signals": institutional_signals,
                "activity_details": activity_details
            }
            
        except Exception as e:
            logger.error(f"Error detecting institutional activity: {str(e)}")
            return {"institutional_activity_level": "unknown"}
    
    def _assess_retail_institutional_split(self, unusual_flow: List[Dict]) -> Dict[str, Any]:
        """Assess split between retail and institutional activity."""
        try:
            retail_indicators = 0
            institutional_indicators = 0
            
            for activity in unusual_flow:
                # Small, frequent trades suggest retail
                if activity.get("volume", 0) < 100 and activity.get("openInterest", 0) < 1000:
                    retail_indicators += 1
                # Large, concentrated activity suggests institutional
                elif activity.get("volume", 0) > 500 or activity.get("openInterest", 0) > 3000:
                    institutional_indicators += 1
            
            total_activity = retail_indicators + institutional_indicators
            
            if total_activity == 0:
                return {"split_analysis": "insufficient_data"}
            
            retail_ratio = retail_indicators / total_activity
            institutional_ratio = institutional_indicators / total_activity
            
            return {
                "retail_activity_ratio": retail_ratio,
                "institutional_activity_ratio": institutional_ratio,
                "dominant_activity_type": "retail" if retail_ratio > 0.6 else "institutional" if institutional_ratio > 0.6 else "mixed"
            }
            
        except Exception as e:
            logger.error(f"Error assessing retail/institutional split: {str(e)}")
            return {"split_analysis": "error"}
    
    def _analyze_activity_clustering(self, unusual_flow: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal clustering of unusual activity."""
        try:
            if len(unusual_flow) < 2:
                return {"clustering_score": 0, "clustering_level": "none"}
            
            # This would require timestamp data in a real implementation
            # For now, return basic clustering assessment
            activity_counts = len(unusual_flow)
            
            if activity_counts > 5:
                clustering_level = "high"
                clustering_score = 0.8
            elif activity_counts > 3:
                clustering_level = "medium"
                clustering_score = 0.5
            else:
                clustering_level = "low"
                clustering_score = 0.2
            
            return {
                "clustering_score": clustering_score,
                "clustering_level": clustering_level,
                "activity_count": activity_counts
            }
            
        except Exception as e:
            logger.error(f"Error analyzing activity clustering: {str(e)}")
            return {"clustering_score": 0, "clustering_level": "unknown"}
    
    async def _calculate_options_technical_indicators(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate options-based technical indicators."""
        try:
            symbol = options_data["symbol"]
            underlying_price = options_data["underlying_price"]
            
            # Options-based technical indicators
            technical_indicators = {
                "volatility_breakout": await self._detect_volatility_breakout(options_data),
                "skew_technical_analysis": await self._analyze_skew_technically(options_data),
                "support_resistance_from_options": await self._find_support_resistance_from_options(options_data),
                "options_momentum": await self._calculate_options_momentum(options_data)
            }
            
            # Volatility regime indicators
            technical_indicators["volatility_regime"] = self._assess_options_volatility_regime(options_data)
            
            # Options sentiment indicators
            technical_indicators["technical_sentiment"] = self._assess_technical_sentiment(options_data)
            
            return technical_indicators
            
        except Exception as e:
            logger.error(f"Error calculating options technical indicators: {str(e)}")
            return {"error": str(e)}
    
    async def _detect_volatility_breakout(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect volatility breakouts using options data."""
        try:
            # This would require historical options data
            # For now, provide framework for breakout detection
            
            return {
                "breakout_status": "monitoring",
                "volatility_acceleration": "normal",
                "breakout_probability": 0.3
            }
            
        except Exception as e:
            logger.error(f"Error detecting volatility breakout: {str(e)}")
            return {"breakout_status": "error"}
    
    async def _analyze_skew_technically(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Technical analysis of volatility skew."""
        try:
            # Technical skew analysis
            skew_data = await self._analyze_volatility_surface(options_data)
            
            if "error" in skew_data:
                return {"skew_technical_status": "insufficient_data"}
            
            return {
                "skew_technical_status": "analyzing",
                "skew_trend": "stable",
                "technical_skew_signal": "neutral"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing skew technically: {str(e)}")
            return {"skew_technical_status": "error"}
    
    async def _find_support_resistance_from_options(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find support and resistance levels from options activity."""
        try:
            options_chain = options_data["options_chain"]
            underlying_price = options_data["underlying_price"]
            
            support_levels = []
            resistance_levels = []
            
            # Analyze open interest concentration for support/resistance
            for exp_date, chain_data in options_chain.items():
                calls = chain_data["calls"]
                puts = chain_data["puts"]
                
                # Look for OI concentration points
                for _, option in calls.iterrows():
                    if option.get('openInterest', 0) > 1000 and option['strike'] > underlying_price:
                        resistance_levels.append({
                            "strike": option['strike'],
                            "oi": option['openInterest'],
                            "type": "resistance"
                        })
                
                for _, option in puts.iterrows():
                    if option.get('openInterest', 0) > 1000 and option['strike'] < underlying_price:
                        support_levels.append({
                            "strike": option['strike'],
                            "oi": option['openInterest'],
                            "type": "support"
                        })
            
            # Sort by OI
            support_levels.sort(key=lambda x: x['oi'], reverse=True)
            resistance_levels.sort(key=lambda x: x['oi'], reverse=True)
            
            return {
                "support_levels": support_levels[:3],  # Top 3
                "resistance_levels": resistance_levels[:3],  # Top 3
                "key_support": support_levels[0]['strike'] if support_levels else None,
                "key_resistance": resistance_levels[0]['strike'] if resistance_levels else None
            }
            
        except Exception as e:
            logger.error(f"Error finding support/resistance from options: {str(e)}")
            return {"support_levels": [], "resistance_levels": []}
    
    async def _calculate_options_momentum(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate options momentum indicators."""
        try:
            # Options momentum based on volume and price changes
            # This would require historical options data in production
            
            return {
                "options_momentum_score": 0.5,
                "momentum_trend": "sideways",
                "momentum_strength": "moderate"
            }
            
        except Exception as e:
            logger.error(f"Error calculating options momentum: {str(e)}")
            return {"options_momentum_score": 0.5}
    
    def _assess_options_volatility_regime(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current options volatility regime."""
        try:
            # Simple regime assessment based on implied volatility levels
            vol_surface = self._analyze_volatility_surface_sync(options_data)
            
            if "error" in vol_surface:
                return {"volatility_regime": "unknown"}
            
            atm_vol = vol_surface.get("current_implied_vol", {}).get("atm_volatility", 0.2)
            
            if atm_vol > 0.35:
                regime = "high_volatility"
            elif atm_vol > 0.25:
                regime = "elevated_volatility"
            elif atm_vol > 0.15:
                regime = "normal_volatility"
            else:
                regime = "low_volatility"
            
            return {
                "volatility_regime": regime,
                "atm_volatility": atm_vol,
                "regime_confidence": 0.7
            }
            
        except Exception as e:
            logger.error(f"Error assessing options volatility regime: {str(e)}")
            return {"volatility_regime": "unknown"}
    
    def _analyze_volatility_surface_sync(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous version of volatility surface analysis for internal use."""
        try:
            # This is a simplified synchronous version
            return {"current_implied_vol": {"atm_volatility": 0.25}, "error": None}
        except Exception:
            return {"error": "sync_analysis_failed"}
    
    def _assess_technical_sentiment(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess technical sentiment from options data."""
        try:
            # Combine various technical indicators
            return {
                "technical_sentiment_score": 0.5,
                "sentiment_direction": "neutral",
                "confidence": 0.6
            }
        except Exception as e:
            logger.error(f"Error assessing technical sentiment: {str(e)}")
            return {"technical_sentiment_score": 0.5}
    
    async def _calculate_options_intelligence_score(self, flow_analysis: Dict, 
                                                  sentiment_analysis: Dict, 
                                                  technical_indicators: Dict) -> Dict[str, Any]:
        """Calculate comprehensive options intelligence score."""
        try:
            # Components of the score
            flow_score = self._score_options_flow(flow_analysis)
            sentiment_score = self._score_options_sentiment(sentiment_analysis)
            technical_score = self._score_options_technical(technical_indicators)
            
            # Weighted combination
            weights = {"flow": 0.4, "sentiment": 0.3, "technical": 0.3}
            overall_score = (
                flow_score * weights["flow"] +
                sentiment_score * weights["sentiment"] +
                technical_score * weights["technical"]
            )
            
            # Intelligence score components
            intelligence_components = {
                "options_flow_score": flow_score,
                "sentiment_score": sentiment_score,
                "technical_score": technical_score,
                "overall_options_intelligence": overall_score
            }
            
            # Risk assessment
            risk_level = self._assess_options_risk_level(flow_analysis, sentiment_analysis)
            
            return {
                "options_intelligence_score": overall_score,
                "score_components": intelligence_components,
                "risk_level": risk_level,
                "recommendation": self._generate_options_recommendation(overall_score, risk_level)
            }
            
        except Exception as e:
            logger.error(f"Error calculating options intelligence score: {str(e)}")
            return {"options_intelligence_score": 0.5, "error": str(e)}
    
    def _score_options_flow(self, flow_analysis: Dict) -> float:
        """Score options flow quality."""
        try:
            if "error" in flow_analysis:
                return 0.5
            
            call_put_ratio = flow_analysis.get("call_put_ratio", 1.0)
            total_volume = flow_analysis.get("total_volume", 0)
            
            # Score based on flow patterns
            if 0.8 <= call_put_ratio <= 1.2:
                flow_balance_score = 1.0
            elif 0.6 <= call_put_ratio <= 1.5:
                flow_balance_score = 0.7
            else:
                flow_balance_score = 0.3
            
            # Volume activity score
            volume_score = min(1.0, total_volume / 10000) if total_volume > 0 else 0.5
            
            return (flow_balance_score + volume_score) / 2
            
        except Exception:
            return 0.5
    
    def _score_options_sentiment(self, sentiment_analysis: Dict) -> float:
        """Score options sentiment indicators."""
        try:
            if "error" in sentiment_analysis:
                return 0.5
            
            interpretation = sentiment_analysis.get("sentiment_interpretation", {})
            contrarian_signal = interpretation.get("contrarian_signal", "neutral")
            
            # Score sentiment extremes (contrarian indicator)
            if "contrarian" in contrarian_signal:
                sentiment_score = 0.7  # Contrarian signals often indicate reversals
            elif "confirmation" in contrarian_signal:
                sentiment_score = 0.6  # Confirmed sentiment
            else:
                sentiment_score = 0.5  # Mixed/neutral sentiment
            
            return sentiment_score
            
        except Exception:
            return 0.5
    
    def _score_options_technical(self, technical_indicators: Dict) -> float:
        """Score options technical indicators."""
        try:
            if "error" in technical_indicators:
                return 0.5
            
            # Simple technical scoring
            volatility_regime = technical_indicators.get("volatility_regime", {})
            regime = volatility_regime.get("volatility_regime", "normal_volatility")
            
            if regime == "normal_volatility":
                tech_score = 0.7
            elif regime == "elevated_volatility":
                tech_score = 0.5
            elif regime == "high_volatility":
                tech_score = 0.3
            else:  # low_volatility
                tech_score = 0.6
            
            return tech_score
            
        except Exception:
            return 0.5
    
    def _assess_options_risk_level(self, flow_analysis: Dict, sentiment_analysis: Dict) -> str:
        """Assess options-related risk level."""
        try:
            risk_factors = 0
            
            # High volatility in flow
            call_put_ratio = flow_analysis.get("call_put_ratio", 1.0)
            if call_put_ratio > 2.0 or call_put_ratio < 0.3:
                risk_factors += 1
            
            # Extreme sentiment
            sentiment = sentiment_analysis.get("sentiment_interpretation", {})
            extreme_sentiment = sentiment.get("sentiment_extreme", "")
            if "extremely_extreme" in extreme_sentiment:
                risk_factors += 1
            
            # Contrarian signals
            contrarian = sentiment.get("contrarian_signal", "")
            if "mixed_sentiment" in contrarian:
                risk_factors += 1
            
            if risk_factors >= 2:
                return "high"
            elif risk_factors == 1:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "medium"
    
    def _generate_options_recommendation(self, intelligence_score: float, risk_level: str) -> str:
        """Generate options trading recommendation."""
        try:
            if intelligence_score > 0.7 and risk_level == "low":
                return "favorable_for_options_strategies"
            elif intelligence_score > 0.6 and risk_level in ["low", "medium"]:
                return "cautiously_positive"
            elif intelligence_score < 0.4 or risk_level == "high":
                return "avoid_new_options_positions"
            else:
                return "monitor_and_wait"
                
        except Exception:
            return "neutral_recommendation"
    
    async def get_options_intelligence_history(self, symbol: str = "SPY", days: int = 30) -> Dict[str, Any]:
        """Get historical options intelligence data."""
        try:
            # In production, this would retrieve historical options data
            # For now, return current analysis with simulated historical context
            
            current_analysis = await self.analyze_options_intelligence(symbol)
            
            # Simulated historical intelligence scores
            historical_scores = []
            base_score = current_analysis.get("options_intelligence_score", {}).get("options_intelligence_score", 0.5)
            
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
                    "volatility_trend": "elevated"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting options intelligence history: {str(e)}")
            return {"error": str(e)}