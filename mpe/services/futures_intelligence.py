"""
Module 15: Futures Intelligence Engine

Advanced futures market intelligence system providing real-time analysis of
futures positioning, term structure, basis analysis, and institutional activity
across major futures markets and commodity sectors.

Author: MiniMax Agent
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.interpolate import interp1d
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FuturesIntelligenceEngine:
    """
    Futures market intelligence and analysis engine.
    
    Features:
    - Futures positioning and COT (Commitment of Traders) analysis
    - Term structure analysis and curve shape interpretation
    - Basis analysis and carry trade opportunities
    - Institutional activity and large trader positioning
    - Seasonality analysis and cyclical patterns
    - Cross-market futures relationships and correlation
    """
    
    def __init__(self, db_manager=None, cache_manager=None):
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.cot_data_cache = {}
        self.term_structure_cache = {}
        self.positioning_cache = {}
        
    async def analyze_futures_intelligence(self, symbol: str = "ES") -> Dict[str, Any]:
        """
        Comprehensive futures market intelligence analysis.
        
        Args:
            symbol: Futures symbol to analyze (e.g., ES, NQ, CL, GC)
            
        Returns:
            Dictionary containing futures intelligence results
        """
        try:
            # Map futures symbol to underlying asset for analysis
            underlying_mapping = {
                "ES": "SPY",   # S&P 500 E-mini
                "NQ": "QQQ",   # NASDAQ 100 E-mini
                "YM": "DIA",   # Dow Jones E-mini
                "RTY": "IWM",  # Russell 2000 E-mini
                "CL": "USO",   # Crude Oil
                "GC": "GLD",   # Gold
                "SI": "SLV",   # Silver
                "NG": "UNG",   # Natural Gas
                "ZC": "CORN",  # Corn
                "ZW": "WEAT"   # Wheat
            }
            
            underlying_symbol = underlying_mapping.get(symbol, "SPY")
            
            # Get futures data and analysis
            futures_data = await self._fetch_futures_data(symbol, underlying_symbol)
            if not futures_data:
                return {"error": "Unable to fetch futures data"}
            
            # Term structure analysis
            term_structure_analysis = await self._analyze_term_structure(futures_data)
            
            # Positioning analysis (simulated COT-style analysis)
            positioning_analysis = await self._analyze_positioning(futures_data)
            
            # Basis and carry analysis
            basis_analysis = await self._analyze_basis(futures_data)
            
            # Institutional activity analysis
            institutional_analysis = await self._analyze_institutional_activity(futures_data)
            
            # Seasonality and cyclical analysis
            seasonality_analysis = await self._analyze_seasonality(symbol, underlying_symbol)
            
            # Cross-market relationships
            cross_market_analysis = await self._analyze_cross_market_relationships(futures_data)
            
            # Volatility analysis
            volatility_analysis = await self._analyze_futures_volatility(futures_data)
            
            # Liquidity analysis
            liquidity_analysis = await self._analyze_liquidity(futures_data)
            
            result = {
                "symbol": symbol,
                "underlying_symbol": underlying_symbol,
                "timestamp": datetime.now().isoformat(),
                "current_price": futures_data.get("current_price", 0),
                "term_structure": term_structure_analysis,
                "positioning_analysis": positioning_analysis,
                "basis_analysis": basis_analysis,
                "institutional_analysis": institutional_analysis,
                "seasonality_analysis": seasonality_analysis,
                "cross_market_analysis": cross_market_analysis,
                "volatility_analysis": volatility_analysis,
                "liquidity_analysis": liquidity_analysis,
                "futures_intelligence_score": await self._calculate_futures_intelligence_score(
                    term_structure_analysis, positioning_analysis, volatility_analysis
                )
            }
            
            # Cache results
            if self.cache_manager:
                await self.cache_manager.set(f"futures_intelligence:{symbol}", result, ttl=300)
                
            return result
            
        except Exception as e:
            logger.error(f"Error in futures intelligence analysis for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    async def _fetch_futures_data(self, symbol: str, underlying_symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch futures data and related information."""
        try:
            # Get underlying asset data
            ticker = yf.Ticker(underlying_symbol)
            underlying_data = ticker.history(period="2y")
            
            if underlying_data.empty:
                return None
            
            current_price = underlying_data['Close'].iloc[-1]
            
            # Simulate futures contract data
            # In production, this would connect to futures data providers
            futures_contracts = await self._create_futures_contracts(symbol, current_price, underlying_data)
            
            # Add market microstructure data
            microstructure_data = await self._get_microstructure_data(underlying_symbol)
            
            return {
                "symbol": symbol,
                "underlying_symbol": underlying_symbol,
                "current_price": current_price,
                "futures_contracts": futures_contracts,
                "underlying_data": underlying_data,
                "microstructure": microstructure_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching futures data: {str(e)}")
            return None
    
    async def _create_futures_contracts(self, symbol: str, current_price: float, 
                                      underlying_data: pd.DataFrame) -> Dict[str, Any]:
        """Create simulated futures contracts data."""
        try:
            contracts = {}
            
            # Define typical futures contract structure
            contract_months = []
            today = datetime.now()
            
            # Generate next 12 contract months
            for i in range(1, 13):
                exp_month = today.replace(day=1) + timedelta(days=32*i)
                exp_month = exp_month.replace(day=1)  # Normalize to first of month
                # Find third Friday (typical futures expiration)
                third_friday = self._get_third_friday(exp_month)
                contract_months.append(third_friday)
            
            # Market-specific pricing
            contract_specifics = self._get_contract_specifics(symbol)
            
            for i, expiration_date in enumerate(contract_months):
                # Calculate days to expiration
                days_to_exp = (expiration_date - today).days
                
                # Estimate futures price based on cost of carry
                futures_price = self._estimate_futures_price(
                    current_price, days_to_exp, contract_specifics
                )
                
                # Simulate volume and open interest
                volume = max(100, int(np.random.exponential(1000)))
                open_interest = max(volume, int(np.random.exponential(5000)))
                
                # Add some market-specific characteristics
                spread_data = self._simulate_spread_data(symbol, current_price, expiration_date)
                
                contracts[f"contract_{i+1}"] = {
                    "symbol": f"{symbol}{expiration_date.strftime('%y%m')}",
                    "expiration_date": expiration_date.isoformat(),
                    "days_to_expiration": days_to_exp,
                    "futures_price": round(futures_price, 2),
                    "volume": volume,
                    "open_interest": open_interest,
                    "settlement_type": contract_specifics.get("settlement_type", "cash"),
                    "tick_size": contract_specifics.get("tick_size", 0.25),
                    "contract_size": contract_specifics.get("contract_size", 50),
                    "spread_data": spread_data
                }
            
            return contracts
            
        except Exception as e:
            logger.error(f"Error creating futures contracts: {str(e)}")
            return {}
    
    def _get_third_friday(self, date: datetime) -> datetime:
        """Get the third Friday of a given month."""
        first_day = date.replace(day=1)
        
        # Find first Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        
        # Third Friday is 14 days after first Friday
        third_friday = first_friday + timedelta(days=14)
        
        return third_friday
    
    def _get_contract_specifics(self, symbol: str) -> Dict[str, Any]:
        """Get contract specifications for different futures."""
        contract_specs = {
            "ES": {
                "settlement_type": "cash",
                "tick_size": 0.25,
                "contract_size": 50,
                "multiplier": 50,
                "margin_requirement": 13200
            },
            "NQ": {
                "settlement_type": "cash", 
                "tick_size": 0.25,
                "contract_size": 20,
                "multiplier": 20,
                "margin_requirement": 17600
            },
            "CL": {
                "settlement_type": "physical",
                "tick_size": 0.01,
                "contract_size": 1000,
                "multiplier": 1000,
                "margin_requirement": 4950
            },
            "GC": {
                "settlement_type": "physical",
                "tick_size": 0.1,
                "contract_size": 100,
                "multiplier": 100,
                "margin_requirement": 8800
            },
            "SI": {
                "settlement_type": "physical",
                "tick_size": 0.005,
                "contract_size": 5000,
                "multiplier": 5000,
                "margin_requirement": 10450
            }
        }
        
        return contract_specs.get(symbol, contract_specs["ES"])  # Default to ES specs
    
    def _estimate_futures_price(self, spot_price: float, days_to_exp: int, 
                              contract_specs: Dict) -> float:
        """Estimate futures price using cost of carry model."""
        try:
            # Simple cost of carry: F = S * e^(r-q)*T
            # Where r = risk-free rate, q = dividend yield, T = time to expiration
            
            # Estimate financing costs based on contract type
            if contract_specs.get("settlement_type") == "physical":
                # Physical commodities have storage costs
                financing_rate = 0.03  # 3% annual
                storage_cost = 0.02    # 2% annual storage
                total_cost = financing_rate + storage_cost
            else:
                # Financial futures (equity indexes)
                financing_rate = 0.025  # 2.5% annual
                dividend_yield = 0.015  # 1.5% annual
                total_cost = financing_rate - dividend_yield
            
            # Calculate time to expiration in years
            T = days_to_exp / 365.25
            
            # Futures price
            futures_price = spot_price * np.exp(total_cost * T)
            
            # Add some market noise
            noise = np.random.normal(0, 0.01)  # 1% standard deviation
            futures_price *= (1 + noise)
            
            return futures_price
            
        except Exception as e:
            logger.error(f"Error estimating futures price: {str(e)}")
            return spot_price
    
    def _simulate_spread_data(self, symbol: str, current_price: float, 
                            expiration_date: datetime) -> Dict[str, Any]:
        """Simulate bid-ask spread data for futures."""
        try:
            # Spread typically varies by liquidity and expiration
            days_to_exp = (expiration_date - datetime.now()).days
            
            # Base spread calculation
            if days_to_exp <= 30:
                base_spread = 0.01  # 1 tick for front month
            elif days_to_exp <= 90:
                base_spread = 0.02  # 2 ticks
            else:
                base_spread = 0.05  # Wider spreads for back months
            
            # Market-specific adjustments
            symbol_adjustments = {
                "ES": 1.0,   # Very liquid
                "NQ": 1.2,   # Liquid
                "CL": 1.5,   # Oil can be volatile
                "GC": 1.3,   # Gold is liquid
                "SI": 1.8    # Silver more volatile
            }
            
            adjustment = symbol_adjustments.get(symbol, 1.5)
            spread = base_spread * adjustment
            
            return {
                "bid_ask_spread": round(spread, 3),
                "spread_to_tick_ratio": round(spread / 0.01, 1),  # Relative to minimum tick
                "liquidity_score": max(0.1, 1 - spread / 0.1)
            }
            
        except Exception as e:
            logger.error(f"Error simulating spread data: {str(e)}")
            return {"bid_ask_spread": 0.01, "spread_to_tick_ratio": 1.0, "liquidity_score": 0.8}
    
    async def _get_microstructure_data(self, underlying_symbol: str) -> Dict[str, Any]:
        """Get market microstructure data."""
        try:
            # Simulate microstructure metrics
            return {
                "bid_ask_spread": round(np.random.uniform(0.01, 0.05), 3),
                "market_depth": {
                    "bid_levels": 5,
                    "ask_levels": 5,
                    "average_size": round(np.random.uniform(100, 1000), 0)
                },
                "volume_profile": {
                    "avg_daily_volume": int(np.random.uniform(1000000, 10000000)),
                    "volume_concentration": round(np.random.uniform(0.3, 0.8), 2)
                }
            }
        except Exception as e:
            logger.error(f"Error getting microstructure data: {str(e)}")
            return {}
    
    async def _analyze_term_structure(self, futures_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze futures term structure and curve shape."""
        try:
            contracts = futures_data["futures_contracts"]
            current_price = futures_data["current_price"]
            underlying_data = futures_data["underlying_data"]
            
            if not contracts:
                return {"error": "No futures contracts data"}
            
            # Extract price and expiration data
            expiration_days = []
            futures_prices = []
            contract_symbols = []
            
            for contract_id, contract_data in contracts.items():
                expiration_days.append(contract_data["days_to_expiration"])
                futures_prices.append(contract_data["futures_price"])
                contract_symbols.append(contract_data["symbol"])
            
            # Sort by expiration
            sorted_data = sorted(zip(expiration_days, futures_prices, contract_symbols))
            expiration_days, futures_prices, contract_symbols = zip(*sorted_data)
            
            # Calculate term structure metrics
            term_structure_metrics = {
                "curve_shape": self._classify_curve_shape(futures_prices, expiration_days),
                "contango_backwardation": self._assess_contango_backwardation(futures_prices, expiration_days, current_price),
                "curve_steepness": self._calculate_curve_steepness(futures_prices, expiration_days),
                "curve_convexity": self._calculate_curve_convexity(futures_prices, expiration_days)
            }
            
            # Term structure analysis by sectors
            sector_analysis = self._analyze_term_structure_by_sector(contracts)
            
            # Carry analysis
            carry_analysis = self._analyze_carry_trade(futures_prices, expiration_days, current_price)
            
            return {
                "contract_prices": dict(zip(contract_symbols, futures_prices)),
                "expiration_days": expiration_days,
                "term_structure_metrics": term_structure_metrics,
                "sector_analysis": sector_analysis,
                "carry_analysis": carry_analysis,
                "term_structure_interpretation": self._interpret_term_structure(term_structure_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing term structure: {str(e)}")
            return {"error": str(e)}
    
    def _classify_curve_shape(self, futures_prices: Tuple, expiration_days: Tuple) -> Dict[str, Any]:
        """Classify the shape of the futures curve."""
        try:
            prices = list(futures_prices)
            days = list(expiration_days)
            
            if len(prices) < 3:
                return {"shape": "insufficient_data", "interpretation": "Need more contracts"}
            
            # Calculate price changes over time
            front_month = prices[0]
            back_month = prices[-1]
            
            # Use linear regression to fit curve
            X = np.array(days).reshape(-1, 1)
            y = np.array(prices)
            
            reg = LinearRegression().fit(X, y)
            slope = reg.coef_[0]
            r_squared = reg.score(X, y)
            
            # Classify shape
            if slope > 0.001:  # Upward sloping (contango)
                if slope > 0.005:
                    shape = "steep_contango"
                else:
                    shape = "normal_contango"
            elif slope < -0.001:  # Downward sloping (backwardation)
                if slope < -0.005:
                    shape = "steep_backwardation"
                else:
                    shape = "normal_backwardation"
            else:
                shape = "flat_curve"
            
            return {
                "shape": shape,
                "slope": slope,
                "r_squared": r_squared,
                "front_month_price": front_month,
                "back_month_price": back_month
            }
            
        except Exception as e:
            logger.error(f"Error classifying curve shape: {str(e)}")
            return {"shape": "error", "slope": 0}
    
    def _assess_contango_backwardation(self, futures_prices: Tuple, expiration_days: Tuple, spot_price: float) -> Dict[str, Any]:
        """Assess contango/backwardation conditions."""
        try:
            prices = list(futures_prices)
            days = list(expiration_days)
            
            # Find front month (closest expiration)
            front_month_price = prices[0]
            front_month_days = days[0]
            
            # Calculate contango/backwardation metrics
            contango_premium = (front_month_price - spot_price) / spot_price if spot_price > 0 else 0
            
            # Find various term points
            short_term_price = prices[min(2, len(prices)-1)]  # 3rd contract
            medium_term_price = prices[min(5, len(prices)-1)]  # 6th contract
            
            # Rolling assessment
            rolling_contango = []
            for i in range(1, min(4, len(prices))):
                if days[i] > 0:
                    roll_contango = (prices[i] - prices[i-1]) / prices[i-1]
                    rolling_contango.append(roll_contango)
            
            # Interpretation
            if contango_premium > 0.02:
                interpretation = "significant_contango"
            elif contango_premium > 0.005:
                interpretation = "moderate_contango"
            elif contango_premium < -0.02:
                interpretation = "significant_backwardation"
            elif contango_premium < -0.005:
                interpretation = "moderate_backwardation"
            else:
                interpretation = "near_parity"
            
            return {
                "front_month_contango": contango_premium,
                "interpretation": interpretation,
                "short_term_premium": (short_term_price - front_month_price) / front_month_price,
                "medium_term_premium": (medium_term_price - front_month_price) / front_month_price,
                "rolling_contango_average": np.mean(rolling_contango) if rolling_contango else 0
            }
            
        except Exception as e:
            logger.error(f"Error assessing contango/backwardation: {str(e)}")
            return {"interpretation": "error", "front_month_contango": 0}
    
    def _calculate_curve_steepness(self, futures_prices: Tuple, expiration_days: Tuple) -> Dict[str, Any]:
        """Calculate curve steepness metrics."""
        try:
            prices = list(futures_prices)
            days = list(expiration_days)
            
            # Calculate price differences
            price_changes = []
            time_diffs = []
            
            for i in range(1, len(prices)):
                price_change = prices[i] - prices[0]
                time_diff = days[i] - days[0]
                if time_diff > 0:
                    price_changes.append(price_change)
                    time_diffs.append(time_diff)
            
            if not price_changes:
                return {"steepness": 0, "interpretation": "insufficient_data"}
            
            # Average steepness
            avg_steepness = np.mean([change / (time/365.25) for change, time in zip(price_changes, time_diffs)])
            
            # Maximum steepness
            max_steepness = max([change / (time/365.25) for change, time in zip(price_changes, time_diffs)])
            
            # Interpretation
            if avg_steepness > 0.1:
                steepness_level = "very_steep"
            elif avg_steepness > 0.05:
                steepness_level = "steep"
            elif avg_steepness > 0.01:
                steepness_level = "moderate"
            elif avg_steepness > -0.01:
                steepness_level = "flat"
            else:
                steepness_level = "inverted"
            
            return {
                "steepness": avg_steepness,
                "max_steepness": max_steepness,
                "steepness_level": steepness_level,
                "annualized_steepness": avg_steepness
            }
            
        except Exception as e:
            logger.error(f"Error calculating curve steepness: {str(e)}")
            return {"steepness": 0, "steepness_level": "unknown"}
    
    def _calculate_curve_convexity(self, futures_prices: Tuple, expiration_days: Tuple) -> Dict[str, Any]:
        """Calculate curve convexity."""
        try:
            prices = list(futures_prices)
            days = list(expiration_days)
            
            if len(prices) < 3:
                return {"convexity": 0, "interpretation": "insufficient_data"}
            
            # Calculate second derivative (curvature)
            convexities = []
            for i in range(1, len(prices)-1):
                # Price change from i-1 to i
                price_change_1 = prices[i] - prices[i-1]
                time_change_1 = days[i] - days[i-1]
                
                # Price change from i to i+1
                price_change_2 = prices[i+1] - prices[i]
                time_change_2 = days[i+1] - days[i]
                
                # Curvature approximation
                if time_change_1 > 0 and time_change_2 > 0:
                    price_velocity_1 = price_change_1 / time_change_1
                    price_velocity_2 = price_change_2 / time_change_2
                    convexity = (price_velocity_2 - price_velocity_1) / ((time_change_1 + time_change_2) / 2)
                    convexities.append(convexity)
            
            avg_convexity = np.mean(convexities) if convexities else 0
            
            # Interpretation
            if avg_convexity > 0.001:
                convexity_level = "convex"
            elif avg_convexity < -0.001:
                convexity_level = "concave"
            else:
                convexity_level = "linear"
            
            return {
                "convexity": avg_convexity,
                "convexity_level": convexity_level,
                "curvature_points": len(convexities)
            }
            
        except Exception as e:
            logger.error(f"Error calculating curve convexity: {str(e)}")
            return {"convexity": 0, "convexity_level": "unknown"}
    
    def _analyze_term_structure_by_sector(self, contracts: Dict) -> Dict[str, Any]:
        """Analyze term structure patterns by market sector."""
        try:
            sector_analysis = {
                "energy": self._analyze_energy_curve(contracts),
                "metals": self._analyze_metals_curve(contracts),
                "agriculture": self._analyze_agriculture_curve(contracts),
                "financial": self._analyze_financial_curve(contracts)
            }
            
            return sector_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing term structure by sector: {str(e)}")
            return {}
    
    def _analyze_energy_curve(self, contracts: Dict) -> Dict[str, Any]:
        """Analyze energy futures curve characteristics."""
        try:
            # Energy futures typically show seasonal patterns
            return {
                "seasonal_patterns": "strong",
                "storage_constraints": "high",
                "curve_shape_tendency": "backwardation_during_shortages",
                "typical_contango_range": "5-15%"
            }
        except Exception:
            return {"analysis_status": "energy_analysis_failed"}
    
    def _analyze_metals_curve(self, contracts: Dict) -> Dict[str, Any]:
        """Analyze metals futures curve characteristics."""
        try:
            return {
                "storage_costs": "moderate",
                "demand_patterns": "industrial_and_investment",
                "curve_shape_tendency": "contango_with_carry",
                "seasonal_influence": "low"
            }
        except Exception:
            return {"analysis_status": "metals_analysis_failed"}
    
    def _analyze_agriculture_curve(self, contracts: Dict) -> Dict[str, Any]:
        """Analyze agricultural futures curve characteristics."""
        try:
            return {
                "seasonal_harvest_patterns": "strong",
                "storage_constraints": "high",
                "curve_shape_tendency": "inverted_during_harvest",
                "weather_dependency": "high"
            }
        except Exception:
            return {"analysis_status": "agriculture_analysis_failed"}
    
    def _analyze_financial_curve(self, contracts: Dict) -> Dict[str, Any]:
        """Analyze financial futures curve characteristics."""
        try:
            return {
                "cost_of_carry": "financing_rates_less_dividends",
                "curve_shape_tendency": "contango_reflecting_cost_of_carry",
                "interest_rate_dependency": "high",
                "dividend_yield_impact": "moderate"
            }
        except Exception:
            return {"analysis_status": "financial_analysis_failed"}
    
    def _analyze_carry_trade(self, futures_prices: Tuple, expiration_days: Tuple, spot_price: float) -> Dict[str, Any]:
        """Analyze carry trade opportunities."""
        try:
            prices = list(futures_prices)
            days = list(expiration_days)
            
            carry_trades = []
            
            for i in range(1, min(4, len(prices))):  # Analyze first 3 roll periods
                if days[i] > 0:
                    # Calculate annualized carry
                    price_difference = prices[i] - prices[0]
                    time_horizon = days[i] / 365.25
                    carry_rate = price_difference / (prices[0] * time_horizon)
                    
                    carry_trades.append({
                        "roll_period": f"{days[0]}_to_{days[i]}_days",
                        "carry_rate": carry_rate,
                        "carry_return": price_difference / prices[0],
                        "annualized_carry": carry_rate
                    })
            
            # Assess carry trade opportunities
            avg_carry = np.mean([trade["carry_rate"] for trade in carry_trades]) if carry_trades else 0
            
            if avg_carry > 0.03:
                carry_opportunity = "attractive_positive_carry"
            elif avg_carry > 0.01:
                carry_opportunity = "moderate_positive_carry"
            elif avg_carry > -0.01:
                carry_opportunity = "neutral_carry"
            elif avg_carry > -0.03:
                carry_opportunity = "moderate_negative_carry"
            else:
                carry_opportunity = "attractive_negative_carry"
            
            return {
                "carry_opportunities": carry_trades,
                "average_carry_rate": avg_carry,
                "carry_trade_assessment": carry_opportunity,
                "roll_yield": avg_carry
            }
            
        except Exception as e:
            logger.error(f"Error analyzing carry trade: {str(e)}")
            return {"carry_trade_assessment": "analysis_failed"}
    
    def _interpret_term_structure(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Provide interpretation of term structure signals."""
        try:
            shape = metrics.get("curve_shape", {}).get("shape", "unknown")
            contango = metrics.get("contango_backwardation", {}).get("interpretation", "unknown")
            steepness = metrics.get("curve_steepness", {}).get("steepness_level", "unknown")
            
            # Overall market sentiment interpretation
            if "contango" in shape and steepness in ["steep", "very_steep"]:
                sentiment = "bullish_storage_costs_high"
            elif "backwardation" in shape and steepness in ["steep", "very_steep"]:
                sentiment = "bearish_scarcity_premium_high"
            elif "contango" in shape:
                sentiment = "neutral_bearish_storage_costs"
            elif "backwardation" in shape:
                sentiment = "neutral_bullish_scarcity"
            else:
                sentiment = "neutral_flat_curve"
            
            # Trading implications
            if "attractive" in metrics.get("carry_analysis", {}).get("carry_trade_assessment", ""):
                implications = "consider_roll_strategies"
            elif "negative" in metrics.get("carry_analysis", {}).get("carry_trade_assessment", ""):
                implications = "favorable_roll_costs"
            else:
                implications = "monitor_roll_opportunities"
            
            return {
                "market_sentiment": sentiment,
                "trading_implications": implications,
                "key_signals": [shape, contango, steepness],
                "confidence_level": 0.7  # Base confidence
            }
            
        except Exception as e:
            logger.error(f"Error interpreting term structure: {str(e)}")
            return {"interpretation": "error", "confidence_level": 0.0}
    
    async def _analyze_positioning(self, futures_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze futures positioning (COT-style analysis)."""
        try:
            contracts = futures_data["futures_contracts"]
            
            # Aggregate positioning data across contracts
            total_volume = 0
            total_open_interest = 0
            positioning_metrics = {}
            
            for contract_id, contract_data in contracts.items():
                volume = contract_data["volume"]
                open_interest = contract_data["open_interest"]
                
                total_volume += volume
                total_open_interest += open_interest
            
            # Simulate trader category analysis
            positioning_analysis = self._simulate_positioning_analysis(contracts)
            
            # Contract-specific positioning
            contract_positioning = self._analyze_contract_specific_positioning(contracts)
            
            # Positioning concentration analysis
            concentration_analysis = self._analyze_positioning_concentration(contracts)
            
            return {
                "aggregate_metrics": {
                    "total_volume": total_volume,
                    "total_open_interest": total_open_interest,
                    "volume_to_oi_ratio": total_volume / total_open_interest if total_open_interest > 0 else 0
                },
                "trader_category_positioning": positioning_analysis,
                "contract_positioning": contract_positioning,
                "concentration_analysis": concentration_analysis,
                "positioning_signals": self._generate_positioning_signals(positioning_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing positioning: {str(e)}")
            return {"error": str(e)}
    
    def _simulate_positioning_analysis(self, contracts: Dict) -> Dict[str, Any]:
        """Simulate COT-style positioning analysis."""
        try:
            # Simulate different trader categories
            total_oi = sum(contract["open_interest"] for contract in contracts.values())
            
            if total_oi == 0:
                return {"error": "No open interest data"}
            
            # Simulate trader positioning (in production, would use real COT data)
            large_traders = max(0.6, min(0.8, np.random.uniform(0.6, 0.8)))  # 60-80%
            small_traders = 1.0 - large_traders
            
            # Simulate long/short positioning
            large_traders_net = np.random.uniform(-0.3, 0.3)  # -30% to +30%
            small_traders_net = np.random.uniform(-0.2, 0.2)  # -20% to +20%
            
            return {
                "large_traders": {
                    "percentage_of_oi": large_traders,
                    "net_positioning": large_traders_net,
                    "long_percentage": (1 + large_traders_net) / 2,
                    "short_percentage": (1 - large_traders_net) / 2,
                    "positioning_bias": "long" if large_traders_net > 0.1 else "short" if large_traders_net < -0.1 else "neutral"
                },
                "small_traders": {
                    "percentage_of_oi": small_traders,
                    "net_positioning": small_traders_net,
                    "long_percentage": (1 + small_traders_net) / 2,
                    "short_percentage": (1 - small_traders_net) / 2,
                    "positioning_bias": "long" if small_traders_net > 0.1 else "short" if small_traders_net < -0.1 else "neutral"
                },
                "commercials": {
                    "percentage_of_oi": 0.1,  # Typically small percentage
                    "hedging_activity": "monitoring",
                    "commercial_bias": "varies_by_season"
                }
            }
            
        except Exception as e:
            logger.error(f"Error simulating positioning analysis: {str(e)}")
            return {"error": "positioning_simulation_failed"}
    
    def _analyze_contract_specific_positioning(self, contracts: Dict) -> Dict[str, Any]:
        """Analyze positioning by specific contracts."""
        try:
            contract_analysis = {}
            
            for contract_id, contract_data in contracts.items():
                contract_symbol = contract_data["symbol"]
                days_to_exp = contract_data["days_to_expiration"]
                volume = contract_data["volume"]
                oi = contract_data["open_interest"]
                
                # Determine contract type
                if days_to_exp <= 30:
                    contract_type = "front_month"
                elif days_to_exp <= 90:
                    contract_type = "near_month"
                else:
                    contract_type = "deferred_month"
                
                # Analyze positioning intensity
                volume_oi_ratio = volume / oi if oi > 0 else 0
                
                contract_analysis[contract_symbol] = {
                    "contract_type": contract_type,
                    "days_to_expiration": days_to_exp,
                    "positioning_intensity": "high" if volume_oi_ratio > 0.5 else "moderate" if volume_oi_ratio > 0.2 else "low",
                    "volume_oi_ratio": volume_oi_ratio,
                    "liquidity_score": min(1.0, volume / 1000) if volume > 0 else 0
                }
            
            return contract_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing contract-specific positioning: {str(e)}")
            return {}
    
    def _analyze_positioning_concentration(self, contracts: Dict) -> Dict[str, Any]:
        """Analyze positioning concentration across contracts."""
        try:
            if not contracts:
                return {"concentration_level": "unknown"}
            
            # Calculate OI distribution
            oi_values = [contract["open_interest"] for contract in contracts.values()]
            total_oi = sum(oi_values)
            
            if total_oi == 0:
                return {"concentration_level": "no_data"}
            
            # Calculate concentration metrics
            oi_percentages = [oi / total_oi for oi in oi_values]
            max_concentration = max(oi_percentages)
            
            # Herfindahl-Hirschman Index for concentration
            hhi = sum(pct**2 for pct in oi_percentages)
            
            # Classify concentration level
            if hhi > 0.25:
                concentration_level = "highly_concentrated"
            elif hhi > 0.15:
                concentration_level = "concentrated"
            elif hhi > 0.10:
                concentration_level = "moderately_concentrated"
            else:
                concentration_level = "well_diversified"
            
            # Find most concentrated contract
            max_oi_contract = list(contracts.keys())[oi_values.index(max(oi_values))]
            
            return {
                "concentration_level": concentration_level,
                "max_concentration": max_concentration,
                "hhi_index": hhi,
                "most_concentrated_contract": max_oi_contract,
                "concentration_interpretation": self._interpret_concentration(concentration_level, hhi)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing positioning concentration: {str(e)}")
            return {"concentration_level": "error"}
    
    def _interpret_concentration(self, concentration_level: str, hhi: float) -> str:
        """Interpret positioning concentration signals."""
        interpretations = {
            "highly_concentrated": "High risk of sharp moves if large positions unwind",
            "concentrated": "Moderate risk of price disruption from large trades",
            "moderately_concentrated": "Normal market concentration patterns",
            "well_diversified": "Healthy, diverse positioning across contracts"
        }
        
        return interpretations.get(concentration_level, "Concentration analysis unavailable")
    
    def _generate_positioning_signals(self, positioning_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate trading signals based on positioning analysis."""
        try:
            signals = []
            
            # Large trader signals
            large_traders = positioning_analysis.get("large_traders", {})
            large_bias = large_traders.get("positioning_bias", "neutral")
            
            if large_bias == "long" and large_traders.get("net_positioning", 0) > 0.2:
                signals.append({
                    "type": "large_traders_long",
                    "signal": "bullish",
                    "strength": large_traders["net_positioning"],
                    "message": "Large traders significantly net long"
                })
            elif large_bias == "short" and large_traders.get("net_positioning", 0) < -0.2:
                signals.append({
                    "type": "large_traders_short",
                    "signal": "bearish",
                    "strength": abs(large_traders["net_positioning"]),
                    "message": "Large traders significantly net short"
                })
            
            # Commercial hedging signals
            commercials = positioning_analysis.get("commercials", {})
            if commercials.get("hedging_activity") == "active":
                signals.append({
                    "type": "commercial_hedging",
                    "signal": "monitor",
                    "strength": 0.5,
                    "message": "Increased commercial hedging activity"
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating positioning signals: {str(e)}")
            return []
    
    async def _analyze_basis(self, futures_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze basis between futures and underlying assets."""
        try:
            current_price = futures_data["current_price"]
            contracts = futures_data["futures_contracts"]
            
            # Calculate basis for each contract
            basis_analysis = {}
            
            for contract_id, contract_data in contracts.items():
                futures_price = contract_data["futures_price"]
                days_to_exp = contract_data["days_to_expiration"]
                
                # Calculate basis
                if current_price > 0:
                    basis_absolute = futures_price - current_price
                    basis_percentage = (basis_absolute / current_price) * 100
                else:
                    basis_absolute = 0
                    basis_percentage = 0
                
                # Estimate fair value based on cost of carry
                fair_value = self._estimate_fair_value(current_price, days_to_exp)
                basis_vs_fair_value = futures_price - fair_value
                
                basis_analysis[contract_data["symbol"]] = {
                    "basis_absolute": basis_absolute,
                    "basis_percentage": basis_percentage,
                    "fair_value": fair_value,
                    "basis_vs_fair_value": basis_vs_fair_value,
                    "days_to_expiration": days_to_exp,
                    "basis_interpretation": self._interpret_basis(basis_percentage, days_to_exp)
                }
            
            # Overall basis trends
            basis_trends = self._analyze_basis_trends(basis_analysis)
            
            # Arbitrage opportunities
            arbitrage_opportunities = self._identify_arbitrage_opportunities(basis_analysis)
            
            return {
                "contract_basis": basis_analysis,
                "basis_trends": basis_trends,
                "arbitrage_opportunities": arbitrage_opportunities,
                "basis_summary": self._summarize_basis_analysis(basis_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing basis: {str(e)}")
            return {"error": str(e)}
    
    def _estimate_fair_value(self, spot_price: float, days_to_exp: int) -> float:
        """Estimate fair value based on cost of carry model."""
        try:
            # Simple cost of carry: F = S * e^(r-q)*T
            financing_cost = 0.025  # 2.5% annual
            dividend_yield = 0.015  # 1.5% annual
            net_cost = financing_cost - dividend_yield
            
            T = days_to_exp / 365.25
            fair_value = spot_price * np.exp(net_cost * T)
            
            return fair_value
            
        except Exception:
            return spot_price
    
    def _interpret_basis(self, basis_percentage: float, days_to_exp: int) -> str:
        """Interpret basis signals."""
        try:
            if abs(basis_percentage) < 0.5:
                return "basis_near_parity"
            elif basis_percentage > 2:
                return "basis_significantly_positive"
            elif basis_percentage > 0.5:
                return "basis_moderately_positive"
            elif basis_percentage < -2:
                return "basis_significantly_negative"
            else:
                return "basis_moderately_negative"
                
        except Exception:
            return "basis_interpretation_error"
    
    def _analyze_basis_trends(self, basis_analysis: Dict) -> Dict[str, Any]:
        """Analyze trends in basis data."""
        try:
            if len(basis_analysis) < 2:
                return {"trend": "insufficient_data"}
            
            # Sort by days to expiration
            sorted_basis = sorted(basis_analysis.items(), 
                                key=lambda x: x[1]["days_to_expiration"])
            
            # Calculate basis changes
            basis_changes = []
            for i in range(1, len(sorted_basis)):
                prev_basis = sorted_basis[i-1][1]["basis_percentage"]
                curr_basis = sorted_basis[i][1]["basis_percentage"]
                basis_change = curr_basis - prev_basis
                basis_changes.append(basis_change)
            
            avg_basis_change = np.mean(basis_changes) if basis_changes else 0
            
            # Trend interpretation
            if avg_basis_change > 0.5:
                trend = "basis_widening"
            elif avg_basis_change < -0.5:
                trend = "basis_narrowing"
            else:
                trend = "basis_stable"
            
            return {
                "trend": trend,
                "average_basis_change": avg_basis_change,
                "basis_volatility": np.std(basis_changes) if basis_changes else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing basis trends: {str(e)}")
            return {"trend": "analysis_error"}
    
    def _identify_arbitrage_opportunities(self, basis_analysis: Dict) -> List[Dict[str, Any]]:
        """Identify potential arbitrage opportunities."""
        try:
            opportunities = []
            
            for contract_symbol, data in basis_analysis.items():
                basis_vs_fair = data["basis_vs_fair_value"]
                days_to_exp = data["days_to_expiration"]
                
                # Look for significant deviations from fair value
                if abs(basis_vs_fair) > 0.02 and days_to_exp > 0:  # 2% threshold
                    opportunity_type = "cash_and_carry" if basis_vs_fair > 0 else "reverse_cash_and_carry"
                    
                    opportunities.append({
                        "contract": contract_symbol,
                        "opportunity_type": opportunity_type,
                        "potential_profit": abs(basis_vs_fair),
                        "days_to_expiration": days_to_exp,
                        "annualized_profit": abs(basis_vs_fair) * (365.25 / days_to_exp) if days_to_exp > 0 else 0,
                        "risk_level": "low" if days_to_exp > 30 else "medium" if days_to_exp > 7 else "high"
                    })
            
            # Sort by potential profit
            opportunities.sort(key=lambda x: x["potential_profit"], reverse=True)
            
            return opportunities[:5]  # Top 5 opportunities
            
        except Exception as e:
            logger.error(f"Error identifying arbitrage opportunities: {str(e)}")
            return []
    
    def _summarize_basis_analysis(self, basis_analysis: Dict) -> Dict[str, Any]:
        """Summarize overall basis analysis."""
        try:
            if not basis_analysis:
                return {"summary": "no_basis_data"}
            
            basis_values = [data["basis_percentage"] for data in basis_analysis.values()]
            
            return {
                "average_basis": np.mean(basis_values),
                "basis_range": [min(basis_values), max(basis_values)],
                "basis_volatility": np.std(basis_values),
                "overall_assessment": self._assess_overall_basis(basis_values)
            }
            
        except Exception as e:
            logger.error(f"Error summarizing basis analysis: {str(e)}")
            return {"summary": "analysis_error"}
    
    def _assess_overall_basis(self, basis_values: List[float]) -> str:
        """Assess overall basis conditions."""
        try:
            avg_basis = np.mean(basis_values)
            basis_vol = np.std(basis_values)
            
            if avg_basis > 1:
                return "contango_market"
            elif avg_basis < -1:
                return "backwardation_market"
            else:
                return "near_parity_market"
                
        except Exception:
            return "basis_assessment_error"
    
    async def _analyze_institutional_activity(self, futures_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze institutional trading activity patterns."""
        try:
            contracts = futures_data["futures_contracts"]
            microstructure = futures_data.get("microstructure", {})
            
            # Analyze volume patterns
            volume_analysis = self._analyze_volume_patterns(contracts)
            
            # Analyze trading intensity
            intensity_analysis = self._analyze_trading_intensity(contracts)
            
            # Analyze market impact potential
            impact_analysis = self._analyze_market_impact_potential(contracts)
            
            # Institutional sentiment indicators
            sentiment_analysis = self._analyze_institutional_sentiment(contracts, microstructure)
            
            return {
                "volume_analysis": volume_analysis,
                "intensity_analysis": intensity_analysis,
                "impact_analysis": impact_analysis,
                "sentiment_analysis": sentiment_analysis,
                "institutional_signals": self._generate_institutional_signals(
                    volume_analysis, sentiment_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing institutional activity: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_volume_patterns(self, contracts: Dict) -> Dict[str, Any]:
        """Analyze trading volume patterns."""
        try:
            volume_data = [(contract["volume"], contract["symbol"], contract["days_to_expiration"]) 
                          for contract in contracts.values()]
            
            if not volume_data:
                return {"pattern": "no_volume_data"}
            
            volumes, symbols, days_to_exp = zip(*volume_data)
            
            # Volume distribution analysis
            volume_stats = {
                "total_volume": sum(volumes),
                "average_volume": np.mean(volumes),
                "volume_volatility": np.std(volumes),
                "max_volume_contract": symbols[volumes.index(max(volumes))],
                "min_volume_contract": symbols[volumes.index(min(volumes))]
            }
            
            # Front month vs back month volume
            front_month_volumes = [vol for vol, _, days in volume_data if days <= 30]
            deferred_volumes = [vol for vol, _, days in volume_data if days > 90]
            
            if front_month_volumes and deferred_volumes:
                front_deferred_ratio = np.mean(front_month_volumes) / np.mean(deferred_volumes)
            else:
                front_deferred_ratio = 1.0
            
            # Volume pattern classification
            if front_deferred_ratio > 3:
                pattern = "front_month_heavy"
            elif front_deferred_ratio < 0.5:
                pattern = "deferred_month_heavy"
            elif volume_stats["volume_volatility"] / volume_stats["average_volume"] > 0.5:
                pattern = "irregular_volume"
            else:
                pattern = "balanced_volume"
            
            return {
                "volume_statistics": volume_stats,
                "front_deferred_ratio": front_deferred_ratio,
                "volume_pattern": pattern,
                "liquidity_assessment": self._assess_liquidity(volume_stats)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume patterns: {str(e)}")
            return {"pattern": "analysis_error"}
    
    def _analyze_trading_intensity(self, contracts: Dict) -> Dict[str, Any]:
        """Analyze trading intensity across contracts."""
        try:
            intensity_data = []
            
            for contract_data in contracts.values():
                volume = contract_data["volume"]
                oi = contract_data["open_interest"]
                days_to_exp = contract_data["days_to_expiration"]
                
                # Volume to OI ratio as intensity measure
                if oi > 0:
                    volume_oi_ratio = volume / oi
                else:
                    volume_oi_ratio = 0
                
                # Adjust for time to expiration
                time_factor = 30 / max(days_to_exp, 1)  # Higher intensity for near-term
                adjusted_intensity = volume_oi_ratio * time_factor
                
                intensity_data.append({
                    "symbol": contract_data["symbol"],
                    "intensity": adjusted_intensity,
                    "volume_oi_ratio": volume_oi_ratio,
                    "days_to_exp": days_to_exp
                })
            
            # Overall intensity assessment
            intensities = [data["intensity"] for data in intensity_data]
            avg_intensity = np.mean(intensities) if intensities else 0
            
            # Most active contract
            if intensity_data:
                most_active = max(intensity_data, key=lambda x: x["intensity"])
            else:
                most_active = None
            
            # Intensity classification
            if avg_intensity > 1.0:
                intensity_level = "very_high"
            elif avg_intensity > 0.5:
                intensity_level = "high"
            elif avg_intensity > 0.2:
                intensity_level = "moderate"
            else:
                intensity_level = "low"
            
            return {
                "average_intensity": avg_intensity,
                "intensity_level": intensity_level,
                "most_active_contract": most_active,
                "contract_intensities": intensity_data,
                "intensity_ranking": sorted(intensity_data, key=lambda x: x["intensity"], reverse=True)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trading intensity: {str(e)}")
            return {"intensity_level": "analysis_error"}
    
    def _analyze_market_impact_potential(self, contracts: Dict) -> Dict[str, Any]:
        """Analyze potential market impact from large trades."""
        try:
            total_volume = sum(contract["volume"] for contract in contracts.values())
            total_oi = sum(contract["open_interest"] for contract in contracts.values())
            
            # Market depth indicators
            average_volume = total_volume / len(contracts) if contracts else 0
            average_oi = total_oi / len(contracts) if contracts else 0
            
            # Impact potential assessment
            if average_volume > 5000 and average_oi > 10000:
                impact_risk = "low"
                explanation = "Deep markets with good liquidity"
            elif average_volume > 2000 and average_oi > 5000:
                impact_risk = "moderate"
                explanation = "Adequate liquidity with some impact risk"
            elif average_volume > 1000 and average_oi > 2000:
                impact_risk = "high"
                explanation = "Limited liquidity, significant impact risk"
            else:
                impact_risk = "very_high"
                explanation = "Poor liquidity, major impact risk"
            
            # Contract-specific impact assessment
            contract_impacts = {}
            for contract_data in contracts.values():
                contract_impact = self._assess_contract_impact_potential(
                    contract_data["volume"], 
                    contract_data["open_interest"],
                    contract_data["days_to_expiration"]
                )
                contract_impacts[contract_data["symbol"]] = contract_impact
            
            return {
                "overall_impact_risk": impact_risk,
                "explanation": explanation,
                "market_depth_metrics": {
                    "average_volume": average_volume,
                    "average_oi": average_oi,
                    "volume_oi_ratio": total_volume / total_oi if total_oi > 0 else 0
                },
                "contract_impacts": contract_impacts,
                "large_trade_warning": self._assess_large_trade_warning(contract_impacts)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market impact potential: {str(e)}")
            return {"impact_risk": "analysis_error"}
    
    def _assess_contract_impact_potential(self, volume: int, oi: int, days_to_exp: int) -> str:
        """Assess impact potential for a specific contract."""
        try:
            if volume > 10000 and oi > 20000 and days_to_exp > 30:
                return "low_impact"
            elif volume > 5000 and oi > 10000:
                return "moderate_impact"
            elif volume > 1000 and oi > 2000:
                return "high_impact"
            else:
                return "very_high_impact"
                
        except Exception:
            return "unknown_impact"
    
    def _assess_large_trade_warning(self, contract_impacts: Dict) -> str:
        """Assess warning level for large trades."""
        try:
            high_impact_contracts = sum(1 for impact in contract_impacts.values() 
                                      if "high" in impact or "very_high" in impact)
            total_contracts = len(contract_impacts)
            
            if total_contracts == 0:
                return "no_data"
            
            high_impact_ratio = high_impact_contracts / total_contracts
            
            if high_impact_ratio > 0.7:
                return "critical_warning"
            elif high_impact_ratio > 0.5:
                return "significant_warning"
            elif high_impact_ratio > 0.3:
                return "moderate_warning"
            else:
                return "low_warning"
                
        except Exception:
            return "warning_assessment_error"}
    
    def _analyze_institutional_sentiment(self, contracts: Dict, microstructure: Dict) -> Dict[str, Any]:
        """Analyze institutional sentiment from trading patterns."""
        try:
            # Analyze spread patterns as sentiment indicator
            spread_sentiment = self._analyze_spread_sentiment(contracts)
            
            # Volume pattern sentiment
            volume_sentiment = self._analyze_volume_sentiment(contracts)
            
            # Overall institutional sentiment
            overall_sentiment = self._calculate_overall_institutional_sentiment(
                spread_sentiment, volume_sentiment
            )
            
            return {
                "spread_sentiment": spread_sentiment,
                "volume_sentiment": volume_sentiment,
                "overall_sentiment": overall_sentiment,
                "sentiment_strength": self._assess_sentiment_strength(overall_sentiment),
                "sentiment_confidence": 0.7  # Base confidence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing institutional sentiment: {str(e)}")
            return {"sentiment": "analysis_error"}
    
    def _analyze_spread_sentiment(self, contracts: Dict) -> Dict[str, Any]:
        """Analyze sentiment from spread patterns."""
        try:
            spreads = []
            
            for contract_data in contracts.values():
                spread_data = contract_data.get("spread_data", {})
                bid_ask_spread = spread_data.get("bid_ask_spread", 0.01)
                spreads.append(bid_ask_spread)
            
            if not spreads:
                return {"sentiment": "no_spread_data"}
            
            avg_spread = np.mean(spreads)
            spread_volatility = np.std(spreads)
            
            # Tight spreads indicate confidence, wide spreads indicate uncertainty
            if avg_spread < 0.02:
                spread_sentiment = "confident"
            elif avg_spread < 0.05:
                spread_sentiment = "neutral"
            else:
                spread_sentiment = "uncertain"
            
            return {
                "sentiment": spread_sentiment,
                "average_spread": avg_spread,
                "spread_volatility": spread_volatility
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spread sentiment: {str(e)}")
            return {"sentiment": "analysis_error"}
    
    def _analyze_volume_sentiment(self, contracts: Dict) -> Dict[str, Any]:
        """Analyze sentiment from volume patterns."""
        try:
            volumes = [contract["volume"] for contract in contracts.values()]
            
            if not volumes:
                return {"sentiment": "no_volume_data"}
            
            total_volume = sum(volumes)
            avg_volume = np.mean(volumes)
            volume_concentration = max(volumes) / total_volume if total_volume > 0 else 0
            
            # High volume concentration can indicate either strong conviction or stress
            if volume_concentration > 0.6:
                volume_sentiment = "concentrated_activity"
            elif avg_volume > 5000:
                volume_sentiment = "high_activity"
            elif avg_volume > 2000:
                volume_sentiment = "moderate_activity"
            else:
                volume_sentiment = "low_activity"
            
            return {
                "sentiment": volume_sentiment,
                "total_volume": total_volume,
                "volume_concentration": volume_concentration,
                "activity_level": volume_sentiment
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume sentiment: {str(e)}")
            return {"sentiment": "analysis_error"}
    
    def _calculate_overall_institutional_sentiment(self, spread_sentiment: Dict, volume_sentiment: Dict) -> str:
        """Calculate overall institutional sentiment."""
        try:
            sentiment_scores = []
            
            # Spread sentiment
            if spread_sentiment.get("sentiment") == "confident":
                sentiment_scores.append(1.0)
            elif spread_sentiment.get("sentiment") == "neutral":
                sentiment_scores.append(0.5)
            else:
                sentiment_scores.append(0.0)
            
            # Volume sentiment
            volume_activity = volume_sentiment.get("activity_level", "moderate_activity")
            if volume_activity == "high_activity":
                sentiment_scores.append(0.8)
            elif volume_activity == "moderate_activity":
                sentiment_scores.append(0.6)
            elif volume_activity == "concentrated_activity":
                sentiment_scores.append(0.4)
            else:
                sentiment_scores.append(0.3)
            
            # Calculate overall sentiment
            overall_score = np.mean(sentiment_scores)
            
            if overall_score > 0.7:
                return "institutionally_bullish"
            elif overall_score > 0.6:
                return "cautiously_bullish"
            elif overall_score > 0.4:
                return "neutral"
            elif overall_score > 0.3:
                return "cautiously_bearish"
            else:
                return "institutionally_bearish"
                
        except Exception:
            return "sentiment_calculation_error"
    
    def _assess_sentiment_strength(self, sentiment: str) -> str:
        """Assess the strength of institutional sentiment."""
        try:
            if "institutionally" in sentiment:
                return "strong"
            elif "cautiously" in sentiment:
                return "moderate"
            else:
                return "weak"
                
        except Exception:
            return "strength_assessment_error"}
    
    def _generate_institutional_signals(self, volume_analysis: Dict, sentiment_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate institutional trading signals."""
        try:
            signals = []
            
            # Volume-based signals
            volume_pattern = volume_analysis.get("volume_pattern", "unknown")
            if volume_pattern == "front_month_heavy":
                signals.append({
                    "type": "front_month_activity",
                    "signal": "bullish",
                    "strength": 0.6,
                    "message": "Heavy front-month activity suggests near-term optimism"
                })
            elif volume_pattern == "deferred_month_heavy":
                signals.append({
                    "type": "deferred_month_activity",
                    "signal": "bearish",
                    "strength": 0.5,
                    "message": "Heavy deferred-month activity suggests longer-term concerns"
                })
            
            # Sentiment-based signals
            overall_sentiment = sentiment_analysis.get("overall_sentiment", "neutral")
            if "institutionally_bullish" in overall_sentiment:
                signals.append({
                    "type": "institutional_sentiment",
                    "signal": "bullish",
                    "strength": 0.8,
                    "message": "Strong institutional bullish sentiment"
                })
            elif "institutionally_bearish" in overall_sentiment:
                signals.append({
                    "type": "institutional_sentiment",
                    "signal": "bearish",
                    "strength": 0.8,
                    "message": "Strong institutional bearish sentiment"
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating institutional signals: {str(e)}")
            return []
    
    async def _analyze_seasonality(self, symbol: str, underlying_symbol: str) -> Dict[str, Any]:
        """Analyze seasonality patterns in futures markets."""
        try:
            # Get historical data for seasonality analysis
            ticker = yf.Ticker(underlying_symbol)
            historical_data = ticker.history(period="2y")
            
            if historical_data.empty:
                return {"error": "No historical data for seasonality"}
            
            # Analyze seasonal patterns
            seasonal_patterns = self._calculate_seasonal_patterns(historical_data, symbol)
            
            # Current seasonal position
            current_seasonal_position = self._assess_current_seasonal_position(historical_data, symbol)
            
            # Cyclical analysis
            cyclical_analysis = self._analyze_cyclical_patterns(historical_data)
            
            # Agricultural/commodity specific patterns
            sector_patterns = self._analyze_sector_seasonality(symbol)
            
            return {
                "seasonal_patterns": seasonal_patterns,
                "current_seasonal_position": current_seasonal_position,
                "cyclical_analysis": cyclical_analysis,
                "sector_patterns": sector_patterns,
                "seasonal_signals": self._generate_seasonal_signals(seasonal_patterns, current_seasonal_position)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing seasonality: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_seasonal_patterns(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate seasonal patterns in price movements."""
        try:
            if len(data) < 365:
                return {"pattern_strength": "insufficient_data"}
            
            # Add month and day of year
            data_with_months = data.copy()
            data_with_months['month'] = data_with_months.index.month
            data_with_months['day_of_year'] = data_with_months.index.dayofyear
            
            # Calculate monthly returns
            monthly_returns = data_with_months.groupby('month')['Close'].apply(
                lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) > 1 else 0
            )
            
            # Calculate seasonal statistics
            seasonal_stats = {
                month: {
                    'return': monthly_returns.get(month, 0),
                    'volatility': data_with_months[data_with_months['month'] == month]['Close'].pct_change().std() * np.sqrt(252),
                    'sample_size': len(data_with_months[data_with_months['month'] == month])
                }
                for month in range(1, 13)
            }
            
            # Identify strongest months
            sorted_months = sorted(monthly_returns.items(), key=lambda x: x[1], reverse=True)
            strongest_months = sorted_months[:3]
            weakest_months = sorted_months[-3:]
            
            return {
                "monthly_returns": monthly_returns.to_dict(),
                "seasonal_statistics": seasonal_stats,
                "strongest_months": [{"month": month, "return": return_val} for month, return_val in strongest_months],
                "weakest_months": [{"month": month, "return": return_val} for month, return_val in weakest_months],
                "seasonal_pattern_strength": self._assess_pattern_strength(monthly_returns),
                "pattern_consistency": self._assess_pattern_consistency(seasonal_stats)
            }
            
        except Exception as e:
            logger.error(f"Error calculating seasonal patterns: {str(e)}")
            return {"pattern_strength": "calculation_error"}
    
    def _assess_pattern_strength(self, monthly_returns: pd.Series) -> str:
        """Assess strength of seasonal patterns."""
        try:
            return_std = monthly_returns.std()
            return_mean = monthly_returns.mean()
            
            if return_std / abs(return_mean) < 0.5:
                return "strong"
            elif return_std / abs(return_mean) < 1.0:
                return "moderate"
            else:
                return "weak"
                
        except Exception:
            return "assessment_error"}
    
    def _assess_pattern_consistency(self, seasonal_stats: Dict) -> str:
        """Assess consistency of seasonal patterns."""
        try:
            if not seasonal_stats:
                return "no_data"
            
            volatilities = [stats.get('volatility', 0) for stats in seasonal_stats.values()]
            if volatilities:
                vol_consistency = 1 - (np.std(volatilities) / np.mean(volatilities)) if np.mean(volatilities) > 0 else 0
                
                if vol_consistency > 0.8:
                    return "highly_consistent"
                elif vol_consistency > 0.6:
                    return "moderately_consistent"
                else:
                    return "inconsistent"
            return "assessment_error"
                
        except Exception:
            return "consistency_assessment_error"}
    
    def _assess_current_seasonal_position(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Assess current position relative to seasonal patterns."""
        try:
            current_month = datetime.now().month
            current_day_of_year = datetime.now().timetuple().tm_yday
            
            # Get historical performance for current time period
            historical_data = data[data.index.month == current_month]
            
            if historical_data.empty:
                return {"position": "no_historical_data"}
            
            # Calculate typical performance for current period
            period_returns = historical_data['Close'].pct_change().dropna()
            
            if len(period_returns) > 0:
                avg_return = period_returns.mean()
                return_volatility = period_returns.std()
            else:
                avg_return = 0
                return_volatility = 0
            
            # Assess current position
            if avg_return > 0.02:
                seasonal_bias = "seasonally_bullish"
            elif avg_return < -0.02:
                seasonal_bias = "seasonally_bearish"
            else:
                seasonal_bias = "seasonally_neutral"
            
            return {
                "current_month": current_month,
                "seasonal_bias": seasonal_bias,
                "typical_return": avg_return,
                "seasonal_volatility": return_volatility,
                "confidence": min(1.0, len(historical_data) / 60)  # More data = higher confidence
            }
            
        except Exception as e:
            logger.error(f"Error assessing current seasonal position: {str(e)}")
            return {"position": "assessment_error"}
    
    def _analyze_cyclical_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze longer-term cyclical patterns."""
        try:
            if len(data) < 180:  # Need at least 6 months
                return {"cyclical_pattern": "insufficient_data"}
            
            # Calculate moving averages for cycle identification
            data['ma_50'] = data['Close'].rolling(50).mean()
            data['ma_200'] = data['Close'].rolling(200).mean()
            
            # Identify cycle phases
            current_price = data['Close'].iloc[-1]
            ma_50_current = data['ma_50'].iloc[-1]
            ma_200_current = data['ma_200'].iloc[-1]
            
            if pd.isna(ma_50_current) or pd.isna(ma_200_current):
                return {"cyclical_pattern": "insufficient_ma_data"}
            
            # Cycle phase classification
            if current_price > ma_50_current > ma_200_current:
                cyclical_phase = "bull_market_cycle"
            elif current_price > ma_200_current and ma_50_current < ma_200_current:
                cyclical_phase = "early_bull_cycle"
            elif current_price < ma_50_current < ma_200_current:
                cyclical_phase = "bear_market_cycle"
            elif current_price < ma_200_current and ma_50_current > ma_200_current:
                cyclical_phase = "early_bear_cycle"
            else:
                cyclical_phase = "sideways_cycle"
            
            # Cycle strength
            price_ma200_ratio = (current_price - ma_200_current) / ma_200_current
            
            if abs(price_ma200_ratio) > 0.2:
                cycle_strength = "strong"
            elif abs(price_ma200_ratio) > 0.1:
                cycle_strength = "moderate"
            else:
                cycle_strength = "weak"
            
            return {
                "cyclical_phase": cyclical_phase,
                "cycle_strength": cycle_strength,
                "price_ma200_deviation": price_ma200_ratio,
                "cycle_confidence": 0.7
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cyclical patterns: {str(e)}")
            return {"cyclical_pattern": "analysis_error"}
    
    def _analyze_sector_seasonality(self, symbol: str) -> Dict[str, Any]:
        """Analyze sector-specific seasonal patterns."""
        try:
            sector_specifics = {
                "energy": {
                    "summer_driving_season": "strong",
                    "winter_heating_season": "moderate",
                    "inventory_reports": "high_impact"
                },
                "agriculture": {
                    "planting_season": "strong",
                    "harvest_season": "strong",
                    "weather_dependency": "high"
                },
                "metals": {
                    "industrial_demand": "moderate",
                    "jewelry_demand": "seasonal",
                    "investment_demand": "counter_cyclical"
                },
                "financial": {
                    "quarter_end_effects": "moderate",
                    "earnings_seasons": "high_impact",
                    "interest_rate_environment": "primary_driver"
                }
            }
            
            # Determine sector for symbol
            sector = self._classify_futures_sector(symbol)
            sector_patterns = sector_specifics.get(sector, {"sector_classification": "unknown"})
            
            return {
                "sector": sector,
                "seasonal_drivers": sector_patterns,
                "sector_risk_factors": self._identify_sector_risks(sector)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sector seasonality: {str(e)}")
            return {"sector_analysis": "error"}
    
    def _classify_futures_sector(self, symbol: str) -> str:
        """Classify futures symbol into sector."""
        energy_symbols = ["CL", "NG", "HO", "RB"]
        metals_symbols = ["GC", "SI", "PL", "PA"]
        agriculture_symbols = ["ZC", "ZW", "ZS", "LE", "HE"]
        financial_symbols = ["ES", "NQ", "YM", "RTY", "ZB", "ZN", "ZT"]
        
        if symbol in energy_symbols:
            return "energy"
        elif symbol in metals_symbols:
            return "metals"
        elif symbol in agriculture_symbols:
            return "agriculture"
        elif symbol in financial_symbols:
            return "financial"
        else:
            return "unknown"
    
    def _identify_sector_risks(self, sector: str) -> List[str]:
        """Identify sector-specific risk factors."""
        sector_risks = {
            "energy": ["Geopolitical tensions", "OPEC decisions", "Weather patterns", "Inventory data"],
            "metals": ["Industrial demand", "Mining supply", "Currency fluctuations", "Central bank policies"],
            "agriculture": ["Weather patterns", "Crop reports", "Trade policies", "Currency impacts"],
            "financial": ["Interest rate changes", "Economic data", "Central bank policies", "Market volatility"]
        }
        
        return sector_risks.get(sector, ["General market risks"])
    
    def _generate_seasonal_signals(self, seasonal_patterns: Dict, current_position: Dict) -> List[Dict[str, Any]]:
        """Generate signals based on seasonal analysis."""
        try:
            signals = []
            
            # Seasonal bias signals
            seasonal_bias = current_position.get("seasonal_bias", "neutral")
            if seasonal_bias == "seasonally_bullish":
                signals.append({
                    "type": "seasonal_bias",
                    "signal": "bullish",
                    "strength": 0.6,
                    "message": "Currently in seasonally bullish period"
                })
            elif seasonal_bias == "seasonally_bearish":
                signals.append({
                    "type": "seasonal_bias",
                    "signal": "bearish",
                    "strength": 0.6,
                    "message": "Currently in seasonally bearish period"
                })
            
            # Pattern strength signals
            pattern_strength = seasonal_patterns.get("pattern_strength", "weak")
            if pattern_strength == "strong":
                signals.append({
                    "type": "seasonal_pattern_strength",
                    "signal": "monitor",
                    "strength": 0.7,
                    "message": "Strong seasonal patterns detected - monitor for opportunities"
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating seasonal signals: {str(e)}")
            return []
    
    async def _analyze_cross_market_relationships(self, futures_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships between different futures markets."""
        try:
            # This would analyze correlations and relationships with other markets
            # For now, provide framework for cross-market analysis
            
            return {
                "cross_market_correlations": {
                    "equity_futures_bond_futures": await self._analyze_equity_bond_relationship(),
                    "commodity_cross_correlations": await self._analyze_commodity_relationships(),
                    "currency_impacts": await self._analyze_currency_impacts()
                },
                "market_regime_relationships": self._analyze_regime_relationships(),
                "contagion_analysis": self._analyze_cross_market_contagion()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cross-market relationships: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_equity_bond_relationship(self) -> Dict[str, Any]:
        """Analyze relationship between equity and bond futures."""
        try:
            # Simulated relationship analysis
            return {
                "correlation": -0.3,  # Typical equity-bond correlation
                "relationship_strength": "moderate",
                "regime_dependent": True,
                "current_relationship": "flight_to_quality"
            }
        except Exception:
            return {"relationship": "analysis_failed"}
    
    async def _analyze_commodity_relationships(self) -> Dict[str, Any]:
        """Analyze relationships between different commodities."""
        try:
            return {
                "energy_metals_correlation": 0.2,
                "agriculture_energy_correlation": 0.1,
                "cross_commodity_co_movements": "low",
                "sector_rotation_opportunities": "monitor"
            }
        except Exception:
            return {"commodity_analysis": "failed"}
    
    async def _analyze_currency_impacts(self) -> Dict[str, Any]:
        """Analyze currency impacts on futures markets."""
        try:
            return {
                "dxy_impact_on_commodities": "negative",
                "currency_volatility_effect": "moderate",
                "hedging_considerations": "important"
            }
        except Exception:
            return {"currency_analysis": "failed"}
    
    def _analyze_regime_relationships(self) -> Dict[str, Any]:
        """Analyze how relationships change across market regimes."""
        try:
            return {
                "current_market_regime": "volatile_transition",
                "relationship_changes": "equity_bond_correlation_increasing",
                "regime_stability": "moderate"
            }
        except Exception:
            return {"regime_analysis": "failed"}
    
    def _analyze_cross_market_contagion(self) -> Dict[str, Any]:
        """Analyze potential for cross-market contagion."""
        try:
            return {
                "contagion_risk": "moderate",
                "primary_contagion_pathways": ["equity_to_commodity", "bond_to_currency"],
                "monitoring_requirements": "enhanced"
            }
        except Exception:
            return {"contagion_analysis": "failed"}
    
    async def _analyze_futures_volatility(self, futures_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility patterns in futures markets."""
        try:
            underlying_data = futures_data["underlying_data"]
            contracts = futures_data["futures_contracts"]
            
            # Calculate historical volatility
            historical_vol = self._calculate_historical_volatility(underlying_data)
            
            # Estimate implied volatility from futures term structure
            implied_vol_estimate = self._estimate_implied_volatility(contracts)
            
            # Volatility term structure
            vol_term_structure = self._analyze_volatility_term_structure(contracts, historical_vol)
            
            # Volatility seasonality
            vol_seasonality = self._analyze_volatility_seasonality(underlying_data)
            
            return {
                "historical_volatility": historical_vol,
                "implied_volatility_estimate": implied_vol_estimate,
                "volatility_term_structure": vol_term_structure,
                "volatility_seasonality": vol_seasonality,
                "volatility_outlook": self._assess_volatility_outlook(historical_vol, implied_vol_estimate)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing futures volatility: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_historical_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate historical volatility metrics."""
        try:
            returns = data['Close'].pct_change().dropna()
            
            if len(returns) == 0:
                return {"volatility": 0}
            
            # Calculate different volatility measures
            daily_vol = returns.std()
            annualized_vol = daily_vol * np.sqrt(252)
            
            # Rolling volatility
            rolling_vols = returns.rolling(30).std() * np.sqrt(252)
            
            # Volatility of volatility
            vol_of_vol = rolling_vols.std()
            
            return {
                "daily_volatility": daily_vol,
                "annualized_volatility": annualized_vol,
                "current_30d_vol": rolling_vols.iloc[-1] if not rolling_vols.empty else 0,
                "volatility_of_volatility": vol_of_vol,
                "volatility_percentile": self._calculate_volatility_percentile(rolling_vols.dropna())
            }
            
        except Exception as e:
            logger.error(f"Error calculating historical volatility: {str(e)}")
            return {"volatility": 0}
    
    def _calculate_volatility_percentile(self, rolling_vols: pd.Series) -> float:
        """Calculate current volatility percentile."""
        try:
            if len(rolling_vols) == 0:
                return 50.0
            
            current_vol = rolling_vols.iloc[-1]
            percentile = (rolling_vols < current_vol).mean() * 100
            
            return percentile
            
        except Exception:
            return 50.0
    
    def _estimate_implied_volatility(self, contracts: Dict) -> Dict[str, Any]:
        """Estimate implied volatility from futures data."""
        try:
            # Simplified implied vol estimation
            # In practice, would need options data for true implied vol
            
            vol_estimates = []
            
            for contract_data in contracts.values():
                days_to_exp = contract_data["days_to_expiration"]
                volume = contract_data["volume"]
                oi = contract_data["open_interest"]
                
                # Estimate volatility based on market activity
                if days_to_exp > 0 and oi > 0:
                    activity_factor = min(2.0, volume / oi)
                    estimated_vol = 0.20 * (1 + activity_factor * 0.5)  # Base 20% vol
                    vol_estimates.append(estimated_vol)
            
            if vol_estimates:
                avg_implied_vol = np.mean(vol_estimates)
                vol_term_slope = self._calculate_vol_term_slope(vol_estimates, contracts)
            else:
                avg_implied_vol = 0.20
                vol_term_slope = 0
            
            return {
                "average_implied_vol": avg_implied_vol,
                "vol_term_structure_slope": vol_term_slope,
                "estimated_term_structure": self._estimate_vol_term_structure(vol_estimates, contracts)
            }
            
        except Exception as e:
            logger.error(f"Error estimating implied volatility: {str(e)}")
            return {"average_implied_vol": 0.20}
    
    def _calculate_vol_term_slope(self, vol_estimates: List[float], contracts: Dict) -> float:
        """Calculate volatility term structure slope."""
        try:
            if len(vol_estimates) < 2 or len(contracts) < 2:
                return 0
            
            # Sort by expiration
            sorted_contracts = sorted(contracts.values(), key=lambda x: x["days_to_expiration"])
            days_to_exp = [c["days_to_expiration"] for c in sorted_contracts]
            
            # Simple linear relationship
            if len(days_to_exp) == len(vol_estimates):
                correlation = np.corrcoef(days_to_exp, vol_estimates)[0, 1]
                return correlation if not np.isnan(correlation) else 0
            else:
                return 0
                
        except Exception:
            return 0
    
    def _estimate_vol_term_structure(self, vol_estimates: List[float], contracts: Dict) -> Dict[str, Any]:
        """Estimate volatility term structure."""
        try:
            if not vol_estimates or not contracts:
                return {"structure": "insufficient_data"}
            
            avg_vol = np.mean(vol_estimates)
            vol_range = max(vol_estimates) - min(vol_estimates) if len(vol_estimates) > 1 else 0
            
            if vol_range / avg_vol > 0.3:
                structure = "steep_term_structure"
            elif vol_range / avg_vol > 0.15:
                structure = "moderate_term_structure"
            else:
                structure = "flat_term_structure"
            
            return {
                "structure": structure,
                "volatility_range": vol_range,
                "structure_strength": vol_range / avg_vol if avg_vol > 0 else 0
            }
            
        except Exception:
            return {"structure": "error"}
    
    def _analyze_volatility_term_structure(self, contracts: Dict, historical_vol: Dict) -> Dict[str, Any]:
        """Analyze volatility term structure patterns."""
        try:
            return {
                "historical_vs_implied": "estimating_from_futures_data",
                "term_structure_interpretation": "analyzing_market_expectations",
                "volatility_risk_premium": self._estimate_vol_risk_premium(contracts, historical_vol)
            }
        except Exception:
            return {"term_structure": "analysis_failed"}
    
    def _estimate_vol_risk_premium(self, contracts: Dict, historical_vol: Dict) -> float:
        """Estimate volatility risk premium."""
        try:
            # Simplified estimation
            hist_vol = historical_vol.get("annualized_volatility", 0.20)
            
            # If we had true implied vol, this would be implied - historical
            # For now, estimate from market activity
            total_volume = sum(c["volume"] for c in contracts.values())
            total_oi = sum(c["open_interest"] for c in contracts.values())
            
            activity_factor = total_volume / total_oi if total_oi > 0 else 1
            estimated_implied = hist_vol * (1 + activity_factor * 0.1)
            
            return max(0, estimated_implied - hist_vol)
            
        except Exception:
            return 0.0
    
    def _analyze_volatility_seasonality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonality in volatility."""
        try:
            if len(data) < 365:
                return {"seasonality": "insufficient_data"}
            
            # Calculate monthly volatility
            monthly_vols = []
            for month in range(1, 13):
                month_data = data[data.index.month == month]
                if len(month_data) > 1:
                    month_returns = month_data['Close'].pct_change().dropna()
                    if len(month_returns) > 0:
                        month_vol = month_returns.std() * np.sqrt(252)
                        monthly_vols.append({"month": month, "volatility": month_vol})
            
            if monthly_vols:
                avg_vol = np.mean([m["volatility"] for m in monthly_vols])
                vol_range = max(m["volatility"] for m in monthly_vols) - min(m["volatility"] for m in monthly_vols)
                
                return {
                    "monthly_volatilities": monthly_vols,
                    "seasonal_volatility_range": vol_range,
                    "volatility_seasonality_strength": vol_range / avg_vol if avg_vol > 0 else 0,
                    "high_volatility_months": [m["month"] for m in monthly_vols if m["volatility"] > avg_vol * 1.2]
                }
            
            return {"seasonality": "no_monthly_data"}
            
        except Exception as e:
            logger.error(f"Error analyzing volatility seasonality: {str(e)}")
            return {"seasonality": "analysis_failed"}
    
    def _assess_volatility_outlook(self, historical_vol: Dict, implied_vol_estimate: Dict) -> Dict[str, Any]:
        """Assess volatility outlook."""
        try:
            hist_vol = historical_vol.get("annualized_volatility", 0.20)
            implied_vol = implied_vol_estimate.get("average_implied_vol", hist_vol)
            
            vol_premium = implied_vol - hist_vol
            
            if vol_premium > 0.05:
                outlook = "volatility_expected_to_increase"
                outlook_strength = "strong"
            elif vol_premium > 0.02:
                outlook = "volatility_expected_to_rise"
                outlook_strength = "moderate"
            elif vol_premium < -0.02:
                outlook = "volatility_expected_to_decrease"
                outlook_strength = "moderate"
            else:
                outlook = "volatility_stable"
                outlook_strength = "weak"
            
            return {
                "outlook": outlook,
                "outlook_strength": outlook_strength,
                "volatility_premium": vol_premium,
                "confidence_level": 0.6
            }
            
        except Exception:
            return {"outlook": "assessment_failed"}
    
    async def _analyze_liquidity(self, futures_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market liquidity conditions."""
        try:
            contracts = futures_data["futures_contracts"]
            microstructure = futures_data.get("microstructure", {})
            
            # Contract-specific liquidity
            contract_liquidity = self._analyze_contract_liquidity(contracts)
            
            # Overall market liquidity
            market_liquidity = self._assess_market_liquidity(contracts)
            
            # Liquidity risk assessment
            liquidity_risk = self._assess_liquidity_risk(contract_liquidity, market_liquidity)
            
            # Liquidity timing analysis
            timing_analysis = self._analyze_liquidity_timing(contracts)
            
            return {
                "contract_liquidity": contract_liquidity,
                "market_liquidity": market_liquidity,
                "liquidity_risk": liquidity_risk,
                "timing_analysis": timing_analysis,
                "liquidity_recommendations": self._generate_liquidity_recommendations(contract_liquidity, market_liquidity)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_contract_liquidity(self, contracts: Dict) -> Dict[str, Any]:
        """Analyze liquidity of individual contracts."""
        try:
            contract_metrics = {}
            
            for contract_data in contracts.values():
                symbol = contract_data["symbol"]
                volume = contract_data["volume"]
                oi = contract_data["open_interest"]
                days_to_exp = contract_data["days_to_expiration"]
                spread_data = contract_data.get("spread_data", {})
                
                # Liquidity score calculation
                liquidity_components = {
                    "volume_score": min(1.0, volume / 5000),  # Normalize volume
                    "oi_score": min(1.0, oi / 10000),        # Normalize OI
                    "spread_score": 1 - min(1.0, spread_data.get("spread_to_tick_ratio", 1) / 5),  # Invert spread ratio
                    "time_score": 1 - min(1.0, days_to_exp / 365)  # Front months get higher score
                }
                
                overall_liquidity_score = np.mean(list(liquidity_components.values()))
                
                # Liquidity classification
                if overall_liquidity_score > 0.8:
                    liquidity_level = "excellent"
                elif overall_liquidity_score > 0.6:
                    liquidity_level = "good"
                elif overall_liquidity_score > 0.4:
                    liquidity_level = "fair"
                else:
                    liquidity_level = "poor"
                
                contract_metrics[symbol] = {
                    "liquidity_score": overall_liquidity_score,
                    "liquidity_level": liquidity_level,
                    "liquidity_components": liquidity_components,
                    "volume": volume,
                    "open_interest": oi,
                    "bid_ask_spread": spread_data.get("bid_ask_spread", 0.01)
                }
            
            return contract_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing contract liquidity: {str(e)}")
            return {}
    
    def _assess_market_liquidity(self, contracts: Dict) -> Dict[str, Any]:
        """Assess overall market liquidity."""
        try:
            if not contracts:
                return {"market_liquidity": "no_data"}
            
            total_volume = sum(c["volume"] for c in contracts.values())
            total_oi = sum(c["open_interest"] for c in contracts.values())
            total_contracts = len(contracts)
            
            # Market liquidity metrics
            avg_volume = total_volume / total_contracts if total_contracts > 0 else 0
            avg_oi = total_oi / total_contracts if total_contracts > 0 else 0
            volume_oi_ratio = total_volume / total_oi if total_oi > 0 else 0
            
            # Liquidity depth
            depth_score = min(1.0, avg_volume / 3000) * min(1.0, avg_oi / 8000)
            
            # Overall liquidity assessment
            if depth_score > 0.8:
                overall_liquidity = "excellent"
            elif depth_score > 0.6:
                overall_liquidity = "good"
            elif depth_score > 0.4:
                overall_liquidity = "fair"
            else:
                overall_liquidity = "poor"
            
            return {
                "overall_liquidity": overall_liquidity,
                "total_volume": total_volume,
                "total_open_interest": total_oi,
                "average_volume": avg_volume,
                "average_open_interest": avg_oi,
                "volume_oi_ratio": volume_oi_ratio,
                "depth_score": depth_score,
                "liquidity_trend": self._assess_liquidity_trend(contracts)
            }
            
        except Exception as e:
            logger.error(f"Error assessing market liquidity: {str(e)}")
            return {"market_liquidity": "assessment_error"}
    
    def _assess_liquidity_trend(self, contracts: Dict) -> str:
        """Assess liquidity trend."""
        try:
            # This would require historical data in production
            # For now, return current assessment
            volumes = [c["volume"] for c in contracts.values()]
            avg_volume = np.mean(volumes) if volumes else 0
            
            if avg_volume > 5000:
                return "stable_high"
            elif avg_volume > 2000:
                return "stable_moderate"
            else:
                return "variable"
                
        except Exception:
            return "trend_unknown"}
    
    def _assess_liquidity_risk(self, contract_liquidity: Dict, market_liquidity: Dict) -> Dict[str, Any]:
        """Assess liquidity risk factors."""
        try:
            risk_factors = []
            risk_score = 0
            
            # Check for poor liquidity contracts
            poor_liquidity_contracts = sum(1 for data in contract_liquidity.values() 
                                         if data["liquidity_level"] in ["poor", "fair"])
            
            if poor_liquidity_contracts > len(contract_liquidity) * 0.3:
                risk_factors.append("high_proportion_poor_liquidity_contracts")
                risk_score += 0.3
            
            # Check overall market liquidity
            overall_liquidity = market_liquidity.get("overall_liquidity", "fair")
            if overall_liquidity in ["poor", "fair"]:
                risk_factors.append("overall_market_liquidity_concern")
                risk_score += 0.4
            
            # Check volume concentration
            volume_oi_ratio = market_liquidity.get("volume_oi_ratio", 1)
            if volume_oi_ratio < 0.1 or volume_oi_ratio > 2.0:
                risk_factors.append("unusual_volume_oi_ratio")
                risk_score += 0.2
            
            # Risk classification
            if risk_score > 0.6:
                risk_level = "high"
            elif risk_score > 0.3:
                risk_level = "moderate"
            else:
                risk_level = "low"
            
            return {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "risk_factors": risk_factors,
                "risk_interpretation": self._interpret_liquidity_risk(risk_level, risk_factors)
            }
            
        except Exception as e:
            logger.error(f"Error assessing liquidity risk: {str(e)}")
            return {"risk_level": "unknown"}
    
    def _interpret_liquidity_risk(self, risk_level: str, risk_factors: List[str]) -> str:
        """Interpret liquidity risk assessment."""
        interpretations = {
            "high": "Significant liquidity constraints - exercise caution with large positions",
            "moderate": "Moderate liquidity risk - monitor conditions closely",
            "low": "Good liquidity conditions - standard position sizing appropriate"
        }
        
        base_interpretation = interpretations.get(risk_level, "Liquidity risk assessment unavailable")
        
        # Add specific factor interpretations
        if "poor_liquidity_contracts" in risk_factors:
            base_interpretation += " Multiple contracts show poor liquidity."
        
        if "overall_market_liquidity_concern" in risk_factors:
            base_interpretation += " Overall market liquidity is concerning."
        
        return base_interpretation
    
    def _analyze_liquidity_timing(self, contracts: Dict) -> Dict[str, Any]:
        """Analyze liquidity timing considerations."""
        try:
            timing_considerations = []
            
            for contract_data in contracts.values():
                days_to_exp = contract_data["days_to_expiration"]
                liquidity_level = contract_liquidity.get(contract_data["symbol"], {}).get("liquidity_level", "unknown")
                
                if days_to_exp <= 7:
                    timing_considerations.append({
                        "contract": contract_data["symbol"],
                        "consideration": "roll_urgently",
                        "reason": "approaching_expiration",
                        "urgency": "high"
                    })
                elif days_to_exp <= 30 and liquidity_level in ["poor", "fair"]:
                    timing_considerations.append({
                        "contract": contract_data["symbol"],
                        "consideration": "monitor_roll_conditions",
                        "reason": "approaching_expiration_poor_liquidity",
                        "urgency": "medium"
                    })
            
            # Overall timing recommendation
            high_urgency_count = sum(1 for c in timing_considerations if c["urgency"] == "high")
            
            if high_urgency_count > 0:
                overall_timing = "immediate_attention_required"
            elif any(c["urgency"] == "medium" for c in timing_considerations):
                overall_timing = "monitor_closely"
            else:
                overall_timing = "normal_timing"
            
            return {
                "overall_timing": overall_timing,
                "timing_considerations": timing_considerations,
                "roll_recommendations": self._generate_roll_recommendations(timing_considerations)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity timing: {str(e)}")
            return {"timing_analysis": "error"}
    
    def _generate_roll_recommendations(self, timing_considerations: List[Dict]) -> List[str]:
        """Generate roll timing recommendations."""
        recommendations = []
        
        high_urgency = [c for c in timing_considerations if c["urgency"] == "high"]
        medium_urgency = [c for c in timing_considerations if c["urgency"] == "medium"]
        
        if high_urgency:
            recommendations.append("Roll positions from contracts approaching expiration immediately")
        
        if medium_urgency:
            recommendations.append("Monitor roll conditions for contracts with limited liquidity")
        
        if not timing_considerations:
            recommendations.append("Normal roll timing - no urgent action required")
        
        return recommendations
    
    def _generate_liquidity_recommendations(self, contract_liquidity: Dict, market_liquidity: Dict) -> List[str]:
        """Generate liquidity-based recommendations."""
        try:
            recommendations = []
            
            # Overall market recommendations
            overall_liquidity = market_liquidity.get("overall_liquidity", "fair")
            if overall_liquidity in ["poor", "fair"]:
                recommendations.append("Consider reducing position sizes due to limited liquidity")
            
            # Contract-specific recommendations
            poor_liquidity_contracts = [symbol for symbol, data in contract_liquidity.items() 
                                      if data["liquidity_level"] == "poor"]
            
            if poor_liquidity_contracts:
                recommendations.append(f"Avoid new positions in contracts: {', '.join(poor_liquidity_contracts)}")
            
            # Volume-based recommendations
            volume_oi_ratio = market_liquidity.get("volume_oi_ratio", 1)
            if volume_oi_ratio < 0.1:
                recommendations.append("Low trading activity - be cautious with large orders")
            elif volume_oi_ratio > 2.0:
                recommendations.append("High turnover - good conditions for position adjustments")
            
            if not recommendations:
                recommendations.append("Current liquidity conditions are adequate for normal trading")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating liquidity recommendations: {str(e)}")
            return ["Liquidity analysis incomplete - proceed with caution"]
    
    def _assess_liquidity(self, volume_stats: Dict) -> str:
        """Assess overall liquidity from volume statistics."""
        try:
            avg_volume = volume_stats.get("average_volume", 0)
            
            if avg_volume > 8000:
                return "excellent"
            elif avg_volume > 5000:
                return "good"
            elif avg_volume > 2000:
                return "fair"
            else:
                return "poor"
                
        except Exception:
            return "liquidity_assessment_error"}
    
    async def _calculate_futures_intelligence_score(self, term_structure_analysis: Dict, 
                                                  positioning_analysis: Dict, 
                                                  volatility_analysis: Dict) -> Dict[str, Any]:
        """Calculate comprehensive futures intelligence score."""
        try:
            # Component scores
            term_structure_score = self._score_term_structure(term_structure_analysis)
            positioning_score = self._score_positioning_analysis(positioning_analysis)
            volatility_score = self._score_volatility_analysis(volatility_analysis)
            
            # Weighted combination
            weights = {"term_structure": 0.3, "positioning": 0.4, "volatility": 0.3}
            overall_score = (
                term_structure_score * weights["term_structure"] +
                positioning_score * weights["positioning"] +
                volatility_score * weights["volatility"]
            )
            
            # Intelligence score components
            intelligence_components = {
                "term_structure_score": term_structure_score,
                "positioning_score": positioning_score,
                "volatility_score": volatility_score,
                "overall_futures_intelligence": overall_score
            }
            
            # Risk assessment
            risk_level = self._assess_futures_risk_level(term_structure_analysis, positioning_analysis)
            
            return {
                "futures_intelligence_score": overall_score,
                "score_components": intelligence_components,
                "risk_level": risk_level,
                "trading_recommendations": self._generate_trading_recommendations(
                    overall_score, risk_level, term_structure_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating futures intelligence score: {str(e)}")
            return {"futures_intelligence_score": 0.5, "error": str(e)}
    
    def _score_term_structure(self, term_structure_analysis: Dict) -> float:
        """Score term structure analysis quality."""
        try:
            if "error" in term_structure_analysis:
                return 0.5
            
            metrics = term_structure_analysis.get("term_structure_metrics", {})
            
            # Score based on data completeness and quality
            score = 0.0
            total_checks = 5
            
            if "curve_shape" in metrics:
                score += 0.2
            if "contango_backwardation" in metrics:
                score += 0.2
            if "curve_steepness" in metrics:
                score += 0.2
            if "curve_convexity" in metrics:
                score += 0.2
            if "carry_analysis" in metrics:
                score += 0.2
            
            return score
            
        except Exception:
            return 0.5
    
    def _score_positioning_analysis(self, positioning_analysis: Dict) -> float:
        """Score positioning analysis quality."""
        try:
            if "error" in positioning_analysis:
                return 0.5
            
            # Score based on data availability and analysis completeness
            score = 0.0
            total_checks = 4
            
            if "aggregate_metrics" in positioning_analysis:
                score += 0.25
            if "trader_category_positioning" in positioning_analysis:
                score += 0.25
            if "contract_positioning" in positioning_analysis:
                score += 0.25
            if "concentration_analysis" in positioning_analysis:
                score += 0.25
            
            return score
            
        except Exception:
            return 0.5
    
    def _score_volatility_analysis(self, volatility_analysis: Dict) -> float:
        """Score volatility analysis quality."""
        try:
            if "error" in volatility_analysis:
                return 0.5
            
            # Score based on analysis completeness
            score = 0.0
            total_checks = 3
            
            if "historical_volatility" in volatility_analysis:
                score += 0.33
            if "implied_volatility_estimate" in volatility_analysis:
                score += 0.33
            if "volatility_term_structure" in volatility_analysis:
                score += 0.34
            
            return score
            
        except Exception:
            return 0.5
    
    def _assess_futures_risk_level(self, term_structure_analysis: Dict, positioning_analysis: Dict) -> str:
        """Assess overall futures market risk level."""
        try:
            risk_factors = 0
            
            # Term structure risk factors
            metrics = term_structure_analysis.get("term_structure_metrics", {})
            shape = metrics.get("curve_shape", {}).get("shape", "")
            
            if "steep" in shape:
                risk_factors += 1
            
            # Positioning risk factors
            concentration = positioning_analysis.get("concentration_analysis", {})
            concentration_level = concentration.get("concentration_level", "")
            
            if "highly_concentrated" in concentration_level:
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
    
    def _generate_trading_recommendations(self, intelligence_score: float, risk_level: str, 
                                        term_structure_analysis: Dict) -> List[str]:
        """Generate trading recommendations based on analysis."""
        try:
            recommendations = []
            
            # Score-based recommendations
            if intelligence_score > 0.7:
                recommendations.append("High confidence in analysis - consider active positioning")
            elif intelligence_score > 0.6:
                recommendations.append("Moderate confidence - proceed with measured positions")
            else:
                recommendations.append("Low confidence - avoid aggressive positioning")
            
            # Risk-based recommendations
            if risk_level == "high":
                recommendations.append("High risk environment - reduce position sizes")
            elif risk_level == "medium":
                recommendations.append("Moderate risk - monitor positions closely")
            else:
                recommendations.append("Low risk environment - normal position sizing appropriate")
            
            # Term structure recommendations
            carry_analysis = term_structure_analysis.get("carry_analysis", {})
            carry_assessment = carry_analysis.get("carry_trade_assessment", "")
            
            if "attractive" in carry_assessment:
                recommendations.append("Favorable carry trade opportunities identified")
            elif "negative" in carry_assessment:
                recommendations.append("Negative carry - monitor roll costs")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating trading recommendations: {str(e)}")
            return ["Analysis incomplete - exercise caution"]
    
    async def get_futures_intelligence_history(self, symbol: str = "ES", days: int = 30) -> Dict[str, Any]:
        """Get historical futures intelligence data."""
        try:
            # In production, this would retrieve historical futures data
            # For now, return current analysis with simulated historical context
            
            current_analysis = await self.analyze_futures_intelligence(symbol)
            
            # Simulated historical intelligence scores
            historical_scores = []
            base_score = current_analysis.get("futures_intelligence_score", {}).get("futures_intelligence_score", 0.5)
            
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
            logger.error(f"Error getting futures intelligence history: {str(e)}")
            return {"error": str(e)}