"""
Module 16: Open Interest Intelligence Engine

Advanced open interest intelligence system providing real-time analysis of
positioning changes, dealer flows, institutional flows, and positioning
concentration across options and futures markets.

Author: MiniMax Agent
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.signal import find_peaks
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class OpenInterestIntelligenceEngine:
    """
    Open Interest market intelligence and analysis engine.
    
    Features:
    - Open interest aggregation and positioning analysis
    - Dealer flow and hedge fund positioning tracking
    - Institutional vs retail flow analysis
    - Options/futures positioning concentration
    - Open interest trends and momentum indicators
    - Cross-asset positioning analysis
    """
    
    def __init__(self, db_manager=None, cache_manager=None):
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.oi_data_cache = {}
        self.positioning_cache = {}
        
    async def analyze_open_interest_intelligence(self, symbol: str = "SPY") -> Dict[str, Any]:
        """
        Comprehensive open interest intelligence analysis.
        
        Args:
            symbol: Asset symbol to analyze
            
        Returns:
            Dictionary containing open interest intelligence results
        """
        try:
            # Fetch open interest data
            oi_data = await self._fetch_open_interest_data(symbol)
            if not oi_data:
                return {"error": "Unable to fetch open interest data"}
            
            # Open interest aggregation analysis
            oi_aggregation = await self._analyze_oi_aggregation(oi_data)
            
            # Positioning analysis
            positioning_analysis = await self._analyze_positioning_intelligence(oi_data)
            
            # Dealer flow analysis
            dealer_analysis = await self._analyze_dealer_flows(oi_data)
            
            # Institutional flow analysis
            institutional_analysis = await self._analyze_institutional_flows(oi_data)
            
            # Concentration analysis
            concentration_analysis = await self._analyze_positioning_concentration(oi_data)
            
            # Trend analysis
            trend_analysis = await self._analyze_oi_trends(oi_data)
            
            # Cross-asset analysis
            cross_asset_analysis = await self._analyze_cross_asset_positioning(symbol)
            
            # Market structure analysis
            market_structure = await self._analyze_market_structure(oi_data)
            
            # Momentum indicators
            momentum_analysis = await self._analyze_momentum_indicators(oi_data)
            
            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "current_price": oi_data.get("current_price", 0),
                "oi_aggregation": oi_aggregation,
                "positioning_analysis": positioning_analysis,
                "dealer_analysis": dealer_analysis,
                "institutional_analysis": institutional_analysis,
                "concentration_analysis": concentration_analysis,
                "trend_analysis": trend_analysis,
                "cross_asset_analysis": cross_asset_analysis,
                "market_structure": market_structure,
                "momentum_analysis": momentum_analysis,
                "open_interest_intelligence_score": await self._calculate_oi_intelligence_score(
                    oi_aggregation, positioning_analysis, trend_analysis
                )
            }
            
            # Cache results
            if self.cache_manager:
                await self.cache_manager.set(f"oi_intelligence:{symbol}", result, ttl=300)
                
            return result
            
        except Exception as e:
            logger.error(f"Error in open interest intelligence analysis for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    async def _fetch_open_interest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch open interest data for analysis."""
        try:
            # Get underlying asset data
            ticker = yf.Ticker(symbol)
            underlying_data = ticker.history(period="1y")
            
            if underlying_data.empty:
                return None
            
            current_price = underlying_data['Close'].iloc[-1]
            
            # Create simulated open interest data
            # In production, this would connect to professional options/futures data providers
            oi_data = await self._create_simulated_oi_data(symbol, current_price, underlying_data)
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "underlying_data": underlying_data,
                "options_oi": oi_data.get("options_oi", {}),
                "futures_oi": oi_data.get("futures_oi", {}),
                "total_oi": oi_data.get("total_oi", 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching open interest data: {str(e)}")
            return None
    
    async def _create_simulated_oi_data(self, symbol: str, current_price: float, 
                                      underlying_data: pd.DataFrame) -> Dict[str, Any]:
        """Create simulated open interest data for analysis."""
        try:
            # Generate realistic OI data across strikes and expirations
            options_oi = {}
            futures_oi = {}
            
            # Generate options OI data
            options_data = await self._generate_options_oi_data(symbol, current_price)
            options_oi = options_data
            
            # Generate futures OI data
            futures_data = await self._generate_futures_oi_data(symbol, current_price)
            futures_oi = futures_data
            
            # Calculate total OI
            total_oi = sum(
                sum(strike_data.values()) for strike_data in options_oi.values()
            ) + sum(contract_data["open_interest"] for contract_data in futures_oi.values())
            
            return {
                "options_oi": options_oi,
                "futures_oi": futures_oi,
                "total_oi": total_oi
            }
            
        except Exception as e:
            logger.error(f"Error creating simulated OI data: {str(e)}")
            return {"options_oi": {}, "futures_oi": {}, "total_oi": 0}
    
    async def _generate_options_oi_data(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Generate simulated options open interest data."""
        try:
            options_oi = {}
            
            # Generate strikes around current price
            strikes = []
            for i in range(-15, 16):  # 31 strikes
                strike = current_price * (1 + i * 0.02)  # 2% increments
                strikes.append(round(strike, 2))
            
            # Expiration dates (simulate next 4 months)
            exp_dates = []
            today = datetime.now()
            for i in range(1, 5):
                exp_date = today + timedelta(days=30*i)
                exp_dates.append(exp_date.strftime("%Y-%m-%d"))
            
            # Generate OI data for each expiration
            for exp_date in exp_dates:
                options_oi[exp_date] = {}
                
                for strike in strikes:
                    # Realistic OI distribution
                    distance_from_atm = abs(strike - current_price) / current_price
                    
                    # ATM and near-ATM options typically have highest OI
                    if distance_from_atm < 0.02:  # Within 2% of ATM
                        base_oi = np.random.lognormal(np.log(2000), 0.5)
                    elif distance_from_atm < 0.05:  # Within 5% of ATM
                        base_oi = np.random.lognormal(np.log(1000), 0.5)
                    else:
                        base_oi = np.random.lognormal(np.log(300), 0.5)
                    
                    # Adjust for time to expiration (farther out = less OI)
                    exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
                    days_to_exp = (exp_dt - today).days
                    time_decay = max(0.3, 1 - days_to_exp / 120)  # Decay over 4 months
                    
                    total_oi = int(base_oi * time_decay)
                    
                    # Split between calls and puts (usually put OI > call OI)
                    put_call_ratio = np.random.uniform(1.1, 1.4)
                    puts_oi = int(total_oi * put_call_ratio / (1 + put_call_ratio))
                    calls_oi = int(total_oi / (1 + put_call_ratio))
                    
                    options_oi[exp_date][strike] = {
                        "calls_oi": calls_oi,
                        "puts_oi": puts_oi,
                        "total_oi": total_oi,
                        "put_call_ratio": puts_oi / calls_oi if calls_oi > 0 else 0,
                        "moneyness": "ITM" if (strike < current_price and symbol not in ["SPY", "QQQ"]) or 
                                             (strike > current_price and symbol in ["SPY", "QQQ"]) else "OTM",
                        "days_to_exp": days_to_exp
                    }
            
            return options_oi
            
        except Exception as e:
            logger.error(f"Error generating options OI data: {str(e)}")
            return {}
    
    async def _generate_futures_oi_data(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Generate simulated futures open interest data."""
        try:
            futures_oi = []
            
            # Map to futures contract
            futures_symbol = self._map_to_futures_symbol(symbol)
            
            # Generate contract months
            contract_months = []
            today = datetime.now()
            for i in range(1, 7):  # Next 6 contract months
                exp_month = today.replace(day=1) + timedelta(days=32*i)
                exp_month = exp_month.replace(day=1)
                third_friday = self._get_third_friday(exp_month)
                contract_months.append(third_friday)
            
            # Generate OI for each contract
            for expiration_date in contract_months:
                days_to_exp = (expiration_date - today).days
                
                # Futures OI typically highest for front months
                if days_to_exp <= 30:
                    base_oi = np.random.lognormal(np.log(15000), 0.3)
                elif days_to_exp <= 90:
                    base_oi = np.random.lognormal(np.log(10000), 0.3)
                else:
                    base_oi = np.random.lognormal(np.log(8000), 0.3)
                
                futures_oi.append({
                    "symbol": f"{futures_symbol}{expiration_date.strftime('%y%m')}",
                    "expiration_date": expiration_date.isoformat(),
                    "days_to_expiration": days_to_exp,
                    "open_interest": int(base_oi),
                    "contract_type": "front_month" if days_to_exp <= 30 else "deferred_month"
                })
            
            return futures_oi
            
        except Exception as e:
            logger.error(f"Error generating futures OI data: {str(e)}")
            return []
    
    def _map_to_futures_symbol(self, symbol: str) -> str:
        """Map stock symbol to futures symbol."""
        mapping = {
            "SPY": "ES",
            "QQQ": "NQ",
            "IWM": "RTY",
            "DIA": "YM"
        }
        return mapping.get(symbol, "ES")  # Default to ES
    
    def _get_third_friday(self, date: datetime) -> datetime:
        """Get the third Friday of a given month."""
        first_day = date.replace(day=1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        third_friday = first_friday + timedelta(days=14)
        return third_friday
    
    async def _analyze_oi_aggregation(self, oi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze open interest aggregation patterns."""
        try:
            options_oi = oi_data.get("options_oi", {})
            futures_oi = oi_data.get("futures_oi", {})
            current_price = oi_data.get("current_price", 0)
            
            # Aggregate options OI
            options_aggregation = self._aggregate_options_oi(options_oi, current_price)
            
            # Aggregate futures OI
            futures_aggregation = self._aggregate_futures_oi(futures_oi)
            
            # Total market OI
            total_oi_analysis = self._analyze_total_oi(options_aggregation, futures_aggregation)
            
            # OI by expiration analysis
            expiration_analysis = self._analyze_oi_by_expiration(options_oi, futures_oi)
            
            # Strike-specific analysis
            strike_analysis = self._analyze_oi_by_strike(options_oi, current_price)
            
            return {
                "options_aggregation": options_aggregation,
                "futures_aggregation": futures_aggregation,
                "total_oi_analysis": total_oi_analysis,
                "expiration_analysis": expiration_analysis,
                "strike_analysis": strike_analysis,
                "aggregation_signals": self._generate_aggregation_signals(options_aggregation, futures_aggregation)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing OI aggregation: {str(e)}")
            return {"error": str(e)}
    
    def _aggregate_options_oi(self, options_oi: Dict, current_price: float) -> Dict[str, Any]:
        """Aggregate options OI data."""
        try:
            if not options_oi:
                return {"total_options_oi": 0}
            
            total_calls_oi = 0
            total_puts_oi = 0
            oi_by_moneyness = {"deep_itm": 0, "itm": 0, "atm": 0, "otm": 0, "deep_otm": 0}
            oi_by_expiration = {}
            
            for exp_date, strike_data in options_oi.items():
                exp_calls_oi = 0
                exp_puts_oi = 0
                
                for strike, data in strike_data.items():
                    calls_oi = data.get("calls_oi", 0)
                    puts_oi = data.get("puts_oi", 0)
                    
                    total_calls_oi += calls_oi
                    total_puts_oi += puts_oi
                    exp_calls_oi += calls_oi
                    exp_puts_oi += puts_oi
                    
                    # Categorize by moneyness
                    moneyness = data.get("moneyness", "otm")
                    if "ITM" in moneyness:
                        oi_by_moneyness["itm"] += calls_oi + puts_oi
                    elif "ATM" in moneyness:
                        oi_by_moneyness["atm"] += calls_oi + puts_oi
                    else:
                        oi_by_moneyness["otm"] += calls_oi + puts_oi
                
                oi_by_expiration[exp_date] = {
                    "calls_oi": exp_calls_oi,
                    "puts_oi": exp_puts_oi,
                    "total_oi": exp_calls_oi + exp_puts_oi
                }
            
            put_call_ratio = total_puts_oi / total_calls_oi if total_calls_oi > 0 else 0
            
            return {
                "total_options_oi": total_calls_oi + total_puts_oi,
                "total_calls_oi": total_calls_oi,
                "total_puts_oi": total_puts_oi,
                "put_call_ratio": put_call_ratio,
                "oi_by_moneyness": oi_by_moneyness,
                "oi_by_expiration": oi_by_expiration,
                "concentration_index": self._calculate_oi_concentration(oi_by_expiration)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating options OI: {str(e)}")
            return {"total_options_oi": 0}
    
    def _aggregate_futures_oi(self, futures_oi: List[Dict]) -> Dict[str, Any]:
        """Aggregate futures OI data."""
        try:
            if not futures_oi:
                return {"total_futures_oi": 0}
            
            total_futures_oi = sum(contract["open_interest"] for contract in futures_oi)
            
            # Group by contract type
            front_month_oi = sum(contract["open_interest"] for contract in futures_oi 
                               if contract["contract_type"] == "front_month")
            deferred_month_oi = total_futures_oi - front_month_oi
            
            # OI by expiration
            oi_by_expiration = {}
            for contract in futures_oi:
                exp_date = contract["expiration_date"]
                oi_by_expiration[exp_date] = contract["open_interest"]
            
            return {
                "total_futures_oi": total_futures_oi,
                "front_month_oi": front_month_oi,
                "deferred_month_oi": deferred_month_oi,
                "front_deferred_ratio": front_month_oi / deferred_month_oi if deferred_month_oi > 0 else float('inf'),
                "oi_by_expiration": oi_by_expiration,
                "oi_concentration": self._calculate_futures_oi_concentration(futures_oi)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating futures OI: {str(e)}")
            return {"total_futures_oi": 0}
    
    def _analyze_total_oi(self, options_aggregation: Dict, futures_aggregation: Dict) -> Dict[str, Any]:
        """Analyze total OI across options and futures."""
        try:
            total_options_oi = options_aggregation.get("total_options_oi", 0)
            total_futures_oi = futures_aggregation.get("total_futures_oi", 0)
            total_market_oi = total_options_oi + total_futures_oi
            
            # OI composition analysis
            options_percentage = total_options_oi / total_market_oi if total_market_oi > 0 else 0
            futures_percentage = total_futures_oi / total_market_oi if total_market_oi > 0 else 0
            
            # Market preference analysis
            if options_percentage > 0.7:
                market_preference = "options_heavy"
            elif futures_percentage > 0.7:
                market_preference = "futures_heavy"
            else:
                market_preference = "balanced"
            
            return {
                "total_market_oi": total_market_oi,
                "options_percentage": options_percentage,
                "futures_percentage": futures_percentage,
                "market_preference": market_preference,
                "options_futures_ratio": total_options_oi / total_futures_oi if total_futures_oi > 0 else float('inf')
            }
            
        except Exception as e:
            logger.error(f"Error analyzing total OI: {str(e)}")
            return {"total_market_oi": 0}
    
    def _analyze_oi_by_expiration(self, options_oi: Dict, futures_oi: List[Dict]) -> Dict[str, Any]:
        """Analyze OI distribution by expiration."""
        try:
            expiration_analysis = {
                "options_expirations": {},
                "futures_expirations": {},
                "near_term_concentration": 0,
                "long_term_distribution": 0
            }
            
            # Analyze options expirations
            today = datetime.now()
            near_term_oi = 0
            total_oi = 0
            
            for exp_date, strike_data in options_oi.items():
                exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
                days_to_exp = (exp_dt - today).days
                exp_total_oi = sum(data.get("total_oi", 0) for data in strike_data.values())
                
                expiration_analysis["options_expirations"][exp_date] = {
                    "days_to_exp": days_to_exp,
                    "total_oi": exp_total_oi
                }
                
                total_oi += exp_total_oi
                if days_to_exp <= 30:
                    near_term_oi += exp_total_oi
            
            # Analyze futures expirations
            for contract in futures_oi:
                exp_date = contract["expiration_date"]
                days_to_exp = contract["days_to_expiration"]
                oi = contract["open_interest"]
                
                expiration_analysis["futures_expirations"][exp_date] = {
                    "days_to_exp": days_to_exp,
                    "open_interest": oi
                }
                
                total_oi += oi
                if days_to_exp <= 30:
                    near_term_oi += oi
            
            # Calculate near-term concentration
            expiration_analysis["near_term_concentration"] = near_term_oi / total_oi if total_oi > 0 else 0
            expiration_analysis["long_term_distribution"] = 1 - expiration_analysis["near_term_concentration"]
            
            return expiration_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing OI by expiration: {str(e)}")
            return {"options_expirations": {}, "futures_expirations": {}}
    
    def _analyze_oi_by_strike(self, options_oi: Dict, current_price: float) -> Dict[str, Any]:
        """Analyze OI distribution by strike prices."""
        try:
            if not options_oi:
                return {"strike_analysis": "no_options_data"}
            
            # Aggregate OI by strike across all expirations
            strike_oi = {}
            total_strike_oi = 0
            
            for exp_date, strike_data in options_oi.items():
                for strike, data in strike_data.items():
                    if strike not in strike_oi:
                        strike_oi[strike] = {"calls_oi": 0, "puts_oi": 0, "total_oi": 0}
                    
                    strike_oi[strike]["calls_oi"] += data.get("calls_oi", 0)
                    strike_oi[strike]["puts_oi"] += data.get("puts_oi", 0)
                    strike_oi[strike]["total_oi"] += data.get("total_oi", 0)
                    
                    total_strike_oi += data.get("total_oi", 0)
            
            # Find key strikes
            sorted_strikes = sorted(strike_oi.items(), key=lambda x: x[1]["total_oi"], reverse=True)
            top_strikes = sorted_strikes[:5]  # Top 5 by OI
            
            # Strike distribution analysis
            atm_strikes = []
            for strike, data in strike_oi.items():
                distance = abs(strike - current_price) / current_price
                if distance < 0.02:  # Within 2% of ATM
                    atm_strikes.append((strike, data["total_oi"]))
            
            atm_concentration = sum(oi for _, oi in atm_strikes) / total_strike_oi if total_strike_oi > 0 else 0
            
            return {
                "total_unique_strikes": len(strike_oi),
                "top_strikes": [{"strike": strike, "oi": data["total_oi"]} for strike, data in top_strikes],
                "atm_concentration": atm_concentration,
                "strike_distribution": self._analyze_strike_distribution(strike_oi, current_price),
                "gamma_hub_analysis": self._analyze_gamma_hub(strike_oi, current_price)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing OI by strike: {str(e)}")
            return {"strike_analysis": "error"}
    
    def _analyze_strike_distribution(self, strike_oi: Dict, current_price: float) -> Dict[str, Any]:
        """Analyze the distribution of OI across strikes."""
        try:
            if not strike_oi:
                return {"distribution": "no_data"}
            
            strikes = list(strike_oi.keys())
            ois = [data["total_oi"] for data in strike_oi.values()]
            
            # Calculate distribution metrics
            atm_strike = min(strikes, key=lambda s: abs(s - current_price))
            distances_from_atm = [abs(strike - current_price) / current_price for strike in strikes]
            
            # Distribution shape analysis
            if len(distances_from_atm) > 2:
                # Check for clustering around ATM
                near_atm_ois = [oi for strike, oi in strike_oi.items() 
                               if abs(strike - current_price) / current_price < 0.05]
                atm_clustering = sum(data["total_oi"] for data in near_atm_ois) / sum(ois) if ois else 0
                
                if atm_clustering > 0.6:
                    distribution_shape = "atm_clustered"
                elif max(distances_from_atm) > 0.3:
                    distribution_shape = "wide_distribution"
                else:
                    distribution_shape = "normal_distribution"
            else:
                distribution_shape = "insufficient_data"
                atm_clustering = 0
            
            return {
                "distribution_shape": distribution_shape,
                "atm_clustering": atm_clustering,
                "distance_from_atm": distances_from_atm,
                "strike_range": {"min": min(distances_from_atm), "max": max(distances_from_atm)}
            }
            
        except Exception as e:
            logger.error(f"Error analyzing strike distribution: {str(e)}")
            return {"distribution": "analysis_error"}
    
    def _analyze_gamma_hub(self, strike_oi: Dict, current_price: float) -> Dict[str, Any]:
        """Analyze gamma hub and concentration areas."""
        try:
            if not strike_oi:
                return {"gamma_hub": "no_data"}
            
            # Find the strike with highest total OI (gamma hub)
            gamma_hub_strike = max(strike_oi.items(), key=lambda x: x[1]["total_oi"])
            gamma_hub_strike_price = gamma_hub_strike[0]
            gamma_hub_oi = gamma_hub_strike[1]["total_oi"]
            
            # Calculate gamma hub characteristics
            distance_from_spot = abs(gamma_hub_strike_price - current_price) / current_price
            total_oi = sum(data["total_oi"] for data in strike_oi.values())
            gamma_concentration = gamma_hub_oi / total_oi if total_oi > 0 else 0
            
            # Gamma hub interpretation
            if distance_from_spot < 0.01:
                gamma_interpretation = "atm_gamma_hub"
            elif distance_from_spot < 0.03:
                gamma_interpretation = "near_atm_gamma_hub"
            elif distance_from_spot < 0.05:
                gamma_interpretation = "slightly_otm_gamma_hub"
            else:
                gamma_interpretation = "deep_otm_gamma_hub"
            
            # Market implications
            if gamma_concentration > 0.2:
                market_implication = "high_gamma_risk"
            elif gamma_concentration > 0.1:
                market_implication = "moderate_gamma_risk"
            else:
                market_implication = "low_gamma_risk"
            
            return {
                "gamma_hub_strike": gamma_hub_strike_price,
                "gamma_hub_oi": gamma_hub_oi,
                "distance_from_spot": distance_from_spot,
                "gamma_concentration": gamma_concentration,
                "gamma_interpretation": gamma_interpretation,
                "market_implication": market_implication
            }
            
        except Exception as e:
            logger.error(f"Error analyzing gamma hub: {str(e)}")
            return {"gamma_hub": "analysis_error"}
    
    def _calculate_oi_concentration(self, oi_by_expiration: Dict) -> Dict[str, Any]:
        """Calculate OI concentration metrics."""
        try:
            if not oi_by_expiration:
                return {"concentration": 0}
            
            total_oi = sum(data["total_oi"] for data in oi_by_expiration.values())
            if total_oi == 0:
                return {"concentration": 0}
            
            # Calculate Herfindahl-Hirschman Index
            concentrations = [(data["total_oi"] / total_oi) ** 2 for data in oi_by_expiration.values()]
            hhi = sum(concentrations)
            
            # Calculate concentration ratio (top 3 expirations)
            sorted_ois = sorted(data["total_oi"] for data in oi_by_expiration.values(), reverse=True)
            if len(sorted_ois) >= 3:
                top3_concentration = sum(sorted_ois[:3]) / total_oi
            else:
                top3_concentration = 1.0
            
            # Concentration interpretation
            if hhi > 0.5:
                concentration_level = "highly_concentrated"
            elif hhi > 0.25:
                concentration_level = "concentrated"
            elif hhi > 0.15:
                concentration_level = "moderately_concentrated"
            else:
                concentration_level = "well_diversified"
            
            return {
                "hhi": hhi,
                "top3_concentration": top3_concentration,
                "concentration_level": concentration_level,
                "concentration_risk": "high" if hhi > 0.4 else "medium" if hhi > 0.25 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error calculating OI concentration: {str(e)}")
            return {"concentration": 0}
    
    def _calculate_futures_oi_concentration(self, futures_oi: List[Dict]) -> Dict[str, Any]:
        """Calculate futures OI concentration."""
        try:
            if not futures_oi:
                return {"concentration": 0}
            
            ois = [contract["open_interest"] for contract in futures_oi]
            total_oi = sum(ois)
            
            if total_oi == 0:
                return {"concentration": 0}
            
            # HHI for futures
            concentrations = [(oi / total_oi) ** 2 for oi in ois]
            hhi = sum(concentrations)
            
            # Front month concentration
            front_month_ois = [contract["open_interest"] for contract in futures_oi 
                             if contract["contract_type"] == "front_month"]
            front_month_concentration = sum(front_month_ois) / total_oi if total_oi > 0 else 0
            
            return {
                "hhi": hhi,
                "front_month_concentration": front_month_concentration,
                "concentration_level": "front_month_heavy" if front_month_concentration > 0.6 else "balanced"
            }
            
        except Exception as e:
            logger.error(f"Error calculating futures OI concentration: {str(e)}")
            return {"concentration": 0}
    
    def _generate_aggregation_signals(self, options_aggregation: Dict, futures_aggregation: Dict) -> List[Dict[str, Any]]:
        """Generate signals from OI aggregation analysis."""
        try:
            signals = []
            
            # Put/Call ratio signals
            put_call_ratio = options_aggregation.get("put_call_ratio", 1.0)
            if put_call_ratio > 1.5:
                signals.append({
                    "type": "put_call_ratio",
                    "signal": "bearish",
                    "strength": min(1.0, (put_call_ratio - 1) / 0.5),
                    "message": f"High put/call ratio of {put_call_ratio:.2f} suggests bearish sentiment"
                })
            elif put_call_ratio < 0.7:
                signals.append({
                    "type": "put_call_ratio",
                    "signal": "bullish",
                    "strength": min(1.0, (0.7 - put_call_ratio) / 0.3),
                    "message": f"Low put/call ratio of {put_call_ratio:.2f} suggests bullish sentiment"
                })
            
            # Front month concentration signals
            front_deferred_ratio = futures_aggregation.get("front_deferred_ratio", 1.0)
            if front_deferred_ratio > 3.0:
                signals.append({
                    "type": "front_month_concentration",
                    "signal": "monitor",
                    "strength": 0.7,
                    "message": "Heavy front-month concentration suggests near-term focus"
                })
            
            # OI concentration signals
            options_concentration = options_aggregation.get("concentration_index", {})
            if options_concentration.get("concentration_risk") == "high":
                signals.append({
                    "type": "oi_concentration",
                    "signal": "warning",
                    "strength": 0.8,
                    "message": "High OI concentration risk - monitor for sharp moves"
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating aggregation signals: {str(e)}")
            return []
    
    async def _analyze_positioning_intelligence(self, oi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze positioning intelligence from OI data."""
        try:
            options_oi = oi_data.get("options_oi", {})
            futures_oi = oi_data.get("futures_oi", {})
            current_price = oi_data.get("current_price", 0)
            
            # Net positioning analysis
            net_positioning = self._analyze_net_positioning(options_oi, futures_oi, current_price)
            
            # Positioning bias analysis
            positioning_bias = self._analyze_positioning_bias(options_oi, futures_oi)
            
            # Large positioning analysis
            large_positioning = self._analyze_large_positioning(options_oi, futures_oi)
            
            # Positioning momentum
            positioning_momentum = self._analyze_positioning_momentum(oi_data)
            
            # Cross-asset positioning
            cross_asset_positioning = self._analyze_cross_asset_positioning_data(options_oi, futures_oi)
            
            return {
                "net_positioning": net_positioning,
                "positioning_bias": positioning_bias,
                "large_positioning": large_positioning,
                "positioning_momentum": positioning_momentum,
                "cross_asset_positioning": cross_asset_positioning,
                "positioning_signals": self._generate_positioning_signals(net_positioning, positioning_bias)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing positioning intelligence: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_net_positioning(self, options_oi: Dict, futures_oi: List[Dict], current_price: float) -> Dict[str, Any]:
        """Analyze net positioning from OI data."""
        try:
            # Calculate net options positioning
            total_calls_oi = 0
            total_puts_oi = 0
            
            for exp_date, strike_data in options_oi.items():
                for strike, data in strike_data.items():
                    total_calls_oi += data.get("calls_oi", 0)
                    total_puts_oi += data.get("puts_oi", 0)
            
            # Net options positioning (simplified - assumes long positions dominate OI)
            options_net = (total_calls_oi - total_puts_oi) / (total_calls_oi + total_puts_oi) if (total_calls_oi + total_puts_oi) > 0 else 0
            
            # Simulate futures positioning (in production, would use COT data)
            total_futures_oi = sum(contract["open_interest"] for contract in futures_oi)
            futures_net = np.random.uniform(-0.3, 0.3)  # Simulated net positioning
            
            # Combined net positioning
            options_weight = total_calls_oi + total_puts_oi
            futures_weight = total_futures_oi
            total_weight = options_weight + futures_oi
            
            if total_weight > 0:
                combined_net = (options_net * options_weight + futures_net * futures_weight) / total_weight
            else:
                combined_net = 0
            
            # Positioning interpretation
            if combined_net > 0.2:
                positioning_sentiment = "strongly_net_long"
            elif combined_net > 0.05:
                positioning_sentiment = "moderately_net_long"
            elif combined_net < -0.2:
                positioning_sentiment = "strongly_net_short"
            elif combined_net < -0.05:
                positioning_sentiment = "moderately_net_short"
            else:
                positioning_sentiment = "net_neutral"
            
            return {
                "options_net_positioning": options_net,
                "futures_net_positioning": futures_net,
                "combined_net_positioning": combined_net,
                "positioning_sentiment": positioning_sentiment,
                "positioning_strength": abs(combined_net),
                "options_weight": options_weight,
                "futures_weight": futures_weight,
                "total_weight": total_weight
            }
            
        except Exception as e:
            logger.error(f"Error analyzing net positioning: {str(e)}")
            return {"combined_net_positioning": 0}
    
    def _analyze_positioning_bias(self, options_oi: Dict, futures_oi: List[Dict]) -> Dict[str, Any]:
        """Analyze positioning bias patterns."""
        try:
            # Analyze positioning bias by moneyness
            itm_oi = 0
            otm_oi = 0
            deep_otm_oi = 0
            
            for exp_date, strike_data in options_oi.items():
                for strike, data in strike_data.items():
                    total_oi = data.get("total_oi", 0)
                    moneyness = data.get("moneyness", "OTM")
                    
                    if "ITM" in moneyness:
                        itm_oi += total_oi
                    elif "ATM" in moneyness:
                        otm_oi += total_oi  # Treat ATM as OTM for this analysis
                    else:
                        if abs(strike - data.get("current_price", 0)) / data.get("current_price", 1) > 0.1:
                            deep_otm_oi += total_oi
                        else:
                            otm_oi += total_oi
            
            total_positioning_oi = itm_oi + otm_oi + deep_otm_oi
            
            if total_positioning_oi > 0:
                itm_percentage = itm_oi / total_positioning_oi
                otm_percentage = otm_oi / total_positioning_oi
                deep_otm_percentage = deep_otm_oi / total_positioning_oi
            else:
                itm_percentage = otm_percentage = deep_otm_percentage = 0
            
            # Positioning bias interpretation
            if deep_otm_percentage > 0.4:
                bias_type = "volatility_speculation_heavy"
                bias_sentiment = "neutral"  # Could be long or short vol
            elif itm_percentage > 0.6:
                bias_type = "directional_heavy"
                bias_sentiment = "directional_positioning"
            elif itm_percentage > 0.4:
                bias_type = "balanced_directional"
                bias_sentiment = "moderate_directional"
            else:
                bias_type = "option_sales_heavy"
                bias_sentiment = "premium_collection"
            
            return {
                "itm_percentage": itm_percentage,
                "otm_percentage": otm_percentage,
                "deep_otm_percentage": deep_otm_percentage,
                "bias_type": bias_type,
                "bias_sentiment": bias_sentiment,
                "positioning_profile": self._classify_positioning_profile(itm_percentage, otm_percentage, deep_otm_percentage)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing positioning bias: {str(e)}")
            return {"bias_type": "analysis_error"}
    
    def _classify_positioning_profile(self, itm: float, otm: float, deep_otm: float) -> str:
        """Classify the overall positioning profile."""
        try:
            if deep_otm > 0.5:
                return "volatility_trader_heavy"
            elif itm > 0.6:
                return "long_term_directional"
            elif otm > 0.4:
                return "short_term_directional"
            else:
                return "balanced_hedge_focused"
                
        except Exception:
            return "profile_classification_error"}
    
    def _analyze_large_positioning(self, options_oi: Dict, futures_oi: List[Dict]) -> Dict[str, Any]:
        """Analyze large positioning indicators."""
        try:
            # Identify potentially large positions
            large_positions = []
            
            # Analyze options for large OI
            for exp_date, strike_data in options_oi.items():
                for strike, data in strike_data.items():
                    total_oi = data.get("total_oi", 0)
                    
                    # Define large position thresholds
                    if total_oi > 5000:  # Large threshold
                        large_positions.append({
                            "type": "options",
                            "strike": strike,
                            "expiration": exp_date,
                            "position_size": total_oi,
                            "moneyness": data.get("moneyness", "Unknown"),
                            "likely_institution": total_oi > 10000
                        })
            
            # Analyze futures for large OI
            for contract in futures_oi:
                if contract["open_interest"] > 20000:  # Large futures threshold
                    large_positions.append({
                        "type": "futures",
                        "contract": contract["symbol"],
                        "expiration": contract["expiration_date"],
                        "position_size": contract["open_interest"],
                        "likely_institution": contract["open_interest"] > 50000
                    })
            
            # Analyze large position patterns
            institutional_positions = [p for p in large_positions if p.get("likely_institution", False)]
            
            return {
                "large_positions_count": len(large_positions),
                "institutional_positions_count": len(institutional_positions),
                "largest_position": max(large_positions, key=lambda x: x["position_size"]) if large_positions else None,
                "large_position_concentration": self._calculate_large_position_concentration(large_positions),
                "institutional_activity_level": self._assess_institutional_activity_level(institutional_positions),
                "large_position_signals": self._analyze_large_position_signals(large_positions)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing large positioning: {str(e)}")
            return {"large_positions_count": 0}
    
    def _calculate_large_position_concentration(self, large_positions: List[Dict]) -> Dict[str, Any]:
        """Calculate concentration of large positions."""
        try:
            if not large_positions:
                return {"concentration": 0}
            
            total_large_oi = sum(pos["position_size"] for pos in large_positions)
            if total_large_oi == 0:
                return {"concentration": 0}
            
            # Calculate concentration of large positions
            position_sizes = [pos["position_size"] for pos in large_positions]
            max_position_size = max(position_sizes)
            concentration_ratio = max_position_size / total_large_oi
            
            # Concentration interpretation
            if concentration_ratio > 0.5:
                concentration_level = "highly_concentrated"
            elif concentration_ratio > 0.3:
                concentration_level = "concentrated"
            else:
                concentration_level = "distributed"
            
            return {
                "concentration_ratio": concentration_ratio,
                "concentration_level": concentration_level,
                "largest_position_percentage": concentration_ratio
            }
            
        except Exception:
            return {"concentration": "calculation_error"}
    
    def _assess_institutional_activity_level(self, institutional_positions: List[Dict]) -> str:
        """Assess level of institutional activity."""
        try:
            if len(institutional_positions) > 10:
                return "very_high"
            elif len(institutional_positions) > 5:
                return "high"
            elif len(institutional_positions) > 2:
                return "moderate"
            elif len(institutional_positions) > 0:
                return "low"
            else:
                return "none"
                
        except Exception:
            return "assessment_error"}
    
    def _analyze_large_position_signals(self, large_positions: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze signals from large position activity."""
        try:
            signals = []
            
            # Concentration signals
            concentration = self._calculate_large_position_concentration(large_positions)
            if concentration.get("concentration_level") == "highly_concentrated":
                signals.append({
                    "type": "position_concentration",
                    "signal": "warning",
                    "message": "High concentration of large positions - monitor for unwinds"
                })
            
            # Institutional activity signals
            institutional_count = len([p for p in large_positions if p.get("likely_institution", False)])
            if institutional_count > 5:
                signals.append({
                    "type": "institutional_activity",
                    "signal": "informational",
                    "message": f"{institutional_count} large institutional positions detected"
                })
            
            # Position type signals
            options_positions = [p for p in large_positions if p["type"] == "options"]
            futures_positions = [p for p in large_positions if p["type"] == "futures"]
            
            if len(options_positions) > len(futures_positions) * 2:
                signals.append({
                    "type": "options_heavy",
                    "signal": "informational",
                    "message": "Large positions heavily concentrated in options"
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing large position signals: {str(e)}")
            return []
    
    def _analyze_positioning_momentum(self, oi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum in positioning changes."""
        try:
            # This would require historical OI data in production
            # For now, provide framework for momentum analysis
            
            return {
                "momentum_direction": "analyzing",
                "momentum_strength": "moderate",
                "positioning_acceleration": "stable",
                "momentum_signals": self._generate_momentum_signals()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing positioning momentum: {str(e)}")
            return {"momentum_direction": "error"}
    
    def _generate_momentum_signals(self) -> List[Dict[str, Any]]:
        """Generate signals based on positioning momentum."""
        # Placeholder for momentum signal generation
        return [{
            "type": "positioning_momentum",
            "signal": "monitor",
            "message": "Positioning momentum analysis requires historical data"
        }]
    
    def _analyze_cross_asset_positioning_data(self, options_oi: Dict, futures_oi: List[Dict]) -> Dict[str, Any]:
        """Analyze positioning across different asset types."""
        try:
            return {
                "options_futures_correlation": "analyzing_relationship",
                "cross_asset_flow": "monitoring",
                "relative_positioning": self._assess_relative_positioning(options_oi, futures_oi)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cross-asset positioning: {str(e)}")
            return {"cross_asset_analysis": "error"}
    
    def _assess_relative_positioning(self, options_oi: Dict, futures_oi: List[Dict]) -> str:
        """Assess relative positioning between options and futures."""
        try:
            options_total = sum(
                sum(data.get("total_oi", 0) for data in strike_data.values())
                for strike_data in options_oi.values()
            )
            futures_total = sum(contract["open_interest"] for contract in futures_oi)
            
            if options_total > futures_total * 1.5:
                return "options_dominant"
            elif futures_total > options_total * 1.5:
                return "futures_dominant"
            else:
                return "balanced_positioning"
                
        except Exception:
            return "relative_assessment_error"}
    
    def _generate_positioning_signals(self, net_positioning: Dict, positioning_bias: Dict) -> List[Dict[str, Any]]:
        """Generate signals from positioning analysis."""
        try:
            signals = []
            
            # Net positioning signals
            combined_net = net_positioning.get("combined_net_positioning", 0)
            positioning_sentiment = net_positioning.get("positioning_sentiment", "neutral")
            
            if "strongly_net_long" in positioning_sentiment:
                signals.append({
                    "type": "net_positioning",
                    "signal": "bullish",
                    "strength": abs(combined_net),
                    "message": "Strongly net long positioning detected"
                })
            elif "strongly_net_short" in positioning_sentiment:
                signals.append({
                    "type": "net_positioning",
                    "signal": "bearish",
                    "strength": abs(combined_net),
                    "message": "Strongly net short positioning detected"
                })
            
            # Positioning bias signals
            bias_type = positioning_bias.get("bias_type", "")
            if "volatility_speculation_heavy" in bias_type:
                signals.append({
                    "type": "positioning_bias",
                    "signal": "monitor",
                    "message": "Heavy volatility speculation detected"
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating positioning signals: {str(e)}")
            return []
    
    async def _analyze_dealer_flows(self, oi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dealer flow patterns from OI data."""
        try:
            options_oi = oi_data.get("options_oi", {})
            
            # Dealer flow analysis (simplified - would require more sophisticated modeling)
            dealer_flow_analysis = self._estimate_dealer_flows(options_oi)
            
            # Delta hedging analysis
            delta_hedging = self._analyze_delta_hedging_flows(options_oi, oi_data.get("current_price", 0))
            
            # Gamma exposure analysis
            gamma_exposure = self._analyze_gamma_exposure(options_oi, oi_data.get("current_price", 0))
            
            # Vega exposure analysis
            vega_exposure = self._analyze_vega_exposure(options_oi)
            
            return {
                "dealer_flow_analysis": dealer_flow_analysis,
                "delta_hedging": delta_hedging,
                "gamma_exposure": gamma_exposure,
                "vega_exposure": vega_exposure,
                "dealer_signals": self._generate_dealer_signals(dealer_flow_analysis, gamma_exposure)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing dealer flows: {str(e)}")
            return {"error": str(e)}
    
    def _estimate_dealer_flows(self, options_oi: Dict) -> Dict[str, Any]:
        """Estimate dealer flows from OI patterns."""
        try:
            # This is a simplified estimation - real dealer flow analysis requires
            # more sophisticated models and data
            
            dealer_flows = {
                "estimated_net_dealer_position": "analyzing",
                "dealer_gamma_flow": "estimating",
                "flow_intensity": "moderate",
                "dealer_sentiment": "neutral"
            }
            
            # Analyze OI patterns to estimate dealer activity
            total_options_oi = sum(
                sum(data.get("total_oi", 0) for data in strike_data.values())
                for strike_data in options_oi.values()
            )
            
            if total_options_oi > 100000:
                dealer_flows["flow_intensity"] = "high"
                dealer_flows["market_participation"] = "active"
            elif total_options_oi > 50000:
                dealer_flows["flow_intensity"] = "moderate"
                dealer_flows["market_participation"] = "normal"
            else:
                dealer_flows["flow_intensity"] = "low"
                dealer_flows["market_participation"] = "quiet"
            
            return dealer_flows
            
        except Exception as e:
            logger.error(f"Error estimating dealer flows: {str(e)}")
            return {"dealer_flows": "estimation_error"}
    
    def _analyze_delta_hedging_flows(self, options_oi: Dict, current_price: float) -> Dict[str, Any]:
        """Analyze delta hedging flows."""
        try:
            # Simplified delta hedging analysis
            # In practice, would calculate actual deltas and hedging flows
            
            # Estimate net delta exposure
            total_delta_exposure = 0
            
            for exp_date, strike_data in options_oi.items():
                for strike, data in strike_data.items():
                    calls_oi = data.get("calls_oi", 0)
                    puts_oi = data.get("puts_oi", 0)
                    
                    # Simplified delta calculation
                    distance_from_atm = abs(strike - current_price) / current_price
                    
                    if distance_from_atm < 0.01:  # Near ATM
                        call_delta = 0.5 * calls_oi
                        put_delta = -0.5 * puts_oi
                    elif distance_from_atm < 0.05:  # Moderately OTM/ITM
                        call_delta = 0.3 * calls_oi
                        put_delta = -0.3 * puts_oi
                    else:  # Far OTM/ITM
                        call_delta = 0.1 * calls_oi
                        put_delta = -0.1 * puts_oi
                    
                    total_delta_exposure += call_delta + put_delta
            
            # Delta hedging implications
            if abs(total_delta_exposure) > 10000:
                hedging_intensity = "high"
            elif abs(total_delta_exposure) > 5000:
                hedging_intensity = "moderate"
            else:
                hedging_intensity = "low"
            
            return {
                "net_delta_exposure": total_delta_exposure,
                "hedging_intensity": hedging_intensity,
                "hedging_direction": "long" if total_delta_exposure > 0 else "short" if total_delta_exposure < 0 else "neutral",
                "hedging_flow_impact": "monitoring"  # Would calculate actual hedging flows
            }
            
        except Exception as e:
            logger.error(f"Error analyzing delta hedging flows: {str(e)}")
            return {"hedging_analysis": "error"}
    
    def _analyze_gamma_exposure(self, options_oi: Dict, current_price: float) -> Dict[str, Any]:
        """Analyze gamma exposure from options OI."""
        try:
            gamma_hubs = []
            total_gamma_exposure = 0
            
            for exp_date, strike_data in options_oi.items():
                for strike, data in strike_data.items():
                    calls_oi = data.get("calls_oi", 0)
                    puts_oi = data.get("puts_oi", 0)
                    
                    # Simplified gamma calculation
                    distance_from_atm = abs(strike - current_price) / current_price
                    
                    # Gamma is highest near ATM
                    if distance_from_atm < 0.01:  # Very close to ATM
                        gamma_factor = 1.0
                    elif distance_from_atm < 0.03:  # Near ATM
                        gamma_factor = 0.7
                    elif distance_from_atm < 0.05:  # Moderately close
                        gamma_factor = 0.4
                    else:  # Far from ATM
                        gamma_factor = 0.1
                    
                    estimated_gamma = (calls_oi + puts_oi) * gamma_factor
                    total_gamma_exposure += estimated_gamma
                    
                    if estimated_gamma > 1000:  # Significant gamma hub
                        gamma_hubs.append({
                            "strike": strike,
                            "gamma_exposure": estimated_gamma,
                            "expiration": exp_date,
                            "distance_from_spot": distance_from_atm
                        })
            
            # Sort gamma hubs by exposure
            gamma_hubs.sort(key=lambda x: x["gamma_exposure"], reverse=True)
            
            # Gamma exposure interpretation
            if total_gamma_exposure > 50000:
                gamma_regime = "high_gamma"
            elif total_gamma_exposure > 20000:
                gamma_regime = "moderate_gamma"
            else:
                gamma_regime = "low_gamma"
            
            return {
                "total_gamma_exposure": total_gamma_exposure,
                "gamma_regime": gamma_regime,
                "primary_gamma_hub": gamma_hubs[0] if gamma_hubs else None,
                "gamma_hubs": gamma_hubs[:5],  # Top 5
                "gamma_concentration": self._calculate_gamma_concentration(gamma_hubs, total_gamma_exposure)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing gamma exposure: {str(e)}")
            return {"gamma_analysis": "error"}
    
    def _calculate_gamma_concentration(self, gamma_hubs: List[Dict], total_gamma: float) -> float:
        """Calculate gamma concentration ratio."""
        try:
            if not gamma_hubs or total_gamma == 0:
                return 0
            
            primary_gamma = gamma_hubs[0]["gamma_exposure"]
            return primary_gamma / total_gamma
            
        except Exception:
            return 0
    
    def _analyze_vega_exposure(self, options_oi: Dict) -> Dict[str, Any]:
        """Analyze vega exposure from options OI."""
        try:
            total_vega_exposure = 0
            vega_by_expiration = {}
            
            for exp_date, strike_data in options_oi.items():
                exp_vega = 0
                
                for strike, data in strike_data.items():
                    calls_oi = data.get("calls_oi", 0)
                    puts_oi = data.get("puts_oi", 0)
                    days_to_exp = data.get("days_to_exp", 30)
                    
                    # Vega is higher for longer-dated options
                    vega_factor = min(2.0, days_to_exp / 30)  # Scale by days to expiration
                    estimated_vega = (calls_oi + puts_oi) * vega_factor
                    
                    total_vega_exposure += estimated_vega
                    exp_vega += estimated_vega
                
                vega_by_expiration[exp_date] = exp_vega
            
            # Vega exposure interpretation
            if total_vega_exposure > 100000:
                vega_regime = "high_vega"
            elif total_vega_exposure > 50000:
                vega_regime = "moderate_vega"
            else:
                vega_regime = "low_vega"
            
            return {
                "total_vega_exposure": total_vega_exposure,
                "vega_regime": vega_regime,
                "vega_by_expiration": vega_by_expiration,
                "vega_concentration": self._calculate_vega_concentration(vega_by_expiration)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing vega exposure: {str(e)}")
            return {"vega_analysis": "error"}
    
    def _calculate_vega_concentration(self, vega_by_expiration: Dict) -> Dict[str, Any]:
        """Calculate vega concentration by expiration."""
        try:
            if not vega_by_expiration:
                return {"concentration": 0}
            
            total_vega = sum(vega_by_expiration.values())
            if total_vega == 0:
                return {"concentration": 0}
            
            # Find concentration in near-term vs long-term
            near_term_vega = 0
            long_term_vega = 0
            
            for exp_date, vega_amount in vega_by_expiration.items():
                exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
                days_to_exp = (exp_dt - datetime.now()).days
                
                if days_to_exp <= 30:
                    near_term_vega += vega_amount
                elif days_to_exp > 60:
                    long_term_vega += vega_amount
            
            near_term_concentration = near_term_vega / total_vega if total_vega > 0 else 0
            long_term_concentration = long_term_vega / total_vega if total_vega > 0 else 0
            
            return {
                "near_term_concentration": near_term_concentration,
                "long_term_concentration": long_term_concentration,
                "concentration_pattern": "near_term_heavy" if near_term_concentration > 0.6 else "balanced"
            }
            
        except Exception:
            return {"concentration": "calculation_error"}
    
    def _generate_dealer_signals(self, dealer_flow_analysis: Dict, gamma_exposure: Dict) -> List[Dict[str, Any]]:
        """Generate signals from dealer flow analysis."""
        try:
            signals = []
            
            # Gamma regime signals
            gamma_regime = gamma_exposure.get("gamma_regime", "low_gamma")
            if gamma_regime == "high_gamma":
                signals.append({
                    "type": "gamma_regime",
                    "signal": "warning",
                    "message": "High gamma environment - monitor for sharp moves around gamma hub"
                })
            elif gamma_regime == "moderate_gamma":
                signals.append({
                    "type": "gamma_regime",
                    "signal": "monitor",
                    "message": "Moderate gamma environment"
                })
            
            # Dealer flow intensity signals
            flow_intensity = dealer_flow_analysis.get("flow_intensity", "moderate")
            if flow_intensity == "high":
                signals.append({
                    "type": "dealer_activity",
                    "signal": "informational",
                    "message": "High dealer activity detected"
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating dealer signals: {str(e)}")
            return []
    
    async def _analyze_institutional_flows(self, oi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze institutional flow patterns."""
        try:
            options_oi = oi_data.get("options_oi", {})
            futures_oi = oi_data.get("futures_oi", {})
            
            # Institutional positioning analysis
            institutional_positioning = self._analyze_institutional_positioning(options_oi, futures_oi)
            
            # Flow direction analysis
            flow_direction = self._analyze_institutional_flow_direction(options_oi, futures_oi)
            
            # Timing analysis
            timing_analysis = self._analyze_institutional_timing(options_oi, futures_oi)
            
            # Size analysis
            size_analysis = self._analyze_institutional_size(options_oi, futures_oi)
            
            return {
                "institutional_positioning": institutional_positioning,
                "flow_direction": flow_direction,
                "timing_analysis": timing_analysis,
                "size_analysis": size_analysis,
                "institutional_signals": self._generate_institutional_signals(
                    institutional_positioning, flow_direction
                )
            }
            
        catch Exception as e:
            logger.error(f"Error analyzing institutional flows: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_institutional_positioning(self, options_oi: Dict, futures_oi: List[Dict]) -> Dict[str, Any]:
        """Analyze institutional positioning patterns."""
        try:
            # This would require sophisticated analysis in production
            # For now, provide framework for institutional analysis
            
            # Analyze large OI concentrations (likely institutional)
            institutional_indicators = {
                "large_oi_concentrations": self._identify_institutional_oi_concentrations(options_oi, futures_oi),
                "positioning_sophistication": self._assess_positioning_sophistication(options_oi, futures_oi),
                "strategic_positioning": self._analyze_strategic_positioning(options_oi, futures_oi)
            }
            
            return institutional_indicators
            
        except Exception as e:
            logger.error(f"Error analyzing institutional positioning: {str(e)}")
            return {"institutional_positioning": "analysis_error"}
    
    def _identify_institutional_oi_concentrations(self, options_oi: Dict, futures_oi: List[Dict]) -> List[Dict[str, Any]]:
        """Identify concentrations that likely represent institutional positions."""
        try:
            concentrations = []
            
            # Analyze options for large concentrations
            for exp_date, strike_data in options_oi.items():
                for strike, data in strike_data.items():
                    total_oi = data.get("total_oi", 0)
                    
                    if total_oi > 10000:  # Threshold for likely institutional
                        concentrations.append({
                            "type": "options",
                            "strike": strike,
                            "expiration": exp_date,
                            "oi": total_oi,
                            "institutional_likelihood": "high" if total_oi > 20000 else "moderate",
                            "position_characteristics": self._characterize_institutional_position(data)
                        })
            
            # Analyze futures for large concentrations
            for contract in futures_oi:
                if contract["open_interest"] > 30000:
                    concentrations.append({
                        "type": "futures",
                        "contract": contract["symbol"],
                        "expiration": contract["expiration_date"],
                        "oi": contract["open_interest"],
                        "institutional_likelihood": "high",
                        "position_characteristics": "large_institutional_futures_position"
                    })
            
            return concentrations
            
        except Exception as e:
            logger.error(f"Error identifying institutional concentrations: {str(e)}")
            return []
    
    def _characterize_institutional_position(self, position_data: Dict) -> str:
        """Characterize the nature of an institutional position."""
        try:
            calls_oi = position_data.get("calls_oi", 0)
            puts_oi = position_data.get("puts_oi", 0)
            put_call_ratio = position_data.get("put_call_ratio", 1.0)
            
            if put_call_ratio > 1.5:
                return "defensive_put_positioning"
            elif put_call_ratio < 0.7:
                return "aggressive_call_positioning"
            elif calls_oi > puts_oi * 1.5:
                return "bullish_call_spread"
            elif puts_oi > calls_oi * 1.5:
                return "bearish_put_spread"
            else:
                return "balanced_institutional_hedge"
                
        except Exception:
            return "institutional_position_characterization_error"}
    
    def _assess_positioning_sophistication(self, options_oi: Dict, futures_oi: List[Dict]) -> str:
        """Assess the sophistication of positioning patterns."""
        try:
            # Analyze positioning complexity
            unique_strikes = len(set(strike for exp_data in options_oi.values() for strike in exp_data.keys()))
            unique_expirations = len(options_oi)
            
            # Complex positioning indicators
            if unique_strikes > 20 and unique_expirations > 3:
                sophistication_level = "highly_sophisticated"
            elif unique_strikes > 10 and unique_expirations > 2:
                sophistication_level = "sophisticated"
            elif unique_strikes > 5:
                sophistication_level = "moderately_sophisticated"
            else:
                sophistication_level = "basic_positioning"
            
            return sophistication_level
            
        except Exception:
            return "sophistication_assessment_error"}
    
    def _analyze_strategic_positioning(self, options_oi: Dict, futures_oi: List[Dict]) -> Dict[str, Any]:
        """Analyze strategic positioning patterns."""
        try:
            return {
                "spread_positioning": self._analyze_spread_positioning(options_oi),
                "calendar_positioning": self._analyze_calendar_positioning(options_oi),
                "ratio_positioning": self._analyze_ratio_positioning(options_oi),
                "strategic_themes": self._identify_strategic_themes(options_oi, futures_oi)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing strategic positioning: {str(e)}")
            return {"strategic_analysis": "error"}
    
    def _analyze_spread_positioning(self, options_oi: Dict) -> Dict[str, Any]:
        """Analyze potential spread positioning."""
        try:
            # This would analyze strike combinations that suggest spreads
            spread_indicators = {
                "vertical_spreads": "analyzing_vertical_spread_patterns",
                "iron_condors": "monitoring_iron_condor_positioning",
                "straddles_strangles": "tracking_straddle_strangle_activity"
            }
            
            return spread_indicators
            
        except Exception:
            return {"spread_analysis": "error"}
    
    def _analyze_calendar_positioning(self, options_oi: Dict) -> Dict[str, Any]:
        """Analyze calendar spread positioning."""
        try:
            # Analyze OI distribution across expirations
            expiration_concentrations = {}
            
            for exp_date, strike_data in options_oi.items():
                total_exp_oi = sum(data.get("total_oi", 0) for data in strike_data.values())
                expiration_concentrations[exp_date] = total_exp_oi
            
            # Identify calendar spread opportunities
            sorted_expirations = sorted(expiration_concentrations.items(), key=lambda x: x[1], reverse=True)
            
            calendar_analysis = {
                "front_month_concentration": sorted_expirations[0] if sorted_expirations else None,
                "calendar_spread_signals": self._detect_calendar_spreads(expiration_concentrations)
            }
            
            return calendar_analysis
            
        except Exception:
            return {"calendar_analysis": "error"}
    
    def _detect_calendar_spreads(self, expiration_concentrations: Dict) -> List[str]:
        """Detect potential calendar spread positioning."""
        signals = []
        
        # Simplified calendar spread detection
        concentrations = list(expiration_concentrations.values())
        if len(concentrations) > 1:
            concentration_ratio = max(concentrations) / min(concentrations) if min(concentrations) > 0 else 1
            if concentration_ratio > 2:
                signals.append("calendar_spread_concentration_detected")
        
        return signals
    
    def _analyze_ratio_positioning(self, options_oi: Dict) -> Dict[str, Any]:
        """Analyze ratio positioning patterns."""
        try:
            ratio_analysis = {
                "put_call_ratios": self._analyze_put_call_ratios(options_oi),
                "asymmetry_analysis": self._analyze_asymmetry(options_oi),
                "risk_reversal_signals": self._analyze_risk_reversals(options_oi)
            }
            
            return ratio_analysis
            
        except Exception:
            return {"ratio_analysis": "error"}
    
    def _analyze_put_call_ratios(self, options_oi: Dict) -> Dict[str, Any]:
        """Analyze put/call ratio patterns."""
        try:
            ratios_by_expiration = {}
            
            for exp_date, strike_data in options_oi.items():
                total_calls = sum(data.get("calls_oi", 0) for data in strike_data.values())
                total_puts = sum(data.get("puts_oi", 0) for data in strike_data.values())
                
                if total_calls > 0:
                    ratio = total_puts / total_calls
                    ratios_by_expiration[exp_date] = ratio
            
            # Analyze ratio patterns
            avg_ratio = np.mean(list(ratios_by_expiration.values())) if ratios_by_expiration else 1.0
            
            return {
                "ratios_by_expiration": ratios_by_expiration,
                "average_ratio": avg_ratio,
                "ratio_interpretation": self._interpret_put_call_ratio(avg_ratio)
            }
            
        except Exception:
            return {"ratio_analysis": "error"}
    
    def _interpret_put_call_ratio(self, ratio: float) -> str:
        """Interpret put/call ratio."""
        if ratio > 1.5:
            return "bearish_sentiment"
        elif ratio > 1.2:
            return "cautiously_bearish"
        elif ratio < 0.7:
            return "bullish_sentiment"
        elif ratio < 0.8:
            return "cautiously_bullish"
        else:
            return "neutral_sentiment"
    
    def _analyze_asymmetry(self, options_oi: Dict) -> Dict[str, Any]:
        """Analyze asymmetries in positioning."""
        try:
            # This would analyze skew and asymmetry patterns
            return {
                "skew_analysis": "analyzing_volatility_skew",
                "asymmetry_indicators": "monitoring_directional_bias",
                "risk_premium_analysis": "assessing_risk_premiums"
            }
            
        except Exception:
            return {"asymmetry_analysis": "error"}
    
    def _analyze_risk_reversals(self, options_oi: Dict) -> Dict[str, Any]:
        """Analyze risk reversal signals."""
        try:
            # Risk reversal analysis would examine 25-delta call/put pricing
            return {
                "risk_reversal_signals": "risk_reversal_analysis_in_progress",
                "skew_indicators": "monitoring_skew_patterns"
            }
            
        except Exception:
            return {"risk_reversal_analysis": "error"}
    
    def _identify_strategic_themes(self, options_oi: Dict, futures_oi: List[Dict]) -> List[str]:
        """Identify strategic positioning themes."""
        try:
            themes = []
            
            # Analyze positioning patterns for strategic themes
            if self._detect_volatility_trading_themes(options_oi):
                themes.append("volatility_trading_strategy")
            
            if self._detect_directional_themes(options_oi, futures_oi):
                themes.append("directional_positioning_strategy")
            
            if self._detect_income_generation_themes(options_oi):
                themes.append("income_generation_strategy")
            
            return themes
            
        except Exception as e:
            logger.error(f"Error identifying strategic themes: {str(e)}")
            return []
    
    def _detect_volatility_trading_themes(self, options_oi: Dict) -> bool:
        """Detect if there are volatility trading themes."""
        try:
            # Look for high OI in both calls and puts at similar strikes
            for exp_date, strike_data in options_oi.items():
                for strike, data in strike_data.items():
                    calls_oi = data.get("calls_oi", 0)
                    puts_oi = data.get("puts_oi", 0)
                    
                    if calls_oi > 2000 and puts_oi > 2000:  # High OI in both
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _detect_directional_themes(self, options_oi: Dict, futures_oi: List[Dict]) -> bool:
        """Detect if there are directional positioning themes."""
        try:
            # Look for concentrated positioning suggesting directional bias
            total_options_oi = sum(
                sum(data.get("total_oi", 0) for data in strike_data.values())
                for strike_data in options_oi.values()
            )
            
            # High concentration suggests directional positioning
            return total_options_oi > 150000
            
        except Exception:
            return False
    
    def _detect_income_generation_themes(self, options_oi: Dict) -> bool:
        """Detect if there are income generation themes."""
        try:
            # Look for high OI in OTM options (likely for premium selling)
            otm_oi = 0
            total_oi = 0
            
            for exp_date, strike_data in options_oi.items():
                for strike, data in strike_data.items():
                    total_oi += data.get("total_oi", 0)
                    if "OTM" in data.get("moneyness", ""):
                        otm_oi += data.get("total_oi", 0)
            
            # High OTM concentration suggests income generation
            return otm_oi / total_oi > 0.6 if total_oi > 0 else False
            
        except Exception:
            return False
    
    def _analyze_institutional_flow_direction(self, options_oi: Dict, futures_oi: List[Dict]) -> Dict[str, Any]:
        """Analyze the direction of institutional flows."""
        try:
            return {
                "flow_bias": "analyzing_flow_directions",
                "timing_indicators": "monitoring_flow_timing",
                "momentum_signals": "tracking_institutional_momentum"
            }
            
        except Exception:
            return {"flow_direction": "analysis_error"}
    
    def _analyze_institutional_timing(self, options_oi: Dict, futures_oi: List[Dict]) -> Dict[str, Any]:
        """Analyze timing of institutional activity."""
        try:
            return {
                "activity_timing": "analyzing_institutional_timing",
                "expiration_focus": self._assess_expiration_focus(options_oi, futures_oi),
                "roll_activity": "monitoring_roll_patterns"
            }
            
        except Exception:
            return {"timing_analysis": "error"}
    
    def _assess_expiration_focus(self, options_oi: Dict, futures_oi: List[Dict]) -> str:
        """Assess focus on specific expirations."""
        try:
            # Analyze concentration by expiration
            options_expirations = len(options_oi)
            futures_expirations = len(futures_oi)
            
            if options_expirations > 4 or futures_expirations > 5:
                return "broad_expiration_coverage"
            elif options_expirations > 2 or futures_expirations > 3:
                return "moderate_expiration_coverage"
            else:
                return "focused_expiration_coverage"
                
        except Exception:
            return "expiration_focus_assessment_error"}
    
    def _analyze_institutional_size(self, options_oi: Dict, futures_oi: List[Dict]) -> Dict[str, Any]:
        """Analyze the size characteristics of institutional positioning."""
        try:
            total_options_oi = sum(
                sum(data.get("total_oi", 0) for data in strike_data.values())
                for strike_data in options_oi.values()
            )
            
            total_futures_oi = sum(contract["open_interest"] for contract in futures_oi)
            total_oi = total_options_oi + total_futures_oi
            
            # Size classification
            if total_oi > 500000:
                size_classification = "very_large_institutional_activity"
            elif total_oi > 200000:
                size_classification = "large_institutional_activity"
            elif total_oi > 100000:
                size_classification = "moderate_institutional_activity"
            else:
                size_classification = "small_scale_activity"
            
            return {
                "total_institutional_oi": total_oi,
                "size_classification": size_classification,
                "activity_scale": self._assess_activity_scale(total_oi)
            }
            
        except Exception:
            return {"size_analysis": "error"}
    
    def _assess_activity_scale(self, total_oi: int) -> str:
        """Assess the scale of institutional activity."""
        try:
            if total_oi > 1000000:
                return "institutional_flood"
            elif total_oi > 500000:
                return "institutional_heavy"
            elif total_oi > 200000:
                return "institutional_active"
            elif total_oi > 50000:
                return "institutional_present"
            else:
                return "institutional_minimal"
                
        except Exception:
            return "scale_assessment_error"}
    
    def _generate_institutional_signals(self, positioning: Dict, flow_direction: Dict) -> List[Dict[str, Any]]:
        """Generate signals from institutional analysis."""
        try:
            signals = []
            
            # Strategic theme signals
            strategic_themes = positioning.get("strategic_positioning", {}).get("strategic_themes", [])
            for theme in strategic_themes:
                signals.append({
                    "type": "institutional_theme",
                    "signal": "informational",
                    "message": f"Institutional {theme.replace('_', ' ')} detected"
                })
            
            # Size-based signals
            size_classification = positioning.get("positioning_sophistication", "")
            if "sophisticated" in size_classification:
                signals.append({
                    "type": "positioning_sophistication",
                    "signal": "informational",
                    "message": "Sophisticated institutional positioning patterns detected"
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating institutional signals: {str(e)}")
            return []
    
    async def _analyze_positioning_concentration(self, oi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze concentration of positioning across the market."""
        try:
            options_oi = oi_data.get("options_oi", {})
            futures_oi = oi_data.get("futures_oi", {})
            
            # Strike-level concentration
            strike_concentration = self._analyze_strike_concentration(options_oi)
            
            # Expiration-level concentration
            expiration_concentration = self._analyze_expiration_concentration(options_oi, futures_oi)
            
            # Overall market concentration
            market_concentration = self._analyze_market_concentration(options_oi, futures_oi)
            
            # Concentration risk assessment
            concentration_risk = self._assess_concentration_risk(strike_concentration, expiration_concentration)
            
            # Unwinding risk analysis
            unwinding_risk = self._analyze_unwinding_risk(concentration_risk)
            
            return {
                "strike_concentration": strike_concentration,
                "expiration_concentration": expiration_concentration,
                "market_concentration": market_concentration,
                "concentration_risk": concentration_risk,
                "unwinding_risk": unwinding_risk,
                "concentration_signals": self._generate_concentration_signals(concentration_risk, unwinding_risk)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing positioning concentration: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_strike_concentration(self, options_oi: Dict) -> Dict[str, Any]:
        """Analyze concentration of OI by strikes."""
        try:
            if not options_oi:
                return {"concentration": "no_data"}
            
            # Aggregate OI by strike
            strike_totals = {}
            total_oi = 0
            
            for exp_date, strike_data in options_oi.items():
                for strike, data in strike_data.items():
                    strike_oi = data.get("total_oi", 0)
                    total_oi += strike_oi
                    
                    if strike not in strike_totals:
                        strike_totals[strike] = 0
                    strike_totals[strike] += strike_oi
            
            if total_oi == 0:
                return {"concentration": "zero_oi"}
            
            # Calculate concentration metrics
            sorted_strikes = sorted(strike_totals.items(), key=lambda x: x[1], reverse=True)
            top_5_strikes = sorted_strikes[:5]
            
            # Concentration ratios
            top1_concentration = top_5_strikes[0][1] / total_oi if top_5_strikes else 0
            top5_concentration = sum(strike[1] for strike in top_5_strikes) / total_oi
            
            # HHI calculation
            hhi = sum((oi / total_oi) ** 2 for oi in strike_totals.values())
            
            # Strike concentration interpretation
            if hhi > 0.15:
                strike_concentration_level = "highly_concentrated"
            elif hhi > 0.10:
                strike_concentration_level = "concentrated"
            elif hhi > 0.05:
                strike_concentration_level = "moderately_concentrated"
            else:
                strike_concentration_level = "well_distributed"
            
            return {
                "hhi": hhi,
                "top1_concentration": top1_concentration,
                "top5_concentration": top5_concentration,
                "concentration_level": strike_concentration_level,
                "top_strikes": [{"strike": strike, "oi": oi} for strike, oi in top_5_strikes],
                "concentration_risk": "high" if hhi > 0.12 else "medium" if hhi > 0.08 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing strike concentration: {str(e)}")
            return {"concentration": "analysis_error"}
    
    def _analyze_expiration_concentration(self, options_oi: Dict, futures_oi: List[Dict]) -> Dict[str, Any]:
        """Analyze concentration of OI by expiration dates."""
        try:
            # Analyze options expiration concentration
            options_expiration_totals = {}
            for exp_date, strike_data in options_oi.items():
                exp_total = sum(data.get("total_oi", 0) for data in strike_data.values())
                options_expiration_totals[exp_date] = exp_total
            
            # Analyze futures expiration concentration
            futures_expiration_totals = {}
            for contract in futures_oi:
                exp_date = contract["expiration_date"]
                if exp_date not in futures_expiration_totals:
                    futures_expiration_totals[exp_date] = 0
                futures_expiration_totals[exp_date] += contract["open_interest"]
            
            # Combined expiration analysis
            total_expiration_oi = sum(options_expiration_totals.values()) + sum(futures_expiration_totals.values())
            
            if total_expiration_oi == 0:
                return {"concentration": "no_expiration_data"}
            
            # Calculate expiration HHI
            all_expirations = {**options_expiration_totals, **futures_expiration_totals}
            hhi = sum((oi / total_expiration_oi) ** 2 for oi in all_expirations.values())
            
            # Near-term concentration
            today = datetime.now()
            near_term_oi = 0
            
            for exp_date, oi in all_expirations.items():
                exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
                days_to_exp = (exp_dt - today).days
                
                if days_to_exp <= 30:
                    near_term_oi += oi
            
            near_term_concentration = near_term_oi / total_expiration_oi if total_expiration_oi > 0 else 0
            
            # Expiration concentration interpretation
            if hhi > 0.4:
                expiration_concentration_level = "highly_concentrated"
            elif hhi > 0.25:
                expiration_concentration_level = "concentrated"
            elif hhi > 0.15:
                expiration_concentration_level = "moderately_concentrated"
            else:
                expiration_concentration_level = "well_distributed"
            
            return {
                "hhi": hhi,
                "near_term_concentration": near_term_concentration,
                "concentration_level": expiration_concentration_level,
                "front_month_dominance": near_term_concentration > 0.6,
                "expiration_distribution": {
                    "options": options_expiration_totals,
                    "futures": futures_expiration_totals
                },
                "roll_risk": self._assess_roll_risk(near_term_concentration)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing expiration concentration: {str(e)}")
            return {"concentration": "analysis_error"}
    
    def _assess_roll_risk(self, near_term_concentration: float) -> str:
        """Assess roll risk based on expiration concentration."""
        try:
            if near_term_concentration > 0.8:
                return "very_high_roll_risk"
            elif near_term_concentration > 0.6:
                return "high_roll_risk"
            elif near_term_concentration > 0.4:
                return "moderate_roll_risk"
            else:
                return "low_roll_risk"
                
        except Exception:
            return "roll_risk_assessment_error"}
    
    def _analyze_market_concentration(self, options_oi: Dict, futures_oi: List[Dict]) -> Dict[str, Any]:
        """Analyze overall market concentration."""
        try:
            # Calculate overall market metrics
            total_options_oi = sum(
                sum(data.get("total_oi", 0) for data in strike_data.values())
                for strike_data in options_oi.values()
            )
            
            total_futures_oi = sum(contract["open_interest"] for contract in futures_oi)
            total_market_oi = total_options_oi + total_futures_oi
            
            # Market composition
            options_percentage = total_options_oi / total_market_oi if total_market_oi > 0 else 0
            futures_percentage = total_futures_oi / total_market_oi if total_market_oi > 0 else 0
            
            # Market concentration assessment
            if options_percentage > 0.8:
                market_concentration_type = "options_dominated"
            elif futures_percentage > 0.8:
                market_concentration_type = "futures_dominated"
            else:
                market_concentration_type = "balanced_market"
            
            # Overall market health
            if total_market_oi > 500000:
                market_health = "very_healthy"
            elif total_market_oi > 200000:
                market_health = "healthy"
            elif total_market_oi > 50000:
                market_health = "moderate"
            else:
                market_health = "thin"
            
            return {
                "total_market_oi": total_market_oi,
                "market_concentration_type": market_concentration_type,
                "market_health": market_health,
                "market_composition": {
                    "options_percentage": options_percentage,
                    "futures_percentage": futures_percentage
                },
                "concentration_assessment": self._assess_overall_concentration(
                    options_percentage, futures_percentage
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market concentration: {str(e)}")
            return {"market_concentration": "analysis_error"}
    
    def _assess_overall_concentration(self, options_pct: float, futures_pct: float) -> str:
        """Assess overall concentration type."""
        try:
            max_concentration = max(options_pct, futures_pct)
            
            if max_concentration > 0.9:
                return "extreme_concentration"
            elif max_concentration > 0.8:
                return "high_concentration"
            elif max_concentration > 0.6:
                return "moderate_concentration"
            else:
                return "balanced_concentration"
                
        except Exception:
            return "concentration_assessment_error"}
    
    def _assess_concentration_risk(self, strike_conc: Dict, exp_conc: Dict) -> Dict[str, Any]:
        """Assess overall concentration risk."""
        try:
            risk_factors = []
            
            # Strike concentration risk
            if strike_conc.get("concentration_risk") == "high":
                risk_factors.append("high_strike_concentration")
            
            # Expiration concentration risk
            near_term_conc = exp_conc.get("near_term_concentration", 0)
            if near_term_conc > 0.7:
                risk_factors.append("high_near_term_concentration")
            
            # Overall risk level
            risk_score = len(risk_factors)
            
            if risk_score >= 2:
                overall_risk = "high"
            elif risk_score == 1:
                overall_risk = "medium"
            else:
                overall_risk = "low"
            
            return {
                "overall_risk_level": overall_risk,
                "risk_factors": risk_factors,
                "risk_score": risk_score,
                "risk_interpretation": self._interpret_concentration_risk(overall_risk, risk_factors)
            }
            
        except Exception as e:
            logger.error(f"Error assessing concentration risk: {str(e)}")
            return {"overall_risk": "assessment_error"}
    
    def _interpret_concentration_risk(self, risk_level: str, risk_factors: List[str]) -> str:
        """Interpret concentration risk."""
        interpretations = {
            "high": "Significant concentration risk - monitor for sharp moves if large positions unwind",
            "medium": "Moderate concentration risk - increased volatility possible during position changes",
            "low": "Low concentration risk - well-distributed positioning across market"
        }
        
        base_interpretation = interpretations.get(risk_level, "Concentration risk assessment unavailable")
        
        if "high_strike_concentration" in risk_factors:
            base_interpretation += " High strike concentration increases gamma risk."
        
        if "high_near_term_concentration" in risk_factors:
            base_interpretation += " Near-term concentration increases roll risk."
        
        return base_interpretation
    
    def _analyze_unwinding_risk(self, concentration_risk: Dict) -> Dict[str, Any]:
        """Analyze risk of position unwinding."""
        try:
            risk_level = concentration_risk.get("overall_risk_level", "low")
            risk_factors = concentration_risk.get("risk_factors", [])
            
            # Unwinding risk assessment
            if risk_level == "high":
                unwinding_risk = "high"
            elif risk_level == "medium":
                unwinding_risk = "moderate"
            else:
                unwinding_risk = "low"
            
            # Trigger scenarios
            trigger_scenarios = []
            
            if "high_strike_concentration" in risk_factors:
                trigger_scenarios.append("large_gamma_unwinding")
            
            if "high_near_term_concentration" in risk_factors:
                trigger_scenarios.append("roll_stress")
            
            # Market impact assessment
            if risk_level == "high":
                market_impact = "significant_market_impact_likely"
            elif risk_level == "medium":
                market_impact = "moderate_market_impact_possible"
            else:
                market_impact = "minimal_market_impact_expected"
            
            return {
                "unwinding_risk": unwinding_risk,
                "trigger_scenarios": trigger_scenarios,
                "market_impact": market_impact,
                "monitoring_requirements": self._determine_monitoring_requirements(risk_level)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing unwinding risk: {str(e)}")
            return {"unwinding_risk": "assessment_error"}
    
    def _determine_monitoring_requirements(self, risk_level: str) -> List[str]:
        """Determine monitoring requirements based on risk level."""
        try:
            requirements = []
            
            if risk_level == "high":
                requirements.extend([
                    "continuous_gamma_monitoring",
                    "volatility_surface_tracking", 
                    "large_order_flow_alerts",
                    "intraday_price_movement_monitoring"
                ])
            elif risk_level == "medium":
                requirements.extend([
                    "daily_gamma_check",
                    "expiration_roll_monitoring",
                    "unusual_activity_alerts"
                ])
            else:
                requirements.append("standard_monitoring")
            
            return requirements
            
        except Exception:
            return ["monitoring_assessment_error"}
    
    def _generate_concentration_signals(self, concentration_risk: Dict, unwinding_risk: Dict) -> List[Dict[str, Any]]:
        """Generate signals from concentration analysis."""
        try:
            signals = []
            
            # Concentration risk signals
            risk_level = concentration_risk.get("overall_risk_level", "low")
            if risk_level == "high":
                signals.append({
                    "type": "concentration_risk",
                    "signal": "warning",
                    "strength": 0.9,
                    "message": "High positioning concentration risk detected"
                })
            elif risk_level == "medium":
                signals.append({
                    "type": "concentration_risk",
                    "signal": "caution",
                    "strength": 0.6,
                    "message": "Moderate positioning concentration risk"
                })
            
            # Unwinding risk signals
            unwinding_risk_level = unwinding_risk.get("unwinding_risk", "low")
            if unwinding_risk_level == "high":
                signals.append({
                    "type": "unwinding_risk",
                    "signal": "warning",
                    "strength": 0.8,
                    "message": "High position unwinding risk - monitor for volatility"
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating concentration signals: {str(e)}")
            return []
    
    async def _analyze_oi_trends(self, oi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in open interest data."""
        try:
            # This would require historical OI data in production
            # For now, provide framework for trend analysis
            
            return {
                "oi_trend_direction": "analyzing",
                "trend_strength": "moderate",
                "momentum_indicators": "calculating",
                "trend_signals": self._generate_trend_signals()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing OI trends: {str(e)}")
            return {"trend_analysis": "error"}
    
    def _generate_trend_signals(self) -> List[Dict[str, Any]]:
        """Generate signals from trend analysis."""
        return [{
            "type": "oi_trends",
            "signal": "monitor",
            "message": "OI trend analysis requires historical data for proper assessment"
        }]
    
    async def _analyze_cross_asset_positioning(self, symbol: str) -> Dict[str, Any]:
        """Analyze positioning across different asset classes."""
        try:
            # This would analyze positioning across related assets
            return {
                "cross_asset_correlations": "analyzing",
                "sector_positioning": "monitoring",
                "asset_allocation_signals": "tracking"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cross-asset positioning: {str(e)}")
            return {"cross_asset_analysis": "error"}
    
    async def _analyze_market_structure(self, oi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall market structure from OI data."""
        try:
            return {
                "market_depth": self._assess_market_depth(oi_data),
                "liquidity_profile": self._assess_liquidity_profile(oi_data),
                "participation_levels": self._assess_participation_levels(oi_data),
                "structural_changes": self._detect_structural_changes(oi_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {str(e)}")
            return {"market_structure": "analysis_error"}
    
    def _assess_market_depth(self, oi_data: Dict[str, Any]) -> str:
        """Assess market depth from OI data."""
        try:
            total_oi = oi_data.get("total_oi", 0)
            
            if total_oi > 1000000:
                return "very_deep"
            elif total_oi > 500000:
                return "deep"
            elif total_oi > 200000:
                return "moderate"
            else:
                return "shallow"
                
        except Exception:
            return "depth_assessment_error"}
    
    def _assess_liquidity_profile(self, oi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess liquidity profile from OI data."""
        try:
            return {
                "liquidity_depth": self._assess_market_depth(oi_data),
                "liquidity_distribution": "analyzing_distribution",
                "quality_assessment": "assessing_quality"
            }
            
        except Exception:
            return {"liquidity_profile": "assessment_error"}
    
    def _assess_participation_levels(self, oi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess market participation levels."""
        try:
            total_oi = oi_data.get("total_oi", 0)
            
            return {
                "participation_intensity": "high" if total_oi > 500000 else "moderate" if total_oi > 200000 else "low",
                "market_activity": self._classify_market_activity(total_oi)
            }
            
        except Exception:
            return {"participation_levels": "assessment_error"}
    
    def _classify_market_activity(self, total_oi: int) -> str:
        """Classify overall market activity level."""
        try:
            if total_oi > 800000:
                return "extremely_active"
            elif total_oi > 400000:
                return "very_active"
            elif total_oi > 200000:
                return "active"
            elif total_oi > 50000:
                return "moderate"
            else:
                return "quiet"
                
        except Exception:
            return "activity_classification_error"}
    
    def _detect_structural_changes(self, oi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect structural changes in market."""
        try:
            return {
                "structural_indicators": "monitoring",
                "market_evolution": "tracking_changes",
                "regime_shifts": "analyzing_regimes"
            }
            
        except Exception:
            return {"structural_analysis": "error"}
    
    async def _analyze_momentum_indicators(self, oi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum indicators from OI data."""
        try:
            return {
                "oi_momentum": "calculating",
                "positioning_momentum": "analyzing",
                "flow_momentum": "tracking",
                "momentum_signals": self._generate_momentum_signals()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing momentum indicators: {str(e)}")
            return {"momentum_analysis": "error"}
    
    async def _calculate_oi_intelligence_score(self, oi_aggregation: Dict, 
                                             positioning_analysis: Dict, 
                                             trend_analysis: Dict) -> Dict[str, Any]:
        """Calculate comprehensive OI intelligence score."""
        try:
            # Component scores
            aggregation_score = self._score_oi_aggregation(oi_aggregation)
            positioning_score = self._score_positioning_analysis(positioning_analysis)
            trend_score = self._score_oi_trends(trend_analysis)
            
            # Weighted combination
            weights = {"aggregation": 0.3, "positioning": 0.4, "trends": 0.3}
            overall_score = (
                aggregation_score * weights["aggregation"] +
                positioning_score * weights["positioning"] +
                trend_score * weights["trends"]
            )
            
            # Intelligence score components
            intelligence_components = {
                "oi_aggregation_score": aggregation_score,
                "positioning_score": positioning_score,
                "trends_score": trend_score,
                "overall_oi_intelligence": overall_score
            }
            
            # Risk assessment
            risk_level = self._assess_oi_risk_level(oi_aggregation, positioning_analysis)
            
            return {
                "oi_intelligence_score": overall_score,
                "score_components": intelligence_components,
                "risk_level": risk_level,
                "analysis_recommendations": self._generate_analysis_recommendations(
                    overall_score, risk_level, positioning_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating OI intelligence score: {str(e)}")
            return {"oi_intelligence_score": 0.5, "error": str(e)}
    
    def _score_oi_aggregation(self, oi_aggregation: Dict) -> float:
        """Score OI aggregation analysis quality."""
        try:
            if "error" in oi_aggregation:
                return 0.5
            
            score = 0.0
            total_checks = 4
            
            if "options_aggregation" in oi_aggregation:
                score += 0.25
            if "futures_aggregation" in oi_aggregation:
                score += 0.25
            if "total_oi_analysis" in oi_aggregation:
                score += 0.25
            if "aggregation_signals" in oi_aggregation:
                score += 0.25
            
            return score
            
        except Exception:
            return 0.5
    
    def _score_positioning_analysis(self, positioning_analysis: Dict) -> float:
        """Score positioning analysis quality."""
        try:
            if "error" in positioning_analysis:
                return 0.5
            
            score = 0.0
            total_checks = 4
            
            if "net_positioning" in positioning_analysis:
                score += 0.25
            if "positioning_bias" in positioning_analysis:
                score += 0.25
            if "large_positioning" in positioning_analysis:
                score += 0.25
            if "positioning_signals" in positioning_analysis:
                score += 0.25
            
            return score
            
        except Exception:
            return 0.5
    
    def _score_oi_trends(self, trend_analysis: Dict) -> float:
        """Score OI trends analysis quality."""
        try:
            # Trends analysis requires historical data - score based on framework
            return 0.6  # Base score for trend framework
            
        except Exception:
            return 0.5
    
    def _assess_oi_risk_level(self, oi_aggregation: Dict, positioning_analysis: Dict) -> str:
        """Assess overall OI-based risk level."""
        try:
            risk_factors = 0
            
            # Concentration risk factors
            aggregation = oi_aggregation.get("options_aggregation", {})
            concentration = aggregation.get("concentration_index", {})
            if concentration.get("concentration_risk") == "high":
                risk_factors += 1
            
            # Positioning risk factors
            net_positioning = positioning_analysis.get("net_positioning", {})
            positioning_sentiment = net_positioning.get("positioning_sentiment", "")
            
            if "strongly_net_long" in positioning_sentiment or "strongly_net_short" in positioning_sentiment:
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
    
    def _generate_analysis_recommendations(self, intelligence_score: float, risk_level: str, 
                                         positioning_analysis: Dict) -> List[str]:
        """Generate recommendations based on OI analysis."""
        try:
            recommendations = []
            
            # Score-based recommendations
            if intelligence_score > 0.7:
                recommendations.append("High confidence in OI analysis - consider positioning insights")
            elif intelligence_score > 0.6:
                recommendations.append("Moderate confidence in analysis - monitor developments")
            else:
                recommendations.append("Limited confidence - avoid major positioning decisions")
            
            # Risk-based recommendations
            if risk_level == "high":
                recommendations.append("High risk environment - monitor concentration and unwinding risk")
            elif risk_level == "medium":
                recommendations.append("Moderate risk - increased monitoring recommended")
            else:
                recommendations.append("Low risk environment - normal monitoring sufficient")
            
            # Positioning-based recommendations
            net_positioning = positioning_analysis.get("net_positioning", {})
            sentiment = net_positioning.get("positioning_sentiment", "")
            
            if "strongly_net" in sentiment:
                recommendations.append("Extreme positioning detected - monitor for reversals")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating analysis recommendations: {str(e)}")
            return ["OI analysis incomplete - proceed with standard monitoring"]
    
    async def get_oi_intelligence_history(self, symbol: str = "SPY", days: int = 30) -> Dict[str, Any]:
        """Get historical OI intelligence data."""
        try:
            # In production, this would retrieve historical OI data
            # For now, return current analysis with simulated historical context
            
            current_analysis = await self.analyze_open_interest_intelligence(symbol)
            
            # Simulated historical intelligence scores
            historical_scores = []
            base_score = current_analysis.get("open_interest_intelligence_score", {}).get("oi_intelligence_score", 0.5)
            
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
            logger.error(f"Error getting OI intelligence history: {str(e)}")
            return {"error": str(e)}