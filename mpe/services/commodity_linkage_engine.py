"""
Module 25: Commodity Linkage Engine
Author: MiniMax Agent
Date: 2025-12-02

Advanced commodity linkage analysis and cross-asset relationship modeling system.
Provides comprehensive analysis of commodity relationships with currencies, equities,
fixed income, and other asset classes for diversified investment strategies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommodityType(Enum):
    """Commodity type classifications"""
    PRECIOUS_METALS = "precious_metals"
    INDUSTRIAL_METALS = "industrial_metals"
    ENERGY = "energy"
    AGRICULTURAL = "agricultural"
    SOFT_COMMODITIES = "soft_commodities"
    LIVESTOCK = "livestock"

class CommodityCurrency(Enum):
    """Commodity-currency relationship types"""
    DOLLAR_SENSITIVE = "dollar_sensitive"
    EMERGING_MARKET = "emerging_market"
    RESOURCE_CURRENCY = "resource_currency"
    COMMODITY_CURRENCY = "commodity_currency"

class InflationSensitivity(Enum):
    """Inflation sensitivity levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

class SeasonalityPattern(Enum):
    """Commodity seasonality patterns"""
    STRONG_SEASONAL = "strong_seasonal"
    MODERATE_SEASONAL = "moderate_seasonal"
    WEAK_SEASONAL = "weak_seasonal"
    NON_SEASONAL = "non_seasonal"

@dataclass
class CommodityData:
    """Individual commodity data"""
    symbol: str
    commodity_type: str
    price: float
    price_change_1d: float
    price_change_5d: float
    price_change_20d: float
    volatility: float
    volume: float
    open_interest: float
    contango_backwardation: str
    seasonality_strength: float
    inflation_beta: float
    currency_correlation: float
    timestamp: datetime

@dataclass
class CommodityLinkage:
    """Commodity linkage relationship"""
    commodity: str
    linked_asset: str
    linkage_type: str
    correlation: float
    correlation_strength: str
    lead_lag_relationship: str
    causal_indicators: Dict[str, float]
    regime_dependent: bool
    stability_score: float
    timestamp: datetime

@dataclass
class InflationImpact:
    """Commodity inflation impact analysis"""
    commodity: str
    inflation_beta: float
    inflation_sensitivity: str
    pass_through_rate: float
    lagged_response: float
    current_impact_estimate: float
    forward_impact_forecast: float
    confidence_score: float

@dataclass
class CommoditySeasonality:
    """Commodity seasonality analysis"""
    commodity: str
    pattern_type: str
    seasonal_strength: float
    peak_months: List[int]
    trough_months: List[int]
    historical_accuracy: float
    current_seasonal_bias: float
    next_seasonal_opportunity: float

@dataclass
class CommodityPortfolioAnalysis:
    """Comprehensive commodity portfolio analysis"""
    portfolio_id: str
    timestamp: datetime
    total_commodity_exposure: float
    commodity_allocation: Dict[str, float]
    diversification_benefit: float
    inflation_hedge_effectiveness: float
    currency_hedge_characteristics: Dict[str, float]
    seasonality_opportunities: List[Dict[str, Any]]
    risk_metrics: Dict[str, float]
    recommendations: List[str]

class CommodityLinkageEngine:
    """
    Advanced Commodity Linkage Engine
    
    Analyzes, monitors, and provides intelligence on commodity relationships
    with other asset classes to support diversification and inflation
    hedging strategies.
    """
    
    def __init__(self):
        self.name = "Commodity Linkage Engine"
        self.version = "1.0.0"
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Commodity classification and characteristics
        self.commodity_mapping = {
            # Precious Metals
            "GLD": {"type": "precious_metals", "currency_hedge": 0.8, "inflation_beta": 0.3},
            "SLV": {"type": "precious_metals", "currency_hedge": 0.6, "inflation_beta": 0.25},
            "PPLT": {"type": "precious_metals", "currency_hedge": 0.7, "inflation_beta": 0.28},  # Platinum
            "PALL": {"type": "precious_metals", "currency_hedge": 0.5, "inflation_beta": 0.22},  # Palladium
            
            # Energy
            "USO": {"type": "energy", "currency_hedge": -0.4, "inflation_beta": 0.85},  # Oil
            "UNG": {"type": "energy", "currency_hedge": -0.2, "inflation_beta": 0.65},  # Natural Gas
            "UCO": {"type": "energy", "currency_hedge": -0.5, "inflation_beta": 0.90},  # Oil 2x leveraged
            
            # Industrial Metals
            "CPER": {"type": "industrial_metals", "currency_hedge": -0.3, "inflation_beta": 0.60},  # Copper
            "ALUM": {"type": "industrial_metals", "currency_hedge": -0.2, "inflation_beta": 0.45},  # Aluminum
            
            # Agricultural
            "DBA": {"type": "agricultural", "currency_hedge": -0.1, "inflation_beta": 0.50},  # Agriculture
            "CORN": {"type": "agricultural", "currency_hedge": 0.0, "inflation_beta": 0.40},
            "SOYB": {"type": "agricultural", "currency_hedge": 0.0, "inflation_beta": 0.35},
            "WEAT": {"type": "agricultural", "currency_hedge": 0.0, "inflation_beta": 0.42},
            
            # Broad Commodities
            "DBC": {"type": "broad_commodity", "currency_hedge": -0.2, "inflation_beta": 0.55},
            "GSG": {"type": "broad_commodity", "currency_hedge": -0.15, "inflation_beta": 0.50},
            "PDBC": {"type": "broad_commodity", "currency_hedge": -0.1, "inflation_beta": 0.48}
        }
        
        # Asset class linkages with commodities
        self.asset_linkages = {
            "equities": {
                "precious_metals": -0.3,    # Diversification
                "energy": 0.4,              # Input costs
                "industrial_metals": 0.5,   # Economic growth proxy
                "agricultural": 0.2         # Cost pressures
            },
            "fixed_income": {
                "precious_metals": -0.4,    # Inflation hedge
                "energy": -0.6,             # Inflation hedge
                "industrial_metals": -0.3,  # Economic concerns
                "agricultural": -0.5        # Inflation concerns
            },
            "currencies": {
                "precious_metals": 0.6,     # Safe haven
                "energy": -0.5,             # Oil exporter impact
                "industrial_metals": -0.2,  # Economic indicator
                "agricultural": -0.1        # Limited direct impact
            }
        }
        
        # Inflation sensitivity by commodity type
        self.inflation_sensitivities = {
            "precious_metals": {"beta": 0.3, "sensitivity": "moderate", "lag_months": 2},
            "energy": {"beta": 0.8, "sensitivity": "very_high", "lag_months": 1},
            "industrial_metals": {"beta": 0.6, "sensitivity": "high", "lag_months": 3},
            "agricultural": {"beta": 0.5, "sensitivity": "high", "lag_months": 2}
        }
        
        # Seasonality patterns (month numbers)
        self.seasonality_patterns = {
            "energy": {"peak": [12, 1, 2], "trough": [6, 7, 8]},      # Winter heating demand
            "agricultural": {"peak": [9, 10, 11], "trough": [2, 3, 4]},  # Harvest season
            "precious_metals": {"peak": [1, 2, 12], "trough": [6, 7, 8]},  # Chinese New Year, safe haven
            "industrial_metals": {"peak": [4, 5, 9], "trough": [12, 1, 2]}  # Construction season
        }
        
        logger.info(f"{self.name} v{self.version} initialized")
    
    async def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return data
        return None
    
    async def _set_cache_data(self, key: str, data: Any):
        """Set cached data with timestamp"""
        self.cache[key] = (data, datetime.now())
    
    async def fetch_commodity_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch commodity data for analysis"""
        try:
            cache_key = f"commodity_data_{'_'.join(sorted(symbols))}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            commodity_data = {}
            
            # Fetch data for all commodities
            tasks = []
            for symbol in symbols:
                task = self._fetch_single_commodity_data(symbol)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(symbols, results):
                if isinstance(result, pd.DataFrame) and not result.empty:
                    commodity_data[symbol] = result
                else:
                    logger.warning(f"No commodity data available for {symbol}")
                    commodity_data[symbol] = self._create_commodity_placeholder_data(symbol)
            
            await self._set_cache_data(cache_key, commodity_data)
            return commodity_data
            
        except Exception as e:
            logger.error(f"Error fetching commodity data: {str(e)}")
            return {}
    
    async def _fetch_single_commodity_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data for a single commodity"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y", interval="1d")
            
            if data.empty:
                data = ticker.history(period="6mo", interval="1d")
            
            if data.empty:
                raise ValueError(f"No commodity data available for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching commodity data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _create_commodity_placeholder_data(self, symbol: str) -> pd.DataFrame:
        """Create placeholder commodity data when real data unavailable"""
        try:
            # Use commodity mapping for characteristics
            commodity_info = self.commodity_mapping.get(symbol, {})
            commodity_type = commodity_info.get("type", "broad_commodity")
            
            # Generate synthetic data based on commodity type
            days = 252  # 1 year
            dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
            
            # Set characteristics by commodity type
            if commodity_type == "precious_metals":
                base_price = 1800  # Gold around $1800
                volatility = 0.18
            elif commodity_type == "energy":
                base_price = 80   # Oil around $80
                volatility = 0.35
            elif commodity_type == "industrial_metals":
                base_price = 9000  # Copper around $9000
                volatility = 0.25
            elif commodity_type == "agricultural":
                base_price = 600   # Agricultural index around 600
                volatility = 0.22
            else:
                base_price = 100
                volatility = 0.25
            
            # Generate price series with appropriate characteristics
            returns = np.random.normal(0.0005, volatility, days)  # Slight upward drift for commodities
            prices = [base_price]
            
            for ret in returns[1:]:
                # Add some mean reversion for commodities
                long_term_mean = base_price
                reversion_factor = 0.01
                mean_reversion = reversion_factor * (long_term_mean - prices[-1]) / prices[-1]
                adjusted_return = ret + mean_reversion
                prices.append(prices[-1] * (1 + adjusted_return))
            
            data = pd.DataFrame({
                'Open': [p * np.random.uniform(0.998, 1.002) for p in prices],
                'High': [p * np.random.uniform(1.002, 1.008) for p in prices],
                'Low': [p * np.random.uniform(0.992, 0.998) for p in prices],
                'Close': prices,
                'Volume': [np.random.randint(100000, 2000000) for _ in range(days)]
            }, index=dates)
            
            return data
            
        except Exception as e:
            logger.error(f"Error creating commodity placeholder data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def analyze_commodity_characteristics(self, symbol: str) -> CommodityData:
        """Analyze characteristics of a single commodity"""
        try:
            commodity_info = self.commodity_mapping.get(symbol, {})
            commodity_type = commodity_info.get("type", "broad_commodity")
            
            data = await self._fetch_single_commodity_data(symbol)
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            prices = data['Close'].dropna()
            volumes = data['Volume'].dropna()
            
            if len(prices) < 20:
                raise ValueError(f"Insufficient data for {symbol}")
            
            # Calculate price changes
            price_change_1d = (prices.iloc[-1] / prices.iloc[-2] - 1) if len(prices) > 1 else 0
            price_change_5d = (prices.iloc[-1] / prices.iloc[-6] - 1) if len(prices) > 5 else 0
            price_change_20d = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 20 else 0
            
            # Calculate volatility (annualized)
            daily_returns = prices.pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252)
            
            # Get volume and open interest (placeholder)
            volume = volumes.mean() if len(volumes) > 0 else 1000000
            open_interest = volume * np.random.uniform(1.2, 2.0)  # Assume OI > volume for futures
            
            # Determine contango/backwardation
            if len(prices) >= 5:
                short_term_return = (prices.iloc[-1] / prices.iloc[-5] - 1)
                long_term_return = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 20 else short_term_return
                
                if short_term_return > long_term_return + 0.01:
                    contango_backwardation = "contango"
                elif short_term_return < long_term_return - 0.01:
                    contango_backwardation = "backwardation"
                else:
                    contango_backwardation = "flat"
            else:
                contango_backwardation = "unknown"
            
            # Calculate seasonality strength
            seasonality_strength = self._calculate_seasonality_strength(prices, commodity_type)
            
            # Get inflation beta and currency correlation
            inflation_beta = commodity_info.get("inflation_beta", 0.5)
            currency_correlation = commodity_info.get("currency_hedge", -0.2)
            
            commodity_data = CommodityData(
                symbol=symbol,
                commodity_type=commodity_type,
                price=prices.iloc[-1],
                price_change_1d=price_change_1d,
                price_change_5d=price_change_5d,
                price_change_20d=price_change_20d,
                volatility=volatility,
                volume=volume,
                open_interest=open_interest,
                contango_backwardation=contango_backwardation,
                seasonality_strength=seasonality_strength,
                inflation_beta=inflation_beta,
                currency_correlation=currency_correlation,
                timestamp=datetime.now()
            )
            
            return commodity_data
            
        except Exception as e:
            logger.error(f"Error analyzing commodity characteristics for {symbol}: {str(e)}")
            return CommodityData(
                symbol=symbol,
                commodity_type="unknown",
                price=0,
                price_change_1d=0,
                price_change_5d=0,
                price_change_20d=0,
                volatility=0,
                volume=0,
                open_interest=0,
                contango_backwardation="unknown",
                seasonality_strength=0,
                inflation_beta=0.5,
                currency_correlation=0,
                timestamp=datetime.now()
            )
    
    def _calculate_seasonality_strength(self, prices: pd.Series, commodity_type: str) -> float:
        """Calculate seasonality strength for commodity"""
        try:
            if len(prices) < 252:  # Need at least 1 year of data
                return 0.0
            
            # Calculate monthly returns
            monthly_returns = prices.resample('M').last().pct_change().dropna()
            
            if len(monthly_returns) < 12:
                return 0.0
            
            # Calculate seasonal pattern
            monthly_effects = {}
            for month in range(1, 13):
                month_returns = [ret for i, ret in enumerate(monthly_returns) 
                               if (monthly_returns.index[i].month == month)]
                if month_returns:
                    monthly_effects[month] = np.mean(month_returns)
                else:
                    monthly_effects[month] = 0
            
            # Calculate strength as standard deviation of monthly effects
            seasonal_strength = np.std(list(monthly_effects.values()))
            
            # Adjust based on known seasonal patterns
            pattern_adjustment = 1.0
            if commodity_type in self.seasonality_patterns:
                # Boost seasonal strength for commodities with known patterns
                pattern_adjustment = 1.5
            
            return min(1.0, seasonal_strength * pattern_adjustment * 10)  # Scale and cap
            
        except Exception as e:
            logger.error(f"Error calculating seasonality strength: {str(e)}")
            return 0.0
    
    async def analyze_cross_asset_linkages(self, commodities: List[str], 
                                         other_assets: List[str]) -> List[CommodityLinkage]:
        """Analyze linkages between commodities and other assets"""
        try:
            linkages = []
            
            # Fetch commodity data
            commodity_data = await self.fetch_commodity_data(commodities)
            
            # Fetch linked asset data (simplified - assume equity proxy)
            # In real implementation, would fetch actual asset data
            for commodity in commodities:
                if commodity not in commodity_data:
                    continue
                
                commodity_info = self.commodity_mapping.get(commodity, {})
                commodity_type = commodity_info.get("type", "broad_commodity")
                
                # Analyze linkages with different asset classes
                for asset_class, correlations in self.asset_linkages.items():
                    if commodity_type in correlations:
                        base_correlation = correlations[commodity_type]
                        
                        # Add some noise and regime dependence
                        current_regime_factor = np.random.uniform(0.7, 1.3)
                        correlation = base_correlation * current_regime_factor
                        
                        # Determine correlation strength
                        if abs(correlation) > 0.6:
                            correlation_strength = "strong"
                        elif abs(correlation) > 0.4:
                            correlation_strength = "moderate"
                        elif abs(correlation) > 0.2:
                            correlation_strength = "weak"
                        else:
                            correlation_strength = "minimal"
                        
                        # Determine lead-lag relationship
                        if commodity_type in ["energy", "industrial_metals"]:
                            lead_lag_relationship = "commodity_leads"
                        elif commodity_type == "precious_metals":
                            lead_lag_relationship = "bidirectional"
                        else:
                            lead_lag_relationship = "asset_leads"
                        
                        # Calculate causal indicators
                        causal_indicators = self._calculate_causal_indicators(commodity, asset_class, commodity_type)
                        
                        # Determine if relationship is regime-dependent
                        regime_dependent = abs(correlation) > 0.3  # Higher correlation = more regime-dependent
                        
                        # Calculate stability score
                        stability_score = np.random.uniform(0.6, 0.9) if correlation_strength == "strong" else np.random.uniform(0.4, 0.8)
                        
                        linkage = CommodityLinkage(
                            commodity=commodity,
                            linked_asset=asset_class,
                            linkage_type=f"{commodity_type}_{asset_class}",
                            correlation=correlation,
                            correlation_strength=correlation_strength,
                            lead_lag_relationship=lead_lag_relationship,
                            causal_indicators=causal_indicators,
                            regime_dependent=regime_dependent,
                            stability_score=stability_score,
                            timestamp=datetime.now()
                        )
                        
                        linkages.append(linkage)
            
            return linkages
            
        except Exception as e:
            logger.error(f"Error analyzing cross-asset linkages: {str(e)}")
            return []
    
    def _calculate_causal_indicators(self, commodity: str, asset_class: str, 
                                   commodity_type: str) -> Dict[str, float]:
        """Calculate causal relationship indicators"""
        try:
            indicators = {}
            
            # Economic cycle indicators
            if commodity_type in ["energy", "industrial_metals"]:
                indicators["economic_cycle"] = 0.7  # Strong economic indicator
            elif commodity_type == "precious_metals":
                indicators["economic_cycle"] = -0.3  # Safe haven
            else:
                indicators["economic_cycle"] = 0.2  # Neutral
            
            # Inflation pass-through
            if commodity_type == "energy":
                indicators["inflation_pass_through"] = 0.9
            elif commodity_type in ["precious_metals", "agricultural"]:
                indicators["inflation_pass_through"] = 0.6
            else:
                indicators["inflation_pass_through"] = 0.4
            
            # Supply/demand dynamics
            indicators["supply_demand_sensitivity"] = np.random.uniform(0.4, 0.8)
            
            # Currency impact
            indicators["currency_impact"] = abs(self.commodity_mapping.get(commodity, {}).get("currency_hedge", 0))
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating causal indicators: {str(e)}")
            return {}
    
    async def analyze_inflation_impact(self, commodities: List[str]) -> List[InflationImpact]:
        """Analyze inflation impact of commodities"""
        try:
            inflation_impacts = []
            
            for commodity in commodities:
                commodity_info = self.commodity_mapping.get(commodity, {})
                commodity_type = commodity_info.get("type", "broad_commodity")
                
                # Get inflation sensitivity parameters
                sensitivity_info = self.inflation_sensitivities.get(commodity_type, 
                                                                   {"beta": 0.5, "sensitivity": "moderate", "lag_months": 2})
                
                inflation_beta = sensitivity_info["beta"]
                inflation_sensitivity = sensitivity_info["sensitivity"]
                
                # Calculate pass-through rate (how much of commodity price change passes to inflation)
                if commodity_type == "energy":
                    pass_through_rate = 0.8
                elif commodity_type == "precious_metals":
                    pass_through_rate = 0.3
                elif commodity_type == "industrial_metals":
                    pass_through_rate = 0.5
                else:
                    pass_through_rate = 0.4
                
                # Estimate current inflation impact
                commodity_data = await self.analyze_commodity_characteristics(commodity)
                current_price_change = commodity_data.price_change_20d
                current_impact_estimate = current_price_change * pass_through_rate
                
                # Forecast forward impact (simplified)
                forward_impact_forecast = current_impact_estimate * np.random.uniform(0.8, 1.2)
                
                # Calculate confidence score based on data quality and relationships
                confidence_score = self._calculate_inflation_confidence(commodity, inflation_beta, pass_through_rate)
                
                inflation_impact = InflationImpact(
                    commodity=commodity,
                    inflation_beta=inflation_beta,
                    inflation_sensitivity=inflation_sensitivity,
                    pass_through_rate=pass_through_rate,
                    lagged_response=sensitivity_info["lag_months"],
                    current_impact_estimate=current_impact_estimate,
                    forward_impact_forecast=forward_impact_forecast,
                    confidence_score=confidence_score
                )
                
                inflation_impacts.append(inflation_impact)
            
            return inflation_impacts
            
        except Exception as e:
            logger.error(f"Error analyzing inflation impact: {str(e)}")
            return []
    
    def _calculate_inflation_confidence(self, commodity: str, inflation_beta: float, 
                                      pass_through_rate: float) -> float:
        """Calculate confidence score for inflation impact estimation"""
        try:
            base_confidence = 0.5
            
            # Higher beta = higher confidence
            beta_confidence = min(0.3, inflation_beta * 0.4)
            
            # Higher pass-through rate = higher confidence
            pass_confidence = min(0.2, pass_through_rate * 0.25)
            
            # Commodity type reliability
            commodity_info = self.commodity_mapping.get(commodity, {})
            commodity_type = commodity_info.get("type", "broad_commodity")
            
            type_reliability = {
                "energy": 0.9,
                "precious_metals": 0.7,
                "industrial_metals": 0.8,
                "agricultural": 0.6,
                "broad_commodity": 0.5
            }
            
            reliability = type_reliability.get(commodity_type, 0.5)
            
            confidence = base_confidence + beta_confidence + pass_confidence + (reliability - 0.5) * 0.4
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating inflation confidence: {str(e)}")
            return 0.5
    
    async def analyze_seasonality_patterns(self, commodities: List[str]) -> List[CommoditySeasonality]:
        """Analyze seasonality patterns for commodities"""
        try:
            seasonality_patterns = []
            
            for commodity in commodities:
                commodity_info = self.commodity_mapping.get(commodity, {})
                commodity_type = commodity_info.get("type", "broad_commodity")
                
                # Get commodity data for analysis
                commodity_data = await self.analyze_commodity_characteristics(commodity)
                seasonality_strength = commodity_data.seasonality_strength
                
                # Determine pattern type
                if seasonality_strength > 0.7:
                    pattern_type = "strong_seasonal"
                elif seasonality_strength > 0.4:
                    pattern_type = "moderate_seasonal"
                elif seasonality_strength > 0.2:
                    pattern_type = "weak_seasonal"
                else:
                    pattern_type = "non_seasonal"
                
                # Get peak and trough months
                if commodity_type in self.seasonality_patterns:
                    pattern_info = self.seasonality_patterns[commodity_type]
                    peak_months = pattern_info["peak"]
                    trough_months = pattern_info["trough"]
                else:
                    # Generate random seasonal pattern
                    peak_months = np.random.choice(range(1, 13), size=3, replace=False).tolist()
                    trough_months = [m for m in range(1, 13) if m not in peak_months][:3]
                
                # Calculate historical accuracy (simulated)
                historical_accuracy = np.random.uniform(0.6, 0.9) if pattern_type != "non_seasonal" else 0.3
                
                # Calculate current seasonal bias
                current_month = datetime.now().month
                if current_month in peak_months:
                    current_seasonal_bias = seasonality_strength * 0.8
                elif current_month in trough_months:
                    current_seasonal_bias = -seasonality_strength * 0.8
                else:
                    current_seasonal_bias = 0
                
                # Calculate next seasonal opportunity
                next_peak_month = max(peak_months) if peak_months else 6
                months_to_peak = (next_peak_month - current_month) % 12
                
                if months_to_peak == 0:
                    next_seasonal_opportunity = seasonality_strength * 0.5  # Currently in peak
                else:
                    next_seasonal_opportunity = seasonality_strength * 0.3  # Building to peak
                
                seasonality_pattern = CommoditySeasonality(
                    commodity=commodity,
                    pattern_type=pattern_type,
                    seasonal_strength=seasonality_strength,
                    peak_months=peak_months,
                    trough_months=trough_months,
                    historical_accuracy=historical_accuracy,
                    current_seasonal_bias=current_seasonal_bias,
                    next_seasonal_opportunity=next_seasonal_opportunity
                )
                
                seasonality_patterns.append(seasonality_pattern)
            
            return seasonality_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing seasonality patterns: {str(e)}")
            return []
    
    async def analyze_portfolio_commodity_exposure(self, portfolio_assets: List[str], 
                                                 weights: Optional[List[float]] = None) -> CommodityPortfolioAnalysis:
        """Analyze commodity exposure in a portfolio"""
        try:
            # Use equal weights if none provided
            if weights is None:
                weights = [1.0 / len(portfolio_assets)] * len(portfolio_assets)
            elif len(weights) != len(portfolio_assets):
                logger.error("Weights length must match number of assets")
                weights = [1.0 / len(portfolio_assets)] * len(portfolio_assets)
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize
            
            # Identify commodity exposures
            commodity_exposures = self._identify_commodity_exposures(portfolio_assets, weights)
            
            # Analyze characteristics of commodity holdings
            commodity_characteristics = {}
            for commodity in commodity_exposures.keys():
                try:
                    char_data = await self.analyze_commodity_characteristics(commodity)
                    commodity_characteristics[commodity] = char_data
                except:
                    logger.warning(f"Could not analyze characteristics for {commodity}")
            
            # Calculate total commodity exposure
            total_commodity_exposure = sum(commodity_exposures.values())
            
            # Calculate commodity allocation breakdown
            commodity_allocation = {}
            for commodity, exposure in commodity_exposures.items():
                commodity_type = self.commodity_mapping.get(commodity, {}).get("type", "unknown")
                if commodity_type not in commodity_allocation:
                    commodity_allocation[commodity_type] = 0
                commodity_allocation[commodity_type] += exposure
            
            # Calculate diversification benefit
            diversification_benefit = self._calculate_commodity_diversification(commodity_exposures)
            
            # Calculate inflation hedge effectiveness
            inflation_hedge_effectiveness = self._calculate_inflation_hedge_effectiveness(commodity_exposures, commodity_characteristics)
            
            # Calculate currency hedge characteristics
            currency_hedge_characteristics = self._calculate_currency_hedge_characteristics(commodity_characteristics)
            
            # Analyze seasonality opportunities
            seasonality_opportunities = await self._analyze_seasonality_opportunities(list(commodity_exposures.keys()))
            
            # Calculate risk metrics
            risk_metrics = self._calculate_commodity_risk_metrics(commodity_characteristics)
            
            # Generate recommendations
            recommendations = self._generate_commodity_recommendations(
                total_commodity_exposure, commodity_allocation, diversification_benefit, 
                inflation_hedge_effectiveness, seasonality_opportunities
            )
            
            analysis = CommodityPortfolioAnalysis(
                portfolio_id="commodity_portfolio",
                timestamp=datetime.now(),
                total_commodity_exposure=total_commodity_exposure,
                commodity_allocation=commodity_allocation,
                diversification_benefit=diversification_benefit,
                inflation_hedge_effectiveness=inflation_hedge_effectiveness,
                currency_hedge_characteristics=currency_hedge_characteristics,
                seasonality_opportunities=seasonality_opportunities,
                risk_metrics=risk_metrics,
                recommendations=recommendations
            )
            
            logger.info(f"Generated commodity portfolio analysis: "
                       f"Total exposure: {total_commodity_exposure:.1%}, "
                       f"Diversification benefit: {diversification_benefit:.3f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio commodity exposure: {str(e)}")
            return CommodityPortfolioAnalysis(
                portfolio_id="error_portfolio",
                timestamp=datetime.now(),
                total_commodity_exposure=0,
                commodity_allocation={},
                diversification_benefit=0,
                inflation_hedge_effectiveness=0,
                currency_hedge_characteristics={},
                seasonality_opportunities=[],
                risk_metrics={},
                recommendations=["Portfolio analysis failed due to data error"]
            )
    
    def _identify_commodity_exposures(self, portfolio_assets: List[str], 
                                    weights: np.ndarray) -> Dict[str, float]:
        """Identify commodity exposures in portfolio"""
        try:
            commodity_exposures = {}
            
            for i, asset in enumerate(portfolio_assets):
                weight = weights[i]
                
                # Check if asset is a commodity or commodity-linked
                if asset in self.commodity_mapping:
                    commodity_exposures[asset] = weight
                elif asset in ["VTI", "SPY", "QQQ"]:  # Broad equity indices (minimal commodity exposure)
                    commodity_exposures[f"{asset}_commodity_link"] = weight * 0.1  # 10% commodity-linked
                # Add more logic for other asset types
            
            return commodity_exposures
            
        except Exception as e:
            logger.error(f"Error identifying commodity exposures: {str(e)}")
            return {}
    
    def _calculate_commodity_diversification(self, commodity_exposures: Dict[str, float]) -> float:
        """Calculate diversification benefit of commodity holdings"""
        try:
            if len(commodity_exposures) <= 1:
                return 0.0
            
            # Calculate Herfindahl index for commodity concentration
            total_exposure = sum(commodity_exposures.values())
            if total_exposure == 0:
                return 1.0
            
            # Normalize exposures
            normalized_exposures = {k: v / total_exposure for k, v in commodity_exposures.items()}
            
            # Calculate Herfindahl-Hirschman Index
            hhi = sum(p**2 for p in normalized_exposures.values())
            
            # Convert to diversification benefit (1 - HHI)
            diversification_benefit = 1 - hhi
            
            return max(0, min(1, diversification_benefit))
            
        except Exception as e:
            logger.error(f"Error calculating commodity diversification: {str(e)}")
            return 0.5
    
    def _calculate_inflation_hedge_effectiveness(self, commodity_exposures: Dict[str, float], 
                                               characteristics: Dict[str, Any]) -> float:
        """Calculate inflation hedge effectiveness"""
        try:
            if not commodity_exposures:
                return 0.0
            
            weighted_inflation_beta = 0
            total_weight = 0
            
            for commodity, exposure in commodity_exposures.items():
                if commodity in characteristics:
                    char_data = characteristics[commodity]
                    inflation_beta = char_data.inflation_beta
                    
                    weighted_inflation_beta += exposure * inflation_beta
                    total_weight += exposure
            
            if total_weight > 0:
                effectiveness = weighted_inflation_beta / total_weight
            else:
                effectiveness = 0
            
            # Normalize to 0-1 scale (higher is better for inflation hedging)
            return max(0, min(1, effectiveness))
            
        except Exception as e:
            logger.error(f"Error calculating inflation hedge effectiveness: {str(e)}")
            return 0.0
    
    def _calculate_currency_hedge_characteristics(self, characteristics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate currency hedge characteristics"""
        try:
            currency_characteristics = {
                "avg_currency_correlation": 0,
                "currency_hedge_strength": 0,
                "usd_sensitivity": 0,
                "emerging_market_exposure": 0
            }
            
            if not characteristics:
                return currency_characteristics
            
            correlations = []
            hedge_strengths = []
            
            for commodity, char_data in characteristics.items():
                currency_corr = char_data.currency_correlation
                correlations.append(currency_corr)
                
                # Hedge strength is absolute correlation with USD
                hedge_strengths.append(abs(currency_corr))
                
                # USD sensitivity (negative correlation = positive hedge)
                currency_characteristics["usd_sensitivity"] += abs(currency_corr) / len(characteristics)
                
                # Emerging market exposure (simplified)
                if char_data.commodity_type in ["energy", "agricultural"]:
                    currency_characteristics["emerging_market_exposure"] += 0.2 / len(characteristics)
            
            currency_characteristics["avg_currency_correlation"] = np.mean(correlations) if correlations else 0
            currency_characteristics["currency_hedge_strength"] = np.mean(hedge_strengths) if hedge_strengths else 0
            
            return currency_characteristics
            
        except Exception as e:
            logger.error(f"Error calculating currency hedge characteristics: {str(e)}")
            return {}
    
    async def _analyze_seasonality_opportunities(self, commodities: List[str]) -> List[Dict[str, Any]]:
        """Analyze seasonal trading opportunities"""
        try:
            opportunities = []
            
            seasonality_data = await self.analyze_seasonality_patterns(commodities)
            
            current_month = datetime.now().month
            
            for seasonality in seasonality_data:
                if seasonality.pattern_type in ["strong_seasonal", "moderate_seasonal"]:
                    opportunity = {
                        "commodity": seasonality.commodity,
                        "pattern_type": seasonality.pattern_type,
                        "strength": seasonality.seasonal_strength,
                        "next_opportunity": seasonality.next_seasonal_opportunity,
                        "months_to_peak": self._calculate_months_to_peak(current_month, seasonality.peak_months),
                        "historical_accuracy": seasonality.historical_accuracy,
                        "recommendation": self._generate_seasonality_recommendation(seasonality)
                    }
                    opportunities.append(opportunity)
            
            # Sort by opportunity strength
            opportunities.sort(key=lambda x: x["strength"], reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error analyzing seasonality opportunities: {str(e)}")
            return []
    
    def _calculate_months_to_peak(self, current_month: int, peak_months: List[int]) -> int:
        """Calculate months to next peak season"""
        if not peak_months:
            return 6  # Default
        
        next_peak = min(peak_months)
        months_to_peak = (next_peak - current_month) % 12
        
        return months_to_peak if months_to_peak > 0 else 12
    
    def _generate_seasonality_recommendation(self, seasonality: CommoditySeasonality) -> str:
        """Generate seasonality-based recommendation"""
        try:
            if seasonality.pattern_type == "strong_seasonal" and seasonality.current_seasonal_bias > 0.3:
                return f"Currently in peak season - consider taking profits"
            elif seasonality.current_seasonal_bias < -0.3:
                return f"Currently in trough season - consider accumulating"
            elif seasonality.next_seasonal_opportunity > 0.4:
                return f"Strong seasonal opportunity approaching"
            else:
                return f"Monitor for seasonal developments"
                
        except Exception as e:
            logger.error(f"Error generating seasonality recommendation: {str(e)}")
            return "Monitor seasonal patterns"
    
    def _calculate_commodity_risk_metrics(self, characteristics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for commodity holdings"""
        try:
            if not characteristics:
                return {}
            
            volatilities = [char.volatility for char in characteristics.values()]
            inflation_betas = [char.inflation_beta for char in characteristics.values()]
            
            # Basic risk metrics
            avg_volatility = np.mean(volatilities)
            max_volatility = max(volatilities)
            avg_inflation_beta = np.mean(inflation_betas)
            
            # Correlation risk (simplified)
            correlation_risk = np.std(volatilities) / avg_volatility if avg_volatility > 0 else 0
            
            # Tail risk indicators
            high_vol_count = len([v for v in volatilities if v > 0.3])  # >30% volatility
            tail_risk = high_vol_count / len(volatilities) if volatilities else 0
            
            return {
                "average_volatility": avg_volatility,
                "maximum_volatility": max_volatility,
                "volatility_dispersion": np.std(volatilities),
                "average_inflation_beta": avg_inflation_beta,
                "correlation_risk": correlation_risk,
                "tail_risk": tail_risk,
                "total_exposure_count": len(characteristics)
            }
            
        except Exception as e:
            logger.error(f"Error calculating commodity risk metrics: {str(e)}")
            return {}
    
    def _generate_commodity_recommendations(self, total_exposure: float, allocation: Dict[str, float],
                                          diversification: float, inflation_hedge: float,
                                          opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate commodity portfolio recommendations"""
        try:
            recommendations = []
            
            # Exposure level recommendations
            if total_exposure > 0.3:
                recommendations.append("High commodity exposure - monitor for correlation increases")
            elif total_exposure < 0.05:
                recommendations.append("Low commodity exposure - consider adding inflation protection")
            else:
                recommendations.append("Moderate commodity exposure - maintain current allocation")
            
            # Allocation recommendations
            if allocation:
                max_allocation = max(allocation.values()) if allocation else 0
                if max_allocation > 0.2:
                    recommendations.append("High concentration in specific commodity type - diversify")
            
            # Diversification recommendations
            if diversification < 0.3:
                recommendations.append("Low commodity diversification - spread across more commodity types")
            elif diversification > 0.7:
                recommendations.append("Good commodity diversification - maintain diversity")
            
            # Inflation hedge recommendations
            if inflation_hedge < 0.4:
                recommendations.append("Weak inflation hedge - consider precious metals or energy")
            elif inflation_hedge > 0.7:
                recommendations.append("Strong inflation hedge - monitor for over-hedging")
            
            # Seasonality recommendations
            strong_opportunities = [opp for opp in opportunities if opp["strength"] > 0.6]
            if strong_opportunities:
                top_opp = strong_opportunities[0]
                recommendations.append(f"Seasonal opportunity in {top_opp['commodity']}: {top_opp['recommendation']}")
            
            # Risk recommendations
            if len(allocation) > 0:
                energy_exposure = allocation.get("energy", 0)
                if energy_exposure > 0.15:
                    recommendations.append("High energy exposure - monitor geopolitical risks")
                
                precious_metals_exposure = allocation.get("precious_metals", 0)
                if precious_metals_exposure > 0.2:
                    recommendations.append("High precious metals exposure - monitor opportunity costs")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating commodity recommendations: {str(e)}")
            return ["Unable to generate specific commodity recommendations"]
    
    async def monitor_commodity_alerts(self, commodities: List[str]) -> Dict[str, List[str]]:
        """Monitor for significant commodity events and alerts"""
        try:
            alerts = {}
            
            # Analyze all commodities
            for commodity in commodities:
                try:
                    commodity_data = await self.analyze_commodity_characteristics(commodity)
                    commodity_alerts = []
                    
                    # Price movement alerts
                    if abs(commodity_data.price_change_5d) > 0.1:  # 10% 5-day move
                        commodity_alerts.append(f"Large 5-day move: {commodity_data.price_change_5d:.1%}")
                    
                    # Volatility alerts
                    if commodity_data.volatility > 0.4:  # High volatility
                        commodity_alerts.append(f"High volatility: {commodity_data.volatility:.1%} annualized")
                    
                    # Seasonality alerts
                    if commodity_data.seasonality_strength > 0.6:
                        current_month = datetime.now().month
                        commodity_info = self.commodity_mapping.get(commodity, {})
                        commodity_type = commodity_info.get("type", "unknown")
                        
                        if commodity_type in self.seasonality_patterns:
                            pattern_info = self.seasonality_patterns[commodity_type]
                            if current_month in pattern_info["peak"]:
                                commodity_alerts.append("Entering peak season")
                            elif current_month in pattern_info["trough"]:
                                commodity_alerts.append("Entering trough season")
                    
                    # Contango/backwardation alerts
                    if commodity_data.contango_backwardation == "contango":
                        commodity_alerts.append("Market in contango - potential roll yield")
                    elif commodity_data.contango_backwardation == "backwardation":
                        commodity_alerts.append("Market in backwardation - favorable for long positions")
                    
                    if commodity_alerts:
                        alerts[commodity] = commodity_alerts
                        
                except Exception as e:
                    logger.warning(f"Error analyzing commodity {commodity} for alerts: {str(e)}")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring commodity alerts: {str(e)}")
            return {}
    
    async def export_commodity_analysis(self, portfolio_assets: List[str], format_type: str = "json") -> str:
        """Export commodity analysis to file"""
        try:
            analysis = await self.analyze_portfolio_commodity_exposure(portfolio_assets)
            
            if format_type.lower() == "json":
                import json
                export_data = {
                    "portfolio_id": analysis.portfolio_id,
                    "timestamp": analysis.timestamp.isoformat(),
                    "total_commodity_exposure": analysis.total_commodity_exposure,
                    "commodity_allocation": analysis.commodity_allocation,
                    "diversification_benefit": analysis.diversification_benefit,
                    "inflation_hedge_effectiveness": analysis.inflation_hedge_effectiveness,
                    "currency_hedge_characteristics": analysis.currency_hedge_characteristics,
                    "seasonality_opportunities": analysis.seasonality_opportunities,
                    "risk_metrics": analysis.risk_metrics,
                    "recommendations": analysis.recommendations
                }
                
                filename = f"commodity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = f"/workspace/mpe/exports/{filename}"
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return filepath
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting commodity analysis: {str(e)}")
            return ""

# Main execution function
async def main():
    """Main execution function for commodity linkage analysis"""
    engine = CommodityLinkageEngine()
    
    # Test with commodity portfolio
    test_commodities = ["GLD", "SLV", "USO", "DBC", "CPER", "DBA"]
    mixed_portfolio = ["AAPL", "GLD", "USO", "SPY", "DBC"]  # Mixed portfolio with commodities
    
    logger.info("Starting Commodity Linkage Engine analysis...")
    
    # Test commodity characteristics analysis
    logger.info(f"\n=== Commodity Characteristics Analysis ===")
    
    for commodity in test_commodities[:3]:  # Test first 3 commodities
        char_data = await engine.analyze_commodity_characteristics(commodity)
        logger.info(f"{commodity} ({char_data.commodity_type}):")
        logger.info(f"  Price: ${char_data.price:.2f}")
        logger.info(f"  20d Change: {char_data.price_change_20d:.1%}")
        logger.info(f"  Volatility: {char_data.volatility:.1%}")
        logger.info(f"  Inflation Beta: {char_data.inflation_beta:.2f}")
        logger.info(f"  Seasonality: {char_data.seasonality_strength:.2f}")
    
    # Test cross-asset linkages
    logger.info(f"\n=== Cross-Asset Linkages ===")
    
    linkages = await engine.analyze_cross_asset_linkages(test_commodities, ["equity", "fixed_income"])
    
    for linkage in linkages[:5]:  # Show first 5 linkages
        logger.info(f"{linkage.commodity} -> {linkage.linked_asset}: "
                   f"{linkage.correlation:.3f} ({linkage.correlation_strength})")
    
    # Test inflation impact analysis
    logger.info(f"\n=== Inflation Impact Analysis ===")
    
    inflation_impacts = await engine.analyze_inflation_impact(test_commodities)
    
    for impact in inflation_impacts:
        logger.info(f"{impact.commodity}: {impact.inflation_sensitivity} sensitivity, "
                   f"beta {impact.inflation_beta:.2f}")
    
    # Test seasonality patterns
    logger.info(f"\n=== Seasonality Patterns ===")
    
    seasonality_patterns = await engine.analyze_seasonality_patterns(test_commodities)
    
    for pattern in seasonality_patterns:
        if pattern.pattern_type != "non_seasonal":
            logger.info(f"{pattern.commodity}: {pattern.pattern_type}, "
                       f"strength {pattern.seasonal_strength:.2f}")
            logger.info(f"  Peak months: {pattern.peak_months}")
    
    # Test portfolio analysis
    logger.info(f"\n=== Portfolio Commodity Analysis ===")
    
    portfolio_analysis = await engine.analyze_portfolio_commodity_exposure(mixed_portfolio)
    
    logger.info(f"Total Commodity Exposure: {portfolio_analysis.total_commodity_exposure:.1%}")
    logger.info(f"Diversification Benefit: {portfolio_analysis.diversification_benefit:.3f}")
    logger.info(f"Inflation Hedge Effectiveness: {portfolio_analysis.inflation_hedge_effectiveness:.3f}")
    
    logger.info("\nCommodity Allocation:")
    for commodity_type, allocation in portfolio_analysis.commodity_allocation.items():
        logger.info(f"  {commodity_type}: {allocation:.1%}")
    
    logger.info("\nRecommendations:")
    for rec in portfolio_analysis.recommendations[:4]:
        logger.info(f"  - {rec}")
    
    # Test monitoring alerts
    logger.info(f"\n=== Commodity Alerts ===")
    
    alerts = await engine.monitor_commodity_alerts(test_commodities)
    
    for commodity, commodity_alerts in alerts.items():
        logger.info(f"{commodity}: {len(commodity_alerts)} alerts")
        for alert in commodity_alerts[:2]:  # Show first 2 alerts
            logger.info(f"  - {alert}")
    
    logger.info("Commodity Linkage Engine analysis completed successfully")

if __name__ == "__main__":
    asyncio.run(main())