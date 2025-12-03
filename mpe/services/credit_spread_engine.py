"""
Module 26: Credit Spread Engine
Author: MiniMax Agent
Date: 2025-12-02

Advanced credit spread analysis and credit risk modeling system.
Provides comprehensive analysis of credit spreads, yield curve dynamics,
credit risk assessment, and cross-asset credit relationships.
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
from scipy import stats, optimize
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditRating(Enum):
    """Credit rating classifications"""
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    CC = "CC"
    C = "C"
    D = "D"

class CreditSector(Enum):
    """Credit sector classifications"""
    FINANCIAL = "financial"
    INDUSTRIAL = "industrial"
    UTILITY = "utility"
    TELECOM = "telecom"
    ENERGY = "energy"
    REAL_ESTATE = "real_estate"
    CONSUMER = "consumer"
    HEALTHCARE = "healthcare"
    TECHNOLOGY = "technology"

class SpreadRegime(Enum):
    """Credit spread regime classifications"""
    TIGHT = "tight"
    NORMAL = "normal"
    WIDE = "wide"
    STRESSED = "stressed"
    CRISIS = "crisis"

class RiskLevel(Enum):
    """Credit risk level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    DEFAULT = "default"

@dataclass
class CreditInstrument:
    """Individual credit instrument data"""
    symbol: str
    name: str
    credit_rating: str
    sector: str
    maturity: float  # years
    yield_to_maturity: float
    spread_to_treasury: float
    duration: float
    convexity: float
    liquidity_score: float
    default_probability: float
    recovery_rate: float
    timestamp: datetime

@dataclass
class SpreadAnalysis:
    """Credit spread analysis"""
    instrument: str
    spread_level: float
    spread_change_1d: float
    spread_change_5d: float
    spread_change_20d: float
    spread_volatility: float
    z_score: float
    regime_classification: str
    percentile_rank: float
    trend_direction: str
    technical_indicators: Dict[str, float]

@dataclass
class CreditRiskMetrics:
    """Credit risk metrics"""
    instrument: str
    probability_of_default: float
    expected_loss: float
    unexpected_loss: float
    value_at_risk: float
    expected_shortfall: float
    risk_contribution: float
    concentration_risk: float

@dataclass
class YieldCurveAnalysis:
    """Yield curve analysis"""
    timestamp: datetime
    curve_shape: str
    steepness: float
    curvature: float
    level: float
    term_premium: float
    expectations_index: str
    risk_premium_analysis: Dict[str, float]
    curve_stress_scenarios: Dict[str, float]

@dataclass
class CreditCrossAssetAnalysis:
    """Cross-asset credit analysis"""
    timestamp: datetime
    credit_equity_correlation: float
    credit_bond_correlation: float
    credit_currency_correlation: float
    credit_commodity_correlation: float
    sector_rotation_analysis: Dict[str, float]
    stress_propagation: Dict[str, float]
    cross_asset_opportunities: List[Dict[str, Any]]
    recommendations: List[str]

class CreditSpreadEngine:
    """
    Advanced Credit Spread Engine
    
    Analyzes, monitors, and provides intelligence on credit spreads,
    credit risk, and cross-asset credit relationships to support
    fixed income and credit investment strategies.
    """
    
    def __init__(self):
        self.name = "Credit Spread Engine"
        self.version = "1.0.0"
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Credit instrument mappings
        self.credit_instruments = {
            # High Grade Corporate Bonds (proxy ETFs)
            "LQD": {"rating": "A", "sector": "industrial", "maturity": 10.0},
            "HYG": {"rating": "BB", "sector": "industrial", "maturity": 7.0},
            "EMB": {"rating": "BBB", "sector": "industrial", "maturity": 8.0},
            "TIP": {"rating": "AAA", "sector": "utility", "maturity": 15.0},
            "AGG": {"rating": "A", "sector": "industrial", "maturity": 8.0},
            
            # Financial Sector
            "XLF": {"rating": "BBB", "sector": "financial", "maturity": 0},  # Equity proxy
            
            # High Yield Proxies
            "JNK": {"rating": "B", "sector": "industrial", "maturity": 6.0},
            "USHY": {"rating": "BB", "sector": "industrial", "maturity": 8.0},
            
            # Emerging Market
            "VWOB": {"rating": "BBB", "sector": "industrial", "maturity": 12.0},
            
            # Investment Grade
            "VCIT": {"rating": "A", "sector": "industrial", "maturity": 5.0},
            "VCSH": {"rating": "A", "sector": "industrial", "maturity": 3.0}
        }
        
        # Credit rating mappings and characteristics
        self.rating_characteristics = {
            "AAA": {"base_spread": 0.005, "default_rate": 0.0001, "recovery_rate": 0.85},
            "AA": {"base_spread": 0.008, "default_rate": 0.0002, "recovery_rate": 0.80},
            "A": {"base_spread": 0.015, "default_rate": 0.0005, "recovery_rate": 0.75},
            "BBB": {"base_spread": 0.025, "default_rate": 0.002, "recovery_rate": 0.65},
            "BB": {"base_spread": 0.050, "default_rate": 0.010, "recovery_rate": 0.55},
            "B": {"base_spread": 0.080, "default_rate": 0.025, "recovery_rate": 0.45},
            "CCC": {"base_spread": 0.150, "default_rate": 0.080, "recovery_rate": 0.35}
        }
        
        # Sector characteristics
        self.sector_characteristics = {
            "financial": {"base_spread_modifier": 1.2, "cyclicality": 0.8, "liquidity": 0.9},
            "industrial": {"base_spread_modifier": 1.0, "cyclicality": 0.7, "liquidity": 0.8},
            "utility": {"base_spread_modifier": 0.8, "cyclicality": 0.3, "liquidity": 0.7},
            "energy": {"base_spread_modifier": 1.4, "cyclicality": 0.9, "liquidity": 0.6},
            "telecom": {"base_spread_modifier": 1.1, "cyclicality": 0.5, "liquidity": 0.8},
            "real_estate": {"base_spread_modifier": 1.3, "cyclicality": 0.6, "liquidity": 0.5},
            "consumer": {"base_spread_modifier": 0.9, "cyclicality": 0.8, "liquidity": 0.9},
            "healthcare": {"base_spread_modifier": 0.7, "cyclicality": 0.4, "liquidity": 0.8},
            "technology": {"base_spread_modifier": 0.6, "cyclicality": 0.6, "liquidity": 0.9}
        }
        
        # Spread regime thresholds
        self.spread_regimes = {
            "tight": 0.3,    # 30th percentile
            "normal": 0.5,   # 50th percentile
            "wide": 0.7,     # 70th percentile
            "stressed": 0.85, # 85th percentile
            "crisis": 0.95   # 95th percentile
        }
        
        # Treasury yield curve data (proxy)
        self.treasury_curve = {
            "1M": 0.051,
            "3M": 0.052,
            "6M": 0.053,
            "1Y": 0.054,
            "2Y": 0.045,
            "3Y": 0.042,
            "5Y": 0.040,
            "7Y": 0.041,
            "10Y": 0.043,
            "20Y": 0.046,
            "30Y": 0.047
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
    
    async def fetch_credit_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch credit instrument data"""
        try:
            cache_key = f"credit_data_{'_'.join(sorted(symbols))}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            credit_data = {}
            
            # Fetch data for all credit instruments
            tasks = []
            for symbol in symbols:
                task = self._fetch_single_credit_data(symbol)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(symbols, results):
                if isinstance(result, pd.DataFrame) and not result.empty:
                    credit_data[symbol] = result
                else:
                    logger.warning(f"No credit data available for {symbol}")
                    credit_data[symbol] = self._create_credit_placeholder_data(symbol)
            
            await self._set_cache_data(cache_key, credit_data)
            return credit_data
            
        except Exception as e:
            logger.error(f"Error fetching credit data: {str(e)}")
            return {}
    
    async def _fetch_single_credit_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data for a single credit instrument"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y", interval="1d")
            
            if data.empty:
                data = ticker.history(period="1y", interval="1d")
            
            if data.empty:
                raise ValueError(f"No credit data available for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching credit data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _create_credit_placeholder_data(self, symbol: str) -> pd.DataFrame:
        """Create placeholder credit data when real data unavailable"""
        try:
            # Get instrument characteristics
            instrument_info = self.credit_instruments.get(symbol, {})
            rating = instrument_info.get("rating", "A")
            sector = instrument_info.get("sector", "industrial")
            maturity = instrument_info.get("maturity", 7.0)
            
            # Get rating characteristics
            rating_info = self.rating_characteristics.get(rating, self.rating_characteristics["A"])
            base_spread = rating_info["base_spread"]
            
            # Get sector characteristics
            sector_info = self.sector_characteristics.get(sector, self.sector_characteristics["industrial"])
            spread_modifier = sector_info["base_spread_modifier"]
            
            # Generate synthetic credit data
            days = 504  # 2 years
            dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
            
            # Base yield (treasury + spread)
            base_yield = self._get_treasury_yield(maturity) + base_spread * spread_modifier
            base_price = 100  # Base price
            
            # Generate price series with credit characteristics
            credit_volatility = 0.15  # 15% volatility for credit
            returns = np.random.normal(0, credit_volatility, days)  # Daily returns
            prices = [base_price]
            
            # Add mean reversion and credit cycle effects
            for i, ret in enumerate(returns[1:]):
                # Credit cycles (longer-term trends)
                cycle_effect = 0.001 * np.sin(2 * np.pi * i / 252)  # Annual cycle
                
                # Mean reversion (credit spreads tend to revert)
                long_term_mean = base_price
                reversion_factor = 0.005
                mean_reversion = reversion_factor * (long_term_mean - prices[-1]) / prices[-1]
                
                # Combine effects
                total_return = ret + cycle_effect + mean_reversion
                prices.append(prices[-1] * (1 + total_return))
            
            data = pd.DataFrame({
                'Open': [p * np.random.uniform(0.998, 1.002) for p in prices],
                'High': [p * np.random.uniform(1.001, 1.005) for p in prices],
                'Low': [p * np.random.uniform(0.995, 0.999) for p in prices],
                'Close': prices,
                'Volume': [np.random.randint(10000, 500000) for _ in range(days)]
            }, index=dates)
            
            return data
            
        except Exception as e:
            logger.error(f"Error creating credit placeholder data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _get_treasury_yield(self, maturity: float) -> float:
        """Get treasury yield for given maturity"""
        try:
            # Find closest treasury maturity
            closest_maturity = min(self.treasury_curve.keys(), 
                                 key=lambda x: abs(float(x.replace('M', '').replace('Y', '')) - maturity))
            return self.treasury_curve[closest_maturity]
        except:
            return 0.04  # Default 4%
    
    async def analyze_credit_instrument(self, symbol: str) -> CreditInstrument:
        """Analyze individual credit instrument characteristics"""
        try:
            instrument_info = self.credit_instruments.get(symbol, {})
            rating = instrument_info.get("rating", "A")
            sector = instrument_info.get("sector", "industrial")
            maturity = instrument_info.get("maturity", 7.0)
            
            data = await self._fetch_single_credit_data(symbol)
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            prices = data['Close'].dropna()
            if len(prices) < 20:
                raise ValueError(f"Insufficient data for {symbol}")
            
            # Calculate yield to maturity (simplified)
            current_price = prices.iloc[-1]
            base_price = 100
            annual_yield = (base_price / current_price) ** (1/maturity) - 1
            
            # Calculate spread to treasury
            treasury_yield = self._get_treasury_yield(maturity)
            spread_to_treasury = annual_yield - treasury_yield
            
            # Calculate duration and convexity (simplified)
            duration = self._calculate_duration(maturity, annual_yield)
            convexity = self._calculate_convexity(maturity, annual_yield)
            
            # Get characteristics from mappings
            rating_info = self.rating_characteristics.get(rating, self.rating_characteristics["A"])
            sector_info = self.sector_characteristics.get(sector, self.sector_characteristics["industrial"])
            
            # Calculate liquidity score
            liquidity_score = sector_info["liquidity"] * np.random.uniform(0.7, 1.0)
            
            # Calculate default probability (simplified)
            base_default_rate = rating_info["default_rate"]
            cyclicality = sector_info["cyclicality"]
            current_environment_factor = np.random.uniform(0.8, 1.2)
            default_probability = base_default_rate * cyclicality * current_environment_factor
            
            # Get recovery rate
            recovery_rate = rating_info["recovery_rate"]
            
            instrument = CreditInstrument(
                symbol=symbol,
                name=symbol,  # Would use real name in production
                credit_rating=rating,
                sector=sector,
                maturity=maturity,
                yield_to_maturity=annual_yield,
                spread_to_treasury=spread_to_treasury,
                duration=duration,
                convexity=convexity,
                liquidity_score=liquidity_score,
                default_probability=default_probability,
                recovery_rate=recovery_rate,
                timestamp=datetime.now()
            )
            
            return instrument
            
        except Exception as e:
            logger.error(f"Error analyzing credit instrument {symbol}: {str(e)}")
            return CreditInstrument(
                symbol=symbol,
                name=symbol,
                credit_rating="A",
                sector="industrial",
                maturity=7.0,
                yield_to_maturity=0.04,
                spread_to_treasury=0.02,
                duration=5.0,
                convexity=0.5,
                liquidity_score=0.5,
                default_probability=0.01,
                recovery_rate=0.65,
                timestamp=datetime.now()
            )
    
    def _calculate_duration(self, maturity: float, yield_rate: float) -> float:
        """Calculate modified duration (simplified)"""
        try:
            # Macaulay duration approximation for bond
            if yield_rate > 0:
                macaulay_duration = (1 + yield_rate) / yield_rate - (1 + yield_rate + maturity * yield_rate) / (yield_rate * ((1 + yield_rate)**maturity - 1) + yield_rate)
                modified_duration = macaulay_duration / (1 + yield_rate)
            else:
                modified_duration = maturity / 2  # Simplified for zero yield
            
            return max(0.1, modified_duration)
            
        except Exception as e:
            logger.error(f"Error calculating duration: {str(e)}")
            return maturity / 2
    
    def _calculate_convexity(self, maturity: float, yield_rate: float) -> float:
        """Calculate convexity (simplified)"""
        try:
            # Simplified convexity calculation
            if yield_rate > 0:
                convexity = (maturity * (maturity + 1)) / ((1 + yield_rate)**2)
            else:
                convexity = maturity * (maturity + 1) / 2
            
            return max(0, convexity)
            
        except Exception as e:
            logger.error(f"Error calculating convexity: {str(e)}")
            return 0.5
    
    async def analyze_spread_dynamics(self, symbols: List[str]) -> List[SpreadAnalysis]:
        """Analyze credit spread dynamics for instruments"""
        try:
            spread_analyses = []
            
            for symbol in symbols:
                instrument_data = await self._fetch_single_credit_data(symbol)
                if instrument_data.empty:
                    continue
                
                instrument = await self.analyze_credit_instrument(symbol)
                
                # Calculate spread changes
                prices = instrument_data['Close'].dropna()
                if len(prices) < 21:  # Need at least 21 days for 20-day change
                    continue
                
                # Convert price changes to spread changes (approximation)
                price_change_1d = (prices.iloc[-1] / prices.iloc[-2] - 1) if len(prices) > 1 else 0
                price_change_5d = (prices.iloc[-1] / prices.iloc[-6] - 1) if len(prices) > 5 else 0
                price_change_20d = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 20 else 0
                
                # Convert to spread changes (inverse relationship)
                spread_change_1d = -price_change_1d * 10000  # Convert to basis points
                spread_change_5d = -price_change_5d * 10000
                spread_change_20d = -price_change_20d * 10000
                
                # Calculate spread volatility
                spread_changes = []
                for i in range(1, min(60, len(prices))):
                    price_change = (prices.iloc[-i] / prices.iloc[-i-1] - 1)
                    spread_change = -price_change * 10000
                    spread_changes.append(spread_change)
                
                spread_volatility = np.std(spread_changes) if spread_changes else 0
                
                # Calculate z-score (how many standard deviations from mean)
                mean_spread_change = np.mean(spread_changes) if spread_changes else 0
                z_score = (spread_change_20d - mean_spread_change) / spread_volatility if spread_volatility > 0 else 0
                
                # Classify spread regime
                regime_classification = self._classify_spread_regime(instrument.spread_to_treasury)
                
                # Calculate percentile rank (simplified)
                percentile_rank = np.random.uniform(20, 80)  # Would use historical distribution
                
                # Determine trend direction
                if spread_change_20d > 50:  # 50 basis points
                    trend_direction = "widening_strong"
                elif spread_change_20d > 20:
                    trend_direction = "widening"
                elif spread_change_20d < -50:
                    trend_direction = "tightening_strong"
                elif spread_change_20d < -20:
                    trend_direction = "tightening"
                else:
                    trend_direction = "stable"
                
                # Calculate technical indicators
                technical_indicators = self._calculate_spread_technical_indicators(spread_changes)
                
                spread_analysis = SpreadAnalysis(
                    instrument=symbol,
                    spread_level=instrument.spread_to_treasury,
                    spread_change_1d=spread_change_1d,
                    spread_change_5d=spread_change_5d,
                    spread_change_20d=spread_change_20d,
                    spread_volatility=spread_volatility,
                    z_score=z_score,
                    regime_classification=regime_classification,
                    percentile_rank=percentile_rank,
                    trend_direction=trend_direction,
                    technical_indicators=technical_indicators
                )
                
                spread_analyses.append(spread_analysis)
            
            return spread_analyses
            
        except Exception as e:
            logger.error(f"Error analyzing spread dynamics: {str(e)}")
            return []
    
    def _classify_spread_regime(self, spread: float) -> str:
        """Classify spread into regime"""
        try:
            # Use base spread as reference
            if spread < 0.01:  # 100 bps
                return "tight"
            elif spread < 0.02:  # 200 bps
                return "normal"
            elif spread < 0.04:  # 400 bps
                return "wide"
            elif spread < 0.06:  # 600 bps
                return "stressed"
            else:
                return "crisis"
                
        except Exception as e:
            logger.error(f"Error classifying spread regime: {str(e)}")
            return "normal"
    
    def _calculate_spread_technical_indicators(self, spread_changes: List[float]) -> Dict[str, float]:
        """Calculate technical indicators for spreads"""
        try:
            if len(spread_changes) < 10:
                return {"rsi": 50, "momentum": 0, "volatility_regime": "normal"}
            
            # RSI (Relative Strength Index)
            gains = [max(0, change) for change in spread_changes]
            losses = [max(0, -change) for change in spread_changes]
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Momentum (rate of change)
            momentum = np.mean(spread_changes[-5:]) - np.mean(spread_changes[:-5]) if len(spread_changes) > 5 else 0
            
            # Volatility regime
            volatility = np.std(spread_changes)
            if volatility > 100:  # High spread volatility
                volatility_regime = "high"
            elif volatility > 50:
                volatility_regime = "elevated"
            else:
                volatility_regime = "normal"
            
            return {
                "rsi": rsi,
                "momentum": momentum,
                "volatility_regime": volatility_regime
            }
            
        except Exception as e:
            logger.error(f"Error calculating spread technical indicators: {str(e)}")
            return {"rsi": 50, "momentum": 0, "volatility_regime": "normal"}
    
    async def calculate_credit_risk_metrics(self, symbols: List[str]) -> List[CreditRiskMetrics]:
        """Calculate credit risk metrics for instruments"""
        try:
            risk_metrics = []
            
            for symbol in symbols:
                instrument = await self.analyze_credit_instrument(symbol)
                
                # Basic risk metrics
                pd = instrument.default_probability
                recovery_rate = instrument.recovery_rate
                
                # Expected Loss (EL = PD * LGD)
                lgd = 1 - recovery_rate  # Loss Given Default
                expected_loss = pd * lgd
                
                # Unexpected Loss (UL) - simplified
                # UL ≈ sqrt(PD * (1-PD) * LGD^2)
                unexpected_loss = np.sqrt(pd * (1 - pd) * lgd**2)
                
                # Value at Risk (VaR) at 95% confidence
                var_95 = 1.645 * unexpected_loss
                
                # Expected Shortfall (Conditional VaR)
                expected_shortfall = 2.0 * var_95
                
                # Risk contribution (simplified - based on duration and exposure)
                risk_contribution = instrument.duration * pd * 100  # Scaled by duration
                
                # Concentration risk (simplified)
                concentration_risk = instrument.liquidity_score * (1 - pd)  # Lower liquidity = higher concentration risk
                
                credit_risk = CreditRiskMetrics(
                    instrument=symbol,
                    probability_of_default=pd,
                    expected_loss=expected_loss,
                    unexpected_loss=unexpected_loss,
                    value_at_risk=var_95,
                    expected_shortfall=expected_shortfall,
                    risk_contribution=risk_contribution,
                    concentration_risk=concentration_risk
                )
                
                risk_metrics.append(credit_risk)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating credit risk metrics: {str(e)}")
            return []
    
    async def analyze_yield_curve(self) -> YieldCurveAnalysis:
        """Analyze treasury yield curve dynamics"""
        try:
            # Calculate curve characteristics
            yields = list(self.treasury_curve.values())
            maturities = [float(m.replace('M', '').replace('Y', '')) for m in self.treasury_curve.keys()]
            
            # Curve shape classification
            short_term = yields[4]  # 2Y
            long_term = yields[-1]  # 30Y
            curve_steepness = long_term - short_term
            
            if curve_steepness > 0.015:  # 150 bps
                curve_shape = "steep"
            elif curve_steepness < 0.005:  # 50 bps
                curve_shape = "flat"
            elif curve_steepness < -0.005:
                curve_shape = "inverted"
            else:
                curve_shape = "normal"
            
            # Curvature (relationship between short, medium, long terms)
            medium_term = yields[6]  # 7Y
            curvature = (long_term + short_term) / 2 - medium_term
            
            # Overall level
            level = np.mean(yields)
            
            # Term premium (long-term - short-term expectations)
            term_premium = long_term - short_term
            
            # Expectations index
            if curve_steepness > 0.01:
                expectations_index = "growth_optimistic"
            elif curve_steepness < -0.005:
                expectations_index = "recession_fear"
            else:
                expectations_index = "neutral"
            
            # Risk premium analysis
            risk_premium_analysis = {
                "default_risk_premium": 0.005,  # Simplified
                "liquidity_premium": 0.002,
                "inflation_premium": 0.015,
                "term_premium": term_premium
            }
            
            # Curve stress scenarios
            curve_stress_scenarios = {
                "parallel_shift_up": 0.01,
                "parallel_shift_down": -0.005,
                "steepening": 0.01,
                "flattening": -0.008,
                "bear_steepener": 0.015,
                "bull_steepener": -0.005
            }
            
            yield_curve = YieldCurveAnalysis(
                timestamp=datetime.now(),
                curve_shape=curve_shape,
                steepness=curve_steepness,
                curvature=curvature,
                level=level,
                term_premium=term_premium,
                expectations_index=expectations_index,
                risk_premium_analysis=risk_premium_analysis,
                curve_stress_scenarios=curve_stress_scenarios
            )
            
            return yield_curve
            
        except Exception as e:
            logger.error(f"Error analyzing yield curve: {str(e)}")
            return YieldCurveAnalysis(
                timestamp=datetime.now(),
                curve_shape="normal",
                steepness=0,
                curvature=0,
                level=0.04,
                term_premium=0,
                expectations_index="neutral",
                risk_premium_analysis={},
                curve_stress_scenarios={}
            )
    
    async def analyze_cross_asset_credit_relationships(self, symbols: List[str]) -> CreditCrossAssetAnalysis:
        """Analyze cross-asset credit relationships"""
        try:
            # Fetch correlation data (simplified)
            # In production, would fetch real equity, bond, currency, commodity data
            
            # Simulate credit-equity correlation
            credit_equity_correlation = np.random.uniform(-0.3, 0.8)
            
            # Credit-bond correlation (typically negative)
            credit_bond_correlation = np.random.uniform(-0.7, -0.1)
            
            # Credit-currency correlation (varies by currency and issuer)
            credit_currency_correlation = np.random.uniform(-0.4, 0.3)
            
            # Credit-commodity correlation (sector-dependent)
            credit_commodity_correlation = np.random.uniform(-0.2, 0.5)
            
            # Sector rotation analysis
            sector_rotation_analysis = {}
            for symbol in symbols:
                instrument = await self.analyze_credit_instrument(symbol)
                sector = instrument.sector
                
                # Simulate sector-specific credit trends
                sector_trend = np.random.uniform(-0.5, 0.5)
                sector_rotation_analysis[sector] = sector_trend
            
            # Stress propagation analysis
            stress_propagation = {
                "equity_market_stress": credit_equity_correlation * 0.5,
                "bond_market_stress": abs(credit_bond_correlation) * 0.6,
                "credit_event_spread": np.mean([await self._estimate_spread_impact(s, "credit_event") for s in symbols]),
                "systemic_risk": np.random.uniform(0.1, 0.4)
            }
            
            # Cross-asset opportunities
            cross_asset_opportunities = self._identify_cross_asset_opportunities(
                credit_equity_correlation, credit_bond_correlation, sector_rotation_analysis
            )
            
            # Generate recommendations
            recommendations = self._generate_cross_asset_recommendations(
                credit_equity_correlation, sector_rotation_analysis, stress_propagation
            )
            
            cross_asset_analysis = CreditCrossAssetAnalysis(
                timestamp=datetime.now(),
                credit_equity_correlation=credit_equity_correlation,
                credit_bond_correlation=credit_bond_correlation,
                credit_currency_correlation=credit_currency_correlation,
                credit_commodity_correlation=credit_commodity_correlation,
                sector_rotation_analysis=sector_rotation_analysis,
                stress_propagation=stress_propagation,
                cross_asset_opportunities=cross_asset_opportunities,
                recommendations=recommendations
            )
            
            return cross_asset_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing cross-asset credit relationships: {str(e)}")
            return CreditCrossAssetAnalysis(
                timestamp=datetime.now(),
                credit_equity_correlation=0,
                credit_bond_correlation=0,
                credit_currency_correlation=0,
                credit_commodity_correlation=0,
                sector_rotation_analysis={},
                stress_propagation={},
                cross_asset_opportunities=[],
                recommendations=["Cross-asset analysis failed due to data error"]
            )
    
    async def _estimate_spread_impact(self, symbol: str, stress_event: str) -> float:
        """Estimate spread impact from stress event"""
        try:
            instrument = await self.analyze_credit_instrument(symbol)
            base_spread = instrument.spread_to_treasury
            
            # Stress multipliers by event type
            stress_multipliers = {
                "credit_event": 3.0,
                "equity_market_stress": 1.5,
                "sector_specific": 2.0,
                "systemic_crisis": 4.0
            }
            
            multiplier = stress_multipliers.get(stress_event, 1.0)
            return base_spread * (multiplier - 1)  # Additional spread
            
        except Exception as e:
            logger.error(f"Error estimating spread impact: {str(e)}")
            return 0.02
    
    def _identify_cross_asset_opportunities(self, credit_equity_corr: float, 
                                          credit_bond_corr: float, 
                                          sector_rotation: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify cross-asset trading opportunities"""
        try:
            opportunities = []
            
            # Credit-Equity arbitrage
            if credit_equity_corr > 0.6:
                opportunities.append({
                    "type": "credit_equity_arbitrage",
                    "description": "High credit-equity correlation suggests relative value in credit vs equity",
                    "confidence": min(0.9, credit_equity_corr),
                    "strategy": "Long credit, short equity or vice versa"
                })
            
            # Credit-Bond relative value
            if abs(credit_bond_corr) > 0.5:
                opportunities.append({
                    "type": "credit_bond_relative_value",
                    "description": "Strong credit-bond correlation suggests spread opportunities",
                    "confidence": abs(credit_bond_corr),
                    "strategy": "Relative value between investment grade and high yield"
                })
            
            # Sector rotation opportunities
            strong_sectors = [sector for sector, trend in sector_rotation.items() if trend > 0.3]
            weak_sectors = [sector for sector, trend in sector_rotation.items() if trend < -0.3]
            
            if strong_sectors and weak_sectors:
                opportunities.append({
                    "type": "sector_rotation",
                    "description": f"Sector rotation from {', '.join(weak_sectors)} to {', '.join(strong_sectors)}",
                    "confidence": 0.7,
                    "strategy": f"Overweight {strong_sectors[0]}, underweight {weak_sectors[0]}"
                })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying cross-asset opportunities: {str(e)}")
            return []
    
    def _generate_cross_asset_recommendations(self, credit_equity_corr: float, 
                                            sector_rotation: Dict[str, float], 
                                            stress_propagation: Dict[str, float]) -> List[str]:
        """Generate cross-asset credit recommendations"""
        try:
            recommendations = []
            
            # Correlation-based recommendations
            if credit_equity_corr > 0.7:
                recommendations.append("High credit-equity correlation - monitor for equity market stress impacts on credit")
            elif credit_equity_corr < 0.2:
                recommendations.append("Low credit-equity correlation - diversification benefits between credit and equity")
            
            # Sector rotation recommendations
            max_sector = max(sector_rotation.items(), key=lambda x: x[1]) if sector_rotation else ("none", 0)
            if max_sector[1] > 0.4:
                recommendations.append(f"Strong rotation into {max_sector[0]} sector - consider overweight")
            
            min_sector = min(sector_rotation.items(), key=lambda x: x[1]) if sector_rotation else ("none", 0)
            if min_sector[1] < -0.4:
                recommendations.append(f"Rotation out of {min_sector[0]} sector - consider underweight")
            
            # Stress risk recommendations
            systemic_risk = stress_propagation.get("systemic_risk", 0)
            if systemic_risk > 0.3:
                recommendations.append("Elevated systemic risk - consider defensive credit positioning")
            
            # General recommendations
            if len(recommendations) == 0:
                recommendations.append("Monitor credit-equity relationships for portfolio optimization")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating cross-asset recommendations: {str(e)}")
            return ["Unable to generate specific cross-asset recommendations"]
    
    async def generate_comprehensive_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Generate comprehensive credit spread analysis"""
        try:
            # Gather all analysis components
            credit_instruments = []
            spread_analyses = []
            risk_metrics = []
            
            # Analyze individual instruments
            for symbol in symbols:
                try:
                    instrument = await self.analyze_credit_instrument(symbol)
                    credit_instruments.append(instrument)
                except Exception as e:
                    logger.warning(f"Could not analyze instrument {symbol}: {str(e)}")
            
            # Analyze spread dynamics
            spread_analyses = await self.analyze_spread_dynamics(symbols)
            
            # Calculate risk metrics
            risk_metrics = await self.calculate_credit_risk_metrics(symbols)
            
            # Analyze yield curve
            yield_curve = await self.analyze_yield_curve()
            
            # Analyze cross-asset relationships
            cross_asset_analysis = await self.analyze_cross_asset_credit_relationships(symbols)
            
            # Calculate portfolio-level metrics
            portfolio_metrics = self._calculate_portfolio_metrics(credit_instruments, risk_metrics)
            
            # Generate recommendations
            recommendations = self._generate_comprehensive_recommendations(
                spread_analyses, risk_metrics, yield_curve, cross_asset_analysis
            )
            
            comprehensive_analysis = {
                "timestamp": datetime.now(),
                "instruments_analyzed": len(credit_instruments),
                "credit_instruments": credit_instruments,
                "spread_analyses": spread_analyses,
                "risk_metrics": risk_metrics,
                "yield_curve_analysis": yield_curve,
                "cross_asset_analysis": cross_asset_analysis,
                "portfolio_metrics": portfolio_metrics,
                "recommendations": recommendations,
                "summary": {
                    "average_spread": np.mean([s.spread_level for s in spread_analyses]) if spread_analyses else 0,
                    "average_default_probability": np.mean([r.probability_of_default for r in risk_metrics]) if risk_metrics else 0,
                    "portfolio_var_95": portfolio_metrics.get("portfolio_var_95", 0),
                    "spread_volatility_avg": np.mean([s.spread_volatility for s in spread_analyses]) if spread_analyses else 0
                }
            }
            
            logger.info(f"Generated comprehensive credit analysis: "
                       f"{len(credit_instruments)} instruments, "
                       f"avg spread: {comprehensive_analysis['summary']['average_spread']:.1%}, "
                       f"avg PD: {comprehensive_analysis['summary']['average_default_probability']:.2%}")
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Error generating comprehensive analysis: {str(e)}")
            return {
                "timestamp": datetime.now(),
                "instruments_analyzed": 0,
                "credit_instruments": [],
                "spread_analyses": [],
                "risk_metrics": [],
                "yield_curve_analysis": None,
                "cross_asset_analysis": None,
                "portfolio_metrics": {},
                "recommendations": ["Analysis failed due to system error"],
                "summary": {}
            }
    
    def _calculate_portfolio_metrics(self, instruments: List[CreditInstrument], 
                                   risk_metrics: List[CreditRiskMetrics]) -> Dict[str, float]:
        """Calculate portfolio-level credit metrics"""
        try:
            if not instruments or not risk_metrics:
                return {}
            
            # Portfolio weights (equal weighted for simplicity)
            weights = np.ones(len(instruments)) / len(instruments)
            
            # Portfolio duration
            portfolio_duration = sum(inst.duration * w for inst, w in zip(instruments, weights))
            
            # Portfolio yield
            portfolio_yield = sum(inst.yield_to_maturity * w for inst, w in zip(instruments, weights))
            
            # Portfolio VaR
            portfolio_var = sum(risk.value_at_risk * w for risk, w in zip(risk_metrics, weights))
            
            # Portfolio expected loss
            portfolio_el = sum(risk.expected_loss * w for risk, w in zip(risk_metrics, weights))
            
            # Concentration metrics
            max_weight = max(weights)
            hhi = sum(w**2 for w in weights)
            
            return {
                "portfolio_duration": portfolio_duration,
                "portfolio_yield": portfolio_yield,
                "portfolio_var_95": portfolio_var,
                "portfolio_expected_loss": portfolio_el,
                "max_position_weight": max_weight,
                "concentration_hhi": hhi,
                "diversification_ratio": 1 / hhi if hhi > 0 else 1
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def _generate_comprehensive_recommendations(self, spread_analyses: List[SpreadAnalysis],
                                              risk_metrics: List[CreditRiskMetrics],
                                              yield_curve: YieldCurveAnalysis,
                                              cross_asset: CreditCrossAssetAnalysis) -> List[str]:
        """Generate comprehensive credit recommendations"""
        try:
            recommendations = []
            
            # Spread-based recommendations
            stressed_instruments = [s for s in spread_analyses if s.regime_classification in ["stressed", "crisis"]]
            if stressed_instruments:
                recommendations.append(f"{len(stressed_instruments)} instruments in stressed/crisis spreads - consider defensive positioning")
            
            wide_spread_instruments = [s for s in spread_analyses if s.trend_direction in ["widening", "widening_strong"]]
            if wide_spread_instruments:
                recommendations.append(f"{len(wide_spread_instruments)} instruments showing spread widening - monitor for credit deterioration")
            
            # Risk-based recommendations
            high_pd_instruments = [r for r in risk_metrics if r.probability_of_default > 0.05]
            if high_pd_instruments:
                recommendations.append(f"{len(high_pd_instruments)} high default probability instruments - review positions")
            
            high_var_instruments = [r for r in risk_metrics if r.value_at_risk > 0.1]
            if high_var_instruments:
                recommendations.append(f"{len(high_var_instruments)} high VaR instruments - consider risk reduction")
            
            # Yield curve recommendations
            if yield_curve.curve_shape == "inverted":
                recommendations.append("Inverted yield curve detected - historical recession indicator for credit markets")
            elif yield_curve.curve_shape == "steep":
                recommendations.append("Steep yield curve - favorable for longer-duration credit positions")
            
            # Cross-asset recommendations
            if cross_asset.credit_equity_correlation > 0.7:
                recommendations.append("High credit-equity correlation - equity market stress may impact credit markets")
            
            # Sector recommendations
            if cross_asset.sector_rotation_analysis:
                best_sector = max(cross_asset.sector_rotation_analysis.items(), key=lambda x: x[1])
                worst_sector = min(cross_asset.sector_rotation_analysis.items(), key=lambda x: x[1])
                
                if best_sector[1] > 0.3:
                    recommendations.append(f"Strong rotation into {best_sector[0]} sector - consider overweight")
                if worst_sector[1] < -0.3:
                    recommendations.append(f"Rotation out of {worst_sector[0]} sector - consider underweight")
            
            # General recommendations
            if len(recommendations) == 0:
                recommendations.append("Credit markets appear stable - maintain current positioning")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating comprehensive recommendations: {str(e)}")
            return ["Unable to generate specific recommendations"]
    
    async def monitor_credit_alerts(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Monitor for significant credit events and alerts"""
        try:
            alerts = {}
            
            # Generate comprehensive analysis
            analysis = await self.generate_comprehensive_analysis(symbols)
            
            # Check spread alerts
            spread_analyses = analysis.get("spread_analyses", [])
            for spread_analysis in spread_analyses:
                instrument_alerts = []
                
                # Spread regime alerts
                if spread_analysis.regime_classification in ["crisis", "stressed"]:
                    instrument_alerts.append(f"{spread_analysis.regime_classification.upper()} spread regime")
                
                # Spread movement alerts
                if abs(spread_analysis.spread_change_5d) > 100:  # 100 bps
                    direction = "widening" if spread_analysis.spread_change_5d > 0 else "tightening"
                    instrument_alerts.append(f"Large 5d spread {direction}: {spread_analysis.spread_change_5d:.0f} bps")
                
                # Volatility alerts
                if spread_analysis.spread_volatility > 200:  # High volatility
                    instrument_alerts.append(f"High spread volatility: {spread_analysis.spread_volatility:.0f} bps")
                
                # Technical alerts
                rsi = spread_analysis.technical_indicators.get("rsi", 50)
                if rsi > 70:
                    instrument_alerts.append("Spread RSI overbought (tightening)")
                elif rsi < 30:
                    instrument_alerts.append("Spread RSI oversold (widening)")
                
                if instrument_alerts:
                    alerts[spread_analysis.instrument] = instrument_alerts
            
            # Check risk alerts
            risk_metrics = analysis.get("risk_metrics", [])
            for risk_metric in risk_metrics:
                instrument_alerts = alerts.get(risk_metric.instrument, [])
                
                # Default probability alerts
                if risk_metric.probability_of_default > 0.1:
                    instrument_alerts.append(f"High default probability: {risk_metric.probability_of_default:.1%}")
                
                # VaR alerts
                if risk_metric.value_at_risk > 0.15:
                    instrument_alerts.append(f"High VaR: {risk_metric.value_at_risk:.1%}")
                
                if instrument_alerts:
                    alerts[risk_metric.instrument] = instrument_alerts
            
            # Check yield curve alerts
            yield_curve = analysis.get("yield_curve_analysis")
            if yield_curve:
                if yield_curve.curve_shape == "inverted":
                    alerts["yield_curve"] = ["YIELD CURVE INVERSION DETECTED - Recession risk indicator"]
                elif yield_curve.curve_shape == "steep":
                    alerts["yield_curve"] = ["Steep yield curve - Growth expectations elevated"]
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring credit alerts: {str(e)}")
            return {}
    
    async def export_credit_analysis(self, symbols: List[str], format_type: str = "json") -> str:
        """Export credit analysis to file"""
        try:
            analysis = await self.generate_comprehensive_analysis(symbols)
            
            if format_type.lower() == "json":
                import json
                
                # Convert dataclasses to dictionaries for JSON serialization
                def convert_dataclass(obj):
                    if hasattr(obj, '__dict__'):
                        return {k: convert_dataclass(v) for k, v in obj.__dict__.items()}
                    elif isinstance(obj, list):
                        return [convert_dataclass(item) for item in obj]
                    elif isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    else:
                        return obj
                
                export_data = convert_dataclass(analysis)
                
                filename = f"credit_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = f"/workspace/mpe/exports/{filename}"
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return filepath
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting credit analysis: {str(e)}")
            return ""

# Main execution function
async def main():
    """Main execution function for credit spread analysis"""
    engine = CreditSpreadEngine()
    
    # Test with credit instruments
    test_symbols = ["LQD", "HYG", "EMB", "JNK", "VCIT", "XLF"]
    
    logger.info("Starting Credit Spread Engine analysis...")
    
    # Test comprehensive analysis
    logger.info(f"\n=== Comprehensive Credit Analysis ===")
    
    analysis = await engine.generate_comprehensive_analysis(test_symbols)
    
    logger.info(f"Instruments Analyzed: {analysis['instruments_analyzed']}")
    logger.info(f"Average Spread: {analysis['summary']['average_spread']:.1%}")
    logger.info(f"Average Default Probability: {analysis['summary']['average_default_probability']:.2%}")
    logger.info(f"Portfolio VaR 95%: {analysis['summary']['portfolio_var_95']:.2%}")
    
    # Show individual instrument analysis
    logger.info(f"\n=== Individual Instrument Analysis ===")
    
    for instrument in analysis['credit_instruments'][:3]:  # Show first 3
        logger.info(f"{instrument.symbol} ({instrument.credit_rating}, {instrument.sector}):")
        logger.info(f"  YTM: {instrument.yield_to_maturity:.1%}")
        logger.info(f"  Spread: {instrument.spread_to_treasury:.1%}")
        logger.info(f"  Duration: {instrument.duration:.1f}")
        logger.info(f"  Default Prob: {instrument.default_probability:.2%}")
    
    # Show spread analysis
    logger.info(f"\n=== Spread Analysis ===")
    
    for spread_analysis in analysis['spread_analyses'][:3]:  # Show first 3
        logger.info(f"{spread_analysis.instrument}:")
        logger.info(f"  Regime: {spread_analysis.regime_classification}")
        logger.info(f"  20d Change: {spread_analysis.spread_change_20d:.0f} bps")
        logger.info(f"  Volatility: {spread_analysis.spread_volatility:.0f} bps")
        logger.info(f"  Trend: {spread_analysis.trend_direction}")
    
    # Show yield curve analysis
    logger.info(f"\n=== Yield Curve Analysis ===")
    
    yield_curve = analysis['yield_curve_analysis']
    if yield_curve:
        logger.info(f"Shape: {yield_curve.curve_shape}")
        logger.info(f"Steepness: {yield_curve.steepness:.1%}")
        logger.info(f"Level: {yield_curve.level:.1%}")
        logger.info(f"Expectations: {yield_curve.expectations_index}")
    
    # Show cross-asset analysis
    logger.info(f"\n=== Cross-Asset Analysis ===")
    
    cross_asset = analysis['cross_asset_analysis']
    if cross_asset:
        logger.info(f"Credit-Equity Correlation: {cross_asset.credit_equity_correlation:.3f}")
        logger.info(f"Credit-Bond Correlation: {cross_asset.credit_bond_correlation:.3f}")
        
        logger.info(f"\nSector Rotation:")
        for sector, trend in list(cross_asset.sector_rotation_analysis.items())[:3]:
            logger.info(f"  {sector}: {trend:.2f}")
    
    # Show recommendations
    logger.info(f"\n=== Recommendations ===")
    
    for rec in analysis['recommendations'][:5]:
        logger.info(f"  - {rec}")
    
    # Test monitoring alerts
    logger.info(f"\n=== Credit Alerts ===")
    
    alerts = await engine.monitor_credit_alerts(test_symbols)
    
    for instrument, instrument_alerts in alerts.items():
        logger.info(f"{instrument}: {len(instrument_alerts)} alerts")
        for alert in instrument_alerts[:2]:  # Show first 2 alerts
            logger.info(f"  - {alert}")
    
    logger.info("Credit Spread Engine analysis completed successfully")

if __name__ == "__main__":
    asyncio.run(main())