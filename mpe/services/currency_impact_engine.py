"""
Module 24: Currency Impact Engine
Author: MiniMax Agent
Date: 2025-12-02

Advanced currency impact analysis and FX risk management system.
Provides comprehensive analysis of currency movements' impact on asset prices,
portfolio returns, and cross-asset relationships for international investing.
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

class CurrencyPair(Enum):
    """Major currency pairs"""
    EURUSD = "EURUSD=X"
    GBPUSD = "GBPUSD=X"
    USDJPY = "JPY=X"
    USDCHF = "CHF=X"
    AUDUSD = "AUDUSD=X"
    USDCAD = "CAD=X"
    NZDUSD = "NZDUSD=X"
    USDCNY = "CNY=X"

class AssetCurrency(Enum):
    """Asset currency classifications"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    AUD = "AUD"
    CAD = "CAD"
    NZD = "NZD"
    CNY = "CNY"
    MULTI_CURRENCY = "multi_currency"

class ExposureType(Enum):
    """Types of currency exposure"""
    DIRECT = "direct"  # Asset denominated in foreign currency
    INDIRECT = "indirect"  # Asset earnings linked to foreign currency
    HEDGED = "hedged"  # Currency exposure hedged
    PARTIAL = "partial"  # Partially hedged exposure

class ImpactLevel(Enum):
    """Currency impact severity levels"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"

@dataclass
class CurrencyExposure:
    """Individual currency exposure data"""
    asset: str
    currency: str
    exposure_amount: float
    exposure_type: str
    hedging_ratio: float
    currency_beta: float
    correlation_with_fx: float
    impact_sensitivity: float
    timestamp: datetime

@dataclass
class FXMovement:
    """Foreign exchange movement data"""
    currency_pair: str
    base_currency: str
    quote_currency: str
    price_change_1d: float
    price_change_5d: float
    price_change_20d: float
    volatility: float
    trend_direction: str
    momentum_score: float
    timestamp: datetime

@dataclass
class CurrencyImpact:
    """Currency impact on asset or portfolio"""
    asset: str
    currency_movement: FXMovement
    estimated_impact: float
    impact_level: str
    confidence_score: float
    hedging_effectiveness: float
    risk_metrics: Dict[str, float]
    scenarios: Dict[str, float]

@dataclass
class PortfolioCurrencyAnalysis:
    """Comprehensive portfolio currency analysis"""
    portfolio_id: str
    timestamp: datetime
    total_fx_exposure: float
    net_currency_positions: Dict[str, float]
    currency_betas: Dict[str, float]
    fx_risk_metrics: Dict[str, float]
    hedging_opportunities: List[Dict[str, Any]]
    impact_scenarios: Dict[str, float]
    recommendations: List[str]

class CurrencyImpactEngine:
    """
    Advanced Currency Impact Engine
    
    Analyzes, monitors, and provides intelligence on currency movements'
    impact on assets and portfolios to support international investment
    and FX risk management strategies.
    """
    
    def __init__(self):
        self.name = "Currency Impact Engine"
        self.version = "1.0.0"
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Major currency mappings
        self.currency_mapping = {
            # US-focused assets
            "AAPL": "USD", "MSFT": "USD", "GOOGL": "USD", "AMZN": "USD",
            "TSLA": "USD", "META": "USD", "NVDA": "USD", "BRK-B": "USD",
            "SPY": "USD", "QQQ": "USD", "IWM": "USD", "VTI": "USD",
            
            # Fixed income
            "TLT": "USD", "IEF": "USD", "SHY": "USD", "LQD": "USD",
            "HYG": "USD", "EMB": "USD", "AGG": "USD", "BND": "USD",
            
            # International ETFs
            "VEA": "EUR",  # Developed Markets ex-US (primarily EUR)
            "VWO": "USD",  # Emerging Markets
            "IEFA": "EUR", # International Developed Markets
            "EEM": "USD",  # Emerging Markets ETF
            "EFA": "EUR",  # EAFE Index
            "VXUS": "EUR", # Total International ex-US
            "VNQ": "USD",  # Real Estate
            
            # Commodity and currency ETFs
            "GLD": "USD", "SLV": "USD", "UUP": "USD", "FXE": "EUR",
            "FXY": "JPY", "FXB": "GBP", "GLDUSD=X": "USD"
        }
        
        # Currency impact thresholds
        self.impact_thresholds = {
            "minimal": 0.005,   # 0.5%
            "low": 0.015,       # 1.5%
            "moderate": 0.03,   # 3%
            "high": 0.06,       # 6%
            "severe": 0.10      # 10%
        }
        
        # FX volatility regimes
        self.fx_volatility_regimes = {
            "low": 0.05,      # 5% annualized
            "normal": 0.10,   # 10% annualized
            "elevated": 0.15, # 15% annualized
            "high": 0.25,     # 25% annualized
            "extreme": 0.40   # 40% annualized
        }
        
        # Correlation assumptions for different asset types
        self.asset_fx_correlations = {
            "equity_domestic": 0.1,
            "equity_international": 0.7,
            "fixed_income_domestic": 0.05,
            "fixed_income_international": 0.5,
            "commodity_dollar": -0.3,
            "commodity_non_dollar": 0.6,
            "real_estate": 0.4,
            "currency_etf": 0.95
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
    
    async def fetch_fx_data(self, currency_pairs: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch FX data for specified currency pairs"""
        try:
            cache_key = f"fx_data_{'_'.join(sorted(currency_pairs))}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            fx_data = {}
            
            # Fetch data for all currency pairs
            tasks = []
            for pair in currency_pairs:
                task = self._fetch_single_fx_data(pair)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for pair, result in zip(currency_pairs, results):
                if isinstance(result, pd.DataFrame) and not result.empty:
                    fx_data[pair] = result
                else:
                    logger.warning(f"No FX data available for {pair}")
                    # Create placeholder data
                    fx_data[pair] = self._create_fx_placeholder_data(pair)
            
            await self._set_cache_data(cache_key, fx_data)
            return fx_data
            
        except Exception as e:
            logger.error(f"Error fetching FX data: {str(e)}")
            return {}
    
    async def _fetch_single_fx_data(self, currency_pair: str) -> pd.DataFrame:
        """Fetch data for a single currency pair"""
        try:
            ticker = yf.Ticker(currency_pair)
            data = ticker.history(period="1y", interval="1d")
            
            if data.empty:
                data = ticker.history(period="6mo", interval="1d")
            
            if data.empty:
                raise ValueError(f"No FX data available for {currency_pair}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching FX data for {currency_pair}: {str(e)}")
            return pd.DataFrame()
    
    def _create_fx_placeholder_data(self, currency_pair: str) -> pd.DataFrame:
        """Create placeholder FX data when real data unavailable"""
        try:
            # Extract currencies from pair
            if "=" in currency_pair:
                base_currency = currency_pair[:3]
                quote_currency = currency_pair[3:6]
            else:
                base_currency = "USD"
                quote_currency = "EUR"
            
            # Generate synthetic FX data
            days = 252  # 1 year
            dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
            
            # Base exchange rates
            if currency_pair == "EURUSD=X":
                base_rate = 1.10
                volatility = 0.12
            elif currency_pair == "GBPUSD=X":
                base_rate = 1.25
                volatility = 0.15
            elif currency_pair == "JPY=X":
                base_rate = 110.0
                volatility = 0.10
            elif currency_pair == "CHF=X":
                base_rate = 0.92
                volatility = 0.08
            elif currency_pair == "AUDUSD=X":
                base_rate = 0.75
                volatility = 0.16
            else:
                base_rate = 1.0
                volatility = 0.12
            
            # Generate returns
            returns = np.random.normal(0, volatility/np.sqrt(252), days)  # Daily returns
            prices = [base_rate]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            data = pd.DataFrame({
                'Open': [p * np.random.uniform(0.9995, 1.0005) for p in prices],
                'High': [p * np.random.uniform(1.001, 1.003) for p in prices],
                'Low': [p * np.random.uniform(0.997, 0.999) for p in prices],
                'Close': prices,
                'Volume': [np.random.randint(100000, 1000000) for _ in range(days)]
            }, index=dates)
            
            return data
            
        except Exception as e:
            logger.error(f"Error creating FX placeholder data for {currency_pair}: {str(e)}")
            return pd.DataFrame()
    
    async def calculate_fx_movements(self, fx_data: Dict[str, pd.DataFrame]) -> List[FXMovement]:
        """Calculate FX movements for all currency pairs"""
        try:
            fx_movements = []
            
            for pair, data in fx_data.items():
                if data.empty:
                    continue
                
                # Extract currencies
                if "=" in pair:
                    base_currency = pair[:3]
                    quote_currency = pair[3:6]
                else:
                    base_currency = "USD"
                    quote_currency = "EUR"
                
                prices = data['Close'].dropna()
                if len(prices) < 20:
                    continue
                
                # Calculate returns for different periods
                price_change_1d = (prices.iloc[-1] / prices.iloc[-2] - 1) if len(prices) > 1 else 0
                price_change_5d = (prices.iloc[-1] / prices.iloc[-6] - 1) if len(prices) > 5 else 0
                price_change_20d = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 20 else 0
                
                # Calculate volatility (annualized)
                daily_returns = prices.pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252)
                
                # Determine trend direction
                if price_change_20d > 0.02:
                    trend_direction = "strong_uptrend"
                elif price_change_20d > 0.005:
                    trend_direction = "uptrend"
                elif price_change_20d < -0.02:
                    trend_direction = "strong_downtrend"
                elif price_change_20d < -0.005:
                    trend_direction = "downtrend"
                else:
                    trend_direction = "sideways"
                
                # Calculate momentum score
                momentum_score = self._calculate_fx_momentum(prices)
                
                movement = FXMovement(
                    currency_pair=pair,
                    base_currency=base_currency,
                    quote_currency=quote_currency,
                    price_change_1d=price_change_1d,
                    price_change_5d=price_change_5d,
                    price_change_20d=price_change_20d,
                    volatility=volatility,
                    trend_direction=trend_direction,
                    momentum_score=momentum_score,
                    timestamp=datetime.now()
                )
                
                fx_movements.append(movement)
            
            return fx_movements
            
        except Exception as e:
            logger.error(f"Error calculating FX movements: {str(e)}")
            return []
    
    def _calculate_fx_momentum(self, prices: pd.Series) -> float:
        """Calculate FX momentum score"""
        try:
            if len(prices) < 20:
                return 0.0
            
            # Multiple momentum indicators
            # 1. Rate of change
            roc_20 = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 20 else 0
            
            # 2. Moving average relationship
            ma_short = prices.rolling(10).mean().iloc[-1]
            ma_long = prices.rolling(30).mean().iloc[-1]
            ma_signal = (ma_short / ma_long - 1) if ma_long != 0 else 0
            
            # 3. Recent trend strength
            recent_prices = prices.tail(10)
            trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            trend_strength = trend_slope / recent_prices.mean() if recent_prices.mean() != 0 else 0
            
            # Combine indicators
            momentum_score = (roc_20 * 0.5 + ma_signal * 0.3 + trend_strength * 0.2)
            
            # Normalize to -1 to 1 range
            return max(-1, min(1, momentum_score * 10))
            
        except Exception as e:
            logger.error(f"Error calculating FX momentum: {str(e)}")
            return 0.0
    
    async def analyze_currency_exposures(self, assets: List[str]) -> List[CurrencyExposure]:
        """Analyze currency exposures for a list of assets"""
        try:
            exposures = []
            
            for asset in assets:
                # Determine asset currency
                asset_currency = self.currency_mapping.get(asset, "USD")
                
                # Determine exposure type
                if asset_currency == "USD":
                    exposure_type = "domestic"
                    base_correlation = self.asset_fx_correlations["equity_domestic"]
                else:
                    exposure_type = "international"
                    base_correlation = self.asset_fx_correlations["equity_international"]
                
                # Calculate exposure amount (assuming full position for simplicity)
                exposure_amount = 1.0  # 100% of position
                
                # Default hedging ratio
                hedging_ratio = 0.0  # No hedging by default
                
                # Calculate currency beta (simplified)
                if asset_currency == "USD":
                    currency_beta = 0.0
                else:
                    currency_beta = 1.0  # Full currency exposure
                
                # Calculate correlation with FX
                correlation_with_fx = base_correlation
                
                # Calculate impact sensitivity
                impact_sensitivity = abs(currency_beta) * (1 - hedging_ratio)
                
                exposure = CurrencyExposure(
                    asset=asset,
                    currency=asset_currency,
                    exposure_amount=exposure_amount,
                    exposure_type=exposure_type,
                    hedging_ratio=hedging_ratio,
                    currency_beta=currency_beta,
                    correlation_with_fx=correlation_with_fx,
                    impact_sensitivity=impact_sensitivity,
                    timestamp=datetime.now()
                )
                
                exposures.append(exposure)
            
            return exposures
            
        except Exception as e:
            logger.error(f"Error analyzing currency exposures: {str(e)}")
            return []
    
    async def calculate_currency_impacts(self, exposures: List[CurrencyExposure], 
                                       fx_movements: List[FXMovement]) -> List[CurrencyImpact]:
        """Calculate currency impacts for all exposures"""
        try:
            currency_impacts = []
            
            for exposure in exposures:
                # Find relevant FX movement
                relevant_fx_movement = None
                for fx_movement in fx_movements:
                    if fx_movement.base_currency == exposure.currency:
                        relevant_fx_movement = fx_movement
                        break
                
                if not relevant_fx_movement:
                    continue
                
                # Calculate estimated impact
                fx_change = relevant_fx_movement.price_change_20d
                estimated_impact = fx_change * exposure.impact_sensitivity * exposure.correlation_with_fx
                
                # Determine impact level
                impact_level = self._classify_impact_level(abs(estimated_impact))
                
                # Calculate confidence score
                confidence_score = self._calculate_impact_confidence(exposure, relevant_fx_movement)
                
                # Calculate hedging effectiveness
                hedging_effectiveness = exposure.hedging_ratio * exposure.currency_beta
                
                # Generate scenarios
                scenarios = self._generate_fx_scenarios(exposure, relevant_fx_movement)
                
                # Calculate risk metrics
                risk_metrics = self._calculate_impact_risk_metrics(exposure, relevant_fx_movement)
                
                impact = CurrencyImpact(
                    asset=exposure.asset,
                    currency_movement=relevant_fx_movement,
                    estimated_impact=estimated_impact,
                    impact_level=impact_level,
                    confidence_score=confidence_score,
                    hedging_effectiveness=hedging_effectiveness,
                    risk_metrics=risk_metrics,
                    scenarios=scenarios
                )
                
                currency_impacts.append(impact)
            
            return currency_impacts
            
        except Exception as e:
            logger.error(f"Error calculating currency impacts: {str(e)}")
            return []
    
    def _classify_impact_level(self, impact: float) -> str:
        """Classify currency impact level"""
        try:
            if impact < self.impact_thresholds["minimal"]:
                return "minimal"
            elif impact < self.impact_thresholds["low"]:
                return "low"
            elif impact < self.impact_thresholds["moderate"]:
                return "moderate"
            elif impact < self.impact_thresholds["high"]:
                return "high"
            else:
                return "severe"
                
        except Exception as e:
            logger.error(f"Error classifying impact level: {str(e)}")
            return "moderate"
    
    def _calculate_impact_confidence(self, exposure: CurrencyExposure, 
                                   fx_movement: FXMovement) -> float:
        """Calculate confidence score for impact estimate"""
        try:
            # Base confidence
            confidence = 0.5
            
            # Adjust for FX volatility (higher volatility = lower confidence)
            vol_regime = self._classify_fx_volatility(fx_movement.volatility)
            volatility_adjustments = {
                "low": 0.2,
                "normal": 0.0,
                "elevated": -0.1,
                "high": -0.2,
                "extreme": -0.3
            }
            confidence += volatility_adjustments.get(vol_regime, 0)
            
            # Adjust for trend strength (stronger trend = higher confidence)
            trend_confidence_boost = min(0.2, abs(fx_movement.price_change_20d) * 2)
            confidence += trend_confidence_boost
            
            # Adjust for exposure stability
            exposure_confidence_boost = exposure.correlation_with_fx * 0.2
            confidence += exposure_confidence_boost
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating impact confidence: {str(e)}")
            return 0.5
    
    def _classify_fx_volatility(self, volatility: float) -> str:
        """Classify FX volatility regime"""
        try:
            if volatility < self.fx_volatility_regimes["low"]:
                return "low"
            elif volatility < self.fx_volatility_regimes["normal"]:
                return "normal"
            elif volatility < self.fx_volatility_regimes["elevated"]:
                return "elevated"
            elif volatility < self.fx_volatility_regimes["high"]:
                return "high"
            else:
                return "extreme"
                
        except Exception as e:
            logger.error(f"Error classifying FX volatility: {str(e)}")
            return "normal"
    
    def _generate_fx_scenarios(self, exposure: CurrencyExposure, 
                             fx_movement: FXMovement) -> Dict[str, float]:
        """Generate FX scenarios and their impacts"""
        try:
            base_impact = fx_movement.price_change_20d * exposure.impact_sensitivity * exposure.correlation_with_fx
            
            scenarios = {
                "current": base_impact,
                " fx_appreciation_5pct": base_impact * 1.2,
                "fx_appreciation_10pct": base_impact * 1.5,
                "fx_depreciation_5pct": base_impact * 0.8,
                "fx_depreciation_10pct": base_impact * 0.5,
                "volatility_spike_2x": abs(base_impact) * 2,
                "trend_reversal": -base_impact * 0.8
            }
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating FX scenarios: {str(e)}")
            return {}
    
    def _calculate_impact_risk_metrics(self, exposure: CurrencyExposure, 
                                     fx_movement: FXMovement) -> Dict[str, float]:
        """Calculate risk metrics for currency impact"""
        try:
            # Value at Risk (VaR) at 95% confidence
            fx_volatility = fx_movement.volatility / np.sqrt(252)  # Daily volatility
            var_95 = 1.645 * fx_volatility * exposure.impact_sensitivity * exposure.correlation_with_fx
            
            # Expected Shortfall (Conditional VaR)
            expected_shortfall = 2.0 * var_95
            
            # Maximum expected loss over 1 month
            max_monthly_loss = var_95 * np.sqrt(22)  # 22 trading days
            
            # Currency stress indicator
            volatility_regime = self._classify_fx_volatility(fx_movement.volatility)
            stress_multipliers = {
                "low": 1.0,
                "normal": 1.2,
                "elevated": 1.5,
                "high": 2.0,
                "extreme": 3.0
            }
            stress_multiplier = stress_multipliers.get(volatility_regime, 1.0)
            
            return {
                "var_95": var_95,
                "expected_shortfall": expected_shortfall,
                "max_monthly_loss": max_monthly_loss,
                "stress_multiplier": stress_multiplier,
                "volatility_contribution": fx_volatility * exposure.impact_sensitivity,
                "correlation_risk": exposure.correlation_with_fx * fx_volatility
            }
            
        except Exception as e:
            logger.error(f"Error calculating impact risk metrics: {str(e)}")
            return {}
    
    async def analyze_portfolio_currency_risk(self, assets: List[str], 
                                            weights: Optional[List[float]] = None) -> PortfolioCurrencyAnalysis:
        """Analyze currency risk for entire portfolio"""
        try:
            # Use equal weights if none provided
            if weights is None:
                weights = [1.0 / len(assets)] * len(assets)
            elif len(weights) != len(assets):
                logger.error("Weights length must match number of assets")
                weights = [1.0 / len(assets)] * len(assets)
            
            weights = np.array(weights)
            
            # Get exposures
            exposures = await self.analyze_currency_exposures(assets)
            
            # Get FX data
            currency_pairs = self._get_relevant_currency_pairs(exposures)
            fx_data = await self.fetch_fx_data(currency_pairs)
            fx_movements = await self.calculate_fx_movements(fx_data)
            
            # Calculate impacts
            impacts = await self.calculate_currency_impacts(exposures, fx_movements)
            
            # Calculate total FX exposure
            total_fx_exposure = sum(
                exp.exposure_amount * weight * (1 - exp.hedging_ratio)
                for exp, weight in zip(exposures, weights)
                if exp.currency != "USD"
            )
            
            # Calculate net currency positions
            net_positions = self._calculate_net_currency_positions(exposures, weights)
            
            # Calculate currency betas
            currency_betas = self._calculate_portfolio_currency_betas(exposures, weights)
            
            # Calculate FX risk metrics
            fx_risk_metrics = self._calculate_portfolio_fx_risk(impacts, weights)
            
            # Identify hedging opportunities
            hedging_opportunities = self._identify_hedging_opportunities(exposures, impacts)
            
            # Generate portfolio scenarios
            impact_scenarios = self._generate_portfolio_scenarios(impacts, weights)
            
            # Generate recommendations
            recommendations = self._generate_currency_recommendations(
                total_fx_exposure, net_positions, fx_risk_metrics, hedging_opportunities
            )
            
            analysis = PortfolioCurrencyAnalysis(
                portfolio_id="portfolio_analysis",
                timestamp=datetime.now(),
                total_fx_exposure=total_fx_exposure,
                net_currency_positions=net_positions,
                currency_betas=currency_betas,
                fx_risk_metrics=fx_risk_metrics,
                hedging_opportunities=hedging_opportunities,
                impact_scenarios=impact_scenarios,
                recommendations=recommendations
            )
            
            logger.info(f"Generated portfolio currency analysis for {len(assets)} assets: "
                       f"FX exposure: {total_fx_exposure:.3f}, "
                       f"VaR 95%: {fx_risk_metrics.get('portfolio_var_95', 0):.4f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio currency risk: {str(e)}")
            return PortfolioCurrencyAnalysis(
                portfolio_id="error_portfolio",
                timestamp=datetime.now(),
                total_fx_exposure=0,
                net_currency_positions={},
                currency_betas={},
                fx_risk_metrics={},
                hedging_opportunities=[],
                impact_scenarios={},
                recommendations=["Portfolio analysis failed due to data error"]
            )
    
    def _get_relevant_currency_pairs(self, exposures: List[CurrencyExposure]) -> List[str]:
        """Get relevant currency pairs for given exposures"""
        try:
            relevant_currencies = set()
            for exposure in exposures:
                if exposure.currency != "USD":
                    relevant_currencies.add(exposure.currency)
            
            # Convert to currency pairs (assuming USD as base)
            currency_pairs = []
            for currency in relevant_currencies:
                pair_symbol = f"{currency}USD=X"
                currency_pairs.append(pair_symbol)
            
            return currency_pairs
            
        except Exception as e:
            logger.error(f"Error getting relevant currency pairs: {str(e)}")
            return []
    
    def _calculate_net_currency_positions(self, exposures: List[CurrencyExposure], 
                                        weights: np.ndarray) -> Dict[str, float]:
        """Calculate net currency positions in portfolio"""
        try:
            net_positions = {}
            
            for exposure in exposures:
                currency = exposure.currency
                if currency == "USD":
                    continue  # Skip USD for net position calculation
                
                # Calculate net exposure for this currency
                net_exposure = exposure.exposure_amount * exposure.currency_beta * (1 - exposure.hedging_ratio)
                
                if currency not in net_positions:
                    net_positions[currency] = 0
                net_positions[currency] += net_exposure
            
            # Normalize by total portfolio value
            total_exposure = sum(abs(pos) for pos in net_positions.values())
            if total_exposure > 0:
                for currency in net_positions:
                    net_positions[currency] = net_positions[currency] / total_exposure
            
            return net_positions
            
        except Exception as e:
            logger.error(f"Error calculating net currency positions: {str(e)}")
            return {}
    
    def _calculate_portfolio_currency_betas(self, exposures: List[CurrencyExposure], 
                                          weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio currency betas"""
        try:
            currency_betas = {}
            
            # Get unique currencies (excluding USD)
            currencies = list(set(exp.currency for exp in exposures if exp.currency != "USD"))
            
            for currency in currencies:
                # Calculate weighted average beta for this currency
                currency_exposures = [exp for exp in exposures if exp.currency == currency]
                if not currency_exposures:
                    continue
                
                # Find weights for these exposures
                currency_weights = []
                currency_beta_values = []
                
                for i, exposure in enumerate(exposures):
                    if exposure.currency == currency:
                        # Find index in original exposures list
                        idx = next((j for j, exp in enumerate(exposures) if exp.asset == exposure.asset), 0)
                        currency_weights.append(weights[idx])
                        currency_beta_values.append(exposure.currency_beta)
                
                if currency_weights and currency_beta_values:
                    total_weight = sum(currency_weights)
                    if total_weight > 0:
                        weighted_beta = sum(w * b for w, b in zip(currency_weights, currency_beta_values)) / total_weight
                        currency_betas[currency] = weighted_beta
            
            return currency_betas
            
        except Exception as e:
            logger.error(f"Error calculating portfolio currency betas: {str(e)}")
            return {}
    
    def _calculate_portfolio_fx_risk(self, impacts: List[CurrencyImpact], 
                                   weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio-level FX risk metrics"""
        try:
            if not impacts:
                return {}
            
            # Aggregate risk metrics
            portfolio_var = sum(impact.risk_metrics.get("var_95", 0) * w 
                              for impact, w in zip(impacts, weights[:len(impacts)]))
            
            portfolio_expected_shortfall = sum(impact.risk_metrics.get("expected_shortfall", 0) * w
                                             for impact, w in zip(impacts, weights[:len(impacts)]))
            
            # Calculate FX correlation risk
            max_individual_var = max([impact.risk_metrics.get("var_95", 0) for impact in impacts]) if impacts else 0
            diversification_benefit = max(0, portfolio_var / max_individual_var - 1) if max_individual_var > 0 else 0
            
            # Calculate currency concentration risk
            high_impact_assets = [impact for impact in impacts if abs(impact.estimated_impact) > 0.02]
            concentration_risk = len(high_impact_assets) / len(impacts) if impacts else 0
            
            return {
                "portfolio_var_95": portfolio_var,
                "portfolio_expected_shortfall": portfolio_expected_shortfall,
                "diversification_benefit": diversification_benefit,
                "concentration_risk": concentration_risk,
                "max_individual_var": max_individual_var,
                "average_hedge_effectiveness": np.mean([impact.hedging_effectiveness for impact in impacts]) if impacts else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio FX risk: {str(e)}")
            return {}
    
    def _identify_hedging_opportunities(self, exposures: List[CurrencyExposure], 
                                      impacts: List[CurrencyImpact]) -> List[Dict[str, Any]]:
        """Identify potential hedging opportunities"""
        try:
            opportunities = []
            
            # High impact exposures that could be hedged
            for impact in impacts:
                if impact.impact_level in ["high", "severe"] and impact.confidence_score > 0.6:
                    opportunity = {
                        "asset": impact.asset,
                        "currency": impact.currency_movement.base_currency,
                        "current_impact": impact.estimated_impact,
                        "potential_hedge_benefit": impact.estimated_impact * 0.8,  # Assume 80% hedge effectiveness
                        "hedging_cost": abs(impact.estimated_impact) * 0.1,  # Assume 10% cost
                        "confidence": impact.confidence_score,
                        "recommendation": f"Hedge {impact.asset} {impact.currency_movement.base_currency} exposure"
                    }
                    opportunities.append(opportunity)
            
            # Sort by potential benefit
            opportunities.sort(key=lambda x: x["potential_hedge_benefit"], reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying hedging opportunities: {str(e)}")
            return []
    
    def _generate_portfolio_scenarios(self, impacts: List[CurrencyImpact], 
                                    weights: np.ndarray) -> Dict[str, float]:
        """Generate portfolio-level scenarios"""
        try:
            scenarios = {}
            
            # Base scenario
            base_impact = sum(impact.estimated_impact * w 
                            for impact, w in zip(impacts, weights[:len(impacts)]))
            scenarios["base_case"] = base_impact
            
            # Extreme FX movements
            for impact in impacts:
                currency = impact.currency_movement.base_currency
                if currency not in [s.replace("base_", "") for s in scenarios.keys()]:
                    # Add scenario for this currency's extreme movement
                    scenario_impact = sum(i.estimated_impact * w for i, w in zip(impacts, weights[:len(impacts)]))
                    
                    if currency == impact.currency_movement.base_currency:
                        scenario_impact = sum(
                            i.scenarios.get("fx_appreciation_10pct", 0) * w
                            for i, w in zip(impacts, weights[:len(impacts)])
                        )
                    
                    scenarios[f"base_{currency}_shock"] = scenario_impact
            
            # Volatility stress scenario
            vol_stress_impact = sum(
                abs(impact.estimated_impact) * w * 2  # Double the impact for volatility stress
                for impact, w in zip(impacts, weights[:len(impacts)])
            )
            scenarios["volatility_stress"] = vol_stress_impact
            
            # Perfect hedge scenario
            perfect_hedge_impact = sum(
                impact.estimated_impact * (1 - impact.hedging_effectiveness) * w
                for impact, w in zip(impacts, weights[:len(impacts)])
            )
            scenarios["perfect_hedge"] = perfect_hedge_impact
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating portfolio scenarios: {str(e)}")
            return {}
    
    def _generate_currency_recommendations(self, total_fx_exposure: float, 
                                         net_positions: Dict[str, float], 
                                         fx_risk_metrics: Dict[str, Any], 
                                         hedging_opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate currency risk management recommendations"""
        try:
            recommendations = []
            
            # Exposure level recommendations
            if total_fx_exposure > 0.5:
                recommendations.append("High FX exposure detected - consider hedging strategies")
            elif total_fx_exposure > 0.2:
                recommendations.append("Moderate FX exposure - monitor currency movements closely")
            else:
                recommendations.append("Low FX exposure - current position manageable")
            
            # Concentration recommendations
            if net_positions:
                max_position = max(abs(pos) for pos in net_positions.values()) if net_positions else 0
                if max_position > 0.3:
                    recommendations.append("High currency concentration risk - diversify FX exposures")
            
            # Risk metric recommendations
            portfolio_var = fx_risk_metrics.get("portfolio_var_95", 0)
            if portfolio_var > 0.05:
                recommendations.append(f"High FX VaR ({portfolio_var:.1%}) - implement risk limits")
            
            concentration_risk = fx_risk_metrics.get("concentration_risk", 0)
            if concentration_risk > 0.5:
                recommendations.append("High currency concentration in risky assets - rebalance")
            
            # Hedging opportunity recommendations
            if hedging_opportunities:
                top_opportunity = hedging_opportunities[0]
                recommendations.append(f"Hedge {top_opportunity['asset']} for potential {top_opportunity['potential_hedge_benefit']:.1%} benefit")
            
            # Diversification recommendations
            diversification_benefit = fx_risk_metrics.get("diversification_benefit", 0)
            if diversification_benefit < 0.2:
                recommendations.append("Low FX diversification benefit - consider more currency-neutral strategies")
            
            # General recommendations
            if total_fx_exposure > 0.3 and not hedging_opportunities:
                recommendations.append("Implement systematic FX hedging for large exposures")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating currency recommendations: {str(e)}")
            return ["Unable to generate specific currency recommendations"]
    
    async def monitor_currency_alerts(self, assets: List[str], 
                                    thresholds: Optional[Dict[str, float]] = None) -> Dict[str, List[str]]:
        """Monitor for significant currency impact alerts"""
        try:
            if thresholds is None:
                thresholds = {
                    "impact_threshold": 0.03,  # 3%
                    "var_threshold": 0.05,     # 5%
                    "exposure_threshold": 0.3   # 30%
                }
            
            alerts = {}
            
            # Analyze portfolio
            analysis = await self.analyze_portfolio_currency_risk(assets)
            
            # Check exposure alerts
            if analysis.total_fx_exposure > thresholds["exposure_threshold"]:
                alerts["high_exposure"] = [f"Total FX exposure: {analysis.total_fx_exposure:.1%}"]
            
            # Check VaR alerts
            portfolio_var = analysis.fx_risk_metrics.get("portfolio_var_95", 0)
            if portfolio_var > thresholds["var_threshold"]:
                alerts["high_var"] = [f"FX VaR exceeds threshold: {portfolio_var:.1%}"]
            
            # Check individual asset alerts
            exposures = await self.analyze_currency_exposures(assets)
            fx_data = await self.fetch_fx_data(self._get_relevant_currency_pairs(exposures))
            fx_movements = await self.calculate_fx_movements(fx_data)
            impacts = await self.calculate_currency_impacts(exposures, fx_movements)
            
            for impact in impacts:
                if abs(impact.estimated_impact) > thresholds["impact_threshold"]:
                    alert_key = "high_impact_assets"
                    if alert_key not in alerts:
                        alerts[alert_key] = []
                    alerts[alert_key].append(
                        f"{impact.asset}: {impact.impact_level} impact ({impact.estimated_impact:.1%})"
                    )
            
            # Check trend alerts
            for fx_movement in fx_movements:
                if fx_movement.trend_direction in ["strong_uptrend", "strong_downtrend"]:
                    alert_key = "fx_trends"
                    if alert_key not in alerts:
                        alerts[alert_key] = []
                    alerts[alert_key].append(
                        f"{fx_movement.currency_pair}: {fx_movement.trend_direction}"
                    )
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring currency alerts: {str(e)}")
            return {}
    
    async def export_currency_analysis(self, assets: List[str], format_type: str = "json") -> str:
        """Export currency analysis to file"""
        try:
            analysis = await self.analyze_portfolio_currency_risk(assets)
            
            if format_type.lower() == "json":
                import json
                export_data = {
                    "portfolio_id": analysis.portfolio_id,
                    "timestamp": analysis.timestamp.isoformat(),
                    "total_fx_exposure": analysis.total_fx_exposure,
                    "net_currency_positions": analysis.net_currency_positions,
                    "currency_betas": analysis.currency_betas,
                    "fx_risk_metrics": analysis.fx_risk_metrics,
                    "hedging_opportunities": analysis.hedging_opportunities,
                    "impact_scenarios": analysis.impact_scenarios,
                    "recommendations": analysis.recommendations
                }
                
                filename = f"currency_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = f"/workspace/mpe/exports/{filename}"
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return filepath
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting currency analysis: {str(e)}")
            return ""

# Main execution function
async def main():
    """Main execution function for currency impact analysis"""
    engine = CurrencyImpactEngine()
    
    # Test with international portfolio
    test_assets = [
        "AAPL", "MSFT", "SPY", "VEA", "VWO", "EEM", "GLD", "TLT", "IEFA"
    ]
    
    logger.info("Starting Currency Impact Engine analysis...")
    
    # Test comprehensive analysis
    logger.info(f"\n=== Portfolio Currency Analysis ===")
    
    analysis = await engine.analyze_portfolio_currency_risk(test_assets)
    
    logger.info(f"Total FX Exposure: {analysis.total_fx_exposure:.1%}")
    logger.info(f"Portfolio VaR 95%: {analysis.fx_risk_metrics.get('portfolio_var_95', 0):.2%}")
    logger.info(f"Concentration Risk: {analysis.fx_risk_metrics.get('concentration_risk', 0):.1%}")
    logger.info(f"Hedging Opportunities: {len(analysis.hedging_opportunities)}")
    
    # Show currency positions
    logger.info("\nNet Currency Positions:")
    for currency, position in analysis.net_currency_positions.items():
        logger.info(f"  {currency}: {position:.1%}")
    
    # Show top hedging opportunities
    if analysis.hedging_opportunities:
        logger.info("\nTop Hedging Opportunities:")
        for opp in analysis.hedging_opportunities[:3]:
            logger.info(f"  {opp['asset']}: Potential benefit {opp['potential_hedge_benefit']:.1%}")
    
    # Show recommendations
    logger.info("\nRecommendations:")
    for rec in analysis.recommendations[:5]:
        logger.info(f"  - {rec}")
    
    # Test monitoring alerts
    logger.info("\n=== Currency Alerts ===")
    alerts = await engine.monitor_currency_alerts(test_assets)
    
    for alert_type, alert_messages in alerts.items():
        logger.info(f"{alert_type.upper()}:")
        for message in alert_messages:
            logger.info(f"  - {message}")
    
    logger.info("Currency Impact Engine analysis completed successfully")

if __name__ == "__main__":
    asyncio.run(main())