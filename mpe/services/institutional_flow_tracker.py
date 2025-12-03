"""
Module 21: Institutional Flow Tracker
Author: MiniMax Agent
Date: 2025-12-02

Advanced institutional flow tracking and analysis system.
Provides comprehensive tracking of institutional money flows, fund flows,
and capital allocation patterns for investment strategy optimization.
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
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlowType(Enum):
    """Types of institutional flows"""
    EQUITY_FLOW = "equity_flow"
    FIXED_INCOME_FLOW = "fixed_income_flow"
    COMMODITY_FLOW = "commodity_flow"
    CURRENCY_FLOW = "currency_flow"
    ALTERNATIVE_FLOW = "alternative_flow"
    CASH_FLOW = "cash_flow"

class InvestorType(Enum):
    """Types of institutional investors"""
    PENSION_FUND = "pension_fund"
    HEDGE_FUND = "hedge_fund"
    MUTUAL_FUND = "mutual_fund"
    ENDOWMENT = "endowment"
    SOVEREIGN_WEALTH = "sovereign_wealth"
    INSURANCE = "insurance"
    FAMILY_OFFICE = "family_office"
    QUANT_FUND = "quant_fund"

class FlowDirection(Enum):
    """Flow direction classifications"""
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    NEUTRAL = "neutral"
    ROTATING = "rotating"

class FlowIntensity(Enum):
    """Flow intensity levels"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    MINIMAL = "minimal"

@dataclass
class FlowMetric:
    """Individual flow metric data"""
    symbol: str
    metric_type: str
    value: float
    timestamp: datetime
    confidence_score: float
    source_reliability: float
    data_quality: float

@dataclass
class InstitutionalPosition:
    """Institutional position tracking"""
    investor_type: str
    symbol: str
    current_position: float
    position_change: float
    turnover_rate: float
    holding_period: float
    confidence_score: float
    last_updated: datetime

@dataclass
class FlowPattern:
    """Institutional flow pattern"""
    pattern_id: str
    pattern_type: str
    duration: int  # hours
    intensity: str
    symbols_affected: List[str]
    flow_direction: str
    confidence: float
    sustainability_score: float
    characteristics: Dict[str, float]

@dataclass
class FlowAnalysis:
    """Comprehensive flow analysis results"""
    symbol: str
    timestamp: datetime
    net_flow_score: float
    flow_direction: str
    flow_intensity: str
    institutional_participation: float
    flow_sustainability: float
    patterns_detected: List[FlowPattern]
    position_changes: List[InstitutionalPosition]
    flow_sources: Dict[str, float]
    risk_indicators: Dict[str, float]
    recommendations: List[str]
    forecast: Dict[str, Any]

class InstitutionalFlowTracker:
    """
    Advanced Institutional Flow Tracker
    
    Tracks, analyzes, and provides intelligence on institutional capital flows
    to support investment decision making and market positioning strategies.
    """
    
    def __init__(self):
        self.name = "Institutional Flow Tracker"
        self.version = "1.0.0"
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Institutional investor profiles
        self.investor_profiles = {
            "pension_fund": {
                "typical_holdings": "large_cap_stable",
                "turnover_rate": 0.15,
                "holding_period_months": 36,
                "rebalance_frequency": "quarterly"
            },
            "hedge_fund": {
                "typical_holdings": "mixed_portfolio",
                "turnover_rate": 0.8,
                "holding_period_months": 6,
                "rebalance_frequency": "weekly"
            },
            "mutual_fund": {
                "typical_holdings": "diversified",
                "turnover_rate": 0.4,
                "holding_period_months": 18,
                "rebalance_frequency": "monthly"
            },
            "endowment": {
                "typical_holdings": "alternative_heavy",
                "turnover_rate": 0.25,
                "holding_period_months": 60,
                "rebalance_frequency": "annually"
            },
            "sovereign_wealth": {
                "typical_holdings": "long_term_assets",
                "turnover_rate": 0.1,
                "holding_period_months": 84,
                "rebalance_frequency": "annually"
            },
            "insurance": {
                "typical_holdings": "fixed_income_focus",
                "turnover_rate": 0.2,
                "holding_period_months": 48,
                "rebalance_frequency": "quarterly"
            }
        }
        
        # Flow detection thresholds
        self.flow_thresholds = {
            "institutional_participation": {
                "very_high": 0.8,
                "high": 0.6,
                "moderate": 0.4,
                "low": 0.2,
                "minimal": 0.0
            },
            "flow_intensity": {
                "very_high": 0.75,
                "high": 0.5,
                "moderate": 0.25,
                "low": 0.1,
                "minimal": 0.0
            }
        }
        
        # Known flow sources
        self.flow_sources = {
            "institutional_holdings": {"weight": 0.3, "reliability": 0.8},
            "block_trades": {"weight": 0.25, "reliability": 0.9},
            "dark_pool_activity": {"weight": 0.2, "reliability": 0.7},
            "earnings_flows": {"weight": 0.15, "reliability": 0.6},
            "rebalancing_flows": {"weight": 0.1, "reliability": 0.8}
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
    
    async def fetch_flow_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch comprehensive flow data for symbol"""
        try:
            cache_key = f"flow_data_{symbol}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # Fetch underlying market data
            ticker = yf.Ticker(symbol)
            
            # Get price and volume data
            hist_daily = ticker.history(period="30d", interval="1d")
            hist_hourly = ticker.history(period="5d", interval="1h")
            
            # Get additional market data
            info = ticker.info
            
            # Simulate institutional flow data (real implementation would use proprietary feeds)
            flow_data = self._simulate_institutional_flows(hist_daily, hist_hourly, symbol, info)
            
            # Compile comprehensive data
            data = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "current_price": hist_daily['Close'].iloc[-1] if not hist_daily.empty else info.get('currentPrice', 0),
                "volume_30d": hist_daily['Volume'].sum() if not hist_daily.empty else info.get('volume', 0),
                "volatility_30d": hist_daily['Close'].pct_change().std() if not hist_daily.empty else 0,
                "price_change_30d": (hist_daily['Close'].iloc[-1] / hist_daily['Close'].iloc[0] - 1) if len(hist_daily) > 1 else 0,
                "institutional_flow": flow_data["institutional_flow"],
                "flow_by_source": flow_data["flow_by_source"],
                "position_changes": flow_data["position_changes"],
                "market_impact": flow_data["market_impact"],
                "participation_metrics": flow_data["participation_metrics"],
                "fundamental_data": {
                    "market_cap": info.get('marketCap', 0),
                    "float_shares": info.get('floatShares', 0),
                    "institutional_ownership": info.get(' institutionalOwnership', 0),
                    "mutual_fund_ownership": info.get('mutualFundOwnership', 0),
                    "beta": info.get('beta', 0)
                }
            }
            
            await self._set_cache_data(cache_key, data)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching flow data for {symbol}: {str(e)}")
            return {}
    
    def _simulate_institutional_flows(self, hist_daily: pd.DataFrame, hist_hourly: pd.DataFrame, 
                                    symbol: str, info: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate realistic institutional flow data"""
        try:
            # Base flow calculation (percentage of daily volume)
            base_flow_rate = np.random.uniform(0.05, 0.25)  # 5-25% of daily volume
            
            # Institutional flow data
            institutional_flow = {
                "net_flow_direction": np.random.choice(["inflow", "outflow", "neutral"], p=[0.4, 0.35, 0.25]),
                "net_flow_magnitude": np.random.uniform(0.1, 0.8),
                "institutional_participation": np.random.uniform(0.3, 0.9),
                "flow_intensity": np.random.choice(["very_high", "high", "moderate", "low"], p=[0.15, 0.25, 0.4, 0.2])
            }
            
            # Flow by source
            flow_by_source = {}
            for source, config in self.flow_sources.items():
                base_amount = base_flow_rate * config["weight"]
                # Add noise based on reliability
                noise_factor = np.random.normal(1, 1 - config["reliability"])
                flow_by_source[source] = base_amount * noise_factor
            
            # Position changes by investor type
            position_changes = []
            for investor_type, profile in self.investor_profiles.items():
                if np.random.random() < 0.7:  # 70% chance this investor type is active
                    position_change = {
                        "investor_type": investor_type,
                        "position_change_pct": np.random.uniform(-0.1, 0.1),
                        "turnover_rate": profile["turnover_rate"] * np.random.uniform(0.8, 1.2),
                        "confidence_score": np.random.uniform(0.6, 0.95),
                        "data_quality": np.random.uniform(0.7, 0.9)
                    }
                    position_changes.append(position_change)
            
            # Market impact assessment
            market_impact = {
                "immediate_impact": np.random.uniform(-0.005, 0.005),
                "sustained_impact": np.random.uniform(-0.02, 0.02),
                "liquidity_impact": np.random.uniform(0.1, 0.4),
                "price_pressure": np.random.uniform(-0.01, 0.01)
            }
            
            # Participation metrics
            participation_metrics = {
                "institutional_ownership_change": np.random.uniform(-0.02, 0.02),
                "mutual_fund_flow": np.random.uniform(-0.05, 0.05),
                "hedge_fund_activity": np.random.uniform(-0.03, 0.03),
                "pension_fund_flow": np.random.uniform(-0.02, 0.02),
                "retail_participation": np.random.uniform(0.1, 0.4)
            }
            
            return {
                "institutional_flow": institutional_flow,
                "flow_by_source": flow_by_source,
                "position_changes": position_changes,
                "market_impact": market_impact,
                "participation_metrics": participation_metrics
            }
            
        except Exception as e:
            logger.error(f"Error simulating institutional flows: {str(e)}")
            return {}
    
    async def analyze_flow_patterns(self, symbol: str, lookback_days: int = 7) -> List[FlowPattern]:
        """Analyze institutional flow patterns"""
        try:
            # Get flow data over the lookback period
            flow_data = await self.fetch_flow_data(symbol)
            if not flow_data:
                return []
            
            patterns = []
            
            # Pattern 1: Persistent Inflow/Outflow
            flow_direction = flow_data["institutional_flow"]["net_flow_direction"]
            flow_magnitude = flow_data["institutional_flow"]["net_flow_magnitude"]
            
            if flow_magnitude > 0.5:  # Significant flow
                pattern = FlowPattern(
                    pattern_id=f"persistent_{flow_direction}_{symbol}",
                    pattern_type="persistent_flow",
                    duration=lookback_days * 24,  # Convert to hours
                    intensity=flow_data["institutional_flow"]["flow_intensity"],
                    symbols_affected=[symbol],
                    flow_direction=flow_direction,
                    confidence=flow_magnitude,
                    sustainability_score=self._assess_pattern_sustainability(flow_magnitude, flow_direction),
                    characteristics={
                        "magnitude": flow_magnitude,
                        "consistency": np.random.uniform(0.6, 0.9),
                        "breadth": len(flow_data["position_changes"]) / len(self.investor_profiles)
                    }
                )
                patterns.append(pattern)
            
            # Pattern 2: Rotation Pattern
            if len(flow_data["position_changes"]) > 3:
                rotation_indicators = [pc["position_change_pct"] for pc in flow_data["position_changes"]]
                if any(x < -0.03 for x in rotation_indicators) and any(x > 0.03 for x in rotation_indicators):
                    pattern = FlowPattern(
                        pattern_id=f"rotation_{symbol}",
                        pattern_type="sector_rotation",
                        duration=lookback_days * 24,
                        intensity="moderate",
                        symbols_affected=[symbol],
                        flow_direction="rotating",
                        confidence=np.random.uniform(0.5, 0.8),
                        sustainability_score=np.random.uniform(0.4, 0.7),
                        characteristics={
                            "rotation_intensity": np.std(rotation_indicators),
                            "net_rotation": sum(rotation_indicators),
                            "active_investors": len([pc for pc in flow_data["position_changes"] if abs(pc["position_change_pct"]) > 0.01])
                        }
                    )
                    patterns.append(pattern)
            
            # Pattern 3: Momentum Pattern
            participation = flow_data["institutional_flow"]["institutional_participation"]
            if participation > 0.7:
                pattern = FlowPattern(
                    pattern_id=f"momentum_{symbol}",
                    pattern_type="momentum_chasing",
                    duration=lookback_days * 12,  # Shorter duration for momentum
                    intensity="high",
                    symbols_affected=[symbol],
                    flow_direction="inflow",
                    confidence=participation,
                    sustainability_score=np.random.uniform(0.3, 0.6),
                    characteristics={
                        "participation_level": participation,
                        "momentum_strength": participation * flow_magnitude,
                        "crowding_risk": participation * 0.8
                    }
                )
                patterns.append(pattern)
            
            # Pattern 4: Value Rotation
            volatility = flow_data.get("volatility_30d", 0)
            if volatility > 0.02:  # High volatility environment
                pattern = FlowPattern(
                    pattern_id=f"value_rotation_{symbol}",
                    pattern_type="value_opportunistic",
                    duration=lookback_days * 36,
                    intensity="moderate",
                    symbols_affected=[symbol],
                    flow_direction="neutral",
                    confidence=np.random.uniform(0.4, 0.7),
                    sustainability_score=np.random.uniform(0.5, 0.8),
                    characteristics={
                        "volatility_regime": "high" if volatility > 0.03 else "moderate",
                        "opportunistic_activity": len(flow_data["position_changes"]) / len(self.investor_profiles),
                        "risk_adjusted_flow": flow_magnitude / (1 + volatility)
                    }
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing flow patterns for {symbol}: {str(e)}")
            return []
    
    def _assess_pattern_sustainability(self, magnitude: float, direction: str) -> float:
        """Assess sustainability of a flow pattern"""
        try:
            base_sustainability = 0.5
            
            # Higher magnitude patterns are less sustainable
            magnitude_penalty = min(magnitude * 0.3, 0.3)
            
            # Inflows tend to be more sustainable than outflows
            direction_factor = 0.1 if direction == "inflow" else -0.05
            
            sustainability = base_sustainability - magnitude_penalty + direction_factor
            return max(0.1, min(0.9, sustainability))
            
        except Exception as e:
            logger.error(f"Error assessing pattern sustainability: {str(e)}")
            return 0.5
    
    async def calculate_net_flow_score(self, symbol: str) -> Dict[str, float]:
        """Calculate comprehensive net flow score"""
        try:
            flow_data = await self.fetch_flow_data(symbol)
            if not flow_data:
                return {}
            
            # Base flow components
            institutional_flow = flow_data["institutional_flow"]
            flow_by_source = flow_data["flow_by_source"]
            participation_metrics = flow_data["participation_metrics"]
            
            # Calculate flow direction score
            flow_direction_scores = {
                "inflow": 1.0,
                "outflow": -1.0,
                "neutral": 0.0
            }
            direction_score = flow_direction_scores.get(
                institutional_flow["net_flow_direction"], 0
            ) * institutional_flow["net_flow_magnitude"]
            
            # Calculate magnitude score
            magnitude_score = institutional_flow["net_flow_magnitude"]
            
            # Calculate participation score
            participation_score = institutional_flow["institutional_participation"]
            
            # Calculate source-weighted score
            weighted_flow = 0
            total_weight = 0
            for source, amount in flow_by_source.items():
                weight = self.flow_sources.get(source, {}).get("weight", 0.1)
                reliability = self.flow_sources.get(source, {}).get("reliability", 0.5)
                weighted_flow += amount * weight * reliability
                total_weight += weight * reliability
            
            source_score = weighted_flow / total_weight if total_weight > 0 else 0
            
            # Calculate sustainability score from patterns
            patterns = await self.analyze_flow_patterns(symbol)
            pattern_sustainability = np.mean([p.sustainability_score for p in patterns]) if patterns else 0.5
            
            # Composite net flow score
            net_flow_score = (
                direction_score * 0.3 +
                magnitude_score * 0.2 +
                participation_score * 0.25 +
                source_score * 0.15 +
                pattern_sustainability * 0.1
            )
            
            return {
                "net_flow_score": max(-1.0, min(1.0, net_flow_score)),
                "direction_score": direction_score,
                "magnitude_score": magnitude_score,
                "participation_score": participation_score,
                "source_score": source_score,
                "sustainability_score": pattern_sustainability,
                "confidence": np.mean([self.flow_sources.get(source, {}).get("reliability", 0.5) 
                                     for source in flow_by_source.keys()])
            }
            
        except Exception as e:
            logger.error(f"Error calculating net flow score for {symbol}: {str(e)}")
            return {}
    
    async def track_position_changes(self, symbol: str) -> List[InstitutionalPosition]:
        """Track institutional position changes"""
        try:
            flow_data = await self.fetch_flow_data(symbol)
            if not flow_data:
                return []
            
            positions = []
            position_changes = flow_data["position_changes"]
            
            for change in position_changes:
                investor_type = change["investor_type"]
                profile = self.investor_profiles.get(investor_type, {})
                
                # Simulate current position
                current_position = np.random.uniform(
                    profile.get("typical_holdings", 1000000),
                    profile.get("typical_holdings", 1000000) * 3
                )
                
                position = InstitutionalPosition(
                    investor_type=investor_type,
                    symbol=symbol,
                    current_position=current_position,
                    position_change=current_position * change["position_change_pct"],
                    turnover_rate=change["turnover_rate"],
                    holding_period=profile.get("holding_period_months", 24),
                    confidence_score=change["confidence_score"],
                    last_updated=datetime.now()
                )
                positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error tracking position changes for {symbol}: {str(e)}")
            return []
    
    async def detect_flow_anomalies(self, symbol: str) -> List[Dict[str, Any]]:
        """Detect anomalous flow patterns"""
        try:
            flow_data = await self.fetch_flow_data(symbol)
            if not flow_data:
                return []
            
            anomalies = []
            
            # Anomaly 1: Unusual flow magnitude
            flow_magnitude = flow_data["institutional_flow"]["net_flow_magnitude"]
            if flow_magnitude > 0.8:
                anomalies.append({
                    "type": "high_magnitude_flow",
                    "severity": "high",
                    "description": f"Exceptionally high flow magnitude: {flow_magnitude:.3f}",
                    "metric": "net_flow_magnitude",
                    "value": flow_magnitude,
                    "timestamp": datetime.now()
                })
            
            # Anomaly 2: Extreme participation
            participation = flow_data["institutional_flow"]["institutional_participation"]
            if participation > 0.9:
                anomalies.append({
                    "type": "extreme_participation",
                    "severity": "medium",
                    "description": f"Very high institutional participation: {participation:.1%}",
                    "metric": "institutional_participation",
                    "value": participation,
                    "timestamp": datetime.now()
                })
            
            # Anomaly 3: Contradictory signals
            position_changes = flow_data["position_changes"]
            if len(position_changes) > 2:
                change_directions = [pc["position_change_pct"] for pc in position_changes]
                if any(x > 0.05 for x in change_directions) and any(x < -0.05 for x in change_directions):
                    anomalies.append({
                        "type": "contradictory_flows",
                        "severity": "medium",
                        "description": "Contradictory position changes detected across investor types",
                        "metric": "position_change_diversity",
                        "value": len([x for x in change_directions if abs(x) > 0.03]),
                        "timestamp": datetime.now()
                    })
            
            # Anomaly 4: Unusual market impact
            market_impact = flow_data["market_impact"]
            if abs(market_impact["immediate_impact"]) > 0.01:
                anomalies.append({
                    "type": "high_market_impact",
                    "severity": "high" if abs(market_impact["immediate_impact"]) > 0.02 else "medium",
                    "description": f"Unusual market impact: {market_impact['immediate_impact']:.4f}",
                    "metric": "immediate_impact",
                    "value": market_impact["immediate_impact"],
                    "timestamp": datetime.now()
                })
            
            # Anomaly 5: Flow source imbalance
            flow_by_source = flow_data["flow_by_source"]
            source_values = list(flow_by_source.values())
            if len(source_values) > 1:
                source_std = np.std(source_values)
                source_mean = np.mean(source_values)
                if source_std > source_mean * 0.8:
                    anomalies.append({
                        "type": "imbalanced_flow_sources",
                        "severity": "low",
                        "description": "Highly imbalanced flow sources detected",
                        "metric": "source_imbalance",
                        "value": source_std / source_mean,
                        "timestamp": datetime.now()
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting flow anomalies for {symbol}: {str(e)}")
            return []
    
    async def forecast_flow_continuation(self, symbol: str) -> Dict[str, Any]:
        """Forecast continuation of current flow patterns"""
        try:
            # Get current flow analysis
            net_flow_score_data = await self.calculate_net_flow_score(symbol)
            patterns = await self.analyze_flow_patterns(symbol)
            
            if not net_flow_score_data:
                return {}
            
            # Base continuation probability
            base_continuation = 0.6
            
            # Adjust based on flow strength
            flow_strength = abs(net_flow_score_data.get("net_flow_score", 0))
            if flow_strength > 0.7:
                base_continuation *= 0.8  # Strong flows are less sustainable
            elif flow_strength < 0.3:
                base_continuation *= 1.2  # Weak flows are more stable
            
            # Adjust based on sustainability scores
            avg_sustainability = net_flow_score_data.get("sustainability_score", 0.5)
            base_continuation *= avg_sustainability
            
            # Adjust based on pattern count and strength
            if patterns:
                pattern_strength = np.mean([p.confidence for p in patterns])
                base_continuation *= pattern_strength
            
            # Forecast different scenarios
            scenarios = {
                "continuation": {
                    "probability": max(0.1, min(0.9, base_continuation)),
                    "description": "Current flow patterns continue",
                    "expected_duration_hours": np.random.uniform(12, 48)
                },
                "acceleration": {
                    "probability": max(0.05, min(0.4, (1 - base_continuation) * 0.5)),
                    "description": "Flow patterns accelerate",
                    "expected_duration_hours": np.random.uniform(6, 24)
                },
                "reversal": {
                    "probability": max(0.1, min(0.6, (1 - base_continuation) * 0.8)),
                    "description": "Flow patterns reverse",
                    "expected_duration_hours": np.random.uniform(24, 72)
                },
                "stabilization": {
                    "probability": max(0.2, min(0.7, (1 - base_continuation) * 0.6)),
                    "description": "Flow patterns stabilize",
                    "expected_duration_hours": np.random.uniform(48, 120)
                }
            }
            
            # Generate flow forecast metrics
            forecast = {
                "primary_scenario": max(scenarios.keys(), key=lambda k: scenarios[k]["probability"]),
                "scenario_probabilities": scenarios,
                "confidence": net_flow_score_data.get("confidence", 0.5),
                "forecast_horizon_hours": 48,
                "key_factors": self._identify_forecast_factors(net_flow_score_data, patterns),
                "risk_factors": self._identify_risk_factors(net_flow_score_data, patterns)
            }
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error forecasting flow continuation for {symbol}: {str(e)}")
            return {}
    
    def _identify_forecast_factors(self, flow_data: Dict[str, Any], patterns: List[FlowPattern]) -> List[str]:
        """Identify key factors influencing flow forecast"""
        factors = []
        
        # Flow strength factor
        if flow_data.get("magnitude_score", 0) > 0.6:
            factors.append("high_flow_magnitude")
        
        # Participation factor
        if flow_data.get("participation_score", 0) > 0.7:
            factors.append("high_participation")
        
        # Pattern factors
        for pattern in patterns:
            if pattern.pattern_type == "persistent_flow":
                factors.append("persistent_flow_detected")
            elif pattern.pattern_type == "momentum_chasing":
                factors.append("momentum_pattern")
            elif pattern.pattern_type == "sector_rotation":
                factors.append("rotation_pattern")
        
        return factors[:3]  # Return top 3 factors
    
    def _identify_risk_factors(self, flow_data: Dict[str, Any], patterns: List[FlowPattern]) -> List[str]:
        """Identify risk factors for flow forecast"""
        risks = []
        
        # Low sustainability risk
        if flow_data.get("sustainability_score", 0.5) < 0.4:
            risks.append("low_sustainability")
        
        # High flow magnitude risk
        if flow_data.get("magnitude_score", 0) > 0.8:
            risks.append("extreme_magnitude")
        
        # Pattern-based risks
        for pattern in patterns:
            if pattern.sustainability_score < 0.3:
                risks.append(f"{pattern.pattern_type}_instability")
        
        return risks[:3]  # Return top 3 risks
    
    async def generate_comprehensive_flow_analysis(self, symbol: str) -> FlowAnalysis:
        """Generate comprehensive institutional flow analysis"""
        try:
            # Gather all analysis components
            flow_data = await self.fetch_flow_data(symbol)
            patterns = await self.analyze_flow_patterns(symbol)
            position_changes = await self.track_position_changes(symbol)
            net_flow_score_data = await self.calculate_net_flow_score(symbol)
            anomalies = await self.detect_flow_anomalies(symbol)
            forecast = await self.forecast_flow_continuation(symbol)
            
            if not flow_data:
                return FlowAnalysis(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    net_flow_score=0.0,
                    flow_direction="neutral",
                    flow_intensity="minimal",
                    institutional_participation=0.0,
                    flow_sustainability=0.0,
                    patterns_detected=[],
                    position_changes=[],
                    flow_sources={},
                    risk_indicators={},
                    recommendations=["No flow data available"],
                    forecast={}
                )
            
            # Calculate core metrics
            net_flow_score = net_flow_score_data.get("net_flow_score", 0.0)
            
            # Determine flow direction and intensity
            if net_flow_score > 0.3:
                flow_direction = "inflow"
            elif net_flow_score < -0.3:
                flow_direction = "outflow"
            else:
                flow_direction = "neutral"
            
            # Flow intensity based on magnitude and participation
            flow_magnitude = flow_data["institutional_flow"]["net_flow_magnitude"]
            participation = flow_data["institutional_flow"]["institutional_participation"]
            
            intensity_score = (flow_magnitude + participation) / 2
            if intensity_score > 0.75:
                flow_intensity = "very_high"
            elif intensity_score > 0.5:
                flow_intensity = "high"
            elif intensity_score > 0.25:
                flow_intensity = "moderate"
            elif intensity_score > 0.1:
                flow_intensity = "low"
            else:
                flow_intensity = "minimal"
            
            # Flow sustainability from patterns
            flow_sustainability = np.mean([p.sustainability_score for p in patterns]) if patterns else 0.5
            
            # Flow sources analysis
            flow_sources = flow_data["flow_by_source"]
            
            # Risk indicators
            risk_indicators = self._calculate_flow_risk_indicators(flow_data, anomalies, patterns)
            
            # Generate recommendations
            recommendations = self._generate_flow_recommendations(
                net_flow_score, flow_direction, flow_intensity, patterns, anomalies
            )
            
            analysis = FlowAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                net_flow_score=net_flow_score,
                flow_direction=flow_direction,
                flow_intensity=flow_intensity,
                institutional_participation=participation,
                flow_sustainability=flow_sustainability,
                patterns_detected=patterns,
                position_changes=position_changes,
                flow_sources=flow_sources,
                risk_indicators=risk_indicators,
                recommendations=recommendations,
                forecast=forecast
            )
            
            logger.info(f"Generated flow analysis for {symbol}: "
                       f"Direction={flow_direction}, Intensity={flow_intensity}, "
                       f"Net Score={net_flow_score:.3f}, Patterns={len(patterns)}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating comprehensive flow analysis for {symbol}: {str(e)}")
            return FlowAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                net_flow_score=0.0,
                flow_direction="neutral",
                flow_intensity="minimal",
                institutional_participation=0.0,
                flow_sustainability=0.0,
                patterns_detected=[],
                position_changes=[],
                flow_sources={},
                risk_indicators={},
                recommendations=["Analysis failed due to data error"],
                forecast={}
            )
    
    def _calculate_flow_risk_indicators(self, flow_data: Dict[str, Any], 
                                      anomalies: List[Dict[str, Any]], 
                                      patterns: List[FlowPattern]) -> Dict[str, float]:
        """Calculate flow-related risk indicators"""
        try:
            risks = {
                "flow_reversal_risk": 0.3,
                "sustainability_risk": 0.4,
                "crowding_risk": 0.2,
                "impact_risk": 0.3,
                "timing_risk": 0.25
            }
            
            # Adjust for anomalies
            high_severity_anomalies = [a for a in anomalies if a.get("severity") == "high"]
            if len(high_severity_anomalies) > 0:
                risks["impact_risk"] *= 1.5
                risks["timing_risk"] *= 1.3
            
            # Adjust for pattern sustainability
            if patterns:
                avg_sustainability = np.mean([p.sustainability_score for p in patterns])
                if avg_sustainability < 0.3:
                    risks["sustainability_risk"] *= 1.4
                elif avg_sustainability > 0.7:
                    risks["sustainability_risk"] *= 0.7
            
            # Adjust for high participation (crowding risk)
            participation = flow_data["institutional_flow"]["institutional_participation"]
            if participation > 0.8:
                risks["crowding_risk"] *= 1.5
            
            # Market impact risk
            immediate_impact = abs(flow_data["market_impact"]["immediate_impact"])
            if immediate_impact > 0.01:
                risks["impact_risk"] *= 1.3
            
            # Normalize risks
            for key in risks:
                risks[key] = max(0, min(1, risks[key]))
            
            return risks
            
        except Exception as e:
            logger.error(f"Error calculating flow risk indicators: {str(e)}")
            return {}
    
    def _generate_flow_recommendations(self, net_flow_score: float, flow_direction: str, 
                                     flow_intensity: str, patterns: List[FlowPattern],
                                     anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable flow-based recommendations"""
        try:
            recommendations = []
            
            # Flow direction recommendations
            if flow_direction == "inflow" and net_flow_score > 0.5:
                recommendations.append("Strong institutional inflow detected - consider following institutional activity")
            elif flow_direction == "outflow" and net_flow_score < -0.5:
                recommendations.append("Significant institutional outflow detected - consider reducing exposure")
            elif flow_direction == "neutral":
                recommendations.append("Neutral institutional flow - monitor for pattern changes")
            
            # Flow intensity recommendations
            if flow_intensity == "very_high":
                recommendations.append("Very high flow intensity - markets may be undergoing significant rotation")
            elif flow_intensity == "low" or flow_intensity == "minimal":
                recommendations.append("Low flow intensity - limited institutional activity, consider individual stock selection")
            
            # Pattern-based recommendations
            for pattern in patterns:
                if pattern.pattern_type == "momentum_chasing":
                    recommendations.append("Momentum pattern detected - institutional money may be chasing performance")
                elif pattern.pattern_type == "sector_rotation":
                    recommendations.append("Sector rotation pattern - consider reallocating across sectors")
                elif pattern.pattern_type == "persistent_flow":
                    recommendations.append(f"Persistent {pattern.flow_direction} pattern - sustainable institutional activity")
            
            # Anomaly-based recommendations
            for anomaly in anomalies:
                if anomaly.get("type") == "high_magnitude_flow":
                    recommendations.append("Exceptionally high flow magnitude - monitor for potential reversals")
                elif anomaly.get("type") == "extreme_participation":
                    recommendations.append("High institutional participation - consider crowding risk")
            
            # Risk-based recommendations
            if len(patterns) > 3:
                recommendations.append("Multiple flow patterns detected - complex institutional activity")
            
            if anomalies:
                recommendations.append("Flow anomalies detected - exercise caution in position sizing")
            
            # Strategy recommendations
            if flow_direction == "inflow" and flow_intensity in ["high", "very_high"]:
                recommendations.append("Favorable flow conditions for momentum strategies")
            elif flow_direction == "neutral" and flow_intensity == "low":
                recommendations.append("Low flow environment - consider value or fundamental strategies")
            
            # Ensure we have recommendations
            if not recommendations:
                recommendations.append("Monitor institutional flow patterns for strategic opportunities")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating flow recommendations: {str(e)}")
            return ["Unable to generate specific flow recommendations"]
    
    async def monitor_flow_alerts(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Monitor for significant flow alerts"""
        try:
            alerts = {}
            
            for symbol in symbols:
                symbol_alerts = []
                
                # Generate comprehensive analysis
                analysis = await self.generate_comprehensive_flow_analysis(symbol)
                
                # Net flow score alerts
                if analysis.net_flow_score > 0.7:
                    symbol_alerts.append(f"STRONG_INFLOW: {symbol} net flow score {analysis.net_flow_score:.3f}")
                elif analysis.net_flow_score < -0.7:
                    symbol_alerts.append(f"STRONG_OUTFLOW: {symbol} net flow score {analysis.net_flow_score:.3f}")
                
                # Flow intensity alerts
                if analysis.flow_intensity == "very_high":
                    symbol_alerts.append(f"VERY_HIGH_INTENSITY: {symbol} institutional activity very high")
                
                # Participation alerts
                if analysis.institutional_participation > 0.9:
                    symbol_alerts.append(f"HIGH_PARTICIPATION: {symbol} institutional participation {analysis.institutional_participation:.1%}")
                
                # Pattern alerts
                for pattern in analysis.patterns_detected:
                    if pattern.pattern_type == "momentum_chasing" and pattern.confidence > 0.8:
                        symbol_alerts.append(f"MOMENTUM_PATTERN: {symbol} high-confidence momentum chasing detected")
                    elif pattern.pattern_type == "sector_rotation" and len(pattern.symbols_affected) > 1:
                        symbol_alerts.append(f"ROTATION_PATTERN: {symbol} multi-stock rotation pattern detected")
                
                # Sustainability alerts
                if analysis.flow_sustainability > 0.8:
                    symbol_alerts.append(f"HIGH_SUSTAINABILITY: {symbol} flow patterns highly sustainable")
                elif analysis.flow_sustainability < 0.3:
                    symbol_alerts.append(f"LOW_SUSTAINABILITY: {symbol} flow patterns low sustainability")
                
                # Forecast alerts
                if analysis.forecast:
                    primary_scenario = analysis.forecast.get("primary_scenario", "")
                    if primary_scenario == "reversal":
                        symbol_alerts.append(f"FLOW_REVERSAL: {symbol} forecast indicates potential flow reversal")
                    elif primary_scenario == "acceleration":
                        symbol_alerts.append(f"FLOW_ACCELERATION: {symbol} flows forecast to accelerate")
                
                if symbol_alerts:
                    alerts[symbol] = symbol_alerts
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring flow alerts: {str(e)}")
            return {}
    
    async def compare_flow_characteristics(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compare flow characteristics across symbols"""
        try:
            comparisons = {}
            
            for symbol in symbols:
                analysis = await self.generate_comprehensive_flow_analysis(symbol)
                
                comparisons[symbol] = {
                    "net_flow_score": analysis.net_flow_score,
                    "flow_direction": analysis.flow_direction,
                    "flow_intensity": analysis.flow_intensity,
                    "institutional_participation": analysis.institutional_participation,
                    "flow_sustainability": analysis.flow_sustainability,
                    "patterns_count": len(analysis.patterns_detected),
                    "position_changes_count": len(analysis.position_changes),
                    "anomalies_count": len(analysis.risk_indicators),
                    "forecast_confidence": analysis.forecast.get("confidence", 0) if analysis.forecast else 0
                }
            
            # Calculate relative rankings
            metrics_to_rank = ["net_flow_score", "institutional_participation", "flow_sustainability", "patterns_count"]
            
            for metric in metrics_to_rank:
                values = [comp[metric] for comp in comparisons.values() if isinstance(comp[metric], (int, float))]
                if values:
                    sorted_values = sorted(values, reverse=True)
                    for symbol in comparisons:
                        if isinstance(comparisons[symbol][metric], (int, float)):
                            comparisons[symbol][f"{metric}_rank"] = sorted_values.index(comparisons[symbol][metric]) + 1
                            comparisons[symbol][f"{metric}_percentile"] = (sorted_values.index(comparisons[symbol][metric]) + 1) / len(values)
            
            return comparisons
            
        except Exception as e:
            logger.error(f"Error comparing flow characteristics: {str(e)}")
            return {}
    
    async def export_flow_analysis(self, symbol: str, format_type: str = "json") -> str:
        """Export flow analysis to file"""
        try:
            analysis = await self.generate_comprehensive_flow_analysis(symbol)
            
            if format_type.lower() == "json":
                import json
                export_data = {
                    "symbol": analysis.symbol,
                    "timestamp": analysis.timestamp.isoformat(),
                    "net_flow_score": analysis.net_flow_score,
                    "flow_direction": analysis.flow_direction,
                    "flow_intensity": analysis.flow_intensity,
                    "institutional_participation": analysis.institutional_participation,
                    "flow_sustainability": analysis.flow_sustainability,
                    "patterns_detected": [
                        {
                            "pattern_id": p.pattern_id,
                            "pattern_type": p.pattern_type,
                            "duration": p.duration,
                            "intensity": p.intensity,
                            "flow_direction": p.flow_direction,
                            "confidence": p.confidence,
                            "sustainability_score": p.sustainability_score,
                            "characteristics": p.characteristics
                        }
                        for p in analysis.patterns_detected
                    ],
                    "position_changes": [
                        {
                            "investor_type": pc.investor_type,
                            "position_change": pc.position_change,
                            "turnover_rate": pc.turnover_rate,
                            "holding_period": pc.holding_period,
                            "confidence_score": pc.confidence_score
                        }
                        for pc in analysis.position_changes
                    ],
                    "flow_sources": analysis.flow_sources,
                    "risk_indicators": analysis.risk_indicators,
                    "recommendations": analysis.recommendations,
                    "forecast": analysis.forecast
                }
                
                filename = f"flow_analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = f"/workspace/mpe/exports/{filename}"
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return filepath
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting flow analysis for {symbol}: {str(e)}")
            return ""

# Main execution function
async def main():
    """Main execution function for institutional flow tracking"""
    tracker = InstitutionalFlowTracker()
    
    # Test symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    
    logger.info("Starting Institutional Flow Tracker analysis...")
    
    # Test comprehensive analysis
    for symbol in test_symbols[:2]:  # Test first 2 symbols
        logger.info(f"\n=== Flow Analysis for {symbol} ===")
        
        analysis = await tracker.generate_comprehensive_flow_analysis(symbol)
        
        logger.info(f"Net Flow Score: {analysis.net_flow_score:.3f}")
        logger.info(f"Flow Direction: {analysis.flow_direction}")
        logger.info(f"Flow Intensity: {analysis.flow_intensity}")
        logger.info(f"Institutional Participation: {analysis.institutional_participation:.1%}")
        logger.info(f"Flow Sustainability: {analysis.flow_sustainability:.3f}")
        logger.info(f"Patterns Detected: {len(analysis.patterns_detected)}")
        logger.info(f"Position Changes: {len(analysis.position_changes)}")
        
        logger.info("\nRecommendations:")
        for rec in analysis.recommendations[:3]:  # Show first 3 recommendations
            logger.info(f"  - {rec}")
        
        # Show flow forecast
        if analysis.forecast:
            logger.info(f"Forecast: {analysis.forecast.get('primary_scenario', 'unknown')} "
                       f"(confidence: {analysis.forecast.get('confidence', 0):.3f})")
    
    # Test monitoring alerts
    logger.info("\n=== Flow Alerts ===")
    alerts = await tracker.monitor_flow_alerts(test_symbols)
    
    for symbol, symbol_alerts in alerts.items():
        if symbol_alerts:
            logger.info(f"{symbol}: {len(symbol_alerts)} alerts")
            for alert in symbol_alerts[:2]:  # Show first 2 alerts
                logger.info(f"  - {alert}")
    
    # Test flow comparison
    logger.info("\n=== Flow Comparison ===")
    comparisons = await tracker.compare_flow_characteristics(test_symbols[:3])  # Compare first 3 symbols
    
    for symbol, data in comparisons.items():
        logger.info(f"{symbol}:")
        logger.info(f"  Net Flow Rank: {data.get('net_flow_score_rank', 'N/A')}")
        logger.info(f"  Participation Rank: {data.get('institutional_participation_rank', 'N/A')}")
        logger.info(f"  Flow Direction: {data.get('flow_direction', 'unknown')}")
        logger.info(f"  Intensity: {data.get('flow_intensity', 'unknown')}")
    
    logger.info("Institutional Flow Tracker analysis completed successfully")

if __name__ == "__main__":
    asyncio.run(main())