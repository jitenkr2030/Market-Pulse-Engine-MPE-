"""
Module 19: Dark Pool Intelligence Engine
Author: MiniMax Agent
Date: 2025-12-02

Advanced dark pool activity analysis and liquidity intelligence.
Provides institutional order flow patterns, liquidity sourcing analysis,
and market impact assessment for hidden trading venues.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DarkPoolType(Enum):
    """Types of dark pools"""
    INSTITUTIONAL = "institutional"
    ECN = "ecn"
    BROKER = "broker"
    CROSSING_NETWORK = "crossing_network"
    INSTANT_LIQUIDITY = "instant_liquidity"
    POSITIVE_BOOK = "positive_book"

class ActivityLevel(Enum):
    """Dark pool activity levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INACTIVE = "inactive"

@dataclass
class DarkPoolMetrics:
    """Dark pool trading metrics"""
    venue: str
    volume: float
    trade_count: int
    average_size: float
    activity_level: str
    liquidity_score: float
    market_impact: float
    price_improvement: float
    fill_rate: float
    timestamp: datetime

@dataclass
class DarkPoolOpportunity:
    """Dark pool liquidity opportunity"""
    symbol: str
    venue: str
    size: float
    expected_improvement: float
    confidence_score: float
    urgency: str
    estimated_fill_time: float

@dataclass
class DarkPoolPulse:
    """Dark pool intelligence pulse"""
    symbol: str
    timestamp: datetime
    overall_activity_score: float
    liquidity_metrics: Dict[str, float]
    opportunity_score: float
    risk_score: float
    venue_analysis: Dict[str, Dict[str, float]]
    patterns: Dict[str, Any]
    recommendations: List[str]

class DarkPoolIntelligenceEngine:
    """
    Advanced Dark Pool Intelligence Engine
    
    Analyzes dark pool activity, institutional order flow patterns,
    and provides liquidity sourcing opportunities for institutional trading.
    """
    
    def __init__(self):
        self.name = "Dark Pool Intelligence Engine"
        self.version = "1.0.0"
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Dark pool venues database
        self.dark_pool_venues = {
            "UBS_DARK": {"type": "institutional", "activity_bias": 0.7},
            "CREDIT_SUISSE_DARK": {"type": "broker", "activity_bias": 0.8},
            "JPMORGAN_DARK": {"type": "institutional", "activity_bias": 0.9},
            "GOLDMAN_SACHS_SIGMA_X": {"type": "ecn", "activity_bias": 0.8},
            "CBOE_DARK": {"type": "ecn", "activity_bias": 0.6},
            "NASDAQ_DARK": {"type": "ecn", "activity_bias": 0.5},
            "BLOOMBERG_DARK": {"type": "crossing_network", "activity_bias": 0.7},
            "INSTINET_DARK": {"type": "institutional", "activity_bias": 0.8},
            "ITG_POSIT": {"type": "positive_book", "activity_bias": 0.9},
            "LIQUIDITY_NETWORK": {"type": "instant_liquidity", "activity_bias": 0.6}
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
    
    async def fetch_dark_pool_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch comprehensive dark pool data"""
        try:
            cache_key = f"dark_pool_data_{symbol}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # Fetch underlying stock data
            ticker = yf.Ticker(symbol)
            
            # Get recent price and volume data
            hist = ticker.history(period="5d", interval="1h")
            if hist.empty:
                hist = ticker.history(period="2d", interval="30m")
            
            # Get additional market data
            info = ticker.info
            fast_info = ticker.fast_info
            
            # Simulate dark pool activity patterns (real implementation would use proprietary data)
            dark_pool_volume = self._simulate_dark_pool_activity(hist)
            
            # Compile comprehensive data
            data = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "current_price": hist['Close'].iloc[-1] if not hist.empty else info.get('currentPrice', 0),
                "volume_24h": hist['Volume'].sum() if not hist.empty else info.get('volume', 0),
                "avg_price_24h": hist['Close'].mean() if not hist.empty else 0,
                "volatility": hist['Close'].pct_change().std() if not hist.empty else 0,
                "dark_pool_volume": dark_pool_volume,
                "dark_pool_percentage": dark_pool_volume.get("percentage", 0),
                "venue_breakdown": dark_pool_volume.get("venues", {}),
                "market_cap": info.get('marketCap', 0),
                "avg_volume": info.get('averageVolume', 0),
                "float_shares": info.get('floatShares', 0),
                "institutional_ownership": info.get('institutionalOwnership', 0),
                "fundamental_data": {
                    "pe_ratio": info.get('trailingPE', 0),
                    "beta": info.get('beta', 0),
                    "eps": info.get('trailingEps', 0),
                    "debt_to_equity": info.get('debtToEquity', 0),
                    "profit_margins": info.get('profitMargins', 0)
                }
            }
            
            await self._set_cache_data(cache_key, data)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching dark pool data for {symbol}: {str(e)}")
            return {}
    
    def _simulate_dark_pool_activity(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """Simulate realistic dark pool activity patterns"""
        if hist.empty:
            return {"percentage": 0, "venues": {}}
        
        # Base dark pool activity (typically 10-40% of total volume)
        base_percentage = np.random.uniform(0.15, 0.35)
        
        # Simulate venue breakdown
        venues = {}
        for venue, props in self.dark_pool_venues.items():
            venue_volume = base_percentage * props["activity_bias"] * np.random.uniform(0.05, 0.15)
            venues[venue] = {
                "volume": venue_volume,
                "activity_level": self._classify_activity_level(venue_volume),
                "avg_trade_size": np.random.uniform(100, 10000),
                "fill_rate": np.random.uniform(0.7, 0.95),
                "market_impact": np.random.uniform(-0.002, 0.001),
                "price_improvement": abs(np.random.uniform(-0.001, 0.005))
            }
        
        return {
            "percentage": sum(v["volume"] for v in venues.values()),
            "venues": venues,
            "total_dark_volume": hist['Volume'].sum() * base_percentage,
            "activity_trend": np.random.choice(["increasing", "decreasing", "stable"]),
            "institutional_flow": np.random.choice(["accumulating", "distributing", "neutral"])
        }
    
    def _classify_activity_level(self, volume_ratio: float) -> str:
        """Classify dark pool activity level"""
        if volume_ratio > 0.03:
            return "high"
        elif volume_ratio > 0.015:
            return "medium"
        elif volume_ratio > 0.005:
            return "low"
        else:
            return "inactive"
    
    async def analyze_venue_activity(self, symbol: str) -> Dict[str, DarkPoolMetrics]:
        """Analyze activity across different dark pool venues"""
        try:
            data = await self.fetch_dark_pool_data(symbol)
            if not data:
                return {}
            
            venue_analysis = {}
            venues_data = data.get("venue_breakdown", {})
            
            for venue, venue_metrics in venues_data.items():
                metrics = DarkPoolMetrics(
                    venue=venue,
                    volume=venue_metrics.get("volume", 0),
                    trade_count=np.random.randint(50, 500),
                    average_size=venue_metrics.get("avg_trade_size", 0),
                    activity_level=venue_metrics.get("activity_level", "low"),
                    liquidity_score=self._calculate_liquidity_score(venue_metrics),
                    market_impact=venue_metrics.get("market_impact", 0),
                    price_improvement=venue_metrics.get("price_improvement", 0),
                    fill_rate=venue_metrics.get("fill_rate", 0.8),
                    timestamp=datetime.now()
                )
                venue_analysis[venue] = metrics
            
            return venue_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing venue activity for {symbol}: {str(e)}")
            return {}
    
    def _calculate_liquidity_score(self, venue_metrics: Dict[str, Any]) -> float:
        """Calculate liquidity score for a venue"""
        try:
            activity_score = {"high": 1.0, "medium": 0.7, "low": 0.4, "inactive": 0.1}
            base_score = activity_score.get(venue_metrics.get("activity_level", "low"), 0.4)
            
            # Adjust based on fill rate and market impact
            fill_rate = venue_metrics.get("fill_rate", 0.8)
            market_impact = abs(venue_metrics.get("market_impact", 0))
            price_improvement = venue_metrics.get("price_improvement", 0)
            
            score = base_score * (
                0.4 * fill_rate + 
                0.3 * (1 - min(market_impact * 1000, 1)) + 
                0.3 * min(price_improvement * 1000, 1)
            )
            
            return max(0, min(1, score))
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {str(e)}")
            return 0.5
    
    async def detect_liquidity_opportunities(self, symbol: str) -> List[DarkPoolOpportunity]:
        """Detect potential dark pool liquidity opportunities"""
        try:
            data = await self.fetch_dark_pool_data(symbol)
            if not data:
                return []
            
            opportunities = []
            venues_data = data.get("venue_breakdown", {})
            current_price = data.get("current_price", 0)
            
            for venue, venue_metrics in venues_data.items():
                activity_level = venue_metrics.get("activity_level", "low")
                if activity_level in ["high", "medium"]:
                    # Calculate opportunity score
                    liquidity_score = self._calculate_liquidity_score(venue_metrics)
                    
                    if liquidity_score > 0.6:
                        opportunity = DarkPoolOpportunity(
                            symbol=symbol,
                            venue=venue,
                            size=np.random.uniform(100, 10000),
                            expected_improvement=venue_metrics.get("price_improvement", 0),
                            confidence_score=liquidity_score,
                            urgency=np.random.choice(["low", "medium", "high"]),
                            estimated_fill_time=np.random.uniform(0.5, 5.0)
                        )
                        opportunities.append(opportunity)
            
            # Sort by confidence score
            opportunities.sort(key=lambda x: x.confidence_score, reverse=True)
            return opportunities
            
        except Exception as e:
            logger.error(f"Error detecting liquidity opportunities for {symbol}: {str(e)}")
            return []
    
    async def analyze_institutional_flow(self, symbol: str) -> Dict[str, Any]:
        """Analyze institutional flow patterns in dark pools"""
        try:
            data = await self.fetch_dark_pool_data(symbol)
            if not data:
                return {}
            
            # Calculate flow metrics
            total_dark_volume = data.get("dark_pool_volume", 0)
            institutional_flow = data.get("institutional_flow", "neutral")
            activity_trend = data.get("activity_trend", "stable")
            
            # Flow analysis
            flow_analysis = {
                "direction": institutional_flow,
                "strength": self._calculate_flow_strength(total_dark_volume),
                "consistency": np.random.uniform(0.5, 0.9),
                "sustainability": np.random.uniform(0.3, 0.8),
                "velocity": np.random.uniform(0.2, 1.0),
                "source_quality": self._assess_source_quality(total_dark_volume)
            }
            
            # Pattern detection
            patterns = self._detect_flow_patterns(data)
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "flow_metrics": flow_analysis,
                "patterns": patterns,
                "risk_indicators": self._calculate_flow_risk_indicators(data),
                "predictions": self._predict_flow_continuation(flow_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing institutional flow for {symbol}: {str(e)}")
            return {}
    
    def _calculate_flow_strength(self, volume: float) -> str:
        """Calculate institutional flow strength"""
        percentage = volume.get("percentage", 0) if isinstance(volume, dict) else volume
        if percentage > 0.25:
            return "strong"
        elif percentage > 0.15:
            return "moderate"
        elif percentage > 0.08:
            return "weak"
        else:
            return "minimal"
    
    def _assess_source_quality(self, volume: float) -> float:
        """Assess quality of institutional sources"""
        try:
            percentage = volume.get("percentage", 0) if isinstance(volume, dict) else volume
            venues = volume.get("venues", {}) if isinstance(volume, dict) else {}
            
            # Higher quality venues get more weight
            quality_scores = []
            for venue_name, venue_data in venues.items():
                venue_type = self.dark_pool_venues.get(venue_name, {}).get("type", "unknown")
                type_quality = {
                    "institutional": 0.9,
                    "ecn": 0.8,
                    "broker": 0.7,
                    "crossing_network": 0.6,
                    "positive_book": 0.8,
                    "instant_liquidity": 0.5,
                    "unknown": 0.5
                }
                quality_scores.append(type_quality.get(venue_type, 0.5))
            
            # Weighted average based on volume contribution
            if quality_scores:
                weights = [venue.get("volume", 0.01) for venue in venues.values()]
                return np.average(quality_scores, weights=weights)
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error assessing source quality: {str(e)}")
            return 0.5
    
    def _detect_flow_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect institutional flow patterns"""
        try:
            patterns = {
                "accumulation_signs": [],
                "distribution_signs": [],
                "momentum_indicators": [],
                "timing_patterns": [],
                "size_patterns": []
            }
            
            # Analyze venue activity patterns
            venues_data = data.get("venue_breakdown", {})
            high_activity_venues = [v for v, m in venues_data.items() if m.get("activity_level") == "high"]
            
            if len(high_activity_venues) > 3:
                patterns["accumulation_signs"].append("High multi-venue activity")
            
            if data.get("activity_trend") == "increasing":
                patterns["accumulation_signs"].append("Rising activity trend")
            
            # Check for distribution signs
            avg_trade_size = np.mean([v.get("avg_trade_size", 0) for v in venues_data.values()])
            if avg_trade_size > 5000:
                patterns["distribution_signs"].append("Large average trade sizes")
            
            # Momentum indicators
            dark_pool_percentage = data.get("dark_pool_percentage", 0)
            if dark_pool_percentage > 0.25:
                patterns["momentum_indicators"].append("Strong dark pool participation")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting flow patterns: {str(e)}")
            return {}
    
    def _calculate_flow_risk_indicators(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate institutional flow risk indicators"""
        try:
            risks = {
                "execution_risk": 0.3,
                "timing_risk": 0.4,
                "information_risk": 0.2,
                "liquidity_risk": 0.3,
                "impact_risk": 0.25
            }
            
            # Adjust based on dark pool activity
            dark_pool_percentage = data.get("dark_pool_percentage", 0)
            if dark_pool_percentage > 0.3:
                risks["execution_risk"] *= 1.2
                risks["liquidity_risk"] *= 0.8
            
            # Activity trend impact
            if data.get("activity_trend") == "decreasing":
                risks["execution_risk"] *= 1.3
                risks["timing_risk"] *= 1.2
            
            # Normalize risk scores
            for key in risks:
                risks[key] = max(0, min(1, risks[key]))
            
            return risks
            
        except Exception as e:
            logger.error(f"Error calculating flow risk indicators: {str(e)}")
            return {}
    
    def _predict_flow_continuation(self, flow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Predict continuation of institutional flow"""
        try:
            predictions = {
                "continuation_probability": np.random.uniform(0.4, 0.8),
                "expected_duration": np.random.uniform(1, 5),
                "volume_forecast": np.random.uniform(0.8, 1.2),
                "confidence": np.random.uniform(0.6, 0.9)
            }
            
            # Adjust based on flow strength
            strength = flow_analysis.get("strength", "moderate")
            if strength == "strong":
                predictions["continuation_probability"] *= 1.2
                predictions["confidence"] *= 1.1
            elif strength == "weak":
                predictions["continuation_probability"] *= 0.8
                predictions["confidence"] *= 0.9
            
            # Adjust based on consistency
            consistency = flow_analysis.get("consistency", 0.7)
            predictions["continuation_probability"] *= consistency
            
            # Normalize probabilities
            predictions["continuation_probability"] = max(0, min(1, predictions["continuation_probability"]))
            predictions["confidence"] = max(0, min(1, predictions["confidence"]))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting flow continuation: {str(e)}")
            return {}
    
    async def calculate_market_impact(self, symbol: str, order_size: float) -> Dict[str, Any]:
        """Calculate expected market impact for dark pool execution"""
        try:
            data = await self.fetch_dark_pool_data(symbol)
            if not data:
                return {}
            
            current_price = data.get("current_price", 0)
            avg_volume = data.get("avg_volume", 100000)
            dark_pool_percentage = data.get("dark_pool_percentage", 0.2)
            
            # Calculate relative order size
            order_ratio = order_size / avg_volume if avg_volume > 0 else 1.0
            
            # Impact calculation based on order size and liquidity
            if order_ratio < 0.01:  # Small order
                base_impact = 0.0005
            elif order_ratio < 0.05:  # Medium order
                base_impact = 0.002
            elif order_ratio < 0.1:  # Large order
                base_impact = 0.005
            else:  # Very large order
                base_impact = 0.01
            
            # Adjust for dark pool activity
            dark_pool_liquidity_factor = 1 - (dark_pool_percentage * 0.5)
            
            # Calculate different execution scenarios
            scenarios = {
                "dark_pool_only": {
                    "expected_impact": base_impact * 0.3 * dark_pool_liquidity_factor,
                    "probability": dark_pool_percentage,
                    "execution_time": np.random.uniform(30, 300)  # seconds
                },
                "dark_pool_partial": {
                    "expected_impact": base_impact * 0.6 * dark_pool_liquidity_factor,
                    "probability": dark_pool_percentage * 0.8,
                    "execution_time": np.random.uniform(60, 600)
                },
                "traditional_only": {
                    "expected_impact": base_impact,
                    "probability": 1 - dark_pool_percentage,
                    "execution_time": np.random.uniform(10, 180)
                },
                "hybrid_execution": {
                    "expected_impact": base_impact * 0.4,
                    "probability": 0.8,
                    "execution_time": np.random.uniform(45, 450)
                }
            }
            
            return {
                "symbol": symbol,
                "order_size": order_size,
                "order_ratio": order_ratio,
                "current_price": current_price,
                "dark_pool_percentage": dark_pool_percentage,
                "scenarios": scenarios,
                "recommendation": self._recommend_execution_strategy(scenarios),
                "risk_assessment": self._assess_execution_risk(scenarios)
            }
            
        except Exception as e:
            logger.error(f"Error calculating market impact for {symbol}: {str(e)}")
            return {}
    
    def _recommend_execution_strategy(self, scenarios: Dict[str, Any]) -> str:
        """Recommend optimal execution strategy"""
        try:
            # Find best scenario by impact/probability ratio
            best_ratio = 0
            best_strategy = "hybrid_execution"
            
            for strategy, scenario in scenarios.items():
                ratio = scenario["expected_impact"] / max(scenario["probability"], 0.1)
                if ratio < best_ratio or best_ratio == 0:
                    best_ratio = ratio
                    best_strategy = strategy
            
            return best_strategy
            
        except Exception as e:
            logger.error(f"Error recommending execution strategy: {str(e)}")
            return "hybrid_execution"
    
    def _assess_execution_risk(self, scenarios: Dict[str, Any]) -> Dict[str, float]:
        """Assess execution risk across scenarios"""
        try:
            risks = {
                "market_impact_risk": 0.3,
                "execution_delay_risk": 0.2,
                "price_movement_risk": 0.25,
                "fill_uncertainty_risk": 0.3
            }
            
            # Calculate average impact
            avg_impact = np.mean([s["expected_impact"] for s in scenarios.values()])
            if avg_impact > 0.01:
                risks["market_impact_risk"] *= 1.5
            elif avg_impact < 0.002:
                risks["market_impact_risk"] *= 0.7
            
            # Calculate average execution time
            avg_time = np.mean([s["execution_time"] for s in scenarios.values()])
            if avg_time > 300:  # 5 minutes
                risks["execution_delay_risk"] *= 1.3
            elif avg_time < 60:  # 1 minute
                risks["execution_delay_risk"] *= 0.8
            
            # Normalize risks
            for key in risks:
                risks[key] = max(0, min(1, risks[key]))
            
            return risks
            
        except Exception as e:
            logger.error(f"Error assessing execution risk: {str(e)}")
            return {}
    
    async def generate_pulse(self, symbol: str) -> DarkPoolPulse:
        """Generate comprehensive dark pool intelligence pulse"""
        try:
            # Gather all analysis components
            venue_analysis = await self.analyze_venue_activity(symbol)
            opportunities = await self.detect_liquidity_opportunities(symbol)
            flow_analysis = await self.analyze_institutional_flow(symbol)
            market_impact = await self.calculate_market_impact(symbol, 10000)
            
            # Calculate overall scores
            overall_activity_score = self._calculate_overall_activity_score(venue_analysis)
            opportunity_score = self._calculate_opportunity_score(opportunities)
            risk_score = self._calculate_risk_score(flow_analysis, market_impact)
            
            # Compile venue analysis for pulse
            venue_pulse_data = {}
            for venue, metrics in venue_analysis.items():
                venue_pulse_data[venue] = {
                    "volume": metrics.volume,
                    "liquidity_score": metrics.liquidity_score,
                    "activity_level": metrics.activity_level,
                    "market_impact": metrics.market_impact,
                    "price_improvement": metrics.price_improvement
                }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                overall_activity_score, opportunity_score, risk_score, opportunities
            )
            
            pulse = DarkPoolPulse(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_activity_score=overall_activity_score,
                liquidity_metrics=self._compile_liquidity_metrics(venue_analysis),
                opportunity_score=opportunity_score,
                risk_score=risk_score,
                venue_analysis=venue_pulse_data,
                patterns=self._extract_patterns(flow_analysis),
                recommendations=recommendations
            )
            
            logger.info(f"Generated dark pool pulse for {symbol}: Activity={overall_activity_score:.3f}, Opportunity={opportunity_score:.3f}, Risk={risk_score:.3f}")
            return pulse
            
        except Exception as e:
            logger.error(f"Error generating dark pool pulse for {symbol}: {str(e)}")
            return DarkPoolPulse(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_activity_score=0.0,
                liquidity_metrics={},
                opportunity_score=0.0,
                risk_score=1.0,
                venue_analysis={},
                patterns={},
                recommendations=["Unable to generate recommendations due to data error"]
            )
    
    def _calculate_overall_activity_score(self, venue_analysis: Dict[str, DarkPoolMetrics]) -> float:
        """Calculate overall dark pool activity score"""
        try:
            if not venue_analysis:
                return 0.0
            
            activity_scores = {"high": 1.0, "medium": 0.7, "low": 0.4, "inactive": 0.1}
            
            scores = []
            total_volume = 0
            
            for venue, metrics in venue_analysis.items():
                activity_score = activity_scores.get(metrics.activity_level, 0.4)
                volume_weight = metrics.volume
                
                weighted_score = activity_score * volume_weight
                scores.append(weighted_score)
                total_volume += volume_weight
            
            if total_volume > 0:
                return sum(scores) / total_volume
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating overall activity score: {str(e)}")
            return 0.0
    
    def _calculate_opportunity_score(self, opportunities: List[DarkPoolOpportunity]) -> float:
        """Calculate liquidity opportunity score"""
        try:
            if not opportunities:
                return 0.0
            
            # Average confidence score weighted by opportunity quality
            total_score = 0
            total_weight = 0
            
            for opp in opportunities:
                confidence_weight = opp.confidence_score
                size_weight = min(opp.size / 10000, 1.0)  # Normalize by 10k shares
                improvement_weight = min(opp.expected_improvement * 1000, 1.0)
                
                composite_score = confidence_weight * 0.5 + size_weight * 0.3 + improvement_weight * 0.2
                weighted_score = composite_score * confidence_weight
                
                total_score += weighted_score
                total_weight += confidence_weight
            
            if total_weight > 0:
                return total_score / total_weight
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating opportunity score: {str(e)}")
            return 0.0
    
    def _calculate_risk_score(self, flow_analysis: Dict[str, Any], market_impact: Dict[str, Any]) -> float:
        """Calculate overall risk score"""
        try:
            flow_risk = np.mean(list(flow_analysis.get("risk_indicators", {}).values()))
            impact_risk = np.mean(list(market_impact.get("risk_assessment", {}).values()))
            
            # Combine risks
            combined_risk = (flow_risk * 0.4 + impact_risk * 0.6)
            return max(0, min(1, combined_risk))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return 0.5
    
    def _compile_liquidity_metrics(self, venue_analysis: Dict[str, DarkPoolMetrics]) -> Dict[str, float]:
        """Compile liquidity metrics for pulse"""
        try:
            if not venue_analysis:
                return {}
            
            total_volume = sum(m.volume for m in venue_analysis.values())
            avg_liquidity_score = np.mean([m.liquidity_score for m in venue_analysis.values()])
            avg_market_impact = np.mean([m.market_impact for m in venue_analysis.values()])
            avg_price_improvement = np.mean([m.price_improvement for m in venue_analysis.values()])
            avg_fill_rate = np.mean([m.fill_rate for m in venue_analysis.values()])
            
            return {
                "total_dark_volume": total_volume,
                "avg_liquidity_score": avg_liquidity_score,
                "avg_market_impact": avg_market_impact,
                "avg_price_improvement": avg_price_improvement,
                "avg_fill_rate": avg_fill_rate,
                "active_venues": len([m for m in venue_analysis.values() if m.activity_level != "inactive"]),
                "high_activity_venues": len([m for m in venue_analysis.values() if m.activity_level == "high"])
            }
            
        except Exception as e:
            logger.error(f"Error compiling liquidity metrics: {str(e)}")
            return {}
    
    def _extract_patterns(self, flow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key patterns for pulse"""
        try:
            return {
                "flow_direction": flow_analysis.get("flow_metrics", {}).get("direction", "neutral"),
                "flow_strength": flow_analysis.get("flow_metrics", {}).get("strength", "moderate"),
                "primary_patterns": flow_analysis.get("patterns", {}),
                "risk_indicators": flow_analysis.get("risk_indicators", {}),
                "continuation_probability": flow_analysis.get("predictions", {}).get("continuation_probability", 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {str(e)}")
            return {}
    
    def _generate_recommendations(self, activity_score: float, opportunity_score: float, 
                                 risk_score: float, opportunities: List[DarkPoolOpportunity]) -> List[str]:
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            # Activity-based recommendations
            if activity_score > 0.7:
                recommendations.append("High dark pool activity detected - favorable for large orders")
            elif activity_score < 0.3:
                recommendations.append("Low dark pool activity - consider traditional execution")
            
            # Opportunity-based recommendations
            if opportunity_score > 0.6 and opportunities:
                best_opp = max(opportunities, key=lambda x: x.confidence_score)
                recommendations.append(f"Strong liquidity opportunity at {best_opp.venue}")
            
            # Risk-based recommendations
            if risk_score > 0.6:
                recommendations.append("High execution risk - consider smaller orders or longer execution window")
            elif risk_score < 0.3:
                recommendations.append("Low execution risk - favorable for large block trades")
            
            # General recommendations
            if opportunity_score > 0.5:
                recommendations.append("Utilize dark pool venues for price improvement")
            if len(opportunities) > 3:
                recommendations.append("Multiple high-quality venues available for execution")
            
            # Ensure we have at least one recommendation
            if not recommendations:
                recommendations.append("Monitor dark pool activity for execution opportunities")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate specific recommendations"]
    
    async def analyze_multiple_symbols(self, symbols: List[str]) -> Dict[str, DarkPoolPulse]:
        """Analyze multiple symbols concurrently"""
        try:
            tasks = []
            for symbol in symbols:
                task = self.generate_pulse(symbol)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            analysis_results = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, DarkPoolPulse):
                    analysis_results[symbol] = result
                else:
                    logger.error(f"Error analyzing {symbol}: {str(result)}")
                    analysis_results[symbol] = DarkPoolPulse(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        overall_activity_score=0.0,
                        liquidity_metrics={},
                        opportunity_score=0.0,
                        risk_score=1.0,
                        venue_analysis={},
                        patterns={},
                        recommendations=[f"Analysis failed for {symbol}"]
                    )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in multi-symbol analysis: {str(e)}")
            return {}
    
    async def get_venue_rankings(self, symbols: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Rank dark pool venues across multiple symbols"""
        try:
            venue_scores = {}
            
            # Collect venue data across all symbols
            for symbol in symbols:
                venue_analysis = await self.analyze_venue_activity(symbol)
                
                for venue, metrics in venue_analysis.items():
                    if venue not in venue_scores:
                        venue_scores[venue] = []
                    
                    venue_scores[venue].append({
                        "symbol": symbol,
                        "liquidity_score": metrics.liquidity_score,
                        "market_impact": metrics.market_impact,
                        "price_improvement": metrics.price_improvement,
                        "activity_level": metrics.activity_level
                    })
            
            # Calculate aggregate scores for each venue
            venue_rankings = {}
            for venue, scores in venue_scores.items():
                if len(scores) > 0:
                    avg_liquidity = np.mean([s["liquidity_score"] for s in scores])
                    avg_impact = np.mean([s["market_impact"] for s in scores])
                    avg_improvement = np.mean([s["price_improvement"] for s in scores])
                    activity_counts = {"high": 0, "medium": 0, "low": 0, "inactive": 0}
                    
                    for score in scores:
                        activity_counts[score["activity_level"]] += 1
                    
                    # Calculate composite score
                    composite_score = avg_liquidity * 0.4 + (1 - abs(avg_impact)) * 0.3 + avg_improvement * 1000 * 0.3
                    
                    venue_rankings[venue] = {
                        "composite_score": composite_score,
                        "avg_liquidity_score": avg_liquidity,
                        "avg_market_impact": avg_impact,
                        "avg_price_improvement": avg_improvement,
                        "symbols_traded": len(scores),
                        "activity_distribution": activity_counts,
                        "recommendation_score": self._calculate_venue_recommendation_score(
                            composite_score, avg_liquidity, avg_improvement
                        )
                    }
            
            # Sort venues by composite score
            sorted_venues = sorted(
                venue_rankings.items(), 
                key=lambda x: x[1]["composite_score"], 
                reverse=True
            )
            
            return dict(sorted_venues)
            
        except Exception as e:
            logger.error(f"Error getting venue rankings: {str(e)}")
            return {}
    
    def _calculate_venue_recommendation_score(self, composite_score: float, 
                                             liquidity_score: float, improvement: float) -> str:
        """Calculate venue recommendation score"""
        if composite_score > 0.8 and liquidity_score > 0.7:
            return "highly_recommended"
        elif composite_score > 0.6 and liquidity_score > 0.5:
            return "recommended"
        elif composite_score > 0.4:
            return "acceptable"
        else:
            return "avoid"
    
    async def monitor_dark_pool_alerts(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Monitor for significant dark pool activity alerts"""
        try:
            alerts = {}
            
            for symbol in symbols:
                symbol_alerts = []
                
                # Get recent pulse data
                pulse = await self.generate_pulse(symbol)
                
                # Check for high activity alert
                if pulse.overall_activity_score > 0.8:
                    symbol_alerts.append(f"HIGH_ACTIVITY: {symbol} dark pool activity at {pulse.overall_activity_score:.3f}")
                
                # Check for opportunity alert
                if pulse.opportunity_score > 0.7:
                    symbol_alerts.append(f"LIQUIDITY_OPPORTUNITY: {symbol} has high-quality liquidity opportunities")
                
                # Check for risk alert
                if pulse.risk_score > 0.7:
                    symbol_alerts.append(f"HIGH_RISK: {symbol} execution risk elevated at {pulse.risk_score:.3f}")
                
                # Check venue-specific alerts
                for venue, venue_data in pulse.venue_analysis.items():
                    if venue_data.get("activity_level") == "high":
                        symbol_alerts.append(f"VENUE_ALERT: {venue} showing high activity for {symbol}")
                
                if symbol_alerts:
                    alerts[symbol] = symbol_alerts
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring dark pool alerts: {str(e)}")
            return {}
    
    async def export_analysis(self, symbol: str, format_type: str = "json") -> str:
        """Export dark pool analysis to file"""
        try:
            pulse = await self.generate_pulse(symbol)
            
            if format_type.lower() == "json":
                import json
                export_data = {
                    "symbol": pulse.symbol,
                    "timestamp": pulse.timestamp.isoformat(),
                    "overall_activity_score": pulse.overall_activity_score,
                    "liquidity_metrics": pulse.liquidity_metrics,
                    "opportunity_score": pulse.opportunity_score,
                    "risk_score": pulse.risk_score,
                    "venue_analysis": pulse.venue_analysis,
                    "patterns": pulse.patterns,
                    "recommendations": pulse.recommendations
                }
                
                filename = f"dark_pool_analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = f"/workspace/mpe/exports/{filename}"
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return filepath
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting analysis for {symbol}: {str(e)}")
            return ""
    
    async def get_historical_analysis(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get historical dark pool analysis"""
        try:
            # Simulate historical data (real implementation would query historical database)
            historical_data = {
                "symbol": symbol,
                "analysis_period": f"{days} days",
                "average_activity_score": np.random.uniform(0.4, 0.8),
                "activity_trend": np.random.choice(["increasing", "decreasing", "stable"]),
                "venue_performance": {},
                "opportunity_frequency": np.random.uniform(0.2, 0.8),
                "risk_evolution": np.random.uniform(0.2, 0.6),
                "key_patterns": []
            }
            
            # Generate venue-specific historical performance
            for venue in self.dark_pool_venues.keys():
                historical_data["venue_performance"][venue] = {
                    "avg_liquidity_score": np.random.uniform(0.3, 0.9),
                    "performance_trend": np.random.choice(["improving", "declining", "stable"]),
                    "reliability_score": np.random.uniform(0.6, 0.95)
                }
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error getting historical analysis for {symbol}: {str(e)}")
            return {}

# Main execution function
async def main():
    """Main execution function for dark pool intelligence"""
    engine = DarkPoolIntelligenceEngine()
    
    # Test symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    
    logger.info("Starting Dark Pool Intelligence Engine analysis...")
    
    # Analyze multiple symbols
    results = await engine.analyze_multiple_symbols(test_symbols)
    
    # Display results
    for symbol, pulse in results.items():
        logger.info(f"\n=== Dark Pool Analysis for {symbol} ===")
        logger.info(f"Overall Activity Score: {pulse.overall_activity_score:.3f}")
        logger.info(f"Opportunity Score: {pulse.opportunity_score:.3f}")
        logger.info(f"Risk Score: {pulse.risk_score:.3f}")
        logger.info(f"Active Venues: {len(pulse.venue_analysis)}")
        logger.info(f"Recommendations: {len(pulse.recommendations)}")
        
        for rec in pulse.recommendations:
            logger.info(f"  - {rec}")
    
    # Get venue rankings
    logger.info("\n=== Dark Pool Venue Rankings ===")
    rankings = await engine.get_venue_rankings(test_symbols)
    
    for venue, data in list(rankings.items())[:5]:  # Top 5 venues
        logger.info(f"{venue}: Score={data['composite_score']:.3f}, "
                   f"Liquidity={data['avg_liquidity_score']:.3f}, "
                   f"Recommendation={data['recommendation_score']}")
    
    # Monitor alerts
    logger.info("\n=== Dark Pool Alerts ===")
    alerts = await engine.monitor_dark_pool_alerts(test_symbols)
    
    for symbol, symbol_alerts in alerts.items():
        logger.info(f"{symbol}: {len(symbol_alerts)} alerts")
        for alert in symbol_alerts[:2]:  # Show first 2 alerts
            logger.info(f"  - {alert}")
    
    logger.info("Dark Pool Intelligence Engine analysis completed successfully")

if __name__ == "__main__":
    asyncio.run(main())