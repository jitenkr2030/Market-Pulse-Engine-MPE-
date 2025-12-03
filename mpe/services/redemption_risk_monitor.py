"""
Module 22: Redemption Risk Monitor
Author: MiniMax Agent
Date: 2025-12-02

Advanced redemption risk monitoring and liquidity stress analysis system.
Provides comprehensive assessment of fund redemption risks, liquidity pressures,
and cash flow stress for institutional risk management.
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FundType(Enum):
    """Types of funds"""
    MUTUAL_FUND = "mutual_fund"
    HEDGE_FUND = "hedge_fund"
    ETF = "etf"
    PENSION_FUND = "pension_fund"
    ENDOWMENT = "endowment"
    INSURANCE_FUND = "insurance_fund"
    PRIVATE_FUND = "private_fund"
    MONEY_MARKET = "money_market"

class LiquidityProfile(Enum):
    """Liquidity profile classifications"""
    HIGH_LIQUIDITY = "high_liquidity"
    MODERATE_LIQUIDITY = "moderate_liquidity"
    LOW_LIQUIDITY = "low_liquidity"
    ILLIQUID = "illiquid"
    STRESSED = "stressed"

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

@dataclass
class LiquidityMetric:
    """Individual liquidity metric"""
    metric_name: str
    value: float
    threshold: float
    status: str  # "normal", "warning", "critical"
    timestamp: datetime
    data_quality: float

@dataclass
class RedemptionPressure:
    """Redemption pressure assessment"""
    fund_identifier: str
    fund_type: str
    current_pressure: float
    pressure_trend: str
    liquidity_coverage: float
    stress_scenarios: Dict[str, float]
    mitigation_capabilities: Dict[str, float]
    timestamp: datetime

@dataclass
class CashFlowStress:
    """Cash flow stress analysis"""
    entity: str
    time_horizon: int  # days
    projected_cash_inflow: float
    projected_cash_outflow: float
    net_cash_flow: float
    liquidity_buffer: float
    stress_multiplier: float
    risk_factors: List[str]

@dataclass
class RedemptionRiskAnalysis:
    """Comprehensive redemption risk analysis"""
    entity: str
    fund_type: str
    timestamp: datetime
    overall_risk_score: float
    risk_level: str
    liquidity_profile: str
    redemption_pressure: RedemptionPressure
    cash_flow_stress: CashFlowStress
    liquidity_metrics: List[LiquidityMetric]
    stress_scenarios: Dict[str, float]
    early_warning_indicators: List[Dict[str, Any]]
    mitigation_strategies: List[str]
    recommendations: List[str]

class RedemptionRiskMonitor:
    """
    Advanced Redemption Risk Monitor
    
    Monitors, analyzes, and provides intelligence on redemption risks
    and liquidity stress to support risk management and capital preservation.
    """
    
    def __init__(self):
        self.name = "Redemption Risk Monitor"
        self.version = "1.0.0"
        self.cache = {}
        self.cache_ttl = 240  # 4 minutes
        
        # Risk thresholds by fund type
        self.risk_thresholds = {
            "mutual_fund": {
                "liquidity_ratio_warning": 0.15,
                "liquidity_ratio_critical": 0.10,
                "redemption_velocity_warning": 0.05,
                "redemption_velocity_critical": 0.10,
                "cash_buffer_warning": 0.02,
                "cash_buffer_critical": 0.01
            },
            "hedge_fund": {
                "liquidity_ratio_warning": 0.20,
                "liquidity_ratio_critical": 0.15,
                "redemption_velocity_warning": 0.08,
                "redemption_velocity_critical": 0.15,
                "cash_buffer_warning": 0.03,
                "cash_buffer_critical": 0.015
            },
            "etf": {
                "liquidity_ratio_warning": 0.10,
                "liquidity_ratio_critical": 0.05,
                "redemption_velocity_warning": 0.03,
                "redemption_velocity_critical": 0.08,
                "cash_buffer_warning": 0.015,
                "cash_buffer_critical": 0.008
            },
            "pension_fund": {
                "liquidity_ratio_warning": 0.25,
                "liquidity_ratio_critical": 0.20,
                "redemption_velocity_warning": 0.02,
                "redemption_velocity_critical": 0.05,
                "cash_buffer_warning": 0.04,
                "cash_buffer_critical": 0.02
            }
        }
        
        # Market stress indicators
        self.stress_indicators = {
            "volatility_spike": {"threshold": 2.0, "weight": 0.3},
            "liquidity_dry_up": {"threshold": 0.7, "weight": 0.25},
            "correlation_breakdown": {"threshold": 0.8, "weight": 0.2},
            "credit_spread_widening": {"threshold": 2.5, "weight": 0.25}
        }
        
        # Liquidity sources and their reliability
        self.liquidity_sources = {
            "cash_equivalents": {"availability": 1.0, "reliability": 1.0},
            "treasury_securities": {"availability": 0.95, "reliability": 0.95},
            "high_quality_bonds": {"availability": 0.85, "reliability": 0.8},
            "large_cap_equities": {"availability": 0.75, "reliability": 0.7},
            "medium_cap_equities": {"availability": 0.6, "reliability": 0.6},
            "small_cap_equities": {"availability": 0.4, "reliability": 0.4},
            "illiquid_alternatives": {"availability": 0.2, "reliability": 0.3},
            "real_estate": {"availability": 0.1, "reliability": 0.2}
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
    
    async def fetch_fund_data(self, fund_identifier: str, fund_type: str) -> Dict[str, Any]:
        """Fetch comprehensive fund data for risk analysis"""
        try:
            cache_key = f"fund_data_{fund_identifier}_{fund_type}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # Simulate comprehensive fund data (real implementation would use fund databases)
            fund_data = self._simulate_fund_data(fund_identifier, fund_type)
            
            # Add market context data
            market_data = await self._get_market_context()
            fund_data["market_context"] = market_data
            
            await self._set_cache_data(cache_key, fund_data)
            return fund_data
            
        except Exception as e:
            logger.error(f"Error fetching fund data for {fund_identifier}: {str(e)}")
            return {}
    
    def _simulate_fund_data(self, fund_identifier: str, fund_type: str) -> Dict[str, Any]:
        """Simulate realistic fund data"""
        try:
            # Base fund characteristics
            if fund_type == "etf":
                total_assets = np.random.uniform(1e9, 1e12)  # $1B - $1T
                daily_volume = total_assets * np.random.uniform(0.001, 0.01)
                cash_position = total_assets * np.random.uniform(0.01, 0.05)
                redemption_notice = 0  # ETFs have immediate redemptions
            elif fund_type == "mutual_fund":
                total_assets = np.random.uniform(1e8, 1e11)  # $100M - $100B
                daily_volume = total_assets * np.random.uniform(0.0005, 0.005)
                cash_position = total_assets * np.random.uniform(0.02, 0.08)
                redemption_notice = np.random.uniform(1, 7)  # 1-7 days
            elif fund_type == "hedge_fund":
                total_assets = np.random.uniform(1e8, 5e10)  # $100M - $50B
                daily_volume = total_assets * np.random.uniform(0.0001, 0.002)
                cash_position = total_assets * np.random.uniform(0.05, 0.15)
                redemption_notice = np.random.uniform(30, 180)  # 30-180 days
            elif fund_type == "pension_fund":
                total_assets = np.random.uniform(1e9, 1e13)  # $1B - $10T
                daily_volume = total_assets * np.random.uniform(0.0001, 0.001)
                cash_position = total_assets * np.random.uniform(0.03, 0.12)
                redemption_notice = np.random.uniform(90, 365)  # 90-365 days
            else:
                total_assets = np.random.uniform(1e7, 1e9)
                daily_volume = total_assets * np.random.uniform(0.0005, 0.003)
                cash_position = total_assets * np.random.uniform(0.03, 0.10)
                redemption_notice = np.random.uniform(7, 90)
            
            # Calculate derived metrics
            liquidity_ratio = cash_position / total_assets
            daily_redemption_capacity = daily_volume * 0.5  # Assume 50% of volume can be redeemed
            
            # Redemption patterns (simulate historical redemption behavior)
            recent_redemptions = self._simulate_redemption_patterns(fund_type, total_assets)
            
            # Asset allocation
            asset_allocation = self._simulate_asset_allocation(fund_type)
            
            # Liquidity profile
            liquidity_profile = self._assess_liquidity_profile(liquidity_ratio, asset_allocation)
            
            return {
                "fund_identifier": fund_identifier,
                "fund_type": fund_type,
                "total_assets": total_assets,
                "cash_position": cash_position,
                "liquidity_ratio": liquidity_ratio,
                "daily_volume": daily_volume,
                "daily_redemption_capacity": daily_redemption_capacity,
                "redemption_notice_period": redemption_notice,
                "recent_redemptions": recent_redemptions,
                "asset_allocation": asset_allocation,
                "liquidity_profile": liquidity_profile,
                "institutional_ownership": np.random.uniform(0.1, 0.8),
                "expense_ratio": np.random.uniform(0.001, 0.03),
                "fund_age_years": np.random.uniform(1, 25),
                "performance_1y": np.random.uniform(-0.2, 0.3),
                "volatility_1y": np.random.uniform(0.05, 0.4)
            }
            
        except Exception as e:
            logger.error(f"Error simulating fund data: {str(e)}")
            return {}
    
    def _simulate_redemption_patterns(self, fund_type: str, total_assets: float) -> Dict[str, Any]:
        """Simulate historical redemption patterns"""
        try:
            # Base redemption rate depends on fund type
            base_rates = {
                "etf": 0.02,           # 2% daily
                "mutual_fund": 0.005,  # 0.5% daily
                "hedge_fund": 0.001,   # 0.1% daily
                "pension_fund": 0.0005 # 0.05% daily
            }
            
            base_redemption_rate = base_rates.get(fund_type, 0.005)
            
            # Generate 30 days of redemption data
            redemption_rates = []
            redemption_amounts = []
            
            for i in range(30):
                # Add some volatility to redemption rates
                daily_rate = base_redemption_rate * np.random.uniform(0.5, 2.0)
                daily_amount = total_assets * daily_rate
                
                redemption_rates.append(daily_rate)
                redemption_amounts.append(daily_amount)
            
            return {
                "daily_rates": redemption_rates,
                "daily_amounts": redemption_amounts,
                "avg_daily_redemption_rate": np.mean(redemption_rates),
                "max_daily_redemption_rate": max(redemption_rates),
                "redemption_volatility": np.std(redemption_rates),
                "trend_30d": np.polyfit(range(30), redemption_rates, 1)[0]
            }
            
        except Exception as e:
            logger.error(f"Error simulating redemption patterns: {str(e)}")
            return {}
    
    def _simulate_asset_allocation(self, fund_type: str) -> Dict[str, float]:
        """Simulate asset allocation by fund type"""
        try:
            allocations = {
                "cash": 0.05,
                "treasuries": 0.15,
                "investment_grade_bonds": 0.20,
                "high_yield_bonds": 0.10,
                "large_cap_equities": 0.25,
                "mid_cap_equities": 0.10,
                "small_cap_equities": 0.05,
                "alternatives": 0.10
            }
            
            # Adjust based on fund type
            if fund_type == "etf":
                allocations["cash"] = 0.02
                allocations["large_cap_equities"] = 0.50
                allocations["alternatives"] = 0.03
            elif fund_type == "mutual_fund":
                allocations["cash"] = 0.05
                allocations["large_cap_equities"] = 0.40
                allocations["alternatives"] = 0.08
            elif fund_type == "hedge_fund":
                allocations["cash"] = 0.10
                allocations["large_cap_equities"] = 0.20
                allocations["alternatives"] = 0.25
            elif fund_type == "pension_fund":
                allocations["cash"] = 0.03
                allocations["treasuries"] = 0.25
                allocations["alternatives"] = 0.15
            
            # Add small random variations
            for key in allocations:
                variation = np.random.uniform(-0.02, 0.02)
                allocations[key] = max(0, min(1, allocations[key] + variation))
            
            # Normalize to sum to 1
            total = sum(allocations.values())
            for key in allocations:
                allocations[key] /= total
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error simulating asset allocation: {str(e)}")
            return {}
    
    def _assess_liquidity_profile(self, liquidity_ratio: float, asset_allocation: Dict[str, float]) -> str:
        """Assess overall liquidity profile"""
        try:
            # Calculate weighted liquidity score
            liquidity_weights = {
                "cash": 1.0,
                "treasuries": 0.95,
                "investment_grade_bonds": 0.85,
                "high_yield_bonds": 0.70,
                "large_cap_equities": 0.75,
                "mid_cap_equities": 0.60,
                "small_cap_equities": 0.40,
                "alternatives": 0.25
            }
            
            weighted_liquidity = sum(
                allocation * liquidity_weights.get(asset_type, 0.5)
                for asset_type, allocation in asset_allocation.items()
            )
            
            # Combine cash ratio and weighted liquidity
            overall_liquidity = (liquidity_ratio * 0.4) + (weighted_liquidity * 0.6)
            
            if overall_liquidity > 0.8:
                return "high_liquidity"
            elif overall_liquidity > 0.6:
                return "moderate_liquidity"
            elif overall_liquidity > 0.4:
                return "low_liquidity"
            elif overall_liquidity > 0.2:
                return "illiquid"
            else:
                return "stressed"
                
        except Exception as e:
            logger.error(f"Error assessing liquidity profile: {str(e)}")
            return "moderate_liquidity"
    
    async def _get_market_context(self) -> Dict[str, float]:
        """Get current market stress context"""
        try:
            # Simulate market stress indicators
            return {
                "market_volatility": np.random.uniform(0.1, 0.5),
                "liquidity_spread": np.random.uniform(0.02, 0.15),
                "correlation_index": np.random.uniform(0.3, 0.9),
                "credit_spread": np.random.uniform(0.01, 0.08),
                "stress_index": np.random.uniform(0.1, 0.7)
            }
            
        except Exception as e:
            logger.error(f"Error getting market context: {str(e)}")
            return {}
    
    async def calculate_liquidity_metrics(self, fund_data: Dict[str, Any]) -> List[LiquidityMetric]:
        """Calculate comprehensive liquidity metrics"""
        try:
            metrics = []
            fund_type = fund_data["fund_type"]
            thresholds = self.risk_thresholds.get(fund_type, self.risk_thresholds["mutual_fund"])
            
            # 1. Liquidity Ratio
            liquidity_ratio = fund_data["liquidity_ratio"]
            if liquidity_ratio < thresholds["liquidity_ratio_critical"]:
                status = "critical"
            elif liquidity_ratio < thresholds["liquidity_ratio_warning"]:
                status = "warning"
            else:
                status = "normal"
            
            metrics.append(LiquidityMetric(
                metric_name="liquidity_ratio",
                value=liquidity_ratio,
                threshold=thresholds["liquidity_ratio_warning"],
                status=status,
                timestamp=datetime.now(),
                data_quality=0.9
            ))
            
            # 2. Redemption Velocity
            recent_redemptions = fund_data["recent_redemptions"]
            avg_redemption_rate = recent_redemptions["avg_daily_redemption_rate"]
            
            if avg_redemption_rate > thresholds["redemption_velocity_critical"]:
                status = "critical"
            elif avg_redemption_rate > thresholds["redemption_velocity_warning"]:
                status = "warning"
            else:
                status = "normal"
            
            metrics.append(LiquidityMetric(
                metric_name="redemption_velocity",
                value=avg_redemption_rate,
                threshold=thresholds["redemption_velocity_warning"],
                status=status,
                timestamp=datetime.now(),
                data_quality=0.8
            ))
            
            # 3. Cash Buffer
            cash_buffer = fund_data["cash_position"] / fund_data["total_assets"]
            
            if cash_buffer < thresholds["cash_buffer_critical"]:
                status = "critical"
            elif cash_buffer < thresholds["cash_buffer_warning"]:
                status = "warning"
            else:
                status = "normal"
            
            metrics.append(LiquidityMetric(
                metric_name="cash_buffer",
                value=cash_buffer,
                threshold=thresholds["cash_buffer_warning"],
                status=status,
                timestamp=datetime.now(),
                data_quality=0.95
            ))
            
            # 4. Asset Liquidity Score
            asset_allocation = fund_data["asset_allocation"]
            asset_liquidity_score = self._calculate_asset_liquidity_score(asset_allocation)
            
            if asset_liquidity_score < 0.3:
                status = "critical"
            elif asset_liquidity_score < 0.5:
                status = "warning"
            else:
                status = "normal"
            
            metrics.append(LiquidityMetric(
                metric_name="asset_liquidity_score",
                value=asset_liquidity_score,
                threshold=0.5,
                status=status,
                timestamp=datetime.now(),
                data_quality=0.85
            ))
            
            # 5. Redemption Capacity
            daily_capacity = fund_data["daily_redemption_capacity"]
            total_assets = fund_data["total_assets"]
            redemption_capacity_ratio = daily_capacity / total_assets
            
            if redemption_capacity_ratio < 0.005:
                status = "critical"
            elif redemption_capacity_ratio < 0.01:
                status = "warning"
            else:
                status = "normal"
            
            metrics.append(LiquidityMetric(
                metric_name="redemption_capacity",
                value=redemption_capacity_ratio,
                threshold=0.01,
                status=status,
                timestamp=datetime.now(),
                data_quality=0.8
            ))
            
            # 6. Liquidity Stress Score
            liquidity_stress = self._calculate_liquidity_stress(fund_data)
            
            if liquidity_stress > 0.8:
                status = "critical"
            elif liquidity_stress > 0.6:
                status = "warning"
            else:
                status = "normal"
            
            metrics.append(LiquidityMetric(
                metric_name="liquidity_stress",
                value=liquidity_stress,
                threshold=0.6,
                status=status,
                timestamp=datetime.now(),
                data_quality=0.75
            ))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {str(e)}")
            return []
    
    def _calculate_asset_liquidity_score(self, asset_allocation: Dict[str, float]) -> float:
        """Calculate weighted asset liquidity score"""
        try:
            liquidity_scores = {
                "cash": 1.0,
                "treasuries": 0.95,
                "investment_grade_bonds": 0.85,
                "high_yield_bonds": 0.70,
                "large_cap_equities": 0.75,
                "mid_cap_equities": 0.60,
                "small_cap_equities": 0.40,
                "alternatives": 0.25
            }
            
            weighted_score = sum(
                allocation * liquidity_scores.get(asset_type, 0.5)
                for asset_type, allocation in asset_allocation.items()
            )
            
            return weighted_score
            
        except Exception as e:
            logger.error(f"Error calculating asset liquidity score: {str(e)}")
            return 0.5
    
    def _calculate_liquidity_stress(self, fund_data: Dict[str, Any]) -> float:
        """Calculate overall liquidity stress score"""
        try:
            stress_components = []
            
            # 1. Cash stress
            cash_ratio = fund_data["liquidity_ratio"]
            cash_stress = max(0, 1 - (cash_ratio / 0.1)) if cash_ratio < 0.1 else 0
            stress_components.append(cash_stress * 0.3)
            
            # 2. Redemption stress
            redemption_trend = fund_data["recent_redemptions"]["trend_30d"]
            redemption_stress = min(1, abs(redemption_trend) * 100)  # Scale trend
            stress_components.append(redemption_stress * 0.25)
            
            # 3. Asset quality stress
            asset_liquidity = self._calculate_asset_liquidity_score(fund_data["asset_allocation"])
            asset_stress = max(0, 1 - asset_liquidity)
            stress_components.append(asset_stress * 0.2)
            
            # 4. Capacity stress
            capacity_ratio = fund_data["daily_redemption_capacity"] / fund_data["total_assets"]
            capacity_stress = max(0, 1 - (capacity_ratio / 0.02)) if capacity_ratio < 0.02 else 0
            stress_components.append(capacity_stress * 0.25)
            
            return sum(stress_components)
            
        except Exception as e:
            logger.error(f"Error calculating liquidity stress: {str(e)}")
            return 0.5
    
    async def assess_redemption_pressure(self, fund_data: Dict[str, Any]) -> RedemptionPressure:
        """Assess current redemption pressure"""
        try:
            recent_redemptions = fund_data["recent_redemptions"]
            total_assets = fund_data["total_assets"]
            
            # Calculate current pressure
            avg_daily_redemption = recent_redemptions["avg_daily_redemption_rate"]
            max_daily_redemption = recent_redemptions["max_daily_redemption_rate"]
            
            # Pressure based on average redemption rate
            pressure_base = avg_daily_redemption * 20  # Scale factor
            pressure_peak = max_daily_redemption * 10
            
            current_pressure = min(1.0, max(pressure_base, pressure_peak))
            
            # Determine pressure trend
            redemption_trend = recent_redemptions["trend_30d"]
            if redemption_trend > 0.001:
                pressure_trend = "increasing"
            elif redemption_trend < -0.001:
                pressure_trend = "decreasing"
            else:
                pressure_trend = "stable"
            
            # Calculate liquidity coverage
            daily_capacity = fund_data["daily_redemption_capacity"]
            liquidity_coverage = daily_capacity / (total_assets * avg_daily_redemption) if avg_daily_redemption > 0 else float('inf')
            
            # Stress scenarios
            stress_scenarios = {
                "moderate_stress": min(1.0, avg_daily_redemption * 2),
                "high_stress": min(1.0, max_daily_redemption * 1.5),
                "extreme_stress": min(1.0, max_daily_redemption * 3),
                "liquidity_crisis": min(1.0, max_daily_redemption * 5)
            }
            
            # Mitigation capabilities
            mitigation_capabilities = {
                "cash_available": fund_data["cash_position"] / total_assets,
                "securable_liquidity": self._calculate_securable_liquidity(fund_data),
                "credit_facilities": np.random.uniform(0.1, 0.3),  # Assume some credit available
                "asset_sales_capacity": self._calculate_sales_capacity(fund_data)
            }
            
            return RedemptionPressure(
                fund_identifier=fund_data["fund_identifier"],
                fund_type=fund_data["fund_type"],
                current_pressure=current_pressure,
                pressure_trend=pressure_trend,
                liquidity_coverage=liquidity_coverage,
                stress_scenarios=stress_scenarios,
                mitigation_capabilities=mitigation_capabilities,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error assessing redemption pressure: {str(e)}")
            return RedemptionPressure(
                fund_identifier=fund_data.get("fund_identifier", "unknown"),
                fund_type=fund_data.get("fund_type", "unknown"),
                current_pressure=0.0,
                pressure_trend="unknown",
                liquidity_coverage=0.0,
                stress_scenarios={},
                mitigation_capabilities={},
                timestamp=datetime.now()
            )
    
    def _calculate_securable_liquidity(self, fund_data: Dict[str, Any]) -> float:
        """Calculate securable liquidity from asset allocation"""
        try:
            asset_allocation = fund_data["asset_allocation"]
            total_assets = fund_data["total_assets"]
            
            # Calculate liquidity that can be secured (sold quickly)
            securable_liquidity = 0
            
            # Cash and cash equivalents
            securable_liquidity += asset_allocation.get("cash", 0) * total_assets
            
            # Treasury securities (assume 95% can be secured quickly)
            securable_liquidity += asset_allocation.get("treasuries", 0) * total_assets * 0.95
            
            # Investment grade bonds (assume 80% can be secured)
            securable_liquidity += asset_allocation.get("investment_grade_bonds", 0) * total_assets * 0.80
            
            # Large cap equities (assume 70% can be secured)
            securable_liquidity += asset_allocation.get("large_cap_equities", 0) * total_assets * 0.70
            
            return securable_liquidity / total_assets
            
        except Exception as e:
            logger.error(f"Error calculating securable liquidity: {str(e)}")
            return 0.5
    
    def _calculate_sales_capacity(self, fund_data: Dict[str, Any]) -> float:
        """Calculate maximum sales capacity under stress"""
        try:
            # Base on daily volume and liquidity profile
            daily_volume = fund_data["daily_volume"]
            base_capacity = daily_volume * 0.8  # Assume 80% of volume can be sold
            
            # Adjust for fund type
            fund_type = fund_data["fund_type"]
            if fund_type == "etf":
                base_capacity *= 1.5  # ETFs typically have higher liquidity
            elif fund_type == "hedge_fund":
                base_capacity *= 0.5  # Hedge funds often have lower daily liquidity
            
            # Adjust for market stress
            market_context = fund_data.get("market_context", {})
            stress_multiplier = 1 - (market_context.get("stress_index", 0) * 0.5)
            base_capacity *= max(0.3, stress_multiplier)
            
            return base_capacity / fund_data["total_assets"]
            
        except Exception as e:
            logger.error(f"Error calculating sales capacity: {str(e)}")
            return 0.01
    
    async def analyze_cash_flow_stress(self, fund_data: Dict[str, Any], time_horizon: int = 30) -> CashFlowStress:
        """Analyze cash flow stress over specified time horizon"""
        try:
            total_assets = fund_data["total_assets"]
            recent_redemptions = fund_data["recent_redemptions"]
            
            # Project cash inflows (fees, investment income)
            avg_redemption_rate = recent_redemptions["avg_daily_redemption_rate"]
            projected_cash_inflow = total_assets * avg_redemption_rate * time_horizon * 0.1  # Assume 10% of redemptions become inflows
            
            # Project cash outflows (redemptions + operating expenses)
            redemption_outflow = total_assets * avg_redemption_rate * time_horizon
            expense_ratio = fund_data.get("expense_ratio", 0.02)
            operating_expenses = total_assets * expense_ratio * (time_horizon / 365)
            
            projected_cash_outflow = redemption_outflow + operating_expenses
            
            # Calculate net cash flow
            net_cash_flow = projected_cash_inflow - projected_cash_outflow
            
            # Calculate liquidity buffer
            cash_position = fund_data["cash_position"]
            securable_liquidity = self._calculate_securable_liquidity(fund_data)
            total_liquidity_buffer = cash_position + (securable_liquidity * total_assets)
            
            # Stress multiplier based on market conditions
            market_context = fund_data.get("market_context", {})
            base_stress = market_context.get("stress_index", 0.3)
            stress_multiplier = 1 + (base_stress * 0.5)  # Increase stress by up to 50%
            
            # Risk factors
            risk_factors = []
            if fund_data["liquidity_ratio"] < 0.05:
                risk_factors.append("low_cash_ratio")
            if recent_redemptions["redemption_volatility"] > 0.01:
                risk_factors.append("high_redemption_volatility")
            if base_stress > 0.5:
                risk_factors.append("high_market_stress")
            if fund_data["liquidity_profile"] == "illiquid":
                risk_factors.append("illiquid_asset_allocation")
            
            return CashFlowStress(
                entity=fund_data["fund_identifier"],
                time_horizon=time_horizon,
                projected_cash_inflow=projected_cash_inflow,
                projected_cash_outflow=projected_cash_outflow,
                net_cash_flow=net_cash_flow,
                liquidity_buffer=total_liquidity_buffer,
                stress_multiplier=stress_multiplier,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"Error analyzing cash flow stress: {str(e)}")
            return CashFlowStress(
                entity=fund_data.get("fund_identifier", "unknown"),
                time_horizon=time_horizon,
                projected_cash_inflow=0,
                projected_cash_outflow=0,
                net_cash_flow=0,
                liquidity_buffer=0,
                stress_multiplier=1.0,
                risk_factors=["analysis_error"]
            )
    
    async def run_stress_scenarios(self, fund_data: Dict[str, Any]) -> Dict[str, float]:
        """Run various stress scenarios"""
        try:
            scenarios = {}
            
            # Scenario 1: Market Crash
            market_crash_impact = {
                "equity_decline": 0.30,
                "bond_spread_widening": 0.02,
                "liquidity_dry_up": 0.50
            }
            scenarios["market_crash"] = self._calculate_stress_impact(fund_data, market_crash_impact)
            
            # Scenario 2: Redemption Surge
            redemption_surge_impact = {
                "redemption_multiplier": 5.0,
                "capacity_reduction": 0.60,
                "liquidity_premium": 0.15
            }
            scenarios["redemption_surge"] = self._calculate_stress_impact(fund_data, redemption_surge_impact)
            
            # Scenario 3: Liquidity Crisis
            liquidity_crisis_impact = {
                "liquidity_bond_widen": 0.05,
                "equity_liquidity": 0.30,
                "alternative_liquidity": 0.10
            }
            scenarios["liquidity_crisis"] = self._calculate_stress_impact(fund_data, liquidity_crisis_impact)
            
            # Scenario 4: Combined Stress
            combined_stress_impact = {
                "equity_decline": 0.20,
                "redemption_multiplier": 3.0,
                "liquidity_bond_widen": 0.03,
                "capacity_reduction": 0.40
            }
            scenarios["combined_stress"] = self._calculate_stress_impact(fund_data, combined_stress_impact)
            
            # Scenario 5: Credit Event
            credit_event_impact = {
                "bond_spread_widening": 0.08,
                "credit_liquidity": 0.20,
                "flight_to_quality": 0.70
            }
            scenarios["credit_event"] = self._calculate_stress_impact(fund_data, credit_event_impact)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error running stress scenarios: {str(e)}")
            return {}
    
    def _calculate_stress_impact(self, fund_data: Dict[str, Any], stress_factors: Dict[str, float]) -> float:
        """Calculate the impact of stress factors on fund liquidity"""
        try:
            base_liquidity = fund_data["liquidity_ratio"]
            asset_allocation = fund_data["asset_allocation"]
            
            # Calculate impact on different asset classes
            impact_components = []
            
            # Equity impact
            equity_impact = (
                asset_allocation.get("large_cap_equities", 0) * stress_factors.get("equity_decline", 0) +
                asset_allocation.get("mid_cap_equities", 0) * stress_factors.get("equity_decline", 0) * 1.2 +
                asset_allocation.get("small_cap_equities", 0) * stress_factors.get("equity_decline", 0) * 1.5
            )
            impact_components.append(equity_impact * 0.4)  # 40% weight for equity impact
            
            # Bond impact
            bond_impact = (
                asset_allocation.get("investment_grade_bonds", 0) * stress_factors.get("bond_spread_widening", 0) +
                asset_allocation.get("high_yield_bonds", 0) * stress_factors.get("bond_spread_widening", 0) * 1.5
            )
            impact_components.append(bond_impact * 0.3)  # 30% weight for bond impact
            
            # Liquidity impact
            liquidity_impact = stress_factors.get("liquidity_dry_up", 0) * base_liquidity
            impact_components.append(liquidity_impact * 0.3)  # 30% weight for liquidity impact
            
            # Redemption multiplier impact
            if "redemption_multiplier" in stress_factors:
                redemption_impact = (stress_factors["redemption_multiplier"] - 1) * 0.1
                impact_components.append(redemption_impact)
            
            return sum(impact_components)
            
        except Exception as e:
            logger.error(f"Error calculating stress impact: {str(e)}")
            return 0.5
    
    async def detect_early_warning_indicators(self, fund_data: Dict[str, Any], 
                                            liquidity_metrics: List[LiquidityMetric]) -> List[Dict[str, Any]]:
        """Detect early warning indicators for redemption risk"""
        try:
            warnings = []
            
            # 1. Declining liquidity ratio
            current_ratio = fund_data["liquidity_ratio"]
            if current_ratio < 0.05:
                warnings.append({
                    "indicator": "critical_liquidity_ratio",
                    "severity": "critical",
                    "description": f"Liquidity ratio critically low: {current_ratio:.1%}",
                    "threshold": 0.05,
                    "current_value": current_ratio
                })
            elif current_ratio < 0.10:
                warnings.append({
                    "indicator": "low_liquidity_ratio",
                    "severity": "high",
                    "description": f"Liquidity ratio declining: {current_ratio:.1%}",
                    "threshold": 0.10,
                    "current_value": current_ratio
                })
            
            # 2. Accelerating redemption velocity
            redemption_trend = fund_data["recent_redemptions"]["trend_30d"]
            if redemption_trend > 0.002:  # 0.2% daily increase
                warnings.append({
                    "indicator": "accelerating_redemptions",
                    "severity": "high",
                    "description": f"Redemption velocity accelerating: {redemption_trend:.4f} daily increase",
                    "threshold": 0.002,
                    "current_value": redemption_trend
                })
            
            # 3. High redemption volatility
            redemption_volatility = fund_data["recent_redemptions"]["redemption_volatility"]
            if redemption_volatility > 0.015:  # 1.5% volatility
                warnings.append({
                    "indicator": "high_redemption_volatility",
                    "severity": "medium",
                    "description": f"High redemption volatility: {redemption_volatility:.1%}",
                    "threshold": 0.015,
                    "current_value": redemption_volatility
                })
            
            # 4. Metric-based warnings
            for metric in liquidity_metrics:
                if metric.status == "critical":
                    warnings.append({
                        "indicator": f"critical_{metric.metric_name}",
                        "severity": "critical",
                        "description": f"Critical level for {metric.metric_name}: {metric.value:.3f}",
                        "threshold": metric.threshold,
                        "current_value": metric.value
                    })
                elif metric.status == "warning":
                    warnings.append({
                        "indicator": f"warning_{metric.metric_name}",
                        "severity": "medium",
                        "description": f"Warning level for {metric.metric_name}: {metric.value:.3f}",
                        "threshold": metric.threshold,
                        "current_value": metric.value
                    })
            
            # 5. Asset allocation concentration
            asset_allocation = fund_data["asset_allocation"]
            max_allocation = max(asset_allocation.values())
            if max_allocation > 0.6:  # 60% concentration
                concentrated_asset = max(asset_allocation, key=asset_allocation.get)
                warnings.append({
                    "indicator": "asset_concentration",
                    "severity": "medium",
                    "description": f"High concentration in {concentrated_asset}: {max_allocation:.1%}",
                    "threshold": 0.60,
                    "current_value": max_allocation
                })
            
            # 6. Market stress indicators
            market_context = fund_data.get("market_context", {})
            market_stress = market_context.get("stress_index", 0)
            if market_stress > 0.7:
                warnings.append({
                    "indicator": "high_market_stress",
                    "severity": "high",
                    "description": f"Market stress elevated: {market_stress:.1%}",
                    "threshold": 0.70,
                    "current_value": market_stress
                })
            
            return warnings
            
        except Exception as e:
            logger.error(f"Error detecting early warning indicators: {str(e)}")
            return []
    
    def generate_mitigation_strategies(self, fund_data: Dict[str, Any], 
                                     risk_analysis: Dict[str, Any]) -> List[str]:
        """Generate liquidity risk mitigation strategies"""
        try:
            strategies = []
            
            # Get current liquidity profile
            liquidity_profile = fund_data["liquidity_profile"]
            liquidity_ratio = fund_data["liquidity_ratio"]
            
            # Base strategies
            if liquidity_ratio < 0.05:
                strategies.extend([
                    "Immediately halt new investments",
                    "Activate emergency liquidity lines",
                    "Consider selective asset sales",
                    "Implement redemption gates if applicable"
                ])
            elif liquidity_ratio < 0.10:
                strategies.extend([
                    "Increase cash allocation",
                    "Establish standby credit facilities",
                    "Review and optimize asset allocation"
                ])
            
            # Profile-specific strategies
            if liquidity_profile in ["illiquid", "stressed"]:
                strategies.extend([
                    "Diversify into more liquid assets",
                    "Establish securities lending programs",
                    "Consider tactical asset rebalancing"
                ])
            
            # Market condition strategies
            market_context = fund_data.get("market_context", {})
            market_stress = market_context.get("stress_index", 0)
            
            if market_stress > 0.6:
                strategies.extend([
                    "Reduce exposure to stressed markets",
                    "Increase allocation to high-quality liquid assets",
                    "Monitor counterparty exposures closely"
                ])
            
            # Fund type specific strategies
            fund_type = fund_data["fund_type"]
            if fund_type == "hedge_fund":
                strategies.extend([
                    "Leverage prime brokerage credit lines",
                    "Utilize derivatives for liquidity management",
                    "Consider side pocket arrangements for illiquid positions"
                ])
            elif fund_type == "mutual_fund":
                strategies.extend([
                    "Implement fair value pricing",
                    "Utilize inter-fund transfers",
                    "Consider temporary suspension mechanisms"
                ])
            elif fund_type == "etf":
                strategies.extend([
                    "Activate authorized participant facilities",
                    "Utilize creation/redemption mechanisms",
                    "Monitor premium/discount dynamics"
                ])
            
            # Risk-specific strategies
            redemption_pressure = risk_analysis.get("redemption_pressure", {})
            if redemption_pressure.get("current_pressure", 0) > 0.7:
                strategies.extend([
                    "Implement gradual redemption processing",
                    "Communicate proactively with investors",
                    "Consider temporary fees or penalties"
                ])
            
            # Stress scenario strategies
            stress_scenarios = risk_analysis.get("stress_scenarios", {})
            for scenario, impact in stress_scenarios.items():
                if impact > 0.6:
                    scenario_strategies = {
                        "market_crash": ["Activate crisis liquidity protocols", "Freeze non-essential trading"],
                        "redemption_surge": ["Implement redemption queues", "Prioritize strategic investors"],
                        "liquidity_crisis": ["Suspend new investments", "Liquidate liquid positions first"],
                        "combined_stress": ["Activate all available liquidity sources", "Implement emergency measures"],
                        "credit_event": ["Review credit exposures", "Implement credit risk limits"]
                    }
                    strategies.extend(scenario_strategies.get(scenario, []))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_strategies = []
            for strategy in strategies:
                if strategy not in seen:
                    seen.add(strategy)
                    unique_strategies.append(strategy)
            
            return unique_strategies[:10]  # Limit to top 10 strategies
            
        except Exception as e:
            logger.error(f"Error generating mitigation strategies: {str(e)}")
            return ["Unable to generate specific mitigation strategies"]
    
    async def generate_comprehensive_risk_analysis(self, fund_identifier: str, fund_type: str) -> RedemptionRiskAnalysis:
        """Generate comprehensive redemption risk analysis"""
        try:
            # Gather all analysis components
            fund_data = await self.fetch_fund_data(fund_identifier, fund_type)
            if not fund_data:
                return RedemptionRiskAnalysis(
                    entity=fund_identifier,
                    fund_type=fund_type,
                    timestamp=datetime.now(),
                    overall_risk_score=1.0,
                    risk_level="critical",
                    liquidity_profile="stressed",
                    redemption_pressure=RedemptionPressure(fund_identifier, fund_type, 0, "unknown", 0, {}, {}, datetime.now()),
                    cash_flow_stress=CashFlowStress(fund_identifier, 30, 0, 0, 0, 0, 1.0, []),
                    liquidity_metrics=[],
                    stress_scenarios={},
                    early_warning_indicators=[],
                    mitigation_strategies=["No data available"],
                    recommendations=["Unable to generate analysis due to data error"]
                )
            
            # Calculate components
            liquidity_metrics = await self.calculate_liquidity_metrics(fund_data)
            redemption_pressure = await self.assess_redemption_pressure(fund_data)
            cash_flow_stress = await self.analyze_cash_flow_stress(fund_data)
            stress_scenarios = await self.run_stress_scenarios(fund_data)
            early_warnings = await self.detect_early_warning_indicators(fund_data, liquidity_metrics)
            
            # Calculate overall risk score
            risk_components = []
            
            # Liquidity ratio component
            liquidity_ratio = fund_data["liquidity_ratio"]
            liquidity_score = max(0, min(1, (0.2 - liquidity_ratio) / 0.2)) if liquidity_ratio < 0.2 else 0
            risk_components.append(liquidity_score * 0.3)
            
            # Redemption pressure component
            pressure_score = redemption_pressure.current_pressure
            risk_components.append(pressure_score * 0.25)
            
            # Cash flow stress component
            cash_stress_score = max(0, min(1, abs(cash_flow_stress.net_cash_flow) / (fund_data["total_assets"] * 0.1)))
            risk_components.append(cash_stress_score * 0.2)
            
            # Early warning indicators component
            critical_warnings = len([w for w in early_warnings if w["severity"] == "critical"])
            warning_score = min(1, critical_warnings * 0.3 + len(early_warnings) * 0.1)
            risk_components.append(warning_score * 0.15)
            
            # Stress scenario component
            avg_stress_impact = np.mean(list(stress_scenarios.values())) if stress_scenarios else 0
            risk_components.append(avg_stress_impact * 0.1)
            
            overall_risk_score = sum(risk_components)
            
            # Determine risk level
            if overall_risk_score > 0.8:
                risk_level = "critical"
            elif overall_risk_score > 0.6:
                risk_level = "very_high"
            elif overall_risk_score > 0.4:
                risk_level = "high"
            elif overall_risk_score > 0.2:
                risk_level = "moderate"
            else:
                risk_level = "low"
            
            # Compile analysis data
            risk_analysis_data = {
                "redemption_pressure": redemption_pressure.__dict__,
                "cash_flow_stress": cash_flow_stress.__dict__,
                "stress_scenarios": stress_scenarios
            }
            
            # Generate mitigation strategies
            mitigation_strategies = self.generate_mitigation_strategies(fund_data, risk_analysis_data)
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(
                overall_risk_score, risk_level, early_warnings, mitigation_strategies
            )
            
            analysis = RedemptionRiskAnalysis(
                entity=fund_identifier,
                fund_type=fund_type,
                timestamp=datetime.now(),
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                liquidity_profile=fund_data["liquidity_profile"],
                redemption_pressure=redemption_pressure,
                cash_flow_stress=cash_flow_stress,
                liquidity_metrics=liquidity_metrics,
                stress_scenarios=stress_scenarios,
                early_warning_indicators=early_warnings,
                mitigation_strategies=mitigation_strategies,
                recommendations=recommendations
            )
            
            logger.info(f"Generated redemption risk analysis for {fund_identifier}: "
                       f"Risk={risk_level}, Score={overall_risk_score:.3f}, "
                       f"Warnings={len(early_warnings)}, Strategies={len(mitigation_strategies)}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating comprehensive risk analysis for {fund_identifier}: {str(e)}")
            return RedemptionRiskAnalysis(
                entity=fund_identifier,
                fund_type=fund_type,
                timestamp=datetime.now(),
                overall_risk_score=1.0,
                risk_level="critical",
                liquidity_profile="stressed",
                redemption_pressure=RedemptionPressure(fund_identifier, fund_type, 0, "unknown", 0, {}, {}, datetime.now()),
                cash_flow_stress=CashFlowStress(fund_identifier, 30, 0, 0, 0, 0, 1.0, []),
                liquidity_metrics=[],
                stress_scenarios={},
                early_warning_indicators=[],
                mitigation_strategies=["Analysis failed"],
                recommendations=["Unable to generate analysis due to system error"]
            )
    
    def _generate_risk_recommendations(self, risk_score: float, risk_level: str, 
                                     early_warnings: List[Dict[str, Any]], 
                                     strategies: List[str]) -> List[str]:
        """Generate risk-based recommendations"""
        try:
            recommendations = []
            
            # Risk level recommendations
            if risk_score > 0.8:
                recommendations.extend([
                    "IMMEDIATE ACTION REQUIRED: Critical redemption risk detected",
                    "Activate emergency liquidity protocols immediately",
                    "Consider temporary suspension of redemptions if legally permissible"
                ])
            elif risk_score > 0.6:
                recommendations.extend([
                    "High redemption risk - implement enhanced monitoring",
                    "Prepare contingency liquidity plans",
                    "Increase communication with key stakeholders"
                ])
            elif risk_score > 0.4:
                recommendations.extend([
                    "Moderate redemption risk - maintain enhanced vigilance",
                    "Review and update liquidity management policies"
                ])
            else:
                recommendations.append("Current risk levels are manageable - maintain standard monitoring")
            
            # Warning-specific recommendations
            for warning in early_warnings:
                if warning["indicator"] == "critical_liquidity_ratio":
                    recommendations.append("URGENT: Increase cash position to minimum 10%")
                elif warning["indicator"] == "accelerating_redemptions":
                    recommendations.append("Monitor redemption patterns closely and prepare for surge")
                elif warning["indicator"] == "high_market_stress":
                    recommendations.append("Review market exposure and consider defensive positioning")
            
            # Strategy-based recommendations
            if len(strategies) > 5:
                recommendations.append("Multiple mitigation strategies required - prioritize implementation")
            
            # Action priorities
            if risk_level == "critical":
                recommendations.insert(0, "Priority 1: Immediate liquidity assessment and mitigation")
            elif risk_level in ["very_high", "high"]:
                recommendations.insert(0, "Priority 2: Enhanced monitoring and preparation")
            
            # General recommendations
            if not any("communication" in r.lower() for r in recommendations):
                recommendations.append("Maintain transparent communication with investors and stakeholders")
            
            return recommendations[:8]  # Limit to top 8 recommendations
            
        except Exception as e:
            logger.error(f"Error generating risk recommendations: {str(e)}")
            return ["Unable to generate specific risk recommendations"]
    
    async def monitor_risk_alerts(self, fund_list: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """Monitor multiple funds for risk alerts"""
        try:
            alerts = {}
            
            for fund_id, fund_type in fund_list:
                fund_alerts = []
                
                # Generate comprehensive analysis
                analysis = await self.generate_comprehensive_risk_analysis(fund_id, fund_type)
                
                # Overall risk alerts
                if analysis.overall_risk_score > 0.8:
                    fund_alerts.append(f"CRITICAL_RISK: {fund_id} risk score {analysis.overall_risk_score:.3f}")
                elif analysis.overall_risk_score > 0.6:
                    fund_alerts.append(f"HIGH_RISK: {fund_id} risk score {analysis.overall_risk_score:.3f}")
                
                # Liquidity profile alerts
                if analysis.liquidity_profile in ["illiquid", "stressed"]:
                    fund_alerts.append(f"LIQUIDITY_ALERT: {fund_id} liquidity profile: {analysis.liquidity_profile}")
                
                # Redemption pressure alerts
                if analysis.redemption_pressure.current_pressure > 0.8:
                    fund_alerts.append(f"HIGH_PRESSURE: {fund_id} redemption pressure {analysis.redemption_pressure.current_pressure:.3f}")
                elif analysis.redemption_pressure.pressure_trend == "increasing":
                    fund_alerts.append(f"PRESSURE_TREND: {fund_id} redemption pressure increasing")
                
                # Critical warnings
                critical_warnings = [w for w in analysis.early_warning_indicators if w["severity"] == "critical"]
                for warning in critical_warnings:
                    fund_alerts.append(f"CRITICAL_WARNING: {fund_id} {warning['indicator']}: {warning['description']}")
                
                # Stress scenario alerts
                for scenario, impact in analysis.stress_scenarios.items():
                    if impact > 0.7:
                        fund_alerts.append(f"STRESS_ALERT: {fund_id} {scenario} impact: {impact:.3f}")
                
                # Cash flow alerts
                if analysis.cash_flow_stress.net_cash_flow < 0:
                    fund_alerts.append(f"CASH_FLOW_NEGATIVE: {fund_id} projected negative cash flow")
                
                if fund_alerts:
                    alerts[fund_id] = fund_alerts
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring risk alerts: {str(e)}")
            return {}
    
    async def compare_fund_risks(self, fund_list: List[Tuple[str, str]]) -> Dict[str, Dict[str, Any]]:
        """Compare redemption risks across multiple funds"""
        try:
            comparisons = {}
            
            for fund_id, fund_type in fund_list:
                analysis = await self.generate_comprehensive_risk_analysis(fund_id, fund_type)
                
                comparisons[fund_id] = {
                    "fund_type": fund_type,
                    "overall_risk_score": analysis.overall_risk_score,
                    "risk_level": analysis.risk_level,
                    "liquidity_profile": analysis.liquidity_profile,
                    "redemption_pressure": analysis.redemption_pressure.current_pressure,
                    "liquidity_ratio": next((m.value for m in analysis.liquidity_metrics if m.metric_name == "liquidity_ratio"), 0),
                    "cash_buffer": next((m.value for m in analysis.liquidity_metrics if m.metric_name == "cash_buffer"), 0),
                    "warning_count": len(analysis.early_warning_indicators),
                    "critical_warning_count": len([w for w in analysis.early_warning_indicators if w["severity"] == "critical"]),
                    "strategy_count": len(analysis.mitigation_strategies),
                    "max_stress_impact": max(analysis.stress_scenarios.values()) if analysis.stress_scenarios else 0
                }
            
            # Calculate relative rankings
            metrics_to_rank = ["overall_risk_score", "redemption_pressure", "warning_count", "max_stress_impact"]
            
            for metric in metrics_to_rank:
                values = [comp[metric] for comp in comparisons.values() if isinstance(comp[metric], (int, float))]
                if values:
                    sorted_values = sorted(values, reverse=True)
                    for fund_id in comparisons:
                        if isinstance(comparisons[fund_id][metric], (int, float)):
                            comparisons[fund_id][f"{metric}_rank"] = sorted_values.index(comparisons[fund_id][metric]) + 1
                            comparisons[fund_id][f"{metric}_percentile"] = (sorted_values.index(comparisons[fund_id][metric]) + 1) / len(values)
            
            return comparisons
            
        except Exception as e:
            logger.error(f"Error comparing fund risks: {str(e)}")
            return {}
    
    async def export_risk_analysis(self, fund_identifier: str, fund_type: str, format_type: str = "json") -> str:
        """Export risk analysis to file"""
        try:
            analysis = await self.generate_comprehensive_risk_analysis(fund_identifier, fund_type)
            
            if format_type.lower() == "json":
                import json
                export_data = {
                    "entity": analysis.entity,
                    "fund_type": analysis.fund_type,
                    "timestamp": analysis.timestamp.isoformat(),
                    "overall_risk_score": analysis.overall_risk_score,
                    "risk_level": analysis.risk_level,
                    "liquidity_profile": analysis.liquidity_profile,
                    "redemption_pressure": {
                        "current_pressure": analysis.redemption_pressure.current_pressure,
                        "pressure_trend": analysis.redemption_pressure.pressure_trend,
                        "liquidity_coverage": analysis.redemption_pressure.liquidity_coverage,
                        "stress_scenarios": analysis.redemption_pressure.stress_scenarios,
                        "mitigation_capabilities": analysis.redemption_pressure.mitigation_capabilities
                    },
                    "cash_flow_stress": {
                        "time_horizon": analysis.cash_flow_stress.time_horizon,
                        "projected_cash_inflow": analysis.cash_flow_stress.projected_cash_inflow,
                        "projected_cash_outflow": analysis.cash_flow_stress.projected_cash_outflow,
                        "net_cash_flow": analysis.cash_flow_stress.net_cash_flow,
                        "liquidity_buffer": analysis.cash_flow_stress.liquidity_buffer,
                        "stress_multiplier": analysis.cash_flow_stress.stress_multiplier,
                        "risk_factors": analysis.cash_flow_stress.risk_factors
                    },
                    "liquidity_metrics": [
                        {
                            "metric_name": m.metric_name,
                            "value": m.value,
                            "threshold": m.threshold,
                            "status": m.status
                        }
                        for m in analysis.liquidity_metrics
                    ],
                    "stress_scenarios": analysis.stress_scenarios,
                    "early_warning_indicators": analysis.early_warning_indicators,
                    "mitigation_strategies": analysis.mitigation_strategies,
                    "recommendations": analysis.recommendations
                }
                
                filename = f"redemption_risk_analysis_{fund_identifier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = f"/workspace/mpe/exports/{filename}"
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return filepath
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting risk analysis for {fund_identifier}: {str(e)}")
            return ""

# Main execution function
async def main():
    """Main execution function for redemption risk monitoring"""
    monitor = RedemptionRiskMonitor()
    
    # Test funds
    test_funds = [
        ("LARGE_MUTUAL_FUND", "mutual_fund"),
        ("TECH_ETF", "etf"),
        ("HEDGE_FUND_ALPHA", "hedge_fund"),
        ("PENSION_FUND_X", "pension_fund")
    ]
    
    logger.info("Starting Redemption Risk Monitor analysis...")
    
    # Test comprehensive analysis
    for fund_id, fund_type in test_funds[:2]:  # Test first 2 funds
        logger.info(f"\n=== Redemption Risk Analysis for {fund_id} ({fund_type}) ===")
        
        analysis = await monitor.generate_comprehensive_risk_analysis(fund_id, fund_type)
        
        logger.info(f"Overall Risk Score: {analysis.overall_risk_score:.3f}")
        logger.info(f"Risk Level: {analysis.risk_level}")
        logger.info(f"Liquidity Profile: {analysis.liquidity_profile}")
        logger.info(f"Redemption Pressure: {analysis.redemption_pressure.current_pressure:.3f}")
        logger.info(f"Early Warnings: {len(analysis.early_warning_indicators)}")
        logger.info(f"Mitigation Strategies: {len(analysis.mitigation_strategies)}")
        
        logger.info("\nKey Metrics:")
        for metric in analysis.liquidity_metrics[:3]:  # Show first 3 metrics
            logger.info(f"  {metric.metric_name}: {metric.value:.3f} ({metric.status})")
        
        logger.info("\nRecommendations:")
        for rec in analysis.recommendations[:3]:  # Show first 3 recommendations
            logger.info(f"  - {rec}")
    
    # Test risk monitoring alerts
    logger.info("\n=== Risk Monitoring Alerts ===")
    alerts = await monitor.monitor_risk_alerts(test_funds)
    
    for fund_id, fund_alerts in alerts.items():
        if fund_alerts:
            logger.info(f"{fund_id}: {len(fund_alerts)} alerts")
            for alert in fund_alerts[:2]:  # Show first 2 alerts
                logger.info(f"  - {alert}")
    
    # Test fund comparison
    logger.info("\n=== Fund Risk Comparison ===")
    comparisons = await monitor.compare_fund_risks(test_funds[:3])  # Compare first 3 funds
    
    for fund_id, data in comparisons.items():
        logger.info(f"{fund_id}:")
        logger.info(f"  Risk Rank: {data.get('overall_risk_score_rank', 'N/A')}")
        logger.info(f"  Risk Level: {data.get('risk_level', 'unknown')}")
        logger.info(f"  Liquidity Profile: {data.get('liquidity_profile', 'unknown')}")
        logger.info(f"  Critical Warnings: {data.get('critical_warning_count', 0)}")
    
    logger.info("Redemption Risk Monitor analysis completed successfully")

if __name__ == "__main__":
    asyncio.run(main())