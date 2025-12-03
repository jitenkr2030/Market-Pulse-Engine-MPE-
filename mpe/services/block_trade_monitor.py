"""
Module 20: Block Trade Monitor
Author: MiniMax Agent
Date: 2025-12-02

Advanced block trade monitoring and analysis system.
Provides real-time tracking of large institutional trades, market impact assessment,
and liquidity analysis for institutional execution strategies.
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
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeType(Enum):
    """Types of block trades"""
    BUY = "buy"
    SELL = "sell"
    CROSS = "cross"
    NEGOTIATED = "negotiated"
    HASHED = "hashed"

class TradeSize(Enum):
    """Trade size categories"""
    SMALL = "small"  # < 10k shares
    MEDIUM = "medium"  # 10k-100k shares
    LARGE = "large"  # 100k-500k shares
    BLOCK = "block"  # 500k-2M shares
    MEGA = "mega"  # > 2M shares

class ExecutionQuality(Enum):
    """Execution quality assessment"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"

@dataclass
class BlockTrade:
    """Block trade data structure"""
    symbol: str
    timestamp: datetime
    trade_type: str
    size: float
    price: float
    venue: str
    execution_quality: str
    market_impact: float
    liquidity_score: float
    participation_rate: float
    confidence_score: float
    id: str = field(default_factory=lambda: f"BT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}")

@dataclass
class TradeCluster:
    """Cluster of related block trades"""
    cluster_id: str
    trades: List[BlockTrade]
    avg_size: float
    total_volume: float
    price_impact: float
    liquidity_characteristics: Dict[str, float]
    execution_pattern: str
    institutional_signature: str

@dataclass
class BlockTradeAnalysis:
    """Block trade analysis results"""
    symbol: str
    timestamp: datetime
    total_block_volume: float
    block_trade_count: int
    avg_trade_size: float
    liquidity_supplied: float
    market_impact_score: float
    participation_rate: float
    execution_quality_score: float
    institutional_activity_level: str
    clusters: List[TradeCluster]
    anomalies: List[Dict[str, Any]]
    patterns: Dict[str, Any]
    recommendations: List[str]

class BlockTradeMonitor:
    """
    Advanced Block Trade Monitor
    
    Monitors, analyzes, and provides intelligence on large block trades
    to support institutional execution strategies and market microstructure analysis.
    """
    
    def __init__(self):
        self.name = "Block Trade Monitor"
        self.version = "1.0.0"
        self.cache = {}
        self.cache_ttl = 180  # 3 minutes
        
        # Block trade thresholds (shares)
        self.size_thresholds = {
            "small": 10000,
            "medium": 100000,
            "large": 500000,
            "block": 2000000,
            "mega": 10000000
        }
        
        # Known block trade venues
        self.block_trade_venues = {
            "POSIT": {"type": "crossing_network", "liquidity_factor": 0.8},
            "DARK_AGGREGATOR": {"type": "dark_pool", "liquidity_factor": 0.9},
            "INSTITUTIONAL_X": {"type": "institutional", "liquidity_factor": 0.7},
            "BLOCK_EXCHANGE": {"type": "exchange", "liquidity_factor": 0.6},
            "PRIVATE_MARKET": {"type": "negotiated", "liquidity_factor": 0.5},
            "CROSSING_SERVICE": {"type": "crossing", "liquidity_factor": 0.8}
        }
        
        # Historical patterns database
        self.institutional_patterns = {
            "momentum_accumulation": {"signature": "increasing_size_uptrend", "duration_hours": 4},
            "value_accumulation": {"signature": "steady_size_sideways", "duration_hours": 8},
            "distribution": {"signature": "decreasing_size_downtrend", "duration_hours": 2},
            "hedge_flow": {"signature": "mixed_size_volatile", "duration_hours": 1},
            "rebalance": {"signature": "balanced_bidirectional", "duration_hours": 6}
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
    
    async def fetch_recent_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch recent block trade data for symbol"""
        try:
            cache_key = f"recent_trades_{symbol}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # Fetch underlying market data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                hist = ticker.history(period="6h", interval="5m")
            
            # Simulate block trade data (real implementation would use proprietary feeds)
            block_trades = self._simulate_block_trades(hist, symbol)
            
            await self._set_cache_data(cache_key, block_trades)
            return block_trades
            
        except Exception as e:
            logger.error(f"Error fetching recent trades for {symbol}: {str(e)}")
            return []
    
    def _simulate_block_trades(self, hist: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """Simulate realistic block trade data"""
        if hist.empty:
            return []
        
        trades = []
        base_volume = hist['Volume'].mean()
        current_price = hist['Close'].iloc[-1]
        
        # Generate 5-20 block trades per day
        num_trades = np.random.randint(5, 21)
        
        for i in range(num_trades):
            # Random timestamp within the trading day
            trade_time = hist.index[np.random.randint(0, len(hist))]
            
            # Determine trade characteristics
            size_category = np.random.choice(
                list(self.size_thresholds.keys()),
                p=[0.4, 0.3, 0.2, 0.08, 0.02]
            )
            
            if size_category == "small":
                size = np.random.uniform(5000, self.size_thresholds["medium"])
            elif size_category == "medium":
                size = np.random.uniform(self.size_thresholds["medium"], self.size_thresholds["large"])
            elif size_category == "large":
                size = np.random.uniform(self.size_thresholds["large"], self.size_thresholds["block"])
            elif size_category == "block":
                size = np.random.uniform(self.size_thresholds["block"], self.size_thresholds["mega"])
            else:  # mega
                size = np.random.uniform(self.size_thresholds["mega"], self.size_thresholds["mega"] * 2)
            
            # Price impact calculation
            size_ratio = size / (base_volume * 1000)  # Normalize by avg volume
            base_impact = size_ratio * 0.02  # Base impact formula
            
            # Random market impact around base
            impact = np.random.normal(0, base_impact * 0.5)
            impact = max(-0.05, min(0.05, impact))  # Cap at ±5%
            
            trade_price = current_price * (1 + impact)
            
            # Select venue based on size
            if size > self.size_thresholds["block"]:
                venue = np.random.choice(["PRIVATE_MARKET", "INSTITUTIONAL_X"], p=[0.6, 0.4])
            elif size > self.size_thresholds["large"]:
                venue = np.random.choice(["POSIT", "DARK_AGGREGATOR"], p=[0.7, 0.3])
            else:
                venue = np.random.choice(list(self.block_trade_venues.keys()))
            
            # Determine execution quality
            liquidity_score = self.block_trade_venues.get(venue, {}).get("liquidity_factor", 0.5)
            quality = self._determine_execution_quality(liquidity_score, size_ratio, impact)
            
            trade = {
                "symbol": symbol,
                "timestamp": trade_time,
                "trade_type": np.random.choice(["buy", "sell", "cross"], p=[0.45, 0.45, 0.1]),
                "size": size,
                "price": trade_price,
                "venue": venue,
                "execution_quality": quality,
                "market_impact": impact,
                "liquidity_score": liquidity_score,
                "participation_rate": min(size_ratio * 100, 25),  # Cap at 25%
                "confidence_score": np.random.uniform(0.6, 0.95)
            }
            
            trades.append(trade)
        
        # Sort by timestamp
        trades.sort(key=lambda x: x["timestamp"])
        return trades
    
    def _determine_execution_quality(self, liquidity_score: float, size_ratio: float, impact: float) -> str:
        """Determine execution quality based on metrics"""
        try:
            # Quality factors
            liquidity_factor = liquidity_score
            size_penalty = min(size_ratio * 5, 0.3)  # Penalty for very large trades
            impact_penalty = abs(impact) * 10  # Penalty for high impact
            
            # Calculate composite quality score
            quality_score = liquidity_factor - size_penalty - impact_penalty
            
            if quality_score > 0.7:
                return "excellent"
            elif quality_score > 0.5:
                return "good"
            elif quality_score > 0.3:
                return "fair"
            elif quality_score > 0.1:
                return "poor"
            else:
                return "very_poor"
                
        except Exception as e:
            logger.error(f"Error determining execution quality: {str(e)}")
            return "fair"
    
    def _classify_trade_size(self, size: float) -> str:
        """Classify trade size category"""
        if size < self.size_thresholds["small"]:
            return "small"
        elif size < self.size_thresholds["medium"]:
            return "medium"
        elif size < self.size_thresholds["large"]:
            return "large"
        elif size < self.size_thresholds["block"]:
            return "block"
        else:
            return "mega"
    
    async def analyze_trade_clusters(self, symbol: str) -> List[TradeCluster]:
        """Analyze clusters of related block trades"""
        try:
            trades_data = await self.fetch_recent_trades(symbol)
            if not trades_data:
                return []
            
            # Convert to DataFrame for clustering
            df = pd.DataFrame(trades_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create features for clustering
            features = []
            for _, trade in df.iterrows():
                feature_vector = [
                    trade['size'],
                    trade['market_impact'],
                    trade['liquidity_score'],
                    trade['timestamp'].hour + trade['timestamp'].minute/60,  # Time of day
                    trade['participation_rate']
                ]
                features.append(feature_vector)
            
            if len(features) < 2:
                return []
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Perform DBSCAN clustering
            eps = 0.5
            min_samples = 2
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features_scaled)
            
            # Group trades by cluster
            clusters = {}
            for i, label in enumerate(clustering.labels_):
                if label == -1:  # Noise
                    continue
                
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(df.iloc[i])
            
            # Create TradeCluster objects
            trade_clusters = []
            for cluster_id, cluster_trades in clusters.items():
                cluster_df = pd.DataFrame(cluster_trades)
                
                # Calculate cluster metrics
                avg_size = cluster_df['size'].mean()
                total_volume = cluster_df['size'].sum()
                price_impact = cluster_df['market_impact'].mean()
                
                # Determine execution pattern
                execution_pattern = self._determine_execution_pattern(cluster_df)
                
                # Calculate institutional signature
                institutional_signature = self._identify_institutional_signature(cluster_df)
                
                # Liquidity characteristics
                liquidity_chars = {
                    "avg_liquidity_score": cluster_df['liquidity_score'].mean(),
                    "participation_rate": cluster_df['participation_rate'].mean(),
                    "venue_diversity": cluster_df['venue'].nunique(),
                    "quality_consistency": 1 - cluster_df['execution_quality'].map(
                        {"excellent": 0, "good": 0.2, "fair": 0.5, "poor": 0.8, "very_poor": 1}
                    ).std()
                }
                
                # Convert trades to BlockTrade objects
                block_trades = []
                for _, trade_data in cluster_df.iterrows():
                    block_trade = BlockTrade(
                        symbol=trade_data['symbol'],
                        timestamp=trade_data['timestamp'],
                        trade_type=trade_data['trade_type'],
                        size=trade_data['size'],
                        price=trade_data['price'],
                        venue=trade_data['venue'],
                        execution_quality=trade_data['execution_quality'],
                        market_impact=trade_data['market_impact'],
                        liquidity_score=trade_data['liquidity_score'],
                        participation_rate=trade_data['participation_rate'],
                        confidence_score=trade_data['confidence_score']
                    )
                    block_trades.append(block_trade)
                
                cluster = TradeCluster(
                    cluster_id=f"cluster_{cluster_id}",
                    trades=block_trades,
                    avg_size=avg_size,
                    total_volume=total_volume,
                    price_impact=price_impact,
                    liquidity_characteristics=liquidity_chars,
                    execution_pattern=execution_pattern,
                    institutional_signature=institutional_signature
                )
                
                trade_clusters.append(cluster)
            
            return trade_clusters
            
        except Exception as e:
            logger.error(f"Error analyzing trade clusters for {symbol}: {str(e)}")
            return []
    
    def _determine_execution_pattern(self, cluster_df: pd.DataFrame) -> str:
        """Determine execution pattern for a cluster"""
        try:
            # Sort by timestamp
            cluster_df = cluster_df.sort_values('timestamp')
            
            # Calculate trends
            size_trend = np.polyfit(range(len(cluster_df)), cluster_df['size'], 1)[0]
            impact_trend = np.polyfit(range(len(cluster_df)), cluster_df['market_impact'], 1)[0]
            
            # Determine pattern
            if size_trend > 1000 and impact_trend < 0:
                return "momentum_accumulation"
            elif abs(size_trend) < 500 and abs(impact_trend) < 0.001:
                return "steady_liquidity"
            elif size_trend < -1000:
                return "distribution"
            elif abs(impact_trend) > 0.005:
                return "impact_driven"
            else:
                return "mixed_pattern"
                
        except Exception as e:
            logger.error(f"Error determining execution pattern: {str(e)}")
            return "unknown"
    
    def _identify_institutional_signature(self, cluster_df: pd.DataFrame) -> str:
        """Identify institutional trading signature"""
        try:
            # Analyze characteristics
            avg_size = cluster_df['size'].mean()
            venue_diversity = cluster_df['venue'].nunique()
            size_consistency = 1 - (cluster_df['size'].std() / cluster_df['size'].mean())
            
            # Determine signature based on characteristics
            if avg_size > self.size_thresholds["block"] and venue_diversity < 3:
                return "traditional_institutional"
            elif avg_size > self.size_thresholds["large"] and venue_diversity > 3:
                return "sophisticated_institutional"
            elif avg_size < self.size_thresholds["medium"] and venue_diversity > 4:
                return "分散_institutional"
            elif size_consistency > 0.7:
                return "systematic_institutional"
            else:
                return "opportunistic_institutional"
                
        except Exception as e:
            logger.error(f"Error identifying institutional signature: {str(e)}")
            return "unknown"
    
    async def detect_anomalies(self, symbol: str) -> List[Dict[str, Any]]:
        """Detect anomalous block trade patterns"""
        try:
            trades_data = await self.fetch_recent_trades(symbol)
            if not trades_data:
                return []
            
            anomalies = []
            
            # Statistical anomaly detection
            df = pd.DataFrame(trades_data)
            
            # Size anomaly detection
            size_z_scores = np.abs(stats.zscore(df['size']))
            size_anomalies = df[size_z_scores > 2.5]
            
            for _, trade in size_anomalies.iterrows():
                anomalies.append({
                    "type": "size_anomaly",
                    "severity": "high" if size_z_scores[trade.name] > 3 else "medium",
                    "trade": trade.to_dict(),
                    "description": f"Unusually large trade: {trade['size']:,.0f} shares",
                    "timestamp": trade['timestamp']
                })
            
            # Impact anomaly detection
            impact_z_scores = np.abs(stats.zscore(df['market_impact']))
            impact_anomalies = df[impact_z_scores > 2.5]
            
            for _, trade in impact_anomalies.iterrows():
                anomalies.append({
                    "type": "impact_anomaly",
                    "severity": "high" if abs(trade['market_impact']) > 0.03 else "medium",
                    "trade": trade.to_dict(),
                    "description": f"Unusual market impact: {trade['market_impact']:.4f}",
                    "timestamp": trade['timestamp']
                })
            
            # Timing anomaly detection
            cluster_analysis = await self.analyze_trade_clusters(symbol)
            
            for cluster in cluster_analysis:
                # Check for clustered trading
                if len(cluster.trades) > 3 and cluster.price_impact > 0.01:
                    anomalies.append({
                        "type": "timing_anomaly",
                        "severity": "medium",
                        "cluster_id": cluster.cluster_id,
                        "description": f"High-impact clustered trading: {len(cluster.trades)} trades",
                        "trades": [t.__dict__ for t in cluster.trades]
                    })
            
            # Venue concentration anomaly
            venue_counts = df['venue'].value_counts()
            max_venue_share = venue_counts.iloc[0] / len(df) if len(venue_counts) > 0 else 0
            
            if max_venue_share > 0.8:
                dominant_venue = venue_counts.index[0]
                anomalies.append({
                    "type": "venue_concentration",
                    "severity": "medium",
                    "venue": dominant_venue,
                    "concentration": max_venue_share,
                    "description": f"High venue concentration: {max_venue_share:.1%} at {dominant_venue}"
                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies for {symbol}: {str(e)}")
            return []
    
    async def calculate_market_impact(self, symbol: str, trade_size: float) -> Dict[str, Any]:
        """Calculate expected market impact for block trade"""
        try:
            trades_data = await self.fetch_recent_trades(symbol)
            hist = (await self._get_cached_data(f"hist_{symbol}")) or []
            
            if not trades_data:
                return {}
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1h")
            current_price = hist['Close'].iloc[-1] if not hist.empty else 0
            avg_volume = hist['Volume'].mean() if not hist.empty else 1000000
            
            # Historical impact analysis
            historical_trades = pd.DataFrame(trades_data)
            size_impact_correlation = historical_trades['size'].corr(historical_trades['market_impact'])
            
            # Calculate impact based on trade size and market conditions
            size_ratio = trade_size / avg_volume if avg_volume > 0 else 1.0
            
            # Base impact calculation
            if size_ratio < 0.01:  # Small trade
                base_impact = 0.0001
            elif size_ratio < 0.05:  # Medium trade
                base_impact = 0.0005
            elif size_ratio < 0.1:  # Large trade
                base_impact = 0.001
            elif size_ratio < 0.2:  # Block trade
                base_impact = 0.002
            else:  # Mega trade
                base_impact = 0.005
            
            # Adjust for correlation
            if not np.isnan(size_impact_correlation):
                correlation_factor = 1 + (size_impact_correlation * 0.2)
            else:
                correlation_factor = 1.0
            
            expected_impact = base_impact * correlation_factor
            
            # Calculate confidence intervals
            impact_std = historical_trades['market_impact'].std() if not historical_trades.empty else 0.001
            confidence_interval = {
                "lower": max(0, expected_impact - 1.96 * impact_std),
                "upper": expected_impact + 1.96 * impact_std
            }
            
            # Execution scenarios
            scenarios = {
                "immediate": {
                    "expected_impact": expected_impact * 1.5,
                    "execution_time": 1,  # minutes
                    "probability": 0.9,
                    "description": "Immediate market execution"
                },
                "auction": {
                    "expected_impact": expected_impact * 0.7,
                    "execution_time": 15,  # minutes
                    "probability": 0.8,
                    "description": "Closing auction execution"
                },
                "dark_pool": {
                    "expected_impact": expected_impact * 0.4,
                    "execution_time": 60,  # minutes
                    "probability": 0.6,
                    "description": "Dark pool execution"
                },
                "twap": {
                    "expected_impact": expected_impact * 0.3,
                    "execution_time": 240,  # minutes
                    "probability": 0.95,
                    "description": "TWAP execution over 4 hours"
                }
            }
            
            return {
                "symbol": symbol,
                "trade_size": trade_size,
                "current_price": current_price,
                "avg_daily_volume": avg_volume,
                "size_ratio": size_ratio,
                "expected_impact": expected_impact,
                "confidence_interval": confidence_interval,
                "size_impact_correlation": size_impact_correlation,
                "execution_scenarios": scenarios,
                "recommendation": self._recommend_execution_strategy(scenarios, trade_size),
                "risk_factors": self._assess_execution_risk_factors(trade_size, expected_impact)
            }
            
        except Exception as e:
            logger.error(f"Error calculating market impact for {symbol}: {str(e)}")
            return {}
    
    def _recommend_execution_strategy(self, scenarios: Dict[str, Any], trade_size: float) -> str:
        """Recommend optimal execution strategy"""
        try:
            # Consider trade size in recommendation
            if trade_size > self.size_thresholds["block"]:
                # Large trades favor dark pool and TWAP
                if scenarios["dark_pool"]["probability"] > 0.7:
                    return "dark_pool"
                else:
                    return "twap"
            elif trade_size > self.size_thresholds["large"]:
                # Medium trades can use auction or dark pool
                if scenarios["auction"]["probability"] > 0.8:
                    return "auction"
                else:
                    return "dark_pool"
            else:
                # Small trades can execute immediately
                return "immediate"
                
        except Exception as e:
            logger.error(f"Error recommending execution strategy: {str(e)}")
            return "immediate"
    
    def _assess_execution_risk_factors(self, trade_size: float, expected_impact: float) -> Dict[str, float]:
        """Assess execution risk factors"""
        try:
            risks = {
                "size_risk": min(trade_size / self.size_thresholds["mega"], 1.0),
                "impact_risk": min(expected_impact * 100, 1.0),
                "timing_risk": 0.3,  # Base timing risk
                "liquidity_risk": 0.2,  # Base liquidity risk
                "information_risk": 0.25  # Base information risk
            }
            
            # Adjust for trade size
            if trade_size > self.size_thresholds["block"]:
                risks["size_risk"] *= 1.5
                risks["information_risk"] *= 1.3
            
            # Adjust for impact
            if expected_impact > 0.005:
                risks["impact_risk"] *= 1.4
            
            # Normalize risks
            for key in risks:
                risks[key] = max(0, min(1, risks[key]))
            
            return risks
            
        except Exception as e:
            logger.error(f"Error assessing execution risk factors: {str(e)}")
            return {}
    
    async def analyze_institutional_activity(self, symbol: str) -> Dict[str, Any]:
        """Analyze institutional block trading activity"""
        try:
            trades_data = await self.fetch_recent_trades(symbol)
            clusters = await self.analyze_trade_clusters(symbol)
            
            if not trades_data:
                return {}
            
            df = pd.DataFrame(trades_data)
            
            # Calculate activity metrics
            total_block_volume = df['size'].sum()
            block_trade_count = len(df)
            avg_trade_size = df['size'].mean()
            
            # Institutional signatures
            signatures = {}
            for cluster in clusters:
                if cluster.institutional_signature not in signatures:
                    signatures[cluster.institutional_signature] = 0
                signatures[cluster.institutional_signature] += len(cluster.trades)
            
            # Activity level assessment
            participation_rates = df['participation_rate'].tolist()
            avg_participation = np.mean(participation_rates) if participation_rates else 0
            
            if avg_participation > 15:
                activity_level = "very_high"
            elif avg_participation > 10:
                activity_level = "high"
            elif avg_participation > 5:
                activity_level = "moderate"
            elif avg_participation > 2:
                activity_level = "low"
            else:
                activity_level = "very_low"
            
            # Flow direction analysis
            buy_volume = df[df['trade_type'] == 'buy']['size'].sum()
            sell_volume = df[df['trade_type'] == 'sell']['size'].sum()
            
            if buy_volume > sell_volume * 1.2:
                flow_direction = "accumulation"
            elif sell_volume > buy_volume * 1.2:
                flow_direction = "distribution"
            else:
                flow_direction = "balanced"
            
            # Execution quality trends
            quality_scores = df['execution_quality'].map({
                "excellent": 1.0, "good": 0.8, "fair": 0.6, "poor": 0.4, "very_poor": 0.2
            })
            quality_trend = np.polyfit(range(len(quality_scores)), quality_scores, 1)[0]
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "total_block_volume": total_block_volume,
                "block_trade_count": block_trade_count,
                "avg_trade_size": avg_trade_size,
                "institutional_signatures": signatures,
                "activity_level": activity_level,
                "avg_participation_rate": avg_participation,
                "flow_direction": flow_direction,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "net_flow": buy_volume - sell_volume,
                "quality_trend": quality_trend,
                "venue_diversity": df['venue'].nunique(),
                "execution_consistency": 1 - df['execution_quality'].map({
                    "excellent": 0, "good": 0.2, "fair": 0.5, "poor": 0.8, "very_poor": 1
                }).std()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing institutional activity for {symbol}: {str(e)}")
            return {}
    
    async def generate_comprehensive_analysis(self, symbol: str) -> BlockTradeAnalysis:
        """Generate comprehensive block trade analysis"""
        try:
            # Gather all analysis components
            trades_data = await self.fetch_recent_trades(symbol)
            clusters = await self.analyze_trade_clusters(symbol)
            anomalies = await self.detect_anomalies(symbol)
            institutional_activity = await self.analyze_institutional_activity(symbol)
            market_impact = await self.calculate_market_impact(symbol, 100000)  # 100k shares example
            
            if not trades_data:
                return BlockTradeAnalysis(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    total_block_volume=0,
                    block_trade_count=0,
                    avg_trade_size=0,
                    liquidity_supplied=0,
                    market_impact_score=0,
                    participation_rate=0,
                    execution_quality_score=0,
                    institutional_activity_level="no_activity",
                    clusters=[],
                    anomalies=[],
                    patterns={},
                    recommendations=["No block trade data available"]
                )
            
            df = pd.DataFrame(trades_data)
            
            # Calculate core metrics
            total_block_volume = df['size'].sum()
            block_trade_count = len(df)
            avg_trade_size = df['size'].mean()
            
            # Liquidity metrics
            liquidity_supplied = df['liquidity_score'].mean()
            participation_rate = df['participation_rate'].mean()
            
            # Impact and quality metrics
            market_impact_score = abs(df['market_impact']).mean()
            execution_quality_scores = df['execution_quality'].map({
                "excellent": 1.0, "good": 0.8, "fair": 0.6, "poor": 0.4, "very_poor": 0.2
            })
            execution_quality_score = execution_quality_scores.mean()
            
            # Activity level from institutional analysis
            activity_level = institutional_activity.get("activity_level", "low")
            
            # Pattern extraction
            patterns = self._extract_trading_patterns(df, clusters)
            
            # Generate recommendations
            recommendations = self._generate_block_trade_recommendations(
                total_block_volume, block_trade_count, market_impact_score, 
                execution_quality_score, activity_level, anomalies
            )
            
            analysis = BlockTradeAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                total_block_volume=total_block_volume,
                block_trade_count=block_trade_count,
                avg_trade_size=avg_trade_size,
                liquidity_supplied=liquidity_supplied,
                market_impact_score=market_impact_score,
                participation_rate=participation_rate,
                execution_quality_score=execution_quality_score,
                institutional_activity_level=activity_level,
                clusters=clusters,
                anomalies=anomalies,
                patterns=patterns,
                recommendations=recommendations
            )
            
            logger.info(f"Generated block trade analysis for {symbol}: "
                       f"{block_trade_count} trades, {total_block_volume:,.0f} shares, "
                       f"impact: {market_impact_score:.4f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating comprehensive analysis for {symbol}: {str(e)}")
            return BlockTradeAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                total_block_volume=0,
                block_trade_count=0,
                avg_trade_size=0,
                liquidity_supplied=0,
                market_impact_score=0,
                participation_rate=0,
                execution_quality_score=0,
                institutional_activity_level="error",
                clusters=[],
                anomalies=[],
                patterns={},
                recommendations=["Analysis failed due to data error"]
            )
    
    def _extract_trading_patterns(self, df: pd.DataFrame, clusters: List[TradeCluster]) -> Dict[str, Any]:
        """Extract key trading patterns"""
        try:
            patterns = {
                "timing_patterns": {},
                "size_patterns": {},
                "venue_patterns": {},
                "quality_patterns": {},
                "impact_patterns": {}
            }
            
            # Timing patterns
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hourly_volume = df.groupby('hour')['size'].sum()
            peak_hour = hourly_volume.idxmax() if not hourly_volume.empty else None
            
            patterns["timing_patterns"] = {
                "peak_hour": peak_hour,
                "trading_concentration": hourly_volume.std() / hourly_volume.mean() if hourly_volume.mean() > 0 else 0,
                "intraday_distribution": hourly_volume.to_dict()
            }
            
            # Size patterns
            size_categories = df['size'].apply(self._classify_trade_size)
            patterns["size_patterns"] = {
                "size_distribution": size_categories.value_counts().to_dict(),
                "avg_size_by_category": df.groupby(size_categories)['size'].mean().to_dict()
            }
            
            # Venue patterns
            patterns["venue_patterns"] = {
                "venue_distribution": df['venue'].value_counts().to_dict(),
                "venue_quality": df.groupby('venue')['execution_quality'].apply(
                    lambda x: x.map({"excellent": 1, "good": 0.8, "fair": 0.6, "poor": 0.4, "very_poor": 0.2}).mean()
                ).to_dict()
            }
            
            # Quality patterns
            quality_dist = df['execution_quality'].value_counts()
            patterns["quality_patterns"] = {
                "quality_distribution": quality_dist.to_dict(),
                "avg_quality_score": df['execution_quality'].map({
                    "excellent": 1, "good": 0.8, "fair": 0.6, "poor": 0.4, "very_poor": 0.2
                }).mean()
            }
            
            # Impact patterns
            patterns["impact_patterns"] = {
                "avg_impact": df['market_impact'].mean(),
                "impact_volatility": df['market_impact'].std(),
                "impact_range": {
                    "min": df['market_impact'].min(),
                    "max": df['market_impact'].max()
                }
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error extracting trading patterns: {str(e)}")
            return {}
    
    def _generate_block_trade_recommendations(self, total_volume: float, trade_count: int, 
                                            impact_score: float, quality_score: float, 
                                            activity_level: str, anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations for block trading"""
        try:
            recommendations = []
            
            # Volume-based recommendations
            if total_volume > 1000000:  # > 1M shares
                recommendations.append("High block trading volume detected - consider timing large orders")
            
            if trade_count > 15:
                recommendations.append("High frequency of block trades - market is active for large orders")
            
            # Impact-based recommendations
            if impact_score > 0.01:
                recommendations.append("High market impact detected - consider dark pool or TWAP execution")
            elif impact_score < 0.002:
                recommendations.append("Low market impact environment - favorable for immediate execution")
            
            # Quality-based recommendations
            if quality_score > 0.8:
                recommendations.append("Excellent execution quality - market conditions are favorable")
            elif quality_score < 0.5:
                recommendations.append("Poor execution quality - consider alternative execution venues")
            
            # Activity level recommendations
            if activity_level in ["very_high", "high"]:
                recommendations.append("High institutional activity - favorable environment for block trades")
            elif activity_level in ["very_low", "low"]:
                recommendations.append("Low institutional activity - consider smaller orders or different timing")
            
            # Anomaly-based recommendations
            for anomaly in anomalies:
                if anomaly.get("type") == "size_anomaly" and anomaly.get("severity") == "high":
                    recommendations.append(f"Large trade anomaly detected: {anomaly.get('description', '')}")
                elif anomaly.get("type") == "impact_anomaly":
                    recommendations.append(f"Market impact anomaly: {anomaly.get('description', '')} - adjust execution strategy")
            
            # Strategy recommendations
            if len([r for r in recommendations if "execution" in r.lower() or "strategy" in r.lower()]) == 0:
                if impact_score < 0.005:
                    recommendations.append("Consider aggressive execution strategies given low market impact")
                else:
                    recommendations.append("Consider passive execution strategies to minimize market impact")
            
            # Ensure we have recommendations
            if not recommendations:
                recommendations.append("Monitor block trade patterns for execution opportunities")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating block trade recommendations: {str(e)}")
            return ["Unable to generate specific recommendations"]
    
    async def monitor_block_trade_alerts(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Monitor for significant block trade alerts"""
        try:
            alerts = {}
            
            for symbol in symbols:
                symbol_alerts = []
                
                # Generate comprehensive analysis
                analysis = await self.generate_comprehensive_analysis(symbol)
                
                # Volume alerts
                if analysis.total_block_volume > 2000000:  # > 2M shares
                    symbol_alerts.append(f"HIGH_VOLUME: {symbol} block volume {analysis.total_block_volume:,.0f} shares")
                
                # Impact alerts
                if analysis.market_impact_score > 0.015:
                    symbol_alerts.append(f"HIGH_IMPACT: {symbol} market impact {analysis.market_impact_score:.4f}")
                
                # Quality alerts
                if analysis.execution_quality_score > 0.9:
                    symbol_alerts.append(f"EXCELLENT_EXECUTION: {symbol} quality score {analysis.execution_quality_score:.3f}")
                elif analysis.execution_quality_score < 0.4:
                    symbol_alerts.append(f"POOR_EXECUTION: {symbol} quality score {analysis.execution_quality_score:.3f}")
                
                # Activity alerts
                if analysis.institutional_activity_level == "very_high":
                    symbol_alerts.append(f"HIGH_ACTIVITY: {symbol} institutional activity very high")
                
                # Anomaly alerts
                for anomaly in analysis.anomalies:
                    if anomaly.get("severity") == "high":
                        symbol_alerts.append(f"ANOMALY: {symbol} {anomaly.get('type', 'unknown')}: {anomaly.get('description', '')}")
                
                # Cluster alerts
                for cluster in analysis.clusters:
                    if cluster.price_impact > 0.02:
                        symbol_alerts.append(f"CLUSTER_IMPACT: {symbol} high-impact cluster with {len(cluster.trades)} trades")
                
                if symbol_alerts:
                    alerts[symbol] = symbol_alerts
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring block trade alerts: {str(e)}")
            return {}
    
    async def compare_symbols(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compare block trading characteristics across symbols"""
        try:
            comparisons = {}
            
            for symbol in symbols:
                analysis = await self.generate_comprehensive_analysis(symbol)
                
                comparisons[symbol] = {
                    "total_volume": analysis.total_block_volume,
                    "trade_count": analysis.block_trade_count,
                    "avg_size": analysis.avg_trade_size,
                    "market_impact": analysis.market_impact_score,
                    "execution_quality": analysis.execution_quality_score,
                    "participation_rate": analysis.participation_rate,
                    "activity_level": analysis.institutional_activity_level,
                    "liquidity_score": analysis.liquidity_supplied
                }
            
            # Calculate relative rankings
            for metric in ["total_volume", "trade_count", "avg_size", "market_impact", 
                          "execution_quality", "participation_rate", "liquidity_score"]:
                values = [comp[metric] for comp in comparisons.values() if isinstance(comp[metric], (int, float))]
                if values:
                    max_val = max(values)
                    for symbol in comparisons:
                        if isinstance(comparisons[symbol][metric], (int, float)):
                            comparisons[symbol][f"{metric}_rank"] = values.index(comparisons[symbol][metric]) + 1
                            comparisons[symbol][f"{metric}_percentile"] = (values.index(comparisons[symbol][metric]) + 1) / len(values)
            
            return comparisons
            
        except Exception as e:
            logger.error(f"Error comparing symbols: {str(e)}")
            return {}
    
    async def export_analysis(self, symbol: str, format_type: str = "json") -> str:
        """Export block trade analysis to file"""
        try:
            analysis = await self.generate_comprehensive_analysis(symbol)
            
            if format_type.lower() == "json":
                import json
                export_data = {
                    "symbol": analysis.symbol,
                    "timestamp": analysis.timestamp.isoformat(),
                    "total_block_volume": analysis.total_block_volume,
                    "block_trade_count": analysis.block_trade_count,
                    "avg_trade_size": analysis.avg_trade_size,
                    "liquidity_supplied": analysis.liquidity_supplied,
                    "market_impact_score": analysis.market_impact_score,
                    "participation_rate": analysis.participation_rate,
                    "execution_quality_score": analysis.execution_quality_score,
                    "institutional_activity_level": analysis.institutional_activity_level,
                    "clusters": [
                        {
                            "cluster_id": c.cluster_id,
                            "avg_size": c.avg_size,
                            "total_volume": c.total_volume,
                            "price_impact": c.price_impact,
                            "execution_pattern": c.execution_pattern,
                            "institutional_signature": c.institutional_signature,
                            "trade_count": len(c.trades)
                        }
                        for c in analysis.clusters
                    ],
                    "anomalies": analysis.anomalies,
                    "patterns": analysis.patterns,
                    "recommendations": analysis.recommendations
                }
                
                filename = f"block_trade_analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = f"/workspace/mpe/exports/{filename}"
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return filepath
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting analysis for {symbol}: {str(e)}")
            return ""
    
    async def get_execution_recommendations(self, symbol: str, order_size: float) -> Dict[str, Any]:
        """Get specific execution recommendations for an order"""
        try:
            # Get current market impact analysis
            impact_analysis = await self.calculate_market_impact(symbol, order_size)
            
            # Get recent execution quality data
            trades_data = await self.fetch_recent_trades(symbol)
            df = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()
            
            # Venue performance analysis
            venue_performance = {}
            if not df.empty:
                for venue in df['venue'].unique():
                    venue_trades = df[df['venue'] == venue]
                    venue_performance[venue] = {
                        "avg_impact": venue_trades['market_impact'].mean(),
                        "quality_score": venue_trades['execution_quality'].map({
                            "excellent": 1, "good": 0.8, "fair": 0.6, "poor": 0.4, "very_poor": 0.2
                        }).mean(),
                        "fill_rate": 0.95,  # Assumed fill rate
                        "avg_size": venue_trades['size'].mean(),
                        "trade_count": len(venue_trades)
                    }
            
            # Timing recommendations
            timing_recs = self._generate_timing_recommendations(df)
            
            # Size-specific recommendations
            size_category = self._classify_trade_size(order_size)
            size_recommendations = self._generate_size_recommendations(size_category, impact_analysis)
            
            return {
                "symbol": symbol,
                "order_size": order_size,
                "size_category": size_category,
                "market_impact_analysis": impact_analysis,
                "venue_performance": venue_performance,
                "timing_recommendations": timing_recs,
                "size_recommendations": size_recommendations,
                "execution_strategy": impact_analysis.get("recommendation", "immediate"),
                "risk_assessment": impact_analysis.get("risk_factors", {})
            }
            
        except Exception as e:
            logger.error(f"Error getting execution recommendations for {symbol}: {str(e)}")
            return {}
    
    def _generate_timing_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate timing-based execution recommendations"""
        try:
            if df.empty:
                return {"optimal_times": [], "avoid_times": [], "recommendation": "Use market hours"}
            
            # Analyze hourly patterns
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hourly_activity = df.groupby('hour').agg({
                'size': 'sum',
                'market_impact': 'mean',
                'execution_quality': lambda x: x.map({
                    "excellent": 1, "good": 0.8, "fair": 0.6, "poor": 0.4, "very_poor": 0.2
                }).mean()
            })
            
            # Find optimal and problematic times
            high_activity_hours = hourly_activity[hourly_activity['size'] > hourly_activity['size'].quantile(0.7)].index.tolist()
            low_impact_hours = hourly_activity[hourly_activity['market_impact'] < hourly_activity['market_impact'].quantile(0.3)].index.tolist()
            high_quality_hours = hourly_activity[hourly_activity['execution_quality'] > hourly_activity['execution_quality'].quantile(0.7)].index.tolist()
            
            optimal_times = list(set(high_activity_hours) & set(low_impact_hours) & set(high_quality_hours))
            avoid_times = hourly_activity[hourly_activity['market_impact'] > hourly_activity['market_impact'].quantile(0.8)].index.tolist()
            
            return {
                "optimal_times": optimal_times,
                "avoid_times": avoid_times,
                "recommendation": f"Execute during hours {optimal_times}" if optimal_times else "No specific optimal times identified",
                "activity_pattern": hourly_activity.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error generating timing recommendations: {str(e)}")
            return {}
    
    def _generate_size_recommendations(self, size_category: str, impact_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate size-specific execution recommendations"""
        recommendations = {
            "small": "Immediate execution recommended due to minimal market impact",
            "medium": "Consider auction execution or dark pool for price improvement",
            "large": "Use TWAP or dark pool execution to minimize market impact",
            "block": "Implement dark pool execution with potential auction component",
            "mega": "Use comprehensive execution strategy with dark pool and TWAP"
        }
        
        base_rec = recommendations.get(size_category, "Standard execution procedures apply")
        
        # Adjust based on current market conditions
        expected_impact = impact_analysis.get("expected_impact", 0)
        if expected_impact > 0.01:
            base_rec += " - High market impact detected, consider alternative execution"
        elif expected_impact < 0.002:
            base_rec += " - Low market impact environment, aggressive execution viable"
        
        return {
            "recommendation": base_rec,
            "strategy": impact_analysis.get("recommendation", "immediate"),
            "risk_level": "high" if expected_impact > 0.01 else "medium" if expected_impact > 0.005 else "low"
        }

# Main execution function
async def main():
    """Main execution function for block trade monitoring"""
    monitor = BlockTradeMonitor()
    
    # Test symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    
    logger.info("Starting Block Trade Monitor analysis...")
    
    # Test comprehensive analysis
    for symbol in test_symbols[:2]:  # Test first 2 symbols
        logger.info(f"\n=== Block Trade Analysis for {symbol} ===")
        
        analysis = await monitor.generate_comprehensive_analysis(symbol)
        
        logger.info(f"Total Block Volume: {analysis.total_block_volume:,.0f} shares")
        logger.info(f"Block Trade Count: {analysis.block_trade_count}")
        logger.info(f"Average Trade Size: {analysis.avg_trade_size:,.0f} shares")
        logger.info(f"Market Impact Score: {analysis.market_impact_score:.4f}")
        logger.info(f"Execution Quality Score: {analysis.execution_quality_score:.3f}")
        logger.info(f"Institutional Activity Level: {analysis.institutional_activity_level}")
        logger.info(f"Clusters Detected: {len(analysis.clusters)}")
        logger.info(f"Anomalies Found: {len(analysis.anomalies)}")
        
        logger.info("\nRecommendations:")
        for rec in analysis.recommendations[:3]:  # Show first 3 recommendations
            logger.info(f"  - {rec}")
    
    # Test monitoring alerts
    logger.info("\n=== Block Trade Alerts ===")
    alerts = await monitor.monitor_block_trade_alerts(test_symbols)
    
    for symbol, symbol_alerts in alerts.items():
        if symbol_alerts:
            logger.info(f"{symbol}: {len(symbol_alerts)} alerts")
            for alert in symbol_alerts[:2]:  # Show first 2 alerts
                logger.info(f"  - {alert}")
    
    # Test execution recommendations
    logger.info("\n=== Execution Recommendations ===")
    recommendations = await monitor.get_execution_recommendations("AAPL", 500000)  # 500k shares
    
    logger.info(f"Order Size: {recommendations.get('order_size', 0):,.0f} shares")
    logger.info(f"Size Category: {recommendations.get('size_category', 'unknown')}")
    logger.info(f"Recommended Strategy: {recommendations.get('execution_strategy', 'unknown')}")
    logger.info(f"Size Recommendation: {recommendations.get('size_recommendations', {}).get('recommendation', 'None')}")
    
    # Test symbol comparison
    logger.info("\n=== Symbol Comparison ===")
    comparisons = await monitor.compare_symbols(test_symbols[:3])  # Compare first 3 symbols
    
    for symbol, data in comparisons.items():
        logger.info(f"{symbol}:")
        logger.info(f"  Volume Rank: {data.get('total_volume_rank', 'N/A')}")
        logger.info(f"  Quality Rank: {data.get('execution_quality_rank', 'N/A')}")
        logger.info(f"  Activity Level: {data.get('activity_level', 'unknown')}")
    
    logger.info("Block Trade Monitor analysis completed successfully")

if __name__ == "__main__":
    asyncio.run(main())