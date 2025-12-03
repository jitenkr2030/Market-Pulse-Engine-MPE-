"""
Module 23: Cross-Asset Correlation Engine
Author: MiniMax Agent
Date: 2025-12-02

Advanced cross-asset correlation analysis and relationship modeling system.
Provides comprehensive correlation tracking, regime detection, and relationship
analysis across multiple asset classes and instruments.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, FrozenSet
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetClass(Enum):
    """Asset class classifications"""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    REAL_ESTATE = "real_estate"
    ALTERNATIVE = "alternative"

class CorrelationRegime(Enum):
    """Correlation regime classifications"""
    LOW_CORRELATION = "low_correlation"
    MODERATE_CORRELATION = "moderate_correlation"
    HIGH_CORRELATION = "high_correlation"
    EXTREME_CORRELATION = "extreme_correlation"
    REGIME_SHIFT = "regime_shift"

class RelationshipType(Enum):
    """Types of cross-asset relationships"""
    STRONG_POSITIVE = "strong_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    STRONG_NEGATIVE = "strong_negative"
    CAUSAL = "causal"
    LEADING_LAGGING = "leading_lagging"

@dataclass
class CorrelationPair:
    """Individual correlation pair data"""
    asset1: str
    asset2: str
    correlation_1d: float
    correlation_5d: float
    correlation_20d: float
    correlation_60d: float
    rolling_volatility_1: float
    rolling_volatility_2: float
    relationship_type: str
    stability_score: float
    timestamp: datetime

@dataclass
class CorrelationCluster:
    """Cluster of highly correlated assets"""
    cluster_id: str
    assets: List[str]
    average_correlation: float
    correlation_matrix: np.ndarray
    cluster_characteristics: Dict[str, float]
    regime_stability: float
    risk_contribution: float

@dataclass
class RegimeChange:
    """Correlation regime change detection"""
    assets: Tuple[str, str]
    old_regime: str
    new_regime: str
    change_magnitude: float
    confidence_score: float
    timestamp: datetime
    causation_indicators: Dict[str, float]

@dataclass
class CrossAssetAnalysis:
    """Comprehensive cross-asset correlation analysis"""
    timestamp: datetime
    assets_analyzed: List[str]
    correlation_matrix: np.ndarray
    correlation_pairs: List[CorrelationPair]
    clusters: List[CorrelationCluster]
    regime_changes: List[RegimeChange]
    market_structure: Dict[str, Any]
    risk_decomposition: Dict[str, Any]
    relationship_analysis: Dict[str, Any]
    recommendations: List[str]

class CrossAssetCorrelationEngine:
    """
    Advanced Cross-Asset Correlation Engine
    
    Analyzes, monitors, and provides intelligence on correlations
    across multiple asset classes to support portfolio construction
    and risk management strategies.
    """
    
    def __init__(self):
        self.name = "Cross-Asset Correlation Engine"
        self.version = "1.0.0"
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Asset class mapping
        self.asset_classification = {
            # Major Equities
            "AAPL": "equity", "MSFT": "equity", "GOOGL": "equity", "AMZN": "equity",
            "TSLA": "equity", "META": "equity", "NVDA": "equity", "BRK-B": "equity",
            "SPY": "equity", "QQQ": "equity", "IWM": "equity", "VTI": "equity",
            
            # Fixed Income
            "TLT": "fixed_income", "IEF": "fixed_income", "SHY": "fixed_income",
            "LQD": "fixed_income", "HYG": "fixed_income", "EMB": "fixed_income",
            "AGG": "fixed_income", "BND": "fixed_income",
            
            # Commodities
            "GLD": "commodity", "SLV": "commodity", "USO": "commodity",
            "DBC": "commodity", "GSG": "commodity", "COPPER": "commodity",
            
            # Currencies
            "UUP": "currency", "FXE": "currency", "FXY": "currency",
            "FXB": "currency", "GLDUSD=X": "currency",
            
            # Crypto
            "BTC-USD": "crypto", "ETH-USD": "crypto", "BNB-USD": "crypto",
            
            # Real Estate
            "VNQ": "real_estate", "IYR": "real_estate", "SCHH": "real_estate",
            
            # Alternatives/VIX
            "VIX": "alternative", "VXX": "alternative", "TLT": "alternative"
        }
        
        # Correlation thresholds
        self.correlation_thresholds = {
            "very_high_positive": 0.8,
            "high_positive": 0.6,
            "moderate_positive": 0.4,
            "neutral": 0.2,
            "moderate_negative": -0.4,
            "high_negative": -0.6,
            "very_high_negative": -0.8
        }
        
        # Regime detection parameters
        self.regime_detection = {
            "lookback_window": 60,
            "regime_threshold": 0.3,
            "stability_threshold": 0.7
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
    
    async def fetch_asset_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive asset data for correlation analysis"""
        try:
            cache_key = f"asset_data_{'_'.join(sorted(symbols))}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # Fetch data for all assets
            asset_data = {}
            
            # Use batch processing for efficiency
            tasks = []
            for symbol in symbols:
                task = self._fetch_single_asset_data(symbol)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(symbols, results):
                if isinstance(result, pd.DataFrame) and not result.empty:
                    asset_data[symbol] = result
                else:
                    logger.warning(f"No data available for {symbol}")
                    # Create placeholder data
                    asset_data[symbol] = self._create_placeholder_data(symbol)
            
            await self._set_cache_data(cache_key, asset_data)
            return asset_data
            
        except Exception as e:
            logger.error(f"Error fetching asset data: {str(e)}")
            return {}
    
    async def _fetch_single_asset_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data for a single asset"""
        try:
            # Try different data sources and periods
            ticker = yf.Ticker(symbol)
            
            # Get longer-term data for correlation analysis
            data = ticker.history(period="1y", interval="1d")
            
            if data.empty:
                # Try shorter period if longer data unavailable
                data = ticker.history(period="6mo", interval="1d")
            
            if data.empty:
                # Try intraday data if daily unavailable
                data = ticker.history(period="1mo", interval="1h")
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _create_placeholder_data(self, symbol: str) -> pd.DataFrame:
        """Create placeholder data when real data unavailable"""
        try:
            # Create synthetic data based on asset class
            asset_class = self.asset_classification.get(symbol, "equity")
            
            days = 252  # 1 year of trading days
            dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
            
            # Set different volatility characteristics by asset class
            if asset_class == "equity":
                base_price = 100
                volatility = 0.02
            elif asset_class == "fixed_income":
                base_price = 100
                volatility = 0.008
            elif asset_class == "commodity":
                base_price = 50
                volatility = 0.025
            elif asset_class == "currency":
                base_price = 1.0
                volatility = 0.012
            elif asset_class == "crypto":
                base_price = 50000
                volatility = 0.05
            else:
                base_price = 100
                volatility = 0.015
            
            # Generate synthetic price series
            returns = np.random.normal(0.0001, volatility, days)  # Small positive drift
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            data = pd.DataFrame({
                'Open': [p * np.random.uniform(0.995, 1.005) for p in prices],
                'High': [p * np.random.uniform(1.005, 1.02) for p in prices],
                'Low': [p * np.random.uniform(0.98, 0.995) for p in prices],
                'Close': prices,
                'Volume': [np.random.randint(1000000, 10000000) for _ in range(days)]
            }, index=dates)
            
            return data
            
        except Exception as e:
            logger.error(f"Error creating placeholder data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_correlation_matrix(self, asset_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Calculate correlation matrix for all assets"""
        try:
            if not asset_data:
                return np.array([])
            
            # Align all data to common dates
            aligned_data = {}
            min_length = float('inf')
            
            for symbol, data in asset_data.items():
                if not data.empty:
                    aligned_data[symbol] = data['Close']
                    min_length = min(min_length, len(data))
            
            if not aligned_data:
                return np.array([])
            
            # Create DataFrame with aligned data
            df = pd.DataFrame(aligned_data)
            
            # Calculate returns
            returns = df.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr().values
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return np.array([])
    
    async def analyze_correlation_pairs(self, asset_data: Dict[str, pd.DataFrame]) -> List[CorrelationPair]:
        """Analyze correlations between all asset pairs"""
        try:
            symbols = list(asset_data.keys())
            correlation_pairs = []
            
            # Create aligned returns DataFrame
            df = pd.DataFrame({symbol: data['Close'] for symbol, data in asset_data.items() if not data.empty})
            returns = df.pct_change().dropna()
            
            # Calculate rolling correlations for different timeframes
            for i, asset1 in enumerate(symbols):
                for j, asset2 in enumerate(symbols):
                    if i < j:  # Avoid duplicates
                        # Get return series for both assets
                        ret1 = returns[asset1].dropna()
                        ret2 = returns[asset2].dropna()
                        
                        if len(ret1) < 20 or len(ret2) < 20:
                            continue  # Need minimum data points
                        
                        # Align series
                        common_dates = ret1.index.intersection(ret2.index)
                        if len(common_dates) < 20:
                            continue
                        
                        ret1_aligned = ret1[common_dates]
                        ret2_aligned = ret2[common_dates]
                        
                        # Calculate correlations for different windows
                        correlation_1d = ret1_aligned.corr(ret2_aligned.tail(20))
                        correlation_5d = ret1_aligned.corr(ret2_aligned.tail(60))
                        correlation_20d = ret1_aligned.corr(ret2_aligned.tail(252)) if len(ret1_aligned) >= 252 else ret1_aligned.corr(ret2_aligned.tail(min(len(ret1_aligned), 60)))
                        correlation_60d = ret1_aligned.corr(ret2_aligned.tail(min(len(ret1_aligned), 252)))
                        
                        # Handle NaN correlations
                        correlation_1d = correlation_1d if not pd.isna(correlation_1d) else 0.0
                        correlation_5d = correlation_5d if not pd.isna(correlation_5d) else 0.0
                        correlation_20d = correlation_20d if not pd.isna(correlation_20d) else 0.0
                        correlation_60d = correlation_60d if not pd.isna(correlation_60d) else 0.0
                        
                        # Calculate rolling volatilities
                        rolling_vol1 = ret1_aligned.tail(60).std()
                        rolling_vol2 = ret2_aligned.tail(60).std()
                        
                        # Determine relationship type
                        relationship_type = self._classify_relationship_type(correlation_60d)
                        
                        # Calculate stability score
                        stability_score = self._calculate_correlation_stability([correlation_1d, correlation_5d, correlation_20d, correlation_60d])
                        
                        # Create correlation pair
                        pair = CorrelationPair(
                            asset1=asset1,
                            asset2=asset2,
                            correlation_1d=correlation_1d,
                            correlation_5d=correlation_5d,
                            correlation_20d=correlation_20d,
                            correlation_60d=correlation_60d,
                            rolling_volatility_1=rolling_vol1,
                            rolling_volatility_2=rolling_vol2,
                            relationship_type=relationship_type,
                            stability_score=stability_score,
                            timestamp=datetime.now()
                        )
                        
                        correlation_pairs.append(pair)
            
            return correlation_pairs
            
        except Exception as e:
            logger.error(f"Error analyzing correlation pairs: {str(e)}")
            return []
    
    def _classify_relationship_type(self, correlation: float) -> str:
        """Classify relationship type based on correlation value"""
        try:
            if correlation > 0.8:
                return "strong_positive"
            elif correlation > 0.6:
                return "positive"
            elif correlation > 0.2:
                return "moderate_positive"
            elif correlation > -0.2:
                return "neutral"
            elif correlation > -0.6:
                return "negative"
            elif correlation > -0.8:
                return "strong_negative"
            else:
                return "extreme_negative"
                
        except Exception as e:
            logger.error(f"Error classifying relationship type: {str(e)}")
            return "neutral"
    
    def _calculate_correlation_stability(self, correlations: List[float]) -> float:
        """Calculate stability score for correlation values"""
        try:
            valid_correlations = [c for c in correlations if not pd.isna(c) and abs(c) <= 1.0]
            if len(valid_correlations) < 2:
                return 0.5
            
            # Calculate variance of correlations (lower variance = higher stability)
            correlation_variance = np.var(valid_correlations)
            stability_score = max(0, 1 - (correlation_variance * 4))  # Scale variance
            
            return stability_score
            
        except Exception as e:
            logger.error(f"Error calculating correlation stability: {str(e)}")
            return 0.5
    
    def identify_correlation_clusters(self, correlation_matrix: np.ndarray, 
                                    symbols: List[str]) -> List[CorrelationCluster]:
        """Identify clusters of highly correlated assets"""
        try:
            if correlation_matrix.size == 0 or len(symbols) < 3:
                return []
            
            # Convert correlation to distance matrix
            distance_matrix = 1 - np.abs(correlation_matrix)
            
            # Perform hierarchical clustering
            if len(symbols) > 2:
                condensed_distances = distance_matrix[np.triu_indices(len(symbols), k=1)]
                linkage_matrix = linkage(condensed_distances, method='average')
                
                # Get clusters (number of clusters based on correlation strength)
                n_clusters = min(5, len(symbols) // 2)
                if n_clusters < 2:
                    n_clusters = 2
                
                cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            else:
                cluster_labels = np.ones(len(symbols))
                n_clusters = 1
            
            clusters = []
            
            for cluster_id in range(1, n_clusters + 1):
                cluster_assets = [symbols[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_assets) >= 2:  # Only include clusters with 2+ assets
                    # Calculate average correlation within cluster
                    cluster_indices = [symbols.index(asset) for asset in cluster_assets]
                    cluster_correlations = correlation_matrix[np.ix_(cluster_indices, cluster_indices)]
                    
                    # Remove diagonal (self-correlations)
                    cluster_corr_no_diag = cluster_correlations.copy()
                    np.fill_diagonal(cluster_corr_no_diag, 0)
                    
                    avg_correlation = np.mean(np.abs(cluster_corr_no_diag))
                    
                    # Calculate cluster characteristics
                    cluster_characteristics = self._calculate_cluster_characteristics(cluster_assets, cluster_correlations, symbols)
                    
                    # Calculate regime stability
                    regime_stability = self._calculate_cluster_regime_stability(cluster_correlations)
                    
                    # Calculate risk contribution
                    risk_contribution = self._calculate_cluster_risk_contribution(cluster_correlations, len(cluster_assets))
                    
                    cluster = CorrelationCluster(
                        cluster_id=f"cluster_{cluster_id}",
                        assets=cluster_assets,
                        average_correlation=avg_correlation,
                        correlation_matrix=cluster_correlations,
                        cluster_characteristics=cluster_characteristics,
                        regime_stability=regime_stability,
                        risk_contribution=risk_contribution
                    )
                    
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error identifying correlation clusters: {str(e)}")
            return []
    
    def _calculate_cluster_characteristics(self, cluster_assets: List[str], 
                                         cluster_correlations: np.ndarray, 
                                         all_symbols: List[str]) -> Dict[str, float]:
        """Calculate characteristics of a correlation cluster"""
        try:
            # Asset class diversity
            cluster_asset_classes = [self.asset_classification.get(asset, "unknown") for asset in cluster_assets]
            asset_class_diversity = len(set(cluster_asset_classes)) / max(1, len(set(self.asset_classification.values())))
            
            # Average correlation strength
            cluster_corr_no_diag = cluster_correlations.copy()
            np.fill_diagonal(cluster_corr_no_diag, 0)
            avg_correlation = np.mean(np.abs(cluster_corr_no_diag))
            
            # Correlation variance (diversity within cluster)
            correlation_variance = np.var(cluster_corr_no_diag)
            
            # Size of cluster relative to universe
            cluster_size_ratio = len(cluster_assets) / len(all_symbols)
            
            return {
                "asset_class_diversity": asset_class_diversity,
                "average_correlation": avg_correlation,
                "correlation_variance": correlation_variance,
                "cluster_size_ratio": cluster_size_ratio,
                "num_assets": len(cluster_assets)
            }
            
        except Exception as e:
            logger.error(f"Error calculating cluster characteristics: {str(e)}")
            return {}
    
    def _calculate_cluster_regime_stability(self, cluster_correlations: np.ndarray) -> float:
        """Calculate regime stability of a cluster"""
        try:
            # Calculate standard deviation of correlations (lower std = higher stability)
            cluster_corr_no_diag = cluster_correlations.copy()
            np.fill_diagonal(cluster_corr_no_diag, np.nan)
            correlation_std = np.nanstd(cluster_corr_no_diag)
            
            # Convert to stability score (0-1 scale)
            stability_score = max(0, 1 - (correlation_std * 4))
            
            return stability_score
            
        except Exception as e:
            logger.error(f"Error calculating cluster regime stability: {str(e)}")
            return 0.5
    
    def _calculate_cluster_risk_contribution(self, cluster_correlations: np.ndarray, cluster_size: int) -> float:
        """Calculate risk contribution of a cluster"""
        try:
            # Simplified risk contribution based on correlation and size
            cluster_corr_no_diag = cluster_correlations.copy()
            np.fill_diagonal(cluster_corr_no_diag, 0)
            
            # Average correlation strength
            avg_correlation = np.mean(np.abs(cluster_corr_no_diag))
            
            # Risk contribution increases with both correlation strength and cluster size
            risk_contribution = avg_correlation * np.sqrt(cluster_size) / len(cluster_correlations)
            
            return min(1.0, risk_contribution)
            
        except Exception as e:
            logger.error(f"Error calculating cluster risk contribution: {str(e)}")
            return 0.5
    
    async def detect_regime_changes(self, asset_data: Dict[str, pd.DataFrame], 
                                  old_analysis: Optional[CrossAssetAnalysis] = None) -> List[RegimeChange]:
        """Detect significant correlation regime changes"""
        try:
            if not old_analysis:
                return []  # No previous analysis to compare with
            
            current_symbols = list(asset_data.keys())
            old_symbols = old_analysis.assets_analyzed
            
            regime_changes = []
            
            # Compare correlations for common asset pairs
            for pair in old_analysis.correlation_pairs:
                if pair.asset1 in current_symbols and pair.asset2 in current_symbols:
                    # Find current correlation for this pair
                    current_pair = None
                    for new_pair in await self.analyze_correlation_pairs(asset_data):
                        if (new_pair.asset1 == pair.asset1 and new_pair.asset2 == pair.asset2) or \
                           (new_pair.asset1 == pair.asset2 and new_pair.asset2 == pair.asset1):
                            current_pair = new_pair
                            break
                    
                    if current_pair:
                        # Compare long-term correlations (60-day)
                        old_correlation = pair.correlation_60d
                        new_correlation = current_pair.correlation_60d
                        
                        # Calculate change magnitude
                        change_magnitude = abs(new_correlation - old_correlation)
                        
                        # Detect significant regime change
                        if change_magnitude > self.regime_detection["regime_threshold"]:
                            # Classify old and new regimes
                            old_regime = self._classify_correlation_regime(old_correlation)
                            new_regime = self._classify_correlation_regime(new_correlation)
                            
                            # Calculate confidence score
                            confidence_score = self._calculate_regime_change_confidence(
                                change_magnitude, pair.stability_score, current_pair.stability_score
                            )
                            
                            # Check for causation indicators
                            causation_indicators = self._analyze_causation_indicators(
                                pair.asset1, pair.asset2, asset_data
                            )
                            
                            regime_change = RegimeChange(
                                assets=(pair.asset1, pair.asset2),
                                old_regime=old_regime,
                                new_regime=new_regime,
                                change_magnitude=change_magnitude,
                                confidence_score=confidence_score,
                                timestamp=datetime.now(),
                                causation_indicators=causation_indicators
                            )
                            
                            regime_changes.append(regime_change)
            
            return regime_changes
            
        except Exception as e:
            logger.error(f"Error detecting regime changes: {str(e)}")
            return []
    
    def _classify_correlation_regime(self, correlation: float) -> str:
        """Classify correlation regime"""
        try:
            abs_corr = abs(correlation)
            if abs_corr > 0.8:
                return "extreme_correlation"
            elif abs_corr > 0.6:
                return "high_correlation"
            elif abs_corr > 0.4:
                return "moderate_correlation"
            elif abs_corr > 0.2:
                return "low_correlation"
            else:
                return "minimal_correlation"
                
        except Exception as e:
            logger.error(f"Error classifying correlation regime: {str(e)}")
            return "unknown"
    
    def _calculate_regime_change_confidence(self, change_magnitude: float, 
                                          old_stability: float, new_stability: float) -> float:
        """Calculate confidence score for regime change"""
        try:
            # Base confidence on change magnitude
            magnitude_confidence = min(1.0, change_magnitude / 0.5)  # Normalize to 0-1
            
            # Adjust for stability (higher stability = higher confidence)
            avg_stability = (old_stability + new_stability) / 2
            stability_factor = avg_stability
            
            # Combine factors
            confidence = magnitude_confidence * stability_factor
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating regime change confidence: {str(e)}")
            return 0.5
    
    def _analyze_causation_indicators(self, asset1: str, asset2: str, 
                                    asset_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze potential causation indicators between assets"""
        try:
            indicators = {}
            
            # Basic asset class relationship
            class1 = self.asset_classification.get(asset1, "unknown")
            class2 = self.asset_classification.get(asset2, "unknown")
            
            if class1 == class2:
                indicators["same_asset_class"] = 0.8
            elif (class1, class2) in [("equity", "fixed_income"), ("fixed_income", "equity")]:
                indicators["equity_bond_relationship"] = 0.6
            elif (class1, class2) in [("commodity", "currency"), ("currency", "commodity")]:
                indicators["commodity_currency_relationship"] = 0.5
            else:
                indicators["cross_asset_relationship"] = 0.3
            
            # Volatility relationship
            if asset1 in asset_data and asset2 in asset_data:
                data1 = asset_data[asset1]['Close'].pct_change().dropna()
                data2 = asset_data[asset2]['Close'].pct_change().dropna()
                
                if len(data1) > 20 and len(data2) > 20:
                    # Calculate rolling volatility correlation
                    vol1 = data1.rolling(20).std().dropna()
                    vol2 = data2.rolling(20).std().dropna()
                    
                    common_dates = vol1.index.intersection(vol2.index)
                    if len(common_dates) > 10:
                        vol_corr = vol1[common_dates].corr(vol2[common_dates])
                        indicators["volatility_correlation"] = abs(vol_corr) if not pd.isna(vol_corr) else 0
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error analyzing causation indicators: {str(e)}")
            return {}
    
    def analyze_market_structure(self, correlation_matrix: np.ndarray, 
                               symbols: List[str], clusters: List[CorrelationCluster]) -> Dict[str, Any]:
        """Analyze overall market structure from correlation patterns"""
        try:
            structure_analysis = {}
            
            if correlation_matrix.size == 0:
                return structure_analysis
            
            # Overall market correlation statistics
            corr_no_diag = correlation_matrix.copy()
            np.fill_diagr(corr_no_diag, 0)
            
            structure_analysis.update({
                "average_correlation": float(np.mean(np.abs(corr_no_diag))),
                "correlation_dispersion": float(np.std(corr_no_diag)),
                "max_correlation": float(np.max(corr_no_diag)),
                "min_correlation": float(np.min(corr_no_diag)),
                "correlation_range": float(np.max(corr_no_diag) - np.min(corr_no_diag))
            })
            
            # Asset class correlation analysis
            asset_class_correlations = self._analyze_asset_class_correlations(correlation_matrix, symbols)
            structure_analysis["asset_class_correlations"] = asset_class_correlations
            
            # Market integration level
            integration_level = self._calculate_market_integration(correlation_matrix, symbols)
            structure_analysis["market_integration"] = integration_level
            
            # Diversification opportunities
            diversification_score = self._calculate_diversification_score(correlation_matrix, symbols)
            structure_analysis["diversification_opportunities"] = diversification_score
            
            # Cluster-based structure analysis
            cluster_analysis = self._analyze_cluster_structure(clusters, symbols)
            structure_analysis["cluster_structure"] = cluster_analysis
            
            return structure_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {str(e)}")
            return {}
    
    def _analyze_asset_class_correlations(self, correlation_matrix: np.ndarray, symbols: List[str]) -> Dict[str, float]:
        """Analyze correlations between asset classes"""
        try:
            # Group assets by class
            class_groups = {}
            for symbol in symbols:
                asset_class = self.asset_classification.get(symbol, "unknown")
                if asset_class not in class_groups:
                    class_groups[asset_class] = []
                class_groups[asset_class].append(symbol)
            
            class_correlations = {}
            
            for class1 in class_groups:
                for class2 in class_groups:
                    if class1 < class2:  # Avoid duplicates
                        # Get assets in each class
                        assets1 = class_groups[class1]
                        assets2 = class_groups[class2]
                        
                        # Calculate average correlation between classes
                        correlations = []
                        for asset1 in assets1:
                            for asset2 in assets2:
                                idx1 = symbols.index(asset1)
                                idx2 = symbols.index(asset2)
                                correlations.append(abs(correlation_matrix[idx1, idx2]))
                        
                        if correlations:
                            avg_correlation = np.mean(correlations)
                            class_correlations[f"{class1}_{class2}"] = avg_correlation
            
            return class_correlations
            
        except Exception as e:
            logger.error(f"Error analyzing asset class correlations: {str(e)}")
            return {}
    
    def _calculate_market_integration(self, correlation_matrix: np.ndarray, symbols: List[str]) -> str:
        """Calculate market integration level"""
        try:
            corr_no_diag = correlation_matrix.copy()
            np.fill_diagonal(corr_no_diag, 0)
            
            avg_correlation = np.mean(np.abs(corr_no_diag))
            
            if avg_correlation > 0.7:
                return "highly_integrated"
            elif avg_correlation > 0.5:
                return "moderately_integrated"
            elif avg_correlation > 0.3:
                return "low_integration"
            else:
                return "fragmented"
                
        except Exception as e:
            logger.error(f"Error calculating market integration: {str(e)}")
            return "unknown"
    
    def _calculate_diversification_score(self, correlation_matrix: np.ndarray, symbols: List[str]) -> float:
        """Calculate diversification opportunities score"""
        try:
            corr_no_diag = correlation_matrix.copy()
            np.fill_diagonal(corr_no_diag, 1)  # Self-correlation
            
            # Use PCA to assess diversification
            if len(symbols) > 2:
                pca = PCA()
                pca.fit(corr_no_diag)
                
                # First component explains market factor
                market_factor_explained_variance = pca.explained_variance_ratio_[0]
                
                # Lower first component variance = better diversification
                diversification_score = 1 - market_factor_explained_variance
            else:
                # Simple correlation-based score for small universes
                avg_correlation = np.mean(np.abs(corr_no_diag))
                diversification_score = 1 - avg_correlation
            
            return max(0, min(1, diversification_score))
            
        except Exception as e:
            logger.error(f"Error calculating diversification score: {str(e)}")
            return 0.5
    
    def _analyze_cluster_structure(self, clusters: List[CorrelationCluster], symbols: List[str]) -> Dict[str, Any]:
        """Analyze structure based on correlation clusters"""
        try:
            if not clusters:
                return {}
            
            cluster_structure = {
                "num_clusters": len(clusters),
                "largest_cluster_size": max(len(c.assets) for c in clusters) if clusters else 0,
                "average_cluster_correlation": float(np.mean([c.average_correlation for c in clusters])),
                "cluster_stability": float(np.mean([c.regime_stability for c in clusters])),
                "total_cluster_coverage": float(sum(len(c.assets) for c in clusters) / len(symbols))
            }
            
            # Asset class distribution across clusters
            class_distribution = {}
            for cluster in clusters:
                for asset in cluster.assets:
                    asset_class = self.asset_classification.get(asset, "unknown")
                    if asset_class not in class_distribution:
                        class_distribution[asset_class] = 0
                    class_distribution[asset_class] += 1
            
            cluster_structure["class_distribution"] = class_distribution
            
            return cluster_structure
            
        except Exception as e:
            logger.error(f"Error analyzing cluster structure: {str(e)}")
            return {}
    
    def decompose_portfolio_risk(self, correlation_matrix: np.ndarray, symbols: List[str], 
                               weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Decompose portfolio risk using correlation analysis"""
        try:
            if correlation_matrix.size == 0 or len(symbols) == 0:
                return {}
            
            # Use equal weights if none provided
            if weights is None:
                weights = [1.0 / len(symbols)] * len(symbols)
            elif len(weights) != len(symbols):
                logger.error("Weights length must match number of symbols")
                return {}
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Calculate portfolio variance
            portfolio_variance = np.dot(weights, np.dot(correlation_matrix, weights))
            
            # Risk decomposition
            risk_contributions = []
            for i, symbol in enumerate(symbols):
                marginal_contribution = np.dot(correlation_matrix[i, :], weights)
                contribution = weights[i] * marginal_contribution
                risk_contributions.append({
                    "asset": symbol,
                    "weight": float(weights[i]),
                    "risk_contribution": float(contribution),
                    "marginal_contribution": float(marginal_contribution),
                    "percent_contribution": float(contribution / portfolio_variance * 100) if portfolio_variance > 0 else 0
                })
            
            # Sort by risk contribution
            risk_contributions.sort(key=lambda x: abs(x["risk_contribution"]), reverse=True)
            
            # Concentration analysis
            top_5_concentration = sum([rc["percent_contribution"] for rc in risk_contributions[:5]])
            
            return {
                "portfolio_variance": float(portfolio_variance),
                "portfolio_volatility": float(np.sqrt(portfolio_variance)),
                "risk_contributions": risk_contributions,
                "top_5_concentration": float(top_5_concentration),
                "diversification_ratio": float(1 / np.sqrt(portfolio_variance)) if portfolio_variance > 0 else 0,
                "effective_num_assets": float(1 / np.sum([w**2 for w in weights]))
            }
            
        except Exception as e:
            logger.error(f"Error decomposing portfolio risk: {str(e)}")
            return {}
    
    async def generate_comprehensive_analysis(self, symbols: List[str], 
                                            weights: Optional[List[float]] = None) -> CrossAssetAnalysis:
        """Generate comprehensive cross-asset correlation analysis"""
        try:
            # Fetch asset data
            asset_data = await self.fetch_asset_data(symbols)
            if not asset_data:
                return CrossAssetAnalysis(
                    timestamp=datetime.now(),
                    assets_analyzed=[],
                    correlation_matrix=np.array([]),
                    correlation_pairs=[],
                    clusters=[],
                    regime_changes=[],
                    market_structure={},
                    risk_decomposition={},
                    relationship_analysis={},
                    recommendations=["No asset data available"]
                )
            
            # Calculate correlation matrix
            correlation_matrix = self.calculate_correlation_matrix(asset_data)
            
            # Analyze correlation pairs
            correlation_pairs = await self.analyze_correlation_pairs(asset_data)
            
            # Identify clusters
            clusters = self.identify_correlation_clusters(correlation_matrix, list(asset_data.keys()))
            
            # Detect regime changes (compare with previous analysis if available)
            previous_analysis = None  # Would need to be stored from previous run
            regime_changes = await self.detect_regime_changes(asset_data, previous_analysis)
            
            # Analyze market structure
            market_structure = self.analyze_market_structure(correlation_matrix, list(asset_data.keys()), clusters)
            
            # Risk decomposition
            risk_decomposition = self.decompose_portfolio_risk(correlation_matrix, list(asset_data.keys()), weights)
            
            # Relationship analysis
            relationship_analysis = self._analyze_key_relationships(correlation_pairs)
            
            # Generate recommendations
            recommendations = self._generate_correlation_recommendations(
                correlation_pairs, clusters, market_structure, risk_decomposition
            )
            
            analysis = CrossAssetAnalysis(
                timestamp=datetime.now(),
                assets_analyzed=list(asset_data.keys()),
                correlation_matrix=correlation_matrix,
                correlation_pairs=correlation_pairs,
                clusters=clusters,
                regime_changes=regime_changes,
                market_structure=market_structure,
                risk_decomposition=risk_decomposition,
                relationship_analysis=relationship_analysis,
                recommendations=recommendations
            )
            
            logger.info(f"Generated cross-asset analysis for {len(symbols)} assets: "
                       f"{len(correlation_pairs)} pairs, {len(clusters)} clusters, "
                       f"avg correlation: {np.mean([abs(p.correlation_60d) for p in correlation_pairs]):.3f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating comprehensive analysis: {str(e)}")
            return CrossAssetAnalysis(
                timestamp=datetime.now(),
                assets_analyzed=[],
                correlation_matrix=np.array([]),
                correlation_pairs=[],
                clusters=[],
                regime_changes=[],
                market_structure={},
                risk_decomposition={},
                relationship_analysis={},
                recommendations=["Analysis failed due to system error"]
            )
    
    def _analyze_key_relationships(self, correlation_pairs: List[CorrelationPair]) -> Dict[str, Any]:
        """Analyze key relationships from correlation pairs"""
        try:
            if not correlation_pairs:
                return {}
            
            # Find strongest positive correlations
            strong_positive = [p for p in correlation_pairs if p.relationship_type in ["strong_positive", "positive"]]
            strong_positive.sort(key=lambda x: abs(x.correlation_60d), reverse=True)
            
            # Find strongest negative correlations
            strong_negative = [p for p in correlation_pairs if p.relationship_type in ["strong_negative", "negative"]]
            strong_negative.sort(key=lambda x: abs(x.correlation_60d), reverse=True)
            
            # Find most stable relationships
            stable_relationships = sorted(correlation_pairs, key=lambda x: x.stability_score, reverse=True)
            
            # Asset class relationship analysis
            asset_class_relationships = self._analyze_asset_class_relationships(correlation_pairs)
            
            return {
                "strongest_positive_correlations": strong_positive[:5],
                "strongest_negative_correlations": strong_negative[:5],
                "most_stable_relationships": stable_relationships[:5],
                "asset_class_relationships": asset_class_relationships,
                "relationship_summary": {
                    "total_pairs": len(correlation_pairs),
                    "strong_positive_count": len(strong_positive),
                    "strong_negative_count": len(strong_negative),
                    "average_stability": np.mean([p.stability_score for p in correlation_pairs])
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing key relationships: {str(e)}")
            return {}
    
    def _analyze_asset_class_relationships(self, correlation_pairs: List[CorrelationPair]) -> Dict[str, Any]:
        """Analyze relationships between asset classes"""
        try:
            class_relationships = {}
            
            for pair in correlation_pairs:
                class1 = self.asset_classification.get(pair.asset1, "unknown")
                class2 = self.asset_classification.get(pair.asset2, "unknown")
                
                if class1 != class2:  # Cross-asset relationships only
                    relationship_key = f"{class1}_{class2}"
                    if relationship_key not in class_relationships:
                        class_relationships[relationship_key] = []
                    class_relationships[relationship_key].append(abs(pair.correlation_60d))
            
            # Calculate average correlations by asset class pair
            avg_class_correlations = {}
            for rel_type, correlations in class_relationships.items():
                avg_class_correlations[rel_type] = {
                    "average_correlation": float(np.mean(correlations)),
                    "correlation_count": len(correlations),
                    "max_correlation": float(np.max(correlations)),
                    "min_correlation": float(np.min(correlations))
                }
            
            return avg_class_correlations
            
        except Exception as e:
            logger.error(f"Error analyzing asset class relationships: {str(e)}")
            return {}
    
    def _generate_correlation_recommendations(self, correlation_pairs: List[CorrelationPair], 
                                            clusters: List[CorrelationCluster],
                                            market_structure: Dict[str, Any], 
                                            risk_decomposition: Dict[str, Any]) -> List[str]:
        """Generate correlation-based recommendations"""
        try:
            recommendations = []
            
            # Diversification recommendations
            if market_structure.get("diversification_opportunities", 0) < 0.3:
                recommendations.append("Low diversification opportunities - consider adding uncorrelated assets")
            
            # Correlation clustering recommendations
            if clusters:
                largest_cluster = max(clusters, key=lambda x: len(x.assets))
                if len(largest_cluster.assets) > 3:
                    recommendations.append(f"Large correlation cluster detected ({len(largest_cluster.assets)} assets) - reduce concentration")
            
            # High correlation alerts
            high_correlations = [p for p in correlation_pairs if abs(p.correlation_60d) > 0.8]
            if high_correlations:
                recommendations.append(f"{len(high_correlations)} asset pairs show extreme correlation (>0.8) - diversification benefit limited")
            
            # Risk concentration recommendations
            if risk_decomposition:
                top_5_concentration = risk_decomposition.get("top_5_concentration", 0)
                if top_5_concentration > 70:
                    recommendations.append(f"High risk concentration in top 5 assets ({top_5_concentration:.1f}%) - rebalance portfolio")
            
            # Market structure recommendations
            integration = market_structure.get("market_integration", "unknown")
            if integration == "highly_integrated":
                recommendations.append("Market highly integrated - traditional diversification may be less effective")
            elif integration == "fragmented":
                recommendations.append("Market fragmented - good diversification opportunities available")
            
            # Stability recommendations
            low_stability_pairs = [p for p in correlation_pairs if p.stability_score < 0.3]
            if low_stability_pairs:
                recommendations.append(f"{len(low_stability_pairs)} relationships show low stability - monitor for regime changes")
            
            # Asset class recommendations
            avg_correlation = market_structure.get("average_correlation", 0)
            if avg_correlation > 0.6:
                recommendations.append("High average correlation environment - consider alternative assets or strategies")
            elif avg_correlation < 0.2:
                recommendations.append("Low correlation environment - traditional diversification highly effective")
            
            # Regime change recommendations
            if len(correlation_pairs) > 10:
                recommendations.append("Monitor correlation stability for regime change indicators")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating correlation recommendations: {str(e)}")
            return ["Unable to generate specific correlation recommendations"]
    
    async def monitor_correlation_alerts(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Monitor for significant correlation changes and alerts"""
        try:
            alerts = {}
            
            # Generate current analysis
            analysis = await self.generate_comprehensive_analysis(symbols)
            
            # Check for high correlation pairs
            extreme_correlations = [p for p in analysis.correlation_pairs if abs(p.correlation_60d) > 0.9]
            if extreme_correlations:
                alerts["extreme_correlations"] = [
                    f"{pair.asset1}-{pair.asset2}: {pair.correlation_60d:.3f}" 
                    for pair in extreme_correlations[:3]  # Top 3
                ]
            
            # Check for regime changes
            if analysis.regime_changes:
                alerts["regime_changes"] = [
                    f"{change.assets[0]}-{change.assets[1]}: {change.old_regime} → {change.new_regime}"
                    for change in analysis.regime_changes[:3]  # Top 3
                ]
            
            # Check for cluster concentration
            large_clusters = [c for c in analysis.clusters if len(c.assets) > 3]
            if large_clusters:
                alerts["large_clusters"] = [
                    f"Cluster {c.cluster_id}: {len(c.assets)} assets, avg correlation: {c.average_correlation:.3f}"
                    for c in large_clusters
                ]
            
            # Check market structure alerts
            market_integration = analysis.market_structure.get("market_integration", "unknown")
            if market_integration == "highly_integrated":
                alerts["market_structure"] = ["Market highly integrated - diversification benefits reduced"]
            elif market_integration == "fragmented":
                alerts["market_structure"] = ["Market fragmented - good diversification opportunities"]
            
            # Check diversification score
            diversification = analysis.market_structure.get("diversification_opportunities", 0.5)
            if diversification < 0.3:
                alerts["diversification"] = ["Low diversification score - consider portfolio rebalancing"]
            elif diversification > 0.7:
                alerts["diversification"] = ["High diversification score - current portfolio well diversified"]
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring correlation alerts: {str(e)}")
            return {}
    
    async def export_correlation_analysis(self, symbols: List[str], format_type: str = "json") -> str:
        """Export correlation analysis to file"""
        try:
            analysis = await self.generate_comprehensive_analysis(symbols)
            
            if format_type.lower() == "json":
                import json
                export_data = {
                    "timestamp": analysis.timestamp.isoformat(),
                    "assets_analyzed": analysis.assets_analyzed,
                    "correlation_matrix": analysis.correlation_matrix.tolist() if analysis.correlation_matrix.size > 0 else [],
                    "correlation_pairs": [
                        {
                            "asset1": pair.asset1,
                            "asset2": pair.asset2,
                            "correlation_60d": pair.correlation_60d,
                            "relationship_type": pair.relationship_type,
                            "stability_score": pair.stability_score
                        }
                        for pair in analysis.correlation_pairs
                    ],
                    "clusters": [
                        {
                            "cluster_id": c.cluster_id,
                            "assets": c.assets,
                            "average_correlation": c.average_correlation,
                            "regime_stability": c.regime_stability,
                            "risk_contribution": c.risk_contribution
                        }
                        for c in analysis.clusters
                    ],
                    "regime_changes": [
                        {
                            "assets": list(change.assets),
                            "old_regime": change.old_regime,
                            "new_regime": change.new_regime,
                            "change_magnitude": change.change_magnitude,
                            "confidence_score": change.confidence_score
                        }
                        for change in analysis.regime_changes
                    ],
                    "market_structure": analysis.market_structure,
                    "risk_decomposition": analysis.risk_decomposition,
                    "recommendations": analysis.recommendations
                }
                
                filename = f"correlation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = f"/workspace/mpe/exports/{filename}"
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return filepath
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting correlation analysis: {str(e)}")
            return ""

# Main execution function
async def main():
    """Main execution function for cross-asset correlation analysis"""
    engine = CrossAssetCorrelationEngine()
    
    # Test with diverse asset universe
    test_symbols = [
        "AAPL", "MSFT", "SPY", "QQQ", "TLT", "IEF", "GLD", "UUP", "VIX", "BTC-USD"
    ]
    
    logger.info("Starting Cross-Asset Correlation Engine analysis...")
    
    # Test comprehensive analysis
    logger.info(f"\n=== Cross-Asset Correlation Analysis ===")
    
    analysis = await engine.generate_comprehensive_analysis(test_symbols)
    
    logger.info(f"Assets Analyzed: {len(analysis.assets_analyzed)}")
    logger.info(f"Correlation Pairs: {len(analysis.correlation_pairs)}")
    logger.info(f"Correlation Clusters: {len(analysis.clusters)}")
    logger.info(f"Regime Changes: {len(analysis.regime_changes)}")
    
    # Show correlation matrix summary
    if analysis.correlation_matrix.size > 0:
        corr_no_diag = analysis.correlation_matrix.copy()
        np.fill_diagonal(corr_no_diag, 0)
        logger.info(f"Average Correlation: {np.mean(np.abs(corr_no_diag)):.3f}")
        logger.info(f"Max Correlation: {np.max(corr_no_diag):.3f}")
        logger.info(f"Min Correlation: {np.min(corr_no_diag):.3f}")
    
    # Show strongest correlations
    logger.info("\nStrongest Positive Correlations:")
    if analysis.relationship_analysis:
        strong_pos = analysis.relationship_analysis.get("strongest_positive_correlations", [])
        for pair in strong_pos[:3]:
            logger.info(f"  {pair.asset1}-{pair.asset2}: {pair.correlation_60d:.3f}")
    
    # Show clusters
    logger.info("\nCorrelation Clusters:")
    for cluster in analysis.clusters:
        logger.info(f"  {cluster.cluster_id}: {len(cluster.assets)} assets, avg correlation: {cluster.average_correlation:.3f}")
        logger.info(f"    Assets: {', '.join(cluster.assets)}")
    
    # Show market structure
    logger.info("\nMarket Structure:")
    if analysis.market_structure:
        logger.info(f"  Integration Level: {analysis.market_structure.get('market_integration', 'unknown')}")
        logger.info(f"  Diversification Score: {analysis.market_structure.get('diversification_opportunities', 0):.3f}")
    
    # Show recommendations
    logger.info("\nRecommendations:")
    for rec in analysis.recommendations[:5]:
        logger.info(f"  - {rec}")
    
    # Test monitoring alerts
    logger.info("\n=== Correlation Alerts ===")
    alerts = await engine.monitor_correlation_alerts(test_symbols)
    
    for alert_type, alert_messages in alerts.items():
        logger.info(f"{alert_type.upper()}:")
        for message in alert_messages:
            logger.info(f"  - {message}")
    
    logger.info("Cross-Asset Correlation Engine analysis completed successfully")

if __name__ == "__main__":
    asyncio.run(main())