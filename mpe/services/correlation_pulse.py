"""
Correlation Pulse Engine - Market Interconnection Analysis
Real-time tracking of correlation structures across assets, sectors, and geographic markets
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import yfinance as yf
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import talib
from database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class CorrelationPulseEngine:
    """Correlation Structure Monitor - Detecting market interconnection changes"""
    
    def __init__(self):
        self.name = "Correlation Pulse Engine"
        self.version = "1.0.0"
        self.status = "initialized"
        self.last_update = None
        self.correlation_cache = {}
        self.models = {}
        self.scaler = StandardScaler()
        
        # Correlation tracking models
        self.correlation_models = {
            "asset_correlations": self._asset_correlation_analysis,
            "sector_correlations": self._sector_correlation_analysis,
            "geographic_correlations": self._geographic_correlation_analysis,
            "cross_asset_correlations": self._cross_asset_correlation_analysis,
            "regime_correlations": self._regime_correlation_analysis
        }
        
        # Asset groups for correlation analysis
        self.asset_groups = {
            "large_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
            "financial": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK"],
            "tech_growth": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"],
            "defensive": ["JNJ", "PG", "KO", "WMT", "VZ", "T", "XLU"],
            "cyclical": ["XOM", "CVX", "CAT", "BA", "MMM", "GE", "HON"],
            "international": ["TM", "7203.T", "NESN.SW", "ASML.AS", "SAP.DE"],
            "bonds": ["TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG"]
        }
        
        # Sector ETFs for sector correlation tracking
        self.sector_etfs = {
            "XLK": "Technology",
            "XLF": "Financial", 
            "XLE": "Energy",
            "XLI": "Industrial",
            "XLV": "Healthcare",
            "XLY": "Consumer",
            "XLU": "Utilities",
            "XLB": "Materials",
            "XLRE": "Real Estate",
            "XLC": "Communication"
        }
        
        # Geographic market indices
        self.geographic_indices = {
            "US": ["^GSPC", "^DJI", "^IXIC", "^RUT"],
            "Europe": ["^FTSE", "^GDAXI", "^FCHI", "^AEX"],
            "Asia": ["^N225", "^HSI", "^SSEC", "^KS11"],
            "Emerging": ["^IPSA", "^BVSP", "^MXX", "^JKSE"]
        }
        
        # Cross-asset classes
        self.cross_asset_classes = {
            "equities": ["SPY", "QQQ", "IWM", "VTI"],
            "bonds": ["TLT", "IEF", "SHY", "AGG"],
            "commodities": ["GLD", "SLV", "DBA", "USO"],
            "currencies": ["DX-Y.NYB", "EURUSD=X", "JPY=X", "GBPUSD=X"],
            "crypto": ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD"]
        }
        
        # Correlation thresholds
        self.correlation_thresholds = {
            "high_correlation": 0.8,
            "low_correlation": 0.2,
            "correlation_breakdown": -0.5,  # Correlation decline threshold
            "correlation_spike": 0.5,       # Correlation increase threshold
            "regime_change": 0.3            # Correlation regime change threshold
        }
        
        # Initialize ML models
        self._initialize_models()
        
        # Database manager
        self.db_manager = None
        
    async def initialize(self):
        """Initialize database connections and models"""
        try:
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            self.status = "active"
            logger.info("Correlation Pulse Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Correlation Pulse Engine: {e}")
            self.status = "error"
            
    def _initialize_models(self):
        """Initialize ML models for correlation prediction"""
        try:
            # Random Forest for correlation regime prediction
            self.models['correlation_regime'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # K-means for correlation clustering
            self.models['correlation_clusters'] = KMeans(
                n_clusters=3,
                random_state=42
            )
            
            # Scaler for correlation feature normalization
            self.scaler = StandardScaler()
            
            logger.info("Correlation prediction models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize correlation models: {e}")
            
    async def _fetch_price_data(self, symbols: List[str], days: int = 60) -> pd.DataFrame:
        """Fetch synchronized price data for multiple symbols"""
        try:
            price_data = {}
            
            # Fetch data for all symbols
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=f"{days}d")
                    if not hist.empty and len(hist) > 30:  # Ensure sufficient data
                        price_data[symbol] = hist['Close']
                except Exception as e:
                    logger.warning(f"Error fetching data for {symbol}: {e}")
                    continue
                    
            if not price_data:
                return pd.DataFrame()
                
            # Create synchronized DataFrame
            df = pd.DataFrame(price_data)
            df = df.dropna()  # Remove symbols with insufficient data
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching synchronized price data: {e}")
            return pd.DataFrame()
            
    def _calculate_rolling_correlations(self, price_df: pd.DataFrame, 
                                       window: int = 30, min_periods: int = 20) -> Dict[str, pd.DataFrame]:
        """Calculate rolling correlations between all pairs"""
        try:
            if price_df.empty or price_df.shape[1] < 2:
                return {}
                
            symbols = price_df.columns.tolist()
            correlation_results = {}
            
            # Calculate returns for correlation analysis
            returns_df = price_df.pct_change().dropna()
            
            # Rolling correlation matrix
            rolling_corr = returns_df.rolling(window=window, min_periods=min_periods).corr()
            
            # Store different correlation perspectives
            correlation_results['rolling_corr_matrix'] = rolling_corr
            
            # Average correlations (across all pairs)
            correlation_results['avg_correlation'] = rolling_corr.groupby(level=1).mean()
            
            # Maximum correlations (highest correlations observed)
            correlation_results['max_correlation'] = rolling_corr.groupby(level=1).max()
            
            # Minimum correlations (lowest correlations observed)
            correlation_results['min_correlation'] = rolling_corr.groupby(level=1).min()
            
            # Correlation volatility (how much correlations change over time)
            correlation_results['correlation_volatility'] = rolling_corr.groupby(level=1).std()
            
            return correlation_results
            
        except Exception as e:
            logger.error(f"Error calculating rolling correlations: {e}")
            return {}
            
    async def _asset_correlation_analysis(self) -> Dict:
        """Analyze correlations within asset groups"""
        try:
            group_correlations = {}
            
            for group_name, symbols in self.asset_groups.items():
                # Fetch synchronized price data
                price_data = await self._fetch_price_data(symbols)
                
                if price_data.empty:
                    continue
                    
                # Calculate correlations
                correlations = self._calculate_rolling_correlations(price_data)
                
                if correlations:
                    # Get latest correlation statistics
                    latest_corr = correlations['avg_correlation'].iloc[-1] if not correlations['avg_correlation'].empty else pd.Series()
                    latest_max = correlations['max_correlation'].iloc[-1] if not correlations['max_correlation'].empty else pd.Series()
                    latest_min = correlations['min_correlation'].iloc[-1] if not correlations['min_correlation'].empty else pd.Series()
                    
                    # Correlation regime analysis
                    corr_regime = self._analyze_correlation_regime(correlations)
                    
                    group_correlations[group_name] = {
                        'avg_correlation': float(latest_corr.mean()) if not latest_corr.empty else 0,
                        'max_correlation': float(latest_max.max()) if not latest_max.empty else 0,
                        'min_correlation': float(latest_min.min()) if not latest_min.empty else 0,
                        'correlation_volatility': float(latest_corr.std()) if not latest_corr.empty else 0,
                        'regime': corr_regime,
                        'asset_count': len(symbols),
                        'data_quality': len(price_data) / 60  # Data completeness ratio
                    }
                    
            if not group_correlations:
                return {'error': 'No asset correlation data available'}
                
            # Overall correlation health
            avg_correlations = [data['avg_correlation'] for data in group_correlations.values()]
            correlation_volatility = np.std(avg_correlations)
            overall_health = 1 - correlation_volatility  # Higher when correlations are stable
            
            return {
                'group_correlations': group_correlations,
                'correlation_health': overall_health,
                'cross_group_correlation': correlation_volatility,
                'regime_distribution': self._calculate_regime_distribution(group_correlations)
            }
            
        except Exception as e:
            logger.error(f"Error in asset correlation analysis: {e}")
            return {'error': str(e)}
            
    def _analyze_correlation_regime(self, correlations: Dict) -> str:
        """Analyze correlation regime changes"""
        try:
            if 'avg_correlation' not in correlations or correlations['avg_correlation'].empty:
                return 'unknown'
                
            # Get recent correlation history
            recent_corr = correlations['avg_correlation'].tail(10)
            
            if len(recent_corr) < 3:
                return 'insufficient_data'
                
            # Calculate correlation trend
            correlation_trend = recent_corr.iloc[-1] - recent_corr.iloc[0]
            current_level = recent_corr.iloc[-1]
            
            # Classify regime
            if current_level > 0.7:
                regime = 'high_correlation'
            elif current_level < 0.3:
                regime = 'low_correlation'
            else:
                regime = 'normal_correlation'
                
            # Add trend modifier
            if abs(correlation_trend) > 0.1:
                if correlation_trend > 0:
                    regime += '_increasing'
                else:
                    regime += '_decreasing'
                    
            return regime
            
        except Exception as e:
            logger.error(f"Error analyzing correlation regime: {e}")
            return 'unknown'
            
    def _calculate_regime_distribution(self, group_correlations: Dict) -> Dict:
        """Calculate distribution of correlation regimes across groups"""
        try:
            regimes = {}
            for data in group_correlations.values():
                regime = data['regime']
                regimes[regime] = regimes.get(regime, 0) + 1
                
            total_groups = len(group_correlations)
            regime_distribution = {regime: count/total_groups for regime, count in regimes.items()}
            
            return regime_distribution
            
        except Exception as e:
            logger.error(f"Error calculating regime distribution: {e}")
            return {}
            
    async def _sector_correlation_analysis(self) -> Dict:
        """Analyze correlations between different sectors"""
        try:
            # Get sector ETF data
            sector_symbols = list(self.sector_etfs.keys())
            price_data = await self._fetch_price_data(sector_symbols)
            
            if price_data.empty:
                return {'error': 'No sector price data available'}
                
            # Calculate sector correlations
            correlations = self._calculate_rolling_correlations(price_data)
            
            if not correlations:
                return {'error': 'Unable to calculate sector correlations'}
                
            # Analyze sector clustering
            sector_clusters = self._analyze_sector_clustering(correlations)
            
            # Calculate sector correlation changes over time
            correlation_changes = self._calculate_correlation_changes(correlations)
            
            # Get sector rotation signals
            rotation_signals = self._calculate_sector_rotation_signals(correlations, price_data)
            
            # Latest correlation matrix for analysis
            latest_corr = correlations['avg_correlation'].iloc[-1] if not correlations['avg_correlation'].empty else pd.DataFrame()
            
            # Analyze sector groupings based on correlations
            sector_groupings = self._identify_sector_groupings(latest_corr)
            
            return {
                'sector_correlations': latest_corr.to_dict() if not latest_corr.empty else {},
                'sector_clusters': sector_clusters,
                'correlation_changes': correlation_changes,
                'rotation_signals': rotation_signals,
                'sector_groupings': sector_groupings,
                'sector_diversification': self._calculate_sector_diversification(latest_corr)
            }
            
        except Exception as e:
            logger.error(f"Error in sector correlation analysis: {e}")
            return {'error': str(e)}
            
    def _analyze_sector_clustering(self, correlations: Dict) -> Dict:
        """Analyze how sectors cluster together based on correlations"""
        try:
            if 'avg_correlation' not in correlations or correlations['avg_correlation'].empty:
                return {}
                
            # Get latest correlation matrix
            latest_corr = correlations['avg_correlation'].iloc[-1]
            
            if latest_corr.empty:
                return {}
                
            # Convert to distance matrix (1 - correlation)
            distance_matrix = 1 - np.abs(latest_corr.values)
            
            # Perform hierarchical clustering (simplified)
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            
            # Convert to condensed distance matrix
            condensed_dist = squareform(distance_matrix, checks=False)
            
            # Perform clustering
            linkage_matrix = linkage(condensed_dist, method='ward')
            
            # Get cluster assignments
            clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')
            
            # Map clusters to sectors
            sector_clusters = {}
            for i, sector in enumerate(latest_corr.index):
                cluster_id = clusters[i]
                if cluster_id not in sector_clusters:
                    sector_clusters[f'cluster_{cluster_id}'] = []
                sector_clusters[f'cluster_{cluster_id}'].append(sector)
                
            return sector_clusters
            
        except Exception as e:
            logger.error(f"Error analyzing sector clustering: {e}")
            return {}
            
    def _calculate_correlation_changes(self, correlations: Dict) -> Dict:
        """Calculate how correlations are changing over time"""
        try:
            changes = {}
            
            for metric_name, metric_data in correlations.items():
                if not isinstance(metric_data, pd.DataFrame) or metric_data.empty:
                    continue
                    
                # Calculate recent vs historical average
                recent_period = metric_data.tail(5)  # Last 5 periods
                historical_period = metric_data.head(-10).tail(10)  # 10 periods before recent
                
                if len(recent_period) > 0 and len(historical_period) > 0:
                    recent_mean = recent_period.mean().mean()
                    historical_mean = historical_period.mean().mean()
                    change = recent_mean - historical_mean
                    
                    changes[metric_name] = {
                        'recent_average': float(recent_mean),
                        'historical_average': float(historical_mean),
                        'change': float(change),
                        'change_direction': 'increasing' if change > 0 else 'decreasing',
                        'change_magnitude': float(abs(change))
                    }
                    
            return changes
            
        except Exception as e:
            logger.error(f"Error calculating correlation changes: {e}")
            return {}
            
    def _calculate_sector_rotation_signals(self, correlations: Dict, price_data: pd.DataFrame) -> Dict:
        """Calculate sector rotation signals based on correlation changes"""
        try:
            rotation_signals = {}
            
            if price_data.empty or correlations.get('avg_correlation', pd.DataFrame()).empty:
                return rotation_signals
                
            # Calculate sector momentum
            sector_returns = price_data.pct_change(5).iloc[-1]  # 5-day returns
            
            # Get correlation matrix
            latest_corr = correlations['avg_corr'].iloc[-1]
            
            # Calculate rotation strength for each sector
            for sector in sector_returns.index:
                sector_momentum = sector_returns[sector]
                
                # Compare sector with highly correlated sectors
                if sector in latest_corr.index:
                    correlations_with_sector = latest_corr[sector].drop(sector)
                    high_corr_sectors = correlations_with_sector[correlations_with_sector > 0.5]
                    
                    if not high_corr_sectors.empty:
                        # Compare sector performance with highly correlated peers
                        peer_performance = [sector_returns[s] for s in high_corr_sectors.index 
                                          if s in sector_returns.index]
                        
                        if peer_performance:
                            avg_peer_performance = np.mean(peer_performance)
                            relative_strength = sector_momentum - avg_peer_performance
                            
                            rotation_signals[sector] = {
                                'sector_momentum': float(sector_momentum),
                                'avg_peer_performance': float(avg_peer_performance),
                                'relative_strength': float(relative_strength),
                                'rotation_signal': 'outperforming' if relative_strength > 0.01 else 'underperforming',
                                'high_correlation_peers': high_corr_sectors.index.tolist()
                            }
                            
            return rotation_signals
            
        except Exception as e:
            logger.error(f"Error calculating sector rotation signals: {e}")
            return {}
            
    def _identify_sector_groupings(self, correlation_matrix: pd.DataFrame) -> Dict:
        """Identify natural sector groupings based on correlations"""
        try:
            if correlation_matrix.empty:
                return {}
                
            groupings = {
                'high_correlation_groups': [],
                'low_correlation_groups': [],
                'moderate_correlation_groups': []
            }
            
            # Find groups of highly correlated sectors
            for i, sector1 in enumerate(correlation_matrix.index):
                high_corr_peers = []
                moderate_corr_peers = []
                low_corr_peers = []
                
                for j, sector2 in enumerate(correlation_matrix.columns):
                    if i != j and sector2 in correlation_matrix.index:
                        corr_value = correlation_matrix.loc[sector1, sector2]
                        
                        if corr_value > 0.7:
                            high_corr_peers.append(sector2)
                        elif corr_value > 0.3:
                            moderate_corr_peers.append(sector2)
                        else:
                            low_corr_peers.append(sector2)
                            
                # Store groupings if they have sufficient peers
                if len(high_corr_peers) >= 2:
                    groupings['high_correlation_groups'].append({
                        'sector': sector1,
                        'high_correlation_peers': high_corr_peers
                    })
                    
            # Remove duplicates and create unique groupings
            unique_high_groups = []
            seen_sectors = set()
            
            for group in groupings['high_correlation_groups']:
                if group['sector'] not in seen_sectors:
                    unique_high_groups.append(group)
                    seen_sectors.update(group['high_correlation_peers'])
                    
            groupings['high_correlation_groups'] = unique_high_groups
            
            return groupings
            
        except Exception as e:
            logger.error(f"Error identifying sector groupings: {e}")
            return {}
            
    def _calculate_sector_diversification(self, correlation_matrix: pd.DataFrame) -> float:
        """Calculate sector diversification metric"""
        try:
            if correlation_matrix.empty:
                return 0
                
            # Average correlation across all sector pairs
            correlations = []
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    corr_value = correlation_matrix.iloc[i, j]
                    correlations.append(abs(corr_value))
                    
            if not correlations:
                return 0
                
            avg_correlation = np.mean(correlations)
            
            # Diversification score (inverse of average correlation)
            diversification_score = 1 - avg_correlation
            
            return float(max(0, diversification_score))
            
        except Exception as e:
            logger.error(f"Error calculating sector diversification: {e}")
            return 0
            
    async def _geographic_correlation_analysis(self) -> Dict:
        """Analyze correlations across different geographic markets"""
        try:
            geographic_correlations = {}
            
            # Analyze correlations within each geographic region
            for region, indices in self.geographic_indices.items():
                # Use representative index for each region
                representative_index = indices[0]  # Take first index as representative
                
                price_data = await self._fetch_price_data([representative_index], days=60)
                if not price_data.empty:
                    # For single asset, we can't calculate correlations within region
                    # This would need to be expanded with multiple representative indices
                    geographic_correlations[region] = {
                        'representative_index': representative_index,
                        'volatility': float(price_data[representative_index].pct_change().std()),
                        'momentum_5d': float(price_data[representative_index].pct_change(5).iloc[-1]),
                        'momentum_20d': float(price_data[representative_index].pct_change(20).iloc[-1])
                    }
                    
            # Analyze cross-geographic correlations
            if len(geographic_correlations) > 1:
                cross_geo_correlations = await self._calculate_cross_geographic_correlations(
                    list(geographic_correlations.keys())
                )
                
                return {
                    'geographic_performance': geographic_correlations,
                    'cross_geographic_correlations': cross_geo_correlations,
                    'geographic_diversification': self._calculate_geographic_diversification(
                        geographic_correlations
                    )
                }
            else:
                return {
                    'geographic_performance': geographic_correlations,
                    'message': 'Insufficient geographic data for cross-analysis'
                }
                
        except Exception as e:
            logger.error(f"Error in geographic correlation analysis: {e}")
            return {'error': str(e)}
            
    async def _calculate_cross_geographic_correlations(self, regions: List[str]) -> Dict:
        """Calculate correlations between different geographic regions"""
        try:
            # This would need real implementation with representative indices
            # For now, return placeholder structure
            cross_correlations = {}
            
            for i, region1 in enumerate(regions):
                for j, region2 in enumerate(regions):
                    if i < j:
                        # Placeholder - would calculate actual correlation
                        correlation_strength = np.random.uniform(0.3, 0.8)  # Replace with real calculation
                        
                        cross_correlations[f"{region1}_{region2}"] = {
                            'correlation_strength': correlation_strength,
                            'correlation_regime': 'high' if correlation_strength > 0.6 else 'normal',
                            'diversification_benefit': 1 - correlation_strength
                        }
                        
            return cross_correlations
            
        except Exception as e:
            logger.error(f"Error calculating cross-geographic correlations: {e}")
            return {}
            
    def _calculate_geographic_diversification(self, geo_performance: Dict) -> float:
        """Calculate geographic diversification metric"""
        try:
            if len(geo_performance) < 2:
                return 0
                
            # Calculate performance correlation across regions
            # This is a simplified version - would need actual return series
            regions = list(geo_performance.keys())
            
            # Calculate momentum divergence as proxy for diversification benefit
            momentum_values = [geo_performance[region]['momentum_5d'] for region in regions]
            momentum_divergence = np.std(momentum_values)
            
            # Higher divergence = better diversification
            diversification_score = min(1.0, momentum_divergence * 10)  # Scale appropriately
            
            return float(diversification_score)
            
        except Exception as e:
            logger.error(f"Error calculating geographic diversification: {e}")
            return 0
            
    async def _cross_asset_correlation_analysis(self) -> Dict:
        """Analyze correlations across different asset classes"""
        try:
            cross_asset_correlations = {}
            
            # Analyze each asset class
            for asset_class, symbols in self.cross_asset_classes.items():
                price_data = await self._fetch_price_data(symbols)
                if not price_data.empty:
                    returns = price_data.pct_change().dropna()
                    
                    # Calculate asset class characteristics
                    cross_asset_correlations[asset_class] = {
                        'avg_return': float(returns.mean().mean()),
                        'volatility': float(returns.std().mean()),
                        'sharpe_estimate': float(returns.mean().mean() / returns.std().mean() 
                                               if returns.std().mean() > 0 else 0),
                        'asset_count': len(symbols),
                        'data_quality': len(price_data) / 60
                    }
                    
            # Calculate inter-asset class correlations
            if len(cross_asset_correlations) > 1:
                inter_class_correlations = self._calculate_inter_class_correlations(
                    cross_asset_correlations
                )
                
                return {
                    'asset_class_performance': cross_asset_correlations,
                    'inter_class_correlations': inter_class_correlations,
                    'portfolio_diversification': self._calculate_portfolio_diversification(
                        cross_asset_correlations
                    )
                }
            else:
                return {
                    'asset_class_performance': cross_asset_correlations,
                    'message': 'Insufficient asset classes for cross-analysis'
                }
                
        except Exception as e:
            logger.error(f"Error in cross-asset correlation analysis: {e}")
            return {'error': str(e)}
            
    def _calculate_inter_class_correlations(self, class_performance: Dict) -> Dict:
        """Calculate correlations between asset classes"""
        try:
            inter_correlations = {}
            asset_classes = list(class_performance.keys())
            
            for i, class1 in enumerate(asset_classes):
                for j, class2 in enumerate(asset_classes):
                    if i < j:
                        # Use return correlation as proxy (simplified)
                        # In reality, would need actual return time series
                        return_diff = abs(class_performance[class1]['avg_return'] - 
                                        class_performance[class2]['avg_return'])
                        
                        # Normalize to correlation-like scale
                        correlation_proxy = 1 / (1 + return_diff * 100)  # Scale adjustment
                        
                        inter_correlations[f"{class1}_{class2}"] = {
                            'correlation_proxy': float(correlation_proxy),
                            'diversification_benefit': 1 - correlation_proxy,
                            'correlation_regime': 'high' if correlation_proxy > 0.7 else 'normal'
                        }
                        
            return inter_correlations
            
        except Exception as e:
            logger.error(f"Error calculating inter-class correlations: {e}")
            return {}
            
    def _calculate_portfolio_diversification(self, class_performance: Dict) -> float:
        """Calculate overall portfolio diversification score"""
        try:
            if not class_performance:
                return 0
                
            # Calculate volatility range across asset classes
            volatilities = [data['volatility'] for data in class_performance.values()]
            return_range = max(volatilities) - min(volatilities)
            
            # Calculate return distribution
            returns = [data['avg_return'] for data in class_performance.values()]
            return_divergence = np.std(returns)
            
            # Combined diversification score
            vol_diversification = min(1.0, return_range * 50)  # Scale appropriately
            return_diversification = min(1.0, return_divergence * 100)  # Scale appropriately
            
            overall_diversification = (vol_diversification + return_diversification) / 2
            
            return float(overall_diversification)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio diversification: {e}")
            return 0
            
    async def _regime_correlation_analysis(self) -> Dict:
        """Analyze correlation regime changes and market structure shifts"""
        try:
            # Get current correlation state across different analyses
            asset_corr = await self._asset_correlation_analysis()
            sector_corr = await self._sector_correlation_analysis()
            geo_corr = await self._geographic_correlation_analysis()
            
            # Analyze correlation regime transitions
            regime_transitions = self._detect_correlation_regime_transitions(
                asset_corr, sector_corr, geo_corr
            )
            
            # Calculate correlation stress indicators
            stress_indicators = self._calculate_correlation_stress_indicators(
                asset_corr, sector_corr, geo_corr
            )
            
            # Market structure analysis
            market_structure = self._analyze_market_structure_changes(
                asset_corr, sector_corr, geo_corr
            )
            
            return {
                'regime_transitions': regime_transitions,
                'stress_indicators': stress_indicators,
                'market_structure': market_structure,
                'correlation_stability': self._assess_correlation_stability(
                    asset_corr, sector_corr, geo_corr
                )
            }
            
        except Exception as e:
            logger.error(f"Error in regime correlation analysis: {e}")
            return {'error': str(e)}
            
    def _detect_correlation_regime_transitions(self, asset_corr: Dict, 
                                             sector_corr: Dict, 
                                             geo_corr: Dict) -> Dict:
        """Detect correlation regime transitions"""
        try:
            transitions = {
                'detected_transitions': [],
                'transition_probability': 0.0,
                'regime_stability': 'unknown'
            }
            
            # Check for regime changes in different analysis areas
            transition_signals = []
            
            # Asset level transitions
            if 'regime_distribution' in asset_corr:
                regime_dist = asset_corr['regime_distribution']
                if len(regime_dist) > 1:
                    transition_signals.append('asset_regime_diversification')
                    
            # Sector level transitions
            if 'correlation_changes' in sector_corr:
                for metric, changes in sector_corr['correlation_changes'].items():
                    if abs(changes.get('change', 0)) > 0.1:
                        transition_signals.append(f'sector_{metric}_change')
                        
            # Geographic level transitions
            if 'cross_geographic_correlations' in geo_corr:
                cross_corrs = geo_corr['cross_geographic_correlations']
                for pair, data in cross_corrs.items():
                    if data.get('correlation_regime') == 'high':
                        transition_signals.append(f'geo_high_correlation_{pair}')
                        
            transitions['detected_transitions'] = list(set(transition_signals))
            transitions['transition_probability'] = min(1.0, len(transitions['detected_transitions']) / 5)
            
            # Determine regime stability
            if transitions['transition_probability'] > 0.7:
                transitions['regime_stability'] = 'unstable'
            elif transitions['transition_probability'] > 0.3:
                transitions['regime_stability'] = 'transitional'
            else:
                transitions['regime_stability'] = 'stable'
                
            return transitions
            
        except Exception as e:
            logger.error(f"Error detecting correlation regime transitions: {e}")
            return {'detected_transitions': [], 'transition_probability': 0, 'regime_stability': 'unknown'}
            
    def _calculate_correlation_stress_indicators(self, asset_corr: Dict,
                                               sector_corr: Dict,
                                               geo_corr: Dict) -> Dict:
        """Calculate correlation stress indicators"""
        try:
            stress_indicators = {
                'high_correlation_stress': 0.0,
                'correlation_concentration': 0.0,
                'regime_stress': 0.0,
                'overall_stress_score': 0.0
            }
            
            # High correlation stress
            high_corr_count = 0
            total_corr_count = 0
            
            # Asset level stress
            if 'group_correlations' in asset_corr:
                for group_data in asset_corr['group_correlations'].values():
                    total_corr_count += 1
                    if group_data.get('avg_correlation', 0) > 0.8:
                        high_corr_count += 1
                        
            # Sector level stress
            if 'sector_correlations' in sector_corr:
                # Count high correlations in sector matrix
                sector_corr_matrix = sector_corr['sector_correlations']
                if isinstance(sector_corr_matrix, dict):
                    for row_key, row_data in sector_corr_matrix.items():
                        if isinstance(row_data, dict):
                            for col_key, corr_value in row_data.items():
                                if row_key != col_key:  # Don't count self-correlations
                                    total_corr_count += 1
                                    if abs(corr_value) > 0.8:
                                        high_corr_count += 1
                                        
            # Calculate high correlation stress
            if total_corr_count > 0:
                stress_indicators['high_correlation_stress'] = high_corr_count / total_corr_count
                
            # Correlation concentration stress
            if 'correlation_health' in asset_corr:
                # Low health indicates high concentration
                health = asset_corr['correlation_health']
                stress_indicators['correlation_concentration'] = 1 - health
                
            # Regime stress
            if 'cross_group_correlation' in asset_corr:
                # High cross-group correlation indicates regime stress
                cross_corr = asset_corr['cross_group_correlation']
                stress_indicators['regime_stress'] = min(1.0, cross_corr * 2)  # Scale appropriately
                
            # Overall stress score
            stress_scores = [
                stress_indicators['high_correlation_stress'],
                stress_indicators['correlation_concentration'],
                stress_indicators['regime_stress']
            ]
            stress_indicators['overall_stress_score'] = np.mean(stress_scores)
            
            return stress_indicators
            
        except Exception as e:
            logger.error(f"Error calculating correlation stress indicators: {e}")
            return {'overall_stress_score': 0.0}
            
    def _analyze_market_structure_changes(self, asset_corr: Dict,
                                        sector_corr: Dict,
                                        geo_corr: Dict) -> Dict:
        """Analyze changes in market structure"""
        try:
            structure_analysis = {
                'structural_shifts': [],
                'market_integration': 0.0,
                'diversification_impact': 0.0
            }
            
            # Check for structural integration
            integration_indicators = []
            
            # Asset class integration
            if 'portfolio_diversification' in sector_corr:
                diversification = sector_corr['portfolio_diversification']
                integration_indicators.append(1 - diversification)  # High integration = low diversification
                
            # Geographic integration
            if 'geographic_diversification' in geo_corr:
                geo_diversification = geo_corr['geographic_diversification']
                integration_indicators.append(1 - geo_diversification)
                
            if integration_indicators:
                structure_analysis['market_integration'] = np.mean(integration_indicators)
                
            # Structural shift detection
            shifts = []
            
            # Detect correlation regime changes as structural shifts
            if 'regime_distribution' in asset_corr:
                regime_dist = asset_corr['regime_distribution']
                if len(regime_dist) > 1:
                    major_regimes = [regime for regime, ratio in regime_dist.items() if ratio > 0.4]
                    if len(major_regimes) > 1:
                        shifts.append('multi_regime_structure')
                        
            structure_analysis['structural_shifts'] = shifts
            
            # Calculate diversification impact
            diversifications = []
            if 'correlation_health' in asset_corr:
                diversifications.append(asset_corr['correlation_health'])
            if 'sector_diversification' in sector_corr:
                diversifications.append(sector_corr['sector_diversification'])
                
            if diversifications:
                structure_analysis['diversification_impact'] = np.mean(diversifications)
                
            return structure_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market structure changes: {e}")
            return {'structural_shifts': [], 'market_integration': 0.0, 'diversification_impact': 0.0}
            
    def _assess_correlation_stability(self, asset_corr: Dict,
                                    sector_corr: Dict,
                                    geo_corr: Dict) -> Dict:
        """Assess overall correlation stability"""
        try:
            stability_assessment = {
                'overall_stability': 0.0,
                'stability_factors': {},
                'instability_risks': []
            }
            
            stability_components = []
            
            # Asset level stability
            if 'correlation_health' in asset_corr:
                asset_health = asset_corr['correlation_health']
                stability_components.append(asset_health)
                stability_assessment['stability_factors']['asset_stability'] = asset_health
                
            # Sector level stability
            if 'correlation_changes' in sector_corr:
                changes = sector_corr['correlation_changes']
                if changes:
                    avg_change_magnitude = np.mean([
                        abs(change.get('change_magnitude', 0)) 
                        for change in changes.values()
                    ])
                    sector_stability = max(0, 1 - avg_change_magnitude * 10)  # Scale appropriately
                    stability_components.append(sector_stability)
                    stability_assessment['stability_factors']['sector_stability'] = sector_stability
                    
            # Geographic stability
            if 'cross_geographic_correlations' in geo_corr:
                cross_corrs = geo_corr['cross_geographic_correlations']
                if cross_corrs:
                    correlation_stabilities = []
                    for corr_data in cross_corrs.values():
                        corr_strength = corr_data.get('correlation_strength', 0)
                        # High correlations can indicate instability
                        geo_stability = max(0, 1 - (corr_strength - 0.5) * 2)
                        correlation_stabilities.append(geo_stability)
                        
                    geo_stability = np.mean(correlation_stabilities) if correlation_stabilities else 0.5
                    stability_components.append(geo_stability)
                    stability_assessment['stability_factors']['geographic_stability'] = geo_stability
                    
            # Calculate overall stability
            if stability_components:
                stability_assessment['overall_stability'] = np.mean(stability_components)
                
            # Identify instability risks
            risks = []
            if stability_assessment['overall_stability'] < 0.5:
                risks.append('low_overall_stability')
                
            for factor, stability in stability_assessment['stability_factors'].items():
                if stability < 0.3:
                    risks.append(f'low_{factor}')
                    
            stability_assessment['instability_risks'] = risks
            
            return stability_assessment
            
        except Exception as e:
            logger.error(f"Error assessing correlation stability: {e}")
            return {'overall_stability': 0.5, 'stability_factors': {}, 'instability_risks': []}
            
    async def get_correlation_pulse(self) -> Dict:
        """Get comprehensive correlation analysis"""
        try:
            # Run all correlation analyses in parallel
            correlation_tasks = [
                self._asset_correlation_analysis(),
                self._sector_correlation_analysis(),
                self._geographic_correlation_analysis(),
                self._cross_asset_correlation_analysis(),
                self._regime_correlation_analysis()
            ]
            
            results = await asyncio.gather(*correlation_tasks, return_exceptions=True)
            (
                asset_corr, sector_corr, geo_corr, 
                cross_asset_corr, regime_corr
            ) = results
            
            # Calculate overall Correlation Momentum Score (CMS)
            cms_components = []
            
            for result in results:
                if isinstance(result, dict) and not result.get('error'):
                    # Extract numeric scores from each analysis
                    score = self._extract_correlation_score(result)
                    if score is not None:
                        cms_components.append(score)
                        
            if cms_components:
                cms_score = np.mean(cms_components)
                cms_volatility = np.std(cms_components)
                
                # Classify correlation state
                if cms_score > 0.7:
                    correlation_state = 'highly_correlated'
                elif cms_score > 0.3:
                    correlation_state = 'moderately_correlated'
                elif cms_score < -0.3:
                    correlation_state = 'decorrelated'
                else:
                    correlation_state = 'normal_correlation'
                    
                return {
                    'correlation_momentum_score': cms_score,
                    'cms_volatility': cms_volatility,
                    'correlation_state': correlation_state,
                    'analysis_breakdown': {
                        'asset_correlations': asset_corr,
                        'sector_correlations': sector_corr,
                        'geographic_correlations': geo_corr,
                        'cross_asset_correlations': cross_asset_corr,
                        'regime_correlations': regime_corr
                    },
                    'timestamp': datetime.utcnow(),
                    'confidence': min(1.0, 1 - (cms_volatility / 2))  # Higher confidence when components agree
                }
            else:
                return {'error': 'Unable to calculate correlation momentum score'}
                
        except Exception as e:
            logger.error(f"Error getting correlation pulse: {e}")
            return {'error': str(e)}
            
    def _extract_correlation_score(self, analysis_result: Dict) -> Optional[float]:
        """Extract numeric correlation score from analysis result"""
        try:
            if 'correlation_health' in analysis_result:
                return analysis_result['correlation_health'] - 0.5  # Center around 0
            elif 'sector_diversification' in analysis_result:
                return analysis_result['sector_diversification'] - 0.5
            elif 'geographic_diversification' in analysis_result:
                return analysis_result['geographic_diversification'] - 0.5
            elif 'portfolio_diversification' in analysis_result:
                return analysis_result['portfolio_diversification'] - 0.5
            elif 'correlation_stability' in analysis_result:
                return analysis_result['correlation_stability'].get('overall_stability', 0) - 0.5
            else:
                return None
                
        except Exception:
            return None
            
    async def store_correlation_data(self, correlation_data: Dict):
        """Store correlation metrics in time-series database"""
        try:
            if self.db_manager and 'timestamp' in correlation_data:
                # Store Correlation Momentum Score
                await self.db_manager.influxdb_client.write_points(
                    database='market_pulse',
                    measurement='correlation_metrics',
                    tags={
                        'engine': 'correlation_pulse',
                        'state': correlation_data.get('correlation_state', 'unknown')
                    },
                    fields={
                        'cms_score': float(correlation_data.get('correlation_momentum_score', 0)),
                        'cms_volatility': float(correlation_data.get('cms_volatility', 0)),
                        'confidence': float(correlation_data.get('confidence', 0))
                    },
                    time=correlation_data['timestamp']
                )
                
                # Store component scores
                for component_name, analysis in correlation_data.get('analysis_breakdown', {}).items():
                    if isinstance(analysis, dict):
                        score = self._extract_correlation_score(analysis)
                        if score is not None:
                            await self.db_manager.influxdb_client.write_points(
                                database='market_pulse',
                                measurement='correlation_components',
                                tags={
                                    'component': component_name,
                                    'engine': 'correlation_pulse'
                                },
                                fields={'component_score': float(score)},
                                time=correlation_data['timestamp']
                            )
                            
            logger.debug("Correlation data stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing correlation data: {e}")
            
    async def get_status(self) -> Dict:
        """Get engine status and health metrics"""
        try:
            current_time = datetime.utcnow()
            time_since_update = (current_time - self.last_update).total_seconds() if self.last_update else None
            
            return {
                'name': self.name,
                'version': self.version,
                'status': self.status,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'time_since_update': time_since_update,
                'cache_size': len(self.correlation_cache),
                'models_loaded': len(self.models),
                'tracked_asset_groups': len(self.asset_groups),
                'tracked_sectors': len(self.sector_etfs),
                'tracked_geographies': len(self.geographic_indices),
                'health_score': self._calculate_engine_health()
            }
            
        except Exception as e:
            logger.error(f"Error getting correlation engine status: {e}")
            return {'name': self.name, 'status': 'error', 'error': str(e)}
            
    def _calculate_engine_health(self) -> float:
        """Calculate overall engine health score"""
        try:
            health_factors = []
            
            # Status health
            health_factors.append(1.0 if self.status == 'active' else 0.0)
            
            # Cache freshness
            if self.last_update:
                minutes_since_update = (datetime.utcnow() - self.last_update).total_seconds() / 60
                cache_freshness = max(0, 1 - (minutes_since_update / 30))  # Decay over 30 minutes
                health_factors.append(cache_freshness)
            
            # Model availability
            health_factors.append(min(1.0, len(self.models) / 3))  # Expect at least 3 models
            
            # Data source availability
            total_data_sources = (len(self.asset_groups) + len(self.sector_etfs) + 
                                len(self.geographic_indices))
            health_factors.append(min(1.0, total_data_sources / 50))  # Scale appropriately
            
            return np.mean(health_factors) if health_factors else 0.0
            
        except Exception:
            return 0.0