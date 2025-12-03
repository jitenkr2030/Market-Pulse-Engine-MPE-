"""
Liquidity Pulse Engine - Capital Flow Pressure Detection
Real-time tracking of market liquidity through ETF flows, volume distribution, and capital rotation patterns
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import yfinance as yf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import talib
from database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class LiquidityPulseEngine:
    """Liquidity Flow Meter - Measuring capital pressure and rotation patterns"""
    
    def __init__(self):
        self.name = "Liquidity Pulse Engine"
        self.version = "1.0.0"
        self.status = "initialized"
        self.last_update = None
        self.liquidity_cache = {}
        self.models = {}
        self.scaler = StandardScaler()
        
        # Liquidity tracking models
        self.liquidity_models = {
            "etf_flows": self._etf_flow_analysis,
            "volume_distribution": self._volume_distribution_analysis,
            "index_rebalancing": self._index_rebalancing_analysis,
            "sector_rotation": self._sector_rotation_analysis,
            "capital_movement": self._capital_movement_analysis
        }
        
        # Key ETFs for flow tracking
        self.tracked_etfs = {
            "broad_market": ["SPY", "VTI", "IVV", "VOO"],
            "sectors": ["XLK", "XLF", "XLE", "XLI", "XLV", "XLY"],
            "growth": ["QQQ", "IWM", "VUG"],
            "value": ["VTV", "IWD"],
            "international": ["EEM", "IEFA", "VEA", "VXUS"],
            "bonds": ["TLT", "IEF", "AGG", "BND"]
        }
        
        # Major indices for volume tracking
        self.tracked_indices = {
            "US": ["^GSPC", "^DJI", "^IXIC", "^RUT"],
            "International": ["^FTSE", "^GDAXI", "^N225", "^HSI"],
            "Emerging": ["^IPSA", "^BVSP", "^MXX"]
        }
        
        # Liquidity metrics thresholds
        self.liquidity_thresholds = {
            "high_flow": 1.5,      # ETF flows 50% above average
            "rotation_intensity": 2.0,  # 2+ standard deviations
            "volume_concentration": 0.7  # 70% volume in top assets
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
            logger.info("Liquidity Pulse Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Liquidity Pulse Engine: {e}")
            self.status = "error"
            
    def _initialize_models(self):
        """Initialize ML models for liquidity prediction"""
        try:
            # Random Forest for liquidity momentum prediction
            self.models['liquidity_momentum'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Scaler for feature normalization
            self.scaler = StandardScaler()
            
            logger.info("Liquidity prediction models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize liquidity models: {e}")
            
    async def _fetch_etf_data(self, etf_symbol: str, days: int = 30) -> Dict:
        """Fetch ETF data including flows and price data"""
        try:
            ticker = yf.Ticker(etf_symbol)
            
            # Get historical data
            hist = ticker.history(period=f"{days}d")
            if hist.empty:
                return {}
                
            # Calculate daily flows (approximate from volume * price change)
            hist['flow_estimate'] = hist['Volume'] * hist['Close'].pct_change()
            
            # Get fund info
            info = ticker.info
            
            # Calculate liquidity metrics
            metrics = {
                'symbol': etf_symbol,
                'net_flow': hist['flow_estimate'].sum(),
                'avg_daily_volume': hist['Volume'].mean(),
                'volume_trend': self._calculate_volume_trend(hist['Volume']),
                'price_momentum': hist['Close'].pct_change(5).iloc[-1] if len(hist) > 5 else 0,
                'assets_under_management': info.get('totalAssets', 0),
                'expense_ratio': info.get('annualReportExpenseRatio', 0),
                'turnover': info.get('annualHoldingsTurnover', 0),
                'timestamp': datetime.utcnow()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching ETF data for {etf_symbol}: {e}")
            return {}
            
    def _calculate_volume_trend(self, volume_series: pd.Series) -> float:
        """Calculate volume trend momentum"""
        try:
            # Calculate volume momentum over different periods
            v_short = volume_series.tail(5).mean()
            v_long = volume_series.tail(20).mean()
            
            if v_long > 0:
                return (v_short / v_long) - 1
            return 0
        except:
            return 0
            
    async def _etf_flow_analysis(self) -> Dict:
        """Analyze ETF flows across different categories"""
        try:
            flow_data = {}
            category_scores = {}
            
            for category, etfs in self.tracked_etfs.items():
                category_flows = []
                
                for etf in etfs:
                    data = await self._fetch_etf_data(etf)
                    if data:
                        category_flows.append(data)
                        
                if category_flows:
                    # Aggregate category metrics
                    avg_flow = np.mean([f['net_flow'] for f in category_flows])
                    avg_volume = np.mean([f['avg_daily_volume'] for f in category_flows])
                    momentum = np.mean([f['volume_trend'] for f in category_flows])
                    
                    # Normalize flow by volume for comparison
                    normalized_flow = avg_flow / avg_volume if avg_volume > 0 else 0
                    
                    category_scores[category] = {
                        'normalized_flow': normalized_flow,
                        'momentum': momentum,
                        'volume_trend': momentum,
                        'active_etfs': len(category_flows)
                    }
                    
            # Calculate overall liquidity score
            if category_scores:
                weights = {
                    'broad_market': 0.3,
                    'sectors': 0.25,
                    'growth': 0.15,
                    'value': 0.1,
                    'international': 0.1,
                    'bonds': 0.1
                }
                
                liquidity_score = 0
                total_weight = 0
                
                for category, score in category_scores.items():
                    weight = weights.get(category, 0.1)
                    # Combine flow and momentum components
                    category_liquidity = (score['normalized_flow'] + score['momentum']) / 2
                    liquidity_score += category_liquidity * weight
                    total_weight += weight
                    
                if total_weight > 0:
                    liquidity_score /= total_weight
                    
                return {
                    'etf_liquidity_score': liquidity_score,
                    'category_breakdown': category_scores,
                    'overall_flow_intensity': abs(liquidity_score),
                    'flow_direction': 'inflow' if liquidity_score > 0 else 'outflow'
                }
            
            return {'error': 'No ETF data available'}
            
        except Exception as e:
            logger.error(f"Error in ETF flow analysis: {e}")
            return {'error': str(e)}
            
    async def _volume_distribution_analysis(self) -> Dict:
        """Analyze volume distribution across markets and sectors"""
        try:
            distribution_data = {}
            
            # Track volume across major indices
            for region, indices in self.tracked_indices.items():
                region_volumes = []
                
                for index in indices:
                    try:
                        ticker = yf.Ticker(index)
                        hist = ticker.history(period="5d")
                        if not hist.empty:
                            avg_volume = hist['Volume'].mean()
                            volume_volatility = hist['Volume'].std()
                            
                            region_volumes.append({
                                'index': index,
                                'avg_volume': avg_volume,
                                'volume_volatility': volume_volatility,
                                'volume_share': 0  # Will calculate after collecting all
                            })
                    except Exception as e:
                        logger.warning(f"Error fetching data for {index}: {e}")
                        continue
                        
                if region_volumes:
                    total_volume = sum([v['avg_volume'] for v in region_volumes])
                    
                    # Calculate volume shares
                    for volume_data in region_volumes:
                        volume_data['volume_share'] = (
                            volume_data['avg_volume'] / total_volume 
                            if total_volume > 0 else 0
                        )
                    
                    # Calculate concentration metrics
                    shares = [v['volume_share'] for v in region_volumes]
                    concentration = max(shares) if shares else 0
                    gini_coefficient = self._calculate_gini_coefficient(shares)
                    
                    distribution_data[region] = {
                        'indices': region_volumes,
                        'total_volume': total_volume,
                        'concentration': concentration,
                        'gini_coefficient': gini_coefficient,
                        'market_efficiency': 1 - concentration  # Higher when volume is spread out
                    }
            
            # Calculate global liquidity metrics
            all_volumes = []
            for region_data in distribution_data.values():
                all_volumes.extend([idx['avg_volume'] for idx in region_data['indices']])
                
            if all_volumes:
                global_volume_trend = np.mean(all_volumes)
                volume_correlation = self._calculate_volume_correlation(distribution_data)
                
                return {
                    'volume_distribution': distribution_data,
                    'global_liquidity': {
                        'total_volume': sum(all_volumes),
                        'volume_trend': global_volume_trend,
                        'cross_market_correlation': volume_correlation,
                        'liquidity_health': self._assess_liquidity_health(distribution_data)
                    }
                }
            
            return {'error': 'No volume distribution data available'}
            
        except Exception as e:
            logger.error(f"Error in volume distribution analysis: {e}")
            return {'error': str(e)}
            
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for volume distribution inequality"""
        try:
            if not values or len(values) < 2:
                return 0
                
            # Sort values
            sorted_values = sorted(values)
            n = len(sorted_values)
            
            # Calculate Gini coefficient
            cumulative_sum = sum((i + 1) * value for i, value in enumerate(sorted_values))
            total_sum = sum(sorted_values)
            
            if total_sum == 0:
                return 0
                
            gini = (2 * cumulative_sum) / (n * total_sum) - (n + 1) / n
            return max(0, min(1, gini))  # Clamp between 0 and 1
            
        except Exception:
            return 0
            
    def _calculate_volume_correlation(self, distribution_data: Dict) -> float:
        """Calculate cross-market volume correlation"""
        try:
            volume_series = {}
            
            # Collect volume time series for each region
            for region, data in distribution_data.items():
                if 'indices' in data:
                    total_volume = sum([idx['avg_volume'] for idx in data['indices']])
                    volume_series[region] = total_volume
                    
            if len(volume_series) < 2:
                return 0
                
            # Calculate correlation coefficient
            volumes = list(volume_series.values())
            correlation = np.corrcoef(volumes)[0, 1] if len(volumes) > 1 else 0
            
            return correlation
            
        except Exception:
            return 0
            
    def _assess_liquidity_health(self, distribution_data: Dict) -> float:
        """Assess overall liquidity health based on distribution"""
        try:
            health_score = 0
            
            for region, data in distribution_data.items():
                # Health is higher when volume is well-distributed
                distribution_health = 1 - data.get('gini_coefficient', 0.5)
                
                # Health is higher when markets move together (systemic health)
                market_stability = data.get('concentration', 0.5)
                stability_health = 1 - abs(market_stability - 0.5) * 2  # Peak health at 50% concentration
                
                region_health = (distribution_health + stability_health) / 2
                health_score += region_health
                
            return health_score / len(distribution_data) if distribution_data else 0
            
        except Exception:
            return 0
            
    async def _index_rebalancing_analysis(self) -> Dict:
        """Analyze index rebalancing flows and impacts"""
        try:
            # Track major index components and their weight changes
            rebalancing_signals = {}
            
            # Major indices to monitor
            major_indices = {
                'S&P 500': '^GSPC',
                'Dow Jones': '^DJI',
                'NASDAQ': '^IXIC',
                'Russell 2000': '^RUT'
            }
            
            for name, symbol in major_indices.items():
                try:
                    # Get index components (approximation using major stocks)
                    # In production, this would use a proper index constituents API
                    component_stocks = self._get_index_components(symbol)
                    
                    rebalancing_flows = []
                    for stock in component_stocks[:50]:  # Top 50 components
                        data = await self._fetch_etf_data(f"{stock}", days=3)  # Recent 3 days
                        if data and abs(data.get('net_flow', 0)) > data.get('avg_daily_volume', 0) * 0.1:
                            rebalancing_flows.append(data)
                    
                    # Analyze rebalancing intensity
                    if rebalancing_flows:
                        total_rebalancing = sum([f['net_flow'] for f in rebalancing_flows])
                        avg_flow = total_rebalancing / len(rebalancing_flows)
                        
                        rebalancing_signals[name] = {
                            'rebalancing_intensity': abs(avg_flow),
                            'flow_direction': 'inflow' if avg_flow > 0 else 'outflow',
                            'active_components': len(rebalancing_flows),
                            'rebalancing_probability': self._calculate_rebalancing_probability(rebalancing_flows)
                        }
                        
                except Exception as e:
                    logger.warning(f"Error analyzing rebalancing for {name}: {e}")
                    continue
                    
            if rebalancing_signals:
                return {
                    'rebalancing_analysis': rebalancing_signals,
                    'overall_rebalancing_pressure': np.mean([
                        signal['rebalancing_intensity'] 
                        for signal in rebalancing_signals.values()
                    ])
                }
            
            return {'message': 'No significant rebalancing detected'}
            
        except Exception as e:
            logger.error(f"Error in rebalancing analysis: {e}")
            return {'error': str(e)}
            
    def _get_index_components(self, index_symbol: str) -> List[str]:
        """Get index components (simplified approach)"""
        # This is a simplified version - in production, use a proper index API
        component_mapping = {
            '^GSPC': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
            '^DJI': ['AAPL', 'MSFT', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA'],
            '^IXIC': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
            '^RUT': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        }
        return component_mapping.get(index_symbol, ['AAPL', 'MSFT', 'GOOGL'])
        
    def _calculate_rebalancing_probability(self, flows: List[Dict]) -> float:
        """Calculate probability of significant rebalancing"""
        try:
            if not flows:
                return 0
                
            # Look for coordinated flows
            flow_directions = [1 if f['net_flow'] > 0 else -1 for f in flows]
            
            # High probability when flows are coordinated (all same direction)
            coordination_score = abs(sum(flow_directions)) / len(flow_directions)
            
            # High probability when flow magnitudes are similar
            magnitudes = [abs(f['net_flow']) for f in flows]
            magnitude_variance = np.var(magnitudes) if len(magnitudes) > 1 else 0
            
            # Normalize variance (lower variance = higher coordination)
            coordination_factor = 1 / (1 + magnitude_variance)
            
            # Combined probability
            probability = (coordination_score + coordination_factor) / 2
            
            return min(1.0, probability)
            
        except Exception:
            return 0
            
    async def _sector_rotation_analysis(self) -> Dict:
        """Analyze sector rotation patterns and capital flows"""
        try:
            sector_data = {}
            
            # ETF sector mapping
            sectors = {
                'Technology': 'XLK',
                'Financial': 'XLF', 
                'Energy': 'XLE',
                'Industrial': 'XLI',
                'Healthcare': 'XLV',
                'Consumer': 'XLY',
                'Utilities': 'XLU',
                'Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Communication': 'XLC'
            }
            
            sector_performance = {}
            
            for sector_name, etf_symbol in sectors.items():
                try:
                    # Get sector performance data
                    data = await self._fetch_etf_data(etf_symbol, days=20)  # 20 days for rotation analysis
                    
                    if data:
                        # Calculate rotation signals
                        momentum_5d = data.get('price_momentum', 0)
                        volume_trend = data.get('volume_trend', 0)
                        
                        # Rotation strength combines price momentum and volume
                        rotation_strength = abs(momentum_5d) + abs(volume_trend)
                        
                        sector_performance[sector_name] = {
                            'momentum': momentum_5d,
                            'volume_trend': volume_trend,
                            'rotation_strength': rotation_strength,
                            'net_flow': data.get('net_flow', 0),
                            'relative_performance': 0  # Will calculate later
                        }
                        
                except Exception as e:
                    logger.warning(f"Error analyzing sector {sector_name}: {e}")
                    continue
                    
            if not sector_performance:
                return {'error': 'No sector data available'}
                
            # Calculate relative performance scores
            momentum_scores = [data['momentum'] for data in sector_performance.values()]
            avg_momentum = np.mean(momentum_scores)
            
            for sector_name in sector_performance:
                sector_performance[sector_name]['relative_performance'] = (
                    sector_performance[sector_name]['momentum'] - avg_momentum
                )
                
            # Identify rotation patterns
            rotation_patterns = self._identify_rotation_patterns(sector_performance)
            
            # Calculate sector flow scores
            sector_flow_scores = {}
            for sector_name, data in sector_performance.items():
                # Flow score combines momentum, volume trend, and relative performance
                flow_score = (
                    data['momentum'] * 0.4 +
                    data['volume_trend'] * 0.3 +
                    data['relative_performance'] * 0.3
                )
                sector_flow_scores[sector_name] = flow_score
                
            return {
                'sector_performance': sector_performance,
                'rotation_patterns': rotation_patterns,
                'sector_flow_scores': sector_flow_scores,
                'rotation_intensity': np.std(list(sector_flow_scores.values())),
                'rotation_direction': 'outperforming' if np.mean(list(sector_flow_scores.values())) > 0 else 'underperforming'
            }
            
        except Exception as e:
            logger.error(f"Error in sector rotation analysis: {e}")
            return {'error': str(e)}
            
    def _identify_rotation_patterns(self, sector_data: Dict) -> Dict:
        """Identify specific rotation patterns in sector data"""
        try:
            patterns = {}
            
            # Sort sectors by performance
            sorted_sectors = sorted(
                sector_data.items(),
                key=lambda x: x[1]['relative_performance'],
                reverse=True
            )
            
            # Identify leadership changes
            top_performers = [sector[0] for sector in sorted_sectors[:3]]
            bottom_performers = [sector[0] for sector in sorted_sectors[-3:]]
            
            patterns['current_leaders'] = top_performers
            patterns['current_laggards'] = bottom_performers
            
            # Rotation momentum
            rotation_momentum = {}
            for sector, data in sector_data.items():
                momentum = data['momentum']
                if momentum > 0.02:  # 2%+ momentum
                    rotation_momentum[sector] = 'accelerating'
                elif momentum < -0.02:
                    rotation_momentum[sector] = 'decelerating'
                else:
                    rotation_momentum[sector] = 'stable'
                    
            patterns['rotation_momentum'] = rotation_momentum
            
            # Sector momentum divergence
            positive_momentum = sum(1 for data in sector_data.values() if data['momentum'] > 0)
            negative_momentum = len(sector_data) - positive_momentum
            
            patterns['market_breadth'] = {
                'advancing_sectors': positive_momentum,
                'declining_sectors': negative_momentum,
                'breadth_ratio': positive_momentum / len(sector_data) if sector_data else 0
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying rotation patterns: {e}")
            return {}
            
    async def _capital_movement_analysis(self) -> Dict:
        """Analyze overall capital movement patterns"""
        try:
            # Combine all liquidity metrics for global capital flow analysis
            etf_flows = await self._etf_flow_analysis()
            volume_dist = await self._volume_distribution_analysis()
            sector_rot = await self._sector_rotation_analysis()
            rebalancing = await self._index_rebalancing_analysis()
            
            # Calculate comprehensive capital movement score
            capital_components = []
            
            # ETF flow component
            if 'etf_liquidity_score' in etf_flows:
                capital_components.append(etf_flows['etf_liquidity_score'])
                
            # Volume distribution component
            if 'volume_distribution' in volume_dist:
                # Higher score for better distributed volume
                global_health = volume_dist['global_liquidity']['liquidity_health']
                capital_components.append(global_health - 0.5)  # Center around 0
                
            # Sector rotation component
            if 'rotation_intensity' in sector_rot:
                # Normalize rotation intensity
                rotation_norm = max(-1, min(1, sector_rot['rotation_intensity']))
                capital_components.append(rotation_norm)
                
            # Rebalancing component
            if 'overall_rebalancing_pressure' in rebalancing:
                rebalancing_norm = max(-1, min(1, rebalancing['overall_rebalancing_pressure']))
                capital_components.append(rebalancing_norm)
                
            # Calculate overall capital movement score
            if capital_components:
                capital_movement_score = np.mean(capital_components)
                
                # Movement direction and strength
                movement_strength = abs(capital_movement_score)
                movement_direction = 'inflow' if capital_movement_score > 0 else 'outflow'
                
                # Market liquidity phase classification
                if movement_strength < 0.2:
                    phase = 'balanced'
                elif capital_movement_score > 0:
                    phase = 'liquidity_injection'
                else:
                    phase = 'liquidity_drain'
                    
                return {
                    'capital_movement_score': capital_movement_score,
                    'movement_strength': movement_strength,
                    'movement_direction': movement_direction,
                    'market_phase': phase,
                    'component_scores': {
                        'etf_flows': etf_flows.get('etf_liquidity_score', 0),
                        'volume_distribution': volume_dist.get('global_liquidity', {}).get('liquidity_health', 0.5) - 0.5,
                        'sector_rotation': sector_rot.get('rotation_intensity', 0),
                        'rebalancing': rebalancing.get('overall_rebalancing_pressure', 0)
                    },
                    'capital_flow_consensus': self._calculate_flow_consensus(capital_components)
                }
            
            return {'error': 'Insufficient data for capital movement analysis'}
            
        except Exception as e:
            logger.error(f"Error in capital movement analysis: {e}")
            return {'error': str(e)}
            
    def _calculate_flow_consensus(self, components: List[float]) -> float:
        """Calculate consensus score across all capital flow components"""
        try:
            if len(components) < 2:
                return 0.5
                
            # Calculate agreement among components
            positive_components = sum(1 for c in components if c > 0)
            negative_components = len(components) - positive_components
            
            # Consensus is high when components agree
            consensus = max(positive_components, negative_components) / len(components)
            
            # Direction consensus
            direction_consensus = 1 if abs(positive_components - negative_components) / len(components) > 0.5 else 0.5
            
            return (consensus + direction_consensus) / 2
            
        except Exception:
            return 0.5
            
    async def get_liquidity_pulse(self) -> Dict:
        """Get comprehensive liquidity pulse analysis"""
        try:
            # Run all liquidity analyses in parallel
            liquidity_tasks = [
                self._etf_flow_analysis(),
                self._volume_distribution_analysis(),
                self._sector_rotation_analysis(),
                self._index_rebalancing_analysis(),
                self._capital_movement_analysis()
            ]
            
            results = await asyncio.gather(*liquidity_tasks, return_exceptions=True)
            (
                etf_flows, volume_dist, sector_rot, 
                rebalancing, capital_movement
            ) = results
            
            # Calculate Liquidity Momentum Score (LMS)
            lms_components = []
            
            for result in results:
                if isinstance(result, dict) and not result.get('error'):
                    # Extract numeric scores from each analysis
                    score = self._extract_liquidity_score(result)
                    if score is not None:
                        lms_components.append(score)
                        
            if lms_components:
                lms_score = np.mean(lms_components)
                lms_strength = np.std(lms_components)
                
                # Classify liquidity state
                if lms_score > 0.3:
                    liquidity_state = 'expanding'
                elif lms_score < -0.3:
                    liquidity_state = 'contracting'
                else:
                    liquidity_state = 'neutral'
                    
                return {
                    'liquidity_momentum_score': lms_score,
                    'lms_strength': lms_strength,
                    'liquidity_state': liquidity_state,
                    'analysis_breakdown': {
                        'etf_flows': etf_flows,
                        'volume_distribution': volume_dist,
                        'sector_rotation': sector_rot,
                        'index_rebalancing': rebalancing,
                        'capital_movement': capital_movement
                    },
                    'timestamp': datetime.utcnow(),
                    'confidence': min(1.0, 1 - (lms_strength / 2))  # Higher confidence when components agree
                }
            else:
                return {'error': 'Unable to calculate liquidity momentum score'}
                
        except Exception as e:
            logger.error(f"Error getting liquidity pulse: {e}")
            return {'error': str(e)}
            
    def _extract_liquidity_score(self, analysis_result: Dict) -> Optional[float]:
        """Extract numeric liquidity score from analysis result"""
        try:
            if 'etf_liquidity_score' in analysis_result:
                return analysis_result['etf_liquidity_score']
            elif 'global_liquidity' in analysis_result:
                return analysis_result['global_liquidity']['liquidity_health'] - 0.5
            elif 'rotation_intensity' in analysis_result:
                return max(-1, min(1, analysis_result['rotation_intensity']))
            elif 'overall_rebalancing_pressure' in analysis_result:
                return max(-1, min(1, analysis_result['overall_rebalancing_pressure']))
            elif 'capital_movement_score' in analysis_result:
                return analysis_result['capital_movement_score']
            else:
                return None
                
        except Exception:
            return None
            
    async def store_liquidity_data(self, liquidity_data: Dict):
        """Store liquidity metrics in time-series database"""
        try:
            if self.db_manager and 'timestamp' in liquidity_data:
                # Store Liquidity Momentum Score
                await self.db_manager.influxdb_client.write_points(
                    database='market_pulse',
                    measurement='liquidity_metrics',
                    tags={
                        'engine': 'liquidity_pulse',
                        'state': liquidity_data.get('liquidity_state', 'unknown')
                    },
                    fields={
                        'lms_score': float(liquidity_data.get('liquidity_momentum_score', 0)),
                        'lms_strength': float(liquidity_data.get('lms_strength', 0)),
                        'confidence': float(liquidity_data.get('confidence', 0))
                    },
                    time=liquidity_data['timestamp']
                )
                
                # Store component scores
                for component_name, analysis in liquidity_data.get('analysis_breakdown', {}).items():
                    if isinstance(analysis, dict):
                        score = self._extract_liquidity_score(analysis)
                        if score is not None:
                            await self.db_manager.influxdb_client.write_points(
                                database='market_pulse',
                                measurement='liquidity_components',
                                tags={
                                    'component': component_name,
                                    'engine': 'liquidity_pulse'
                                },
                                fields={'component_score': float(score)},
                                time=liquidity_data['timestamp']
                            )
                            
            logger.debug("Liquidity data stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing liquidity data: {e}")
            
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
                'cache_size': len(self.liquidity_cache),
                'models_loaded': len(self.models),
                'tracked_etfs': sum(len(etfs) for etfs in self.tracked_etfs.values()),
                'tracked_indices': sum(len(indices) for indices in self.tracked_indices.values()),
                'health_score': self._calculate_engine_health()
            }
            
        except Exception as e:
            logger.error(f"Error getting liquidity engine status: {e}")
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
            
            return np.mean(health_factors) if health_factors else 0.0
            
        except Exception:
            return 0.0