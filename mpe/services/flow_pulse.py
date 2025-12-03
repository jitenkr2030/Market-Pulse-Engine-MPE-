"""
Flow Pulse Engine - Institutional Money Flow Detection
Real-time tracking of institutional money flows, hedge fund positioning, and capital allocation patterns
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import yfinance as yf
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import talib
from database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class FlowPulseEngine:
    """Money Flow Monitor - Detecting institutional capital movements"""
    
    def __init__(self):
        self.name = "Flow Pulse Engine"
        self.version = "1.0.0"
        self.status = "initialized"
        self.last_update = None
        self.flow_cache = {}
        self.models = {}
        self.scaler = StandardScaler()
        
        # Flow tracking models
        self.flow_models = {
            "institutional_flows": self._institutional_flow_analysis,
            "smart_money_flows": self._smart_money_flow_analysis,
            "insider_trading": self._insider_trading_analysis,
            "hedge_fund_flows": self._hedge_fund_flow_analysis,
            "mutual_fund_flows": self._mutual_fund_flow_analysis
        }
        
        # Institutional tracking categories
        self.institutional_categories = {
            "hedge_funds": {
                "etfs": ["SPY", "QQQ", "IWM", "VTI"],
                "characteristics": "High frequency, contrarian, concentrated"
            },
            "mutual_funds": {
                "etfs": ["VTI", "VOO", "IVV", "VTV"],
                "characteristics": "Lower frequency, trend following, diversified"
            },
            "pension_funds": {
                "etfs": ["VTI", "AGG", "VEA", "VWO"],
                "characteristics": "Very long-term, strategic, large positions"
            },
            "sovereign_wealth": {
                "etfs": ["SPY", "VEA", "EEM", "VWO"],
                "characteristics": "Strategic, large positions, long-term"
            },
            "retail_flows": {
                "etfs": ["QQQ", "TQQQ", "SQQQ", "ARKK"],
                "characteristics": "Momentum driven, speculative, high beta"
            }
        }
        
        # Smart money indicators
        self.smart_money_indicators = {
            "dark_pool_activity": "Dark pool volume ratios",
            "block_trades": "Large institutional trades",
            "options_sweeps": "Options market manipulation",
            "earnings_revisions": "Analyst estimate changes",
            "sec_filings": "13F institutional holdings",
            "insider_trading": "Corporate insider transactions"
        }
        
        # Flow pattern recognition
        self.flow_patterns = {
            "accumulation": "Smart money buying while retail selling",
            "distribution": "Smart money selling while retail buying",
            "rotation": "Flow shifting between sectors",
            "momentum": "Follow-through flows in trending markets",
            "contrarian": "Flows opposite to price movements"
        }
        
        # Market microstructure data
        self.microstructure_data = {
            "volume_profile": "Price-volume relationships",
            "time_of_day_flows": "Intraday flow patterns",
            "spread_changes": "Bid-ask spread dynamics",
            "order_book_imbalance": "Market depth analysis"
        }
        
        # Flow thresholds
        self.flow_thresholds = {
            "institutional_surge": 2.0,     # 2x normal volume
            "dark_pool_spike": 1.5,         # 50% above average
            "insider_buying": 0.7,          # Buy/sell ratio threshold
            "hedge_fund_concentration": 0.3, # Position concentration
            "retail_excitement": 3.0        # Retail-specific activity
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
            logger.info("Flow Pulse Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Flow Pulse Engine: {e}")
            self.status = "error"
            
    def _initialize_models(self):
        """Initialize ML models for flow prediction"""
        try:
            # Random Forest for flow momentum prediction
            self.models['flow_momentum'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Classification model for flow types
            self.models['flow_classifier'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=8
            )
            
            # Scaler for feature normalization
            self.scaler = StandardScaler()
            
            logger.info("Flow prediction models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize flow models: {e}")
            
    async def _fetch_market_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch comprehensive market data for flow analysis"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist = ticker.history(period=period)
            if hist.empty:
                return pd.DataFrame()
                
            # Calculate flow metrics
            hist['volume_ma_5'] = hist['Volume'].rolling(5).mean()
            hist['volume_ma_20'] = hist['Volume'].rolling(20).mean()
            hist['price_change'] = hist['Close'].pct_change()
            hist['volume_ratio'] = hist['Volume'] / hist['volume_ma_20']
            
            # Volume-weighted metrics
            hist['vwap'] = (hist['Close'] * hist['Volume']).cumsum() / hist['Volume'].cumsum()
            hist['price_vwap_ratio'] = hist['Close'] / hist['vwap']
            
            # Flow momentum indicators
            hist['flow_momentum'] = hist['volume_ratio'] * hist['price_change']
            
            # Volume pattern analysis
            hist['volume_percentile'] = hist['Volume'].rolling(60).rank(pct=True)
            
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()
            
    async def _institutional_flow_analysis(self) -> Dict:
        """Analyze institutional money flows across different categories"""
        try:
            institutional_flows = {}
            
            for category, config in self.institutional_categories.items():
                category_flows = []
                
                for etf in config["etfs"]:
                    try:
                        data = await self._fetch_market_data(etf)
                        if not data.empty:
                            # Calculate institutional flow indicators
                            flow_metrics = self._calculate_institutional_flow_metrics(data)
                            flow_metrics['etf'] = etf
                            category_flows.append(flow_metrics)
                    except Exception as e:
                        logger.warning(f"Error analyzing flow for {etf}: {e}")
                        continue
                        
                if category_flows:
                    # Aggregate category metrics
                    avg_volume_ratio = np.mean([f['volume_ratio'] for f in category_flows])
                    avg_flow_momentum = np.mean([f['flow_momentum'] for f in category_flows])
                    institutional_strength = np.mean([f['institutional_signature'] for f in category_flows])
                    
                    # Flow direction analysis
                    positive_flows = sum(1 for f in category_flows if f['flow_momentum'] > 0)
                    flow_direction = "inflow" if positive_flows > len(category_flows) / 2 else "outflow"
                    
                    institutional_flows[category] = {
                        'avg_volume_ratio': float(avg_volume_ratio),
                        'avg_flow_momentum': float(avg_flow_momentum),
                        'institutional_strength': float(institutional_strength),
                        'flow_direction': flow_direction,
                        'flow_consensus': positive_flows / len(category_flows),
                        'active_securities': len(category_flows),
                        'category_characteristics': config["characteristics"]
                    }
                    
            if not institutional_flows:
                return {'error': 'No institutional flow data available'}
                
            # Calculate overall institutional flow score
            flow_scores = []
            for category_data in institutional_flows.values():
                # Combine volume ratio and flow momentum
                score = (category_data['avg_volume_ratio'] - 1) * 0.5 + category_data['avg_flow_momentum'] * 0.5
                flow_scores.append(score)
                
            overall_institutional_score = np.mean(flow_scores) if flow_scores else 0
            
            # Flow regime analysis
            flow_regime = self._determine_flow_regime(institutional_flows)
            
            return {
                'institutional_flows': institutional_flows,
                'overall_institutional_score': overall_institutional_score,
                'flow_regime': flow_regime,
                'flow_intensity': np.std(flow_scores) if len(flow_scores) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in institutional flow analysis: {e}")
            return {'error': str(e)}
            
    def _calculate_institutional_flow_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate institutional flow metrics from market data"""
        try:
            if data.empty:
                return {}
                
            # Get latest metrics
            latest = data.iloc[-1]
            recent_data = data.tail(10)  # Last 10 periods
            historical_data = data.head(-20).tail(20)  # 20 periods before recent
            
            # Volume ratio (institutional vs retail)
            volume_ratio = latest['volume_ratio']
            
            # Flow momentum
            flow_momentum = latest['flow_momentum']
            
            # Volume price analysis
            price_volume_trend = latest['price_vwap_ratio']
            
            # Institutional signature (higher for institutional flows)
            # Institutions tend to have consistent volume patterns
            volume_consistency = 1 - recent_data['volume_ratio'].std()
            institutional_signature = min(1.0, volume_consistency * volume_ratio)
            
            # Flow persistence (sustained flows vs one-time events)
            recent_flows = recent_data['flow_momentum'].values
            flow_persistence = np.mean(np.sign(recent_flows)) * np.mean(np.abs(recent_flows))
            
            # Dark pool activity proxy (higher volume with minimal price impact)
            if len(recent_data) > 1:
                price_impact = np.std(recent_data['price_change'])
                volume_intensity = np.mean(recent_data['volume_ratio'])
                dark_pool_proxy = volume_intensity / (1 + price_impact * 10)  # Lower price impact = higher dark pool activity
            else:
                dark_pool_proxy = 0
                
            return {
                'volume_ratio': volume_ratio,
                'flow_momentum': flow_momentum,
                'price_volume_trend': price_volume_trend,
                'institutional_signature': institutional_signature,
                'flow_persistence': flow_persistence,
                'dark_pool_proxy': dark_pool_proxy,
                'timestamp': data.index[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating institutional flow metrics: {e}")
            return {}
            
    def _determine_flow_regime(self, institutional_flows: Dict) -> str:
        """Determine the current flow regime"""
        try:
            if not institutional_flows:
                return 'unknown'
                
            # Analyze flow patterns
            inflow_categories = [cat for cat, data in institutional_flows.items() 
                               if data['flow_direction'] == 'inflow']
            outflow_categories = [cat for cat, data in institutional_flows.items() 
                                if data['flow_direction'] == 'outflow']
            
            # Regime classification
            if len(inflow_categories) >= len(outflow_categories) * 2:
                return 'institutional_inflow_regime'
            elif len(outflow_categories) >= len(inflow_categories) * 2:
                return 'institutional_outflow_regime'
            elif len(inflow_categories) == len(outflow_categories):
                return 'flow_rotation_regime'
            else:
                return 'mixed_flow_regime'
                
        except Exception as e:
            logger.error(f"Error determining flow regime: {e}")
            return 'unknown'
            
    async def _smart_money_flow_analysis(self) -> Dict:
        """Analyze smart money flows and institutional positioning"""
        try:
            # Key ETFs that smart money typically trades
            smart_money_etfs = ["SPY", "QQQ", "IWM", "VTI", "TLT", "GLD"]
            
            smart_money_data = {}
            
            for etf in smart_money_etfs:
                try:
                    data = await self._fetch_market_data(etf)
                    if not data.empty:
                        # Analyze smart money characteristics
                        smart_indicators = self._analyze_smart_money_indicators(data)
                        smart_indicators['etf'] = etf
                        smart_money_data[etf] = smart_indicators
                except Exception as e:
                    logger.warning(f"Error analyzing smart money flow for {etf}: {e}")
                    continue
                    
            if not smart_money_data:
                return {'error': 'No smart money flow data available'}
                
            # Calculate smart money consensus
            smart_consensus = self._calculate_smart_money_consensus(smart_money_data)
            
            # Detect smart money contrarian signals
            contrarian_signals = self._detect_contrarian_signals(smart_money_data)
            
            # Smart money vs retail flow divergence
            flow_divergence = self._analyze_flow_divergence(smart_money_data)
            
            # Market timing signals
            timing_signals = self._calculate_timing_signals(smart_money_data)
            
            return {
                'smart_money_data': smart_money_data,
                'smart_consensus': smart_consensus,
                'contrarian_signals': contrarian_signals,
                'flow_divergence': flow_divergence,
                'timing_signals': timing_signals
            }
            
        except Exception as e:
            logger.error(f"Error in smart money flow analysis: {e}")
            return {'error': str(e)}
            
    def _analyze_smart_money_indicators(self, data: pd.DataFrame) -> Dict:
        """Analyze smart money indicators in market data"""
        try:
            if data.empty:
                return {}
                
            latest = data.iloc[-1]
            recent_data = data.tail(5)
            
            # Smart money characteristics
            indicators = {}
            
            # 1. Volume consistency (smart money has consistent flows)
            volume_consistency = 1 - recent_data['volume_ratio'].std()
            indicators['volume_consistency'] = float(volume_consistency)
            
            # 2. Price impact minimization (smart money minimizes market impact)
            if len(recent_data) > 1:
                avg_price_change = np.mean(np.abs(recent_data['price_change']))
                avg_volume_ratio = np.mean(recent_data['volume_ratio'])
                impact_efficiency = avg_volume_ratio / (1 + avg_price_change * 20)
                indicators['impact_efficiency'] = float(impact_efficiency)
            else:
                indicators['impact_efficiency'] = 0
                
            # 3. Timing precision (smart money enters at optimal times)
            # Look for entries before price movements
            price_momentum = recent_data['price_change'].iloc[-1]
            volume_surge = latest['volume_ratio'] - 1
            timing_precision = min(1.0, abs(volume_surge) * abs(price_momentum) * 10)
            indicators['timing_precision'] = float(timing_precision)
            
            # 4. Sustained positioning (smart money holds positions)
            flow_persistence = np.mean(np.sign(recent_data['flow_momentum'])) * np.std(recent_data['flow_momentum'])
            indicators['position_sustainability'] = float(abs(flow_persistence))
            
            # 5. Dark pool activity (large orders executed quietly)
            dark_pool_activity = latest['dark_pool_proxy']
            indicators['dark_pool_activity'] = float(dark_pool_activity)
            
            # Smart money signature score
            smart_score = np.mean(list(indicators.values()))
            indicators['smart_money_score'] = float(smart_score)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error analyzing smart money indicators: {e}")
            return {}
            
    def _calculate_smart_money_consensus(self, smart_data: Dict) -> Dict:
        """Calculate smart money consensus indicators"""
        try:
            if not smart_data:
                return {}
                
            # Get consensus metrics
            consensus_scores = [data['smart_money_score'] for data in smart_data.values()]
            flow_directions = []
            volume_levels = []
            
            for data in smart_data.values():
                # Determine flow direction
                if data['flow_momentum'] > 0:
                    flow_directions.append('bullish')
                elif data['flow_momentum'] < 0:
                    flow_directions.append('bearish')
                else:
                    flow_directions.append('neutral')
                    
                # Volume level classification
                if data['volume_ratio'] > 2:
                    volume_levels.append('surge')
                elif data['volume_ratio'] > 1.5:
                    volume_levels.append('elevated')
                elif data['volume_ratio'] > 1:
                    volume_levels.append('above_average')
                else:
                    volume_levels.append('normal')
                    
            # Calculate consensus metrics
            bullish_ratio = flow_directions.count('bullish') / len(flow_directions)
            surge_ratio = volume_levels.count('surge') / len(volume_levels)
            
            # Consensus strength (agreement among smart money)
            consensus_strength = 1 - np.std(consensus_scores)
            
            return {
                'consensus_strength': float(consensus_strength),
                'bullish_consensus': float(bullish_ratio),
                'volume_surge_consensus': float(surge_ratio),
                'overall_direction': 'bullish' if bullish_ratio > 0.6 else 'bearish' if bullish_ratio < 0.4 else 'mixed',
                'confidence': float(max(bullish_ratio, 1 - bullish_ratio))
            }
            
        except Exception as e:
            logger.error(f"Error calculating smart money consensus: {e}")
            return {}
            
    def _detect_contrarian_signals(self, smart_data: Dict) -> Dict:
        """Detect contrarian signals from smart money flows"""
        try:
            contrarian_signals = {
                'detected_signals': [],
                'contrarian_strength': 0.0,
                'market_timing': 'neutral'
            }
            
            contrarian_indicators = []
            
            for etf, data in smart_data.items():
                # Look for smart money going opposite to recent price action
                price_momentum = data['flow_momentum']  # Flow momentum
                
                # Smart money contrarian signals
                if data['smart_money_score'] > 0.7:  # High confidence smart money
                    if data['flow_momentum'] > 0.02 and data['price_change'] < -0.01:
                        contrarian_indicators.append('smart_money_buying_weakness')
                        contrarian_signals['detected_signals'].append(f'{etf}: buying weakness')
                    elif data['flow_momentum'] < -0.02 and data['price_change'] > 0.01:
                        contrarian_indicators.append('smart_money_selling_strength')
                        contrarian_signals['detected_signals'].append(f'{etf}: selling strength')
                        
            # Calculate contrarian strength
            contrarian_signals['contrarian_strength'] = len(contrarian_signals['detected_signals']) / len(smart_data)
            
            # Market timing assessment
            if contrarian_signals['contrarian_strength'] > 0.5:
                contrarian_signals['market_timing'] = 'contrarian_active'
            elif len(contrarian_indicators) > 2:
                contrarian_signals['market_timing'] = 'contrarian_moderate'
            else:
                contrarian_signals['market_timing'] = 'trend_following'
                
            return contrarian_signals
            
        except Exception as e:
            logger.error(f"Error detecting contrarian signals: {e}")
            return {'detected_signals': [], 'contrarian_strength': 0.0, 'market_timing': 'neutral'}
            
    def _analyze_flow_divergence(self, smart_data: Dict) -> Dict:
        """Analyze divergence between different flow types"""
        try:
            divergence_analysis = {
                'smart_vs_retail_divergence': 0.0,
                'institutional_divergence': 0.0,
                'sector_flow_divergence': {}
            }
            
            if not smart_data:
                return divergence_analysis
                
            # Calculate smart money metrics
            smart_scores = [data['smart_money_score'] for data in smart_data.values()]
            volume_ratios = [data['volume_ratio'] for data in smart_data.values()]
            
            # Smart vs retail divergence (simplified proxy)
            avg_smart_score = np.mean(smart_scores)
            smart_volume_consistency = 1 - np.std(volume_ratios)
            
            # Higher smart money score + higher volume consistency = institutional divergence
            institutional_divergence = avg_smart_score * smart_volume_consistency
            divergence_analysis['institutional_divergence'] = float(institutional_divergence)
            
            # Smart vs retail proxy
            # Smart money tends to be more consistent and less emotional
            retail_excitement_proxy = np.std(volume_ratios)  # High volatility = retail excitement
            smart_consistency_proxy = 1 - retail_excitement_proxy
            
            divergence_analysis['smart_vs_retail_divergence'] = float(smart_consistency_proxy)
            
            # Sector flow divergence (if we had sector data)
            divergence_analysis['sector_flow_divergence'] = {
                'technology': float(np.mean([s for i, s in enumerate(smart_scores) if list(smart_data.keys())[i].lower() in ['qqq', 'qqqe']])),
                'broad_market': float(np.mean([s for i, s in enumerate(smart_scores) if list(smart_data.keys())[i].lower() in ['spy', 'vti']])),
                'bonds': float(np.mean([s for i, s in enumerate(smart_scores) if list(smart_data.keys())[i].lower() in ['tlt', 'agg']]))
            }
            
            return divergence_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing flow divergence: {e}")
            return {'smart_vs_retail_divergence': 0.0, 'institutional_divergence': 0.0, 'sector_flow_divergence': {}}
            
    def _calculate_timing_signals(self, smart_data: Dict) -> Dict:
        """Calculate market timing signals from smart money flows"""
        try:
            timing_signals = {
                'entry_signals': [],
                'exit_signals': [],
                'overall_timing': 'neutral',
                'timing_confidence': 0.0
            }
            
            if not smart_data:
                return timing_signals
                
            # Analyze timing patterns
            entry_signals = []
            exit_signals = []
            
            for etf, data in smart_data.items():
                # Entry signals
                if (data['smart_money_score'] > 0.6 and 
                    data['flow_momentum'] > 0.01 and 
                    data['volume_ratio'] > 1.3):
                    entry_signals.append(etf)
                    
                # Exit signals
                if (data['smart_money_score'] > 0.6 and 
                    data['flow_momentum'] < -0.01 and 
                    data['volume_ratio'] > 1.3):
                    exit_signals.append(etf)
                    
            timing_signals['entry_signals'] = entry_signals
            timing_signals['exit_signals'] = exit_signals
            
            # Overall timing assessment
            total_securities = len(smart_data)
            entry_ratio = len(entry_signals) / total_securities if total_securities > 0 else 0
            exit_ratio = len(exit_signals) / total_securities if total_securities > 0 else 0
            
            if entry_ratio > exit_ratio * 1.5:
                timing_signals['overall_timing'] = 'accumulation'
            elif exit_ratio > entry_ratio * 1.5:
                timing_signals['overall_timing'] = 'distribution'
            else:
                timing_signals['overall_timing'] = 'neutral'
                
            # Timing confidence
            timing_signals['timing_confidence'] = max(entry_ratio, exit_ratio)
            
            return timing_signals
            
        except Exception as e:
            logger.error(f"Error calculating timing signals: {e}")
            return {'entry_signals': [], 'exit_signals': [], 'overall_timing': 'neutral', 'timing_confidence': 0.0}
            
    async def _insider_trading_analysis(self) -> Dict:
        """Analyze insider trading patterns and signals"""
        try:
            # Note: This would require access to insider trading databases
            # For now, we'll create a framework and use proxy indicators
            
            insider_analysis = {
                'insider_activity_level': 'low',
                'buy_sell_ratio': 0.5,
                'insider_sentiment': 'neutral',
                'key_insider_moves': [],
                'sector_insider_activity': {}
            }
            
            # Proxy indicators for insider activity
            # In a real implementation, this would connect to actual insider trading APIs
            
            # Generate synthetic insider signals based on volume patterns
            # This simulates insider activity patterns
            volume_anomalies = []
            
            # Analyze for volume anomalies that might indicate insider activity
            major_etfs = ["SPY", "QQQ", "IWM"]
            
            for etf in major_etfs:
                try:
                    data = await self._fetch_market_data(etf, period="3mo")
                    if not data.empty:
                        # Look for volume spikes without corresponding price moves
                        recent_volume = data['Volume'].tail(5).mean()
                        historical_volume = data['Volume'].head(-5).tail(20).mean()
                        
                        if recent_volume > historical_volume * 2:
                            price_change = data['Close'].pct_change().tail(5).mean()
                            if abs(price_change) < 0.01:  # Low price change despite volume spike
                                volume_anomalies.append({
                                    'symbol': etf,
                                    'volume_spike': recent_volume / historical_volume,
                                    'price_impact': abs(price_change),
                                    'potential_insider': True
                                })
                except Exception as e:
                    logger.warning(f"Error analyzing insider activity for {etf}: {e}")
                    continue
                    
            # Calculate insider activity metrics
            if volume_anomalies:
                insider_analysis['insider_activity_level'] = 'high'
                insider_analysis['key_insider_moves'] = volume_anomalies
                
                # Calculate proxy buy/sell ratio
                buy_indicators = sum(1 for anomaly in volume_anomalies if anomaly['potential_insider'])
                total_indicators = len(volume_anomalies)
                insider_analysis['buy_sell_ratio'] = buy_indicators / total_indicators
                
                # Insider sentiment
                if insider_analysis['buy_sell_ratio'] > 0.6:
                    insider_analysis['insider_sentiment'] = 'bullish'
                elif insider_analysis['buy_sell_ratio'] < 0.4:
                    insider_analysis['insider_sentiment'] = 'bearish'
                else:
                    insider_analysis['insider_sentiment'] = 'neutral'
            else:
                insider_analysis['insider_activity_level'] = 'normal'
                
            return insider_analysis
            
        except Exception as e:
            logger.error(f"Error in insider trading analysis: {e}")
            return {'error': str(e)}
            
    async def _hedge_fund_flow_analysis(self) -> Dict:
        """Analyze hedge fund positioning and flows"""
        try:
            # Hedge fund style ETFs and indicators
            hedge_fund_indicators = {
                'long_short': ["HDG", "MLP", "QAI"],  # Long/short strategies
                'market_neutral': ["CSM", "MVV"],     # Market neutral strategies
                'alternative_beta': ["HDG", "AIY"],  # Alternative beta strategies
                'tail_risk': ["TIP", "VXX"]           # Tail risk hedging
            }
            
            hedge_fund_data = {}
            
            for strategy, etfs in hedge_fund_indicators.items():
                strategy_flows = []
                
                for etf in etfs:
                    try:
                        data = await self._fetch_market_data(etf)
                        if not data.empty:
                            # Analyze hedge fund specific patterns
                            hf_metrics = self._calculate_hedge_fund_metrics(data)
                            hf_metrics['etf'] = etf
                            strategy_flows.append(hf_metrics)
                    except Exception as e:
                        logger.warning(f"Error analyzing hedge fund flow for {etf}: {e}")
                        continue
                        
                if strategy_flows:
                    hedge_fund_data[strategy] = {
                        'strategy_flows': strategy_flows,
                        'avg_flow_momentum': np.mean([f['flow_momentum'] for f in strategy_flows]),
                        'strategy_sentiment': self._calculate_strategy_sentiment(strategy_flows)
                    }
                    
            if not hedge_fund_data:
                return {'error': 'No hedge fund flow data available'}
                
            # Calculate hedge fund positioning
            positioning_analysis = self._calculate_hedge_fund_positioning(hedge_fund_data)
            
            # Hedge fund conviction signals
            conviction_signals = self._detect_hedge_fund_conviction(hedge_fund_data)
            
            return {
                'hedge_fund_data': hedge_fund_data,
                'positioning_analysis': positioning_analysis,
                'conviction_signals': conviction_signals
            }
            
        except Exception as e:
            logger.error(f"Error in hedge fund flow analysis: {e}")
            return {'error': str(e)}
            
    def _calculate_hedge_fund_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate hedge fund specific metrics"""
        try:
            if data.empty:
                return {}
                
            latest = data.iloc[-1]
            recent_data = data.tail(10)
            
            metrics = {
                'flow_momentum': latest['flow_momentum'],
                'volume_ratio': latest['volume_ratio'],
                'price_volatility': recent_data['price_change'].std(),
                'hedge_fund_signature': self._calculate_hf_signature(recent_data)
            }
            
            return {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                   for k, v in metrics.items()}
            
        except Exception as e:
            logger.error(f"Error calculating hedge fund metrics: {e}")
            return {}
            
    def _calculate_hf_signature(self, data: pd.DataFrame) -> float:
        """Calculate hedge fund signature (pattern recognition)"""
        try:
            # Hedge funds tend to have more consistent, lower-impact trading
            volume_consistency = 1 - data['volume_ratio'].std()
            price_impact_consistency = 1 - data['price_change'].std() * 10
            
            # Hedge fund signature combines consistency metrics
            hf_signature = (volume_consistency + price_impact_consistency) / 2
            
            return float(max(0, min(1, hf_signature)))
            
        except Exception as e:
            logger.error(f"Error calculating hedge fund signature: {e}")
            return 0
            
    def _calculate_strategy_sentiment(self, strategy_flows: List[Dict]) -> str:
        """Calculate sentiment for a hedge fund strategy"""
        try:
            if not strategy_flows:
                return 'neutral'
                
            avg_momentum = np.mean([f['flow_momentum'] for f in strategy_flows])
            
            if avg_momentum > 0.02:
                return 'bullish'
            elif avg_momentum < -0.02:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error calculating strategy sentiment: {e}")
            return 'neutral'
            
    def _calculate_hedge_fund_positioning(self, hf_data: Dict) -> Dict:
        """Calculate overall hedge fund positioning"""
        try:
            positioning = {
                'overall_sentiment': 'neutral',
                'risk_positioning': 'neutral',
                'conviction_level': 0.0
            }
            
            # Aggregate sentiment across strategies
            sentiments = [data['strategy_sentiment'] for data in hf_data.values()]
            bullish_count = sentiments.count('bullish')
            bearish_count = sentiments.count('bearish')
            
            if bullish_count > bearish_count:
                positioning['overall_sentiment'] = 'bullish'
            elif bearish_count > bullish_count:
                positioning['overall_sentiment'] = 'bearish'
                
            # Calculate conviction level
            momentum_values = [data['avg_flow_momentum'] for data in hf_data.values()]
            conviction = np.std(momentum_values) if len(momentum_values) > 1 else 0
            positioning['conviction_level'] = float(conviction)
            
            # Risk positioning (based on alternative beta flows)
            if 'alternative_beta' in hf_data:
                alt_beta_sentiment = hf_data['alternative_beta']['strategy_sentiment']
                positioning['risk_positioning'] = alt_beta_sentiment
                
            return positioning
            
        except Exception as e:
            logger.error(f"Error calculating hedge fund positioning: {e}")
            return {'overall_sentiment': 'neutral', 'risk_positioning': 'neutral', 'conviction_level': 0.0}
            
    def _detect_hedge_fund_conviction(self, hf_data: Dict) -> Dict:
        """Detect hedge fund conviction signals"""
        try:
            conviction_signals = {
                'high_conviction_positions': [],
                'conviction_trends': {},
                'positioning_changes': []
            }
            
            for strategy, data in hf_data.items():
                # High conviction: high flow momentum + consistent volume
                avg_momentum = data['avg_flow_momentum']
                flow_consistency = 1 - np.std([f['flow_momentum'] for f in data['strategy_flows']])
                
                if abs(avg_momentum) > 0.02 and flow_consistency > 0.5:
                    conviction_signals['high_conviction_positions'].append(strategy)
                    
                # Conviction trends
                if avg_momentum > 0.02:
                    conviction_signals['conviction_trends'][strategy] = 'building_long'
                elif avg_momentum < -0.02:
                    conviction_signals['conviction_trends'][strategy] = 'building_short'
                else:
                    conviction_signals['conviction_trends'][strategy] = 'neutral'
                    
            return conviction_signals
            
        except Exception as e:
            logger.error(f"Error detecting hedge fund conviction: {e}")
            return {'high_conviction_positions': [], 'conviction_trends': {}, 'positioning_changes': []}
            
    async def _mutual_fund_flow_analysis(self) -> Dict:
        """Analyze mutual fund flows and positioning"""
        try:
            # Broad market mutual fund proxies
            mutual_fund_proxies = ["VTI", "VOO", "IVV", "VTV", "IWD"]
            
            mutual_fund_data = {}
            
            for fund in mutual_fund_proxies:
                try:
                    data = await self._fetch_market_data(fund)
                    if not data.empty:
                        # Analyze mutual fund specific patterns
                        mf_metrics = self._calculate_mutual_fund_metrics(data)
                        mf_metrics['fund'] = fund
                        mutual_fund_data[fund] = mf_metrics
                except Exception as e:
                    logger.warning(f"Error analyzing mutual fund flow for {fund}: {e}")
                    continue
                    
            if not mutual_fund_data:
                return {'error': 'No mutual fund flow data available'}
                
            # Calculate mutual fund positioning
            positioning = self._calculate_mutual_fund_positioning(mutual_fund_data)
            
            # Flow sustainability analysis
            sustainability = self._analyze_flow_sustainability(mutual_fund_data)
            
            # Risk appetite signals
            risk_appetite = self._calculate_mutual_fund_risk_appetite(mutual_fund_data)
            
            return {
                'mutual_fund_data': mutual_fund_data,
                'positioning': positioning,
                'sustainability': sustainability,
                'risk_appetite': risk_appetite
            }
            
        except Exception as e:
            logger.error(f"Error in mutual fund flow analysis: {e}")
            return {'error': str(e)}
            
    def _calculate_mutual_fund_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate mutual fund specific metrics"""
        try:
            if data.empty:
                return {}
                
            latest = data.iloc[-1]
            recent_data = data.tail(10)
            
            # Mutual funds tend to have more stable, trend-following flows
            metrics = {
                'flow_momentum': latest['flow_momentum'],
                'volume_ratio': latest['volume_ratio'],
                'trend_following_score': self._calculate_trend_following_score(recent_data),
                'fund_flow_signature': self._calculate_fund_signature(recent_data)
            }
            
            return {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                   for k, v in metrics.items()}
            
        except Exception as e:
            logger.error(f"Error calculating mutual fund metrics: {e}")
            return {}
            
    def _calculate_trend_following_score(self, data: pd.DataFrame) -> float:
        """Calculate trend following score for mutual funds"""
        try:
            # Mutual funds tend to follow trends with less contrarian behavior
            price_momentum = data['price_change'].iloc[-1]
            flow_momentum = data['flow_momentum'].iloc[-1]
            
            # High correlation between price and flow = trend following
            if len(data) > 2:
                correlation = np.corrcoef(data['price_change'], data['flow_momentum'])[0, 1]
                trend_score = abs(correlation) if not np.isnan(correlation) else 0
            else:
                trend_score = 0
                
            return float(max(0, min(1, trend_score)))
            
        except Exception as e:
            logger.error(f"Error calculating trend following score: {e}")
            return 0
            
    def _calculate_fund_signature(self, data: pd.DataFrame) -> float:
        """Calculate mutual fund flow signature"""
        try:
            # Mutual funds have more gradual, sustainable flows
            flow_volatility = data['flow_momentum'].std()
            volume_stability = 1 - data['volume_ratio'].std()
            
            # Lower volatility + stable volume = mutual fund signature
            fund_signature = (1 - flow_volatility + volume_stability) / 2
            
            return float(max(0, min(1, fund_signature)))
            
        except Exception as e:
            logger.error(f"Error calculating fund signature: {e}")
            return 0
            
    def _calculate_mutual_fund_positioning(self, mf_data: Dict) -> Dict:
        """Calculate overall mutual fund positioning"""
        try:
            positioning = {
                'overall_sentiment': 'neutral',
                'positioning_intensity': 0.0,
                'positioning_consensus': 0.0
            }
            
            # Calculate sentiment
            sentiments = []
            momentum_values = []
            
            for data in mf_data.values():
                sentiment = 'bullish' if data['flow_momentum'] > 0 else 'bearish'
                sentiments.append(sentiment)
                momentum_values.append(data['flow_momentum'])
                
            # Overall sentiment
            bullish_ratio = sentiments.count('bullish') / len(sentiments)
            if bullish_ratio > 0.6:
                positioning['overall_sentiment'] = 'bullish'
            elif bullish_ratio < 0.4:
                positioning['overall_sentiment'] = 'bearish'
                
            # Positioning intensity
            positioning['positioning_intensity'] = float(np.mean([abs(m) for m in momentum_values]))
            
            # Positioning consensus
            positioning['positioning_consensus'] = 1 - np.std(momentum_values) if len(momentum_values) > 1 else 0
            
            return positioning
            
        except Exception as e:
            logger.error(f"Error calculating mutual fund positioning: {e}")
            return {'overall_sentiment': 'neutral', 'positioning_intensity': 0.0, 'positioning_consensus': 0.0}
            
    def _analyze_flow_sustainability(self, mf_data: Dict) -> Dict:
        """Analyze sustainability of mutual fund flows"""
        try:
            sustainability = {
                'sustainability_score': 0.0,
                'sustainable_flows': [],
                'unsustainable_flows': []
            }
            
            for fund, data in mf_data.items():
                # Sustainable flows: consistent volume, gradual momentum changes
                volume_consistency = 1 - abs(data['volume_ratio'] - 1)
                momentum_gradual = 1 - min(1, abs(data['flow_momentum']) * 10)
                
                sustainable_score = (volume_consistency + momentum_gradual) / 2
                
                if sustainable_score > 0.7:
                    sustainability['sustainable_flows'].append(fund)
                else:
                    sustainability['unsustainable_flows'].append(fund)
                    
            sustainability['sustainability_score'] = (
                len(sustainability['sustainable_flows']) / 
                len(mf_data) if mf_data else 0
            )
            
            return sustainability
            
        except Exception as e:
            logger.error(f"Error analyzing flow sustainability: {e}")
            return {'sustainability_score': 0.0, 'sustainable_flows': [], 'unsustainable_flows': []}
            
    def _calculate_mutual_fund_risk_appetite(self, mf_data: Dict) -> Dict:
        """Calculate mutual fund risk appetite"""
        try:
            risk_appetite = {
                'risk_appetite_level': 'moderate',
                'risk_appetite_trend': 'stable',
                'defensive_positioning': 0.0
            }
            
            # Calculate risk metrics
            risk_indicators = []
            
            for data in mf_data.values():
                # Higher volume ratios might indicate increased risk appetite
                volume_risk = data['volume_ratio']
                momentum_risk = abs(data['flow_momentum'])
                risk_indicators.append((volume_risk + momentum_risk) / 2)
                
            avg_risk = np.mean(risk_indicators) if risk_indicators else 0
            
            # Risk appetite level
            if avg_risk > 2:
                risk_appetite['risk_appetite_level'] = 'high'
            elif avg_risk > 1.5:
                risk_appetite['risk_appetite_level'] = 'moderate_high'
            elif avg_risk < 0.8:
                risk_appetite['risk_appetite_level'] = 'low'
            else:
                risk_appetite['risk_appetite_level'] = 'moderate'
                
            # Defensive positioning
            risk_appetite['defensive_positioning'] = float(1 - avg_risk / 3)  # Scale to 0-1
            
            return risk_appetite
            
        except Exception as e:
            logger.error(f"Error calculating mutual fund risk appetite: {e}")
            return {'risk_appetite_level': 'moderate', 'risk_appetite_trend': 'stable', 'defensive_positioning': 0.0}
            
    async def get_flow_pulse(self) -> Dict:
        """Get comprehensive flow analysis"""
        try:
            # Run all flow analyses in parallel
            flow_tasks = [
                self._institutional_flow_analysis(),
                self._smart_money_flow_analysis(),
                self._insider_trading_analysis(),
                self._hedge_fund_flow_analysis(),
                self._mutual_fund_flow_analysis()
            ]
            
            results = await asyncio.gather(*flow_tasks, return_exceptions=True)
            (
                institutional_flows, smart_money_flows,
                insider_analysis, hedge_fund_flows,
                mutual_fund_flows
            ) = results
            
            # Calculate overall Flow Momentum Score (FMS)
            fms_components = []
            
            for result in results:
                if isinstance(result, dict) and not result.get('error'):
                    # Extract numeric scores from each analysis
                    score = self._extract_flow_score(result)
                    if score is not None:
                        fms_components.append(score)
                        
            if fms_components:
                fms_score = np.mean(fms_components)
                fms_volatility = np.std(fms_components)
                
                # Classify flow state
                if fms_score > 0.3:
                    flow_state = 'strong_inflows'
                elif fms_score < -0.3:
                    flow_state = 'strong_outflows'
                elif fms_volatility > 0.5:
                    flow_state = 'flow_rotation'
                else:
                    flow_state = 'balanced_flows'
                    
                return {
                    'flow_momentum_score': fms_score,
                    'fms_volatility': fms_volatility,
                    'flow_state': flow_state,
                    'analysis_breakdown': {
                        'institutional_flows': institutional_flows,
                        'smart_money_flows': smart_money_flows,
                        'insider_analysis': insider_analysis,
                        'hedge_fund_flows': hedge_fund_flows,
                        'mutual_fund_flows': mutual_fund_flows
                    },
                    'timestamp': datetime.utcnow(),
                    'confidence': min(1.0, 1 - (fms_volatility / 2))
                }
            else:
                return {'error': 'Unable to calculate flow momentum score'}
                
        except Exception as e:
            logger.error(f"Error getting flow pulse: {e}")
            return {'error': str(e)}
            
    def _extract_flow_score(self, analysis_result: Dict) -> Optional[float]:
        """Extract numeric flow score from analysis result"""
        try:
            if 'overall_institutional_score' in analysis_result:
                return analysis_result['overall_institutional_score']
            elif 'smart_consensus' in analysis_result and 'consensus_strength' in analysis_result['smart_consensus']:
                return analysis_result['smart_consensus']['consensus_strength'] - 0.5
            elif 'buy_sell_ratio' in analysis_result:
                return analysis_result['buy_sell_ratio'] - 0.5
            elif 'positioning_analysis' in analysis_result and 'conviction_level' in analysis_result['positioning_analysis']:
                return analysis_result['positioning_analysis']['conviction_level'] - 0.5
            elif 'positioning' in analysis_result and 'positioning_intensity' in analysis_result['positioning']:
                return analysis_result['positioning']['positioning_intensity'] - 0.5
            else:
                return None
                
        except Exception:
            return None
            
    async def store_flow_data(self, flow_data: Dict):
        """Store flow metrics in time-series database"""
        try:
            if self.db_manager and 'timestamp' in flow_data:
                # Store Flow Momentum Score
                await self.db_manager.influxdb_client.write_points(
                    database='market_pulse',
                    measurement='flow_metrics',
                    tags={
                        'engine': 'flow_pulse',
                        'state': flow_data.get('flow_state', 'unknown')
                    },
                    fields={
                        'fms_score': float(flow_data.get('flow_momentum_score', 0)),
                        'fms_volatility': float(flow_data.get('fms_volatility', 0)),
                        'confidence': float(flow_data.get('confidence', 0))
                    },
                    time=flow_data['timestamp']
                )
                
                # Store component scores
                for component_name, analysis in flow_data.get('analysis_breakdown', {}).items():
                    if isinstance(analysis, dict):
                        score = self._extract_flow_score(analysis)
                        if score is not None:
                            await self.db_manager.influxdb_client.write_points(
                                database='market_pulse',
                                measurement='flow_components',
                                tags={
                                    'component': component_name,
                                    'engine': 'flow_pulse'
                                },
                                fields={'component_score': float(score)},
                                time=flow_data['timestamp']
                            )
                            
            logger.debug("Flow data stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing flow data: {e}")
            
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
                'cache_size': len(self.flow_cache),
                'models_loaded': len(self.models),
                'tracked_institutional_categories': len(self.institutional_categories),
                'smart_money_indicators': len(self.smart_money_indicators),
                'health_score': self._calculate_engine_health()
            }
            
        except Exception as e:
            logger.error(f"Error getting flow engine status: {e}")
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
                cache_freshness = max(0, 1 - (minutes_since_update / 30))
                health_factors.append(cache_freshness)
            
            # Model availability
            health_factors.append(min(1.0, len(self.models) / 3))
            
            # Data source coverage
            total_data_sources = len(self.institutional_categories)
            health_factors.append(min(1.0, total_data_sources / 5))
            
            return np.mean(health_factors) if health_factors else 0.0
            
        except Exception:
            return 0.0