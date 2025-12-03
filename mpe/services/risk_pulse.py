"""
Risk Pulse Engine - Market Stress & Risk Accumulation Detection
Real-time tracking of systemic risk, stress indicators, and risk-on/risk-off flows
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
from sklearn.cluster import KMeans
import talib
from database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class RiskPulseEngine:
    """Risk Stress Monitor - Detecting systemic risk accumulation and stress events"""
    
    def __init__(self):
        self.name = "Risk Pulse Engine"
        self.version = "1.0.0"
        self.status = "initialized"
        self.last_update = None
        self.risk_cache = {}
        self.models = {}
        self.scaler = StandardScaler()
        
        # Risk tracking models
        self.risk_models = {
            "systemic_risk": self._systemic_risk_analysis,
            "stress_indicators": self._stress_indicators_analysis,
            "risk_on_off": self._risk_on_off_analysis,
            "volatility_regime": self._volatility_regime_analysis,
            "correlation_breakdown": self._correlation_breakdown_analysis
        }
        
        # Systemic risk indicators
        self.systemic_indicators = {
            "vix_spy_spread": "VIX vs SPY implied correlation",
            "credit_spreads": "Credit spread widening",
            "volatility_surface": "Options volatility skew",
            "cross_asset_correlations": "Correlation spikes",
            "funding_stress": "Money market stress",
            "leverage_indicators": "Margin debt and leverage",
            "liquidity_indicators": "Bid-ask spreads and depth"
        }
        
        # Risk-on/Risk-off assets
        self.risk_assets = {
            "risk_on": ["SPY", "QQQ", "IWM", "EEM", "GLD"],
            "risk_off": ["TLT", "IEF", "SHY", "DXY", "VIX"]
        }
        
        # Safe haven assets
        self.safe_havens = {
            "bonds": ["TLT", "IEF", "SHY"],
            "currencies": ["DXY", "JPY", "CHF"],
            "commodities": ["GLD", "SLV", "VIX"],
            "defensive_stocks": ["XLU", "XLV", "XLP"]
        }
        
        # Stress indicators and thresholds
        self.stress_thresholds = {
            "vix_level": 30,           # VIX above 30 = stress
            "credit_spread": 2.0,      # Credit spread widening
            "correlation_spike": 0.8,  # High correlation regime
            "volatility_regime": 1.5,  # Volatility regime change
            "liquidity_stress": 2.0,   # Liquidity stress indicator
            "systemic_pressure": 0.7   # Overall systemic risk
        }
        
        # Volatility regimes
        self.volatility_regimes = {
            "low_vol": {"vix_range": [10, 15], "characteristics": "Calm markets, risk-taking"},
            "normal_vol": {"vix_range": [15, 25], "characteristics": "Normal market conditions"},
            "elevated_vol": {"vix_range": [25, 35], "characteristics": "Heightened uncertainty"},
            "high_vol": {"vix_range": [35, 50], "characteristics": "Market stress, risk aversion"},
            "crisis_vol": {"vix_range": [50, 100], "characteristics": "Market crisis, panic"}
        }
        
        # Risk clustering
        self.risk_clusters = {
            "tail_risk": "Extreme downside events",
            "liquidity_risk": "Market liquidity stress",
            "credit_risk": "Credit market stress",
            "currency_risk": "Foreign exchange stress",
            "commodity_risk": "Commodity price stress"
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
            logger.info("Risk Pulse Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Risk Pulse Engine: {e}")
            self.status = "error"
            
    def _initialize_models(self):
        """Initialize ML models for risk prediction"""
        try:
            # Random Forest for risk regime prediction
            self.models['risk_regime'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Random Forest for stress event prediction
            self.models['stress_prediction'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=8
            )
            
            # K-means for risk clustering
            self.models['risk_clusters'] = KMeans(
                n_clusters=4,
                random_state=42
            )
            
            # Scaler for risk feature normalization
            self.scaler = StandardScaler()
            
            logger.info("Risk prediction models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize risk models: {e}")
            
    async def _fetch_risk_data(self, symbols: List[str], period: str = "3mo") -> pd.DataFrame:
        """Fetch risk-related market data"""
        try:
            risk_data = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    if not hist.empty and len(hist) > 30:
                        risk_data[symbol] = hist
                except Exception as e:
                    logger.warning(f"Error fetching risk data for {symbol}: {e}")
                    continue
                    
            if not risk_data:
                return pd.DataFrame()
                
            # Combine all data
            combined_data = pd.DataFrame()
            
            for symbol, data in risk_data.items():
                for col in data.columns:
                    combined_data[f"{symbol}_{col}"] = data[col]
                    
            return combined_data
            
        except Exception as e:
            logger.error(f"Error fetching risk data: {e}")
            return pd.DataFrame()
            
    async def _systemic_risk_analysis(self) -> Dict:
        """Analyze systemic risk indicators"""
        try:
            # Fetch key risk indicators
            risk_symbols = ["SPY", "QQQ", "TLT", "GLD", "VXX", "TLT"]
            risk_data = await self._fetch_risk_data(risk_symbols)
            
            if risk_data.empty:
                return {'error': 'No risk data available'}
                
            systemic_indicators = {}
            
            # VIX (stress indicator)
            if 'VXX_Close' in risk_data.columns:
                vix_data = risk_data['VXX_Close']
                vix_current = vix_data.iloc[-1]
                vix_ma = vix_data.rolling(20).mean().iloc[-1]
                vix_percentile = (vix_data.rolling(60).rank(pct=True).iloc[-1]) * 100
                
                systemic_indicators['vix'] = {
                    'current_level': float(vix_current),
                    'moving_average': float(vix_ma),
                    'percentile': float(vix_percentile),
                    'stress_level': self._calculate_vix_stress(vix_current, vix_ma)
                }
                
            # SPY volatility
            if 'SPY_Close' in risk_data.columns:
                spy_data = risk_data['SPY_Close']
                spy_returns = spy_data.pct_change().dropna()
                spy_vol = spy_returns.rolling(20).std() * np.sqrt(252)  # Annualized
                spy_vol_current = spy_vol.iloc[-1]
                spy_vol_percentile = spy_vol.rolling(60).rank(pct=True).iloc[-1] * 100
                
                systemic_indicators['spy_volatility'] = {
                    'current_level': float(spy_vol_current),
                    'percentile': float(spy_vol_percentile),
                    'volatility_regime': self._determine_volatility_regime(spy_vol_current)
                }
                
            # Risk-on/Risk-off ratio
            if all(col in risk_data.columns for col in ['SPY_Close', 'TLT_Close']):
                spy_prices = risk_data['SPY_Close']
                tlt_prices = risk_data['TLT_Close']
                
                # Calculate risk-on/off spread
                spy_normalized = spy_prices / spy_prices.rolling(20).mean()
                tlt_normalized = tlt_prices / tlt_prices.rolling(20).mean()
                risk_spread = spy_normalized - tlt_normalized
                
                systemic_indicators['risk_spread'] = {
                    'current_spread': float(risk_spread.iloc[-1]),
                    'spread_trend': float(risk_spread.diff(5).iloc[-1]),
                    'risk_sentiment': 'risk_on' if risk_spread.iloc[-1] > 0 else 'risk_off'
                }
                
            # Safe haven demand
            if 'GLD_Close' in risk_data.columns:
                gold_data = risk_data['GLD_Close']
                gold_momentum = gold_data.pct_change(5).iloc[-1]
                gold_volume = risk_data['GLD_Volume'].iloc[-1] if 'GLD_Volume' in risk_data.columns else 0
                gold_volume_ma = risk_data['GLD_Volume'].rolling(20).mean().iloc[-1] if 'GLD_Volume' in risk_data.columns else 1
                
                systemic_indicators['safe_haven_demand'] = {
                    'gold_momentum': float(gold_momentum),
                    'volume_surge': float(gold_volume / gold_volume_ma) if gold_volume_ma > 0 else 0,
                    'safe_haven_activity': 'elevated' if gold_momentum > 0.01 or gold_volume / gold_volume_ma > 1.5 else 'normal'
                }
                
            # Calculate overall systemic risk score
            risk_score = self._calculate_systemic_risk_score(systemic_indicators)
            
            return {
                'systemic_indicators': systemic_indicators,
                'overall_risk_score': risk_score,
                'risk_level': self._classify_risk_level(risk_score),
                'stress_signals': self._identify_stress_signals(systemic_indicators)
            }
            
        except Exception as e:
            logger.error(f"Error in systemic risk analysis: {e}")
            return {'error': str(e)}
            
    def _calculate_vix_stress(self, vix_current: float, vix_ma: float) -> str:
        """Calculate VIX-based stress level"""
        try:
            if vix_current > 30:
                return 'high_stress'
            elif vix_current > 25:
                return 'moderate_stress'
            elif vix_current < 15:
                return 'low_stress'
            else:
                return 'normal'
        except:
            return 'unknown'
            
    def _determine_volatility_regime(self, vol_current: float) -> str:
        """Determine current volatility regime"""
        try:
            if vol_current > 0.5:
                return 'high_vol'
            elif vol_current > 0.3:
                return 'elevated_vol'
            elif vol_current > 0.2:
                return 'normal_vol'
            else:
                return 'low_vol'
        except:
            return 'unknown'
            
    def _calculate_systemic_risk_score(self, indicators: Dict) -> float:
        """Calculate overall systemic risk score"""
        try:
            risk_components = []
            
            # VIX stress component
            if 'vix' in indicators:
                vix_stress = indicators['vix']['stress_level']
                if vix_stress == 'high_stress':
                    risk_components.append(0.8)
                elif vix_stress == 'moderate_stress':
                    risk_components.append(0.5)
                elif vix_stress == 'normal':
                    risk_components.append(0.3)
                else:
                    risk_components.append(0.1)
                    
            # Volatility component
            if 'spy_volatility' in indicators:
                vol_regime = indicators['spy_volatility']['volatility_regime']
                if vol_regime == 'high_vol':
                    risk_components.append(0.9)
                elif vol_regime == 'elevated_vol':
                    risk_components.append(0.6)
                elif vol_regime == 'normal_vol':
                    risk_components.append(0.4)
                else:
                    risk_components.append(0.2)
                    
            # Risk sentiment component
            if 'risk_spread' in indicators:
                risk_sentiment = indicators['risk_spread']['risk_sentiment']
                if risk_sentiment == 'risk_off':
                    risk_components.append(0.7)
                else:
                    risk_components.append(0.3)
                    
            # Safe haven demand component
            if 'safe_haven_demand' in indicators:
                haven_activity = indicators['safe_haven_demand']['safe_haven_activity']
                if haven_activity == 'elevated':
                    risk_components.append(0.6)
                else:
                    risk_components.append(0.2)
                    
            return np.mean(risk_components) if risk_components else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating systemic risk score: {e}")
            return 0.5
            
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify overall risk level"""
        try:
            if risk_score > 0.8:
                return 'extreme_risk'
            elif risk_score > 0.6:
                return 'high_risk'
            elif risk_score > 0.4:
                return 'moderate_risk'
            elif risk_score > 0.2:
                return 'low_risk'
            else:
                return 'minimal_risk'
        except:
            return 'unknown'
            
    def _identify_stress_signals(self, indicators: Dict) -> List[str]:
        """Identify current stress signals"""
        try:
            stress_signals = []
            
            # Check each indicator for stress signals
            for indicator_name, indicator_data in indicators.items():
                if isinstance(indicator_data, dict):
                    for metric_name, metric_value in indicator_data.items():
                        # Check for stress conditions
                        if 'stress_level' in metric_name and metric_value in ['high_stress', 'moderate_stress']:
                            stress_signals.append(f'{indicator_name}_stress')
                        elif 'regime' in metric_name and metric_value in ['high_vol', 'elevated_vol']:
                            stress_signals.append(f'{indicator_name}_volatility_stress')
                        elif 'activity' in metric_name and metric_value == 'elevated':
                            stress_signals.append(f'{indicator_name}_activity_surge')
                            
            return stress_signals
            
        except Exception as e:
            logger.error(f"Error identifying stress signals: {e}")
            return []
            
    async def _stress_indicators_analysis(self) -> Dict:
        """Analyze various stress indicators across markets"""
        try:
            # Get stress-related data
            stress_symbols = ["SPY", "TLT", "GLD", "DXY", "VXX"]
            stress_data = await self._fetch_risk_data(stress_symbols)
            
            if stress_data.empty:
                return {'error': 'No stress data available'}
                
            stress_indicators = {}
            
            # Liquidity stress
            if 'TLT_Volume' in stress_data.columns:
                bond_volume = stress_data['TLT_Volume'].rolling(5).mean().iloc[-1]
                bond_volume_ma = stress_data['TLT_Volume'].rolling(20).mean().iloc[-1]
                liquidity_stress = max(0, min(1, (bond_volume - bond_volume_ma) / bond_volume_ma)) if bond_volume_ma > 0 else 0
                
                stress_indicators['liquidity_stress'] = {
                    'bond_volume_ratio': float(bond_volume / bond_volume_ma) if bond_volume_ma > 0 else 1,
                    'stress_level': 'high' if liquidity_stress > 0.5 else 'moderate' if liquidity_stress > 0.2 else 'normal'
                }
                
            # Currency stress (DXY proxy)
            if 'DXY_Close' in stress_data.columns:
                dxy_data = stress_data['DXY_Close']
                dxy_volatility = dxy_data.pct_change().rolling(20).std().iloc[-1]
                dxy_trend = dxy_data.pct_change(5).iloc[-1]
                
                stress_indicators['currency_stress'] = {
                    'volatility_level': float(dxy_volatility),
                    'trend_momentum': float(dxy_trend),
                    'stress_indicators': self._assess_currency_stress(dxy_volatility, dxy_trend)
                }
                
            # Commodity stress (Gold proxy)
            if 'GLD_Close' in stress_data.columns:
                gold_data = stress_data['GLD_Close']
                gold_momentum = gold_data.pct_change(10).iloc[-1]
                gold_volatility = gold_data.pct_change().rolling(10).std().iloc[-1]
                
                stress_indicators['commodity_stress'] = {
                    'gold_momentum': float(gold_momentum),
                    'volatility_level': float(gold_volatility),
                    'stress_pattern': 'safe_haven_buying' if gold_momentum > 0.02 else 'normal'
                }
                
            # Equity market stress
            if 'SPY_Close' in stress_data.columns:
                spy_data = stress_data['SPY_Close']
                spy_drawdown = (spy_data.iloc[-1] / spy_data.expanding().max().iloc[-1]) - 1
                spy_vol_spike = spy_data.pct_change().rolling(10).std().iloc[-1]
                spy_ma = spy_data.rolling(50).mean().iloc[-1]
                
                stress_indicators['equity_stress'] = {
                    'current_drawdown': float(spy_drawdown),
                    'volatility_spike': float(spy_vol_spike),
                    'trend_health': 'bearish' if spy_data.iloc[-1] < spy_ma * 0.95 else 'bullish',
                    'stress_level': self._assess_equity_stress(spy_drawdown, spy_vol_spike)
                }
                
            # Overall stress assessment
            stress_level = self._calculate_overall_stress_level(stress_indicators)
            
            return {
                'stress_indicators': stress_indicators,
                'overall_stress_level': stress_level,
                'stress_sources': self._identify_stress_sources(stress_indicators),
                'stress_dispersion': self._calculate_stress_dispersion(stress_indicators)
            }
            
        except Exception as e:
            logger.error(f"Error in stress indicators analysis: {e}")
            return {'error': str(e)}
            
    def _assess_currency_stress(self, volatility: float, trend: float) -> str:
        """Assess currency-related stress"""
        try:
            if volatility > 0.01:  # High volatility
                return 'high_volatility_stress'
            elif abs(trend) > 0.02:  # Strong trend
                return 'trend_stress'
            else:
                return 'normal'
        except:
            return 'normal'
            
    def _assess_equity_stress(self, drawdown: float, vol_spike: float) -> str:
        """Assess equity market stress"""
        try:
            if drawdown < -0.1 or vol_spike > 0.025:
                return 'high_stress'
            elif drawdown < -0.05 or vol_spike > 0.02:
                return 'moderate_stress'
            else:
                return 'normal'
        except:
            return 'normal'
            
    def _calculate_overall_stress_level(self, indicators: Dict) -> str:
        """Calculate overall stress level across indicators"""
        try:
            stress_scores = []
            
            for indicator_name, data in indicators.items():
                if isinstance(data, dict):
                    if 'stress_level' in data:
                        level = data['stress_level']
                        if level == 'high':
                            stress_scores.append(0.8)
                        elif level == 'moderate':
                            stress_scores.append(0.5)
                        else:
                            stress_scores.append(0.2)
                            
            if stress_scores:
                avg_stress = np.mean(stress_scores)
                
                if avg_stress > 0.7:
                    return 'extreme_stress'
                elif avg_stress > 0.5:
                    return 'high_stress'
                elif avg_stress > 0.3:
                    return 'moderate_stress'
                else:
                    return 'low_stress'
            else:
                return 'unknown'
                
        except Exception:
            return 'unknown'
            
    def _identify_stress_sources(self, indicators: Dict) -> List[str]:
        """Identify main sources of market stress"""
        try:
            stress_sources = []
            
            for indicator_name, data in indicators.items():
                if isinstance(data, dict):
                    # Check for stress conditions
                    if data.get('stress_level') == 'high':
                        stress_sources.append(f'{indicator_name}_high_stress')
                    elif data.get('stress_level') == 'moderate':
                        stress_sources.append(f'{indicator_name}_moderate_stress')
                        
            return stress_sources
            
        except Exception:
            return []
            
    def _calculate_stress_dispersion(self, indicators: Dict) -> float:
        """Calculate how stress is dispersed across different markets"""
        try:
            stress_levels = []
            
            for data in indicators.values():
                if isinstance(data, dict) and 'stress_level' in data:
                    level = data['stress_level']
                    if level == 'high':
                        stress_levels.append(1.0)
                    elif level == 'moderate':
                        stress_levels.append(0.5)
                    else:
                        stress_levels.append(0.0)
                        
            # High dispersion = stress concentrated in specific areas
            # Low dispersion = widespread stress
            if len(stress_levels) > 1:
                stress_dispersion = 1 - np.std(stress_levels)  # Lower std = more dispersed
            else:
                stress_dispersion = 0.5
                
            return float(max(0, min(1, stress_dispersion)))
            
        except Exception:
            return 0.5
            
    async def _risk_on_off_analysis(self) -> Dict:
        """Analyze risk-on/risk-off market sentiment"""
        try:
            # Get risk-on and risk-off asset data
            risk_on_symbols = self.risk_assets["risk_on"]
            risk_off_symbols = self.risk_assets["risk_off"]
            
            all_symbols = risk_on_symbols + risk_off_symbols
            risk_data = await self._fetch_risk_data(all_symbols)
            
            if risk_data.empty:
                return {'error': 'No risk-on/off data available'}
                
            risk_analysis = {}
            
            # Analyze risk-on assets
            risk_on_performance = {}
            for symbol in risk_on_symbols:
                close_col = f"{symbol}_Close"
                if close_col in risk_data.columns:
                    price_data = risk_data[close_col]
                    momentum_5d = price_data.pct_change(5).iloc[-1]
                    momentum_20d = price_data.pct_change(20).iloc[-1]
                    
                    risk_on_performance[symbol] = {
                        'momentum_5d': float(momentum_5d),
                        'momentum_20d': float(momentum_20d)
                    }
                    
            # Analyze risk-off assets
            risk_off_performance = {}
            for symbol in risk_off_symbols:
                close_col = f"{symbol}_Close"
                if close_col in risk_data.columns:
                    price_data = risk_data[close_col]
                    momentum_5d = price_data.pct_change(5).iloc[-1]
                    momentum_20d = price_data.pct_change(20).iloc[-1]
                    
                    risk_off_performance[symbol] = {
                        'momentum_5d': float(momentum_5d),
                        'momentum_20d': float(momentum_20d)
                    }
                    
            # Calculate risk sentiment
            sentiment = self._calculate_risk_sentiment(risk_on_performance, risk_off_performance)
            
            # Risk rotation analysis
            rotation = self._analyze_risk_rotation(risk_on_performance, risk_off_performance)
            
            # Safe haven flows
            safe_haven_flows = self._analyze_safe_haven_flows(risk_off_performance)
            
            # Risk momentum
            momentum_analysis = self._analyze_risk_momentum(risk_on_performance, risk_off_performance)
            
            return {
                'risk_on_performance': risk_on_performance,
                'risk_off_performance': risk_off_performance,
                'risk_sentiment': sentiment,
                'risk_rotation': rotation,
                'safe_haven_flows': safe_haven_flows,
                'momentum_analysis': momentum_analysis,
                'risk_regime': self._determine_risk_regime(sentiment, rotation)
            }
            
        except Exception as e:
            logger.error(f"Error in risk-on/off analysis: {e}")
            return {'error': str(e)}
            
    def _calculate_risk_sentiment(self, risk_on: Dict, risk_off: Dict) -> Dict:
        """Calculate overall risk sentiment"""
        try:
            # Calculate average momentum for each category
            on_momentum_5d = np.mean([data['momentum_5d'] for data in risk_on.values()]) if risk_on else 0
            off_momentum_5d = np.mean([data['momentum_5d'] for data in risk_off.values()]) if risk_off else 0
            
            on_momentum_20d = np.mean([data['momentum_20d'] for data in risk_on.values()]) if risk_on else 0
            off_momentum_20d = np.mean([data['momentum_20d'] for data in risk_off.values()]) if risk_off else 0
            
            # Risk sentiment score
            sentiment_5d = on_momentum_5d - off_momentum_5d
            sentiment_20d = on_momentum_20d - off_momentum_20d
            
            return {
                'sentiment_5d': float(sentiment_5d),
                'sentiment_20d': float(sentiment_20d),
                'sentiment_direction': 'risk_on' if sentiment_5d > 0 else 'risk_off',
                'sentiment_strength': float(abs(sentiment_5d)),
                'sentiment_consensus': 'strong' if abs(sentiment_5d) > 0.02 else 'moderate' if abs(sentiment_5d) > 0.01 else 'weak'
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk sentiment: {e}")
            return {'sentiment_direction': 'neutral', 'sentiment_strength': 0.0}
            
    def _analyze_risk_rotation(self, risk_on: Dict, risk_off: Dict) -> Dict:
        """Analyze risk rotation patterns"""
        try:
            rotation_analysis = {
                'rotation_detected': False,
                'rotation_strength': 0.0,
                'rotation_direction': 'stable',
                'rotation_assets': []
            }
            
            # Look for rotation signals
            if risk_on and risk_off:
                # Check for divergence between short and long-term momentum
                on_short_term = np.mean([data['momentum_5d'] for data in risk_on.values()])
                on_long_term = np.mean([data['momentum_20d'] for data in risk_on.values()])
                
                off_short_term = np.mean([data['momentum_5d'] for data in risk_off.values()])
                off_long_term = np.mean([data['momentum_20d'] for data in risk_off.values()])
                
                # Rotation strength (divergence between short and long term)
                rotation_strength = abs((on_short_term - on_long_term) - (off_short_term - off_long_term))
                
                if rotation_strength > 0.01:  # Threshold for rotation detection
                    rotation_analysis['rotation_detected'] = True
                    rotation_analysis['rotation_strength'] = float(rotation_strength)
                    
                    # Determine rotation direction
                    if on_short_term > on_long_term:
                        rotation_analysis['rotation_direction'] = 'increasing_risk_on'
                    else:
                        rotation_analysis['rotation_direction'] = 'increasing_risk_off'
                        
            return rotation_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing risk rotation: {e}")
            return {'rotation_detected': False, 'rotation_strength': 0.0}
            
    def _analyze_safe_haven_flows(self, risk_off: Dict) -> Dict:
        """Analyze safe haven asset flows"""
        try:
            safe_haven_analysis = {
                'demand_level': 'normal',
                'flow_momentum': 0.0,
                'leading_safe_havens': []
            }
            
            if not risk_off:
                return safe_haven_analysis
                
            # Analyze safe haven momentum
            momentum_values = [data['momentum_5d'] for data in risk_off.values()]
            avg_momentum = np.mean(momentum_values)
            
            # Demand level classification
            if avg_momentum > 0.02:
                safe_haven_analysis['demand_level'] = 'high'
                safe_haven_analysis['flow_momentum'] = float(avg_momentum)
            elif avg_momentum > 0.01:
                safe_haven_analysis['demand_level'] = 'elevated'
                safe_haven_analysis['flow_momentum'] = float(avg_momentum)
            else:
                safe_haven_analysis['demand_level'] = 'normal'
                safe_haven_analysis['flow_momentum'] = float(avg_momentum)
                
            # Identify leading safe havens
            sorted_havens = sorted(risk_off.items(), key=lambda x: x[1]['momentum_5d'], reverse=True)
            safe_haven_analysis['leading_safe_havens'] = [symbol for symbol, _ in sorted_havens[:2]]
            
            return safe_haven_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing safe haven flows: {e}")
            return {'demand_level': 'normal', 'flow_momentum': 0.0}
            
    def _analyze_risk_momentum(self, risk_on: Dict, risk_off: Dict) -> Dict:
        """Analyze risk momentum patterns"""
        try:
            momentum_analysis = {
                'momentum_direction': 'stable',
                'momentum_acceleration': 0.0,
                'momentum_consistency': 0.0
            }
            
            if risk_on and risk_off:
                # Calculate momentum consistency
                on_momentum_std = np.std([data['momentum_5d'] for data in risk_on.values()])
                off_momentum_std = np.std([data['momentum_5d'] for data in risk_off.values()])
                
                combined_std = np.sqrt(on_momentum_std**2 + off_momentum_std**2)
                momentum_consistency = 1 / (1 + combined_std)  # Higher consistency = lower std
                
                momentum_analysis['momentum_consistency'] = float(momentum_consistency)
                
                # Determine momentum direction
                on_avg_momentum = np.mean([data['momentum_5d'] for data in risk_on.values()])
                off_avg_momentum = np.mean([data['momentum_5d'] for data in risk_off.values()])
                
                if on_avg_momentum > 0.01 and off_avg_momentum < -0.01:
                    momentum_analysis['momentum_direction'] = 'strong_risk_on'
                elif on_avg_momentum < -0.01 and off_avg_momentum > 0.01:
                    momentum_analysis['momentum_direction'] = 'strong_risk_off'
                elif abs(on_avg_momentum) < 0.005 and abs(off_avg_momentum) < 0.005:
                    momentum_analysis['momentum_direction'] = 'momentum_pause'
                else:
                    momentum_analysis['momentum_direction'] = 'mixed_momentum'
                    
                # Momentum acceleration
                momentum_acceleration = (on_avg_momentum - off_avg_momentum)
                momentum_analysis['momentum_acceleration'] = float(momentum_acceleration)
                
            return momentum_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing risk momentum: {e}")
            return {'momentum_direction': 'stable', 'momentum_acceleration': 0.0}
            
    def _determine_risk_regime(self, sentiment: Dict, rotation: Dict) -> str:
        """Determine current risk regime"""
        try:
            sentiment_direction = sentiment.get('sentiment_direction', 'neutral')
            rotation_detected = rotation.get('rotation_detected', False)
            rotation_strength = rotation.get('rotation_strength', 0)
            
            # Regime classification
            if sentiment_direction == 'risk_on' and not rotation_detected:
                return 'risk_on_regime'
            elif sentiment_direction == 'risk_off' and not rotation_detected:
                return 'risk_off_regime'
            elif rotation_detected and rotation_strength > 0.02:
                return 'risk_rotation_regime'
            else:
                return 'neutral_regime'
                
        except Exception as e:
            logger.error(f"Error determining risk regime: {e}")
            return 'unknown_regime'
            
    async def _volatility_regime_analysis(self) -> Dict:
        """Analyze current volatility regime and stress"""
        try:
            # Get volatility-related data
            vol_symbols = ["SPY", "VXX", "TLT", "GLD"]
            vol_data = await self._fetch_risk_data(vol_symbols)
            
            if vol_data.empty:
                return {'error': 'No volatility data available'}
                
            regime_analysis = {}
            
            # SPY volatility regime
            if 'SPY_Close' in vol_data.columns:
                spy_data = vol_data['SPY_Close']
                spy_returns = spy_data.pct_change().dropna()
                spy_vol_5d = spy_returns.rolling(5).std() * np.sqrt(252)
                spy_vol_20d = spy_returns.rolling(20).std() * np.sqrt(252)
                spy_vol_current = spy_vol_5d.iloc[-1]
                spy_vol_ma = spy_vol_20d.iloc[-1]
                
                regime_analysis['equity_volatility'] = {
                    'current_level': float(spy_vol_current),
                    'moving_average': float(spy_vol_ma),
                    'regime': self._determine_vol_regime(spy_vol_current),
                    'vol_trend': 'increasing' if spy_vol_current > spy_vol_ma * 1.1 else 'decreasing' if spy_vol_current < spy_vol_ma * 0.9 else 'stable'
                }
                
            # VIX-based stress analysis
            if 'VXX_Close' in vol_data.columns:
                vxx_data = vol_data['VXX_Close']
                vxx_momentum = vxx_data.pct_change(5).iloc[-1]
                vxx_level = vxx_data.iloc[-1]
                
                regime_analysis['fear_gauge'] = {
                    'current_level': float(vxx_level),
                    'momentum': float(vxx_momentum),
                    'stress_level': self._calculate_vix_stress_level(vxx_level),
                    'fear_sentiment': 'high_fear' if vxx_momentum > 0.1 else 'calming_fear' if vxx_momentum < -0.1 else 'stable_fear'
                }
                
            # Cross-asset volatility
            vol_correlations = self._calculate_cross_asset_vol_correlations(vol_data)
            
            # Volatility clustering
            clustering_analysis = self._analyze_volatility_clustering(vol_data)
            
            # Overall volatility assessment
            vol_assessment = self._assess_overall_volatility(regime_analysis, vol_correlations)
            
            return {
                'regime_analysis': regime_analysis,
                'cross_asset_correlations': vol_correlations,
                'clustering_analysis': clustering_analysis,
                'volatility_assessment': vol_assessment,
                'stress_forecast': self._forecast_volatility_stress(regime_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error in volatility regime analysis: {e}")
            return {'error': str(e)}
            
    def _determine_vol_regime(self, vol_level: float) -> str:
        """Determine volatility regime based on level"""
        try:
            if vol_level > 0.5:
                return 'crisis_regime'
            elif vol_level > 0.35:
                return 'high_regime'
            elif vol_level > 0.25:
                return 'elevated_regime'
            elif vol_level > 0.15:
                return 'normal_regime'
            else:
                return 'low_regime'
        except:
            return 'unknown'
            
    def _calculate_vix_stress_level(self, vix_level: float) -> str:
        """Calculate VIX-based stress level"""
        try:
            if vix_level > 40:
                return 'extreme_stress'
            elif vix_level > 30:
                return 'high_stress'
            elif vix_level > 25:
                return 'elevated_stress'
            elif vix_level < 15:
                return 'low_stress'
            else:
                return 'normal_stress'
        except:
            return 'unknown'
            
    def _calculate_cross_asset_vol_correlations(self, vol_data: pd.DataFrame) -> Dict:
        """Calculate cross-asset volatility correlations"""
        try:
            correlations = {}
            
            # Get return series for volatility calculation
            return_columns = [col for col in vol_data.columns if col.endswith('_Close')]
            
            for i, col1 in enumerate(return_columns):
                for col2 in return_columns[i+1:]:
                    # Calculate returns
                    returns1 = vol_data[col1].pct_change().dropna()
                    returns2 = vol_data[col2].pct_change().dropna()
                    
                    # Align data
                    aligned_data = pd.concat([returns1, returns2], axis=1).dropna()
                    
                    if len(aligned_data) > 10:
                        corr = aligned_data.corr().iloc[0, 1]
                        pair_name = f"{col1.replace('_Close', '')}_{col2.replace('_Close', '')}"
                        correlations[pair_name] = float(corr)
                        
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating cross-asset vol correlations: {e}")
            return {}
            
    def _analyze_volatility_clustering(self, vol_data: pd.DataFrame) -> Dict:
        """Analyze volatility clustering patterns"""
        try:
            clustering = {
                'clustering_detected': False,
                'clustering_strength': 0.0,
                'clustering_assets': [],
                'persistence': 0.0
            }
            
            # Use SPY as proxy for volatility clustering
            if 'SPY_Close' in vol_data.columns:
                spy_returns = vol_data['SPY_Close'].pct_change().dropna()
                spy_vol = spy_returns.rolling(10).std()
                
                # Check for volatility clustering
                high_vol_periods = spy_vol > spy_vol.quantile(0.8)
                vol_transitions = high_vol_periods.astype(int).diff()
                
                # Clustering strength (persistence of high vol periods)
                if high_vol_periods.sum() > 0:
                    clustering_strength = high_vol_periods.sum() / len(high_vol_periods)
                    clustering['clustering_detected'] = clustering_strength > 0.3
                    clustering['clustering_strength'] = float(clustering_strength)
                    clustering['clustering_assets'] = ['SPY']  # Using SPY as proxy
                    
                    # Persistence calculation
                    if len(high_vol_periods) > 10:
                        recent_high_vol = high_vol_periods.tail(10)
                        persistence = recent_high_vol.sum() / len(recent_high_vol)
                        clustering['persistence'] = float(persistence)
                        
            return clustering
            
        except Exception as e:
            logger.error(f"Error analyzing volatility clustering: {e}")
            return {'clustering_detected': False, 'clustering_strength': 0.0}
            
    def _assess_overall_volatility(self, regime_analysis: Dict, correlations: Dict) -> Dict:
        """Assess overall volatility conditions"""
        try:
            assessment = {
                'overall_regime': 'normal',
                'stress_indicators': [],
                'regime_stability': 'stable',
                'volatility_health': 'healthy'
            }
                
            # Determine overall regime
            if 'equity_volatility' in regime_analysis:
                vol_regime = regime_analysis['equity_volatility']['regime']
                assessment['overall_regime'] = vol_regime
                
            if 'fear_gauge' in regime_analysis:
                fear_level = regime_analysis['fear_gauge']['stress_level']
                if fear_level in ['extreme_stress', 'high_stress']:
                    assessment['stress_indicators'].append('high_fear_level')
                    
            # Check regime stability
            if 'equity_volatility' in regime_analysis:
                vol_trend = regime_analysis['equity_volatility']['vol_trend']
                assessment['regime_stability'] = 'unstable' if vol_trend == 'increasing' else 'stable'
                
            # Volatility health assessment
            if len(assessment['stress_indicators']) > 2:
                assessment['volatility_health'] = 'stressed'
            elif len(assessment['stress_indicators']) > 0:
                assessment['volatility_health'] = 'concerned'
            else:
                assessment['volatility_health'] = 'healthy'
                
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing overall volatility: {e}")
            return {'overall_regime': 'normal', 'stress_indicators': []}
            
    def _forecast_volatility_stress(self, regime_analysis: Dict) -> Dict:
        """Forecast potential volatility stress"""
        try:
            forecast = {
                'stress_probability': 0.0,
                'time_horizon': '1_week',
                'key_indicators': [],
                'forecast_confidence': 0.0
            }
            
            stress_indicators = []
            
            # Analyze each indicator for stress signals
            for indicator_name, data in regime_analysis.items():
                if isinstance(data, dict):
                    # Check for escalating stress patterns
                    if indicator_name == 'equity_volatility':
                        if data.get('vol_trend') == 'increasing':
                            stress_indicators.append('volatility_increasing')
                    elif indicator_name == 'fear_gauge':
                        if data.get('fear_sentiment') == 'high_fear':
                            stress_indicators.append('fear_escalation')
                            
            # Calculate stress probability
            forecast['stress_probability'] = min(1.0, len(stress_indicators) / 4)  # Max 4 indicators
            forecast['key_indicators'] = stress_indicators
            forecast['forecast_confidence'] = max(0.3, 1 - (len(stress_indicators) / 10))  # Higher confidence with fewer stress indicators
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error forecasting volatility stress: {e}")
            return {'stress_probability': 0.0, 'forecast_confidence': 0.0}
            
    async def _correlation_breakdown_analysis(self) -> Dict:
        """Analyze correlation breakdown and market structure changes"""
        try:
            # Get correlation-related data
            corr_symbols = ["SPY", "QQQ", "TLT", "GLD", "DXY"]
            corr_data = await self._fetch_risk_data(corr_symbols)
            
            if corr_data.empty:
                return {'error': 'No correlation data available'}
                
            breakdown_analysis = {}
            
            # Calculate correlations
            correlation_matrix = {}
            return_columns = [col for col in corr_data.columns if col.endswith('_Close')]
            
            for i, col1 in enumerate(return_columns):
                for col2 in return_columns[i+1:]:
                    # Calculate rolling correlations
                    returns1 = corr_data[col1].pct_change().dropna()
                    returns2 = corr_data[col2].pct_change().dropna()
                    
                    aligned_data = pd.concat([returns1, returns2], axis=1).dropna()
                    
                    if len(aligned_data) > 20:
                        rolling_corr = aligned_data.rolling(20).corr().iloc[:, 1]
                        current_corr = rolling_corr.iloc[-1]
                        avg_corr = rolling_corr.mean()
                        corr_std = rolling_corr.std()
                        
                        asset1 = col1.replace('_Close', '')
                        asset2 = col2.replace('_Close', '')
                        pair_name = f"{asset1}_{asset2}"
                        
                        correlation_matrix[pair_name] = {
                            'current_correlation': float(current_corr),
                            'average_correlation': float(avg_corr),
                            'correlation_volatility': float(corr_std),
                            'correlation_regime': self._determine_corr_regime(current_corr, avg_corr),
                            'breakdown_risk': float(max(0, (current_corr - avg_corr) / corr_std)) if corr_std > 0 else 0
                        }
                        
            # Overall correlation health
            health_metrics = self._assess_correlation_health(correlation_matrix)
            
            # Correlation clustering
            clustering = self._analyze_correlation_clustering(correlation_matrix)
            
            # Market structure changes
            structure_changes = self._detect_structure_changes(correlation_matrix)
            
            return {
                'correlation_matrix': correlation_matrix,
                'health_metrics': health_metrics,
                'correlation_clustering': clustering,
                'structure_changes': structure_changes,
                'systemic_correlation_risk': self._calculate_systemic_correlation_risk(correlation_matrix)
            }
            
        except Exception as e:
            logger.error(f"Error in correlation breakdown analysis: {e}")
            return {'error': str(e)}
            
    def _determine_corr_regime(self, current_corr: float, avg_corr: float) -> str:
        """Determine correlation regime"""
        try:
            if current_corr > avg_corr + 0.2:
                return 'high_correlation_regime'
            elif current_corr < avg_corr - 0.2:
                return 'low_correlation_regime'
            elif current_corr > 0.7:
                return 'elevated_correlation'
            elif current_corr < 0.3:
                return 'decorrelated_regime'
            else:
                return 'normal_correlation'
        except:
            return 'unknown'
            
    def _assess_correlation_health(self, matrix: Dict) -> Dict:
        """Assess overall correlation health"""
        try:
            health = {
                'health_score': 0.5,
                'unhealthy_pairs': [],
                'correlation_stress': 'normal'
            }
            
            if not matrix:
                return health
                
            health_scores = []
            unhealthy_pairs = []
            
            for pair, data in matrix.items():
                # Calculate health score for pair
                corr_vol = data.get('correlation_volatility', 0)
                breakdown_risk = data.get('breakdown_risk', 0)
                
                pair_health = 1 - (corr_vol * 0.5 + breakdown_risk * 0.5)
                health_scores.append(pair_health)
                
                if pair_health < 0.5:
                    unhealthy_pairs.append(pair)
                    
            health['health_score'] = float(np.mean(health_scores)) if health_scores else 0.5
            health['unhealthy_pairs'] = unhealthy_pairs
            
            # Stress level
            if health['health_score'] > 0.7:
                health['correlation_stress'] = 'low'
            elif health['health_score'] > 0.4:
                health['correlation_stress'] = 'moderate'
            else:
                health['correlation_stress'] = 'high'
                
            return health
            
        except Exception as e:
            logger.error(f"Error assessing correlation health: {e}")
            return {'health_score': 0.5, 'correlation_stress': 'normal'}
            
    def _analyze_correlation_clustering(self, matrix: Dict) -> Dict:
        """Analyze correlation clustering patterns"""
        try:
            clustering = {
                'clustering_detected': False,
                'high_corr_clusters': [],
                'correlation_clusters': []
            }
            
            # Identify high correlation pairs
            high_corr_pairs = []
            for pair, data in matrix.items():
                if data.get('current_correlation', 0) > 0.7:
                    high_corr_pairs.append(pair)
                    
            if len(high_corr_pairs) >= 3:
                clustering['clustering_detected'] = True
                clustering['high_corr_clusters'] = high_corr_pairs
                
                # Analyze cluster structure
                cluster_assets = set()
                for pair in high_corr_pairs:
                    assets = pair.split('_')
                    cluster_assets.update(assets)
                    
                clustering['correlation_clusters'] = list(cluster_assets)
                
            return clustering
            
        except Exception as e:
            logger.error(f"Error analyzing correlation clustering: {e}")
            return {'clustering_detected': False, 'high_corr_clusters': []}
            
    def _detect_structure_changes(self, matrix: Dict) -> Dict:
        """Detect market structure changes"""
        try:
            changes = {
                'structure_changes_detected': False,
                'change_indicators': [],
                'regime_shift_probability': 0.0
            }
            
            change_indicators = []
            
            for pair, data in matrix.items():
                breakdown_risk = data.get('breakdown_risk', 0)
                if breakdown_risk > 1.5:  # High breakdown risk
                    change_indicators.append(f'{pair}_breakdown_risk')
                    
                corr_regime = data.get('correlation_regime', 'normal_correlation')
                if corr_regime in ['high_correlation_regime', 'low_correlation_regime']:
                    change_indicators.append(f'{pair}_{corr_regime}')
                    
            if change_indicators:
                changes['structure_changes_detected'] = True
                changes['change_indicators'] = change_indicators
                changes['regime_shift_probability'] = min(1.0, len(change_indicators) / 10)
                
            return changes
            
        except Exception as e:
            logger.error(f"Error detecting structure changes: {e}")
            return {'structure_changes_detected': False, 'change_indicators': []}
            
    def _calculate_systemic_correlation_risk(self, matrix: Dict) -> float:
        """Calculate systemic correlation risk score"""
        try:
            if not matrix:
                return 0.5
                
            risk_scores = []
            
            for data in matrix.values():
                breakdown_risk = data.get('breakdown_risk', 0)
                correlation_vol = data.get('correlation_volatility', 0)
                
                # Higher breakdown risk and volatility = higher systemic risk
                risk_score = min(1.0, breakdown_risk * 0.6 + correlation_vol * 2)
                risk_scores.append(risk_score)
                
            return float(np.mean(risk_scores)) if risk_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating systemic correlation risk: {e}")
            return 0.5
            
    async def get_risk_pulse(self) -> Dict:
        """Get comprehensive risk analysis"""
        try:
            # Run all risk analyses in parallel
            risk_tasks = [
                self._systemic_risk_analysis(),
                self._stress_indicators_analysis(),
                self._risk_on_off_analysis(),
                self._volatility_regime_analysis(),
                self._correlation_breakdown_analysis()
            ]
            
            results = await asyncio.gather(*risk_tasks, return_exceptions=True)
            (
                systemic_risk, stress_indicators,
                risk_on_off, vol_regime,
                corr_breakdown
            ) = results
            
            # Calculate overall Risk Momentum Score (RMS)
            rms_components = []
            
            for result in results:
                if isinstance(result, dict) and not result.get('error'):
                    # Extract numeric scores from each analysis
                    score = self._extract_risk_score(result)
                    if score is not None:
                        rms_components.append(score)
                        
            if rms_components:
                rms_score = np.mean(rms_components)
                rms_volatility = np.std(rms_components)
                
                # Classify risk state
                if rms_score > 0.7:
                    risk_state = 'extreme_stress'
                elif rms_score > 0.5:
                    risk_state = 'high_stress'
                elif rms_score > 0.3:
                    risk_state = 'moderate_stress'
                elif rms_score < 0.1:
                    risk_state = 'low_stress'
                else:
                    risk_state = 'normal_stress'
                    
                return {
                    'risk_momentum_score': rms_score,
                    'rms_volatility': rms_volatility,
                    'risk_state': risk_state,
                    'analysis_breakdown': {
                        'systemic_risk': systemic_risk,
                        'stress_indicators': stress_indicators,
                        'risk_on_off': risk_on_off,
                        'volatility_regime': vol_regime,
                        'correlation_breakdown': corr_breakdown
                    },
                    'timestamp': datetime.utcnow(),
                    'confidence': min(1.0, 1 - (rms_volatility / 2))
                }
            else:
                return {'error': 'Unable to calculate risk momentum score'}
                
        except Exception as e:
            logger.error(f"Error getting risk pulse: {e}")
            return {'error': str(e)}
            
    def _extract_risk_score(self, analysis_result: Dict) -> Optional[float]:
        """Extract numeric risk score from analysis result"""
        try:
            if 'overall_risk_score' in analysis_result:
                return analysis_result['overall_risk_score']
            elif 'overall_stress_level' in analysis_result:
                stress_level = analysis_result['overall_stress_level']
                if stress_level == 'extreme_stress':
                    return 0.9
                elif stress_level == 'high_stress':
                    return 0.7
                elif stress_level == 'moderate_stress':
                    return 0.5
                elif stress_level == 'low_stress':
                    return 0.2
                else:
                    return 0.3
            elif 'sentiment' in analysis_result and 'sentiment_strength' in analysis_result['sentiment']:
                # Invert risk sentiment (risk_off = higher risk score)
                sentiment_strength = analysis_result['sentiment']['sentiment_strength']
                if analysis_result['sentiment']['sentiment_direction'] == 'risk_off':
                    return min(1.0, sentiment_strength + 0.5)
                else:
                    return max(0.0, 0.5 - sentiment_strength)
            elif 'volatility_assessment' in analysis_result:
                vol_assessment = analysis_result['volatility_assessment']
                if vol_assessment.get('volatility_health') == 'stressed':
                    return 0.8
                elif vol_assessment.get('volatility_health') == 'concerned':
                    return 0.6
                else:
                    return 0.3
            elif 'systemic_correlation_risk' in analysis_result:
                return analysis_result['systemic_correlation_risk']
            else:
                return None
                
        except Exception:
            return None
            
    async def store_risk_data(self, risk_data: Dict):
        """Store risk metrics in time-series database"""
        try:
            if self.db_manager and 'timestamp' in risk_data:
                # Store Risk Momentum Score
                await self.db_manager.influxdb_client.write_points(
                    database='market_pulse',
                    measurement='risk_metrics',
                    tags={
                        'engine': 'risk_pulse',
                        'state': risk_data.get('risk_state', 'unknown')
                    },
                    fields={
                        'rms_score': float(risk_data.get('risk_momentum_score', 0)),
                        'rms_volatility': float(risk_data.get('rms_volatility', 0)),
                        'confidence': float(risk_data.get('confidence', 0))
                    },
                    time=risk_data['timestamp']
                )
                
                # Store component scores
                for component_name, analysis in risk_data.get('analysis_breakdown', {}).items():
                    if isinstance(analysis, dict):
                        score = self._extract_risk_score(analysis)
                        if score is not None:
                            await self.db_manager.influxdb_client.write_points(
                                database='market_pulse',
                                measurement='risk_components',
                                tags={
                                    'component': component_name,
                                    'engine': 'risk_pulse'
                                },
                                fields={'component_score': float(score)},
                                time=risk_data['timestamp']
                            )
                            
            logger.debug("Risk data stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing risk data: {e}")
            
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
                'cache_size': len(self.risk_cache),
                'models_loaded': len(self.models),
                'systemic_indicators': len(self.systemic_indicators),
                'risk_asset_pairs': len(self.risk_assets),
                'health_score': self._calculate_engine_health()
            }
            
        except Exception as e:
            logger.error(f"Error getting risk engine status: {e}")
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
            total_data_sources = len(self.systemic_indicators)
            health_factors.append(min(1.0, total_data_sources / 10))
            
            return np.mean(health_factors) if health_factors else 0.0
            
        except Exception:
            return 0.0