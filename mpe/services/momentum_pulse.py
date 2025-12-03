"""
Momentum Pulse Engine - Price Momentum & Trend Detection
Real-time tracking of price momentum, trend strength, and acceleration across asset classes
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

class MomentumPulseEngine:
    """Momentum Trend Monitor - Detecting price momentum and trend acceleration"""
    
    def __init__(self):
        self.name = "Momentum Pulse Engine"
        self.version = "1.0.0"
        self.status = "initialized"
        self.last_update = None
        self.momentum_cache = {}
        self.models = {}
        self.scaler = StandardScaler()
        
        # Momentum tracking models
        self.momentum_models = {
            "price_momentum": self._price_momentum_analysis,
            "trend_strength": self._trend_strength_analysis,
            "momentum_acceleration": self._momentum_acceleration_analysis,
            "sector_momentum": self._sector_momentum_analysis,
            "cross_asset_momentum": self._cross_asset_momentum_analysis
        }
        
        # Momentum indicators configuration
        self.momentum_indicators = {
            "rsi": "Relative Strength Index",
            "macd": "Moving Average Convergence Divergence",
            "stochastic": "Stochastic Oscillator",
            "williams_r": "Williams %R",
            "roc": "Rate of Change",
            "mfi": "Money Flow Index"
        }
        
        # Asset classes for momentum analysis
        self.asset_classes = {
            "equities": {
                "symbols": ["SPY", "QQQ", "IWM", "VTI", "VOO"],
                "characteristics": "Equity market momentum patterns"
            },
            "growth_stocks": {
                "symbols": ["QQQ", "TQQQ", "ARKK", "VUG"],
                "characteristics": "High-beta growth momentum"
            },
            "value_stocks": {
                "symbols": ["VTV", "IWD", "VBR"],
                "characteristics": "Value stock momentum patterns"
            },
            "international": {
                "symbols": ["VEA", "VWO", "IEFA", "EEM"],
                "characteristics": "International equity momentum"
            },
            "bonds": {
                "symbols": ["TLT", "IEF", "SHY", "AGG", "BND"],
                "characteristics": "Fixed income momentum"
            },
            "commodities": {
                "symbols": ["GLD", "SLV", "DBA", "USO"],
                "characteristics": "Commodity momentum patterns"
            }
        }
        
        # Sector momentum tracking
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
        
        # Momentum thresholds
        self.momentum_thresholds = {
            "strong_momentum": 0.05,     # 5% momentum over period
            "momentum_breakdown": -0.03, # -3% momentum breakdown
            "acceleration": 0.02,        # 2% acceleration
            "momentum_divergence": 0.03, # 3% divergence threshold
            "trend_strength": 0.7        # 70% trend strength threshold
        }
        
        # Trend classification
        self.trend_classes = {
            "strong_uptrend": {"momentum_range": [0.03, 1.0], "characteristics": "Bullish trend with strong momentum"},
            "uptrend": {"momentum_range": [0.01, 0.03], "characteristics": "Bullish trend with moderate momentum"},
            "sideways": {"momentum_range": [-0.01, 0.01], "characteristics": "Sideways/consolidation"},
            "downtrend": {"momentum_range": [-0.03, -0.01], "characteristics": "Bearish trend with moderate momentum"},
            "strong_downtrend": {"momentum_range": [-1.0, -0.03], "characteristics": "Bearish trend with strong momentum"}
        }
        
        # Momentum clustering
        self.momentum_clusters = {
            "leading_momentum": "Assets leading the market higher",
            "lagging_momentum": "Assets lagging the market",
            "contrarian_momentum": "Assets moving opposite to trend",
            "acceleration_momentum": "Assets showing increasing momentum",
            "deceleration_momentum": "Assets showing decreasing momentum"
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
            logger.info("Momentum Pulse Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Momentum Pulse Engine: {e}")
            self.status = "error"
            
    def _initialize_models(self):
        """Initialize ML models for momentum prediction"""
        try:
            # Random Forest for momentum prediction
            self.models['momentum_predictor'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Random Forest for trend classification
            self.models['trend_classifier'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=8
            )
            
            # K-means for momentum clustering
            self.models['momentum_clusters'] = KMeans(
                n_clusters=4,
                random_state=42
            )
            
            # Scaler for momentum feature normalization
            self.scaler = StandardScaler()
            
            logger.info("Momentum prediction models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize momentum models: {e}")
            
    async def _fetch_momentum_data(self, symbols: List[str], period: str = "6mo") -> pd.DataFrame:
        """Fetch price data for momentum analysis"""
        try:
            momentum_data = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    if not hist.empty and len(hist) > 30:
                        momentum_data[symbol] = hist
                except Exception as e:
                    logger.warning(f"Error fetching momentum data for {symbol}: {e}")
                    continue
                    
            if not momentum_data:
                return pd.DataFrame()
                
            # Combine all data
            combined_data = pd.DataFrame()
            
            for symbol, data in momentum_data.items():
                for col in data.columns:
                    combined_data[f"{symbol}_{col}"] = data[col]
                    
            return combined_data
            
        except Exception as e:
            logger.error(f"Error fetching momentum data: {e}")
            return pd.DataFrame()
            
    def _calculate_momentum_indicators(self, price_data: pd.Series) -> Dict:
        """Calculate comprehensive momentum indicators"""
        try:
            if price_data.empty or len(price_data) < 20:
                return {}
                
            indicators = {}
            
            # RSI calculation
            try:
                rsi = talib.RSI(price_data.values, timeperiod=14)
                indicators['rsi'] = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
            except:
                indicators['rsi'] = 50.0
                
            # MACD calculation
            try:
                macd, macdsignal, macdhist = talib.MACD(price_data.values)
                indicators['macd'] = float(macd[-1]) if not np.isnan(macd[-1]) else 0.0
                indicators['macd_signal'] = float(macdsignal[-1]) if not np.isnan(macdsignal[-1]) else 0.0
                indicators['macd_histogram'] = float(macdhist[-1]) if not np.isnan(macdhist[-1]) else 0.0
            except:
                indicators['macd'] = 0.0
                indicators['macd_signal'] = 0.0
                indicators['macd_histogram'] = 0.0
                
            # Stochastic calculation
            try:
                high_col = price_data.name.replace('Close', 'High')
                low_col = price_data.name.replace('Close', 'Low')
                # This would need high/low data, simplified for now
                indicators['stoch'] = 50.0  # Placeholder
            except:
                indicators['stoch'] = 50.0
                
            # Rate of Change (ROC)
            try:
                roc = talib.ROC(price_data.values, timeperiod=10)
                indicators['roc'] = float(roc[-1]) if not np.isnan(roc[-1]) else 0.0
            except:
                indicators['roc'] = 0.0
                
            # Williams %R
            try:
                willr = talib.WILLR(price_data.values, high=price_data.values, low=price_data.values, timeperiod=14)
                indicators['williams_r'] = float(willr[-1]) if not np.isnan(willr[-1]) else -50.0
            except:
                indicators['williams_r'] = -50.0
                
            # Price momentum over different periods
            try:
                momentum_5 = price_data.pct_change(5).iloc[-1]
                momentum_10 = price_data.pct_change(10).iloc[-1]
                momentum_20 = price_data.pct_change(20).iloc[-1]
                
                indicators['momentum_5d'] = float(momentum_5) if not np.isnan(momentum_5) else 0.0
                indicators['momentum_10d'] = float(momentum_10) if not np.isnan(momentum_10) else 0.0
                indicators['momentum_20d'] = float(momentum_20) if not np.isnan(momentum_20) else 0.0
            except:
                indicators['momentum_5d'] = 0.0
                indicators['momentum_10d'] = 0.0
                indicators['momentum_20d'] = 0.0
                
            # Trend strength indicators
            try:
                # Average Directional Index (ADX)
                adx = talib.ADX(price_data.values, high=price_data.values, low=price_data.values, timeperiod=14)
                indicators['adx'] = float(adx[-1]) if not np.isnan(adx[-1]) else 25.0
                
                # Parabolic SAR
                sar = talib.SAR(price_data.values, high=price_data.values, low=price_data.values)
                indicators['sar_position'] = float(1 if price_data.iloc[-1] > sar[-1] else -1) if not np.isnan(sar[-1]) else 0.0
            except:
                indicators['adx'] = 25.0
                indicators['sar_position'] = 0.0
                
            # Momentum divergence analysis
            try:
                # Price vs momentum divergence
                price_change = price_data.pct_change(5).iloc[-1]
                momentum_change = indicators['momentum_5d']
                
                if abs(price_change) > 0.01 and abs(momentum_change) > 0.01:
                    divergence = price_change - momentum_change
                    indicators['momentum_divergence'] = float(divergence)
                    indicators['divergence_strength'] = 'strong' if abs(divergence) > 0.02 else 'moderate' if abs(divergence) > 0.01 else 'weak'
                else:
                    indicators['momentum_divergence'] = 0.0
                    indicators['divergence_strength'] = 'none'
            except:
                indicators['momentum_divergence'] = 0.0
                indicators['divergence_strength'] = 'none'
                
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return {}
            
    async def _price_momentum_analysis(self) -> Dict:
        """Analyze price momentum across different asset classes"""
        try:
            price_momentum_data = {}
            
            # Analyze momentum for each asset class
            for class_name, config in self.asset_classes.items():
                class_momentum_data = []
                
                for symbol in config["symbols"]:
                    try:
                        # Get price data
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="3mo")
                        if not hist.empty:
                            # Calculate momentum indicators
                            momentum_indicators = self._calculate_momentum_indicators(hist['Close'])
                            
                            if momentum_indicators:
                                momentum_data = {
                                    'symbol': symbol,
                                    'momentum_5d': momentum_indicators['momentum_5d'],
                                    'momentum_10d': momentum_indicators['momentum_10d'],
                                    'momentum_20d': momentum_indicators['momentum_20d'],
                                    'rsi': momentum_indicators['rsi'],
                                    'macd': momentum_indicators['macd'],
                                    'roc': momentum_indicators['roc'],
                                    'adx': momentum_indicators['adx'],
                                    'trend_strength': self._calculate_trend_strength(momentum_indicators),
                                    'momentum_class': self._classify_momentum(momentum_indicators)
                                }
                                class_momentum_data.append(momentum_data)
                    except Exception as e:
                        logger.warning(f"Error analyzing momentum for {symbol}: {e}")
                        continue
                        
                if class_momentum_data:
                    # Aggregate class metrics
                    avg_momentum_5d = np.mean([d['momentum_5d'] for d in class_momentum_data])
                    avg_momentum_10d = np.mean([d['momentum_10d'] for d in class_momentum_data])
                    momentum_consensus = self._calculate_momentum_consensus(class_momentum_data)
                    
                    price_momentum_data[class_name] = {
                        'class_assets': class_momentum_data,
                        'avg_momentum_5d': float(avg_momentum_5d),
                        'avg_momentum_10d': float(avg_momentum_10d),
                        'momentum_consensus': momentum_consensus,
                        'class_characteristics': config["characteristics"],
                        'strongest_momentum_asset': max(class_momentum_data, key=lambda x: x['momentum_5d'])['symbol'],
                        'weakest_momentum_asset': min(class_momentum_data, key=lambda x: x['momentum_5d'])['symbol']
                    }
                    
            if not price_momentum_data:
                return {'error': 'No price momentum data available'}
                
            # Overall momentum assessment
            momentum_assessment = self._assess_overall_momentum(price_momentum_data)
            
            # Momentum leader identification
            momentum_leaders = self._identify_momentum_leaders(price_momentum_data)
            
            # Cross-asset momentum correlation
            momentum_correlation = self._analyze_cross_asset_momentum_correlation(price_momentum_data)
            
            return {
                'price_momentum_data': price_momentum_data,
                'momentum_assessment': momentum_assessment,
                'momentum_leaders': momentum_leaders,
                'momentum_correlation': momentum_correlation
            }
            
        except Exception as e:
            logger.error(f"Error in price momentum analysis: {e}")
            return {'error': str(e)}
            
    def _calculate_trend_strength(self, indicators: Dict) -> float:
        """Calculate trend strength score"""
        try:
            # Combine multiple strength indicators
            rsi_strength = abs(indicators.get('rsi', 50) - 50) / 50  # Distance from neutral
            adx_strength = min(1.0, indicators.get('adx', 25) / 50)  # ADX normalized
            macd_strength = abs(indicators.get('macd', 0)) * 100  # MACD magnitude
            momentum_strength = abs(indicators.get('momentum_10d', 0)) * 10  # Momentum magnitude
            
            # Combine with weights
            trend_strength = (
                rsi_strength * 0.2 +
                adx_strength * 0.3 +
                min(1.0, macd_strength) * 0.3 +
                min(1.0, momentum_strength) * 0.2
            )
            
            return float(min(1.0, trend_strength))
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.5
            
    def _classify_momentum(self, indicators: Dict) -> str:
        """Classify momentum type"""
        try:
            momentum_5d = indicators.get('momentum_5d', 0)
            momentum_10d = indicators.get('momentum_10d', 0)
            rsi = indicators.get('rsi', 50)
            
            # Classification logic
            if momentum_5d > 0.03 and momentum_10d > 0.02:
                if rsi > 70:
                    return 'strong_bullish_overbought'
                else:
                    return 'strong_bullish'
            elif momentum_5d > 0.01:
                return 'moderate_bullish'
            elif momentum_5d < -0.03 and momentum_10d < -0.02:
                if rsi < 30:
                    return 'strong_bearish_oversold'
                else:
                    return 'strong_bearish'
            elif momentum_5d < -0.01:
                return 'moderate_bearish'
            else:
                return 'sideways'
                
        except Exception as e:
            logger.error(f"Error classifying momentum: {e}")
            return 'unknown'
            
    def _calculate_momentum_consensus(self, momentum_data: List[Dict]) -> Dict:
        """Calculate momentum consensus within asset class"""
        try:
            if not momentum_data:
                return {'consensus': 'neutral', 'strength': 0.0}
                
            # Count momentum directions
            bullish_count = sum(1 for d in momentum_data if d['momentum_5d'] > 0)
            bearish_count = sum(1 for d in momentum_data if d['momentum_5d'] < 0)
            
            total_count = len(momentum_data)
            bullish_ratio = bullish_count / total_count
            bearish_ratio = bearish_count / total_count
            
            # Calculate consensus strength
            if bullish_ratio > 0.7:
                consensus = 'strong_bullish'
                strength = bullish_ratio
            elif bullish_ratio > 0.5:
                consensus = 'moderate_bullish'
                strength = bullish_ratio
            elif bearish_ratio > 0.7:
                consensus = 'strong_bearish'
                strength = bearish_ratio
            elif bearish_ratio > 0.5:
                consensus = 'moderate_bearish'
                strength = bearish_ratio
            else:
                consensus = 'neutral'
                strength = 0.5
                
            return {
                'consensus': consensus,
                'strength': float(strength),
                'bullish_ratio': float(bullish_ratio),
                'bearish_ratio': float(bearish_ratio)
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum consensus: {e}")
            return {'consensus': 'neutral', 'strength': 0.0}
            
    def _assess_overall_momentum(self, momentum_data: Dict) -> Dict:
        """Assess overall market momentum"""
        try:
            assessment = {
                'overall_momentum': 'neutral',
                'momentum_strength': 0.0,
                'momentum_distribution': {},
                'leading_sectors': [],
                'lagging_sectors': []
            }
            
            # Aggregate momentum across asset classes
            all_momentum_values = []
            class_momentum_scores = {}
            
            for class_name, data in momentum_data.items():
                avg_momentum = data['avg_momentum_5d']
                all_momentum_values.append(avg_momentum)
                class_momentum_scores[class_name] = avg_momentum
                
            # Overall momentum assessment
            if all_momentum_values:
                overall_momentum = np.mean(all_momentum_values)
                momentum_volatility = np.std(all_momentum_values)
                
                assessment['overall_momentum'] = self._classify_overall_momentum(overall_momentum)
                assessment['momentum_strength'] = float(abs(overall_momentum))
                assessment['momentum_consistency'] = float(1 - momentum_volatility)  # Higher consistency = lower volatility
                
            # Leading and lagging sectors
            sorted_classes = sorted(class_momentum_scores.items(), key=lambda x: x[1], reverse=True)
            assessment['leading_sectors'] = [item[0] for item in sorted_classes[:2]]
            assessment['lagging_sectors'] = [item[0] for item in sorted_classes[-2:]]
            
            # Momentum distribution
            positive_momentum = sum(1 for score in class_momentum_scores.values() if score > 0)
            negative_momentum = len(class_momentum_scores) - positive_momentum
            
            assessment['momentum_distribution'] = {
                'positive_momentum_classes': positive_momentum,
                'negative_momentum_classes': negative_momentum,
                'momentum_breadth': positive_momentum / len(class_momentum_scores) if class_momentum_scores else 0
            }
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing overall momentum: {e}")
            return {'overall_momentum': 'neutral', 'momentum_strength': 0.0}
            
    def _classify_overall_momentum(self, momentum: float) -> str:
        """Classify overall momentum direction"""
        try:
            if momentum > 0.02:
                return 'strong_bullish'
            elif momentum > 0.01:
                return 'bullish'
            elif momentum < -0.02:
                return 'strong_bearish'
            elif momentum < -0.01:
                return 'bearish'
            else:
                return 'neutral'
        except:
            return 'neutral'
            
    def _identify_momentum_leaders(self, momentum_data: Dict) -> Dict:
        """Identify momentum leaders and laggards"""
        try:
            leaders = {
                'strongest_momentum': None,
                'momentum_leaders': [],
                'momentum_laggards': [],
                'momentum_divergence': 0.0
            }
            
            # Collect all momentum values
            all_momentum_data = []
            for class_name, data in momentum_data.items():
                for asset in data['class_assets']:
                    all_momentum_data.append({
                        'asset': asset['symbol'],
                        'class': class_name,
                        'momentum_5d': asset['momentum_5d'],
                        'momentum_class': asset['momentum_class']
                    })
                    
            if not all_momentum_data:
                return leaders
                
            # Sort by momentum
            sorted_momentum = sorted(all_momentum_data, key=lambda x: x['momentum_5d'], reverse=True)
            
            # Identify leaders and laggards
            leaders['strongest_momentum'] = sorted_momentum[0]
            leaders['momentum_leaders'] = sorted_momentum[:3]
            leaders['momentum_laggards'] = sorted_momentum[-3:]
            
            # Calculate momentum divergence
            momentum_range = sorted_momentum[0]['momentum_5d'] - sorted_momentum[-1]['momentum_5d']
            leaders['momentum_divergence'] = float(momentum_range)
            
            return leaders
            
        except Exception as e:
            logger.error(f"Error identifying momentum leaders: {e}")
            return {'momentum_leaders': [], 'momentum_laggards': []}
            
    def _analyze_cross_asset_momentum_correlation(self, momentum_data: Dict) -> Dict:
        """Analyze momentum correlation across asset classes"""
        try:
            correlation_analysis = {
                'momentum_correlation': 0.0,
                'correlation_regime': 'normal',
                'synchronized_momentum': False
            }
            
            # Calculate momentum correlation
            momentum_values = [data['avg_momentum_5d'] for data in momentum_data.values()]
            
            if len(momentum_values) > 1:
                # Use standard deviation as proxy for correlation
                momentum_std = np.std(momentum_values)
                momentum_mean = np.mean(momentum_values)
                
                # Higher standard deviation = lower correlation
                correlation_proxy = 1 / (1 + momentum_std * 10) if momentum_std > 0 else 1
                correlation_analysis['momentum_correlation'] = float(correlation_proxy)
                
                # Correlation regime
                if correlation_proxy > 0.8:
                    correlation_analysis['correlation_regime'] = 'high_correlation'
                elif correlation_proxy < 0.3:
                    correlation_analysis['correlation_regime'] = 'low_correlation'
                else:
                    correlation_analysis['correlation_regime'] = 'normal_correlation'
                    
                # Synchronized momentum detection
                positive_momentum = sum(1 for m in momentum_values if m > 0)
                negative_momentum = len(momentum_values) - positive_momentum
                
                if positive_momentum >= len(momentum_values) * 0.8 or negative_momentum >= len(momentum_values) * 0.8:
                    correlation_analysis['synchronized_momentum'] = True
                    
            return correlation_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing cross-asset momentum correlation: {e}")
            return {'momentum_correlation': 0.0, 'correlation_regime': 'normal'}
            
    async def _trend_strength_analysis(self) -> Dict:
        """Analyze trend strength and persistence"""
        try:
            # Analyze trend strength across major indices
            trend_symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
            trend_data = await self._fetch_momentum_data(trend_symbols)
            
            if trend_data.empty:
                return {'error': 'No trend data available'}
                
            trend_analysis = {}
            
            for symbol in trend_symbols:
                close_col = f"{symbol}_Close"
                if close_col in trend_data.columns:
                    price_data = trend_data[close_col].dropna()
                    
                    if len(price_data) > 20:
                        # Calculate trend strength indicators
                        trend_metrics = self._calculate_trend_metrics(price_data)
                        trend_analysis[symbol] = trend_metrics
                        
            if not trend_analysis:
                return {'error': 'No trend analysis possible'}
                
            # Trend strength assessment
            strength_assessment = self._assess_trend_strength(trend_analysis)
            
            # Trend persistence analysis
            persistence_analysis = self._analyze_trend_persistence(trend_analysis)
            
            # Trend acceleration/deceleration
            acceleration_analysis = self._analyze_trend_acceleration(trend_analysis)
            
            return {
                'trend_analysis': trend_analysis,
                'strength_assessment': strength_assessment,
                'persistence_analysis': persistence_analysis,
                'acceleration_analysis': acceleration_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in trend strength analysis: {e}")
            return {'error': str(e)}
            
    def _calculate_trend_metrics(self, price_data: pd.Series) -> Dict:
        """Calculate comprehensive trend metrics"""
        try:
            if len(price_data) < 20:
                return {}
                
            metrics = {}
            
            # Linear trend slope
            try:
                x = np.arange(len(price_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, price_data.values)
                metrics['trend_slope'] = float(slope)
                metrics['trend_r_squared'] = float(r_value**2)
            except:
                metrics['trend_slope'] = 0.0
                metrics['trend_r_squared'] = 0.0
                
            # Moving average trends
            ma_20 = price_data.rolling(20).mean()
            ma_50 = price_data.rolling(50).mean()
            
            metrics['ma_20_trend'] = float(ma_20.iloc[-1] - ma_20.iloc[-5]) if len(ma_20) > 5 else 0.0
            metrics['ma_50_trend'] = float(ma_50.iloc[-1] - ma_50.iloc[-10]) if len(ma_50) > 10 else 0.0
            
            # Trend direction
            current_price = price_data.iloc[-1]
            ma_20_current = ma_20.iloc[-1] if not np.isnan(ma_20.iloc[-1]) else current_price
            ma_50_current = ma_50.iloc[-1] if not np.isnan(ma_50.iloc[-1]) else current_price
            
            if current_price > ma_20_current > ma_50_current:
                metrics['trend_direction'] = 'strong_uptrend'
            elif current_price > ma_20_current:
                metrics['trend_direction'] = 'uptrend'
            elif current_price < ma_20_current < ma_50_current:
                metrics['trend_direction'] = 'strong_downtrend'
            elif current_price < ma_20_current:
                metrics['trend_direction'] = 'downtrend'
            else:
                metrics['trend_direction'] = 'sideways'
                
            # Trend strength score
            r_squared = metrics['trend_r_squared']
            slope_magnitude = abs(metrics['trend_slope']) / price_data.mean() if price_data.mean() > 0 else 0
            
            metrics['trend_strength_score'] = float((r_squared * 0.6) + (min(1.0, slope_magnitude * 100) * 0.4))
            
            # Support and resistance levels
            recent_high = price_data.tail(20).max()
            recent_low = price_data.tail(20).min()
            
            metrics['support_level'] = float(recent_low)
            metrics['resistance_level'] = float(recent_high)
            metrics['price_position'] = float((current_price - recent_low) / (recent_high - recent_low)) if recent_high > recent_low else 0.5
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating trend metrics: {e}")
            return {}
            
    def _assess_trend_strength(self, trend_analysis: Dict) -> Dict:
        """Assess overall trend strength"""
        try:
            strength_assessment = {
                'overall_trend_strength': 'moderate',
                'trending_assets': [],
                'sideways_assets': [],
                'trend_consensus': 'mixed'
            }
            
            if not trend_analysis:
                return strength_assessment
                
            # Categorize assets by trend strength
            strong_trends = []
            weak_trends = []
            sideways_trends = []
            
            for symbol, metrics in trend_analysis.items():
                trend_direction = metrics.get('trend_direction', 'sideways')
                strength_score = metrics.get('trend_strength_score', 0.5)
                
                if strength_score > 0.7 and trend_direction in ['strong_uptrend', 'strong_downtrend']:
                    strong_trends.append(symbol)
                elif strength_score < 0.3 or trend_direction == 'sideways':
                    sideways_trends.append(symbol)
                else:
                    weak_trends.append(symbol)
                    
            strength_assessment['trending_assets'] = strong_trends
            strength_assessment['sideways_assets'] = sideways_trends
            
            # Overall trend consensus
            if len(strong_trends) >= len(weak_trends) * 2:
                strength_assessment['overall_trend_strength'] = 'strong'
                strength_assessment['trend_consensus'] = 'aligned'
            elif len(sideways_trends) >= len(strong_trends) * 2:
                strength_assessment['overall_trend_strength'] = 'weak'
                strength_assessment['trend_consensus'] = 'range_bound'
            else:
                strength_assessment['overall_trend_strength'] = 'moderate'
                strength_assessment['trend_consensus'] = 'mixed'
                
            return strength_assessment
            
        except Exception as e:
            logger.error(f"Error assessing trend strength: {e}")
            return {'overall_trend_strength': 'moderate', 'trending_assets': []}
            
    def _analyze_trend_persistence(self, trend_analysis: Dict) -> Dict:
        """Analyze trend persistence patterns"""
        try:
            persistence = {
                'persistence_score': 0.0,
                'persistent_trends': [],
                'trending_trends': [],
                'momentum_fading': []
            }
            
            if not trend_analysis:
                return persistence
                
            persistent_count = 0
            fading_count = 0
            
            for symbol, metrics in trend_analysis.items():
                # Check trend persistence
                ma_20_trend = metrics.get('ma_20_trend', 0)
                ma_50_trend = metrics.get('ma_50_trend', 0)
                trend_direction = metrics.get('trend_direction', 'sideways')
                
                # Persistent trends show consistent movement across timeframes
                if ma_20_trend > 0 and ma_50_trend > 0:
                    if trend_direction in ['strong_uptrend', 'uptrend']:
                        persistent_count += 1
                        persistence['persistent_trends'].append(symbol)
                elif ma_20_trend < 0 and ma_50_trend < 0:
                    if trend_direction in ['strong_downtrend', 'downtrend']:
                        persistent_count += 1
                        persistence['persistent_trends'].append(symbol)
                        
                # Momentum fading (trend direction changes or weakens)
                elif abs(ma_20_trend) > abs(ma_50_trend) * 2:
                    fading_count += 1
                    persistence['momentum_fading'].append(symbol)
                else:
                    persistence['trending_trends'].append(symbol)
                    
            # Calculate persistence score
            total_assets = len(trend_analysis)
            if total_assets > 0:
                persistence['persistence_score'] = float(persistent_count / total_assets)
                
            return persistence
            
        except Exception as e:
            logger.error(f"Error analyzing trend persistence: {e}")
            return {'persistence_score': 0.0, 'persistent_trends': []}
            
    def _analyze_trend_acceleration(self, trend_analysis: Dict) -> Dict:
        """Analyze trend acceleration and deceleration"""
        try:
            acceleration = {
                'acceleration_score': 0.0,
                'accelerating_trends': [],
                'decelerating_trends': [],
                'acceleration_consensus': 'mixed'
            }
            
            if not trend_analysis:
                return acceleration
                
            accelerating_trends = []
            decelerating_trends = []
            
            for symbol, metrics in trend_analysis.items():
                ma_20_trend = metrics.get('ma_20_trend', 0)
                ma_50_trend = metrics.get('ma_50_trend', 0)
                
                # Acceleration: recent trend stronger than longer-term trend
                if abs(ma_20_trend) > abs(ma_50_trend) * 1.5:
                    if ma_20_trend > 0:
                        accelerating_trends.append((symbol, 'accelerating_up'))
                    else:
                        accelerating_trends.append((symbol, 'accelerating_down'))
                        
                # Deceleration: trend weakening
                elif abs(ma_20_trend) < abs(ma_50_trend) * 0.7:
                    if ma_50_trend > 0:
                        decelerating_trends.append((symbol, 'decelerating_up'))
                    else:
                        decelerating_trends.append((symbol, 'decelerating_down'))
                        
            acceleration['accelerating_trends'] = [item[0] for item in accelerating_trends]
            acceleration['decelerating_trends'] = [item[0] for item in decelerating_trends]
            
            # Acceleration consensus
            total_analysis = len(trend_analysis)
            acceleration_ratio = len(accelerating_trends) / total_analysis if total_analysis > 0 else 0
            deceleration_ratio = len(decelerating_trends) / total_analysis if total_analysis > 0 else 0
            
            if acceleration_ratio > 0.6:
                acceleration['acceleration_consensus'] = 'accelerating'
            elif deceleration_ratio > 0.6:
                acceleration['acceleration_consensus'] = 'decelerating'
            elif abs(acceleration_ratio - deceleration_ratio) < 0.2:
                acceleration['acceleration_consensus'] = 'mixed'
            else:
                acceleration['acceleration_consensus'] = 'neutral'
                
            # Overall acceleration score
            acceleration['acceleration_score'] = float(acceleration_ratio - deceleration_ratio)
            
            return acceleration
            
        except Exception as e:
            logger.error(f"Error analyzing trend acceleration: {e}")
            return {'acceleration_score': 0.0, 'acceleration_consensus': 'mixed'}
            
    async def _momentum_acceleration_analysis(self) -> Dict:
        """Analyze momentum acceleration patterns"""
        try:
            # Get acceleration-related data
            accel_symbols = ["SPY", "QQQ", "TLT", "GLD"]
            accel_data = await self._fetch_momentum_data(accel_symbols, period="2mo")
            
            if accel_data.empty:
                return {'error': 'No acceleration data available'}
                
            acceleration_analysis = {}
            
            for symbol in accel_symbols:
                close_col = f"{symbol}_Close"
                if close_col in accel_data.columns:
                    price_data = accel_data[close_col].dropna()
                    
                    if len(price_data) > 10:
                        # Calculate momentum acceleration
                        accel_metrics = self._calculate_acceleration_metrics(price_data)
                        acceleration_analysis[symbol] = accel_metrics
                        
            if not acceleration_analysis:
                return {'error': 'No acceleration analysis possible'}
                
            # Acceleration assessment
            accel_assessment = self._assess_momentum_acceleration(acceleration_analysis)
            
            # Momentum change detection
            change_detection = self._detect_momentum_changes(acceleration_analysis)
            
            # Acceleration clustering
            clustering = self._analyze_acceleration_clustering(acceleration_analysis)
            
            return {
                'acceleration_analysis': acceleration_analysis,
                'acceleration_assessment': accel_assessment,
                'change_detection': change_detection,
                'acceleration_clustering': clustering
            }
            
        except Exception as e:
            logger.error(f"Error in momentum acceleration analysis: {e}")
            return {'error': str(e)}
            
    def _calculate_acceleration_metrics(self, price_data: pd.Series) -> Dict:
        """Calculate momentum acceleration metrics"""
        try:
            if len(price_data) < 10:
                return {}
                
            metrics = {}
            
            # Calculate momentum over different periods
            momentum_5d = price_data.pct_change(5).iloc[-1]
            momentum_10d = price_data.pct_change(10).iloc[-1]
            momentum_20d = price_data.pct_change(20).iloc[-1] if len(price_data) > 20 else momentum_10d
            
            metrics['momentum_5d'] = float(momentum_5d)
            metrics['momentum_10d'] = float(momentum_10d)
            metrics['momentum_20d'] = float(momentum_20d)
            
            # Calculate acceleration (change in momentum)
            accel_short = momentum_5d - momentum_10d
            accel_long = momentum_10d - momentum_20d
            
            metrics['acceleration_short'] = float(accel_short)
            metrics['acceleration_long'] = float(accel_long)
            metrics['overall_acceleration'] = float((accel_short + accel_long) / 2)
            
            # Acceleration classification
            if metrics['overall_acceleration'] > 0.01:
                metrics['acceleration_type'] = 'accelerating'
            elif metrics['overall_acceleration'] < -0.01:
                metrics['acceleration_type'] = 'decelerating'
            else:
                metrics['acceleration_type'] = 'stable'
                
            # Momentum volatility (change in momentum intensity)
            recent_momentum = price_data.pct_change(5).tail(5)
            momentum_volatility = recent_momentum.std()
            metrics['momentum_volatility'] = float(momentum_volatility)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating acceleration metrics: {e}")
            return {}
            
    def _assess_momentum_acceleration(self, acceleration_analysis: Dict) -> Dict:
        """Assess overall momentum acceleration"""
        try:
            assessment = {
                'overall_acceleration': 'stable',
                'acceleration_strength': 0.0,
                'accelerating_assets': [],
                'decelerating_assets': []
            }
            
            if not acceleration_analysis:
                return assessment
                
            acceleration_values = []
            
            for symbol, metrics in acceleration_analysis.items():
                overall_accel = metrics.get('overall_acceleration', 0)
                acceleration_values.append(overall_accel)
                
                if overall_accel > 0.005:
                    assessment['accelerating_assets'].append(symbol)
                elif overall_accel < -0.005:
                    assessment['decelerating_assets'].append(symbol)
                    
            # Overall acceleration assessment
            if acceleration_values:
                avg_acceleration = np.mean(acceleration_values)
                assessment['overall_acceleration'] = (
                    'strongly_accelerating' if avg_acceleration > 0.02 else
                    'accelerating' if avg_acceleration > 0.005 else
                    'strongly_decelerating' if avg_acceleration < -0.02 else
                    'decelerating' if avg_acceleration < -0.005 else
                    'stable'
                )
                assessment['acceleration_strength'] = float(abs(avg_acceleration))
                
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing momentum acceleration: {e}")
            return {'overall_acceleration': 'stable', 'acceleration_strength': 0.0}
            
    def _detect_momentum_changes(self, acceleration_analysis: Dict) -> Dict:
        """Detect momentum change patterns"""
        try:
            change_detection = {
                'momentum_shifts': [],
                'change_intensity': 0.0,
                'predictable_changes': []
            }
            
            if not acceleration_analysis:
                return change_detection
                
            shifts = []
            change_intensities = []
            
            for symbol, metrics in acceleration_analysis.items():
                accel_short = metrics.get('acceleration_short', 0)
                accel_long = metrics.get('acceleration_long', 0)
                
                # Detect momentum shifts (significant change in acceleration)
                if abs(accel_short) > 0.02:
                    if accel_short > 0:
                        shifts.append(f'{symbol}_momentum_increasing')
                    else:
                        shifts.append(f'{symbol}_momentum_decreasing')
                        
                    change_intensities.append(abs(accel_short))
                    
            change_detection['momentum_shifts'] = shifts
            change_detection['change_intensity'] = float(np.mean(change_intensities)) if change_intensities else 0.0
            
            # Identify predictable changes (consistent acceleration patterns)
            predictable = []
            for symbol, metrics in acceleration_analysis.items():
                accel_short = metrics.get('acceleration_short', 0)
                accel_long = metrics.get('acceleration_long', 0)
                
                # If short and long acceleration are in same direction
                if accel_short * accel_long > 0 and abs(accel_short) > 0.005:
                    predictable.append(symbol)
                    
            change_detection['predictable_changes'] = predictable
            
            return change_detection
            
        except Exception as e:
            logger.error(f"Error detecting momentum changes: {e}")
            return {'momentum_shifts': [], 'change_intensity': 0.0}
            
    def _analyze_acceleration_clustering(self, acceleration_analysis: Dict) -> Dict:
        """Analyze acceleration clustering patterns"""
        try:
            clustering = {
                'acceleration_clustering': False,
                'synchronized_acceleration': 0.0,
                'cluster_strength': 0.0
            }
            
            if not acceleration_analysis or len(acceleration_analysis) < 2:
                return clustering
                
            # Analyze synchronization of acceleration
            acceleration_values = [metrics.get('overall_acceleration', 0) for metrics in acceleration_analysis.values()]
            
            # Calculate acceleration synchronization
            positive_accel = sum(1 for a in acceleration_values if a > 0.005)
            negative_accel = sum(1 for a in acceleration_values if a < -0.005)
            total_assets = len(acceleration_values)
            
            if total_assets > 0:
                synchronization = max(positive_accel, negative_accel) / total_assets
                clustering['synchronized_acceleration'] = float(synchronization)
                
                # High synchronization indicates clustering
                if synchronization > 0.6:
                    clustering['acceleration_clustering'] = True
                    clustering['cluster_strength'] = float(synchronization)
                else:
                    clustering['cluster_strength'] = float(1 - synchronization)
                    
            return clustering
            
        except Exception as e:
            logger.error(f"Error analyzing acceleration clustering: {e}")
            return {'acceleration_clustering': False, 'cluster_strength': 0.0}
            
    async def _sector_momentum_analysis(self) -> Dict:
        """Analyze momentum across different sectors"""
        try:
            sector_momentum_data = {}
            
            # Analyze momentum for each sector
            for sector_etf, sector_name in self.sector_etfs.items():
                try:
                    ticker = yf.Ticker(sector_etf)
                    hist = ticker.history(period="3mo")
                    if not hist.empty:
                        # Calculate sector momentum
                        momentum_indicators = self._calculate_momentum_indicators(hist['Close'])
                        
                        if momentum_indicators:
                            sector_momentum_data[sector_etf] = {
                                'sector_name': sector_name,
                                'momentum_5d': momentum_indicators['momentum_5d'],
                                'momentum_10d': momentum_indicators['momentum_10d'],
                                'momentum_20d': momentum_indicators['momentum_20d'],
                                'rsi': momentum_indicators['rsi'],
                                'trend_strength': self._calculate_trend_strength(momentum_indicators),
                                'momentum_classification': self._classify_sector_momentum(momentum_indicators),
                                'sector_performance': self._calculate_sector_performance(hist['Close'])
                            }
                except Exception as e:
                    logger.warning(f"Error analyzing sector momentum for {sector_etf}: {e}")
                    continue
                    
            if not sector_momentum_data:
                return {'error': 'No sector momentum data available'}
                
            # Sector momentum ranking
            momentum_ranking = self._rank_sector_momentum(sector_momentum_data)
            
            # Sector rotation analysis
            rotation_analysis = self._analyze_sector_rotation(sector_momentum_data)
            
            # Sector momentum clustering
            clustering = self._analyze_sector_momentum_clustering(sector_momentum_data)
            
            return {
                'sector_momentum_data': sector_momentum_data,
                'momentum_ranking': momentum_ranking,
                'rotation_analysis': rotation_analysis,
                'momentum_clustering': clustering
            }
            
        except Exception as e:
            logger.error(f"Error in sector momentum analysis: {e}")
            return {'error': str(e)}
            
    def _classify_sector_momentum(self, indicators: Dict) -> str:
        """Classify sector momentum type"""
        try:
            momentum_5d = indicators.get('momentum_5d', 0)
            rsi = indicators.get('rsi', 50)
            trend_strength = indicators.get('trend_strength', 0.5)
            
            # Enhanced classification for sectors
            if momentum_5d > 0.04 and rsi > 65 and trend_strength > 0.7:
                return 'strong_sector_leader'
            elif momentum_5d > 0.02 and rsi > 55:
                return 'sector_outperformer'
            elif momentum_5d > 0.01:
                return 'moderate_sector_performer'
            elif momentum_5d < -0.04 and rsi < 35 and trend_strength > 0.7:
                return 'sector_laggard'
            elif momentum_5d < -0.02 and rsi < 45:
                return 'sector_underperformer'
            elif momentum_5d < -0.01:
                return 'sector_weakness'
            else:
                return 'sector_neutral'
                
        except Exception as e:
            logger.error(f"Error classifying sector momentum: {e}")
            return 'unknown'
            
    def _calculate_sector_performance(self, price_data: pd.Series) -> Dict:
        """Calculate sector performance metrics"""
        try:
            if len(price_data) < 10:
                return {}
                
            performance = {}
            
            # Performance over different periods
            performance['1w'] = float(price_data.pct_change(5).iloc[-1])
            performance['1m'] = float(price_data.pct_change(20).iloc[-1])
            performance['3m'] = float(price_data.pct_change(60).iloc[-1]) if len(price_data) > 60 else float(price_data.pct_change(len(price_data)-1).iloc[-1])
            
            # Performance consistency
            returns = price_data.pct_change().dropna()
            performance['volatility'] = float(returns.std() * np.sqrt(252))  # Annualized
            performance['sharpe'] = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # Maximum drawdown
            peak = price_data.expanding().max()
            drawdown = (price_data - peak) / peak
            performance['max_drawdown'] = float(drawdown.min())
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating sector performance: {e}")
            return {}
            
    def _rank_sector_momentum(self, sector_data: Dict) -> Dict:
        """Rank sectors by momentum strength"""
        try:
            # Create ranking based on momentum and performance
            sector_scores = {}
            
            for sector_etf, data in sector_data.items():
                # Calculate composite momentum score
                momentum_score = data['momentum_5d'] * 0.4 + data['momentum_10d'] * 0.3 + data['momentum_20d'] * 0.3
                
                # Adjust for trend strength and RSI
                trend_bonus = data['trend_strength'] * 0.1
                rsi_adjustment = (data['rsi'] - 50) / 500  # Small adjustment for RSI level
                
                composite_score = momentum_score + trend_bonus + rsi_adjustment
                sector_scores[sector_etf] = {
                    'sector_name': data['sector_name'],
                    'momentum_score': float(composite_score),
                    'raw_momentum_5d': data['momentum_5d'],
                    'classification': data['momentum_classification']
                }
                
            # Sort by momentum score
            sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1]['momentum_score'], reverse=True)
            
            ranking = {
                'top_momentum_sectors': [item[0] for item in sorted_sectors[:3]],
                'bottom_momentum_sectors': [item[0] for item in sorted_sectors[-3:]],
                'momentum_leaders': sorted_sectors[:5],
                'momentum_laggards': sorted_sectors[-5:]
            }
            
            return ranking
            
        except Exception as e:
            logger.error(f"Error ranking sector momentum: {e}")
            return {'top_momentum_sectors': [], 'momentum_leaders': []}
            
    def _analyze_sector_rotation(self, sector_data: Dict) -> Dict:
        """Analyze sector rotation patterns"""
        try:
            rotation = {
                'rotation_detected': False,
                'rotation_strength': 0.0,
                'rotation_direction': 'stable',
                'rotating_sectors': []
            }
            
            if len(sector_data) < 3:
                return rotation
                
            # Calculate momentum momentum (change in momentum)
            momentum_changes = []
            for data in sector_data.values():
                momentum_change = data['momentum_5d'] - data['momentum_10d']
                momentum_changes.append(momentum_change)
                
            # Detect rotation patterns
            if len(momentum_changes) > 0:
                momentum_variance = np.var(momentum_changes)
                avg_momentum_change = np.mean(momentum_changes)
                
                # High variance indicates rotation
                if momentum_variance > 0.001:  # Threshold for rotation detection
                    rotation['rotation_detected'] = True
                    rotation['rotation_strength'] = float(momentum_variance)
                    
                    # Determine rotation direction
                    if avg_momentum_change > 0.005:
                        rotation['rotation_direction'] = 'momentum_building'
                    elif avg_momentum_change < -0.005:
                        rotation['rotation_direction'] = 'momentum_fading'
                    else:
                        rotation['rotation_direction'] = 'mixed_rotation'
                        
                    # Identify sectors with significant momentum changes
                    for sector_etf, data in sector_data.items():
                        momentum_change = data['momentum_5d'] - data['momentum_10d']
                        if abs(momentum_change) > 0.01:  # Significant change threshold
                            direction = 'building' if momentum_change > 0 else 'fading'
                            rotation['rotating_sectors'].append(f"{sector_etf}_{direction}")
                            
            return rotation
            
        except Exception as e:
            logger.error(f"Error analyzing sector rotation: {e}")
            return {'rotation_detected': False, 'rotation_strength': 0.0}
            
    def _analyze_sector_momentum_clustering(self, sector_data: Dict) -> Dict:
        """Analyze sector momentum clustering"""
        try:
            clustering = {
                'clustering_detected': False,
                'momentum_clusters': [],
                'cross_sector_correlation': 0.0
            }
            
            if len(sector_data) < 3:
                return clustering
                
            # Calculate momentum correlation across sectors
            momentum_values = [data['momentum_5d'] for data in sector_data.values()]
            
            # Use standard deviation as proxy for clustering
            momentum_std = np.std(momentum_values)
            momentum_mean = np.mean(momentum_values)
            
            # High standard deviation = low clustering (more divergence)
            # Low standard deviation = high clustering (more synchronization)
            correlation_proxy = 1 / (1 + momentum_std * 20) if momentum_std > 0 else 1
            clustering['cross_sector_correlation'] = float(correlation_proxy)
            
            # Detect momentum clusters
            if correlation_proxy > 0.7:
                clustering['clustering_detected'] = True
                
                # Identify clusters based on momentum levels
                high_momentum_sectors = [etf for etf, data in sector_data.items() if data['momentum_5d'] > momentum_mean + momentum_std]
                low_momentum_sectors = [etf for etf, data in sector_data.items() if data['momentum_5d'] < momentum_mean - momentum_std]
                
                if high_momentum_sectors:
                    clustering['momentum_clusters'].append({'type': 'high_momentum', 'sectors': high_momentum_sectors})
                if low_momentum_sectors:
                    clustering['momentum_clusters'].append({'type': 'low_momentum', 'sectors': low_momentum_sectors})
                    
            return clustering
            
        except Exception as e:
            logger.error(f"Error analyzing sector momentum clustering: {e}")
            return {'clustering_detected': False, 'cross_sector_correlation': 0.0}
            
    async def _cross_asset_momentum_analysis(self) -> Dict:
        """Analyze momentum across different asset classes"""
        try:
            # Define cross-asset momentum groups
            cross_asset_groups = {
                "equities": ["SPY", "QQQ", "IWM"],
                "fixed_income": ["TLT", "IEF", "SHY"],
                "commodities": ["GLD", "SLV", "DBA"],
                "international": ["VEA", "VWO", "IEFA"]
            }
            
            cross_asset_momentum = {}
            
            for asset_class, symbols in cross_asset_groups.items():
                class_momentum_data = []
                
                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="3mo")
                        if not hist.empty:
                            momentum_indicators = self._calculate_momentum_indicators(hist['Close'])
                            if momentum_indicators:
                                class_momentum_data.append({
                                    'symbol': symbol,
                                    'momentum_5d': momentum_indicators['momentum_5d'],
                                    'momentum_consensus': self._calculate_momentum_consensus([momentum_indicators])
                                })
                    except Exception as e:
                        logger.warning(f"Error analyzing cross-asset momentum for {symbol}: {e}")
                        continue
                        
                if class_momentum_data:
                    avg_momentum = np.mean([d['momentum_5d'] for d in class_momentum_data])
                    momentum_consensus = class_momentum_data[0]['momentum_consensus']
                    
                    cross_asset_momentum[asset_class] = {
                        'class_momentum': float(avg_momentum),
                        'consensus': momentum_consensus,
                        'assets': class_momentum_data
                    }
                    
            if not cross_asset_momentum:
                return {'error': 'No cross-asset momentum data available'}
                
            # Cross-asset momentum assessment
            assessment = self._assess_cross_asset_momentum(cross_asset_momentum)
            
            # Inter-asset momentum correlation
            correlation = self._calculate_inter_asset_momentum_correlation(cross_asset_momentum)
            
            # Asset class rotation
            rotation = self._analyze_cross_asset_rotation(cross_asset_momentum)
            
            return {
                'cross_asset_momentum': cross_asset_momentum,
                'momentum_assessment': assessment,
                'asset_correlation': correlation,
                'cross_asset_rotation': rotation
            }
            
        except Exception as e:
            logger.error(f"Error in cross-asset momentum analysis: {e}")
            return {'error': str(e)}
            
    def _assess_cross_asset_momentum(self, momentum_data: Dict) -> Dict:
        """Assess momentum across asset classes"""
        try:
            assessment = {
                'leading_asset_class': None,
                'momentum_leaders': [],
                'momentum_laggards': [],
                'cross_asset_trend': 'neutral',
                'diversification_benefit': 0.0
            }
                
            # Rank asset classes by momentum
            class_momentum_scores = {k: v['class_momentum'] for k, v in momentum_data.items()}
            
            if class_momentum_scores:
                sorted_classes = sorted(class_momentum_scores.items(), key=lambda x: x[1], reverse=True)
                
                assessment['leading_asset_class'] = sorted_classes[0][0]
                assessment['momentum_leaders'] = [item[0] for item in sorted_classes[:2]]
                assessment['momentum_laggards'] = [item[0] for item in sorted_classes[-2:]]
                
                # Determine cross-asset trend
                positive_classes = sum(1 for score in class_momentum_scores.values() if score > 0)
                negative_classes = len(class_momentum_scores) - positive_classes
                
                if positive_classes >= len(class_momentum_scores) * 0.75:
                    assessment['cross_asset_trend'] = 'broadly_positive'
                elif negative_classes >= len(class_momentum_scores) * 0.75:
                    assessment['cross_asset_trend'] = 'broadly_negative'
                else:
                    assessment['cross_asset_trend'] = 'mixed'
                    
                # Calculate diversification benefit
                momentum_variance = np.var(list(class_momentum_scores.values()))
                assessment['diversification_benefit'] = float(momentum_variance * 10)  # Scale appropriately
                
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing cross-asset momentum: {e}")
            return {'cross_asset_trend': 'neutral', 'diversification_benefit': 0.0}
            
    def _calculate_inter_asset_momentum_correlation(self, momentum_data: Dict) -> Dict:
        """Calculate momentum correlation between asset classes"""
        try:
            correlation = {
                'correlation_strength': 0.0,
                'correlation_regime': 'normal',
                'synchronized_momentum': False
            }
                
            momentum_values = list(momentum_data.values())
            if len(momentum_values) < 2:
                return correlation
                
            # Calculate momentum correlation
            class_momentum_list = [data['class_momentum'] for data in momentum_values]
            
            # Use standard deviation as proxy for correlation
            momentum_std = np.std(class_momentum_list)
            momentum_mean = np.mean(class_momentum_list)
            
            # Higher std = lower correlation
            correlation_proxy = 1 / (1 + momentum_std * 10) if momentum_std > 0 else 1
            correlation['correlation_strength'] = float(correlation_proxy)
            
            # Correlation regime
            if correlation_proxy > 0.8:
                correlation['correlation_regime'] = 'high_correlation'
            elif correlation_proxy < 0.3:
                correlation['correlation_regime'] = 'low_correlation'
            else:
                correlation['correlation_regime'] = 'normal_correlation'
                
            # Synchronized momentum detection
            positive_momentum = sum(1 for m in class_momentum_list if m > 0)
            negative_momentum = len(class_momentum_list) - positive_momentum
            
            if positive_momentum >= len(class_momentum_list) * 0.75 or negative_momentum >= len(class_momentum_list) * 0.75:
                correlation['synchronized_momentum'] = True
                
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating inter-asset momentum correlation: {e}")
            return {'correlation_strength': 0.0, 'correlation_regime': 'normal'}
            
    def _analyze_cross_asset_rotation(self, momentum_data: Dict) -> Dict:
        """Analyze rotation between asset classes"""
        try:
            rotation = {
                'rotation_detected': False,
                'rotation_strength': 0.0,
                'rotation_assets': [],
                'rotation_direction': 'stable'
            }
                
            # This would need historical momentum data to detect rotation
            # For now, we'll create a framework based on current momentum dispersion
            momentum_values = [data['class_momentum'] for data in momentum_data.values()]
            
            if len(momentum_values) > 1:
                momentum_range = max(momentum_values) - min(momentum_values)
                momentum_std = np.std(momentum_values)
                
                # High dispersion indicates rotation
                if momentum_range > 0.04:  # 4% range threshold
                    rotation['rotation_detected'] = True
                    rotation['rotation_strength'] = float(momentum_range)
                    
                    # Identify rotating assets
                    mean_momentum = np.mean(momentum_values)
                    for asset_class, data in momentum_data.items():
                        momentum_deviation = data['class_momentum'] - mean_momentum
                        if abs(momentum_deviation) > 0.02:  # 2% deviation threshold
                            direction = 'outperforming' if momentum_deviation > 0 else 'underperforming'
                            rotation['rotation_assets'].append(f"{asset_class}_{direction}")
                            
                    # Rotation direction
                    if momentum_std > 0.02:
                        rotation['rotation_direction'] = 'high_volatility_rotation'
                    else:
                        rotation['rotation_direction'] = 'moderate_rotation'
                        
            return rotation
            
        except Exception as e:
            logger.error(f"Error analyzing cross-asset rotation: {e}")
            return {'rotation_detected': False, 'rotation_strength': 0.0}
            
    async def get_momentum_pulse(self) -> Dict:
        """Get comprehensive momentum analysis"""
        try:
            # Run all momentum analyses in parallel
            momentum_tasks = [
                self._price_momentum_analysis(),
                self._trend_strength_analysis(),
                self._momentum_acceleration_analysis(),
                self._sector_momentum_analysis(),
                self._cross_asset_momentum_analysis()
            ]
            
            results = await asyncio.gather(*momentum_tasks, return_exceptions=True)
            (
                price_momentum, trend_strength,
                momentum_acceleration, sector_momentum,
                cross_asset_momentum
            ) = results
            
            # Calculate overall Momentum Momentum Score (MMS)
            mms_components = []
            
            for result in results:
                if isinstance(result, dict) and not result.get('error'):
                    # Extract numeric scores from each analysis
                    score = self._extract_momentum_score(result)
                    if score is not None:
                        mms_components.append(score)
                        
            if mms_components:
                mms_score = np.mean(mms_components)
                mms_volatility = np.std(mms_components)
                
                # Classify momentum state
                if mms_score > 0.3:
                    momentum_state = 'strong_momentum'
                elif mms_score > 0.1:
                    momentum_state = 'moderate_momentum'
                elif mms_score < -0.3:
                    momentum_state = 'strong_momentum_reversal'
                elif mms_score < -0.1:
                    momentum_state = 'momentum_fading'
                else:
                    momentum_state = 'momentum_neutral'
                    
                return {
                    'momentum_momentum_score': mms_score,
                    'mms_volatility': mms_volatility,
                    'momentum_state': momentum_state,
                    'analysis_breakdown': {
                        'price_momentum': price_momentum,
                        'trend_strength': trend_strength,
                        'momentum_acceleration': momentum_acceleration,
                        'sector_momentum': sector_momentum,
                        'cross_asset_momentum': cross_asset_momentum
                    },
                    'timestamp': datetime.utcnow(),
                    'confidence': min(1.0, 1 - (mms_volatility / 2))
                }
            else:
                return {'error': 'Unable to calculate momentum momentum score'}
                
        except Exception as e:
            logger.error(f"Error getting momentum pulse: {e}")
            return {'error': str(e)}
            
    def _extract_momentum_score(self, analysis_result: Dict) -> Optional[float]:
        """Extract numeric momentum score from analysis result"""
        try:
            if 'momentum_assessment' in analysis_result and 'momentum_strength' in analysis_result['momentum_assessment']:
                return analysis_result['momentum_assessment']['momentum_strength']
            elif 'strength_assessment' in analysis_result:
                # Convert trend strength to momentum score
                strength = analysis_result['strength_assessment'].get('overall_trend_strength', 'moderate')
                if strength == 'strong':
                    return 0.7
                elif strength == 'moderate':
                    return 0.4
                else:
                    return 0.1
            elif 'acceleration_assessment' in analysis_result:
                return analysis_result['acceleration_assessment'].get('acceleration_strength', 0)
            elif 'momentum_ranking' in analysis_result and analysis_result['momentum_ranking'].get('top_momentum_sectors'):
                # Calculate momentum score from sector ranking
                top_sectors = analysis_result['momentum_ranking']['top_momentum_sectors']
                return min(1.0, len(top_sectors) / 5)  # Normalize to 0-1
            elif 'momentum_assessment' in analysis_result and 'cross_asset_trend' in analysis_result['momentum_assessment']:
                trend = analysis_result['momentum_assessment']['cross_asset_trend']
                if trend == 'broadly_positive':
                    return 0.6
                elif trend == 'broadly_negative':
                    return -0.6
                else:
                    return 0.0
            else:
                return None
                
        except Exception:
            return None
            
    async def store_momentum_data(self, momentum_data: Dict):
        """Store momentum metrics in time-series database"""
        try:
            if self.db_manager and 'timestamp' in momentum_data:
                # Store Momentum Momentum Score
                await self.db_manager.influxdb_client.write_points(
                    database='market_pulse',
                    measurement='momentum_metrics',
                    tags={
                        'engine': 'momentum_pulse',
                        'state': momentum_data.get('momentum_state', 'unknown')
                    },
                    fields={
                        'mms_score': float(momentum_data.get('momentum_momentum_score', 0)),
                        'mms_volatility': float(momentum_data.get('mms_volatility', 0)),
                        'confidence': float(momentum_data.get('confidence', 0))
                    },
                    time=momentum_data['timestamp']
                )
                
                # Store component scores
                for component_name, analysis in momentum_data.get('analysis_breakdown', {}).items():
                    if isinstance(analysis, dict):
                        score = self._extract_momentum_score(analysis)
                        if score is not None:
                            await self.db_manager.influxdb_client.write_points(
                                database='market_pulse',
                                measurement='momentum_components',
                                tags={
                                    'component': component_name,
                                    'engine': 'momentum_pulse'
                                },
                                fields={'component_score': float(score)},
                                time=momentum_data['timestamp']
                            )
                            
            logger.debug("Momentum data stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing momentum data: {e}")
            
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
                'cache_size': len(self.momentum_cache),
                'models_loaded': len(self.models),
                'tracked_asset_classes': len(self.asset_classes),
                'tracked_sectors': len(self.sector_etfs),
                'momentum_indicators': len(self.momentum_indicators),
                'health_score': self._calculate_engine_health()
            }
            
        except Exception as e:
            logger.error(f"Error getting momentum engine status: {e}")
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
            total_data_sources = len(self.asset_classes) + len(self.sector_etfs)
            health_factors.append(min(1.0, total_data_sources / 20))
            
            return np.mean(health_factors) if health_factors else 0.0
            
        except Exception:
            return 0.0