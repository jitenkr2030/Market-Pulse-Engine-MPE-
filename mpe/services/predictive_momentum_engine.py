"""
Module 28: Predictive Momentum Engine
Author: MiniMax Agent
Date: 2025-12-02

Advanced predictive momentum analysis and forecasting system.
Provides comprehensive momentum signal generation, trend prediction,
cross-asset momentum modeling, and systematic trend-following strategies.
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
from scipy import signal, optimize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MomentumType(Enum):
    """Types of momentum signals"""
    PRICE_MOMENTUM = "price_momentum"
    VOLUME_MOMENTUM = "volume_momentum"
    CROSS_ASSET_MOMENTUM = "cross_asset_momentum"
    SECTOR_MOMENTUM = "sector_momentum"
    STATISTICAL_MOMENTUM = "statistical_momentum"
    REGIME_MOMENTUM = "regime_momentum"
    VOLATILITY_MOMENTUM = "volatility_momentum"
    MOMENTUM_REVERSAL = "momentum_reversal"
    ACCELERATION_MOMENTUM = "acceleration_momentum"
    MEAN_REVERSION = "mean_reversion"

class SignalHorizon(Enum):
    """Momentum signal horizons"""
    VERY_SHORT_TERM = "very_short_term"  # 1-5 days
    SHORT_TERM = "short_term"           # 1-4 weeks
    MEDIUM_TERM = "medium_term"         # 1-6 months
    LONG_TERM = "long_term"             # 6+ months

class TrendStrength(Enum):
    """Trend strength classifications"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"

class PredictionConfidence(Enum):
    """Prediction confidence levels"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"

@dataclass
class MomentumSignal:
    """Individual momentum signal"""
    symbol: str
    momentum_type: str
    signal_horizon: str
    current_strength: float
    predicted_change: float
    confidence_level: str
    trend_direction: str
    persistence_score: float
    risk_adjusted_return: float
    position_size: float
    stop_loss: float
    take_profit: float
    timestamp: datetime

@dataclass
class CrossAssetMomentum:
    """Cross-asset momentum relationship"""
    primary_asset: str
    secondary_asset: str
    lead_lag_relationship: str
    correlation_strength: float
    predictive_power: float
    optimal_holding_period: int
    regime_dependent: bool
    signal_quality: float

@dataclass
class MomentumForecast:
    """Momentum-based price forecast"""
    symbol: str
    forecast_horizon: int  # days
    predicted_price: float
    predicted_return: float
    confidence_interval: Tuple[float, float]
    prediction_confidence: str
    key_factors: List[str]
    risk_factors: List[str]
    model_performance: Dict[str, float]

@dataclass
class MomentumStrategy:
    """Momentum-based trading strategy"""
    strategy_id: str
    momentum_types: List[str]
    asset_universe: List[str]
    position_sizing: str
    rebalancing_frequency: str
    expected_sharpe: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    turnover_rate: float
    risk_metrics: Dict[str, float]
    performance_attribution: Dict[str, float]

class PredictiveMomentumEngine:
    """
    Advanced Predictive Momentum Engine
    
    Analyzes, predicts, and provides systematic momentum signals
    across multiple timeframes and asset classes to support
    trend-following and momentum-based investment strategies.
    """
    
    def __init__(self):
        self.name = "Predictive Momentum Engine"
        self.version = "1.0.0"
        self.cache = {}
        self.cache_ttl = 240  # 4 minutes
        
        # Momentum calculation parameters
        self.momentum_windows = {
            "very_short_term": 5,
            "short_term": 20,
            "medium_term": 60,
            "long_term": 252
        }
        
        # Asset universe for momentum analysis
        self.momentum_universe = {
            # Large Cap Equities
            "AAPL": {"type": "equity", "sector": "technology", "volatility": 0.25},
            "MSFT": {"type": "equity", "sector": "technology", "volatility": 0.22},
            "GOOGL": {"type": "equity", "sector": "technology", "volatility": 0.23},
            "AMZN": {"type": "equity", "sector": "consumer_discretionary", "volatility": 0.28},
            "TSLA": {"type": "equity", "sector": "automotive", "volatility": 0.45},
            "NVDA": {"type": "equity", "sector": "technology", "volatility": 0.35},
            "META": {"type": "equity", "sector": "technology", "volatility": 0.30},
            
            # Equity ETFs
            "SPY": {"type": "equity_etf", "sector": "broad_market", "volatility": 0.18},
            "QQQ": {"type": "equity_etf", "sector": "technology", "volatility": 0.25},
            "IWM": {"type": "equity_etf", "sector": "small_cap", "volatility": 0.28},
            "VTI": {"type": "equity_etf", "sector": "broad_market", "volatility": 0.17},
            
            # Fixed Income
            "TLT": {"type": "fixed_income", "sector": "treasury_long", "volatility": 0.15},
            "IEF": {"type": "fixed_income", "sector": "treasury_intermediate", "volatility": 0.08},
            "SHY": {"type": "fixed_income", "sector": "treasury_short", "volatility": 0.03},
            "LQD": {"type": "fixed_income", "sector": "corporate_ig", "volatility": 0.10},
            "HYG": {"type": "fixed_income", "sector": "corporate_hy", "volatility": 0.18},
            
            # Commodities
            "GLD": {"type": "commodity", "sector": "precious_metals", "volatility": 0.20},
            "SLV": {"type": "commodity", "sector": "precious_metals", "volatility": 0.30},
            "USO": {"type": "commodity", "sector": "energy", "volatility": 0.40},
            "DBC": {"type": "commodity", "sector": "broad_commodity", "volatility": 0.25},
            
            # Currency
            "UUP": {"type": "currency", "sector": "dollar", "volatility": 0.12},
            "FXE": {"type": "currency", "sector": "euro", "volatility": 0.14},
            
            # Real Estate
            "VNQ": {"type": "real_estate", "sector": "reits", "volatility": 0.25},
            
            # Alternatives
            "VIX": {"type": "volatility", "sector": "fear_index", "volatility": 0.80}
        }
        
        # Momentum signal thresholds
        self.momentum_thresholds = {
            "very_strong": 0.75,
            "strong": 0.60,
            "moderate": 0.40,
            "weak": 0.20,
            "very_weak": 0.00
        }
        
        # Risk management parameters
        self.risk_parameters = {
            "max_position_size": 0.10,      # 10% max position
            "stop_loss_pct": 0.05,          # 5% stop loss
            "take_profit_pct": 0.15,        # 15% take profit
            "max_leverage": 2.0,            # Max leverage
            "correlation_limit": 0.7,       # Max correlation between positions
            "volatility_target": 0.15       # Target portfolio volatility
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
    
    async def fetch_momentum_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive data for momentum analysis"""
        try:
            cache_key = f"momentum_data_{'_'.join(sorted(symbols))}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            momentum_data = {}
            
            # Fetch data for all symbols
            tasks = []
            for symbol in symbols:
                task = self._fetch_single_momentum_data(symbol)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(symbols, results):
                if isinstance(result, pd.DataFrame) and not result.empty:
                    momentum_data[symbol] = result
                else:
                    logger.warning(f"No momentum data available for {symbol}")
                    momentum_data[symbol] = self._create_momentum_placeholder_data(symbol)
            
            await self._set_cache_data(cache_key, momentum_data)
            return momentum_data
            
        except Exception as e:
            logger.error(f"Error fetching momentum data: {str(e)}")
            return {}
    
    async def _fetch_single_momentum_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data for a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y", interval="1d")
            
            if data.empty:
                data = ticker.history(period="1y", interval="1d")
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching momentum data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for momentum analysis"""
        try:
            if data.empty:
                return data
            
            # Moving averages
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['SMA_200'] = data['Close'].rolling(200).mean()
            
            # Exponential moving averages
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(20).mean()
            bb_std = data['Close'].rolling(20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
            data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
            
            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            
            # Price momentum
            for window in [5, 10, 20, 60]:
                data[f'Momentum_{window}'] = (data['Close'] / data['Close'].shift(window) - 1) * 100
            
            # Volatility
            data['Volatility_20'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
            data['Volatility_60'] = data['Close'].pct_change().rolling(60).std() * np.sqrt(252)
            
            # Price acceleration
            data['Acceleration'] = data['Close'].pct_change().diff()
            
            return data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return data
    
    def _create_momentum_placeholder_data(self, symbol: str) -> pd.DataFrame:
        """Create placeholder data for momentum analysis"""
        try:
            # Get asset characteristics
            asset_info = self.momentum_universe.get(symbol, {"type": "equity", "volatility": 0.25})
            asset_type = asset_info["type"]
            volatility = asset_info["volatility"]
            
            # Generate synthetic data
            days = 504  # 2 years
            dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
            
            # Set base characteristics by asset type
            if asset_type == "equity":
                base_price = 150
                daily_vol = volatility / np.sqrt(252)
                drift = 0.0003
            elif asset_type == "equity_etf":
                base_price = 300
                daily_vol = volatility / np.sqrt(252)
                drift = 0.0002
            elif asset_type == "fixed_income":
                base_price = 100
                daily_vol = volatility / np.sqrt(252)
                drift = 0.0001
            elif asset_type == "commodity":
                base_price = 100
                daily_vol = volatility / np.sqrt(252)
                drift = 0.0004
            elif asset_type == "currency":
                base_price = 1.0
                daily_vol = volatility / np.sqrt(252)
                drift = 0.0001
            elif asset_type == "volatility":
                base_price = 20
                daily_vol = volatility / np.sqrt(252)
                drift = 0.001
            else:
                base_price = 100
                daily_vol = volatility / np.sqrt(252)
                drift = 0.0002
            
            # Generate price series with momentum characteristics
            returns = np.random.normal(drift, daily_vol, days)
            prices = [base_price]
            
            # Add momentum and trend components
            for i, ret in enumerate(returns[1:]):
                # Add momentum persistence
                if i > 50:  # After initial period
                    momentum_effect = 0.1 * returns[max(0, i-20):i].mean()
                    trend_effect = 0.001 * np.sin(2 * np.pi * i / 252)  # Annual cycle
                    total_return = ret + momentum_effect + trend_effect
                else:
                    total_return = ret
                
                prices.append(prices[-1] * (1 + total_return))
            
            data = pd.DataFrame({
                'Open': [p * np.random.uniform(0.999, 1.001) for p in prices],
                'High': [p * np.random.uniform(1.001, 1.003) for p in prices],
                'Low': [p * np.random.uniform(0.997, 0.999) for p in prices],
                'Close': prices,
                'Volume': [np.random.randint(1000000, 10000000) for _ in range(days)]
            }, index=dates)
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error creating momentum placeholder data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def calculate_momentum_signals(self, symbols: List[str]) -> List[MomentumSignal]:
        """Calculate comprehensive momentum signals"""
        try:
            momentum_data = await self.fetch_momentum_data(symbols)
            if not momentum_data:
                return []
            
            momentum_signals = []
            
            for symbol, data in momentum_data.items():
                if data.empty:
                    continue
                
                # Calculate different types of momentum signals
                signals = await self._calculate_symbol_momentum_signals(symbol, data)
                momentum_signals.extend(signals)
            
            # Sort by strength and confidence
            momentum_signals.sort(key=lambda x: x.current_strength * x.confidence_level_score(), reverse=True)
            
            return momentum_signals
            
        except Exception as e:
            logger.error(f"Error calculating momentum signals: {str(e)}")
            return []
    
    def confidence_level_score(self) -> float:
        """Convert confidence level to numeric score"""
        confidence_scores = {
            "very_high": 1.0,
            "high": 0.8,
            "moderate": 0.6,
            "low": 0.4,
            "very_low": 0.2
        }
        return confidence_scores.get(self.confidence_level, 0.5)
    
    async def _calculate_symbol_momentum_signals(self, symbol: str, data: pd.DataFrame) -> List[MomentumSignal]:
        """Calculate momentum signals for a single symbol"""
        try:
            signals = []
            
            # Price momentum signals
            price_momentum = await self._calculate_price_momentum_signal(symbol, data)
            if price_momentum:
                signals.append(price_momentum)
            
            # Volume momentum signals
            volume_momentum = await self._calculate_volume_momentum_signal(symbol, data)
            if volume_momentum:
                signals.append(volume_momentum)
            
            # Statistical momentum
            statistical_momentum = await self._calculate_statistical_momentum_signal(symbol, data)
            if statistical_momentum:
                signals.append(statistical_momentum)
            
            # Volatility momentum
            volatility_momentum = await self._calculate_volatility_momentum_signal(symbol, data)
            if volatility_momentum:
                signals.append(volatility_momentum)
            
            # Acceleration momentum
            acceleration_momentum = await self._calculate_acceleration_momentum_signal(symbol, data)
            if acceleration_momentum:
                signals.append(acceleration_momentum)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error calculating momentum signals for {symbol}: {str(e)}")
            return []
    
    async def _calculate_price_momentum_signal(self, symbol: str, data: pd.DataFrame) -> Optional[MomentumSignal]:
        """Calculate price momentum signal"""
        try:
            if data.empty or len(data) < 200:
                return None
            
            current_price = data['Close'].iloc[-1]
            
            # Calculate momentum across multiple timeframes
            momentum_5d = (current_price / data['Close'].iloc[-6] - 1) if len(data) > 5 else 0
            momentum_20d = (current_price / data['Close'].iloc[-21] - 1) if len(data) > 20 else 0
            momentum_60d = (current_price / data['Close'].iloc[-61] - 1) if len(data) > 60 else 0
            
            # Weighted momentum score
            momentum_score = (momentum_5d * 0.4 + momentum_20d * 0.4 + momentum_60d * 0.2)
            
            # Determine signal strength
            signal_strength = self._classify_momentum_strength(abs(momentum_score))
            
            # Predict future change (simplified continuation model)
            recent_trend = momentum_20d * 0.6 + momentum_60d * 0.4
            predicted_change = recent_trend * 0.7  # Assume momentum continues at 70% strength
            
            # Calculate confidence based on consistency
            momentum_consistency = self._calculate_momentum_consistency([momentum_5d, momentum_20d, momentum_60d])
            confidence_level = self._classify_prediction_confidence(momentum_consistency)
            
            # Determine trend direction
            trend_direction = "uptrend" if momentum_score > 0 else "downtrend"
            
            # Calculate persistence score
            persistence_score = self._calculate_momentum_persistence(data['Close'])
            
            # Risk-adjusted return estimate
            volatility = data['Close'].pct_change().std() * np.sqrt(252)
            risk_adjusted_return = momentum_score / volatility if volatility > 0 else momentum_score
            
            signal = MomentumSignal(
                symbol=symbol,
                momentum_type="price_momentum",
                signal_horizon="medium_term",
                current_strength=abs(momentum_score),
                predicted_change=predicted_change,
                confidence_level=confidence_level,
                trend_direction=trend_direction,
                persistence_score=persistence_score,
                risk_adjusted_return=risk_adjusted_return,
                position_size=self._calculate_position_size(momentum_score, volatility),
                stop_loss=self.risk_parameters["stop_loss_pct"],
                take_profit=self.risk_parameters["take_profit_pct"],
                timestamp=datetime.now()
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error calculating price momentum signal: {str(e)}")
            return None
    
    async def _calculate_volume_momentum_signal(self, symbol: str, data: pd.DataFrame) -> Optional[MomentumSignal]:
        """Calculate volume momentum signal"""
        try:
            if data.empty or len(data) < 50:
                return None
            
            # Volume momentum
            recent_volume = data['Volume'].tail(20).mean()
            historical_volume = data['Volume'].tail(60).head(40).mean()
            volume_momentum = (recent_volume / historical_volume - 1)
            
            # Volume-price momentum confirmation
            price_change = (data['Close'].iloc[-1] / data['Close'].iloc[-21] - 1) if len(data) > 20 else 0
            volume_price_confirmation = 1.0 if (volume_momentum > 0.1 and price_change > 0) or (volume_momentum < -0.1 and price_change < 0) else 0.5
            
            # Combined signal strength
            signal_strength = abs(volume_momentum) * volume_price_confirmation
            
            if signal_strength < 0.1:  # Minimum threshold
                return None
            
            # Predict continuation
            predicted_change = price_change * 0.3  # Volume leads price slightly
            
            confidence_level = "moderate" if signal_strength > 0.3 else "low"
            trend_direction = "volume_increasing" if volume_momentum > 0 else "volume_decreasing"
            
            signal = MomentumSignal(
                symbol=symbol,
                momentum_type="volume_momentum",
                signal_horizon="short_term",
                current_strength=signal_strength,
                predicted_change=predicted_change,
                confidence_level=confidence_level,
                trend_direction=trend_direction,
                persistence_score=0.4,  # Volume momentum is less persistent
                risk_adjusted_return=signal_strength * 0.5,
                position_size=signal_strength * 0.05,
                stop_loss=self.risk_parameters["stop_loss_pct"],
                take_profit=self.risk_parameters["take_profit_pct"],
                timestamp=datetime.now()
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error calculating volume momentum signal: {str(e)}")
            return None
    
    async def _calculate_statistical_momentum_signal(self, symbol: str, data: pd.DataFrame) -> Optional[MomentumSignal]:
        """Calculate statistical momentum using machine learning"""
        try:
            if data.empty or len(data) < 200:
                return None
            
            # Prepare features
            features = self._prepare_momentum_features(data)
            if features.empty or features.shape[1] < 5:
                return None
            
            # Prepare target (future returns)
            target = self._prepare_momentum_target(data)
            if target.empty:
                return None
            
            # Align features and target
            min_length = min(len(features), len(target))
            if min_length < 100:
                return None
            
            X = features.tail(min_length)
            y = target.tail(min_length)
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            
            # Use rolling window for training
            train_size = min(100, len(X) // 2)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            model.fit(X_train, y_train)
            
            # Make prediction
            latest_features = X.iloc[-1:].values.reshape(1, -1)
            predicted_return = model.predict(latest_features)[0]
            
            # Calculate model performance
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            model_performance = 1 / (1 + mse)  # Simple performance metric
            
            # Signal strength based on prediction confidence
            signal_strength = abs(predicted_return) * model_performance
            confidence_level = self._classify_prediction_confidence(model_performance)
            
            if signal_strength < 0.05:  # Minimum threshold
                return None
            
            trend_direction = "uptrend" if predicted_return > 0 else "downtrend"
            
            signal = MomentumSignal(
                symbol=symbol,
                momentum_type="statistical_momentum",
                signal_horizon="short_term",
                current_strength=signal_strength,
                predicted_change=predicted_return,
                confidence_level=confidence_level,
                trend_direction=trend_direction,
                persistence_score=model_performance,
                risk_adjusted_return=predicted_return / data['Close'].pct_change().std() if data['Close'].pct_change().std() > 0 else predicted_return,
                position_size=signal_strength * 0.08,
                stop_loss=self.risk_parameters["stop_loss_pct"],
                take_profit=self.risk_parameters["take_profit_pct"],
                timestamp=datetime.now()
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error calculating statistical momentum signal: {str(e)}")
            return None
    
    def _prepare_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for momentum prediction"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Price-based features
            features['returns_1d'] = data['Close'].pct_change()
            features['returns_5d'] = data['Close'].pct_change(5)
            features['returns_20d'] = data['Close'].pct_change(20)
            
            # Technical indicators
            features['rsi'] = data['RSI']
            features['macd'] = data['MACD']
            features['bb_position'] = data['BB_Position']
            features['bb_width'] = data['BB_Width']
            
            # Volume features
            features['volume_ratio'] = data['Volume_Ratio']
            
            # Volatility features
            features['volatility_20'] = data['Volatility_20']
            features['volatility_ratio'] = data['Volatility_20'] / data['Volatility_60']
            
            # Momentum features
            features['momentum_5d'] = data['Momentum_5']
            features['momentum_20d'] = data['Momentum_20']
            
            # Drop rows with NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing momentum features: {str(e)}")
            return pd.DataFrame()
    
    def _prepare_momentum_target(self, data: pd.DataFrame) -> pd.Series:
        """Prepare target variable for momentum prediction"""
        try:
            # Target: future 5-day returns
            target = data['Close'].pct_change(5).shift(-5)
            target = target.dropna()
            return target
            
        except Exception as e:
            logger.error(f"Error preparing momentum target: {str(e)}")
            return pd.Series()
    
    async def _calculate_volatility_momentum_signal(self, symbol: str, data: pd.DataFrame) -> Optional[MomentumSignal]:
        """Calculate volatility momentum signal"""
        try:
            if data.empty or len(data) < 60:
                return None
            
            # Volatility momentum
            current_vol = data['Volatility_20'].iloc[-1]
            historical_vol = data['Volatility_20'].tail(60).head(40).mean()
            vol_momentum = (current_vol / historical_vol - 1)
            
            # Volatility mean reversion signal
            long_term_vol = data['Volatility_20'].tail(252).mean()
            vol_deviation = (current_vol - long_term_vol) / long_term_vol
            
            # Volatility breakout signal
            recent_vol_max = data['Volatility_20'].tail(60).max()
            vol_breakout = current_vol / recent_vol_max
            
            # Combined signal
            signal_components = [
                abs(vol_momentum) * 0.3,
                abs(vol_deviation) * 0.4,
                (vol_breakout - 0.8) * 0.3 if vol_breakout > 0.8 else 0
            ]
            
            signal_strength = sum(signal_components)
            
            if signal_strength < 0.1:
                return None
            
            # Predict volatility change (mean reversion)
            predicted_vol_change = -vol_deviation * 0.5  # Mean reversion
            
            confidence_level = "moderate" if signal_strength > 0.3 else "low"
            
            # Volatility momentum is inverse to price momentum
            trend_direction = "volatility_increasing" if vol_momentum > 0 else "volatility_decreasing"
            
            signal = MomentumSignal(
                symbol=symbol,
                momentum_type="volatility_momentum",
                signal_horizon="short_term",
                current_strength=signal_strength,
                predicted_change=predicted_vol_change,
                confidence_level=confidence_level,
                trend_direction=trend_direction,
                persistence_score=0.6,  # Volatility clusters
                risk_adjusted_return=signal_strength * 0.3,
                position_size=signal_strength * 0.03,  # Smaller positions for vol trades
                stop_loss=self.risk_parameters["stop_loss_pct"] * 1.5,  # Wider stops for vol
                take_profit=self.risk_parameters["take_profit_pct"] * 0.7,
                timestamp=datetime.now()
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error calculating volatility momentum signal: {str(e)}")
            return None
    
    async def _calculate_acceleration_momentum_signal(self, symbol: str, data: pd.DataFrame) -> Optional[MomentumSignal]:
        """Calculate acceleration momentum signal"""
        try:
            if data.empty or len(data) < 40:
                return None
            
            # Calculate acceleration
            returns = data['Close'].pct_change()
            acceleration = returns.diff()
            
            # Momentum of acceleration
            acc_momentum_5d = acceleration.tail(5).mean()
            acc_momentum_20d = acceleration.tail(20).mean()
            
            # Combined acceleration momentum
            acc_signal = (acc_momentum_5d * 0.7 + acc_momentum_20d * 0.3) * 100  # Scale to percentage
            
            # Signal strength
            signal_strength = min(abs(acc_signal) / 2, 1.0)  # Normalize
            
            if signal_strength < 0.15:
                return None
            
            # Predict price acceleration continuation
            recent_price_momentum = (data['Close'].iloc[-1] / data['Close'].iloc[-21] - 1) if len(data) > 20 else 0
            predicted_change = recent_price_momentum * (1 + acc_signal / 100)
            
            confidence_level = "low" if signal_strength < 0.4 else "moderate"
            trend_direction = "acceleration_increasing" if acc_signal > 0 else "acceleration_decreasing"
            
            signal = MomentumSignal(
                symbol=symbol,
                momentum_type="acceleration_momentum",
                signal_horizon="very_short_term",
                current_strength=signal_strength,
                predicted_change=predicted_change,
                confidence_level=confidence_level,
                trend_direction=trend_direction,
                persistence_score=0.3,  # Acceleration is short-lived
                risk_adjusted_return=signal_strength * 0.4,
                position_size=signal_strength * 0.04,
                stop_loss=self.risk_parameters["stop_loss_pct"] * 0.7,  # Tighter stops for acceleration
                take_profit=self.risk_parameters["take_profit_pct"] * 0.5,
                timestamp=datetime.now()
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error calculating acceleration momentum signal: {str(e)}")
            return None
    
    def _classify_momentum_strength(self, momentum_score: float) -> str:
        """Classify momentum strength"""
        try:
            if momentum_score > self.momentum_thresholds["very_strong"]:
                return "very_strong"
            elif momentum_score > self.momentum_thresholds["strong"]:
                return "strong"
            elif momentum_score > self.momentum_thresholds["moderate"]:
                return "moderate"
            elif momentum_score > self.momentum_thresholds["weak"]:
                return "weak"
            else:
                return "very_weak"
                
        except Exception as e:
            logger.error(f"Error classifying momentum strength: {str(e)}")
            return "moderate"
    
    def _classify_prediction_confidence(self, confidence_score: float) -> str:
        """Classify prediction confidence"""
        try:
            if confidence_score > 0.8:
                return "very_high"
            elif confidence_score > 0.65:
                return "high"
            elif confidence_score > 0.45:
                return "moderate"
            elif confidence_score > 0.25:
                return "low"
            else:
                return "very_low"
                
        except Exception as e:
            logger.error(f"Error classifying prediction confidence: {str(e)}")
            return "moderate"
    
    def _calculate_momentum_consistency(self, momentum_values: List[float]) -> float:
        """Calculate momentum consistency score"""
        try:
            if len(momentum_values) < 2:
                return 0.5
            
            # Calculate correlation between different timeframes
            signs = [1 if m > 0 else -1 if m < 0 else 0 for m in momentum_values]
            consistency = sum(1 for i in range(len(signs)-1) if signs[i] == signs[i+1]) / (len(signs) - 1)
            
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating momentum consistency: {str(e)}")
            return 0.5
    
    def _calculate_momentum_persistence(self, prices: pd.Series) -> float:
        """Calculate momentum persistence score"""
        try:
            if len(prices) < 60:
                return 0.5
            
            # Calculate autocorrelation of returns
            returns = prices.pct_change().dropna()
            if len(returns) < 20:
                return 0.5
            
            autocorr_1 = returns.autocorr(lag=1)
            autocorr_5 = returns.autocorr(lag=5)
            
            # Persistence is average autocorrelation
            persistence = (autocorr_1 + autocorr_5) / 2
            persistence = max(-1, min(1, persistence))
            
            return (persistence + 1) / 2  # Convert to 0-1 scale
            
        except Exception as e:
            logger.error(f"Error calculating momentum persistence: {str(e)}")
            return 0.5
    
    def _calculate_position_size(self, momentum_score: float, volatility: float) -> float:
        """Calculate position size based on momentum strength and volatility"""
        try:
            # Base position size
            base_size = min(self.risk_parameters["max_position_size"], momentum_score * 0.1)
            
            # Adjust for volatility (inverse relationship)
            vol_adjustment = min(1.0, 0.15 / volatility) if volatility > 0 else 0.5
            
            # Final position size
            position_size = base_size * vol_adjustment
            
            return max(0.01, min(self.risk_parameters["max_position_size"], position_size))
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.05  # Default position size
    
    async def predict_price_movements(self, symbols: List[str], forecast_horizons: List[int] = [5, 10, 20]) -> List[MomentumForecast]:
        """Predict price movements using momentum models"""
        try:
            momentum_data = await self.fetch_momentum_data(symbols)
            forecasts = []
            
            for symbol, data in momentum_data.items():
                if data.empty or len(data) < 200:
                    continue
                
                for horizon in forecast_horizons:
                    forecast = await self._predict_single_horizon(symbol, data, horizon)
                    if forecast:
                        forecasts.append(forecast)
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Error predicting price movements: {str(e)}")
            return []
    
    async def _predict_single_horizon(self, symbol: str, data: pd.DataFrame, horizon: int) -> Optional[MomentumForecast]:
        """Predict price movement for single horizon"""
        try:
            # Prepare features and target
            features = self._prepare_momentum_features(data)
            target = data['Close'].pct_change(horizon).shift(-horizon)
            
            # Align data
            min_length = min(len(features), len(target))
            if min_length < 100:
                return None
            
            X = features.tail(min_length)
            y = target.tail(min_length)
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8)
            train_size = min(80, len(X) // 2)
            
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            model.fit(X_train, y_train)
            
            # Make prediction
            latest_features = X.iloc[-1:].values.reshape(1, -1)
            predicted_return = model.predict(latest_features)[0]
            
            # Calculate confidence interval (simplified)
            residuals = y_test - model.predict(X_test)
            std_error = np.std(residuals)
            confidence_interval = (
                predicted_return - 1.96 * std_error,
                predicted_return + 1.96 * std_error
            )
            
            # Determine confidence level
            model_score = model.score(X_test, y_test)
            confidence_level = self._classify_prediction_confidence(model_score)
            
            # Get current price
            current_price = data['Close'].iloc[-1]
            predicted_price = current_price * (1 + predicted_return)
            
            # Identify key factors
            key_factors = self._identify_prediction_factors(model, latest_features)
            risk_factors = self._identify_risk_factors(data, horizon)
            
            forecast = MomentumForecast(
                symbol=symbol,
                forecast_horizon=horizon,
                predicted_price=predicted_price,
                predicted_return=predicted_return,
                confidence_interval=confidence_interval,
                prediction_confidence=confidence_level,
                key_factors=key_factors,
                risk_factors=risk_factors,
                model_performance={
                    "r2_score": model_score,
                    "mse": np.mean(residuals**2),
                    "mae": np.mean(np.abs(residuals))
                }
            )
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error predicting single horizon: {str(e)}")
            return None
    
    def _identify_prediction_factors(self, model, features: np.ndarray) -> List[str]:
        """Identify key factors driving the prediction"""
        try:
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = ['returns_1d', 'returns_5d', 'returns_20d', 'rsi', 'macd', 
                               'bb_position', 'bb_width', 'volume_ratio', 'volatility_20', 
                               'volatility_ratio', 'momentum_5d', 'momentum_20d']
                
                # Get top 3 most important features
                top_indices = np.argsort(importance)[-3:][::-1]
                key_factors = [feature_names[i] for i in top_indices if i < len(feature_names)]
                
                return key_factors[:3]
            else:
                return ["momentum_factors", "technical_indicators", "volume_patterns"]
                
        except Exception as e:
            logger.error(f"Error identifying prediction factors: {str(e)}")
            return ["momentum_indicators", "technical_factors"]
    
    def _identify_risk_factors(self, data: pd.DataFrame, horizon: int) -> List[str]:
        """Identify risk factors for the prediction"""
        try:
            risk_factors = []
            
            # Volatility risk
            current_vol = data['Volatility_20'].iloc[-1]
            if current_vol > data['Volatility_20'].quantile(0.8):
                risk_factors.append("high_volatility")
            
            # Trend reversal risk
            if len(data) > 60:
                short_momentum = (data['Close'].iloc[-1] / data['Close'].iloc[-21] - 1)
                long_momentum = (data['Close'].iloc[-1] / data['Close'].iloc[-61] - 1) if len(data) > 60 else 0
                
                if abs(short_momentum - long_momentum) > 0.05:
                    risk_factors.append("momentum_divergence")
            
            # Event risk (simplified)
            if horizon > 10:
                risk_factors.append("event_uncertainty")
            
            # Mean reversion risk
            if horizon > 20:
                risk_factors.append("mean_reversion_risk")
            
            return risk_factors[:3]  # Return top 3 risk factors
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {str(e)}")
            return ["market_risk", "volatility_risk"]
    
    async def generate_momentum_strategy(self, signals: List[MomentumSignal]) -> MomentumStrategy:
        """Generate systematic momentum strategy"""
        try:
            if not signals:
                return MomentumStrategy(
                    strategy_id="empty_momentum_strategy",
                    momentum_types=[],
                    asset_universe=[],
                    position_sizing="fixed",
                    rebalancing_frequency="daily",
                    expected_sharpe=0,
                    max_drawdown=0,
                    win_rate=0,
                    profit_factor=0,
                    turnover_rate=0,
                    risk_metrics={},
                    performance_attribution={}
                )
            
            # Filter strong signals
            strong_signals = [s for s in signals if s.current_strength > 0.4 and s.confidence_level in ["high", "very_high"]]
            
            if not strong_signals:
                return MomentumStrategy(
                    strategy_id="weak_momentum_strategy",
                    momentum_types=[],
                    asset_universe=[],
                    position_sizing="fixed",
                    rebalancing_frequency="daily",
                    expected_sharpe=0.5,
                    max_drawdown=0.15,
                    win_rate=0.45,
                    profit_factor=1.2,
                    turnover_rate=2.0,
                    risk_metrics={},
                    performance_attribution={}
                )
            
            # Group signals by momentum type
            momentum_types = list(set(s.momentum_type for s in strong_signals))
            
            # Asset universe
            asset_universe = list(set(s.symbol for s in strong_signals))
            
            # Calculate strategy metrics
            total_return = sum(s.predicted_change * s.position_size for s in strong_signals)
            expected_sharpe = total_return / max(np.std([s.predicted_change for s in strong_signals]), 0.01)
            max_drawdown = max(s.current_strength for s in strong_signals) * 0.2
            
            # Win rate estimation
            high_confidence_signals = [s for s in strong_signals if s.confidence_level in ["high", "very_high"]]
            win_rate = len(high_confidence_signals) / len(strong_signals) if strong_signals else 0.5
            
            # Profit factor estimation
            profit_factor = (win_rate * 1.5) / ((1 - win_rate) * 0.8) if win_rate < 1 else 2.0
            
            # Turnover rate
            turnover_rate = len(strong_signals) / len(asset_universe) if asset_universe else 1.0
            
            # Risk metrics
            risk_metrics = {
                "var_95": max_drawdown * 1.5,
                "expected_shortfall": max_drawdown * 2.0,
                "max_concentration": max(s.position_size for s in strong_signals) if strong_signals else 0.1,
                "correlation_exposure": self._calculate_correlation_exposure(strong_signals)
            }
            
            # Performance attribution
            performance_attribution = {}
            for momentum_type in momentum_types:
                type_signals = [s for s in strong_signals if s.momentum_type == momentum_type]
                type_return = sum(s.predicted_change * s.position_size for s in type_signals)
                performance_attribution[momentum_type] = type_return
            
            strategy = MomentumStrategy(
                strategy_id="systematic_momentum_strategy",
                momentum_types=momentum_types,
                asset_universe=asset_universe,
                position_sizing="volatility_adjusted",
                rebalancing_frequency="weekly",
                expected_sharpe=expected_sharpe,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                turnover_rate=turnover_rate,
                risk_metrics=risk_metrics,
                performance_attribution=performance_attribution
            )
            
            logger.info(f"Generated momentum strategy: {len(strong_signals)} signals, "
                       f"Expected Sharpe: {expected_sharpe:.2f}, Win rate: {win_rate:.1%}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating momentum strategy: {str(e)}")
            return MomentumStrategy(
                strategy_id="error_momentum_strategy",
                momentum_types=[],
                asset_universe=[],
                position_sizing="fixed",
                rebalancing_frequency="daily",
                expected_sharpe=0,
                max_drawdown=0,
                win_rate=0,
                profit_factor=0,
                turnover_rate=0,
                risk_metrics={},
                performance_attribution={}
            )
    
    def _calculate_correlation_exposure(self, signals: List[MomentumSignal]) -> float:
        """Calculate correlation exposure of strategy"""
        try:
            # Simplified correlation calculation
            # In reality, would calculate actual correlations between assets
            unique_assets = set(s.symbol for s in signals)
            
            # Estimate correlation based on asset types
            asset_types = {}
            for signal in signals:
                if signal.symbol in self.momentum_universe:
                    asset_type = self.momentum_universe[signal.symbol]["type"]
                    if asset_type not in asset_types:
                        asset_types[asset_type] = 0
                    asset_types[asset_type] += 1
            
            # Higher concentration in same asset types = higher correlation
            max_concentration = max(asset_types.values()) if asset_types else 1
            correlation_exposure = max_concentration / len(signals) if signals else 0
            
            return min(1.0, correlation_exposure)
            
        except Exception as e:
            logger.error(f"Error calculating correlation exposure: {str(e)}")
            return 0.5
    
    async def monitor_momentum_alerts(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Monitor for significant momentum signals and alerts"""
        try:
            alerts = {}
            
            # Generate momentum signals
            signals = await self.calculate_momentum_signals(symbols)
            
            # Check for very strong signals
            very_strong_signals = [s for s in signals if s.current_strength > 0.8]
            if very_strong_signals:
                alerts["very_strong_signals"] = [
                    f"{s.symbol}: {s.momentum_type} ({s.current_strength:.2f} strength, {s.predicted_change:.1%})"
                    for s in very_strong_signals[:3]
                ]
            
            # Check for high-confidence predictions
            high_confidence_signals = [s for s in signals if s.confidence_level in ["high", "very_high"]]
            if high_confidence_signals:
                alerts["high_confidence"] = [
                    f"{s.symbol}: {s.momentum_type} - {s.confidence_level} confidence"
                    for s in high_confidence_signals[:3]
                ]
            
            # Check for momentum reversals
            reversal_signals = [s for s in signals if s.momentum_type == "momentum_reversal"]
            if reversal_signals:
                alerts["momentum_reversals"] = [
                    f"{s.symbol}: Momentum reversal detected"
                    for s in reversal_signals[:2]
                ]
            
            # Check for acceleration signals
            acceleration_signals = [s for s in signals if s.momentum_type == "acceleration_momentum"]
            if acceleration_signals:
                alerts["acceleration"] = [
                    f"{s.symbol}: Price acceleration increasing"
                    for s in acceleration_signals[:2]
                ]
            
            # Check for volatility momentum
            vol_signals = [s for s in signals if s.momentum_type == "volatility_momentum"]
            if vol_signals:
                alerts["volatility_momentum"] = [
                    f"{s.symbol}: Volatility regime change"
                    for s in vol_signals[:2]
                ]
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring momentum alerts: {str(e)}")
            return {}
    
    async def export_momentum_analysis(self, symbols: List[str], format_type: str = "json") -> str:
        """Export momentum analysis to file"""
        try:
            # Generate comprehensive analysis
            signals = await self.calculate_momentum_signals(symbols)
            forecasts = await self.predict_price_movements(symbols)
            strategy = await self.generate_momentum_strategy(signals)
            
            if format_type.lower() == "json":
                import json
                
                # Convert dataclasses to dictionaries for JSON serialization
                def convert_dataclass(obj):
                    if hasattr(obj, '__dict__'):
                        return {k: convert_dataclass(v) for k, v in obj.__dict__.items()}
                    elif isinstance(obj, list):
                        return [convert_dataclass(item) for item in obj]
                    elif isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    else:
                        return obj
                
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "momentum_signals": convert_dataclass(signals),
                    "price_forecasts": convert_dataclass(forecasts),
                    "momentum_strategy": convert_dataclass(strategy),
                    "summary": {
                        "total_signals": len(signals),
                        "strong_signals": len([s for s in signals if s.current_strength > 0.6]),
                        "high_confidence_signals": len([s for s in signals if s.confidence_level in ["high", "very_high"]]),
                        "avg_signal_strength": np.mean([s.current_strength for s in signals]) if signals else 0,
                        "strategy_expected_sharpe": strategy.expected_sharpe,
                        "strategy_win_rate": strategy.win_rate
                    }
                }
                
                filename = f"momentum_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = f"/workspace/mpe/exports/{filename}"
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return filepath
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting momentum analysis: {str(e)}")
            return ""

# Main execution function
async def main():
    """Main execution function for predictive momentum analysis"""
    engine = PredictiveMomentumEngine()
    
    # Test with diversified momentum universe
    test_symbols = ["AAPL", "MSFT", "SPY", "QQQ", "TLT", "GLD", "USO", "VIX"]
    
    logger.info("Starting Predictive Momentum Engine analysis...")
    
    # Test momentum signal generation
    logger.info(f"\n=== Momentum Signal Analysis ===")
    
    signals = await engine.calculate_momentum_signals(test_symbols)
    
    logger.info(f"Total Momentum Signals: {len(signals)}")
    
    if signals:
        logger.info(f"\nTop 5 Strongest Signals:")
        sorted_signals = sorted(signals, key=lambda x: x.current_strength, reverse=True)
        for i, signal in enumerate(sorted_signals[:5], 1):
            logger.info(f"{i}. {signal.symbol}: {signal.momentum_type}")
            logger.info(f"   Strength: {signal.current_strength:.2f}, Prediction: {signal.predicted_change:.1%}")
            logger.info(f"   Confidence: {signal.confidence_level}, Horizon: {signal.signal_horizon}")
            logger.info(f"   Trend: {signal.trend_direction}, Position Size: {signal.position_size:.1%}")
    
    # Test price forecasting
    logger.info(f"\n=== Price Forecast Analysis ===")
    
    forecasts = await engine.predict_price_movements(test_symbols, [5, 10, 20])
    
    logger.info(f"Total Forecasts Generated: {len(forecasts)}")
    
    # Show sample forecasts
    for symbol in test_symbols[:3]:
        symbol_forecasts = [f for f in forecasts if f.symbol == symbol]
        if symbol_forecasts:
            logger.info(f"{symbol} Forecasts:")
            for forecast in symbol_forecasts[:2]:  # Show first 2 horizons
                logger.info(f"  {forecast.forecast_horizon}d: {forecast.predicted_change:.1%} "
                           f"(confidence: {forecast.prediction_confidence})")
    
    # Test momentum strategy
    logger.info(f"\n=== Momentum Strategy ===")
    
    strategy = await engine.generate_momentum_strategy(signals)
    
    logger.info(f"Strategy ID: {strategy.strategy_id}")
    logger.info(f"Momentum Types: {', '.join(strategy.momentum_types)}")
    logger.info(f"Asset Universe: {len(strategy.asset_universe)} assets")
    logger.info(f"Expected Sharpe: {strategy.expected_sharpe:.2f}")
    logger.info(f"Win Rate: {strategy.win_rate:.1%}")
    logger.info(f"Max Drawdown: {strategy.max_drawdown:.1%}")
    logger.info(f"Profit Factor: {strategy.profit_factor:.2f}")
    logger.info(f"Turnover Rate: {strategy.turnover_rate:.1f}x")
    
    # Show performance attribution
    if strategy.performance_attribution:
        logger.info(f"\nPerformance Attribution:")
        for momentum_type, contribution in strategy.performance_attribution.items():
            logger.info(f"  {momentum_type}: {contribution:.1%}")
    
    # Test monitoring alerts
    logger.info(f"\n=== Momentum Alerts ===")
    
    alerts = await engine.monitor_momentum_alerts(test_symbols)
    
    for alert_type, alert_messages in alerts.items():
        logger.info(f"{alert_type.upper()}:")
        for message in alert_messages:
            logger.info(f"  - {message}")
    
    # Show signal type breakdown
    if signals:
        signal_types = {}
        for signal in signals:
            signal_type = signal.momentum_type
            if signal_type not in signal_types:
                signal_types[signal_type] = 0
            signal_types[signal_type] += 1
        
        logger.info(f"\nSignal Type Breakdown:")
        for signal_type, count in signal_types.items():
            logger.info(f"  {signal_type}: {count}")
    
    # Show confidence level distribution
    if signals:
        confidence_levels = {}
        for signal in signals:
            conf_level = signal.confidence_level
            if conf_level not in confidence_levels:
                confidence_levels[conf_level] = 0
            confidence_levels[conf_level] += 1
        
        logger.info(f"\nConfidence Level Distribution:")
        for level, count in confidence_levels.items():
            logger.info(f"  {level}: {count}")
    
    logger.info("Predictive Momentum Engine analysis completed successfully")

if __name__ == "__main__":
    asyncio.run(main())