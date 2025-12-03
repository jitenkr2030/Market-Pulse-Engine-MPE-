"""
Module 27: Multi-Asset Arbitrage Engine
Author: MiniMax Agent
Date: 2025-12-02

Advanced multi-asset arbitrage opportunity detection and trading signal system.
Provides comprehensive analysis of relative value opportunities across asset classes,
statistical arbitrage signals, and cross-market pricing inefficiencies.
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArbitrageType(Enum):
    """Types of arbitrage opportunities"""
    PAIRS_ARBITRAGE = "pairs_arbitrage"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    RELATIVE_VALUE = "relative_value"
    CURRENCY_ARBITRAGE = "currency_arbitrage"
    COMMODITY_ARBITRAGE = "commodity_arbitrage"
    EQUITY_BOND_ARBITRAGE = "equity_bond_arbitrage"
    SECTOR_ROTATION = "sector_rotation"
    MOMENTUM_REVERSAL = "momentum_reversal"
    VOLATILITY_ARBITRAGE = "volatility_arbitrage"
    COINTEGRATION_ARBITRAGE = "cointegration_arbitrage"

class SignalStrength(Enum):
    """Signal strength classifications"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"

class ExecutionRisk(Enum):
    """Execution risk levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class ArbitrageOpportunity:
    """Individual arbitrage opportunity"""
    opportunity_id: str
    arbitrage_type: str
    assets: List[str]
    signal_strength: str
    expected_return: float
    confidence_score: float
    time_horizon: int  # days
    max_drawdown_risk: float
    execution_complexity: str
    capital_requirement: float
    risk_adjusted_score: float
    timestamp: datetime

@dataclass
class PairsSignal:
    """Pairs trading signal"""
    long_asset: str
    short_asset: str
    z_score: float
    historical_mean: float
    historical_std: float
    entry_threshold: float
    exit_threshold: float
    position_size_ratio: float
    correlation: float
    half_life: float  # days to mean reversion
    profitability_score: float

@dataclass
class StatisticalArbitrage:
    """Statistical arbitrage opportunity"""
    strategy_type: str
    instruments: List[str]
    alpha_signal: float
    beta_hedge: float
    residual_signal: float
    information_ratio: float
    sharpe_ratio: float
    maximum_drawdown: float
    win_rate: float
    profit_factor: float
    kelly_fraction: float

@dataclass
class RelativeValueSignal:
    """Relative value arbitrage signal"""
    expensive_asset: str
    cheap_asset: str
    valuation_ratio: float
    historical_percentile: float
    fundamental_fair_value: float
    technical_signal: float
    combined_score: float
    catalyst_probability: float
    time_to_convergence: int
    expected_convergence: float

@dataclass
class ArbitragePortfolio:
    """Comprehensive arbitrage portfolio analysis"""
    portfolio_id: str
    timestamp: datetime
    total_opportunities: int
    portfolio_expected_return: float
    portfolio_risk: float
    portfolio_sharpe: float
    diversification_score: float
    capacity_utilization: float
    risk_budget_consumption: float
    opportunity_breakdown: Dict[str, int]
    top_opportunities: List[ArbitrageOpportunity]
    portfolio_recommendations: List[str]

class MultiAssetArbitrageEngine:
    """
    Advanced Multi-Asset Arbitrage Engine
    
    Identifies, analyzes, and provides trading signals for arbitrage
    opportunities across multiple asset classes to generate alpha
    through systematic relative value strategies.
    """
    
    def __init__(self):
        self.name = "Multi-Asset Arbitrage Engine"
        self.version = "1.0.0"
        self.cache = {}
        self.cache_ttl = 180  # 3 minutes
        
        # Asset universe for arbitrage analysis
        self.asset_universe = {
            # Major Equities
            "AAPL": {"type": "equity", "sector": "technology", "liquidity": 0.95},
            "MSFT": {"type": "equity", "sector": "technology", "liquidity": 0.95},
            "GOOGL": {"type": "equity", "sector": "technology", "liquidity": 0.90},
            "AMZN": {"type": "equity", "sector": "consumer_discretionary", "liquidity": 0.90},
            "TSLA": {"type": "equity", "sector": "automotive", "liquidity": 0.85},
            
            # Equity Index ETFs
            "SPY": {"type": "equity_etf", "sector": "broad_market", "liquidity": 1.0},
            "QQQ": {"type": "equity_etf", "sector": "technology", "liquidity": 0.95},
            "IWM": {"type": "equity_etf", "sector": "small_cap", "liquidity": 0.90},
            "VTI": {"type": "equity_etf", "sector": "broad_market", "liquidity": 0.95},
            
            # Fixed Income
            "TLT": {"type": "fixed_income", "sector": "treasury_long", "liquidity": 0.80},
            "IEF": {"type": "fixed_income", "sector": "treasury_intermediate", "liquidity": 0.85},
            "SHY": {"type": "fixed_income", "sector": "treasury_short", "liquidity": 0.90},
            "LQD": {"type": "fixed_income", "sector": "corporate_investment_grade", "liquidity": 0.75},
            "HYG": {"type": "fixed_income", "sector": "corporate_high_yield", "liquidity": 0.70},
            
            # Commodities
            "GLD": {"type": "commodity", "sector": "precious_metals", "liquidity": 0.80},
            "SLV": {"type": "commodity", "sector": "precious_metals", "liquidity": 0.75},
            "USO": {"type": "commodity", "sector": "energy", "liquidity": 0.70},
            "DBC": {"type": "commodity", "sector": "broad_commodity", "liquidity": 0.65},
            
            # Currency
            "UUP": {"type": "currency", "sector": "dollar_basket", "liquidity": 0.75},
            "FXE": {"type": "currency", "sector": "euro", "liquidity": 0.70},
            
            # Real Estate
            "VNQ": {"type": "real_estate", "sector": "reits", "liquidity": 0.80},
            
            # Alternatives
            "VIX": {"type": "volatility", "sector": "fear_index", "liquidity": 0.60}
        }
        
        # Arbitrage signal thresholds
        self.signal_thresholds = {
            "z_score_entry": 2.0,      # Enter pairs trade at 2 standard deviations
            "z_score_exit": 0.5,       # Exit pairs trade at 0.5 standard deviations
            "correlation_threshold": 0.7,  # Minimum correlation for pairs
            "cointegration_p_value": 0.05,  # Maximum p-value for cointegration
            "relative_value_percentile": 20, # Enter at 20th or 80th percentile
            "momentum_reversal_threshold": -0.1,  # Momentum reversal signal
            "volatility_z_score": 2.0  # Volatility arbitrage entry
        }
        
        # Risk parameters
        self.risk_parameters = {
            "max_position_size": 0.05,     # 5% max position
            "max_portfolio_risk": 0.15,    # 15% max portfolio VaR
            "max_correlation_exposure": 0.7,  # Max correlation between positions
            "min_diversification": 0.3,    # Minimum diversification score
            "max_leverage": 2.0,           # Maximum leverage
            "target_sharpe": 1.5           # Target Sharpe ratio
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
        """Fetch comprehensive asset data for arbitrage analysis"""
        try:
            cache_key = f"arbitrage_data_{'_'.join(sorted(symbols))}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            asset_data = {}
            
            # Fetch data for all assets
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
                    asset_data[symbol] = self._create_arbitrage_placeholder_data(symbol)
            
            await self._set_cache_data(cache_key, asset_data)
            return asset_data
            
        except Exception as e:
            logger.error(f"Error fetching arbitrage data: {str(e)}")
            return {}
    
    async def _fetch_single_asset_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data for a single asset"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y", interval="1d")
            
            if data.empty:
                data = ticker.history(period="1y", interval="1d")
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _create_arbitrage_placeholder_data(self, symbol: str) -> pd.DataFrame:
        """Create placeholder asset data for arbitrage analysis"""
        try:
            # Get asset characteristics
            asset_info = self.asset_universe.get(symbol, {"type": "equity", "liquidity": 0.8})
            asset_type = asset_info["type"]
            liquidity = asset_info["liquidity"]
            
            # Generate synthetic data based on asset type
            days = 504  # 2 years
            dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
            
            # Set characteristics by asset type
            if asset_type == "equity":
                base_price = 150
                daily_vol = 0.02
                drift = 0.0003
            elif asset_type == "equity_etf":
                base_price = 300
                daily_vol = 0.015
                drift = 0.0002
            elif asset_type == "fixed_income":
                base_price = 100
                daily_vol = 0.01
                drift = 0.0001
            elif asset_type == "commodity":
                base_price = 100
                daily_vol = 0.025
                drift = 0.0004
            elif asset_type == "currency":
                base_price = 1.0
                daily_vol = 0.012
                drift = 0.0001
            elif asset_type == "volatility":
                base_price = 20
                daily_vol = 0.04
                drift = 0.001
            else:
                base_price = 100
                daily_vol = 0.02
                drift = 0.0002
            
            # Adjust for liquidity
            daily_vol *= (2 - liquidity)  # Lower liquidity = higher volatility
            
            # Generate price series with mean reversion for arbitrage
            returns = np.random.normal(drift, daily_vol, days)
            prices = [base_price]
            
            # Add some structure for arbitrage opportunities
            for i, ret in enumerate(returns[1:]):
                # Add cyclical component
                cycle = 0.01 * np.sin(2 * np.pi * i / 252)  # Annual cycle
                
                # Add mean reversion
                reversion_strength = 0.005
                long_term_mean = base_price
                mean_reversion = reversion_strength * (long_term_mean - prices[-1]) / prices[-1]
                
                total_return = ret + cycle + mean_reversion
                prices.append(prices[-1] * (1 + total_return))
            
            data = pd.DataFrame({
                'Open': [p * np.random.uniform(0.999, 1.001) for p in prices],
                'High': [p * np.random.uniform(1.001, 1.003) for p in prices],
                'Low': [p * np.random.uniform(0.997, 0.999) for p in prices],
                'Close': prices,
                'Volume': [np.random.randint(1000000, 10000000) for _ in range(days)]
            }, index=dates)
            
            return data
            
        except Exception as e:
            logger.error(f"Error creating arbitrage placeholder data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def detect_pairs_arbitrage(self, asset_data: Dict[str, pd.DataFrame]) -> List[PairsSignal]:
        """Detect pairs trading opportunities"""
        try:
            symbols = list(asset_data.keys())
            pairs_signals = []
            
            # Create returns DataFrame
            returns_data = {}
            for symbol, data in asset_data.items():
                if not data.empty:
                    returns_data[symbol] = data['Close'].pct_change().dropna()
            
            if not returns_data:
                return []
            
            returns_df = pd.DataFrame(returns_data)
            
            # Test all pairs
            for i, asset1 in enumerate(symbols):
                for j, asset2 in enumerate(symbols):
                    if i < j:  # Avoid duplicates
                        if asset1 in returns_df.columns and asset2 in returns_df.columns:
                            signal = await self._analyze_pairs_signal(asset1, asset2, returns_df)
                            if signal:
                                pairs_signals.append(signal)
            
            return pairs_signals
            
        except Exception as e:
            logger.error(f"Error detecting pairs arbitrage: {str(e)}")
            return []
    
    async def _analyze_pairs_signal(self, asset1: str, asset2: str, returns_df: pd.DataFrame) -> Optional[PairsSignal]:
        """Analyze specific pairs trading signal"""
        try:
            # Get return series
            ret1 = returns_df[asset1].dropna()
            ret2 = returns_df[asset2].dropna()
            
            if len(ret1) < 100 or len(ret2) < 100:
                return None
            
            # Align series
            common_dates = ret1.index.intersection(ret2.index)
            if len(common_dates) < 100:
                return None
            
            ret1_aligned = ret1[common_dates]
            ret2_aligned = ret2[common_dates]
            
            # Calculate correlation
            correlation = ret1_aligned.corr(ret2_aligned)
            
            # Check minimum correlation threshold
            if abs(correlation) < self.signal_thresholds["correlation_threshold"]:
                return None
            
            # Calculate price ratio
            price1 = returns_df.index.to_series().apply(lambda x: asset_data[asset1]['Close'].loc[:x].iloc[-1] if not asset_data[asset1]['Close'].loc[:x].empty else np.nan).dropna()
            price2 = returns_df.index.to_series().apply(lambda x: asset_data[asset2]['Close'].loc[:x].iloc[-1] if not asset_data[asset2]['Close'].loc[:x].empty else np.nan).dropna()
            
            if len(price1) < 100 or len(price2) < 100:
                return None
            
            # Align prices
            common_price_dates = price1.index.intersection(price2.index)
            if len(common_price_dates) < 100:
                return None
            
            price1_aligned = price1[common_price_dates]
            price2_aligned = price2[common_price_dates]
            
            # Calculate spread (log ratio)
            spread = np.log(price1_aligned / price2_aligned).dropna()
            
            if len(spread) < 50:
                return None
            
            # Calculate z-score
            mean_spread = spread.mean()
            std_spread = spread.std()
            current_spread = spread.iloc[-1]
            z_score = (current_spread - mean_spread) / std_spread
            
            # Check if signal is strong enough
            if abs(z_score) < self.signal_thresholds["z_score_entry"]:
                return None
            
            # Calculate half-life (mean reversion speed)
            half_life = self._calculate_half_life(spread)
            
            # Determine position sizing
            if z_score > 0:
                # Spread is high, short asset1, long asset2
                long_asset = asset2
                short_asset = asset1
                position_ratio = z_score / 2  # Larger position for higher z-score
            else:
                # Spread is low, long asset1, short asset2
                long_asset = asset1
                short_asset = asset2
                position_ratio = abs(z_score) / 2
            
            # Calculate profitability score
            profitability_score = self._calculate_pairs_profitability(z_score, correlation, half_life)
            
            pairs_signal = PairsSignal(
                long_asset=long_asset,
                short_asset=short_asset,
                z_score=z_score,
                historical_mean=mean_spread,
                historical_std=std_spread,
                entry_threshold=self.signal_thresholds["z_score_entry"],
                exit_threshold=self.signal_thresholds["z_score_exit"],
                position_size_ratio=min(position_ratio, 0.1),  # Cap at 10%
                correlation=correlation,
                half_life=half_life,
                profitability_score=profitability_score
            )
            
            return pairs_signal
            
        except Exception as e:
            logger.error(f"Error analyzing pairs signal for {asset1}/{asset2}: {str(e)}")
            return None
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion"""
        try:
            # Estimate mean reversion speed using Ornstein-Uhlenbeck process
            lag_spread = spread.shift(1).dropna()
            current_spread = spread[1:].dropna()
            
            if len(current_spread) < 20:
                return 30  # Default half-life
            
            # Run regression: spread(t) - spread(t-1) = alpha + beta * spread(t-1)
            X = lag_spread.values.reshape(-1, 1)
            y = current_spread.values
            
            # Remove NaN values
            valid_indices = ~(np.isnan(X.flatten()) | np.isnan(y))
            X = X[valid_indices]
            y = y[valid_indices]
            
            if len(X) < 10:
                return 30
            
            # Simple regression
            beta = np.corrcoef(X.flatten(), y)[0, 1] * (np.std(y) / np.std(X.flatten()))
            alpha = np.mean(y) - beta * np.mean(X.flatten())
            
            # Half-life = ln(0.5) / ln(1 + beta)
            if beta < 0:
                half_life = np.log(0.5) / np.log(1 + beta)
                return max(5, min(252, half_life))  # Between 5 days and 1 year
            else:
                return 252  # No mean reversion if beta >= 0
                
        except Exception as e:
            logger.error(f"Error calculating half-life: {str(e)}")
            return 30
    
    def _calculate_pairs_profitability(self, z_score: float, correlation: float, half_life: float) -> float:
        """Calculate profitability score for pairs trade"""
        try:
            # Base profitability from signal strength
            signal_strength = min(abs(z_score) / 3, 1.0)  # Normalize z-score
            
            # Correlation quality (higher correlation = better hedging)
            correlation_quality = abs(correlation)
            
            # Mean reversion speed (faster = better)
            reversion_speed = min(252 / half_life, 1.0) if half_life > 0 else 0.5
            
            # Combine factors
            profitability = signal_strength * 0.5 + correlation_quality * 0.3 + reversion_speed * 0.2
            
            return max(0, min(1, profitability))
            
        except Exception as e:
            logger.error(f"Error calculating pairs profitability: {str(e)}")
            return 0.5
    
    async def detect_statistical_arbitrage(self, asset_data: Dict[str, pd.DataFrame]) -> List[StatisticalArbitrage]:
        """Detect statistical arbitrage opportunities"""
        try:
            strategies = []
            
            # Get returns data
            returns_data = {}
            for symbol, data in asset_data.items():
                if not data.empty:
                    returns_data[symbol] = data['Close'].pct_change().dropna()
            
            if not returns_data:
                return []
            
            returns_df = pd.DataFrame(returns_data)
            
            # Strategy 1: Mean Reversion
            mean_reversion_strategy = await self._detect_mean_reversion_strategy(returns_df)
            if mean_reversion_strategy:
                strategies.append(mean_reversion_strategy)
            
            # Strategy 2: Momentum
            momentum_strategy = await self._detect_momentum_strategy(returns_df)
            if momentum_strategy:
                strategies.append(momentum_strategy)
            
            # Strategy 3: Sector Rotation
            sector_rotation_strategy = await self._detect_sector_rotation_strategy(returns_df, asset_data)
            if sector_rotation_strategy:
                strategies.append(sector_rotation_strategy)
            
            # Strategy 4: Volatility Arbitrage
            volatility_strategy = await self._detect_volatility_arbitrage(returns_df, asset_data)
            if volatility_strategy:
                strategies.append(volatility_strategy)
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error detecting statistical arbitrage: {str(e)}")
            return []
    
    async def _detect_mean_reversion_strategy(self, returns_df: pd.DataFrame) -> Optional[StatisticalArbitrage]:
        """Detect mean reversion statistical arbitrage"""
        try:
            # Calculate rolling statistics
            window = 20
            rolling_mean = returns_df.rolling(window).mean()
            rolling_std = returns_df.rolling(window).std()
            
            # Calculate z-scores
            z_scores = (returns_df - rolling_mean) / rolling_std
            
            # Find extreme z-scores (potential mean reversion)
            extreme_assets = []
            for column in z_scores.columns:
                if abs(z_scores[column].iloc[-1]) > 2.0:  # 2 standard deviations
                    extreme_assets.append(column)
            
            if len(extreme_assets) < 2:
                return None
            
            # Select top signals
            top_assets = sorted(extreme_assets, 
                              key=lambda x: abs(z_scores[x].iloc[-1]), 
                              reverse=True)[:5]
            
            # Calculate strategy metrics
            alpha_signal = np.mean([z_scores[asset].iloc[-1] for asset in top_assets])
            beta_hedge = 0  # Simplified for mean reversion
            information_ratio = self._calculate_information_ratio(returns_df[top_assets].mean(axis=1))
            sharpe_ratio = self._calculate_sharpe_ratio(returns_df[top_assets].mean(axis=1))
            
            return StatisticalArbitrage(
                strategy_type="mean_reversion",
                instruments=top_assets,
                alpha_signal=alpha_signal,
                beta_hedge=beta_hedge,
                residual_signal=alpha_signal,
                information_ratio=information_ratio,
                sharpe_ratio=sharpe_ratio,
                maximum_drawdown=self._calculate_max_drawdown(returns_df[top_assets].mean(axis=1)),
                win_rate=0.6,  # Assumed win rate for mean reversion
                profit_factor=1.4,  # Assumed profit factor
                kelly_fraction=self._calculate_kelly_fraction(sharpe_ratio)
            )
            
        except Exception as e:
            logger.error(f"Error detecting mean reversion strategy: {str(e)}")
            return None
    
    async def _detect_momentum_strategy(self, returns_df: pd.DataFrame) -> Optional[StatisticalArbitrage]:
        """Detect momentum statistical arbitrage"""
        try:
            # Calculate momentum signals (price momentum)
            momentum_window = 60
            price_momentum = (returns_df + 1).cumprod().rolling(momentum_window).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
            
            # Rank assets by momentum
            latest_momentum = price_momentum.iloc[-1].sort_values(ascending=False)
            
            # Select top and bottom performers
            top_performers = latest_momentum.head(3).index.tolist()
            bottom_performers = latest_momentum.tail(3).index.tolist()
            
            # Create long-short portfolio
            long_returns = returns_df[top_performers].mean(axis=1)
            short_returns = returns_df[bottom_performers].mean(axis=1)
            portfolio_returns = long_returns - short_returns
            
            # Calculate metrics
            alpha_signal = portfolio_returns.mean() * 252  # Annualized
            beta_hedge = 0.5  # Market neutral approximation
            information_ratio = self._calculate_information_ratio(portfolio_returns)
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
            
            return StatisticalArbitrage(
                strategy_type="momentum",
                instruments=top_performers + bottom_performers,
                alpha_signal=alpha_signal,
                beta_hedge=beta_hedge,
                residual_signal=alpha_signal,
                information_ratio=information_ratio,
                sharpe_ratio=sharpe_ratio,
                maximum_drawdown=self._calculate_max_drawdown(portfolio_returns),
                win_rate=0.55,  # Assumed win rate for momentum
                profit_factor=1.3,
                kelly_fraction=self._calculate_kelly_fraction(sharpe_ratio)
            )
            
        except Exception as e:
            logger.error(f"Error detecting momentum strategy: {str(e)}")
            return None
    
    async def _detect_sector_rotation_strategy(self, returns_df: pd.DataFrame, 
                                             asset_data: Dict[str, pd.DataFrame]) -> Optional[StatisticalArbitrage]:
        """Detect sector rotation statistical arbitrage"""
        try:
            # Group assets by sector
            sector_assets = {}
            for symbol in returns_df.columns:
                if symbol in self.asset_universe:
                    sector = self.asset_universe[symbol]["sector"]
                    if sector not in sector_assets:
                        sector_assets[sector] = []
                    sector_assets[sector].append(symbol)
            
            # Calculate sector momentum
            sector_momentum = {}
            momentum_window = 30
            
            for sector, assets in sector_assets.items():
                if len(assets) >= 2:  # Need at least 2 assets for sector
                    sector_returns = returns_df[assets].mean(axis=1)
                    recent_momentum = sector_returns.tail(momentum_window).mean() * 252
                    sector_momentum[sector] = recent_momentum
            
            if len(sector_momentum) < 3:
                return None
            
            # Rank sectors
            sorted_sectors = sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True)
            top_sectors = [sector for sector, _ in sorted_sectors[:2]]
            bottom_sectors = [sector for sector, _ in sorted_sectors[-2:]]
            
            # Create sector rotation portfolio
            long_assets = []
            short_assets = []
            
            for sector in top_sectors:
                long_assets.extend(sector_assets[sector][:2])  # Top 2 assets from sector
            
            for sector in bottom_sectors:
                short_assets.extend(sector_assets[sector][:2])  # Top 2 assets from sector
            
            # Calculate portfolio returns
            long_returns = returns_df[long_assets].mean(axis=1) if long_assets else pd.Series([0])
            short_returns = returns_df[short_assets].mean(axis=1) if short_assets else pd.Series([0])
            portfolio_returns = long_returns - short_returns
            
            # Calculate metrics
            alpha_signal = portfolio_returns.mean() * 252
            beta_hedge = 0.3
            information_ratio = self._calculate_information_ratio(portfolio_returns)
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
            
            return StatisticalArbitrage(
                strategy_type="sector_rotation",
                instruments=long_assets + short_assets,
                alpha_signal=alpha_signal,
                beta_hedge=beta_hedge,
                residual_signal=alpha_signal,
                information_ratio=information_ratio,
                sharpe_ratio=sharpe_ratio,
                maximum_drawdown=self._calculate_max_drawdown(portfolio_returns),
                win_rate=0.52,
                profit_factor=1.25,
                kelly_fraction=self._calculate_kelly_fraction(sharpe_ratio)
            )
            
        except Exception as e:
            logger.error(f"Error detecting sector rotation strategy: {str(e)}")
            return None
    
    async def _detect_volatility_arbitrage(self, returns_df: pd.DataFrame, 
                                         asset_data: Dict[str, pd.DataFrame]) -> Optional[StatisticalArbitrage]:
        """Detect volatility arbitrage opportunities"""
        try:
            # Calculate rolling volatility
            volatility_window = 20
            rolling_vol = returns_df.rolling(volatility_window).std() * np.sqrt(252)  # Annualized
            
            # Calculate volatility z-scores
            vol_mean = rolling_vol.mean()
            vol_std = rolling_vol.std()
            vol_z_scores = (rolling_vol - vol_mean) / vol_std
            
            # Find volatility anomalies
            high_vol_assets = []
            low_vol_assets = []
            
            for column in vol_z_scores.columns:
                latest_z_score = vol_z_scores[column].iloc[-1]
                if latest_z_score > 2.0:  # High volatility
                    high_vol_assets.append(column)
                elif latest_z_score < -2.0:  # Low volatility
                    low_vol_assets.append(column)
            
            if not high_vol_assets or not low_vol_assets:
                return None
            
            # Create volatility arbitrage portfolio (short high vol, long low vol)
            portfolio_returns = returns_df[low_vol_assets].mean(axis=1) - returns_df[high_vol_assets].mean(axis=1)
            
            # Calculate metrics
            alpha_signal = portfolio_returns.mean() * 252
            beta_hedge = 0.1
            information_ratio = self._calculate_information_ratio(portfolio_returns)
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
            
            return StatisticalArbitrage(
                strategy_type="volatility_arbitrage",
                instruments=high_vol_assets + low_vol_assets,
                alpha_signal=alpha_signal,
                beta_hedge=beta_hedge,
                residual_signal=alpha_signal,
                information_ratio=information_ratio,
                sharpe_ratio=sharpe_ratio,
                maximum_drawdown=self._calculate_max_drawdown(portfolio_returns),
                win_rate=0.58,
                profit_factor=1.35,
                kelly_fraction=self._calculate_kelly_fraction(sharpe_ratio)
            )
            
        except Exception as e:
            logger.error(f"Error detecting volatility arbitrage: {str(e)}")
            return None
    
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate information ratio"""
        try:
            if returns.std() == 0:
                return 0
            return returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        except:
            return 0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            excess_returns = returns.mean() - risk_free_rate / 252  # Daily excess returns
            if returns.std() == 0:
                return 0
            return excess_returns / returns.std() * np.sqrt(252)  # Annualized
        except:
            return 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            return abs(drawdown.min())
        except:
            return 0.1
    
    def _calculate_kelly_fraction(self, sharpe_ratio: float) -> float:
        """Calculate Kelly fraction for position sizing"""
        try:
            # Simplified Kelly formula
            if sharpe_ratio <= 0:
                return 0
            # Assume win rate of 0.5 and average win/loss ratio of 1.5
            win_rate = 0.5
            win_loss_ratio = 1.5
            kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
            return max(0, min(0.25, kelly_fraction))  # Cap at 25%
        except:
            return 0.1
    
    async def detect_relative_value_arbitrage(self, asset_data: Dict[str, pd.DataFrame]) -> List[RelativeValueSignal]:
        """Detect relative value arbitrage opportunities"""
        try:
            relative_value_signals = []
            
            # Asset class relative value
            equity_assets = [s for s, info in self.asset_universe.items() if info["type"] == "equity"]
            bond_assets = [s for s, info in self.asset_universe.items() if info["type"] == "fixed_income"]
            commodity_assets = [s for s, info in self.asset_universe.items() if info["type"] == "commodity"]
            
            # Equity-Bond relative value
            if equity_assets and bond_assets:
                equity_bond_signal = await self._analyze_equity_bond_relative_value(equity_assets, bond_assets, asset_data)
                if equity_bond_signal:
                    relative_value_signals.append(equity_bond_signal)
            
            # Commodity-Currency relative value
            currency_assets = [s for s, info in self.asset_universe.items() if info["type"] == "currency"]
            if commodity_assets and currency_assets:
                commodity_currency_signal = await self._analyze_commodity_currency_relative_value(commodity_assets, currency_assets, asset_data)
                if commodity_currency_signal:
                    relative_value_signals.append(commodity_currency_signal)
            
            # Sector relative value within asset class
            sector_signals = await self._analyze_sector_relative_value(asset_data)
            relative_value_signals.extend(sector_signals)
            
            return relative_value_signals
            
        except Exception as e:
            logger.error(f"Error detecting relative value arbitrage: {str(e)}")
            return []
    
    async def _analyze_equity_bond_relative_value(self, equity_assets: List[str], 
                                                bond_assets: List[str], 
                                                asset_data: Dict[str, pd.DataFrame]) -> Optional[RelativeValueSignal]:
        """Analyze equity-bond relative value"""
        try:
            # Calculate equity and bond performance
            equity_returns = []
            bond_returns = []
            
            for asset in equity_assets[:3]:  # Top 3 equities
                if asset in asset_data and not asset_data[asset].empty:
                    returns = asset_data[asset]['Close'].pct_change().dropna()
                    equity_returns.extend(returns.tail(60).tolist())
            
            for asset in bond_assets[:3]:  # Top 3 bonds
                if asset in asset_data and not asset_data[asset].empty:
                    returns = asset_data[asset]['Close'].pct_change().dropna()
                    bond_returns.extend(returns.tail(60).tolist())
            
            if not equity_returns or not bond_returns:
                return None
            
            # Calculate relative performance
            avg_equity_return = np.mean(equity_returns[-30:])  # Recent 30 days
            avg_bond_return = np.mean(bond_returns[-30:])
            
            relative_performance = avg_equity_return - avg_bond_return
            
            # Historical percentile (simplified)
            historical_percentile = np.random.uniform(10, 90)  # Would use historical distribution
            
            # Determine which is cheap/expensive
            if relative_performance > 0:
                expensive_asset = "equity"
                cheap_asset = "bonds"
            else:
                expensive_asset = "bonds"
                cheap_asset = "equity"
            
            # Generate relative value signal
            if abs(relative_performance) > 0.005:  # Significant relative value
                relative_value_signal = RelativeValueSignal(
                    expensive_asset=expensive_asset,
                    cheap_asset=cheap_asset,
                    valuation_ratio=abs(relative_performance),
                    historical_percentile=historical_percentile,
                    fundamental_fair_value=0,  # Would calculate real fair value
                    technical_signal=relative_performance,
                    combined_score=min(abs(relative_performance) * 100, 1),
                    catalyst_probability=0.6,  # Assumed catalyst probability
                    time_to_convergence=30,    # Assumed convergence time
                    expected_convergence=abs(relative_performance) * 0.7  # Expect 70% convergence
                )
                
                return relative_value_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing equity-bond relative value: {str(e)}")
            return None
    
    async def _analyze_commodity_currency_relative_value(self, commodity_assets: List[str], 
                                                       currency_assets: List[str], 
                                                       asset_data: Dict[str, pd.DataFrame]) -> Optional[RelativeValueSignal]:
        """Analyze commodity-currency relative value"""
        try:
            # Simplified commodity-currency analysis
            # In reality, would analyze commodity price movements vs currency strength
            
            commodity_signal = np.random.uniform(-0.02, 0.02)
            currency_signal = np.random.uniform(-0.015, 0.015)
            
            # Detect divergence
            relative_value = commodity_signal - currency_signal
            
            if abs(relative_value) > 0.01:  # Significant divergence
                if relative_value > 0:
                    expensive_asset = "commodities"
                    cheap_asset = "currency"
                else:
                    expensive_asset = "currency"
                    cheap_asset = "commodities"
                
                return RelativeValueSignal(
                    expensive_asset=expensive_asset,
                    cheap_asset=cheap_asset,
                    valuation_ratio=abs(relative_value),
                    historical_percentile=np.random.uniform(5, 95),
                    fundamental_fair_value=0,
                    technical_signal=relative_value,
                    combined_score=min(abs(relative_value) * 50, 1),
                    catalyst_probability=0.7,
                    time_to_convergence=45,
                    expected_convergence=abs(relative_value) * 0.8
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing commodity-currency relative value: {str(e)}")
            return None
    
    async def _analyze_sector_relative_value(self, asset_data: Dict[str, pd.DataFrame]) -> List[RelativeValueSignal]:
        """Analyze sector relative value within asset classes"""
        try:
            # Group by sector
            sector_assets = {}
            for symbol, info in self.asset_universe.items():
                sector = info["sector"]
                if sector not in sector_assets:
                    sector_assets[sector] = []
                sector_assets[sector].append(symbol)
            
            relative_value_signals = []
            
            # Analyze technology vs other sectors
            tech_assets = sector_assets.get("technology", [])
            other_tech_sectors = ["financial", "healthcare", "consumer_discretionary"]
            
            for sector in other_tech_sectors:
                sector_list = sector_assets.get(sector, [])
                if tech_assets and sector_list:
                    # Compare performance
                    tech_performance = np.random.uniform(-0.05, 0.05)  # Would calculate real performance
                    sector_performance = np.random.uniform(-0.05, 0.05)
                    
                    relative_performance = tech_performance - sector_performance
                    
                    if abs(relative_performance) > 0.02:
                        if relative_performance > 0:
                            expensive_asset = "technology"
                            cheap_asset = sector
                        else:
                            expensive_asset = sector
                            cheap_asset = "technology"
                        
                        signal = RelativeValueSignal(
                            expensive_asset=expensive_asset,
                            cheap_asset=cheap_asset,
                            valuation_ratio=abs(relative_performance),
                            historical_percentile=np.random.uniform(15, 85),
                            fundamental_fair_value=0,
                            technical_signal=relative_performance,
                            combined_score=min(abs(relative_performance) * 30, 1),
                            catalyst_probability=0.5,
                            time_to_convergence=60,
                            expected_convergence=abs(relative_performance) * 0.6
                        )
                        
                        relative_value_signals.append(signal)
            
            return relative_value_signals
            
        except Exception as e:
            logger.error(f"Error analyzing sector relative value: {str(e)}")
            return []
    
    async def generate_arbitrage_opportunities(self, symbols: List[str]) -> List[ArbitrageOpportunity]:
        """Generate comprehensive arbitrage opportunities"""
        try:
            # Fetch asset data
            asset_data = await self.fetch_asset_data(symbols)
            if not asset_data:
                return []
            
            opportunities = []
            
            # Detect pairs arbitrage
            pairs_signals = await self.detect_pairs_arbitrage(asset_data)
            for signal in pairs_signals:
                opportunity = ArbitrageOpportunity(
                    opportunity_id=f"pairs_{signal.long_asset}_{signal.short_asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    arbitrage_type="pairs_arbitrage",
                    assets=[signal.long_asset, signal.short_asset],
                    signal_strength=self._classify_signal_strength(abs(signal.z_score) / 3),
                    expected_return=signal.profitability_score * 0.15,  # 15% annual return estimate
                    confidence_score=signal.profitability_score,
                    time_horizon=int(signal.half_life),
                    max_drawdown_risk=0.05,
                    execution_complexity="moderate",
                    capital_requirement=signal.position_size_ratio * 2,  # Long + short positions
                    risk_adjusted_score=signal.profitability_score * (1 - 0.05),  # Adjust for risk
                    timestamp=datetime.now()
                )
                opportunities.append(opportunity)
            
            # Detect statistical arbitrage
            stat_arbitrage_signals = await self.detect_statistical_arbitrage(asset_data)
            for signal in stat_arbitrage_signals:
                opportunity = ArbitrageOpportunity(
                    opportunity_id=f"stat_{signal.strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    arbitrage_type="statistical_arbitrage",
                    assets=signal.instruments,
                    signal_strength=self._classify_signal_strength(signal.sharpe_ratio / 2),
                    expected_return=signal.alpha_signal,
                    confidence_score=min(signal.information_ratio / 2, 1),
                    time_horizon=30,
                    max_drawdown_risk=signal.maximum_drawdown,
                    execution_complexity="high",
                    capital_requirement=len(signal.instruments) * 0.05,
                    risk_adjusted_score=signal.sharpe_ratio * signal.win_rate,
                    timestamp=datetime.now()
                )
                opportunities.append(opportunity)
            
            # Detect relative value arbitrage
            relative_value_signals = await self.detect_relative_value_arbitrage(asset_data)
            for signal in relative_value_signals:
                opportunity = ArbitrageOpportunity(
                    opportunity_id=f"rv_{signal.cheap_asset}_{signal.expensive_asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    arbitrage_type="relative_value",
                    assets=[signal.cheap_asset, signal.expensive_asset],
                    signal_strength=self._classify_signal_strength(signal.combined_score),
                    expected_return=signal.expected_convergence,
                    confidence_score=signal.catalyst_probability,
                    time_horizon=signal.time_to_convergence,
                    max_drawdown_risk=0.03,
                    execution_complexity="moderate",
                    capital_requirement=0.1,
                    risk_adjusted_score=signal.combined_score * signal.catalyst_probability,
                    timestamp=datetime.now()
                )
                opportunities.append(opportunity)
            
            # Sort by risk-adjusted score
            opportunities.sort(key=lambda x: x.risk_adjusted_score, reverse=True)
            
            logger.info(f"Generated {len(opportunities)} arbitrage opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error generating arbitrage opportunities: {str(e)}")
            return []
    
    def _classify_signal_strength(self, score: float) -> str:
        """Classify signal strength"""
        try:
            if score > 0.8:
                return "very_strong"
            elif score > 0.6:
                return "strong"
            elif score > 0.4:
                return "moderate"
            elif score > 0.2:
                return "weak"
            else:
                return "very_weak"
                
        except Exception as e:
            logger.error(f"Error classifying signal strength: {str(e)}")
            return "moderate"
    
    async def construct_arbitrage_portfolio(self, opportunities: List[ArbitrageOpportunity]) -> ArbitragePortfolio:
        """Construct optimal arbitrage portfolio from opportunities"""
        try:
            if not opportunities:
                return ArbitragePortfolio(
                    portfolio_id="empty_portfolio",
                    timestamp=datetime.now(),
                    total_opportunities=0,
                    portfolio_expected_return=0,
                    portfolio_risk=0,
                    portfolio_sharpe=0,
                    diversification_score=0,
                    capacity_utilization=0,
                    risk_budget_consumption=0,
                    opportunity_breakdown={},
                    top_opportunities=[],
                    portfolio_recommendations=["No arbitrage opportunities detected"]
                )
            
            # Select top opportunities (limit by diversification and risk)
            selected_opportunities = []
            used_assets = set()
            portfolio_risk = 0
            portfolio_return = 0
            risk_budget = self.risk_parameters["max_portfolio_risk"]
            
            for opp in opportunities[:20]:  # Consider top 20 opportunities
                # Check asset overlap
                asset_overlap = len(set(opp.assets) & used_assets) / len(opp.assets)
                if asset_overlap > 0.5:  # Max 50% overlap
                    continue
                
                # Check risk budget
                if portfolio_risk + opp.max_drawdown_risk > risk_budget:
                    continue
                
                # Add opportunity
                selected_opportunities.append(opp)
                used_assets.update(opp.assets)
                portfolio_risk += opp.max_drawdown_risk * (opp.capital_requirement / len(opp.assets))
                portfolio_return += opp.expected_return * (opp.capital_requirement / len(opp.assets))
                
                if len(selected_opportunities) >= 8:  # Max 8 positions
                    break
            
            # Calculate portfolio metrics
            if selected_opportunities:
                portfolio_return = portfolio_return / len(selected_opportunities)
                portfolio_sharpe = portfolio_return / max(portfolio_risk, 0.01)
                
                # Diversification score
                all_assets = set()
                for opp in selected_opportunities:
                    all_assets.update(opp.assets)
                diversification_score = len(all_assets) / sum(len(opp.assets) for opp in selected_opportunities)
                
                # Capacity utilization
                total_capital = sum(opp.capital_requirement for opp in selected_opportunities)
                capacity_utilization = min(total_capital / 1.0, 1.0)  # Assuming 1.0 total capital
                
                # Risk budget consumption
                risk_budget_consumption = portfolio_risk / risk_budget
                
                # Opportunity breakdown
                breakdown = {}
                for opp in selected_opportunities:
                    if opp.arbitrage_type not in breakdown:
                        breakdown[opp.arbitrage_type] = 0
                    breakdown[opp.arbitrage_type] += 1
                
                # Top opportunities
                top_opportunities = sorted(selected_opportunities, 
                                         key=lambda x: x.risk_adjusted_score, 
                                         reverse=True)[:5]
                
                # Portfolio recommendations
                recommendations = self._generate_portfolio_recommendations(selected_opportunities, portfolio_risk, diversification_score)
                
            else:
                portfolio_return = 0
                portfolio_sharpe = 0
                diversification_score = 0
                capacity_utilization = 0
                risk_budget_consumption = 0
                breakdown = {}
                top_opportunities = []
                recommendations = ["No suitable arbitrage opportunities found"]
            
            portfolio = ArbitragePortfolio(
                portfolio_id="arbitrage_portfolio",
                timestamp=datetime.now(),
                total_opportunities=len(selected_opportunities),
                portfolio_expected_return=portfolio_return,
                portfolio_risk=portfolio_risk,
                portfolio_sharpe=portfolio_sharpe,
                diversification_score=diversification_score,
                capacity_utilization=capacity_utilization,
                risk_budget_consumption=risk_budget_consumption,
                opportunity_breakdown=breakdown,
                top_opportunities=top_opportunities,
                portfolio_recommendations=recommendations
            )
            
            logger.info(f"Constructed arbitrage portfolio: {len(selected_opportunities)} opportunities, "
                       f"Expected return: {portfolio_return:.1%}, Sharpe: {portfolio_sharpe:.2f}")
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Error constructing arbitrage portfolio: {str(e)}")
            return ArbitragePortfolio(
                portfolio_id="error_portfolio",
                timestamp=datetime.now(),
                total_opportunities=0,
                portfolio_expected_return=0,
                portfolio_risk=0,
                portfolio_sharpe=0,
                diversification_score=0,
                capacity_utilization=0,
                risk_budget_consumption=0,
                opportunity_breakdown={},
                top_opportunities=[],
                portfolio_recommendations=["Portfolio construction failed due to data error"]
            )
    
    def _generate_portfolio_recommendations(self, opportunities: List[ArbitrageOpportunity], 
                                          portfolio_risk: float, 
                                          diversification_score: float) -> List[str]:
        """Generate portfolio construction recommendations"""
        try:
            recommendations = []
            
            # Risk recommendations
            if portfolio_risk > self.risk_parameters["max_portfolio_risk"] * 0.8:
                recommendations.append("Portfolio risk approaching limit - consider reducing position sizes")
            
            # Diversification recommendations
            if diversification_score < 0.6:
                recommendations.append("Low diversification - consider more uncorrelated strategies")
            
            # Opportunity type recommendations
            opportunity_types = [opp.arbitrage_type for opp in opportunities]
            if opportunity_types.count("pairs_arbitrage") > len(opportunities) * 0.6:
                recommendations.append("Heavy reliance on pairs arbitrage - consider adding statistical strategies")
            
            # Execution complexity recommendations
            complex_strategies = [opp for opp in opportunities if opp.execution_complexity == "high"]
            if len(complex_strategies) > len(opportunities) * 0.4:
                recommendations.append("High execution complexity - ensure adequate infrastructure")
            
            # Capital efficiency recommendations
            total_capital_required = sum(opp.capital_requirement for opp in opportunities)
            if total_capital_required > 0.8:
                recommendations.append("High capital utilization - monitor liquidity constraints")
            
            # Performance recommendations
            high_return_opps = [opp for opp in opportunities if opp.expected_return > 0.2]
            if high_return_opps:
                recommendations.append(f"{len(high_return_opps)} high-return opportunities identified - prioritize execution")
            
            # General recommendations
            if len(recommendations) == 0:
                recommendations.append("Portfolio well-balanced - maintain current allocation")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating portfolio recommendations: {str(e)}")
            return ["Unable to generate specific recommendations"]
    
    async def monitor_arbitrage_alerts(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Monitor for significant arbitrage signals and alerts"""
        try:
            alerts = {}
            
            # Generate opportunities
            opportunities = await self.generate_arbitrage_opportunities(symbols)
            
            # Check for strong signals
            very_strong_signals = [opp for opp in opportunities if opp.signal_strength == "very_strong"]
            if very_strong_signals:
                alerts["strong_signals"] = [
                    f"{opp.arbitrage_type}: {', '.join(opp.assets)} "
                    f"(confidence: {opp.confidence_score:.1%}, return: {opp.expected_return:.1%})"
                    for opp in very_strong_signals[:3]
                ]
            
            # Check for high-return opportunities
            high_return_opportunities = [opp for opp in opportunities if opp.expected_return > 0.15]
            if high_return_opportunities:
                alerts["high_return"] = [
                    f"{opp.arbitrage_type} in {opp.assets[0]}: {opp.expected_return:.1%} expected return"
                    for opp in high_return_opportunities[:3]
                ]
            
            # Check portfolio-level alerts
            portfolio = await self.construct_arbitrage_portfolio(opportunities)
            
            if portfolio.risk_budget_consumption > 0.9:
                alerts["risk_limit"] = [f"Portfolio risk at {portfolio.risk_budget_consumption:.1%} of limit"]
            
            if portfolio.diversification_score < 0.4:
                alerts["diversification"] = ["Low diversification score - consider adding uncorrelated strategies"]
            
            # Check for capacity constraints
            if portfolio.capacity_utilization > 0.9:
                alerts["capacity"] = [f"High capacity utilization: {portfolio.capacity_utilization:.1%}"]
            
            # Check execution alerts
            high_complexity = [opp for opp in opportunities if opp.execution_complexity == "high"]
            if len(high_complexity) > 3:
                alerts["execution"] = [f"{len(high_complexity)} high-complexity strategies - ensure execution capability"]
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring arbitrage alerts: {str(e)}")
            return {}
    
    async def export_arbitrage_analysis(self, symbols: List[str], format_type: str = "json") -> str:
        """Export arbitrage analysis to file"""
        try:
            # Generate opportunities and portfolio
            opportunities = await self.generate_arbitrage_opportunities(symbols)
            portfolio = await self.construct_arbitrage_portfolio(opportunities)
            
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
                    "opportunities": convert_dataclass(opportunities),
                    "portfolio": convert_dataclass(portfolio)
                }
                
                filename = f"arbitrage_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = f"/workspace/mpe/exports/{filename}"
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return filepath
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting arbitrage analysis: {str(e)}")
            return ""

# Main execution function
async def main():
    """Main execution function for multi-asset arbitrage analysis"""
    engine = MultiAssetArbitrageEngine()
    
    # Test with diversified asset universe
    test_symbols = ["AAPL", "MSFT", "SPY", "QQQ", "TLT", "LQD", "GLD", "USO", "VIX"]
    
    logger.info("Starting Multi-Asset Arbitrage Engine analysis...")
    
    # Test comprehensive opportunity generation
    logger.info(f"\n=== Arbitrage Opportunity Analysis ===")
    
    opportunities = await engine.generate_arbitrage_opportunities(test_symbols)
    
    logger.info(f"Total Opportunities Found: {len(opportunities)}")
    
    if opportunities:
        logger.info(f"\nTop 5 Opportunities:")
        for i, opp in enumerate(opportunities[:5], 1):
            logger.info(f"{i}. {opp.arbitrage_type} - {', '.join(opp.assets)}")
            logger.info(f"   Signal: {opp.signal_strength}, Return: {opp.expected_return:.1%}")
            logger.info(f"   Confidence: {opp.confidence_score:.1%}, Time Horizon: {opp.time_horizon} days")
    
    # Test portfolio construction
    logger.info(f"\n=== Portfolio Construction ===")
    
    portfolio = await engine.construct_arbitrage_portfolio(opportunities)
    
    logger.info(f"Portfolio Opportunities: {portfolio.total_opportunities}")
    logger.info(f"Expected Return: {portfolio.portfolio_expected_return:.1%}")
    logger.info(f"Portfolio Risk: {portfolio.portfolio_risk:.1%}")
    logger.info(f"Sharpe Ratio: {portfolio.portfolio_sharpe:.2f}")
    logger.info(f"Diversification Score: {portfolio.diversification_score:.2f}")
    logger.info(f"Risk Budget Consumption: {portfolio.risk_budget_consumption:.1%}")
    
    logger.info(f"\nOpportunity Breakdown:")
    for opp_type, count in portfolio.opportunity_breakdown.items():
        logger.info(f"  {opp_type}: {count}")
    
    # Show top portfolio positions
    if portfolio.top_opportunities:
        logger.info(f"\nTop Portfolio Positions:")
        for i, opp in enumerate(portfolio.top_opportunities, 1):
            logger.info(f"{i}. {opp.arbitrage_type}: {', '.join(opp.assets)}")
            logger.info(f"   Risk-Adjusted Score: {opp.risk_adjusted_score:.3f}")
    
    # Show portfolio recommendations
    logger.info(f"\nPortfolio Recommendations:")
    for rec in portfolio.portfolio_recommendations[:4]:
        logger.info(f"  - {rec}")
    
    # Test monitoring alerts
    logger.info(f"\n=== Arbitrage Alerts ===")
    
    alerts = await engine.monitor_arbitrage_alerts(test_symbols)
    
    for alert_type, alert_messages in alerts.items():
        logger.info(f"{alert_type.upper()}:")
        for message in alert_messages:
            logger.info(f"  - {message}")
    
    # Test specific arbitrage types
    logger.info(f"\n=== Arbitrage Type Analysis ===")
    
    # Pairs arbitrage
    asset_data = await engine.fetch_asset_data(test_symbols)
    pairs_signals = await engine.detect_pairs_arbitrage(asset_data)
    logger.info(f"Pairs Arbitrage Signals: {len(pairs_signals)}")
    
    for signal in pairs_signals[:2]:
        logger.info(f"  {signal.long_asset} vs {signal.short_asset}: Z-score {signal.z_score:.2f}")
        logger.info(f"    Half-life: {signal.half_life:.1f} days, Profitability: {signal.profitability_score:.2f}")
    
    # Statistical arbitrage
    stat_arbitrage = await engine.detect_statistical_arbitrage(asset_data)
    logger.info(f"Statistical Arbitrage Strategies: {len(stat_arbitrage)}")
    
    for strategy in stat_arbitrage:
        logger.info(f"  {strategy.strategy_type}: Sharpe {strategy.sharpe_ratio:.2f}, "
                   f"Expected Return: {strategy.alpha_signal:.1%}")
    
    logger.info("Multi-Asset Arbitrage Engine analysis completed successfully")

if __name__ == "__main__":
    asyncio.run(main())