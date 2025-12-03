"""
Configuration settings for Market Pulse Engine
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = "Market Pulse Engine (MPE)"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Database
    database_url: str = "postgresql://user:password@localhost/mpe"
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = "your-token"
    influxdb_org: str = "mpe"
    influxdb_bucket: str = "market_data"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Financial Data APIs
    alpha_vantage_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    fred_api_key: Optional[str] = None
    
    # Crypto APIs
    binance_api_key: Optional[str] = None
    binance_secret_key: Optional[str] = None
    coinbase_api_key: Optional[str] = None
    coinbase_secret_key: Optional[str] = None
    
    # News & Sentiment APIs
    news_api_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    
    # Real-time Data Sources
    websocket_endpoints: List[str] = [
        "wss://stream.binance.com:9443/ws",
        "wss://ws-feed.exchange.coinbase.com"
    ]
    
    # Market Data Assets
    equities: List[str] = ["SPY", "QQQ", "IWM", "VTI", "DIA", "XLF", "XLK"]
    bonds: List[str] = ["TLT", "IEF", "SHY", "TLH", "TIP"]
    commodities: List[str] = ["GLD", "SLV", "USO", "UNG", "DBC"]
    crypto: List[str] = ["BTCUSD", "ETHUSD", "BNBUSD", "ADAUSD"]
    forex: List[str] = ["DXY", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    
    # Pulse Engine Settings
    sentiment_pulse_interval: int = 5  # seconds
    volatility_pulse_interval: int = 10  # seconds
    liquidity_pulse_interval: int = 15  # seconds
    correlation_pulse_interval: int = 20  # seconds
    flow_pulse_interval: int = 12  # seconds
    risk_pulse_interval: int = 8  # seconds
    momentum_pulse_interval: int = 6  # seconds
    
    # Machine Learning Settings
    ml_model_path: str = "./models/"
    prediction_horizons: List[str] = ["5min", "15min", "30min", "1h", "4h", "1d"]
    lookback_periods: int = 252  # Trading days
    
    # Risk Management
    max_position_size: float = 0.1  # 10% of portfolio
    risk_threshold: float = 0.05  # 5% VaR threshold
    volatility_threshold: float = 0.25  # 25% annualized volatility
    
    # Real-time Processing
    websocket_timeout: int = 30  # seconds
    message_queue_size: int = 1000
    processing_threads: int = 4
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Market Pulse Engine specific configurations
class PulseEngineConfig:
    """Configuration for each pulse engine"""
    
    # Core Pulse Engines (1-7)
    SENTIMENT_PULSE = {
        "sources": ["news", "social", "analyst", "political"],
        "weights": {"news": 0.3, "social": 0.25, "analyst": 0.3, "political": 0.15},
        "sentiment_threshold": 0.7,
        "update_frequency": 5
    }
    
    VOLATILITY_PULSE = {
        "volatility_models": ["garch", "ewma", "realized"],
        "horizons": [5, 15, 30, 60, 120],
        "percentiles": [5, 25, 50, 75, 95],
        "compression_threshold": 0.8,
        "expansion_threshold": 1.2
    }
    
    LIQUIDITY_PULSE = {
        "sources": ["etf", "mutual_fund", "derivatives", "on_chain"],
        "metrics": ["flow_volume", "turnover", "bid_ask", "market_depth"],
        "aggregation_periods": [1, 5, 15, 60],
        "liquidity_threshold": 0.1
    }
    
    CORRELATION_PULSE = {
        "asset_pairs": [
            ("SPY", "TLT"),
            ("QQQ", "GLD"),
            ("DXY", "EURUSD"),
            ("BTCUSD", "SPY")
        ],
        "correlation_thresholds": {"tightening": 0.8, "weakening": 0.3},
        "lookback_window": 20
    }
    
    FLOW_PULSE = {
        "flow_sources": ["institutional", "retail", "hft", "foreign"],
        "sector_rotation": ["technology", "finance", "healthcare", "energy"],
        "factor_rotation": ["value", "growth", "momentum", "quality"],
        "momentum_threshold": 0.05
    }
    
    RISK_PULSE = {
        "risk_metrics": ["var", "expected_shortfall", "drawdown", "volatility"],
        "leverage_indicators": ["margin_debt", "options_volume", "derivative_positioning"],
        "stress_scenarios": ["liquidity_crisis", "correlation_break", "volatility_spike"],
        "alert_thresholds": {"high": 0.8, "medium": 0.5, "low": 0.2}
    }
    
    MOMENTUM_PULSE = {
        "timeframes": [5, 15, 30, 60, 120, 240],
        "momentum_indicators": ["rsi", "macd", "stochastic", "williams_r"],
        "regime_phases": ["early", "mid", "late", "reversal"],
        "exhaustion_threshold": 0.9
    }
    
    # Intelligence Engines (8-13)
    MACRO_PULSE = {
        "indicators": ["fed_rate", "inflation", "gdp", "unemployment"],
        "sensitivity_assets": ["TLT", "SPY", "DXY", "GLD"],
        "regime_indicators": ["quantitative_easing", "quantitative_tightening", "normal"]
    }
    
    NARRATIVE_INTELLIGENCE = {
        "sources": ["mainstream_media", "social_media", "influencer_posts"],
        "topics": ["monetary_policy", "geopolitics", "technology", "climate"],
        "sentiment_tracking": ["bullish", "bearish", "neutral", "conflicted"],
        "viral_threshold": 1000  # Engagement threshold
    }
    
    EVENT_SHOCKWAVE = {
        "event_types": ["fed_announcement", "earnings", "geopolitical", "natural_disaster"],
        "impact_measurement": ["price_change", "volume_spike", "volatility_jump"],
        "duration": [1, 5, 15, 60, 240]  # Minutes
    }
    
    # Derivatives Intelligence (14-18)
    OPTIONS_SURFACE = {
        "strikes_range": 0.2,  # ±20% from current price
        "expirations": [7, 14, 30, 60, 90, 180, 365],
        "surface_fitting": "svi",
        "greeks_calculation": ["delta", "gamma", "vega", "theta", "rho"]
    }
    
    OPTIONS_FLOW = {
        "volume_threshold": 100,
        "block_trade_size": 1000,
        "unusual_activity_threshold": 2.0,  # Standard deviations
        "time_decay_analysis": True
    }
    
    FUTURES_POSITIONING = {
        "participant_types": ["hedge_funds", "commodity_traders", "commercials", "small_traders"],
        "position_sizing_threshold": 100,
        "positioning_indicators": ["long_ratio", "short_ratio", "spread_ratio"]
    }
    
    # Liquidity & Cash Flow (19-22)
    ETF_FLOW_TRACKER = {
        "etf_universes": ["broad_market", "sector", "theme", "leverage"],
        "flow_threshold": 0.05,  # 5% of AUM
        "redemption_indicators": ["premium_discount", "creation_redemption_flow"],
        "sector_etfs": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLB", "XLU"]
    }
    
    MUTUAL_FUND_FLOW = {
        "fund_categories": ["equity", "bond", "hybrid", "money_market"],
        "flow_indicators": ["net_sales", "net_redemptions", "systematic_flow"],
        "institutional_flows": True
    }
    
    STABLECOIN_LIQUIDITY = {
        "stablecoins": ["USDT", "USDC", "DAI", "BUSD"],
        "monitoring_exchanges": ["binance", "coinbase", "huobi", "kraken"],
        "liquidity_metrics": ["market_cap", "trading_volume", "exchange_reserves"],
        "withdrawal_indicators": True
    }
    
    CROSS_MARKET_LIQUIDITY = {
        "markets": ["equity", "bond", "commodity", "crypto", "forex"],
        "liquidity_ratios": ["equity_bond", "equity_commodity", "crypto_fiat"],
        "stress_indicators": ["flight_to_quality", "flight_to_safety", "risk_on_risk_off"]
    }
    
    # Cross-Asset Intelligence (23-27)
    COMMODITY_PULSE = {
        "commodity_groups": ["metals", "energy", "agriculture"],
        "supercycle_indicators": ["supply_demand", "inventory_levels", "capex_spending"],
        "correlation_analysis": True
    }
    
    FX_PULSE = {
        "major_pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
        "carry_trade_indicators": ["interest_rate_differential", "volatility_spread"],
        "safe_haven_analysis": True
    }
    
    FIXED_INCOME_PULSE = {
        "bond_indices": ["treasury", "corporate", "municipal", "high_yield"],
        "yield_curve_analysis": ["steepening", "flattening", "inversion"],
        "credit_spread_analysis": True
    }
    
    EQUITY_FACTOR = {
        "factors": ["value", "growth", "momentum", "quality", "size", "low_volatility"],
        "factor_rotation": True,
        "factor_correlation": True
    }
    
    CRYPTO_MACRO = {
        "crypto_indicators": ["on_chain_metrics", "institutional_flows", "network_activity"],
        "macro_correlations": ["bitcoin_gold", "crypto_stocks", "crypto_vix"],
        "defi_indicators": ["tvl", "yield_farming", "dex_volume"]
    }
    
    # Predictive & Forecasting (28-30)
    MARKET_PULSE_FORECASTER = {
        "forecast_models": ["lstm", "transformer", "ensemble"],
        "feature_importance": True,
        "backtesting": True,
        "confidence_intervals": True
    }
    
    SCENARIO_SIMULATION = {
        "scenarios": ["liquidity_squeeze", "volatility_shock", "correlation_break", "narrative_flip"],
        "monte_carlo_runs": 1000,
        "stress_testing": True
    }
    
    MARKET_STRESS_PROBABILITY = {
        "stress_indicators": ["crash_probability", "volatility_spike", "correlation_breakdown"],
        "probability_models": ["logistic_regression", "svm", "random_forest"],
        "early_warning": True
    }


# Data source configurations
class DataSourceConfig:
    """Configuration for data sources"""
    
    # Free data sources
    YAHOO_FINANCE = {
        "base_url": "https://query1.finance.yahoo.com/v8/finance/chart/",
        "rate_limit": 2000,  # requests per hour
        "data_types": ["price", "volume", "fundamentals"]
    }
    
    FRED = {
        "base_url": "https://api.stlouisfed.org/fred/series/observations",
        "rate_limit": 120,  # requests per minute
        "key_series": ["GDP", "UNRATE", "CPIAUCSL", "DGS10"]
    }
    
    BINANCE = {
        "base_url": "https://api.binance.com/api/v3",
        "websocket_url": "wss://stream.binance.com:9443/ws",
        "rate_limit": 1200,  # requests per minute
        "data_types": ["trade", "kline", "depth"]
    }
    
    COINBASE = {
        "base_url": "https://api.exchange.coinbase.com",
        "websocket_url": "wss://ws-feed.exchange.coinbase.com",
        "rate_limit": 300,  # requests per minute
        "data_types": ["ticker", "trade", "level2"]
    }
    
    # News sources
    NEWS_API = {
        "base_url": "https://newsapi.org/v2",
        "sources": ["reuters", "bloomberg", "cnbc", "marketwatch"],
        "rate_limit": 1000,  # requests per day
        "keywords": ["market", "finance", "economy", "federal_reserve"]
    }
    
    # Social sentiment
    TWITTER_API = {
        "base_url": "https://api.twitter.com/2",
        "rate_limit": 300,  # requests per 15 minutes
        "endpoints": ["tweets/search", "users/by/username", "tweets"]
    }