"""
Real-Time Processor - Core data processing and aggregation engine
Handles real-time data streaming, processing, and distribution
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import logging
import yfinance as yf
from database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class RealTimeProcessor:
    """Real-time data processing and market data aggregation"""
    
    def __init__(self):
        self.name = "Real-Time Processor"
        self.version = "1.0.0"
        self.status = "active"
        self.last_process = None
        
        # Data sources configuration
        self.data_sources = {
            "equities": ["SPY", "QQQ", "IWM", "VTI", "DIA"],
            "bonds": ["TLT", "IEF", "SHY"],
            "commodities": ["GLD", "SLV", "USO"],
            "crypto": ["BTC-USD", "ETH-USD"],
            "forex": ["DX-Y.NYB", "EURUSD=X"]
        }
        
        # Processing intervals (seconds)
        self.processing_intervals = {
            "tick_data": 1,
            "minute_data": 60,
            "hourly_data": 3600,
            "daily_data": 86400
        }
        
        # Real-time data cache
        self.data_cache = {}
        self.processed_data = {}
        
        # Data processing functions
        self.processors = {
            "price_update": self._process_price_update,
            "volume_analysis": self._analyze_volume,
            "volatility_calc": self._calculate_volatility,
            "technical_indicators": self._calculate_technical_indicators,
            "cross_asset_correlation": self._calculate_correlations
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get processor status"""
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "last_process": self.last_process.isoformat() if self.last_process else None,
            "data_sources": len(self.data_sources),
            "cache_size": len(self.data_cache)
        }
    
    async def process_realtime_data(self):
        """Main real-time data processing function"""
        try:
            current_time = datetime.utcnow()
            
            # Collect data from all sources
            market_data = await self._collect_market_data()
            
            # Process data through all processors
            processed_results = {}
            for processor_name, processor_func in self.processors.items():
                try:
                    result = await processor_func(market_data)
                    processed_results[processor_name] = result
                except Exception as e:
                    logger.error(f"Error in {processor_name} processor: {e}")
            
            # Update cache
            self.data_cache = market_data
            self.processed_data = processed_results
            self.last_process = current_time
            
            logger.debug(f"Real-time processing completed at {current_time}")
            
        except Exception as e:
            logger.error(f"Error in real-time processing: {e}")
    
    async def _collect_market_data(self) -> Dict[str, Dict[str, Any]]:
        """Collect market data from all configured sources"""
        market_data = {}
        
        for category, symbols in self.data_sources.items():
            market_data[category] = {}
            
            for symbol in symbols:
                try:
                    # Get real-time data
                    ticker = yf.Ticker(symbol)
                    
                    # Get current info
                    info = ticker.info
                    
                    # Get recent price data
                    hist = ticker.history(period="2d", interval="1m")
                    
                    if len(hist) > 0:
                        latest = hist.iloc[-1]
                        previous = hist.iloc[-2] if len(hist) > 1 else latest
                        
                        # Calculate price changes
                        price_change = latest['Close'] - previous['Close']
                        price_change_pct = (price_change / previous['Close']) * 100
                        
                        # Calculate volume metrics
                        volume = latest['Volume']
                        avg_volume = hist['Volume'].tail(20).mean() if len(hist) > 20 else volume
                        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
                        
                        market_data[category][symbol] = {
                            "symbol": symbol,
                            "price": float(latest['Close']),
                            "open": float(latest['Open']),
                            "high": float(latest['High']),
                            "low": float(latest['Low']),
                            "volume": int(volume),
                            "price_change": float(price_change),
                            "price_change_pct": float(price_change_pct),
                            "volume_ratio": float(volume_ratio),
                            "timestamp": latest.name,
                            "market_cap": info.get("marketCap"),
                            "bid_ask_spread": info.get("bidAskSpread", 0),
                            "data_source": "yfinance"
                        }
                        
                except Exception as e:
                    logger.error(f"Error collecting data for {symbol}: {e}")
                    market_data[category][symbol] = {
                        "symbol": symbol,
                        "error": str(e),
                        "timestamp": datetime.utcnow()
                    }
        
        return market_data
    
    async def _process_price_update(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Process price updates and detect significant movements"""
        price_updates = {}
        significant_moves = []
        
        for category, symbols_data in market_data.items():
            for symbol, data in symbols_data.items():
                if "error" in data:
                    continue
                
                # Detect significant price movements
                if abs(data["price_change_pct"]) > 0.5:  # >0.5% move
                    significant_moves.append({
                        "symbol": symbol,
                        "category": category,
                        "price_change_pct": data["price_change_pct"],
                        "direction": "up" if data["price_change_pct"] > 0 else "down"
                    })
                
                price_updates[symbol] = {
                    "current_price": data["price"],
                    "change": data["price_change"],
                    "change_pct": data["price_change_pct"],
                    "volume_ratio": data["volume_ratio"],
                    "timestamp": data["timestamp"].isoformat()
                }
        
        return {
            "price_updates": price_updates,
            "significant_moves": significant_moves,
            "total_assets": sum(len(symbols) for symbols in market_data.values()),
            "processing_time": datetime.utcnow().isoformat()
        }
    
    async def _analyze_volume(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze volume patterns and anomalies"""
        volume_analysis = {}
        volume_alerts = []
        
        for category, symbols_data in market_data.items():
            volume_analysis[category] = {}
            
            for symbol, data in symbols_data.items():
                if "error" in data:
                    continue
                
                volume_ratio = data["volume_ratio"]
                
                # Volume analysis
                if volume_ratio > 2.0:  # 2x average volume
                    volume_analysis[symbol] = "high_volume"
                    volume_alerts.append({
                        "symbol": symbol,
                        "type": "high_volume",
                        "ratio": volume_ratio,
                        "category": category
                    })
                elif volume_ratio < 0.5:  # 50% of average
                    volume_analysis[symbol] = "low_volume"
                else:
                    volume_analysis[symbol] = "normal_volume"
        
        return {
            "volume_analysis": volume_analysis,
            "volume_alerts": volume_alerts,
            "total_alerts": len(volume_alerts)
        }
    
    async def _calculate_volatility(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate real-time volatility metrics"""
        volatility_data = {}
        
        for category, symbols_data in market_data.items():
            for symbol, data in symbols_data.items():
                if "error" in data:
                    continue
                
                # Simple volatility calculation based on price changes
                # In production, this would use more sophisticated methods
                
                price_change_pct = abs(data["price_change_pct"])
                
                # Classify volatility level
                if price_change_pct > 1.0:
                    vol_level = "high"
                elif price_change_pct > 0.5:
                    vol_level = "medium"
                else:
                    vol_level = "low"
                
                volatility_data[symbol] = {
                    "volatility_level": vol_level,
                    "intraday_vol": price_change_pct,
                    "volume_confirmed": data["volume_ratio"] > 1.2
                }
        
        return {
            "volatility_data": volatility_data,
            "high_vol_assets": [s for s, v in volatility_data.items() if v["volatility_level"] == "high"],
            "medium_vol_assets": [s for s, v in volatility_data.items() if v["volatility_level"] == "medium"]
        }
    
    async def _calculate_technical_indicators(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic technical indicators"""
        technical_data = {}
        
        for category, symbols_data in market_data.items():
            for symbol, data in symbols_data.items():
                if "error" in data:
                    continue
                
                # Simple technical analysis
                price = data["price"]
                price_change_pct = data["price_change_pct"]
                
                # Price momentum
                if price_change_pct > 1.0:
                    momentum = "strong_bullish"
                elif price_change_pct > 0.2:
                    momentum = "bullish"
                elif price_change_pct < -1.0:
                    momentum = "strong_bearish"
                elif price_change_pct < -0.2:
                    momentum = "bearish"
                else:
                    momentum = "neutral"
                
                technical_data[symbol] = {
                    "momentum": momentum,
                    "price_trend": "up" if price_change_pct > 0 else "down",
                    "strength": abs(price_change_pct),
                    "volume_confirmation": data["volume_ratio"] > 1.1
                }
        
        return {
            "technical_indicators": technical_data,
            "bullish_assets": [s for s, t in technical_data.items() if "bullish" in t["momentum"]],
            "bearish_assets": [s for s, t in technical_data.items() if "bearish" in t["momentum"]]
        }
    
    async def _calculate_correlations(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate cross-asset correlations"""
        correlation_data = {}
        
        # Get all asset prices
        all_prices = {}
        for category, symbols_data in market_data.items():
            for symbol, data in symbols_data.items():
                if "error" in data:
                    continue
                all_prices[symbol] = data["price"]
        
        # Calculate pairwise correlations (simplified)
        symbols = list(all_prices.keys())
        correlations = {}
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                # Simplified correlation based on price movements
                price1 = all_prices[symbol1]
                price2 = all_prices[symbol2]
                
                # Mock correlation calculation
                # In production, this would use historical price data
                correlation = np.random.uniform(-1, 1)  # Placeholder
                
                correlations[f"{symbol1}_{symbol2}"] = {
                    "correlation": correlation,
                    "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
                }
        
        # Find strong correlations
        strong_correlations = []
        for pair, data in correlations.items():
            if abs(data["correlation"]) > 0.7:
                strong_correlations.append({
                    "pair": pair,
                    "correlation": data["correlation"],
                    "direction": "positive" if data["correlation"] > 0 else "negative"
                })
        
        return {
            "correlations": correlations,
            "strong_correlations": strong_correlations,
            "correlation_matrix_size": len(symbols)
        }
    
    async def get_processed_data(self) -> Dict[str, Any]:
        """Get latest processed data"""
        return {
            "timestamp": self.last_process.isoformat() if self.last_process else None,
            "market_data": self.data_cache,
            "processed_data": self.processed_data,
            "status": "active"
        }
    
    async def register_custom_processor(self, name: str, processor_func: Callable):
        """Register a custom data processor"""
        self.processors[name] = processor_func
        logger.info(f"Registered custom processor: {name}")
    
    async def cleanup(self):
        """Cleanup resources"""
        self.data_cache.clear()
        self.processed_data.clear()
        logger.info("Real-Time Processor cleaned up")