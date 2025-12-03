"""
Sentiment Pulse Engine - Core Market Emotion Detection
Real-time sentiment analysis across news, social media, and analyst publications
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import requests
import json
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import yfinance as yf
from database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class SentimentPulseEngine:
    """Market Emotion Meter - Real-time sentiment detection across multiple sources"""
    
    def __init__(self):
        self.name = "Sentiment Pulse Engine"
        self.version = "1.0.0"
        self.status = "initialized"
        self.last_update = None
        self.sentiment_cache = {}
        self.model = None
        self.vectorizer = None
        self.sentiment_data = {}
        
        # Initialize ML models
        self._initialize_models()
        
        # Initialize data sources
        self.data_sources = {
            "news": ["reuters", "bloomberg", "cnbc", "marketwatch"],
            "social": ["twitter", "reddit", "stocktwits"],
            "analyst": ["earnings_reports", "research_reports"],
            "political": ["fed_speeches", "political_news"]
        }
        
        # Sentiment weights
        self.source_weights = {
            "news": 0.30,
            "social": 0.25,
            "analyst": 0.30,
            "political": 0.15
        }
        
        # Keywords for market sentiment analysis
        self.market_keywords = {
            "bullish": ["bull", "bullish", "rise", "rally", "gain", "profit", "growth", "positive", "optimistic"],
            "bearish": ["bear", "bearish", "fall", "decline", "loss", "negative", "pessimistic", "crash", "drop"],
            "fear": ["fear", "panic", "selloff", "volatile", "uncertainty", "recession", "crisis"],
            "greed": ["greed", "bubble", "euphoria", "hype", "fomo", "rally", "buy", "invest"]
        }
    
    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        try:
            # Simple sentiment model based on TextBlob + keywords
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            logger.info("Sentiment models initialized")
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "data_sources": list(self.data_sources.keys()),
            "cache_size": len(self.sentiment_cache)
        }
    
    async def get_pulse_data(self, assets: List[str] = None) -> Dict[str, Any]:
        """Get comprehensive sentiment pulse data"""
        if not assets:
            assets = ["SPY", "QQQ", "IWM", "DXY", "TLT", "GLD", "BTCUSD"]
        
        try:
            # Collect sentiment from all sources
            sentiment_scores = await self._collect_sentiment_data(assets)
            
            # Calculate sentiment pulse
            pulse_data = await self._calculate_sentiment_pulse(sentiment_scores)
            
            # Update cache
            self.sentiment_cache = pulse_data
            self.last_update = datetime.utcnow()
            
            return {
                "timestamp": self.last_update.isoformat(),
                "engine": self.name,
                "assets": assets,
                "sentiment_pulse": pulse_data,
                "source_breakdown": sentiment_scores,
                "market_sentiment": self._get_market_sentiment_summary(pulse_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment pulse data: {e}")
            return {"error": str(e)}
    
    async def _collect_sentiment_data(self, assets: List[str]) -> Dict[str, Dict[str, float]]:
        """Collect sentiment data from all sources"""
        sentiment_data = {}
        
        for asset in assets:
            sentiment_data[asset] = {}
            
            # Collect from each source
            for source in self.data_sources.keys():
                try:
                    if source == "news":
                        sentiment_data[asset][source] = await self._get_news_sentiment(asset)
                    elif source == "social":
                        sentiment_data[asset][source] = await self._get_social_sentiment(asset)
                    elif source == "analyst":
                        sentiment_data[asset][source] = await self._get_analyst_sentiment(asset)
                    elif source == "political":
                        sentiment_data[asset][source] = await self._get_political_sentiment(asset)
                    
                except Exception as e:
                    logger.error(f"Error collecting {source} sentiment for {asset}: {e}")
                    sentiment_data[asset][source] = 0.0
        
        return sentiment_data
    
    async def _get_news_sentiment(self, asset: str) -> float:
        """Get sentiment from financial news"""
        try:
            # Mock news sentiment analysis
            # In production, this would connect to news APIs
            
            # Simulate news sentiment based on recent price action
            ticker = yf.Ticker(asset)
            info = ticker.info
            
            # Get recent price data
            hist = ticker.history(period="5d")
            if len(hist) > 1:
                recent_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                
                # Convert price change to sentiment score
                sentiment_score = np.tanh(recent_change * 10)  # Scale and normalize
                
                # Add some noise to simulate real news sentiment
                noise = np.random.normal(0, 0.1)
                sentiment_score = np.clip(sentiment_score + noise, -1, 1)
                
                return sentiment_score
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {asset}: {e}")
            return 0.0
    
    async def _get_social_sentiment(self, asset: str) -> float:
        """Get sentiment from social media"""
        try:
            # Mock social media sentiment
            # In production, this would analyze Twitter, Reddit, StockTwits
            
            # Simulate social sentiment with some correlation to price volatility
            ticker = yf.Ticker(asset)
            hist = ticker.history(period="5d")
            
            if len(hist) > 1:
                # Calculate volatility as proxy for social media activity
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                
                # High volatility often correlates with more social media activity
                # This is a simplified model
                social_activity_factor = min(volatility * 2, 1.0)
                
                # Generate sentiment based on social activity
                sentiment_score = np.random.normal(0, 0.3) * social_activity_factor
                
                return np.clip(sentiment_score, -1, 1)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting social sentiment for {asset}: {e}")
            return 0.0
    
    async def _get_analyst_sentiment(self, asset: str) -> float:
        """Get sentiment from analyst reports and earnings"""
        try:
            # Mock analyst sentiment
            # In production, this would analyze earnings calls, analyst ratings
            
            ticker = yf.Ticker(asset)
            info = ticker.info
            
            # Get analyst recommendations if available
            if 'recommendationMean' in info:
                recommendation = info['recommendationMean']
                
                # Convert recommendation to sentiment score
                # 1 = Strong Buy, 2 = Buy, 3 = Hold, 4 = Sell, 5 = Strong Sell
                if recommendation <= 2:
                    sentiment_score = 0.5  # Bullish
                elif recommendation <= 3:
                    sentiment_score = 0.0  # Neutral
                else:
                    sentiment_score = -0.5  # Bearish
                
                return sentiment_score
            
            # Fallback to price-based sentiment
            hist = ticker.history(period="30d")
            if len(hist) > 1:
                recent_performance = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                sentiment_score = np.tanh(recent_performance * 5)
                return sentiment_score
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting analyst sentiment for {asset}: {e}")
            return 0.0
    
    async def _get_political_sentiment(self, asset: str) -> float:
        """Get sentiment from political and macro news"""
        try:
            # Mock political sentiment
            # In production, this would analyze Fed speeches, political news
            
            # Simulate political sentiment based on market regime
            political_sentiment = 0.0
            
            # Add some random variation to simulate political events
            event_impact = np.random.choice([-0.3, -0.1, 0.0, 0.1, 0.3], p=[0.1, 0.2, 0.4, 0.2, 0.1])
            
            return np.clip(political_sentiment + event_impact, -1, 1)
            
        except Exception as e:
            logger.error(f"Error getting political sentiment for {asset}: {e}")
            return 0.0
    
    async def _calculate_sentiment_pulse(self, sentiment_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate overall sentiment pulse for each asset"""
        pulse_data = {}
        
        for asset, sources in sentiment_data.items():
            # Calculate weighted sentiment
            weighted_sentiment = 0.0
            source_sentiments = {}
            
            for source, sentiment in sources.items():
                weight = self.source_weights.get(source, 0.1)
                source_sentiments[source] = sentiment
                weighted_sentiment += sentiment * weight
            
            # Calculate sentiment momentum (change over time)
            momentum = self._calculate_sentiment_momentum(asset, sources)
            
            # Calculate sentiment volatility
            volatility = np.std(list(sources.values()))
            
            # Determine sentiment regime
            regime = self._classify_sentiment_regime(weighted_sentiment, volatility)
            
            # Calculate confidence score
            confidence = self._calculate_sentiment_confidence(sources, volatility)
            
            pulse_data[asset] = {
                "sentiment_score": weighted_sentiment,
                "sentiment_momentum": momentum,
                "sentiment_volatility": volatility,
                "sentiment_regime": regime,
                "confidence": confidence,
                "source_breakdown": source_sentiments,
                "fear_greed_index": self._calculate_fear_greed_index(sources),
                "sentiment_acceleration": self._calculate_sentiment_acceleration(asset)
            }
        
        return pulse_data
    
    def _calculate_sentiment_momentum(self, asset: str, sources: Dict[str, float]) -> float:
        """Calculate sentiment momentum (rate of change)"""
        # Simple momentum calculation
        # In production, this would compare with historical sentiment
        
        if asset in self.sentiment_data:
            previous_sentiment = self.sentiment_data[asset].get("sentiment_score", 0)
            current_sentiment = np.mean(list(sources.values()))
            momentum = current_sentiment - previous_sentiment
        else:
            momentum = 0.0
        
        # Update sentiment history
        self.sentiment_data[asset] = {
            "sentiment_score": np.mean(list(sources.values())),
            "timestamp": datetime.utcnow()
        }
        
        return momentum
    
    def _classify_sentiment_regime(self, sentiment: float, volatility: float) -> str:
        """Classify market sentiment regime"""
        if sentiment > 0.3 and volatility < 0.2:
            return "bullish_stable"
        elif sentiment > 0.3 and volatility >= 0.2:
            return "bullish_volatile"
        elif sentiment < -0.3 and volatility < 0.2:
            return "bearish_stable"
        elif sentiment < -0.3 and volatility >= 0.2:
            return "bearish_volatile"
        elif abs(sentiment) <= 0.1:
            return "neutral"
        else:
            return "mixed"
    
    def _calculate_sentiment_confidence(self, sources: Dict[str, float], volatility: float) -> float:
        """Calculate confidence in sentiment measurement"""
        # Confidence based on agreement between sources and low volatility
        source_count = len(sources)
        avg_sentiment = np.mean(list(sources.values()))
        
        # Calculate agreement (how close sources are to average)
        if source_count > 1:
            agreements = [abs(sentiment - avg_sentiment) for sentiment in sources.values()]
            agreement_score = 1 - np.mean(agreements)
        else:
            agreement_score = 1.0
        
        # Combine agreement and low volatility for confidence
        confidence = agreement_score * (1 - volatility)
        
        return np.clip(confidence, 0, 1)
    
    def _calculate_fear_greed_index(self, sources: Dict[str, float]) -> float:
        """Calculate Fear & Greed Index"""
        avg_sentiment = np.mean(list(sources.values()))
        
        # Fear & Greed Index (0 = Extreme Fear, 100 = Extreme Greed)
        # Scale sentiment score to this range
        fear_greed_index = ((avg_sentiment + 1) / 2) * 100
        
        return np.clip(fear_greed_index, 0, 100)
    
    def _calculate_sentiment_acceleration(self, asset: str) -> float:
        """Calculate sentiment acceleration (rate of momentum change)"""
        # Simplified acceleration calculation
        # In production, this would track momentum changes over time
        
        if asset in self.sentiment_data:
            # Mock acceleration calculation
            acceleration = np.random.normal(0, 0.1)
            return acceleration
        
        return 0.0
    
    def _get_market_sentiment_summary(self, pulse_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get overall market sentiment summary"""
        if not pulse_data:
            return {"overall_sentiment": 0.0, "confidence": 0.0, "regime": "neutral"}
        
        # Calculate overall market sentiment
        sentiments = [data["sentiment_score"] for data in pulse_data.values()]
        overall_sentiment = np.mean(sentiments)
        
        # Calculate overall confidence
        confidences = [data["confidence"] for data in pulse_data.values()]
        overall_confidence = np.mean(confidences)
        
        # Determine dominant regime
        regimes = [data["sentiment_regime"] for data in pulse_data.values()]
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        dominant_regime = max(regime_counts.keys(), key=regime_counts.get)
        
        # Market fear/greed summary
        fear_greed_indices = [data["fear_greed_index"] for data in pulse_data.values()]
        avg_fear_greed = np.mean(fear_greed_indices)
        
        return {
            "overall_sentiment": overall_sentiment,
            "overall_confidence": overall_confidence,
            "dominant_regime": dominant_regime,
            "fear_greed_index": avg_fear_greed,
            "sentiment_distribution": regime_counts,
            "most_bullish_asset": max(pulse_data.keys(), key=lambda x: pulse_data[x]["sentiment_score"]),
            "most_bearish_asset": min(pulse_data.keys(), key=lambda x: pulse_data[x]["sentiment_score"])
        }
    
    async def get_sentiment_alerts(self, thresholds: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Generate sentiment-based alerts"""
        if not thresholds:
            thresholds = {
                "extreme_fear": 20,
                "extreme_greed": 80,
                "high_volatility": 0.5,
                "sentiment_shift": 0.3
            }
        
        alerts = []
        
        if self.sentiment_cache:
            for asset, data in self.sentiment_cache.items():
                # Fear/Greed alerts
                if data["fear_greed_index"] <= thresholds["extreme_fear"]:
                    alerts.append({
                        "type": "extreme_fear",
                        "asset": asset,
                        "severity": "high",
                        "message": f"Extreme fear detected for {asset} (Fear & Greed: {data['fear_greed_index']:.1f})",
                        "data": data
                    })
                
                elif data["fear_greed_index"] >= thresholds["extreme_greed"]:
                    alerts.append({
                        "type": "extreme_greed",
                        "asset": asset,
                        "severity": "medium",
                        "message": f"Extreme greed detected for {asset} (Fear & Greed: {data['fear_greed_index']:.1f})",
                        "data": data
                    })
                
                # Volatility alerts
                if data["sentiment_volatility"] >= thresholds["high_volatility"]:
                    alerts.append({
                        "type": "high_sentiment_volatility",
                        "asset": asset,
                        "severity": "medium",
                        "message": f"High sentiment volatility for {asset}",
                        "data": data
                    })
                
                # Sentiment shift alerts
                if abs(data["sentiment_momentum"]) >= thresholds["sentiment_shift"]:
                    direction = "bullish" if data["sentiment_momentum"] > 0 else "bearish"
                    alerts.append({
                        "type": "sentiment_shift",
                        "asset": asset,
                        "severity": "high",
                        "message": f"Significant {direction} sentiment shift for {asset}",
                        "data": data
                    })
        
        return alerts
    
    async def cleanup(self):
        """Cleanup resources"""
        self.sentiment_cache.clear()
        self.sentiment_data.clear()
        logger.info("Sentiment Pulse Engine cleaned up")