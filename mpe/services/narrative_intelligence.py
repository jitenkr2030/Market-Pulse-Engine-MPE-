"""
Narrative Intelligence Engine - Market Narrative & Story Analysis
Real-time tracking of dominant market narratives, story cycles, and narrative-driven sentiment shifts
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import yfinance as yf
from scipy import stats
import requests
import json
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class NarrativeIntelligenceEngine:
    """Market Narrative Monitor - Tracking dominant stories and narrative cycles"""
    
    def __init__(self):
        self.name = "Narrative Intelligence Engine"
        self.version = "1.0.0"
        self.status = "initialized"
        self.last_update = None
        self.narrative_cache = {}
        self.models = {}
        self.scaler = StandardScaler()
        
        # Narrative tracking models
        self.narrative_models = {
            "dominant_narratives": self._dominant_narratives_analysis,
            "narrative_momentum": self._narrative_momentum_analysis,
            "narrative_lifecycle": self._narrative_lifecycle_analysis,
            "story_cycle_detection": self._story_cycle_detection,
            "narrative_impact": self._narrative_impact_analysis
        }
        
        # Market narrative categories
        self.narrative_categories = {
            "economic_narratives": {
                "recession_fears": "Economic downturn and recession concerns",
                "inflation_story": "Inflation dynamics and central bank response",
                "growth_optimism": "Economic growth and expansion themes",
                "employment_surge": "Labor market strength narratives",
                "monetary_policy": "Interest rate and Fed policy stories"
            },
            "market_narratives": {
                "bubble_concerns": "Asset bubble and valuation worries",
                "tech_revolution": "Technology disruption and innovation themes",
                "climate_investing": "ESG and climate-focused narratives",
                "crypto_mainstream": "Cryptocurrency adoption stories",
                "market_democratization": "Retail trading and democratization"
            },
            "geopolitical_narratives": {
                "trade_war": "International trade conflict stories",
                "global_power_shift": "Geopolitical realignment themes",
                "energy_transition": "Energy policy and transition narratives",
                "supply_chain": "Global supply chain disruption stories",
                "regional_conflicts": "Regional instability and conflict"
            },
            "corporate_narratives": {
                "earnings_seasons": "Quarterly earnings and corporate performance",
                "merger_mania": "M&A and consolidation themes",
                "corporate_governance": "Board and governance storylines",
                "innovation_pipeline": "R&D and innovation narratives",
                "competitive_advantage": "Market position and moat stories"
            },
            "behavioral_narratives": {
                "fear_greed": "Market psychology and sentiment cycles",
                "momentum_mania": "Momentum and trend-following behaviors",
                "contrarian_opportunities": "Value and contrarian investment themes",
                "fomo_dynamics": "Fear of missing out and herding behaviors",
                "generational_wealth": "Wealth transfer and generational investing"
            }
        }
        
        # Narrative lifecycle stages
        self.narrative_stages = {
            "emergence": "Early stages, limited awareness, organic growth",
            "growth": "Accelerating adoption, increasing coverage, viral spread",
            "maturation": "Peak attention, mainstream adoption, saturated coverage",
            "saturation": "Overexposure, diminishing returns, narrative fatigue",
            "decline": "Fading interest, alternative narratives emerging, closure",
            "resurrection": "Renewed interest, cyclical returns, narrative reinvention"
        }
        
        # Narrative impact factors
        self.narrative_impacts = {
            "market_sentiment": "Impact on overall market sentiment",
            "sector_rotation": "Narrative-driven sector allocation changes",
            "asset_allocation": "Portfolio allocation shifts based on narratives",
            "volatility_trigger": "Narratives that drive volatility spikes",
            "trend_acceleration": "Stories that accelerate existing trends"
        }
        
        # Narrative sentiment indicators
        self.sentiment_indicators = {
            "news_coverage_intensity": "Frequency and prominence of narrative mentions",
            "social_media_buzz": "Social media engagement and viral spread",
            "expert_opinion": "Analyst and expert commentary alignment",
            "mainstream_adoption": "Narrative reach beyond financial media",
            "contrarian_detection": "Opposing viewpoints and dissenting opinions"
        }
        
        # Narrative thresholds
        self.narrative_thresholds = {
            "viral_threshold": 5.0,      # 5x normal mention frequency
            "saturation_point": 0.8,     # 80% narrative saturation
            "cycle_length_min": 30,      # Minimum 30-day cycle
            "impact_threshold": 0.7,     # 70% impact threshold
            "narrative_fatigue": 10      # Days of declining mentions
        }
        
        # Initialize NLP models
        self._initialize_nlp_models()
        
        # Database manager
        self.db_manager = None
        
    async def initialize(self):
        """Initialize database connections and NLP models"""
        try:
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            
            # Initialize NLP models
            self._initialize_nlp_models()
            
            self.status = "active"
            logger.info("Narrative Intelligence Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Narrative Intelligence Engine: {e}")
            self.status = "error"
            
    def _initialize_nlp_models(self):
        """Initialize NLP models for narrative analysis"""
        try:
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
                
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon')
                
            # Initialize sentiment analyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize TF-IDF vectorizer for narrative clustering
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            # Initialize clustering model for narrative grouping
            self.narrative_clustering = KMeans(
                n_clusters=8,  # Match narrative categories
                random_state=42
            )
            
            # Initialize Random Forest for narrative impact prediction
            self.models['narrative_impact_predictor'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Initialize narrative lifecycle classifier
            self.models['narrative_lifecycle_classifier'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=8
            )
            
            logger.info("NLP models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NLP models: {e}")
            
    async def _fetch_market_narrative_data(self, symbols: List[str], period: str = "1mo") -> Dict:
        """Fetch market data that might be narrative-driven"""
        try:
            narrative_data = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    if not hist.empty and len(hist) > 10:
                        # Calculate narrative-sensitive metrics
                        price_data = hist['Close']
                        volume_data = hist['Volume']
                        
                        # Volatility spikes (potential narrative events)
                        volatility = price_data.pct_change().rolling(5).std()
                        vol_spikes = volatility > volatility.quantile(0.9)
                        
                        # Volume anomalies (attention-grabbing events)
                        volume_ma = volume_data.rolling(10).mean()
                        volume_anomalies = volume_data > volume_ma * 2
                        
                        narrative_data[symbol] = {
                            'price_data': price_data,
                            'volume_data': volume_data,
                            'volatility_spikes': vol_spikes,
                            'volume_anomalies': volume_anomalies,
                            'narrative_proxies': {
                                'volatility_regime': 'high' if vol_spikes.sum() > 2 else 'normal',
                                'attention_level': 'elevated' if volume_anomalies.sum() > 3 else 'normal',
                                'price_acceleration': price_data.pct_change(5).iloc[-1] if len(price_data) > 5 else 0
                            }
                        }
                except Exception as e:
                    logger.warning(f"Error fetching narrative data for {symbol}: {e}")
                    continue
                    
            return narrative_data
            
        except Exception as e:
            logger.error(f"Error fetching market narrative data: {e}")
            return {}
            
    async def _dominant_narratives_analysis(self) -> Dict:
        """Analyze currently dominant market narratives"""
        try:
            # Get market data to identify narrative-driven moves
            narrative_symbols = ["SPY", "QQQ", "TLT", "GOLD", "BTC-USD", "TSLA", "AAPL"]
            market_data = await self._fetch_market_narrative_data(narrative_symbols)
            
            dominant_narratives = {}
            
            # Analyze each asset for narrative signals
            for symbol, data in market_data.items():
                narrative_signals = self._extract_narrative_signals(symbol, data)
                if narrative_signals:
                    dominant_narratives[symbol] = narrative_signals
                    
            if not dominant_narratives:
                return {'error': 'No dominant narrative data available'}
                
            # Classify narratives by category
            narrative_classification = self._classify_narratives(dominant_narratives)
            
            # Assess narrative dominance
            dominance_assessment = self._assess_narrative_dominance(narrative_classification)
            
            # Identify narrative leaders
            narrative_leaders = self._identify_narrative_leaders(dominant_narratives)
            
            # Cross-narrative interaction analysis
            interactions = self._analyze_narrative_interactions(narrative_classification)
            
            return {
                'dominant_narratives': dominant_narratives,
                'narrative_classification': narrative_classification,
                'dominance_assessment': dominance_assessment,
                'narrative_leaders': narrative_leaders,
                'narrative_interactions': interactions
            }
            
        except Exception as e:
            logger.error(f"Error in dominant narratives analysis: {e}")
            return {'error': str(e)}
            
    def _extract_narrative_signals(self, symbol: str, market_data: Dict) -> Dict:
        """Extract narrative signals from market data"""
        try:
            signals = {}
            
            # Volatility spike analysis (narrative-breaking events)
            vol_spikes = market_data.get('volatility_spikes', pd.Series())
            if vol_spikes.sum() > 0:
                recent_vol_spike = vol_spikes.tail(3).sum() > 0
                signals['volatility_narrative'] = {
                    'active': recent_vol_spike,
                    'intensity': vol_spikes.sum() / len(vol_spikes),
                    'event_type': 'breakout' if recent_vol_spike else 'stable'
                }
                
            # Volume anomaly analysis (narrative attention)
            volume_anomalies = market_data.get('volume_anomalies', pd.Series())
            if volume_anomalies.sum() > 0:
                recent_attention = volume_anomalies.tail(5).sum() > 0
                signals['attention_narrative'] = {
                    'active': recent_attention,
                    'attention_level': volume_anomalies.sum() / len(volume_anomalies),
                    'viral_potential': 'high' if volume_anomalies.sum() > 3 else 'moderate'
                }
                
            # Price acceleration analysis (narrative momentum)
            narrative_proxies = market_data.get('narrative_proxies', {})
            price_acceleration = narrative_proxies.get('price_acceleration', 0)
            
            if abs(price_acceleration) > 0.02:  # 2% acceleration threshold
                signals['momentum_narrative'] = {
                    'active': True,
                    'acceleration_strength': abs(price_acceleration),
                    'direction': 'bullish' if price_acceleration > 0 else 'bearish'
                }
                
            # Combine signals for overall narrative strength
            if signals:
                narrative_strength = np.mean([
                    sig.get('intensity', 0) if 'intensity' in sig else
                    sig.get('attention_level', 0) if 'attention_level' in sig else
                    sig.get('acceleration_strength', 0) if 'acceleration_strength' in sig else 0
                    for sig in signals.values()
                ])
                
                signals['overall_narrative_strength'] = float(narrative_strength)
                
            return signals
            
        except Exception as e:
            logger.error(f"Error extracting narrative signals for {symbol}: {e}")
            return {}
            
    def _classify_narratives(self, narratives: Dict) -> Dict:
        """Classify narratives into categories"""
        try:
            classification = {
                'economic': [],
                'market': [],
                'geopolitical': [],
                'corporate': [],
                'behavioral': []
            }
            
            # Simple classification based on symbol patterns and signals
            symbol_classifications = {
                'SPY': 'market', 'QQQ': 'market', 'TLT': 'economic', 'GOLD': 'economic',
                'BTC-USD': 'market', 'TSLA': 'corporate', 'AAPL': 'corporate'
            }
            
            for symbol, signals in narratives.items():
                category = symbol_classifications.get(symbol, 'market')
                
                # Analyze signals to determine sub-category
                signal_strength = signals.get('overall_narrative_strength', 0)
                volatility_active = signals.get('volatility_narrative', {}).get('active', False)
                attention_active = signals.get('attention_narrative', {}).get('active', False)
                momentum_active = signals.get('momentum_narrative', {}).get('active', False)
                
                classification[category].append({
                    'symbol': symbol,
                    'narrative_strength': signal_strength,
                    'signal_types': {
                        'volatility': volatility_active,
                        'attention': attention_active,
                        'momentum': momentum_active
                    }
                })
                
            # Calculate category scores
            category_scores = {}
            for category, items in classification.items():
                if items:
                    avg_strength = np.mean([item['narrative_strength'] for item in items])
                    category_scores[category] = {
                        'average_strength': float(avg_strength),
                        'narrative_count': len(items),
                        'dominance_rank': 0  # Will calculate later
                    }
                    
            return {
                'category_breakdown': classification,
                'category_scores': category_scores
            }
            
        except Exception as e:
            logger.error(f"Error classifying narratives: {e}")
            return {'category_breakdown': {}, 'category_scores': {}}
            
    def _assess_narrative_dominance(self, classification: Dict) -> Dict:
        """Assess which narratives are most dominant"""
        try:
            assessment = {
                'most_dominant_category': None,
                'dominance_hierarchy': [],
                'narrative_concentration': 0.0,
                'dominant_narratives': []
            }
            
            category_scores = classification.get('category_scores', {})
            
            if not category_scores:
                return assessment
                
            # Rank categories by dominance
            ranked_categories = sorted(
                category_scores.items(),
                key=lambda x: x[1]['average_strength'],
                reverse=True
            )
            
            assessment['most_dominant_category'] = ranked_categories[0][0]
            assessment['dominance_hierarchy'] = [
                {'category': cat, 'strength': data['average_strength']}
                for cat, data in ranked_categories
            ]
            
            # Calculate narrative concentration
            strengths = [data['average_strength'] for data in category_scores.values()]
            concentration = max(strengths) / np.sum(strengths) if np.sum(strengths) > 0 else 0
            assessment['narrative_concentration'] = float(concentration)
            
            # Identify dominant narratives (above threshold)
            threshold = np.mean(strengths) if strengths else 0
            assessment['dominant_narratives'] = [
                cat for cat, data in ranked_categories
                if data['average_strength'] > threshold
            ]
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing narrative dominance: {e}")
            return {'most_dominant_category': None, 'dominance_hierarchy': []}
            
    def _identify_narrative_leaders(self, narratives: Dict) -> Dict:
        """Identify assets leading each narrative"""
        try:
            leaders = {
                'narrative_leaders': [],
                'followers': [],
                'laggards': [],
                'narrative_acceleration': []
            }
            
            # Rank assets by narrative strength
            ranked_assets = sorted(
                narratives.items(),
                key=lambda x: x[1].get('overall_narrative_strength', 0),
                reverse=True
            )
            
            total_assets = len(ranked_assets)
            
            # Classify assets
            if total_assets >= 3:
                leaders['narrative_leaders'] = [asset[0] for asset in ranked_assets[:max(1, total_assets//3)]]
                leaders['followers'] = [asset[0] for asset in ranked_assets[max(1, total_assets//3):-max(1, total_assets//3)]]
                leaders['laggards'] = [asset[0] for asset in ranked_assets[-max(1, total_assets//3):]]
            else:
                leaders['narrative_leaders'] = [asset[0] for asset in ranked_assets]
                
            # Identify acceleration patterns
            for symbol, data in narratives.items():
                if 'momentum_narrative' in data and data['momentum_narrative']['active']:
                    leaders['narrative_acceleration'].append({
                        'symbol': symbol,
                        'acceleration_type': data['momentum_narrative']['direction'],
                        'strength': data['momentum_narrative']['acceleration_strength']
                    })
                    
            return leaders
            
        except Exception as e:
            logger.error(f"Error identifying narrative leaders: {e}")
            return {'narrative_leaders': [], 'followers': [], 'laggards': []}
            
    def _analyze_narrative_interactions(self, classification: Dict) -> Dict:
        """Analyze how different narratives interact"""
        try:
            interactions = {
                'synergistic_pairs': [],
                'conflicting_pairs': [],
                'narrative_spillover': [],
                'interaction_intensity': 0.0
            }
            
            category_scores = classification.get('category_scores', {})
            
            if len(category_scores) < 2:
                return interactions
                
            # Analyze pairwise interactions
            categories = list(category_scores.keys())
            interaction_scores = []
            
            for i, cat1 in enumerate(categories):
                for j, cat2 in enumerate(categories[i+1:], i+1):
                    strength1 = category_scores[cat1]['average_strength']
                    strength2 = category_scores[cat2]['average_strength']
                    
                    # Calculate interaction based on strength correlation
                    interaction_strength = (strength1 + strength2) / 2
                    
                    # Simple heuristic: strong + strong = synergistic, strong + weak = spillover
                    if strength1 > 0.3 and strength2 > 0.3:
                        interactions['synergistic_pairs'].append({
                            'categories': [cat1, cat2],
                            'strength': interaction_strength
                        })
                    elif abs(strength1 - strength2) > 0.2:
                        interactions['narrative_spillover'].append({
                            'dominant_category': cat1 if strength1 > strength2 else cat2,
                            'receiving_category': cat2 if strength1 > strength2 else cat1,
                            'spillover_strength': interaction_strength
                        })
                        
                    interaction_scores.append(interaction_strength)
                    
            # Calculate overall interaction intensity
            interactions['interaction_intensity'] = float(np.mean(interaction_scores)) if interaction_scores else 0.0
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error analyzing narrative interactions: {e}")
            return {'interaction_intensity': 0.0, 'synergistic_pairs': []}
            
    async def _narrative_momentum_analysis(self) -> Dict:
        """Analyze momentum and acceleration in narratives"""
        try:
            # Get historical data for momentum analysis
            momentum_symbols = ["SPY", "QQQ", "TLT", "GOLD"]
            historical_data = await self._fetch_market_narrative_data(momentum_symbols, period="3mo")
            
            momentum_analysis = {}
            
            for symbol, data in historical_data.items():
                momentum_signals = self._calculate_narrative_momentum(symbol, data)
                if momentum_signals:
                    momentum_analysis[symbol] = momentum_signals
                    
            if not momentum_analysis:
                return {'error': 'No narrative momentum data available'}
                
            # Overall momentum assessment
            momentum_assessment = self._assess_narrative_momentum(momentum_analysis)
            
            # Momentum acceleration/deceleration
            acceleration_analysis = self._analyze_momentum_acceleration(momentum_analysis)
            
            # Cross-asset momentum correlation
            correlation_analysis = self._analyze_momentum_correlation(momentum_analysis)
            
            return {
                'momentum_analysis': momentum_analysis,
                'momentum_assessment': momentum_assessment,
                'acceleration_analysis': acceleration_analysis,
                'correlation_analysis': correlation_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in narrative momentum analysis: {e}")
            return {'error': str(e)}
            
    def _calculate_narrative_momentum(self, symbol: str, data: Dict) -> Dict:
        """Calculate narrative momentum for an asset"""
        try:
            if not data:
                return {}
                
            momentum_metrics = {}
            
            # Calculate momentum over different timeframes
            price_data = data.get('price_data', pd.Series())
            if len(price_data) > 20:
                momentum_5d = price_data.pct_change(5).iloc[-1]
                momentum_10d = price_data.pct_change(10).iloc[-1]
                momentum_20d = price_data.pct_change(20).iloc[-1]
                
                momentum_metrics['short_term_momentum'] = float(momentum_5d)
                momentum_metrics['medium_term_momentum'] = float(momentum_10d)
                momentum_metrics['long_term_momentum'] = float(momentum_20d)
                
                # Momentum acceleration (change in momentum)
                momentum_acceleration = momentum_5d - momentum_10d
                momentum_metrics['momentum_acceleration'] = float(momentum_acceleration)
                
                # Narrative momentum classification
                if momentum_5d > 0.02 and momentum_acceleration > 0.01:
                    momentum_metrics['momentum_class'] = 'accelerating_bullish'
                elif momentum_5d > 0.01:
                    momentum_metrics['momentum_class'] = 'steady_bullish'
                elif momentum_5d < -0.02 and momentum_acceleration < -0.01:
                    momentum_metrics['momentum_class'] = 'accelerating_bearish'
                elif momentum_5d < -0.01:
                    momentum_metrics['momentum_class'] = 'steady_bearish'
                else:
                    momentum_metrics['momentum_class'] = 'momentum_neutral'
                    
            # Volume momentum analysis
            volume_data = data.get('volume_data', pd.Series())
            if len(volume_data) > 10:
                volume_ma_5 = volume_data.rolling(5).mean().iloc[-1]
                volume_ma_10 = volume_data.rolling(10).mean().iloc[-1]
                
                volume_momentum = (volume_ma_5 / volume_ma_10 - 1) if volume_ma_10 > 0 else 0
                momentum_metrics['volume_momentum'] = float(volume_momentum)
                
            return momentum_metrics
            
        except Exception as e:
            logger.error(f"Error calculating narrative momentum for {symbol}: {e}")
            return {}
            
    def _assess_narrative_momentum(self, momentum_data: Dict) -> Dict:
        """Assess overall narrative momentum"""
        try:
            assessment = {
                'overall_momentum_direction': 'neutral',
                'momentum_strength': 0.0,
                'momentum_consensus': 0.0,
                'leading_momentum_assets': [],
                'momentum_divergence': 0.0
            }
                
            if not momentum_data:
                return assessment
                
            # Collect momentum values
            momentum_values = []
            momentum_classes = []
            
            for symbol, metrics in momentum_data.items():
                short_mom = metrics.get('short_term_momentum', 0)
                medium_mom = metrics.get('medium_term_momentum', 0)
                
                momentum_values.append(short_mom)
                momentum_classes.append(metrics.get('momentum_class', 'neutral'))
                
            # Overall momentum direction
            avg_momentum = np.mean(momentum_values) if momentum_values else 0
            
            assessment['overall_momentum_direction'] = (
                'strongly_bullish' if avg_momentum > 0.03 else
                'bullish' if avg_momentum > 0.01 else
                'strongly_bearish' if avg_momentum < -0.03 else
                'bearish' if avg_momentum < -0.01 else
                'neutral'
            )
            
            assessment['momentum_strength'] = float(abs(avg_momentum))
            
            # Momentum consensus
            bullish_classes = [cls for cls in momentum_classes if 'bullish' in cls]
            bearish_classes = [cls for cls in momentum_classes if 'bearish' in cls]
            
            consensus_ratio = max(len(bullish_classes), len(bearish_classes)) / len(momentum_classes) if momentum_classes else 0
            assessment['momentum_consensus'] = float(consensus_ratio)
            
            # Leading momentum assets
            sorted_assets = sorted(
                momentum_data.items(),
                key=lambda x: x[1].get('short_term_momentum', 0),
                reverse=True
            )
            assessment['leading_momentum_assets'] = [asset[0] for asset in sorted_assets[:2]]
            
            # Momentum divergence
            momentum_std = np.std(momentum_values) if len(momentum_values) > 1 else 0
            assessment['momentum_divergence'] = float(momentum_std)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing narrative momentum: {e}")
            return {'overall_momentum_direction': 'neutral', 'momentum_strength': 0.0}
            
    def _analyze_momentum_acceleration(self, momentum_data: Dict) -> Dict:
        """Analyze momentum acceleration and deceleration patterns"""
        try:
            acceleration = {
                'acceleration_pattern': 'stable',
                'accelerating_assets': [],
                'decelerating_assets': [],
                'acceleration_intensity': 0.0
            }
                
            accelerating_assets = []
            decelerating_assets = []
            acceleration_strengths = []
            
            for symbol, metrics in momentum_data.items():
                momentum_acceleration = metrics.get('momentum_acceleration', 0)
                
                if momentum_acceleration > 0.005:  # Positive acceleration threshold
                    accelerating_assets.append(symbol)
                    acceleration_strengths.append(momentum_acceleration)
                elif momentum_acceleration < -0.005:  # Negative acceleration threshold
                    decelerating_assets.append(symbol)
                    acceleration_strengths.append(abs(momentum_acceleration))
                    
            acceleration['accelerating_assets'] = accelerating_assets
            acceleration['decelerating_assets'] = decelerating_assets
            
            # Overall acceleration pattern
            total_assets = len(momentum_data)
            if total_assets > 0:
                acceleration_ratio = len(accelerating_assets) / total_assets
                deceleration_ratio = len(decelerating_assets) / total_assets
                
                if acceleration_ratio > 0.6:
                    acceleration['acceleration_pattern'] = 'broadly_accelerating'
                elif deceleration_ratio > 0.6:
                    acceleration['acceleration_pattern'] = 'broadly_decelerating'
                elif acceleration_ratio > 0.3:
                    acceleration['acceleration_pattern'] = 'selectively_accelerating'
                elif deceleration_ratio > 0.3:
                    acceleration['acceleration_pattern'] = 'selectively_decelerating'
                else:
                    acceleration['acceleration_pattern'] = 'stable'
                    
                acceleration['acceleration_intensity'] = float(
                    np.mean(acceleration_strengths) if acceleration_strengths else 0
                )
                
            return acceleration
            
        except Exception as e:
            logger.error(f"Error analyzing momentum acceleration: {e}")
            return {'acceleration_pattern': 'stable', 'acceleration_intensity': 0.0}
            
    def _analyze_momentum_correlation(self, momentum_data: Dict) -> Dict:
        """Analyze correlation between different asset momentum"""
        try:
            correlation = {
                'momentum_correlation': 0.0,
                'synchronized_momentum': False,
                'correlation_regime': 'normal'
            }
                
            if not momentum_data or len(momentum_data) < 2:
                return correlation
                
            # Calculate momentum correlation using standard deviation as proxy
            momentum_values = [metrics.get('short_term_momentum', 0) for metrics in momentum_data.values()]
            
            if len(momentum_values) > 1:
                momentum_std = np.std(momentum_values)
                momentum_mean = np.mean(momentum_values)
                
                # Lower standard deviation = higher correlation
                correlation_proxy = 1 / (1 + momentum_std * 20) if momentum_std > 0 else 1
                correlation['momentum_correlation'] = float(correlation_proxy)
                
                # Synchronized momentum detection
                positive_momentum = sum(1 for m in momentum_values if m > 0)
                negative_momentum = len(momentum_values) - positive_momentum
                
                if positive_momentum >= len(momentum_values) * 0.7 or negative_momentum >= len(momentum_values) * 0.7:
                    correlation['synchronized_momentum'] = True
                    
                # Correlation regime
                if correlation_proxy > 0.8:
                    correlation['correlation_regime'] = 'high_correlation'
                elif correlation_proxy < 0.3:
                    correlation['correlation_regime'] = 'low_correlation'
                else:
                    correlation['correlation_regime'] = 'normal_correlation'
                    
            return correlation
            
        except Exception as e:
            logger.error(f"Error analyzing momentum correlation: {e}")
            return {'momentum_correlation': 0.0, 'correlation_regime': 'normal'}
            
    async def _narrative_lifecycle_analysis(self) -> Dict:
        """Analyze the lifecycle stages of dominant narratives"""
        try:
            # Analyze narrative evolution over time
            lifecycle_symbols = ["SPY", "QQQ", "TLT", "GOLD"]
            lifecycle_data = await self._fetch_market_narrative_data(lifecycle_symbols, period="2mo")
            
            lifecycle_analysis = {}
            
            for symbol, data in lifecycle_data.items():
                lifecycle_metrics = self._analyze_narrative_lifecycle(symbol, data)
                if lifecycle_metrics:
                    lifecycle_analysis[symbol] = lifecycle_metrics
                    
            if not lifecycle_analysis:
                return {'error': 'No narrative lifecycle data available'}
                
            # Overall lifecycle assessment
            lifecycle_assessment = self._assess_narrative_lifecycle(lifecycle_analysis)
            
            # Narrative transition detection
            transitions = self._detect_narrative_transitions(lifecycle_analysis)
            
            # Lifecycle synchronization
            synchronization = self._analyze_lifecycle_synchronization(lifecycle_analysis)
            
            return {
                'lifecycle_analysis': lifecycle_analysis,
                'lifecycle_assessment': lifecycle_assessment,
                'transitions': transitions,
                'synchronization': synchronization
            }
            
        except Exception as e:
            logger.error(f"Error in narrative lifecycle analysis: {e}")
            return {'error': str(e)}
            
    def _analyze_narrative_lifecycle(self, symbol: str, data: Dict) -> Dict:
        """Analyze lifecycle stage for a specific narrative"""
        try:
            if not data:
                return {}
                
            lifecycle_metrics = {}
            
            # Analyze attention patterns (proxy for narrative stage)
            volume_data = data.get('volume_data', pd.Series())
            if len(volume_data) > 20:
                # Calculate volume momentum and saturation
                volume_ma_short = volume_data.rolling(5).mean()
                volume_ma_long = volume_data.rolling(15).mean()
                
                current_volume = volume_data.iloc[-1]
                volume_ratio = current_volume / volume_ma_long.iloc[-1] if volume_ma_long.iloc[-1] > 0 else 1
                
                # Volume trend (acceleration/deceleration)
                volume_trend = (volume_ma_short.iloc[-1] / volume_ma_long.iloc[-1] - 1) if volume_ma_long.iloc[-1] > 0 else 0
                
                lifecycle_metrics['attention_level'] = float(volume_ratio)
                lifecycle_metrics['attention_trend'] = float(volume_trend)
                
                # Narrative saturation proxy
                recent_high_volume = volume_data.tail(10).quantile(0.8)
                saturation_proxy = current_volume / recent_high_volume if recent_high_volume > 0 else 1
                lifecycle_metrics['saturation_level'] = float(saturation_proxy)
                
            # Analyze momentum evolution
            price_data = data.get('price_data', pd.Series())
            if len(price_data) > 20:
                momentum_5d = price_data.pct_change(5).iloc[-1]
                momentum_15d = price_data.pct_change(15).iloc[-1]
                
                momentum_change = momentum_5d - momentum_15d
                lifecycle_metrics['momentum_evolution'] = float(momentum_change)
                
            # Lifecycle stage classification
            attention_level = lifecycle_metrics.get('attention_level', 1)
            saturation_level = lifecycle_metrics.get('saturation_level', 1)
            momentum_trend = lifecycle_metrics.get('attention_trend', 0)
            
            if attention_level > 2 and saturation_level < 0.8:
                lifecycle_stage = 'emergence'
            elif attention_level > 1.5 and momentum_trend > 0.1:
                lifecycle_stage = 'growth'
            elif saturation_level > 0.9 and momentum_trend > 0:
                lifecycle_stage = 'maturation'
            elif saturation_level > 0.8 and momentum_trend < -0.1:
                lifecycle_stage = 'saturation'
            elif attention_level < 1.2 and saturation_level > 0.8:
                lifecycle_stage = 'decline'
            else:
                lifecycle_stage = 'transition'
                
            lifecycle_metrics['lifecycle_stage'] = lifecycle_stage
            
            # Lifecycle strength score
            lifecycle_score = (attention_level * 0.4 + (1 - saturation_level) * 0.3 + abs(momentum_trend) * 0.3)
            lifecycle_metrics['lifecycle_strength'] = float(min(1.0, lifecycle_score))
            
            return lifecycle_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing narrative lifecycle for {symbol}: {e}")
            return {}
            
    def _assess_narrative_lifecycle(self, lifecycle_data: Dict) -> Dict:
        """Assess overall narrative lifecycle status"""
        try:
            assessment = {
                'dominant_stage': 'mixed',
                'lifecycle_distribution': {},
                'narrative_health': 'stable',
                'stage_transition_probability': 0.0
            }
            
            if not lifecycle_data:
                return assessment
                
            # Count narratives in each stage
            stage_counts = {}
            stage_strengths = {}
            
            for symbol, metrics in lifecycle_data.items():
                stage = metrics.get('lifecycle_stage', 'unknown')
                strength = metrics.get('lifecycle_strength', 0)
                
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
                
                if stage not in stage_strengths:
                    stage_strengths[stage] = []
                stage_strengths[stage].append(strength)
                
            # Determine dominant stage
            if stage_counts:
                dominant_stage = max(stage_counts.items(), key=lambda x: x[1])[0]
                assessment['dominant_stage'] = dominant_stage
                
            # Lifecycle distribution
            total_narratives = len(lifecycle_data)
            assessment['lifecycle_distribution'] = {
                stage: count / total_narratives for stage, count in stage_counts.items()
            }
            
            # Narrative health assessment
            avg_strength = np.mean([metrics.get('lifecycle_strength', 0) for metrics in lifecycle_data.values()])
            
            if avg_strength > 0.7:
                assessment['narrative_health'] = 'vibrant'
            elif avg_strength > 0.4:
                assessment['narrative_health'] = 'stable'
            elif avg_strength > 0.2:
                assessment['narrative_health'] = 'declining'
            else:
                assessment['narrative_health'] = 'weak'
                
            # Stage transition probability
            transition_indicators = 0
            for metrics in lifecycle_data.values():
                attention_trend = metrics.get('attention_trend', 0)
                if abs(attention_trend) > 0.15:  # Strong trend indicator
                    transition_indicators += 1
                    
            transition_probability = transition_indicators / len(lifecycle_data) if lifecycle_data else 0
            assessment['stage_transition_probability'] = float(transition_probability)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing narrative lifecycle: {e}")
            return {'dominant_stage': 'mixed', 'narrative_health': 'stable'}
            
    def _detect_narrative_transitions(self, lifecycle_data: Dict) -> Dict:
        """Detect narratives in transition between stages"""
        try:
            transitions = {
                'transitions_detected': False,
                'transitioning_narratives': [],
                'transition_type': 'stable',
                'transition_intensity': 0.0
            }
                
            transitioning_narratives = []
            transition_intensities = []
            
            for symbol, metrics in lifecycle_data.items():
                attention_trend = metrics.get('attention_trend', 0)
                momentum_evolution = metrics.get('momentum_evolution', 0)
                
                # Detect strong transition signals
                if abs(attention_trend) > 0.2 or abs(momentum_evolution) > 0.02:
                    transition_type = 'growth' if attention_trend > 0 else 'decline'
                    transition_intensity = abs(attention_trend)
                    
                    transitioning_narratives.append({
                        'symbol': symbol,
                        'transition_type': transition_type,
                        'intensity': transition_intensity
                    })
                    
                    transition_intensities.append(transition_intensity)
                    
            if transitioning_narratives:
                transitions['transitions_detected'] = True
                transitions['transitioning_narratives'] = transitioning_narratives
                
                # Determine overall transition type
                growth_transitions = sum(1 for n in transitioning_narratives if n['transition_type'] == 'growth')
                decline_transitions = len(transitioning_narratives) - growth_transitions
                
                if growth_transitions > decline_transitions:
                    transitions['transition_type'] = 'expansionary'
                elif decline_transitions > growth_transitions:
                    transitions['transition_type'] = 'contractionary'
                else:
                    transitions['transition_type'] = 'mixed'
                    
                transitions['transition_intensity'] = float(np.mean(transition_intensities))
                
            return transitions
            
        except Exception as e:
            logger.error(f"Error detecting narrative transitions: {e}")
            return {'transitions_detected': False, 'transition_intensity': 0.0}
            
    def _analyze_lifecycle_synchronization(self, lifecycle_data: Dict) -> Dict:
        """Analyze synchronization across narrative lifecycles"""
        try:
            synchronization = {
                'synchronization_level': 0.0,
                'synchronized_stages': [],
                'lifecycle_coordination': 'mixed',
                'synchronization_strength': 0.0
            }
                
            if not lifecycle_data or len(lifecycle_data) < 2:
                return synchronization
                
            # Analyze stage synchronization
            stages = [metrics.get('lifecycle_stage', 'unknown') for metrics in lifecycle_data.values()]
            
            # Count stage occurrences
            stage_counts = Counter(stages)
            most_common_stage = stage_counts.most_common(1)[0] if stage_counts else ('unknown', 0)
            
            synchronization_ratio = most_common_stage[1] / len(stages)
            synchronization['synchronization_level'] = float(synchronization_ratio)
            
            # Synchronized stages
            synchronized_stages = [stage for stage, count in stage_counts.items() if count > 1]
            synchronization['synchronized_stages'] = synchronized_stages
            
            # Lifecycle coordination
            if synchronization_ratio > 0.7:
                synchronization['lifecycle_coordination'] = 'highly_coordinated'
            elif synchronization_ratio > 0.5:
                synchronization['lifecycle_coordination'] = 'moderately_coordinated'
            else:
                synchronization['lifecycle_coordination'] = 'fragmented'
                
            # Synchronization strength
            stage_consistency = 1 - len(set(stages)) / len(stages) if stages else 0
            synchronization['synchronization_strength'] = float(stage_consistency)
            
            return synchronization
            
        except Exception as e:
            logger.error(f"Error analyzing lifecycle synchronization: {e}")
            return {'synchronization_level': 0.0, 'lifecycle_coordination': 'mixed'}
            
    async def _story_cycle_detection(self) -> Dict:
        """Detect recurring story cycles and patterns"""
        try:
            # Analyze cyclical patterns in narrative data
            cycle_symbols = ["SPY", "QQQ", "TLT", "GOLD"]
            cycle_data = await self._fetch_market_narrative_data(cycle_symbols, period="6mo")
            
            cycle_detection = {}
            
            for symbol, data in cycle_data.items():
                cycle_patterns = self._detect_story_cycles(symbol, data)
                if cycle_patterns:
                    cycle_detection[symbol] = cycle_patterns
                    
            if not cycle_detection:
                return {'error': 'No story cycle data available'}
                
            # Overall cycle analysis
            cycle_assessment = self._assess_story_cycles(cycle_detection)
            
            # Cycle pattern classification
            pattern_classification = self._classify_cycle_patterns(cycle_detection)
            
            # Cycle prediction signals
            prediction_signals = self._generate_cycle_predictions(cycle_detection)
            
            return {
                'cycle_detection': cycle_detection,
                'cycle_assessment': cycle_assessment,
                'pattern_classification': pattern_classification,
                'prediction_signals': prediction_signals
            }
            
        except Exception as e:
            logger.error(f"Error in story cycle detection: {e}")
            return {'error': str(e)}
            
    def _detect_story_cycles(self, symbol: str, data: Dict) -> Dict:
        """Detect cyclical patterns in narrative data"""
        try:
            if not data:
                return {}
                
            cycle_patterns = {}
            
            # Analyze price cycles (narrative-driven)
            price_data = data.get('price_data', pd.Series())
            if len(price_data) > 30:
                # Calculate price momentum cycles
                momentum_series = price_data.pct_change(5).dropna()
                
                # Detect momentum peaks and troughs
                peaks, _ = find_peaks(momentum_series, height=0.01)
                troughs, _ = find_peaks(-momentum_series, height=0.01)
                
                if len(peaks) > 1 or len(troughs) > 1:
                    # Calculate average cycle length
                    if len(peaks) > 1:
                        peak_intervals = np.diff(peaks)
                        avg_peak_cycle = np.mean(peak_intervals) if len(peak_intervals) > 0 else 0
                    else:
                        avg_peak_cycle = 0
                        
                    if len(troughs) > 1:
                        trough_intervals = np.diff(troughs)
                        avg_trough_cycle = np.mean(trough_intervals) if len(trough_intervals) > 0 else 0
                    else:
                        avg_trough_cycle = 0
                        
                    cycle_patterns['price_cycles'] = {
                        'peak_frequency': len(peaks),
                        'trough_frequency': len(troughs),
                        'avg_peak_cycle_length': float(avg_peak_cycle) if avg_peak_cycle > 0 else 0,
                        'avg_trough_cycle_length': float(avg_trough_cycle) if avg_trough_cycle > 0 else 0,
                        'cycle_regularity': float(1 - np.std([avg_peak_cycle, avg_trough_cycle]) / np.mean([avg_peak_cycle, avg_trough_cycle]) if np.mean([avg_peak_cycle, avg_trough_cycle]) > 0 else 0)
                    }
                    
            # Analyze volume cycles (attention cycles)
            volume_data = data.get('volume_data', pd.Series())
            if len(volume_data) > 20:
                # Detect volume spikes
                volume_ma = volume_data.rolling(10).mean()
                volume_spikes = volume_data > volume_ma * 2
                spike_positions = volume_spikes[volume_spikes].index
                
                if len(spike_positions) > 1:
                    spike_intervals = np.diff(range(len(spike_positions)))
                    avg_spike_cycle = np.mean(spike_intervals) if len(spike_intervals) > 0 else 0
                    
                    cycle_patterns['attention_cycles'] = {
                        'spike_frequency': len(spike_positions),
                        'avg_attention_cycle_length': float(avg_spike_cycle),
                        'attention_intensity': float(volume_data[volume_spikes].mean() / volume_ma.mean() if volume_ma.mean() > 0 else 1)
                    }
                    
            # Calculate overall cycle strength
            if cycle_patterns:
                cycle_strength_scores = []
                
                if 'price_cycles' in cycle_patterns:
                    price_cycle = cycle_patterns['price_cycles']
                    cycle_strength = (price_cycle['peak_frequency'] + price_cycle['trough_frequency']) / 2
                    cycle_strength_scores.append(min(1.0, cycle_strength / 5))  # Normalize
                    
                if 'attention_cycles' in cycle_patterns:
                    attention_cycle = cycle_patterns['attention_cycles']
                    cycle_strength_scores.append(min(1.0, attention_cycle['spike_frequency'] / 3))
                    
                if cycle_strength_scores:
                    cycle_patterns['overall_cycle_strength'] = float(np.mean(cycle_strength_scores))
                    
            return cycle_patterns
            
        except Exception as e:
            logger.error(f"Error detecting story cycles for {symbol}: {e}")
            return {}
            
    def _assess_story_cycles(self, cycle_data: Dict) -> Dict:
        """Assess overall story cycle patterns"""
        try:
            assessment = {
                'cycle_activity_level': 'low',
                'cycle_synchronization': 0.0,
                'cycle_predictability': 'low',
                'dominant_cycle_pattern': 'none',
                'cycle_health': 'stable'
            }
                
            if not cycle_data:
                return assessment
                
            # Analyze cycle activity
            activity_levels = []
            cycle_strengths = []
            
            for symbol, patterns in cycle_data.items():
                strength = patterns.get('overall_cycle_strength', 0)
                cycle_strengths.append(strength)
                
                # Calculate activity level based on frequency
                price_cycles = patterns.get('price_cycles', {})
                attention_cycles = patterns.get('attention_cycles', {})
                
                total_activity = (price_cycles.get('peak_frequency', 0) + 
                                price_cycles.get('trough_frequency', 0) + 
                                attention_cycles.get('spike_frequency', 0))
                activity_levels.append(total_activity)
                
            # Overall cycle activity
            avg_activity = np.mean(activity_levels) if activity_levels else 0
            avg_strength = np.mean(cycle_strengths) if cycle_strengths else 0
            
            if avg_activity > 5:
                assessment['cycle_activity_level'] = 'high'
            elif avg_activity > 2:
                assessment['cycle_activity_level'] = 'moderate'
            else:
                assessment['cycle_activity_level'] = 'low'
                
            # Cycle synchronization
            activity_std = np.std(activity_levels) if len(activity_levels) > 1 else 0
            synchronization = 1 / (1 + activity_std) if activity_std > 0 else 1
            assessment['cycle_synchronization'] = float(synchronization)
            
            # Cycle predictability
            if avg_strength > 0.7:
                assessment['cycle_predictability'] = 'high'
            elif avg_strength > 0.4:
                assessment['cycle_predictability'] = 'moderate'
            else:
                assessment['cycle_predictability'] = 'low'
                
            # Dominant cycle pattern
            if avg_activity > 4 and avg_strength > 0.6:
                assessment['dominant_cycle_pattern'] = 'active_recurring'
            elif avg_activity > 2:
                assessment['dominant_cycle_pattern'] = 'moderate_recurring'
            elif avg_strength > 0.3:
                assessment['dominant_cycle_pattern'] = 'weak_recurring'
            else:
                assessment['dominant_cycle_pattern'] = 'erratic'
                
            # Cycle health
            if avg_strength > 0.8 and synchronization > 0.7:
                assessment['cycle_health'] = 'strong'
            elif avg_strength > 0.5 and synchronization > 0.4:
                assessment['cycle_health'] = 'stable'
            elif avg_strength > 0.2:
                assessment['cycle_health'] = 'weak'
            else:
                assessment['cycle_health'] = 'deteriorating'
                
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing story cycles: {e}")
            return {'cycle_activity_level': 'low', 'cycle_health': 'stable'}
            
    def _classify_cycle_patterns(self, cycle_data: Dict) -> Dict:
        """Classify different types of cycle patterns"""
        try:
            classification = {
                'cycle_types': [],
                'pattern_diversity': 0.0,
                'pattern_complexity': 'simple',
                'cycle_stability': 'unstable'
            }
                
            cycle_types = []
            
            for symbol, patterns in cycle_data.items():
                pattern_type = self._determine_cycle_type(patterns)
                if pattern_type:
                    cycle_types.append({
                        'symbol': symbol,
                        'pattern_type': pattern_type
                    })
                    
            classification['cycle_types'] = cycle_types
            
            # Pattern diversity
            unique_patterns = set(pt['pattern_type'] for pt in cycle_types)
            pattern_diversity = len(unique_patterns) / len(cycle_types) if cycle_types else 0
            classification['pattern_diversity'] = float(pattern_diversity)
            
            # Pattern complexity
            if pattern_diversity > 0.8:
                classification['pattern_complexity'] = 'highly_complex'
            elif pattern_diversity > 0.5:
                classification['pattern_complexity'] = 'moderately_complex'
            else:
                classification['pattern_complexity'] = 'simple'
                
            # Cycle stability
            regular_cycles = sum(1 for pt in cycle_types if 'regular' in pt['pattern_type'])
            stability_ratio = regular_cycles / len(cycle_types) if cycle_types else 0
            
            if stability_ratio > 0.7:
                classification['cycle_stability'] = 'very_stable'
            elif stability_ratio > 0.4:
                classification['cycle_stability'] = 'stable'
            else:
                classification['cycle_stability'] = 'unstable'
                
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying cycle patterns: {e}")
            return {'cycle_types': [], 'pattern_diversity': 0.0}
            
    def _determine_cycle_type(self, patterns: Dict) -> Optional[str]:
        """Determine the type of cycle pattern"""
        try:
            price_cycles = patterns.get('price_cycles', {})
            attention_cycles = patterns.get('attention_cycles', {})
            overall_strength = patterns.get('overall_cycle_strength', 0)
            
            if overall_strength < 0.3:
                return 'weak_cycle'
                
            # Determine pattern characteristics
            has_regular_price_cycle = (price_cycles.get('cycle_regularity', 0) > 0.6 and 
                                     price_cycles.get('peak_frequency', 0) > 2)
            
            has_regular_attention_cycle = (attention_cycles.get('spike_frequency', 0) > 2 and
                                         attention_cycles.get('attention_intensity', 1) > 2)
            
            # Classify pattern type
            if has_regular_price_cycle and has_regular_attention_cycle:
                return 'dual_regular_cycle'
            elif has_regular_price_cycle:
                return 'price_regular_cycle'
            elif has_regular_attention_cycle:
                return 'attention_regular_cycle'
            elif overall_strength > 0.6:
                return 'strong_irregular_cycle'
            else:
                return 'weak_irregular_cycle'
                
        except Exception as e:
            logger.error(f"Error determining cycle type: {e}")
            return None
            
    def _generate_cycle_predictions(self, cycle_data: Dict) -> Dict:
        """Generate cycle-based predictions"""
        try:
            predictions = {
                'next_cycle_probability': 0.0,
                'cycle_direction': 'stable',
                'prediction_confidence': 'low',
                'key_prediction_factors': []
            }
                
            if not cycle_data:
                return predictions
                
            # Analyze cycle patterns for predictions
            cycle_strengths = []
            cycle_types = []
            
            for symbol, patterns in cycle_data.items():
                strength = patterns.get('overall_cycle_strength', 0)
                cycle_types.append(self._determine_cycle_type(patterns))
                cycle_strengths.append(strength)
                
            # Calculate prediction probability
            avg_strength = np.mean(cycle_strengths) if cycle_strengths else 0
            regular_patterns = sum(1 for ct in cycle_types if ct and 'regular' in ct)
            
            predictions['next_cycle_probability'] = float(min(1.0, avg_strength * 0.8 + regular_patterns * 0.2))
            
            # Predict cycle direction
            if avg_strength > 0.7:
                predictions['cycle_direction'] = 'highly_active'
            elif avg_strength > 0.4:
                predictions['cycle_direction'] = 'moderately_active'
            elif avg_strength < 0.2:
                predictions['cycle_direction'] = 'inactive'
            else:
                predictions['cycle_direction'] = 'stable'
                
            # Prediction confidence
            prediction_variance = np.var(cycle_strengths) if len(cycle_strengths) > 1 else 0
            consistency_score = 1 / (1 + prediction_variance) if prediction_variance > 0 else 1
            
            if consistency_score > 0.8:
                predictions['prediction_confidence'] = 'high'
            elif consistency_score > 0.5:
                predictions['prediction_confidence'] = 'moderate'
            else:
                predictions['prediction_confidence'] = 'low'
                
            # Key prediction factors
            if avg_strength > 0.6:
                predictions['key_prediction_factors'].append('strong_cycle_activity')
            if regular_patterns > len(cycle_types) * 0.5:
                predictions['key_prediction_factors'].append('regular_patterns')
            if len(cycle_types) > 3:
                predictions['key_prediction_factors'].append('diverse_cycle_sources')
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating cycle predictions: {e}")
            return {'next_cycle_probability': 0.0, 'prediction_confidence': 'low'}
            
    async def _narrative_impact_analysis(self) -> Dict:
        """Analyze the market impact of dominant narratives"""
        try:
            # Analyze impact across different market segments
            impact_symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLF", "XLK"]
            impact_data = await self._fetch_market_narrative_data(impact_symbols)
            
            impact_analysis = {}
            
            for symbol, data in impact_data.items():
                impact_metrics = self._calculate_narrative_impact(symbol, data)
                if impact_metrics:
                    impact_analysis[symbol] = impact_metrics
                    
            if not impact_analysis:
                return {'error': 'No narrative impact data available'}
                
            # Overall impact assessment
            impact_assessment = self._assess_narrative_impact(impact_analysis)
            
            # Sector impact analysis
            sector_impact = self._analyze_sector_impact(impact_analysis)
            
            # Cross-asset impact correlation
            impact_correlation = self._analyze_impact_correlation(impact_analysis)
            
            return {
                'impact_analysis': impact_analysis,
                'impact_assessment': impact_assessment,
                'sector_impact': sector_impact,
                'impact_correlation': impact_correlation
            }
            
        except Exception as e:
            logger.error(f"Error in narrative impact analysis: {e}")
            return {'error': str(e)}
            
    def _calculate_narrative_impact(self, symbol: str, data: Dict) -> Dict:
        """Calculate narrative impact for an asset"""
        try:
            if not data:
                return {}
                
            impact_metrics = {}
            
            # Volatility impact (narrative-driven volatility)
            price_data = data.get('price_data', pd.Series())
            if len(price_data) > 10:
                returns = price_data.pct_change().dropna()
                current_volatility = returns.rolling(5).std().iloc[-1]
                historical_volatility = returns.rolling(20).std().mean()
                
                volatility_impact = current_volatility / historical_volatility if historical_volatility > 0 else 1
                impact_metrics['volatility_impact'] = float(volatility_impact)
                
            # Volume impact (attention-driven trading)
            volume_data = data.get('volume_data', pd.Series())
            if len(volume_data) > 10:
                current_volume = volume_data.iloc[-1]
                avg_volume = volume_data.rolling(20).mean().iloc[-1]
                
                volume_impact = current_volume / avg_volume if avg_volume > 0 else 1
                impact_metrics['volume_impact'] = float(volume_impact)
                
            # Price impact (narrative-driven price moves)
            if len(price_data) > 10:
                price_change_5d = price_data.pct_change(5).iloc[-1]
                price_change_20d = price_data.pct_change(20).iloc[-1]
                
                # Short-term vs long-term momentum (narrative acceleration)
                momentum_divergence = abs(price_change_5d - price_change_20d)
                impact_metrics['price_impact'] = float(abs(price_change_5d))
                impact_metrics['momentum_divergence'] = float(momentum_divergence)
                
            # Overall impact score
            impact_components = [
                impact_metrics.get('volatility_impact', 1),
                impact_metrics.get('volume_impact', 1),
                impact_metrics.get('price_impact', 0)
            ]
            
            # Normalize and combine
            normalized_impact = np.mean([
                min(2.0, comp) for comp in impact_components if comp is not None
            ])
            impact_metrics['overall_impact_score'] = float(min(2.0, normalized_impact))
            
            # Impact classification
            overall_impact = impact_metrics['overall_impact_score']
            if overall_impact > 1.5:
                impact_metrics['impact_classification'] = 'high_impact'
            elif overall_impact > 1.0:
                impact_metrics['impact_classification'] = 'moderate_impact'
            else:
                impact_metrics['impact_classification'] = 'low_impact'
                
            return impact_metrics
            
        except Exception as e:
            logger.error(f"Error calculating narrative impact for {symbol}: {e}")
            return {}
            
    def _assess_narrative_impact(self, impact_data: Dict) -> Dict:
        """Assess overall narrative impact across markets"""
        try:
            assessment = {
                'overall_impact_level': 'moderate',
                'impact_intensity': 0.0,
                'impact_distribution': {},
                'high_impact_assets': [],
                'impact_concentration': 0.0
            }
                
            if not impact_data:
                return assessment
                
            # Calculate impact metrics
            impact_scores = []
            classification_counts = {}
            
            for symbol, metrics in impact_data.items():
                overall_impact = metrics.get('overall_impact_score', 0)
                impact_scores.append(overall_impact)
                
                classification = metrics.get('impact_classification', 'low_impact')
                classification_counts[classification] = classification_counts.get(classification, 0) + 1
                
            # Overall impact level
            avg_impact = np.mean(impact_scores) if impact_scores else 0
            
            if avg_impact > 1.4:
                assessment['overall_impact_level'] = 'high'
            elif avg_impact > 1.0:
                assessment['overall_impact_level'] = 'moderate'
            else:
                assessment['overall_impact_level'] = 'low'
                
            assessment['impact_intensity'] = float(avg_impact)
            
            # Impact distribution
            total_assets = len(impact_data)
            assessment['impact_distribution'] = {
                classification: count / total_assets 
                for classification, count in classification_counts.items()
            }
            
            # High impact assets
            sorted_assets = sorted(
                impact_data.items(),
                key=lambda x: x[1].get('overall_impact_score', 0),
                reverse=True
            )
            assessment['high_impact_assets'] = [asset[0] for asset in sorted_assets[:3]]
            
            # Impact concentration
            max_impact = max(impact_scores) if impact_scores else 0
            impact_concentration = max_impact / np.sum(impact_scores) if np.sum(impact_scores) > 0 else 0
            assessment['impact_concentration'] = float(impact_concentration)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing narrative impact: {e}")
            return {'overall_impact_level': 'moderate', 'impact_intensity': 0.0}
            
    def _analyze_sector_impact(self, impact_data: Dict) -> Dict:
        """Analyze narrative impact by sector"""
        try:
            # Define sector mappings
            sector_mapping = {
                'SPY': 'broad_market', 'QQQ': 'growth', 'IWM': 'small_cap',
                'TLT': 'bonds', 'GLD': 'commodities',
                'XLF': 'financials', 'XLK': 'technology'
            }
            
            sector_impact = {}
            
            for symbol, metrics in impact_data.items():
                sector = sector_mapping.get(symbol, 'other')
                
                if sector not in sector_impact:
                    sector_impact[sector] = []
                    
                sector_impact[sector].append(metrics.get('overall_impact_score', 0))
                
            # Calculate sector-level impact
            sector_assessment = {}
            for sector, impact_scores in sector_impact.items():
                avg_impact = np.mean(impact_scores) if impact_scores else 0
                max_impact = max(impact_scores) if impact_scores else 0
                
                sector_assessment[sector] = {
                    'average_impact': float(avg_impact),
                    'maximum_impact': float(max_impact),
                    'impact_volatility': float(np.std(impact_scores)) if len(impact_scores) > 1 else 0
                }
                
            # Rank sectors by impact
            ranked_sectors = sorted(
                sector_assessment.items(),
                key=lambda x: x[1]['average_impact'],
                reverse=True
            )
            
            return {
                'sector_assessment': sector_assessment,
                'most_impacted_sectors': [sector[0] for sector in ranked_sectors[:3]],
                'sector_impact_spread': float(np.std([s[1]['average_impact'] for s in ranked_sectors]))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sector impact: {e}")
            return {'sector_assessment': {}, 'most_impacted_sectors': []}
            
    def _analyze_impact_correlation(self, impact_data: Dict) -> Dict:
        """Analyze correlation between narrative impacts"""
        try:
            correlation = {
                'impact_correlation': 0.0,
                'synchronized_impact': False,
                'correlation_regime': 'normal',
                'impact_coordination': 'mixed'
            }
                
            if not impact_data or len(impact_data) < 2:
                return correlation
                
            impact_scores = [metrics.get('overall_impact_score', 0) for metrics in impact_data.values()]
            
            if len(impact_scores) > 1:
                # Use standard deviation as proxy for correlation
                impact_std = np.std(impact_scores)
                impact_mean = np.mean(impact_scores)
                
                # Lower std = higher correlation
                correlation_proxy = 1 / (1 + impact_std * 5) if impact_std > 0 else 1
                correlation['impact_correlation'] = float(correlation_proxy)
                
                # Synchronized impact detection
                high_impact_count = sum(1 for score in impact_scores if score > 1.2)
                synchronized_ratio = high_impact_count / len(impact_scores)
                
                if synchronized_ratio > 0.6:
                    correlation['synchronized_impact'] = True
                    
                # Correlation regime
                if correlation_proxy > 0.8:
                    correlation['correlation_regime'] = 'high_correlation'
                elif correlation_proxy < 0.3:
                    correlation['correlation_regime'] = 'low_correlation'
                else:
                    correlation['correlation_regime'] = 'normal_correlation'
                    
                # Impact coordination
                if synchronized_ratio > 0.7:
                    correlation['impact_coordination'] = 'highly_coordinated'
                elif synchronized_ratio > 0.4:
                    correlation['impact_coordination'] = 'moderately_coordinated'
                else:
                    correlation['impact_coordination'] = 'fragmented'
                    
            return correlation
            
        except Exception as e:
            logger.error(f"Error analyzing impact correlation: {e}")
            return {'impact_correlation': 0.0, 'correlation_regime': 'normal'}
            
    async def get_narrative_pulse(self) -> Dict:
        """Get comprehensive narrative intelligence analysis"""
        try:
            # Run all narrative analyses in parallel
            narrative_tasks = [
                self._dominant_narratives_analysis(),
                self._narrative_momentum_analysis(),
                self._narrative_lifecycle_analysis(),
                self._story_cycle_detection(),
                self._narrative_impact_analysis()
            ]
            
            results = await asyncio.gather(*narrative_tasks, return_exceptions=True)
            (
                dominant_narratives, narrative_momentum,
                narrative_lifecycle, story_cycles,
                narrative_impact
            ) = results
            
            # Calculate overall Narrative Intelligence Score (NIS)
            nis_components = []
            
            for result in results:
                if isinstance(result, dict) and not result.get('error'):
                    # Extract numeric scores from each analysis
                    score = self._extract_narrative_score(result)
                    if score is not None:
                        nis_components.append(score)
                        
            if nis_components:
                nis_score = np.mean(nis_components)
                nis_volatility = np.std(nis_components)
                
                # Classify narrative state
                if nis_score > 0.6:
                    narrative_state = 'dominant_narrative_active'
                elif nis_score > 0.3:
                    narrative_state = 'strong_narrative_flow'
                elif nis_score < -0.6:
                    narrative_state = 'narrative_vacuum'
                elif nis_score < -0.3:
                    narrative_state = 'narrative_weakness'
                else:
                    narrative_state = 'balanced_narrative_environment'
                    
                return {
                    'narrative_intelligence_score': nis_score,
                    'nis_volatility': nis_volatility,
                    'narrative_state': narrative_state,
                    'analysis_breakdown': {
                        'dominant_narratives': dominant_narratives,
                        'narrative_momentum': narrative_momentum,
                        'narrative_lifecycle': narrative_lifecycle,
                        'story_cycles': story_cycles,
                        'narrative_impact': narrative_impact
                    },
                    'timestamp': datetime.utcnow(),
                    'confidence': min(1.0, 1 - (nis_volatility / 2))
                }
            else:
                return {'error': 'Unable to calculate narrative intelligence score'}
                
        except Exception as e:
            logger.error(f"Error getting narrative pulse: {e}")
            return {'error': str(e)}
            
    def _extract_narrative_score(self, analysis_result: Dict) -> Optional[float]:
        """Extract numeric narrative score from analysis result"""
        try:
            if 'dominance_assessment' in analysis_result:
                dominance = analysis_result['dominance_assessment']
                concentration = dominance.get('narrative_concentration', 0)
                return (concentration - 0.5) * 2  # Normalize to -1 to 1
            elif 'momentum_assessment' in analysis_result:
                momentum = analysis_result['momentum_assessment']
                strength = momentum.get('momentum_strength', 0)
                direction = momentum.get('overall_momentum_direction', 'neutral')
                if 'bullish' in direction:
                    return min(1.0, strength * 10)
                elif 'bearish' in direction:
                    return -min(1.0, abs(strength) * 10)
                else:
                    return 0.0
            elif 'lifecycle_assessment' in analysis_result:
                lifecycle = analysis_result['lifecycle_assessment']
                health = lifecycle.get('narrative_health', 'stable')
                transition_prob = lifecycle.get('stage_transition_probability', 0)
                if health == 'vibrant':
                    return 0.6 + transition_prob * 0.2
                elif health == 'stable':
                    return 0.2 + transition_prob * 0.1
                elif health == 'declining':
                    return -0.2 - transition_prob * 0.3
                else:
                    return -0.6 - transition_prob * 0.4
            elif 'cycle_assessment' in analysis_result:
                cycle = analysis_result['cycle_assessment']
                activity_level = cycle.get('cycle_activity_level', 'low')
                health = cycle.get('cycle_health', 'stable')
                if activity_level == 'high' and health == 'strong':
                    return 0.7
                elif activity_level == 'moderate' or health == 'stable':
                    return 0.3
                elif activity_level == 'low' or health == 'deteriorating':
                    return -0.5
                else:
                    return 0.0
            elif 'impact_assessment' in analysis_result:
                impact = analysis_result['impact_assessment']
                intensity = impact.get('impact_intensity', 0)
                level = impact.get('overall_impact_level', 'moderate')
                if level == 'high':
                    return 0.5 + intensity * 0.3
                elif level == 'moderate':
                    return intensity * 0.3
                else:
                    return -0.3 + intensity * 0.1
            else:
                return None
                
        except Exception:
            return None
            
    async def store_narrative_data(self, narrative_data: Dict):
        """Store narrative metrics in time-series database"""
        try:
            if self.db_manager and 'timestamp' in narrative_data:
                # Store Narrative Intelligence Score
                await self.db_manager.influxdb_client.write_points(
                    database='market_pulse',
                    measurement='narrative_metrics',
                    tags={
                        'engine': 'narrative_intelligence',
                        'state': narrative_data.get('narrative_state', 'unknown')
                    },
                    fields={
                        'nis_score': float(narrative_data.get('narrative_intelligence_score', 0)),
                        'nis_volatility': float(narrative_data.get('nis_volatility', 0)),
                        'confidence': float(narrative_data.get('confidence', 0))
                    },
                    time=narrative_data['timestamp']
                )
                
                # Store component scores
                for component_name, analysis in narrative_data.get('analysis_breakdown', {}).items():
                    if isinstance(analysis, dict):
                        score = self._extract_narrative_score(analysis)
                        if score is not None:
                            await self.db_manager.influxdb_client.write_points(
                                database='market_pulse',
                                measurement='narrative_components',
                                tags={
                                    'component': component_name,
                                    'engine': 'narrative_intelligence'
                                },
                                fields={'component_score': float(score)},
                                time=narrative_data['timestamp']
                            )
                            
            logger.debug("Narrative data stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing narrative data: {e}")
            
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
                'cache_size': len(self.narrative_cache),
                'models_loaded': len(self.models),
                'narrative_categories': len(self.narrative_categories),
                'narrative_stages': len(self.narrative_stages),
                'sentiment_indicators': len(self.sentiment_indicators),
                'health_score': self._calculate_engine_health()
            }
            
        except Exception as e:
            logger.error(f"Error getting narrative engine status: {e}")
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
            health_factors.append(min(1.0, len(self.models) / 2))
            
            # NLP model availability
            nlp_models_available = (
                hasattr(self, 'sentiment_analyzer') and 
                hasattr(self, 'vectorizer') and 
                hasattr(self, 'narrative_clustering')
            )
            health_factors.append(1.0 if nlp_models_available else 0.0)
            
            return np.mean(health_factors) if health_factors else 0.0
            
        except Exception:
            return 0.0