"""
Event Shockwave Engine - Market Event Detection & Impact Analysis
Real-time detection of market-moving events, shockwave propagation, and event-driven volatility
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

class EventShockwaveEngine:
    """Market Event Monitor - Detecting shockwaves and event-driven market movements"""
    
    def __init__(self):
        self.name = "Event Shockwave Engine"
        self.version = "1.0.0"
        self.status = "initialized"
        self.last_update = None
        self.event_cache = {}
        self.models = {}
        self.scaler = StandardScaler()
        
        # Event detection models
        self.event_models = {
            "volatility_events": self._volatility_event_detection,
            "volume_events": self._volume_event_detection,
            "correlation_events": self._correlation_event_detection,
            "sector_rotation_events": self._sector_rotation_event_detection,
            "cross_asset_events": self._cross_asset_event_detection
        }
        
        # Event categories
        self.event_categories = {
            "earnings_events": {
                "earnings_surprises": "Positive/negative earnings surprises",
                "guidance_changes": "Management guidance revisions",
                "analyst_revisions": "Analyst estimate changes"
            },
            "macroeconomic_events": {
                "fed_decisions": "Federal Reserve policy decisions",
                "economic_releases": "Key economic data releases",
                "geopolitical_events": "Political and geopolitical events"
            },
            "corporate_events": {
                "mergers_acquisitions": "M&A announcements and completions",
                "corporate_actions": "Dividends, splits, buybacks",
                "insider_activity": "Insider trading and corporate filings"
            },
            "market_structure_events": {
                "flash_crashes": "Sudden price movements and halts",
                "liquidity_events": "Market liquidity disruptions",
                "circuit_breaker_events": "Trading halts and circuit breakers"
            },
            "sentiment_events": {
                "fear_greed_extremes": "Extreme fear or greed events",
                "momentum_breakdowns": "Sudden momentum reversals",
                "trend_breaks": "Major trend line breaks"
            }
        }
        
        # Shockwave characteristics
        self.shockwave_phases = {
            "initial_impact": "Immediate market reaction to event",
            "shock_propagation": "Event impact spreads across assets",
            "contagion": "Effects spread to related markets",
            "absorption": "Markets begin to digest and normalize",
            "residual_effects": "Longer-term impacts and adjustments"
        }
        
        # Event impact metrics
        self.impact_metrics = {
            "volatility_spike": "Sudden increase in market volatility",
            "volume_surge": "Unusual trading volume activity",
            "correlation_breakdown": "Disruption in normal correlations",
            "sector_rotation": "Capital reallocation between sectors",
            "liquidity_drain": "Temporary reduction in market liquidity"
        }
        
        # Event detection thresholds
        self.event_thresholds = {
            "volatility_spike": 3.0,      # 3x normal volatility
            "volume_surge": 5.0,          # 5x normal volume
            "correlation_change": 0.5,    # 50% correlation change
            "price_movement": 0.05,       # 5% price movement
            "duration_minutes": 30        # Minimum 30-minute duration
        }
        
        # Shockwave intensity levels
        self.shockwave_intensities = {
            "minor": {"volatility_multiplier": [1.5, 2.5], "duration_hours": [0.5, 2]},
            "moderate": {"volatility_multiplier": [2.5, 4.0], "duration_hours": [2, 6]},
            "major": {"volatility_multiplier": [4.0, 6.0], "duration_hours": [6, 24]},
            "extreme": {"volatility_multiplier": [6.0, 10.0], "duration_hours": [24, 72]}
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
            logger.info("Event Shockwave Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Event Shockwave Engine: {e}")
            self.status = "error"
            
    def _initialize_models(self):
        """Initialize ML models for event detection"""
        try:
            # Random Forest for event classification
            self.models['event_classifier'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Random Forest for shockwave prediction
            self.models['shockwave_predictor'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=8
            )
            
            # Scaler for event feature normalization
            self.scaler = StandardScaler()
            
            logger.info("Event detection models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize event models: {e}")
            
    async def _fetch_event_data(self, symbols: List[str], period: str = "1mo") -> Dict:
        """Fetch market data for event detection"""
        try:
            event_data = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    if not hist.empty and len(hist) > 20:
                        event_data[symbol] = hist
                except Exception as e:
                    logger.warning(f"Error fetching event data for {symbol}: {e}")
                    continue
                    
            return event_data
            
        except Exception as e:
            logger.error(f"Error fetching event data: {e}")
            return {}
            
    async def _volatility_event_detection(self) -> Dict:
        """Detect volatility-related market events"""
        try:
            # Get volatility-sensitive assets
            volatility_symbols = ["VIX", "SPY", "QQQ", "TLT", "VXX"]
            event_data = await self._fetch_event_data(volatility_symbols)
            
            volatility_events = {}
            
            for symbol, data in event_data.items():
                events = self._detect_volatility_events(symbol, data)
                if events:
                    volatility_events[symbol] = events
                    
            if not volatility_events:
                return {'error': 'No volatility events detected'}
                
            # Overall volatility event assessment
            event_assessment = self._assess_volatility_events(volatility_events)
            
            # Shockwave propagation analysis
            propagation = self._analyze_shockwave_propagation(volatility_events)
            
            return {
                'volatility_events': volatility_events,
                'event_assessment': event_assessment,
                'shockwave_propagation': propagation
            }
            
        except Exception as e:
            logger.error(f"Error in volatility event detection: {e}")
            return {'error': str(e)}
            
    def _detect_volatility_events(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Detect volatility events for a specific asset"""
        try:
            if data.empty or len(data) < 20:
                return {}
                
            events = {}
            
            # Calculate returns and volatility
            returns = data['Close'].pct_change().dropna()
            volatility = returns.rolling(5).std()  # 5-day rolling volatility
            
            # Detect volatility spikes
            vol_ma = volatility.rolling(20).mean()
            vol_std = volatility.rolling(20).std()
            
            # Identify volatility spike events
            vol_spikes = volatility > (vol_ma + 2 * vol_std)
            
            # Group consecutive spike days
            spike_groups = []
            current_group = []
            
            for i, is_spike in enumerate(vol_spikes):
                if is_spike:
                    current_group.append(i)
                else:
                    if current_group:
                        spike_groups.append(current_group)
                        current_group = []
                        
            # Add final group if it exists
            if current_group:
                spike_groups.append(current_group)
                
            # Analyze each spike group as an event
            event_id = 0
            for spike_group in spike_groups:
                if len(spike_group) >= 1:  # Minimum 1 day duration
                    event_start = spike_group[0]
                    event_end = spike_group[-1]
                    
                    # Calculate event metrics
                    event_duration = event_end - event_start + 1
                    max_volatility = volatility.iloc[event_start:event_end+1].max()
                    avg_volatility = volatility.iloc[event_start:event_end+1].mean()
                    
                    # Calculate price impact during event
                    price_before = data['Close'].iloc[event_start-1] if event_start > 0 else data['Close'].iloc[0]
                    price_after = data['Close'].iloc[event_end] if event_end < len(data)-1 else data['Close'].iloc[-1]
                    price_impact = abs(price_after - price_before) / price_before if price_before > 0 else 0
                    
                    # Classify event intensity
                    intensity = self._classify_event_intensity(max_volatility, vol_ma.iloc[event_end] if not pd.isna(vol_ma.iloc[event_end]) else vol_ma.mean())
                    
                    events[f'event_{event_id}'] = {
                        'start_date': data.index[event_start],
                        'end_date': data.index[event_end],
                        'duration_days': event_duration,
                        'max_volatility': float(max_volatility),
                        'avg_volatility': float(avg_volatility),
                        'price_impact': float(price_impact),
                        'intensity': intensity,
                        'volatility_multiplier': float(max_volatility / vol_ma.mean() if vol_ma.mean() > 0 else 1)
                    }
                    
                    event_id += 1
                    
            # Recent events (last 7 days)
            recent_events = {}
            cutoff_date = data.index[-7] if len(data) > 7 else data.index[0]
            
            for event_id, event_data in events.items():
                if event_data['start_date'] >= cutoff_date:
                    recent_events[event_id] = event_data
                    
            if recent_events:
                events['recent_events_summary'] = {
                    'count': len(recent_events),
                    'max_intensity': max(event['intensity'] for event in recent_events.values()),
                    'total_price_impact': sum(event['price_impact'] for event in recent_events.values()),
                    'latest_event': max(recent_events.keys(), key=lambda x: recent_events[x]['start_date'])
                }
                
            return events
            
        except Exception as e:
            logger.error(f"Error detecting volatility events for {symbol}: {e}")
            return {}
            
    def _classify_event_intensity(self, event_vol: float, normal_vol: float) -> str:
        """Classify event intensity based on volatility"""
        try:
            multiplier = event_vol / normal_vol if normal_vol > 0 else 1
            
            if multiplier >= 6.0:
                return 'extreme'
            elif multiplier >= 4.0:
                return 'major'
            elif multiplier >= 2.5:
                return 'moderate'
            elif multiplier >= 1.5:
                return 'minor'
            else:
                return 'normal'
                
        except Exception:
            return 'normal'
            
    def _assess_volatility_events(self, volatility_events: Dict) -> Dict:
        """Assess overall volatility event patterns"""
        try:
            assessment = {
                'total_events_detected': 0,
                'recent_activity_level': 'low',
                'dominant_intensity': 'normal',
                'average_impact': 0.0,
                'event_clusters': [],
                'market_stress_level': 'normal'
            }
                
            # Aggregate events across all symbols
            all_events = []
            recent_events = []
            intensity_counts = {'minor': 0, 'moderate': 0, 'major': 0, 'extreme': 0}
            
            for symbol, events in volatility_events.items():
                if 'recent_events_summary' in events:
                    recent_count = events['recent_events_summary']['count']
                    recent_events.append(recent_count)
                    assessment['total_events_detected'] += recent_count
                    
                # Collect all individual events
                for event_id, event_data in events.items():
                    if not event_id.startswith('recent_events_'):
                        all_events.append(event_data)
                        intensity = event_data.get('intensity', 'normal')
                        if intensity in intensity_counts:
                            intensity_counts[intensity] += 1
                            
            # Recent activity level
            total_recent = sum(recent_events)
            if total_recent >= 5:
                assessment['recent_activity_level'] = 'high'
            elif total_recent >= 2:
                assessment['recent_activity_level'] = 'moderate'
            else:
                assessment['recent_activity_level'] = 'low'
                
            # Dominant intensity
            max_intensity = max(intensity_counts.items(), key=lambda x: x[1])
            assessment['dominant_intensity'] = max_intensity[0] if max_intensity[1] > 0 else 'normal'
            
            # Average impact
            if all_events:
                avg_impact = np.mean([event['price_impact'] for event in all_events])
                assessment['average_impact'] = float(avg_impact)
                
            # Market stress level
            extreme_count = intensity_counts['extreme']
            major_count = intensity_counts['major']
            
            if extreme_count >= 2:
                assessment['market_stress_level'] = 'extreme_stress'
            elif major_count >= 3 or extreme_count >= 1:
                assessment['market_stress_level'] = 'high_stress'
            elif major_count >= 1:
                assessment['market_stress_level'] = 'moderate_stress'
            else:
                assessment['market_stress_level'] = 'normal'
                
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing volatility events: {e}")
            return {'total_events_detected': 0, 'market_stress_level': 'normal'}
            
    def _analyze_shockwave_propagation(self, volatility_events: Dict) -> Dict:
        """Analyze how volatility events propagate across markets"""
        try:
            propagation = {
                'propagation_detected': False,
                'propagation_pattern': 'isolated',
                'cross_market_impact': 0.0,
                'propagation_timing': {}
            }
                
            if len(volatility_events) < 2:
                return propagation
                
            # Analyze timing of events across different assets
            event_timings = {}
            
            for symbol, events in volatility_events.items():
                if 'recent_events_summary' in events:
                    latest_event = events['recent_events_summary'].get('latest_event', '')
                    if latest_event and latest_event in events:
                        event_timings[symbol] = events[latest_event]['start_date']
                        
            # Detect propagation patterns
            if len(event_timings) >= 2:
                # Sort by timing
                sorted_timings = sorted(event_timings.items(), key=lambda x: x[1])
                
                # Check for sequential propagation
                time_diffs = []
                for i in range(len(sorted_timings) - 1):
                    time_diff = (sorted_timings[i+1][1] - sorted_timings[i][1]).total_seconds() / 3600  # hours
                    time_diffs.append(time_diff)
                    
                # Propagation criteria
                if time_diffs and all(diff < 24 for diff in time_diffs):  # Within 24 hours
                    propagation['propagation_detected'] = True
                    propagation['propagation_pattern'] = 'sequential'
                    propagation['cross_market_impact'] = float(1 - (max(time_diffs) - min(time_diffs)) / 24)
                elif len(time_diffs) > 1 and np.std(time_diffs) < 6:  # Similar timing
                    propagation['propagation_detected'] = True
                    propagation['propagation_pattern'] = 'simultaneous'
                    propagation['cross_market_impact'] = 0.8
                    
            # Propagating assets
            propagating_assets = list(event_timings.keys())
            propagation['propagating_assets'] = propagating_assets
            
            return propagation
            
        except Exception as e:
            logger.error(f"Error analyzing shockwave propagation: {e}")
            return {'propagation_detected': False, 'cross_market_impact': 0.0}
            
    async def _volume_event_detection(self) -> Dict:
        """Detect volume-based market events"""
        try:
            # Get assets with significant volume patterns
            volume_symbols = ["SPY", "QQQ", "TLT", "GOLD", "BTC-USD"]
            event_data = await self._fetch_event_data(volume_symbols)
            
            volume_events = {}
            
            for symbol, data in event_data.items():
                events = self._detect_volume_events(symbol, data)
                if events:
                    volume_events[symbol] = events
                    
            if not volume_events:
                return {'error': 'No volume events detected'}
                
            # Volume event assessment
            assessment = self._assess_volume_events(volume_events)
            
            return {
                'volume_events': volume_events,
                'assessment': assessment
            }
            
        except Exception as e:
            logger.error(f"Error in volume event detection: {e}")
            return {'error': str(e)}
            
    def _detect_volume_events(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Detect volume events for a specific asset"""
        try:
            if data.empty or len(data) < 20:
                return {}
                
            events = {}
            
            # Calculate volume metrics
            volume = data['Volume']
            price = data['Close']
            
            # Volume baseline and anomalies
            volume_ma = volume.rolling(20).mean()
            volume_std = volume.rolling(20).std()
            
            # Detect volume spikes (5x normal)
            volume_spikes = volume > (volume_ma + 5 * volume_std)
            
            # Group consecutive spike days
            spike_groups = []
            current_group = []
            
            for i, is_spike in enumerate(volume_spikes):
                if is_spike:
                    current_group.append(i)
                else:
                    if current_group:
                        spike_groups.append(current_group)
                        current_group = []
                        
            if current_group:
                spike_groups.append(current_group)
                
            # Analyze each spike group
            event_id = 0
            for spike_group in spike_groups:
                if len(spike_group) >= 1:
                    event_start = spike_group[0]
                    event_end = spike_group[-1]
                    
                    # Event metrics
                    event_duration = event_end - event_start + 1
                    max_volume = volume.iloc[event_start:event_end+1].max()
                    avg_volume = volume.iloc[event_start:event_end+1].mean()
                    baseline_volume = volume_ma.iloc[event_end]
                    
                    # Price impact during event
                    price_before = price.iloc[event_start-1] if event_start > 0 else price.iloc[0]
                    price_after = price.iloc[event_end] if event_end < len(price)-1 else price.iloc[-1]
                    price_impact = (price_after - price_before) / price_before if price_before > 0 else 0
                    
                    # Volume intensity
                    volume_intensity = max_volume / baseline_volume if baseline_volume > 0 else 1
                    
                    events[f'volume_event_{event_id}'] = {
                        'start_date': data.index[event_start],
                        'end_date': data.index[event_end],
                        'duration_days': event_duration,
                        'max_volume': float(max_volume),
                        'avg_volume': float(avg_volume),
                        'baseline_volume': float(baseline_volume),
                        'price_impact': float(price_impact),
                        'volume_intensity': float(volume_intensity),
                        'event_type': self._classify_volume_event(volume_intensity, price_impact)
                    }
                    
                    event_id += 1
                    
            return events
            
        except Exception as e:
            logger.error(f"Error detecting volume events for {symbol}: {e}")
            return {}
            
    def _classify_volume_event(self, volume_intensity: float, price_impact: float) -> str:
        """Classify volume event type"""
        try:
            if volume_intensity > 10 and abs(price_impact) > 0.05:
                return 'breakout'
            elif volume_intensity > 8:
                return 'accumulation'
            elif abs(price_impact) > 0.03:
                return 'price_driven'
            elif volume_intensity > 5:
                return 'attention_spike'
            else:
                return 'normal_volume'
                
        except Exception:
            return 'unknown'
            
    def _assess_volume_events(self, volume_events: Dict) -> Dict:
        """Assess overall volume event patterns"""
        try:
            assessment = {
                'total_volume_events': 0,
                'event_frequency': 'low',
                'dominant_event_type': 'normal',
                'average_volume_intensity': 0.0,
                'breakout_events': 0,
                'accumulation_events': 0
            }
                
            # Aggregate event data
            event_types = {}
            volume_intensities = []
            
            for symbol, events in volume_events.items():
                for event_id, event_data in events.items():
                    if not event_id.startswith('volume_event_'):
                        continue
                        
                    assessment['total_volume_events'] += 1
                    
                    # Track event types
                    event_type = event_data.get('event_type', 'normal')
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                    
                    # Track intensities
                    intensity = event_data.get('volume_intensity', 1)
                    volume_intensities.append(intensity)
                    
                    # Count specific event types
                    if event_type == 'breakout':
                        assessment['breakout_events'] += 1
                    elif event_type == 'accumulation':
                        assessment['accumulation_events'] += 1
                        
            # Event frequency
            total_events = assessment['total_volume_events']
            if total_events >= 10:
                assessment['event_frequency'] = 'high'
            elif total_events >= 5:
                assessment['event_frequency'] = 'moderate'
            else:
                assessment['event_frequency'] = 'low'
                
            # Dominant event type
            if event_types:
                dominant_type = max(event_types.items(), key=lambda x: x[1])[0]
                assessment['dominant_event_type'] = dominant_type
                
            # Average intensity
            if volume_intensities:
                assessment['average_volume_intensity'] = float(np.mean(volume_intensities))
                
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing volume events: {e}")
            return {'total_volume_events': 0, 'event_frequency': 'low'}
            
    async def _correlation_event_detection(self) -> Dict:
        """Detect correlation breakdown events"""
        try:
            # Get multiple assets for correlation analysis
            correlation_symbols = ["SPY", "QQQ", "TLT", "GLD", "XLF", "XLK"]
            event_data = await self._fetch_event_data(correlation_symbols)
            
            correlation_events = {}
            
            # Calculate correlations between pairs
            symbols = list(event_data.keys())
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    events = self._detect_correlation_events(symbol1, symbol2, event_data[symbol1], event_data[symbol2])
                    if events:
                        pair_key = f"{symbol1}_{symbol2}"
                        correlation_events[pair_key] = events
                        
            if not correlation_events:
                return {'error': 'No correlation events detected'}
                
            return {
                'correlation_events': correlation_events,
                'assessment': self._assess_correlation_events(correlation_events)
            }
            
        except Exception as e:
            logger.error(f"Error in correlation event detection: {e}")
            return {'error': str(e)}
            
    def _detect_correlation_events(self, symbol1: str, symbol2: str, data1: pd.DataFrame, data2: pd.DataFrame) -> Dict:
        """Detect correlation breakdown events between two assets"""
        try:
            if data1.empty or data2.empty:
                return {}
                
            # Calculate returns
            returns1 = data1['Close'].pct_change().dropna()
            returns2 = data2['Close'].pct_change().dropna()
            
            # Align data
            combined_data = pd.concat([returns1, returns2], axis=1).dropna()
            if combined_data.empty:
                return {}
                
            # Calculate rolling correlations
            rolling_corr = combined_data.rolling(20).corr().iloc[:, 1]  # Get correlation column
            
            # Calculate baseline correlation
            baseline_corr = rolling_corr.rolling(60).mean()
            
            # Detect correlation breakdowns (significant deviations from baseline)
            corr_threshold = 0.3  # 30% deviation threshold
            correlation_breakdowns = abs(rolling_corr - baseline_corr) > corr_threshold
            
            # Group consecutive breakdown periods
            breakdown_groups = []
            current_group = []
            
            for i, is_breakdown in enumerate(correlation_breakdowns):
                if is_breakdown:
                    current_group.append(i)
                else:
                    if current_group:
                        breakdown_groups.append(current_group)
                        current_group = []
                        
            if current_group:
                breakdown_groups.append(current_group)
                
            # Analyze each breakdown group
            events = {}
            event_id = 0
            
            for breakdown_group in breakdown_groups:
                if len(breakdown_group) >= 2:  # Minimum 2 days
                    event_start = breakdown_group[0]
                    event_end = breakdown_group[-1]
                    
                    # Event metrics
                    event_duration = event_end - event_start + 1
                    correlation_before = rolling_corr.iloc[event_start-5:event_start].mean() if event_start >= 5 else rolling_corr.iloc[:event_start].mean()
                    correlation_during = rolling_corr.iloc[event_start:event_end+1].mean()
                    correlation_after = rolling_corr.iloc[event_end+1:event_end+6].mean() if event_end+6 < len(rolling_corr) else rolling_corr.iloc[event_end+1:].mean()
                    
                    # Correlation impact
                    max_breakdown = abs(rolling_corr.iloc[event_start:event_end+1] - baseline_corr.iloc[event_start:event_end+1]).max()
                    
                    events[f'correlation_breakdown_{event_id}'] = {
                        'asset_pair': f"{symbol1}_{symbol2}",
                        'start_date': combined_data.index[event_start],
                        'end_date': combined_data.index[event_end],
                        'duration_days': event_duration,
                        'correlation_before': float(correlation_before),
                        'correlation_during': float(correlation_during),
                        'correlation_after': float(correlation_after),
                        'max_breakdown': float(max_breakdown),
                        'breakdown_type': self._classify_correlation_breakdown(correlation_before, correlation_during)
                    }
                    
                    event_id += 1
                    
            return events
            
        except Exception as e:
            logger.error(f"Error detecting correlation events between {symbol1} and {symbol2}: {e}")
            return {}
            
    def _classify_correlation_breakdown(self, corr_before: float, corr_during: float) -> str:
        """Classify type of correlation breakdown"""
        try:
            if pd.isna(corr_before) or pd.isna(corr_during):
                return 'unknown'
                
            change = corr_during - corr_before
            
            if abs(change) > 0.7:
                return 'complete_breakdown'
            elif abs(change) > 0.4:
                return 'major_disruption'
            elif abs(change) > 0.2:
                return 'moderate_shift'
            else:
                return 'minor_fluctuation'
                
        except Exception:
            return 'unknown'
            
    def _assess_correlation_events(self, correlation_events: Dict) -> Dict:
        """Assess overall correlation event patterns"""
        try:
            assessment = {
                'total_correlation_events': 0,
                'breakdown_frequency': 'low',
                'dominant_breakdown_type': 'minor',
                'correlation_stability': 'stable',
                'most_affected_pairs': []
            }
                
            # Aggregate event data
            breakdown_types = {}
            affected_pairs = []
            
            for pair, events in correlation_events.items():
                affected_pairs.append(pair)
                assessment['total_correlation_events'] += len(events)
                
                for event_id, event_data in events.items():
                    breakdown_type = event_data.get('breakdown_type', 'unknown')
                    breakdown_types[breakdown_type] = breakdown_types.get(breakdown_type, 0) + 1
                    
            # Breakdown frequency
            total_events = assessment['total_correlation_events']
            if total_events >= 8:
                assessment['breakdown_frequency'] = 'high'
            elif total_events >= 4:
                assessment['breakdown_frequency'] = 'moderate'
            else:
                assessment['breakdown_frequency'] = 'low'
                
            # Dominant breakdown type
            if breakdown_types:
                dominant_type = max(breakdown_types.items(), key=lambda x: x[1])[0]
                assessment['dominant_breakdown_type'] = dominant_type
                
            # Correlation stability
            major_disruptions = breakdown_types.get('major_disruption', 0) + breakdown_types.get('complete_breakdown', 0)
            total_pairs = len(affected_pairs)
            
            if major_disruptions >= total_pairs * 0.5:
                assessment['correlation_stability'] = 'unstable'
            elif major_disruptions >= total_pairs * 0.3:
                assessment['correlation_stability'] = 'moderate'
            else:
                assessment['correlation_stability'] = 'stable'
                
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing correlation events: {e}")
            return {'total_correlation_events': 0, 'correlation_stability': 'stable'}
            
    async def _sector_rotation_event_detection(self) -> Dict:
        """Detect sector rotation events"""
        try:
            # Get sector ETF data
            sector_symbols = ["XLK", "XLF", "XLE", "XLI", "XLV", "XLY", "XLU", "XLB", "XLRE", "XLC"]
            event_data = await self._fetch_event_data(sector_symbols)
            
            rotation_events = self._detect_sector_rotation_events(event_data)
            
            return {
                'rotation_events': rotation_events,
                'assessment': self._assess_rotation_events(rotation_events)
            }
            
        except Exception as e:
            logger.error(f"Error in sector rotation event detection: {e}")
            return {'error': str(e)}
            
    def _detect_sector_rotation_events(self, event_data: Dict) -> Dict:
        """Detect sector rotation events"""
        try:
            if not event_data or len(event_data) < 3:
                return {}
                
            # Calculate sector performance rankings over time
            rotation_signals = {}
            
            # Get rolling performance for each sector
            sector_performance = {}
            for symbol, data in event_data.items():
                if len(data) > 10:
                    returns = data['Close'].pct_change(5).rolling(5).mean()  # 5-day performance
                    sector_performance[symbol] = returns
                    
            if len(sector_performance) < 3:
                return {}
                
            # Detect ranking changes (rotation signals)
            dates = sector_performance[symbol].dropna().index
            
            for i in range(5, len(dates)):  # Need history to detect changes
                current_date = dates[i]
                
                # Current rankings
                current_rankings = {}
                for symbol, perf in sector_performance.items():
                    if current_date in perf.index:
                        current_rankings[symbol] = perf.loc[current_date]
                        
                # Previous rankings (5 days ago)
                prev_date = dates[i-5] if i >= 5 else dates[0]
                prev_rankings = {}
                for symbol, perf in sector_performance.items():
                    if prev_date in perf.index:
                        prev_rankings[symbol] = perf.loc[prev_date]
                        
                if len(current_rankings) >= 3 and len(prev_rankings) >= 3:
                    # Calculate ranking changes
                    sorted_current = sorted(current_rankings.items(), key=lambda x: x[1], reverse=True)
                    sorted_prev = sorted(prev_rankings.items(), key=lambda x: x[1], reverse=True)
                    
                    # Calculate rank changes
                    rank_changes = {}
                    for symbol in current_rankings:
                        current_rank = next(i for i, (s, _) in enumerate(sorted_current) if s == symbol)
                        prev_rank = next(i for i, (s, _) in enumerate(sorted_prev) if s == symbol) if symbol in dict(sorted_prev) else current_rank
                        rank_changes[symbol] = current_rank - prev_rank
                        
                    # Detect significant rotation (top 3 sectors changing by 2+ positions)
                    significant_rotations = {symbol: change for symbol, change in rank_changes.items() if abs(change) >= 2}
                    
                    if significant_rotations:
                        rotation_signals[current_date] = {
                            'rank_changes': rank_changes,
                            'significant_rotations': significant_rotations,
                            'rotation_strength': np.mean([abs(change) for change in significant_rotations.values()])
                        }
                        
            return rotation_signals
            
        except Exception as e:
            logger.error(f"Error detecting sector rotation events: {e}")
            return {}
            
    def _assess_rotation_events(self, rotation_events: Dict) -> Dict:
        """Assess sector rotation patterns"""
        try:
            assessment = {
                'total_rotation_events': len(rotation_events),
                'rotation_frequency': 'low',
                'rotation_intensity': 'weak',
                'rotation_consistency': 'variable',
                'sector_leadership_changes': 0
            }
                
            if not rotation_events:
                return assessment
                
            # Rotation frequency
            event_count = len(rotation_events)
            if event_count >= 10:
                assessment['rotation_frequency'] = 'high'
            elif event_count >= 5:
                assessment['rotation_frequency'] = 'moderate'
            else:
                assessment['rotation_frequency'] = 'low'
                
            # Rotation intensity
            rotation_strengths = [event['rotation_strength'] for event in rotation_events.values()]
            avg_strength = np.mean(rotation_strengths) if rotation_strengths else 0
            
            if avg_strength >= 3:
                assessment['rotation_intensity'] = 'strong'
            elif avg_strength >= 2:
                assessment['rotation_intensity'] = 'moderate'
            else:
                assessment['rotation_intensity'] = 'weak'
                
            # Rotation consistency
            strength_std = np.std(rotation_strengths) if len(rotation_strengths) > 1 else 0
            if strength_std < 0.5:
                assessment['rotation_consistency'] = 'consistent'
            elif strength_std < 1:
                assessment['rotation_consistency'] = 'moderate'
            else:
                assessment['rotation_consistency'] = 'variable'
                
            # Count leadership changes (top sector changes)
            leadership_changes = 0
            for event_data in rotation_events.values():
                significant = event_data.get('significant_rotations', {})
                for symbol, change in significant.items():
                    if change <= -2:  # Moved up significantly
                        leadership_changes += 1
                        
            assessment['sector_leadership_changes'] = leadership_changes
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing rotation events: {e}")
            return {'total_rotation_events': 0, 'rotation_frequency': 'low'}
            
    async def _cross_asset_event_detection(self) -> Dict:
        """Detect cross-asset class events"""
        try:
            # Define cross-asset categories
            cross_asset_categories = {
                "equities": ["SPY", "QQQ"],
                "bonds": ["TLT", "IEF"],
                "commodities": ["GLD", "SLV"],
                "alternatives": ["VXX", "BTC-USD"]
            }
            
            # Get data for each category
            category_data = {}
            for category, symbols in cross_asset_categories.items():
                category_events = []
                for symbol in symbols:
                    event_data = await self._fetch_event_data([symbol])
                    if symbol in event_data:
                        category_events.append(event_data[symbol])
                        
                if category_events:
                    category_data[category] = category_events
                    
            # Detect cross-asset events
            cross_asset_events = self._analyze_cross_asset_coordination(category_data)
            
            return {
                'cross_asset_events': cross_asset_events,
                'assessment': self._assess_cross_asset_events(cross_asset_events)
            }
            
        except Exception as e:
            logger.error(f"Error in cross-asset event detection: {e}")
            return {'error': str(e)}
            
    def _analyze_cross_asset_coordination(self, category_data: Dict) -> Dict:
        """Analyze coordination across asset classes"""
        try:
            coordination_events = {}
            
            # Calculate performance for each asset class
            class_performance = {}
            for category, data_list in category_data.items():
                category_performance = []
                for data in data_list:
                    if len(data) > 5:
                        perf = data['Close'].pct_change(5).iloc[-1]  # 5-day performance
                        category_performance.append(perf)
                        
                if category_performance:
                    class_performance[category] = np.mean(category_performance)
                    
            # Detect coordinated moves
            if len(class_performance) >= 2:
                dates = []
                for category, data_list in category_data.items():
                    for data in data_list:
                        if len(data) > 20:
                            dates.extend(data.index[-20:].tolist())
                            
                unique_dates = sorted(set(dates))
                
                for date in unique_dates[-10:]:  # Last 10 periods
                    date_performance = {}
                    
                    for category, data_list in category_data.items():
                        category_perf = []
                        for data in data_list:
                            if date in data.index:
                                idx = data.index.get_loc(date)
                                if idx >= 5:
                                    perf = data['Close'].pct_change(5).iloc[idx]
                                    category_perf.append(perf)
                                    
                        if category_perf:
                            date_performance[category] = np.mean(category_perf)
                            
                    if len(date_performance) >= 2:
                        # Check for coordinated moves
                        performances = list(date_performance.values())
                        positive_count = sum(1 for p in performances if p > 0)
                        negative_count = len(performances) - positive_count
                        
                        # Coordinated move if 75% of assets move in same direction
                        if positive_count >= len(performances) * 0.75:
                            coordination_type = 'broadly_positive'
                        elif negative_count >= len(performances) * 0.75:
                            coordination_type = 'broadly_negative'
                        else:
                            coordination_type = 'mixed'
                            
                        coordination_events[date] = {
                            'coordination_type': coordination_type,
                            'performance_by_class': date_performance,
                            'coordination_strength': max(positive_count, negative_count) / len(performances)
                        }
                        
            return coordination_events
            
        except Exception as e:
            logger.error(f"Error analyzing cross-asset coordination: {e}")
            return {}
            
    def _assess_cross_asset_events(self, cross_asset_events: Dict) -> Dict:
        """Assess cross-asset event patterns"""
        try:
            assessment = {
                'coordination_events': len(cross_asset_events),
                'coordination_frequency': 'low',
                'dominant_coordination_type': 'mixed',
                'market_coordination': 'fragmented',
                'synchronization_level': 0.0
            }
                
            if not cross_asset_events:
                return assessment
                
            # Coordination frequency
            event_count = len(cross_asset_events)
            if event_count >= 5:
                assessment['coordination_frequency'] = 'high'
            elif event_count >= 2:
                assessment['coordination_frequency'] = 'moderate'
            else:
                assessment['coordination_frequency'] = 'low'
                
            # Dominant coordination type
            coordination_types = {}
            for event in cross_asset_events.values():
                coord_type = event.get('coordination_type', 'mixed')
                coordination_types[coord_type] = coordination_types.get(coord_type, 0) + 1
                
            if coordination_types:
                dominant_type = max(coordination_types.items(), key=lambda x: x[1])[0]
                assessment['dominant_coordination_type'] = dominant_type
                
            # Market coordination level
            coordinated_events = sum(1 for event in cross_asset_events.values() 
                                  if event.get('coordination_type') in ['broadly_positive', 'broadly_negative'])
            total_events = len(cross_asset_events)
            
            coordination_ratio = coordinated_events / total_events if total_events > 0 else 0
            assessment['synchronization_level'] = float(coordination_ratio)
            
            if coordination_ratio > 0.7:
                assessment['market_coordination'] = 'highly_coordinated'
            elif coordination_ratio > 0.4:
                assessment['market_coordination'] = 'moderately_coordinated'
            else:
                assessment['market_coordination'] = 'fragmented'
                
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing cross-asset events: {e}")
            return {'coordination_events': 0, 'market_coordination': 'fragmented'}
            
    async def get_event_pulse(self) -> Dict:
        """Get comprehensive event shockwave analysis"""
        try:
            # Run all event analyses in parallel
            event_tasks = [
                self._volatility_event_detection(),
                self._volume_event_detection(),
                self._correlation_event_detection(),
                self._sector_rotation_event_detection(),
                self._cross_asset_event_detection()
            ]
            
            results = await asyncio.gather(*event_tasks, return_exceptions=True)
            (
                volatility_events, volume_events,
                correlation_events, rotation_events,
                cross_asset_events
            ) = results
            
            # Calculate overall Event Impact Score (EIS)
            eis_components = []
            
            for result in results:
                if isinstance(result, dict) and not result.get('error'):
                    # Extract numeric scores from each analysis
                    score = self._extract_event_score(result)
                    if score is not None:
                        eis_components.append(score)
                        
            if eis_components:
                eis_score = np.mean(eis_components)
                eis_volatility = np.std(eis_components)
                
                # Classify event state
                if eis_score > 0.7:
                    event_state = 'high_event_activity'
                elif eis_score > 0.4:
                    event_state = 'moderate_event_flow'
                elif eis_score < -0.7:
                    event_state = 'event_drought'
                elif eis_score < -0.4:
                    event_state = 'low_event_activity'
                else:
                    event_state = 'normal_event_environment'
                    
                return {
                    'event_impact_score': eis_score,
                    'eis_volatility': eis_volatility,
                    'event_state': event_state,
                    'analysis_breakdown': {
                        'volatility_events': volatility_events,
                        'volume_events': volume_events,
                        'correlation_events': correlation_events,
                        'rotation_events': rotation_events,
                        'cross_asset_events': cross_asset_events
                    },
                    'timestamp': datetime.utcnow(),
                    'confidence': min(1.0, 1 - (eis_volatility / 2))
                }
            else:
                return {'error': 'Unable to calculate event impact score'}
                
        except Exception as e:
            logger.error(f"Error getting event pulse: {e}")
            return {'error': str(e)}
            
    def _extract_event_score(self, analysis_result: Dict) -> Optional[float]:
        """Extract numeric event score from analysis result"""
        try:
            if 'event_assessment' in analysis_result:
                assessment = analysis_result['event_assessment']
                stress_level = assessment.get('market_stress_level', 'normal')
                if stress_level == 'extreme_stress':
                    return 0.9
                elif stress_level == 'high_stress':
                    return 0.7
                elif stress_level == 'moderate_stress':
                    return 0.4
                else:
                    return 0.1
            elif 'assessment' in analysis_result:
                assessment = analysis_result['assessment']
                # Volume events
                if 'event_frequency' in assessment:
                    frequency = assessment['event_frequency']
                    if frequency == 'high':
                        return 0.6
                    elif frequency == 'moderate':
                        return 0.3
                    else:
                        return 0.1
                # Correlation events
                elif 'breakdown_frequency' in assessment:
                    frequency = assessment['breakdown_frequency']
                    if frequency == 'high':
                        return 0.8
                    elif frequency == 'moderate':
                        return 0.5
                    else:
                        return 0.2
                # Rotation events
                elif 'rotation_frequency' in assessment:
                    frequency = assessment['rotation_frequency']
                    if frequency == 'high':
                        return 0.7
                    elif frequency == 'moderate':
                        return 0.4
                    else:
                        return 0.1
                # Cross-asset events
                elif 'coordination_frequency' in assessment:
                    frequency = assessment['coordination_frequency']
                    if frequency == 'high':
                        return 0.6
                    elif frequency == 'moderate':
                        return 0.3
                    else:
                        return 0.1
            else:
                return None
                
        except Exception:
            return None
            
    async def store_event_data(self, event_data: Dict):
        """Store event metrics in time-series database"""
        try:
            if self.db_manager and 'timestamp' in event_data:
                # Store Event Impact Score
                await self.db_manager.influxdb_client.write_points(
                    database='market_pulse',
                    measurement='event_metrics',
                    tags={
                        'engine': 'event_shockwave',
                        'state': event_data.get('event_state', 'unknown')
                    },
                    fields={
                        'eis_score': float(event_data.get('event_impact_score', 0)),
                        'eis_volatility': float(event_data.get('eis_volatility', 0)),
                        'confidence': float(event_data.get('confidence', 0))
                    },
                    time=event_data['timestamp']
                )
                
                # Store component scores
                for component_name, analysis in event_data.get('analysis_breakdown', {}).items():
                    if isinstance(analysis, dict):
                        score = self._extract_event_score(analysis)
                        if score is not None:
                            await self.db_manager.influxdb_client.write_points(
                                database='market_pulse',
                                measurement='event_components',
                                tags={
                                    'component': component_name,
                                    'engine': 'event_shockwave'
                                },
                                fields={'component_score': float(score)},
                                time=event_data['timestamp']
                            )
                            
            logger.debug("Event data stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing event data: {e}")
            
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
                'cache_size': len(self.event_cache),
                'models_loaded': len(self.models),
                'event_categories': len(self.event_categories),
                'shockwave_phases': len(self.shockwave_phases),
                'health_score': self._calculate_engine_health()
            }
            
        except Exception as e:
            logger.error(f"Error getting event engine status: {e}")
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
                cache_freshness = max(0, 1 - (minutes_since_update / 15))  # Events require frequent updates
                health_factors.append(cache_freshness)
            
            # Model availability
            health_factors.append(min(1.0, len(self.models) / 2))
            
            return np.mean(health_factors) if health_factors else 0.0
            
        except Exception:
            return 0.0