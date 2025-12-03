"""
Capital Rotation Engine - Asset Allocation & Capital Flow Analysis
Real-time tracking of capital movements between asset classes, sectors, and investment styles
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class CapitalRotationEngine:
    """Capital Flow Monitor - Tracking allocation changes and rotation patterns"""
    
    def __init__(self):
        self.name = "Capital Rotation Engine"
        self.version = "1.0.0"
        self.status = "initialized"
        self.last_update = None
        self.rotation_cache = {}
        self.models = {}
        self.scaler = StandardScaler()
        
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
            logger.info("Capital Rotation Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Capital Rotation Engine: {e}")
            self.status = "error"
            
    def _initialize_models(self):
        """Initialize ML models for rotation prediction"""
        try:
            self.models['rotation_predictor'] = RandomForestRegressor(
                n_estimators=50,
                random_state=42,
                max_depth=8
            )
            logger.info("Capital rotation models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize rotation models: {e}")
            
    async def get_capital_pulse(self) -> Dict:
        """Get capital rotation analysis"""
        try:
            # Asset class rotation analysis
            asset_rotation = await self._analyze_asset_rotation()
            
            # Sector rotation analysis  
            sector_rotation = await self._analyze_sector_rotation()
            
            # Style rotation analysis
            style_rotation = await self._analyze_style_rotation()
            
            # Calculate overall Capital Rotation Score
            rotation_score = self._calculate_rotation_score(asset_rotation, sector_rotation, style_rotation)
            
            return {
                'capital_rotation_score': rotation_score,
                'asset_rotation': asset_rotation,
                'sector_rotation': sector_rotation,
                'style_rotation': style_rotation,
                'rotation_state': self._classify_rotation_state(rotation_score),
                'timestamp': datetime.utcnow(),
                'confidence': 0.7
            }
            
        except Exception as e:
            logger.error(f"Error getting capital pulse: {e}")
            return {'error': str(e)}
            
    async def _analyze_asset_rotation(self) -> Dict:
        """Analyze rotation between asset classes"""
        try:
            # Major asset classes
            asset_classes = {
                'equities': ['SPY', 'QQQ', 'IWM'],
                'bonds': ['TLT', 'IEF', 'SHY'],
                'commodities': ['GLD', 'SLV', 'DBA'],
                'alternatives': ['VXX', 'BTC-USD']
            }
            
            rotation_data = {}
            for asset_class, symbols in asset_classes.items():
                class_performance = []
                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="3mo")
                        if not hist.empty:
                            perf = hist['Close'].pct_change(20).iloc[-1]
                            class_performance.append(perf)
                    except:
                        continue
                        
                if class_performance:
                    rotation_data[asset_class] = {
                        'avg_performance': float(np.mean(class_performance)),
                        'performance_spread': float(np.std(class_performance))
                    }
                    
            return {
                'asset_class_performance': rotation_data,
                'rotation_leaders': self._identify_rotation_leaders(rotation_data),
                'rotation_intensity': self._calculate_rotation_intensity(rotation_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing asset rotation: {e}")
            return {'error': str(e)}
            
    async def _analyze_sector_rotation(self) -> Dict:
        """Analyze sector rotation patterns"""
        try:
            sectors = {
                'XLK': 'Technology',
                'XLF': 'Financial',
                'XLE': 'Energy',
                'XLI': 'Industrial',
                'XLV': 'Healthcare',
                'XLY': 'Consumer',
                'XLU': 'Utilities'
            }
            
            sector_performance = {}
            for etf, sector_name in sectors.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period="2mo")
                    if not hist.empty:
                        perf_1m = hist['Close'].pct_change(20).iloc[-1]
                        perf_2m = hist['Close'].pct_change(40).iloc[-1]
                        
                        sector_performance[sector_name] = {
                            'performance_1m': float(perf_1m),
                            'performance_2m': float(perf_2m),
                            'momentum_acceleration': float(perf_1m - perf_2m)
                        }
                except:
                    continue
                    
            return {
                'sector_performance': sector_performance,
                'rotation_winners': self._identify_rotation_winners(sector_performance),
                'sector_momentum': self._analyze_sector_momentum(sector_performance)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sector rotation: {e}")
            return {'error': str(e)}
            
    async def _analyze_style_rotation(self) -> Dict:
        """Analyze style rotation (growth vs value, etc.)"""
        try:
            styles = {
                'growth': ['QQQ', 'VUG'],
                'value': ['VTV', 'IWD'],
                'quality': ['QUAL', 'SPHQ'],
                'momentum': ['MTUM', 'SPMO']
            }
            
            style_performance = {}
            for style, symbols in styles.items():
                style_returns = []
                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="2mo")
                        if not hist.empty:
                            perf = hist['Close'].pct_change(20).iloc[-1]
                            style_returns.append(perf)
                    except:
                        continue
                        
                if style_returns:
                    style_performance[style] = {
                        'avg_return': float(np.mean(style_returns)),
                        'return_consistency': float(1 - np.std(style_returns))
                    }
                    
            return {
                'style_performance': style_performance,
                'style_leaders': self._identify_style_leaders(style_performance),
                'style_momentum': self._calculate_style_momentum(style_performance)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing style rotation: {e}")
            return {'error': str(e)}
            
    def _identify_rotation_leaders(self, rotation_data: Dict) -> List[str]:
        """Identify leading asset classes"""
        if not rotation_data:
            return []
            
        sorted_performance = sorted(
            rotation_data.items(),
            key=lambda x: x[1]['avg_performance'],
            reverse=True
        )
        
        return [item[0] for item in sorted_performance[:2]]
        
    def _identify_rotation_winners(self, sector_performance: Dict) -> List[str]:
        """Identify sector rotation winners"""
        if not sector_performance:
            return []
            
        # Sort by momentum acceleration
        sorted_momentum = sorted(
            sector_performance.items(),
            key=lambda x: x[1]['momentum_acceleration'],
            reverse=True
        )
        
        return [item[0] for item in sorted_momentum[:3]]
        
    def _identify_style_leaders(self, style_performance: Dict) -> List[str]:
        """Identify style rotation leaders"""
        if not style_performance:
            return []
            
        sorted_styles = sorted(
            style_performance.items(),
            key=lambda x: x[1]['avg_return'],
            reverse=True
        )
        
        return [item[0] for item in sorted_styles[:2]]
        
    def _calculate_rotation_intensity(self, rotation_data: Dict) -> float:
        """Calculate rotation intensity"""
        if not rotation_data:
            return 0.0
            
        performances = [data['avg_performance'] for data in rotation_data.values()]
        return float(np.std(performances))
        
    def _analyze_sector_momentum(self, sector_performance: Dict) -> Dict:
        """Analyze sector momentum patterns"""
        if not sector_performance:
            return {'momentum_state': 'neutral'}
            
        accelerations = [data['momentum_acceleration'] for data in sector_performance.values()]
        avg_acceleration = np.mean(accelerations)
        
        return {
            'momentum_state': 'accelerating' if avg_acceleration > 0.01 else 'decelerating' if avg_acceleration < -0.01 else 'stable',
            'momentum_strength': float(abs(avg_acceleration))
        }
        
    def _calculate_style_momentum(self, style_performance: Dict) -> Dict:
        """Calculate style momentum"""
        if not style_performance:
            return {'style_state': 'neutral'}
            
        returns = [data['avg_return'] for data in style_performance.values()]
        avg_return = np.mean(returns)
        
        return {
            'style_state': 'growth_leading' if returns and max(style_performance.items(), key=lambda x: x[1]['avg_return'])[0] == 'growth' else 'value_leading',
            'style_momentum': float(avg_return)
        }
        
    def _calculate_rotation_score(self, asset_rotation: Dict, sector_rotation: Dict, style_rotation: Dict) -> float:
        """Calculate overall capital rotation score"""
        components = []
        
        # Asset rotation component
        if 'rotation_intensity' in asset_rotation:
            components.append(asset_rotation['rotation_intensity'])
            
        # Sector momentum component
        if 'sector_momentum' in sector_rotation:
            momentum_strength = sector_rotation['sector_momentum'].get('momentum_strength', 0)
            components.append(momentum_strength)
            
        # Style momentum component
        if 'style_momentum' in style_rotation:
            style_momentum = style_rotation['style_momentum'].get('style_momentum', 0)
            components.append(abs(style_momentum))
            
        return float(np.mean(components) if components else 0.0)
        
    def _classify_rotation_state(self, rotation_score: float) -> str:
        """Classify current rotation state"""
        if rotation_score > 0.05:
            return 'active_rotation'
        elif rotation_score > 0.02:
            return 'moderate_rotation'
        elif rotation_score < 0.01:
            return 'rotation_pause'
        else:
            return 'stable_allocation'
            
    async def store_rotation_data(self, rotation_data: Dict):
        """Store rotation metrics in database"""
        try:
            if self.db_manager and 'timestamp' in rotation_data:
                await self.db_manager.influxdb_client.write_points(
                    database='market_pulse',
                    measurement='rotation_metrics',
                    tags={'engine': 'capital_rotation'},
                    fields={
                        'rotation_score': float(rotation_data.get('capital_rotation_score', 0)),
                        'confidence': float(rotation_data.get('confidence', 0))
                    },
                    time=rotation_data['timestamp']
                )
        except Exception as e:
            logger.error(f"Error storing rotation data: {e}")
            
    async def get_status(self) -> Dict:
        """Get engine status"""
        try:
            return {
                'name': self.name,
                'version': self.version,
                'status': self.status,
                'cache_size': len(self.rotation_cache),
                'models_loaded': len(self.models)
            }
        except Exception as e:
            logger.error(f"Error getting rotation engine status: {e}")
            return {'name': self.name, 'status': 'error'}