"""
Macro Pulse Engine - Macroeconomic Environment Analysis
Real-time tracking of macroeconomic indicators, policy changes, and economic sentiment
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import talib
from database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class MacroPulseEngine:
    """Macroeconomic Environment Monitor - Tracking economic policy and sentiment"""
    
    def __init__(self):
        self.name = "Macro Pulse Engine"
        self.version = "1.0.0"
        self.status = "initialized"
        self.last_update = None
        self.macro_cache = {}
        self.models = {}
        self.scaler = StandardScaler()
        
        # Macro tracking models
        self.macro_models = {
            "monetary_policy": self._monetary_policy_analysis,
            "fiscal_policy": self._fiscal_policy_analysis,
            "economic_indicators": self._economic_indicators_analysis,
            "policy_impact": self._policy_impact_analysis,
            "economic_sentiment": self._economic_sentiment_analysis
        }
        
        # Key macroeconomic indicators
        self.macro_indicators = {
            "interest_rates": {
                "fed_funds_rate": "Federal Funds Rate",
                "10y_treasury": "10-Year Treasury Yield",
                "2y_treasury": "2-Year Treasury Yield",
                "real_rates": "Real Interest Rates"
            },
            "inflation_measures": {
                "cpi": "Consumer Price Index",
                "ppi": "Producer Price Index", 
                "core_cpi": "Core CPI",
                "pce": "Personal Consumption Expenditures"
            },
            "employment": {
                "unemployment_rate": "Unemployment Rate",
                "jobless_claims": "Initial Jobless Claims",
                "payroll_growth": "Non-Farm Payrolls",
                "wage_growth": "Average Hourly Earnings"
            },
            "growth_indicators": {
                "gdp_growth": "GDP Growth Rate",
                "industrial_production": "Industrial Production",
                "retail_sales": "Retail Sales",
                "consumer_confidence": "Consumer Confidence"
            },
            "housing": {
                "housing_starts": "Housing Starts",
                "home_sales": "Existing Home Sales",
                "mortgage_rates": "Mortgage Rates",
                "home_price_index": "Case-Shiller Home Price Index"
            }
        }
        
        # Economic sentiment indicators
        self.sentiment_indicators = {
            "consumer_survey": "Consumer sentiment surveys",
            "business_survey": "Business confidence surveys",
            "manufacturing_survey": "Manufacturing PMI",
            "services_survey": "Services PMI"
        }
        
        # Policy analysis frameworks
        self.policy_frameworks = {
            "accommodative": "Ultra-low rates, QE, forward guidance",
            "neutral": "Balanced monetary policy stance",
            "restrictive": "Rate hikes, QT, hawkish stance",
            "data_dependent": "Policy reacts to incoming data"
        }
        
        # Economic regimes
        self.economic_regimes = {
            "expansion": "Strong growth, low unemployment, rising inflation",
            "recovery": "Growth acceleration, improving labor market",
            "peak_cycle": "Mature expansion, peak growth rates",
            "recession": "Negative growth, rising unemployment",
            "disinflation": "Falling inflation, stable growth",
            "stagflation": "High inflation, stagnant growth"
        }
        
        # Economic correlations
        self.economic_correlations = {
            "rates_bonds": "Interest rates vs bond prices",
            "rates_equities": "Interest rates vs equity valuations",
            "growth_equities": "Economic growth vs equity returns",
            "inflation_equities": "Inflation vs equity performance"
        }
        
        # Threshold levels for macro indicators
        self.macro_thresholds = {
            "cpi_acceleration": 0.3,      # 0.3% CPI acceleration
            "growth_slowdown": -1.0,      # 1% GDP growth slowdown
            "unemployment_surge": 2.0,    # 2% unemployment increase
            "yield_curve_steepening": 1.0, # 100bp yield curve steepening
            "sentiment_extreme": 0.8      # 80% extreme sentiment
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
            logger.info("Macro Pulse Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Macro Pulse Engine: {e}")
            self.status = "error"
            
    def _initialize_models(self):
        """Initialize ML models for macro prediction"""
        try:
            # Random Forest for economic regime prediction
            self.models['regime_classifier'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Random Forest for policy impact prediction
            self.models['policy_impact_predictor'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=8
            )
            
            # K-means for economic clustering
            self.models['economic_clusters'] = KMeans(
                n_clusters=3,
                random_state=42
            )
            
            # Scaler for macro feature normalization
            self.scaler = StandardScaler()
            
            logger.info("Macro prediction models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize macro models: {e}")
            
    async def _fetch_macro_data(self, symbols: List[str], period: str = "1y") -> Dict:
        """Fetch macro-related market data"""
        try:
            macro_data = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    if not hist.empty:
                        macro_data[symbol] = hist
                except Exception as e:
                    logger.warning(f"Error fetching macro data for {symbol}: {e}")
                    continue
                    
            return macro_data
            
        except Exception as e:
            logger.error(f"Error fetching macro data: {e}")
            return {}
            
    async def _monetary_policy_analysis(self) -> Dict:
        """Analyze current monetary policy stance and outlook"""
        try:
            # Get key rate-sensitive assets
            policy_symbols = ["TLT", "IEF", "SHY", "VIX", "TLT", "SPY"]
            policy_data = await self._fetch_macro_data(policy_symbols)
            
            monetary_policy = {}
            
            # Analyze yield curve dynamics
            if "TLT" in policy_data and "SHY" in policy_data:
                long_rates = policy_data["TLT"]["Close"]
                short_rates = policy_data["SHY"]["Close"]
                
                # Yield curve proxy (Long-term vs Short-term relationship)
                yield_curve_proxy = (long_rates / long_rates.rolling(20).mean()) - (short_rates / short_rates.rolling(20).mean())
                current_curve_slope = yield_curve_proxy.iloc[-1]
                
                monetary_policy['yield_curve'] = {
                    'current_slope': float(current_curve_slope),
                    'slope_trend': float(yield_curve_proxy.diff(5).iloc[-1]),
                    'curve_steepening': current_curve_slope > 0.1,
                    'curve_flattening': current_curve_slope < -0.1
                }
                
            # Interest rate expectations (via bond price momentum)
            if "TLT" in policy_data:
                tlt_data = policy_data["TLT"]["Close"]
                rate_momentum_1m = tlt_data.pct_change(20).iloc[-1]  # 1-month rate momentum
                rate_momentum_3m = tlt_data.pct_change(60).iloc[-1]  # 3-month rate momentum
                
                monetary_policy['rate_expectations'] = {
                    'short_term_momentum': float(rate_momentum_1m),
                    'medium_term_momentum': float(rate_momentum_3m),
                    'rate_cut_probability_proxy': float(max(0, -rate_momentum_1m)),  # Inverse of price change
                    'rate_hike_probability_proxy': float(max(0, rate_momentum_1m))
                }
                
            # Central bank policy proxy (via VIX and equity volatility)
            if "VIX" in policy_data:
                vix_data = policy_data["VIX"]["Close"]
                vix_current = vix_data.iloc[-1]
                vix_trend = vix_data.pct_change(10).iloc[-1]
                
                # High VIX with falling trend might indicate policy accommodation
                # High VIX with rising trend might indicate policy concern
                policy_proxy = 'accommodative' if (vix_current > 20 and vix_trend < 0) else \
                              'restrictive' if (vix_current > 25 and vix_trend > 0) else 'neutral'
                
                monetary_policy['policy_proxy'] = {
                    'current_regime': policy_proxy,
                    'vix_level': float(vix_current),
                    'vix_trend': float(vix_trend),
                    'policy_stress': 'high' if vix_current > 30 else 'moderate' if vix_current > 20 else 'low'
                }
                
            # Fed policy stance analysis
            policy_stance = self._analyze_fed_policy_stance(monetary_policy)
            
            # Policy transmission analysis
            transmission = self._analyze_policy_transmission(monetary_policy)
            
            # Policy outlook assessment
            outlook = self._assess_policy_outlook(monetary_policy)
            
            return {
                'monetary_policy': monetary_policy,
                'policy_stance': policy_stance,
                'transmission': transmission,
                'policy_outlook': outlook
            }
            
        except Exception as e:
            logger.error(f"Error in monetary policy analysis: {e}")
            return {'error': str(e)}
            
    def _analyze_fed_policy_stance(self, policy_data: Dict) -> Dict:
        """Analyze current Fed policy stance"""
        try:
            stance_analysis = {
                'current_stance': 'neutral',
                'policy_bias': 'balanced',
                'next_move_likelihood': 'data_dependent',
                'policy_consensus': 'mixed'
            }
            
            # Analyze yield curve signals
            if 'yield_curve' in policy_data:
                curve_slope = policy_data['yield_curve']['current_slope']
                
                if curve_slope > 0.2:
                    stance_analysis['current_stance'] = 'accommodative'
                elif curve_slope < -0.2:
                    stance_analysis['current_stance'] = 'restrictive'
                else:
                    stance_analysis['current_stance'] = 'neutral'
                    
            # Analyze rate expectations
            if 'rate_expectations' in policy_data:
                rate_momentum = policy_data['rate_expectations']['short_term_momentum']
                
                if rate_momentum < -0.02:  # Bond prices rising (rates falling)
                    stance_analysis['policy_bias'] = 'dovish'
                    stance_analysis['next_move_likelihood'] = 'rate_cut'
                elif rate_momentum > 0.02:  # Bond prices falling (rates rising)
                    stance_analysis['policy_bias'] = 'hawkish'
                    stance_analysis['next_move_likelihood'] = 'rate_hike'
                else:
                    stance_analysis['policy_bias'] = 'balanced'
                    stance_analysis['next_move_likelihood'] = 'hold'
                    
            # Assess policy consensus
            if 'policy_proxy' in policy_data:
                policy_stress = policy_data['policy_proxy'].get('policy_stress', 'moderate')
                
                if policy_stress == 'high':
                    stance_analysis['policy_consensus'] = 'stressed'
                elif policy_stress == 'low':
                    stance_analysis['policy_consensus'] = 'aligned'
                else:
                    stance_analysis['policy_consensus'] = 'mixed'
                    
            return stance_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Fed policy stance: {e}")
            return {'current_stance': 'neutral', 'policy_bias': 'balanced'}
            
    def _analyze_policy_transmission(self, policy_data: Dict) -> Dict:
        """Analyze how policy changes are transmitted through markets"""
        try:
            transmission = {
                'transmission_strength': 'normal',
                'market_impact': 'balanced',
                'transmission_channels': []
            }
            
            # Analyze transmission through different channels
            if 'rate_expectations' in policy_data:
                rate_impact = policy_data['rate_expectations']
                
                # Strong rate expectations suggest clear transmission
                if abs(rate_impact.get('short_term_momentum', 0)) > 0.03:
                    transmission['transmission_strength'] = 'strong'
                    transmission['transmission_channels'].append('interest_rate_channel')
                    
            if 'policy_proxy' in policy_data:
                policy_stress = policy_data['policy_proxy'].get('policy_stress', 'moderate')
                
                if policy_stress == 'high':
                    transmission['market_impact'] = 'volatile'
                    transmission['transmission_channels'].append('volatility_channel')
                elif policy_stress == 'low':
                    transmission['market_impact'] = 'stable'
                    transmission['transmission_channels'].append('confidence_channel')
                    
            return transmission
            
        except Exception as e:
            logger.error(f"Error analyzing policy transmission: {e}")
            return {'transmission_strength': 'normal', 'market_impact': 'balanced'}
            
    def _assess_policy_outlook(self, policy_data: Dict) -> Dict:
        """Assess monetary policy outlook"""
        try:
            outlook = {
                'outlook_bias': 'neutral',
                'policy_change_probability': 0.0,
                'time_horizon': 'near_term',
                'key_risks': []
            }
            
            # Assess based on multiple signals
            change_signals = 0
            total_signals = 0
            
            # Yield curve signals
            if 'yield_curve' in policy_data:
                curve_trend = policy_data['yield_curve'].get('slope_trend', 0)
                if abs(curve_trend) > 0.05:
                    change_signals += 1
                total_signals += 1
                
            # Rate expectation signals
            if 'rate_expectations' in policy_data:
                rate_momentum = abs(policy_data['rate_expectations'].get('short_term_momentum', 0))
                if rate_momentum > 0.02:
                    change_signals += 1
                total_signals += 1
                
            # Calculate change probability
            if total_signals > 0:
                outlook['policy_change_probability'] = float(change_signals / total_signals)
                
                if outlook['policy_change_probability'] > 0.7:
                    outlook['outlook_bias'] = 'high_change_probability'
                elif outlook['policy_change_probability'] > 0.3:
                    outlook['outlook_bias'] = 'moderate_change_probability'
                else:
                    outlook['outlook_bias'] = 'low_change_probability'
                    
            return outlook
            
        except Exception as e:
            logger.error(f"Error assessing policy outlook: {e}")
            return {'outlook_bias': 'neutral', 'policy_change_probability': 0.0}
            
    async def _fiscal_policy_analysis(self) -> Dict:
        """Analyze fiscal policy environment and market impact"""
        try:
            # Get fiscal policy sensitive assets
            fiscal_symbols = ["TLT", "VTI", "XLF", "XLU", "XLV"]
            fiscal_data = await self._fetch_macro_data(fiscal_symbols)
            
            fiscal_policy = {}
            
            # Treasury market analysis (fiscal impact proxy)
            if "TLT" in fiscal_data:
                bond_data = fiscal_data["TLT"]["Close"]
                bond_volatility = bond_data.pct_change().rolling(20).std().iloc[-1]
                bond_momentum = bond_data.pct_change(10).iloc[-1]
                
                fiscal_policy['bond_market'] = {
                    'volatility_level': float(bond_volatility),
                    'price_momentum': float(bond_momentum),
                    'fiscal_stress_proxy': 'elevated' if bond_volatility > 0.015 else 'normal'
                }
                
            # Growth-sensitive assets (fiscal support proxy)
            growth_assets = ["VTI", "XLF"]  # Broad market + financials
            growth_performance = {}
            
            for asset in growth_assets:
                if asset in fiscal_data:
                    performance = fiscal_data[asset]["Close"].pct_change(20).iloc[-1]
                    growth_performance[asset] = float(performance)
                    
            if growth_performance:
                avg_growth_performance = np.mean(list(growth_performance.values()))
                fiscal_policy['growth_sensitivity'] = {
                    'average_performance': float(avg_growth_performance),
                    'fiscal_optimism_proxy': 'positive' if avg_growth_performance > 0.02 else 'negative' if avg_growth_performance < -0.02 else 'neutral'
                }
                
            # Defensive vs cyclical rotation (fiscal priorities proxy)
            if "XLU" in fiscal_data and "XLF" in fiscal_data:
                defensive_performance = fiscal_data["XLU"]["Close"].pct_change(20).iloc[-1]
                cyclical_performance = fiscal_data["XLF"]["Close"].pct_change(20).iloc[-1]
                
                rotation_signal = cyclical_performance - defensive_performance
                
                fiscal_policy['sector_rotation'] = {
                    'defensive_momentum': float(defensive_performance),
                    'cyclical_momentum': float(cyclical_performance),
                    'rotation_signal': float(rotation_signal),
                    'fiscal_priority': 'infrastructure' if rotation_signal > 0.01 else 'social_spending' if rotation_signal < -0.01 else 'balanced'
                }
                
            # Fiscal policy impact assessment
            impact_assessment = self._assess_fiscal_impact(fiscal_policy)
            
            # Market expectations analysis
            expectations = self._analyze_fiscal_expectations(fiscal_policy)
            
            return {
                'fiscal_policy': fiscal_policy,
                'impact_assessment': impact_assessment,
                'market_expectations': expectations
            }
            
        except Exception as e:
            logger.error(f"Error in fiscal policy analysis: {e}")
            return {'error': str(e)}
            
    def _assess_fiscal_impact(self, fiscal_data: Dict) -> Dict:
        """Assess fiscal policy impact on markets"""
        try:
            impact = {
                'overall_impact': 'neutral',
                'impact_strength': 'moderate',
                'primary_channels': [],
                'secondary_effects': []
            }
            
            # Bond market impact
            if 'bond_market' in fiscal_data:
                bond_stress = fiscal_data['bond_market'].get('fiscal_stress_proxy', 'normal')
                
                if bond_stress == 'elevated':
                    impact['primary_channels'].append('debt_concerns')
                    impact['impact_strength'] = 'elevated'
                    
            # Growth impact
            if 'growth_sensitivity' in fiscal_data:
                optimism = fiscal_data['growth_sensitivity'].get('fiscal_optimism_proxy', 'neutral')
                
                if optimism == 'positive':
                    impact['secondary_effects'].append('growth_optimism')
                elif optimism == 'negative':
                    impact['secondary_effects'].append('growth_concerns')
                    
            # Overall impact assessment
            if len(impact['primary_channels']) > 0:
                impact['overall_impact'] = 'significant'
            elif len(impact['secondary_effects']) > 0:
                impact['overall_impact'] = 'moderate'
            else:
                impact['overall_impact'] = 'minimal'
                
            return impact
            
        except Exception as e:
            logger.error(f"Error assessing fiscal impact: {e}")
            return {'overall_impact': 'neutral', 'impact_strength': 'moderate'}
            
    def _analyze_fiscal_expectations(self, fiscal_data: Dict) -> Dict:
        """Analyze market expectations for fiscal policy"""
        try:
            expectations = {
                'policy_expectations': 'stable',
                'expectation_shift': 'minimal',
                'confidence_level': 'moderate'
            }
            
            # Based on bond volatility (policy uncertainty proxy)
            if 'bond_market' in fiscal_data:
                bond_vol = fiscal_data['bond_market'].get('volatility_level', 0.01)
                
                if bond_vol > 0.02:  # High bond volatility = high policy uncertainty
                    expectations['policy_expectations'] = 'uncertain'
                    expectations['confidence_level'] = 'low'
                elif bond_vol < 0.01:  # Low bond volatility = stable expectations
                    expectations['policy_expectations'] = 'stable'
                    expectations['confidence_level'] = 'high'
                    
            # Growth expectations
            if 'growth_sensitivity' in fiscal_data:
                momentum = fiscal_data['growth_sensitivity'].get('average_performance', 0)
                
                if momentum > 0.03:  # Strong growth performance
                    expectations['expectation_shift'] = 'expansionary_bias'
                elif momentum < -0.03:  # Weak growth performance
                    expectations['expectation_shift'] = 'austerity_bias'
                    
            return expectations
            
        except Exception as e:
            logger.error(f"Error analyzing fiscal expectations: {e}")
            return {'policy_expectations': 'stable', 'expectation_shift': 'minimal'}
            
    async def _economic_indicators_analysis(self) -> Dict:
        """Analyze key economic indicators and trends"""
        try:
            # Get economic indicator proxies
            indicator_symbols = ["SPY", "TLT", "GLD", "VIX", "XLU"]
            indicator_data = await self._fetch_macro_data(indicator_symbols)
            
            economic_indicators = {}
            
            # Growth indicator (SPY performance)
            if "SPY" in indicator_data:
                spy_data = indicator_data["SPY"]["Close"]
                growth_indicators = {
                    'gdp_proxy_1m': float(spy_data.pct_change(20).iloc[-1]),
                    'gdp_proxy_3m': float(spy_data.pct_change(60).iloc[-1]),
                    'growth_trend': 'accelerating' if spy_data.pct_change(10).iloc[-1] > spy_data.pct_change(20).iloc[-1] else 'decelerating'
                }
                economic_indicators['growth'] = growth_indicators
                
            # Inflation indicator (TIPS/Gold ratio)
            if "TLT" in indicator_data and "GLD" in indicator_data:
                bond_data = indicator_data["TLT"]["Close"]
                gold_data = indicator_data["GLD"]["Close"]
                
                # Real rates proxy (Gold vs Bond performance)
                bond_performance = bond_data.pct_change(20).iloc[-1]
                gold_performance = gold_data.pct_change(20).iloc[-1]
                real_rates_proxy = bond_performance - gold_performance
                
                inflation_indicators = {
                    'real_rates_proxy': float(real_rates_proxy),
                    'inflation_expectation': 'rising' if real_rates_proxy < -0.01 else 'falling' if real_rates_proxy > 0.01 else 'stable',
                    'inflation_proxy_trend': float(gold_data.pct_change(10).iloc[-1] - bond_data.pct_change(10).iloc[-1])
                }
                economic_indicators['inflation'] = inflation_indicators
                
            # Employment proxy (Defensive sector performance)
            if "XLU" in indicator_data:
                xlu_data = indicator_data["XLU"]["Close"]
                xlu_momentum = xlu_data.pct_change(20).iloc[-1]
                
                employment_indicators = {
                    'employment_proxy': float(xlu_momentum),
                    'employment_sentiment': 'strong' if xlu_momentum > 0.02 else 'weak' if xlu_momentum < -0.02 else 'stable'
                }
                economic_indicators['employment'] = employment_indicators
                
            # Market sentiment (VIX levels)
            if "VIX" in indicator_data:
                vix_data = indicator_data["VIX"]["Close"]
                vix_current = vix_data.iloc[-1]
                vix_trend = vix_data.pct_change(10).iloc[-1]
                
                sentiment_indicators = {
                    'vix_level': float(vix_current),
                    'sentiment_trend': 'improving' if vix_trend < -0.1 else 'deteriorating' if vix_trend > 0.1 else 'stable',
                    'economic_confidence': 'high' if vix_current < 15 else 'low' if vix_current > 25 else 'moderate'
                }
                economic_indicators['sentiment'] = sentiment_indicators
                
            # Economic momentum assessment
            momentum_assessment = self._assess_economic_momentum(economic_indicators)
            
            # Economic health score
            health_score = self._calculate_economic_health_score(economic_indicators)
            
            return {
                'economic_indicators': economic_indicators,
                'momentum_assessment': momentum_assessment,
                'economic_health_score': health_score
            }
            
        except Exception as e:
            logger.error(f"Error in economic indicators analysis: {e}")
            return {'error': str(e)}
            
    def _assess_economic_momentum(self, indicators: Dict) -> Dict:
        """Assess overall economic momentum"""
        try:
            momentum_assessment = {
                'overall_momentum': 'neutral',
                'momentum_strength': 0.0,
                'leading_indicators': [],
                'lagging_indicators': []
            }
            
            # Collect momentum signals
            momentum_signals = []
            
            if 'growth' in indicators:
                gdp_proxy = indicators['growth'].get('gdp_proxy_1m', 0)
                momentum_signals.append(('growth', gdp_proxy))
                
            if 'inflation' in indicators:
                inflation_trend = indicators['inflation'].get('inflation_proxy_trend', 0)
                # Inverted: falling real rates = inflationary pressure = negative momentum
                momentum_signals.append(('inflation', -inflation_trend))
                
            if 'employment' in indicators:
                employment_proxy = indicators['employment'].get('employment_proxy', 0)
                momentum_signals.append(('employment', employment_proxy))
                
            if 'sentiment' in indicators:
                # Convert VIX trend to confidence momentum
                sentiment_data = indicators['sentiment']
                if sentiment_data.get('sentiment_trend') == 'improving':
                    momentum_signals.append(('sentiment', 0.02))
                elif sentiment_data.get('sentiment_trend') == 'deteriorating':
                    momentum_signals.append(('sentiment', -0.02))
                    
            # Calculate overall momentum
            if momentum_signals:
                momentum_values = [signal[1] for signal in momentum_signals]
                overall_momentum = np.mean(momentum_values)
                
                momentum_assessment['overall_momentum'] = (
                    'strong_positive' if overall_momentum > 0.02 else
                    'positive' if overall_momentum > 0.005 else
                    'strong_negative' if overall_momentum < -0.02 else
                    'negative' if overall_momentum < -0.005 else
                    'neutral'
                )
                momentum_assessment['momentum_strength'] = float(abs(overall_momentum))
                
                # Identify leading/lagging indicators
                sorted_signals = sorted(momentum_signals, key=lambda x: x[1], reverse=True)
                momentum_assessment['leading_indicators'] = [signal[0] for signal in sorted_signals[:2]]
                momentum_assessment['lagging_indicators'] = [signal[0] for signal in sorted_signals[-2:]]
                
            return momentum_assessment
            
        except Exception as e:
            logger.error(f"Error assessing economic momentum: {e}")
            return {'overall_momentum': 'neutral', 'momentum_strength': 0.0}
            
    def _calculate_economic_health_score(self, indicators: Dict) -> Dict:
        """Calculate overall economic health score"""
        try:
            health_score = {
                'overall_health': 50.0,  # Base score
                'health_components': {},
                'risk_factors': [],
                'positive_factors': []
            }
            
            # Growth health
            if 'growth' in indicators:
                growth_momentum = indicators['growth'].get('gdp_proxy_1m', 0)
                if growth_momentum > 0.02:
                    health_score['positive_factors'].append('strong_growth_momentum')
                    health_score['health_components']['growth'] = 75
                elif growth_momentum > 0:
                    health_score['health_components']['growth'] = 60
                else:
                    health_score['risk_factors'].append('weak_growth_momentum')
                    health_score['health_components']['growth'] = 40
                    
            # Inflation health
            if 'inflation' in indicators:
                inflation_sentiment = indicators['inflation'].get('inflation_expectation', 'stable')
                if inflation_sentiment == 'stable':
                    health_score['positive_factors'].append('stable_inflation_expectations')
                    health_score['health_components']['inflation'] = 65
                else:
                    health_score['risk_factors'].append('inflation_expectations_volatility')
                    health_score['health_components']['inflation'] = 45
                    
            # Employment health
            if 'employment' in indicators:
                employment_sentiment = indicators['employment'].get('employment_sentiment', 'stable')
                if employment_sentiment == 'strong':
                    health_score['positive_factors'].append('strong_employment_sentiment')
                    health_score['health_components']['employment'] = 70
                elif employment_sentiment == 'weak':
                    health_score['risk_factors'].append('weak_employment_sentiment')
                    health_score['health_components']['employment'] = 40
                else:
                    health_score['health_components']['employment'] = 55
                    
            # Sentiment health
            if 'sentiment' in indicators:
                confidence = indicators['sentiment'].get('economic_confidence', 'moderate')
                if confidence == 'high':
                    health_score['positive_factors'].append('high_economic_confidence')
                    health_score['health_components']['sentiment'] = 75
                elif confidence == 'low':
                    health_score['risk_factors'].append('low_economic_confidence')
                    health_score['health_components']['sentiment'] = 35
                else:
                    health_score['health_components']['sentiment'] = 55
                    
            # Calculate overall health
            if health_score['health_components']:
                health_score['overall_health'] = float(np.mean(list(health_score['health_components'].values())))
                
            return health_score
            
        except Exception as e:
            logger.error(f"Error calculating economic health score: {e}")
            return {'overall_health': 50.0, 'health_components': {}}
            
    async def _policy_impact_analysis(self) -> Dict:
        """Analyze the impact of policy changes on different sectors"""
        try:
            # Get policy-sensitive sector data
            sector_symbols = ["XLF", "XLV", "XLU", "XLE", "XLK", "XLI"]
            policy_impact_data = await self._fetch_macro_data(sector_symbols)
            
            policy_impact = {}
            
            # Analyze policy sensitivity by sector
            policy_sensitive_sectors = {
                'XLF': 'Monetary Policy Sensitive',
                'XLV': 'Healthcare Policy Sensitive', 
                'XLU': 'Defensive/Interest Rate Sensitive',
                'XLE': 'Energy Policy Sensitive',
                'XLK': 'Technology/Trade Policy Sensitive',
                'XLI': 'Infrastructure Policy Sensitive'
            }
            
            for sector_etf, policy_type in policy_sensitive_sectors.items():
                if sector_etf in policy_impact_data:
                    sector_data = policy_impact_data[sector_etf]["Close"]
                    
                    # Calculate policy sensitivity
                    policy_sensitivity = self._calculate_policy_sensitivity(sector_data)
                    
                    policy_impact[sector_etf] = {
                        'sector_name': policy_type,
                        'policy_sensitivity': policy_sensitivity,
                        'recent_performance': float(sector_data.pct_change(20).iloc[-1]),
                        'policy_impact_score': self._calculate_policy_impact_score(policy_sensitivity)
                    }
                    
            # Overall policy impact assessment
            impact_assessment = self._assess_overall_policy_impact(policy_impact)
            
            # Policy transmission effectiveness
            transmission_effectiveness = self._assess_policy_transmission_effectiveness(policy_impact)
            
            return {
                'policy_impact': policy_impact,
                'impact_assessment': impact_assessment,
                'transmission_effectiveness': transmission_effectiveness
            }
            
        except Exception as e:
            logger.error(f"Error in policy impact analysis: {e}")
            return {'error': str(e)}
            
    def _calculate_policy_sensitivity(self, price_data: pd.Series) -> str:
        """Calculate policy sensitivity for a sector"""
        try:
            if len(price_data) < 30:
                return 'unknown'
                
            # Calculate volatility during policy-sensitive periods
            returns = price_data.pct_change().dropna()
            
            # High volatility periods (policy uncertainty)
            high_vol_periods = returns > returns.quantile(0.9)
            vol_spike_frequency = high_vol_periods.sum() / len(high_vol_periods)
            
            # Trend analysis
            recent_trend = price_data.pct_change(10).iloc[-1]
            historical_trend = price_data.pct_change(60).iloc[-1] if len(price_data) > 60 else recent_trend
            
            trend_divergence = abs(recent_trend - historical_trend)
            
            # Policy sensitivity classification
            if vol_spike_frequency > 0.15:  # 15% of periods show high volatility
                return 'highly_sensitive'
            elif vol_spike_frequency > 0.08:  # 8% of periods show high volatility
                return 'moderately_sensitive'
            elif trend_divergence > 0.05:  # 5% trend divergence
                return 'trend_sensitive'
            else:
                return 'low_sensitive'
                
        except Exception as e:
            logger.error(f"Error calculating policy sensitivity: {e}")
            return 'unknown'
            
    def _calculate_policy_impact_score(self, sensitivity: str) -> float:
        """Calculate policy impact score"""
        try:
            sensitivity_scores = {
                'highly_sensitive': 0.8,
                'moderately_sensitive': 0.6,
                'trend_sensitive': 0.4,
                'low_sensitive': 0.2,
                'unknown': 0.5
            }
            return sensitivity_scores.get(sensitivity, 0.5)
            
        except Exception as e:
            logger.error(f"Error calculating policy impact score: {e}")
            return 0.5
            
    def _assess_overall_policy_impact(self, impact_data: Dict) -> Dict:
        """Assess overall policy impact across sectors"""
        try:
            assessment = {
                'overall_impact': 'moderate',
                'impact_intensity': 'normal',
                'sectors_most_affected': [],
                'policy_effectiveness': 'mixed'
            }
            
            if not impact_data:
                return assessment
                
            # Calculate average impact score
            impact_scores = [data['policy_impact_score'] for data in impact_data.values()]
            avg_impact = np.mean(impact_scores)
            
            # Impact intensity
            if avg_impact > 0.7:
                assessment['impact_intensity'] = 'high'
                assessment['overall_impact'] = 'significant'
            elif avg_impact > 0.5:
                assessment['impact_intensity'] = 'moderate'
                assessment['overall_impact'] = 'moderate'
            else:
                assessment['impact_intensity'] = 'low'
                assessment['overall_impact'] = 'minimal'
                
            # Most affected sectors
            sorted_impacts = sorted(impact_data.items(), key=lambda x: x[1]['policy_impact_score'], reverse=True)
            assessment['sectors_most_affected'] = [item[0] for item in sorted_impacts[:3]]
            
            # Policy effectiveness
            affected_sectors = len([score for score in impact_scores if score > 0.6])
            if affected_sectors > len(impact_scores) * 0.7:
                assessment['policy_effectiveness'] = 'widespread'
            elif affected_sectors < len(impact_scores) * 0.3:
                assessment['policy_effectiveness'] = 'targeted'
            else:
                assessment['policy_effectiveness'] = 'mixed'
                
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing overall policy impact: {e}")
            return {'overall_impact': 'moderate', 'impact_intensity': 'normal'}
            
    def _assess_policy_transmission_effectiveness(self, impact_data: Dict) -> Dict:
        """Assess how effectively policy is being transmitted"""
        try:
            effectiveness = {
                'transmission_rate': 'normal',
                'sector_coordination': 'mixed',
                'timeliness': 'adequate',
                'coordination_strength': 0.0
            }
                
            if not impact_data:
                return effectiveness
                
            # Analyze transmission timing
            performance_spread = []
            for data in impact_data.values():
                recent_perf = data['recent_performance']
                sensitivity = data['policy_sensitivity']
                
                # Different sensitivities should show different impacts
                if sensitivity == 'highly_sensitive':
                    performance_spread.append(abs(recent_perf))
                    
            # Transmission rate assessment
            if performance_spread:
                avg_transmission = np.mean(performance_spread)
                
                if avg_transmission > 0.03:  # 3% average performance difference
                    effectiveness['transmission_rate'] = 'strong'
                elif avg_transmission > 0.015:  # 1.5% average performance difference
                    effectiveness['transmission_rate'] = 'moderate'
                else:
                    effectiveness['transmission_rate'] = 'weak'
                    
                # Coordination strength
                effectiveness['coordination_strength'] = float(min(1.0, avg_transmission * 20))
                
            # Sector coordination
            positive_performance = sum(1 for data in impact_data.values() if data['recent_performance'] > 0)
            total_sectors = len(impact_data)
            
            if positive_performance > total_sectors * 0.7:
                effectiveness['sector_coordination'] = 'coordinated'
            elif positive_performance < total_sectors * 0.3:
                effectiveness['sector_coordination'] = 'divergent'
            else:
                effectiveness['sector_coordination'] = 'mixed'
                
            return effectiveness
            
        except Exception as e:
            logger.error(f"Error assessing policy transmission effectiveness: {e}")
            return {'transmission_rate': 'normal', 'coordination_strength': 0.0}
            
    async def _economic_sentiment_analysis(self) -> Dict:
        """Analyze economic sentiment indicators"""
        try:
            # Get sentiment-sensitive assets
            sentiment_symbols = ["SPY", "VIX", "GLD", "TLT"]
            sentiment_data = await self._fetch_macro_data(sentiment_symbols)
            
            economic_sentiment = {}
            
            # Market-based sentiment indicators
            if "SPY" in sentiment_data:
                spy_data = sentiment_data["SPY"]["Close"]
                spy_performance = spy_data.pct_change(20).iloc[-1]
                spy_momentum = spy_data.pct_change(5).iloc[-1]
                
                market_sentiment = {
                    'performance_sentiment': 'optimistic' if spy_performance > 0.03 else 'pessimistic' if spy_performance < -0.03 else 'neutral',
                    'momentum_sentiment': 'building' if spy_momentum > spy_performance else 'fading' if spy_momentum < spy_performance else 'stable',
                    'risk_appetite_proxy': float(max(0, spy_performance + 0.5))  # Positive performance = higher risk appetite
                }
                economic_sentiment['market_sentiment'] = market_sentiment
                
            # Fear/greed indicator (VIX levels)
            if "VIX" in sentiment_data:
                vix_data = sentiment_data["VIX"]["Close"]
                vix_current = vix_data.iloc[-1]
                vix_trend = vix_data.pct_change(10).iloc[-1]
                
                fear_greed_indicator = {
                    'fear_level': 'extreme_fear' if vix_current > 30 else 'high_fear' if vix_current > 20 else 'low_fear' if vix_current < 12 else 'normal_fear',
                    'greed_level': 'extreme_greed' if vix_current < 12 else 'high_greed' if vix_current < 15 else 'normal_greed' if vix_current < 20 else 'low_greed',
                    'sentiment_momentum': 'fear_growing' if vix_trend > 0.1 else 'greed_growing' if vix_trend < -0.1 else 'sentiment_stable'
                }
                economic_sentiment['fear_greed'] = fear_greed_indicator
                
            # Safe haven demand (Gold vs bond performance)
            if "GLD" in sentiment_data and "TLT" in sentiment_data:
                gold_performance = sentiment_data["GLD"]["Close"].pct_change(20).iloc[-1]
                bond_performance = sentiment_data["TLT"]["Close"].pct_change(20).iloc[-1]
                
                safe_haven_flows = {
                    'gold_demand': 'high' if gold_performance > 0.02 else 'low' if gold_performance < -0.02 else 'normal',
                    'bond_demand': 'high' if bond_performance > 0.01 else 'low' if bond_performance < -0.01 else 'normal',
                    'flight_to_quality': 'active' if (gold_performance > 0.01 and bond_performance > 0.005) else 'inactive'
                }
                economic_sentiment['safe_haven'] = safe_haven_flows
                
            # Overall sentiment assessment
            sentiment_assessment = self._assess_overall_economic_sentiment(economic_sentiment)
            
            # Sentiment momentum analysis
            momentum_analysis = self._analyze_sentiment_momentum(economic_sentiment)
            
            # Sentiment risk assessment
            risk_assessment = self._assess_sentiment_risks(economic_sentiment)
            
            return {
                'economic_sentiment': economic_sentiment,
                'sentiment_assessment': sentiment_assessment,
                'momentum_analysis': momentum_analysis,
                'risk_assessment': risk_assessment
            }
            
        except Exception as e:
            logger.error(f"Error in economic sentiment analysis: {e}")
            return {'error': str(e)}
            
    def _assess_overall_economic_sentiment(self, sentiment_data: Dict) -> Dict:
        """Assess overall economic sentiment"""
        try:
            assessment = {
                'overall_sentiment': 'neutral',
                'sentiment_strength': 0.0,
                'confidence_level': 'moderate',
                'sentiment_divergence': 0.0
            }
            
            # Collect sentiment signals
            sentiment_signals = []
            
            # Market sentiment signals
            if 'market_sentiment' in sentiment_data:
                market_data = sentiment_data['market_sentiment']
                
                if market_data.get('performance_sentiment') == 'optimistic':
                    sentiment_signals.append(0.6)
                elif market_data.get('performance_sentiment') == 'pessimistic':
                    sentiment_signals.append(-0.6)
                else:
                    sentiment_signals.append(0.0)
                    
            # Fear/greed signals
            if 'fear_greed' in sentiment_data:
                fear_greed_data = sentiment_data['fear_greed']
                fear_level = fear_greed_data.get('fear_level', 'normal_fear')
                
                if fear_level in ['extreme_fear', 'high_fear']:
                    sentiment_signals.append(-0.7)
                elif fear_level in ['low_fear']:
                    sentiment_signals.append(0.5)
                else:
                    sentiment_signals.append(0.0)
                    
            # Safe haven signals
            if 'safe_haven' in sentiment_data:
                safe_haven_data = sentiment_data['safe_haven']
                
                if safe_haven_data.get('flight_to_quality') == 'active':
                    sentiment_signals.append(-0.5)  # Flight to quality = negative sentiment
                else:
                    sentiment_signals.append(0.0)
                    
            # Calculate overall sentiment
            if sentiment_signals:
                overall_sentiment_score = np.mean(sentiment_signals)
                
                assessment['overall_sentiment'] = (
                    'extremely_positive' if overall_sentiment_score > 0.5 else
                    'positive' if overall_sentiment_score > 0.1 else
                    'extremely_negative' if overall_sentiment_score < -0.5 else
                    'negative' if overall_sentiment_score < -0.1 else
                    'neutral'
                )
                assessment['sentiment_strength'] = float(abs(overall_sentiment_score))
                
                # Confidence level
                sentiment_std = np.std(sentiment_signals)
                if sentiment_std < 0.2:
                    assessment['confidence_level'] = 'high'
                elif sentiment_std > 0.4:
                    assessment['confidence_level'] = 'low'
                else:
                    assessment['confidence_level'] = 'moderate'
                    
                # Sentiment divergence
                assessment['sentiment_divergence'] = float(sentiment_std)
                
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing overall economic sentiment: {e}")
            return {'overall_sentiment': 'neutral', 'sentiment_strength': 0.0}
            
    def _analyze_sentiment_momentum(self, sentiment_data: Dict) -> Dict:
        """Analyze sentiment momentum and changes"""
        try:
            momentum_analysis = {
                'momentum_direction': 'stable',
                'momentum_strength': 0.0,
                'sentiment_acceleration': False,
                'key_shifts': []
            }
                
            # This would need historical sentiment data to properly analyze momentum
            # For now, we'll create a framework based on current signals
            momentum_indicators = []
            
            if 'market_sentiment' in sentiment_data:
                market_momentum = sentiment_data['market_sentiment'].get('momentum_sentiment', 'stable')
                if market_momentum == 'building':
                    momentum_indicators.append('positive')
                elif market_momentum == 'fading':
                    momentum_indicators.append('negative')
                    
            if 'fear_greed' in sentiment_data:
                fear_greed_momentum = sentiment_data['fear_greed'].get('sentiment_momentum', 'sentiment_stable')
                if fear_greed_momentum == 'greed_growing':
                    momentum_indicators.append('positive')
                elif fear_greed_momentum == 'fear_growing':
                    momentum_indicators.append('negative')
                    
            # Determine momentum direction
            positive_momentum = momentum_indicators.count('positive')
            negative_momentum = momentum_indicators.count('negative')
            
            if positive_momentum > negative_momentum:
                momentum_analysis['momentum_direction'] = 'improving'
                momentum_analysis['momentum_strength'] = float(positive_momentum / len(momentum_indicators)) if momentum_indicators else 0
            elif negative_momentum > positive_momentum:
                momentum_analysis['momentum_direction'] = 'deteriorating'
                momentum_analysis['momentum_strength'] = float(negative_momentum / len(momentum_indicators)) if momentum_indicators else 0
            else:
                momentum_analysis['momentum_direction'] = 'stable'
                
            # Sentiment acceleration detection
            if len(momentum_indicators) > 1 and momentum_analysis['momentum_strength'] > 0.6:
                momentum_analysis['sentiment_acceleration'] = True
                
            return momentum_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment momentum: {e}")
            return {'momentum_direction': 'stable', 'momentum_strength': 0.0}
            
    def _assess_sentiment_risks(self, sentiment_data: Dict) -> Dict:
        """Assess sentiment-related risks"""
        try:
            risk_assessment = {
                'risk_level': 'moderate',
                'risk_factors': [],
                'sentiment_crash_probability': 0.0,
                'euphoria_risk': False
            }
                
            risk_factors = []
            risk_score = 0
            
            # Extreme sentiment risks
            if 'fear_greed' in sentiment_data:
                fear_greed_data = sentiment_data['fear_greed']
                
                if fear_greed_data.get('fear_level') == 'extreme_fear':
                    risk_factors.append('extreme_fear_conditions')
                    risk_score += 0.3
                elif fear_greed_data.get('greed_level') == 'extreme_greed':
                    risk_factors.append('euphoria_conditions')
                    risk_assessment['euphoria_risk'] = True
                    risk_score += 0.2
                    
            # Flight to quality risks
            if 'safe_haven' in sentiment_data:
                safe_haven_data = sentiment_data['safe_haven']
                
                if safe_haven_data.get('flight_to_quality') == 'active':
                    risk_factors.append('active_flight_to_quality')
                    risk_score += 0.2
                    
            # Market sentiment extremes
            if 'market_sentiment' in sentiment_data:
                market_data = sentiment_data['market_sentiment']
                
                if market_data.get('performance_sentiment') == 'pessimistic':
                    risk_factors.append('extreme_pessimism')
                    risk_score += 0.2
                    
            # Calculate risk level
            risk_assessment['risk_factors'] = risk_factors
            
            if risk_score > 0.6:
                risk_assessment['risk_level'] = 'high'
                risk_assessment['sentiment_crash_probability'] = float(min(1.0, risk_score))
            elif risk_score > 0.3:
                risk_assessment['risk_level'] = 'elevated'
                risk_assessment['sentiment_crash_probability'] = float(risk_score * 0.7)
            else:
                risk_assessment['risk_level'] = 'moderate'
                risk_assessment['sentiment_crash_probability'] = float(risk_score * 0.5)
                
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error assessing sentiment risks: {e}")
            return {'risk_level': 'moderate', 'sentiment_crash_probability': 0.0}
            
    async def get_macro_pulse(self) -> Dict:
        """Get comprehensive macroeconomic analysis"""
        try:
            # Run all macro analyses in parallel
            macro_tasks = [
                self._monetary_policy_analysis(),
                self._fiscal_policy_analysis(),
                self._economic_indicators_analysis(),
                self._policy_impact_analysis(),
                self._economic_sentiment_analysis()
            ]
            
            results = await asyncio.gather(*macro_tasks, return_exceptions=True)
            (
                monetary_policy, fiscal_policy,
                economic_indicators, policy_impact,
                economic_sentiment
            ) = results
            
            # Calculate overall Macro Momentum Score (MMS)
            mms_components = []
            
            for result in results:
                if isinstance(result, dict) and not result.get('error'):
                    # Extract numeric scores from each analysis
                    score = self._extract_macro_score(result)
                    if score is not None:
                        mms_components.append(score)
                        
            if mms_components:
                mms_score = np.mean(mms_components)
                mms_volatility = np.std(mms_components)
                
                # Classify macro state
                if mms_score > 0.4:
                    macro_state = 'expansionary_environment'
                elif mms_score > 0.1:
                    macro_state = 'supportive_environment'
                elif mms_score < -0.4:
                    macro_state = 'challenging_environment'
                elif mms_score < -0.1:
                    macro_state = 'cautious_environment'
                else:
                    macro_state = 'neutral_environment'
                    
                return {
                    'macro_momentum_score': mms_score,
                    'mms_volatility': mms_volatility,
                    'macro_state': macro_state,
                    'analysis_breakdown': {
                        'monetary_policy': monetary_policy,
                        'fiscal_policy': fiscal_policy,
                        'economic_indicators': economic_indicators,
                        'policy_impact': policy_impact,
                        'economic_sentiment': economic_sentiment
                    },
                    'timestamp': datetime.utcnow(),
                    'confidence': min(1.0, 1 - (mms_volatility / 2))
                }
            else:
                return {'error': 'Unable to calculate macro momentum score'}
                
        except Exception as e:
            logger.error(f"Error getting macro pulse: {e}")
            return {'error': str(e)}
            
    def _extract_macro_score(self, analysis_result: Dict) -> Optional[float]:
        """Extract numeric macro score from analysis result"""
        try:
            if 'policy_stance' in analysis_result:
                stance = analysis_result['policy_stance'].get('current_stance', 'neutral')
                if stance == 'accommodative':
                    return 0.6
                elif stance == 'restrictive':
                    return -0.6
                else:
                    return 0.0
            elif 'economic_health_score' in analysis_result:
                health = analysis_result['economic_health_score']
                return (health.get('overall_health', 50) - 50) / 50  # Normalize to -1 to 1
            elif 'impact_assessment' in analysis_result:
                impact = analysis_result['impact_assessment']
                overall_impact = impact.get('overall_impact', 'neutral')
                if overall_impact == 'significant':
                    return 0.5
                elif overall_impact == 'moderate':
                    return 0.2
                elif overall_impact == 'minimal':
                    return -0.2
                else:
                    return 0.0
            elif 'sentiment_assessment' in analysis_result:
                sentiment = analysis_result['sentiment_assessment']
                overall_sentiment = sentiment.get('overall_sentiment', 'neutral')
                if 'positive' in overall_sentiment:
                    return 0.4 if 'extremely' not in overall_sentiment else 0.6
                elif 'negative' in overall_sentiment:
                    return -0.4 if 'extremely' not in overall_sentiment else -0.6
                else:
                    return 0.0
            else:
                return None
                
        except Exception:
            return None
            
    async def store_macro_data(self, macro_data: Dict):
        """Store macro metrics in time-series database"""
        try:
            if self.db_manager and 'timestamp' in macro_data:
                # Store Macro Momentum Score
                await self.db_manager.influxdb_client.write_points(
                    database='market_pulse',
                    measurement='macro_metrics',
                    tags={
                        'engine': 'macro_pulse',
                        'state': macro_data.get('macro_state', 'unknown')
                    },
                    fields={
                        'mms_score': float(macro_data.get('macro_momentum_score', 0)),
                        'mms_volatility': float(macro_data.get('mms_volatility', 0)),
                        'confidence': float(macro_data.get('confidence', 0))
                    },
                    time=macro_data['timestamp']
                )
                
                # Store component scores
                for component_name, analysis in macro_data.get('analysis_breakdown', {}).items():
                    if isinstance(analysis, dict):
                        score = self._extract_macro_score(analysis)
                        if score is not None:
                            await self.db_manager.influxdb_client.write_points(
                                database='market_pulse',
                                measurement='macro_components',
                                tags={
                                    'component': component_name,
                                    'engine': 'macro_pulse'
                                },
                                fields={'component_score': float(score)},
                                time=macro_data['timestamp']
                            )
                            
            logger.debug("Macro data stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing macro data: {e}")
            
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
                'cache_size': len(self.macro_cache),
                'models_loaded': len(self.models),
                'macro_indicators': len(self.macro_indicators),
                'policy_frameworks': len(self.policy_frameworks),
                'economic_regimes': len(self.economic_regimes),
                'health_score': self._calculate_engine_health()
            }
            
        except Exception as e:
            logger.error(f"Error getting macro engine status: {e}")
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
                cache_freshness = max(0, 1 - (minutes_since_update / 60))  # Macroeconomic data updates less frequently
                health_factors.append(cache_freshness)
            
            # Model availability
            health_factors.append(min(1.0, len(self.models) / 3))
            
            # Data source coverage
            total_data_sources = len(self.macro_indicators)
            health_factors.append(min(1.0, total_data_sources / 20))
            
            return np.mean(health_factors) if health_factors else 0.0
            
        except Exception:
            return 0.0