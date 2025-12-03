#!/usr/bin/env python3
"""
Diagnostic test to identify the exact source of data processing errors
"""

import sys
import logging
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_market_regime_forecaster():
    """Test market regime forecaster with detailed error analysis."""
    print("🔍 Testing market_regime_forecaster with detailed analysis...")
    
    try:
        # Import and initialize
        sys.path.append('/workspace/mpe/services')
        from market_regime_forecaster import MarketRegimeForecaster
        
        forecaster = MarketRegimeForecaster()
        
        # Create test data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        print(f"Fetching data from {start_date.date()} to {end_date.date()}")
        
        # Get test data
        data = yf.download(['SPY'], start=start_date, end=end_date, progress=False)
        if data.empty:
            print("❌ Failed to fetch data")
            return False
            
        # Extract returns - handle different Yahoo Finance formats
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif ('Close', 'SPY') in data.columns:
            prices = data[('Close', 'SPY')]
        else:
            prices = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]  # Fallback to first column
        
        returns = prices.pct_change().dropna()
        
        print(f"Data shape: {returns.shape}")
        print(f"Returns data type: {returns.dtype}")
        print(f"Returns first few values: {returns.head()}")
        
        # Test each component individually
        print("\n🧪 Testing individual components...")
        
        try:
            print("  - Testing volatility regime...")
            vol_regime = forecaster.indicators.volatility_regime(returns, window=20)
            print(f"    ✅ Volatility regime OK: {vol_regime.shape}")
        except Exception as e:
            print(f"    ❌ Volatility regime error: {e}")
            traceback.print_exc()
        
        try:
            print("  - Testing correlation regime...")
            corr_data = {'SPY': returns, 'QQQ': returns * 0.8}  # Create synthetic correlation data
            corr_regime = forecaster.indicators.cross_sectional_regime(corr_data, window=20)
            print(f"    ✅ Correlation regime OK: {corr_regime.shape}")
        except Exception as e:
            print(f"    ❌ Correlation regime error: {e}")
            traceback.print_exc()
        
        try:
            print("  - Testing liquidity regime...")
            volume = pd.Series(np.random.lognormal(10, 1, len(returns)))
            volume.index = returns.index
            liquidity_regime = forecaster.indicators.liquidity_regime(returns, volume, window=20)
            print(f"    ✅ Liquidity regime OK: {liquidity_regime.shape}")
        except Exception as e:
            print(f"    ❌ Liquidity regime error: {e}")
            traceback.print_exc()
        
        try:
            print("  - Testing microstructure regime...")
            volume = pd.Series(np.random.lognormal(10, 1, len(returns)))
            volume.index = returns.index
            microstructure_regime = forecaster.indicators.market_microstructure_regime(returns, volume, window=20)
            print(f"    ✅ Microstructure regime OK: {microstructure_regime.shape}")
        except Exception as e:
            print(f"    ❌ Microstructure regime error: {e}")
            traceback.print_exc()
        
        try:
            print("  - Testing change point detection...")
            change_points = forecaster.transition_detector.change_point_detection(returns, method='bayesian')
            print(f"    ✅ Change point detection OK: {len(change_points)} points found")
        except Exception as e:
            print(f"    ❌ Change point detection error: {e}")
            traceback.print_exc()
        
        # Test the full analyze method
        try:
            print("\n🎯 Testing full analyze method...")
            symbols_tuple = ('SPY',)
            result = forecaster.analyze(symbols_tuple, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            print(f"    ✅ Full analysis OK: {type(result)}")
            return True
        except Exception as e:
            print(f"    ❌ Full analysis error: {e}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        traceback.print_exc()
        return False

def test_liquidity_prediction_engine():
    """Test liquidity prediction engine with detailed error analysis."""
    print("\n🔍 Testing liquidity_prediction_engine with detailed analysis...")
    
    try:
        # Import and initialize
        sys.path.append('/workspace/mpe/services')
        from liquidity_prediction_engine import LiquidityPredictionEngine
        
        engine = LiquidityPredictionEngine()
        
        # Create test data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        print(f"Fetching data from {start_date.date()} to {end_date.date()}")
        
        # Get test data
        data = yf.download(['SPY'], start=start_date, end=end_date, progress=False)
        if data.empty:
            print("❌ Failed to fetch data")
            return False
            
        print(f"Data shape: {data.shape}")
        
        # Create factors DataFrame
        factors = pd.DataFrame(index=data.index)
        # Handle different Yahoo Finance data formats
        if 'Adj Close' in data.columns:
            price_data = data['Adj Close']
        elif ('Close', 'SPY') in data.columns:
            price_data = data[('Close', 'SPY')]
        else:
            price_data = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]  # Fallback to first column
        
        factors['returns'] = price_data.pct_change()
        factors['volume'] = data['Volume']
        factors['volatility'] = factors['returns'].rolling(20).std()
        factors['volume_ratio'] = factors['volume'] / factors['volume'].rolling(20).mean()
        
        # Create target Series
        target = factors['volume_ratio'].shift(-1)  # Next period volume ratio
        
        print(f"Factors shape: {factors.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Factors dtypes: {factors.dtypes}")
        
        # Test prepare_features step by step
        try:
            print("\n🧪 Testing prepare_features step by step...")
            
            # Step 1: Align data
            common_index = factors.index.intersection(target.index)
            X = factors.loc[common_index]
            y = target.loc[common_index]
            print(f"  Aligned X shape: {X.shape}, y shape: {y.shape}")
            
            # Step 2: Create lag features
            lag_features = pd.DataFrame(index=X.index)
            for col in X.columns:
                col_data = X[col].astype(float)
                for lag in [1, 2, 3, 5, 10]:
                    lag_features[f'{col}_lag_{lag}'] = col_data.shift(lag)
            print(f"  Lag features shape: {lag_features.shape}")
            
            # Step 3: Create rolling statistics
            for col in X.columns:
                col_data = X[col].astype(float)
                lag_features[f'{col}_ma_5'] = col_data.rolling(5).mean()
                lag_features[f'{col}_ma_10'] = col_data.rolling(10).mean()
                lag_features[f'{col}_std_5'] = col_data.rolling(5).std()
                lag_features[f'{col}_vol_5'] = col_data.rolling(5).std()
            print(f"  Final lag features shape: {lag_features.shape}")
            
            # Step 4: Combine features
            X_clean = X.fillna(0)
            lag_clean = lag_features.fillna(0)
            
            common_index = X_clean.index.intersection(lag_clean.index)
            X_aligned = X_clean.loc[common_index]
            lag_aligned = lag_clean.loc[common_index]
            
            print(f"  X_aligned shape: {X_aligned.shape}")
            print(f"  lag_aligned shape: {lag_aligned.shape}")
            
            if X_aligned.shape[1] != lag_aligned.shape[1]:
                print(f"  ⚠️  Column count mismatch: {X_aligned.shape[1]} vs {lag_aligned.shape[1]}")
                print(f"     X_aligned columns: {list(X_aligned.columns)}")
                print(f"     lag_aligned columns: {list(lag_aligned.columns)}")
            
            # Test concatenation
            X_combined = pd.concat([X_aligned, lag_aligned], axis=1, sort=False)
            print(f"  X_combined shape: {X_combined.shape}")
            
            y_aligned = y.reindex(X_combined.index, method='ffill').fillna(0)
            print(f"  y_aligned shape: {y_aligned.shape}")
            
            print("  ✅ prepare_features step-by-step OK")
            
        except Exception as e:
            print(f"  ❌ prepare_features step-by-step error: {e}")
            traceback.print_exc()
        
        # Test the full analyze method
        try:
            print("\n🎯 Testing full analyze method...")
            symbols_tuple = ('SPY',)
            result = engine.analyze(symbols_tuple, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            print(f"    ✅ Full analysis OK: {type(result)}")
            return True
        except Exception as e:
            print(f"    ❌ Full analysis error: {e}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Phase 3C: Diagnostic Testing for Data Processing Errors")
    print("=" * 60)
    
    success1 = test_market_regime_forecaster()
    success2 = test_liquidity_prediction_engine()
    
    print("\n" + "=" * 60)
    print(f"🎯 Results:")
    print(f"  market_regime_forecaster: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    print(f"  liquidity_prediction_engine: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\n🎉 Both engines are now operational!")
    else:
        print("\n🔧 Additional fixes needed based on diagnostic results")