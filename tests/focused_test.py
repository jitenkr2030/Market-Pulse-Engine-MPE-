#!/usr/bin/env python3
"""
Focused test for specific data processing errors
"""

import sys
import asyncio
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_specific_error():
    """Test the specific methods causing issues."""
    print("🔧 Focused Testing for Data Processing Errors")
    print("=" * 60)
    
    try:
        # Import the engines
        sys.path.append('/workspace/mpe/services')
        from market_regime_forecaster import RegimeIndicators
        from liquidity_prediction_engine import LiquidityPredictor
        
        # Create instances
        indicators = RegimeIndicators()
        liquidity_predictor = LiquidityPredictor()
        
        # Test data with the same date range as synthesizer
        symbols = ['SPY']
        start_date = '2025-09-01'
        end_date = '2025-12-03'
        
        print(f"Testing with period: {start_date} to {end_date}")
        print()
        
        # Download data like the engines do
        print("📊 Downloading data...")
        import yfinance as yf
        
        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            print(f"Downloaded {len(hist)} days of data for {symbol}")
            
            if not hist.empty and len(hist) > 50:
                data[symbol] = hist
            else:
                print(f"⚠️ Insufficient data for {symbol}: {len(hist)} days")
                continue
        
        if not data:
            print("❌ No data available for testing")
            return
            
        # Calculate returns for market_regime_forecaster test
        print("\n🧪 Testing market_regime_forecaster components...")
        returns_data = {}
        for symbol, df in data.items():
            close_prices = df['Close']
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.iloc[:, 0]
            returns_data[symbol] = close_prices.pct_change().dropna()
            
        print(f"Returns data shape: {len(returns_data['SPY'])}")
        
        # Test correlation regime (this is where the error likely occurs)
        if len(returns_data) > 1:
            print("\n🔍 Testing correlation regime...")
            try:
                returns_df = pd.DataFrame(returns_data)
                corr_regime = indicators.correlation_regime(returns_df, window=20)
                print(f"✅ Correlation regime OK: {len(corr_regime)}")
            except Exception as e:
                print(f"❌ Correlation regime error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n⚠️ Skipping correlation regime (need multiple symbols)")
        
        # Test liquidity_prediction_engine
        print("\n🧪 Testing liquidity_prediction_engine components...")
        
        # Create factors like the engine does
        for symbol, df in data.items():
            close_prices = df['Close']
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.iloc[:, 0]
            volume = df['Volume']
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]
                
            returns = close_prices.pct_change().dropna()
            volatility = returns.rolling(20).std()
            volume_ratio = volume / volume.rolling(20).mean()
            
            factors = pd.DataFrame({
                'returns': returns,
                'volume': volume,
                'volatility': volatility,
                'volume_ratio': volume_ratio
            }).dropna()
            
            target = returns.shift(-1).fillna(0)
            
            print(f"Factors shape: {factors.shape}")
            print(f"Target shape: {target.shape}")
            
            print("\n🔍 Testing prepare_features...")
            try:
                X, y = liquidity_predictor.prepare_features({symbol: returns}, {symbol: target})
                print(f"✅ prepare_features OK: X={X.shape}, y={y.shape}")
            except Exception as e:
                print(f"❌ prepare_features error: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("✅ Focused test completed")
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import pandas as pd
    asyncio.run(test_specific_error())