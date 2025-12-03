"""
Realistic Data Test - Test with actual yfinance data format
"""
import asyncio
import logging
import traceback
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from mpe.services.market_regime_forecaster import MarketRegimeForecaster

# Set up logging
logging.basicConfig(level=logging.DEBUG)

async def test_with_real_data():
    """Test with real yfinance data"""
    print("\n🧠 Testing with real yfinance data")
    print("="*60)
    
    try:
        # Download real data
        print("📊 Downloading real data...")
        spy = yf.download('SPY', start='2025-09-01', end='2025-12-03')
        
        if spy.empty:
            print("❌ No data downloaded")
            return
            
        print(f"✅ Data downloaded: {spy.shape}")
        print(f"📊 Columns: {spy.columns.tolist()}")
        print(f"📊 Index type: {type(spy.index)}")
        
        # Extract returns and volume in the format expected
        # Handle MultiIndex columns
        close_prices = spy['Close']['SPY'] if ('Close', 'SPY') in spy.columns else spy['Close'].iloc[:, 0]
        returns = close_prices.pct_change().dropna()
        
        volume = spy['Volume']['SPY'] if ('Volume', 'SPY') in spy.columns else spy['Volume'].iloc[:, 0]
        volume = volume.loc[returns.index]
        
        print(f"\n📊 Data prepared:")
        print(f"  Returns shape: {returns.shape}")
        print(f"  Volume shape: {volume.shape}")
        print(f"  Returns dtype: {returns.dtype}")
        print(f"  Volume dtype: {volume.dtype}")
        print(f"  Returns index type: {type(returns.index)}")
        print(f"  Volume index type: {type(volume.index)}")
        
        # Test the market_microstructure_regime method directly
        print("\n🔍 Calling market_microstructure_regime method...")
        from mpe.services.market_regime_forecaster import RegimeIndicators
        result = RegimeIndicators.market_microstructure_regime(returns, volume)
        print(f"✅ Success! Result shape: {result.shape}")
        print(f"📊 Result dtype: {result.dtype}")
        print(f"📊 Unique regimes: {result.unique()}")
        
    except Exception as e:
        print(f"💥 EXCEPTION in real data test:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print("\n🔍 FULL TRACEBACK:")
        traceback.print_exc()

async def main():
    await test_with_real_data()
    print("\n🏁 Real data test completed")

if __name__ == "__main__":
    asyncio.run(main())