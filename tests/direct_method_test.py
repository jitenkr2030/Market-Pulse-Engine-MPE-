"""
Direct Method Test - Bypass error handling to get full traceback
"""
import asyncio
import logging
import traceback
import sys
import pandas as pd
import numpy as np
from mpe.services.market_regime_forecaster import MarketRegimeForecaster

# Set up logging
logging.basicConfig(level=logging.DEBUG)

async def test_division_operation_directly():
    """Test the division operation that's causing the error"""
    print("\n🧠 Testing division operation directly")
    print("="*60)
    
    try:
        # Get some sample data
        engine = MarketRegimeForecaster()
        
        # Create test data similar to what the engine would use
        dates = pd.date_range('2025-09-01', '2025-12-03', freq='D')
        test_data = pd.DataFrame({
            'SPY': np.random.randn(len(dates)) * 0.02 + 0.001,
            'QQQ': np.random.randn(len(dates)) * 0.025 + 0.001,
            'SPY_Volume': np.random.randint(50000000, 200000000, len(dates)),
            'QQQ_Volume': np.random.randint(30000000, 150000000, len(dates))
        }, index=dates)
        
        # Extract returns and volume for first symbol
        returns = test_data['SPY'].pct_change().dropna()
        volume = test_data['SPY_Volume'].loc[returns.index]
        
        print(f"📊 Data prepared:")
        print(f"  Returns type: {type(returns)}, dtype: {returns.dtype}")
        print(f"  Volume type: {type(volume)}, dtype: {volume.dtype}")
        print(f"  Returns shape: {returns.shape}")
        print(f"  Volume shape: {volume.shape}")
        
        # Test the market_microstructure_regime method directly
        print("\n🔍 Calling market_microstructure_regime method...")
        result = MarketRegimeForecaster.market_microstructure_regime(returns, volume)
        print(f"✅ Success! Result shape: {result.shape}")
        
    except Exception as e:
        print(f"💥 EXCEPTION in division operation:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print("\n🔍 FULL TRACEBACK:")
        traceback.print_exc()
        print("\n🔍 TRACEBACK WITH LINE NUMBERS:")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_list = traceback.extract_tb(exc_traceback)
        for frame in tb_list:
            if 'market_regime_forecaster' in frame.filename:
                print(f"   📍 File: {frame.filename}")
                print(f"   📍 Line: {frame.lineno}")
                print(f"   📍 Function: {frame.name}")
                print(f"   📍 Code: {frame.line}")
                print()

async def main():
    await test_division_operation_directly()
    print("\n🏁 Direct method test completed")

if __name__ == "__main__":
    asyncio.run(main())