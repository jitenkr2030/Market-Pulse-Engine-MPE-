"""
Full Pipeline Test - Test the complete analyze method with debug logging
"""
import asyncio
import logging
import traceback
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from mpe.services.market_regime_forecaster import MarketRegimeForecaster

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

async def test_full_pipeline():
    """Test the complete analyze method"""
    print("\n🧠 Testing full MarketRegimeForecaster pipeline")
    print("="*60)
    
    try:
        engine = MarketRegimeForecaster()
        
        print("📊 Calling analyze method...")
        result = await engine.analyze(('SPY', 'QQQ'), '2025-09-01', '2025-12-03')
        
        print(f"✅ Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"📊 Result keys: {list(result.keys())}")
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print("✅ No error found!")
                # Print some details
                if 'regime_dimensions' in result:
                    regime_dims = result['regime_dimensions']
                    print(f"📈 Regime dimensions: {list(regime_dims.keys())}")
        
    except Exception as e:
        print(f"💥 EXCEPTION in full pipeline test:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print("\n🔍 FULL TRACEBACK:")
        traceback.print_exc()
        
        # Try to get more details about the exception
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_list = traceback.extract_tb(exc_traceback)
        print("\n🔍 Relevant stack frames:")
        for frame in tb_list:
            if 'market_regime_forecaster' in frame.filename:
                print(f"   📍 File: {frame.filename}")
                print(f"   📍 Line: {frame.lineno}")
                print(f"   📍 Function: {frame.name}")
                print(f"   📍 Code: {frame.line}")
                print()

async def main():
    await test_full_pipeline()
    print("\n🏁 Full pipeline test completed")

if __name__ == "__main__":
    asyncio.run(main())