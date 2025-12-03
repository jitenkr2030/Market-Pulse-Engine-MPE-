"""
Detailed Traceback Test - Show exact line numbers where errors occur
"""
import asyncio
import logging
import traceback
import sys
from mpe.services.market_regime_forecaster import MarketRegimeForecaster
from mpe.services.liquidity_prediction_engine import LiquidityPredictionEngine

# Set up logging to capture all details
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

async def test_market_regime_forecaster():
    """Test market_regime_forecaster with full traceback"""
    print("\n🧠 Testing market_regime_forecaster with detailed traceback")
    print("="*70)
    
    try:
        engine = MarketRegimeForecaster()
        result = await engine.analyze(('SPY', 'QQQ'), '2025-09-01', '2025-12-03')
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        
    except Exception as e:
        print(f"💥 EXCEPTION in market_regime_forecaster:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print("\n🔍 FULL TRACEBACK:")
        traceback.print_exc()
        print("\n🔍 TRACEBACK WITH LINE NUMBERS:")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_list = traceback.extract_tb(exc_traceback)
        for frame in tb_list:
            print(f"   📍 File: {frame.filename}")
            print(f"   📍 Line: {frame.lineno}")
            print(f"   📍 Function: {frame.name}")
            print(f"   📍 Code: {frame.line}")
            print()

async def test_liquidity_prediction_engine():
    """Test liquidity_prediction_engine with full traceback"""
    print("\n💧 Testing liquidity_prediction_engine with detailed traceback")
    print("="*70)
    
    try:
        engine = LiquidityPredictionEngine()
        result = await engine.analyze(('SPY', 'QQQ'), '2025-09-01', '2025-12-03')
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        
    except Exception as e:
        print(f"💥 EXCEPTION in liquidity_prediction_engine:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print("\n🔍 FULL TRACEBACK:")
        traceback.print_exc()
        print("\n🔍 TRACEBACK WITH LINE NUMBERS:")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_list = traceback.extract_tb(exc_traceback)
        for frame in tb_list:
            print(f"   📍 File: {frame.filename}")
            print(f"   📍 Line: {frame.lineno}")
            print(f"   📍 Function: {frame.name}")
            print(f"   📍 Code: {frame.line}")
            print()

async def main():
    await test_market_regime_forecaster()
    await test_liquidity_prediction_engine()
    print("\n🏁 Detailed traceback analysis completed")

if __name__ == "__main__":
    asyncio.run(main())