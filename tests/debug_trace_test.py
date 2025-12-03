"""
Debug Trace Test - Enhanced error tracing for statistical calculation issues
"""
import asyncio
import logging
import sys
import traceback
from mpe.services.market_pulse_synthesizer import CompleteMarketPulseSynthesizer

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

async def debug_market_regime_forecaster():
    """Debug market_regime_forecaster with detailed tracing"""
    print("\n🧠 DEBUGGING MARKET REGIME FORECASTER")
    print("="*60)
    
    try:
        from mpe.services.market_regime_forecaster import MarketRegimeForecaster
        engine = MarketRegimeForecaster()
        
        # Test with debug info
        print("📊 Testing with symbols: ('SPY', 'QQQ')")
        print("📅 Period: 2025-09-01 to 2025-12-03")
        
        result = await engine.analyze(('SPY', 'QQQ'), '2025-09-01', '2025-12-03')
        
        print(f"✅ Result type: {type(result)}")
        print(f"📊 Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if 'error' in result:
            print(f"❌ Error found: {result['error']}")
        else:
            print("✅ No error found!")
            if isinstance(result, dict):
                for key, value in result.items():
                    print(f"   📈 {key}: {type(value)}")
        
    except Exception as e:
        print(f"💥 Exception occurred: {type(e).__name__}: {e}")
        print("🔍 Full traceback:")
        traceback.print_exc()

async def debug_liquidity_prediction_engine():
    """Debug liquidity_prediction_engine with detailed tracing"""
    print("\n💧 DEBUGGING LIQUIDITY PREDICTION ENGINE")
    print("="*60)
    
    try:
        from mpe.services.liquidity_prediction_engine import LiquidityPredictionEngine
        engine = LiquidityPredictionEngine()
        
        # Test with debug info
        print("📊 Testing with symbols: ('SPY', 'QQQ')")
        print("📅 Period: 2025-09-01 to 2025-12-03")
        
        result = await engine.analyze(('SPY', 'QQQ'), '2025-09-01', '2025-12-03')
        
        print(f"✅ Result type: {type(result)}")
        print(f"📊 Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if 'error' in result:
            print(f"❌ Error found: {result['error']}")
        else:
            print("✅ No error found!")
            if isinstance(result, dict):
                for key, value in result.items():
                    print(f"   📈 {key}: {type(value)}")
        
    except Exception as e:
        print(f"💥 Exception occurred: {type(e).__name__}: {e}")
        print("🔍 Full traceback:")
        traceback.print_exc()

async def main():
    """Main debug function"""
    print("🔧 DEBUG TRACE TEST - Enhanced error analysis")
    print("="*60)
    
    await debug_market_regime_forecaster()
    await debug_liquidity_prediction_engine()
    
    print("\n🏁 Debug tracing completed")

if __name__ == "__main__":
    asyncio.run(main())