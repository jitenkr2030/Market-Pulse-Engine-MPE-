"""
Full Pipeline Test for Liquidity Prediction Engine
"""
import asyncio
import logging
import traceback
import sys
from mpe.services.liquidity_prediction_engine import LiquidityPredictionEngine

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

async def test_liquidity_full_pipeline():
    """Test the complete liquidity prediction pipeline"""
    print("\n💧 Testing full LiquidityPredictionEngine pipeline")
    print("="*60)
    
    try:
        engine = LiquidityPredictionEngine()
        
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
                if 'signals' in result:
                    signals = result['signals']
                    print(f"📈 Signal keys: {list(signals.keys())}")
        
    except Exception as e:
        print(f"💥 EXCEPTION in full liquidity pipeline test:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print("\n🔍 FULL TRACEBACK:")
        traceback.print_exc()

async def main():
    await test_liquidity_full_pipeline()
    print("\n🏁 Full liquidity pipeline test completed")

if __name__ == "__main__":
    asyncio.run(main())