#!/usr/bin/env python3
"""
Test the engines specifically through the CompleteMarketPulseSynthesizer interface
"""

import asyncio
import sys
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_engine_through_synthesizer():
    """Test individual engines through the synthesizer interface."""
    print("🔧 Testing engines through CompleteMarketPulseSynthesizer interface")
    print("=" * 60)
    
    try:
        # Import the synthesizer
        sys.path.append('/workspace/mpe/services')
        from market_pulse_synthesizer import CompleteMarketPulseSynthesizer
        
        # Initialize synthesizer
        synthesizer = CompleteMarketPulseSynthesizer()
        print("✅ Synthesizer initialized")
        
        # Test parameters - use period with 60+ days (above 50-day minimum)
        symbols = ['SPY']
        end_date = '2025-12-03'
        start_date = '2025-09-01'
        
        print(f"Testing with symbols: {symbols}")
        print(f"Period: {start_date} to {end_date}")
        print()
        
        # Test market_regime_forecaster
        print("🧠 Testing market_regime_forecaster...")
        try:
            engine = synthesizer.engines['market_regime_forecaster']
            result = await synthesizer._safe_engine_call(
                'market_regime_forecaster', engine, tuple(symbols), start_date, end_date
            )
            print(f"   ✅ SUCCESS: {type(result)}")
            if 'error' in result:
                print(f"   ⚠️  Contains error: {result['error']}")
            else:
                print(f"   📊 Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
        
        print()
        
        # Test liquidity_prediction_engine
        print("💧 Testing liquidity_prediction_engine...")
        try:
            engine = synthesizer.engines['liquidity_prediction_engine']
            result = await synthesizer._safe_engine_call(
                'liquidity_prediction_engine', engine, tuple(symbols), start_date, end_date
            )
            print(f"   ✅ SUCCESS: {type(result)}")
            if 'error' in result:
                print(f"   ⚠️  Contains error: {result['error']}")
            else:
                print(f"   📊 Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
        
        print()
        
        # Test redemption_risk_monitor (which works)
        print("⚠️  Testing redemption_risk_monitor (control test)...")
        try:
            engine = synthesizer.engines['redemption_risk_monitor']
            result = await synthesizer._safe_engine_call(
                'redemption_risk_monitor', engine, tuple(symbols), start_date, end_date
            )
            print(f"   ✅ SUCCESS: {type(result)}")
            if 'error' in result:
                print(f"   ⚠️  Contains error: {result['error']}")
            else:
                print(f"   📊 Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            
        print()
        print("=" * 60)
        print("✅ Test completed")
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_engine_through_synthesizer())