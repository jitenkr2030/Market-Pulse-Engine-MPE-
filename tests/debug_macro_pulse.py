#!/usr/bin/env python3
"""
Debug test for Market Intelligence engines
"""

import asyncio
import sys
sys.path.insert(0, '/workspace')

from mpe.services.market_pulse_synthesizer import CompleteMarketPulseSynthesizer

async def debug_engine_call():
    """Debug the engine call process"""
    synthesizer = CompleteMarketPulseSynthesizer()
    
    # Test macro_pulse specifically
    engine_name = 'macro_pulse'
    engine = synthesizer.engines[engine_name]
    
    print(f"Engine name: {engine_name}")
    print(f"Engine class: {type(engine).__name__}")
    print(f"Engine has get_macro_pulse: {hasattr(engine, 'get_macro_pulse')}")
    
    # Debug the _safe_engine_call logic
    symbols = ('SPY', 'QQQ', 'IWM')
    start_date = "2024-06-01"
    end_date = "2024-12-01"
    
    # Test through _safe_engine_call method
    print(f"\n=== Testing through _safe_engine_call ===")
    try:
        result = await synthesizer._safe_engine_call(engine_name, engine, symbols, start_date, end_date)
        if result and not result.get('error'):
            print(f"✅ _safe_engine_call succeeded!")
            print(f"📊 Result keys: {list(result.keys())}")
        else:
            print(f"❌ _safe_engine_call failed: {result}")
    except Exception as e:
        print(f"❌ _safe_engine_call exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_engine_call())