#!/usr/bin/env python3
"""
Debug volatility_pulse engine call to understand the 'list' object is not callable error
"""

import asyncio
import sys
sys.path.insert(0, '/workspace/mpe/services')

from volatility_pulse import VolatilityPulseEngine

async def debug_volatility_pulse_call():
    """Debug the exact issue with volatility_pulse engine"""
    
    print("🔬 Debugging Volatility Pulse Engine Call")
    print("=" * 50)
    
    # Simulate the same call pattern as the detailed test
    engine_name = 'volatility_pulse'
    symbols = ('SPY', 'QQQ', 'IWM', 'TLT', 'GLD')
    start_date = '2025-06-06'
    end_date = '2025-12-03'
    
    engine = VolatilityPulseEngine()
    
    print(f"Engine: {engine_name}")
    print(f"Symbols: {symbols}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    
    # Check what methods are available
    methods = [method for method in dir(engine) if method.startswith('analyze') or method.startswith('generate') or method.startswith('predict')]
    print(f"Available methods starting with analyze/generate/predict: {methods}")
    
    print(f"Has get_pulse_data: {hasattr(engine, 'get_pulse_data')}")
    
    try:
        # This is what the test script is trying to do
        if hasattr(engine, 'get_pulse_data'):
            method = getattr(engine, 'get_pulse_data')
            print(f"Method object: {method}")
            print(f"Method type: {type(method)}")
            print(f"Is coroutine function: {asyncio.iscoroutinefunction(method)}")
            
            # This should work
            print("\n🔍 Calling get_pulse_data with list(symbols)...")
            result = await method(list(symbols))
            print(f"SUCCESS: Got result of type {type(result)}")
            
        else:
            print("No get_pulse_data method found!")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_volatility_pulse_call())