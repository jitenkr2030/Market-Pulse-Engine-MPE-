#!/usr/bin/env python3
"""
Test volatility_pulse engine specifically to debug the 'list' object is not callable error
"""

import asyncio
import sys
sys.path.insert(0, '/workspace/mpe/services')

from volatility_pulse import VolatilityPulseEngine

async def test_volatility_pulse():
    """Test volatility_pulse engine directly"""
    
    print("🔬 Testing Volatility Pulse Engine Directly")
    print("=" * 50)
    
    try:
        # Initialize the engine
        engine = VolatilityPulseEngine()
        print("✅ VolatilityPulseEngine initialized")
        
        # Test different ways of calling get_pulse_data
        symbols = ['SPY', 'QQQ', 'IWM']
        
        print(f"\n🧪 Testing with symbols: {symbols}")
        
        # Test 1: Call with symbols list
        try:
            print("\n🔍 Test 1: Calling get_pulse_data with symbols list...")
            result1 = await engine.get_pulse_data(symbols)
            print(f"  ✅ SUCCESS with symbols list: {type(result1)}")
        except Exception as e1:
            print(f"  ❌ FAILED with symbols list: {e1}")
        
        # Test 2: Call with None (default)
        try:
            print("\n🔍 Test 2: Calling get_pulse_data with None...")
            result2 = await engine.get_pulse_data(None)
            print(f"  ✅ SUCCESS with None: {type(result2)}")
        except Exception as e2:
            print(f"  ❌ FAILED with None: {e2}")
        
        # Test 3: Call with no parameters
        try:
            print("\n🔍 Test 3: Calling get_pulse_data with no parameters...")
            result3 = await engine.get_pulse_data()
            print(f"  ✅ SUCCESS with no params: {type(result3)}")
        except Exception as e3:
            print(f"  ❌ FAILED with no params: {e3}")
            
        # Test 4: Check engine attributes
        print(f"\n🔍 Engine attributes:")
        print(f"  • Type: {type(engine)}")
        print(f"  • Has get_pulse_data: {hasattr(engine, 'get_pulse_data')}")
        
        # Check if there's something wrong with the method itself
        if hasattr(engine, 'get_pulse_data'):
            method = getattr(engine, 'get_pulse_data')
            print(f"  • get_pulse_data type: {type(method)}")
            print(f"  • Is callable: {callable(method)}")
            print(f"  • Is coroutine function: {asyncio.iscoroutinefunction(method)}")
            
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_volatility_pulse())