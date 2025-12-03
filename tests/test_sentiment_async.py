#!/usr/bin/env python3
"""
Test sentiment_pulse engine with proper async handling
"""

import asyncio
import sys
import traceback

async def test_sentiment_pulse_async():
    """Test sentiment_pulse engine directly with proper async handling"""
    
    print("🔬 TESTING SENTIMENT PULSE WITH ASYNC")
    print("=" * 50)
    
    try:
        sys.path.insert(0, '/workspace/mpe/services')
        from sentiment_pulse import SentimentPulseEngine
        
        engine = SentimentPulseEngine()
        print("✅ SentimentPulseEngine initialized")
        
        # Test with symbols
        symbols = ['SPY', 'QQQ', 'IWM']
        print(f"🧪 Testing with symbols: {symbols}")
        
        # Test get_pulse_data
        try:
            result = await engine.get_pulse_data(symbols)
            print(f"✅ get_pulse_data succeeded: {type(result)}")
            
            if isinstance(result, dict):
                print(f"  • Keys: {list(result.keys())}")
                
                # Check each key
                for key, value in result.items():
                    print(f"  • {key}: {type(value)}")
                    if isinstance(value, dict):
                        print(f"    • Sub-keys: {list(value.keys())}")
                        
                # Check for errors
                if "error" in result:
                    print(f"  ❌ ERROR FOUND: {result['error']}")
                else:
                    print(f"  ✅ No errors in result")
            else:
                print(f"  ❌ Result is not a dict: {result}")
                        
        except Exception as e:
            print(f"❌ get_pulse_data failed: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ SentimentPulseEngine initialization failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_sentiment_pulse_async())