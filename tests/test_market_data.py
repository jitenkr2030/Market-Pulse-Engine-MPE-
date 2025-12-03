#!/usr/bin/env python3
"""
Test market data fetching to understand the "Failed to download data" errors
"""

import yfinance as yf
import sys
import traceback

def test_market_data():
    """Test if yfinance can fetch data for our symbols"""
    
    print("🔍 TESTING MARKET DATA FETCHING")
    print("=" * 50)
    
    symbols = ['SPY', 'QQQ', 'IWM']
    
    for symbol in symbols:
        print(f"\n🧪 Testing {symbol}:")
        
        try:
            # Test basic ticker creation
            ticker = yf.Ticker(symbol)
            print(f"  ✅ Ticker created successfully")
            
            # Test info
            try:
                info = ticker.info
                print(f"  ✅ Info retrieved: {len(info)} fields")
                if info:
                    print(f"    • Long name: {info.get('longName', 'N/A')}")
                    print(f"    • Market cap: {info.get('marketCap', 'N/A')}")
            except Exception as e:
                print(f"  ❌ Info failed: {e}")
            
            # Test recent history
            try:
                hist = ticker.history(period="5d")
                print(f"  ✅ History retrieved: {len(hist)} days")
                if len(hist) > 0:
                    print(f"    • Latest price: ${hist['Close'].iloc[-1]:.2f}")
                    print(f"    • Date range: {hist.index[0].date()} to {hist.index[-1].date()}")
                else:
                    print(f"  ⚠️ No historical data returned")
            except Exception as e:
                print(f"  ❌ History failed: {e}")
                traceback.print_exc()
            
            # Test longer history
            try:
                long_hist = ticker.history(period="1mo")
                print(f"  ✅ Long history retrieved: {len(long_hist)} days")
            except Exception as e:
                print(f"  ❌ Long history failed: {e}")
                
        except Exception as e:
            print(f"  ❌ Ticker creation failed: {e}")
            traceback.print_exc()

def test_sentiment_pulse_direct():
    """Test sentiment_pulse engine directly with debug output"""
    
    print(f"\n\n🔬 TESTING SENTIMENT PULSE DIRECTLY")
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
            result = engine.get_pulse_data(symbols)
            print(f"✅ get_pulse_data succeeded: {type(result)}")
            print(f"  • Keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            
            if isinstance(result, dict):
                for key, value in result.items():
                    print(f"  • {key}: {type(value)}")
                    if isinstance(value, dict):
                        print(f"    • {list(value.keys())}")
                        
        except Exception as e:
            print(f"❌ get_pulse_data failed: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ SentimentPulseEngine initialization failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_market_data()
    test_sentiment_pulse_direct()