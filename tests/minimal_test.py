#!/usr/bin/env python3
"""
Minimal test to isolate the exact length mismatch error
"""

import sys
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_market_regime_step_by_step():
    """Test each step of market regime analysis to find the exact error."""
    print("🔧 Step-by-step testing for length mismatch error")
    print("=" * 60)
    
    try:
        # Import and create instances
        sys.path.append('/workspace/mpe/services')
        from market_regime_forecaster import RegimeIndicators
        import yfinance as yf
        import pandas as pd
        import numpy as np
        
        indicators = RegimeIndicators()
        
        # Test data
        symbols = ['SPY']
        start_date = '2025-09-01'
        end_date = '2025-12-03'
        
        print(f"Testing with period: {start_date} to {end_date}")
        
        # Download data
        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            print(f"Downloaded {len(hist)} days of data for {symbol}")
            
            if not hist.empty and len(hist) > 50:
                data[symbol] = hist
        
        if not data:
            print("❌ No data available")
            return
        
        # Calculate returns
        returns_data = {}
        for symbol, df in data.items():
            close_prices = df['Close']
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.iloc[:, 0]
            returns = close_prices.pct_change().dropna()
            returns_data[symbol] = returns
            print(f"Returns for {symbol}: {len(returns)} values")
        
        print(f"Returns data: {list(returns_data.keys())}")
        print(f"SPY returns shape: {returns_data['SPY'].shape}")
        
        # Test each regime method step by step
        print("\n🧪 Testing individual regime methods...")
        
        # 1. Test volatility regime
        print("\n1️⃣ Testing volatility regime...")
        try:
            vol_regime = indicators.volatility_regime(returns_data['SPY'], window=20)
            print(f"✅ Volatility regime: {len(vol_regime)} values")
        except Exception as e:
            print(f"❌ Volatility regime error: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. Test trend regime
        print("\n2️⃣ Testing trend regime...")
        try:
            trend_regime = indicators.trend_regime(returns_data['SPY'], window=20)
            print(f"✅ Trend regime: {len(trend_regime)} values")
        except Exception as e:
            print(f"❌ Trend regime error: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. Test liquidity regime
        print("\n3️⃣ Testing liquidity regime...")
        try:
            volume = data['SPY']['Volume']
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]
            liquid_regime = indicators.liquidity_regime(returns_data['SPY'], volume, window=20)
            print(f"✅ Liquidity regime: {len(liquid_regime)} values")
        except Exception as e:
            print(f"❌ Liquidity regime error: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. Test microstructure regime
        print("\n4️⃣ Testing microstructure regime...")
        try:
            micro_regime = indicators.market_microstructure_regime(returns_data['SPY'], volume, window=20)
            print(f"✅ Microstructure regime: {len(micro_regime)} values")
        except Exception as e:
            print(f"❌ Microstructure regime error: {e}")
            import traceback
            traceback.print_exc()
        
        # 5. Test correlation regime (only if we have multiple symbols)
        if len(returns_data) > 1:
            print("\n5️⃣ Testing correlation regime...")
            try:
                returns_df = pd.DataFrame(returns_data)
                corr_regime = indicators.correlation_regime(returns_df, window=20)
                print(f"✅ Correlation regime: {len(corr_regime)} values")
            except Exception as e:
                print(f"❌ Correlation regime error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n5️⃣ Skipping correlation regime (need multiple symbols)")
        
        print("\n" + "=" * 60)
        print("✅ Step-by-step test completed")
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_market_regime_step_by_step()