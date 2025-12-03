#!/usr/bin/env python3
"""
Comprehensive MPE System Test with Real Market Data
Tests all operational engines with live market data for major symbols
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import yfinance as yf

# Add the MPE services directory to Python path
sys.path.insert(0, '/workspace/mpe/services')

from market_pulse_synthesizer import CompleteMarketPulseSynthesizer

def fetch_market_data(symbols: List[str], days: int = 252) -> Dict[str, pd.DataFrame]:
    """Fetch historical market data for specified symbols"""
    print(f"📊 Fetching market data for {len(symbols)} symbols...")
    
    data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days * 1.2)  # Extra buffer for calculations
    
    for symbol in symbols:
        try:
            print(f"  Fetching {symbol} data...")
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(start=start_date, end=end_date)
            
            if len(hist_data) > 50:  # Ensure sufficient data
                data[symbol] = hist_data
                print(f"  ✅ {symbol}: {len(hist_data)} days of data")
            else:
                print(f"  ⚠️ {symbol}: Insufficient data ({len(hist_data)} days)")
        except Exception as e:
            print(f"  ❌ {symbol}: Error fetching data - {str(e)}")
    
    return data

def prepare_market_data(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Prepare and structure market data for MPE analysis"""
    print("\n🔧 Preparing market data for analysis...")
    
    # Combine all symbols into unified dataset
    all_data = {}
    
    for symbol, df in market_data.items():
        symbol_data = {
            'prices': df['Close'].to_dict(),
            'volumes': df['Volume'].to_dict() if 'Volume' in df.columns else {},
            'highs': df['High'].to_dict(),
            'lows': df['Low'].to_dict(),
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'returns': df['Close'].pct_change().dropna().to_dict()
        }
        all_data[symbol] = symbol_data
        print(f"  ✅ {symbol}: Prepared {len(symbol_data['dates'])} data points")
    
    # Create portfolio and market data
    market_summary = {
        'symbols': list(market_data.keys()),
        'combined_returns': pd.concat([pd.Series(data['returns']) for data in all_data.values()], axis=1, keys=all_data.keys()),
        'portfolio_value': 100000,  # Simulated portfolio
        'cash_position': 20000,
        'risk_budget': 0.15
    }
    
    return {
        'individual_symbols': all_data,
        'market_summary': market_summary,
        'timestamp': datetime.now().isoformat()
    }

async def run_comprehensive_mpe_analysis(symbols: List[str]) -> Dict[str, Any]:
    """Run complete MPE system analysis"""
    print("\n🚀 Initializing MPE System...")
    
    try:
        # Initialize the complete market pulse synthesizer
        synthesizer = CompleteMarketPulseSynthesizer()
        print("✅ MPE Synthesizer initialized")
        
        print("\n📈 Running comprehensive market analysis...")
        print("=" * 60)
        
        # Prepare date range for analysis
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=252)).strftime('%Y-%m-%d')  # 1 year ago
        
        print(f"📅 Analysis period: {start_date} to {end_date}")
        print(f"📊 Analyzing symbols: {', '.join(symbols)}")
        
        # Generate complete market pulse analysis
        result = await synthesizer.generate_complete_market_pulse(
            symbols=tuple(symbols),
            start_date=start_date,
            end_date=end_date
        )
        
        return result
        
    except Exception as e:
        print(f"❌ Error in MPE analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def display_results(results: Dict[str, Any]):
    """Display comprehensive analysis results"""
    if not results:
        print("❌ No results to display")
        return
    
    print("\n" + "=" * 80)
    print("🎯 MARKET PULSE ENGINE - COMPREHENSIVE ANALYSIS RESULTS")
    print("=" * 80)
    
    # Market Pulse Index (MPI)
    mpi_score = results.get('market_pulse_index', 'N/A')
    print(f"\n📊 MARKET PULSE INDEX (MPI): {mpi_score}")
    
    # Market Regime
    regime = results.get('market_regime', 'N/A')
    print(f"🎭 MARKET REGIME: {regime}")
    
    # Confidence Level
    confidence = results.get('confidence_level', 'N/A')
    print(f"🎯 CONFIDENCE LEVEL: {confidence}")
    
    # Individual Engine Outputs
    print(f"\n🔧 INDIVIDUAL ENGINE ANALYSIS:")
    print("-" * 50)
    
    engine_outputs = results.get('engine_outputs', {})
    for engine_name, output in engine_outputs.items():
        if isinstance(output, dict):
            print(f"\n🔍 {engine_name.upper()}:")
            for key, value in output.items():
                if isinstance(value, (int, float)):
                    print(f"  • {key}: {value:.4f}" if abs(value) < 1 else f"  • {key}: {value:.2f}")
                else:
                    print(f"  • {key}: {value}")
        else:
            print(f"  • {engine_name}: {output}")
    
    # Generated Signals
    print(f"\n🚨 GENERATED SIGNALS:")
    print("-" * 30)
    
    signals = results.get('signals', [])
    if signals:
        for signal in signals:
            print(f"  📢 {signal}")
    else:
        print("  ℹ️ No specific signals generated")
    
    # Risk Assessment
    print(f"\n⚠️ RISK ASSESSMENT:")
    print("-" * 25)
    
    risk_metrics = results.get('risk_metrics', {})
    for metric, value in risk_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  • {metric}: {value:.4f}" if abs(value) < 1 else f"  • {metric}: {value:.2f}")
        else:
            print(f"  • {metric}: {value}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    recommendations = results.get('recommendations', [])
    if recommendations:
        for rec in recommendations:
            print(f"  💭 {rec}")
    else:
        print("  ℹ️ No specific recommendations generated")
    
    # Performance Metrics
    print(f"\n📈 PERFORMANCE METRICS:")
    print("-" * 28)
    
    performance = results.get('performance_metrics', {})
    for metric, value in performance.items():
        if isinstance(value, (int, float)):
            print(f"  • {metric}: {value:.4f}" if abs(value) < 1 else f"  • {metric}: {value:.2f}")
        else:
            print(f"  • {metric}: {value}")

async def main():
    """Main test execution"""
    print("🎯 COMPREHENSIVE MPE SYSTEM TEST")
    print("=" * 50)
    
    # Define major market symbols for analysis
    symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
    
    print(f"📋 Testing with symbols: {', '.join(symbols)}")
    print(f"⏰ Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Fetch real market data
    market_data = fetch_market_data(symbols, days=252)  # 1 year of data
    
    if not market_data:
        print("❌ Failed to fetch market data")
        return
    
    print(f"\n✅ Successfully fetched data for {len(market_data)} symbols")
    
    # Step 2: Run comprehensive MPE analysis
    results = await run_comprehensive_mpe_analysis(symbols)
    
    # Step 4: Display results
    display_results(results)
    
    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE MPE TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    # Run the comprehensive test
    asyncio.run(main())