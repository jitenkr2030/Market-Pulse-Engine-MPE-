#!/usr/bin/env python3
"""
Detailed MPE Engine Analysis
Shows individual engine outputs and detailed MPI calculation
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the MPE services directory to Python path
sys.path.insert(0, '/workspace/mpe/services')

from market_pulse_synthesizer import CompleteMarketPulseSynthesizer

async def detailed_engine_analysis():
    """Run detailed analysis showing individual engine outputs"""
    print("🔬 DETAILED MPE ENGINE ANALYSIS")
    print("=" * 60)
    
    # Initialize MPE System
    synthesizer = CompleteMarketPulseSynthesizer()
    
    # Set analysis parameters
    symbols = ('SPY', 'QQQ', 'IWM', 'TLT', 'GLD')
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')  # 6 months
    
    print(f"📅 Analysis Period: {start_date} to {end_date}")
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"⚙️ Engines Active: {len(synthesizer.engines)}/30")
    print()
    
    # Test each engine individually
    engine_results = {}
    
    print("🔧 TESTING INDIVIDUAL ENGINES:")
    print("-" * 40)
    
    for engine_name, engine in synthesizer.engines.items():
        try:
            print(f"\n🧪 Testing {engine_name}...")
            
            # Get the engine's analysis method
            if hasattr(engine, 'analyze_market'):
                result = await engine.analyze_market(symbols, start_date, end_date)
            elif hasattr(engine, 'generate_signals'):
                result = await engine.generate_signals(symbols, start_date, end_date)
            elif hasattr(engine, 'predict'):
                result = await engine.predict(symbols, start_date, end_date)
            elif hasattr(engine, 'analyze'):
                result = await engine.analyze(symbols, start_date, end_date)
            elif hasattr(engine, 'get_pulse_data'):
                # Special handling for pulse engines - prioritize get_pulse_data
                method = getattr(engine, 'get_pulse_data')
                if asyncio.iscoroutinefunction(method):
                    result = await method(list(symbols))
                else:
                    result = method(list(symbols))
            else:
                # Try to find any other async analysis method
                methods = [method for method in dir(engine) if method.startswith('analyze') or method.startswith('generate') or method.startswith('predict')]
                if methods:
                    method = getattr(engine, methods[0])
                    if asyncio.iscoroutinefunction(method):
                        result = await method(symbols, start_date, end_date)
                    else:
                        result = method(symbols, start_date, end_date)
                else:
                    result = "No analysis method found"
            
            engine_results[engine_name] = result
            
            if isinstance(result, dict):
                print(f"  ✅ {engine_name}: {len(result)} data points")
                # Show key metrics
                for key, value in list(result.items())[:3]:  # Show first 3 keys
                    if isinstance(value, (int, float)):
                        print(f"    • {key}: {value:.4f}")
                    else:
                        print(f"    • {key}: {str(value)[:50]}...")
            else:
                print(f"  ✅ {engine_name}: {type(result).__name__}")
                
        except Exception as e:
            print(f"  ❌ {engine_name}: Error - {str(e)[:80]}...")
            engine_results[engine_name] = f"Error: {str(e)}"
    
    print(f"\n📊 INDIVIDUAL ENGINE SUMMARY:")
    print("-" * 40)
    
    working_engines = 0
    failed_engines = 0
    
    for engine_name, result in engine_results.items():
        if isinstance(result, str) and result.startswith("Error"):
            print(f"  ❌ {engine_name}")
            failed_engines += 1
        else:
            print(f"  ✅ {engine_name}")
            working_engines += 1
    
    print(f"\n📈 ENGINE STATUS SUMMARY:")
    print(f"  Working: {working_engines}")
    print(f"  Failed: {failed_engines}")
    print(f"  Success Rate: {working_engines/(working_engines+failed_engines)*100:.1f}%")
    
    # Run full MPE analysis
    print(f"\n🚀 RUNNING COMPLETE MPE ANALYSIS:")
    print("=" * 50)
    
    try:
        full_result = await synthesizer.generate_complete_market_pulse(symbols, start_date, end_date)
        
        print(f"\n🎯 FINAL MPE RESULTS:")
        print("-" * 30)
        
        # Market Pulse Index
        mpi_data = full_result.get('market_pulse_index', {})
        print(f"📊 MARKET PULSE INDEX:")
        print(f"  Score: {mpi_data.get('mpi_score', 'N/A')}")
        print(f"  Raw MPI: {mpi_data.get('raw_mpi', 'N/A')}")
        print(f"  Confidence: {mpi_data.get('confidence', 'N/A')}")
        print(f"  Interpretation: {mpi_data.get('interpretation', 'N/A')}")
        print(f"  Signal Strength: {mpi_data.get('signal_strength', 'N/A')}")
        
        # Market Regime
        regime_data = full_result.get('market_regime', {})
        print(f"\n🎭 MARKET REGIME:")
        print(f"  Regime: {regime_data.get('regime', 'N/A')}")
        print(f"  Confidence: {regime_data.get('confidence', 'N/A')}")
        print(f"  Description: {regime_data.get('description', 'N/A')}")
        
        # Signal Generation
        signals_data = full_result.get('signals', [])
        print(f"\n🚨 GENERATED SIGNALS:")
        if signals_data:
            for i, signal in enumerate(signals_data, 1):
                print(f"  {i}. {signal}")
        else:
            print("  No signals generated")
        
        # Risk Assessment
        risk_data = full_result.get('risk_metrics', {})
        print(f"\n⚠️ RISK METRICS:")
        if risk_data:
            for metric, value in risk_data.items():
                if isinstance(value, (int, float)):
                    print(f"  • {metric}: {value:.4f}")
                else:
                    print(f"  • {metric}: {value}")
        else:
            print("  No risk metrics available")
        
        # Performance Metrics
        perf_data = full_result.get('performance_metrics', {})
        print(f"\n📈 PERFORMANCE METRICS:")
        if perf_data:
            for metric, value in perf_data.items():
                if isinstance(value, (int, float)):
                    print(f"  • {metric}: {value:.4f}")
                else:
                    print(f"  • {metric}: {value}")
        else:
            print("  No performance metrics available")
            
    except Exception as e:
        print(f"❌ Full MPE analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "=" * 60)
    print("✅ DETAILED MPE ANALYSIS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(detailed_engine_analysis())