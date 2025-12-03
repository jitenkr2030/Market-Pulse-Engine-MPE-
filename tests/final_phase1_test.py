#!/usr/bin/env python3
"""
Final comprehensive test with proper async handling to show Phase 1 progress
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the MPE services directory to Python path
sys.path.insert(0, '/workspace/mpe/services')

from market_pulse_synthesizer import CompleteMarketPulseSynthesizer

async def final_phase1_test():
    """Final comprehensive test of Phase 1 progress"""
    
    print("🎯 PHASE 1 FINAL TEST - COMPREHENSIVE ENGINE ANALYSIS")
    print("=" * 70)
    
    # Initialize MPE System
    synthesizer = CompleteMarketPulseSynthesizer()
    
    # Set analysis parameters
    symbols = ('SPY', 'QQQ', 'IWM')
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print(f"📅 Analysis Period: {start_date} to {end_date}")
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"⚙️ Total Engines: {len(synthesizer.engines)}")
    print()
    
    # Test each engine individually with proper async handling
    engine_results = {}
    working_engines = 0
    failed_engines = 0
    
    print("🔧 INDIVIDUAL ENGINE TESTING:")
    print("-" * 50)
    
    for engine_name, engine in synthesizer.engines.items():
        try:
            print(f"\n🧪 Testing {engine_name}...")
            
            # Use the synthesizer's _safe_engine_call method
            result = await synthesizer._safe_engine_call(engine_name, engine, symbols, start_date, end_date)
            
            if isinstance(result, dict) and "error" not in result:
                print(f"  ✅ {engine_name}: SUCCESS")
                engine_results[engine_name] = result
                working_engines += 1
                
                # Show some key metrics for working engines
                if isinstance(result, dict):
                    keys = list(result.keys())[:3]  # Show first 3 keys
                    print(f"    • Keys: {keys}")
            else:
                print(f"  ❌ {engine_name}: {str(result)[:60]}...")
                engine_results[engine_name] = result
                failed_engines += 1
                
        except Exception as e:
            print(f"  ❌ {engine_name}: Exception - {str(e)[:60]}...")
            engine_results[engine_name] = {"error": str(e)}
            failed_engines += 1
    
    print(f"\n📊 ENGINE SUMMARY:")
    print(f"  ✅ Working: {working_engines}")
    print(f"  ❌ Failed: {failed_engines}")
    print(f"  📈 Success Rate: {working_engines/(working_engines+failed_engines)*100:.1f}%")
    
    # Test complete MPE analysis
    print(f"\n🚀 COMPLETE MPE ANALYSIS:")
    print("=" * 40)
    
    try:
        # Generate complete market pulse
        full_result = await synthesizer.generate_complete_market_pulse(symbols, start_date, end_date)
        
        # Extract MPI data
        mpi_data = full_result.get('market_pulse_index', {})
        
        print(f"📊 FINAL MPI RESULTS:")
        print(f"  📈 MPI Score: {mpi_data.get('mpi_score', 'N/A')}")
        print(f"  📊 Raw MPI: {mpi_data.get('raw_mpi', 'N/A')}")
        print(f"  🎯 Confidence: {mpi_data.get('confidence', 'N/A')}")
        print(f"  📋 Modules Contributing: {mpi_data.get('modules_contributing', 'N/A')}")
        print(f"  💡 Interpretation: {mpi_data.get('interpretation', 'N/A')}")
        
        # Show component breakdown
        component_breakdown = mpi_data.get('component_breakdown', {})
        if component_breakdown:
            print(f"\n🔍 COMPONENT BREAKDOWN:")
            for engine_name, component_data in component_breakdown.items():
                print(f"  • {engine_name}: Score={component_data.get('raw_score', 'N/A'):.3f}, Weight={component_data.get('weight', 'N/A')}")
        else:
            print(f"\n⚠️ No component breakdown (modules_contributing = 0)")
        
        # Market regime
        regime_data = full_result.get('market_regime', {})
        print(f"\n🎭 MARKET REGIME:")
        print(f"  Regime: {regime_data.get('regime', 'N/A')}")
        print(f"  Confidence: {regime_data.get('confidence', 'N/A')}")
        
        # Signals
        signals = full_result.get('comprehensive_signals', [])
        print(f"\n🚨 GENERATED SIGNALS:")
        if signals:
            for i, signal in enumerate(signals[:3], 1):  # Show first 3 signals
                print(f"  {i}. {signal}")
        else:
            print("  No signals generated")
            
    except Exception as e:
        print(f"❌ Complete MPE analysis failed: {e}")
    
    print(f"\n" + "=" * 70)
    print("🎉 PHASE 1 FINAL TEST COMPLETED")
    print("=" * 70)
    
    return {
        'working_engines': working_engines,
        'failed_engines': failed_engines,
        'success_rate': working_engines/(working_engines+failed_engines)*100,
        'engine_results': engine_results
    }

if __name__ == "__main__":
    asyncio.run(final_phase1_test())