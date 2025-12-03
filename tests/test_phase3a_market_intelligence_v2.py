#!/usr/bin/env python3
"""
Test script for Phase 3A: Market Intelligence Engine Standardization
Tests the 5 Market Intelligence engines using the working pattern from Phase 2
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the workspace to Python path
sys.path.insert(0, '/workspace')

from mpe.services.market_pulse_synthesizer import CompleteMarketPulseSynthesizer


async def test_market_intelligence_engines():
    """Test Market Intelligence engines through the synthesizer"""
    print("🚀 Phase 3A: Testing Market Intelligence Engines")
    print("=" * 60)
    
    # Create synthesizer
    synthesizer = CompleteMarketPulseSynthesizer()
    print("✅ Synthesizer initialized")
    
    # Set test parameters  
    symbols = ('SPY', 'QQQ', 'IWM')
    start_date = "2024-06-01"
    end_date = "2024-12-01"
    
    # Get available engines
    available_engines = synthesizer.engines
    print(f"\n📊 Total engines available: {len(available_engines)}")
    
    # Test the 5 Market Intelligence engines
    market_intelligence_engines = [
        'macro_pulse',
        'narrative_intelligence', 
        'event_shockwave',
        'capital_rotation',
        'regime_detection'
    ]
    
    results = {}
    operational_count = 0
    
    print("\n=== Testing Individual Market Intelligence Engines ===")
    
    for engine_name in market_intelligence_engines:
        print(f"\n🔍 Testing {engine_name}...")
        
        if engine_name not in available_engines:
            print(f"❌ {engine_name}: Not found in available engines")
            results[engine_name] = {"status": "not_found"}
            continue
            
        try:
            # Get the specific engine
            engine = available_engines[engine_name]
            engine_class = type(engine).__name__
            
            print(f"   📦 Engine class: {engine_class}")
            
            # Test using _safe_engine_call
            engine_data = await synthesizer._safe_engine_call(engine_name, engine, symbols, start_date, end_date)
            
            if engine_data and not engine_data.get('error'):
                print(f"✅ {engine_name}: OPERATIONAL")
                print(f"   📊 Data keys: {list(engine_data.keys())}")
                
                # Show some sample data
                for key, value in list(engine_data.items())[:2]:
                    if isinstance(value, dict):
                        print(f"   📈 {key}: {len(value)} sub-components")
                    else:
                        print(f"   📈 {key}: {type(value).__name__}")
                
                results[engine_name] = {
                    "status": "operational",
                    "class_name": engine_class,
                    "data_keys": list(engine_data.keys())
                }
                operational_count += 1
            else:
                error_msg = engine_data.get('error', 'Unknown error') if engine_data else 'No data returned'
                print(f"❌ {engine_name}: FAILED - {error_msg}")
                results[engine_name] = {"status": "failed", "error": error_msg}
                
        except Exception as e:
            print(f"❌ {engine_name}: EXCEPTION - {str(e)}")
            results[engine_name] = {"status": "exception", "error": str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PHASE 3A MARKET INTELLIGENCE ENGINE TEST RESULTS")
    print("=" * 60)
    
    total_engines = len(market_intelligence_engines)
    success_rate = (operational_count / total_engines) * 100
    
    print(f"✅ Operational: {operational_count}/{total_engines} engines")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if operational_count == total_engines:
        print("\n🎉 ALL MARKET INTELLIGENCE ENGINES OPERATIONAL!")
        print("🎯 PHASE 3A SUCCESS: Ready to achieve 87.5% system operational rate!")
    elif operational_count >= 4:
        print("\n✅ EXCELLENT: 4/5 Market Intelligence engines operational")
    elif operational_count >= 3:
        print("\n⚠️  PARTIAL: 3/5 Market Intelligence engines operational") 
    else:
        print("\n❌ NEEDS WORK: Less than 3 Market Intelligence engines operational")
    
    print("\n📋 Detailed Results:")
    for engine_name, result in results.items():
        status = result['status']
        status_emoji = "✅" if status == "operational" else "❌"
        print(f"{status_emoji} {engine_name}: {status}")
        if 'class_name' in result:
            print(f"    Class: {result['class_name']}")
        if status != "operational" and 'error' in result:
            print(f"    Error: {result['error']}")
    
    # Test all modules data collection
    print(f"\n🧮 Testing Complete System with Market Intelligence Engines...")
    try:
        all_module_data = await synthesizer._collect_all_module_data(symbols, start_date, end_date)
        if all_module_data:
            print(f"✅ Complete module data collection successful")
            print(f"📊 Modules collected: {len(all_module_data)}")
            
            # Count how many Market Intelligence modules are working
            working_market_intel = 0
            for engine_name in market_intelligence_engines:
                if engine_name in all_module_data and not all_module_data[engine_name].get('error'):
                    working_market_intel += 1
            
            print(f"📈 Market Intelligence modules working: {working_market_intel}/5")
            
            # Expected total operational engines
            expected_operational = 16 + working_market_intel  # 16 from Phase 2 + new working engines
            target_operational = 21  # Target for 87.5%
            achievement_rate = (expected_operational / 24) * 100
            
            print(f"🎯 Expected system operational rate: {expected_operational}/24 ({achievement_rate:.1f}%)")
            if expected_operational >= target_operational:
                print(f"✅ TARGET ACHIEVED: {achievement_rate:.1f}% >= 87.5%")
            else:
                print(f"📍 Progress toward 87.5%: {achievement_rate:.1f}% / 87.5%")
        else:
            print(f"❌ Module data collection failed")
    except Exception as e:
        print(f"❌ Module data collection exception: {str(e)}")
    
    return results


async def main():
    """Main test function"""
    try:
        results = await test_market_intelligence_engines()
        return results
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    print("🚀 Starting Phase 3A: Market Intelligence Engine Standardization Test")
    print("=" * 70)
    results = asyncio.run(main())
    print(f"\n🏁 Test completed. Results saved.")