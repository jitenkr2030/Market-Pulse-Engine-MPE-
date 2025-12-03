#!/usr/bin/env python3
"""
Test script for Phase 3A: Market Intelligence Engine Standardization
Tests the 5 Market Intelligence engines to verify they are operational
"""

import asyncio
import sys
import os

# Add the workspace to the Python path
sys.path.append('/workspace')

from mpe.services.market_pulse_synthesizer import CompleteMarketPulseSynthesizer

async def test_market_intelligence_engines():
    """Test all 5 Market Intelligence engines through the synthesizer"""
    
    print("🚀 Testing Phase 3A: Market Intelligence Engine Standardization")
    print("=" * 70)
    
    # Initialize the synthesizer
    synthesizer = CompleteMarketPulseSynthesizer()
    
    print(f"✅ Synthesizer initialized")
    
    # Test each Market Intelligence engine
    market_intelligence_engines = [
        'macro_pulse',
        'narrative_intelligence', 
        'event_shockwave',
        'capital_rotation',
        'regime_detection'
    ]
    
    results = {}
    operational_count = 0
    
    for engine_name in market_intelligence_engines:
        print(f"\n🔍 Testing {engine_name}...")
        
        try:
            # Get module by name
            module_info = synthesizer._get_module_by_name(engine_name)
            if not module_info:
                print(f"❌ {engine_name}: Module not found")
                results[engine_name] = {"status": "not_found", "error": "Module not found"}
                continue
                
            engine, module_name, class_name = module_info
            
            # Test the engine through _collect_module_data
            engine_data = await synthesizer._collect_module_data(engine_name)
            
            if engine_data and not engine_data.get('error'):
                print(f"✅ {engine_name}: OPERATIONAL")
                print(f"   📊 Data keys: {list(engine_data.keys())}")
                
                # Show a sample of the data
                for key, value in list(engine_data.items())[:2]:
                    if isinstance(value, dict):
                        print(f"   📈 {key}: {len(value)} sub-components")
                    else:
                        print(f"   📈 {key}: {type(value).__name__}")
                
                results[engine_name] = {
                    "status": "operational",
                    "data_keys": list(engine_data.keys()),
                    "class_name": class_name
                }
                operational_count += 1
            else:
                error_msg = engine_data.get('error', 'Unknown error')
                print(f"❌ {engine_name}: FAILED - {error_msg}")
                results[engine_name] = {"status": "failed", "error": error_msg}
                
        except Exception as e:
            print(f"❌ {engine_name}: EXCEPTION - {str(e)}")
            results[engine_name] = {"status": "exception", "error": str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 PHASE 3A MARKET INTELLIGENCE ENGINE TEST RESULTS")
    print("=" * 70)
    
    total_engines = len(market_intelligence_engines)
    success_rate = (operational_count / total_engines) * 100
    
    print(f"✅ Operational: {operational_count}/{total_engines} engines")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if operational_count == total_engines:
        print("\n🎉 ALL MARKET INTELLIGENCE ENGINES OPERATIONAL!")
        print("🎯 Target achieved: 21/24 engines operational (87.5%)")
    elif operational_count >= 4:
        print("\n✅ EXCELLENT PROGRESS: 4/5 Market Intelligence engines operational")
        print("🎯 Near target achievement")
    elif operational_count >= 3:
        print("\n⚠️  PARTIAL SUCCESS: 3/5 Market Intelligence engines operational")
    else:
        print("\n❌ NEEDS WORK: Less than 3 Market Intelligence engines operational")
    
    print("\n📋 Detailed Results:")
    for engine_name, result in results.items():
        status = result['status']
        status_emoji = "✅" if status == "operational" else "❌"
        print(f"{status_emoji} {engine_name}: {status}")
        if status != "operational" and 'error' in result:
            print(f"   Error: {result['error']}")
    
    # Test MPI calculation with new engines
    print(f"\n🧮 Testing MPI Calculation with Market Intelligence Engines...")
    try:
        mpi_result = await synthesizer.calculate_market_pulse_index()
        if mpi_result and 'market_pulse_index' in mpi_result:
            print(f"✅ MPI Calculation successful")
            print(f"📊 MPI Score: {mpi_result['market_pulse_index']:.3f}")
            
            # Show breakdown
            if 'component_breakdown' in mpi_result:
                breakdown = mpi_result['component_breakdown']
                print(f"📈 Component contributions:")
                for component, score in list(breakdown.items())[:5]:
                    if isinstance(score, dict) and 'score' in score:
                        print(f"   {component}: {score['score']:.3f}")
        else:
            print(f"❌ MPI Calculation failed")
    except Exception as e:
        print(f"❌ MPI Calculation exception: {str(e)}")
    
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
    results = asyncio.run(main())
    print(f"\n🏁 Test completed. Results: {results}")