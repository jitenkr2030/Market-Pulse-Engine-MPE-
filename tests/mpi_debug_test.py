#!/usr/bin/env python3
"""
Debug MPI calculation to see why modules_contributing is 0
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the MPE services directory to Python path
sys.path.insert(0, '/workspace/mpe/services')

from market_pulse_synthesizer import CompleteMarketPulseSynthesizer

async def debug_mpi_calculation():
    """Debug why engines are not contributing to MPI"""
    
    print("🔍 DEBUGGING MPI CALCULATION")
    print("=" * 50)
    
    # Initialize MPE System
    synthesizer = CompleteMarketPulseSynthesizer()
    
    # Set analysis parameters
    symbols = ('SPY', 'QQQ', 'IWM')
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # Shorter period
    
    print(f"📅 Analysis Period: {start_date} to {end_date}")
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"⚙️ Engines Available: {len(synthesizer.engines)}")
    
    # Step 1: Collect data from engines
    print(f"\n🔧 STEP 1: Collecting data from all engines...")
    
    try:
        # Call the internal method to see what we get
        all_engine_data = await synthesizer._collect_all_module_data(symbols, start_date, end_date)
        
        print(f"📊 Engine Data Collected:")
        working_engines = 0
        failed_engines = 0
        
        for engine_name, engine_data in all_engine_data.items():
            if isinstance(engine_data, dict) and "error" not in engine_data:
                print(f"  ✅ {engine_name}: {len(engine_data) if isinstance(engine_data, dict) else 'N/A'} data points")
                working_engines += 1
                
                # Show first few keys
                if isinstance(engine_data, dict):
                    keys = list(engine_data.keys())[:3]
                    print(f"    • Keys: {keys}")
            else:
                print(f"  ❌ {engine_name}: {str(engine_data)[:80]}...")
                failed_engines += 1
        
        print(f"\n📈 Engine Data Summary:")
        print(f"  Working: {working_engines}")
        print(f"  Failed: {failed_engines}")
        print(f"  Total: {len(all_engine_data)}")
        
        # Step 2: Check MPI calculation
        print(f"\n🔧 STEP 2: Testing MPI calculation...")
        
        # Show the module weights
        print(f"📊 Module Weights:")
        for engine_name, weight in synthesizer.module_weights.items():
            if engine_name in all_engine_data and "error" not in all_engine_data[engine_name]:
                print(f"  ✅ {engine_name}: {weight}")
            else:
                print(f"  ❌ {engine_name}: {weight} (engine failed)")
        
        # Test MPI calculation directly
        try:
            mpi_data = await synthesizer._calculate_market_pulse_index(all_engine_data)
            
            print(f"\n📊 MPI Calculation Result:")
            print(f"  Raw MPI: {mpi_data.get('raw_mpi', 'N/A')}")
            print(f"  Final MPI: {mpi_data.get('mpi_score', 'N/A')}")
            print(f"  Modules Contributing: {mpi_data.get('modules_contributing', 'N/A')}")
            print(f"  Total Weight: {mpi_data.get('total_weight', 'N/A')}")
            print(f"  Confidence: {mpi_data.get('confidence', 'N/A')}")
            
            component_breakdown = mpi_data.get('component_breakdown', {})
            if component_breakdown:
                print(f"\n🔍 Component Breakdown:")
                for engine_name, component_data in component_breakdown.items():
                    print(f"  • {engine_name}: {component_data}")
            else:
                print(f"\n⚠️ No component breakdown - this is why modules_contributing is 0")
                
        except Exception as e:
            print(f"❌ MPI calculation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 3: Test signal extraction
        print(f"\n🔧 STEP 3: Testing signal extraction...")
        
        # Test signal extraction for a working engine
        working_engines_list = [name for name, data in all_engine_data.items() 
                               if isinstance(data, dict) and "error" not in data]
        
        if working_engines_list:
            test_engine = working_engines_list[0]
            engine_data = all_engine_data[test_engine]
            
            print(f"🧪 Testing signal extraction for {test_engine}:")
            print(f"  Engine data type: {type(engine_data)}")
            print(f"  Engine data keys: {list(engine_data.keys()) if isinstance(engine_data, dict) else 'N/A'}")
            
            # Test the extraction method
            try:
                signal = synthesizer._extract_primary_signal(test_engine, engine_data)
                print(f"  Extracted signal: {signal}")
                
                confidence = synthesizer._extract_confidence(test_engine, engine_data)
                print(f"  Extracted confidence: {confidence}")
                
            except Exception as e:
                print(f"  ❌ Signal extraction failed: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_mpi_calculation())