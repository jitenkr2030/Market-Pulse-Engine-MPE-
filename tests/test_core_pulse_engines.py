#!/usr/bin/env python3
"""
Test script for Phase 2: Core Pulse Engine Fixes
Tests the 5 core pulse engines that were showing "No compatible analysis method found" errors.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the workspace to Python path
sys.path.insert(0, '/workspace')

from mpe.services.market_pulse_synthesizer import CompleteMarketPulseSynthesizer
from mpe.services.liquidity_pulse import LiquidityPulse
from mpe.services.correlation_pulse import CorrelationPulse
from mpe.services.flow_pulse import FlowPulse
from mpe.services.risk_pulse import RiskPulse
from mpe.services.momentum_pulse import MomentumPulse


async def test_core_pulse_engine(engine_name: str, engine):
    """Test a single core pulse engine"""
    print(f"\n=== Testing {engine_name} ===")
    
    try:
        # Initialize the engine
        await engine.initialize()
        print(f"✅ {engine_name} initialized successfully")
        
        # Test the specific pulse method
        pulse_method = getattr(engine, f'get_{engine_name.replace("_pulse", "")}_pulse')
        print(f"✅ Found {pulse_method.__name__} method")
        
        # Call the pulse method
        result = await pulse_method()
        
        if result and isinstance(result, dict):
            print(f"✅ {engine_name} returned valid result")
            print(f"   Result type: {type(result)}")
            
            # Check for key indicators
            if 'overall_pulse' in result:
                print(f"   Overall Pulse: {result['overall_pulse']}")
            elif 'pulse_score' in result:
                print(f"   Pulse Score: {result['pulse_score']}")
            elif 'score' in result:
                print(f"   Score: {result['score']}")
            elif 'signal_strength' in result:
                print(f"   Signal Strength: {result['signal_strength']}")
            
            return True
        else:
            print(f"❌ {engine_name} returned invalid result: {result}")
            return False
            
    except Exception as e:
        print(f"❌ {engine_name} failed: {str(e)}")
        return False


async def test_core_engines_in_synthesizer():
    """Test all core pulse engines through the synthesizer"""
    print("\n=== Testing Core Pulse Engines in Synthesizer ===")
    
    # Create synthesizer
    synthesizer = CompleteMarketPulseSynthesizer()
    
    # Initialize
    await synthesizer.initialize()
    print("✅ Synthesizer initialized")
    
    # Set test parameters
    symbols = ('SPY', 'QQQ', 'IWM')
    start_date = "2024-06-01"
    end_date = "2024-12-01"
    
    # Get available engines
    available_engines = await synthesizer.get_available_modules()
    print(f"\nAvailable engines: {list(available_engines.keys())}")
    
    # Test each core pulse engine through synthesizer
    core_engines = ['liquidity_pulse', 'correlation_pulse', 'flow_pulse', 'risk_pulse', 'momentum_pulse']
    
    results = {}
    
    for engine_name in core_engines:
        if engine_name in available_engines:
            print(f"\n--- Testing {engine_name} through synthesizer ---")
            
            try:
                # Get the specific engine
                engine = available_engines[engine_name]
                
                # Test using _safe_engine_call
                result = await synthesizer._safe_engine_call(engine_name, engine, symbols, start_date, end_date)
                
                if "error" not in result:
                    print(f"✅ {engine_name} working through synthesizer!")
                    results[engine_name] = True
                else:
                    print(f"❌ {engine_name} failed: {result.get('error', 'Unknown error')}")
                    results[engine_name] = False
                    
            except Exception as e:
                print(f"❌ {engine_name} crashed: {str(e)}")
                results[engine_name] = False
        else:
            print(f"⚠️  {engine_name} not available in synthesizer")
            results[engine_name] = False
    
    return results


async def main():
    """Main test function"""
    print("=== Phase 2: Core Pulse Engine Testing ===")
    
    # Test 1: Direct engine testing
    print("\n### Test 1: Direct Core Pulse Engine Testing ###")
    
    engines_to_test = [
        ('liquidity_pulse', LiquidityPulse()),
        ('correlation_pulse', CorrelationPulse()),
        ('flow_pulse', FlowPulse()),
        ('risk_pulse', RiskPulse()),
        ('momentum_pulse', MomentumPulse())
    ]
    
    direct_results = {}
    for engine_name, engine in engines_to_test:
        direct_results[engine_name] = await test_core_pulse_engine(engine_name, engine)
    
    # Test 2: Synthesizer integration testing
    print("\n\n### Test 2: Synthesizer Integration Testing ###")
    synth_results = await test_core_engines_in_synthesizer()
    
    # Summary
    print("\n\n=== PHASE 2 TEST SUMMARY ===")
    print(f"Direct Engine Tests:")
    for engine, status in direct_results.items():
        status_str = "✅ WORKING" if status else "❌ FAILED"
        print(f"  {engine}: {status_str}")
    
    print(f"\nSynthesizer Integration Tests:")
    for engine, status in synth_results.items():
        status_str = "✅ WORKING" if status else "❌ FAILED"
        print(f"  {engine}: {status_str}")
    
    # Count successes
    working_direct = sum(1 for result in direct_results.values() if result)
    working_synth = sum(1 for result in synth_results.values() if result)
    
    print(f"\nDirect Engine Success: {working_direct}/{len(direct_results)} ({working_direct/len(direct_results)*100:.1f}%)")
    print(f"Synthesizer Success: {working_synth}/{len(synth_results)} ({working_synth/len(synth_results)*100:.1f}%)")
    
    if working_direct == len(direct_results) and working_synth == len(synth_results):
        print("\n🎉 PHASE 2 SUCCESS: All core pulse engines are now working!")
        return True
    else:
        print(f"\n⚠️  Phase 2 Partial Success: {working_direct + working_synth}/{len(direct_results) + len(synth_results)} engines working")
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)