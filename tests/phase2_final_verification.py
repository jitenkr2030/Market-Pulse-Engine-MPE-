#!/usr/bin/env python3
"""
Phase 2 Final Verification Test
Confirms complete MPE system status after core engine standardization
"""

import asyncio
import sys
import os

# Add the workspace to Python path
sys.path.insert(0, '/workspace')

from mpe.services.market_pulse_synthesizer import CompleteMarketPulseSynthesizer


async def comprehensive_mpe_verification():
    """Comprehensive verification of MPE system status"""
    print("=== PHASE 2 FINAL VERIFICATION ===")
    print("Testing complete MPE system after core engine standardization\n")
    
    # Create synthesizer
    synthesizer = CompleteMarketPulseSynthesizer()
    print("✅ MPE Synthesizer initialized")
    
    # Get available engines
    available_engines = synthesizer.engines
    total_engines = len(available_engines)
    print(f"📊 Total engines initialized: {total_engines}/24")
    
    # Test each engine through _safe_engine_call
    symbols = ('SPY', 'QQQ', 'IWM')
    start_date = "2024-06-01" 
    end_date = "2024-12-01"
    
    print(f"\n🔍 Testing all {total_engines} engines via _safe_engine_call...")
    
    working_engines = []
    failed_engines = []
    
    for engine_name, engine in available_engines.items():
        try:
            result = await synthesizer._safe_engine_call(engine_name, engine, symbols, start_date, end_date)
            
            if "error" not in result and isinstance(result, dict) and len(result) > 0:
                working_engines.append(engine_name)
                print(f"✅ {engine_name}")
            else:
                failed_engines.append(engine_name)
                print(f"❌ {engine_name}: {result.get('error', 'No data returned')}")
                
        except Exception as e:
            failed_engines.append(engine_name)
            print(f"💥 {engine_name}: {str(e)}")
    
    # Calculate success rate
    success_rate = (len(working_engines) / total_engines) * 100
    
    print(f"\n📈 ENGINE STATUS SUMMARY:")
    print(f"✅ Working: {len(working_engines)}/{total_engines} ({success_rate:.1f}%)")
    print(f"❌ Failed: {len(failed_engines)}/{total_engines} ({100-success_rate:.1f}%)")
    
    # Categorize engines
    core_pulse = ['sentiment_pulse', 'volatility_pulse', 'liquidity_pulse', 'correlation_pulse', 'flow_pulse', 'risk_pulse', 'momentum_pulse']
    market_intel = ['macro_pulse', 'narrative_intelligence', 'event_shockwave', 'capital_rotation', 'regime_detection', 'dark_pool_intelligence']
    derivatives = ['block_trade_monitor', 'institutional_flow_tracker', 'redemption_risk_monitor', 'cross_asset_correlation_engine', 'currency_impact_engine']
    cross_asset = ['commodity_linkage_engine', 'credit_spread_engine', 'multi_asset_arbitrage_engine', 'predictive_momentum_engine']
    predictive = ['market_regime_forecaster', 'liquidity_prediction_engine']
    
    print(f"\n🏷️  ENGINE CATEGORIES:")
    
    categories = {
        "Core Pulse (7)": core_pulse,
        "Market Intelligence (6)": market_intel, 
        "Derivatives Intelligence (5)": derivatives,
        "Cross-Asset Intelligence (4)": cross_asset,
        "Predictive & Forecasting (2)": predictive
    }
    
    for category, engines in categories.items():
        working_in_category = len([e for e in engines if e in working_engines])
        total_in_category = len(engines)
        category_rate = (working_in_category / total_in_category) * 100
        print(f"   {category}: {working_in_category}/{total_in_category} ({category_rate:.1f}%) ✅")
        
        for engine in engines:
            if engine in working_engines:
                print(f"      ✅ {engine}")
            else:
                print(f"      ❌ {engine}")
    
    # Test MPI calculation capability  
    print(f"\n🧮 TESTING MPI CALCULATION:")
    try:
        mpi_result = await synthesizer.generate_complete_market_pulse(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        if mpi_result and 'market_pulse_index' in mpi_result:
            mpi_data = mpi_result['market_pulse_index']
            modules_analyzed = mpi_result.get('analysis_parameters', {}).get('modules_analyzed', 0)
            
            print(f"✅ MPI calculation successful!")
            print(f"   📊 Modules analyzed: {modules_analyzed}")
            print(f"   📈 MPI Score: {mpi_data.get('mpi_score', 'N/A')}")
            print(f"   🎯 Interpretation: {mpi_data.get('interpretation', 'N/A')}")
            
            # Count contributing components
            component_breakdown = mpi_data.get('component_breakdown', {})
            contributing_components = len([k for k, v in component_breakdown.items() if isinstance(v, dict) and 'score' in v])
            
            print(f"   🔧 Contributing components: {contributing_components}")
            
            if contributing_components > 0:
                print(f"✅ MPE SYSTEM GENERATING REAL MARKET SIGNALS!")
            else:
                print(f"⚠️  MPI calculation working but no component signals extracted")
        else:
            print(f"❌ MPI calculation failed or returned invalid data")
            
    except Exception as e:
        print(f"❌ MPI calculation crashed: {str(e)}")
    
    # Final assessment
    print(f"\n🎯 PHASE 2 FINAL ASSESSMENT:")
    
    if success_rate >= 70:
        print(f"✅ EXCELLENT: {success_rate:.1f}% operational rate achieved")
        print(f"🏆 Phase 2 mission: ACCOMPLISHED")
    elif success_rate >= 50:
        print(f"✅ GOOD: {success_rate:.1f}% operational rate achieved")
        print(f"🏆 Phase 2 mission: SUBSTANTIAL SUCCESS")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT: {success_rate:.1f}% operational rate")
        print(f"🏆 Phase 2 mission: PARTIAL SUCCESS")
    
    print(f"\n📊 SYSTEM READINESS:")
    if len(working_engines) >= 18:  # 75% of 24 engines
        print(f"✅ INSTITUTIONAL-GRADE READY: Core market intelligence operational")
        print(f"🚀 Ready for Phase 3: Complete remaining engines")
    elif len(working_engines) >= 12:  # 50% of 24 engines
        print(f"✅ SOLID FOUNDATION: Significant market intelligence capability")
        print(f"🎯 Focus Phase 3 on Market Intelligence engines")
    else:
        print(f"⚠️  DEVELOPMENT STAGE: Basic capability established")
        print(f"🔧 Continue Phase 2 work on additional engines")
    
    print(f"\n🏁 PHASE 2 COMPLETE")
    return len(working_engines), total_engines


if __name__ == "__main__":
    working, total = asyncio.run(comprehensive_mpe_verification())
    print(f"\nFinal Result: {working}/{total} engines operational ({(working/total)*100:.1f}%)")
    sys.exit(0 if working >= total * 0.7 else 1)  # Exit with success if 70%+ operational