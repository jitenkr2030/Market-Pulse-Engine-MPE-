"""
Market Pulse Engine (MPE) - Complete 30-Module Production System

The comprehensive Real-Time Global Market Awareness & Pulse Forecasting System
integrating all 30 completed MPE modules for complete market intelligence.

Author: MiniMax Agent
Date: December 2025
Version: 2.0.0
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import List, Dict, Any, Optional
import logging
import asyncio
from datetime import datetime
import os

# Import the complete synthesizer that integrates all 30 modules
from services.market_pulse_synthesizer import CompleteMarketPulseSynthesizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for engine management
connection_manager = WebSocketConnectionManager()
synthesizer = CompleteMarketPulseSynthesizer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Complete MPE System (30 modules)...")
    
    try:
        # Get synthesizer status
        status = await synthesizer.get_status()
        logger.info(f"Synthesizer status: {status}")
        
        logger.info("Complete MPE System started successfully")
    except Exception as e:
        logger.error(f"Error starting MPE system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Complete MPE System...")
    logger.info("Complete MPE System shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Market Pulse Engine (MPE) - Complete System",
    description="The Real-Time Global Market Awareness & Pulse Forecasting System with all 30 modules",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket Connection Manager
class WebSocketConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.user_connections[user_id] = websocket
        logger.info(f"User {user_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id in self.user_connections:
            del self.user_connections[user_id]
        logger.info(f"User {user_id} disconnected. Remaining connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected users"""
        if self.active_connections:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            
            # Clean up disconnected clients
            for conn in disconnected:
                self.active_connections.remove(conn)
    
    async def send_personal_message(self, message: Dict[str, Any], user_id: str):
        """Send message to specific user"""
        if user_id in self.user_connections:
            try:
                await self.user_connections[user_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {e}")
                self.disconnect(self.user_connections[user_id], user_id)

# ===============================
# API Routes
# ===============================

@app.get("/")
async def root():
    """Health check and system information endpoint"""
    try:
        status = await synthesizer.get_status()
        return {
            "message": "Market Pulse Engine (MPE) - Complete 30-Module System",
            "version": "2.0.0",
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "modules_initialized": status.get("engines_initialized", 0),
            "modules_expected": 30,
            "system_info": {
                "name": status.get("name"),
                "last_update": status.get("last_update"),
                "engines_ready": status.get("engines_ready", [])
            }
        }
    except Exception as e:
        return {
            "message": "Market Pulse Engine (MPE)",
            "version": "2.0.0",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/v1/status")
async def get_comprehensive_status():
    """Get comprehensive system status including all modules"""
    try:
        status = await synthesizer.get_status()
        
        return {
            "system": {
                "status": "operational",
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat(),
                "uptime": "active"
            },
            "synthesizer": status,
            "real_time": {
                "websockets_connected": len(connection_manager.active_connections),
                "active_users": len(connection_manager.user_connections)
            },
            "modules_summary": {
                "total_modules": 30,
                "initialized_modules": status.get("engines_initialized", 0),
                "initialization_rate": f"{(status.get('engines_initialized', 0) / 30 * 100):.1f}%"
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "system": {"status": "error", "error": str(e)},
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/v1/market-pulse")
async def get_complete_market_pulse(
    symbols: Optional[List[str]] = Query(None, description="List of symbols to analyze"),
    start_date: str = Query("2023-01-01", description="Start date for analysis (YYYY-MM-DD)"),
    end_date: str = Query("2024-12-01", description="End date for analysis (YYYY-MM-DD)")
):
    """
    Get complete market pulse analysis from all 30 modules
    
    This endpoint provides comprehensive market intelligence including:
    - Market Pulse Index (MPI) from all modules
    - Market regime analysis and forecasting
    - Cross-module correlation analysis
    - Comprehensive trading signals
    - Risk assessment and recommendations
    """
    try:
        # Default symbols if none provided
        if not symbols:
            symbols = ["SPY", "QQQ", "IWM", "DXY", "TLT", "GLD", "BTCUSD"]
        
        # Convert to tuple for caching
        symbols_tuple = tuple(symbols)
        
        # Generate complete market pulse
        result = await synthesizer.generate_complete_market_pulse(
            symbols=symbols_tuple,
            start_date=start_date,
            end_date=end_date
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in market pulse analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Market pulse analysis failed: {str(e)}")

@app.get("/api/v1/mpi/{mpi_score:.2f}")
async def interpret_mpi_score(mpi_score: float):
    """
    Interpret Market Pulse Index score with detailed analysis
    
    Args:
        mpi_score: MPI score between 0.0 and 1.0
    
    Returns:
        Detailed interpretation of the MPI score
    """
    if not 0.0 <= mpi_score <= 1.0:
        raise HTTPException(status_code=400, detail="MPI score must be between 0.0 and 1.0")
    
    interpretation = _get_mpi_interpretation(mpi_score)
    
    return {
        "mpi_score": mpi_score,
        "interpretation": interpretation,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/regime/{regime}")
async def get_regime_analysis(regime: str):
    """
    Get detailed analysis for a specific market regime
    
    Args:
        regime: Market regime (BULL_MARKET, BEAR_MARKET, CRISIS, EUPHORIA, etc.)
    """
    regime_info = _get_regime_analysis(regime)
    
    if not regime_info:
        raise HTTPException(status_code=404, detail=f"Unknown regime: {regime}")
    
    return {
        "regime": regime,
        "analysis": regime_info,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/signals")
async def get_signal_recommendations(
    mpi_score: float = Query(..., description="Market Pulse Index score"),
    regime: str = Query(..., description="Current market regime"),
    conviction: str = Query("MEDIUM", description="Signal conviction level")
):
    """
    Get trading signal recommendations based on MPI and regime
    
    Args:
        mpi_score: Current MPI score
        regime: Current market regime
        conviction: Signal conviction level (LOW, MEDIUM, HIGH)
    """
    if not 0.0 <= mpi_score <= 1.0:
        raise HTTPException(status_code=400, detail="MPI score must be between 0.0 and 1.0")
    
    recommendations = _generate_signal_recommendations(mpi_score, regime, conviction)
    
    return {
        "mpi_score": mpi_score,
        "regime": regime,
        "conviction": conviction,
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat()
    }

# ===============================
# WebSocket Endpoints
# ===============================

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time market data streaming"""
    await connection_manager.connect(websocket, user_id)
    try:
        # Send welcome message
        await connection_manager.send_personal_message({
            "action": "welcome",
            "message": "Connected to MPE Complete System",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }, user_id)
        
        while True:
            # Handle incoming messages
            data = await websocket.receive_json()
            await handle_websocket_message(websocket, user_id, data)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, user_id)
        logger.info(f"User {user_id} disconnected via WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        connection_manager.disconnect(websocket, user_id)

async def handle_websocket_message(websocket: WebSocket, user_id: str, data: Dict[str, Any]):
    """Handle incoming WebSocket messages"""
    try:
        action = data.get("action")
        
        if action == "subscribe":
            await handle_subscription(websocket, user_id, data)
        elif action == "unsubscribe":
            await handle_unsubscription(websocket, user_id, data)
        elif action == "get_market_pulse":
            await handle_market_pulse_request(websocket, user_id, data)
        elif action == "ping":
            await connection_manager.send_personal_message({
                "action": "pong",
                "timestamp": datetime.now().isoformat()
            }, user_id)
        else:
            await connection_manager.send_personal_message({
                "action": "error",
                "message": f"Unknown action: {action}",
                "timestamp": datetime.now().isoformat()
            }, user_id)
            
    except Exception as e:
        logger.error(f"Error handling WebSocket message from {user_id}: {e}")
        await connection_manager.send_personal_message({
            "action": "error",
            "message": f"Message processing failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }, user_id)

async def handle_subscription(websocket: WebSocket, user_id: str, data: Dict[str, Any]):
    """Handle subscription to market data feeds"""
    symbols = data.get("symbols", ["SPY", "QQQ", "IWM"])
    frequency = data.get("frequency", 30)  # seconds
    
    await connection_manager.send_personal_message({
        "action": "subscription_confirmed",
        "symbols": symbols,
        "frequency": frequency,
        "message": f"Subscribed to market pulse updates for {symbols}",
        "timestamp": datetime.now().isoformat()
    }, user_id)
    
    # Start real-time updates for this user
    asyncio.create_task(stream_market_pulse_to_user(user_id, symbols, frequency))

async def handle_unsubscription(websocket: WebSocket, user_id: str, data: Dict[str, Any]):
    """Handle unsubscription from market data feeds"""
    await connection_manager.send_personal_message({
        "action": "unsubscription_confirmed",
        "message": "Unsubscribed from market pulse updates",
        "timestamp": datetime.now().isoformat()
    }, user_id)

async def handle_market_pulse_request(websocket: WebSocket, user_id: str, data: Dict[str, Any]):
    """Handle one-time market pulse request"""
    symbols = data.get("symbols", ["SPY", "QQQ", "IWM"])
    
    try:
        result = await synthesizer.generate_complete_market_pulse(
            symbols=tuple(symbols),
            start_date="2023-01-01",
            end_date="2024-12-01"
        )
        
        await connection_manager.send_personal_message({
            "action": "market_pulse_data",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }, user_id)
        
    except Exception as e:
        await connection_manager.send_personal_message({
            "action": "error",
            "message": f"Market pulse request failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }, user_id)

async def stream_market_pulse_to_user(user_id: str, symbols: List[str], frequency: int):
    """Stream market pulse updates to a specific user"""
    while user_id in connection_manager.user_connections:
        try:
            result = await synthesizer.generate_complete_market_pulse(
                symbols=tuple(symbols),
                start_date="2023-01-01", 
                end_date="2024-12-01"
            )
            
            await connection_manager.send_personal_message({
                "action": "market_pulse_update",
                "data": result,
                "frequency": frequency,
                "timestamp": datetime.now().isoformat()
            }, user_id)
            
            await asyncio.sleep(frequency)
            
        except Exception as e:
            logger.error(f"Error streaming market pulse to {user_id}: {e}")
            await asyncio.sleep(frequency)

# ===============================
# Utility Functions
# ===============================

def _get_mpi_interpretation(mpi_score: float) -> Dict[str, Any]:
    """Get detailed MPI interpretation"""
    if mpi_score >= 0.8:
        return {
            "category": "EXTREME_BULLISH",
            "interpretation": "Strong bullish conditions with high risk of euphoria and potential reversal",
            "action": "Consider taking profits or reducing position sizes",
            "risk_level": "HIGH",
            "watch_for": "Signs of capitulation by bears, extreme optimism, volume spikes"
        }
    elif mpi_score >= 0.65:
        return {
            "category": "BULLISH",
            "interpretation": "Positive market conditions with good momentum and liquidity",
            "action": "Maintain bullish positioning with normal risk management",
            "risk_level": "MODERATE",
            "watch_for": "Continuation of trends, earnings surprises, macro developments"
        }
    elif mpi_score >= 0.35:
        return {
            "category": "NEUTRAL",
            "interpretation": "Mixed signals with no clear directional bias",
            "action": "Maintain current positions, await clearer signals",
            "risk_level": "MODERATE",
            "watch_for": "Breakout or breakdown signals, regime changes"
        }
    elif mpi_score >= 0.2:
        return {
            "category": "BEARISH",
            "interpretation": "Caution advised as negative signals are emerging",
            "action": "Consider defensive positioning and risk reduction",
            "risk_level": "HIGH",
            "watch_for": "Further weakness, capitulation signals, crisis indicators"
        }
    else:
        return {
            "category": "EXTREME_BEARISH",
            "interpretation": "Crisis conditions - defensive positioning highly recommended",
            "action": "Focus on capital preservation, look for oversold opportunities",
            "risk_level": "EXTREME",
            "watch_for": "Bottom formation signals, capitulation, extreme fear"
        }

def _get_regime_analysis(regime: str) -> Optional[Dict[str, Any]]:
    """Get detailed regime analysis"""
    regime_info = {
        "BULL_MARKET": {
            "description": "Sustained upward price movement with positive sentiment",
            "characteristics": ["Rising prices", "High confidence", "Strong inflows", "Low volatility"],
            "strategies": ["Buy on dips", "Momentum strategies", "Risk-on positioning"],
            "risks": ["Complacency", "Overvaluation", "Sudden reversals"]
        },
        "BEAR_MARKET": {
            "description": "Sustained downward price movement with negative sentiment",
            "characteristics": ["Falling prices", "Risk aversion", "Outflows", "High volatility"],
            "strategies": ["Defensive positioning", "Short strategies", "Safe havens"],
            "risks": ["Continued decline", "Liquidity issues", "Systemic stress"]
        },
        "CRISIS": {
            "description": "Extreme market stress with severe price declines",
            "characteristics": ["Panic selling", "Liquidity dry-up", "Correlation spike", "Flight to safety"],
            "strategies": ["Capital preservation", "Extreme risk management", "Quality focus"],
            "risks": ["Systemic failure", "Contagion", "Extended bear market"]
        },
        "EUPHORIA": {
            "description": "Extreme optimism with potential for sharp reversals",
            "characteristics": ["Excessive optimism", "High speculation", "Low volatility", "Momentum divergence"],
            "strategies": ["Profit taking", "Hedging", "Contrarian approaches"],
            "risks": ["Sharp reversal", "Bubble burst", "Regime change"]
        },
        "NEUTRAL": {
            "description": "Balanced market conditions with mixed signals",
            "characteristics": ["Sideways movement", "Balanced sentiment", "Normal volatility", "Uncertain direction"],
            "strategies": ["Range trading", "Option strategies", "Selective positioning"],
            "risks": ["False breakouts", "Whipsaws", "Regime transitions"]
        }
    }
    
    return regime_info.get(regime.upper())

def _generate_signal_recommendations(mpi_score: float, regime: str, conviction: str) -> Dict[str, Any]:
    """Generate trading signal recommendations"""
    
    # Base recommendations
    if mpi_score >= 0.75:
        primary_action = "CONSIDER_PROFIT_TAKING"
        position_sizing = "REDUCE"
    elif mpi_score >= 0.6:
        primary_action = "MAINTAIN_BULLISH"
        position_sizing = "NORMAL"
    elif mpi_score <= 0.25:
        primary_action = "LOOK_FOR_BUY_OPPORTUNITIES"
        position_sizing = "INCREASE"
    elif mpi_score <= 0.4:
        primary_action = "DEFENSIVE_POSITIONING"
        position_sizing = "REDUCE"
    else:
        primary_action = "MAINTAIN_NEUTRAL"
        position_sizing = "NORMAL"
    
    # Adjust for regime
    regime_adjustments = {
        "CRISIS": {
            "primary_action": "CAPITAL_PRESERVATION",
            "position_sizing": "MINIMAL",
            "additional_actions": ["Focus on quality assets", "Reduce leverage", "Increase cash"]
        },
        "EUPHORIA": {
            "primary_action": "PREPARE_FOR_REVERSAL",
            "position_sizing": "REDUCE",
            "additional_actions": ["Take profits", "Increase hedging", "Watch for signals"]
        }
    }
    
    if regime.upper() in regime_adjustments:
        adjustments = regime_adjustments[regime.upper()]
        primary_action = adjustments["primary_action"]
        position_sizing = adjustments["position_sizing"]
    
    return {
        "primary_action": primary_action,
        "position_sizing": position_sizing,
        "conviction_level": conviction,
        "risk_management": {
            "stop_loss_tightening": "YES" if conviction == "HIGH" else "NO",
            "hedging_recommendation": "YES" if abs(mpi_score - 0.5) > 0.3 else "MONITOR"
        },
        "time_horizons": {
            "intraday": "TRADE_WITH_TREND",
            "short_term": primary_action,
            "medium_term": "MONITOR_REGIME_CHANGE",
            "long_term": "STRATEGIC_POSITIONING"
        }
    }

# ===============================
# Background Tasks
# ===============================

@app.on_event("startup")
async def startup_event():
    """Start background tasks when application starts"""
    logger.info("Starting background tasks...")
    # Additional startup initialization can be added here

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )