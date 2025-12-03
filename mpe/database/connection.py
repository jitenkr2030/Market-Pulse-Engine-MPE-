"""
Database connection and management for Market Pulse Engine
"""

import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import os
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import redis
import json

logger = logging.getLogger(__name__)

# PostgreSQL Models
Base = declarative_base()

class MarketData(Base):
    """Market data storage model"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Float, default=0)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    data_source = Column(String(50), default="yahoo")
    created_at = Column(DateTime, default=datetime.utcnow)

class PulseData(Base):
    """Pulse engine data storage model"""
    __tablename__ = "pulse_data"
    
    id = Column(Integer, primary_key=True, index=True)
    pulse_type = Column(String(50), index=True, nullable=False)
    symbol = Column(String(20), index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    pulse_value = Column(Float, nullable=False)
    confidence = Column(Float, default=1.0)
    additional_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class MarketForecast(Base):
    """Market forecast storage model"""
    __tablename__ = "market_forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    forecast_horizon = Column(String(10), nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    forecast_value = Column(Float, nullable=False)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    model_version = Column(String(50))
    accuracy_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class SentimentData(Base):
    """Sentiment data storage model"""
    __tablename__ = "sentiment_data"
    
    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(50), index=True, nullable=False)
    source_id = Column(String(100), index=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    sentiment_label = Column(String(20))
    content = Column(Text)
    keywords = Column(JSON)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class RiskMetrics(Base):
    """Risk metrics storage model"""
    __tablename__ = "risk_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    var_1d = Column(Float)
    var_5d = Column(Float)
    volatility = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    beta = Column(Float)
    correlation_matrix = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserPreferences(Base):
    """User preferences and settings"""
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), unique=True, index=True, nullable=False)
    watchlist = Column(JSON, default=[])
    alert_thresholds = Column(JSON, default={})
    dashboard_layout = Column(JSON, default={})
    notification_settings = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SystemAlerts(Base):
    """System alerts and notifications"""
    __tablename__ = "system_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String(50), index=True, nullable=False)
    severity = Column(String(20), index=True, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    affected_symbols = Column(JSON)
    alert_data = Column(JSON)
    acknowledged = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Database connection and management class"""
    
    def __init__(self):
        self.postgres_engine = None
        self.postgres_session = None
        self.influx_client = None
        self.redis_client = None
        
    async def initialize(self):
        """Initialize database connections"""
        try:
            # PostgreSQL connection
            postgres_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/mpe")
            self.postgres_engine = create_async_engine(
                postgres_url.replace("postgresql://", "postgresql+asyncpg://"),
                echo=False,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            self.postgres_session = async_sessionmaker(
                self.postgres_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.postgres_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            # InfluxDB connection
            influx_url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
            influx_token = os.getenv("INFLUXDB_TOKEN", "your-token")
            influx_org = os.getenv("INFLUXDB_ORG", "mpe")
            
            self.influx_client = InfluxDBClient(
                url=influx_url,
                token=influx_token,
                org=influx_org
            )
            
            # Redis connection
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def get_postgres_session(self) -> AsyncSession:
        """Get PostgreSQL session"""
        async with self.postgres_session() as session:
            yield session
    
    async def write_influx_point(self, measurement: str, fields: Dict[str, Any], tags: Dict[str, str] = None):
        """Write data point to InfluxDB"""
        try:
            if not self.influx_client:
                logger.warning("InfluxDB client not initialized")
                return
                
            point = Point(measurement)
            
            # Add tags
            if tags:
                for tag_key, tag_value in tags.items():
                    point.tag(tag_key, tag_value)
            
            # Add fields
            for field_key, field_value in fields.items():
                if isinstance(field_value, (int, float)):
                    point.field(field_key, field_value)
                else:
                    point.field(field_key, str(field_value))
            
            # Add timestamp
            point.time(datetime.utcnow())
            
            write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            bucket = os.getenv("INFLUXDB_BUCKET", "market_data")
            
            write_api.write(bucket=bucket, record=point)
            
        except Exception as e:
            logger.error(f"Failed to write to InfluxDB: {e}")
    
    async def query_influx_data(self, measurement: str, time_range: str = "-1h", filters: Dict[str, str] = None):
        """Query data from InfluxDB"""
        try:
            if not self.influx_client:
                logger.warning("InfluxDB client not initialized")
                return []
            
            query_api = self.influx_client.query_api()
            bucket = os.getenv("INFLUXDB_BUCKET", "market_data")
            
            # Build query
            query_filter = ""
            if filters:
                filter_parts = []
                for key, value in filters.items():
                    filter_parts.append(f'r["{key}"] == "{value}"')
                query_filter = f" |> filter(fn: (r) => {' and '.join(filter_parts)})"
            
            flux_query = f'''
            from(bucket: "{bucket}")
                |> range(start: {time_range})
                |> filter(fn: (r) => r["_measurement"] == "{measurement}")
                {query_filter}
            '''
            
            result = query_api.query(flux_query)
            return result
            
        except Exception as e:
            logger.error(f"Failed to query InfluxDB: {e}")
            return []
    
    async def cache_set(self, key: str, value: Any, expire: int = 300):
        """Set value in Redis cache"""
        try:
            if self.redis_client:
                serialized_value = json.dumps(value, default=str)
                self.redis_client.setex(key, expire, serialized_value)
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            if self.redis_client:
                cached_value = self.redis_client.get(key)
                if cached_value:
                    return json.loads(cached_value)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve cached data: {e}")
            return None
    
    async def cache_delete(self, key: str):
        """Delete value from Redis cache"""
        try:
            if self.redis_client:
                self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Failed to delete cached data: {e}")
    
    async def store_market_data(self, symbol: str, data: Dict[str, Any]):
        """Store market data in both PostgreSQL and InfluxDB"""
        try:
            # Store in PostgreSQL
            async with self.get_postgres_session() as session:
                market_record = MarketData(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    price=data.get('price', 0),
                    volume=data.get('volume', 0),
                    open=data.get('open'),
                    high=data.get('high'),
                    low=data.get('low'),
                    close=data.get('close'),
                    data_source=data.get('source', 'yahoo')
                )
                session.add(market_record)
                await session.commit()
            
            # Store in InfluxDB
            await self.write_influx_point(
                measurement="market_data",
                fields={
                    "price": data.get('price', 0),
                    "volume": data.get('volume', 0),
                    "open": data.get('open', 0),
                    "high": data.get('high', 0),
                    "low": data.get('low', 0),
                    "close": data.get('close', 0)
                },
                tags={
                    "symbol": symbol,
                    "source": data.get('source', 'yahoo')
                }
            )
            
            logger.debug(f"Stored market data for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to store market data: {e}")
    
    async def store_pulse_data(self, pulse_type: str, symbol: str, pulse_value: float, 
                              confidence: float = 1.0, additional_data: Dict = None):
        """Store pulse engine data"""
        try:
            # Store in PostgreSQL
            async with self.get_postgres_session() as session:
                pulse_record = PulseData(
                    pulse_type=pulse_type,
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    pulse_value=pulse_value,
                    confidence=confidence,
                    additional_data=additional_data or {}
                )
                session.add(pulse_record)
                await session.commit()
            
            # Store in InfluxDB
            await self.write_influx_point(
                measurement="pulse_data",
                fields={
                    "pulse_value": pulse_value,
                    "confidence": confidence
                },
                tags={
                    "pulse_type": pulse_type,
                    "symbol": symbol
                }
            )
            
            logger.debug(f"Stored {pulse_type} pulse data for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to store pulse data: {e}")
    
    async def store_forecast(self, symbol: str, horizon: str, forecast_value: float,
                           confidence_lower: float = None, confidence_upper: float = None,
                           model_version: str = "1.0", accuracy_score: float = None):
        """Store market forecast data"""
        try:
            # Store in PostgreSQL
            async with self.get_postgres_session() as session:
                forecast_record = MarketForecast(
                    symbol=symbol,
                    forecast_horizon=horizon,
                    timestamp=datetime.utcnow(),
                    forecast_value=forecast_value,
                    confidence_interval_lower=confidence_lower,
                    confidence_interval_upper=confidence_upper,
                    model_version=model_version,
                    accuracy_score=accuracy_score
                )
                session.add(forecast_record)
                await session.commit()
            
            # Store in InfluxDB
            await self.write_influx_point(
                measurement="market_forecasts",
                fields={
                    "forecast_value": forecast_value,
                    "confidence_lower": confidence_lower or forecast_value,
                    "confidence_upper": confidence_upper or forecast_value,
                    "accuracy_score": accuracy_score or 0.0
                },
                tags={
                    "symbol": symbol,
                    "horizon": horizon,
                    "model_version": model_version
                }
            )
            
            logger.debug(f"Stored forecast for {symbol} - {horizon}")
            
        except Exception as e:
            logger.error(f"Failed to store forecast: {e}")
    
    async def get_recent_data(self, symbol: str, table: str, limit: int = 100):
        """Get recent data from PostgreSQL"""
        try:
            model_map = {
                "market_data": MarketData,
                "pulse_data": PulseData,
                "market_forecasts": MarketForecast,
                "sentiment_data": SentimentData,
                "risk_metrics": RiskMetrics
            }
            
            if table not in model_map:
                raise ValueError(f"Unknown table: {table}")
            
            async with self.get_postgres_session() as session:
                model = model_map[table]
                result = await session.execute(
                    f"SELECT * FROM {table} WHERE symbol = '{symbol}' ORDER BY timestamp DESC LIMIT {limit}"
                )
                return result.fetchall()
                
        except Exception as e:
            logger.error(f"Failed to get recent data: {e}")
            return []
    
    async def cleanup(self):
        """Clean up database connections"""
        try:
            if self.postgres_engine:
                await self.postgres_engine.dispose()
            
            if self.influx_client:
                self.influx_client.close()
            
            if self.redis_client:
                self.redis_client.close()
            
            logger.info("Database connections closed successfully")
            
        except Exception as e:
            logger.error(f"Database cleanup error: {e}")

# Database utility functions
async def create_tables():
    """Create database tables"""
    db_manager = DatabaseManager()
    await db_manager.initialize()
    await db_manager.cleanup()

async def get_database_manager() -> DatabaseManager:
    """Get database manager instance"""
    db_manager = DatabaseManager()
    await db_manager.initialize()
    return db_manager