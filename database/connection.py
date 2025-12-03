"""
Mock Database Module for MPE Testing
Provides minimal DatabaseManager interface for engines that need it.
"""

import asyncio
from typing import Dict, Any, Optional


class DatabaseManager:
    """Mock Database Manager for testing MPE engines"""
    
    def __init__(self):
        self.connected = False
        self.data_cache = {}
    
    async def connect(self):
        """Mock connection"""
        self.connected = True
        return True
    
    async def disconnect(self):
        """Mock disconnection"""
        self.connected = False
    
    async def store_data(self, collection: str, data: Dict[str, Any]) -> bool:
        """Mock data storage"""
        if collection not in self.data_cache:
            self.data_cache[collection] = []
        self.data_cache[collection].append(data)
        return True
    
    async def retrieve_data(self, collection: str, query: Dict[str, Any] = None) -> list:
        """Mock data retrieval"""
        if collection in self.data_cache:
            return self.data_cache[collection]
        return []
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock query execution"""
        return {"success": True, "data": []}
    
    def get_status(self) -> Dict[str, Any]:
        """Get mock database status"""
        return {
            "connected": self.connected,
            "cache_size": len(self.data_cache),
            "collections": list(self.data_cache.keys())
        }


# Global database manager instance
db_manager = DatabaseManager()