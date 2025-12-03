"""
Mock database connection module for MPE system
"""

class DatabaseManager:
    def __init__(self):
        self.connected = False
    
    def connect(self):
        self.connected = True
        return self
    
    def close(self):
        self.connected = False
    
    def execute_query(self, query):
        return []
    
    def store_metrics(self, metrics):
        pass

# Make DatabaseManager available at module level for import compatibility
__all__ = ['DatabaseManager']