import sqlite3
import os
import json
from datetime import datetime

class QueryCache:
    def __init__(self, db_path="backend/query_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        # Create table if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    query TEXT PRIMARY KEY,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get(self, query: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT response FROM cache WHERE query = ?", (query.strip().lower(),))
            row = cursor.fetchone()
            if row:
                return row[0]
            return None

    def set(self, query: str, response: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO cache (query, response, timestamp)
                VALUES (?, ?, ?)
            """, (query.strip().lower(), response, datetime.now()))
            conn.commit()
