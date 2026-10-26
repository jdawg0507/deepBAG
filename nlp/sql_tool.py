import sqlite3
import os

class HistoricalSQLTool:
    def __init__(self, db_path=None):
        if db_path:
            self.db_path = db_path
        else:
            candidates = [
                "db/historical_nba.db",
                "../backend/db/historical_nba.db",
                "backend/db/historical_nba.db",
                os.path.join(os.path.dirname(__file__), "..", "db", "historical_nba.db"),
                os.path.join(os.path.dirname(__file__), "..", "..", "backend", "db", "historical_nba.db"),
            ]
            self.db_path = next((p for p in candidates if os.path.exists(p)), "db/historical_nba.db")
        
    def execute_sql(self, query: str):
        """Executes a SQL query on the local Kaggle NBA SQLite database."""
        if not os.path.exists(self.db_path):
            return {"error": f"Local database not found at {self.db_path}. Historical data offline."}

            
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Read-only query enforcement
                query_lower = query.strip().lower()
                if any(word in query_lower for word in ["insert", "update", "delete", "drop", "alter", "create"]):
                    return {"error": "Only SELECT queries are allowed."}
                    
                cursor = conn.cursor()
                cursor.execute(query)
                columns = [description[0] for description in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                
                # Convert rows to list of dicts for easy reading
                result_data = [dict(zip(columns, row)) for row in rows]
                
                return {
                    "total_rows_available": len(result_data),
                    "columns": columns,
                    "sample_data": result_data[:50],  # Limit to 50 rows to prevent blowing up tokens
                }
        except Exception as e:
            return {"error": str(e)}

def get_sql_tool_declaration():
    return {
        "name": "query_historical_database",
        "description": "Execute a raw SQL query against the local SQLite database containing historical NBA Box Scores. AVAILABLE TABLES AND SCHEMA: 1) `players` (personId, firstName, lastName, heightInches, bodyWeightLbs, jersey, etc.) 2) `playerstatistics` (firstName, lastName, personId, gameId, gameDate, points, assists, reboundsTotal, gameType, etc.) 3) `games` (gameId, hometeamName, awayteamName, homeScore, awayScore, gameType, gameDate). Note: gameType typically indicates 'Playoffs' or 'Regular Season'. Write queries using these exact table names.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "sql_query": {
                    "type": "STRING",
                    "description": "The exact SQL query to execute (e.g. SELECT * FROM playerstatistics WHERE firstName = 'LeBron' AND lastName = 'James' AND gameType = 'Playoffs' ORDER BY points DESC LIMIT 10)"
                }
            },
            "required": ["sql_query"]
        }
    }
