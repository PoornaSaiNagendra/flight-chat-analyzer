import sqlite3
import pandas as pd
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        self.cursor = self.conn.cursor()
        self._setup_tables()

    def _setup_tables(self):
        """Create necessary tables in the in-memory database"""
        try:
            # Create bookings table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS bookings (
                    booking_id TEXT PRIMARY KEY,
                    airline_id TEXT,
                    airline_name TEXT,
                    flight_id TEXT,
                    departure_datetime DATETIME,
                    arrival_datetime DATETIME,
                    departure_airport TEXT,
                    arrival_airport TEXT,
                    flight_status TEXT,
                    available_seats INTEGER,
                    total_capacity INTEGER,
                    occupancy_rate FLOAT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create airlines table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS airlines (
                    airline_id TEXT PRIMARY KEY,
                    airline_name TEXT,
                    country TEXT,
                    fleet_size INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            self.conn.commit()
        except Exception as e:
            logger.error(f"Error setting up database tables: {str(e)}")
            raise

    def load_data(self, bookings_df: pd.DataFrame, airlines_df: pd.DataFrame):
        """Load data into the in-memory database"""
        try:
            # Load bookings data
            bookings_df.to_sql('bookings', self.conn, if_exists='replace', index=False)
            
            # Load airlines data
            airlines_df.to_sql('airlines', self.conn, if_exists='replace', index=False)
            
            # Create indexes for better query performance
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_airline_id ON bookings(airline_id)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_departure_datetime ON bookings(departure_datetime)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_flight_status ON bookings(flight_status)')
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error loading data into database: {str(e)}")
            raise

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame"""
        try:
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    def get_table_schema(self) -> Dict:
        """Get the schema of all tables"""
        try:
            schema = {}
            for table in ['bookings', 'airlines']:
                self.cursor.execute(f"PRAGMA table_info({table})")
                columns = self.cursor.fetchall()
                schema[table] = [
                    {
                        'name': col[1],
                        'type': col[2],
                        'nullable': not col[3],
                        'default': col[4]
                    }
                    for col in columns
                ]
            return schema
        except Exception as e:
            logger.error(f"Error getting table schema: {str(e)}")
            raise

    def get_sample_data(self, table: str, limit: int = 5) -> pd.DataFrame:
        """Get sample data from a table"""
        try:
            return pd.read_sql_query(f"SELECT * FROM {table} LIMIT {limit}", self.conn)
        except Exception as e:
            logger.error(f"Error getting sample data: {str(e)}")
            raise

    def close(self):
        """Close the database connection"""
        self.conn.close() 