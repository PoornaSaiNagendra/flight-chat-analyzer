import pytest
import pandas as pd
import os
from datetime import datetime
from app.services.data_processor import FlightDataProcessor
from app.services.db_service import DatabaseService
from app.services.sql_llm_service import SQLLLMService

# Test data
SAMPLE_BOOKINGS = pd.DataFrame({
    'booking_id': ['B001', 'B002', 'B003'],
    'airline_id': ['A001', 'A001', 'A002'],
    'flight_id': ['F001', 'F002', 'F003'],
    'departure_datetime': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00'],
    'arrival_datetime': ['2024-01-01 12:00:00', '2024-01-01 13:00:00', '2024-01-01 14:00:00'],
    'departure_airport': ['JFK', 'LAX', 'SFO'],
    'arrival_airport': ['LAX', 'SFO', 'JFK'],
    'flight_status': ['Completed', 'Cancelled', 'Completed'],
    'available_seats': [50, 100, 75],
    'total_capacity': [100, 100, 100]
})

SAMPLE_AIRLINES = pd.DataFrame({
    'airline_id': ['A001', 'A002'],
    'airline_name': ['Test Airline 1', 'Test Airline 2'],
    'country': ['USA', 'Canada'],
    'fleet_size': [10, 5]
})

@pytest.fixture
def data_processor():
    return FlightDataProcessor()

@pytest.fixture
def db_service():
    return DatabaseService()

@pytest.fixture
def sql_llm_service():
    return SQLLLMService()

class TestDataProcessor:
    def test_clean_data(self, data_processor):
        # Test data cleaning
        cleaned_df = data_processor.clean_data(SAMPLE_BOOKINGS)
        
        # Check if datetime columns are properly converted
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df['departure_datetime'])
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df['arrival_datetime'])
        
        # Check if occupancy rate is calculated
        assert 'occupancy_rate' in cleaned_df.columns
        assert cleaned_df['occupancy_rate'].dtype == 'float64'
        
        # Check if values are within expected ranges
        assert cleaned_df['occupancy_rate'].min() >= 0
        assert cleaned_df['occupancy_rate'].max() <= 1

    def test_validate_data(self, data_processor):
        # Test data validation
        validation_result = data_processor.validate_data(SAMPLE_BOOKINGS)
        assert validation_result['is_valid']
        assert len(validation_result['issues']) == 0

class TestDatabaseService:
    def test_setup_tables(self, db_service):
        # Test table creation
        db_service._setup_tables()
        
        # Check if tables exist
        cursor = db_service.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]
        assert 'bookings' in tables
        assert 'airlines' in tables

    def test_load_data(self, db_service):
        # Test data loading
        db_service.load_data(SAMPLE_BOOKINGS, SAMPLE_AIRLINES)
        
        # Check if data is loaded correctly
        bookings_df = pd.read_sql_query("SELECT * FROM bookings", db_service.conn)
        airlines_df = pd.read_sql_query("SELECT * FROM airlines", db_service.conn)
        
        assert len(bookings_df) == len(SAMPLE_BOOKINGS)
        assert len(airlines_df) == len(SAMPLE_AIRLINES)

    def test_execute_query(self, db_service):
        # Load test data
        db_service.load_data(SAMPLE_BOOKINGS, SAMPLE_AIRLINES)
        
        # Test simple query
        query = "SELECT COUNT(*) as count FROM bookings"
        result = db_service.execute_query(query)
        assert result['count'].iloc[0] == len(SAMPLE_BOOKINGS)
        
        # Test join query
        query = """
        SELECT a.airline_name, COUNT(*) as flight_count
        FROM bookings b
        JOIN airlines a ON b.airline_id = a.airline_id
        GROUP BY a.airline_name
        """
        result = db_service.execute_query(query)
        assert len(result) == len(SAMPLE_AIRLINES)

    def test_get_table_schema(self, db_service):
        # Test schema retrieval
        schema = db_service.get_table_schema()
        
        assert 'bookings' in schema
        assert 'airlines' in schema
        assert len(schema['bookings']) > 0
        assert len(schema['airlines']) > 0

class TestSQLLLMService:
    def test_generate_sql_query(self, sql_llm_service):
        # Test SQL query generation
        user_query = "Show me the top 3 most frequented destinations"
        schema = {
            'bookings': [
                {'name': 'arrival_airport', 'type': 'TEXT'},
                {'name': 'departure_airport', 'type': 'TEXT'}
            ]
        }
        sample_data = {
            'bookings': SAMPLE_BOOKINGS.head().to_dict()
        }
        
        result = sql_llm_service.generate_sql_query(user_query, schema, sample_data)
        
        assert 'sql_query' in result
        assert 'explanation' in result
        assert 'potential_issues' in result
        assert 'optimization_notes' in result

    def test_validate_query(self, sql_llm_service):
        # Test query validation
        sql_query = "SELECT arrival_airport, COUNT(*) as count FROM bookings GROUP BY arrival_airport"
        schema = {
            'bookings': [
                {'name': 'arrival_airport', 'type': 'TEXT'}
            ]
        }
        
        result = sql_llm_service.validate_query(sql_query, schema)
        
        assert 'is_valid' in result
        assert 'validation_notes' in result
        assert 'performance_notes' in result
        assert 'security_notes' in result

    def test_explain_results(self, sql_llm_service):
        # Test results explanation
        sql_query = "SELECT arrival_airport, COUNT(*) as count FROM bookings GROUP BY arrival_airport"
        query_results = pd.DataFrame({
            'arrival_airport': ['LAX', 'SFO', 'JFK'],
            'count': [1, 1, 1]
        })
        
        result = sql_llm_service.explain_results(sql_query, query_results, "Show me the most frequented destinations")
        
        assert 'business_interpretation' in result
        assert 'insights' in result
        assert 'recommendations' in result
        assert 'risks' in result 