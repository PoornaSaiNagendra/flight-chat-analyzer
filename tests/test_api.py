import pytest
from fastapi.testclient import TestClient
import pandas as pd
import os
from app.main import app
from app.services.data_processor import FlightDataProcessor

# Test client
client = TestClient(app)

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
def sample_files(tmp_path):
    # Create temporary files
    booking_file = tmp_path / "test_bookings.csv"
    airline_file = tmp_path / "test_airlines.csv"
    
    # Save sample data to files
    SAMPLE_BOOKINGS.to_csv(booking_file, index=False)
    SAMPLE_AIRLINES.to_csv(airline_file, index=False)
    
    return str(booking_file), str(airline_file)

def test_process_data(sample_files):
    booking_file, airline_file = sample_files
    
    # Test data processing endpoint
    response = client.post(
        "/analysis/process-data",
        params={
            "booking_file": booking_file,
            "airline_mapping_file": airline_file
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "schema" in data
    assert "sample_data" in data
    assert data["message"] == "Data processed and loaded into database successfully"

def test_query_endpoint(sample_files):
    booking_file, airline_file = sample_files
    
    # First process the data
    client.post(
        "/analysis/process-data",
        params={
            "booking_file": booking_file,
            "airline_mapping_file": airline_file
        }
    )
    
    # Test query endpoint
    response = client.post(
        "/analysis/query",
        json={
            "query": "Show me the top 3 most frequented destinations",
            "max_results": 1000
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "query_info" in data
    assert "validation" in data
    assert "results" in data
    assert "business_insights" in data
    
    # Check results structure
    assert "data" in data["results"]
    assert "total_rows" in data["results"]
    assert "columns" in data["results"]

def test_schema_endpoint(sample_files):
    booking_file, airline_file = sample_files
    
    # First process the data
    client.post(
        "/analysis/process-data",
        params={
            "booking_file": booking_file,
            "airline_mapping_file": airline_file
        }
    )
    
    # Test schema endpoint
    response = client.get("/analysis/schema")
    
    assert response.status_code == 200
    schema = response.json()
    assert "bookings" in schema
    assert "airlines" in schema

def test_sample_data_endpoint(sample_files):
    booking_file, airline_file = sample_files
    
    # First process the data
    client.post(
        "/analysis/process-data",
        params={
            "booking_file": booking_file,
            "airline_mapping_file": airline_file
        }
    )
    
    # Test sample data endpoint for bookings
    response = client.get("/analysis/sample/bookings")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    
    # Test sample data endpoint for airlines
    response = client.get("/analysis/sample/airlines")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_error_handling():
    # Test query without processing data
    response = client.post(
        "/analysis/query",
        json={
            "query": "Show me the top 3 most frequented destinations",
            "max_results": 1000
        }
    )
    
    assert response.status_code == 400
    assert "detail" in response.json()
    
    # Test invalid table name
    response = client.get("/analysis/sample/invalid_table")
    assert response.status_code == 500
    
    # Test invalid query
    response = client.post(
        "/analysis/query",
        json={
            "query": "Invalid query that should fail",
            "max_results": 1000
        }
    )
    
    assert response.status_code == 500
    assert "detail" in response.json() 