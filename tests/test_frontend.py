import pytest
import streamlit as st
import pandas as pd
import os
from frontend.app import (
    process_uploaded_files,
    display_message,
    display_analysis_results
)

# Test data
SAMPLE_ANALYSIS_RESULTS = {
    'query_info': {
        'generated_sql': 'SELECT arrival_airport, COUNT(*) as count FROM bookings GROUP BY arrival_airport',
        'explanation': 'Counts flights by destination',
        'potential_issues': [
            {
                'issue': 'Missing data',
                'severity': 'low',
                'mitigation': 'NULL handling included'
            }
        ],
        'optimization_notes': [
            {
                'optimization': 'Index on arrival_airport',
                'impact': 'high'
            }
        ]
    },
    'validation': {
        'is_valid': True,
        'validation_notes': [
            {
                'aspect': 'syntax',
                'status': 'pass',
                'details': 'Valid SQL syntax'
            }
        ]
    },
    'results': {
        'data': [
            {'arrival_airport': 'LAX', 'count': 1},
            {'arrival_airport': 'SFO', 'count': 1},
            {'arrival_airport': 'JFK', 'count': 1}
        ],
        'total_rows': 3,
        'columns': ['arrival_airport', 'count']
    },
    'business_insights': {
        'business_interpretation': {
            'summary': 'Equal distribution of flights across destinations',
            'key_findings': ['All destinations have same number of flights'],
            'context': 'Current flight distribution'
        },
        'insights': [
            {
                'insight': 'Balanced route network',
                'significance': 'Good for market coverage',
                'confidence': 'high'
            }
        ],
        'recommendations': [
            {
                'recommendation': 'Consider expanding to more destinations',
                'priority': 'medium',
                'implementation': 'Market research and route planning'
            }
        ],
        'risks': [
            {
                'risk': 'Limited market coverage',
                'severity': 'low',
                'mitigation': 'Regular route analysis'
            }
        ]
    }
}

@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state"""
    return {
        'messages': [],
        'data_processed': False,
        'schema': None,
        'sample_data': None
    }

@pytest.fixture
def sample_files(tmp_path):
    """Create sample files for testing"""
    # Create upload directory
    upload_dir = tmp_path / "uploaded"
    upload_dir.mkdir()
    
    # Create sample booking data
    booking_data = pd.DataFrame({
        'booking_id': ['B001', 'B002'],
        'airline_id': ['A001', 'A002'],
        'flight_id': ['F001', 'F002'],
        'departure_datetime': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
        'arrival_datetime': ['2024-01-01 12:00:00', '2024-01-01 13:00:00'],
        'departure_airport': ['JFK', 'LAX'],
        'arrival_airport': ['LAX', 'SFO'],
        'flight_status': ['Completed', 'Completed'],
        'available_seats': [50, 75],
        'total_capacity': [100, 100]
    })
    
    # Create sample airline data
    airline_data = pd.DataFrame({
        'airline_id': ['A001', 'A002'],
        'airline_name': ['Test Airline 1', 'Test Airline 2'],
        'country': ['USA', 'Canada'],
        'fleet_size': [10, 5]
    })
    
    # Save files
    booking_file = upload_dir / "test_bookings.csv"
    airline_file = upload_dir / "test_airlines.csv"
    
    booking_data.to_csv(booking_file, index=False)
    airline_data.to_csv(airline_file, index=False)
    
    return str(booking_file), str(airline_file)

def test_process_uploaded_files(sample_files, monkeypatch):
    """Test file processing functionality"""
    booking_file, airline_file = sample_files
    
    # Mock requests.post
    def mock_post(*args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.status_code = 200
                self._json = {
                    'message': 'Data processed successfully',
                    'schema': {'bookings': [], 'airlines': []},
                    'sample_data': {'bookings': [], 'airlines': []}
                }
            
            def json(self):
                return self._json
        
        return MockResponse()
    
    monkeypatch.setattr('requests.post', mock_post)
    
    # Test successful processing
    assert process_uploaded_files() is True
    
    # Test with invalid files
    monkeypatch.setattr('os.listdir', lambda x: [])
    assert process_uploaded_files() is False

def test_display_message(capsys):
    """Test message display functionality"""
    # Test user message
    display_message("Test user message", is_user=True)
    captured = capsys.readouterr()
    assert "Test user message" in captured.out
    
    # Test assistant message
    display_message("Test assistant message", is_user=False)
    captured = capsys.readouterr()
    assert "Test assistant message" in captured.out

def test_display_analysis_results(capsys):
    """Test analysis results display functionality"""
    display_analysis_results(SAMPLE_ANALYSIS_RESULTS)
    captured = capsys.readouterr()
    
    # Check if key components are displayed
    assert "Generated SQL Query" in captured.out
    assert "Business Insights" in captured.out
    assert "Query Results" in captured.out
    assert "Total rows: 3" in captured.out

def test_session_state_management(mock_session_state):
    """Test session state management"""
    # Test initial state
    assert not mock_session_state['data_processed']
    assert mock_session_state['schema'] is None
    assert mock_session_state['sample_data'] is None
    
    # Test state updates
    mock_session_state['data_processed'] = True
    mock_session_state['schema'] = {'bookings': [], 'airlines': []}
    mock_session_state['sample_data'] = {'bookings': [], 'airlines': []}
    
    assert mock_session_state['data_processed']
    assert mock_session_state['schema'] is not None
    assert mock_session_state['sample_data'] is not None

def test_error_handling(mock_session_state, monkeypatch):
    """Test error handling in frontend"""
    # Mock requests.post to simulate error
    def mock_post_error(*args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.status_code = 500
                self._json = {'detail': 'Test error'}
            
            def json(self):
                return self._json
        
        return MockResponse()
    
    monkeypatch.setattr('requests.post', mock_post_error)
    
    # Test error handling in file processing
    assert process_uploaded_files() is False
    
    # Test error handling in query processing
    response = requests.post(
        "http://localhost:8000/analysis/query",
        json={"query": "Test query", "max_results": 1000}
    )
    assert response.status_code == 500
    assert "detail" in response.json() 