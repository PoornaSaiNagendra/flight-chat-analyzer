import pytest
import pandas as pd
from datetime import datetime, timedelta
from app.services.analysis_service import FlightAnalysisService

@pytest.fixture
def sample_data():
    # Create sample flight data
    data = {
        'airline_name': ['Airline A', 'Airline A', 'Airline B', 'Airline B', 'Airline C'],
        'flight_id': ['FL001', 'FL002', 'FL003', 'FL004', 'FL005'],
        'departure_datetime': [
            datetime.now(),
            datetime.now() + timedelta(hours=2),
            datetime.now() + timedelta(hours=4),
            datetime.now() + timedelta(hours=6),
            datetime.now() + timedelta(hours=8)
        ],
        'arrival_datetime': [
            datetime.now() + timedelta(hours=1),
            datetime.now() + timedelta(hours=3),
            datetime.now() + timedelta(hours=5),
            datetime.now() + timedelta(hours=7),
            datetime.now() + timedelta(hours=9)
        ],
        'arrival_airport': ['JFK', 'LAX', 'JFK', 'SFO', 'LAX'],
        'flight_status': ['On Time', 'Cancelled', 'On Time', 'On Time', 'Cancelled'],
        'available_seats': [50, 100, 75, 25, 150],
        'total_capacity': [100, 200, 150, 50, 300]
    }
    return pd.DataFrame(data)

def test_get_airline_with_most_flights(sample_data):
    service = FlightAnalysisService(sample_data)
    result = service.get_airline_with_most_flights()
    assert result['airline'] == 'Airline A'
    assert result['flight_count'] == 2

def test_get_top_destinations(sample_data):
    service = FlightAnalysisService(sample_data)
    result = service.get_top_destinations(n=2)
    assert len(result) == 2
    assert result[0]['airport'] == 'JFK'
    assert result[0]['count'] == 2

def test_get_average_delay_by_airline(sample_data):
    service = FlightAnalysisService(sample_data)
    result = service.get_average_delay_by_airline()
    assert len(result) == 3
    for airline_delay in result:
        assert 'airline' in airline_delay
        assert 'avg_delay_minutes' in airline_delay

def test_analyze_cancellation_patterns(sample_data):
    service = FlightAnalysisService(sample_data)
    result = service.analyze_cancellation_patterns()
    assert 'top_cancellation_days' in result
    assert 'airline_cancellation_rates' in result

def test_analyze_seat_occupancy(sample_data):
    service = FlightAnalysisService(sample_data)
    result = service.analyze_seat_occupancy()
    assert 'most_popular_flights' in result
    assert 'least_popular_flights' in result
    assert len(result['most_popular_flights']) == 3
    assert len(result['least_popular_flights']) == 3 