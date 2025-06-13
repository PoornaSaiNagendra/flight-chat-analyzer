import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_upload_files():
    booking_csv = "tests/test_data/booking.csv"
    airline_csv = "tests/test_data/airlines.csv"

    with open(booking_csv, "rb") as b_file, open(airline_csv, "rb") as a_file:
        response = client.post(
            "/upload/",
            files={
                "booking_file": ("booking.csv", b_file, "text/csv"),
                "airline_file": ("airlines.csv", a_file, "text/csv"),
            },
        )
    assert response.status_code == 200
    assert "Files uploaded and processed successfully." in response.json()["message"]

    assert os.path.exists("uploaded/cleaned_booking.csv")
    assert os.path.exists("uploaded/data_dictionary.json")
