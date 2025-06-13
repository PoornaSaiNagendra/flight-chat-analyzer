import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime
from app.utils.cleaning import DataCleaner
from app.utils.llm_analysis import LLMAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlightDataProcessor:
    def __init__(self):
        self.cleaner = DataCleaner()
        self.analyzer = LLMAnalyzer()
        self.column_mappings = {
            'airline_id': 'airline_code',
            'flight_number': 'flight_id',
            'departure_time': 'departure_datetime',
            'arrival_time': 'arrival_datetime',
            'origin': 'departure_airport',
            'destination': 'arrival_airport',
            'status': 'flight_status',
            'seats_available': 'available_seats',
            'total_seats': 'total_capacity'
        }

    def load_data(self, booking_file: str, airline_mapping_file: str) -> tuple:
        """Load and merge booking data with airline mappings"""
        try:
            bookings_df = pd.read_csv(booking_file)
            airlines_df = pd.read_csv(airline_mapping_file)
            
            # Merge airline data
            merged_df = pd.merge(
                bookings_df,
                airlines_df,
                on='airline_id',
                how='left'
            )
            
            return merged_df, airlines_df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        try:
            # Rename columns
            df = df.rename(columns=self.column_mappings)
            
            # Convert datetime columns
            datetime_columns = ['departure_datetime', 'arrival_datetime']
            for col in datetime_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Handle missing values
            df['flight_status'] = df['flight_status'].fillna('Unknown')
            df['available_seats'] = df['available_seats'].fillna(0)
            
            # Calculate derived columns
            if 'total_capacity' in df.columns and 'available_seats' in df.columns:
                df['occupancy_rate'] = (df['total_capacity'] - df['available_seats']) / df['total_capacity']
            
            return df
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise

    def validate_data(self, df: pd.DataFrame) -> dict:
        """Validate data quality"""
        validation = {
            "is_valid": True,
            "issues": []
        }
        
        # Check for required columns
        required_columns = {
            'bookings': ['booking_id', 'airline_id', 'flight_id', 'departure_datetime', 
                        'arrival_datetime', 'departure_airport', 'arrival_airport', 
                        'flight_status', 'available_seats', 'total_capacity'],
            'airlines': ['airline_id', 'airline_name', 'country', 'fleet_size']
        }
        
        # Determine which set of columns to check
        if 'flight_id' in df.columns:
            required = required_columns['bookings']
        else:
            required = required_columns['airlines']
        
        # Check for missing columns
        missing_columns = [col for col in required if col not in df.columns]
        if missing_columns:
            validation["is_valid"] = False
            validation["issues"].append(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        null_counts = df.isnull().sum()
        if null_counts.any():
            validation["issues"].append(f"Columns with missing values: {null_counts[null_counts > 0].to_dict()}")
        
        # Check for data type issues
        if 'departure_datetime' in df.columns:
            try:
                pd.to_datetime(df['departure_datetime'])
            except:
                validation["is_valid"] = False
                validation["issues"].append("Invalid datetime format in departure_datetime")
        
        if 'arrival_datetime' in df.columns:
            try:
                pd.to_datetime(df['arrival_datetime'])
            except:
                validation["is_valid"] = False
                validation["issues"].append("Invalid datetime format in arrival_datetime")
        
        return validation

    def process_data(self, booking_file: str, airline_file: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """Process and clean flight booking data"""
        # Load data
        bookings_df = pd.read_csv(booking_file)
        airlines_df = pd.read_csv(airline_file)
        
        # Clean data using LLM
        cleaned_bookings, bookings_report = self.cleaner.clean_data(bookings_df)
        cleaned_airlines, airlines_report = self.cleaner.clean_data(airlines_df)
        
        # Generate data dictionary
        bookings_dict = self.analyzer.generate_data_dictionary(cleaned_bookings)
        airlines_dict = self.analyzer.generate_data_dictionary(cleaned_airlines)
        
        # Combine reports
        report = f"""
        Booking Data Cleaning Report:
        {bookings_report}
        
        Airline Data Cleaning Report:
        {airlines_report}
        
        Data Dictionary:
        Bookings: {bookings_dict}
        Airlines: {airlines_dict}
        """
        
        return cleaned_bookings, cleaned_airlines, report 