import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from app.utils.llm_analysis import LLMAnalyzer
from app.services.db_service import DatabaseService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisService:
    def __init__(self):
        self.analyzer = LLMAnalyzer()
        self.db_service = DatabaseService()

    def process_query(self, query: str, max_results: int = 1000) -> Dict[str, Any]:
        """Process a natural language query"""
        # Get schema and sample data
        schema = self.db_service.get_table_schema()
        sample_data = {
            'bookings': self.db_service.get_sample_data('bookings'),
            'airlines': self.db_service.get_sample_data('airlines')
        }
        
        # Analyze query and get SQL
        analysis = self.analyzer.analyze_query(query, schema, sample_data)
        
        # Add limit to SQL if specified
        if max_results:
            if 'LIMIT' not in analysis['sql_query'].upper():
                analysis['sql_query'] += f" LIMIT {max_results}"
        
        # Execute query
        results_df = self.db_service.execute_query(analysis['sql_query'])
        
        # Generate visualization if needed
        response = {
            "query_info": {
                "generated_sql": analysis['sql_query'],
                "explanation": analysis['reasoning']
            },
            "results": {
                "data": results_df.to_dict(orient='records'),
                "total_rows": len(results_df),
                "columns": results_df.columns.tolist()
            }
        }
        
        if analysis['needs_visualization']:
            viz_image = self.analyzer.generate_visualization(results_df, query)
            response["visualization"] = {
                "type": analysis['visualization_type'],
                "image": viz_image,
                "title": analysis.get('suggested_chart_title', 'Data Visualization')
            }
        
        return response

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the data"""
        summary = {
            "bookings": {},
            "airlines": {}
        }
        
        # Get booking statistics
        booking_stats = self.db_service.execute_query("""
            SELECT 
                COUNT(*) as total_bookings,
                COUNT(DISTINCT airline_id) as unique_airlines,
                COUNT(DISTINCT departure_airport) as unique_departure_airports,
                COUNT(DISTINCT arrival_airport) as unique_arrival_airports,
                AVG(available_seats) as avg_available_seats,
                AVG(total_capacity) as avg_total_capacity
            FROM bookings
        """)
        summary["bookings"] = booking_stats.to_dict(orient='records')[0]
        
        # Get airline statistics
        airline_stats = self.db_service.execute_query("""
            SELECT 
                COUNT(*) as total_airlines,
                COUNT(DISTINCT country) as unique_countries,
                AVG(fleet_size) as avg_fleet_size
            FROM airlines
        """)
        summary["airlines"] = airline_stats.to_dict(orient='records')[0]
        
        return summary

class FlightAnalysisService:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_airline_with_most_flights(self) -> Dict:
        """Find the airline with the most flights"""
        try:
            airline_counts = self.df['airline_name'].value_counts()
            top_airline = airline_counts.index[0]
            return {
                'airline': top_airline,
                'flight_count': int(airline_counts.iloc[0])
            }
        except Exception as e:
            logger.error(f"Error finding top airline: {str(e)}")
            raise

    def get_top_destinations(self, n: int = 3) -> List[Dict]:
        """Find the top N most frequented destinations"""
        try:
            destination_counts = self.df['arrival_airport'].value_counts().head(n)
            return [
                {'airport': airport, 'count': int(count)}
                for airport, count in destination_counts.items()
            ]
        except Exception as e:
            logger.error(f"Error finding top destinations: {str(e)}")
            raise

    def get_airline_bookings_yesterday(self, airline_name: str) -> int:
        """Get number of bookings for a specific airline yesterday"""
        try:
            yesterday = datetime.now() - timedelta(days=1)
            yesterday_str = yesterday.strftime('%Y-%m-%d')
            
            mask = (
                (self.df['airline_name'] == airline_name) &
                (self.df['departure_datetime'].dt.strftime('%Y-%m-%d') == yesterday_str)
            )
            return int(self.df[mask].shape[0])
        except Exception as e:
            logger.error(f"Error getting yesterday's bookings: {str(e)}")
            raise

    def get_average_delay_by_airline(self) -> List[Dict]:
        """Calculate average flight delay per airline"""
        try:
            if 'arrival_datetime' in self.df.columns and 'departure_datetime' in self.df.columns:
                self.df['delay_minutes'] = (
                    self.df['arrival_datetime'] - self.df['departure_datetime']
                ).dt.total_seconds() / 60
                
                avg_delays = self.df.groupby('airline_name')['delay_minutes'].mean()
                return [
                    {'airline': airline, 'avg_delay_minutes': float(delay)}
                    for airline, delay in avg_delays.items()
                ]
            return []
        except Exception as e:
            logger.error(f"Error calculating average delays: {str(e)}")
            raise

    def get_month_with_highest_bookings(self) -> Dict:
        """Find the month with the highest number of bookings"""
        try:
            self.df['month'] = self.df['departure_datetime'].dt.strftime('%Y-%m')
            monthly_bookings = self.df['month'].value_counts()
            top_month = monthly_bookings.index[0]
            return {
                'month': top_month,
                'booking_count': int(monthly_bookings.iloc[0])
            }
        except Exception as e:
            logger.error(f"Error finding top booking month: {str(e)}")
            raise

    def analyze_cancellation_patterns(self) -> Dict:
        """Analyze patterns in booking cancellations"""
        try:
            # Group by day of week and airline
            self.df['day_of_week'] = self.df['departure_datetime'].dt.day_name()
            cancellations = self.df[self.df['flight_status'] == 'Cancelled']
            
            # Calculate cancellation rates
            total_flights = self.df.groupby(['day_of_week', 'airline_name']).size()
            cancelled_flights = cancellations.groupby(['day_of_week', 'airline_name']).size()
            
            cancellation_rates = (cancelled_flights / total_flights * 100).fillna(0)
            
            # Get top 3 days with highest cancellation rates
            top_days = cancellation_rates.groupby('day_of_week').mean().nlargest(3)
            
            return {
                'top_cancellation_days': [
                    {'day': day, 'rate': float(rate)}
                    for day, rate in top_days.items()
                ],
                'airline_cancellation_rates': [
                    {
                        'airline': airline,
                        'day': day,
                        'rate': float(rate)
                    }
                    for (day, airline), rate in cancellation_rates.items()
                ]
            }
        except Exception as e:
            logger.error(f"Error analyzing cancellation patterns: {str(e)}")
            raise

    def analyze_seat_occupancy(self) -> Dict:
        """Analyze seat occupancy patterns"""
        try:
            if 'occupancy_rate' in self.df.columns:
                # Find most and least popular flights
                avg_occupancy = self.df.groupby('flight_id')['occupancy_rate'].mean()
                
                most_popular = avg_occupancy.nlargest(3)
                least_popular = avg_occupancy.nsmallest(3)
                
                return {
                    'most_popular_flights': [
                        {'flight_id': flight_id, 'occupancy_rate': float(rate)}
                        for flight_id, rate in most_popular.items()
                    ],
                    'least_popular_flights': [
                        {'flight_id': flight_id, 'occupancy_rate': float(rate)}
                        for flight_id, rate in least_popular.items()
                    ]
                }
            return {}
        except Exception as e:
            logger.error(f"Error analyzing seat occupancy: {str(e)}")
            raise 