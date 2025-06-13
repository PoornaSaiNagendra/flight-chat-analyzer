from typing import Dict, List, Optional, Union
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import logging
import os
import json
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLLLMService:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self._setup_prompts()
        self._setup_common_queries()

    def _setup_common_queries(self):
        """Setup common query patterns for reference"""
        self.common_queries = {
            'airline_analysis': """
                SELECT 
                    a.airline_name,
                    COUNT(*) as total_flights,
                    AVG(CASE WHEN b.flight_status = 'Cancelled' THEN 1 ELSE 0 END) as cancellation_rate,
                    AVG(julianday(b.arrival_datetime) - julianday(b.departure_datetime)) * 24 * 60 as avg_delay_minutes
                FROM bookings b
                JOIN airlines a ON b.airline_id = a.airline_id
                GROUP BY a.airline_name
                ORDER BY total_flights DESC
            """,
            'route_analysis': """
                SELECT 
                    departure_airport,
                    arrival_airport,
                    COUNT(*) as route_frequency,
                    AVG(occupancy_rate) as avg_occupancy
                FROM bookings
                GROUP BY departure_airport, arrival_airport
                ORDER BY route_frequency DESC
            """,
            'time_analysis': """
                SELECT 
                    strftime('%Y-%m', departure_datetime) as month,
                    COUNT(*) as total_bookings,
                    AVG(CASE WHEN flight_status = 'Cancelled' THEN 1 ELSE 0 END) as cancellation_rate
                FROM bookings
                GROUP BY month
                ORDER BY month
            """
        }

    def _setup_prompts(self):
        """Setup all LLM prompts for SQL generation"""
        self.prompts = {
            'query_generation': PromptTemplate(
                input_variables=["user_query", "schema", "sample_data", "common_queries"],
                template="""
                You are an expert SQL query generator specializing in flight booking data analysis.
                Your task is to convert natural language queries into optimized SQLite SQL queries.

                Database Schema:
                {schema}

                Sample Data:
                {sample_data}

                Common Query Patterns (for reference):
                {common_queries}

                User Query:
                {user_query}

                Requirements:
                1. Generate a valid SQLite SQL query that answers the user's question
                2. Use appropriate JOINs when accessing multiple tables
                3. Apply proper date/time functions (julianday, strftime) for datetime operations
                4. Include error handling for NULL values and edge cases
                5. Optimize the query using indexes and efficient joins
                6. Consider using CTEs for complex queries
                7. Add appropriate comments explaining the query logic

                Return the response in the following JSON format:
                {{
                    "sql_query": "the generated SQL query with comments",
                    "explanation": "detailed explanation of what the query does and how it works",
                    "potential_issues": [
                        {{
                            "issue": "description of potential issue",
                            "severity": "high/medium/low",
                            "mitigation": "how the query handles this issue"
                        }}
                    ],
                    "optimization_notes": [
                        {{
                            "optimization": "description of optimization",
                            "impact": "expected performance impact"
                        }}
                    ],
                    "suggested_improvements": [
                        "list of potential improvements for future iterations"
                    ]
                }}
                """
            ),
            
            'query_validation': PromptTemplate(
                input_variables=["sql_query", "schema", "common_queries"],
                template="""
                You are a SQL query validator specializing in flight booking data analysis.
                Validate the following SQL query against the database schema and provide comprehensive feedback.

                Database Schema:
                {schema}

                Common Query Patterns (for reference):
                {common_queries}

                SQL Query to Validate:
                {sql_query}

                Please provide:
                1. Syntax validation
                2. Schema compatibility check
                3. Performance analysis
                4. Security assessment
                5. Edge case handling
                6. Comparison with common patterns

                Return the response in the following JSON format:
                {{
                    "is_valid": true/false,
                    "validation_notes": [
                        {{
                            "aspect": "syntax/schema/performance/security",
                            "status": "pass/fail/warning",
                            "details": "detailed explanation"
                        }}
                    ],
                    "performance_notes": [
                        {{
                            "concern": "description of performance concern",
                            "severity": "high/medium/low",
                            "recommendation": "how to address it"
                        }}
                    ],
                    "security_notes": [
                        {{
                            "concern": "description of security concern",
                            "severity": "high/medium/low",
                            "recommendation": "how to address it"
                        }}
                    ],
                    "suggested_improvements": [
                        {{
                            "improvement": "description of improvement",
                            "priority": "high/medium/low",
                            "implementation": "how to implement it"
                        }}
                    ]
                }}
                """
            ),
            
            'query_explanation': PromptTemplate(
                input_variables=["sql_query", "query_results", "user_query"],
                template="""
                You are a business analyst specializing in flight booking data.
                Explain the results of the following SQL query in business terms, focusing on actionable insights.

                Original User Query:
                {user_query}

                SQL Query:
                {sql_query}

                Query Results:
                {query_results}

                Please provide:
                1. Business interpretation of the results
                2. Key insights and trends
                3. Potential business implications
                4. Actionable recommendations
                5. Risk factors and considerations
                6. Comparison with industry benchmarks (if applicable)

                Return the response in the following JSON format:
                {{
                    "business_interpretation": {{
                        "summary": "high-level summary of the results",
                        "key_findings": ["list of key findings"],
                        "context": "business context for the results"
                    }},
                    "insights": [
                        {{
                            "insight": "description of insight",
                            "significance": "why it matters",
                            "confidence": "high/medium/low"
                        }}
                    ],
                    "implications": [
                        {{
                            "implication": "description of business implication",
                            "impact": "high/medium/low",
                            "timeframe": "short/medium/long term"
                        }}
                    ],
                    "recommendations": [
                        {{
                            "recommendation": "specific actionable recommendation",
                            "priority": "high/medium/low",
                            "implementation": "how to implement it"
                        }}
                    ],
                    "risks": [
                        {{
                            "risk": "description of risk",
                            "severity": "high/medium/low",
                            "mitigation": "how to mitigate it"
                        }}
                    ]
                }}
                """
            )
        }

    def _parse_json_response(self, response: str) -> Dict:
        """Safely parse JSON response from LLM"""
        try:
            # First try direct JSON parsing
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                # Try to find JSON in the response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
            except:
                logger.error(f"Failed to parse JSON response: {response}")
                raise ValueError("Invalid JSON response from LLM")

    def generate_sql_query(self, user_query: str, schema: Dict, sample_data: Dict) -> Dict:
        """Generate SQL query from natural language"""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.prompts['query_generation'])
            result = chain.run(
                user_query=user_query,
                schema=schema,
                sample_data=sample_data,
                common_queries=self.common_queries
            )
            return self._parse_json_response(result)
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            raise

    def validate_query(self, sql_query: str, schema: Dict) -> Dict:
        """Validate generated SQL query"""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.prompts['query_validation'])
            result = chain.run(
                sql_query=sql_query,
                schema=schema,
                common_queries=self.common_queries
            )
            return self._parse_json_response(result)
        except Exception as e:
            logger.error(f"Error validating SQL query: {str(e)}")
            raise

    def explain_results(self, sql_query: str, query_results: pd.DataFrame, user_query: str) -> Dict:
        """Explain query results in business terms"""
        try:
            # Convert DataFrame to string representation with formatting
            results_str = query_results.to_string(index=False)
            
            chain = LLMChain(llm=self.llm, prompt=self.prompts['query_explanation'])
            result = chain.run(
                sql_query=sql_query,
                query_results=results_str,
                user_query=user_query
            )
            return self._parse_json_response(result)
        except Exception as e:
            logger.error(f"Error explaining query results: {str(e)}")
            raise 