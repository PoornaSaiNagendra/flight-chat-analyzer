from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Optional
from app.services.data_processor import FlightDataProcessor
from app.services.db_service import DatabaseService
from app.services.sql_llm_service import SQLLLMService
import pandas as pd
import os
from pydantic import BaseModel

router = APIRouter()

# Global variables to store services
db_service = None
sql_llm_service = None

class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 1000

@router.post("/process-data")
async def process_data(booking_file: str, airline_mapping_file: str):
    """Process and load data into in-memory database"""
    try:
        # Initialize services
        global db_service, sql_llm_service
        db_service = DatabaseService()
        sql_llm_service = SQLLLMService()
        
        # Load and process data
        processor = FlightDataProcessor()
        bookings_df, airlines_df = processor.load_data(booking_file, airline_mapping_file)
        
        # Clean data
        bookings_df = processor.clean_data(bookings_df)
        
        # Load into database
        db_service.load_data(bookings_df, airlines_df)
        
        # Get schema and sample data for LLM context
        schema = db_service.get_table_schema()
        sample_data = {
            'bookings': db_service.get_sample_data('bookings').to_dict(),
            'airlines': db_service.get_sample_data('airlines').to_dict()
        }
        
        return {
            "message": "Data processed and loaded into database successfully",
            "schema": schema,
            "sample_data": sample_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def execute_query(request: QueryRequest):
    """Execute a natural language query using LLM-generated SQL"""
    if not db_service or not sql_llm_service:
        raise HTTPException(status_code=400, detail="Data not processed yet")
    
    try:
        # Get schema and sample data for context
        schema = db_service.get_table_schema()
        sample_data = {
            'bookings': db_service.get_sample_data('bookings').to_dict(),
            'airlines': db_service.get_sample_data('airlines').to_dict()
        }
        
        # Generate SQL query
        query_generation = sql_llm_service.generate_sql_query(
            user_query=request.query,
            schema=schema,
            sample_data=sample_data
        )
        
        # Validate the generated query
        validation = sql_llm_service.validate_query(
            sql_query=query_generation['sql_query'],
            schema=schema
        )
        
        if not validation['is_valid']:
            return {
                "error": "Generated query is not valid",
                "validation_notes": validation['validation_notes'],
                "suggested_improvements": validation['suggested_improvements']
            }
        
        # Execute the query with limit
        query = query_generation['sql_query']
        if request.max_results:
            # Add LIMIT clause if not present
            if 'LIMIT' not in query.upper():
                query = f"{query} LIMIT {request.max_results}"
        
        results = db_service.execute_query(query)
        
        # Get business explanation of results
        explanation = sql_llm_service.explain_results(
            sql_query=query,
            query_results=results,
            user_query=request.query
        )
        
        return {
            "query_info": {
                "generated_sql": query_generation['sql_query'],
                "explanation": query_generation['explanation'],
                "potential_issues": query_generation['potential_issues'],
                "optimization_notes": query_generation['optimization_notes']
            },
            "validation": validation,
            "results": {
                "data": results.to_dict(orient='records'),
                "total_rows": len(results),
                "columns": list(results.columns)
            },
            "business_insights": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/schema")
async def get_schema():
    """Get the current database schema"""
    if not db_service:
        raise HTTPException(status_code=400, detail="Data not processed yet")
    return db_service.get_table_schema()

@router.get("/sample/{table}")
async def get_sample_data(table: str, limit: int = 5):
    """Get sample data from a specific table"""
    if not db_service:
        raise HTTPException(status_code=400, detail="Data not processed yet")
    try:
        return db_service.get_sample_data(table, limit).to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 