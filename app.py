from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.services.autonomous_agent import AutonomousAgent
from app.utils.logger_config import setup_logger
import os
import tempfile
import shutil
import uuid
from typing import Dict, Any, Optional

# Setup logger
logger = setup_logger(__name__)

# Create necessary directories
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploaded")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Flight Data Analysis API",
    description="API for analyzing flight booking and airline data using LLM",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/uploaded", StaticFiles(directory=UPLOAD_DIR), name="uploaded")

# Initialize the autonomous agent
agent = AutonomousAgent()

def get_session_id(x_session_id: Optional[str] = Header(None)) -> str:
    """Get or create session ID"""
    if not x_session_id:
        return str(uuid.uuid4())
    return x_session_id

@app.post("/analysis/process-data")
async def process_data(
    booking_file: UploadFile = File(...),
    airline_file: UploadFile = File(...),
    session_id: str = Depends(get_session_id)
) -> Dict[str, Any]:
    """
    Process booking and airline data files.
    This endpoint:
    1. Saves uploaded files temporarily
    2. Analyzes column descriptions
    3. Cleans the data
    4. Loads into in-memory SQLite database
    """
    try:
        logger.info("Starting data processing")
        logger.info(f"Received files - booking_file: {booking_file.filename}, airline_file: {airline_file.filename}")
        
        # Save files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as booking_temp, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as airline_temp:
            
            # Save booking file
            shutil.copyfileobj(booking_file.file, booking_temp)
            booking_path = booking_temp.name
            logger.info(f"Saved booking file to: {booking_path}")
            
            # Save airline file
            shutil.copyfileobj(airline_file.file, airline_temp)
            airline_path = airline_temp.name
            logger.info(f"Saved airline file to: {airline_path}")
        
        try:
            # Process files
            logger.info("Starting file processing with agent")
            result = agent.process_files(booking_path, airline_path, session_id)
            logger.info("Data processing completed successfully")
            return result
            
        finally:
            # Clean up temporary files
            logger.info("Cleaning up temporary files")
            os.unlink(booking_path)
            os.unlink(airline_path)
            
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/query")
async def process_query(
    query: str,
    session_id: str = Depends(get_session_id)
) -> Dict[str, Any]:
    """
    Process a natural language query about the data.
    This endpoint:
    1. Determines query intent (insight/visualization)
    2. Generates and executes SQL
    3. Returns either insights or visualization
    """
    try:
        logger.info(f"Processing query: {query}")
        result = agent.process_query(query, session_id)
        logger.info("Query processed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/chat-history")
async def get_chat_history(
    session_id: str = Depends(get_session_id),
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get chat history for a session.
    """
    try:
        logger.info(f"Retrieving chat history for session {session_id}")
        history = agent.get_chat_history(session_id, limit)
        return {
            "session_id": session_id,
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/analysis/chat-history")
async def clear_chat_history(
    session_id: str = Depends(get_session_id)
) -> Dict[str, Any]:
    """
    Clear chat history for a session.
    """
    try:
        logger.info(f"Clearing chat history for session {session_id}")
        agent.clear_chat_history(session_id)
        return {
            "message": "Chat history cleared successfully",
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/column-descriptions")
async def get_column_descriptions(
    session_id: str = Depends(get_session_id)
) -> Dict[str, Any]:
    """
    Get descriptions of all columns in the loaded datasets.
    """
    try:
        logger.info("Retrieving column descriptions")
        return {
            "column_descriptions": agent.column_descriptions
        }
        
    except Exception as e:
        logger.error(f"Error retrieving column descriptions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Flight Data Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "analysis": "/analysis",
            "upload": "/upload",
            "chat": "/chat"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Application startup initiated")
    logger.info(f"Upload directory: {UPLOAD_DIR}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Application shutdown initiated") 