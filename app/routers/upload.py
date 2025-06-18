from fastapi import APIRouter, UploadFile, File, Depends, Header
from typing import Dict, Any, Optional
import os
from app.services.autonomous_agent import AutonomousAgent
from app.utils.logger_config import setup_logger
import uuid
import tempfile
import shutil

# Setup logger
logger = setup_logger(__name__)

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploaded")
BOOKING_PATH = os.path.join(UPLOAD_DIR, "booking.csv")
AIRLINE_PATH = os.path.join(UPLOAD_DIR, "airlines.csv")
CLEANED_PATH = os.path.join(UPLOAD_DIR, "cleaned_booking.csv")
DATA_DICT_PATH = os.path.join(UPLOAD_DIR, "data_dictionary.json")

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.info(f"Upload directory created/verified at: {UPLOAD_DIR}")

router = APIRouter()

# Initialize the autonomous agent
agent = AutonomousAgent()

def get_session_id(x_session_id: Optional[str] = Header(None)) -> str:
    """Get or create session ID"""
    if not x_session_id:
        return str(uuid.uuid4())
    return x_session_id

@router.post("/")
async def upload_files(
    booking_file: UploadFile = File(...),
    airline_file: UploadFile = File(...),
    session_id: str = Depends(get_session_id)
) -> Dict[str, Any]:
    """
    Upload and process booking and airline files.
    This endpoint:
    1. Saves uploaded files temporarily
    2. Processes files using the autonomous agent
    3. Maintains session context
    """
    try:
        logger.info("Starting file upload process")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as booking_temp, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as airline_temp:
            
            # Save booking file
            logger.info("Saving booking file")
            shutil.copyfileobj(booking_file.file, booking_temp)
            booking_path = booking_temp.name
            
            # Save airline file
            logger.info("Saving airline file")
            shutil.copyfileobj(airline_file.file, airline_temp)
            airline_path = airline_temp.name
        
        try:
            # Process files using autonomous agent
            logger.info("Processing files with autonomous agent")
            result = agent.process_files(booking_path, airline_path, session_id)
            logger.info("File processing completed successfully")
            
            return {
                "message": "Files uploaded and processed successfully",
                "session_id": session_id,
                **result
            }
            
        finally:
            # Clean up temporary files
            os.unlink(booking_path)
            os.unlink(airline_path)
            
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}", exc_info=True)
        raise
