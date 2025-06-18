from fastapi import APIRouter, Query, Depends, Header
from typing import Dict, Any, Optional
from app.services.autonomous_agent import AutonomousAgent
from app.utils.logger_config import setup_logger
import uuid

router = APIRouter()
logger = setup_logger(__name__)

# Initialize the autonomous agent
agent = AutonomousAgent()

def get_session_id(x_session_id: Optional[str] = Header(None)) -> str:
    """Get or create session ID"""
    if not x_session_id:
        return str(uuid.uuid4())
    return x_session_id

@router.get("/")
async def chat_with_data(
    query: str = Query(...),
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
        raise
