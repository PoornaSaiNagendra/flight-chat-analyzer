from typing import Dict, List, Optional
from datetime import datetime
import json
from app.utils.logger_config import setup_logger

logger = setup_logger(__name__)

class ChatHistory:
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
        self.current_context: Dict[str, Any] = {}
        
    def start_conversation(self, session_id: str) -> None:
        """Start a new conversation session"""
        try:
            logger.info(f"Starting new conversation session: {session_id}")
            self.conversations[session_id] = []
            self.current_context = {
                "session_id": session_id,
                "start_time": datetime.now().isoformat(),
                "last_query": None,
                "column_descriptions": {},
                "data_summary": {},
                "visualization_history": []
            }
        except Exception as e:
            logger.error(f"Error starting conversation: {str(e)}", exc_info=True)
            raise

    def add_message(self, session_id: str, role: str, content: Dict) -> None:
        """Add a message to the conversation history"""
        try:
            logger.info(f"Adding message to session {session_id}")
            if session_id not in self.conversations:
                self.start_conversation(session_id)
            
            message = {
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": content
            }
            
            self.conversations[session_id].append(message)
            
            # Update context
            if role == "user":
                self.current_context["last_query"] = content.get("query")
            elif role == "assistant":
                if content.get("type") == "visualization":
                    self.current_context["visualization_history"].append({
                        "query": self.current_context["last_query"],
                        "visualization_type": content.get("visualization_type"),
                        "timestamp": message["timestamp"]
                    })
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}", exc_info=True)
            raise

    def update_context(self, session_id: str, updates: Dict) -> None:
        """Update the conversation context"""
        try:
            logger.info(f"Updating context for session {session_id}")
            if session_id not in self.conversations:
                self.start_conversation(session_id)
            
            self.current_context.update(updates)
        except Exception as e:
            logger.error(f"Error updating context: {str(e)}", exc_info=True)
            raise

    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get conversation history for a session"""
        try:
            logger.info(f"Retrieving conversation history for session {session_id}")
            if session_id not in self.conversations:
                return []
            
            history = self.conversations[session_id]
            if limit:
                history = history[-limit:]
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}", exc_info=True)
            raise

    def get_context(self, session_id: str) -> Dict:
        """Get current context for a session"""
        try:
            logger.info(f"Retrieving context for session {session_id}")
            if session_id not in self.conversations:
                self.start_conversation(session_id)
            
            return self.current_context
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
            raise

    def clear_conversation(self, session_id: str) -> None:
        """Clear conversation history for a session"""
        try:
            logger.info(f"Clearing conversation for session {session_id}")
            if session_id in self.conversations:
                del self.conversations[session_id]
            self.current_context = {}
        except Exception as e:
            logger.error(f"Error clearing conversation: {str(e)}", exc_info=True)
            raise 