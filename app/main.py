from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import analysis, upload, chat
from app.utils.logger_config import setup_logger
import os

# Setup logger
logger = setup_logger(__name__)

# Create necessary directories
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploaded")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Flight Data Analysis API",
    description="API for analyzing flight booking and airline data using LLM",
    version="1.0.0"
)

# Log application startup
logger.info("Starting Flight Booking Chat Analyzer application")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log middleware configuration
logger.info("CORS middleware configured")

# Mount static files directory
app.mount("/uploaded", StaticFiles(directory=UPLOAD_DIR), name="uploaded")

# Log middleware configuration
logger.info("Static files directory mounted")

# Include routers with correct prefixes
app.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])

# Log router configuration
logger.info("API routers configured: analysis, upload, chat")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Application startup initiated")
    logger.info(f"Upload directory: {UPLOAD_DIR}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Application shutdown initiated")

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
