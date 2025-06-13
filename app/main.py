from fastapi import FastAPI
from app.routers import upload, chat, analysis
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Flight Booking Chat Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/upload")
app.include_router(chat.router, prefix="/chat")
app.include_router(analysis.router, prefix="/analysis")
