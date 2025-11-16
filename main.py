#!/usr/bin/env python3
"""
GenAI Assistant FastAPI - Main Entry Point

This is the main entry point for the GenAI Assistant FastAPI application.
It imports and runs the FastAPI app from the app package.
"""

import os
import uvicorn
from dotenv import load_dotenv
from app.main import app

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get port from environment variable, default to 5000
    web_port = int(os.getenv('WEB_PORT') or os.getenv('PORT') or '5000')
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=web_port,
        reload=True,
        log_level="info"
    )
