"""
Logging configuration for FastAPI GenAI Assistant

Provides comprehensive logging for all API requests, responses, and system events.
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request, Response
from fastapi.logger import logger
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

# Create logger
api_logger = logging.getLogger("genai_api")

class APILogger:
    """Custom API logger for request/response logging."""
    
    def __init__(self):
        self.logger = api_logger
    
    def log_request(self, request: Request, user: Optional[str] = None):
        """Log incoming request details."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "user": user or "anonymous"
        }
        
        # Note: Cannot log request body here as request.body() is async and this method is not
        # Request body logging is handled in the middleware where async is available
        
        self.logger.info(f"REQUEST: {json.dumps(log_data, indent=2)}")
    
    def log_response(self, response: Response, response_data: Any, processing_time: float, user: Optional[str] = None):
        """Log response details."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "status_code": response.status_code,
            "processing_time_ms": round(processing_time * 1000, 2),
            "user": user or "anonymous"
        }
        
        # Log response data (limit size for large responses)
        if isinstance(response_data, dict):
            # Remove sensitive data
            safe_data = response_data.copy()
            if "access_token" in safe_data:
                safe_data["access_token"] = "***"
            log_data["response_data"] = safe_data
        elif isinstance(response_data, str):
            log_data["response_data"] = response_data[:1000]  # Limit size
        else:
            log_data["response_type"] = type(response_data).__name__
        
        self.logger.info(f"RESPONSE: {json.dumps(log_data, indent=2)}")
    
    def log_error(self, error: Exception, request: Optional[Request], user: Optional[str] = None):
        """Log error details."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "user": user or "anonymous"
        }
        
        if request:
            log_data.update({
                "method": request.method,
                "url": str(request.url)
            })
        
        self.logger.error(f"ERROR: {json.dumps(log_data, indent=2)}")
    
    def log_system_event(self, event: str, details: Dict[str, Any] = None):
        """Log system events."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "details": details or {}
        }
        
        self.logger.info(f"SYSTEM: {json.dumps(log_data, indent=2)}")

# Global logger instance
api_logger_instance = APILogger()

def get_api_logger() -> APILogger:
    """Get the global API logger instance."""
    return api_logger_instance 