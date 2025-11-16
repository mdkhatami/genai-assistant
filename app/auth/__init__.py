"""
Authentication package for GenAI Assistant FastAPI

Contains authentication and authorization functionality.
"""

from .auth import *

__all__ = [
    'authenticate_user', 'create_access_token', 'get_current_user', 'User'
]
