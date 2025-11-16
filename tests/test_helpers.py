"""
Test helper utilities for deployment testing.

This module provides common utilities for testing including environment
variable loading, port checking, URL validation, and test configuration.
"""

import os
import sys
import time
import subprocess
import socket
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable
from dotenv import load_dotenv
import requests


def get_project_root() -> Path:
    """Get the project root directory."""
    # This file is in tests/, so go up one level
    return Path(__file__).parent.parent.resolve()


def load_test_env(env_file: Optional[str] = None) -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Path to .env file (defaults to project root/.env)
        
    Returns:
        Dictionary of environment variables
    """
    project_root = get_project_root()
    
    if env_file:
        env_path = Path(env_file)
    else:
        env_path = project_root / ".env"
    
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at {env_path}")
    
    # Load environment variables
    load_dotenv(env_path, override=True)
    
    # Return as dict
    return dict(os.environ)


def get_server_port() -> int:
    """Get the server port from environment variables."""
    return int(os.getenv('WEB_PORT') or os.getenv('PORT') or '5000')


def get_webapp_port() -> int:
    """Get the webapp port from environment variables."""
    return int(os.getenv('WEBAPP_PORT') or '8080')


def get_server_url(port: Optional[int] = None) -> str:
    """Get the server URL."""
    if port is None:
        port = get_server_port()
    return f"http://localhost:{port}"


def get_webapp_url(port: Optional[int] = None) -> str:
    """Get the webapp URL."""
    if port is None:
        port = get_webapp_port()
    return f"http://localhost:{port}"


def check_port_available(port: int) -> bool:
    """
    Check if a port is available.
    
    Args:
        port: Port number to check
        
    Returns:
        True if port is available, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False


def wait_for_url(url: str, timeout: int = 30, interval: int = 2) -> bool:
    """
    Wait for a URL to become accessible.
    
    Args:
        url: URL to check
        timeout: Maximum time to wait in seconds
        interval: Time between checks in seconds
        
    Returns:
        True if URL becomes accessible, False otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code < 500:  # Accept any non-server-error status
                return True
        except (requests.RequestException, ConnectionError):
            pass
        time.sleep(interval)
    return False


def check_health_endpoint(base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Check the health endpoint and return status.
    
    Args:
        base_url: Base URL of the server (defaults to get_server_url())
        
    Returns:
        Dictionary with 'success' (bool) and 'data' (dict) or 'error' (str)
    """
    if base_url is None:
        base_url = get_server_url()
    
    health_url = f"{base_url}/health"
    
    try:
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        return {
            'success': True,
            'data': response.json(),
            'status_code': response.status_code
        }
    except requests.RequestException as e:
        return {
            'success': False,
            'error': str(e),
            'status_code': getattr(e.response, 'status_code', None)
        }


def authenticate(base_url: Optional[str] = None, 
                 username: Optional[str] = None, 
                 password: Optional[str] = None) -> Dict[str, Any]:
    """
    Authenticate and get JWT token.
    
    Args:
        base_url: Base URL of the server
        username: Username (defaults to ADMIN_USERNAME from env)
        password: Password (defaults to ADMIN_PASSWORD from env)
        
    Returns:
        Dictionary with 'success' (bool) and 'token' (str) or 'error' (str)
    """
    if base_url is None:
        base_url = get_server_url()
    
    if username is None:
        username = os.getenv('ADMIN_USERNAME', 'admin')
    if password is None:
        password = os.getenv('ADMIN_PASSWORD', '')
    
    login_url = f"{base_url}/auth/login"
    
    try:
        response = requests.post(
            login_url,
            json={'username': username, 'password': password},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return {
            'success': True,
            'token': data.get('access_token'),
            'data': data
        }
    except requests.RequestException as e:
        return {
            'success': False,
            'error': str(e),
            'status_code': getattr(e.response, 'status_code', None)
        }


def make_authenticated_request(method: str, url: str, token: str, **kwargs) -> requests.Response:
    """
    Make an authenticated HTTP request.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        url: URL to request
        token: JWT token
        **kwargs: Additional arguments to pass to requests
        
    Returns:
        Response object
    """
    headers = kwargs.pop('headers', {})
    headers['Authorization'] = f'Bearer {token}'
    return requests.request(method, url, headers=headers, **kwargs)


def check_cors_headers(response: requests.Response, origin: str) -> bool:
    """
    Check if CORS headers are present and correct.
    
    Args:
        response: Response object
        origin: Expected origin
        
    Returns:
        True if CORS headers are correct
    """
    access_control_allow_origin = response.headers.get('Access-Control-Allow-Origin')
    access_control_allow_credentials = response.headers.get('Access-Control-Allow-Credentials')
    
    # Check if origin is allowed (could be '*' or specific origin)
    if access_control_allow_origin == '*' or access_control_allow_origin == origin:
        return True
    
    # Check if origin is in comma-separated list
    if access_control_allow_origin and origin in access_control_allow_origin.split(','):
        return True
    
    return False


def validate_env_var(name: str, required: bool = True, 
                    validator: Optional[Callable[[str], Tuple[bool, Optional[str]]]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate an environment variable.
    
    Args:
        name: Environment variable name
        required: Whether the variable is required
        validator: Optional validator function that takes value and returns (bool, error_msg)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    value = os.getenv(name)
    
    if required and not value:
        return False, f"Required environment variable {name} is not set"
    
    if value and validator:
        is_valid, error = validator(value)
        if not is_valid:
            return False, f"Invalid value for {name}: {error}"
    
    return True, None


def validate_port(port_str: str) -> Tuple[bool, Optional[str]]:
    """Validate that a string is a valid port number."""
    try:
        port = int(port_str)
        if port < 1 or port > 65535:
            return False, f"Port {port} is out of range (1-65535)"
        return True, None
    except ValueError:
        return False, f"'{port_str}' is not a valid port number"


def validate_jwt_secret(secret: str) -> Tuple[bool, Optional[str]]:
    """Validate JWT secret key."""
    if not secret or secret == "your-secret-key-change-in-production":
        return False, "JWT_SECRET_KEY must be set to a non-default value"
    if len(secret) < 32:
        return False, "JWT_SECRET_KEY should be at least 32 characters"
    return True, None


def validate_openai_key(key: str) -> Tuple[bool, Optional[str]]:
    """Validate OpenAI API key format."""
    if not key or key == "your_openai_api_key_here":
        return False, "OPENAI_API_KEY appears to be a placeholder"
    if not key.startswith("sk-"):
        return False, "OpenAI API key should start with 'sk-'"
    return True, None


def validate_cors_origins(origins: str) -> Tuple[bool, Optional[str]]:
    """Validate CORS origins format."""
    if not origins:
        return False, "CORS_ORIGINS cannot be empty"
    
    # Split by comma and validate each origin
    origin_list = [o.strip() for o in origins.split(',')]
    for origin in origin_list:
        if not origin.startswith('http://') and not origin.startswith('https://'):
            return False, f"Invalid origin format: {origin} (must start with http:// or https://)"
    
    return True, None


def run_command(cmd: list, cwd: Optional[Path] = None, 
                timeout: Optional[int] = None, 
                capture_output: bool = True) -> Tuple[int, str, str]:
    """
    Run a shell command and return result.
    
    Args:
        cmd: Command as list of strings
        cwd: Working directory
        timeout: Timeout in seconds
        capture_output: Whether to capture stdout/stderr
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    if cwd is None:
        cwd = get_project_root()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            capture_output=capture_output,
            text=True
        )
        stdout = result.stdout if capture_output else ""
        stderr = result.stderr if capture_output else ""
        return result.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return -1, "", str(e)

