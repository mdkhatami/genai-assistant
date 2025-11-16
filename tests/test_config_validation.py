"""
Configuration validation tests.

This module tests that the .env configuration file is valid and contains
all required settings with proper formats.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_helpers import (
    load_test_env,
    validate_env_var,
    validate_port,
    validate_jwt_secret,
    validate_openai_key,
    validate_cors_origins,
    check_port_available,
    get_project_root
)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = get_project_root()
        self.env_file = self.project_root / ".env"
        
        # Load environment variables
        if not self.env_file.exists():
            self.skipTest(".env file not found")
        
        load_test_env(str(self.env_file))
    
    def test_required_env_vars_present(self):
        """Test that all required environment variables are present."""
        required_vars = [
            'JWT_SECRET_KEY',
            'ADMIN_USERNAME',
            'ADMIN_PASSWORD'
        ]
        
        missing_vars = []
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
        
        self.assertEqual(len(missing_vars), 0, 
                        f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def test_jwt_secret_key_valid(self):
        """Test that JWT_SECRET_KEY is valid."""
        jwt_secret = os.getenv('JWT_SECRET_KEY', '')
        is_valid, error = validate_jwt_secret(jwt_secret)
        self.assertTrue(is_valid, f"JWT_SECRET_KEY validation failed: {error}")
    
    def test_admin_credentials_set(self):
        """Test that admin credentials are set (not default)."""
        username = os.getenv('ADMIN_USERNAME', '')
        password = os.getenv('ADMIN_PASSWORD', '')
        
        # Username should not be empty
        self.assertNotEqual(username, '', "ADMIN_USERNAME should be set")
        
        # Password should not be empty or default
        self.assertNotEqual(password, '', "ADMIN_PASSWORD should be set")
        self.assertNotEqual(password, 'change-this-password-in-production', 
                          "ADMIN_PASSWORD should be changed from default")
    
    def test_port_numbers_valid(self):
        """Test that port numbers are valid integers."""
        port_vars = ['WEB_PORT', 'PORT', 'WEBAPP_PORT', 'OLLAMA_PORT']
        
        for var in port_vars:
            port_str = os.getenv(var)
            if port_str:  # Optional vars might not be set
                is_valid, error = validate_port(port_str)
                self.assertTrue(is_valid, f"{var} validation failed: {error}")
    
    def test_cors_origins_valid(self):
        """Test that CORS_ORIGINS format is valid."""
        cors_origins = os.getenv('CORS_ORIGINS', '')
        
        if cors_origins:  # CORS_ORIGINS might have defaults
            is_valid, error = validate_cors_origins(cors_origins)
            self.assertTrue(is_valid, f"CORS_ORIGINS validation failed: {error}")
    
    def test_openai_key_format(self):
        """Test OpenAI API key format if present."""
        openai_key = os.getenv('OPENAI_API_KEY', '')
        
        if openai_key and openai_key != 'your_openai_api_key_here':
            is_valid, error = validate_openai_key(openai_key)
            self.assertTrue(is_valid, f"OPENAI_API_KEY validation failed: {error}")
    
    def test_port_availability(self):
        """Test that configured ports are available (or warn if in use)."""
        web_port = int(os.getenv('WEB_PORT') or os.getenv('PORT') or '5000')
        webapp_port = int(os.getenv('WEBAPP_PORT') or '8080')
        
        # Check if ports are available (warn if not, but don't fail)
        web_port_available = check_port_available(web_port)
        webapp_port_available = check_port_available(webapp_port)
        
        if not web_port_available:
            self.fail(f"WEB_PORT {web_port} is already in use")
        
        if not webapp_port_available:
            self.fail(f"WEBAPP_PORT {webapp_port} is already in use")
    
    def test_env_file_exists(self):
        """Test that .env file exists."""
        self.assertTrue(self.env_file.exists(), 
                       f".env file not found at {self.env_file}")


if __name__ == '__main__':
    unittest.main()

