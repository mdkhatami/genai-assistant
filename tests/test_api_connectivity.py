"""
API connectivity tests.

This module tests that the frontend can communicate with the backend,
including authentication, CORS, and protected endpoints.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_helpers import (
    load_test_env,
    get_server_url,
    check_health_endpoint,
    authenticate,
    make_authenticated_request,
    check_cors_headers,
    wait_for_url
)


class TestAPIConnectivity(unittest.TestCase):
    """Test API connectivity."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env"
        
        if not env_file.exists():
            raise unittest.SkipTest(".env file not found")
        
        load_test_env(str(env_file))
        cls.base_url = get_server_url()
        
        # Wait for server to be available
        if not wait_for_url(f"{cls.base_url}/health", timeout=10):
            raise unittest.SkipTest("Server is not available. Please start the server first.")
    
    def setUp(self):
        """Set up for each test."""
        self.base_url = self.__class__.base_url
    
    def test_health_endpoint(self):
        """Test that health endpoint is accessible without authentication."""
        result = check_health_endpoint(self.base_url)
        
        self.assertTrue(result['success'], 
                       f"Health endpoint failed: {result.get('error')}")
        self.assertIn('status', result['data'])
        self.assertIn('timestamp', result['data'])
    
    def test_health_endpoint_cors(self):
        """Test that health endpoint returns CORS headers."""
        import requests
        
        # Make request with origin header
        origin = f"http://localhost:{os.getenv('WEBAPP_PORT', '8080')}"
        response = requests.get(
            f"{self.base_url}/health",
            headers={'Origin': origin},
            timeout=10
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(check_cors_headers(response, origin),
                       "CORS headers are missing or incorrect")
    
    def test_authentication(self):
        """Test authentication endpoint."""
        username = os.getenv('ADMIN_USERNAME', 'admin')
        password = os.getenv('ADMIN_PASSWORD', '')
        
        result = authenticate(self.base_url, username, password)
        
        self.assertTrue(result['success'], 
                       f"Authentication failed: {result.get('error')}")
        self.assertIsNotNone(result['token'], "Token not returned")
        self.assertIsInstance(result['token'], str)
        self.assertGreater(len(result['token']), 0)
    
    def test_authentication_invalid_credentials(self):
        """Test authentication with invalid credentials."""
        result = authenticate(self.base_url, 'invalid_user', 'invalid_pass')
        
        self.assertFalse(result['success'], "Authentication should fail with invalid credentials")
        self.assertIsNone(result.get('token'))
    
    def test_protected_endpoint_without_token(self):
        """Test that protected endpoints require authentication."""
        import requests
        
        # Try to access protected endpoint without token
        response = requests.get(
            f"{self.base_url}/api/llm/ollama/models",
            timeout=10
        )
        
        self.assertEqual(response.status_code, 401, 
                        "Protected endpoint should return 401 without token")
    
    def test_protected_endpoint_with_token(self):
        """Test that protected endpoints work with valid token."""
        # Authenticate first
        username = os.getenv('ADMIN_USERNAME', 'admin')
        password = os.getenv('ADMIN_PASSWORD', '')
        auth_result = authenticate(self.base_url, username, password)
        
        self.assertTrue(auth_result['success'], "Authentication must succeed")
        token = auth_result['token']
        
        # Access protected endpoint
        response = make_authenticated_request(
            'GET',
            f"{self.base_url}/api/llm/ollama/models",
            token
        )
        
        self.assertEqual(response.status_code, 200, 
                       f"Protected endpoint failed: {response.text}")
    
    def test_cors_preflight_request(self):
        """Test CORS preflight (OPTIONS) request."""
        import requests
        
        origin = f"http://localhost:{os.getenv('WEBAPP_PORT', '8080')}"
        response = requests.options(
            f"{self.base_url}/api/llm/openai",
            headers={
                'Origin': origin,
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type,Authorization'
            },
            timeout=10
        )
        
        # Should return 200 or 204 for OPTIONS
        self.assertIn(response.status_code, [200, 204, 405], 
                     "CORS preflight should succeed")
        
        # Check CORS headers
        if response.status_code != 405:  # 405 means OPTIONS not supported
            access_control_allow_origin = response.headers.get('Access-Control-Allow-Origin')
            self.assertIsNotNone(access_control_allow_origin,
                               "Access-Control-Allow-Origin header missing")
    
    def test_port_configuration_change(self):
        """Test that port configuration changes don't break connectivity."""
        # This test verifies that the server URL is correctly constructed
        # from environment variables
        
        # Get current port
        current_port = int(os.getenv('WEB_PORT') or os.getenv('PORT') or '5000')
        current_url = get_server_url(current_port)
        
        # Verify URL is correct
        self.assertEqual(current_url, f"http://localhost:{current_port}")
        
        # Verify health endpoint works with current port
        result = check_health_endpoint(current_url)
        self.assertTrue(result['success'], 
                       "Health endpoint should work with configured port")


if __name__ == '__main__':
    unittest.main()

