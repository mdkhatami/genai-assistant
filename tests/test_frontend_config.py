"""
Frontend configuration tests.

This module tests that the frontend configuration is correct and
can communicate with the backend.
"""

import os
import sys
import unittest
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_helpers import (
    load_test_env,
    get_server_url,
    get_webapp_url,
    get_server_port,
    get_webapp_port
)


class TestFrontendConfig(unittest.TestCase):
    """Test frontend configuration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env"
        
        if not env_file.exists():
            raise unittest.SkipTest(".env file not found")
        
        load_test_env(str(env_file))
        cls.project_root = project_root
        cls.config_file = project_root / "standalone_webapp" / "config.js"
    
    def test_config_file_exists(self):
        """Test that config.js file exists."""
        self.assertTrue(self.config_file.exists(), 
                      f"config.js not found at {self.config_file}")
    
    def test_config_file_readable(self):
        """Test that config.js can be read."""
        try:
            content = self.config_file.read_text()
            self.assertGreater(len(content), 0, "config.js is empty")
        except Exception as e:
            self.fail(f"Failed to read config.js: {e}")
    
    def test_config_structure(self):
        """Test that config.js has expected structure."""
        content = self.config_file.read_text()
        
        # Check for key configuration objects
        self.assertIn('CONFIG', content, "CONFIG object not found")
        self.assertIn('connectionPresets', content, "connectionPresets not found")
        self.assertIn('server', content, "server configuration not found")
    
    def test_cors_configuration_match(self):
        """Test that CORS configuration matches between frontend and backend."""
        # Get CORS origins from .env
        cors_origins = os.getenv('CORS_ORIGINS', '')
        webapp_port = get_webapp_port()
        
        # Expected origin
        expected_origin = f"http://localhost:{webapp_port}"
        
        # Check if expected origin is in CORS_ORIGINS
        if cors_origins:
            cors_list = [o.strip() for o in cors_origins.split(',')]
            self.assertIn(expected_origin, cors_list,
                         f"Frontend origin {expected_origin} not in CORS_ORIGINS")
    
    def test_port_configuration_consistency(self):
        """Test that port configuration is consistent."""
        server_port = get_server_port()
        webapp_port = get_webapp_port()
        
        # Ports should be different
        self.assertNotEqual(server_port, webapp_port,
                          "Server and webapp ports should be different")
        
        # Ports should be valid
        self.assertGreater(server_port, 0)
        self.assertLess(server_port, 65536)
        self.assertGreater(webapp_port, 0)
        self.assertLess(webapp_port, 65536)
    
    def test_frontend_can_detect_backend_url(self):
        """Test that frontend can construct backend URL."""
        server_port = get_server_port()
        expected_url = f"http://localhost:{server_port}"
        actual_url = get_server_url()
        
        self.assertEqual(actual_url, expected_url,
                        "Backend URL construction is incorrect")
    
    def test_webapp_directory_exists(self):
        """Test that standalone_webapp directory exists."""
        webapp_dir = self.project_root / "standalone_webapp"
        self.assertTrue(webapp_dir.exists(), 
                       f"standalone_webapp directory not found at {webapp_dir}")
        self.assertTrue(webapp_dir.is_dir(), 
                       "standalone_webapp is not a directory")
    
    def test_webapp_index_exists(self):
        """Test that index.html exists in webapp directory."""
        index_file = self.project_root / "standalone_webapp" / "index.html"
        self.assertTrue(index_file.exists(), 
                      f"index.html not found at {index_file}")
    
    def test_config_export_format(self):
        """Test that config.js exports configuration correctly."""
        content = self.config_file.read_text()
        
        # Check for module.exports or window assignment
        has_module_export = 'module.exports' in content or 'module.exports =' in content
        has_window_assign = 'window.GenAIConfig' in content or 'window.GenAIConfig =' in content
        
        self.assertTrue(has_module_export or has_window_assign,
                      "config.js should export CONFIG via module.exports or window.GenAIConfig")


if __name__ == '__main__':
    unittest.main()

