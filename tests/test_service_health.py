"""
Service health tests.

This module tests the /health endpoint and verifies that all services
are properly initialized and accessible.
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
    wait_for_url
)


class TestServiceHealth(unittest.TestCase):
    """Test service health endpoint."""
    
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
    
    def test_health_endpoint_accessible(self):
        """Test that health endpoint is accessible."""
        result = check_health_endpoint(self.base_url)
        
        self.assertTrue(result['success'], 
                       f"Health endpoint failed: {result.get('error')}")
        self.assertEqual(result['status_code'], 200)
    
    def test_health_response_structure(self):
        """Test that health response has expected structure."""
        result = check_health_endpoint(self.base_url)
        
        self.assertTrue(result['success'])
        data = result['data']
        
        # Check required fields
        self.assertIn('status', data)
        self.assertIn('timestamp', data)
        self.assertIn('version', data)
        self.assertIn('components', data)
        
        # Check status is valid
        self.assertIn(data['status'], ['healthy', 'degraded', 'unavailable'])
    
    def test_health_components_present(self):
        """Test that health response includes component status."""
        result = check_health_endpoint(self.base_url)
        
        self.assertTrue(result['success'])
        components = result['data'].get('components', {})
        
        # Check for expected components
        expected_components = [
            'openai_llm',
            'ollama_llm',
            'image_generator',
            'transcriber'
        ]
        
        for component in expected_components:
            self.assertIn(component, components,
                         f"Component {component} not found in health response")
    
    def test_openai_llm_status(self):
        """Test OpenAI LLM component status."""
        result = check_health_endpoint(self.base_url)
        
        self.assertTrue(result['success'])
        components = result['data'].get('components', {})
        openai_status = components.get('openai_llm', '')
        
        # Status should be a string describing the state
        self.assertIsInstance(openai_status, str)
        self.assertGreater(len(openai_status), 0)
        
        # If API key is configured, should not be "unavailable"
        openai_key = os.getenv('OPENAI_API_KEY', '')
        if openai_key and openai_key != 'your_openai_api_key_here':
            # Should be healthy or have a specific status
            self.assertNotEqual(openai_status.lower(), 'unavailable',
                              "OpenAI LLM should be available if API key is configured")
    
    def test_ollama_llm_status(self):
        """Test Ollama LLM component status."""
        result = check_health_endpoint(self.base_url)
        
        self.assertTrue(result['success'])
        components = result['data'].get('components', {})
        ollama_status = components.get('ollama_llm', '')
        
        # Status should be a string
        self.assertIsInstance(ollama_status, str)
        self.assertGreater(len(ollama_status), 0)
        
        # Note: Ollama might be unavailable if service is not running
        # This is acceptable, we just check that status is reported
    
    def test_image_generator_status(self):
        """Test image generator component status."""
        result = check_health_endpoint(self.base_url)
        
        self.assertTrue(result['success'])
        components = result['data'].get('components', {})
        image_status = components.get('image_generator', '')
        
        # Status should be a string
        self.assertIsInstance(image_status, str)
        self.assertGreater(len(image_status), 0)
        
        # Should indicate GPU availability if CUDA is available
        if 'GPU' in image_status or 'CPU' in image_status:
            # Status includes device information
            pass
    
    def test_transcriber_status(self):
        """Test transcriber component status."""
        result = check_health_endpoint(self.base_url)
        
        self.assertTrue(result['success'])
        components = result['data'].get('components', {})
        transcriber_status = components.get('transcriber', '')
        
        # Status should be a string
        self.assertIsInstance(transcriber_status, str)
        self.assertGreater(len(transcriber_status), 0)
    
    def test_overall_status(self):
        """Test that overall status is reasonable."""
        result = check_health_endpoint(self.base_url)
        
        self.assertTrue(result['success'])
        status = result['data'].get('status', '')
        
        # Status should be one of the expected values
        self.assertIn(status, ['healthy', 'degraded', 'unavailable'])
        
        # If all components are unavailable, overall should be unavailable
        components = result['data'].get('components', {})
        unavailable_count = sum(1 for v in components.values() 
                               if 'unavailable' in str(v).lower())
        
        if unavailable_count == len(components) and len(components) > 0:
            self.assertEqual(status, 'unavailable',
                           "Overall status should be unavailable if all components are unavailable")
    
    def test_health_endpoint_performance(self):
        """Test that health endpoint responds quickly."""
        import time
        
        start_time = time.time()
        result = check_health_endpoint(self.base_url)
        elapsed = time.time() - start_time
        
        self.assertTrue(result['success'])
        # Health endpoint should respond quickly (within 5 seconds)
        self.assertLess(elapsed, 5.0,
                       f"Health endpoint took {elapsed:.2f}s, should be < 5s")


if __name__ == '__main__':
    unittest.main()

