"""
Integration tests for core functionality.

This module tests the core functionality end-to-end using actual API calls.
These tests use real API keys from .env (not mocks) per project requirements.
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
    authenticate,
    make_authenticated_request,
    wait_for_url
)


class TestIntegration(unittest.TestCase):
    """Integration tests for core functionality."""
    
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
        
        # Authenticate once for all tests
        username = os.getenv('ADMIN_USERNAME', 'admin')
        password = os.getenv('ADMIN_PASSWORD', '')
        auth_result = authenticate(cls.base_url, username, password)
        
        if not auth_result['success']:
            raise unittest.SkipTest("Authentication failed. Check ADMIN_USERNAME and ADMIN_PASSWORD.")
        
        cls.token = auth_result['token']
    
    def setUp(self):
        """Set up for each test."""
        self.base_url = self.__class__.base_url
        self.token = self.__class__.token
    
    @unittest.skipUnless(
        os.getenv('OPENAI_API_KEY') and 
        os.getenv('OPENAI_API_KEY') != 'your_openai_api_key_here',
        "OpenAI API key not configured"
    )
    def test_openai_llm_api(self):
        """Test OpenAI LLM API with a minimal prompt."""
        response = make_authenticated_request(
            'POST',
            f"{self.base_url}/api/llm/openai",
            self.token,
            json={
                'prompt': 'Say "Hello, this is a test"',
                'model': 'gpt-3.5-turbo',
                'max_tokens': 50
            }
        )
        
        self.assertEqual(response.status_code, 200,
                        f"OpenAI LLM API failed: {response.text}")
        
        data = response.json()
        self.assertIn('response', data)
        self.assertIn('model', data)
        self.assertIsInstance(data['response'], str)
        self.assertGreater(len(data['response']), 0)
    
    def test_ollama_llm_api(self):
        """Test Ollama LLM API (if Ollama service is available)."""
        # First check if Ollama is available via health endpoint
        import requests
        health_response = requests.get(f"{self.base_url}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            ollama_status = health_data.get('components', {}).get('ollama_llm', '')
            if 'unavailable' in ollama_status.lower():
                self.skipTest("Ollama service is not available")
        
        # Try to get models first
        models_response = make_authenticated_request(
            'GET',
            f"{self.base_url}/api/llm/ollama/models",
            self.token
        )
        
        if models_response.status_code != 200:
            self.skipTest("Ollama service is not available")
        
        # Test with a simple prompt
        response = make_authenticated_request(
            'POST',
            f"{self.base_url}/api/llm/ollama",
            self.token,
            json={
                'prompt': 'Say "Hello"',
                'model': 'llama2',  # Common default model
                'max_tokens': 20
            }
        )
        
        # Accept 200 (success) or 503 (service unavailable)
        if response.status_code == 503:
            self.skipTest("Ollama LLM service is unavailable")
        
        self.assertEqual(response.status_code, 200,
                        f"Ollama LLM API failed: {response.text}")
        
        data = response.json()
        self.assertIn('response', data)
        self.assertIsInstance(data['response'], str)
    
    @unittest.skipUnless(
        os.getenv('HUGGINGFACE_TOKEN') and 
        os.getenv('HUGGINGFACE_TOKEN') != 'your_huggingface_token_here',
        "Hugging Face token not configured"
    )
    def test_image_generation_api(self):
        """Test image generation API with a small test image."""
        response = make_authenticated_request(
            'POST',
            f"{self.base_url}/api/image/generate",
            self.token,
            json={
                'prompt': 'A simple red circle on white background',
                'model': 'flux-dev',
                'width': 256,
                'height': 256,
                'steps': 10,  # Reduced for faster testing
                'guidance_scale': 7.5,
                'num_images': 1
            }
        )
        
        # Accept 200 (success) or 503 (service unavailable)
        if response.status_code == 503:
            self.skipTest("Image generation service is unavailable")
        
        self.assertEqual(response.status_code, 200,
                        f"Image generation API failed: {response.text}")
        
        data = response.json()
        self.assertIn('images', data)
        self.assertIsInstance(data['images'], list)
        self.assertGreater(len(data['images']), 0)
        
        # Check image data
        image_data = data['images'][0]
        self.assertIn('image_data', image_data)
        self.assertIn('image_path', image_data)
    
    def test_transcription_api(self):
        """Test transcription API with sample audio file."""
        project_root = Path(__file__).parent.parent
        sample_audio = project_root / "sample_data" / "m_hajibandeh_en_v1.mp3"
        
        if not sample_audio.exists():
            self.skipTest(f"Sample audio file not found at {sample_audio}")
        
        # Upload file for transcription
        with open(sample_audio, 'rb') as f:
            files = {'file': (sample_audio.name, f, 'audio/mpeg')}
            data = {
                'model_type': 'faster-whisper',
                'model_name': 'base',
                'language': 'auto',
                'task': 'transcribe'
            }
            
            import requests
            response = requests.post(
                f"{self.base_url}/api/transcribe",
                headers={'Authorization': f'Bearer {self.token}'},
                files=files,
                data=data,
                timeout=120  # Transcription can take time
            )
        
        # Accept 200 (success) or 503 (service unavailable)
        if response.status_code == 503:
            self.skipTest("Transcription service is unavailable")
        
        self.assertEqual(response.status_code, 200,
                        f"Transcription API failed: {response.text}")
        
        result = response.json()
        self.assertIn('text', result)
        self.assertIn('language', result)
        self.assertIn('model', result)
        self.assertIsInstance(result['text'], str)
    
    def test_ollama_models_list(self):
        """Test Ollama models list endpoint."""
        response = make_authenticated_request(
            'GET',
            f"{self.base_url}/api/llm/ollama/models",
            self.token
        )
        
        # Accept 200 (success) or 503 (service unavailable)
        if response.status_code == 503:
            self.skipTest("Ollama service is unavailable")
        
        self.assertEqual(response.status_code, 200,
                        f"Ollama models API failed: {response.text}")
        
        data = response.json()
        # Response should have models list or error info
        self.assertIn('success', data)
        self.assertIn('models', data)
    
    def test_image_models_list(self):
        """Test image models list endpoint."""
        response = make_authenticated_request(
            'GET',
            f"{self.base_url}/api/image/models",
            self.token
        )
        
        # Accept 200 (success) or 503 (service unavailable)
        if response.status_code == 503:
            self.skipTest("Image generation service is unavailable")
        
        self.assertEqual(response.status_code, 200,
                        f"Image models API failed: {response.text}")
        
        data = response.json()
        self.assertIn('models', data)
        self.assertIn('count', data)
        self.assertIsInstance(data['models'], list)


if __name__ == '__main__':
    unittest.main()

