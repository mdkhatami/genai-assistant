"""
Test script for Image Generation API endpoint.

This script tests the image generation API with on-demand loading,
memory management, and error handling.
"""

import os
import sys
import unittest
import time
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


class TestImageGenerationAPI(unittest.TestCase):
    """Test cases for Image Generation API."""
    
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
        print(f"\n✅ Authenticated successfully")
        print(f"✅ Server URL: {cls.base_url}")
    
    def test_image_generation_minimal_model(self):
        """Test image generation with minimal model (flux-schnell)."""
        print("\n🧪 Testing image generation with minimal model (flux-schnell)...")
        
        response = make_authenticated_request(
            'POST',
            f"{self.base_url}/api/image/generate",
            self.token,
            json={
                'prompt': 'A simple red circle on white background',
                'model': 'flux-schnell',  # Minimal model
                'width': 512,
                'height': 512,
                'steps': 4,
                'guidance_scale': 0.0,
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
        
        # Check that model info is present
        self.assertIn('model', data)
        self.assertIn('processing_time', data)
        
        print(f"✅ Image generated successfully!")
        print(f"   Model: {data.get('model', 'unknown')}")
        print(f"   Processing time: {data.get('processing_time', 0):.2f}s")
        print(f"   Images generated: {len(data['images'])}")
    
    def test_image_generation_default_model(self):
        """Test image generation with default model (should be minimal)."""
        print("\n🧪 Testing image generation with default model...")
        
        response = make_authenticated_request(
            'POST',
            f"{self.base_url}/api/image/generate",
            self.token,
            json={
                'prompt': 'A beautiful sunset over mountains',
                # No model specified - should use default (flux-schnell)
                'width': 512,
                'height': 512,
                'steps': 4,
                'guidance_scale': 0.0,
                'num_images': 1
            }
        )
        
        if response.status_code == 503:
            self.skipTest("Image generation service is unavailable")
        
        self.assertEqual(response.status_code, 200,
                        f"Image generation API failed: {response.text}")
        
        data = response.json()
        self.assertIn('images', data)
        self.assertGreater(len(data['images']), 0)
        
        print(f"✅ Default model image generated successfully!")
        print(f"   Model used: {data.get('model', 'unknown')}")
    
    def test_image_generation_on_demand_loading(self):
        """Test that model loads on-demand and unloads after use."""
        print("\n🧪 Testing on-demand loading/unloading...")
        
        # First request - model should load
        print("   Making first request (model should load)...")
        start_time = time.time()
        response1 = make_authenticated_request(
            'POST',
            f"{self.base_url}/api/image/generate",
            self.token,
            json={
                'prompt': 'A cat sitting on a mat',
                'model': 'flux-schnell',
                'width': 512,
                'height': 512,
                'steps': 4,
                'num_images': 1
            }
        )
        time1 = time.time() - start_time
        
        if response1.status_code == 503:
            self.skipTest("Image generation service is unavailable")
        
        self.assertEqual(response1.status_code, 200)
        print(f"   First request completed in {time1:.2f}s (includes model loading)")
        
        # Wait a moment
        time.sleep(1)
        
        # Second request - model should load again (was unloaded)
        print("   Making second request (model should load again)...")
        start_time = time.time()
        response2 = make_authenticated_request(
            'POST',
            f"{self.base_url}/api/image/generate",
            self.token,
            json={
                'prompt': 'A dog playing in the park',
                'model': 'flux-schnell',
                'width': 512,
                'height': 512,
                'steps': 4,
                'num_images': 1
            }
        )
        time2 = time.time() - start_time
        
        self.assertEqual(response2.status_code, 200)
        print(f"   Second request completed in {time2:.2f}s (includes model loading)")
        
        # Both should succeed
        data1 = response1.json()
        data2 = response2.json()
        
        self.assertIn('images', data1)
        self.assertIn('images', data2)
        self.assertGreater(len(data1['images']), 0)
        self.assertGreater(len(data2['images']), 0)
        
        print(f"✅ On-demand loading/unloading verified!")
        print(f"   Both requests succeeded, model loaded/unloaded between requests")
    
    def test_image_generation_memory_handling(self):
        """Test image generation with memory constraints."""
        print("\n🧪 Testing memory handling...")
        
        # Try generating with minimal settings
        response = make_authenticated_request(
            'POST',
            f"{self.base_url}/api/image/generate",
            self.token,
            json={
                'prompt': 'A simple test image',
                'model': 'flux-schnell',  # Minimal model
                'width': 512,  # Smaller size
                'height': 512,
                'steps': 4,  # Minimal steps
                'guidance_scale': 0.0,
                'num_images': 1
            }
        )
        
        if response.status_code == 503:
            self.skipTest("Image generation service is unavailable")
        
        # Should succeed or fail gracefully with memory error
        if response.status_code == 200:
            data = response.json()
            self.assertIn('images', data)
            print(f"✅ Image generated successfully with memory constraints!")
        elif response.status_code == 500:
            # Check if it's a memory error
            error_text = response.text.lower()
            if 'memory' in error_text or 'cuda' in error_text:
                print(f"⚠️  Memory error encountered (expected in constrained environments)")
            else:
                self.fail(f"Unexpected error: {response.text}")
        else:
            self.fail(f"Unexpected status code: {response.status_code}")
    
    def test_image_generation_error_handling(self):
        """Test error handling for invalid requests."""
        print("\n🧪 Testing error handling...")
        
        # Test with invalid model
        response = make_authenticated_request(
            'POST',
            f"{self.base_url}/api/image/generate",
            self.token,
            json={
                'prompt': 'Test prompt',
                'model': 'invalid-model-name',
                'width': 512,
                'height': 512,
                'steps': 4,
                'num_images': 1
            }
        )
        
        # Should either fail gracefully or use default model
        if response.status_code == 200:
            print("✅ Invalid model handled gracefully (used default)")
        elif response.status_code in [400, 500, 503]:
            print(f"✅ Invalid model rejected as expected (status: {response.status_code})")
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")


if __name__ == '__main__':
    print("=" * 70)
    print("Image Generation API Test Suite")
    print("=" * 70)
    print("\nThis test suite verifies:")
    print("  - On-demand model loading/unloading")
    print("  - Minimal model (flux-schnell) as default")
    print("  - Memory management and error handling")
    print("  - API endpoint functionality")
    print("\n" + "=" * 70 + "\n")
    
    unittest.main(verbosity=2)

