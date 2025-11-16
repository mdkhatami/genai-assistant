#!/usr/bin/env python3
"""
Test API endpoint for image generation to verify memory optimization works.
"""

import requests
import json
import sys
import time
from pathlib import Path

# Configuration
API_URL = "http://localhost:5000"
USERNAME = "admin"
PASSWORD = "SecureAdmin@2024!"  # Update if different

def login():
    """Login and get auth token."""
    response = requests.post(
        f"{API_URL}/auth/login",
        json={"username": USERNAME, "password": PASSWORD}
    )
    if response.status_code != 200:
        print(f"❌ Login failed: {response.status_code}")
        print(response.text)
        return None
    return response.json()["access_token"]

def test_image_generation(token):
    """Test image generation via API."""
    print("=" * 80)
    print("API Image Generation Test")
    print("=" * 80)
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": "a beautiful sunset over mountains, high quality",
        "model": "flux-schnell",
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance_scale": 0.0,
        "num_images": 1
    }
    
    print(f"\n📤 Sending request:")
    print(f"   Model: {payload['model']}")
    print(f"   Prompt: {payload['prompt']}")
    print(f"   Resolution: {payload['width']}x{payload['height']}")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/api/image/generate",
            headers=headers,
            json=payload,
            timeout=120  # 2 minute timeout
        )
        elapsed = time.time() - start_time
        
        print(f"\n📥 Response received in {elapsed:.2f}s")
        print(f"   Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Request failed: {response.text}")
            return False
        
        data = response.json()
        
        print(f"\n✅ Request successful!")
        print(f"   Model used: {data.get('model', 'unknown')}")
        print(f"   Processing time: {data.get('processing_time', 0):.2f}s")
        print(f"   Images generated: {len(data.get('images', []))}")
        
        # Check images
        images = data.get('images', [])
        if not images:
            print("❌ No images in response")
            return False
        
        for i, img_data in enumerate(images):
            if img_data.get('error'):
                print(f"❌ Image {i+1} error: {img_data['error']}")
                return False
            
            has_data = bool(img_data.get('image_data'))
            print(f"   Image {i+1}: {'✅ Has image data' if has_data else '❌ No image data'}")
            if has_data:
                data_len = len(img_data['image_data'])
                print(f"      Data length: {data_len} chars")
                print(f"      Generation time: {img_data.get('generation_time', 0):.2f}s")
        
        print(f"\n✅ SUCCESS: Image generation via API works!")
        return True
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>120s)")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_endpoint(token):
    """Test memory info endpoint."""
    print("\n" + "=" * 80)
    print("Memory Info Endpoint Test")
    print("=" * 80)
    
    headers = {
        "Authorization": f"Bearer {token}",
    }
    
    try:
        response = requests.get(
            f"{API_URL}/api/image/memory",
            headers=headers
        )
        
        if response.status_code != 200:
            print(f"❌ Request failed: {response.status_code}")
            return False
        
        data = response.json()
        
        print("\n📊 Memory Information:")
        print(f"   Image Generator:")
        print(f"      Model: {data.get('image_generator', {}).get('model_name', 'unknown')}")
        print(f"      Is Loaded: {data.get('image_generator', {}).get('is_loaded', False)}")
        print(f"      CPU Offload: {data.get('image_generator', {}).get('cpu_offload', False)}")
        
        gpu_memory = data.get('gpu_memory', {})
        if gpu_memory.get('cuda_available'):
            print(f"   GPU Memory:")
            print(f"      Total: {gpu_memory.get('total_gb', 0):.2f}GB")
            print(f"      Allocated: {gpu_memory.get('allocated_gb', 0):.2f}GB")
            print(f"      Reserved: {gpu_memory.get('reserved_gb', 0):.2f}GB")
            print(f"      Available: {gpu_memory.get('available_gb', 0):.2f}GB")
            print(f"      Utilization: {gpu_memory.get('utilization_percent', 0):.1f}%")
        
        transcription = data.get('transcription_cache', {})
        if 'total_cached' in transcription:
            print(f"   Transcription Cache:")
            print(f"      Total cached models: {transcription.get('total_cached', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing API endpoints...")
    
    # Login
    token = login()
    if not token:
        sys.exit(1)
    
    print("✅ Login successful\n")
    
    # Test memory endpoint first
    test_memory_endpoint(token)
    
    # Test image generation
    success = test_image_generation(token)
    
    # Test memory endpoint after generation
    print("\n")
    test_memory_endpoint(token)
    
    sys.exit(0 if success else 1)

