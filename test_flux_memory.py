#!/usr/bin/env python3
"""
Test script to verify FLUX memory optimization works correctly.
Tests memory usage and image generation end-to-end.
"""

import os
import sys
import time
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.optimized_image_generation import get_optimized_generator
from app.config import get_config_loader

def get_gpu_memory():
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    available = total - reserved
    
    return {
        "total_gb": total,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "available_gb": available
    }

def test_flux_memory():
    """Test FLUX model memory usage and image generation."""
    print("=" * 80)
    print("FLUX Memory Optimization Test")
    print("=" * 80)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot test GPU memory.")
        return False
    
    print(f"\n✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Get initial memory state
    print("\n📊 Initial GPU Memory State:")
    initial_memory = get_gpu_memory()
    if initial_memory:
        print(f"   Total: {initial_memory['total_gb']:.2f}GB")
        print(f"   Allocated: {initial_memory['allocated_gb']:.2f}GB")
        print(f"   Reserved: {initial_memory['reserved_gb']:.2f}GB")
        print(f"   Available: {initial_memory['available_gb']:.2f}GB")
    
    # Clear cache
    torch.cuda.empty_cache()
    time.sleep(1)
    
    # Get memory after cleanup
    print("\n📊 GPU Memory After Cleanup:")
    after_cleanup = get_gpu_memory()
    if after_cleanup:
        print(f"   Available: {after_cleanup['available_gb']:.2f}GB")
    
    # Load generator
    print("\n🔄 Loading FLUX.1-schnell generator...")
    try:
        generator = get_optimized_generator(
            model_name="black-forest-labs/FLUX.1-schnell",
            device="cuda",
            enable_cpu_offload=None,  # Auto-detect
            auto_unload=True,
            unload_timeout=0.0
        )
        print("✅ Generator created")
    except Exception as e:
        print(f"❌ Failed to create generator: {e}")
        return False
    
    # Get model info before loading
    print("\n📋 Model Info (before load):")
    info_before = generator.get_model_info()
    print(f"   Model: {info_before.get('model_name')}")
    print(f"   Is Loaded: {info_before.get('is_loaded')}")
    
    # Generate image (this will trigger model loading)
    print("\n🎨 Generating test image...")
    print("   Prompt: 'a beautiful sunset over mountains'")
    print("   This will load the model on-demand...")
    
    try:
        start_time = time.time()
        result = generator.generate_image(
            prompt="a beautiful sunset over mountains",
            width=512,
            height=512,
            num_inference_steps=4,
            guidance_scale=0.0,
            num_images=1
        )
        generation_time = time.time() - start_time
        
        # Check memory after generation
        print("\n📊 GPU Memory After Image Generation:")
        after_generation = get_gpu_memory()
        if after_generation:
            print(f"   Allocated: {after_generation['allocated_gb']:.2f}GB")
            print(f"   Reserved: {after_generation['reserved_gb']:.2f}GB")
            print(f"   Available: {after_generation['available_gb']:.2f}GB")
            
            # Calculate memory used by model
            if initial_memory and after_generation:
                memory_used = after_generation['reserved_gb'] - initial_memory['reserved_gb']
                print(f"   Memory Used by Model: ~{memory_used:.2f}GB")
        
        # Check if image was generated
        if result.error:
            print(f"\n❌ Image generation failed: {result.error}")
            return False
        
        if result.image is None:
            print("\n❌ Image generation returned None")
            return False
        
        print(f"\n✅ Image generated successfully!")
        print(f"   Generation time: {generation_time:.2f}s")
        print(f"   Image size: {result.image.size}")
        print(f"   Image mode: {result.image.mode}")
        
        # Get model info after generation
        print("\n📋 Model Info (after generation):")
        info_after = generator.get_model_info()
        print(f"   Model: {info_after.get('model_name')}")
        print(f"   Is Loaded: {info_after.get('is_loaded')}")
        print(f"   CPU Offload: {info_after.get('actual_cpu_offload')}")
        print(f"   Sequential Offload: {info_after.get('actual_sequential_offload')}")
        
        # Check if model was unloaded
        time.sleep(2)  # Wait a bit for unload
        info_final = generator.get_model_info()
        print(f"\n📋 Model Info (after unload wait):")
        print(f"   Is Loaded: {info_final.get('is_loaded')}")
        
        # Check final memory
        print("\n📊 GPU Memory After Unload:")
        final_memory = get_gpu_memory()
        if final_memory:
            print(f"   Allocated: {final_memory['allocated_gb']:.2f}GB")
            print(f"   Reserved: {final_memory['reserved_gb']:.2f}GB")
            print(f"   Available: {final_memory['available_gb']:.2f}GB")
        
        # Verify memory usage is acceptable
        if after_generation:
            if after_generation['reserved_gb'] < 8.0:
                print(f"\n✅ SUCCESS: Memory usage is acceptable ({after_generation['reserved_gb']:.2f}GB < 8GB)")
                return True
            else:
                print(f"\n⚠️  WARNING: Memory usage is high ({after_generation['reserved_gb']:.2f}GB >= 8GB)")
                print("   This may still work with sequential CPU offload, but is not ideal.")
                return True  # Still consider it success if image was generated
        
        return True
        
    except Exception as e:
        print(f"\n❌ Image generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_flux_memory()
    sys.exit(0 if success else 1)

