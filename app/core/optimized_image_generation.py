"""
Optimized Image Generation Core Module

This module provides high-performance image generation functionality using the FLUX.1-dev model
with proper GPU memory management, model persistence, and optimized resource usage.
"""

import os
import sys
import logging
import time
import gc
import atexit
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading

import torch
from diffusers import FluxPipeline
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImageGenerationResult:
    """Data class for standardized image generation result format."""
    image: Image.Image
    prompt: str
    model: str
    generation_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class OptimizedImageGenerator:
    """
    Optimized Image Generation using FLUX.1-dev model with GPU persistence.
    
    Features:
    - Model persistence (loads once, keeps in GPU memory)
    - Proper GPU memory management
    - Optimized for FLUX.1-dev model
    - Thread-safe operations
    - Automatic cleanup on shutdown
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self,
                 model_name: str = "black-forest-labs/FLUX.1-dev",
                 device: str = "cuda",
                 gpu_index: Optional[int] = None,
                 torch_dtype: torch.dtype = torch.bfloat16,
                 enable_cpu_offload: bool = False,
                 max_sequence_length: int = 512,
                 **kwargs):
        """
        Initialize Optimized Image Generator.
        
        Args:
            model_name: Model name (FLUX.1-dev)
            device: Device to run on (cuda recommended)
            gpu_index: Specific GPU index to use
            torch_dtype: Data type (bfloat16 recommended for FLUX)
            enable_cpu_offload: Whether to use CPU offloading (not recommended for speed)
            max_sequence_length: Maximum sequence length for text encoder
        """
        
        # Prevent re-initialization
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            logger.info("🚀 Initializing Optimized Image Generator...")
            
            self.model_name = model_name
            self.device = device
            self.gpu_index = gpu_index
            self.torch_dtype = torch_dtype
            self.enable_cpu_offload = enable_cpu_offload
            self.max_sequence_length = max_sequence_length
            
            # Model state
            self.pipeline = None
            self.is_loaded = False
            self.target_device = None
            
            # Performance tracking
            self.generation_count = 0
            self.total_generation_time = 0.0
            
            # Setup device
            self._setup_device()
            
            # Load model
            self._load_model()
            
            # Register cleanup
            atexit.register(self.cleanup)
            
            self._initialized = True
            logger.info("✅ Optimized Image Generator initialized successfully")
    
    def _setup_device(self):
        """Setup the target device for model execution."""
        # Check CUDA availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("⚠️  CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.gpu_index = None

        if self.device == "cuda":
            available_gpus = torch.cuda.device_count()
            logger.info(f"🔍 Available GPUs: {available_gpus}")

            # Use specified GPU index if valid, otherwise use first available
            if self.gpu_index is not None:
                if self.gpu_index < available_gpus:
                    self.target_device = f"cuda:{self.gpu_index}"
                    torch.cuda.set_device(self.gpu_index)
                    gpu_name = torch.cuda.get_device_name(self.gpu_index)
                    logger.info(f"🎯 Using GPU {self.gpu_index}: {gpu_name}")
                else:
                    logger.warning(f"⚠️  GPU {self.gpu_index} not available (only {available_gpus} GPUs), using cuda:0")
                    self.target_device = "cuda:0"
                    self.gpu_index = 0
                    torch.cuda.set_device(0)
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"🎯 Using GPU 0: {gpu_name}")
            else:
                # No specific GPU requested, use first available
                self.target_device = "cuda:0"
                self.gpu_index = 0
                torch.cuda.set_device(0)
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🎯 Using GPU 0: {gpu_name}")
        else:
            # CPU mode
            self.target_device = "cpu"
            logger.info(f"🎯 Using device: CPU")

        # Clear GPU memory before starting
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("🧹 Cleared GPU memory cache")
    
    def _load_model(self):
        """Load the FLUX.1-dev model with optimal settings."""
        try:
            logger.info(f"📥 Loading FLUX.1-dev model: {self.model_name}")
            start_time = time.time()
            
            # Get Hugging Face token
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            
            # Optimal settings for FLUX.1-dev
            pipeline_kwargs = {
                "torch_dtype": self.torch_dtype,
                "use_safetensors": True,
                "token": hf_token,
                "max_sequence_length": self.max_sequence_length,
                "device_map": None,  # We'll handle device placement manually
            }
            
            # Load the pipeline
            self.pipeline = FluxPipeline.from_pretrained(
                self.model_name,
                **pipeline_kwargs
            )
            
            # Move to target device
            logger.info(f"📱 Moving model to {self.target_device}")
            self.pipeline = self.pipeline.to(self.target_device)
            
            # Apply optimizations (but NOT CPU offloading for maximum speed)
            if not self.enable_cpu_offload:
                logger.info("⚡ Keeping model on GPU for maximum speed")
                
                # Enable memory efficient attention if available
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("✅ Enabled xFormers memory efficient attention")
                except Exception as e:
                    logger.info("ℹ️  xFormers not available, using default attention")
                
                # Enable VAE slicing for memory efficiency
                try:
                    self.pipeline.enable_vae_slicing()
                    logger.info("✅ Enabled VAE slicing")
                except Exception as e:
                    logger.warning(f"Could not enable VAE slicing: {e}")
            
            else:
                # Only use CPU offloading if explicitly requested (slower but uses less VRAM)
                logger.info("🔄 Enabling CPU offloading (slower but memory efficient)")
                self.pipeline.enable_model_cpu_offload()
            
            # Compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self.device == "cuda":
                try:
                    logger.info("🔥 Compiling model for faster inference...")
                    self.pipeline.transformer = torch.compile(
                        self.pipeline.transformer, 
                        mode="reduce-overhead",
                        fullgraph=True
                    )
                    logger.info("✅ Model compiled successfully")
                except Exception as e:
                    logger.info(f"ℹ️  Could not compile model: {e}")
            
            load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"🎉 Model loaded successfully in {load_time:.2f} seconds")
            
            # Log GPU memory usage
            if self.device == "cuda":
                self._log_gpu_memory()
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Failed to load FLUX.1-dev model: {e}")
    
    def _log_gpu_memory(self):
        """Log current GPU memory usage."""
        if self.device == "cuda":
            try:
                # Get the logical device index
                logical_device = int(self.target_device.split(':')[1]) if ':' in self.target_device else 0
                with torch.cuda.device(logical_device):
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    physical_gpu = self.gpu_index if self.gpu_index is not None else "unknown"
                    logger.info(f"📊 GPU {physical_gpu} (logical {logical_device}) Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            except Exception as e:
                logger.warning(f"Could not log GPU memory: {e}")
    
    def generate_image(self,
                      prompt: str,
                      width: int = 1024,
                      height: int = 1024,
                      num_inference_steps: int = 4,
                      guidance_scale: float = 0.0,
                      num_images: int = 1,
                      seed: Optional[int] = None,
                      max_sequence_length: Optional[int] = None,
                      **kwargs) -> Union[ImageGenerationResult, List[ImageGenerationResult]]:
        """
        Generate image(s) from text prompt using FLUX.1-dev.
        
        Args:
            prompt: Text description of the image
            width: Image width (default 1024 for FLUX)
            height: Image height (default 1024 for FLUX)
            num_inference_steps: Number of denoising steps (4 is optimal for FLUX)
            guidance_scale: Guidance scale (0.0 is optimal for FLUX.1-dev)
            num_images: Number of images to generate (default 1)
            seed: Random seed for reproducibility
            max_sequence_length: Override default max sequence length
            
        Returns:
            ImageGenerationResult or List[ImageGenerationResult] containing the generated image(s)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Please initialize the generator first.")
        
        start_time = time.time()
        
        try:
            logger.info(f"🎨 Generating {num_images} image{'s' if num_images > 1 else ''}: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if self.device == "cuda":
                    torch.cuda.manual_seed_all(seed)  # Set for all CUDA devices

            # Generate parameters
            generation_kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images,
                "generator": torch.Generator(device=self.target_device).manual_seed(seed) if seed else None,
            }

            # Add max_sequence_length if specified
            if max_sequence_length:
                generation_kwargs["max_sequence_length"] = max_sequence_length

            # Add any additional kwargs
            generation_kwargs.update(kwargs)

            # Generate image
            with torch.no_grad():
                result = self.pipeline(**generation_kwargs)
            
            generation_time = time.time() - start_time
            
            # Update statistics
            self.generation_count += 1
            self.total_generation_time += generation_time
            
            logger.info(f"✅ {num_images} image{'s' if num_images > 1 else ''} generated in {generation_time:.2f}s (avg: {self.total_generation_time/self.generation_count:.2f}s)")
            
            # Create result object(s)
            if num_images == 1:
                image_result = ImageGenerationResult(
                    image=result.images[0],
                    prompt=prompt,
                    model=self.model_name,
                    generation_time=generation_time,
                    metadata={
                        "width": width,
                        "height": height,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "num_images": num_images,
                        "seed": seed,
                        "device": self.target_device,
                        "generation_count": self.generation_count
                    }
                )
                return image_result
            else:
                # Multiple images
                results = []
                for i, image in enumerate(result.images):
                    image_result = ImageGenerationResult(
                        image=image,
                        prompt=prompt,
                        model=self.model_name,
                        generation_time=generation_time,
                        metadata={
                            "width": width,
                            "height": height,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                            "num_images": num_images,
                            "image_index": i,
                            "seed": seed,
                            "device": self.target_device,
                            "generation_count": self.generation_count
                        }
                    )
                    results.append(image_result)
                return results
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ Image generation failed after {generation_time:.2f}s: {e}")
            
            error_result = ImageGenerationResult(
                image=None,
                prompt=prompt,
                model=self.model_name,
                generation_time=generation_time,
                error=str(e)
            )
            
            if num_images == 1:
                return error_result
            else:
                return [error_result] * num_images
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.target_device,
            "is_loaded": self.is_loaded,
            "torch_dtype": str(self.torch_dtype),
            "generation_count": self.generation_count,
            "average_generation_time": self.total_generation_time / max(self.generation_count, 1),
            "enable_cpu_offload": self.enable_cpu_offload,
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return [
            "flux",
            "flux-dev", 
            "flux-dev-8bit",
            "flux-dev-4bit"
        ]
    
    def cleanup(self):
        """Clean up GPU memory and resources."""
        logger.info("🧹 Cleaning up GPU resources...")

        try:
            if self.pipeline is not None:
                # Move pipeline to CPU to free GPU memory
                if self.device == "cuda":
                    self.pipeline = self.pipeline.to("cpu")

                # Delete pipeline
                del self.pipeline
                self.pipeline = None

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Also collect IPC memory
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()

            self.is_loaded = False
            logger.info("✅ GPU resources cleaned up successfully")

        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, 'is_loaded') and self.is_loaded:
            self.cleanup()


# Global instance
_generator_instance = None
_generator_lock = threading.Lock()


def get_optimized_generator(
    model_name: str = "black-forest-labs/FLUX.1-dev",
    device: str = "cuda", 
    gpu_index: Optional[int] = None,
    **kwargs
) -> OptimizedImageGenerator:
    """
    Get the optimized image generator instance (singleton).
    
    Args:
        model_name: Model name
        device: Device to use
        gpu_index: GPU index
        **kwargs: Additional arguments
        
    Returns:
        OptimizedImageGenerator instance
    """
    global _generator_instance
    
    if _generator_instance is None:
        with _generator_lock:
            if _generator_instance is None:
                _generator_instance = OptimizedImageGenerator(
                    model_name=model_name,
                    device=device,
                    gpu_index=gpu_index,
                    **kwargs
                )
    
    return _generator_instance


def cleanup_generator():
    """Clean up the global generator instance."""
    global _generator_instance
    
    if _generator_instance is not None:
        with _generator_lock:
            if _generator_instance is not None:
                _generator_instance.cleanup()
                _generator_instance = None


# Register cleanup on module exit
atexit.register(cleanup_generator)
