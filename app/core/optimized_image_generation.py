"""
Optimized Image Generation Core Module

This module provides high-performance image generation functionality using the FLUX.1-dev model
with on-demand loading/unloading, proper GPU memory management, and optimized resource usage.
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

# Set PyTorch CUDA allocator config for better memory management
if os.getenv('PYTORCH_CUDA_ALLOC_CONF') is None:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


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
    Optimized Image Generation using FLUX.1-dev model with on-demand loading.
    
    Features:
    - On-demand model loading (loads when first request comes)
    - Automatic model unloading after use
    - Proper GPU memory management
    - Memory-aware loading with CPU offload fallback
    - Optimized for FLUX.1-dev model
    - Thread-safe operations
    - Automatic cleanup on shutdown
    """
    
    def __init__(self,
                 model_name: str = "black-forest-labs/FLUX.1-schnell",
                 device: str = "cuda",
                 gpu_index: Optional[int] = None,
                 torch_dtype: torch.dtype = torch.bfloat16,
                 enable_cpu_offload: Optional[bool] = None,  # None = auto-detect
                 max_sequence_length: int = 512,
                 auto_unload: bool = True,
                 unload_timeout: float = 0.0,  # 0 = unload immediately
                 **kwargs):
        """
        Initialize Optimized Image Generator (on-demand loading).
        
        Args:
            model_name: Model name (default: FLUX.1-schnell - minimal model)
            device: Device to run on (cuda recommended)
            gpu_index: Specific GPU index to use
            torch_dtype: Data type (bfloat16 recommended for FLUX)
            enable_cpu_offload: Whether to use CPU offloading (None = auto-detect based on memory)
            max_sequence_length: Maximum sequence length for text encoder
            auto_unload: Whether to automatically unload model after use
            unload_timeout: Seconds to keep model loaded after last use (0 = immediate unload)
        """
        
        logger.info("🚀 Initializing Optimized Image Generator (on-demand loading)...")
        
        self.model_name = model_name
        self.device = device
        self.gpu_index = gpu_index
        self.torch_dtype = torch_dtype
        self.enable_cpu_offload = enable_cpu_offload
        self.max_sequence_length = max_sequence_length
        self.auto_unload = auto_unload
        self.unload_timeout = unload_timeout
        
        # Model state
        self.pipeline = None
        self.is_loaded = False
        self.target_device = None
        self._load_lock = threading.Lock()
        self._last_use_time = None
        self._actual_cpu_offload = False  # Track actual CPU offload state
        self._actual_sequential_offload = False  # Track sequential offload state
        self._use_sequential_offload = False  # Will be set during load
        
        # Performance tracking
        self.generation_count = 0
        self.total_generation_time = 0.0
        self.load_count = 0
        self.unload_count = 0
        
        # Setup device (but don't load model yet)
        self._setup_device()
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        logger.info("✅ Optimized Image Generator initialized (model will load on first request)")
    
    def _setup_device(self):
        """Setup the target device for model execution."""
        # Check CUDA availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("⚠️  CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.gpu_index = None

        # Check MPS availability (macOS)
        if self.device == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.target_device = "mps"
                logger.info("🎯 Using device: Apple Silicon GPU (MPS)")
            else:
                logger.warning("⚠️  MPS not available, falling back to CPU")
                self.device = "cpu"
                self.gpu_index = None
                self.target_device = "cpu"
                logger.info("🎯 Using device: CPU")
                return

        if self.device == "cuda":
            available_gpus = torch.cuda.device_count()
            logger.info(f"🔍 Available CUDA GPUs: {available_gpus}")

            # Use specified GPU index if valid, otherwise use first available
            if self.gpu_index is not None:
                if self.gpu_index < available_gpus:
                    self.target_device = f"cuda:{self.gpu_index}"
                    torch.cuda.set_device(self.gpu_index)
                    gpu_name = torch.cuda.get_device_name(self.gpu_index)
                    logger.info(f"🎯 Using CUDA GPU {self.gpu_index}: {gpu_name}")
                else:
                    logger.warning(f"⚠️  CUDA GPU {self.gpu_index} not available (only {available_gpus} GPUs), using cuda:0")
                    self.target_device = "cuda:0"
                    self.gpu_index = 0
                    torch.cuda.set_device(0)
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"🎯 Using CUDA GPU 0: {gpu_name}")
            else:
                # No specific GPU requested, use first available
                self.target_device = "cuda:0"
                self.gpu_index = 0
                torch.cuda.set_device(0)
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🎯 Using CUDA GPU 0: {gpu_name}")
        elif self.device == "cpu":
            # CPU mode
            self.target_device = "cpu"
            logger.info(f"🎯 Using device: CPU")
        elif self.device == "mps":
            # MPS already handled above
            pass
        else:
            # Unknown device type, fallback to CPU
            logger.warning(f"⚠️  Unknown device type '{self.device}', falling back to CPU")
            self.device = "cpu"
            self.target_device = "cpu"
            logger.info(f"🎯 Using device: CPU")
    
    def _check_available_memory(self, required_gb: float = 2.0) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Check if enough GPU memory is available with detailed diagnostics.
        
        Args:
            required_gb: Required memory in GB
            
        Returns:
            (has_enough_memory, available_gb, memory_info_dict)
        """
        memory_info = {
            "total_gb": 0.0,
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "available_gb": 0.0,
            "required_gb": required_gb,
            "device": self.target_device
        }
        
        if self.device != "cuda" or not torch.cuda.is_available():
            memory_info["available_gb"] = float('inf')
            return True, float('inf'), memory_info  # CPU mode, assume enough memory
        
        try:
            logical_device = int(self.target_device.split(':')[1]) if ':' in self.target_device else 0
            with torch.cuda.device(logical_device):
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(logical_device).total_memory / 1024**3
                available = total - reserved
                
                memory_info.update({
                    "total_gb": total,
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "available_gb": available
                })
                
                # Get detailed memory stats if available
                try:
                    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                    memory_info["max_allocated_gb"] = max_allocated
                except:
                    pass
                
                logger.info(f"📊 GPU Memory Check - Total: {total:.2f}GB, Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Available: {available:.2f}GB, Required: {required_gb:.2f}GB")
                
                return available >= required_gb, available, memory_info
        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}")
            memory_info["error"] = str(e)
            return True, float('inf'), memory_info  # Assume OK if check fails
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """
        Get detailed GPU memory information for diagnostics.
        
        Returns:
            Dictionary with memory statistics
        """
        if self.device != "cuda" or not torch.cuda.is_available():
            return {"device": "cpu", "cuda_available": False}
        
        try:
            logical_device = int(self.target_device.split(':')[1]) if ':' in self.target_device else 0
            with torch.cuda.device(logical_device):
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(logical_device).total_memory / 1024**3
                available = total - reserved
                
                info = {
                    "device": self.target_device,
                    "gpu_index": logical_device,
                    "gpu_name": torch.cuda.get_device_name(logical_device),
                    "total_gb": total,
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "available_gb": available,
                    "utilization_percent": (reserved / total * 100) if total > 0 else 0,
                    "cuda_available": True
                }
                
                # Try to get max memory stats
                try:
                    info["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
                except:
                    pass
                
                return info
        except Exception as e:
            return {"device": self.target_device, "error": str(e), "cuda_available": False}
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded (on-demand loading)."""
        if self.is_loaded and self.pipeline is not None:
            self._last_use_time = time.time()
            return
        
        with self._load_lock:
            # Double-check after acquiring lock
            if self.is_loaded and self.pipeline is not None:
                self._last_use_time = time.time()
                return
            
            self._load_model()
    
    def _load_model(self):
        """Load the FLUX model with optimal settings and memory management."""
        try:
            logger.info(f"📥 Loading model: {self.model_name} (on-demand)")
            start_time = time.time()
            
            # Log memory state before cleanup
            if self.device == "cuda":
                self._log_gpu_memory("before cleanup")
            
            # Aggressive memory cleanup before loading
            self._aggressive_memory_cleanup()
            
            # Log memory state after cleanup
            if self.device == "cuda":
                self._log_gpu_memory("after cleanup")
            
            # Check available memory and determine if CPU offload is needed
            # FLUX.1-schnell actually needs ~12-15GB VRAM (not 2GB as previously assumed)
            # FLUX.1-dev needs ~24GB VRAM
            is_schnell_model = "schnell" in self.model_name.lower()
            required_memory_gb = 12.0 if is_schnell_model else 24.0
            logger.info(f"🔍 Model: {self.model_name}, Type: {'schnell' if is_schnell_model else 'dev'}, Required VRAM: ~{required_memory_gb}GB")
            has_enough_memory, available_gb, memory_info = self._check_available_memory(required_memory_gb)
            
            # Memory-aware model selection: reject dev model if insufficient memory
            if "dev" in self.model_name.lower() and not "schnell" in self.model_name.lower():
                if available_gb < 8.0:  # Need at least 8GB for dev model
                    error_msg = (f"Insufficient GPU memory for FLUX.1-dev model. "
                               f"Available: {available_gb:.2f}GB, Required: ~12GB. "
                               f"Please use FLUX.1-schnell model instead or free up GPU memory.")
                    logger.error(f"❌ {error_msg}")
                    raise RuntimeError(error_msg)
            
            # Auto-determine CPU offload if not explicitly set
            # FLUX.1-schnell needs ~12-15GB, so we should ALWAYS use CPU offload for limited memory GPUs
            use_cpu_offload = self.enable_cpu_offload
            use_sequential_offload = False  # Will be set based on memory
            
            if use_cpu_offload is None:
                # Force CPU offload for schnell model when memory is limited
                # Sequential offload is more memory-efficient than model offload
                is_schnell = "schnell" in self.model_name.lower()
                memory_threshold = 16.0  # Use sequential offload if less than 16GB available
                
                if is_schnell:
                    # For schnell model, ALWAYS use sequential CPU offload if < 16GB available
                    # Sequential offload can work with as little as 6GB VRAM
                    if available_gb < memory_threshold:
                        use_cpu_offload = True
                        use_sequential_offload = True  # Sequential is more memory efficient
                        reason = f"schnell model with limited memory (available: {available_gb:.2f}GB < {memory_threshold}GB) - using sequential CPU offload"
                    elif not has_enough_memory:
                        use_cpu_offload = True
                        use_sequential_offload = True
                        reason = f"schnell model insufficient memory (available: {available_gb:.2f}GB < required: {required_memory_gb:.2f}GB) - using sequential CPU offload"
                    else:
                        use_cpu_offload = False
                        reason = f"sufficient memory for schnell model (available: {available_gb:.2f}GB >= {memory_threshold}GB)"
                else:
                    # For dev model, use CPU offload if memory is low
                    use_cpu_offload = not has_enough_memory or available_gb < 16.0
                    use_sequential_offload = use_cpu_offload  # Use sequential for dev too
                    reason = f"dev model - {'low memory' if use_cpu_offload else 'sufficient memory'} (available: {available_gb:.2f}GB)"
                
                if use_cpu_offload:
                    logger.info(f"🔄 Auto-enabling CPU offload ({reason})")
                else:
                    logger.info(f"✅ Sufficient memory available ({available_gb:.2f}GB), using GPU directly")
            
            # Store offload type
            self._use_sequential_offload = use_sequential_offload
            
            # Get Hugging Face token
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            
            # Optimal settings for FLUX models
            pipeline_kwargs = {
                "torch_dtype": self.torch_dtype,
                "use_safetensors": True,
                "token": hf_token,
                "device_map": None,  # We'll handle device placement manually
            }
            
            # Only add max_sequence_length if model supports it (schnell doesn't)
            if "schnell" not in self.model_name.lower():
                pipeline_kwargs["max_sequence_length"] = self.max_sequence_length
            
            # Load the pipeline
            self.pipeline = FluxPipeline.from_pretrained(
                self.model_name,
                **pipeline_kwargs
            )
            
            # Apply optimizations
            if use_cpu_offload:
                # CPU offloading (slower but uses less VRAM)
                # Sequential offload is more memory-efficient than model offload
                if use_sequential_offload:
                    logger.info("🔄 Enabling sequential CPU offloading (most memory efficient, works with ~6GB VRAM)")
                    try:
                        self.pipeline.enable_sequential_cpu_offload()
                        logger.info("✅ Sequential CPU offload enabled successfully")
                    except Exception as e:
                        logger.warning(f"⚠️  Sequential CPU offload failed, falling back to model CPU offload: {e}")
                        self.pipeline.enable_model_cpu_offload()
                        logger.info("✅ Model CPU offload enabled (fallback)")
                else:
                    logger.info("🔄 Enabling model CPU offloading (memory efficient)")
                    self.pipeline.enable_model_cpu_offload()
                
                # Even with CPU offload, enable VAE slicing and tiling for additional memory savings
                try:
                    self.pipeline.enable_vae_slicing()
                    logger.info("✅ Enabled VAE slicing (with CPU offload)")
                except Exception as e:
                    logger.debug(f"Could not enable VAE slicing with CPU offload: {e}")
                
                # Enable VAE tiling for even more memory efficiency
                try:
                    self.pipeline.enable_vae_tiling()
                    logger.info("✅ Enabled VAE tiling (with CPU offload)")
                except Exception as e:
                    logger.debug(f"Could not enable VAE tiling with CPU offload: {e}")
                
                # Enable attention slicing for transformer layers
                try:
                    self.pipeline.enable_attention_slicing()
                    logger.info("✅ Enabled attention slicing (with CPU offload)")
                except Exception as e:
                    logger.debug(f"Could not enable attention slicing with CPU offload: {e}")
            else:
                # Move to target device and apply GPU optimizations
                logger.info(f"📱 Moving model to {self.target_device}")
                self.pipeline = self.pipeline.to(self.target_device)
                
                # Enable memory efficient attention if available
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("✅ Enabled xFormers memory efficient attention")
                except Exception as e:
                    logger.info("ℹ️  xFormers not available, using default attention")
                
                # Enable VAE slicing for memory efficiency (especially important for schnell)
                try:
                    self.pipeline.enable_vae_slicing()
                    logger.info("✅ Enabled VAE slicing")
                except Exception as e:
                    logger.warning(f"Could not enable VAE slicing: {e}")
                
                # Enable VAE tiling for additional memory savings
                try:
                    self.pipeline.enable_vae_tiling()
                    logger.info("✅ Enabled VAE tiling")
                except Exception as e:
                    logger.debug(f"Could not enable VAE tiling: {e}")
                
                # Enable attention slicing for transformer layers
                try:
                    self.pipeline.enable_attention_slicing()
                    logger.info("✅ Enabled attention slicing")
                except Exception as e:
                    logger.debug(f"Could not enable attention slicing: {e}")
            
            # Disable model compilation for schnell model to save memory
            # Compilation can use significant memory and schnell is already fast
            is_schnell = "schnell" in self.model_name.lower()
            should_compile = not use_cpu_offload and not is_schnell and hasattr(torch, 'compile') and self.device == "cuda"
            
            if should_compile:
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
            elif is_schnell:
                logger.info("ℹ️  Skipping model compilation for schnell model (memory optimization)")
            
            load_time = time.time() - start_time
            self.is_loaded = True
            self.load_count += 1
            self._last_use_time = time.time()
            
            offload_type = "sequential CPU" if (use_cpu_offload and use_sequential_offload) else ("CPU" if use_cpu_offload else "GPU")
            logger.info(f"🎉 Model loaded successfully in {load_time:.2f} seconds (Device: {offload_type})")
            logger.info(f"📋 Model Details: {self.model_name}, Device: {self.target_device}, Offload: {offload_type}")
            
            # Store CPU offload state for info
            self._actual_cpu_offload = use_cpu_offload
            self._actual_sequential_offload = use_sequential_offload if use_cpu_offload else False
            
            # Log GPU memory usage after load
            if self.device == "cuda":
                self._log_gpu_memory("after model load")
                # Verify memory usage is acceptable
                memory_info_after = self.get_gpu_memory_info()
                if "available_gb" in memory_info_after:
                    if memory_info_after["available_gb"] < 2.0:
                        logger.warning(f"⚠️  Low GPU memory after model load: {memory_info_after['available_gb']:.2f}GB available")
                    else:
                        logger.info(f"✅ GPU memory after load: {memory_info_after['available_gb']:.2f}GB available")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.is_loaded = False
            self.pipeline = None
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def _aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup before loading model."""
        logger.info("🧹 Performing aggressive memory cleanup...")
        
        # Try to clear other services' memory if possible
        try:
            # Clear transcription models if available - this is critical for freeing GPU memory
            from app.core.robust_transcription import get_robust_transcription_manager
            trans_manager = get_robust_transcription_manager()
            if trans_manager:
                # Get info about cached models before clearing
                model_info = trans_manager.get_model_info()
                cached_count = (len(model_info.get('whisper_models_loaded', [])) + 
                              len(model_info.get('faster_whisper_models_loaded', [])))
                if cached_count > 0:
                    logger.info(f"📋 Found {cached_count} cached transcription model(s), clearing...")
                trans_manager.clear_cache()
                logger.info("✅ Cleared transcription model cache")
        except Exception as e:
            logger.debug(f"Could not clear transcription cache: {e}")
        
        # Force garbage collection multiple times to ensure Python objects are freed
        for i in range(3):
            collected = gc.collect()
            if i == 0 and collected > 0:
                logger.debug(f"🧹 Garbage collection: freed {collected} objects")
        
        # Clear CUDA cache aggressively
        if torch.cuda.is_available():
            # Get memory before clearing
            try:
                logical_device = int(self.target_device.split(':')[1]) if ':' in self.target_device else 0
                with torch.cuda.device(logical_device):
                    before_allocated = torch.cuda.memory_allocated() / 1024**3
                    before_reserved = torch.cuda.memory_reserved() / 1024**3
            except:
                before_allocated = before_reserved = 0
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all operations to complete
            
            # Also collect IPC memory
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
            
            # Try to free unused memory
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                try:
                    torch.cuda.reset_peak_memory_stats()
                except:
                    pass
            
            # Log memory freed
            try:
                with torch.cuda.device(logical_device):
                    after_allocated = torch.cuda.memory_allocated() / 1024**3
                    after_reserved = torch.cuda.memory_reserved() / 1024**3
                    freed_allocated = before_allocated - after_allocated
                    freed_reserved = before_reserved - after_reserved
                    if freed_allocated > 0.01 or freed_reserved > 0.01:  # Only log if significant
                        logger.info(f"🧹 CUDA cache cleared: freed {freed_allocated:.2f}GB allocated, {freed_reserved:.2f}GB reserved")
            except:
                pass
        
        # Clear MPS cache (macOS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        logger.info("✅ Memory cleanup completed")
    
    def _log_gpu_memory(self, context: str = "after load"):
        """Log current GPU memory usage with context."""
        if self.device == "cuda":
            try:
                memory_info = self.get_gpu_memory_info()
                if "error" not in memory_info:
                    physical_gpu = self.gpu_index if self.gpu_index is not None else "unknown"
                    logger.info(f"📊 GPU {physical_gpu} Memory ({context}) - "
                              f"Total: {memory_info['total_gb']:.2f}GB, "
                              f"Allocated: {memory_info['allocated_gb']:.2f}GB, "
                              f"Reserved: {memory_info['reserved_gb']:.2f}GB, "
                              f"Available: {memory_info['available_gb']:.2f}GB, "
                              f"Utilization: {memory_info.get('utilization_percent', 0):.1f}%")
                else:
                    logger.warning(f"Could not log GPU memory: {memory_info.get('error')}")
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
        Generate image(s) from text prompt using FLUX model (on-demand loading).
        
        Args:
            prompt: Text description of the image
            width: Image width (default 1024 for FLUX)
            height: Image height (default 1024 for FLUX)
            num_inference_steps: Number of denoising steps (4 is optimal for FLUX)
            guidance_scale: Guidance scale (0.0 is optimal for FLUX.1-dev/schnell)
            num_images: Number of images to generate (default 1)
            seed: Random seed for reproducibility
            max_sequence_length: Override default max sequence length
            
        Returns:
            ImageGenerationResult or List[ImageGenerationResult] containing the generated image(s)
        """
        # Ensure model is loaded (on-demand)
        self._ensure_model_loaded()
        
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
            
            # Validate result structure
            if not hasattr(result, 'images') or not result.images:
                raise ValueError("Pipeline result does not contain images")
            
            # Extract images before model unload (important for CPU offload)
            extracted_images = []
            for img in result.images:
                if img is None:
                    logger.error("Generated image is None!")
                    raise ValueError("Generated image is None")
                
                # Ensure image is a PIL Image
                from PIL import Image
                if not isinstance(img, Image.Image):
                    logger.error(f"Generated image is not a PIL Image, type: {type(img)}")
                    raise ValueError(f"Generated image is not a PIL Image, type: {type(img)}")
                
                # Log image info for debugging
                logger.debug(f"Extracted image: size={img.size}, mode={img.mode}, format={img.format}")
                extracted_images.append(img)
            
            # Create result object(s) - images are now safely extracted
            if num_images == 1:
                image_result = ImageGenerationResult(
                    image=extracted_images[0],
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
                for i, image in enumerate(extracted_images):
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
        finally:
            # Update last use time
            self._last_use_time = time.time()
            
            # Auto-unload if enabled
            if self.auto_unload and self.unload_timeout == 0:
                # Immediate unload
                self.unload_model()
            elif self.auto_unload and self.unload_timeout > 0:
                # Schedule unload after timeout (could be done in background thread)
                # For now, we'll unload immediately if timeout is 0
                pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "model_name": self.model_name,
            "device": self.target_device,
            "is_loaded": self.is_loaded,
            "torch_dtype": str(self.torch_dtype),
            "generation_count": self.generation_count,
            "average_generation_time": self.total_generation_time / max(self.generation_count, 1),
            "enable_cpu_offload": self.enable_cpu_offload,
            "actual_cpu_offload": getattr(self, '_actual_cpu_offload', False),
            "actual_sequential_offload": getattr(self, '_actual_sequential_offload', False),
            "load_count": self.load_count,
            "unload_count": self.unload_count,
            "auto_unload": self.auto_unload,
        }
        
        # Add GPU memory info if available
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                memory_info = self.get_gpu_memory_info()
                info["gpu_memory"] = memory_info
            except:
                pass
        
        return info
    
    def get_available_models(self) -> List[str]:
        """Get list of available models (minimal first)."""
        return [
            "flux-schnell",  # Minimal model - should be first/default
            "flux-dev",
            "flux",
            "flux-dev-8bit",
            "flux-dev-4bit"
        ]
    
    def unload_model(self):
        """Unload the model to free memory."""
        if not self.is_loaded or self.pipeline is None:
            return
        
        with self._load_lock:
            if not self.is_loaded or self.pipeline is None:
                return
            
            logger.info("📤 Unloading model to free memory...")
            
            try:
                # Move pipeline to CPU to free GPU memory
                if self.device == "cuda" and self.pipeline is not None:
                    try:
                        self.pipeline = self.pipeline.to("cpu")
                    except Exception as e:
                        logger.warning(f"Could not move pipeline to CPU: {e}")

                # Delete pipeline
                if self.pipeline is not None:
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

                # Clear MPS cache (macOS)
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                self.is_loaded = False
                self.unload_count += 1
                logger.info("✅ Model unloaded successfully")

            except Exception as e:
                logger.error(f"❌ Error during model unload: {e}")
                self.is_loaded = False
                self.pipeline = None
    
    def cleanup(self):
        """Clean up GPU memory and resources."""
        logger.info("🧹 Cleaning up GPU resources...")
        self.unload_model()
        logger.info("✅ Cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, 'is_loaded') and self.is_loaded:
            self.cleanup()


# Global instances cache (keyed by model_name + device)
_generator_instances: Dict[str, OptimizedImageGenerator] = {}
_generator_lock = threading.Lock()


def get_optimized_generator(
    model_name: str = "black-forest-labs/FLUX.1-schnell",
    device: str = "cuda", 
    gpu_index: Optional[int] = None,
    **kwargs
) -> OptimizedImageGenerator:
    """
    Get or create an optimized image generator instance (on-demand, not singleton).
    
    Args:
        model_name: Model name (default: FLUX.1-schnell - minimal model)
        device: Device to use
        gpu_index: GPU index
        **kwargs: Additional arguments
        
    Returns:
        OptimizedImageGenerator instance
    """
    global _generator_instances
    
    # Create cache key
    cache_key = f"{model_name}:{device}:{gpu_index}"
    
    # Check if instance exists
    if cache_key in _generator_instances:
        return _generator_instances[cache_key]
    
    # Create new instance
    with _generator_lock:
        # Double-check after acquiring lock
        if cache_key in _generator_instances:
            return _generator_instances[cache_key]
        
        generator = OptimizedImageGenerator(
            model_name=model_name,
            device=device,
            gpu_index=gpu_index,
            **kwargs
        )
        
        _generator_instances[cache_key] = generator
        return generator


def cleanup_generator(model_name: Optional[str] = None, device: Optional[str] = None):
    """
    Clean up generator instance(s).
    
    Args:
        model_name: Specific model to cleanup (None = all)
        device: Specific device to cleanup (None = all)
    """
    global _generator_instances
    
    with _generator_lock:
        if model_name is None and device is None:
            # Cleanup all
            for generator in _generator_instances.values():
                generator.cleanup()
            _generator_instances.clear()
        else:
            # Cleanup specific instances
            keys_to_remove = []
            for key, generator in _generator_instances.items():
                if (model_name is None or model_name in key) and (device is None or device in key):
                    generator.cleanup()
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del _generator_instances[key]


# Register cleanup on module exit
atexit.register(cleanup_generator)
