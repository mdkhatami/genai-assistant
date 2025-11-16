"""
Device Manager Utility

Centralized device (GPU/CPU) management for GenAI Assistant.
Provides automatic device detection, selection, and allocation without
hardcoded GPU indices or environment variable manipulation.

Supports:
- CUDA GPUs (Linux/Windows)
- Metal Performance Shaders (MPS) GPUs (macOS Apple Silicon)
- CPU fallback on all platforms
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Information about an available compute device."""
    device_type: str  # "cuda", "mps", or "cpu"
    device_index: Optional[int] = None  # GPU index if CUDA/MPS, None for CPU
    device_name: str = ""  # GPU name or "CPU"
    total_memory: Optional[int] = None  # Total memory in bytes
    available_memory: Optional[int] = None  # Available memory in bytes
    compute_capability: Optional[tuple] = None  # (major, minor) for CUDA


class DeviceManager:
    """
    Centralized device management for GPU/CPU selection.

    Features:
    - Automatic GPU detection (CUDA on Linux/Windows, MPS on macOS)
    - CPU fallback when GPU unavailable
    - Thread-safe device allocation
    - Memory availability checking
    - No environment variable manipulation
    - No hardcoded GPU indices
    - Cross-platform support (macOS, Linux, Windows)
    """

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        """Singleton pattern to ensure single instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # Initialize _initialized flag on the instance
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize device manager (only once)."""
        # Use instance-level _initialized flag for thread safety
        if hasattr(self, '_initialized') and self._initialized:
            return

        with self._lock:
            # Double-check after acquiring lock
            if hasattr(self, '_initialized') and self._initialized:
                return

            self._initialized = True
            self.available_devices: List[DeviceInfo] = []
            self.cpu_info: Optional[DeviceInfo] = None
            self.torch_available = False
            self.platform = sys.platform
            self.is_macos = sys.platform == "darwin"

            # Detect available devices
            self._detect_devices()

            platform_info = f" ({self.platform})" if self.is_macos else ""
            logger.info(f"🔧 DeviceManager initialized{platform_info}: {len(self.available_devices)} GPU(s), CPU available: {self.cpu_info is not None}")

    def _detect_devices(self):
        """Detect available GPUs and CPU."""
        # Try to import torch to detect GPUs
        try:
            import torch
            self.torch_available = True

            # Detect CUDA GPUs (Linux/Windows)
            if torch.cuda.is_available():
                self._detect_cuda_devices(torch)
            else:
                logger.info("ℹ️  CUDA not available - checking for other GPU types")

            # Detect MPS GPUs (macOS Apple Silicon)
            if self.is_macos:
                self._detect_mps_devices(torch)

        except ImportError:
            logger.info("ℹ️  PyTorch not available - will use CPU mode")
            self.torch_available = False

        # Always detect CPU as fallback
        self._detect_cpu()

    def _detect_cuda_devices(self, torch):
        """Detect CUDA GPUs."""
        try:
            gpu_count = torch.cuda.device_count()
            logger.info(f"✅ CUDA available: {gpu_count} GPU(s) detected")

            for i in range(gpu_count):
                try:
                    device_name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)

                    device_info = DeviceInfo(
                        device_type="cuda",
                        device_index=i,
                        device_name=device_name,
                        total_memory=props.total_memory,
                        compute_capability=(props.major, props.minor)
                    )

                    # Get available memory (accounting for reserved memory)
                    try:
                        torch.cuda.set_device(i)
                        # Use memory_reserved() for more accurate available memory calculation
                        allocated = torch.cuda.memory_allocated(i)
                        reserved = torch.cuda.memory_reserved(i)
                        total = props.total_memory
                        # Available = total - reserved (more accurate than total - allocated)
                        device_info.available_memory = max(0, total - reserved)
                    except Exception as e:
                        logger.warning(f"  Could not get memory info for CUDA GPU {i}: {e}")
                        device_info.available_memory = device_info.total_memory

                    self.available_devices.append(device_info)

                    logger.info(f"  CUDA GPU {i}: {device_name} ({self._format_memory(device_info.total_memory)} total, "
                              f"{self._format_memory(device_info.available_memory)} available)")

                except Exception as e:
                    logger.warning(f"  Could not get info for CUDA GPU {i}: {e}")
                    # Continue detecting other GPUs even if one fails

        except Exception as e:
            logger.warning(f"  Error detecting CUDA devices: {e}")

    def _detect_mps_devices(self, torch):
        """Detect macOS Metal Performance Shaders (MPS) GPUs."""
        try:
            # Check if MPS is available (macOS 12.3+ with Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("✅ MPS (Metal) available: Apple Silicon GPU detected")

                # MPS typically has one device, but we'll treat it as device 0
                # Get device info if available
                device_info = DeviceInfo(
                    device_type="mps",
                    device_index=0,
                    device_name="Apple Silicon GPU (MPS)",
                    total_memory=None,  # MPS doesn't expose total memory easily
                    available_memory=None
                )

                # Try to get memory info if possible
                try:
                    # MPS memory info is not directly accessible via PyTorch
                    # We'll leave it as None and let services handle it
                    pass
                except Exception:
                    pass

                self.available_devices.append(device_info)
                logger.info(f"  MPS GPU: Apple Silicon GPU (Metal Performance Shaders)")
            else:
                if self.is_macos:
                    logger.info("ℹ️  MPS not available - macOS Intel or older macOS version")

        except Exception as e:
            logger.warning(f"  Error detecting MPS devices: {e}")

    def _detect_cpu(self):
        """Detect CPU and available RAM."""
        try:
            import psutil

            # Get system memory
            mem = psutil.virtual_memory()

            self.cpu_info = DeviceInfo(
                device_type="cpu",
                device_name="CPU",
                total_memory=mem.total,
                available_memory=mem.available
            )

            logger.info(f"  CPU: {self._format_memory(mem.total)} RAM total, "
                       f"{self._format_memory(mem.available)} available")

        except ImportError:
            logger.warning("  psutil not available - cannot detect CPU memory")
            # Still create basic CPU info
            self.cpu_info = DeviceInfo(
                device_type="cpu",
                device_name="CPU"
            )

    def _format_memory(self, bytes_val: Optional[int]) -> str:
        """Format memory size in human-readable format."""
        if bytes_val is None:
            return "Unknown"

        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.2f} PB"

    def get_best_device(self,
                       device_preference: str = "auto",
                       min_memory_gb: Optional[float] = None,
                       service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the best available device based on preference and requirements.

        Args:
            device_preference: "auto", "gpu", "cuda", or "cpu"
            min_memory_gb: Minimum memory required in GB (optional)
            service_name: Service name for logging (optional)

        Returns:
            Dictionary with:
                - device: "cuda" or "cpu"
                - device_index: GPU index if CUDA, None for CPU
                - device_str: "cuda:0", "cuda:1", or "cpu"
                - device_name: Human-readable device name
                - available_memory: Available memory in bytes
        """
        service_label = f" for {service_name}" if service_name else ""

        # Normalize preference
        if device_preference.lower() in ["auto", "gpu", "cuda", "mps"]:
            # Prefer GPU if available
            if self.available_devices:
                # Find GPU with most available memory (or first if memory unknown)
                best_gpu = max(self.available_devices,
                             key=lambda d: d.available_memory if d.available_memory is not None else 0)

                # Check memory requirement (skip if memory is unknown, e.g., MPS)
                if min_memory_gb is not None and best_gpu.available_memory is not None:
                    min_memory_bytes = min_memory_gb * 1024 * 1024 * 1024
                    if best_gpu.available_memory < min_memory_bytes:
                        logger.warning(f"⚠️  GPU has insufficient memory{service_label}: "
                                     f"{self._format_memory(best_gpu.available_memory)} < "
                                     f"{self._format_memory(int(min_memory_bytes))} - falling back to CPU")
                        return self._get_cpu_device(service_label)

                # Build device string based on device type
                if best_gpu.device_type == "mps":
                    device_str = "mps"
                elif best_gpu.device_type == "cuda":
                    device_str = f"cuda:{best_gpu.device_index}" if best_gpu.device_index is not None else "cuda"
                else:
                    device_str = f"{best_gpu.device_type}:{best_gpu.device_index}" if best_gpu.device_index is not None else best_gpu.device_type

                logger.info(f"✅ Selected{service_label}: {best_gpu.device_name} ({device_str})")

                return {
                    "device": best_gpu.device_type,
                    "device_index": best_gpu.device_index,
                    "device_str": device_str,
                    "device_name": best_gpu.device_name,
                    "available_memory": best_gpu.available_memory
                }
            else:
                # No GPU available, fallback to CPU
                if device_preference.lower() in ["gpu", "cuda", "mps"]:
                    logger.warning(f"⚠️  GPU requested{service_label} but none available - using CPU")
                return self._get_cpu_device(service_label)

        elif device_preference.lower() == "cpu":
            return self._get_cpu_device(service_label)

        else:
            logger.warning(f"⚠️  Unknown device preference '{device_preference}'{service_label} - using auto")
            return self.get_best_device("auto", min_memory_gb, service_name)

    def _get_cpu_device(self, service_label: str = "") -> Dict[str, Any]:
        """Get CPU device info."""
        if not self.cpu_info:
            # Fallback if CPU detection failed
            self.cpu_info = DeviceInfo(device_type="cpu", device_name="CPU")

        logger.info(f"ℹ️  Selected{service_label}: CPU")

        return {
            "device": "cpu",
            "device_index": None,
            "device_str": "cpu",
            "device_name": "CPU",
            "available_memory": self.cpu_info.available_memory
        }

    def get_device_for_service(self,
                              service_name: str,
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get device for a specific service based on configuration.

        Args:
            service_name: Name of the service (e.g., "image_generation", "transcription")
            config: Service configuration dictionary with optional 'device' and 'min_memory_gb'

        Returns:
            Device info dictionary (same as get_best_device)
        """
        device_pref = config.get('device', 'auto')
        min_memory = config.get('min_memory_gb')

        return self.get_best_device(
            device_preference=device_pref,
            min_memory_gb=min_memory,
            service_name=service_name
        )

    def validate_device_requirements(self,
                                    service_name: str,
                                    min_memory_gb: float,
                                    device_type: str = "auto") -> tuple[bool, str]:
        """
        Validate if device requirements can be met.

        Args:
            service_name: Service name
            min_memory_gb: Minimum memory required in GB
            device_type: Required device type ("auto", "gpu", "cpu")

        Returns:
            (is_valid, message) tuple
        """
        device = self.get_best_device(device_type, min_memory_gb, service_name)

        if device['available_memory'] is None:
            return True, f"{service_name}: Device {device['device_name']} selected (memory unknown)"

        min_memory_bytes = min_memory_gb * 1024 * 1024 * 1024

        if device['available_memory'] >= min_memory_bytes:
            return True, (f"{service_name}: ✅ {device['device_name']} has sufficient memory "
                         f"({self._format_memory(device['available_memory'])} >= "
                         f"{self._format_memory(int(min_memory_bytes))})")
        else:
            return False, (f"{service_name}: ❌ {device['device_name']} has insufficient memory "
                          f"({self._format_memory(device['available_memory'])} < "
                          f"{self._format_memory(int(min_memory_bytes))})")

    def get_summary(self) -> str:
        """Get a summary of available devices."""
        lines = ["Device Summary:"]
        
        if self.is_macos:
            lines.append(f"  Platform: macOS ({self.platform})")

        if self.available_devices:
            lines.append(f"  GPUs: {len(self.available_devices)}")
            for dev in self.available_devices:
                mem_str = self._format_memory(dev.total_memory) if dev.total_memory else "Unknown"
                index_str = f" {dev.device_index}" if dev.device_index is not None else ""
                lines.append(f"    - {dev.device_type.upper()} GPU{index_str}: {dev.device_name} "
                           f"({mem_str})")
        else:
            lines.append("  GPUs: None available")

        if self.cpu_info:
            mem_str = self._format_memory(self.cpu_info.total_memory) if self.cpu_info.total_memory else "Unknown"
            lines.append(f"  CPU: Available ({mem_str} RAM)")
        else:
            lines.append("  CPU: Available (memory unknown)")

        return "\n".join(lines)

    def clear_gpu_memory(self):
        """Clear GPU memory cache (if torch available)."""
        if not self.torch_available:
            return

        try:
            import torch
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🧹 CUDA memory cache cleared")
            # Clear MPS cache (macOS)
            if self.is_macos and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.info("🧹 MPS memory cache cleared")
        except Exception as e:
            logger.warning(f"Could not clear GPU memory: {e}")


# Global instance
_device_manager_instance = None


def get_device_manager() -> DeviceManager:
    """Get the global DeviceManager instance."""
    global _device_manager_instance
    if _device_manager_instance is None:
        _device_manager_instance = DeviceManager()
    return _device_manager_instance


def initialize_device_manager() -> DeviceManager:
    """Initialize and return the DeviceManager instance."""
    return get_device_manager()
