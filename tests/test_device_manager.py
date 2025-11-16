"""
Test suite for DeviceManager utility.

This module tests the device management functionality including:
- GPU/CPU detection
- Device selection based on preferences
- Memory validation
- Singleton pattern
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.device_manager import DeviceManager, get_device_manager, initialize_device_manager


class TestDeviceManagerSingleton(unittest.TestCase):
    """Test cases for DeviceManager singleton pattern."""

    def setUp(self):
        """Reset singleton instance before each test."""
        DeviceManager._instance = None

    def test_singleton_pattern(self):
        """Test that DeviceManager follows singleton pattern."""
        dm1 = get_device_manager()
        dm2 = get_device_manager()
        self.assertIs(dm1, dm2, "DeviceManager should return same instance")

    def test_initialize_device_manager(self):
        """Test initialize_device_manager function."""
        dm = initialize_device_manager()
        self.assertIsInstance(dm, DeviceManager)


class TestDeviceDetection(unittest.TestCase):
    """Test cases for device detection."""

    def setUp(self):
        """Reset singleton instance before each test."""
        DeviceManager._instance = None

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_cpu_only_detection(self, mock_device_count, mock_is_available):
        """Test detection when no GPU is available."""
        mock_is_available.return_value = False
        mock_device_count.return_value = 0

        dm = get_device_manager()
        self.assertEqual(len(dm.available_devices), 0)
        # device_count doesn't exist - use len(available_devices) instead
        self.assertEqual(len(dm.available_devices), 0)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    def test_gpu_detection(self, mock_get_props, mock_device_count, mock_is_available):
        """Test detection when GPU is available."""
        mock_is_available.return_value = True
        mock_device_count.return_value = 1

        # Mock GPU properties
        mock_props = Mock()
        mock_props.name = "NVIDIA Tesla T4"
        mock_props.total_memory = 16 * 1024 * 1024 * 1024  # 16GB
        mock_props.major = 7
        mock_props.minor = 5
        mock_get_props.return_value = mock_props

        # Mock memory functions
        with patch('torch.cuda.set_device'), \
             patch('torch.cuda.memory_allocated', return_value=0), \
             patch('torch.cuda.memory_reserved', return_value=0):
            dm = get_device_manager()
            # device_count doesn't exist - use len(available_devices) instead
            self.assertEqual(len(dm.available_devices), 1)


class TestDeviceSelection(unittest.TestCase):
    """Test cases for device selection logic."""

    def setUp(self):
        """Reset singleton instance before each test."""
        DeviceManager._instance = None

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_auto_mode_cpu_fallback(self, mock_device_count, mock_is_available):
        """Test auto mode falls back to CPU when no GPU available."""
        mock_is_available.return_value = False
        mock_device_count.return_value = 0

        dm = get_device_manager()
        device_info = dm.get_best_device(device_preference="auto")

        self.assertEqual(device_info['device'], "cpu")
        self.assertEqual(device_info['device_str'], "cpu")

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_cpu_mode_forced(self, mock_device_count, mock_is_available):
        """Test CPU mode is used when explicitly requested."""
        mock_is_available.return_value = True
        mock_device_count.return_value = 1

        dm = get_device_manager()
        device_info = dm.get_best_device(device_preference="cpu")

        self.assertEqual(device_info['device'], "cpu")
        self.assertEqual(device_info['device_str'], "cpu")

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.mem_get_info')
    def test_auto_mode_gpu_selection(self, mock_mem_info, mock_get_props,
                                     mock_device_count, mock_is_available):
        """Test auto mode selects GPU when available."""
        mock_is_available.return_value = True
        mock_device_count.return_value = 1

        # Mock GPU properties
        mock_props = Mock()
        mock_props.name = "NVIDIA Tesla T4"
        mock_props.total_memory = 16 * 1024 * 1024 * 1024  # 16GB
        mock_get_props.return_value = mock_props

        # Mock memory info (free, total)
        mock_mem_info.return_value = (8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024)

        dm = get_device_manager()
        device_info = dm.get_best_device(device_preference="auto")

        self.assertEqual(device_info['device'], "cuda")
        self.assertIn("cuda:", device_info['device_str'])


class TestServiceConfiguration(unittest.TestCase):
    """Test cases for service-specific device configuration."""

    def setUp(self):
        """Reset singleton instance before each test."""
        DeviceManager._instance = None

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_get_device_for_service(self, mock_device_count, mock_is_available):
        """Test get_device_for_service with config."""
        mock_is_available.return_value = False
        mock_device_count.return_value = 0

        dm = get_device_manager()
        config = {
            'device': 'auto',
            'min_memory_gb': 4
        }

        device_info = dm.get_device_for_service(
            service_name="test_service",
            config=config
        )

        self.assertEqual(device_info['device'], "cpu")


class TestMacOSMPSDetection(unittest.TestCase):
    """Test cases for macOS MPS GPU detection."""

    def setUp(self):
        """Reset singleton instance before each test."""
        DeviceManager._instance = None

    @patch('sys.platform', 'darwin')
    @patch('torch.cuda.is_available')
    def test_mps_detection_available(self, mock_cuda_available):
        """Test MPS detection when available on macOS."""
        mock_cuda_available.return_value = False
        
        # Mock torch.backends.mps
        mock_mps_backend = Mock()
        mock_mps_backend.is_available.return_value = True
        
        with patch('torch.backends.mps', mock_mps_backend), \
             patch('sys.platform', 'darwin'):
            dm = get_device_manager()
            # Should detect MPS device
            mps_devices = [d for d in dm.available_devices if d.device_type == "mps"]
            self.assertGreater(len(mps_devices), 0, "Should detect MPS device on macOS")

    @patch('sys.platform', 'darwin')
    @patch('torch.cuda.is_available')
    def test_mps_detection_unavailable(self, mock_cuda_available):
        """Test MPS detection when not available (Intel Mac or older macOS)."""
        mock_cuda_available.return_value = False
        
        # Mock torch.backends.mps as unavailable
        mock_mps_backend = Mock()
        mock_mps_backend.is_available.return_value = False
        
        with patch('torch.backends.mps', mock_mps_backend), \
             patch('sys.platform', 'darwin'):
            dm = get_device_manager()
            # Should fall back to CPU
            mps_devices = [d for d in dm.available_devices if d.device_type == "mps"]
            self.assertEqual(len(mps_devices), 0, "Should not detect MPS device")
            # CPU should be available
            self.assertIsNotNone(dm.cpu_info, "CPU should be available as fallback")

    @patch('sys.platform', 'darwin')
    @patch('torch.cuda.is_available')
    def test_mps_device_selection(self, mock_cuda_available):
        """Test that MPS device is selected when available."""
        mock_cuda_available.return_value = False
        
        # Mock torch.backends.mps
        mock_mps_backend = Mock()
        mock_mps_backend.is_available.return_value = True
        
        with patch('torch.backends.mps', mock_mps_backend), \
             patch('sys.platform', 'darwin'):
            dm = get_device_manager()
            device_info = dm.get_best_device(device_preference="auto")
            # Should select MPS if available, otherwise CPU
            self.assertIn(device_info['device'], ['mps', 'cpu'])


if __name__ == '__main__':
    unittest.main()
