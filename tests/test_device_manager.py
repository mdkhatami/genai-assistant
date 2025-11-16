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
        self.assertEqual(dm.device_count, 0)

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
        mock_get_props.return_value = mock_props

        dm = get_device_manager()
        self.assertEqual(dm.device_count, 1)
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


if __name__ == '__main__':
    unittest.main()
