"""
Test suite for Image Generation Core functionality.

This module tests the image generation functionality with comprehensive
coverage of all features including error handling, parameter validation, and
result formatting.

Note: These are unit tests using mocks. For integration tests that use actual
API calls, see test_integration.py. To run only integration tests:
    python -m pytest tests/test_integration.py -v
"""

import os
import sys
import unittest
import threading
from unittest.mock import Mock, patch
import tempfile
from PIL import Image
import torch

# Add the parent directory to the path to import core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.optimized_image_generation import OptimizedImageGenerator, ImageGenerationResult, get_optimized_generator


class TestImageGenerationResult(unittest.TestCase):
    """Test cases for ImageGenerationResult data class."""
    
    def test_image_generation_result_creation(self):
        """Test ImageGenerationResult object creation with all parameters."""
        # Create a test image
        test_image = Image.new('RGB', (512, 512), color='red')
        
        result = ImageGenerationResult(
            image=test_image,
            prompt="Test prompt",
            model="black-forest-labs/FLUX.1-dev",
            generation_time=1.5,
            error=None,
            metadata={"test": "data"}
        )
        
        self.assertEqual(result.prompt, "Test prompt")
        self.assertEqual(result.model, "black-forest-labs/FLUX.1-dev")
        self.assertEqual(result.generation_time, 1.5)
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["test"], "data")
        self.assertEqual(result.image.size, (512, 512))
    
    def test_image_generation_result_minimal(self):
        """Test ImageGenerationResult object creation with minimal parameters."""
        test_image = Image.new('RGB', (256, 256), color='blue')
        
        result = ImageGenerationResult(
            image=test_image,
            prompt="Test prompt",
            model="black-forest-labs/FLUX.1-dev"
        )
        
        self.assertEqual(result.prompt, "Test prompt")
        self.assertEqual(result.model, "black-forest-labs/FLUX.1-dev")
        self.assertIsNone(result.generation_time)
        self.assertIsNone(result.error)
        self.assertIsNone(result.metadata)
        self.assertEqual(result.image.size, (256, 256))


class TestOptimizedImageGenerator(unittest.TestCase):
    """Test cases for Optimized Image Generator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        # Reset singleton instance before each test
        import app.core.optimized_image_generation as img_gen_module
        img_gen_module._generator_instance = None
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Reset singleton instance after each test
        import app.core.optimized_image_generation as img_gen_module
        img_gen_module._generator_instance = None
    
    @patch('app.core.optimized_image_generation.FluxPipeline')
    def test_generator_initialization(self, mock_pipeline):
        """Test OptimizedImageGenerator initialization."""
        # Mock pipeline
        mock_pipe = Mock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipeline.from_pretrained.return_value = mock_pipe
        
        generator = OptimizedImageGenerator(
            model_name="black-forest-labs/FLUX.1-dev",
            device="cpu",
            gpu_index=None
        )
        
        self.assertEqual(generator.model_name, "black-forest-labs/FLUX.1-dev")
        self.assertEqual(generator.device, "cpu")
        self.assertTrue(generator.is_loaded)
        mock_pipeline.from_pretrained.assert_called_once()
    
    @patch('app.core.optimized_image_generation.FluxPipeline')
    def test_generator_cuda_fallback(self, mock_pipeline):
        """Test OptimizedImageGenerator CUDA fallback to CPU."""
        # Mock CUDA not available
        with patch('torch.cuda.is_available', return_value=False):
            mock_pipe = Mock()
            mock_pipe.to.return_value = mock_pipe
            mock_pipeline.from_pretrained.return_value = mock_pipe
            
            generator = OptimizedImageGenerator(device="cuda", gpu_index=None)
            
            self.assertEqual(generator.device, "cpu")
    
    @patch('app.core.optimized_image_generation.FluxPipeline')
    def test_get_available_models(self, mock_pipeline):
        """Test getting available models."""
        mock_pipe = Mock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipeline.from_pretrained.return_value = mock_pipe
        
        generator = OptimizedImageGenerator(device="cpu", gpu_index=None)
        models = generator.get_available_models()
        
        self.assertIsInstance(models, list)
        self.assertIn("flux", models)
        self.assertIn("flux-dev", models)
    
    @patch('app.core.optimized_image_generation.FluxPipeline')
    def test_get_model_info(self, mock_pipeline):
        """Test getting model information."""
        mock_pipe = Mock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipeline.from_pretrained.return_value = mock_pipe
        
        generator = OptimizedImageGenerator(device="cpu", gpu_index=None)
        info = generator.get_model_info()
        
        self.assertIn("model_name", info)
        self.assertIn("device", info)
        self.assertIn("is_loaded", info)
        self.assertEqual(info["model_name"], "black-forest-labs/FLUX.1-dev")
    
    @patch('app.core.optimized_image_generation.FluxPipeline')
    def test_generate_image_success(self, mock_pipeline):
        """Test successful image generation."""
        # Mock pipeline and result
        mock_pipe = Mock()
        mock_pipe.to.return_value = mock_pipe
        
        # Create test image and mock the pipeline call
        test_image = Image.new('RGB', (1024, 1024), color='green')
        mock_result = Mock()
        mock_result.images = [test_image]
        mock_pipe.return_value = mock_result
        
        mock_pipeline.from_pretrained.return_value = mock_pipe
        
        generator = OptimizedImageGenerator(device="cpu", gpu_index=None)
        generator.pipeline = mock_pipe
        
        result = generator.generate_image("Test prompt", width=1024, height=1024)
        
        self.assertIsInstance(result, ImageGenerationResult)
        self.assertEqual(result.prompt, "Test prompt")
        self.assertEqual(result.model, "black-forest-labs/FLUX.1-dev")
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.generation_time)
        self.assertEqual(result.image.size, (1024, 1024))
    
    @patch('app.core.optimized_image_generation.FluxPipeline')
    def test_generate_multiple_images(self, mock_pipeline):
        """Test generating multiple images."""
        mock_pipe = Mock()
        mock_pipe.to.return_value = mock_pipe
        
        # Create test images
        test_images = [
            Image.new('RGB', (1024, 1024), color='red'),
            Image.new('RGB', (1024, 1024), color='blue'),
            Image.new('RGB', (1024, 1024), color='green')
        ]
        mock_result = Mock()
        mock_result.images = test_images
        mock_pipe.return_value = mock_result
        
        mock_pipeline.from_pretrained.return_value = mock_pipe
        
        generator = OptimizedImageGenerator(device="cpu", gpu_index=None)
        generator.pipeline = mock_pipe
        
        results = generator.generate_image("Test prompt", num_images=3)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result.prompt, "Test prompt")
            self.assertEqual(result.metadata["image_index"], i)
    
    @patch('app.core.optimized_image_generation.FluxPipeline')
    def test_generate_image_error_handling(self, mock_pipeline):
        """Test error handling in image generation."""
        mock_pipe = Mock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipeline.from_pretrained.return_value = mock_pipe
        
        generator = OptimizedImageGenerator(device="cpu", gpu_index=None)
        generator.pipeline = mock_pipe
        
        # Simulate pipeline error during generation
        mock_pipe.side_effect = Exception("Pipeline error")
        
        result = generator.generate_image("Test prompt")
        
        self.assertIsInstance(result, ImageGenerationResult)
        self.assertIsNotNone(result.error)
        self.assertIn("Pipeline error", result.error)


if __name__ == '__main__':
    unittest.main()
