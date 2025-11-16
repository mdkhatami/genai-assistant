"""
Test suite for Transcription Core functionality.

This module tests the transcription functionality with comprehensive
coverage of all features including error handling, parameter validation, and
result formatting.

Note: These are unit tests using mocks. For integration tests that use actual
API calls, see test_integration.py. To run only integration tests:
    python -m pytest tests/test_integration.py -v
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch
import tempfile

# Add the parent directory to the path to import core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.robust_transcription import (
    get_robust_transcription_manager, 
    TranscriptionConfig, 
    TranscriptionResult,
    RobustTranscriptionManager
)


class TestTranscriptionResult(unittest.TestCase):
    """Test cases for TranscriptionResult data class."""
    
    def test_transcription_result_creation(self):
        """Test TranscriptionResult object creation with all parameters."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            model="whisper-base",
            segments=[{"start": 0, "end": 2, "text": "Hello"}],
            transcription_time=1.5,
            error=None,
            metadata={"test": "data"}
        )
        
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.model, "whisper-base")
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.transcription_time, 1.5)
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["test"], "data")
    
    def test_transcription_result_minimal(self):
        """Test TranscriptionResult object creation with minimal parameters."""
        result = TranscriptionResult(
            text="Test transcription",
            language="en",
            model="whisper-base"
        )
        
        self.assertEqual(result.text, "Test transcription")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.model, "whisper-base")
        self.assertIsNone(result.segments)
        self.assertIsNone(result.transcription_time)
        self.assertIsNone(result.error)
        self.assertIsNone(result.metadata)


class TestTranscriptionConfig(unittest.TestCase):
    """Test cases for TranscriptionConfig data class."""
    
    def test_transcription_config_creation(self):
        """Test TranscriptionConfig object creation."""
        config = TranscriptionConfig(
            audio_path="/path/to/audio.wav",
            model_type="faster-whisper",
            model_name="base",
            language="en",
            task="transcribe"
        )
        
        self.assertEqual(config.audio_path, "/path/to/audio.wav")
        self.assertEqual(config.model_type, "faster-whisper")
        self.assertEqual(config.model_name, "base")
        self.assertEqual(config.language, "en")
        self.assertEqual(config.task, "transcribe")
    
    def test_transcription_config_defaults(self):
        """Test TranscriptionConfig with default values."""
        config = TranscriptionConfig(audio_path="/path/to/audio.wav")
        
        self.assertEqual(config.model_type, "faster-whisper")
        self.assertEqual(config.model_name, "base")
        self.assertIsNone(config.language)
        self.assertEqual(config.task, "transcribe")
        self.assertEqual(config.device, "cuda")


class TestRobustTranscriptionManager(unittest.TestCase):
    """Test cases for RobustTranscriptionManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        # Create a dummy audio file for testing
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        with open(self.test_audio_path, 'wb') as f:
            f.write(b"dummy audio file content")
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test RobustTranscriptionManager initialization."""
        manager = RobustTranscriptionManager()
        
        self.assertIsInstance(manager.available_models, list)
        self.assertIn("base", manager.available_models)
        self.assertIn("large", manager.available_models)
        self.assertIsInstance(manager.supported_languages, list)
        self.assertIn("en", manager.supported_languages)
    
    def test_get_robust_transcription_manager_singleton(self):
        """Test that get_robust_transcription_manager returns singleton."""
        manager1 = get_robust_transcription_manager()
        manager2 = get_robust_transcription_manager()
        
        self.assertIs(manager1, manager2)
    
    @patch('app.core.robust_transcription.WhisperTranscriber')
    def test_transcribe_with_whisper(self, mock_whisper_class):
        """Test transcription with OpenAI Whisper."""
        # Mock transcriber
        mock_transcriber = Mock()
        mock_transcriber.transcribe.return_value = TranscriptionResult(
            text="Hello world",
            language="en",
            model="openai-whisper-base",
            segments=[],
            transcription_time=1.0
        )
        mock_whisper_class.return_value = mock_transcriber
        
        manager = RobustTranscriptionManager()
        config = TranscriptionConfig(
            audio_path=self.test_audio_path,
            model_type="whisper",
            model_name="base",
            device="cpu"
        )
        
        result = manager.transcribe(config)
        
        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.model, "openai-whisper-base")
    
    @patch('app.core.robust_transcription.FasterWhisperTranscriber')
    def test_transcribe_with_faster_whisper(self, mock_faster_whisper_class):
        """Test transcription with Faster-Whisper."""
        # Mock transcriber
        mock_transcriber = Mock()
        mock_transcriber.transcribe.return_value = TranscriptionResult(
            text="Hello world",
            language="en",
            model="faster-whisper-base",
            segments=[],
            transcription_time=0.5
        )
        mock_faster_whisper_class.return_value = mock_transcriber
        
        manager = RobustTranscriptionManager()
        config = TranscriptionConfig(
            audio_path=self.test_audio_path,
            model_type="faster-whisper",
            model_name="base",
            device="cpu"
        )
        
        result = manager.transcribe(config)
        
        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.model, "faster-whisper-base")
    
    def test_transcribe_file_not_found(self):
        """Test transcription with non-existent file."""
        manager = RobustTranscriptionManager()
        config = TranscriptionConfig(
            audio_path="non_existent_file.wav",
            model_type="faster-whisper",
            model_name="base"
        )
        
        result = manager.transcribe(config)
        
        self.assertIsInstance(result, TranscriptionResult)
        self.assertIsNotNone(result.error)
        self.assertIn("not found", result.error.lower())
    
    def test_validate_config_invalid_model_type(self):
        """Test validation with invalid model type."""
        manager = RobustTranscriptionManager()
        config = TranscriptionConfig(
            audio_path=self.test_audio_path,
            model_type="invalid-type",
            model_name="base"
        )
        
        result = manager.transcribe(config)
        
        self.assertIsInstance(result, TranscriptionResult)
        self.assertIsNotNone(result.error)
    
    def test_get_model_info(self):
        """Test getting model information."""
        manager = RobustTranscriptionManager()
        info = manager.get_model_info()
        
        self.assertIn("whisper_models_loaded", info)
        self.assertIn("faster_whisper_models_loaded", info)
        self.assertIn("available_models", info)
        self.assertIn("supported_languages", info)
    
    def test_clear_cache(self):
        """Test clearing model cache."""
        manager = RobustTranscriptionManager()
        
        # Should not raise an exception
        manager.clear_cache()


if __name__ == '__main__':
    unittest.main()
