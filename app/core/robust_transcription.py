"""
Robust Transcription System

This module provides a production-ready transcription system that properly handles
both OpenAI Whisper and Faster-Whisper with correct parameter mapping and error handling.
"""

import os
import logging
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Standardized transcription result."""
    text: str
    language: str
    model: str
    segments: Optional[List[Dict[str, Any]]] = None
    transcription_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TranscriptionConfig:
    """Configuration for transcription request."""
    # Core parameters
    audio_path: str
    model_type: str = "faster-whisper"  # "whisper" or "faster-whisper"
    model_name: str = "base"
    language: Optional[str] = None  # None for auto-detect
    task: str = "transcribe"  # "transcribe" or "translate"
    
    # Prompting
    initial_prompt: Optional[str] = None
    
    # Advanced options
    condition_on_previous_text: bool = True
    word_timestamps: bool = False
    
    # Device configuration
    device: str = "cuda"
    gpu_index: Optional[int] = 0
    
    # Faster-Whisper specific
    compute_type: str = "float16"
    cpu_threads: int = 4
    num_workers: int = 1
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    temperature: Union[float, List[float]] = 0.0
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6


class WhisperTranscriber:
    """OpenAI Whisper implementation."""
    
    def __init__(self, model_name: str = "base", device: str = "cuda", gpu_index: Optional[int] = 0):
        """Initialize OpenAI Whisper model."""
        import whisper

        self.model_name = model_name
        self.device = device
        self.gpu_index = gpu_index
        self.target_device = "cpu"  # Default to CPU

        logger.info(f"Loading OpenAI Whisper model: {model_name}")

        try:
            # Determine target device
            if device == "cuda":
                import torch
                if torch.cuda.is_available():
                    # Use specified GPU index, default to 0
                    if gpu_index is not None and gpu_index < torch.cuda.device_count():
                        self.target_device = f"cuda:{gpu_index}"
                        logger.info(f"Will use GPU {gpu_index} for OpenAI Whisper")
                    else:
                        self.target_device = "cuda:0"
                        logger.info(f"Will use default GPU (cuda:0) for OpenAI Whisper")
                else:
                    logger.warning("CUDA not available, falling back to CPU")
                    self.device = "cpu"
                    self.target_device = "cpu"
            else:
                logger.info("OpenAI Whisper using CPU")
                self.target_device = "cpu"

            # Load model
            self.model = whisper.load_model(model_name)

            # Move to target device
            self.model = self.model.to(self.target_device)
            logger.info(f"OpenAI Whisper {model_name} loaded on {self.target_device}")

        except Exception as e:
            logger.error(f"Failed to load OpenAI Whisper: {e}")
            raise
    
    def transcribe(self, config: TranscriptionConfig) -> TranscriptionResult:
        """Transcribe audio using OpenAI Whisper."""
        start_time = time.time()
        
        try:
            # Prepare parameters for OpenAI Whisper
            whisper_params = {
                'language': config.language,
                'task': config.task,
                'condition_on_previous_text': config.condition_on_previous_text,
                'word_timestamps': config.word_timestamps,
            }
            
            # Add initial prompt if provided
            if config.initial_prompt:
                whisper_params['initial_prompt'] = config.initial_prompt
            
            # Add temperature
            if isinstance(config.temperature, (int, float)):
                whisper_params['temperature'] = config.temperature
            elif isinstance(config.temperature, list) and config.temperature:
                whisper_params['temperature'] = config.temperature[0]
            
            # Remove None values
            whisper_params = {k: v for k, v in whisper_params.items() if v is not None}
            
            logger.info(f"Transcribing with OpenAI Whisper: {whisper_params}")
            
            # Transcribe
            result = self.model.transcribe(config.audio_path, **whisper_params)
            
            transcription_time = time.time() - start_time
            
            return TranscriptionResult(
                text=result["text"],
                language=result.get("language", config.language or "unknown"),
                model=f"openai-whisper-{self.model_name}",
                segments=result.get("segments", []),
                transcription_time=transcription_time,
                metadata={
                    "task": config.task,
                    "device": self.device,
                    "parameters": whisper_params
                }
            )
            
        except Exception as e:
            error_msg = f"OpenAI Whisper transcription failed: {str(e)}"
            logger.error(error_msg)
            return TranscriptionResult(
                text="",
                language=config.language or "unknown",
                model=f"openai-whisper-{self.model_name}",
                error=error_msg,
                transcription_time=time.time() - start_time
            )


class FasterWhisperTranscriber:
    """Faster-Whisper implementation."""
    
    def __init__(self, model_name: str = "base", device: str = "cuda", gpu_index: Optional[int] = 0,
                 compute_type: str = "float16", cpu_threads: int = 4, num_workers: int = 1):
        """Initialize Faster-Whisper model."""
        from faster_whisper import WhisperModel

        self.model_name = model_name
        self.device = device
        self.gpu_index = gpu_index
        self.compute_type = compute_type

        logger.info(f"Loading Faster-Whisper model: {model_name}")

        try:
            # Determine target device and adjust compute_type if needed
            target_device = device
            target_compute_type = compute_type

            if device == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Validate GPU index
                        if gpu_index is not None and gpu_index < torch.cuda.device_count():
                            target_device = f"cuda:{gpu_index}"
                            logger.info(f"Using GPU {gpu_index} for Faster-Whisper")
                        else:
                            target_device = "cuda:0"
                            logger.info(f"Using default GPU (cuda:0) for Faster-Whisper")
                    else:
                        logger.warning("CUDA requested but not available, falling back to CPU")
                        target_device = "cpu"
                        self.device = "cpu"
                        # Force float32 for CPU
                        if compute_type == "float16":
                            target_compute_type = "float32"
                            logger.info("Adjusted compute_type to float32 for CPU")
                except ImportError:
                    logger.warning("PyTorch not available for CUDA check, using CPU")
                    target_device = "cpu"
                    self.device = "cpu"
                    if compute_type == "float16":
                        target_compute_type = "float32"
            else:
                # CPU mode - ensure compatible compute type
                if compute_type == "float16":
                    target_compute_type = "float32"
                    logger.info("Adjusted compute_type to float32 for CPU")

            # Faster-Whisper expects simple "cuda" or "cpu", will use CUDA_VISIBLE_DEVICES if set
            # We'll use device_index parameter if Faster-Whisper supports it, otherwise just "cuda"
            simple_device = "cuda" if target_device.startswith("cuda") else "cpu"

            self.model = WhisperModel(
                model_name,
                device=simple_device,
                device_index=gpu_index if simple_device == "cuda" and gpu_index is not None else 0,
                compute_type=target_compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers
            )

            logger.info(f"Faster-Whisper {model_name} loaded on {target_device} with compute_type={target_compute_type}")

        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper: {e}")
            raise
    
    def transcribe(self, config: TranscriptionConfig) -> TranscriptionResult:
        """Transcribe audio using Faster-Whisper."""
        start_time = time.time()
        
        try:
            # Prepare parameters for Faster-Whisper
            faster_whisper_params = {
                'language': config.language,
                'task': config.task,
                'condition_on_previous_text': config.condition_on_previous_text,
                'word_timestamps': config.word_timestamps,
                'beam_size': config.beam_size,
                'best_of': config.best_of,
                'patience': config.patience,
                'length_penalty': config.length_penalty,
                'repetition_penalty': config.repetition_penalty,
                'temperature': config.temperature,
                'compression_ratio_threshold': config.compression_ratio_threshold,
                'log_prob_threshold': config.log_prob_threshold,
                'no_speech_threshold': config.no_speech_threshold,
            }
            
            # Add initial prompt if provided
            if config.initial_prompt:
                faster_whisper_params['initial_prompt'] = config.initial_prompt
            
            # Remove None values
            faster_whisper_params = {k: v for k, v in faster_whisper_params.items() if v is not None}
            
            logger.info(f"Transcribing with Faster-Whisper: {faster_whisper_params}")
            
            # Transcribe
            segments, info = self.model.transcribe(config.audio_path, **faster_whisper_params)
            
            # Collect results
            text = ""
            segment_list = []
            
            for segment in segments:
                text += segment.text
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
                if config.word_timestamps and hasattr(segment, 'words') and segment.words:
                    segment_dict["words"] = [
                        {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": word.probability
                        }
                        for word in segment.words
                    ]
                segment_list.append(segment_dict)
            
            transcription_time = time.time() - start_time
            
            return TranscriptionResult(
                text=text,
                language=info.language,
                model=f"faster-whisper-{self.model_name}",
                segments=segment_list,
                transcription_time=transcription_time,
                metadata={
                    "task": config.task,
                    "device": self.device,
                    "language_probability": info.language_probability,
                    "duration": info.duration,
                    "parameters": faster_whisper_params
                }
            )
            
        except Exception as e:
            error_msg = f"Faster-Whisper transcription failed: {str(e)}"
            logger.error(error_msg)
            return TranscriptionResult(
                text="",
                language=config.language or "unknown",
                model=f"faster-whisper-{self.model_name}",
                error=error_msg,
                transcription_time=time.time() - start_time
            )


class RobustTranscriptionManager:
    """
    Production-ready transcription manager that handles both Whisper implementations.
    Includes model caching with LRU eviction policy and memory limits.
    """

    def __init__(self, max_cached_models: int = 3):
        """
        Initialize the transcription manager.

        Args:
            max_cached_models: Maximum number of models to cache (default: 3)
                              When exceeded, least recently used models are evicted
        """
        self._whisper_models: Dict[str, WhisperTranscriber] = {}
        self._faster_whisper_models: Dict[str, FasterWhisperTranscriber] = {}
        self._model_access_order: List[str] = []  # Track LRU order
        self.max_cached_models = max_cached_models

        self.available_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        self.supported_languages = [
            "auto", "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl",
            "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da",
            "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te",
            "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne",
            "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af",
            "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk",
            "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba",
            "jw", "su"
        ]

        logger.info(f"RobustTranscriptionManager initialized (max_cached_models={max_cached_models})")
    
    def _get_model_key(self, model_type: str, model_name: str, device: str, gpu_index: Optional[int]) -> str:
        """Generate cache key for model."""
        return f"{model_type}-{model_name}-{device}-{gpu_index}"
    
    def _evict_lru_model_if_needed(self):
        """Evict least recently used model if cache is full."""
        total_models = len(self._whisper_models) + len(self._faster_whisper_models)

        if total_models >= self.max_cached_models and self._model_access_order:
            # Find least recently used model
            lru_key = self._model_access_order[0]

            # Remove from cache
            if lru_key in self._whisper_models:
                logger.info(f"Evicting LRU Whisper model: {lru_key}")
                model = self._whisper_models.pop(lru_key)
                # Clean up GPU memory
                try:
                    import torch
                    if hasattr(model, 'model'):
                        del model.model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Could not clean up model memory: {e}")

            elif lru_key in self._faster_whisper_models:
                logger.info(f"Evicting LRU Faster-Whisper model: {lru_key}")
                model = self._faster_whisper_models.pop(lru_key)
                # Clean up
                try:
                    if hasattr(model, 'model'):
                        del model.model
                except Exception as e:
                    logger.warning(f"Could not clean up model: {e}")

            # Remove from access order
            self._model_access_order.remove(lru_key)

    def _update_access_order(self, key: str):
        """Update model access order for LRU tracking."""
        if key in self._model_access_order:
            self._model_access_order.remove(key)
        self._model_access_order.append(key)

    def _get_or_create_whisper_model(self, config: TranscriptionConfig) -> WhisperTranscriber:
        """Get or create OpenAI Whisper model."""
        key = self._get_model_key("whisper", config.model_name, config.device, config.gpu_index)

        if key not in self._whisper_models:
            # Evict LRU model if cache is full
            self._evict_lru_model_if_needed()

            logger.info(f"Creating new OpenAI Whisper model: {key}")
            self._whisper_models[key] = WhisperTranscriber(
                model_name=config.model_name,
                device=config.device,
                gpu_index=config.gpu_index
            )

        # Update access order
        self._update_access_order(key)

        return self._whisper_models[key]
    
    def _get_or_create_faster_whisper_model(self, config: TranscriptionConfig) -> FasterWhisperTranscriber:
        """Get or create Faster-Whisper model."""
        key = self._get_model_key("faster-whisper", config.model_name, config.device, config.gpu_index)

        if key not in self._faster_whisper_models:
            # Evict LRU model if cache is full
            self._evict_lru_model_if_needed()

            logger.info(f"Creating new Faster-Whisper model: {key}")
            self._faster_whisper_models[key] = FasterWhisperTranscriber(
                model_name=config.model_name,
                device=config.device,
                gpu_index=config.gpu_index,
                compute_type=config.compute_type,
                cpu_threads=config.cpu_threads,
                num_workers=config.num_workers
            )

        # Update access order
        self._update_access_order(key)

        return self._faster_whisper_models[key]
    
    def transcribe(self, config: TranscriptionConfig) -> TranscriptionResult:
        """
        Transcribe audio using the specified configuration.
        
        Args:
            config: TranscriptionConfig with all parameters
            
        Returns:
            TranscriptionResult
        """
        try:
            # Validate configuration
            self._validate_config(config)
            
            # Get appropriate model and transcribe
            if config.model_type == "whisper":
                model = self._get_or_create_whisper_model(config)
                return model.transcribe(config)
            elif config.model_type == "faster-whisper":
                model = self._get_or_create_faster_whisper_model(config)
                return model.transcribe(config)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptionResult(
                text="",
                language=config.language or "unknown",
                model=f"{config.model_type}-{config.model_name}",
                error=str(e),
                transcription_time=0.0
            )
    
    def _validate_config(self, config: TranscriptionConfig) -> None:
        """Validate transcription configuration."""
        if not os.path.exists(config.audio_path):
            raise FileNotFoundError(f"Audio file not found: {config.audio_path}")
        
        if config.model_type not in ["whisper", "faster-whisper"]:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        if config.model_name not in self.available_models:
            raise ValueError(f"Unsupported model: {config.model_name}")
        
        if config.task not in ["transcribe", "translate"]:
            raise ValueError(f"Unsupported task: {config.task}")
        
        if config.language and config.language not in self.supported_languages:
            logger.warning(f"Language '{config.language}' may not be supported")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "whisper_models_loaded": len(self._whisper_models),
            "faster_whisper_models_loaded": len(self._faster_whisper_models),
            "available_models": self.available_models,
            "supported_languages": len(self.supported_languages),
            "whisper_model_keys": list(self._whisper_models.keys()),
            "faster_whisper_model_keys": list(self._faster_whisper_models.keys())
        }
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        # Clear GPU memory for whisper models
        for model in self._whisper_models.values():
            try:
                import torch
                if hasattr(model, 'model'):
                    del model.model
                torch.cuda.empty_cache()
            except:
                pass
        
        # Clear faster-whisper models
        for model in self._faster_whisper_models.values():
            try:
                if hasattr(model, 'model'):
                    del model.model
            except:
                pass
        
        self._whisper_models.clear()
        self._faster_whisper_models.clear()
        logger.info("Model cache cleared")


# Global manager instance
_manager = None

def get_robust_transcription_manager() -> RobustTranscriptionManager:
    """Get the global robust transcription manager."""
    global _manager
    if _manager is None:
        _manager = RobustTranscriptionManager()
    return _manager
