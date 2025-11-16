"""
Pydantic Models for FastAPI Request/Response Validation

This module contains all Pydantic models used for API request and response validation
in the GenAI Assistant FastAPI application.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


# ============================================================================
# Authentication Models
# ============================================================================

class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., description="Username for authentication")
    password: str = Field(..., description="Password for authentication")


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    username: str = Field(..., description="Authenticated username")


# ============================================================================
# LLM Models
# ============================================================================

class LLMRequest(BaseModel):
    """LLM request model."""
    prompt: str = Field(..., description="The prompt to send to the LLM")
    system_message: Optional[str] = Field(None, description="System message for context")
    model: Optional[str] = Field(None, description="Model to use (overrides default)")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Sampling temperature (0-2)")
    top_p: Optional[float] = Field(None, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(None, description="Frequency penalty (-2 to 2)")
    presence_penalty: Optional[float] = Field(None, description="Presence penalty (-2 to 2)")


class LLMResponse(BaseModel):
    """LLM response model."""
    response: str = Field(..., description="Generated response text")
    model: str = Field(..., description="Model used for generation")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


# ============================================================================
# Image Generation Models
# ============================================================================

class ImageGenerationRequest(BaseModel):
    """Image generation request model."""
    prompt: str = Field(..., description="Text prompt for image generation")
    model: Optional[str] = Field(None, description="Model to use (e.g., flux-dev)")
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")
    steps: Optional[int] = Field(None, description="Number of inference steps")
    guidance_scale: Optional[float] = Field(None, description="Guidance scale for generation")
    num_images: Optional[int] = Field(1, description="Number of images to generate")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")


class ImageItem(BaseModel):
    """Single image item in response."""
    image_path: str = Field(..., description="Path to saved image file")
    image_data: str = Field(..., description="Base64 encoded image data")
    generation_time: float = Field(..., description="Time taken to generate this image")
    error: Optional[str] = Field(None, description="Error message if generation failed")


class ImageGenerationResponse(BaseModel):
    """Image generation response model."""
    images: List[ImageItem] = Field(..., description="List of generated images")
    prompt: str = Field(..., description="Original prompt used")
    model: str = Field(..., description="Model used for generation")
    parameters: Dict[str, Any] = Field(..., description="Generation parameters used")
    processing_time: float = Field(..., description="Total processing time in seconds")


# ============================================================================
# Transcription Models
# ============================================================================

class TranscriptionRequest(BaseModel):
    """Transcription request model."""
    model_type: Optional[str] = Field("faster-whisper", description="Model type (whisper or faster-whisper)")
    model_name: Optional[str] = Field("base", description="Model size (tiny, base, small, medium, large)")
    language: Optional[str] = Field("auto", description="Language code or 'auto' for auto-detection")
    task: Optional[str] = Field("transcribe", description="Task type (transcribe or translate)")
    condition_on_previous_text: Optional[bool] = Field(True, description="Condition on previous text")
    initial_prompt: Optional[str] = Field(None, description="Initial prompt for context")
    word_timestamps: Optional[bool] = Field(False, description="Include word-level timestamps")
    device: Optional[str] = Field("auto", description="Device to use (auto, cuda, cpu)")
    gpu_index: Optional[int] = Field(None, description="GPU index to use")
    compute_type: Optional[str] = Field("float16", description="Compute type (float16, float32, int8)")
    cpu_threads: Optional[int] = Field(4, description="Number of CPU threads")
    num_workers: Optional[int] = Field(1, description="Number of workers")


class TranscriptionResponse(BaseModel):
    """Transcription response model."""
    text: str = Field(..., description="Transcribed text")
    language: Optional[str] = Field(None, description="Detected language")
    model: str = Field(..., description="Model used for transcription")
    processing_time: float = Field(..., description="Processing time in seconds")
    segments: Optional[List[Dict[str, Any]]] = Field(None, description="Detailed segments with timestamps")


# ============================================================================
# Model List Models
# ============================================================================

class ModelInfo(BaseModel):
    """Model information."""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type (llm, image_generation, transcription)")
    description: Optional[str] = Field(None, description="Model description")


class ModelListResponse(BaseModel):
    """Model list response."""
    models: List[ModelInfo] = Field(..., description="List of available models")
    count: int = Field(..., description="Total number of models")


# ============================================================================
# Health Check Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall system status (healthy, degraded, unavailable)")
    timestamp: datetime = Field(..., description="Timestamp of health check")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Status of individual components")


# ============================================================================
# File Upload Models
# ============================================================================

class FileUploadResponse(BaseModel):
    """File upload response model."""
    filename: str = Field(..., description="Uploaded filename")
    file_path: str = Field(..., description="Path where file is stored")
    file_size: int = Field(..., description="File size in bytes")
    content_type: Optional[str] = Field(None, description="File content type")
    upload_time: datetime = Field(..., description="Upload timestamp")

