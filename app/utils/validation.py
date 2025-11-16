"""
Validation utilities for file uploads and inputs.
"""

import os
from typing import List, Tuple
from fastapi import UploadFile, HTTPException, status


# Allowed file extensions for audio/video transcription
ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.opus'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
ALLOWED_TRANSCRIPTION_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# Allowed MIME types for audio/video
ALLOWED_AUDIO_MIME_TYPES = {
    'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/wave', 'audio/x-wav',
    'audio/mp4', 'audio/m4a', 'audio/x-m4a', 'audio/ogg', 'audio/vorbis',
    'audio/flac', 'audio/x-flac', 'audio/aac', 'audio/x-aac',
    'audio/wma', 'audio/x-ms-wma', 'audio/opus'
}
ALLOWED_VIDEO_MIME_TYPES = {
    'video/mp4', 'video/x-msvideo', 'video/avi', 'video/quicktime',
    'video/x-matroska', 'video/webm', 'video/x-flv', 'video/x-ms-wmv',
    'video/m4v'
}
ALLOWED_TRANSCRIPTION_MIME_TYPES = ALLOWED_AUDIO_MIME_TYPES | ALLOWED_VIDEO_MIME_TYPES


def validate_file_size(file: UploadFile, max_size_bytes: int) -> None:
    """
    Validate file size before reading.
    
    Args:
        file: The uploaded file
        max_size_bytes: Maximum allowed file size in bytes
        
    Raises:
        HTTPException: If file size exceeds limit
    """
    # Check content-length header if available
    if hasattr(file, 'size') and file.size and file.size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({file.size} bytes) exceeds maximum allowed size ({max_size_bytes} bytes)"
        )


def validate_file_type(file: UploadFile, allowed_extensions: set, allowed_mime_types: set) -> Tuple[str, str]:
    """
    Validate file type by extension and MIME type.
    
    Args:
        file: The uploaded file
        allowed_extensions: Set of allowed file extensions (e.g., {'.mp3', '.wav'})
        allowed_mime_types: Set of allowed MIME types
        
    Returns:
        Tuple of (file_extension, mime_type)
        
    Raises:
        HTTPException: If file type is not allowed
    """
    # Get file extension
    filename = file.filename or ""
    file_extension = os.path.splitext(filename.lower())[1]
    
    # Get MIME type
    mime_type = file.content_type or ""
    
    # Validate extension
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{file_extension}' is not allowed. Allowed types: {', '.join(sorted(allowed_extensions))}"
        )
    
    # Validate MIME type (if provided)
    if mime_type and mime_type not in allowed_mime_types:
        # Log warning but don't fail if extension is valid (MIME types can be unreliable)
        # In production, you might want to be stricter
        pass
    
    return file_extension, mime_type


def validate_transcription_file(file: UploadFile, max_size_bytes: int = 16 * 1024 * 1024) -> Tuple[str, str]:
    """
    Validate file for transcription (audio/video).
    
    Args:
        file: The uploaded file
        max_size_bytes: Maximum allowed file size in bytes (default: 16MB)
        
    Returns:
        Tuple of (file_extension, mime_type)
        
    Raises:
        HTTPException: If file is invalid
    """
    # Validate file size
    validate_file_size(file, max_size_bytes)
    
    # Validate file type
    return validate_file_type(
        file,
        ALLOWED_TRANSCRIPTION_EXTENSIONS,
        ALLOWED_TRANSCRIPTION_MIME_TYPES
    )

