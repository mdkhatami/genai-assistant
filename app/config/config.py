"""
Configuration Management for GenAI Assistant

This module provides centralized configuration management for all components
of the GenAI assistant, including environment variables, defaults, and validation.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM components."""
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_max_tokens: int = 1000
    openai_temperature: float = 0.7
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"


@dataclass
class ImageConfig:
    """Configuration for image generation."""
    model_name: str = "flux-dev"
    device: str = "cuda"
    gpu_index: Optional[int] = None
    default_width: int = 512
    default_height: int = 512
    default_steps: int = 20
    default_guidance_scale: float = 7.5


@dataclass
class TranscriptionConfig:
    """Configuration for transcription."""
    model_name: str = "base"
    model_type: str = "faster-whisper"
    device: str = "cuda"
    gpu_index: Optional[int] = 0
    language: Optional[str] = None


@dataclass
class WebConfig:
    """Configuration for web interface."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    max_file_size: int = 16 * 1024 * 1024  # 16MB


class Config:
    """Main configuration class for GenAI Assistant."""
    
    def __init__(self):
        self.llm = LLMConfig()
        self.image = ImageConfig()
        self.transcription = TranscriptionConfig()
        self.web = WebConfig()
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        
        # LLM Configuration
        self.llm.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.llm.openai_model = os.getenv('OPENAI_MODEL', self.llm.openai_model)
        self.llm.openai_max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', self.llm.openai_max_tokens))
        self.llm.openai_temperature = float(os.getenv('OPENAI_TEMPERATURE', self.llm.openai_temperature))
        
        # Ollama base URL: prefer OLLAMA_BASE_URL, otherwise construct from OLLAMA_PORT
        ollama_base_url = os.getenv('OLLAMA_BASE_URL')
        if not ollama_base_url:
            ollama_port = os.getenv('OLLAMA_PORT', '11434')
            ollama_base_url = f"http://localhost:{ollama_port}"
        self.llm.ollama_base_url = ollama_base_url
        self.llm.ollama_model = os.getenv('OLLAMA_MODEL', self.llm.ollama_model)
        
        # Image Generation Configuration
        self.image.model_name = os.getenv('IMAGE_GENERATION_MODEL', self.image.model_name)
        self.image.device = os.getenv('IMAGE_GENERATION_DEVICE', self.image.device)
        gpu_index_str = os.getenv('IMAGE_GENERATION_GPU_INDEX')
        self.image.gpu_index = int(gpu_index_str) if gpu_index_str else None
        self.image.default_width = int(os.getenv('IMAGE_GENERATION_WIDTH', self.image.default_width))
        self.image.default_height = int(os.getenv('IMAGE_GENERATION_HEIGHT', self.image.default_height))
        self.image.default_steps = int(os.getenv('IMAGE_GENERATION_STEPS', self.image.default_steps))
        self.image.default_guidance_scale = float(os.getenv('IMAGE_GENERATION_GUIDANCE_SCALE', self.image.default_guidance_scale))
        
        # Transcription Configuration
        self.transcription.model_name = os.getenv('TRANSCRIPTION_MODEL', self.transcription.model_name)
        self.transcription.model_type = os.getenv('TRANSCRIPTION_MODEL_TYPE', self.transcription.model_type)
        self.transcription.device = os.getenv('TRANSCRIPTION_DEVICE', self.transcription.device)
        gpu_index_str = os.getenv('TRANSCRIPTION_GPU_INDEX')
        self.transcription.gpu_index = int(gpu_index_str) if gpu_index_str else None
        self.transcription.language = os.getenv('TRANSCRIPTION_LANGUAGE')
        
        # Web Configuration
        # Support both WEB_PORT and PORT for backward compatibility
        web_port = os.getenv('WEB_PORT') or os.getenv('PORT')
        if web_port:
            self.web.port = int(web_port)
        self.web.host = os.getenv('WEB_HOST', self.web.host)
        self.web.debug = os.getenv('WEB_DEBUG', 'false').lower() == 'true'
        self.web.max_file_size = int(os.getenv('WEB_MAX_FILE_SIZE', self.web.max_file_size))
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return status."""
        status = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'components': {}
        }
        
        # Validate LLM configuration
        llm_status = {'openai': False, 'ollama': False}
        
        if not self.llm.openai_api_key:
            status['warnings'].append("OpenAI API key not set - OpenAI LLM will not be available")
        else:
            llm_status['openai'] = True
        
        # Check if Ollama is accessible (basic check)
        try:
            import requests
            response = requests.get(f"{self.llm.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                llm_status['ollama'] = True
            else:
                status['warnings'].append("Ollama service not accessible - Ollama LLM will not be available")
        except Exception:
            status['warnings'].append("Ollama service not accessible - Ollama LLM will not be available")
        
        status['components']['llm'] = llm_status
        
        # Validate Image Generation configuration
        image_status = {'available': False}
        try:
            import torch
            if self.image.device == "cuda" and not torch.cuda.is_available():
                status['warnings'].append("CUDA not available - falling back to CPU for image generation")
                self.image.device = "cpu"
            image_status['available'] = True
        except ImportError:
            status['errors'].append("PyTorch not installed - image generation not available")
        
        status['components']['image'] = image_status
        
        # Validate Transcription configuration
        transcription_status = {'available': False}
        try:
            import whisper
            transcription_status['available'] = True
        except ImportError:
            status['errors'].append("Whisper not installed - transcription not available")
        
        status['components']['transcription'] = transcription_status
        
        # Check for critical errors
        if status['errors']:
            status['valid'] = False
        
        return status
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration as dictionary."""
        return {
            'api_key': self.llm.openai_api_key,
            'model': self.llm.openai_model,
            'max_tokens': self.llm.openai_max_tokens,
            'temperature': self.llm.openai_temperature
        }
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama configuration as dictionary."""
        return {
            'base_url': self.llm.ollama_base_url,
            'model': self.llm.ollama_model,
            'max_tokens': self.llm.openai_max_tokens,
            'temperature': self.llm.openai_temperature
        }
    
    def get_image_config(self) -> Dict[str, Any]:
        """Get image generation configuration as dictionary."""
        return {
            'model_name': self.image.model_name,
            'device': self.image.device,
            'gpu_index': self.image.gpu_index,
            'default_width': self.image.default_width,
            'default_height': self.image.default_height,
            'default_steps': self.image.default_steps,
            'default_guidance_scale': self.image.default_guidance_scale
        }
    
    def get_transcription_config(self) -> Dict[str, Any]:
        """Get transcription configuration as dictionary."""
        return {
            'model_name': self.transcription.model_name,
            'model_type': self.transcription.model_type,
            'device': self.transcription.device,
            'gpu_index': self.transcription.gpu_index,
            'language': self.transcription.language
        }
    
    def get_web_config(self) -> Dict[str, Any]:
        """Get web interface configuration as dictionary."""
        return {
            'host': self.web.host,
            'port': self.web.port,
            'debug': self.web.debug,
            'max_file_size': self.web.max_file_size
        }
    
    def print_status(self):
        """Print configuration status."""
        status = self.validate()
        
        print("🤖 GenAI Assistant Configuration Status")
        print("=" * 50)
        
        if status['valid']:
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration has errors:")
            for error in status['errors']:
                print(f"  - {error}")
        
        if status['warnings']:
            print("\n⚠️  Warnings:")
            for warning in status['warnings']:
                print(f"  - {warning}")
        
        print("\n📋 Component Status:")
        for component, comp_status in status['components'].items():
            if component == 'llm':
                print(f"  LLM:")
                print(f"    OpenAI: {'✅' if comp_status['openai'] else '❌'}")
                print(f"    Ollama: {'✅' if comp_status['ollama'] else '❌'}")
            else:
                status_icon = '✅' if comp_status['available'] else '❌'
                print(f"  {component.title()}: {status_icon}")
        
        print("\n🔧 Current Configuration:")
        print(f"  OpenAI Model: {self.llm.openai_model}")
        print(f"  Ollama URL: {self.llm.ollama_base_url}")
        print(f"  Image Device: {self.image.device}" + (f" (GPU {self.image.gpu_index})" if self.image.gpu_index is not None else ""))
        print(f"  Transcription Model: {self.transcription.model_name}")
        print(f"  Transcription Device: {self.transcription.device}" + (f" (GPU {self.transcription.gpu_index})" if self.transcription.gpu_index is not None else ""))
        print(f"  Web Interface: {self.web.host}:{self.web.port}")


def create_env_template():
    """Create a template .env file."""
    template = """# GenAI Assistant Environment Configuration

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.7

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Image Generation Configuration
IMAGE_GENERATION_MODEL=black-forest-flux
IMAGE_GENERATION_DEVICE=cuda
IMAGE_GENERATION_GPU_INDEX=3
IMAGE_GENERATION_WIDTH=512
IMAGE_GENERATION_HEIGHT=512
IMAGE_GENERATION_STEPS=20
IMAGE_GENERATION_GUIDANCE_SCALE=7.5

# Transcription Configuration
TRANSCRIPTION_MODEL=base
TRANSCRIPTION_MODEL_TYPE=whisper
TRANSCRIPTION_DEVICE=cuda
TRANSCRIPTION_GPU_INDEX=3
TRANSCRIPTION_LANGUAGE=en

# Web Interface Configuration
WEB_HOST=0.0.0.0
WEB_PORT=5000
WEB_DEBUG=false
WEB_MAX_FILE_SIZE=16777216

# Port Configuration
# All port numbers should be configured here - no hardcoded ports in the codebase
WEBAPP_PORT=8080
OLLAMA_PORT=11434
NGINX_HTTP_PORT=80
NGINX_HTTPS_PORT=443
"""
    
    env_path = Path('.env')
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(template)
        print("✅ Created .env template file")
        print("📝 Please edit .env with your actual configuration values")
    else:
        print("⚠️  .env file already exists")


if __name__ == "__main__":
    # Create configuration instance
    config = Config()
    
    # Print status
    config.print_status()
    
    # Create .env template if it doesn't exist
    create_env_template() 