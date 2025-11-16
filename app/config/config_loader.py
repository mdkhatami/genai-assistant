"""
Configuration Loader for GenAI Assistant

This module provides a centralized configuration system that reads from:
- .env file: Sensitive tokens and API keys
- config.yaml: All other configuration settings

This makes debugging easier by separating sensitive data from configuration.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

class ConfigLoader:
    """Centralized configuration loader for GenAI Assistant."""
    
    def __init__(self, config_path: str = "config.yaml", env_path: str = ".env"):
        """Initialize the configuration loader."""
        self.config_path = Path(config_path)
        self.env_path = Path(env_path)
        self._config = None
        self._env_vars = None
        
        # Load environment variables
        self._load_env_vars()
        
        # Load YAML configuration
        self._load_yaml_config()
    
    def _load_env_vars(self):
        """Load environment variables from .env file."""
        if self.env_path.exists():
            load_dotenv(self.env_path)
        
        self._env_vars = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'HUGGINGFACE_TOKEN': os.getenv('HUGGINGFACE_TOKEN'),
            'OLLAMA_GPU_INDEX': os.getenv('OLLAMA_GPU_INDEX'),
            'TRANSCRIPTION_GPU_INDEX': os.getenv('TRANSCRIPTION_GPU_INDEX'),
            'IMAGE_GENERATION_GPU_INDEX': os.getenv('IMAGE_GENERATION_GPU_INDEX'),
            'CUDA_VISIBLE_DEVICES': os.getenv('CUDA_VISIBLE_DEVICES'),
            # Port configuration
            'WEB_PORT': os.getenv('WEB_PORT') or os.getenv('PORT'),
            'WEBAPP_PORT': os.getenv('WEBAPP_PORT'),
            'OLLAMA_PORT': os.getenv('OLLAMA_PORT'),
            'OLLAMA_BASE_URL': os.getenv('OLLAMA_BASE_URL'),
            'NGINX_HTTP_PORT': os.getenv('NGINX_HTTP_PORT'),
            'NGINX_HTTPS_PORT': os.getenv('NGINX_HTTPS_PORT'),
        }
    
    def _load_yaml_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            self._config = yaml.safe_load(file)
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration."""
        config = self._config.get('openai', {}).copy()
        config['api_key'] = self._env_vars.get('OPENAI_API_KEY')
        return config
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama configuration with environment variable overrides."""
        config = self._config.get('ollama', {}).copy()
        
        # Override base_url from environment variable
        # Prefer OLLAMA_BASE_URL, otherwise construct from OLLAMA_PORT
        ollama_base_url = self._env_vars.get('OLLAMA_BASE_URL')
        if not ollama_base_url:
            ollama_port = self._env_vars.get('OLLAMA_PORT') or '11434'
            ollama_base_url = f"http://localhost:{ollama_port}"
        if ollama_base_url:
            config['base_url'] = ollama_base_url
        
        # Override GPU index from environment variable
        if self._env_vars.get('OLLAMA_GPU_INDEX'):
            try:
                config['gpu_index'] = int(self._env_vars['OLLAMA_GPU_INDEX'])
            except ValueError:
                pass
        
        return config
    
    def get_image_generation_config(self) -> Dict[str, Any]:
        """Get image generation configuration with environment variable overrides."""
        config = self._config.get('image_generation', {}).copy()
        
        # Override GPU index from environment variable
        if self._env_vars.get('IMAGE_GENERATION_GPU_INDEX'):
            try:
                config['gpu_index'] = int(self._env_vars['IMAGE_GENERATION_GPU_INDEX'])
            except ValueError:
                pass
        
        return config
    
    def get_transcription_config(self) -> Dict[str, Any]:
        """Get transcription configuration with environment variable overrides."""
        config = self._config.get('transcription', {}).copy()
        
        # Override GPU index from environment variable
        if self._env_vars.get('TRANSCRIPTION_GPU_INDEX'):
            try:
                config['gpu_index'] = int(self._env_vars['TRANSCRIPTION_GPU_INDEX'])
            except ValueError:
                pass
        
        return config
    
    def get_web_config(self) -> Dict[str, Any]:
        """Get web interface configuration with environment variable overrides."""
        config = self._config.get('web', {}).copy()
        
        # Override port from environment variable (support both WEB_PORT and PORT)
        web_port = self._env_vars.get('WEB_PORT') or self._env_vars.get('PORT')
        if web_port:
            try:
                config['port'] = int(web_port)
            except ValueError:
                pass
        
        # Override host from environment variable
        web_host = os.getenv('WEB_HOST')
        if web_host:
            config['host'] = web_host
        
        return config
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration."""
        return self._config.get('server', {})
    
    def get_huggingface_token(self) -> Optional[str]:
        """Get Hugging Face token."""
        return self._env_vars.get('HUGGINGFACE_TOKEN')
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        return {
            'openai': self.get_openai_config(),
            'ollama': self.get_ollama_config(),
            'image_generation': self.get_image_generation_config(),
            'transcription': self.get_transcription_config(),
            'web': self.get_web_config(),
            'server': self.get_server_config(),
            'huggingface_token': self.get_huggingface_token(),
        }
    
    def validate_config(self) -> bool:
        """Validate that all required configuration is present."""
        required_env_vars = ['OPENAI_API_KEY']
        required_config_sections = ['openai', 'transcription', 'web']
        
        # Check environment variables
        for var in required_env_vars:
            if not self._env_vars.get(var):
                print(f"❌ Missing required environment variable: {var}")
                return False
        
        # Check configuration sections
        for section in required_config_sections:
            if section not in self._config:
                print(f"❌ Missing required configuration section: {section}")
                return False
        
        print("✅ Configuration validation passed")
        return True
    
    def print_config_summary(self):
        """Print a summary of the current configuration."""
        print("\n🔧 Configuration Summary:")
        print("=" * 50)
        
        # Environment variables (masked)
        print("🔐 Environment Variables:")
        for key, value in self._env_vars.items():
            if value:
                masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"  {key}: {masked_value}")
            else:
                print(f"  {key}: ❌ Not set")
        
        # Configuration sections
        print("\n⚙️  Configuration Sections:")
        for section, config in self._config.items():
            print(f"  {section}: ✅ Loaded ({len(config)} settings)")
        
        print("=" * 50)

# Global configuration instance
_config_loader = None

def get_config_loader() -> ConfigLoader:
    """Get the global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader

def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    return get_config_loader().get_all_config()

def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI configuration."""
    return get_config_loader().get_openai_config()

def get_transcription_config() -> Dict[str, Any]:
    """Get transcription configuration."""
    return get_config_loader().get_transcription_config()

def get_image_generation_config() -> Dict[str, Any]:
    """Get image generation configuration."""
    return get_config_loader().get_image_generation_config()

def get_web_config() -> Dict[str, Any]:
    """Get web configuration."""
    return get_config_loader().get_web_config()

if __name__ == "__main__":
    # Test the configuration loader
    loader = ConfigLoader()
    loader.validate_config()
    loader.print_config_summary()
