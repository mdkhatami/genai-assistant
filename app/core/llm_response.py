"""
LLM Response Core Module

This module provides comprehensive LLM functionality with OpenAI ChatGPT and Ollama integration.
All functions include robust error handling, comprehensive parameters with sensible defaults,
and production-ready logging and monitoring capabilities.
"""

import os
import logging
import time
import json
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import openai
from openai import OpenAI
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Data class for standardized LLM response format."""
    content: str
    model: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    def __init__(self, **kwargs):
        self.model = kwargs.get('model', 'default')
        self.max_tokens = kwargs.get('max_tokens', 1000)
        self.temperature = kwargs.get('temperature', 0.7)
        self.top_p = kwargs.get('top_p', 1.0)
        self.frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        self.presence_penalty = kwargs.get('presence_penalty', 0.0)
        self.stop_sequences = kwargs.get('stop_sequences', None)
        self.log_requests = kwargs.get('log_requests', True)
        
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from the LLM."""
        pass
    
    def _log_request(self, prompt: str, response: LLMResponse, **kwargs):
        """Log request details for monitoring."""
        if self.log_requests:
            logger.info(f"LLM Request - Model: {response.model}, "
                       f"Tokens: {response.tokens_used}, "
                       f"Time: {response.response_time:.2f}s")
    
    def _handle_error(self, error: Exception, prompt: str) -> LLMResponse:
        """Handle errors and return standardized error response."""
        error_msg = f"Error generating response: {str(error)}"
        logger.error(error_msg)
        return LLMResponse(
            content="",
            model=self.model,
            error=error_msg,
            response_time=0.0
        )


class OpenAILLM(BaseLLM):
    """
    OpenAI ChatGPT integration with comprehensive functionality.
    
    Features:
    - Multiple model support (GPT-4, GPT-3.5-turbo, etc.)
    - Comprehensive parameter control
    - Streaming support
    - Function calling
    - Robust error handling
    - Request logging and monitoring
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 max_tokens: int = 1000,
                 temperature: float = 0.7,
                 top_p: float = 1.0,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 stop_sequences: Optional[List[str]] = None,
                 log_requests: bool = True,
                 timeout: int = 60,
                 max_retries: int = 3,
                 **kwargs):
        """
        Initialize OpenAI LLM client.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            stop_sequences: Sequences to stop generation
            log_requests: Whether to log requests
        """
        # Call parent constructor with base parameters
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            log_requests=log_requests,
            **kwargs
        )
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        # Additional OpenAI-specific parameters
        self.functions = kwargs.get('functions', None)
        self.function_call = kwargs.get('function_call', None)
        self.stream = kwargs.get('stream', False)
        self.user = kwargs.get('user', None)
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Vision model support
        self.vision_models = ["gpt-4-vision-preview", "gpt-4o", "gpt-4o-mini"]
        
        logger.info(f"OpenAI LLM initialized with model: {self.model}")
    
    def generate_response(self, 
                        prompt: str,
                        system_message: Optional[str] = None,
                        messages: Optional[List[Dict[str, str]]] = None,
                        images: Optional[List[str]] = None,
                        **kwargs) -> LLMResponse:
        """
        Generate response using OpenAI ChatGPT.
        
        Args:
            prompt: The input prompt
            system_message: System message to set context
            messages: List of message dictionaries
            **kwargs: Additional parameters to override defaults
            
        Returns:
            LLMResponse object with generated content and metadata
        """
        start_time = time.time()
        
        try:
            # Prepare messages
            if messages is None:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                
                # Handle vision models
                if images and self.model in self.vision_models:
                    content = [{"type": "text", "text": prompt}]
                    for image in images:
                        if image.startswith('http'):
                            content.append({"type": "image_url", "image_url": {"url": image}})
                        else:
                            # Assume base64 image
                            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "user", "content": prompt})
            
            # Prepare parameters
            params = {
                "model": kwargs.get('model', self.model),
                "messages": messages,
                "max_tokens": kwargs.get('max_tokens', self.max_tokens),
                "temperature": kwargs.get('temperature', self.temperature),
                "top_p": kwargs.get('top_p', self.top_p),
                "frequency_penalty": kwargs.get('frequency_penalty', self.frequency_penalty),
                "presence_penalty": kwargs.get('presence_penalty', self.presence_penalty),
                "stream": kwargs.get('stream', self.stream)
            }
            
            # Add optional parameters
            if kwargs.get('stop_sequences') or self.stop_sequences:
                params["stop"] = kwargs.get('stop_sequences', self.stop_sequences)
            
            if kwargs.get('functions') or self.functions:
                params["functions"] = kwargs.get('functions', self.functions)
            
            if kwargs.get('function_call') or self.function_call:
                params["function_call"] = kwargs.get('function_call', self.function_call)
            
            if kwargs.get('user') or self.user:
                params["user"] = kwargs.get('user', self.user)
            
            # Make API call
            response = self.client.chat.completions.create(**params)
            
            # Extract response content
            content = response.choices[0].message.content or ""
            
            # Create response object
            llm_response = LLMResponse(
                content=content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                response_time=time.time() - start_time,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "function_call": response.choices[0].message.function_call,
                    "tool_calls": response.choices[0].message.tool_calls
                }
            )
            
            self._log_request(prompt, llm_response, **kwargs)
            return llm_response
            
        except Exception as e:
            return self._handle_error(e, prompt)
    
    def analyze_image(self, 
                     image_path: str, 
                     prompt: str = "Describe this image in detail",
                     system_message: Optional[str] = None,
                     **kwargs) -> LLMResponse:
        """
        Analyze an image using vision models.
        
        Args:
            image_path: Path to the image file or image URL
            prompt: Prompt for image analysis
            system_message: System message
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with image analysis
        """
        if self.model not in self.vision_models:
            # Switch to a vision model
            kwargs['model'] = 'gpt-4o'
        
        return self.generate_response(
            prompt=prompt,
            system_message=system_message,
            images=[image_path],
            **kwargs
        )
    
    def generate_streaming_response(self, 
                                 prompt: str,
                                 system_message: Optional[str] = None,
                                 **kwargs):
        """
        Generate streaming response from OpenAI.
        
        Args:
            prompt: The input prompt
            system_message: System message to set context
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the response as they are generated
        """
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            params = {
                "model": kwargs.get('model', self.model),
                "messages": messages,
                "max_tokens": kwargs.get('max_tokens', self.max_tokens),
                "temperature": kwargs.get('temperature', self.temperature),
                "stream": True
            }
            
            stream = self.client.chat.completions.create(**params)
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield f"Error: {str(e)}"


class OllamaLLM(BaseLLM):
    """
    Ollama integration for local LLM inference.
    
    Features:
    - Local model inference
    - Multiple model support
    - Streaming support
    - Comprehensive parameter control
    - Robust error handling
    """
    
    def __init__(self,
                 base_url: Optional[str] = None,
                 model: str = "llama2",
                 max_tokens: int = 1000,
                 temperature: float = 0.7,
                 top_p: float = 1.0,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 stop_sequences: Optional[List[str]] = None,
                 log_requests: bool = True,
                 timeout: int = 60,
                 max_retries: int = 3,
                 **kwargs):
        """
        Initialize Ollama LLM client.
        
        Args:
            base_url: Ollama server URL
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop_sequences: Sequences to stop generation
            log_requests: Whether to log requests
        """
        # Call parent constructor with base parameters
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            log_requests=log_requests,
            **kwargs
        )
        
        # Set base_url: use provided, or construct from OLLAMA_PORT env var, or use default
        if base_url is None:
            ollama_base_url = os.getenv('OLLAMA_BASE_URL')
            if not ollama_base_url:
                ollama_port = os.getenv('OLLAMA_PORT', '11434')
                base_url = f"http://localhost:{ollama_port}"
            else:
                base_url = ollama_base_url
        
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Cache for model information
        self._model_cache = {}
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code != 200:
                raise ConnectionError(f"Could not connect to Ollama at {self.base_url}")
            logger.info(f"Ollama LLM initialized with model: {self.model}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {str(e)}")
    
    def generate_response(self, 
                        prompt: str,
                        system_message: Optional[str] = None,
                        **kwargs) -> LLMResponse:
        """
        Generate response using Ollama.
        
        Args:
            prompt: The input prompt
            system_message: System message to set context
            **kwargs: Additional parameters to override defaults
            
        Returns:
            LLMResponse object with generated content and metadata
        """
        start_time = time.time()
        
        try:
            # Prepare request payload
            payload = {
                "model": kwargs.get('model', self.model),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": kwargs.get('max_tokens', self.max_tokens),
                    "temperature": kwargs.get('temperature', self.temperature),
                    "top_p": kwargs.get('top_p', self.top_p),
                    "repeat_penalty": 1.0 + kwargs.get('frequency_penalty', self.frequency_penalty),
                    "stop": kwargs.get('stop_sequences', self.stop_sequences) or []
                }
            }
            
            if system_message:
                payload["system"] = system_message
            
            # Make API call
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            response_data = response.json()
            
            # Create response object
            llm_response = LLMResponse(
                content=response_data.get('response', ''),
                model=response_data.get('model', self.model),
                tokens_used=response_data.get('eval_count'),
                response_time=time.time() - start_time,
                metadata={
                    "done": response_data.get('done', True),
                    "context": response_data.get('context', [])
                }
            )
            
            self._log_request(prompt, llm_response, **kwargs)
            return llm_response
            
        except Exception as e:
            return self._handle_error(e, prompt)
    
    def generate_streaming_response(self, 
                                 prompt: str,
                                 system_message: Optional[str] = None,
                                 **kwargs):
        """
        Generate streaming response from Ollama.
        
        Args:
            prompt: The input prompt
            system_message: System message to set context
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the response as they are generated
        """
        try:
            payload = {
                "model": kwargs.get('model', self.model),
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": kwargs.get('max_tokens', self.max_tokens),
                    "temperature": kwargs.get('temperature', self.temperature),
                    "top_p": kwargs.get('top_p', self.top_p),
                    "repeat_penalty": 1.0 + kwargs.get('frequency_penalty', self.frequency_penalty),
                    "stop": kwargs.get('stop_sequences', self.stop_sequences) or []
                }
            }
            
            if system_message:
                payload["system"] = system_message
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=300
            )
            
            for line in response.iter_lines():
                if line:
                    data = line.decode('utf-8')
                    if data.startswith('data: '):
                        try:
                            chunk_data = data[6:]  # Remove 'data: ' prefix
                            if chunk_data.strip() == '[DONE]':
                                break
                            chunk = json.loads(chunk_data)
                            if 'response' in chunk:
                                yield chunk['response']
                        except Exception as e:
                            logger.warning(f"Error parsing streaming chunk: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield f"Error: {str(e)}"
    
    def list_models(self) -> Dict[str, Any]:
        """List available models on the Ollama server with detailed information."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                # Format model information
                model_list = []
                for model in models:
                    model_info = {
                        'name': model.get('name', 'Unknown'),
                        'size': model.get('size', 0),
                        'digest': model.get('digest', ''),
                        'modified_at': model.get('modified_at', ''),
                        'size_gb': round(model.get('size', 0) / (1024**3), 2) if model.get('size') else 0
                    }
                    model_list.append(model_info)
                
                return {
                    'success': True,
                    'models': model_list,
                    'total_models': len(model_list),
                    'server_url': self.base_url
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'models': [],
                    'total_models': 0,
                    'server_url': self.base_url
                }
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'models': [],
                'total_models': 0,
                'server_url': self.base_url
            }
    
    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a model to the Ollama server."""
        try:
            logger.info(f"Pulling model: {model_name}")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=600  # 10 minutes for model download
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'message': f"Model {model_name} pulled successfully",
                    'model_name': model_name
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'model_name': model_name
                }
                
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
    
    def check_model_exists(self, model_name: str) -> bool:
        """Check if a model exists on the Ollama server."""
        try:
            models_info = self.list_models()
            if models_info['success']:
                model_names = [model['name'] for model in models_info['models']]
                return model_name in model_names
            return False
        except Exception as e:
            logger.error(f"Error checking model existence: {str(e)}")
            return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get Ollama server information."""
        try:
            # Try to get server version/info
            response = requests.get(f"{self.base_url}/api/version", timeout=self.timeout)
            if response.status_code == 200:
                version_info = response.json()
            else:
                version_info = {"version": "unknown"}
            
            # Get model list for additional info
            models_info = self.list_models()
            
            return {
                'success': True,
                'server_url': self.base_url,
                'version': version_info.get('version', 'unknown'),
                'status': 'connected',
                'total_models': models_info.get('total_models', 0),
                'models_available': models_info.get('success', False)
            }
            
        except Exception as e:
            logger.error(f"Error getting server info: {str(e)}")
            return {
                'success': False,
                'server_url': self.base_url,
                'status': 'disconnected',
                'error': str(e)
            } 