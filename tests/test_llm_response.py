"""
Test suite for LLM Response Core functionality.

This module tests both OpenAI and Ollama LLM integrations with comprehensive
coverage of all features including error handling, parameter validation, and
response formatting.

Note: These are unit tests using mocks. For integration tests that use actual
API calls, see test_integration.py. To run only integration tests:
    python -m pytest tests/test_integration.py -v
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

# Add the parent directory to the path to import core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.llm_response import OpenAILLM, OllamaLLM, LLMResponse, BaseLLM


class TestLLMResponse(unittest.TestCase):
    """Test cases for LLMResponse data class."""
    
    def test_llm_response_creation(self):
        """Test LLMResponse object creation with all parameters."""
        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            tokens_used=100,
            response_time=1.5,
            error=None,
            metadata={"test": "data"}
        )
        
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.model, "gpt-4")
        self.assertEqual(response.tokens_used, 100)
        self.assertEqual(response.response_time, 1.5)
        self.assertIsNone(response.error)
        self.assertEqual(response.metadata["test"], "data")
    
    def test_llm_response_minimal(self):
        """Test LLMResponse object creation with minimal parameters."""
        response = LLMResponse(
            content="Test response",
            model="gpt-4"
        )
        
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.model, "gpt-4")
        self.assertIsNone(response.tokens_used)
        self.assertIsNone(response.response_time)
        self.assertIsNone(response.error)
        self.assertIsNone(response.metadata)


class TestOpenAILLM(unittest.TestCase):
    """Test cases for OpenAI LLM integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock environment variable
        self.env_patcher = patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up test environment."""
        self.env_patcher.stop()
    
    @patch('core.llm_response.OpenAI')
    def test_openai_llm_initialization(self, mock_openai):
        """Test OpenAI LLM initialization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        llm = OpenAILLM(
            model="gpt-4",
            max_tokens=500,
            temperature=0.8
        )
        
        self.assertEqual(llm.model, "gpt-4")
        self.assertEqual(llm.max_tokens, 500)
        self.assertEqual(llm.temperature, 0.8)
        mock_openai.assert_called_once_with(api_key='test-key')
    
    def test_openai_llm_no_api_key(self):
        """Test OpenAI LLM initialization without API key."""
        # Remove API key from environment
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                OpenAILLM()
    
    @patch('core.llm_response.OpenAI')
    def test_generate_response_success(self, mock_openai):
        """Test successful response generation."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_choice = Mock()
        mock_choice.message.content = "Test response"
        mock_choice.finish_reason = "stop"
        mock_choice.message.function_call = None
        mock_choice.message.tool_calls = None
        
        mock_usage = Mock()
        mock_usage.total_tokens = 150
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4"
        mock_response.usage = mock_usage
        
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = OpenAILLM()
        response = llm.generate_response("Test prompt")
        
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.model, "gpt-4")
        self.assertEqual(response.tokens_used, 150)
        self.assertIsNone(response.error)
        self.assertIsNotNone(response.response_time)
    
    @patch('core.llm_response.OpenAI')
    def test_generate_response_with_system_message(self, mock_openai):
        """Test response generation with system message."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_choice = Mock()
        mock_choice.message.content = "System response"
        mock_choice.finish_reason = "stop"
        mock_choice.message.function_call = None
        mock_choice.message.tool_calls = None
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4"
        mock_response.usage = None
        
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = OpenAILLM()
        response = llm.generate_response(
            "Test prompt",
            system_message="You are a helpful assistant."
        )
        
        # Verify that the system message was included in the API call
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[0]['content'], 'You are a helpful assistant.')
        self.assertEqual(messages[1]['role'], 'user')
        self.assertEqual(messages[1]['content'], 'Test prompt')
    
    @patch('core.llm_response.OpenAI')
    def test_generate_response_error_handling(self, mock_openai):
        """Test error handling in response generation."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Simulate API error
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        llm = OpenAILLM()
        response = llm.generate_response("Test prompt")
        
        self.assertEqual(response.content, "")
        self.assertIsNotNone(response.error)
        self.assertIn("API Error", response.error)
    
    @patch('core.llm_response.OpenAI')
    def test_streaming_response(self, mock_openai):
        """Test streaming response generation."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock streaming response
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        
        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta.content = " World"
        
        mock_chunk3 = Mock()
        mock_chunk3.choices = [Mock()]
        mock_chunk3.choices[0].delta.content = None
        
        mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]
        
        llm = OpenAILLM()
        responses = list(llm.generate_streaming_response("Test prompt"))
        
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0], "Hello")
        self.assertEqual(responses[1], " World")


class TestOllamaLLM(unittest.TestCase):
    """Test cases for Ollama LLM integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Get Ollama base URL from environment or use default
        import os
        from dotenv import load_dotenv
        load_dotenv()
        ollama_base_url = os.getenv('OLLAMA_BASE_URL')
        if not ollama_base_url:
            ollama_port = os.getenv('OLLAMA_PORT', '11434')
            ollama_base_url = f"http://localhost:{ollama_port}"
        self.base_url = ollama_base_url
    
    @patch('requests.get')
    def test_ollama_llm_initialization_success(self, mock_get):
        """Test successful Ollama LLM initialization."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        llm = OllamaLLM(
            base_url=self.base_url,
            model="llama2",
            max_tokens=500,
            temperature=0.8
        )
        
        self.assertEqual(llm.base_url, self.base_url)
        self.assertEqual(llm.model, "llama2")
        self.assertEqual(llm.max_tokens, 500)
        self.assertEqual(llm.temperature, 0.8)
    
    @patch('requests.get')
    def test_ollama_llm_initialization_failure(self, mock_get):
        """Test Ollama LLM initialization with connection failure."""
        mock_get.side_effect = Exception("Connection failed")
        
        # Should not raise exception, just log warning
        llm = OllamaLLM(base_url=self.base_url)
        self.assertEqual(llm.base_url, self.base_url)
    
    @patch('requests.post')
    def test_generate_response_success(self, mock_post):
        """Test successful Ollama response generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Test response',
            'model': 'llama2',
            'eval_count': 100,
            'done': True,
            'context': []
        }
        mock_post.return_value = mock_response
        
        llm = OllamaLLM(base_url=self.base_url)
        response = llm.generate_response("Test prompt")
        
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.model, "llama2")
        self.assertEqual(response.tokens_used, 100)
        self.assertIsNone(response.error)
        self.assertIsNotNone(response.response_time)
        
        # Verify API call parameters
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], f"{self.base_url}/api/generate")
        
        payload = call_args[1]['json']
        self.assertEqual(payload['model'], 'llama2')
        self.assertEqual(payload['prompt'], 'Test prompt')
        self.assertFalse(payload['stream'])
    
    @patch('requests.post')
    def test_generate_response_with_system_message(self, mock_post):
        """Test Ollama response generation with system message."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'System response',
            'model': 'llama2',
            'eval_count': 50,
            'done': True,
            'context': []
        }
        mock_post.return_value = mock_response
        
        llm = OllamaLLM(base_url=self.base_url)
        response = llm.generate_response(
            "Test prompt",
            system_message="You are a helpful assistant."
        )
        
        # Verify system message was included
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        self.assertEqual(payload['system'], 'You are a helpful assistant.')
    
    @patch('requests.post')
    def test_generate_response_error_handling(self, mock_post):
        """Test error handling in Ollama response generation."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        llm = OllamaLLM(base_url=self.base_url)
        response = llm.generate_response("Test prompt")
        
        self.assertEqual(response.content, "")
        self.assertIsNotNone(response.error)
        self.assertIn("500", response.error)
    
    @patch('requests.post')
    def test_streaming_response(self, mock_post):
        """Test Ollama streaming response generation."""
        # Mock streaming response
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            b'data: {"response": "Hello", "done": false}',
            b'data: {"response": " World", "done": false}',
            b'data: {"response": "", "done": true}',
            b'data: [DONE]'
        ]
        mock_post.return_value = mock_response
        
        llm = OllamaLLM(base_url=self.base_url)
        responses = list(llm.generate_streaming_response("Test prompt"))
        
        self.assertEqual(len(responses), 3)
        self.assertEqual(responses[0], "Hello")
        self.assertEqual(responses[1], " World")
        self.assertEqual(responses[2], "")
    
    @patch('requests.get')
    def test_list_models(self, mock_get):
        """Test listing available models."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [
                {'name': 'llama2'},
                {'name': 'gpt4all'},
                {'name': 'mistral'}
            ]
        }
        mock_get.return_value = mock_response
        
        llm = OllamaLLM(base_url=self.base_url)
        models = llm.list_models()
        
        self.assertEqual(len(models), 3)
        self.assertIn('llama2', models)
        self.assertIn('gpt4all', models)
        self.assertIn('mistral', models)
    
    @patch('requests.post')
    def test_pull_model(self, mock_post):
        """Test pulling a model."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        llm = OllamaLLM(base_url=self.base_url)
        success = llm.pull_model("llama2")
        
        self.assertTrue(success)
        
        # Verify API call
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], f"{self.base_url}/api/pull")
        self.assertEqual(call_args[1]['json'], {"name": "llama2"})


class TestBaseLLM(unittest.TestCase):
    """Test cases for BaseLLM abstract class."""
    
    def test_base_llm_initialization(self):
        """Test BaseLLM initialization with default parameters."""
        # Create a concrete implementation for testing
        class TestLLM(BaseLLM):
            def generate_response(self, prompt: str, **kwargs):
                return LLMResponse(content="test", model="test")
        
        llm = TestLLM()
        
        self.assertEqual(llm.model, "default")
        self.assertEqual(llm.max_tokens, 1000)
        self.assertEqual(llm.temperature, 0.7)
        self.assertEqual(llm.top_p, 1.0)
        self.assertEqual(llm.frequency_penalty, 0.0)
        self.assertEqual(llm.presence_penalty, 0.0)
        self.assertIsNone(llm.stop_sequences)
        self.assertTrue(llm.log_requests)
    
    def test_base_llm_custom_parameters(self):
        """Test BaseLLM initialization with custom parameters."""
        class TestLLM(BaseLLM):
            def generate_response(self, prompt: str, **kwargs):
                return LLMResponse(content="test", model="test")
        
        llm = TestLLM(
            model="custom-model",
            max_tokens=500,
            temperature=0.9,
            top_p=0.8,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            stop_sequences=["END", "STOP"],
            log_requests=False
        )
        
        self.assertEqual(llm.model, "custom-model")
        self.assertEqual(llm.max_tokens, 500)
        self.assertEqual(llm.temperature, 0.9)
        self.assertEqual(llm.top_p, 0.8)
        self.assertEqual(llm.frequency_penalty, 0.5)
        self.assertEqual(llm.presence_penalty, 0.3)
        self.assertEqual(llm.stop_sequences, ["END", "STOP"])
        self.assertFalse(llm.log_requests)


if __name__ == '__main__':
    unittest.main() 