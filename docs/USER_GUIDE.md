# 🚀 GenAI Assistant - Complete User Guide

A comprehensive GenAI assistant with core functionalities for LLM responses, image generation, and transcription. Features GPU acceleration, multiple model support, and both CLI and web interfaces.

## 📋 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Web Interface](#web-interface)
- [Command Line Interface](#command-line-interface)
- [Configuration](#configuration)
- [API Usage](#api-usage)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Features

### Core Functionality
- **🤖 LLM Response Core**: OpenAI ChatGPT and Ollama integration with vision model support
- **🎨 Image Generation Core**: Multiple models including Black Forest Flux, Stable Diffusion, and SDXL
- **🎤 Transcription Core**: Whisper and Fast Whisper for local audio transcription
- **⚙️ GPU Acceleration**: Configurable GPU support with automatic fallback to CPU

### Interfaces
- **💻 Command Line Interface**: Comprehensive CLI with subcommands for all functionalities
- **🌐 Web Interface**: Modern, responsive FastAPI web application with real-time streaming
- **📱 Standalone Webapp**: Browser-based client for easy access
- **🔧 Programmatic API**: Direct Python API for integration into other projects

### Advanced Features
- **👁️ Vision Analysis**: OpenAI vision model support for image analysis
- **🔄 Model Management**: Local model caching and management
- **⚡ Streaming Responses**: Real-time streaming for LLM responses
- **🎯 Batch Processing**: Support for batch image generation and transcription
- **🔒 Error Handling**: Robust error handling with graceful fallbacks
- **🔐 Authentication**: JWT-based authentication for secure access

---

## 📦 Installation

### 1. Clone and Setup
```bash
git clone <repository-url>
cd genai_assitant
```

### 2. Create Virtual Environment
```bash
# Using the setup script (recommended)
./setup_venv.sh

# Or manually:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Copy environment template
cp .env_example .env

# Edit .env with your configuration
# Required: OPENAI_API_KEY for OpenAI features
# Optional: GPU configuration for acceleration
```

### 4. Get API Keys

#### OpenAI API Key:
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Go to "API Keys" section
4. Create a new API key
5. Copy the key and paste it in your `.env` file

#### Hugging Face Token (for Image Generation):
1. Go to [Hugging Face](https://huggingface.co/)
2. Sign up or log in
3. Go to "Settings" → "Access Tokens"
4. Create a new token
5. Copy the token and paste it in your `.env` file

---

## 🚀 Quick Start

### Start the Server
```bash
# Using the startup script (recommended)
./start_server.sh

# Or directly with Python
python main.py

# Or with uvicorn
# Port is configurable via WEB_PORT in .env (default: 5000)
uvicorn app.main:app --host 0.0.0.0 --port ${WEB_PORT:-5000} --reload
```

The server will be available at:
- **API**: http://localhost:${WEB_PORT:-5000} (configurable via WEB_PORT in .env)
- **Documentation**: http://localhost:${WEB_PORT:-5000}/docs
- **Default credentials**: admin / admin123

---

## 🌐 Web Interface

### Standalone Web Application

The GenAI Assistant includes a modern, flexible web client with enhanced connection management.

#### 🚀 Features
- **🔌 Flexible Connection Management**: Quick connect to common server configurations
- **🎨 Enhanced User Experience**: Password visibility toggle, real-time connection status
- **🔒 Security Features**: No hardcoded credentials, secure storage
- **📱 Responsive Design**: Works on desktop and mobile devices

#### 📋 Prerequisites
- A running GenAI Assistant FastAPI server
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Network access to the server

#### 🛠️ Installation & Usage

**Option 1: Direct File Access**
1. Download the `standalone_webapp` folder
2. Open `index.html` in your web browser
3. Configure your server connection

**Option 2: Local Web Server (Recommended)**
```bash
cd standalone_webapp
python3 -m http.server ${WEBAPP_PORT:-8080}
# Then open http://localhost:${WEBAPP_PORT:-8080} in your browser (configurable via WEBAPP_PORT in .env)
```

**Option 3: Production Deployment**
1. Upload the `standalone_webapp` folder to your web server
2. Configure your server connection

#### 🎯 Quick Test

**Test LLM Chat**
1. Go to **"LLM Chat"** tab
2. Select **"Ollama"** provider
3. Choose **"llama3.3:latest"** model
4. Enter: `"Hello, how are you?"`
5. Click **"Send"**

**Test Image Generation**
1. Go to **"Image Generation"** tab
2. Enter prompt: `"a beautiful sunset over mountains"`
3. Click **"Generate Image"**

**Test Transcription**
1. Go to **"Transcription"** tab
2. Upload an audio file
3. Click **"Transcribe"**

### Web Interface Features

#### 🔐 Authentication Interface
- **Connection Modal**: Server URL, credentials input
- **Connection Presets**: Local, Production, Docker, Custom
- **Security Features**: Password visibility toggle, credential validation
- **Status Indicators**: Real-time connection status

#### 🤖 LLM Chat Interface
**Tab**: Primary interface for text generation

**Features**:
- **Provider Selection**: OpenAI vs Ollama dropdown
- **Model Selection**: Dynamic model loading
- **Prompt Interface**: Multi-line text input
- **Response Display**: Formatted text output
- **Settings**: Temperature, max tokens, system messages

**Supported Models**:
- **OpenAI**: GPT-4, GPT-3.5-turbo, GPT-4-turbo
- **Ollama**: llama3.3:latest, qwen3:30b, llama3.2-vision:latest

#### 🎨 Image Generation Interface
**Tab**: AI image generation interface

**Features**:
- **Model Selection**: Flux model variants
- **Prompt Input**: Detailed image descriptions
- **Parameter Control**: Resolution, count, steps, guidance
- **Batch Generation**: 1-8 images per request
- **Result Display**: Grid layout with metadata

**Configuration Options**:
- **Resolution**: 512x512, 768x768, 1024x1024
- **Steps**: 1-100 (default: 20)
- **Guidance Scale**: 0.1-20 (default: 3.5)
- **Image Count**: 1-8 images

#### 🎤 Audio Transcription Interface
**Tab**: Audio/video transcription with dual engines

**Features**:
- **File Upload**: Drag-and-drop or file picker
- **Dual Engines**: Faster-Whisper and OpenAI Whisper
- **Language Support**: 100+ languages with auto-detection
- **Advanced Options**: Collapsible parameter controls
- **Result Display**: Formatted transcription with timestamps

**Engine Comparison**:
- **Faster-Whisper**: Local processing, GPU acceleration
- **OpenAI Whisper**: Cloud processing, higher accuracy

---

## 💻 Command Line Interface

### Basic Usage
```bash
# LLM responses
python -m app.cli.cli llm openai "What is artificial intelligence?"
python -m app.cli.cli llm ollama "Explain quantum computing"

# Image generation
python -m app.cli.cli image "A beautiful sunset over mountains" -o sunset.png

# Audio transcription
python -m app.cli.cli transcribe audio.wav -o transcript.txt

# Vision analysis
python -m app.cli.cli llm openai "Describe this image" --analyze-image photo.jpg
```

### CLI Commands

#### LLM Commands
```bash
# OpenAI LLM
python -m app.cli.cli llm openai "Your prompt here" --model gpt-4 --max-tokens 1000

# Ollama LLM
python -m app.cli.cli llm ollama "Your prompt here" --model llama2 --temperature 0.7
```

#### Image Generation
```bash
# Basic image generation
python -m app.cli.cli image "A beautiful landscape" -o output.png

# Advanced options
python -m app.cli.cli image "A futuristic city" -o city.png --steps 30 --guidance 7.5 --resolution 1024x1024
```

#### Transcription
```bash
# Basic transcription
python -m app.cli.cli transcribe audio.wav -o transcript.txt

# Advanced options
python -m app.cli.cli transcribe audio.wav -o transcript.txt --model large --language en --device cuda
```

---

## ⚙️ Configuration

### Environment Variables
```bash
# OpenAI Configuration (Required for LLM)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.7

# Ollama Configuration (Alternative LLM - Optional)
OLLAMA_BASE_URL=http://localhost:${OLLAMA_PORT:-11434}
# Or set OLLAMA_PORT separately (default: 11434)
OLLAMA_MODEL=llama2

# Hugging Face Configuration (Required for Image Generation)
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Image Generation Configuration
IMAGE_GENERATION_MODEL=flux-dev
IMAGE_GENERATION_DEVICE=cuda  # or cpu
IMAGE_GENERATION_GPU_INDEX=3
IMAGE_GENERATION_STEPS=20
IMAGE_GENERATION_GUIDANCE_SCALE=7.5

# Transcription Configuration
TRANSCRIPTION_MODEL=base
TRANSCRIPTION_DEVICE=cuda  # or cpu
TRANSCRIPTION_GPU_INDEX=2
TRANSCRIPTION_LANGUAGE=en

# Web Interface Configuration
WEB_HOST=0.0.0.0
WEB_PORT=5000  # Configurable - change in .env if needed
WEBAPP_PORT=8080  # Standalone webapp port (configurable)
OLLAMA_PORT=11434  # Ollama service port (configurable)
WEB_DEBUG=false
WEB_MAX_FILE_SIZE=16777216
```

### Multi-GPU Configuration
The system is configured to use multiple GPUs:
- **GPU 1**: Local LLM (Ollama)
- **GPU 2**: Transcription (Whisper)
- **GPU 3**: Image Generation (FLUX.1-dev)

---

## 🔧 API Usage

### Programmatic Usage
```python
from app.core import OpenAILLM, OllamaLLM, ImageGenerator, Transcriber

# LLM responses
llm = OpenAILLM()
response = llm.generate_response("Hello, how are you?")

# Image generation
generator = ImageGenerator()
result = generator.generate_image("A beautiful landscape")

# Transcription
transcriber = Transcriber()
text = transcriber.transcribe_audio("audio_file.wav")
```

### API Endpoints

#### Authentication
- `POST /auth/login` - Login and get JWT token

#### Health & Info
- `GET /health` - Server health check
- `GET /docs` - API documentation

#### LLM Services
- `POST /api/llm/openai` - OpenAI LLM requests
- `POST /api/llm/ollama` - Ollama LLM requests
- `GET /api/llm/ollama/models` - List Ollama models

#### Image Generation
- `POST /api/image/generate` - Generate images
- `GET /api/image/models` - List available models

#### Transcription
- `POST /api/transcribe` - Transcribe audio files
- `GET /api/transcribe/models` - List transcription models

#### File Access
- `GET /generated/{filename}` - Access generated files
- `GET /uploads/{filename}` - Access uploaded files

---

## 🧪 Testing

### Run All Tests
```bash
# Test LLM functionality
python scripts/test_llm_complete.py

# Test image generation
python scripts/test_all_image_models.py

# Test transcription
python scripts/final_transcription_test.py

# Test server integration
python scripts/test_llm_with_server.py

# Test multi-GPU configuration
python scripts/test_multi_gpu_config.py
```

### Test Results
After restructuring, all core functionality is working:
- ✅ **LLM**: OpenAI and Ollama working perfectly
- ✅ **Server**: FastAPI server starts and runs correctly
- ✅ **Authentication**: JWT authentication working
- ✅ **API**: All endpoints functional
- ⚠️ **GPU Issues**: Some GPU-related issues (cuDNN, memory) are pre-existing

---

## 🚨 Troubleshooting

### Common Issues

#### "Failed to fetch" Error
1. Check that your FastAPI server is running: `curl http://localhost:${WEB_PORT:-5000}/health` (uses WEB_PORT from .env)
2. Verify API keys are set: `env | grep OPENAI`
3. Check server logs for specific errors

#### GPU Memory Issues
1. Clear GPU memory: `python -c "import torch; torch.cuda.empty_cache()"`
2. Restart the system to free up GPU memory
3. Use CPU fallback if GPU is unavailable

#### cuDNN Errors
1. This is a known compatibility issue
2. Use CPU transcription as a workaround
3. Consider updating cuDNN libraries

#### Authentication Issues
1. Check credentials match your `.env` file (ADMIN_USERNAME and ADMIN_PASSWORD)
2. Verify JWT token is valid
3. Reconnect to the server

### Server Startup Issues
1. Ensure virtual environment is activated
2. Check all dependencies are installed
3. Verify configuration files exist
4. Check port ${WEB_PORT:-5000} is available (configure WEB_PORT in .env if needed)

### Connection Issues
1. **CORS Errors**: Server configured with `allow_origins=["*"]`
2. **Authentication Failures**: Verify credentials in `.env` file
3. **Network Timeouts**: Check server status with `/health` endpoint

### Model Loading Issues
1. **Ollama Models Not Found**: Check Ollama server at `http://localhost:${OLLAMA_PORT:-11434}` (configure OLLAMA_PORT in .env)
2. **Image Generation Failures**: Verify GPU 3 is available
3. **Transcription Errors**: Check GPU 2 availability

### Performance Issues
1. **Slow Responses**: Check GPU memory usage
2. **Memory Errors**: Restart server to clear GPU memory
3. **Timeout Errors**: Increase timeout in config.js

---

## 🔄 Alternative: Use Ollama (Local LLM)

If you don't want to use OpenAI, you can use Ollama for local LLM:

1. Install Ollama: https://ollama.ai/
2. Run: `ollama pull llama2`
3. Make sure Ollama is running: `ollama serve`
4. In your standalone webapp, select "Ollama" as the LLM provider

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:${WEB_PORT:-5000}/docs (configurable via WEB_PORT in .env)
- **Developer Guide**: See `docs/DEVELOPER_GUIDE.md`
- **Deployment Guide**: See `docs/DEPLOYMENT_GUIDE.md`
- **Test Reports**: Available in `docs/` directory
- **Configuration Examples**: See `.env_example` in the project root

---

## 🤝 Contributing

1. Follow the project structure in `docs/DEVELOPER_GUIDE.md`
2. Run tests before submitting changes
3. Update documentation as needed
4. Follow FastAPI best practices

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Last Updated**: 2025-01-14  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
