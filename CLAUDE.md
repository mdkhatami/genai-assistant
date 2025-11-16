# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GenAI Assistant is a comprehensive AI service hub providing LLM chat, image generation, and audio transcription capabilities. The system features a FastAPI backend with GPU-accelerated inference, JWT authentication, and both web and CLI interfaces.

## Key Architecture

### Device Management (GPU/CPU)

The application automatically detects and uses available hardware:

- **Auto-Detection**: Uses GPU when available, falls back to CPU
- **Single Device**: All services share a single GPU/CPU (simplified from multi-GPU)
- **Configuration**: Set `DEVICE_PREFERENCE=auto` (default), `gpu`, or `cpu` in `.env`

The `DeviceManager` utility ([app/utils/device_manager.py](app/utils/device_manager.py)) handles device detection and allocation. Services automatically select the best available device based on memory requirements.

### Core Service Architecture

```
FastAPI Server (app/main.py)
├── Authentication Layer (app/auth/)
│   └── JWT-based auth with admin credentials
├── Core Services (app/core/)
│   ├── llm_response.py - OpenAI & Ollama integration
│   ├── optimized_image_generation.py - FLUX.1-dev with singleton pattern
│   └── robust_transcription.py - Whisper & Faster-Whisper
├── Configuration (app/config/)
│   ├── config_loader.py - YAML + env var merging
│   └── logging_config.py - Structured logging
└── API Models (app/models/)
    └── Pydantic models for requests/responses
```

### Configuration System

The system uses a **two-layer configuration**:

1. **YAML base**: `config.yaml` provides defaults
2. **Environment override**: `.env` variables take precedence

Port configuration is centralized in `.env` with these variables:
- `WEB_PORT` - FastAPI server (default: 5000)
- `WEBAPP_PORT` - Standalone web UI (default: 8080)
- `OLLAMA_PORT` - Ollama service (default: 11434)

The `config_loader.py` module handles merging these layers.

### Singleton Patterns

**Critical**: Several core services use singleton patterns for GPU memory efficiency:

- `OptimizedImageGenerator` in [app/core/optimized_image_generation.py](app/core/optimized_image_generation.py) - Loads model once, persists in GPU
- `RobustTranscriptionManager` in [app/core/robust_transcription.py](app/core/robust_transcription.py) - Model caching

When modifying these, preserve the singleton pattern to avoid GPU OOM errors.

## Development Commands

### Setup

```bash
# Local development (recommended)
./setup_local.sh

# Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env_example .env
# Edit .env with your API keys
```

### Running

```bash
# Development server (auto-reload)
./start_development.sh

# Production server
./start_production.sh

# Manual run
source venv/bin/activate
python main.py
```

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_llm_response.py

# Run with verbose output
python -m pytest -v tests/

# Test specific service
python -m pytest tests/test_image_generation.py
python -m pytest tests/test_transcription.py
```

### Docker

```bash
# Setup and start
./setup_docker.sh

# Manual Docker operations
docker compose up -d
docker compose down
docker compose logs -f
docker compose restart
```

### Service Management (Production)

```bash
# Status check
sudo ./service_management.sh status

# Restart service
sudo ./service_management.sh restart

# View logs
sudo journalctl -u genai-assistant -f
```

## API Structure

### Authentication Flow

All endpoints except `/health` and `/auth/login` require JWT bearer token:

1. POST `/auth/login` with `{"username": "admin", "password": "..."}`
2. Receive `{"access_token": "...", "token_type": "bearer"}`
3. Include `Authorization: Bearer <token>` in subsequent requests

Default credentials are in `.env` (`ADMIN_USERNAME`, `ADMIN_PASSWORD`).

### Main Endpoints

- `POST /api/llm/openai` - OpenAI ChatGPT response
- `POST /api/llm/ollama` - Ollama local LLM response
- `GET /api/llm/ollama/models` - List available Ollama models
- `POST /api/image/generate` - Generate image(s) with FLUX
- `GET /api/image/models` - List image generation models
- `POST /api/transcribe` - Transcribe audio file
- `GET /api/transcribe/models` - Get transcription model info
- `GET /health` - Health check (no auth required)

### Key Request/Response Models

All models are defined in [app/models/models.py](app/models/models.py):

- `LLMRequest`, `LLMResponse` - LLM interactions
- `ImageGenerationRequest`, `ImageGenerationResponse` - Image generation
- `TranscriptionRequest`, `TranscriptionResponse` - Audio transcription

## Common Development Tasks

### Adding a New LLM Provider

1. Create new class inheriting from `BaseLLM` in [app/core/llm_response.py](app/core/llm_response.py)
2. Implement `generate_response()` method returning `LLMResponse`
3. Add getter function in [app/main.py](app/main.py) (e.g., `get_xyz_llm()`)
4. Create endpoint using `handle_llm_request()` pattern
5. Add configuration to [config.yaml](config.yaml)
6. Update health check in [app/main.py](app/main.py):234-322

### Modifying Device Configuration

Device selection is managed centrally by `DeviceManager`:

- **Auto Mode (Default)**: `DeviceManager` selects best device automatically
- **Manual Override**: Can specify `device="cpu"` or `device="cuda"` in config
- **Per-Service Config**: Each service can specify minimum memory requirements

Configuration is in [.env_example](.env_example) and [config.yaml](config.yaml). The `DeviceManager` is initialized at startup in [app/main.py](app/main.py).

### Adding New API Endpoints

1. Define request/response models in [app/models/models.py](app/models/models.py)
2. Add endpoint function in [app/main.py](app/main.py)
3. Use `current_user: User = Depends(get_current_user)` for auth
4. Log errors with `logger.log_error(e, request, current_user.username)`
5. Test with `/docs` interactive API documentation

### Working with Configuration

Config loading hierarchy (highest priority first):

1. Environment variables from `.env`
2. YAML values in `config.yaml`

To add new config:
1. Add to [config.yaml](config.yaml) with sensible default
2. Add to [.env_example](.env_example) with documentation
3. Update `get_*_config()` methods in [app/config/config_loader.py](app/config/config_loader.py)

## Important Patterns

### Error Handling

All core services return result objects with `.error` field (not exceptions):

```python
result = generator.generate_image(prompt)
if result.error:
    # Handle error case
```

API endpoints convert errors to `HTTPException` with appropriate status codes.

### Logging

Use the centralized logger from [app/config/logging_config.py](app/config/logging_config.py):

```python
from app.config import get_api_logger
logger = get_api_logger()
logger.log_error(exception, request, username)
logger.log_system_event("event_name", {"key": "value"})
```

### GPU Memory Management

- Use `torch.cuda.empty_cache()` between operations in development
- Singleton pattern prevents multiple model loads
- `cleanup_generator()` in [app/core/optimized_image_generation.py](app/core/optimized_image_generation.py) handles shutdown

### File Uploads

File uploads use FastAPI's `UploadFile`:

```python
from fastapi import UploadFile, File

@app.post("/endpoint")
async def upload(file: UploadFile = File(...)):
    # Validate with app.utils.validation
    # Save to uploads/ directory
    # Process and cleanup
```

See [app/main.py](app/main.py):541-655 for transcription endpoint example.

## Deployment Considerations

### Environment Variables Validation

[app/main.py](app/main.py):737-784 validates critical env vars on startup:

- `JWT_SECRET_KEY` - Must be set (use `openssl rand -hex 32`)
- `OPENAI_API_KEY` - Warning if missing
- `HUGGINGFACE_TOKEN` - Warning if missing

### CORS Configuration

CORS origins are configurable via `CORS_ORIGINS` env var. Default allows localhost ports for development.

### Cleanup on Shutdown

The `@app.on_event("shutdown")` handler calls `cleanup_generator()` to release GPU memory properly.

## Standalone Web Application

The `standalone_webapp/` directory contains a vanilla JavaScript web interface:

- **No build step** - pure HTML/CSS/JS
- Served via Python's `http.server` on `WEBAPP_PORT`
- Connects to FastAPI backend on `WEB_PORT`
- All files in: [standalone_webapp/](standalone_webapp/)

Configuration in [standalone_webapp/config.js](standalone_webapp/config.js) reads `WEB_PORT` from environment.

## Testing Notes

- Tests use mocking for external dependencies (OpenAI, Ollama)
- GPU tests may require actual GPU hardware
- Set test environment variables in test files
- See [tests/test_llm_response.py](tests/test_llm_response.py) for mocking patterns
