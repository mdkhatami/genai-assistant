"""
FastAPI GenAI Assistant

Main FastAPI application providing REST API endpoints for LLM, image generation,
and transcription services with JWT authentication and comprehensive logging.
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, Request, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer
import uvicorn

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from app.auth import authenticate_user, create_access_token, get_current_user, User
from app.models import LoginRequest, LoginResponse
from app.models import (
    LLMRequest, LLMResponse, ImageGenerationRequest, ImageGenerationResponse,
    TranscriptionRequest, TranscriptionResponse, ModelListResponse, HealthResponse,
    FileUploadResponse
)
from app.config import get_api_logger, get_config_loader
from app.core import OpenAILLM, OllamaLLM
from app.core.optimized_image_generation import get_optimized_generator, cleanup_generator
from app.utils.device_manager import get_device_manager, initialize_device_manager

# Initialize FastAPI app
app = FastAPI(
    title="GenAI Assistant API",
    description="REST API for GenAI Assistant with LLM, image generation, and transcription services",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
# Load allowed origins from environment variable, default to localhost for development
# Use WEBAPP_PORT from env if CORS_ORIGINS not explicitly set
webapp_port = os.getenv('WEBAPP_PORT', '8080')
default_cors_origins = f"http://localhost:{webapp_port},http://localhost:3000,http://127.0.0.1:{webapp_port},http://127.0.0.1:3000"
cors_origins_str = os.getenv("CORS_ORIGINS", default_cors_origins)
cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config_loader = get_config_loader()
logger = get_api_logger()

# Global instances
llm_openai = None
llm_ollama = None
image_generator = None
device_manager = None  # Initialized on startup
# transcriber removed - using robust_transcription module directly

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("generated", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_openai_llm():
    """Get OpenAI LLM instance."""
    global llm_openai
    if llm_openai is None:
        try:
            llm_openai = OpenAILLM()
        except Exception as e:
            logger.log_system_event("openai_llm_init_error", {"error": str(e)})
            return None
    return llm_openai

def get_ollama_llm():
    """Get Ollama LLM instance."""
    global llm_ollama
    if llm_ollama is None:
        try:
            ollama_config = config_loader.get_ollama_config()
            # Get base_url from config (which already handles env vars), with fallback
            ollama_base_url = ollama_config.get('base_url')
            if not ollama_base_url:
                # Fallback: construct from OLLAMA_PORT env var
                ollama_port = os.getenv('OLLAMA_PORT', '11434')
                ollama_base_url = f"http://localhost:{ollama_port}"
            llm_ollama = OllamaLLM(
                base_url=ollama_base_url,
                model=ollama_config.get('model', 'llama2')
            )
            
            # Log GPU configuration if specified
            if 'gpu_index' in ollama_config:
                logger.log_system_event("ollama_gpu_config", {
                    "gpu_index": ollama_config['gpu_index'],
                    "device": ollama_config.get('device', 'cuda')
                })
                
        except Exception as e:
            logger.log_system_event("ollama_llm_init_error", {"error": str(e)})
            return None
    return llm_ollama

def get_image_generator(model_name=None):
    """Get optimized image generator instance (on-demand loading)."""
    try:
        device_mgr = get_device_manager()
        image_config = config_loader.get_image_generation_config()

        # Get best device for image generation from DeviceManager
        device_info = device_mgr.get_device_for_service(
            service_name="image_generation",
            config=image_config
        )

        # Use the optimized generator with proper model name mapping
        # Default to flux-schnell (minimal model) if not specified
        model_mapping = {
            "flux": "black-forest-labs/FLUX.1-dev",
            "flux-dev": "black-forest-labs/FLUX.1-dev",
            "flux-dev-8bit": "black-forest-labs/FLUX.1-dev",
            "flux-dev-4bit": "black-forest-labs/FLUX.1-dev",
            "flux-schnell": "black-forest-labs/FLUX.1-schnell"
        }

        # Get model name from request or config, default to flux-schnell (minimal)
        actual_model_name = model_name or image_config.get('model', 'flux-schnell')
        
        # Ensure minimal model is used if no specific model requested
        if not model_name and actual_model_name not in model_mapping:
            actual_model_name = 'flux-schnell'
            logger.info(f"Using minimal model (flux-schnell) as default")
        
        hf_model_name = model_mapping.get(actual_model_name, actual_model_name)

        # Check for force CPU offload environment variable
        force_cpu_offload = os.getenv('IMAGE_GENERATION_FORCE_CPU_OFFLOAD', 'false').lower() == 'true'
        enable_cpu_offload = True if force_cpu_offload else None  # None = auto-detect
        
        # For on-demand loading, CPU offload will be auto-detected unless forced
        # CPU offload is automatically enabled if memory is low or for schnell model
        if force_cpu_offload:
            logger.info("🔄 Force CPU offload enabled via IMAGE_GENERATION_FORCE_CPU_OFFLOAD")
        
        generator = get_optimized_generator(
            model_name=hf_model_name,
            device=device_info['device'],  # From DeviceManager
            gpu_index=device_info['device_index'],  # Auto-selected
            enable_cpu_offload=enable_cpu_offload,  # Auto-detect or forced
            auto_unload=True,  # Enable automatic unloading after use
            unload_timeout=0.0  # Unload immediately after generation
        )

        return generator

    except Exception as e:
        logger.log_system_event("optimized_image_generator_init_error", {"error": str(e)})
        return None

# Transcriber initialization removed - using robust_transcription module directly in endpoints

# Middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses."""
    start_time = time.time()
    
    # Get user from token if available
    user = None
    try:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            from app.auth import verify_token
            payload = verify_token(token)
            if payload:
                user = payload.get("sub")
    except Exception:
        pass
    
    # Log request
    logger.log_request(request, user)
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Log response (we'll need to capture response data)
    try:
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        
        # Reconstruct response
        response = Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
        
        # Try to parse response data for logging
        response_data = None
        try:
            if response.media_type == "application/json":
                response_data = response_body.decode()
        except Exception:
            response_data = f"<{len(response_body)} bytes>"
        
        logger.log_response(response, response_data, processing_time, user)
        
    except Exception as e:
        logger.log_error(e, request, user)
    
    return response

# Authentication endpoints
@app.post("/auth/login", response_model=LoginResponse)
async def login(login_data: LoginRequest):
    """Authenticate user and return JWT token."""
    user = authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        username=user.username
    )

# Health check endpoint (no authentication required)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and component status with actual connectivity tests."""
    components = {}
    
    # Check OpenAI LLM with actual connectivity test
    try:
        openai_llm = get_openai_llm()
        if openai_llm:
            # Test actual connectivity by checking API key
            openai_config = config_loader.get_openai_config()
            if openai_config.get('api_key') and openai_config['api_key'] != 'your_openai_api_key_here':
                components["openai_llm"] = "healthy"
            else:
                components["openai_llm"] = "unavailable (no API key)"
        else:
            components["openai_llm"] = "unavailable"
    except Exception as e:
        components["openai_llm"] = f"error: {str(e)[:50]}"
    
    # Check Ollama LLM with actual connectivity test
    try:
        ollama_llm = get_ollama_llm()
        if ollama_llm:
            # Test actual connectivity
            ollama_config = config_loader.get_ollama_config()
            import requests
            try:
                ollama_base_url = ollama_config.get('base_url')
                if not ollama_base_url:
                    ollama_port = os.getenv('OLLAMA_PORT', '11434')
                    ollama_base_url = f"http://localhost:{ollama_port}"
                response = requests.get(f"{ollama_base_url}/api/tags", timeout=2)
                if response.status_code == 200:
                    components["ollama_llm"] = "healthy"
                else:
                    components["ollama_llm"] = f"unavailable (status: {response.status_code})"
            except requests.RequestException:
                components["ollama_llm"] = "unavailable (connection failed)"
        else:
            components["ollama_llm"] = "unavailable"
    except Exception as e:
        components["ollama_llm"] = f"error: {str(e)[:50]}"
    
    # Check image generator with GPU availability
    try:
        img_gen = get_image_generator()
        if img_gen:
            # Check GPU availability
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    components["image_generator"] = f"healthy (GPU: {gpu_count} available)"
                else:
                    components["image_generator"] = "healthy (CPU mode)"
            except ImportError:
                components["image_generator"] = "healthy"
        else:
            components["image_generator"] = "unavailable"
    except Exception as e:
        components["image_generator"] = f"error: {str(e)[:50]}"
    
    # Check transcriber
    try:
        from app.core.robust_transcription import get_robust_transcription_manager
        trans = get_robust_transcription_manager()
        if trans:
            # Get model info to verify it's actually working
            info = trans.get_model_info()
            components["transcriber"] = f"healthy (models loaded: {info.get('whisper_models_loaded', 0) + info.get('faster_whisper_models_loaded', 0)})"
        else:
            components["transcriber"] = "unavailable"
    except Exception as e:
        components["transcriber"] = f"error: {str(e)[:50]}"
    
    # Determine overall status
    overall_status = "healthy"
    if any("error" in str(v).lower() for v in components.values()):
        overall_status = "degraded"
    elif all("unavailable" in str(v).lower() for v in components.values()):
        overall_status = "unavailable"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version="1.0.0",
        components=components
    )

# Helper function for LLM requests
async def handle_llm_request(request: LLMRequest, provider: str, current_user: User) -> LLMResponse:
    """Common handler for LLM requests (OpenAI or Ollama)."""
    start_time = time.time()
    
    try:
        if provider == 'openai':
            # Get OpenAI configuration
            openai_config = config_loader.get_openai_config()
            model_to_use = request.model or openai_config.get('model', 'gpt-4')
            
            # Create OpenAI LLM instance
            from app.core.llm_response import OpenAILLM
            llm = OpenAILLM(
                model=model_to_use,
                max_tokens=request.max_tokens or openai_config.get('max_tokens', 1000),
                temperature=request.temperature or openai_config.get('temperature', 0.7)
            )
        elif provider == 'ollama':
            # Get Ollama configuration
            ollama_config = config_loader.get_ollama_config()
            model_to_use = request.model or ollama_config.get('model', 'llama2')
            
            # Create Ollama LLM instance
            from app.core.llm_response import OllamaLLM
            ollama_base_url = ollama_config.get('base_url')
            if not ollama_base_url:
                ollama_port = os.getenv('OLLAMA_PORT', '11434')
                ollama_base_url = f"http://localhost:{ollama_port}"
            llm = OllamaLLM(
                base_url=ollama_base_url,
                model=model_to_use,
                max_tokens=request.max_tokens or ollama_config.get('max_tokens', 1000),
                temperature=request.temperature or ollama_config.get('temperature', 0.7)
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Generate response
        response = llm.generate_response(
            prompt=request.prompt,
            system_message=request.system_message,
            model=model_to_use,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        processing_time = time.time() - start_time
        
        return LLMResponse(
            response=response.content,
            model=response.model,
            tokens_used=response.tokens_used,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.log_error(e, None, current_user.username)
        raise HTTPException(status_code=500, detail=str(e))

# LLM endpoints
@app.post("/api/llm/openai", response_model=LLMResponse)
async def llm_openai_api(request: LLMRequest, current_user: User = Depends(get_current_user)):
    """Generate response using OpenAI LLM."""
    return await handle_llm_request(request, 'openai', current_user)

@app.post("/api/llm/ollama", response_model=LLMResponse)
async def llm_ollama_api(request: LLMRequest, current_user: User = Depends(get_current_user)):
    """Generate response using Ollama LLM."""
    return await handle_llm_request(request, 'ollama', current_user)

@app.get("/api/llm/ollama/models")
async def llm_ollama_models_api(current_user: User = Depends(get_current_user)):
    """Get available Ollama models with detailed information."""
    try:
        llm = get_ollama_llm()
        if not llm:
            raise HTTPException(status_code=503, detail="Ollama LLM not available")
        
        # Use the enhanced list_models method
        models_info = llm.list_models()
        
        # Also get server info
        server_info = llm.get_server_info()
        
        # Combine the information
        response = {
            "success": models_info.get('success', False),
            "models": models_info.get('models', []),
            "total_models": models_info.get('total_models', 0),
            "server_url": models_info.get('server_url', 'Unknown'),
            "server_status": server_info.get('status', 'unknown'),
            "server_version": server_info.get('version', 'unknown'),
            "error": models_info.get('error') or server_info.get('error')
        }
        
        return response
    
    except Exception as e:
        logger.log_error(e, None, current_user.username)
        raise HTTPException(status_code=500, detail=str(e))

# Image generation endpoints
@app.post("/api/image/generate", response_model=ImageGenerationResponse)
async def image_generate_api(
    request: ImageGenerationRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate image using specified model (on-demand loading)."""
    start_time = time.time()
    generator = None
    
    try:
        # Log the requested model for debugging
        requested_model = request.model or "default (from config)"
        logger.logger.info(f"🎨 Image generation request - Model: {requested_model}, Prompt: '{request.prompt[:50]}{'...' if len(request.prompt) > 50 else ''}'")
        
        # Get generator (model will load on-demand when generate_image is called)
        generator = get_image_generator(request.model)
        if not generator:
            raise HTTPException(status_code=503, detail="Image generator not available")
        
        # Log which model will actually be used
        model_info = generator.get_model_info()
        logger.logger.info(f"📋 Using model: {model_info.get('model_name', 'unknown')}, Device: {model_info.get('device', 'unknown')}")
        
        # Get default parameters from config
        image_config = config_loader.get_image_generation_config()
        
        # Validate and limit number of images
        num_images = min(max(request.num_images or 1, 1), 8)  # Limit between 1-8 images
        
        # Generate image (model loads on-demand here)
        result = generator.generate_image(
            prompt=request.prompt,
            width=request.width or image_config['width'],
            height=request.height or image_config['height'],
            num_inference_steps=request.steps or image_config['steps'],
            guidance_scale=request.guidance_scale or image_config['guidance_scale'],
            num_images=num_images,
            negative_prompt=request.negative_prompt
        )
        
        # Model will be automatically unloaded after generation (auto_unload=True)
        
        processing_time = time.time() - start_time
        
        # Handle both single and multiple images
        import uuid
        import base64
        from pathlib import Path
        from io import BytesIO
        
        # Create generated directory if it doesn't exist
        generated_dir = Path("generated")
        generated_dir.mkdir(exist_ok=True)
        
        # Process results (could be single result or list of results)
        results_list = result if isinstance(result, list) else [result]
        image_items = []
        
        for i, img_result in enumerate(results_list):
            if img_result.error:
                # Handle error case
                logger.logger.warning(f"Image generation error for image {i+1}: {img_result.error}")
                image_items.append({
                    "image_path": "",
                    "image_data": "",
                    "generation_time": img_result.generation_time,
                    "error": img_result.error
                })
            else:
                # Validate PIL Image object
                if img_result.image is None:
                    logger.logger.error(f"Image {i+1}: PIL Image object is None!")
                    image_items.append({
                        "image_path": "",
                        "image_data": "",
                        "generation_time": img_result.generation_time,
                        "error": "Generated image is None"
                    })
                    continue
                
                try:
                    # Generate unique filename
                    filename = f"{uuid.uuid4().hex}.png"
                    image_path = generated_dir / filename
                    
                    # Save the image to file
                    img_result.image.save(image_path, "PNG")
                    logger.logger.debug(f"Image {i+1}: Saved to {image_path}")
                    
                    # Convert to base64 for direct display in frontend
                    buffer = BytesIO()
                    img_result.image.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    image_data_url = f"data:image/png;base64,{img_base64}"
                    
                    # Debug logging
                    logger.logger.debug(f"Image {i+1}: Base64 length: {len(img_base64)} chars, Data URL length: {len(image_data_url)} chars")
                    logger.logger.debug(f"Image {i+1}: Data URL prefix: {image_data_url[:50]}...")
                    
                    if not img_base64 or len(img_base64) < 100:
                        logger.logger.error(f"Image {i+1}: Base64 data is too short or empty! Length: {len(img_base64)}")
                    
                    image_items.append({
                        "image_path": str(image_path),
                        "image_data": image_data_url,
                        "generation_time": img_result.generation_time
                    })
                except Exception as e:
                    logger.logger.error(f"Image {i+1}: Error processing image: {e}", exc_info=True)
                    image_items.append({
                        "image_path": "",
                        "image_data": "",
                        "generation_time": img_result.generation_time,
                        "error": str(e)
                    })
        
        # Debug logging for response
        logger.logger.debug(f"Image generation response: {len(image_items)} image(s) prepared")
        for i, item in enumerate(image_items):
            has_data = bool(item.get('image_data'))
            data_length = len(item.get('image_data', ''))
            logger.logger.debug(f"  Image {i+1}: has_data={has_data}, data_length={data_length}, error={item.get('error')}")
        
        response = ImageGenerationResponse(
            images=image_items,
            prompt=request.prompt,
            model=results_list[0].model,
            parameters={
                "width": request.width or image_config['width'],
                "height": request.height or image_config['height'],
                "steps": request.steps or image_config['steps'],
                "guidance_scale": request.guidance_scale or image_config['guidance_scale'],
                "num_images": num_images,
                "negative_prompt": request.negative_prompt
            },
            processing_time=processing_time
        )
        
        return response
    
    except Exception as e:
        logger.log_error(e, None, current_user.username)
        # Ensure model is unloaded even on error
        if generator:
            try:
                generator.unload_model()
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/image/models", response_model=ModelListResponse)
async def image_models_api(current_user: User = Depends(get_current_user)):
    """Get available image generation models."""
    try:
        generator = get_image_generator()
        if not generator:
            raise HTTPException(status_code=503, detail="Image generator not available")
        
        models = generator.get_available_models()
        model_list = [
            {"name": model, "type": "image_generation", "description": f"Image generation model: {model}"}
            for model in models
        ]
        
        return ModelListResponse(models=model_list, count=len(model_list))
    
    except Exception as e:
        logger.log_error(e, None, current_user.username)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/image/memory")
async def image_memory_info_api(current_user: User = Depends(get_current_user)):
    """Get GPU memory information for image generation diagnostics."""
    try:
        generator = get_image_generator()
        if not generator:
            raise HTTPException(status_code=503, detail="Image generator not available")
        
        # Get model info which includes memory info
        model_info = generator.get_model_info()
        
        # Get detailed memory info
        memory_info = {}
        if hasattr(generator, 'get_gpu_memory_info'):
            memory_info = generator.get_gpu_memory_info()
        
        # Get transcription manager info if available
        transcription_info = {}
        try:
            from app.core.robust_transcription import get_robust_transcription_manager
            trans_manager = get_robust_transcription_manager()
            if trans_manager:
                trans_info = trans_manager.get_model_info()
                transcription_info = {
                    "cached_models": {
                        "whisper": len(trans_info.get('whisper_models_loaded', [])),
                        "faster_whisper": len(trans_info.get('faster_whisper_models_loaded', []))
                    },
                    "total_cached": (len(trans_info.get('whisper_models_loaded', [])) + 
                                   len(trans_info.get('faster_whisper_models_loaded', [])))
                }
        except Exception as e:
            transcription_info = {"error": str(e)}
        
        return {
            "image_generator": {
                "model_name": model_info.get("model_name"),
                "is_loaded": model_info.get("is_loaded"),
                "device": model_info.get("device"),
                "cpu_offload": model_info.get("actual_cpu_offload", False),
                "gpu_memory": model_info.get("gpu_memory", {})
            },
            "gpu_memory": memory_info,
            "transcription_cache": transcription_info
        }
    
    except Exception as e:
        logger.log_error(e, None, current_user.username)
        raise HTTPException(status_code=500, detail=str(e))

# Transcription endpoints
@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_api(
    file: UploadFile = File(...),
    model_type: str = Form("faster-whisper"),
    model_name: str = Form("base"),
    language: str = Form("auto"),
    task: str = Form("transcribe"),
    condition_on_previous_text: bool = Form(True),
    initial_prompt: Optional[str] = Form(None),
    word_timestamps: bool = Form(False),
    device: str = Form("auto"),
    gpu_index: Optional[int] = Form(None),
    compute_type: str = Form("float16"),
    cpu_threads: int = Form(4),
    num_workers: int = Form(1),
    current_user: User = Depends(get_current_user)
):
    """Transcribe audio file with full parameter control."""
    start_time = time.time()
    
    try:
        from app.core.robust_transcription import get_robust_transcription_manager, TranscriptionConfig
        from app.utils.validation import validate_transcription_file
        from app.config import get_config_loader
        
        # Get configuration for file size limit
        config_loader = get_config_loader()
        web_config = config_loader.get_web_config()
        max_file_size = web_config.get('max_file_size', 16 * 1024 * 1024)
        
        # Validate file type and size before processing
        validate_transcription_file(file, max_file_size)
        
        manager = get_robust_transcription_manager()
        
        # Ensure uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        
        # Save uploaded file with size validation during read
        file_path = f"uploads/{file.filename}"
        content_size = 0
        with open(file_path, "wb") as buffer:
            # Read file in chunks to validate size during read
            while True:
                chunk = await file.read(8192)  # Read 8KB chunks
                if not chunk:
                    break
                content_size += len(chunk)
                if content_size > max_file_size:
                    # Clean up partial file
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
                    raise HTTPException(
                        status_code=413,
                        detail=f"File size exceeds maximum allowed size ({max_file_size} bytes)"
                    )
                buffer.write(chunk)
        
        # Get configuration for defaults
        from app.config import get_transcription_config
        env_config = get_transcription_config()

        # Use DeviceManager for "auto" mode
        if device == "auto":
            device_mgr = get_device_manager()
            device_info = device_mgr.get_device_for_service(
                service_name="transcription",
                config=env_config
            )
            device = device_info['device']
            gpu_index = device_info.get('device_index')
        elif gpu_index is None:
            # Fallback to config defaults for backward compatibility
            gpu_index = env_config.get('gpu_index')

        # Use float32 for CPU, float16 for CUDA
        if device == "cpu" and compute_type == "float16":
            compute_type = "float32"
        
        # Create transcription configuration
        config = TranscriptionConfig(
            audio_path=file_path,
            model_type=model_type,
            model_name=model_name,
            language=None if language == "auto" else language,
            task=task,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=word_timestamps,
            device=device,
            gpu_index=gpu_index,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers
        )
        
        # Transcribe
        result = manager.transcribe(config)
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except OSError:
            pass  # File might already be removed
        
        processing_time = time.time() - start_time
        
        if result.error:
            raise HTTPException(status_code=500, detail=result.error)
        
        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            model=result.model,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.log_error(e, None, current_user.username)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/transcribe/models")
async def get_transcription_models(current_user: User = Depends(get_current_user)):
    """Get available transcription models and parameters."""
    try:
        from app.core.robust_transcription import get_robust_transcription_manager
        
        manager = get_robust_transcription_manager()
        
        return {
            "available_models": {
                "whisper": manager.available_models,
                "faster-whisper": manager.available_models
            },
            "supported_languages": manager.supported_languages,
            "default_parameters": {
                "whisper": {
                    "model_name": "base",
                    "language": "auto",
                    "task": "transcribe",
                    "condition_on_previous_text": True,
                    "word_timestamps": False,
                    "device": "cuda",
                    "gpu_index": 3
                },
                "faster-whisper": {
                    "model_name": "base",
                    "language": "auto", 
                    "task": "transcribe",
                    "condition_on_previous_text": True,
                    "word_timestamps": False,
                    "device": "cuda",
                    "gpu_index": 3,
                    "compute_type": "float16",
                    "cpu_threads": 4,
                    "num_workers": 1,
                    "beam_size": 5,
                    "temperature": 0.0
                }
            },
            "tasks": ["transcribe", "translate"],
            "devices": ["cuda", "cpu"],
            "compute_types": ["float16", "float32", "int8"]
        }
    
    except Exception as e:
        logger.log_error(e, None, current_user.username)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/transcribe/info")
async def get_transcription_info(current_user: User = Depends(get_current_user)):
    """Get transcription manager information."""
    try:
        from app.core.robust_transcription import get_robust_transcription_manager
        
        manager = get_robust_transcription_manager()
        return manager.get_model_info()
    
    except Exception as e:
        logger.log_error(e, None, current_user.username)
        raise HTTPException(status_code=500, detail=str(e))

# File serving endpoints
@app.get("/generated/{filename}")
async def serve_generated_file(filename: str, current_user: User = Depends(get_current_user)):
    """Serve generated files."""
    file_path = f"generated/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@app.get("/uploads/{filename}")
async def serve_uploaded_file(filename: str, current_user: User = Depends(get_current_user)):
    """Serve uploaded files."""
    file_path = f"uploads/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup."""
    global device_manager

    # Initialize DeviceManager
    device_manager = initialize_device_manager()
    logger.logger.info(f"DeviceManager initialized: {len(device_manager.available_devices)} GPU(s)")

    validation_errors = []
    validation_warnings = []
    
    # Validate JWT secret key
    jwt_secret = os.getenv("JWT_SECRET_KEY")
    if not jwt_secret or jwt_secret == "your-secret-key-change-in-production":
        validation_errors.append(
            "JWT_SECRET_KEY environment variable must be set. "
            "Generate a secure key with: openssl rand -hex 32"
        )
    
    # Validate OpenAI API key (warning, not error - service can work without it)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here" or openai_key == "OPENAI_API_KEY":
        validation_warnings.append(
            "OPENAI_API_KEY not configured - OpenAI LLM features will not be available"
        )
    
    # Validate Hugging Face token (warning for image generation)
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token or hf_token == "your_huggingface_token_here" or hf_token == "HUGGINGFACE_TOKEN":
        validation_warnings.append(
            "HUGGINGFACE_TOKEN not configured - Image generation may not work"
        )
    
    # Log validation results
    if validation_errors:
        logger.log_system_event("startup_error", {
            "errors": validation_errors,
            "warnings": validation_warnings
        })
        raise ValueError("\n".join(validation_errors))
    
    if validation_warnings:
        logger.log_system_event("startup_warnings", {
            "warnings": validation_warnings
        })
        for warning in validation_warnings:
            logger.logger.warning(f"Startup warning: {warning}")
    
    logger.log_system_event("api_startup", {
        "version": "1.0.0",
        "warnings": len(validation_warnings),
        "status": "started successfully"
    })

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.log_system_event("api_shutdown", {"message": "Cleaning up GPU resources"})
    cleanup_generator()

    # Clear GPU memory cache if torch is available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.logger.info("GPU memory cache cleared")
    except ImportError:
        pass

if __name__ == "__main__":
    # Run the application
    web_config = config_loader.get_web_config()
    uvicorn.run(
        "app.main:app",
        host=web_config.get('host', '0.0.0.0'),
        port=web_config.get('port', 5000),
        reload=web_config.get('debug', False)
    ) 