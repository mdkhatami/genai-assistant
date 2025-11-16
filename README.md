# 🚀 GenAI Assistant

A comprehensive GenAI assistant with core functionalities for LLM responses, image generation, and transcription. Features GPU acceleration, multiple model support, and both CLI and web interfaces.

## 📋 Quick Start

### Option 1: Local Development (Recommended for Development)

One script handles everything - no Docker required:

```bash
git clone <repository-url>
cd genai_assitant
./setup_local.sh
```

This script will:
- ✅ Check Python 3.8+ installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Setup .env file (if missing)
- ✅ Start FastAPI server (default port: 5000, configurable via WEB_PORT in .env)
- ✅ Serve webapp (default port: 8080, configurable via WEBAPP_PORT in .env)
- ✅ Open browser automatically

**Server**: http://localhost:${WEB_PORT:-5000}  
**Webapp**: http://localhost:${WEBAPP_PORT:-8080}  
**API Docs**: http://localhost:${WEB_PORT:-5000}/docs

> **Note**: All port numbers are configurable via environment variables in `.env`. See `.env_example` for port configuration options.

### Option 2: Docker Setup (Recommended for Testing)

For users with Docker installed:

```bash
git clone <repository-url>
cd genai_assitant
./setup_docker.sh
```

This script will:
- ✅ Check Docker and docker-compose installation
- ✅ Setup .env file (if missing)
- ✅ Build Docker image
- ✅ Start containers
- ✅ Serve webapp (default port: 8080, configurable via WEBAPP_PORT in .env)
- ✅ Open browser automatically

**Server**: http://localhost:${WEB_PORT:-5000}  
**Webapp**: http://localhost:${WEBAPP_PORT:-8080}

> **Note**: All port numbers are configurable via environment variables in `.env`. See `.env_example` for port configuration options.

### Option 3: Production Deployment

For production servers with nginx:

```bash
git clone <repository-url>
cd genai_assitant
sudo ./setup_production.sh
```

This script will:
- ✅ Setup systemd service
- ✅ Configure nginx reverse proxy
- ✅ Setup SSL certificates (optional)
- ✅ Configure firewall
- ✅ Start and enable services

**Access**: http://your-domain (or http://localhost)

### Manual Setup (Alternative)

If you prefer manual setup:

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup .env file
cp .env_example .env
# Edit .env with your API keys

# 4. Start server
./start_development.sh

# 5. In another terminal, serve webapp
cd standalone_webapp
python3 -m http.server ${WEBAPP_PORT:-8080}
```

## 📚 Documentation

### 📖 [User Guide](docs/USER_GUIDE.md)
Complete user guide covering:
- Features and capabilities
- Installation and setup
- Web interface usage
- Command line interface
- Configuration options
- Troubleshooting

### 🛠️ [Developer Guide](docs/DEVELOPER_GUIDE.md)
Developer documentation including:
- Project structure and architecture
- API reference
- Core components
- Development setup
- Testing procedures
- Contributing guidelines

### 🚀 [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
Production deployment guide covering:
- Production setup
- Service management
- Server monitoring
- Configuration
- Troubleshooting
- Maintenance

## 🎯 Key Features

- **🤖 LLM Integration**: OpenAI ChatGPT and Ollama local models
- **🎨 Image Generation**: Flux models with GPU acceleration
- **🎤 Audio Transcription**: Whisper and Faster-Whisper
- **🌐 Web Interface**: Modern standalone web application
- **💻 CLI Interface**: Command-line tools for all features
- **🔐 Authentication**: JWT-based secure access
- **⚡ GPU Acceleration**: Multi-GPU support for optimal performance

## 🏗️ Architecture

```
Client Layer (Web/CLI) → FastAPI Server → Core Services → AI Providers
                                    ↓
                            GPU Infrastructure
```

## 🚀 Quick Commands

### Setup Scripts

```bash
# Local development (all-in-one)
./setup_local.sh

# Docker setup
./setup_docker.sh

# Production deployment
sudo ./setup_production.sh
```

### Manual Server Management

```bash
# Start development server
./start_development.sh

# Start production server
./start_production.sh

# Check service status (production)
sudo ./service_management.sh status

# Test API (uses WEB_PORT from .env, default: 5000)
curl http://localhost:${WEB_PORT:-5000}/health
```

### Docker Management

```bash
# Start containers
docker compose up -d

# Stop containers
docker compose down

# View logs
docker compose logs -f

# Restart
docker compose restart
```

## 📊 Current Status

✅ **Production Ready**  
✅ **All Components Healthy**  
✅ **GPU Accelerated**  
✅ **Multi-Provider Support**  

## 🔧 Configuration

### Environment Variables

The setup scripts will automatically create a `.env` file from the template if it doesn't exist. Edit `.env` to configure:

- **OPENAI_API_KEY**: Your OpenAI API key for LLM features
- **HUGGINGFACE_TOKEN**: Your Hugging Face token for image generation
- **CUDA_VISIBLE_DEVICES**: GPU configuration (default: 1,2,3)
- **OLLAMA_GPU_INDEX**: GPU for Ollama LLM (default: 1)
- **TRANSCRIPTION_GPU_INDEX**: GPU for transcription (default: 2)
- **IMAGE_GENERATION_GPU_INDEX**: GPU for image generation (default: 3)
- **ADMIN_USERNAME** / **ADMIN_PASSWORD**: Admin credentials
- **JWT_SECRET_KEY**: Secret key for JWT tokens

### Manual Configuration

```bash
# Copy template
cp .env_example .env

# Edit with your preferred editor
nano .env
# or
vim .env
```

## 🐛 Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Check what's using the port
lsof -i :${WEB_PORT:-5000}
# or
netstat -tulpn | grep ${WEB_PORT:-5000}

# Kill the process or change port in .env
```

**Python version issues:**
```bash
# Check Python version
python3 --version

# Should be 3.8 or higher
```

**Docker issues:**
```bash
# Check Docker is running
docker info

# Check container logs
docker compose logs
```

**Service not starting (production):**
```bash
# Check service status
sudo systemctl status genai-assistant

# View logs
sudo journalctl -u genai-assistant -f

# Check nginx
sudo nginx -t
sudo systemctl status nginx
```

**Permission errors:**
```bash
# Make scripts executable
chmod +x setup_*.sh
chmod +x start_*.sh
```

## 📞 Support

- **Documentation**: See `docs/` directory
- **Issues**: Check troubleshooting sections in guides
- **API Docs**: http://localhost:${WEB_PORT:-5000}/docs (when server running)

### Port Configuration

All port numbers are configurable via environment variables in the `.env` file:

- `WEB_PORT` - Main FastAPI server port (default: 5000)
- `WEBAPP_PORT` - Standalone webapp HTTP server port (default: 8080)
- `OLLAMA_PORT` - Ollama service port (default: 11434)
- `NGINX_HTTP_PORT` - Nginx HTTP listener port (default: 80)
- `NGINX_HTTPS_PORT` - Nginx HTTPS listener port (default: 443)

See `.env_example` for all configuration options.
- **Setup Scripts**: Run with `--help` or check script comments

---

**Version**: 1.0.0  
**Status**: ✅ Production Ready  
**Last Updated**: 2025-01-14
