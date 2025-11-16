#!/bin/bash
# Development startup script (with reload)
# GenAI Assistant FastAPI Server - Development Mode

echo "🚀 Starting GenAI Assistant FastAPI Server (Development Mode)..."
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment 'venv' not found!"
    echo "Please create a virtual environment first using uv:"
    echo "  uv venv venv"
    echo "  source venv/bin/activate"
    echo "  uv pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found in current directory!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Clear any existing GPU memory
echo "🧹 Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache(); print('✅ GPU memory cleared')" 2>/dev/null || echo "ℹ️  Torch not available or no GPU (will use CPU mode)"

# Load configuration from .env file if it exists
if [ -f ".env" ]; then
    echo "📄 Loading configuration from .env..."
    source .env
fi

# Get port from environment variable, default to 5000
WEB_PORT=${WEB_PORT:-${PORT:-5000}}

echo ""
echo "🔧 Device Configuration:"
echo "  - Device selection: Auto-detect (GPU if available, CPU otherwise)"
echo "  - Configuration: See .env file or defaults in config.yaml"
echo "  - Server host: 0.0.0.0"
echo "  - Server port: $WEB_PORT"
echo "  - Environment: Development (Auto-Reload Enabled)"

echo ""
echo "🌟 Starting FastAPI server in development mode..."
echo "📡 Server will be available at: http://localhost:$WEB_PORT"
echo "📚 API Documentation: http://localhost:$WEB_PORT/docs"
echo ""
echo "👤 Default credentials:"
echo "  Username: admin"
echo "  Password: admin123"
echo ""
echo "🔄 Auto-reload is enabled - server will restart when files change"
echo "🛑 To stop the server, press Ctrl+C"
echo "================================================"
echo ""

# Start the FastAPI server with reload enabled for development
python main.py
