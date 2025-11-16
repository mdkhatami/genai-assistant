#!/bin/bash
# GenAI Assistant - Local Development Setup Script
# This script sets up and runs the GenAI Assistant locally without Docker

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
HELPERS="$PROJECT_ROOT/scripts/helpers.sh"

# Source helper functions
if [ -f "$HELPERS" ]; then
    source "$HELPERS"
else
    echo "Error: helpers.sh not found at $HELPERS"
    exit 1
fi

# Configuration
VENV_DIR="$PROJECT_ROOT/venv"
PID_FILE="$PROJECT_ROOT/.pids"

# Load port configuration from .env if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

# Get ports from environment variables with defaults
SERVER_PORT=${WEB_PORT:-${PORT:-5000}}
WEBAPP_PORT=${WEBAPP_PORT:-8080}
WEBAPP_URL="http://localhost:$WEBAPP_PORT"

# Change to project root
cd "$PROJECT_ROOT"

# Print header
echo "================================================"
echo "🚀 GenAI Assistant - Local Development Setup"
echo "================================================"
echo ""

# Setup cleanup trap
setup_cleanup_trap "$PID_FILE"

# Function to check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    if ! check_python_version; then
        print_error "Please install Python 3.8 or higher"
        exit 1
    fi
    
    if ! check_command pip3; then
        print_error "pip3 is not installed. Please install pip."
        exit 1
    fi
    
    print_success "All prerequisites met"
}

# Function to setup virtual environment
setup_venv() {
    print_step "Setting up virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        print_info "Virtual environment already exists"
    else
        print_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --quiet --upgrade pip
}

# Function to install dependencies
install_dependencies() {
    print_step "Installing dependencies..."
    
    if [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    print_info "Installing packages from requirements.txt..."
    pip install --quiet -r "$PROJECT_ROOT/requirements.txt"
    print_success "Dependencies installed"
}

# Function to check and create .env file
setup_env_file() {
    print_step "Checking .env file..."
    
    if ! create_env_from_template "$PROJECT_ROOT"; then
        print_warning "Could not create .env file automatically"
    fi
    
    if [ -f "$PROJECT_ROOT/.env" ]; then
        print_success ".env file exists"
        # Check if API keys are set (basic check)
        if grep -q "your_openai_api_key_here\|OPENAI_API_KEY" "$PROJECT_ROOT/.env" && ! grep -q "^OPENAI_API_KEY=sk-" "$PROJECT_ROOT/.env"; then
            print_warning "Please update .env file with your API keys before using the service"
        fi
    else
        print_error ".env file not found and could not be created"
        exit 1
    fi
}

# Function to check ports
check_ports() {
    print_step "Checking if ports are available..."
    
    if ! check_port $SERVER_PORT; then
        print_warning "Port $SERVER_PORT is already in use. The server might already be running."
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    if ! check_port $WEBAPP_PORT; then
        print_warning "Port $WEBAPP_PORT is already in use. Trying to use it anyway..."
    fi
}

# Function to start FastAPI server
start_server() {
    print_step "Starting FastAPI server..."
    
    # Activate venv if not already activated
    source "$VENV_DIR/bin/activate"

    # Load .env file (device configuration now in .env or auto-detected)
    if [ -f "$PROJECT_ROOT/.env" ]; then
        set -a
        source "$PROJECT_ROOT/.env"
        set +a
    fi
    
    # Start server in background
    print_info "Starting server on port $SERVER_PORT..."
    python "$PROJECT_ROOT/main.py" > "$PROJECT_ROOT/server.log" 2>&1 &
    SERVER_PID=$!
    store_pid $SERVER_PID "$PID_FILE"
    
    print_success "Server started (PID: $SERVER_PID)"
    print_info "Server logs: $PROJECT_ROOT/server.log"
    
    # Wait for server to be ready
    sleep 3
    if wait_for_url "http://localhost:$SERVER_PORT/health" 30 2; then
        print_success "Server is ready and responding"
    else
        print_error "Server did not start properly. Check logs: $PROJECT_ROOT/server.log"
        exit 1
    fi
}

# Function to start webapp server
start_webapp() {
    print_step "Starting webapp server..."
    
    if [ ! -d "$PROJECT_ROOT/standalone_webapp" ]; then
        print_error "standalone_webapp directory not found"
        exit 1
    fi
    
    print_info "Starting webapp server on port $WEBAPP_PORT..."
    cd "$PROJECT_ROOT/standalone_webapp"
    python3 -m http.server $WEBAPP_PORT > "$PROJECT_ROOT/webapp.log" 2>&1 &
    WEBAPP_PID=$!
    store_pid $WEBAPP_PID "$PID_FILE"
    cd "$PROJECT_ROOT"
    
    print_success "Webapp server started (PID: $WEBAPP_PID)"
    print_info "Webapp logs: $PROJECT_ROOT/webapp.log"
    
    # Wait a moment for server to start
    sleep 2
}

# Main execution
main() {
    check_prerequisites
    setup_venv
    install_dependencies
    setup_env_file
    check_ports
    start_server
    start_webapp
    
    echo ""
    echo "================================================"
    print_success "Setup complete!"
    echo "================================================"
    echo ""
    echo "📡 Server: http://localhost:$SERVER_PORT"
    echo "🌐 Webapp: $WEBAPP_URL"
    echo "📚 API Docs: http://localhost:$SERVER_PORT/docs"
    echo ""
    echo "📝 Logs:"
    echo "   - Server: $PROJECT_ROOT/server.log"
    echo "   - Webapp: $PROJECT_ROOT/webapp.log"
    echo ""
    echo "🛑 To stop: Press Ctrl+C or run: pkill -f 'python.*main.py|python.*http.server'"
    echo ""
    
    # Open browser
    sleep 2
    open_browser "$WEBAPP_URL"
    
    # Keep script running
    print_info "Press Ctrl+C to stop all services..."
    wait
}

# Run main function
main

