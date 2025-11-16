#!/bin/bash
# GenAI Assistant - Docker Setup Script
# This script sets up and runs the GenAI Assistant using Docker

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
PID_FILE="$PROJECT_ROOT/.pids"

# Load port configuration from .env if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

# Get ports from environment variables with defaults
SERVER_PORT=${WEB_PORT:-${PORT:-5000}}
WEBAPP_PORT=${WEBAPP_PORT:-8080}
WEBAPP_URL="http://localhost:$WEBAPP_PORT"
CONTAINER_NAME="genai-assistant"

# Change to project root
cd "$PROJECT_ROOT"

# Print header
echo "================================================"
echo "🐳 GenAI Assistant - Docker Setup"
echo "================================================"
echo ""

# Setup cleanup trap
setup_cleanup_trap "$PID_FILE"

# Function to check Docker prerequisites
check_docker() {
    print_step "Checking Docker installation..."
    
    if ! check_command docker; then
        print_error "Docker is not installed"
        print_info "Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    print_success "Docker found: $(docker --version)"
    
    # Check if docker-compose is available (either as plugin or standalone)
    if docker compose version >/dev/null 2>&1; then
        DOCKER_COMPOSE_CMD="docker compose"
        print_success "Docker Compose (plugin) found"
    elif check_command docker-compose; then
        DOCKER_COMPOSE_CMD="docker-compose"
        print_success "Docker Compose (standalone) found"
    else
        print_error "Docker Compose is not installed"
        print_info "Please install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running"
        print_info "Please start Docker and try again"
        exit 1
    fi
    
    print_success "Docker daemon is running"
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
        print_warning "Port $SERVER_PORT is already in use"
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

# Function to build and start Docker containers
build_and_start() {
    print_step "Building and starting Docker containers..."
    
    if [ ! -f "$PROJECT_ROOT/docker-compose.yml" ]; then
        print_error "docker-compose.yml not found"
        exit 1
    fi
    
    if [ ! -f "$PROJECT_ROOT/Dockerfile" ]; then
        print_error "Dockerfile not found"
        exit 1
    fi
    
    # Stop existing containers if running
    print_info "Stopping existing containers (if any)..."
    $DOCKER_COMPOSE_CMD down 2>/dev/null || true
    
    # Build and start containers
    print_info "Building Docker image (this may take a while)..."
    $DOCKER_COMPOSE_CMD build --quiet
    
    print_info "Starting containers..."
    $DOCKER_COMPOSE_CMD up -d
    
    print_success "Containers started"
    
    # Show container status
    print_info "Container status:"
    $DOCKER_COMPOSE_CMD ps
}

# Function to wait for server health
wait_for_server() {
    print_step "Waiting for server to be ready..."
    
    if wait_for_url "http://localhost:$SERVER_PORT/health" 60 3; then
        print_success "Server is ready and responding"
    else
        print_error "Server did not start properly"
        print_info "Checking container logs..."
        $DOCKER_COMPOSE_CMD logs --tail=50
        exit 1
    fi
}

# Function to start webapp server (local Python server)
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

# Function to show logs
show_logs() {
    print_info "Container logs (last 20 lines):"
    $DOCKER_COMPOSE_CMD logs --tail=20
}

# Main execution
main() {
    check_docker
    setup_env_file
    check_ports
    build_and_start
    wait_for_server
    start_webapp
    
    echo ""
    echo "================================================"
    print_success "Docker setup complete!"
    echo "================================================"
    echo ""
    echo "📡 Server: http://localhost:$SERVER_PORT"
    echo "🌐 Webapp: $WEBAPP_URL"
    echo "📚 API Docs: http://localhost:$SERVER_PORT/docs"
    echo ""
    echo "📝 Useful commands:"
    echo "   - View logs: $DOCKER_COMPOSE_CMD logs -f"
    echo "   - Stop containers: $DOCKER_COMPOSE_CMD down"
    echo "   - Restart: $DOCKER_COMPOSE_CMD restart"
    echo "   - Status: $DOCKER_COMPOSE_CMD ps"
    echo ""
    echo "🛑 To stop: Press Ctrl+C (will stop webapp server)"
    echo "   To stop containers: $DOCKER_COMPOSE_CMD down"
    echo ""
    
    # Open browser
    sleep 2
    open_browser "$WEBAPP_URL"
    
    # Keep script running and show logs
    print_info "Press Ctrl+C to stop webapp server (containers will keep running)..."
    print_info "To view container logs, run: $DOCKER_COMPOSE_CMD logs -f"
    wait
}

# Run main function
main

