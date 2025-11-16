#!/bin/bash
# Shared helper functions for GenAI Assistant setup scripts

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_step() {
    echo -e "${CYAN}▶️  $1${NC}"
}

# Check if a command exists
check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check if a port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 || netstat -an 2>/dev/null | grep -q ":$port.*LISTEN"; then
        return 1  # Port is in use
    else
        return 0  # Port is available
    fi
}

# Wait for a URL to be accessible
wait_for_url() {
    local url=$1
    local max_attempts=${2:-30}
    local delay=${3:-2}
    local attempt=0

    print_info "Waiting for $url to be accessible..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            print_success "URL $url is accessible"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep $delay
    done

    print_error "URL $url did not become accessible after $((max_attempts * delay)) seconds"
    return 1
}

# Check Python version (requires 3.8+)
check_python_version() {
    if ! check_command python3; then
        print_error "Python 3 is not installed"
        return 1
    fi

    local version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    local major=$(echo $version | cut -d. -f1)
    local minor=$(echo $version | cut -d. -f2)

    if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 8 ]); then
        print_error "Python 3.8+ is required, but found Python $version"
        return 1
    fi

    print_success "Python $version found"
    return 0
}

# Create .env file from template if it doesn't exist
create_env_from_template() {
    local project_root=$1
    local env_file="$project_root/.env"
    # Prefer root .env_example, fallback to docs/env_example.txt for backward compatibility
    local template_file="$project_root/.env_example"
    local fallback_template="$project_root/docs/env_example.txt"

    if [ -f "$env_file" ]; then
        print_info ".env file already exists"
        return 0
    fi

    # Use fallback if root .env_example doesn't exist
    if [ ! -f "$template_file" ] && [ -f "$fallback_template" ]; then
        template_file="$fallback_template"
    fi

    if [ ! -f "$template_file" ]; then
        print_warning "Template file not found at $template_file"
        print_info "Creating basic .env file..."
        cat > "$env_file" << EOF
# GenAI Assistant Environment Configuration
# Please fill in your API keys and configuration

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face Configuration
HUGGINGFACE_TOKEN=your_huggingface_token_here

# GPU Configuration
CUDA_VISIBLE_DEVICES=1,2,3
OLLAMA_GPU_INDEX=1
TRANSCRIPTION_GPU_INDEX=2
IMAGE_GENERATION_GPU_INDEX=3

# CORS Configuration
# Comma-separated list of allowed origins for CORS
# For production, specify exact origins (e.g., "https://yourdomain.com,https://app.yourdomain.com")
# For development, defaults to localhost origins
CORS_ORIGINS=http://localhost:8080,http://localhost:3000,http://127.0.0.1:8080,http://127.0.0.1:3000

# JWT Configuration
# Generate a secure secret key: openssl rand -hex 32
JWT_SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || echo "change-this-secret-key-in-production")

# Admin Configuration
# IMPORTANT: Change these default credentials in production!
ADMIN_USERNAME=admin
ADMIN_PASSWORD=change-this-password-in-production
ADMIN_EMAIL=admin@example.com
ADMIN_FULL_NAME=Administrator

# Server Configuration
HOST=0.0.0.0
PORT=5000
DEBUG=false
LOG_LEVEL=INFO

# Port Configuration
# All port numbers should be configured here - no hardcoded ports in the codebase
WEB_PORT=5000
WEBAPP_PORT=8080
OLLAMA_PORT=11434
NGINX_HTTP_PORT=80
NGINX_HTTPS_PORT=443

# Ollama Configuration
# OLLAMA_BASE_URL can be set directly, or will be constructed from OLLAMA_PORT
OLLAMA_BASE_URL=http://localhost:11434
EOF
    else
        print_info "Creating .env file from template..."
        cp "$template_file" "$env_file"
    fi

    print_warning ".env file created. Please edit it with your API keys and configuration."
    return 0
}

# Open browser (cross-platform)
open_browser() {
    local url=$1
    print_info "Opening browser to $url..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open "$url"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if check_command xdg-open; then
            xdg-open "$url" 2>/dev/null &
        elif check_command gnome-open; then
            gnome-open "$url" 2>/dev/null &
        else
            print_warning "Could not open browser automatically. Please open $url manually"
        fi
    else
        print_warning "Could not detect OS to open browser. Please open $url manually"
    fi
}

# Get project root directory
get_project_root() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "$(cd "$script_dir/.." && pwd)"
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script requires root/sudo privileges"
        return 1
    fi
    return 0
}

# Store PID to file
store_pid() {
    local pid=$1
    local pid_file=$2
    echo "$pid" >> "$pid_file"
}

# Kill processes by PID file
kill_pids_from_file() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        while read -r pid; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                print_info "Stopping process $pid..."
                kill "$pid" 2>/dev/null || true
            fi
        done < "$pid_file"
        rm -f "$pid_file"
    fi
}

# Cleanup function for trap
cleanup_on_exit() {
    local pid_file=$1
    print_info "Cleaning up..."
    kill_pids_from_file "$pid_file"
    exit 0
}

# Setup signal traps
setup_cleanup_trap() {
    local pid_file=$1
    trap "cleanup_on_exit $pid_file" INT TERM EXIT
}

