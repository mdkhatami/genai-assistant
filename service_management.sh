#!/bin/bash
# GenAI Assistant FastAPI Service Management Script
# This script provides easy commands to manage the service

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to show usage
show_usage() {
    echo "GenAI Assistant FastAPI Service Management"
    echo "=========================================="
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     - Start the service"
    echo "  stop      - Stop the service"
    echo "  restart   - Restart the service"
    echo "  status    - Show service status"
    echo "  logs      - Show service logs (follow mode)"
    echo "  logs-n    - Show last 50 log lines"
    echo "  enable    - Enable service to start on boot"
    echo "  disable   - Disable service from starting on boot"
    echo "  test      - Test the API health endpoint"
    echo "  update    - Pull latest code and restart"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 restart"
    echo "  $0 logs"
    echo "  $0 update"
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This command needs to be run with sudo privileges"
        echo "Usage: sudo $0 $1"
        exit 1
    fi
}

# Function to test API
test_api() {
    # Load port from .env if it exists
    if [ -f ".env" ]; then
        source .env
    fi
    
    # Get port from environment variable, default to 5000
    WEB_PORT=${WEB_PORT:-${PORT:-5000}}
    
    print_info "Testing API health endpoint..."
    if curl -s http://localhost:$WEB_PORT/health > /dev/null; then
        print_status "API is responding correctly"
        echo "Health check response:"
        curl -s http://localhost:$WEB_PORT/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:$WEB_PORT/health
    else
        print_error "API is not responding"
        echo "Make sure the service is running: sudo $0 status"
    fi
}

# Function to update code and restart
update_service() {
    print_info "Updating code and restarting service..."
    
    # Check if we're in a git repository
    if [ ! -d ".git" ]; then
        print_warning "Not a git repository. Skipping git pull."
    else
        print_info "Pulling latest code..."
        git pull origin main || git pull origin master
    fi
    
    # Install dependencies if requirements.txt changed
    if [ -f "requirements.txt" ]; then
        print_info "Installing/updating dependencies..."
        source venv/bin/activate
        pip install -r requirements.txt
    fi
    
    # Restart service
    print_info "Restarting service..."
    systemctl restart genai-assistant
    
    # Wait and test
    sleep 5
    test_api
}

# Main command handling
case "$1" in
    start)
        check_root "start"
        print_info "Starting GenAI Assistant service..."
        systemctl start genai-assistant
        sleep 2
        systemctl status genai-assistant --no-pager
        ;;
    stop)
        check_root "stop"
        print_info "Stopping GenAI Assistant service..."
        systemctl stop genai-assistant
        print_status "Service stopped"
        ;;
    restart)
        check_root "restart"
        print_info "Restarting GenAI Assistant service..."
        systemctl restart genai-assistant
        sleep 3
        systemctl status genai-assistant --no-pager
        test_api
        ;;
    status)
        print_info "GenAI Assistant service status:"
        systemctl status genai-assistant --no-pager
        ;;
    logs)
        print_info "Showing service logs (follow mode)..."
        print_info "Press Ctrl+C to exit log view"
        journalctl -u genai-assistant -f
        ;;
    logs-n)
        print_info "Showing last 50 log lines:"
        journalctl -u genai-assistant -n 50 --no-pager
        ;;
    enable)
        check_root "enable"
        print_info "Enabling GenAI Assistant service to start on boot..."
        systemctl enable genai-assistant
        print_status "Service enabled"
        ;;
    disable)
        check_root "disable"
        print_info "Disabling GenAI Assistant service from starting on boot..."
        systemctl disable genai-assistant
        print_status "Service disabled"
        ;;
    test)
        test_api
        ;;
    update)
        check_root "update"
        update_service
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
