#!/bin/bash
# Test script for local setup validation
# This script validates that the system can be started successfully using setup_local.sh

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
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
PID_FILE="$PROJECT_ROOT/.test_pids"
TEST_LOG="$PROJECT_ROOT/test_startup_local.log"

# Load port configuration from .env if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

# Get ports from environment variables with defaults
SERVER_PORT=${WEB_PORT:-${PORT:-5000}}
WEBAPP_PORT=${WEBAPP_PORT:-8080}

# Change to project root
cd "$PROJECT_ROOT"

# Cleanup function
cleanup() {
    print_info "Cleaning up test processes..."
    if [ -f "$PID_FILE" ]; then
        kill_pids_from_file "$PID_FILE" 2>/dev/null || true
        rm -f "$PID_FILE"
    fi
    # Kill any remaining test processes
    pkill -f "python.*main.py" 2>/dev/null || true
    pkill -f "python.*http.server.*$WEBAPP_PORT" 2>/dev/null || true
}

# Setup cleanup trap
trap cleanup EXIT INT TERM

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Test function
run_test() {
    local test_name="$1"
    local test_func="$2"
    
    print_step "Running: $test_name"
    if $test_func; then
        print_success "$test_name passed"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        print_error "$test_name failed"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Test 1: Check prerequisites
test_prerequisites() {
    if ! check_python_version; then
        return 1
    fi
    
    if ! check_command pip3; then
        print_error "pip3 is not installed"
        return 1
    fi
    
    return 0
}

# Test 2: Check .env file
test_env_file() {
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        print_error ".env file not found"
        return 1
    fi
    
    # Check for required variables
    if ! grep -q "^JWT_SECRET_KEY=" "$PROJECT_ROOT/.env" || \
       grep -q "^JWT_SECRET_KEY=your-secret-key-change-in-production" "$PROJECT_ROOT/.env"; then
        print_error "JWT_SECRET_KEY not properly configured"
        return 1
    fi
    
    if ! grep -q "^ADMIN_USERNAME=" "$PROJECT_ROOT/.env"; then
        print_error "ADMIN_USERNAME not found in .env"
        return 1
    fi
    
    if ! grep -q "^ADMIN_PASSWORD=" "$PROJECT_ROOT/.env" || \
       grep -q "^ADMIN_PASSWORD=change-this-password-in-production" "$PROJECT_ROOT/.env"; then
        print_error "ADMIN_PASSWORD not properly configured"
        return 1
    fi
    
    return 0
}

# Test 3: Check port availability
test_ports() {
    if ! check_port $SERVER_PORT; then
        print_error "Port $SERVER_PORT is already in use"
        return 1
    fi
    
    if ! check_port $WEBAPP_PORT; then
        print_error "Port $WEBAPP_PORT is already in use"
        return 1
    fi
    
    return 0
}

# Test 4: Setup virtual environment
test_venv_setup() {
    if [ -d "$VENV_DIR" ]; then
        print_info "Virtual environment already exists, using it"
    else
        print_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR" || return 1
    fi
    
    # Activate and verify
    source "$VENV_DIR/bin/activate"
    if [ -z "$VIRTUAL_ENV" ]; then
        print_error "Failed to activate virtual environment"
        return 1
    fi
    
    return 0
}

# Test 5: Install dependencies
test_install_dependencies() {
    if [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
        print_error "requirements.txt not found"
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    print_info "Installing dependencies (this may take a while)..."
    pip install --quiet --upgrade pip > "$TEST_LOG" 2>&1
    pip install --quiet -r "$PROJECT_ROOT/requirements.txt" >> "$TEST_LOG" 2>&1 || return 1
    
    return 0
}

# Test 6: Start FastAPI server
test_start_server() {
    source "$VENV_DIR/bin/activate"
    
    # Load .env file
    if [ -f "$PROJECT_ROOT/.env" ]; then
        set -a
        source "$PROJECT_ROOT/.env"
        set +a
    fi
    
    print_info "Starting FastAPI server on port $SERVER_PORT..."
    python "$PROJECT_ROOT/main.py" > "$PROJECT_ROOT/test_server.log" 2>&1 &
    SERVER_PID=$!
    store_pid $SERVER_PID "$PID_FILE"
    
    # Wait for server to be ready
    sleep 3
    if wait_for_url "http://localhost:$SERVER_PORT/health" 30 2; then
        print_success "Server is ready and responding"
        return 0
    else
        print_error "Server did not start properly. Check logs: $PROJECT_ROOT/test_server.log"
        tail -20 "$PROJECT_ROOT/test_server.log"
        return 1
    fi
}

# Test 7: Start webapp server
test_start_webapp() {
    if [ ! -d "$PROJECT_ROOT/standalone_webapp" ]; then
        print_error "standalone_webapp directory not found"
        return 1
    fi
    
    print_info "Starting webapp server on port $WEBAPP_PORT..."
    cd "$PROJECT_ROOT/standalone_webapp"
    python3 -m http.server $WEBAPP_PORT > "$PROJECT_ROOT/test_webapp.log" 2>&1 &
    WEBAPP_PID=$!
    store_pid $WEBAPP_PID "$PID_FILE"
    cd "$PROJECT_ROOT"
    
    # Wait a moment for server to start
    sleep 2
    
    # Check if webapp is accessible
    if curl -sf "http://localhost:$WEBAPP_PORT" >/dev/null 2>&1; then
        print_success "Webapp server is accessible"
        return 0
    else
        print_error "Webapp server is not accessible"
        return 1
    fi
}

# Test 8: Verify health endpoint
test_health_endpoint() {
    if wait_for_url "http://localhost:$SERVER_PORT/health" 10 1; then
        HEALTH_RESPONSE=$(curl -sf "http://localhost:$SERVER_PORT/health")
        if echo "$HEALTH_RESPONSE" | grep -q '"status"'; then
            print_success "Health endpoint returns valid JSON"
            return 0
        else
            print_error "Health endpoint response is invalid"
            return 1
        fi
    else
        print_error "Health endpoint is not accessible"
        return 1
    fi
}

# Main test execution
main() {
    echo "================================================"
    echo "🧪 Local Setup Validation Tests"
    echo "================================================"
    echo ""
    
    # Run tests
    run_test "Prerequisites check" test_prerequisites
    run_test ".env file validation" test_env_file
    run_test "Port availability check" test_ports
    run_test "Virtual environment setup" test_venv_setup
    run_test "Dependencies installation" test_install_dependencies
    run_test "FastAPI server startup" test_start_server
    run_test "Webapp server startup" test_start_webapp
    run_test "Health endpoint verification" test_health_endpoint
    
    # Summary
    echo ""
    echo "================================================"
    echo "Test Summary"
    echo "================================================"
    echo "✅ Passed: $TESTS_PASSED"
    echo "❌ Failed: $TESTS_FAILED"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        print_success "All tests passed!"
        exit 0
    else
        print_error "Some tests failed"
        exit 1
    fi
}

# Run main function
main

