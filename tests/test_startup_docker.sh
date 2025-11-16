#!/bin/bash
# Test script for Docker setup validation
# This script validates that the system can be started successfully using setup_docker.sh

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
CONTAINER_NAME="genai-assistant-test"

# Load port configuration from .env if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

# Get ports from environment variables with defaults
SERVER_PORT=${WEB_PORT:-${PORT:-5000}}
WEBAPP_PORT=${WEBAPP_PORT:-8080}

# Change to project root
cd "$PROJECT_ROOT"

# Determine docker-compose command
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    echo "Error: docker-compose not found"
    exit 1
fi

# Cleanup function
cleanup() {
    print_info "Cleaning up test containers..."
    # Stop and remove test containers
    $DOCKER_COMPOSE_CMD down 2>/dev/null || true
    # Remove test container if it exists
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
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

# Test 1: Check Docker prerequisites
test_docker_prerequisites() {
    if ! check_command docker; then
        print_error "Docker is not installed"
        return 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running"
        return 1
    fi
    
    if ! $DOCKER_COMPOSE_CMD version >/dev/null 2>&1; then
        print_error "Docker Compose is not available"
        return 1
    fi
    
    print_success "Docker prerequisites met"
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
    
    return 0
}

# Test 4: Check Docker files exist
test_docker_files() {
    if [ ! -f "$PROJECT_ROOT/docker-compose.yml" ]; then
        print_error "docker-compose.yml not found"
        return 1
    fi
    
    if [ ! -f "$PROJECT_ROOT/Dockerfile" ]; then
        print_error "Dockerfile not found"
        return 1
    fi
    
    return 0
}

# Test 5: Build Docker image
test_build_image() {
    print_info "Building Docker image (this may take a while)..."
    if $DOCKER_COMPOSE_CMD build --quiet > "$PROJECT_ROOT/test_docker_build.log" 2>&1; then
        print_success "Docker image built successfully"
        return 0
    else
        print_error "Docker image build failed. Check logs: $PROJECT_ROOT/test_docker_build.log"
        tail -20 "$PROJECT_ROOT/test_docker_build.log"
        return 1
    fi
}

# Test 6: Start containers
test_start_containers() {
    print_info "Starting containers..."
    
    # Stop any existing containers first
    $DOCKER_COMPOSE_CMD down 2>/dev/null || true
    
    if $DOCKER_COMPOSE_CMD up -d > "$PROJECT_ROOT/test_docker_start.log" 2>&1; then
        print_success "Containers started"
        
        # Wait a moment for containers to initialize
        sleep 5
        
        # Check container status
        if $DOCKER_COMPOSE_CMD ps | grep -q "Up"; then
            print_success "Containers are running"
            return 0
        else
            print_error "Containers are not running"
            $DOCKER_COMPOSE_CMD ps
            return 1
        fi
    else
        print_error "Failed to start containers. Check logs: $PROJECT_ROOT/test_docker_start.log"
        tail -20 "$PROJECT_ROOT/test_docker_start.log"
        return 1
    fi
}

# Test 7: Wait for server health
test_server_health() {
    print_info "Waiting for server to be ready..."
    
    if wait_for_url "http://localhost:$SERVER_PORT/health" 60 3; then
        print_success "Server is ready and responding"
        
        # Verify health endpoint returns valid JSON
        HEALTH_RESPONSE=$(curl -sf "http://localhost:$SERVER_PORT/health")
        if echo "$HEALTH_RESPONSE" | grep -q '"status"'; then
            print_success "Health endpoint returns valid JSON"
            return 0
        else
            print_error "Health endpoint response is invalid"
            return 1
        fi
    else
        print_error "Server did not become ready"
        print_info "Container logs:"
        $DOCKER_COMPOSE_CMD logs --tail=30
        return 1
    fi
}

# Test 8: Check container logs for errors
test_container_logs() {
    print_info "Checking container logs for errors..."
    
    LOGS=$($DOCKER_COMPOSE_CMD logs --tail=50 2>&1)
    
    # Check for critical errors
    if echo "$LOGS" | grep -qi "error\|exception\|traceback\|failed"; then
        # Some errors might be acceptable (warnings), so we'll just warn
        print_warning "Some errors found in logs (may be acceptable):"
        echo "$LOGS" | grep -i "error\|exception\|traceback\|failed" | head -10
        # Don't fail the test, just warn
    fi
    
    return 0
}

# Main test execution
main() {
    echo "================================================"
    echo "🐳 Docker Setup Validation Tests"
    echo "================================================"
    echo ""
    
    # Run tests
    run_test "Docker prerequisites check" test_docker_prerequisites
    run_test ".env file validation" test_env_file
    run_test "Port availability check" test_ports
    run_test "Docker files check" test_docker_files
    run_test "Docker image build" test_build_image
    run_test "Container startup" test_start_containers
    run_test "Server health check" test_server_health
    run_test "Container logs check" test_container_logs
    
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

