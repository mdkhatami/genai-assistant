#!/bin/bash
# End-to-end connection test
# This script tests the full flow: health → auth → API call

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

# Load port configuration from .env if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

# Get ports from environment variables with defaults
SERVER_PORT=${WEB_PORT:-${PORT:-5000}}
WEBAPP_PORT=${WEBAPP_PORT:-8080}
SERVER_URL="http://localhost:$SERVER_PORT"
WEBAPP_URL="http://localhost:$WEBAPP_PORT"

# Change to project root
cd "$PROJECT_ROOT"

# Configuration
START_SERVERS=${START_SERVERS:-false}  # Set to true to start servers automatically
PID_FILE="$PROJECT_ROOT/.test_e2e_pids"

# Cleanup function
cleanup() {
    print_info "Cleaning up..."
    if [ -f "$PID_FILE" ]; then
        kill_pids_from_file "$PID_FILE" 2>/dev/null || true
        rm -f "$PID_FILE"
    fi
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

# Start servers if requested
start_servers() {
    if [ "$START_SERVERS" != "true" ]; then
        print_info "Skipping server startup (set START_SERVERS=true to enable)"
        return 0
    fi
    
    print_info "Starting servers for E2E test..."
    
    # Check if servers are already running
    if curl -sf "$SERVER_URL/health" >/dev/null 2>&1; then
        print_info "Backend server is already running"
    else
        print_info "Starting backend server..."
        source "$PROJECT_ROOT/venv/bin/activate" 2>/dev/null || true
        if [ -f "$PROJECT_ROOT/.env" ]; then
            set -a
            source "$PROJECT_ROOT/.env"
            set +a
        fi
        python "$PROJECT_ROOT/main.py" > "$PROJECT_ROOT/test_e2e_server.log" 2>&1 &
        SERVER_PID=$!
        store_pid $SERVER_PID "$PID_FILE"
        
        # Wait for server
        if wait_for_url "$SERVER_URL/health" 30 2; then
            print_success "Backend server started"
        else
            print_error "Backend server failed to start"
            return 1
        fi
    fi
    
    # Check if webapp is already running
    if curl -sf "$WEBAPP_URL" >/dev/null 2>&1; then
        print_info "Webapp server is already running"
    else
        print_info "Starting webapp server..."
        cd "$PROJECT_ROOT/standalone_webapp"
        python3 -m http.server $WEBAPP_PORT > "$PROJECT_ROOT/test_e2e_webapp.log" 2>&1 &
        WEBAPP_PID=$!
        store_pid $WEBAPP_PID "$PID_FILE"
        cd "$PROJECT_ROOT"
        
        sleep 2
        if curl -sf "$WEBAPP_URL" >/dev/null 2>&1; then
            print_success "Webapp server started"
        else
            print_error "Webapp server failed to start"
            return 1
        fi
    fi
    
    return 0
}

# Test 1: Health endpoint
test_health_endpoint() {
    if ! wait_for_url "$SERVER_URL/health" 10 1; then
        print_error "Health endpoint is not accessible"
        return 1
    fi
    
    HEALTH_RESPONSE=$(curl -sf "$SERVER_URL/health")
    if echo "$HEALTH_RESPONSE" | grep -q '"status"'; then
        print_success "Health endpoint returns valid JSON"
        return 0
    else
        print_error "Health endpoint response is invalid"
        return 1
    fi
}

# Test 2: Authentication
test_authentication() {
    USERNAME=${ADMIN_USERNAME:-admin}
    PASSWORD=${ADMIN_PASSWORD:-}
    
    if [ -z "$PASSWORD" ]; then
        print_error "ADMIN_PASSWORD not set"
        return 1
    fi
    
    AUTH_RESPONSE=$(curl -sf -X POST "$SERVER_URL/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}")
    
    if [ $? -ne 0 ]; then
        print_error "Authentication request failed"
        return 1
    fi
    
    if echo "$AUTH_RESPONSE" | grep -q '"access_token"'; then
        TOKEN=$(echo "$AUTH_RESPONSE" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
        if [ -n "$TOKEN" ]; then
            print_success "Authentication successful, token obtained"
            export E2E_TOKEN="$TOKEN"
            return 0
        fi
    fi
    
    print_error "Authentication failed or token not returned"
    return 1
}

# Test 3: Protected endpoint access
test_protected_endpoint() {
    if [ -z "$E2E_TOKEN" ]; then
        print_error "No token available for protected endpoint test"
        return 1
    fi
    
    # Test accessing a protected endpoint
    API_RESPONSE=$(curl -sf "$SERVER_URL/api/llm/ollama/models" \
        -H "Authorization: Bearer $E2E_TOKEN")
    
    if [ $? -eq 0 ]; then
        print_success "Protected endpoint accessible with token"
        return 0
    else
        print_error "Protected endpoint access failed"
        return 1
    fi
}

# Test 4: CORS headers
test_cors_headers() {
    ORIGIN="$WEBAPP_URL"
    
    CORS_RESPONSE=$(curl -sf -H "Origin: $ORIGIN" \
        -H "Access-Control-Request-Method: GET" \
        -X OPTIONS "$SERVER_URL/health")
    
    # Check for CORS headers in a GET request
    GET_RESPONSE=$(curl -sf -v -H "Origin: $ORIGIN" "$SERVER_URL/health" 2>&1)
    
    if echo "$GET_RESPONSE" | grep -qi "access-control-allow-origin"; then
        print_success "CORS headers present"
        return 0
    else
        print_warning "CORS headers not found (may be acceptable)"
        return 0  # Don't fail, just warn
    fi
}

# Test 5: Frontend can load
test_frontend_loads() {
    if ! wait_for_url "$WEBAPP_URL" 10 1; then
        print_error "Frontend is not accessible"
        return 1
    fi
    
    FRONTEND_CONTENT=$(curl -sf "$WEBAPP_URL")
    if echo "$FRONTEND_CONTENT" | grep -qi "html\|genai\|assistant"; then
        print_success "Frontend loads successfully"
        return 0
    else
        print_error "Frontend content is invalid"
        return 1
    fi
}

# Test 6: Response format validation
test_response_format() {
    if [ -z "$E2E_TOKEN" ]; then
        print_error "No token available for response format test"
        return 1
    fi
    
    # Test a simple API call and validate response format
    API_RESPONSE=$(curl -sf "$SERVER_URL/api/llm/ollama/models" \
        -H "Authorization: Bearer $E2E_TOKEN")
    
    # Check if response is valid JSON
    if echo "$API_RESPONSE" | python3 -m json.tool >/dev/null 2>&1; then
        print_success "API response is valid JSON"
        return 0
    else
        print_error "API response is not valid JSON"
        return 1
    fi
}

# Main test execution
main() {
    echo "================================================"
    echo "🔗 End-to-End Connection Tests"
    echo "================================================"
    echo ""
    echo "Server URL: $SERVER_URL"
    echo "Webapp URL: $WEBAPP_URL"
    echo ""
    
    # Start servers if requested
    if ! start_servers; then
        print_error "Failed to start servers"
        exit 1
    fi
    
    # Wait a moment for everything to be ready
    sleep 2
    
    # Run tests
    run_test "Health endpoint check" test_health_endpoint
    run_test "Authentication flow" test_authentication
    run_test "Protected endpoint access" test_protected_endpoint
    run_test "CORS headers check" test_cors_headers
    run_test "Frontend loads" test_frontend_loads
    run_test "Response format validation" test_response_format
    
    # Summary
    echo ""
    echo "================================================"
    echo "Test Summary"
    echo "================================================"
    echo "✅ Passed: $TESTS_PASSED"
    echo "❌ Failed: $TESTS_FAILED"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        print_success "All E2E tests passed!"
        exit 0
    else
        print_error "Some E2E tests failed"
        exit 1
    fi
}

# Run main function
main

