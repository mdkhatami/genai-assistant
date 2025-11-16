#!/bin/bash
# Master test runner for deployment tests
# Runs tests in priority order: Tier 1 → Tier 2 → Tier 3

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
ENV_FILE="$PROJECT_ROOT/.env"
STOP_ON_FAILURE=${STOP_ON_FAILURE:-true}
TEST_REPORT="$PROJECT_ROOT/test_report.txt"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Parse arguments
TIER=""
CUSTOM_ENV_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --tier)
            TIER="$2"
            shift 2
            ;;
        --env-file)
            CUSTOM_ENV_FILE="$2"
            shift 2
            ;;
        --no-stop-on-failure)
            STOP_ON_FAILURE=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tier N              Run only tier N (1, 2, or 3)"
            echo "  --env-file PATH       Use custom .env file"
            echo "  --no-stop-on-failure  Continue running tests after failure"
            echo "  --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run all tests"
            echo "  $0 --tier 1                  # Run only startup validation"
            echo "  $0 --tier 2                  # Run only connectivity tests"
            echo "  $0 --env-file /path/to/.env # Use custom .env file"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Use custom env file if provided
if [ -n "$CUSTOM_ENV_FILE" ]; then
    ENV_FILE="$CUSTOM_ENV_FILE"
fi

# Initialize test report
echo "================================================" > "$TEST_REPORT"
echo "Deployment Test Report" >> "$TEST_REPORT"
echo "Generated: $TIMESTAMP" >> "$TEST_REPORT"
echo "================================================" >> "$TEST_REPORT"
echo "" >> "$TEST_REPORT"

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Function to run a test and track results
run_test_suite() {
    local tier_name="$1"
    local test_name="$2"
    local test_command="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo "" >> "$TEST_REPORT"
    echo "--- $tier_name: $test_name ---" >> "$TEST_REPORT"
    echo "Command: $test_command" >> "$TEST_REPORT"
    
    print_step "Running: $tier_name - $test_name"
    
    if eval "$test_command" >> "$TEST_REPORT" 2>&1; then
        print_success "$test_name passed"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo "Result: PASSED" >> "$TEST_REPORT"
        return 0
    else
        print_error "$test_name failed"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo "Result: FAILED" >> "$TEST_REPORT"
        
        if [ "$STOP_ON_FAILURE" = "true" ]; then
            echo ""
            print_error "Stopping tests due to failure (use --no-stop-on-failure to continue)"
            return 1
        fi
        return 0
    fi
}

# Function to run Python test module
run_python_test() {
    local tier_name="$1"
    local test_name="$2"
    local test_module="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo "" >> "$TEST_REPORT"
    echo "--- $tier_name: $test_name ---" >> "$TEST_REPORT"
    echo "Module: $test_module" >> "$TEST_REPORT"
    
    print_step "Running: $tier_name - $test_name"
    
    # Activate venv if it exists
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    fi
    
    if python3 -m pytest "$SCRIPT_DIR/$test_module" -v >> "$TEST_REPORT" 2>&1; then
        print_success "$test_name passed"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo "Result: PASSED" >> "$TEST_REPORT"
        return 0
    else
        # Try unittest if pytest is not available
        if python3 -m unittest "$test_module" >> "$TEST_REPORT" 2>&1; then
            print_success "$test_name passed"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            echo "Result: PASSED" >> "$TEST_REPORT"
            return 0
        else
            print_error "$test_name failed"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            echo "Result: FAILED" >> "$TEST_REPORT"
            
            if [ "$STOP_ON_FAILURE" = "true" ]; then
                echo ""
                print_error "Stopping tests due to failure (use --no-stop-on-failure to continue)"
                return 1
            fi
            return 0
        fi
    fi
}

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    print_error ".env file not found at $ENV_FILE"
    print_info "Please create .env file or use --env-file to specify a different location"
    exit 1
fi

# Main execution
main() {
    echo "================================================"
    echo "🧪 Deployment Test Suite"
    echo "================================================"
    echo ""
    echo "Environment file: $ENV_FILE"
    echo "Stop on failure: $STOP_ON_FAILURE"
    echo "Test report: $TEST_REPORT"
    echo ""
    
    # Tier 1: Startup Validation
    if [ -z "$TIER" ] || [ "$TIER" = "1" ]; then
        echo "================================================"
        echo "Tier 1: Startup Validation"
        echo "================================================"
        echo ""
        
        run_python_test "Tier 1" "Configuration Validation" "test_config_validation.py" || exit 1
        
        # Check if we should run local startup tests
        if command -v python3 >/dev/null 2>&1; then
            run_test_suite "Tier 1" "Local Setup Validation" \
                "\"$SCRIPT_DIR/test_startup_local.sh\"" || exit 1
        else
            print_warning "Skipping local setup tests (Python 3 not found)"
            SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
        fi
        
        # Check if Docker is available
        if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
            run_test_suite "Tier 1" "Docker Setup Validation" \
                "\"$SCRIPT_DIR/test_startup_docker.sh\"" || exit 1
        else
            print_warning "Skipping Docker setup tests (Docker not available)"
            SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
        fi
    fi
    
    # Tier 2: Frontend-Backend Connectivity
    if [ -z "$TIER" ] || [ "$TIER" = "2" ]; then
        echo ""
        echo "================================================"
        echo "Tier 2: Frontend-Backend Connectivity"
        echo "================================================"
        echo ""
        
        # Check if server is running
        SERVER_PORT=${WEB_PORT:-${PORT:-5000}}
        if ! curl -sf "http://localhost:$SERVER_PORT/health" >/dev/null 2>&1; then
            print_warning "Server is not running. Some connectivity tests may fail."
            print_info "Start the server before running Tier 2 tests for best results."
        fi
        
        run_python_test "Tier 2" "API Connectivity" "test_api_connectivity.py" || exit 1
        run_python_test "Tier 2" "Frontend Configuration" "test_frontend_config.py" || exit 1
        run_test_suite "Tier 2" "End-to-End Connection" \
            "\"$SCRIPT_DIR/test_e2e_connection.sh\"" || exit 1
    fi
    
    # Tier 3: Functional Tests
    if [ -z "$TIER" ] || [ "$TIER" = "3" ]; then
        echo ""
        echo "================================================"
        echo "Tier 3: Functional Tests"
        echo "================================================"
        echo ""
        
        # Check if server is running
        SERVER_PORT=${WEB_PORT:-${PORT:-5000}}
        if ! curl -sf "http://localhost:$SERVER_PORT/health" >/dev/null 2>&1; then
            print_error "Server is not running. Functional tests require a running server."
            print_info "Start the server before running Tier 3 tests."
            exit 1
        fi
        
        run_python_test "Tier 3" "Service Health" "test_service_health.py" || exit 1
        run_python_test "Tier 3" "Integration Tests" "test_integration.py" || exit 1
    fi
    
    # Summary
    echo ""
    echo "================================================"
    echo "Test Summary"
    echo "================================================"
    echo "✅ Passed: $PASSED_TESTS"
    echo "❌ Failed: $FAILED_TESTS"
    echo "⏭️  Skipped: $SKIPPED_TESTS"
    echo "📊 Total: $TOTAL_TESTS"
    echo ""
    echo "Test report saved to: $TEST_REPORT"
    echo ""
    
    # Add summary to report
    echo "" >> "$TEST_REPORT"
    echo "================================================" >> "$TEST_REPORT"
    echo "Summary" >> "$TEST_REPORT"
    echo "================================================" >> "$TEST_REPORT"
    echo "Passed: $PASSED_TESTS" >> "$TEST_REPORT"
    echo "Failed: $FAILED_TESTS" >> "$TEST_REPORT"
    echo "Skipped: $SKIPPED_TESTS" >> "$TEST_REPORT"
    echo "Total: $TOTAL_TESTS" >> "$TEST_REPORT"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        print_success "All tests passed!"
        exit 0
    else
        print_error "$FAILED_TESTS test(s) failed"
        exit 1
    fi
}

# Run main function
main

