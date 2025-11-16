# Deployment Testing Guide

This directory contains a comprehensive test suite for validating the GenAI Assistant system before deployment. The tests are organized into three tiers, prioritizing startup validation and connectivity before functional tests.

## Test Structure

### Tier 1: Startup Validation (Highest Priority)

Validates that the system can be started successfully in both local and Docker environments.

- **`test_config_validation.py`** - Validates `.env` configuration file
  - Checks required environment variables
  - Validates port numbers, API keys, CORS settings
  - Detects port conflicts

- **`test_startup_local.sh`** - Local setup validation
  - Prerequisites check (Python, pip, venv)
  - Virtual environment setup
  - Dependencies installation
  - Server startup and health check
  - Webapp server startup

- **`test_startup_docker.sh`** - Docker setup validation
  - Docker prerequisites check
  - Docker image build
  - Container startup
  - Health endpoint verification
  - Container logs check

### Tier 2: Frontend-Backend Connectivity (High Priority)

Ensures frontend can communicate with backend reliably, especially after configuration changes.

- **`test_api_connectivity.py`** - API connectivity tests
  - Health endpoint accessibility
  - Authentication flow
  - CORS headers validation
  - Protected endpoint access
  - Port configuration changes

- **`test_frontend_config.py`** - Frontend configuration tests
  - Config file existence and readability
  - CORS configuration matching
  - Port configuration consistency
  - Frontend-backend URL construction

- **`test_e2e_connection.sh`** - End-to-end connection test
  - Full flow: health → auth → API call
  - Response format validation
  - Multiple port configurations

### Tier 3: Functional Tests (Secondary Priority)

Verifies core functionality works end-to-end using actual API calls.

- **`test_service_health.py`** - Service health endpoint tests
  - Health endpoint structure
  - Component status reporting
  - Overall system status

- **`test_integration.py`** - Integration test suite
  - LLM API tests (OpenAI and Ollama)
  - Image generation API tests
  - Transcription API tests
  - Uses actual API keys from `.env` (not mocks)

### Helper Utilities

- **`test_helpers.py`** - Common test utilities
  - Environment variable loading
  - Port checking
  - URL validation
  - Authentication helpers
  - CORS validation

### Master Test Runner

- **`run_deployment_tests.sh`** - Master test runner
  - Runs tests in priority order (Tier 1 → Tier 2 → Tier 3)
  - Supports running individual tiers
  - Generates test reports
  - Configurable stop-on-failure behavior

## Usage

### Prerequisites

1. Ensure `.env` file exists and is properly configured
2. For Tier 2 and Tier 3 tests, the server should be running (or use `START_SERVERS=true` for E2E tests)

### Running All Tests

```bash
# Run all tests in order
./tests/run_deployment_tests.sh

# Run with custom .env file
./tests/run_deployment_tests.sh --env-file /path/to/.env

# Continue running after failures
./tests/run_deployment_tests.sh --no-stop-on-failure
```

### Running Individual Tiers

```bash
# Run only startup validation (Tier 1)
./tests/run_deployment_tests.sh --tier 1

# Run only connectivity tests (Tier 2)
./tests/run_deployment_tests.sh --tier 2

# Run only functional tests (Tier 3)
./tests/run_deployment_tests.sh --tier 3
```

### Running Individual Test Files

```bash
# Python tests
python3 -m unittest tests.test_config_validation
python3 -m pytest tests/test_api_connectivity.py -v

# Shell script tests
./tests/test_startup_local.sh
./tests/test_startup_docker.sh
./tests/test_e2e_connection.sh START_SERVERS=true
```

## Test Execution Flow

1. **Tier 1: Startup Validation**
   - Validates configuration
   - Tests local setup
   - Tests Docker setup
   - **Stops here if any test fails** (unless `--no-stop-on-failure`)

2. **Tier 2: Connectivity**
   - Tests API connectivity
   - Tests frontend configuration
   - Tests end-to-end connection
   - **Requires server to be running** (or use E2E script with `START_SERVERS=true`)

3. **Tier 3: Functional Tests**
   - Tests service health
   - Tests integration with real API calls
   - **Requires server to be running**

## Test Reports

Test results are saved to `test_report.txt` in the project root. The report includes:
- Test execution timestamps
- Command output
- Pass/fail status
- Summary statistics

## Configuration

Tests read configuration from the `.env` file in the project root. Key variables:

- `JWT_SECRET_KEY` - Required, must be non-default
- `ADMIN_USERNAME` - Required
- `ADMIN_PASSWORD` - Required, must be non-default
- `WEB_PORT` / `PORT` - Server port (default: 5000)
- `WEBAPP_PORT` - Webapp port (default: 8080)
- `OPENAI_API_KEY` - For OpenAI LLM tests
- `HUGGINGFACE_TOKEN` - For image generation tests
- `CORS_ORIGINS` - CORS configuration

## Success Criteria

- **Tier 1**: All startup scripts complete without errors
- **Tier 2**: All connectivity tests pass with default and modified port configurations
- **Tier 3**: At least one functional test per service passes (graceful degradation if services unavailable)

## Notes

- Tests use actual API keys from `.env` (per project requirements, no mocks)
- Tests are designed to be runnable in CI/CD pipelines
- Tests provide clear error messages for debugging
- Tests clean up after themselves (stop servers, remove containers)

## Troubleshooting

### Server Not Running

If Tier 2 or Tier 3 tests fail because the server is not running:

```bash
# Start server manually, then run tests
./setup_local.sh &
# Wait for server to start, then:
./tests/run_deployment_tests.sh --tier 2

# Or use E2E test with auto-start
START_SERVERS=true ./tests/test_e2e_connection.sh
```

### Port Conflicts

If tests fail due to port conflicts:

1. Check what's using the ports: `lsof -i :5000` or `netstat -an | grep 5000`
2. Stop conflicting services or change ports in `.env`
3. Re-run tests

### Docker Tests Failing

If Docker tests fail:

1. Ensure Docker daemon is running: `docker info`
2. Check Docker Compose is available: `docker compose version`
3. Review container logs: `docker compose logs`

### Authentication Failures

If authentication tests fail:

1. Verify `ADMIN_USERNAME` and `ADMIN_PASSWORD` are set in `.env`
2. Ensure passwords are not default values
3. Check server logs for authentication errors

## Integration with CI/CD

The test suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run Deployment Tests
  run: |
    ./tests/run_deployment_tests.sh --tier 1
    # Start server
    ./setup_local.sh &
    sleep 30
    # Run connectivity and functional tests
    ./tests/run_deployment_tests.sh --tier 2
    ./tests/run_deployment_tests.sh --tier 3
```

## Related Files

- `scripts/helpers.sh` - Shared helper functions used by test scripts
- `.env_example` - Example environment configuration
- `setup_local.sh` - Local development setup script
- `setup_docker.sh` - Docker setup script

