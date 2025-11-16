#!/bin/bash
# GenAI Assistant - Production Deployment Script
# This script sets up the GenAI Assistant for production with systemd and nginx

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
SERVICE_NAME="genai-assistant"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
NGINX_CONF_DIR="/etc/nginx"
NGINX_SITE_CONF="${NGINX_CONF_DIR}/sites-available/${SERVICE_NAME}"
NGINX_SITE_ENABLED="${NGINX_CONF_DIR}/sites-enabled/${SERVICE_NAME}"

# Load port configuration from .env if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

# Get ports from environment variables with defaults
SERVER_PORT=${WEB_PORT:-${PORT:-5000}}
NGINX_HTTP_PORT=${NGINX_HTTP_PORT:-80}
NGINX_HTTPS_PORT=${NGINX_HTTPS_PORT:-443}

# Change to project root
cd "$PROJECT_ROOT"

# Print header
echo "================================================"
echo "🚀 GenAI Assistant - Production Deployment"
echo "================================================"
echo ""

# Function to check root privileges
check_root_privileges() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script must be run as root or with sudo"
        print_info "Usage: sudo $0"
        exit 1
    fi
    print_success "Running with root privileges"
}

# Function to check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    if ! check_python_version; then
        print_error "Python 3.8+ is required"
        exit 1
    fi
    
    if ! check_command nginx; then
        print_warning "nginx is not installed"
        print_info "Installing nginx..."
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update
            apt-get install -y nginx
        elif command -v yum >/dev/null 2>&1; then
            yum install -y nginx
        else
            print_error "Could not install nginx automatically. Please install it manually."
            exit 1
        fi
    fi
    
    print_success "All prerequisites met"
}

# Function to setup virtual environment and dependencies
setup_application() {
    print_step "Setting up application..."
    
    local venv_dir="$PROJECT_ROOT/venv"
    
    if [ ! -d "$venv_dir" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv "$venv_dir"
    else
        print_info "Virtual environment already exists"
    fi
    
    print_info "Installing dependencies..."
    source "$venv_dir/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet -r "$PROJECT_ROOT/requirements.txt"
    deactivate
    
    print_success "Application setup complete"
}

# Function to setup .env file
setup_env_file() {
    print_step "Checking .env file..."
    
    if ! create_env_from_template "$PROJECT_ROOT"; then
        print_warning "Could not create .env file automatically"
    fi
    
    if [ -f "$PROJECT_ROOT/.env" ]; then
        print_success ".env file exists"
        print_warning "Please ensure .env file has correct production values"
    else
        print_error ".env file not found and could not be created"
        exit 1
    fi
}

# Function to setup systemd service
setup_systemd_service() {
    print_step "Setting up systemd service..."
    
    print_info "Creating systemd service file..."
    
    # Create system user if it doesn't exist
    if ! id "genai" &>/dev/null; then
        print_info "Creating system user 'genai'..."
        useradd --system --create-home --shell /bin/bash genai
    fi
    
    # Set ownership
    chown -R genai:genai "$PROJECT_ROOT"
    
    # Create service file
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=GenAI Assistant FastAPI
After=network.target

[Service]
Type=simple
User=genai
Group=genai
WorkingDirectory=$PROJECT_ROOT
Environment=PATH=$PROJECT_ROOT/venv/bin
EnvironmentFile=$PROJECT_ROOT/.env
ExecStart=$PROJECT_ROOT/venv/bin/python $PROJECT_ROOT/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd
    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"
    
    print_success "Systemd service created and enabled"
}

# Function to configure nginx
configure_nginx() {
    print_step "Configuring nginx..."
    
    # Check if nginx config exists in project
    local nginx_project_conf="$PROJECT_ROOT/nginx/nginx.conf"
    
    if [ -f "$nginx_project_conf" ]; then
        print_info "Using nginx configuration from project..."
        
        # Create sites-available directory if it doesn't exist
        mkdir -p "${NGINX_CONF_DIR}/sites-available"
        mkdir -p "${NGINX_CONF_DIR}/sites-enabled"
        
        # Copy and adapt nginx config
        # Note: The project nginx.conf is a full config, we need to adapt it
        # For now, create a simple reverse proxy config
        cat > "$NGINX_SITE_CONF" << NGINX_EOF
server {
    listen ${NGINX_HTTP_PORT};
    server_name _;

    # Logging
    access_log /var/log/nginx/genai-assistant-access.log;
    error_log /var/log/nginx/genai-assistant-error.log;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;

    # Main location
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://127.0.0.1:${SERVER_PORT};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
        
        # File upload size
        client_max_body_size 50M;
    }

    # Auth endpoints with stricter rate limiting
    location /auth/ {
        limit_req zone=login burst=5 nodelay;
        
        proxy_pass http://127.0.0.1:${SERVER_PORT};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:${SERVER_PORT};
        access_log off;
    }
}
NGINX_EOF
        
    else
        print_error "nginx.conf not found in project"
        exit 1
    fi
    
    # Enable site
    if [ -L "$NGINX_SITE_ENABLED" ]; then
        print_info "Site already enabled"
    else
        ln -s "$NGINX_SITE_CONF" "$NGINX_SITE_ENABLED"
    fi
    
    # Test nginx configuration
    print_info "Testing nginx configuration..."
    if nginx -t; then
        print_success "Nginx configuration is valid"
    else
        print_error "Nginx configuration test failed"
        exit 1
    fi
    
    print_success "Nginx configured"
}

# Function to setup SSL (optional)
setup_ssl() {
    print_step "SSL Certificate Setup"
    
    read -p "Do you want to setup SSL/HTTPS? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping SSL setup"
        return 0
    fi
    
    print_info "SSL setup options:"
    echo "1. Let's Encrypt (certbot) - Recommended for production"
    echo "2. Self-signed certificate - For testing"
    echo "3. Skip SSL setup"
    read -p "Choose option (1-3): " ssl_option
    
    case $ssl_option in
        1)
            if ! check_command certbot; then
                print_info "Installing certbot..."
                if command -v apt-get >/dev/null 2>&1; then
                    apt-get install -y certbot python3-certbot-nginx
                elif command -v yum >/dev/null 2>&1; then
                    yum install -y certbot python3-certbot-nginx
                fi
            fi
            
            read -p "Enter your domain name: " domain_name
            if [ -z "$domain_name" ]; then
                print_warning "No domain name provided, skipping SSL setup"
                return 0
            fi
            
            print_info "Running certbot for $domain_name..."
            certbot --nginx -d "$domain_name" --non-interactive --agree-tos --register-unsafely-without-email || {
                print_warning "Certbot setup failed. You can run it manually later."
            }
            ;;
        2)
            print_info "Generating self-signed certificate..."
            mkdir -p "$PROJECT_ROOT/nginx/ssl"
            openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
                -keyout "$PROJECT_ROOT/nginx/ssl/key.pem" \
                -out "$PROJECT_ROOT/nginx/ssl/cert.pem" \
                -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
            
            print_warning "Self-signed certificate created. Update nginx config manually to use it."
            ;;
        *)
            print_info "Skipping SSL setup"
            ;;
    esac
}

# Function to configure firewall
configure_firewall() {
    print_step "Configuring firewall..."
    
    if check_command ufw; then
        print_info "Configuring UFW firewall..."
        ufw allow 22/tcp comment 'SSH'
        ufw allow ${NGINX_HTTP_PORT}/tcp comment 'HTTP'
        ufw allow ${NGINX_HTTPS_PORT}/tcp comment 'HTTPS'
        print_info "Firewall rules added. Run 'ufw enable' to activate if needed."
    elif check_command firewall-cmd; then
        print_info "Configuring firewalld..."
        firewall-cmd --permanent --add-service=http
        firewall-cmd --permanent --add-service=https
        firewall-cmd --reload
        print_success "Firewall configured"
    else
        print_warning "No supported firewall found (ufw or firewalld)"
        print_info "Please configure firewall manually to allow ports ${NGINX_HTTP_PORT} and ${NGINX_HTTPS_PORT}"
    fi
}

# Function to start services
start_services() {
    print_step "Starting services..."
    
    # Start systemd service
    print_info "Starting $SERVICE_NAME service..."
    systemctl start "$SERVICE_NAME"
    sleep 3
    
    # Check service status
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        print_success "Service is running"
    else
        print_error "Service failed to start"
        systemctl status "$SERVICE_NAME" --no-pager
        exit 1
    fi
    
    # Start nginx
    print_info "Starting nginx..."
    systemctl start nginx
    systemctl enable nginx
    
    if systemctl is-active --quiet nginx; then
        print_success "Nginx is running"
    else
        print_error "Nginx failed to start"
        systemctl status nginx --no-pager
        exit 1
    fi
}

# Function to verify deployment
verify_deployment() {
    print_step "Verifying deployment..."
    
    # Wait a moment for services to be ready
    sleep 5
    
    # Check service health
    if wait_for_url "http://localhost:$SERVER_PORT/health" 30 2; then
        print_success "Service health check passed"
    else
        print_warning "Service health check failed, but continuing..."
    fi
    
    # Check nginx
    if curl -sf "http://localhost/health" >/dev/null 2>&1; then
        print_success "Nginx reverse proxy is working"
    else
        print_warning "Nginx reverse proxy check failed"
    fi
}

# Main execution
main() {
    check_root_privileges
    check_prerequisites
    setup_application
    setup_env_file
    setup_systemd_service
    configure_nginx
    setup_ssl
    configure_firewall
    start_services
    verify_deployment
    
    echo ""
    echo "================================================"
    print_success "Production deployment complete!"
    echo "================================================"
    echo ""
    echo "📡 Service: $SERVICE_NAME"
    echo "🌐 Nginx: http://localhost (or your domain)"
    echo "📚 API: http://localhost/api"
    echo "📊 Health: http://localhost/health"
    echo ""
    echo "📝 Useful commands:"
    echo "   - Service status: systemctl status $SERVICE_NAME"
    echo "   - Service logs: journalctl -u $SERVICE_NAME -f"
    echo "   - Restart service: systemctl restart $SERVICE_NAME"
    echo "   - Nginx status: systemctl status nginx"
    echo "   - Nginx logs: tail -f /var/log/nginx/genai-assistant-*.log"
    echo "   - Test nginx: nginx -t"
    echo ""
    echo "🔒 Security reminders:"
    echo "   - Update .env file with secure values"
    echo "   - Configure SSL/HTTPS for production"
    echo "   - Review firewall settings"
    echo "   - Keep system and dependencies updated"
    echo ""
}

# Run main function
main

