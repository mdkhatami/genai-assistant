#!/bin/bash
# Quick restart script for GenAI Assistant FastAPI Server
# This script restarts the systemd service and shows status

echo "🔄 Restarting GenAI Assistant FastAPI Server..."
echo "================================================"

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script needs to be run with sudo privileges"
    echo "Usage: sudo ./restart_server.sh"
    exit 1
fi

# Stop the service
echo "⏹️  Stopping service..."
systemctl stop genai-assistant

# Wait a moment
sleep 2

# Start the service
echo "▶️  Starting service..."
systemctl start genai-assistant

# Wait for service to fully start
sleep 3

# Check status
echo "📊 Service status:"
systemctl status genai-assistant --no-pager

# Load port from .env if it exists
if [ -f ".env" ]; then
    source .env
fi

# Get port from environment variable, default to 5000
WEB_PORT=${WEB_PORT:-${PORT:-5000}}

# Test health endpoint
echo ""
echo "🏥 Testing health endpoint..."
if curl -s http://localhost:$WEB_PORT/health > /dev/null; then
    echo "✅ Server is responding to health checks"
else
    echo "⚠️  Server might still be starting up..."
    echo "   You can check logs with: sudo journalctl -u genai-assistant -f"
fi

echo ""
echo "✅ Server restart completed!"
echo "📝 To view logs: sudo journalctl -u genai-assistant -f"
echo "🌐 Server URL: http://localhost:$WEB_PORT"
echo "================================================"
