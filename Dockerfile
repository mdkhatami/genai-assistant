# Use Python 3.12 slim image for smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash genai && \
    chown -R genai:genai /app
USER genai

# Copy requirements first for better caching
COPY --chown=genai:genai requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Add user's local bin to PATH
ENV PATH="/home/genai/.local/bin:${PATH}"

# Copy application code
COPY --chown=genai:genai . .

# Create directories for uploads and generated files
RUN mkdir -p uploads generated static

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port (can be overridden via build arg)
ARG WEB_PORT=5000
EXPOSE ${WEB_PORT}

# Health check (uses WEB_PORT environment variable at runtime)
# Note: WEB_PORT should be set as ENV variable, not just ARG
ENV WEB_PORT=${WEB_PORT:-5000}
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD sh -c "curl -f http://localhost:$$WEB_PORT/health || exit 1"

# Run the application
CMD ["python", "main.py"]
