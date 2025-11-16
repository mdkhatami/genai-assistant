// GenAI Assistant Standalone Client
class GenAIClient {
    constructor() {
        this.serverUrl = '';
        this.authToken = '';
        this.isConnected = false;
        this.connectionModal = null;
        this.ollamaModels = []; // Cache for Ollama models
        this.ollamaServerInfo = null; // Cache for Ollama server info
        this.healthCheckInterval = null; // Periodic health check interval
        this.lastSuccessfulConnection = null; // Timestamp of last successful connection
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadSavedConnection();
        this.setupConnectionPresets();
        this.attemptAutoConnection();
    }

    async attemptAutoConnection() {
        // Check if auto-connect is enabled in config
        if (window.GenAIConfig && !window.GenAIConfig.connection.autoConnect) {
            this.showConnectionModal();
            return;
        }

        const saved = localStorage.getItem('genai_connection_settings');
        if (saved) {
            const settings = JSON.parse(saved);
            if (settings.serverUrl) {
                try {
                    // Try to ping the health endpoint to test connectivity
                    const response = await this.fetchWithRetry(
                        () => fetch(`${settings.serverUrl}/health`, {
                            method: 'GET',
                            timeout: 5000
                        }),
                        { maxRetries: 1, initialDelay: 500 }
                    );
                    
                    if (response.ok) {
                        this.showInfo('Server detected. Please enter your credentials to connect.');
                        this.showConnectionModal();
                        return;
                    }
                } catch (error) {
                    console.log('Auto-connection failed, showing connection modal');
                }
            }
        }
        
        this.showConnectionModal();
    }

    setupEventListeners() {
        // Connection modal events
        document.getElementById('connectBtn').addEventListener('click', () => this.connectToServer());
        document.getElementById('testConnectionBtn').addEventListener('click', () => this.testConnection());
        document.getElementById('togglePassword').addEventListener('click', () => this.togglePasswordVisibility());
        
        // Settings events
        document.getElementById('changeConnectionBtn').addEventListener('click', () => this.showConnectionModal());
        document.getElementById('clearSettingsBtn').addEventListener('click', () => this.clearSettings());
        
        // Service navigation
        document.querySelectorAll('[data-service]').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchService(e.target.closest('[data-service]').dataset.service);
            });
        });
        
        // LLM events
        document.getElementById('sendLlmBtn').addEventListener('click', () => this.sendLlmRequest());
        document.getElementById('llmProvider').addEventListener('change', () => this.updateLlmModels());
        
        // Add Ollama-specific event listeners if elements exist
        const loadOllamaModelsBtn = document.getElementById('loadOllamaModelsBtn');
        if (loadOllamaModelsBtn) {
            loadOllamaModelsBtn.addEventListener('click', () => this.loadOllamaModels());
        }
        
        const refreshOllamaBtn = document.getElementById('refreshOllamaBtn');
        if (refreshOllamaBtn) {
            refreshOllamaBtn.addEventListener('click', () => this.refreshOllamaInfo());
        }
        
        // Image generation events
        document.getElementById('generateImageBtn').addEventListener('click', () => this.generateImage());
        
        // Transcription events
        document.getElementById('transcribeBtn').addEventListener('click', () => this.transcribeAudio());
        document.getElementById('loadModelsBtn').addEventListener('click', () => this.loadTranscriptionModels());
        document.getElementById('clearResultsBtn').addEventListener('click', () => this.clearTranscriptionResults());
        
        // Form submission prevention
        document.getElementById('connectionForm').addEventListener('submit', (e) => e.preventDefault());
        
        // Initialize UI state - load languages on first connection
        // this.toggleFasterWhisperOptions(); // Removed as no longer needed
    }

    setupConnectionPresets() {
        const presetsContainer = document.getElementById('connectionPresets');
        if (!presetsContainer || !window.GenAIConfig) return;

        const presets = window.GenAIConfig.connectionPresets;
        presetsContainer.innerHTML = '';

        Object.entries(presets).forEach(([key, preset]) => {
            const presetCard = document.createElement('div');
            presetCard.className = 'col-md-6 mb-2';
            presetCard.innerHTML = `
                <div class="card h-100 preset-card" data-preset="${key}">
                    <div class="card-body">
                        <h6 class="card-title">${preset.name}</h6>
                        <p class="card-text small text-muted">${preset.description}</p>
                        <small class="text-muted">${preset.url}</small>
                        <div class="d-grid">
                            <button class="btn btn-outline-primary btn-sm preset-btn" data-preset="${key}">
                                <i class="fas fa-bolt"></i> Use ${preset.name}
                            </button>
                        </div>
                    </div>
                </div>
            `;
            presetsContainer.appendChild(presetCard);
        });

        // Add event listeners to preset buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const presetKey = e.target.dataset.preset;
                this.applyPreset(presetKey);
            });
        });
    }

    applyPreset(presetKey) {
        const presets = window.GenAIConfig.connectionPresets;
        const preset = presets[presetKey];
        
        if (preset) {
            document.getElementById('serverUrl').value = preset.url;
            if (presetKey === 'custom') {
                document.getElementById('serverUrl').focus();
            }
        }
    }

    /**
     * Get user-friendly error message from technical error
     * @param {Error|string} error - The error object or message
     * @param {number} statusCode - HTTP status code if available
     * @returns {string} - User-friendly error message
     */
    getUserFriendlyError(error, statusCode = null) {
        const errorMessage = typeof error === 'string' ? error : error.message || 'An unknown error occurred';
        const status = statusCode || (error.response && error.response.status);
        
        // Network errors
        if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError') || errorMessage.includes('Network request failed')) {
            return {
                message: 'Unable to connect to the server. Please check:',
                suggestions: [
                    'Ensure the server is running',
                    'Verify the server URL is correct',
                    'Check your internet connection',
                    'Try again in a few moments'
                ]
            };
        }
        
        // HTTP status code errors
        if (status === 401) {
            return {
                message: 'Your session has expired or credentials are invalid.',
                suggestions: [
                    'Please reconnect to the server',
                    'Check your username and password',
                    'Your session may have timed out'
                ]
            };
        }
        
        if (status === 403) {
            return {
                message: 'Access denied. You don\'t have permission to perform this action.',
                suggestions: [
                    'Check your account permissions',
                    'Contact your administrator if you believe this is an error'
                ]
            };
        }
        
        if (status === 404) {
            return {
                message: 'The requested resource was not found.',
                suggestions: [
                    'Check the server URL',
                    'Verify the endpoint exists',
                    'The server may have been updated'
                ]
            };
        }
        
        if (status === 413) {
            return {
                message: 'File size is too large.',
                suggestions: [
                    'Try uploading a smaller file',
                    'Maximum file size is 16MB',
                    'Compress or split large files'
                ]
            };
        }
        
        if (status === 429) {
            return {
                message: 'Too many requests. Please slow down.',
                suggestions: [
                    'Wait a few moments before trying again',
                    'Reduce the frequency of your requests'
                ]
            };
        }
        
        if (status >= 500) {
            return {
                message: 'The server encountered an error processing your request.',
                suggestions: [
                    'This may be a temporary issue - try again in a moment',
                    'Check if the server is experiencing high load',
                    'If the problem persists, contact the administrator',
                    'Verify your API keys are configured correctly'
                ]
            };
        }
        
        // API-specific errors
        if (errorMessage.includes('API key') || errorMessage.includes('OPENAI_API_KEY')) {
            return {
                message: 'API key is missing or invalid.',
                suggestions: [
                    'Check your API key configuration',
                    'Verify the API key in your .env file',
                    'Ensure the API key has the required permissions'
                ]
            };
        }
        
        if (errorMessage.includes('transcription') || errorMessage.includes('audio')) {
            return {
                message: 'There was a problem processing your audio file.',
                suggestions: [
                    'Check that the file format is supported',
                    'Ensure the file is not corrupted',
                    'Try a different audio format',
                    'Verify the file size is within limits'
                ]
            };
        }
        
        if (errorMessage.includes('image generation') || errorMessage.includes('image')) {
            return {
                message: 'Image generation failed.',
                suggestions: [
                    'Check your prompt is valid',
                    'Try reducing image resolution or number of images',
                    'Verify GPU resources are available',
                    'Try again in a moment if the server is busy'
                ]
            };
        }
        
        // Default fallback
        return {
            message: 'An error occurred while processing your request.',
            suggestions: [
                'Please try again',
                'If the problem persists, check the technical details below',
                'Contact support if you need assistance'
            ],
            technicalDetails: errorMessage
        };
    }

    /**
     * Retry wrapper for fetch requests with exponential backoff
     * @param {Function} fetchFn - Function that returns a fetch promise
     * @param {Object} options - Retry options
     * @param {number} options.maxRetries - Maximum number of retries (default: 3)
     * @param {number} options.initialDelay - Initial delay in ms (default: 1000)
     * @param {number} options.maxDelay - Maximum delay in ms (default: 10000)
     * @param {Function} options.shouldRetry - Function to determine if error should be retried
     * @returns {Promise} - Fetch response
     */
    async fetchWithRetry(fetchFn, options = {}) {
        const {
            maxRetries = 3,
            initialDelay = 1000,
            maxDelay = 10000,
            shouldRetry = (error, response) => {
                // Retry on network errors or 5xx status codes
                if (error) return true;
                if (response && response.status >= 500) return true;
                return false;
            }
        } = options;

        let lastError = null;
        let lastResponse = null;

        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                const response = await fetchFn();
                
                // Check if we should retry based on response status
                if (shouldRetry(null, response)) {
                    if (attempt < maxRetries) {
                        const delay = Math.min(initialDelay * Math.pow(2, attempt), maxDelay);
                        console.log(`Request failed, retrying in ${delay}ms (attempt ${attempt + 1}/${maxRetries})...`);
                        await new Promise(resolve => setTimeout(resolve, delay));
                        continue;
                    }
                }
                
                return response;
            } catch (error) {
                lastError = error;
                
                // Check if we should retry based on error
                if (shouldRetry(error, null)) {
                    if (attempt < maxRetries) {
                        const delay = Math.min(initialDelay * Math.pow(2, attempt), maxDelay);
                        console.log(`Request failed with error: ${error.message}, retrying in ${delay}ms (attempt ${attempt + 1}/${maxRetries})...`);
                        await new Promise(resolve => setTimeout(resolve, delay));
                        continue;
                    }
                }
                
                // Don't retry for this error
                throw error;
            }
        }

        // If we exhausted retries, throw the last error or return last response
        if (lastError) {
            throw lastError;
        }
        return lastResponse;
    }

    togglePasswordVisibility() {
        const passwordInput = document.getElementById('password');
        const toggleBtn = document.getElementById('togglePassword');
        const icon = toggleBtn.querySelector('i');
        
        if (passwordInput.type === 'password') {
            passwordInput.type = 'text';
            icon.className = 'fas fa-eye-slash';
        } else {
            passwordInput.type = 'password';
            icon.className = 'fas fa-eye';
        }
    }

    async testConnection() {
        const serverUrl = this.getServerUrl();
        if (!serverUrl) {
            this.showConnectionStatus('Please enter a server URL', 'warning');
            return;
        }

        this.showConnectionStatus('Testing connection...', 'info');
        document.getElementById('testConnectionBtn').disabled = true;

        try {
            const response = await this.fetchWithRetry(
                () => fetch(`${serverUrl}/health`, {
                    method: 'GET',
                    timeout: 5000
                }),
                { maxRetries: 2, initialDelay: 500 }
            );

            if (response.ok) {
                const data = await response.json();
                this.showConnectionStatus(`✅ Server is reachable! Status: ${data.status}`, 'success');
            } else {
                this.showConnectionStatus(`❌ Server responded with status: ${response.status}`, 'danger');
            }
        } catch (error) {
            this.showConnectionStatus(`❌ Connection failed: ${error.message}`, 'danger');
        } finally {
            document.getElementById('testConnectionBtn').disabled = false;
        }
    }

    getServerUrl() {
        let serverUrl = document.getElementById('serverUrl').value.trim();
        const port = document.getElementById('serverPort').value.trim();
        
        if (!serverUrl) return '';
        
        // Add protocol if missing
        if (!serverUrl.startsWith('http://') && !serverUrl.startsWith('https://')) {
            serverUrl = 'http://' + serverUrl;
        }
        
        // Add port if provided
        if (port) {
            const url = new URL(serverUrl);
            url.port = port;
            serverUrl = url.toString();
        }
        
        return serverUrl;
    }

    showConnectionStatus(message, type = 'info') {
        const statusDiv = document.getElementById('connectionStatus');
        const statusText = document.getElementById('connectionStatusText');
        
        statusDiv.className = `alert alert-${type}`;
        statusText.textContent = message;
        statusDiv.style.display = 'block';
        
        // Auto-hide after 5 seconds for success/info messages
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        }
    }

    showConnectionModal() {
        this.connectionModal = new bootstrap.Modal(document.getElementById('connectionModal'));
        this.connectionModal.show();
    }

    async connectToServer() {
        const serverUrl = this.getServerUrl();
        const username = document.getElementById('username').value.trim();
        const password = document.getElementById('password').value.trim();
        const rememberConnection = document.getElementById('rememberConnection').checked;
        const testConnection = document.getElementById('testConnection').checked;

        if (!serverUrl || !username || !password) {
            this.showError('Please fill in all fields');
            return;
        }

        // Test connection first if requested
        if (testConnection) {
            this.showConnectionStatus('Testing connection before login...', 'info');
            try {
                const response = await this.fetchWithRetry(
                    () => fetch(`${serverUrl}/health`, {
                        method: 'GET',
                        timeout: 5000
                    }),
                    { maxRetries: 1, initialDelay: 500 }
                );
                
                if (!response.ok) {
                    this.showConnectionStatus(`❌ Server test failed: ${response.status}`, 'danger');
                    return;
                }
            } catch (error) {
                this.showConnectionStatus(`❌ Connection test failed: ${error.message}`, 'danger');
                return;
            }
        }

        this.showLoading(true);
        this.updateConnectionStatus('connecting');

        try {
            // Test connection and authenticate
            const response = await this.fetchWithRetry(
                () => fetch(`${serverUrl}/auth/login`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ username, password })
                }),
                {
                    maxRetries: 2,
                    initialDelay: 1000,
                    shouldRetry: (error, response) => {
                        // Don't retry on 401/403 (auth errors)
                        if (response && (response.status === 401 || response.status === 403)) {
                            return false;
                        }
                        // Retry on network errors or 5xx
                        return error || (response && response.status >= 500);
                    }
                }
            );

            if (!response.ok) {
                throw new Error(`Authentication failed: ${response.status}`);
            }

            const data = await response.json();
            this.authToken = data.access_token;
            this.serverUrl = serverUrl;
            this.isConnected = true;

            // Save connection settings if requested
            if (rememberConnection) {
                this.saveConnectionSettings(serverUrl, username);
            }

            this.connectionModal.hide();
            this.showMainApp();
            this.updateConnectionStatus('connected');
            this.lastSuccessfulConnection = new Date();
            this.startHealthCheck();
            this.showSuccess('Successfully connected to server');
            
            // Initialize LLM models after successful connection
            this.updateLlmModels();

        } catch (error) {
            console.error('Connection error:', error);
            let errorMessage = 'Connection failed';
            
            if (error.message.includes('Failed to fetch')) {
                errorMessage = 'Unable to connect to server. Please check the URL and ensure the server is running.';
            } else if (error.message.includes('401') || error.message.includes('Authentication failed')) {
                errorMessage = 'Invalid username or password. Please check your credentials.';
            } else if (error.message.includes('403')) {
                errorMessage = 'Access forbidden. Please check your credentials.';
            } else if (error.message.includes('404')) {
                errorMessage = 'Server not found. Please check the URL and port.';
            } else if (error.message.includes('500')) {
                errorMessage = 'Server error. Please try again later or contact the administrator.';
            } else {
                errorMessage = `Connection failed: ${error.message}`;
            }
            
            this.showError(errorMessage);
            this.updateConnectionStatus('disconnected');
        } finally {
            this.showLoading(false);
        }
    }

    async sendLlmRequest() {
        if (!this.isConnected) {
            this.showError('Not connected to server');
            return;
        }

        const provider = document.getElementById('llmProvider').value;
        const model = document.getElementById('llmModel').value;
        const prompt = document.getElementById('llmPrompt').value.trim();

        if (!prompt) {
            this.showError('Please enter a prompt');
            return;
        }

        this.showLoading(true);
        const responseDiv = document.getElementById('llmResponse');
        responseDiv.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Processing...</div>';

        try {
            const endpoint = provider === 'openai' ? '/api/llm/openai' : '/api/llm/ollama';
            const response = await this.fetchWithRetry(
                () => fetch(`${this.serverUrl}${endpoint}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.authToken}`
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        model: model
                    })
                }),
                {
                    maxRetries: 2,
                    shouldRetry: (error, response) => {
                        // Don't retry on 401 (auth errors)
                        if (response && response.status === 401) return false;
                        // Retry on network errors or 5xx
                        return error || (response && response.status >= 500);
                    }
                }
            );

            if (!response.ok) {
                const errorText = await response.text();
                console.error('API Error Response:', errorText);
                throw new Error(`Request failed: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            
            // Handle different response formats
            let responseContent = data.content || data.response || data.message || 'No response content';
            let metadata = '';
            
            if (data.model) {
                metadata += `<strong>Model:</strong> ${data.model}<br>`;
            }
            if (data.tokens_used) {
                metadata += `<strong>Tokens:</strong> ${data.tokens_used}<br>`;
            }
            if (data.response_time) {
                metadata += `<strong>Response Time:</strong> ${data.response_time.toFixed(2)}s<br>`;
            }
            
            responseDiv.innerHTML = `
                <div class="response-container success">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h6 class="mb-0"><i class="fas fa-robot"></i> Response:</h6>
                        ${provider === 'ollama' ? '<span class="badge bg-success">🦙 Ollama</span>' : '<span class="badge bg-primary">🤖 OpenAI</span>'}
                    </div>
                    <div class="llm-response mb-2">${responseContent}</div>
                    ${metadata ? `<div class="metadata"><small class="text-muted">${metadata}</small></div>` : ''}
                </div>
            `;

        } catch (error) {
            console.error('LLM request error:', error);
            const statusCode = error.response?.status || (error.message.match(/\d{3}/)?.[0] ? parseInt(error.message.match(/\d{3}/)[0]) : null);
            const friendlyError = this.getUserFriendlyError(error, statusCode);
            
            let suggestionsHtml = '';
            if (friendlyError.suggestions && friendlyError.suggestions.length > 0) {
                suggestionsHtml = `
                    <div class="mt-3">
                        <strong>Suggestions:</strong>
                        <ul class="mb-0">
                            ${friendlyError.suggestions.map(s => `<li>${s}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            const technicalDetails = friendlyError.technicalDetails || error.message;
            
            responseDiv.innerHTML = `
                <div class="response-container error">
                    <h6><i class="fas fa-exclamation-triangle"></i> Error:</h6>
                    <div>${friendlyError.message}</div>
                    ${suggestionsHtml}
                    <div class="mt-2">
                        <details>
                            <summary class="text-muted" style="cursor: pointer;">Technical details</summary>
                            <small class="text-muted">${technicalDetails}</small>
                        </details>
                    </div>
                </div>
            `;
        } finally {
            this.showLoading(false);
        }
    }

    async generateImage() {
        if (!this.isConnected) {
            this.showError('Not connected to server');
            return;
        }

        const model = document.getElementById('imageModel').value;
        const prompt = document.getElementById('imagePrompt').value.trim();
        const resolution = document.getElementById('imageResolution').value;
        const steps = parseInt(document.getElementById('imageSteps').value);
        const guidance = parseFloat(document.getElementById('imageGuidance').value);
        const numImages = parseInt(document.getElementById('imageCount').value);

        if (!prompt) {
            this.showError('Please enter a prompt');
            return;
        }

        this.showLoading(true);
        const resultDiv = document.getElementById('imageResult');
        resultDiv.innerHTML = `<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Generating ${numImages} image${numImages > 1 ? 's' : ''}...</div>`;

        try {
            const [width, height] = resolution.split('x').map(Number);
            
            const response = await this.fetchWithRetry(
                () => fetch(`${this.serverUrl}/api/image/generate`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.authToken}`
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        model: model,
                        width: width,
                        height: height,
                        steps: steps,
                        guidance_scale: guidance,
                        num_images: numImages
                    })
                }),
                {
                    maxRetries: 1, // Image generation is slow, only retry once
                    initialDelay: 2000
                }
            );

            if (!response.ok) {
                throw new Error(`Image generation failed: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.images && data.images.length > 0) {
                let imagesHtml = '<div class="row">';
                
                data.images.forEach((image, index) => {
                    const colClass = numImages === 1 ? 'col-12' : 
                                   numImages <= 2 ? 'col-md-6' : 
                                   numImages <= 4 ? 'col-md-6 col-lg-3' : 'col-md-4 col-lg-2';
                    
                    imagesHtml += `
                        <div class="${colClass} mb-3">
                            <div class="text-center">
                                <img src="${image.image_data}" alt="Generated image ${index + 1}" class="generated-image img-fluid">
                                <div class="mt-2">
                                    <small class="text-muted">
                                        Image ${index + 1}
                                        ${image.generation_time ? ` • ${image.generation_time.toFixed(2)}s` : ''}
                                    </small>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                imagesHtml += '</div>';
                imagesHtml += `
                    <div class="text-center mt-3">
                        <small class="text-muted">
                            Total processing time: ${data.processing_time ? data.processing_time.toFixed(2) : 'N/A'}s
                            • ${data.images.length} image${data.images.length > 1 ? 's' : ''} generated
                        </small>
                    </div>
                `;
                
                resultDiv.innerHTML = imagesHtml;
            } else {
                throw new Error('No images generated');
            }

        } catch (error) {
            console.error('Image generation error:', error);
            const statusCode = error.response?.status || (error.message.match(/\d{3}/)?.[0] ? parseInt(error.message.match(/\d{3}/)[0]) : null);
            const friendlyError = this.getUserFriendlyError(error, statusCode);
            
            let suggestionsHtml = '';
            if (friendlyError.suggestions && friendlyError.suggestions.length > 0) {
                suggestionsHtml = `
                    <div class="mt-3">
                        <strong>Suggestions:</strong>
                        <ul class="mb-0">
                            ${friendlyError.suggestions.map(s => `<li>${s}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            const technicalDetails = friendlyError.technicalDetails || error.message;
            let suggestionsHtml = '';
            if (friendlyError.suggestions && friendlyError.suggestions.length > 0) {
                suggestionsHtml = `
                    <div class="mt-3">
                        <strong>Suggestions:</strong>
                        <ul class="mb-0">
                            ${friendlyError.suggestions.map(s => `<li>${s}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            resultDiv.innerHTML = `
                <div class="alert alert-danger">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        <div>
                            <strong>Image Generation Failed</strong><br>
                            <span>${friendlyError.message}</span>
                        </div>
                    </div>
                    ${suggestionsHtml}
                    <div class="mt-2">
                        <details>
                            <summary class="text-muted" style="cursor: pointer;">Technical details</summary>
                            <small class="text-muted">${technicalDetails}</small>
                        </details>
                    </div>
                </div>
            `;
        } finally {
            this.showLoading(false);
        }
    }

    async loadTranscriptionModels() {
        if (!this.isConnected) {
            this.showError('Not connected to server');
            return;
        }

        try {
            const response = await this.fetchWithRetry(
                () => fetch(`${this.serverUrl}/api/transcribe/models`, {
                    headers: {
                        'Authorization': `Bearer ${this.authToken}`
                    }
                })
            );

            if (!response.ok) {
                throw new Error(`Failed to load models: ${response.status}`);
            }

            const data = await response.json();
            this.populateLanguageDropdowns(data.supported_languages);
            this.showInfo('Available languages loaded successfully');
            
        } catch (error) {
            console.error('Error loading models:', error);
            this.showError(`Failed to load models: ${error.message}`);
        }
    }

    populateLanguageDropdowns(languages) {
        const fasterLanguageSelect = document.getElementById('fasterLanguage');
        const whisperLanguageSelect = document.getElementById('whisperLanguage');
        
        // Get language names with flags
        const getLanguageOption = (code) => {
            const names = {
                'auto': '🌍 Auto-detect',
                'en': '🇺🇸 English', 'es': '🇪🇸 Spanish', 'fr': '🇫🇷 French', 'de': '🇩🇪 German', 'it': '🇮🇹 Italian',
                'pt': '🇵🇹 Portuguese', 'ru': '🇷🇺 Russian', 'ja': '🇯🇵 Japanese', 'ko': '🇰🇷 Korean', 'zh': '🇨🇳 Chinese',
                'ar': '🇸🇦 Arabic', 'hi': '🇮🇳 Hindi', 'nl': '🇳🇱 Dutch', 'sv': '🇸🇪 Swedish', 'pl': '🇵🇱 Polish',
                'tr': '🇹🇷 Turkish', 'ca': '🏴󠁥󠁳󠁣󠁴󠁿 Catalan', 'fi': '🇫🇮 Finnish', 'vi': '🇻🇳 Vietnamese', 'he': '🇮🇱 Hebrew',
                'uk': '🇺🇦 Ukrainian', 'el': '🇬🇷 Greek', 'cs': '🇨🇿 Czech', 'ro': '🇷🇴 Romanian', 'da': '🇩🇰 Danish',
                'hu': '🇭🇺 Hungarian', 'ta': '🇮🇳 Tamil', 'no': '🇳🇴 Norwegian', 'th': '🇹🇭 Thai', 'ur': '🇵🇰 Urdu'
            };
            return names[code] || `🌐 ${code.toUpperCase()}`;
        };

        // Clear and populate both dropdowns
        [fasterLanguageSelect, whisperLanguageSelect].forEach(select => {
            const currentValue = select.value;
            
            // Keep only the first 13 common languages
            while (select.children.length > 13) {
                select.removeChild(select.lastChild);
            }
            
            // Add all other supported languages
            const commonLanguages = ['auto', 'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi'];
            languages.forEach(lang => {
                if (!commonLanguages.includes(lang)) {
                    const option = document.createElement('option');
                    option.value = lang;
                    option.textContent = getLanguageOption(lang);
                    select.appendChild(option);
                }
            });
            
            // Restore selection
            select.value = currentValue;
        });
    }

    getCurrentModelType() {
        const activeTab = document.querySelector('#transcriptionTabs .nav-link.active');
        return activeTab.id === 'faster-whisper-tab' ? 'faster-whisper' : 'whisper';
    }

    getTranscriptionParameters() {
        const modelType = this.getCurrentModelType();
        const fileInput = document.getElementById('audioFile');
        const file = fileInput.files[0];

        if (!file) {
            throw new Error('Please select an audio file');
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('model_type', modelType);

        if (modelType === 'faster-whisper') {
            formData.append('model_name', document.getElementById('fasterModelName').value);
            formData.append('language', document.getElementById('fasterLanguage').value);
            formData.append('task', document.getElementById('fasterTask').value);
            formData.append('condition_on_previous_text', document.getElementById('fasterConditionOnPreviousText').checked);
            formData.append('word_timestamps', document.getElementById('fasterWordTimestamps').checked);
            formData.append('device', document.getElementById('fasterDevice').value);
            formData.append('gpu_index', document.getElementById('fasterGpuIndex').value);
            formData.append('compute_type', document.getElementById('fasterComputeType').value);
            formData.append('cpu_threads', document.getElementById('fasterCpuThreads').value);
            formData.append('num_workers', document.getElementById('fasterNumWorkers').value);
            formData.append('beam_size', document.getElementById('fasterBeamSize').value);
            formData.append('temperature', document.getElementById('fasterTemperature').value);
            formData.append('vad_filter', document.getElementById('fasterVadFilter').value === 'true');
            
            // Add additional Faster-Whisper parameters if they exist
            const bestOf = document.getElementById('fasterBestOf');
            if (bestOf) formData.append('best_of', bestOf.value);
            
            const patience = document.getElementById('fasterPatience');
            if (patience) formData.append('patience', patience.value);
            
            const lengthPenalty = document.getElementById('fasterLengthPenalty');
            if (lengthPenalty) formData.append('length_penalty', lengthPenalty.value);
            
            const repetitionPenalty = document.getElementById('fasterRepetitionPenalty');
            if (repetitionPenalty) formData.append('repetition_penalty', repetitionPenalty.value);
            
            const initialPrompt = document.getElementById('fasterInitialPrompt').value;
            if (initialPrompt) formData.append('initial_prompt', initialPrompt);
        } else {
            formData.append('model_name', document.getElementById('whisperModelName').value);
            formData.append('language', document.getElementById('whisperLanguage').value);
            formData.append('task', document.getElementById('whisperTask').value);
            formData.append('condition_on_previous_text', document.getElementById('whisperConditionOnPreviousText').checked);
            formData.append('word_timestamps', document.getElementById('whisperWordTimestamps').checked);
            formData.append('device', document.getElementById('whisperDevice').value);
            formData.append('gpu_index', document.getElementById('whisperGpuIndex').value);
            
            const initialPrompt = document.getElementById('whisperInitialPrompt').value;
            if (initialPrompt) formData.append('initial_prompt', initialPrompt);
        }

        return formData;
    }

    async transcribeAudio() {
        if (!this.isConnected) {
            this.showError('Not connected to server');
            return;
        }

        try {
            const formData = this.getTranscriptionParameters();
            const modelType = this.getCurrentModelType();

            this.showLoading(true);
            const resultDiv = document.getElementById('transcriptionResult');
            resultDiv.innerHTML = `
                <div class="text-center p-4">
                    <div class="spinner-border text-primary mb-3" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <h5>Transcribing with ${modelType === 'faster-whisper' ? 'Faster-Whisper' : 'OpenAI Whisper'}...</h5>
                    <p class="text-muted">This may take a few moments depending on audio length and model size.</p>
                </div>
            `;

            const response = await this.fetchWithRetry(
                () => fetch(`${this.serverUrl}/api/transcribe`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.authToken}`
                    },
                    body: formData
                }),
                {
                    maxRetries: 1, // Transcription is slow, only retry once
                    initialDelay: 2000
                }
            );

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Transcription failed: ${response.status}`);
            }

            const data = await response.json();
            
            resultDiv.innerHTML = `
                <div class="card border-success">
                    <div class="card-header bg-success text-white">
                        <div class="d-flex justify-content-between align-items-center">
                            <h6 class="mb-0"><i class="fas fa-check-circle"></i> Transcription Complete</h6>
                            <div class="badge bg-light text-dark">
                                ${data.processing_time.toFixed(2)}s
                            </div>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="row mb-3">
                            <div class="col-md-4">
                                <small class="text-muted">Model:</small><br>
                                <strong>${data.model}</strong>
                            </div>
                            <div class="col-md-4">
                                <small class="text-muted">Language:</small><br>
                                <strong>${data.language}</strong>
                            </div>
                            <div class="col-md-4">
                                <small class="text-muted">Processing Time:</small><br>
                                <strong>${data.processing_time.toFixed(2)}s</strong>
                            </div>
                        </div>
                        <hr>
                        <div class="transcription-text">
                            <h6><i class="fas fa-quote-left"></i> Transcribed Text:</h6>
                            <div class="p-3 bg-light rounded border">
                                <p class="mb-0" style="white-space: pre-wrap; line-height: 1.6;">
                                    ${data.text || 'No text was transcribed from the audio.'}
                                </p>
                            </div>
                        </div>
                        <div class="mt-3 text-end">
                            <button class="btn btn-outline-primary btn-sm" onclick="navigator.clipboard.writeText('${data.text.replace(/'/g, "\\'")}')">
                                <i class="fas fa-copy"></i> Copy Text
                            </button>
                        </div>
                    </div>
                </div>
            `;

        } catch (error) {
            console.error('Transcription error:', error);
            const resultDiv = document.getElementById('transcriptionResult');
            const statusCode = error.response?.status || (error.message.match(/\d{3}/)?.[0] ? parseInt(error.message.match(/\d{3}/)[0]) : null);
            const friendlyError = this.getUserFriendlyError(error, statusCode);
            
            let suggestionsHtml = '';
            if (friendlyError.suggestions && friendlyError.suggestions.length > 0) {
                suggestionsHtml = `
                    <div class="mt-3">
                        <strong>Suggestions:</strong>
                        <ul class="mb-0">
                            ${friendlyError.suggestions.map(s => `<li>${s}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            const technicalDetails = friendlyError.technicalDetails || error.message;
            
            resultDiv.innerHTML = `
                <div class="card border-danger">
                    <div class="card-header bg-danger text-white">
                        <h6 class="mb-0"><i class="fas fa-exclamation-triangle"></i> Transcription Failed</h6>
                    </div>
                    <div class="card-body">
                        <p class="text-danger mb-0">${friendlyError.message}</p>
                        ${suggestionsHtml}
                        <div class="mt-2">
                            <details>
                                <summary class="text-muted" style="cursor: pointer;">Technical details</summary>
                                <small class="text-muted">${technicalDetails}</small>
                            </details>
                        </div>
                    </div>
                </div>
            `;
        } finally {
            this.showLoading(false);
        }
    }

    clearTranscriptionResults() {
        document.getElementById('transcriptionResult').innerHTML = '';
        this.showInfo('Results cleared');
    }

    switchService(service) {
        // Hide all service content
        document.querySelectorAll('.service-content').forEach(content => {
            content.style.display = 'none';
        });

        // Show selected service
        document.getElementById(`${service}Service`).style.display = 'block';

        // Update navigation
        document.querySelectorAll('[data-service]').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-service="${service}"]`).classList.add('active');
    }

    async updateLlmModels() {
        const provider = document.getElementById('llmProvider').value;
        const modelSelect = document.getElementById('llmModel');
        const ollamaInfo = document.getElementById('ollamaInfo');
        
        modelSelect.innerHTML = '<option value="">Loading...</option>';
        
        if (provider === 'openai') {
            // OpenAI models are static
            modelSelect.innerHTML = `
                <option value="gpt-4">GPT-4</option>
                <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                <option value="gpt-4-turbo">GPT-4 Turbo</option>
            `;
            
            // Hide Ollama info
            if (ollamaInfo) {
                ollamaInfo.style.display = 'none';
            }
        } else if (provider === 'ollama') {
            // Show Ollama info section
            if (ollamaInfo) {
                ollamaInfo.style.display = 'block';
            }
            
            // Load Ollama models dynamically
            await this.loadOllamaModels();
        }
    }

    async loadOllamaModels() {
        if (!this.isConnected) {
            this.showError('Not connected to server');
            return;
        }

        const modelSelect = document.getElementById('llmModel');
        const ollamaStatusDiv = document.getElementById('ollamaStatus');
        
        try {
            modelSelect.innerHTML = '<option value="">Loading models...</option>';
            
            if (ollamaStatusDiv) {
                ollamaStatusDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading Ollama information...';
            }

            // Get Ollama models from the API
            const response = await this.fetchWithRetry(
                () => fetch(`${this.serverUrl}/api/llm/ollama/models`, {
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${this.authToken}`
                    }
                })
            );

            if (!response.ok) {
                throw new Error(`Failed to fetch Ollama models: ${response.status}`);
            }

            const data = await response.json();
            this.ollamaModels = data.models || [];
            this.ollamaServerInfo = data;

            // Update the model dropdown
            if (this.ollamaModels.length > 0) {
                modelSelect.innerHTML = this.ollamaModels.map(model => 
                    `<option value="${model.name}">${model.name} (${model.size_gb} GB)</option>`
                ).join('');
                
                // Select the default model if available
                const defaultModel = window.GenAIConfig.models.llm.ollama.default;
                const defaultOption = modelSelect.querySelector(`option[value="${defaultModel}"]`);
                if (defaultOption) {
                    defaultOption.selected = true;
                }
            } else {
                // Use fallback models if no models are loaded
                const fallbackModels = window.GenAIConfig.models.llm.ollama.fallback;
                modelSelect.innerHTML = fallbackModels.map(model => 
                    `<option value="${model}">${model} (Not loaded)</option>`
                ).join('');
            }

            // Update status display
            this.updateOllamaStatus(data);

        } catch (error) {
            console.error('Error loading Ollama models:', error);
            
            // Fallback to default models
            const fallbackModels = window.GenAIConfig.models.llm.ollama.fallback;
            modelSelect.innerHTML = fallbackModels.map(model => 
                `<option value="${model}">${model} (Fallback)</option>`
            ).join('');
            
            if (ollamaStatusDiv) {
                ollamaStatusDiv.innerHTML = `
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle"></i>
                        Unable to load Ollama models: ${error.message}
                    </div>
                `;
            }
        }
    }

    async refreshOllamaInfo() {
        await this.loadOllamaModels();
        this.showInfo('Ollama information refreshed');
    }

    updateOllamaStatus(serverInfo) {
        const ollamaStatusDiv = document.getElementById('ollamaStatus');
        if (!ollamaStatusDiv) return;

        const isConnected = serverInfo.success !== false;
        const modelCount = serverInfo.total_models || 0;
        const serverUrl = serverInfo.server_url || 'Unknown';
        
        let statusHtml = '';
        
        if (isConnected) {
            statusHtml = `
                <div class="alert alert-success">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <i class="fas fa-check-circle"></i>
                            <strong>Ollama Server Connected</strong>
                        </div>
                        <button type="button" class="btn btn-sm btn-outline-success" id="refreshOllamaBtn">
                            <i class="fas fa-sync-alt"></i> Refresh
                        </button>
                    </div>
                    <div class="mt-2">
                        <small>
                            <strong>Server:</strong> ${serverUrl}<br>
                            <strong>Available Models:</strong> ${modelCount}<br>
                            <strong>Status:</strong> ${serverInfo.status || 'Connected'}
                        </small>
                    </div>
                </div>
            `;
            
            if (modelCount === 0) {
                statusHtml += `
                    <div class="alert alert-warning">
                        <i class="fas fa-info-circle"></i>
                        <strong>No models found.</strong> You may need to pull models using Ollama CLI:
                        <br><code>ollama pull llama3.3</code>
                    </div>
                `;
            }
        } else {
            statusHtml = `
                <div class="alert alert-danger">
                    <i class="fas fa-times-circle"></i>
                    <strong>Ollama Server Disconnected</strong>
                    <div class="mt-2">
                        <small>Error: ${serverInfo.error || 'Unknown error'}</small>
                    </div>
                </div>
            `;
        }

        ollamaStatusDiv.innerHTML = statusHtml;
        
        // Re-attach event listener for refresh button
        const refreshBtn = document.getElementById('refreshOllamaBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshOllamaInfo());
        }
    }

    showMainApp() {
        document.getElementById('mainApp').style.display = 'block';
    }

    showLoading(show) {
        document.getElementById('loadingSpinner').style.display = show ? 'flex' : 'none';
    }

    /**
     * Start periodic health check to monitor connection status
     */
    startHealthCheck() {
        // Clear any existing interval
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
        }
        
        // Check health every 30 seconds
        this.healthCheckInterval = setInterval(async () => {
            if (!this.isConnected || !this.serverUrl) {
                return;
            }
            
            try {
                const response = await this.fetchWithRetry(
                    () => fetch(`${this.serverUrl}/health`, {
                        method: 'GET',
                        timeout: 3000
                    }),
                    { maxRetries: 1, initialDelay: 500 }
                );
                
                if (response.ok) {
                    this.lastSuccessfulConnection = new Date();
                    this.updateConnectionStatus('connected');
                } else {
                    this.updateConnectionStatus('disconnected');
                }
            } catch (error) {
                this.updateConnectionStatus('disconnected');
            }
        }, 30000); // Check every 30 seconds
    }
    
    /**
     * Stop periodic health check
     */
    stopHealthCheck() {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
            this.healthCheckInterval = null;
        }
    }

    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connectionStatus');
        if (!statusElement) return;
        
        const icon = statusElement.querySelector('i');
        if (!icon) return;
        
        icon.className = 'fas fa-circle';
        icon.classList.remove('text-success', 'text-danger', 'text-warning');
        
        let statusText = '';
        let tooltip = '';
        
        switch (status) {
            case 'connected':
                icon.classList.add('text-success');
                statusText = 'Connected';
                if (this.lastSuccessfulConnection) {
                    const timeAgo = Math.floor((new Date() - this.lastSuccessfulConnection) / 1000);
                    tooltip = `Last successful connection: ${timeAgo < 60 ? `${timeAgo}s ago` : `${Math.floor(timeAgo / 60)}m ago`}`;
                }
                break;
            case 'connecting':
                icon.classList.add('text-warning');
                statusText = 'Connecting...';
                break;
            case 'disconnected':
                icon.classList.add('text-danger');
                statusText = 'Disconnected';
                tooltip = 'Lost connection to server. Please reconnect.';
                break;
            default:
                icon.classList.add('text-warning');
                statusText = 'Unknown';
        }
        
        statusElement.innerHTML = `<i class="fas fa-circle ${icon.classList.toString().replace('fas fa-circle ', '')}"></i> ${statusText}`;
        if (tooltip) {
            statusElement.setAttribute('title', tooltip);
        }
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showInfo(message) {
        this.showNotification(message, 'info');
    }

    showNotification(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        let alertClass = 'success';
        if (type === 'error') alertClass = 'danger';
        else if (type === 'info') alertClass = 'info';
        
        notification.className = `alert alert-${alertClass} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    saveConnectionSettings(serverUrl, username) {
        const settings = { serverUrl, username };
        localStorage.setItem('genai_connection_settings', JSON.stringify(settings));
    }

    loadSavedConnection() {
        const saved = localStorage.getItem('genai_connection_settings');
        if (saved) {
            const settings = JSON.parse(saved);
            
            // Parse URL to separate host and port
            if (settings.serverUrl) {
                try {
                    const url = new URL(settings.serverUrl);
                    document.getElementById('serverUrl').value = `${url.protocol}//${url.hostname}`;
                    if (url.port) {
                        document.getElementById('serverPort').value = url.port;
                    }
                } catch (e) {
                    // If URL parsing fails, just use the saved URL as is
                    document.getElementById('serverUrl').value = settings.serverUrl;
                }
            }
            
            document.getElementById('username').value = settings.username || '';
        } else {
            // Set default username if no saved settings
            if (window.GenAIConfig && window.GenAIConfig.server && window.GenAIConfig.server.defaultCredentials) {
                document.getElementById('username').value = window.GenAIConfig.server.defaultCredentials.username || '';
            }
        }
    }

    clearSettings() {
        localStorage.removeItem('genai_connection_settings');
        this.stopHealthCheck();
        this.isConnected = false;
        this.serverUrl = '';
        this.authToken = '';
        this.showSuccess('Settings cleared');
        this.showConnectionModal();
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new GenAIClient();
});
