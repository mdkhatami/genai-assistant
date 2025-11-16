// Configuration file for GenAI Assistant Standalone Client
// This file contains default settings that can be overridden by user input

const CONFIG = {
    // Connection presets for different environments
    // Ports are configurable - defaults shown, but users can override
    connectionPresets: {
        local: {
            name: 'Local Development',
            url: '',  // Empty - user will enter, or will be auto-detected from server
            description: 'Local server running on your machine (default: http://localhost:5000)'
        },
        production: {
            name: 'Production Server',
            url: '',  // Empty - user will enter their production server URL
            description: 'Production server (enter your server URL)'
        },
        docker: {
            name: 'Docker Container',
            url: '',  // Empty - user will enter, or will use WEB_PORT from env
            description: 'Server running in Docker container'
        },
        custom: {
            name: 'Custom Server',
            url: '',
            description: 'Enter your own server details'
        }
    },
    
    // Connection settings
    connection: {
        timeout: 10000,               // Connection timeout in milliseconds
        retryAttempts: 3,             // Number of retry attempts for failed requests
        autoReconnect: true,          // Automatically reconnect if connection is lost
        rememberCredentials: true,    // Remember server URL and username (password is never saved)
        autoConnect: false            // Don't auto-connect on startup for security
    },
    
    // UI settings
    ui: {
        theme: 'light',               // 'light' or 'dark'
        language: 'en',               // Language code
        autoSave: true,               // Auto-save responses
        maxResponseLength: 10000,     // Maximum length of responses to display
        showTimestamps: true,         // Show timestamps on responses
        showConnectionPresets: true   // Show connection presets in modal
    },
    
    // API settings
    api: {
        maxFileSize: 16 * 1024 * 1024, // Maximum file size for uploads (16MB)
        supportedAudioFormats: ['mp3', 'wav', 'm4a', 'ogg', 'flac'],
        supportedVideoFormats: ['mp4', 'avi', 'mov', 'mkv'],
        imageFormats: ['png', 'jpg', 'jpeg', 'webp']
    },
    
    // Model defaults - Updated to match current server configuration
    models: {
        llm: {
            openai: {
                default: 'gpt-4',
                available: ['gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo']
            },
            ollama: {
                default: 'llama3.3:latest',
                available: [], // Will be dynamically loaded from server
                fallback: [
                    'llama3.3:latest', 
                    'qwen3:30b', 
                    'llama3.2-vision:latest', 
                    'qwen2.5:32b', 
                    'llama2', 
                    'mistral', 
                    'codellama'
                ]
            }
        },
        image: {
            default: 'flux-dev',
            available: ['flux', 'flux-dev', 'flux-dev-8bit', 'flux-dev-4bit']
        },
        transcription: {
            default: 'base',
            available: ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
        }
    },
    
    // Feature flags
    features: {
        llm: true,
        imageGeneration: true,
        transcription: true,
        fileUpload: true,
        batchProcessing: false,
        realTimeChat: false
    },
    
    // Server configuration - Updated to match current setup
    server: {
        // NOTE: Default credentials are for development only
        // In production, users should enter their own credentials
        // These should match the ADMIN_USERNAME and ADMIN_PASSWORD from your .env file
        defaultCredentials: {
            username: 'admin',  // Change this to match your .env ADMIN_USERNAME
            password: ''        // Leave empty - users must enter their password
        },
        gpuConfiguration: {
            ollama: 1,        // GPU 1 for Ollama LLM
            transcription: 2, // GPU 2 for transcription
            imageGeneration: 3 // GPU 3 for image generation
        }
    }
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
} else {
    // Make available globally for browser
    window.GenAIConfig = CONFIG;
}
