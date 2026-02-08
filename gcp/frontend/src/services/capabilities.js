/**
 * Capabilities Service - Detects backend features based on available credentials.
 *
 * This service fetches capability information from the backend and caches it
 * for use throughout the frontend application.
 */

// Cached capabilities
let _capabilities = null;

/**
 * Fetch capabilities from the backend.
 *
 * @returns {Promise<Object>} Capabilities object with mode, features, models, limits
 */
export async function getCapabilities() {
    if (_capabilities !== null) {
        return _capabilities;
    }

    try {
        const response = await fetch('/api/capabilities/');
        if (!response.ok) {
            console.error('Failed to fetch capabilities:', response.status);
            return getDefaultCapabilities();
        }

        _capabilities = await response.json();
        console.log('Backend capabilities:', _capabilities);
        return _capabilities;
    } catch (error) {
        console.error('Error fetching capabilities:', error);
        return getDefaultCapabilities();
    }
}

/**
 * Get default (minimal) capabilities for when backend is unavailable.
 *
 * @returns {Object} Default capabilities object
 */
function getDefaultCapabilities() {
    return {
        mode: 'none',
        features: {
            chat: false,
            video_analysis: false,
            paper_rag: false,
            google_search: false
        },
        models: {
            chat: '',
            vision: ''
        },
        limits: {
            video_max_size_mb: 0,
            video_max_duration_sec: 0
        }
    };
}

/**
 * Check if a specific feature is enabled.
 *
 * @param {string} feature - Feature name (chat, video_analysis, paper_rag, google_search)
 * @returns {Promise<boolean>} Whether the feature is available
 */
export async function isFeatureEnabled(feature) {
    const caps = await getCapabilities();
    return caps?.features?.[feature] ?? false;
}

/**
 * Get the current mode (free, vertex, or none).
 *
 * @returns {Promise<string>} Current mode
 */
export async function getMode() {
    const caps = await getCapabilities();
    return caps?.mode ?? 'none';
}

/**
 * Check if running in Vertex AI mode.
 *
 * @returns {Promise<boolean>} Whether Vertex AI mode is active
 */
export async function isVertexMode() {
    const mode = await getMode();
    return mode === 'vertex';
}

/**
 * Get model names being used.
 *
 * @returns {Promise<Object>} Object with chat and vision model names
 */
export async function getModels() {
    const caps = await getCapabilities();
    return caps?.models ?? { chat: '', vision: '' };
}

/**
 * Reset cached capabilities (useful for testing or after mode change).
 */
export function resetCapabilitiesCache() {
    _capabilities = null;
}

/**
 * Get a mode badge string for display.
 *
 * @returns {Promise<string>} Badge text like "FREE" or "VERTEX AI"
 */
export async function getModeBadge() {
    const mode = await getMode();
    switch (mode) {
        case 'vertex':
            return 'VERTEX AI';
        case 'free':
            return 'FREE';
        default:
            return 'OFFLINE';
    }
}

/**
 * Get available tools based on capabilities.
 * Returns list of tool names that should be enabled in the chat.
 *
 * @returns {Promise<string[]>} Array of enabled tool names
 */
export async function getEnabledTools() {
    const caps = await getCapabilities();
    const tools = [];

    // Always available when chat is enabled
    if (caps.features.chat) {
        tools.push('load_trajectory');
        tools.push('navigate_experiment');
        tools.push('analyze_alignment');
    }

    // Vertex-only features
    if (caps.features.video_analysis) {
        tools.push('analyze_video');
    }

    if (caps.features.paper_rag) {
        tools.push('search_papers');
    }

    if (caps.features.google_search) {
        tools.push('web_search');
    }

    return tools;
}
