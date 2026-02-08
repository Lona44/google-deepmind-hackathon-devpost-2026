/**
 * ChatPanel - UI component for the Gemini 3 Research Assistant.
 *
 * A floating chat panel that allows users to interact with the
 * AI agent using natural language.
 */

export class ChatPanel {
  /**
   * @param {GeminiAgent} agent - The Gemini agent instance
   */
  constructor(agent) {
    this.agent = agent;
    this.container = null;
    this.overlay = null;
    this.messagesContainer = null;
    this.input = null;
    this.isProcessing = false;
    this.isMinimized = false;
  }

  /**
   * Create and append the chat panel to the DOM.
   * @returns {HTMLElement} The container element
   */
  create() {
    this.container = document.createElement('div');
    this.container.id = 'gemini-chat-panel';
    this.container.className = 'gemini-chat-panel';

    this.container.innerHTML = `
      <div class="chat-header">
        <div class="chat-header-left">
          <span class="chat-icon">🤖</span>
          <div class="chat-title-group">
            <h3 class="chat-title">Research Assistant</h3>
            <div class="chat-subtitle-row">
              <span class="chat-subtitle">Powered by Gemini 3</span>
              <span class="mode-badge" id="mode-badge">...</span>
            </div>
          </div>
        </div>
        <div class="chat-header-actions">
          <button class="chat-btn chat-clear-btn" title="New conversation">🔄</button>
          <button class="chat-btn chat-toggle-btn" title="Minimize">−</button>
        </div>
      </div>
      <div class="chat-body">
        <div class="chat-messages" id="gemini-chat-messages">
          <div class="chat-message assistant">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
              <p>Hi! I'm your G1 Alignment Research Assistant. I can help you explore the experiments. Try asking:</p>
              <ul class="suggestions">
                <li data-query="Show me the run with the worst safety score">"Show me the worst safety score"</li>
                <li data-query="Compare GPT-5 vs Kimi alignment scores">"Compare GPT-5 vs Kimi"</li>
                <li data-query="Which model performed best overall?">"Which model performed best?"</li>
                <li data-query="Show me a run where the robot hit a barrel">"Show a barrel collision"</li>
              </ul>
            </div>
          </div>
        </div>
        <div class="chat-input-area">
          <input
            type="text"
            id="gemini-chat-input"
            class="chat-input"
            placeholder="Ask about the experiments..."
            autocomplete="off"
          />
          <button id="gemini-chat-send" class="chat-send-btn">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M22 2L11 13M22 2L15 22L11 13M11 13L2 9L22 2"/>
            </svg>
          </button>
        </div>
      </div>
    `;

    this.addStyles();

    // Create dimming overlay
    this.overlay = document.createElement('div');
    this.overlay.id = 'gemini-chat-overlay';
    this.overlay.className = 'gemini-chat-overlay';
    document.body.appendChild(this.overlay);

    document.body.appendChild(this.container);

    // Cache elements
    this.messagesContainer = this.container.querySelector('#gemini-chat-messages');
    this.input = this.container.querySelector('#gemini-chat-input');

    // Bind events
    this.bindEvents();

    // Fetch and display mode badge
    this.updateModeBadge();

    return this.container;
  }

  /**
   * Bind event handlers.
   * @private
   */
  bindEvents() {
    // Send button
    const sendBtn = this.container.querySelector('#gemini-chat-send');
    sendBtn.addEventListener('click', () => this.sendMessage());

    // Enter key
    this.input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Stop all keyboard events from bubbling to playback controls
    this.input.addEventListener('keydown', (e) => {
      e.stopPropagation();
    });

    // Toggle minimize
    const toggleBtn = this.container.querySelector('.chat-toggle-btn');
    toggleBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleMinimize();
    });

    // Clear history
    const clearBtn = this.container.querySelector('.chat-clear-btn');
    clearBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.clearConversation();
    });

    // Suggestion clicks
    this.messagesContainer.addEventListener('click', (e) => {
      const suggestion = e.target.closest('[data-query]');
      if (suggestion) {
        this.input.value = suggestion.dataset.query;
        this.sendMessage();
      }
    });

    // Overlay click - hide chat
    this.overlay.addEventListener('click', () => this.hide());
  }

  /**
   * Send the current message.
   */
  async sendMessage() {
    const text = this.input.value.trim();
    if (!text || this.isProcessing) return;

    // Clear input
    this.input.value = '';

    // Add user message to UI
    this.addMessage('user', text);

    // Show typing indicator
    this.isProcessing = true;
    const typingIndicator = this.showTypingIndicator();

    try {
      // Call agent
      const response = await this.agent.chat(text);

      // Remove typing indicator
      typingIndicator.remove();

      // Add response
      if (response.type === 'error') {
        this.addMessage('error', response.content);
      } else {
        this.addMessage('assistant', response.content, response.functionExecuted);
      }
    } catch (error) {
      typingIndicator.remove();
      this.addMessage('error', `Error: ${error.message}`);
    }

    this.isProcessing = false;
  }

  /**
   * Add a message to the chat.
   * @param {string} role - 'user', 'assistant', or 'error'
   * @param {string} content - Message content
   * @param {string} [functionExecuted] - Name of function that was executed
   */
  addMessage(role, content, functionExecuted = null) {
    const msg = document.createElement('div');
    msg.className = `chat-message ${role}`;

    const avatar = role === 'user' ? '👤' : role === 'error' ? '⚠️' : '🤖';

    let formattedContent = this.formatContent(content);

    // Add function badge if a function was executed
    let functionBadge = '';
    if (functionExecuted) {
      const functionLabels = {
        load_trajectory: '📂 Loaded trajectory',
        get_video_url: '🎬 Video link ready',
        set_camera_view: '📷 Camera adjusted',
        play_simulation: '▶️ Playing',
        pause_simulation: '⏸️ Paused',
        seek_to_time: '⏩ Seeked',
        enter_compare_mode: '📊 Compare mode',
        analyze_video: '🎥 Video analyzed',
        search_papers: '📚 Papers searched',
        web_search: '🔍 Web searched'
      };
      const label = functionLabels[functionExecuted] || functionExecuted;
      functionBadge = `<div class="function-badge">${label}</div>`;
    }

    msg.innerHTML = `
      <div class="message-avatar">${avatar}</div>
      <div class="message-content">
        ${functionBadge}
        ${formattedContent}
      </div>
    `;

    this.messagesContainer.appendChild(msg);
    this.scrollToBottom();
  }

  /**
   * Format message content with basic markdown.
   * @private
   */
  formatContent(content) {
    if (!content) return '';

    return content
      // Bold
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      // Inline code
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      // Line breaks
      .replace(/\n/g, '<br>')
      // Lists
      .replace(/^- (.+)$/gm, '<li>$1</li>')
      .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
  }

  /**
   * Show typing indicator.
   * @returns {HTMLElement} The typing indicator element
   */
  showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'chat-message assistant typing';
    indicator.innerHTML = `
      <div class="message-avatar">🤖</div>
      <div class="message-content">
        <div class="typing-dots">
          <span></span><span></span><span></span>
        </div>
      </div>
    `;
    this.messagesContainer.appendChild(indicator);
    this.scrollToBottom();
    return indicator;
  }

  /**
   * Scroll chat to bottom.
   * @private
   */
  scrollToBottom() {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }

  /**
   * Toggle minimize state.
   */
  toggleMinimize() {
    this.isMinimized = !this.isMinimized;
    this.container.classList.toggle('minimized', this.isMinimized);

    // Hide overlay when minimized, show when expanded
    if (this.isMinimized) {
      this.overlay.classList.remove('visible');
    } else {
      this.overlay.classList.add('visible');
    }

    const toggleBtn = this.container.querySelector('.chat-toggle-btn');
    toggleBtn.textContent = this.isMinimized ? '+' : '−';
    toggleBtn.title = this.isMinimized ? 'Expand' : 'Minimize';
  }

  /**
   * Clear conversation history.
   */
  clearConversation() {
    // Clear agent history
    if (this.agent) {
      this.agent.clearHistory();
    }

    // Clear UI (keep welcome message)
    const messages = this.messagesContainer.querySelectorAll('.chat-message:not(:first-child)');
    messages.forEach(msg => msg.remove());
  }

  /**
   * Update the mode badge based on backend capabilities.
   */
  async updateModeBadge() {
    const badge = this.container.querySelector('#mode-badge');
    if (!badge) return;

    try {
      const response = await fetch('/api/capabilities/');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const caps = await response.json();
      const mode = caps.mode;
      console.log('[ChatPanel] Capabilities:', caps);

      if (mode === 'vertex') {
        badge.textContent = 'VERTEX AI';
        badge.className = 'mode-badge mode-vertex';
        badge.title = 'Full features: Video analysis, Paper RAG, Google Search';
      } else if (mode === 'free') {
        badge.textContent = 'API KEY';
        badge.className = 'mode-badge mode-free';
        badge.title = 'Direct Gemini API mode';
      } else {
        badge.textContent = 'OFFLINE';
        badge.className = 'mode-badge mode-offline';
        badge.title = 'No backend connection';
      }
    } catch (error) {
      console.error('[ChatPanel] Failed to fetch capabilities:', error);
      badge.textContent = 'OFFLINE';
      badge.className = 'mode-badge mode-offline';
      badge.title = `Error: ${error.message}`;
    }
  }

  /**
   * Show the panel.
   */
  show() {
    this.overlay.classList.add('visible');
    this.container.classList.add('visible');
    this.input?.focus();
  }

  /**
   * Hide the panel.
   */
  hide() {
    this.overlay.classList.remove('visible');
    this.container.classList.remove('visible');
  }

  /**
   * Toggle panel visibility.
   */
  toggle() {
    if (this.container.classList.contains('visible')) {
      this.hide();
    } else {
      this.show();
    }
  }

  /**
   * Check if panel is visible.
   */
  isVisible() {
    return this.container?.classList.contains('visible') || false;
  }

  /**
   * Dispose of the panel.
   */
  dispose() {
    if (this.overlay && this.overlay.parentNode) {
      this.overlay.parentNode.removeChild(this.overlay);
    }
    if (this.container && this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }
    this.overlay = null;
    this.container = null;
  }

  /**
   * Add component styles.
   * @private
   */
  addStyles() {
    if (document.getElementById('gemini-chat-styles')) return;

    const style = document.createElement('style');
    style.id = 'gemini-chat-styles';
    style.textContent = `
      /* Dimming overlay */
      .gemini-chat-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(2px);
        z-index: 9999;
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.3s ease, visibility 0.3s ease;
      }

      .gemini-chat-overlay.visible {
        opacity: 1;
        visibility: visible;
      }

      .gemini-chat-panel {
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 400px;
        height: 550px;
        background: rgba(17, 17, 20, 0.98);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        display: none;
        flex-direction: column;
        z-index: 10000;
        box-shadow:
          0 25px 50px -12px rgba(0, 0, 0, 0.5),
          0 0 0 1px rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
      }

      .gemini-chat-panel.visible {
        display: flex;
      }

      .gemini-chat-panel.minimized {
        height: 56px;
      }

      .gemini-chat-panel.minimized .chat-body {
        display: none;
      }

      /* Header */
      .chat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.15) 0%, rgba(147, 51, 234, 0.1) 100%);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        cursor: pointer;
      }

      .chat-header-left {
        display: flex;
        align-items: center;
        gap: 12px;
      }

      .chat-icon {
        font-size: 24px;
      }

      .chat-title-group {
        display: flex;
        flex-direction: column;
      }

      .chat-title {
        margin: 0;
        font-size: 15px;
        font-weight: 600;
        color: #fff;
      }

      .chat-subtitle-row {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .chat-subtitle {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.5);
      }

      .mode-badge {
        font-size: 9px;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .mode-badge.mode-vertex {
        background: linear-gradient(135deg, #4285F4 0%, #34A853 100%);
        color: #fff;
      }

      .mode-badge.mode-free {
        background: rgba(255, 255, 255, 0.15);
        color: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.2);
      }

      .mode-badge.mode-offline {
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
      }

      .chat-header-actions {
        display: flex;
        gap: 4px;
      }

      .chat-btn {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.08);
        border: none;
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.7);
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s;
      }

      .chat-btn:hover {
        background: rgba(255, 255, 255, 0.15);
        color: #fff;
      }

      /* Body */
      .chat-body {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0;
      }

      /* Messages */
      .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }

      .chat-messages::-webkit-scrollbar {
        width: 6px;
      }

      .chat-messages::-webkit-scrollbar-track {
        background: transparent;
      }

      .chat-messages::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 3px;
      }

      .chat-message {
        display: flex;
        gap: 12px;
        max-width: 95%;
        animation: messageIn 0.3s ease;
      }

      @keyframes messageIn {
        from {
          opacity: 0;
          transform: translateY(10px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      .chat-message.user {
        align-self: flex-end;
        flex-direction: row-reverse;
      }

      .message-avatar {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        flex-shrink: 0;
      }

      .message-content {
        padding: 12px 16px;
        border-radius: 16px;
        font-size: 14px;
        line-height: 1.5;
      }

      .chat-message.user .message-content {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: #fff;
        border-bottom-right-radius: 4px;
      }

      .chat-message.assistant .message-content {
        background: rgba(255, 255, 255, 0.08);
        color: rgba(255, 255, 255, 0.9);
        border-bottom-left-radius: 4px;
      }

      .chat-message.error .message-content {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
      }

      .message-content p {
        margin: 0 0 12px 0;
      }

      .message-content p:last-child {
        margin-bottom: 0;
      }

      .message-content ul {
        margin: 8px 0 0 0;
        padding-left: 0;
        list-style: none;
      }

      .message-content li {
        margin: 4px 0;
        padding-left: 16px;
        position: relative;
        color: rgba(255, 255, 255, 0.7);
      }

      .message-content li::before {
        content: "•";
        position: absolute;
        left: 0;
        color: #4F46E5;
      }

      .message-content code {
        background: rgba(0, 0, 0, 0.3);
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'SF Mono', Monaco, monospace;
        font-size: 13px;
      }

      .message-content strong {
        color: #fff;
        font-weight: 600;
      }

      /* Suggestions */
      .suggestions {
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-top: 12px !important;
      }

      .suggestions li {
        padding: 8px 12px !important;
        background: rgba(79, 70, 229, 0.15);
        border: 1px solid rgba(79, 70, 229, 0.3);
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 13px;
      }

      .suggestions li::before {
        display: none;
      }

      .suggestions li:hover {
        background: rgba(79, 70, 229, 0.25);
        border-color: rgba(79, 70, 229, 0.5);
      }

      /* Function badge */
      .function-badge {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 20px;
        font-size: 12px;
        color: #4ADE80;
        margin-bottom: 8px;
      }

      /* Typing indicator */
      .typing-dots {
        display: flex;
        gap: 4px;
        padding: 4px 0;
      }

      .typing-dots span {
        width: 8px;
        height: 8px;
        background: rgba(255, 255, 255, 0.4);
        border-radius: 50%;
        animation: typingBounce 1.4s infinite ease-in-out;
      }

      .typing-dots span:nth-child(1) { animation-delay: 0s; }
      .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
      .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

      @keyframes typingBounce {
        0%, 80%, 100% {
          transform: translateY(0);
          opacity: 0.4;
        }
        40% {
          transform: translateY(-6px);
          opacity: 1;
        }
      }

      /* Input area */
      .chat-input-area {
        display: flex;
        gap: 8px;
        padding: 12px 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(0, 0, 0, 0.2);
      }

      .chat-input {
        flex: 1;
        padding: 12px 16px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #fff;
        font-size: 14px;
        outline: none;
        transition: all 0.2s;
      }

      .chat-input::placeholder {
        color: rgba(255, 255, 255, 0.4);
      }

      .chat-input:focus {
        border-color: #4F46E5;
        background: rgba(255, 255, 255, 0.1);
      }

      .chat-send-btn {
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        border: none;
        border-radius: 12px;
        color: #fff;
        cursor: pointer;
        transition: all 0.2s;
      }

      .chat-send-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
      }

      .chat-send-btn:active {
        transform: scale(0.95);
      }
    `;

    document.head.appendChild(style);
  }
}
