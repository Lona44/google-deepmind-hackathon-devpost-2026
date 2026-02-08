"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  MessageSquare,
  Minimize2,
  Maximize2,
  X,
  Send,
  RefreshCw,
  Bot,
  User,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/useAppStore";

interface ChatPanelProps {
  className?: string;
}

/**
 * Suggestion button component
 */
interface SuggestionButtonProps {
  label: string;
  onClick: () => void;
}

function SuggestionButton({ label, onClick }: SuggestionButtonProps) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-sm text-white/70 hover:bg-white/10 hover:text-white hover:border-white/20 transition-all"
    >
      {label}
    </button>
  );
}

/**
 * Chat message component
 */
interface MessageProps {
  role: "user" | "assistant" | "error";
  content: string;
  toolCalls?: Array<{ name: string; result?: string }>;
}

function Message({ role, content, toolCalls }: MessageProps) {
  const icons = {
    user: <User className="w-4 h-4" />,
    assistant: <Bot className="w-4 h-4" />,
    error: <AlertCircle className="w-4 h-4" />,
  };

  const colors = {
    user: "bg-[var(--color-accent-secondary)]/20 border-[var(--color-accent-secondary)]/30",
    assistant: "bg-white/5 border-white/10",
    error: "bg-[var(--color-accent-danger)]/20 border-[var(--color-accent-danger)]/30",
  };

  return (
    <div className={cn("flex gap-3 p-3 rounded-lg border", colors[role])}>
      <div
        className={cn(
          "w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0",
          role === "user" && "bg-[var(--color-accent-secondary)]/30",
          role === "assistant" && "bg-white/10",
          role === "error" && "bg-[var(--color-accent-danger)]/30"
        )}
      >
        {icons[role]}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-white/90 whitespace-pre-wrap break-words">
          {content}
        </p>
        {toolCalls && toolCalls.length > 0 && (
          <div className="mt-2 space-y-1">
            {toolCalls.map((call, i) => (
              <div
                key={i}
                className="text-xs bg-black/20 px-2 py-1 rounded inline-flex items-center gap-1.5"
              >
                <span className="text-[var(--color-accent-primary)]">
                  {call.name}
                </span>
                {call.result && (
                  <span className="text-white/50 truncate max-w-[200px]">
                    {call.result}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Typing indicator animation
 */
function TypingIndicator() {
  return (
    <div className="flex gap-1 px-3 py-2">
      <span className="w-2 h-2 rounded-full bg-white/30 animate-bounce [animation-delay:0ms]" />
      <span className="w-2 h-2 rounded-full bg-white/30 animate-bounce [animation-delay:150ms]" />
      <span className="w-2 h-2 rounded-full bg-white/30 animate-bounce [animation-delay:300ms]" />
    </div>
  );
}

/**
 * Backend mode badge
 */
interface ModeBadgeProps {
  mode: "vertex" | "api_key" | "offline";
}

function ModeBadge({ mode }: ModeBadgeProps) {
  const labels = {
    vertex: "VERTEX AI",
    api_key: "API KEY",
    offline: "OFFLINE",
  };

  const colors = {
    vertex: "bg-[var(--color-accent-primary)]/20 text-[var(--color-accent-primary)]",
    api_key: "bg-[var(--color-accent-secondary)]/20 text-[var(--color-accent-secondary)]",
    offline: "bg-white/10 text-white/50",
  };

  return (
    <span
      className={cn(
        "px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide",
        colors[mode]
      )}
    >
      {labels[mode]}
    </span>
  );
}

/**
 * Chat panel component - AI research assistant interface
 */
export function ChatPanel({ className }: ChatPanelProps) {
  const {
    isChatPanelOpen,
    isChatMinimized,
    chatMessages,
    chatBackendMode,
    toggleChatPanel,
    toggleChatMinimized,
    addChatMessage,
    clearChatMessages,
  } = useAppStore();

  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  // Focus input when panel opens
  useEffect(() => {
    if (isChatPanelOpen && !isChatMinimized) {
      inputRef.current?.focus();
    }
  }, [isChatPanelOpen, isChatMinimized]);

  // Suggestion prompts
  const suggestions = [
    "Show worst safety runs",
    "Compare model performance",
    "What patterns do you see?",
    "Search alignment papers",
  ];

  // Handle send message
  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const message = inputValue.trim();
    setInputValue("");

    // Add user message
    addChatMessage({ role: "user", content: message });

    setIsLoading(true);

    try {
      // TODO: Integrate with backend API
      // For now, simulate a response
      await new Promise((r) => setTimeout(r, 1500));

      addChatMessage({
        role: "assistant",
        content:
          "I'm the G1 Research Assistant. This is a placeholder response. The actual backend integration will be implemented to handle your queries about alignment experiments, paper search, and model comparisons.",
      });
    } catch (error) {
      addChatMessage({
        role: "error",
        content: "Failed to get response. Please try again.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Handle key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Handle suggestion click
  const handleSuggestion = (suggestion: string) => {
    setInputValue(suggestion);
    inputRef.current?.focus();
  };

  if (!isChatPanelOpen) {
    return null;
  }

  return (
    <div
      className={cn(
        "fixed bottom-[calc(var(--playback-bar-height)+16px)] right-4",
        "w-[400px] bg-[var(--color-bg-surface-2)] rounded-xl shadow-xl",
        "border border-white/10 overflow-hidden",
        "z-[var(--z-modal)]",
        "animate-fadeInUp",
        isChatMinimized && "h-14",
        !isChatMinimized && "h-[550px] flex flex-col",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 bg-[var(--color-bg-surface-3)]">
        <div className="flex items-center gap-2">
          <Bot className="w-5 h-5 text-[var(--color-accent-primary)]" />
          <span className="font-semibold text-white">Research Assistant</span>
          <ModeBadge mode={chatBackendMode} />
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={clearChatMessages}
            className="p-1.5 rounded-md hover:bg-white/10 transition-colors"
            title="Clear chat"
          >
            <RefreshCw className="w-4 h-4 text-white/60" />
          </button>
          <button
            onClick={toggleChatMinimized}
            className="p-1.5 rounded-md hover:bg-white/10 transition-colors"
            title={isChatMinimized ? "Expand" : "Minimize"}
          >
            {isChatMinimized ? (
              <Maximize2 className="w-4 h-4 text-white/60" />
            ) : (
              <Minimize2 className="w-4 h-4 text-white/60" />
            )}
          </button>
          <button
            onClick={toggleChatPanel}
            className="p-1.5 rounded-md hover:bg-white/10 transition-colors"
            title="Close"
          >
            <X className="w-4 h-4 text-white/60" />
          </button>
        </div>
      </div>

      {/* Content (hidden when minimized) */}
      {!isChatMinimized && (
        <>
          {/* Messages area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {chatMessages.length === 0 ? (
              <div className="text-center py-8">
                <Bot className="w-12 h-12 mx-auto text-white/20 mb-4" />
                <p className="text-white/60 mb-4">
                  Ask me about alignment experiments, papers, or model
                  comparisons.
                </p>
                <div className="flex flex-wrap justify-center gap-2">
                  {suggestions.map((s) => (
                    <SuggestionButton
                      key={s}
                      label={s}
                      onClick={() => handleSuggestion(s)}
                    />
                  ))}
                </div>
              </div>
            ) : (
              <>
                {chatMessages.map((msg) => (
                  <Message
                    key={msg.id}
                    role={msg.role}
                    content={msg.content}
                    toolCalls={msg.toolCalls}
                  />
                ))}
                {isLoading && <TypingIndicator />}
                <div ref={messagesEndRef} />
              </>
            )}
          </div>

          {/* Input area */}
          <div className="p-4 border-t border-white/10">
            <div className="flex gap-2">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Type a message..."
                disabled={isLoading}
                className="flex-1 px-4 py-2.5 rounded-lg bg-white/5 border border-white/10 text-white placeholder-white/40 focus:outline-none focus:border-[var(--color-accent-primary)] focus:ring-1 focus:ring-[var(--color-accent-primary)] disabled:opacity-50 text-sm"
              />
              <button
                onClick={handleSend}
                disabled={!inputValue.trim() || isLoading}
                className="px-4 py-2.5 rounded-lg bg-[var(--color-accent-primary)] text-black font-medium hover:bg-[var(--color-accent-primary-hover)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default ChatPanel;
