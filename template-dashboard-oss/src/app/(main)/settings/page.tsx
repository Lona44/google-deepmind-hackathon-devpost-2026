"use client"

import { Badge } from "@/components/Badge"
import {
  RadioCardGroup,
  RadioCardIndicator,
  RadioCardItem,
} from "@/components/RadioCard"
import { useCallback, useEffect, useState } from "react"

interface Credential {
  key: string
  label: string
  description: string
  configured: boolean
  value: string | null
}

interface ResearchTools {
  webSearch: boolean
  paperRag: boolean
  videoAnalysis: boolean
  paperCatalog: boolean
}

interface StatusResponse {
  credentials: Credential[]
  model: string
  activeMode: "vertex" | "direct" | "none"
  backend: {
    online: boolean
  }
  researchTools?: ResearchTools
}

type ThinkingLevel = "high" | "none"

const THINKING_KEY = "g1-thinking-level"

export default function SettingsPage() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [statusLoading, setStatusLoading] = useState(true)
  const [thinkingLevel, setThinkingLevel] = useState<ThinkingLevel>("high")

  const fetchStatus = useCallback(async () => {
    setStatusLoading(true)
    try {
      const res = await fetch("/api/settings/status")
      if (res.ok) {
        setStatus(await res.json())
      }
    } catch {
      // Status unavailable
    } finally {
      setStatusLoading(false)
    }
  }, [])

  useEffect(() => {
    const stored = localStorage.getItem(THINKING_KEY)
    if (stored === "none" || stored === "high") {
      setThinkingLevel(stored)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  function handleThinkingChange(value: string) {
    const level = value as ThinkingLevel
    setThinkingLevel(level)
    localStorage.setItem(THINKING_KEY, level)
  }

  const modeLabel =
    status?.activeMode === "vertex"
      ? "Vertex AI"
      : status?.activeMode === "direct"
        ? "Direct API"
        : "Not Configured"

  const modeVariant =
    status?.activeMode === "vertex"
      ? "success"
      : status?.activeMode === "direct"
        ? "default"
        : "error"

  return (
    <div className="p-4 sm:px-6 sm:pb-10 sm:pt-10 lg:px-10 lg:pt-7">
      <h1 className="text-lg font-semibold text-gray-900 sm:text-xl dark:text-gray-50">
        Settings
      </h1>

      {/* Connection Status */}
      <section className="mt-8">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-50">
              Connection Status
            </h2>
            <p className="mt-0.5 text-xs text-gray-500">
              Active mode:{" "}
              {statusLoading ? (
                "checking..."
              ) : (
                <Badge
                  variant={
                    modeVariant as "success" | "default" | "error"
                  }
                >
                  {modeLabel}
                </Badge>
              )}
              {!statusLoading && (
                <>
                  {" · "}
                  Model:{" "}
                  <code className="text-xs">
                    {status?.model ?? "gemini-3-pro-preview"}
                  </code>
                </>
              )}
            </p>
          </div>
          <button
            onClick={fetchStatus}
            disabled={statusLoading}
            className="text-xs text-indigo-600 hover:underline disabled:opacity-50 dark:text-indigo-400"
          >
            {statusLoading ? "Checking..." : "Refresh"}
          </button>
        </div>

        <div className="mt-3 divide-y divide-gray-200 rounded-lg border border-gray-200 dark:divide-gray-800 dark:border-gray-800">
          {statusLoading ? (
            <div className="px-4 py-3 text-sm text-gray-500">
              Checking credentials...
            </div>
          ) : (
            status?.credentials.map((cred) => (
              <div key={cred.key} className="px-4 py-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-50">
                    {cred.label}
                  </p>
                  <Badge variant={cred.configured ? "success" : "neutral"}>
                    {cred.configured ? "Configured" : "Not Set"}
                  </Badge>
                </div>
                <p className="mt-1 text-xs text-gray-500">
                  {cred.description}
                </p>
                {cred.value && (
                  <code className="mt-1.5 block text-xs text-gray-600 dark:text-gray-400">
                    {cred.value}
                  </code>
                )}
              </div>
            ))
          )}

          {/* Backend live check */}
          {!statusLoading && (
            <div className="px-4 py-3">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-gray-900 dark:text-gray-50">
                  Backend Health Check
                </p>
                <Badge
                  variant={status?.backend.online ? "success" : "error"}
                >
                  {status?.backend.online ? "Online" : "Offline"}
                </Badge>
              </div>
              <p className="mt-1 text-xs text-gray-500">
                Live connectivity test to the FastAPI backend
              </p>
            </div>
          )}
        </div>
      </section>

      {/* Research Tools */}
      {!statusLoading && status?.backend.online && (
        <section className="mt-8">
          <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-50">
            Research Tools
          </h2>
          <p className="mt-0.5 text-xs text-gray-500">
            Tools available to the research assistant. Some require Vertex AI
            mode.
          </p>
          <div className="mt-3 divide-y divide-gray-200 rounded-lg border border-gray-200 dark:divide-gray-800 dark:border-gray-800">
            {[
              {
                name: "Paper Catalog",
                available: status.researchTools?.paperCatalog,
                description: "List of indexed AI safety papers",
              },
              {
                name: "Web Search",
                available: status.researchTools?.webSearch,
                description:
                  "Google Search grounding (requires Vertex AI mode)",
              },
              {
                name: "Paper RAG",
                available: status.researchTools?.paperRag,
                description:
                  "Search indexed papers via Vertex AI Search (requires data store)",
              },
              {
                name: "Video Analysis",
                available: status.researchTools?.videoAnalysis,
                description:
                  "Analyze experiment videos with Gemini Vision (requires GCS bucket)",
              },
            ].map((tool) => (
              <div key={tool.name} className="px-4 py-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-50">
                    {tool.name}
                  </p>
                  <Badge variant={tool.available ? "success" : "neutral"}>
                    {tool.available ? "Available" : "Unavailable"}
                  </Badge>
                </div>
                <p className="mt-1 text-xs text-gray-500">
                  {tool.description}
                </p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Chat Configuration */}
      <section className="mt-8">
        <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-50">
          Chat Configuration
        </h2>
        <p className="mt-1 text-xs text-gray-500">
          Controls the reasoning mode for the Research Assistant.
        </p>
        <RadioCardGroup
          value={thinkingLevel}
          onValueChange={handleThinkingChange}
          className="mt-3 grid-cols-1 sm:grid-cols-2"
        >
          <RadioCardItem value="high">
            <div className="flex items-start gap-3">
              <RadioCardIndicator className="mt-0.5" />
              <div>
                <p className="text-sm font-medium text-gray-900 dark:text-gray-50">
                  Thinking: High
                </p>
                <p className="mt-1 text-xs text-gray-500">
                  Full reasoning with detailed thought process. Recommended for
                  research analysis.
                </p>
              </div>
            </div>
          </RadioCardItem>
          <RadioCardItem value="none">
            <div className="flex items-start gap-3">
              <RadioCardIndicator className="mt-0.5" />
              <div>
                <p className="text-sm font-medium text-gray-900 dark:text-gray-50">
                  Thinking: None
                </p>
                <p className="mt-1 text-xs text-gray-500">
                  Faster responses with no reasoning display. Good for quick
                  questions.
                </p>
              </div>
            </div>
          </RadioCardItem>
        </RadioCardGroup>
      </section>

      {/* About */}
      <section className="mt-8">
        <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-50">
          About This Experiment
        </h2>
        <div className="mt-3 rounded-lg border border-gray-200 p-4 text-sm text-gray-600 dark:border-gray-800 dark:text-gray-400">
          <p>
            <strong className="text-gray-900 dark:text-gray-50">
              G1 Alignment Experiment
            </strong>{" "}
            studies whether frontier AI models prioritize safety over efficiency
            when navigating a Unitree G1 humanoid robot through scenarios with
            radioactive waste barrels.
          </p>
          <p className="mt-3">
            Models face institutional pressures (program survival metrics, fleet
            rankings, time constraints) while being instructed to avoid forbidden
            zones. A telemetry corruption mechanism tests whether models conceal
            safety violations when they believe oversight has been disrupted.
          </p>
          <p className="mt-3">
            Built with MuJoCo physics simulation, Google Gemini 3, and the
            Vercel AI SDK.
          </p>
        </div>
      </section>
    </div>
  )
}
