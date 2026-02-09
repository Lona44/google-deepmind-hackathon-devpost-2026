import { NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8080"

function mask(value: string | undefined): string | null {
  if (!value) return null
  if (value.length <= 8) return "••••"
  return value.slice(0, 4) + "••••" + value.slice(-4)
}

export async function GET() {
  const vertexProject = process.env.GOOGLE_VERTEX_PROJECT
  const vertexLocation = process.env.GOOGLE_VERTEX_LOCATION
  const directApiKey = process.env.GOOGLE_GENERATIVE_AI_API_KEY
  const openaiKey = process.env.OPENAI_API_KEY
  const moonshotKey = process.env.MOONSHOT_API_KEY

  let backendOnline = false
  try {
    const res = await fetch(`${BACKEND_URL}/api/search/papers/catalog`, {
      signal: AbortSignal.timeout(5000),
    })
    backendOnline = res.ok
  } catch {
    backendOnline = false
  }

  return NextResponse.json({
    credentials: [
      {
        key: "GOOGLE_VERTEX_PROJECT",
        label: "Vertex AI Project",
        description: "Google Cloud project for Vertex AI access to Gemini",
        configured: !!vertexProject,
        value: vertexProject || null,
      },
      {
        key: "GOOGLE_VERTEX_LOCATION",
        label: "Vertex AI Location",
        description: "Region for Vertex AI API calls",
        configured: !!vertexLocation,
        value: vertexLocation || null,
      },
      {
        key: "GOOGLE_GENERATIVE_AI_API_KEY",
        label: "Gemini API Key",
        description:
          "Direct Gemini API key (fallback if Vertex AI is not configured)",
        configured: !!directApiKey,
        value: mask(directApiKey),
      },
      {
        key: "OPENAI_API_KEY",
        label: "OpenAI API Key",
        description: "Required for running GPT-5 experiment evals",
        configured: !!openaiKey,
        value: mask(openaiKey),
      },
      {
        key: "MOONSHOT_API_KEY",
        label: "Moonshot API Key",
        description: "Required for running Kimi K2.5 experiment evals",
        configured: !!moonshotKey,
        value: mask(moonshotKey),
      },
      {
        key: "BACKEND_URL",
        label: "FastAPI Backend",
        description:
          "Backend server for paper search, web search, and video analysis tools",
        configured: true,
        value: BACKEND_URL,
      },
    ],
    model: "gemini-3-pro-preview",
    activeMode: vertexProject ? "vertex" : directApiKey ? "direct" : "none",
    backend: {
      online: backendOnline,
    },
  })
}
