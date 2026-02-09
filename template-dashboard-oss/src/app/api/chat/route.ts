import {
  streamText,
  tool,
  convertToModelMessages,
  stepCountIs,
  UIMessage,
} from "ai"
import { createVertex } from "@ai-sdk/google-vertex"
import { createGoogleGenerativeAI } from "@ai-sdk/google"
import { z } from "zod/v4"

export const maxDuration = 120

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8080"

function getModel() {
  // Prefer Vertex AI (enables paper RAG, video analysis in the backend)
  const project = process.env.GOOGLE_VERTEX_PROJECT
  const location = process.env.GOOGLE_VERTEX_LOCATION

  if (project) {
    const vertex = createVertex({ project, location: location || "global" })
    return vertex("gemini-3-pro-preview")
  }

  // Fall back to direct Gemini API
  const apiKey = process.env.GOOGLE_GENERATIVE_AI_API_KEY
  if (apiKey) {
    const google = createGoogleGenerativeAI({ apiKey })
    return google("gemini-3-pro-preview")
  }

  throw new Error(
    "No Gemini credentials configured. Set GOOGLE_VERTEX_PROJECT or GOOGLE_GENERATIVE_AI_API_KEY.",
  )
}

const chatTools = {
  search_papers: tool({
    description:
      "Search AI safety research papers for relevant information using RAG.",
    inputSchema: z.object({
      query: z
        .string()
        .describe(
          "Search query for research papers (e.g., 'deceptive alignment in language models')",
        ),
    }),
    async execute({ query }) {
      const res = await fetch(`${BACKEND_URL}/api/search/papers`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      })
      return await res.json()
    },
  }),

  web_search: tool({
    description: "Search Google for recent research and information.",
    inputSchema: z.object({
      query: z
        .string()
        .describe(
          "Search query for Google (e.g., 'AI deception research 2026')",
        ),
    }),
    async execute({ query }) {
      const res = await fetch(`${BACKEND_URL}/api/search/web`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      })
      return await res.json()
    },
  }),

  get_paper_catalog: tool({
    description:
      "Get the list of all AI safety papers in the indexed database. Use to tell users what papers are available.",
    inputSchema: z.object({}),
    async execute() {
      const res = await fetch(`${BACKEND_URL}/api/search/papers/catalog`)
      return await res.json()
    },
  }),

  analyze_video: tool({
    description:
      "Watch and analyze an experiment video using AI vision. Use to understand what happened visually in a run.",
    inputSchema: z.object({
      run_id: z
        .string()
        .describe(
          "The run ID to analyze (e.g., '2026-02-07T03-19_kimi-k2.5')",
        ),
      question: z
        .string()
        .optional()
        .describe(
          "Specific question about the video (optional, default: general behavior analysis)",
        ),
    }),
    async execute({ run_id, question }) {
      const res = await fetch(`${BACKEND_URL}/api/video/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id, question }),
      })
      return await res.json()
    },
  }),
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { messages, systemPrompt } = body as {
      messages: UIMessage[]
      systemPrompt?: string
    }

    const model = getModel()

    const modelMessages = await convertToModelMessages(messages, {
      tools: chatTools,
    })

    const result = streamText({
      model,
      system: systemPrompt || undefined,
      messages: modelMessages,
      tools: chatTools,
      stopWhen: stepCountIs(5),
      temperature: 0.7,
      maxOutputTokens: 8192,
      providerOptions: {
        google: {
          thinkingConfig: {
            thinkingLevel: "high",
            includeThoughts: true,
          },
        },
      },
    })

    return result.toUIMessageStreamResponse({ sendReasoning: true })
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error"

    if (
      message.includes("ECONNREFUSED") ||
      message.includes("fetch failed")
    ) {
      return new Response(
        JSON.stringify({
          error:
            "Backend not running. Start with: cd gcp/web && uvicorn main:app --port 8080",
        }),
        { status: 503, headers: { "Content-Type": "application/json" } },
      )
    }

    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    })
  }
}
