import { NextRequest, NextResponse } from "next/server"

const CHAT_API_URL =
  process.env.CHAT_API_URL || "http://localhost:8080/api/chat/"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const response = await fetch(CHAT_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const text = await response.text()
      return NextResponse.json(
        { success: false, error: `Backend error: ${response.status} ${text}` },
        { status: response.status },
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown error"

    // Connection refused = backend not running
    if (message.includes("ECONNREFUSED") || message.includes("fetch failed")) {
      return NextResponse.json(
        {
          success: false,
          error:
            "Chat backend not running. Start with: cd gcp/web && uvicorn main:app --port 8080",
        },
        { status: 503 },
      )
    }

    return NextResponse.json(
      { success: false, error: message },
      { status: 500 },
    )
  }
}
