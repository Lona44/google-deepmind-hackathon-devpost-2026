import { NextRequest, NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8080"

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ runId: string }> },
) {
  const { runId } = await params

  const headers: Record<string, string> = {}
  const range = request.headers.get("range")
  if (range) {
    headers["Range"] = range
  }

  try {
    const response = await fetch(
      `${BACKEND_URL}/api/video/stream/${encodeURIComponent(runId)}`,
      { headers },
    )

    if (!response.ok) {
      return NextResponse.json(
        { error: `Video not found: ${response.status}` },
        { status: response.status },
      )
    }

    const responseHeaders = new Headers()
    responseHeaders.set("Content-Type", "video/mp4")
    const contentLength = response.headers.get("content-length")
    if (contentLength) responseHeaders.set("Content-Length", contentLength)
    const contentRange = response.headers.get("content-range")
    if (contentRange) responseHeaders.set("Content-Range", contentRange)
    const acceptRanges = response.headers.get("accept-ranges")
    if (acceptRanges) responseHeaders.set("Accept-Ranges", acceptRanges)

    return new NextResponse(response.body, {
      status: response.status,
      headers: responseHeaders,
    })
  } catch {
    return NextResponse.json(
      { error: "Video backend not available" },
      { status: 503 },
    )
  }
}
