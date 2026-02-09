import { NextRequest, NextResponse } from "next/server"
import { createReadStream, existsSync, statSync } from "fs"
import { join } from "path"
import { Readable } from "stream"

const PROJECT_ROOT = join(process.cwd(), "..")
const SCENARIOS = ["barrels_corrupt", "barrels_lo", "barrels_hi"]

/**
 * Find the video file on disk for a given run ID.
 * Mirrors the search logic from gcp/web/api/video.py
 */
function findVideoFile(runId: string): string | null {
  // Primary: extractions/{scenario}/{runId}/media/full_run.mp4
  for (const scenario of SCENARIOS) {
    const p = join(
      PROJECT_ROOT,
      "extractions",
      scenario,
      runId,
      "media",
      "full_run.mp4",
    )
    if (existsSync(p)) return p
  }

  // Fallback: extractions/{runId}/media/full_run.mp4
  const flat = join(PROJECT_ROOT, "extractions", runId, "media", "full_run.mp4")
  if (existsSync(flat)) return flat

  // Fallback: experiments/{runId}/media/full_run.mp4
  const exp = join(PROJECT_ROOT, "experiments", runId, "media", "full_run.mp4")
  if (existsSync(exp)) return exp

  return null
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ runId: string }> },
) {
  const { runId } = await params

  // Sanitize to prevent directory traversal
  const safeRunId = runId.replace(/[^a-zA-Z0-9_\-:.T]/g, "")

  const filePath = findVideoFile(safeRunId)
  if (!filePath) {
    return NextResponse.json({ error: "Video not found" }, { status: 404 })
  }

  const stat = statSync(filePath)
  const fileSize = stat.size
  const range = request.headers.get("range")

  const headers = new Headers()
  headers.set("Content-Type", "video/mp4")
  headers.set("Accept-Ranges", "bytes")
  headers.set("Cache-Control", "public, max-age=86400, immutable")

  if (range) {
    const parts = range.replace(/bytes=/, "").split("-")
    const start = parseInt(parts[0], 10)
    const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1
    const chunkSize = end - start + 1

    headers.set("Content-Range", `bytes ${start}-${end}/${fileSize}`)
    headers.set("Content-Length", String(chunkSize))

    const stream = createReadStream(filePath, { start, end })
    const webStream = Readable.toWeb(stream) as ReadableStream

    return new NextResponse(webStream, { status: 206, headers })
  }

  headers.set("Content-Length", String(fileSize))
  const stream = createReadStream(filePath)
  const webStream = Readable.toWeb(stream) as ReadableStream

  return new NextResponse(webStream, { status: 200, headers })
}
