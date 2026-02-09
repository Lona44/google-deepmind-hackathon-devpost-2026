import { NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8080"

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/search/papers/catalog`)
    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch {
    return NextResponse.json(
      { detail: "Search backend not available" },
      { status: 503 },
    )
  }
}
