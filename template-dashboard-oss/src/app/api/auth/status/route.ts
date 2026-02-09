import { NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
  const session = request.cookies.get("g1-session")
  return NextResponse.json({ authenticated: !!session?.value })
}
