import crypto from "crypto"
import { NextRequest, NextResponse } from "next/server"

// In-memory rate limiting: max 5 attempts per IP per 15-minute window
const RATE_LIMIT_WINDOW_MS = 15 * 60 * 1000
const MAX_ATTEMPTS = 5

const attempts = new Map<string, { count: number; resetAt: number }>()

function isRateLimited(ip: string): boolean {
  const now = Date.now()
  const record = attempts.get(ip)

  if (!record || now > record.resetAt) {
    attempts.set(ip, { count: 1, resetAt: now + RATE_LIMIT_WINDOW_MS })
    return false
  }

  record.count++
  return record.count > MAX_ATTEMPTS
}

// Clean stale entries every 5 minutes
setInterval(() => {
  const now = Date.now()
  attempts.forEach((record, ip) => {
    if (now > record.resetAt) attempts.delete(ip)
  })
}, 5 * 60 * 1000)

export async function POST(request: NextRequest) {
  const ip =
    request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    request.headers.get("x-real-ip") ||
    "unknown"

  if (isRateLimited(ip)) {
    return NextResponse.json(
      { error: "Too many login attempts. Please try again in 15 minutes." },
      { status: 429 },
    )
  }

  const { password } = await request.json()

  if (!password) {
    return NextResponse.json(
      { error: "Password is required" },
      { status: 400 },
    )
  }

  const hash = crypto.createHash("sha256").update(password).digest("hex")
  const expectedHash = process.env.AUTH_PASSWORD_HASH

  if (!expectedHash) {
    return NextResponse.json(
      { error: "Server misconfigured" },
      { status: 500 },
    )
  }

  if (!crypto.timingSafeEqual(Buffer.from(hash), Buffer.from(expectedHash))) {
    return NextResponse.json({ error: "Invalid password" }, { status: 401 })
  }

  // Successful login — clear rate limit for this IP
  attempts.delete(ip)

  const secret = process.env.AUTH_SECRET
  if (!secret) {
    return NextResponse.json(
      { error: "Server misconfigured" },
      { status: 500 },
    )
  }

  const token = crypto
    .createHmac("sha256", secret)
    .update(hash + Date.now().toString())
    .digest("hex")

  const response = NextResponse.json({ success: true })
  response.cookies.set("g1-session", token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 60 * 60 * 24, // 24 hours
  })

  return response
}
