import crypto from "crypto"
import { NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
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
