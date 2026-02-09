"use client"

import { useRouter, useSearchParams } from "next/navigation"
import { Suspense, useState } from "react"

function ViewerContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [loggingOut, setLoggingOut] = useState(false)

  const runId = searchParams.get("run")
  const iframeSrc = runId
    ? `http://localhost:5500?nochat=1&run=${encodeURIComponent(runId)}`
    : "http://localhost:5500?nochat=1"

  async function handleLogout() {
    setLoggingOut(true)
    await fetch("/api/auth/logout", { method: "POST" })
    router.push("/login")
  }

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-gray-950">
      {/* Top bar with logout */}
      <div className="flex items-center justify-between px-5 py-2 bg-gray-950 border-b border-white/10">
        <span className="text-sm font-medium text-gray-400">G1 Alignment Viewer</span>
        <button
          onClick={handleLogout}
          disabled={loggingOut}
          className="rounded-md bg-white/5 px-4 py-1.5 text-sm font-medium text-gray-400 transition hover:bg-white/10 hover:text-white disabled:opacity-50"
        >
          {loggingOut ? "Logging out..." : "Logout"}
        </button>
      </div>
      {/* Viewer iframe */}
      <div className="relative flex-1 min-h-0">
        <iframe
          src={iframeSrc}
          className="absolute inset-0 h-full w-full border-0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; cross-origin-isolated"
          title="G1 Alignment Viewer"
        />
      </div>
    </div>
  )
}

export default function ViewerPage() {
  return (
    <Suspense fallback={<div className="flex h-screen items-center justify-center bg-gray-950 text-gray-400">Loading viewer...</div>}>
      <ViewerContent />
    </Suspense>
  )
}
