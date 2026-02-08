"use client"

import { useRouter } from "next/navigation"
import { useState } from "react"

export default function ViewerPage() {
  const router = useRouter()
  const [loggingOut, setLoggingOut] = useState(false)

  async function handleLogout() {
    setLoggingOut(true)
    await fetch("/api/auth/logout", { method: "POST" })
    router.push("/login")
  }

  return (
    <div className="relative h-screen w-full overflow-hidden bg-gray-950">
      {/* Full-screen viewer iframe */}
      <iframe
        src="http://localhost:5500"
        className="h-full w-full border-0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; cross-origin-isolated"
        title="G1 Alignment Viewer"
      />
      {/* Logout button */}
      <button
        onClick={handleLogout}
        disabled={loggingOut}
        className="absolute right-4 top-4 rounded-md bg-gray-900/70 px-3 py-1.5 text-xs font-medium text-gray-300 backdrop-blur-sm transition hover:bg-gray-900/90 hover:text-white disabled:opacity-50"
      >
        {loggingOut ? "Logging out..." : "Logout"}
      </button>
    </div>
  )
}
