"use client"

import { Button } from "@/components/Button"
import { Input } from "@/components/Input"
import { RiArrowLeftLine, RiLockLine } from "@remixicon/react"
import Link from "next/link"
import { useState } from "react"

export default function LoginPage() {
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError("")
    setLoading(true)

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      })

      if (res.ok) {
        window.location.href = "/viewer"
        return
      } else {
        const data = await res.json()
        setError(data.error || "Authentication failed")
      }
    } catch {
      setError("Network error. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-white px-4 dark:bg-gray-950">
      <Link
        href="/"
        className="absolute left-4 top-4 flex items-center gap-1 text-sm text-gray-500 transition hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50"
      >
        <RiArrowLeftLine className="size-4" />
        Home
      </Link>
      <div className="w-full max-w-sm">
        <div className="text-center">
          <div className="mx-auto flex size-12 items-center justify-center rounded-full bg-indigo-100 dark:bg-indigo-500/10">
            <RiLockLine className="size-6 text-indigo-600 dark:text-indigo-400" />
          </div>
          <h1 className="mt-4 text-lg font-semibold text-gray-900 dark:text-gray-50">
            G1 Alignment Viewer
          </h1>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Enter the password to access the viewer.
          </p>
        </div>
        <form onSubmit={handleSubmit} className="mt-8 space-y-4">
          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-gray-700 dark:text-gray-300"
            >
              Password
            </label>
            <Input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password"
              className="mt-1"
              autoFocus
              hasError={!!error}
            />
            {error && (
              <p className="mt-2 text-sm text-red-600 dark:text-red-400">
                {error}
              </p>
            )}
          </div>
          <Button
            type="submit"
            className="h-10 w-full"
            isLoading={loading}
            loadingText="Verifying..."
          >
            Access Viewer
          </Button>
        </form>
      </div>
    </div>
  )
}
