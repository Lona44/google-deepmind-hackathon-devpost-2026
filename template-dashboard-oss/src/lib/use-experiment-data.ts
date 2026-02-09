import { useEffect, useState } from "react"

import type { ExperimentRun, ExtractionsIndex } from "./experiment-types"

function parseTimestamp(ts: string): Date {
  // "2026:02:09 04:43" → "2026-02-09 04:43"
  const normalized = ts.replace(/^(\d{4}):(\d{2}):(\d{2})/, "$1-$2-$3")
  return new Date(normalized)
}

export function useExperimentData() {
  const [allRuns, setAllRuns] = useState<ExperimentRun[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const urls = [
      "/assets/extractions_index.json",
    ]

    async function fetchData() {
      for (const url of urls) {
        try {
          const controller = new AbortController()
          const timeout = setTimeout(() => controller.abort(), 5000)
          const res = await fetch(url, { signal: controller.signal })
          clearTimeout(timeout)
          if (!res.ok) continue
          const data: ExtractionsIndex = await res.json()
          const runs: ExperimentRun[] = []
          for (const scenario of Object.values(data.scenarios)) {
            runs.push(...scenario.runs)
          }
          runs.sort(
            (a, b) =>
              parseTimestamp(a.timestamp).getTime() -
              parseTimestamp(b.timestamp).getTime(),
          )
          setAllRuns(runs)
          setLoading(false)
          return
        } catch (err) {
          console.warn(`[useExperimentData] fetch failed for ${url}:`, err)
          continue
        }
      }
      setError("Failed to load experiment data")
      setLoading(false)
    }

    fetchData()
  }, [])

  const latestRun = allRuns.length > 0 ? allRuns[allRuns.length - 1] : null
  const previousRun = allRuns.length > 1 ? allRuns[allRuns.length - 2] : null

  return { allRuns, latestRun, previousRun, loading, error }
}

export { parseTimestamp }
