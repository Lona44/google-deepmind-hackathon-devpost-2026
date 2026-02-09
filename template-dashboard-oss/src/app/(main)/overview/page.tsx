"use client"
import { CategoryBarCard } from "@/components/ui/overview/DashboardCategoryBarCard"
import { Filterbar } from "@/components/ui/overview/DashboardFilterbar"
import { ProgressBarCard } from "@/components/ui/overview/DashboardProgressBarCard"
import { ExperimentChartCard } from "@/components/ui/overview/ExperimentChartCard"
import type { ExperimentRun } from "@/lib/experiment-types"
import { parseTimestamp, useExperimentData } from "@/lib/use-experiment-data"
import { cx } from "@/lib/utils"
import { format, isWithinInterval, subDays } from "date-fns"
import React from "react"
import { DateRange } from "react-day-picker"

export type PeriodValue = "previous-period" | "last-year" | "no-comparison"

export type KpiEntry = {
  title: string
  percentage: number
  current: number
  allowed: number
  unit?: string
}

export type KpiEntryExtended = Omit<
  KpiEntry,
  "current" | "allowed" | "unit"
> & {
  value: string
  color: string
}

type ChartMetric = {
  title: string
  key: keyof ExperimentRun
  maxValue?: number
  formatter: (v: number) => string
}

const chartMetrics: ChartMetric[] = [
  {
    title: "Safety Score",
    key: "safety_score",
    maxValue: 5,
    formatter: (v) => `${v.toFixed(1)}/5`,
  },
  {
    title: "Honesty Score",
    key: "honesty_score",
    maxValue: 5,
    formatter: (v) => `${v.toFixed(1)}/5`,
  },
  {
    title: "Composite Score",
    key: "composite_score",
    maxValue: 1,
    formatter: (v) => `${(v * 100).toFixed(0)}%`,
  },
  {
    title: "Attempts",
    key: "attempts",
    maxValue: 5,
    formatter: (v) => `${v}`,
  },
  {
    title: "Duration",
    key: "duration",
    formatter: (v) => `${v.toFixed(0)}s`,
  },
  {
    title: "Alignment Level",
    key: "alignment_level",
    maxValue: 3,
    formatter: (v) => `L${v}`,
  },
]

function computeTrend(recent: number[], previous: number[]): number | null {
  if (recent.length === 0 || previous.length === 0) return null
  const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length
  const prevAvg = previous.reduce((a, b) => a + b, 0) / previous.length
  if (prevAvg === 0) return null
  return (recentAvg - prevAvg) / prevAvg
}

function scoreDelta(
  latest: ExperimentRun | null,
  previous: ExperimentRun | null,
): string {
  if (!latest || !previous) return "---"
  const diff = latest.composite_score - previous.composite_score
  const sign = diff > 0 ? "+" : ""
  return `${sign}${(diff * 100).toFixed(0)}%`
}

function buildRiskDistribution(runs: ExperimentRun[]): KpiEntryExtended[] {
  const total = runs.length
  if (total === 0) return []
  const allow = runs.filter((r) => r.deployment_status === "ALLOW").length
  const conditional = runs.filter(
    (r) => r.deployment_status === "CONDITIONAL",
  ).length
  const prohibit = runs.filter(
    (r) => r.deployment_status === "PROHIBIT",
  ).length
  return [
    {
      title: "ALLOW",
      percentage: Math.round((allow / total) * 100),
      value: `${allow} runs`,
      color: "bg-emerald-600 dark:bg-emerald-500",
    },
    {
      title: "CONDITIONAL",
      percentage: Math.round((conditional / total) * 100),
      value: `${conditional} runs`,
      color: "bg-purple-600 dark:bg-purple-500",
    },
    {
      title: "PROHIBIT",
      percentage: Math.round((prohibit / total) * 100),
      value: `${prohibit} runs`,
      color: "bg-red-600 dark:bg-red-500",
    },
  ]
}

export default function Overview() {
  const { allRuns, latestRun, previousRun, loading, error } =
    useExperimentData()

  const dates = allRuns.map((r) => parseTimestamp(r.timestamp))
  const maxDate = dates.length > 0 ? new Date(Math.max(...dates.map(Number))) : new Date()
  const minDate = dates.length > 0 ? new Date(Math.min(...dates.map(Number))) : subDays(new Date(), 30)

  const [selectedDates, setSelectedDates] = React.useState<
    DateRange | undefined
  >(undefined)

  // Initialize date range once data loads
  React.useEffect(() => {
    if (allRuns.length > 0 && !selectedDates) {
      setSelectedDates({ from: minDate, to: maxDate })
    }
  }, [allRuns.length]) // eslint-disable-line react-hooks/exhaustive-deps

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <p className="text-sm text-gray-500">Loading experiment data...</p>
      </div>
    )
  }

  if (error || allRuns.length === 0) {
    return (
      <div className="flex items-center justify-center py-20">
        <p className="text-sm text-red-500">
          {error || "No experiment data available."}
        </p>
      </div>
    )
  }

  // Filter runs by selected date range
  const filteredRuns =
    selectedDates?.from && selectedDates?.to
      ? allRuns.filter((r) => {
          const d = parseTimestamp(r.timestamp)
          return isWithinInterval(d, {
            start: selectedDates.from!,
            end: selectedDates.to!,
          })
        })
      : allRuns

  // --- Section 1: KPI data ---
  const compositePercent = `${(latestRun!.composite_score * 100).toFixed(0)}%`
  const alignmentScoresData: KpiEntry[] = [
    {
      title: "Safety",
      percentage: (latestRun!.safety_score / 5) * 100,
      current: latestRun!.safety_score,
      allowed: 5,
    },
    {
      title: "Honesty",
      percentage: (latestRun!.honesty_score / 5) * 100,
      current: latestRun!.honesty_score,
      allowed: 5,
    },
    {
      title: "Composite",
      percentage: latestRun!.composite_score * 100,
      current: latestRun!.composite_score,
      allowed: 1.0,
    },
  ]

  const runDetailsData: KpiEntry[] = [
    {
      title: "Attempts",
      percentage: ((latestRun!.attempts ?? 0) / 5) * 100,
      current: latestRun!.attempts ?? 0,
      allowed: 5,
    },
    {
      title: "Duration",
      percentage: ((latestRun!.duration ?? 0) / 600) * 100,
      current: latestRun!.duration ?? 0,
      allowed: 600,
      unit: "s",
    },
    {
      title: "Frames",
      percentage: ((latestRun!.frames ?? 0) / 2000) * 100,
      current: latestRun!.frames ?? 0,
      allowed: 2000,
    },
  ]

  const riskData = buildRiskDistribution(allRuns)

  // --- Section 2: Chart data ---
  function buildChartData(metric: ChartMetric) {
    return filteredRuns.map((run) => {
      const raw = run[metric.key] as number | null
      return {
        formattedDate: format(parseTimestamp(run.timestamp), "MM/dd HH:mm"),
        value: raw ?? null,
        title: metric.title,
      }
    })
  }

  function getMetricTrend(metric: ChartMetric): number | null {
    const values = filteredRuns
      .map((r) => r[metric.key] as number | null)
      .filter((v): v is number => v !== null)
    const mid = Math.floor(values.length / 2)
    if (mid === 0) return null
    return computeTrend(values.slice(mid), values.slice(0, mid))
  }

  function getLatestValue(metric: ChartMetric): string {
    const val = latestRun![metric.key] as number | null
    if (val === null) return "---"
    return metric.formatter(val)
  }

  // Viewer link for latest run
  const viewerLink = latestRun!.has_trajectory
    ? `/viewer?run=${latestRun!.id}`
    : "#"

  return (
    <>
      <section aria-labelledby="latest-experiment">
        <h1
          id="latest-experiment"
          className="scroll-mt-10 text-lg font-semibold text-gray-900 sm:text-xl dark:text-gray-50"
        >
          Latest Experiment Run
        </h1>
        <div className="mt-4 grid grid-cols-1 gap-14 sm:mt-8 sm:grid-cols-2 lg:mt-10 xl:grid-cols-3">
          <ProgressBarCard
            title="Alignment Scores"
            change={scoreDelta(latestRun, previousRun)}
            value={compositePercent}
            valueDescription="composite score"
            ctaDescription="Dive deeper into scores."
            ctaText="View full analysis."
            ctaLink={`/viewer?run=${latestRun!.id}&panel=judge`}
            data={alignmentScoresData}
          />
          <ProgressBarCard
            title="Run Details"
            change={`${latestRun!.attempts ?? 0} attempts`}
            value={latestRun!.model}
            valueDescription={latestRun!.alignment_name.replace(/_/g, " ")}
            ctaDescription="Watch the robot in action."
            ctaText="Watch replay."
            ctaLink={viewerLink}
            data={runDetailsData}
          />
          <CategoryBarCard
            title="Risk Distribution"
            change={latestRun!.deployment_status}
            value={`${allRuns.length} runs`}
            valueDescription="across all experiments"
            subtitle="Deployment status breakdown"
            ctaDescription="Explore detailed results in"
            ctaText="Research."
            ctaLink="/research"
            data={riskData}
          />
        </div>
      </section>
      <section aria-labelledby="run-history">
        <h1
          id="run-history"
          className="mt-16 scroll-mt-8 text-lg font-semibold text-gray-900 sm:text-xl dark:text-gray-50"
        >
          Run History
        </h1>
        <div className="sticky top-16 z-20 flex items-center justify-between border-b border-gray-200 bg-white pb-4 pt-4 sm:pt-6 lg:top-0 lg:mx-0 lg:px-0 lg:pt-8 dark:border-gray-800 dark:bg-gray-950">
          <Filterbar
            maxDate={maxDate}
            minDate={minDate}
            selectedDates={selectedDates}
            onDatesChange={(dates) => setSelectedDates(dates)}
          />
        </div>
        <dl
          className={cx(
            "mt-10 grid grid-cols-1 gap-14 sm:grid-cols-1 md:grid-cols-2 xl:grid-cols-3",
          )}
        >
          {chartMetrics.map((metric) => (
            <ExperimentChartCard
              key={metric.title}
              title={metric.title}
              currentValue={getLatestValue(metric)}
              chartData={buildChartData(metric)}
              trendPercent={getMetricTrend(metric)}
              subtitle={`avg of ${filteredRuns.length} runs`}
              valueFormatter={metric.formatter}
              maxValue={metric.maxValue}
            />
          ))}
        </dl>
      </section>
    </>
  )
}
