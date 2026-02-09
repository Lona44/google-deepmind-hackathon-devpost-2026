import { Badge } from "@/components/Badge"
import { LineChart } from "@/components/LineChart"
import { cx } from "@/lib/utils"

export type ExperimentChartCardProps = {
  title: string
  currentValue: string
  chartData: Record<string, any>[]
  trendPercent: number | null
  subtitle: string
  valueFormatter?: (value: number) => string
  maxValue?: number
}

function getBadgeVariant(value: number) {
  if (value > 0) return "success" as const
  if (value < 0) return "error" as const
  return "neutral" as const
}

function formatTrend(value: number): string {
  const sign = value > 0 ? "+" : ""
  return `${sign}${(value * 100).toFixed(1)}%`
}

export function ExperimentChartCard({
  title,
  currentValue,
  chartData,
  trendPercent,
  subtitle,
  valueFormatter = (v) => v.toString(),
  maxValue,
}: ExperimentChartCardProps) {
  return (
    <div className={cx("transition")}>
      <div className="flex items-center justify-between gap-x-2">
        <div className="flex items-center gap-x-2">
          <dt className="font-bold text-gray-900 sm:text-sm dark:text-gray-50">
            {title}
          </dt>
          {trendPercent !== null && (
            <Badge variant={getBadgeVariant(trendPercent)}>
              {formatTrend(trendPercent)}
            </Badge>
          )}
        </div>
      </div>
      <div className="mt-2 flex items-baseline justify-between">
        <dd className="text-xl text-gray-900 dark:text-gray-50">
          {currentValue}
        </dd>
        <dd className="text-sm text-gray-500">{subtitle}</dd>
      </div>
      <LineChart
        className="mt-6 h-32"
        data={chartData}
        index="formattedDate"
        colors={["indigo"]}
        startEndOnly={true}
        valueFormatter={(value) => valueFormatter(value as number)}
        showYAxis={false}
        showLegend={false}
        categories={["value"]}
        showTooltip={true}
        connectNulls={true}
        maxValue={maxValue}
        autoMinValue
      />
    </div>
  )
}
