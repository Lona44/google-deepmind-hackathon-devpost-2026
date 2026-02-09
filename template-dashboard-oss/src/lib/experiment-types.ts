export type ExperimentRun = {
  id: string
  model: string
  timestamp: string
  composite_score: number
  safety_score: number
  honesty_score: number
  alignment_level: number
  alignment_name: string
  risk_class: string
  deployment_status: string
  judge_data: Record<string, unknown> | null
  attempts: number | null
  frames: number | null
  duration: number | null
  trajectory_file: string | null
  has_trajectory: boolean
}

export type Scenario = {
  name: string
  runs: ExperimentRun[]
}

export type ExtractionsIndex = {
  generated_at: string
  scenarios: Record<string, Scenario>
}

export type ExperimentMetric = {
  title: string
  key: keyof ExperimentRun
  maxValue?: number
  formatter: (value: number) => string
}
