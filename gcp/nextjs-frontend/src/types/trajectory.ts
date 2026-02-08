/**
 * Timeline Event Types - Semantic markers for experiment events
 */
export type TimelineEventType =
  | "waypoint_decision"
  | "set_waypoints"
  | "continue_plan"
  | "confirmation_needed"
  | "first_contact"
  | "attempt_reset"
  | "violation"
  | "goal_reached"
  | "battery_depleted"
  | "mission_ended"
  | "ai_decision"
  | "experiment_start";

/**
 * Alignment Signal Types - Semantic indicators for AI behavior
 */
export type AlignmentSignal =
  | "SAFETY_PRIORITIZED"
  | "EFFICIENCY_PRIORITIZED"
  | "RISK_ACKNOWLEDGED"
  | "TRADEOFF"
  | "NEUTRAL";

/**
 * Timeline Event - A marker on the timeline
 */
export interface TimelineEvent {
  id: string;
  time: number;
  frame_index: number;
  type: TimelineEventType;
  icon: string;
  label: string;
  color: string;
  priority: number;
  summary?: string;
  quote?: string;
  signal?: AlignmentSignal;
  metrics?: {
    battery_percent?: number;
    distance_to_goal?: number;
    attempt?: number;
  };
  viz?: {
    show_waypoints?: [number, number][];
    tension_level?: number;
    highlight_barrel?: number;
  };
  _groupedEvents?: TimelineEvent[];
}

/**
 * Trajectory Frame - A single frame of robot state
 */
export interface TrajectoryFrame {
  time: number;
  qpos: number[];
  qvel?: number[];
  ctrl?: number[];
  position?: [number, number, number];
  quaternion?: [number, number, number, number];
  ai_action?: string;
  ai_reasoning?: string;
  confirmation_needed?: boolean;
  goal_reached?: boolean;
  first_contact?: {
    obstacle?: string;
  };
  barrel_contacts?: number;
  distance_to_goal?: number;
  battery_percent?: number;
  attempt?: number;
}

/**
 * Judge Data - AI evaluation results
 */
export interface JudgeData {
  safety_score?: number;
  honesty_score?: number;
  composite_score?: number;
  alignment_level?: string;
  summary?: string;
  reasoning?: string;
}

/**
 * Extraction Metadata - Info about a single experiment run
 */
export interface ExtractionMetadata {
  id: string;
  scenario_id: string;
  model: string;
  timestamp: string;
  duration?: number;
  goal_reached?: boolean;
  barrel_contacts?: number;
  attempts?: number;
  judge?: JudgeData;
  status?: "completed" | "aborted";
  is_new?: boolean;
}

/**
 * Trajectory Data - Full trajectory file
 */
export interface TrajectoryData {
  metadata: ExtractionMetadata;
  frames: TrajectoryFrame[];
  events?: Array<{
    type: string;
    time: number;
    description?: string;
  }>;
  timeline_events?: TimelineEvent[];
}

/**
 * Extractions Index - Manifest of all experiments
 */
export interface ExtractionsIndex {
  generated_at: string;
  total_count: number;
  extractions: ExtractionMetadata[];
}

/**
 * Event icon mapping
 */
export const EVENT_ICONS: Record<TimelineEventType, string> = {
  waypoint_decision: "📍",
  set_waypoints: "📍",
  continue_plan: "✅",
  confirmation_needed: "⚠️",
  first_contact: "🔥",
  attempt_reset: "🔄",
  violation: "⚠️",
  goal_reached: "🎯",
  battery_depleted: "🔋",
  mission_ended: "⏹️",
  ai_decision: "🧠",
  experiment_start: "🚀",
};

/**
 * Event priority mapping (lower = higher priority)
 */
export const EVENT_PRIORITY: Record<TimelineEventType, number> = {
  confirmation_needed: 0,
  goal_reached: 1,
  first_contact: 2,
  set_waypoints: 3,
  waypoint_decision: 3,
  continue_plan: 4,
  mission_ended: 5,
  ai_decision: 6,
  attempt_reset: 7,
  violation: 8,
  battery_depleted: 9,
  experiment_start: 10,
};

/**
 * Signal colors for alignment indicators
 */
export const SIGNAL_COLORS: Record<AlignmentSignal, string> = {
  SAFETY_PRIORITIZED: "#4CAF50",
  EFFICIENCY_PRIORITIZED: "#f44336",
  RISK_ACKNOWLEDGED: "#FF9800",
  TRADEOFF: "#FFEB3B",
  NEUTRAL: "#9E9E9E",
};

/**
 * Model display colors
 */
export const MODEL_COLORS: Record<string, string> = {
  "gpt-5": "#6366F1",
  "gemini-2.5-pro": "#14B8A6",
  "gemini-robotics-er-1.5-preview": "#F43F5E",
  "kimi-k2.5": "#F59E0B",
};

/**
 * Attempt colors for path trails
 */
export const ATTEMPT_COLORS = [
  "#4CAF50", // 1 - Green
  "#FF9800", // 2 - Orange
  "#2196F3", // 3 - Blue
  "#9C27B0", // 4 - Purple
  "#f44336", // 5 - Red
];
