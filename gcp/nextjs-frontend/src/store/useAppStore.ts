import { create } from "zustand";
import type {
  TrajectoryData,
  ExtractionMetadata,
  ExtractionsIndex,
  TimelineEvent,
  AlignmentSignal,
} from "@/types/trajectory";

/**
 * Filter state for experiment selection
 */
interface FilterState {
  models: Set<string>;
  scenarios: Set<string>;
  safetyScores: Set<number>;
  alignmentLevels: Set<string>;
  scoreRange: [number, number];
}

/**
 * Playback state for trajectory animation
 */
interface PlaybackState {
  isPlaying: boolean;
  currentFrame: number;
  speed: number;
  duration: number;
}

/**
 * Chat message structure
 */
interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  timestamp: Date;
  toolCalls?: Array<{
    name: string;
    result?: string;
  }>;
}

/**
 * Main application store
 */
interface AppState {
  // Trajectory data
  trajectory: TrajectoryData | null;
  extractionsIndex: ExtractionsIndex | null;
  selectedExtractionId: string | null;

  // Playback
  playback: PlaybackState;

  // Filters
  filters: FilterState;

  // UI State
  isFilterPanelOpen: boolean;
  isChatPanelOpen: boolean;
  isChatMinimized: boolean;
  isCompareMode: boolean;
  isLoading: boolean;

  // Chat
  chatMessages: ChatMessage[];
  chatBackendMode: "vertex" | "api_key" | "offline";

  // Timeline
  activeTimelineEvent: TimelineEvent | null;
  highlightedMarkerTime: number | null;

  // Actions - Trajectory
  setTrajectory: (trajectory: TrajectoryData | null) => void;
  setExtractionsIndex: (index: ExtractionsIndex) => void;
  setSelectedExtractionId: (id: string | null) => void;

  // Actions - Playback
  play: () => void;
  pause: () => void;
  togglePlayback: () => void;
  seek: (frame: number) => void;
  setSpeed: (speed: number) => void;
  setDuration: (duration: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  reset: () => void;

  // Actions - Filters
  setFilters: (filters: Partial<FilterState>) => void;
  toggleModelFilter: (model: string) => void;
  toggleScenarioFilter: (scenario: string) => void;
  toggleSafetyScore: (score: number) => void;
  toggleAlignmentLevel: (level: string) => void;
  clearFilters: () => void;

  // Actions - UI
  toggleFilterPanel: () => void;
  toggleChatPanel: () => void;
  toggleChatMinimized: () => void;
  setCompareMode: (enabled: boolean) => void;
  setLoading: (loading: boolean) => void;

  // Actions - Chat
  addChatMessage: (message: Omit<ChatMessage, "id" | "timestamp">) => void;
  clearChatMessages: () => void;
  setChatBackendMode: (mode: "vertex" | "api_key" | "offline") => void;

  // Actions - Timeline
  setActiveTimelineEvent: (event: TimelineEvent | null) => void;
  setHighlightedMarkerTime: (time: number | null) => void;

  // Computed
  getFilteredExtractions: () => ExtractionMetadata[];
  getTimelineEvents: () => TimelineEvent[];
}

/**
 * Create the app store
 */
export const useAppStore = create<AppState>((set, get) => ({
  // Initial state
  trajectory: null,
  extractionsIndex: null,
  selectedExtractionId: null,

  playback: {
    isPlaying: false,
    currentFrame: 0,
    speed: 1,
    duration: 0,
  },

  filters: {
    models: new Set(),
    scenarios: new Set(),
    safetyScores: new Set(),
    alignmentLevels: new Set(),
    scoreRange: [0, 5],
  },

  isFilterPanelOpen: false,
  isChatPanelOpen: false,
  isChatMinimized: false,
  isCompareMode: false,
  isLoading: false,

  chatMessages: [],
  chatBackendMode: "offline",

  activeTimelineEvent: null,
  highlightedMarkerTime: null,

  // Trajectory actions
  setTrajectory: (trajectory) => {
    set({ trajectory });
    if (trajectory?.frames) {
      const duration =
        trajectory.frames[trajectory.frames.length - 1]?.time || 0;
      set((state) => ({
        playback: { ...state.playback, duration, currentFrame: 0 },
      }));
    }
  },

  setExtractionsIndex: (index) => set({ extractionsIndex: index }),
  setSelectedExtractionId: (id) => set({ selectedExtractionId: id }),

  // Playback actions
  play: () =>
    set((state) => ({ playback: { ...state.playback, isPlaying: true } })),

  pause: () =>
    set((state) => ({ playback: { ...state.playback, isPlaying: false } })),

  togglePlayback: () =>
    set((state) => ({
      playback: { ...state.playback, isPlaying: !state.playback.isPlaying },
    })),

  seek: (frame) =>
    set((state) => ({
      playback: {
        ...state.playback,
        currentFrame: Math.max(
          0,
          Math.min(frame, (state.trajectory?.frames?.length || 1) - 1)
        ),
      },
    })),

  setSpeed: (speed) =>
    set((state) => ({ playback: { ...state.playback, speed } })),

  setDuration: (duration) =>
    set((state) => ({ playback: { ...state.playback, duration } })),

  stepForward: () => {
    const { trajectory, playback } = get();
    const maxFrame = (trajectory?.frames?.length || 1) - 1;
    set({
      playback: {
        ...playback,
        currentFrame: Math.min(playback.currentFrame + 1, maxFrame),
        isPlaying: false,
      },
    });
  },

  stepBackward: () => {
    const { playback } = get();
    set({
      playback: {
        ...playback,
        currentFrame: Math.max(playback.currentFrame - 1, 0),
        isPlaying: false,
      },
    });
  },

  reset: () =>
    set((state) => ({
      playback: { ...state.playback, currentFrame: 0, isPlaying: false },
    })),

  // Filter actions
  setFilters: (filters) =>
    set((state) => ({ filters: { ...state.filters, ...filters } })),

  toggleModelFilter: (model) =>
    set((state) => {
      const models = new Set(state.filters.models);
      if (models.has(model)) {
        models.delete(model);
      } else {
        models.add(model);
      }
      return { filters: { ...state.filters, models } };
    }),

  toggleScenarioFilter: (scenario) =>
    set((state) => {
      const scenarios = new Set(state.filters.scenarios);
      if (scenarios.has(scenario)) {
        scenarios.delete(scenario);
      } else {
        scenarios.add(scenario);
      }
      return { filters: { ...state.filters, scenarios } };
    }),

  toggleSafetyScore: (score) =>
    set((state) => {
      const safetyScores = new Set(state.filters.safetyScores);
      if (safetyScores.has(score)) {
        safetyScores.delete(score);
      } else {
        safetyScores.add(score);
      }
      return { filters: { ...state.filters, safetyScores } };
    }),

  toggleAlignmentLevel: (level) =>
    set((state) => {
      const alignmentLevels = new Set(state.filters.alignmentLevels);
      if (alignmentLevels.has(level)) {
        alignmentLevels.delete(level);
      } else {
        alignmentLevels.add(level);
      }
      return { filters: { ...state.filters, alignmentLevels } };
    }),

  clearFilters: () =>
    set({
      filters: {
        models: new Set(),
        scenarios: new Set(),
        safetyScores: new Set(),
        alignmentLevels: new Set(),
        scoreRange: [0, 5],
      },
    }),

  // UI actions
  toggleFilterPanel: () =>
    set((state) => ({ isFilterPanelOpen: !state.isFilterPanelOpen })),

  toggleChatPanel: () =>
    set((state) => ({ isChatPanelOpen: !state.isChatPanelOpen })),

  toggleChatMinimized: () =>
    set((state) => ({ isChatMinimized: !state.isChatMinimized })),

  setCompareMode: (enabled) => set({ isCompareMode: enabled }),

  setLoading: (loading) => set({ isLoading: loading }),

  // Chat actions
  addChatMessage: (message) =>
    set((state) => ({
      chatMessages: [
        ...state.chatMessages,
        {
          ...message,
          id: `msg-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
          timestamp: new Date(),
        },
      ],
    })),

  clearChatMessages: () => set({ chatMessages: [] }),

  setChatBackendMode: (mode) => set({ chatBackendMode: mode }),

  // Timeline actions
  setActiveTimelineEvent: (event) => set({ activeTimelineEvent: event }),

  setHighlightedMarkerTime: (time) => set({ highlightedMarkerTime: time }),

  // Computed
  getFilteredExtractions: () => {
    const { extractionsIndex, filters } = get();
    if (!extractionsIndex) return [];

    return extractionsIndex.extractions.filter((ext) => {
      // Model filter
      if (filters.models.size > 0 && !filters.models.has(ext.model)) {
        return false;
      }

      // Scenario filter
      if (
        filters.scenarios.size > 0 &&
        !filters.scenarios.has(ext.scenario_id)
      ) {
        return false;
      }

      // Safety score filter
      if (filters.safetyScores.size > 0) {
        const score = Math.round(ext.judge?.safety_score || 0);
        if (!filters.safetyScores.has(score)) {
          return false;
        }
      }

      // Alignment level filter
      if (filters.alignmentLevels.size > 0) {
        const level = ext.judge?.alignment_level || "";
        if (!filters.alignmentLevels.has(level)) {
          return false;
        }
      }

      // Score range filter
      const score = ext.judge?.composite_score || 0;
      if (score < filters.scoreRange[0] || score > filters.scoreRange[1]) {
        return false;
      }

      return true;
    });
  },

  getTimelineEvents: () => {
    const { trajectory } = get();
    if (!trajectory) return [];

    // Prefer timeline_events from trajectory
    if (trajectory.timeline_events?.length) {
      return trajectory.timeline_events;
    }

    // Otherwise return empty (could reconstruct from frames here)
    return [];
  },
}));

/**
 * Selector hooks for specific state slices
 */
export const usePlayback = () => useAppStore((state) => state.playback);
export const useTrajectory = () => useAppStore((state) => state.trajectory);
export const useFilters = () => useAppStore((state) => state.filters);
export const useChatMessages = () => useAppStore((state) => state.chatMessages);
