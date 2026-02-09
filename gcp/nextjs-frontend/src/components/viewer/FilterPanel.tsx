"use client";

import React, { useMemo } from "react";
import { X, Filter, RotateCcw } from "lucide-react";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/useAppStore";

interface FilterPanelProps {
  className?: string;
}

/**
 * Checkbox item for filter lists
 */
interface CheckboxItemProps {
  label: string;
  checked: boolean;
  onChange: () => void;
  count?: number;
  color?: string;
}

function CheckboxItem({
  label,
  checked,
  onChange,
  count,
  color,
}: CheckboxItemProps) {
  return (
    <label className="flex items-center gap-2.5 py-1.5 cursor-pointer group">
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="w-4 h-4 rounded border-white/30 bg-transparent text-[var(--color-accent-primary)] focus:ring-[var(--color-accent-primary)] focus:ring-offset-0 cursor-pointer"
      />
      {color && (
        <span
          className="w-2.5 h-2.5 rounded-full flex-shrink-0"
          style={{ backgroundColor: color }}
        />
      )}
      <span className="text-sm text-white/80 group-hover:text-white transition-colors flex-1">
        {label}
      </span>
      {count !== undefined && (
        <span className="text-xs text-white/40">{count}</span>
      )}
    </label>
  );
}

/**
 * Section header with optional all/none buttons
 */
interface SectionHeaderProps {
  title: string;
  onSelectAll?: () => void;
  onSelectNone?: () => void;
}

function SectionHeader({
  title,
  onSelectAll,
  onSelectNone,
}: SectionHeaderProps) {
  return (
    <div className="flex items-center justify-between mb-3">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-white/50">
        {title}
      </h3>
      {onSelectAll && onSelectNone && (
        <div className="flex gap-2">
          <button
            onClick={onSelectAll}
            className="text-xs text-white/40 hover:text-white transition-colors"
          >
            All
          </button>
          <span className="text-white/20">/</span>
          <button
            onClick={onSelectNone}
            className="text-xs text-white/40 hover:text-white transition-colors"
          >
            None
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * Safety score button grid
 */
interface SafetyScoreGridProps {
  selectedScores: Set<number>;
  onToggle: (score: number) => void;
}

function SafetyScoreGrid({ selectedScores, onToggle }: SafetyScoreGridProps) {
  const scores = [1, 2, 3, 4, 5];

  return (
    <div className="flex gap-2">
      {scores.map((score) => (
        <button
          key={score}
          onClick={() => onToggle(score)}
          className={cn(
            "w-10 h-10 rounded-lg font-semibold transition-all duration-150",
            "border text-sm",
            selectedScores.has(score)
              ? "bg-[var(--color-accent-primary)] border-[var(--color-accent-primary)] text-black"
              : "bg-white/5 border-white/10 text-white/60 hover:bg-white/10 hover:text-white"
          )}
        >
          {score}
        </button>
      ))}
    </div>
  );
}

/**
 * Main filter panel component
 */
export function FilterPanel({ className }: FilterPanelProps) {
  const {
    extractionsIndex,
    filters,
    isFilterPanelOpen,
    toggleFilterPanel,
    toggleModelFilter,
    toggleScenarioFilter,
    toggleSafetyScore,
    toggleAlignmentLevel,
    clearFilters,
    setFilters,
    getFilteredExtractions,
  } = useAppStore();

  // Extract unique values from extractions
  const { models, scenarios, alignmentLevels, modelCounts, scenarioCounts } =
    useMemo(() => {
      if (!extractionsIndex) {
        return {
          models: [] as string[],
          scenarios: [] as string[],
          alignmentLevels: [] as string[],
          modelCounts: {} as Record<string, number>,
          scenarioCounts: {} as Record<string, number>,
        };
      }

      const modelSet = new Set<string>();
      const scenarioSet = new Set<string>();
      const alignmentSet = new Set<string>();
      const mCounts: Record<string, number> = {};
      const sCounts: Record<string, number> = {};

      for (const ext of extractionsIndex.extractions) {
        modelSet.add(ext.model);
        scenarioSet.add(ext.scenario_id);
        if (ext.judge?.alignment_level) {
          alignmentSet.add(ext.judge.alignment_level);
        }
        mCounts[ext.model] = (mCounts[ext.model] || 0) + 1;
        sCounts[ext.scenario_id] = (sCounts[ext.scenario_id] || 0) + 1;
      }

      return {
        models: Array.from(modelSet).sort(),
        scenarios: Array.from(scenarioSet).sort(),
        alignmentLevels: Array.from(alignmentSet).sort(),
        modelCounts: mCounts,
        scenarioCounts: sCounts,
      };
    }, [extractionsIndex]);

  // Model colors (matching original)
  const modelColors: Record<string, string> = {
    "gpt-5": "#6366F1",
    "gemini-2.5-pro": "#14B8A6",
    "gemini-robotics-er-1.5-preview": "#F43F5E",
    "kimi-k2.5": "#F59E0B",
  };

  // Filtered count
  const filteredExtractions = getFilteredExtractions();
  const totalCount = extractionsIndex?.total_count || 0;
  const filteredCount = filteredExtractions.length;

  // Check if any filters are active
  const hasActiveFilters =
    filters.models.size > 0 ||
    filters.scenarios.size > 0 ||
    filters.safetyScores.size > 0 ||
    filters.alignmentLevels.size > 0;

  if (!isFilterPanelOpen) {
    return null;
  }

  return (
    <div
      className={cn(
        "fixed top-[var(--header-height)] right-0 bottom-[var(--playback-bar-height)]",
        "w-[360px] bg-[var(--color-bg-surface-2)] border-l border-white/10",
        "flex flex-col z-[var(--z-fixed)]",
        "animate-[slideInFromRight_0.25s_ease-out]",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-white/60" />
          <h2 className="font-semibold text-white">Filters</h2>
        </div>
        <button
          onClick={toggleFilterPanel}
          className="p-1.5 rounded-md hover:bg-white/10 transition-colors"
          title="Close (Escape)"
        >
          <X className="w-5 h-5 text-white/60" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-6">
        {/* Models */}
        <div>
          <SectionHeader
            title="Models"
            onSelectAll={() =>
              setFilters({ models: new Set(models) })
            }
            onSelectNone={() => setFilters({ models: new Set() })}
          />
          <div className="space-y-1">
            {models.map((model) => (
              <CheckboxItem
                key={model}
                label={model}
                checked={filters.models.has(model)}
                onChange={() => toggleModelFilter(model)}
                count={modelCounts[model]}
                color={modelColors[model]}
              />
            ))}
          </div>
        </div>

        {/* Scenarios */}
        <div>
          <SectionHeader
            title="Scenarios"
            onSelectAll={() =>
              setFilters({ scenarios: new Set(scenarios) })
            }
            onSelectNone={() => setFilters({ scenarios: new Set() })}
          />
          <div className="space-y-1">
            {scenarios.map((scenario) => (
              <CheckboxItem
                key={scenario}
                label={scenario}
                checked={filters.scenarios.has(scenario)}
                onChange={() => toggleScenarioFilter(scenario)}
                count={scenarioCounts[scenario]}
              />
            ))}
          </div>
        </div>

        {/* Safety Score */}
        <div>
          <SectionHeader title="Safety Score" />
          <SafetyScoreGrid
            selectedScores={filters.safetyScores}
            onToggle={toggleSafetyScore}
          />
        </div>

        {/* Alignment Level */}
        {alignmentLevels.length > 0 && (
          <div>
            <SectionHeader title="Alignment Level" />
            <div className="space-y-1">
              {alignmentLevels.map((level) => (
                <CheckboxItem
                  key={level}
                  label={level}
                  checked={filters.alignmentLevels.has(level)}
                  onChange={() => toggleAlignmentLevel(level)}
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-white/10 space-y-3">
        {/* Result count */}
        <div className="text-sm text-white/60">
          <span className="text-white font-medium">{filteredCount}</span>
          <span> / {totalCount} runs</span>
        </div>

        {/* Clear button */}
        {hasActiveFilters && (
          <button
            onClick={clearFilters}
            className="flex items-center gap-2 w-full px-3 py-2 rounded-md bg-white/5 hover:bg-white/10 text-white/60 hover:text-white transition-colors text-sm"
          >
            <RotateCcw className="w-4 h-4" />
            Clear all filters
          </button>
        )}
      </div>
    </div>
  );
}

export default FilterPanel;
