"use client";

import React from "react";
import {
  RefreshCw,
  Filter,
  GitCompare,
  MessageSquare,
  ChevronDown,
  HelpCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/useAppStore";
import type { ExtractionMetadata } from "@/types/trajectory";

interface HeaderProps {
  className?: string;
}

/**
 * Extraction selector dropdown
 */
interface ExtractionSelectorProps {
  extractions: ExtractionMetadata[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRefresh: () => void;
}

function ExtractionSelector({
  extractions,
  selectedId,
  onSelect,
  onRefresh,
}: ExtractionSelectorProps) {
  const [isOpen, setIsOpen] = React.useState(false);

  // Group extractions by scenario
  const grouped = React.useMemo(() => {
    const groups: Record<string, ExtractionMetadata[]> = {};
    for (const ext of extractions) {
      if (!groups[ext.scenario_id]) {
        groups[ext.scenario_id] = [];
      }
      groups[ext.scenario_id].push(ext);
    }
    return groups;
  }, [extractions]);

  const selected = extractions.find((e) => e.id === selectedId);

  return (
    <div className="relative flex items-center gap-2">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[var(--color-bg-surface-2)] border border-white/10 hover:border-white/20 transition-colors min-w-[280px]"
      >
        <span className="text-sm text-white/80 flex-1 text-left truncate">
          {selected ? `${selected.scenario_id} - ${selected.model}` : "Select experiment..."}
        </span>
        <ChevronDown className="w-4 h-4 text-white/50" />
      </button>

      <button
        onClick={onRefresh}
        className="p-2 rounded-lg bg-[var(--color-bg-surface-2)] border border-white/10 hover:border-white/20 hover:bg-white/5 transition-colors"
        title="Refresh manifest"
      >
        <RefreshCw className="w-4 h-4 text-white/60" />
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-[99]"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute top-full left-0 mt-1 w-[400px] max-h-[400px] overflow-y-auto bg-[var(--color-bg-surface-3)] border border-white/10 rounded-lg shadow-xl z-[100]">
            {Object.entries(grouped).map(([scenario, exts]) => (
              <div key={scenario}>
                <div className="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-white/40 bg-black/20 sticky top-0">
                  {scenario}
                </div>
                {exts.map((ext) => (
                  <button
                    key={ext.id}
                    onClick={() => {
                      onSelect(ext.id);
                      setIsOpen(false);
                    }}
                    className={cn(
                      "w-full px-3 py-2.5 text-left hover:bg-white/5 transition-colors flex items-center gap-3",
                      ext.id === selectedId && "bg-white/10"
                    )}
                  >
                    <span className="flex-1 text-sm text-white/80 truncate">
                      {ext.model}
                    </span>
                    {ext.is_new && (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[var(--color-accent-primary)]/20 text-[var(--color-accent-primary)]">
                        NEW
                      </span>
                    )}
                    {ext.status === "aborted" && (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[var(--color-accent-warning)]/20 text-[var(--color-accent-warning)]">
                        ABORTED
                      </span>
                    )}
                    {ext.judge?.safety_score && (
                      <span className="text-xs text-white/50">
                        Safety: {ext.judge.safety_score.toFixed(1)}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

/**
 * Header button component
 */
interface HeaderButtonProps {
  onClick: () => void;
  active?: boolean;
  badge?: number;
  title?: string;
  children: React.ReactNode;
}

function HeaderButton({
  onClick,
  active,
  badge,
  title,
  children,
}: HeaderButtonProps) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={cn(
        "relative p-2 rounded-lg transition-colors",
        active
          ? "bg-[var(--color-accent-primary)]/20 text-[var(--color-accent-primary)]"
          : "hover:bg-white/10 text-white/60 hover:text-white"
      )}
    >
      {children}
      {badge !== undefined && badge > 0 && (
        <span className="absolute -top-1 -right-1 min-w-[18px] h-[18px] rounded-full bg-[var(--color-accent-primary)] text-black text-[10px] font-bold flex items-center justify-center px-1">
          {badge}
        </span>
      )}
    </button>
  );
}

/**
 * Header component
 */
export function Header({ className }: HeaderProps) {
  const {
    extractionsIndex,
    selectedExtractionId,
    isFilterPanelOpen,
    isChatPanelOpen,
    isCompareMode,
    filters,
    setSelectedExtractionId,
    toggleFilterPanel,
    toggleChatPanel,
    setCompareMode,
    getFilteredExtractions,
  } = useAppStore();

  const filteredExtractions = getFilteredExtractions();

  // Count active filters
  const activeFilterCount =
    filters.models.size +
    filters.scenarios.size +
    filters.safetyScores.size +
    filters.alignmentLevels.size;

  const handleRefresh = async () => {
    // TODO: Implement manifest refresh
    console.log("Refreshing manifest...");
  };

  const handleSelectExtraction = (id: string) => {
    setSelectedExtractionId(id);
    // TODO: Load trajectory data
  };

  return (
    <header
      className={cn(
        "fixed top-0 left-0 right-0 h-[var(--header-height)]",
        "bg-[var(--color-bg-surface-1)] border-b border-white/10",
        "flex items-center justify-between px-4",
        "z-[var(--z-sticky)]",
        className
      )}
    >
      {/* Left side - Title and selector */}
      <div className="flex items-center gap-4">
        <h1 className="text-lg font-semibold text-white">
          G1 Alignment Viewer
        </h1>

        <ExtractionSelector
          extractions={filteredExtractions}
          selectedId={selectedExtractionId}
          onSelect={handleSelectExtraction}
          onRefresh={handleRefresh}
        />
      </div>

      {/* Right side - Action buttons */}
      <div className="flex items-center gap-2">
        <HeaderButton
          onClick={toggleFilterPanel}
          active={isFilterPanelOpen}
          badge={activeFilterCount}
          title="Filters (F)"
        >
          <Filter className="w-5 h-5" />
        </HeaderButton>

        <HeaderButton
          onClick={() => setCompareMode(!isCompareMode)}
          active={isCompareMode}
          title="Compare mode"
        >
          <GitCompare className="w-5 h-5" />
        </HeaderButton>

        <HeaderButton
          onClick={toggleChatPanel}
          active={isChatPanelOpen}
          title="Chat (G)"
        >
          <MessageSquare className="w-5 h-5" />
        </HeaderButton>

        <HeaderButton onClick={() => {}} title="Help (?)">
          <HelpCircle className="w-5 h-5" />
        </HeaderButton>
      </div>
    </header>
  );
}

export default Header;
