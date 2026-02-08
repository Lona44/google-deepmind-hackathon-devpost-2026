"use client";

import React, { useState, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import { formatTime } from "@/lib/utils";
import type { TimelineEvent, AlignmentSignal } from "@/types/trajectory";
import { SIGNAL_COLORS } from "@/types/trajectory";

interface TimelineProps {
  events: TimelineEvent[];
  duration: number;
  currentTime: number;
  progress: number;
  onSeek: (frameIndex: number) => void;
  onEventClick?: (event: TimelineEvent) => void;
  className?: string;
}

/**
 * Format signal name for display
 */
function formatSignal(signal: AlignmentSignal): string {
  const words = signal.toLowerCase().split("_");
  return words[0].charAt(0).toUpperCase() + words[0].slice(1);
}

/**
 * Get signal color class
 */
function getSignalColorClass(signal: AlignmentSignal): string {
  const classes: Record<AlignmentSignal, string> = {
    SAFETY_PRIORITIZED: "bg-green-500",
    EFFICIENCY_PRIORITIZED: "bg-red-500",
    RISK_ACKNOWLEDGED: "bg-orange-500",
    TRADEOFF: "bg-yellow-400",
    NEUTRAL: "bg-gray-500",
  };
  return classes[signal] || "bg-gray-500";
}

/**
 * Event marker tooltip component
 */
interface TooltipProps {
  event: TimelineEvent;
  style: React.CSSProperties;
}

function Tooltip({ event, style }: TooltipProps) {
  const groupedEvents = event._groupedEvents || [event];

  return (
    <div
      className="absolute z-[100] pointer-events-none animate-fadeIn"
      style={style}
    >
      <div className="bg-[rgba(20,20,24,0.95)] border border-white/15 rounded-xl px-5 py-4 shadow-lg -translate-x-1/2 whitespace-nowrap">
        {/* Meta info */}
        <div className="flex gap-2.5 items-center mb-2">
          <span className="font-mono text-xs text-white/50">
            {formatTime(event.time)}
          </span>
          {groupedEvents.length > 1 && (
            <span className="text-xs text-white/50 italic">
              {groupedEvents.length} events
            </span>
          )}
        </div>

        {/* Events list */}
        <div className="flex flex-col gap-3">
          {groupedEvents.map((evt, i) => (
            <div
              key={evt.id || i}
              className={cn(
                "p-2.5 bg-white/5 rounded-lg border-l-[3px]",
                i === 0
                  ? "border-l-[var(--color-accent-secondary)]"
                  : "border-l-white/30 opacity-90"
              )}
            >
              {/* Header */}
              <div className="flex items-center gap-2.5 mb-1">
                <span className="text-xl">{evt.icon}</span>
                <span className="text-base font-semibold">{evt.label}</span>
                {evt.signal && (
                  <span
                    className={cn(
                      "text-[11px] font-semibold px-2.5 py-1 rounded uppercase tracking-wide",
                      evt.signal === "SAFETY_PRIORITIZED" &&
                        "bg-green-500/30 text-green-300",
                      evt.signal === "EFFICIENCY_PRIORITIZED" &&
                        "bg-red-500/30 text-red-300",
                      evt.signal === "RISK_ACKNOWLEDGED" &&
                        "bg-orange-500/30 text-orange-300",
                      evt.signal === "TRADEOFF" &&
                        "bg-yellow-400/30 text-yellow-200",
                      evt.signal === "NEUTRAL" && "bg-gray-500/30 text-gray-300"
                    )}
                  >
                    {formatSignal(evt.signal)}
                  </span>
                )}
              </div>

              {/* Summary */}
              {evt.summary && (
                <div className="mt-1.5 text-sm text-white/80 max-w-[320px] whitespace-normal leading-relaxed">
                  {evt.summary}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/**
 * Individual timeline marker
 */
interface MarkerProps {
  event: TimelineEvent;
  position: number;
  isActive: boolean;
  onHover: (event: TimelineEvent | null) => void;
  onClick: () => void;
}

function Marker({ event, position, isActive, onHover, onClick }: MarkerProps) {
  const groupSize = event._groupedEvents?.length || 1;

  return (
    <div
      className={cn(
        "absolute bottom-0 cursor-pointer z-[3]",
        "transition-transform duration-150 ease-out",
        "hover:scale-[1.2]",
        isActive && "scale-[1.4] animate-[markerPulse_1s_ease-in-out_infinite]"
      )}
      style={{
        left: `${position}%`,
        transform: `translateX(-50%)${isActive ? " scale(1.4)" : ""}`,
      }}
      onClick={onClick}
      onMouseEnter={() => onHover(event)}
      onMouseLeave={() => onHover(null)}
      title={`${event.type} @ ${event.time.toFixed(1)}s`}
    >
      {/* Stem line */}
      <div className="absolute bottom-[-6px] left-1/2 w-[3px] h-[10px] bg-white/30 -translate-x-1/2" />

      {/* Icon */}
      <span
        className="text-[38px] block leading-none"
        style={{
          filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.9))",
        }}
      >
        {event.icon}
      </span>

      {/* Signal badge */}
      {event.signal && (
        <span
          className={cn(
            "absolute bottom-[-2px] right-[-2px] w-2 h-2 rounded-full",
            "border-[1.5px] border-black/50 shadow-sm",
            getSignalColorClass(event.signal)
          )}
          title={event.signal.replace(/_/g, " ")}
        />
      )}

      {/* Group badge */}
      {groupSize > 1 && (
        <span className="absolute top-[-6px] right-[-8px] min-w-[22px] h-[22px] bg-orange-500 text-black text-[13px] font-bold rounded-full flex items-center justify-center px-1 shadow-md">
          {groupSize}
        </span>
      )}
    </div>
  );
}

/**
 * Timeline component - Enhanced timeline with emoji event markers
 */
export function Timeline({
  events,
  duration,
  currentTime,
  progress,
  onSeek,
  onEventClick,
  className,
}: TimelineProps) {
  const [hoveredEvent, setHoveredEvent] = useState<TimelineEvent | null>(null);
  const [activeEventId, setActiveEventId] = useState<string | null>(null);

  // Group events by time slot and keep highest priority
  const groupedEvents = useMemo(() => {
    if (!events.length || !duration) return [];

    // Sort by time, then priority
    const sorted = [...events].sort(
      (a, b) => a.time - b.time || a.priority - b.priority
    );

    // Group by 0.5s slots
    const bySlot = new Map<number, TimelineEvent[]>();
    for (const evt of sorted) {
      const slot = Math.round(evt.time * 2) / 2;
      if (!bySlot.has(slot)) {
        bySlot.set(slot, []);
      }
      bySlot.get(slot)!.push(evt);
    }

    // For each slot, use highest priority but attach all events
    return Array.from(bySlot.values()).map((slotEvents) => {
      const primary = slotEvents[0];
      return {
        ...primary,
        _groupedEvents: slotEvents,
      };
    });
  }, [events, duration]);

  const handleMarkerClick = useCallback(
    (event: TimelineEvent) => {
      setActiveEventId(event.id);
      if (event.frame_index !== undefined) {
        onSeek(event.frame_index);
      }
      onEventClick?.(event);
    },
    [onSeek, onEventClick]
  );

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = Number(e.target.value);
      // Assuming we have total frames from parent
      const frameIndex = Math.round(value / 1000);
      onSeek(frameIndex);
    },
    [onSeek]
  );

  return (
    <div className={cn("relative flex items-center pt-8 overflow-visible", className)}>
      {/* Track */}
      <div className="relative w-full h-2 bg-white/15 rounded-full overflow-visible">
        {/* Progress bar */}
        <div
          className="absolute top-0 left-0 h-full bg-[var(--color-accent-primary)] rounded-full pointer-events-none z-[1] transition-[width] duration-50 linear"
          style={{ width: `${progress * 100}%` }}
        />

        {/* Markers container */}
        <div className="absolute -top-[52px] left-0 right-0 h-[52px] pointer-events-none z-10">
          {groupedEvents.map((event) => {
            const position = (event.time / duration) * 100;
            return (
              <Marker
                key={event.id}
                event={event}
                position={position}
                isActive={activeEventId === event.id}
                onHover={setHoveredEvent}
                onClick={() => handleMarkerClick(event)}
              />
            );
          })}
        </div>

        {/* Slider (invisible but interactive) */}
        <input
          type="range"
          min="0"
          max="1000"
          value={Math.round(progress * 1000)}
          onChange={handleSliderChange}
          className="absolute top-0 left-0 w-full h-full appearance-none bg-transparent cursor-pointer z-[4] m-0
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:w-[14px]
            [&::-webkit-slider-thumb]:h-[14px]
            [&::-webkit-slider-thumb]:bg-[var(--color-accent-primary)]
            [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:shadow-md
            [&::-webkit-slider-thumb]:transition-transform
            [&::-webkit-slider-thumb]:duration-150
            [&::-webkit-slider-thumb]:mt-[-3px]
            [&::-webkit-slider-thumb]:hover:scale-[1.2]
            [&::-webkit-slider-runnable-track]:h-1.5
            [&::-webkit-slider-runnable-track]:bg-transparent
            [&::-webkit-slider-runnable-track]:rounded-full
            [&::-moz-range-thumb]:w-[14px]
            [&::-moz-range-thumb]:h-[14px]
            [&::-moz-range-thumb]:bg-[var(--color-accent-primary)]
            [&::-moz-range-thumb]:rounded-full
            [&::-moz-range-thumb]:cursor-pointer
            [&::-moz-range-thumb]:shadow-md
            [&::-moz-range-thumb]:border-none
            [&::-moz-range-track]:h-1.5
            [&::-moz-range-track]:bg-transparent
            [&::-moz-range-track]:rounded-full"
        />
      </div>

      {/* Tooltip */}
      {hoveredEvent && (
        <Tooltip
          event={hoveredEvent}
          style={{
            left: `${(hoveredEvent.time / duration) * 100}%`,
            bottom: "32px",
          }}
        />
      )}
    </div>
  );
}

export default Timeline;
