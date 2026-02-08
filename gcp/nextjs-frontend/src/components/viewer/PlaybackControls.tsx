"use client";

import React, { useCallback, useEffect } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  Eye,
  ChevronDown,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatTime } from "@/lib/utils";
import { Timeline } from "./Timeline";
import { useAppStore } from "@/store/useAppStore";
import type { TimelineEvent } from "@/types/trajectory";

interface PlaybackControlsProps {
  className?: string;
}

const SPEED_OPTIONS = [0.25, 0.5, 1, 2, 4];

/**
 * Icon button component for playback controls
 */
interface IconButtonProps {
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
  title?: string;
  children: React.ReactNode;
  className?: string;
}

function IconButton({
  onClick,
  disabled,
  active,
  title,
  children,
  className,
}: IconButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={cn(
        "p-2 rounded-md transition-all duration-150",
        "text-white/70 hover:text-white hover:bg-white/10",
        "disabled:opacity-40 disabled:cursor-not-allowed",
        active && "text-[var(--color-accent-primary)] bg-white/10",
        className
      )}
    >
      {children}
    </button>
  );
}

/**
 * Speed selector dropdown
 */
interface SpeedSelectorProps {
  speed: number;
  onSpeedChange: (speed: number) => void;
}

function SpeedSelector({ speed, onSpeedChange }: SpeedSelectorProps) {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1 px-3 py-1.5 rounded-md bg-white/10 text-white/80 hover:bg-white/15 hover:text-white transition-colors text-sm font-medium"
      >
        {speed}x
        <ChevronDown className="w-4 h-4" />
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-[99]"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute bottom-full mb-1 left-0 bg-[var(--color-bg-surface-3)] border border-white/10 rounded-md shadow-lg z-[100] overflow-hidden min-w-[80px]">
            {SPEED_OPTIONS.map((s) => (
              <button
                key={s}
                onClick={() => {
                  onSpeedChange(s);
                  setIsOpen(false);
                }}
                className={cn(
                  "w-full px-4 py-2 text-left text-sm hover:bg-white/10 transition-colors",
                  s === speed
                    ? "text-[var(--color-accent-primary)] bg-white/5"
                    : "text-white/80"
                )}
              >
                {s}x
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

/**
 * Playback controls bar component
 */
export function PlaybackControls({ className }: PlaybackControlsProps) {
  const {
    trajectory,
    playback,
    play,
    pause,
    togglePlayback,
    seek,
    setSpeed,
    stepForward,
    stepBackward,
    reset,
    setActiveTimelineEvent,
  } = useAppStore();

  const [followMode, setFollowMode] = React.useState(false);

  // Calculate current time and progress
  const currentFrame = trajectory?.frames?.[playback.currentFrame];
  const currentTime = currentFrame?.time || 0;
  const totalFrames = trajectory?.frames?.length || 0;
  const progress = totalFrames > 1 ? playback.currentFrame / (totalFrames - 1) : 0;

  // Get timeline events
  const timelineEvents = useAppStore((state) => state.getTimelineEvents());

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      switch (e.key) {
        case " ":
          e.preventDefault();
          togglePlayback();
          break;
        case "ArrowLeft":
          e.preventDefault();
          stepBackward();
          break;
        case "ArrowRight":
          e.preventDefault();
          stepForward();
          break;
        case "[":
          e.preventDefault();
          const currentSpeedIndex = SPEED_OPTIONS.indexOf(playback.speed);
          if (currentSpeedIndex > 0) {
            setSpeed(SPEED_OPTIONS[currentSpeedIndex - 1]);
          }
          break;
        case "]":
          e.preventDefault();
          const nextSpeedIndex = SPEED_OPTIONS.indexOf(playback.speed);
          if (nextSpeedIndex < SPEED_OPTIONS.length - 1) {
            setSpeed(SPEED_OPTIONS[nextSpeedIndex + 1]);
          }
          break;
        case "r":
        case "R":
          e.preventDefault();
          reset();
          break;
        case "f":
        case "F":
          if (!e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            setFollowMode(!followMode);
          }
          break;
        case "j":
        case "J":
          e.preventDefault();
          // Jump to next event
          jumpToNextEvent();
          break;
        case "k":
        case "K":
          e.preventDefault();
          // Jump to previous event
          jumpToPrevEvent();
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    togglePlayback,
    stepForward,
    stepBackward,
    setSpeed,
    reset,
    playback.speed,
    followMode,
  ]);

  // Jump to next timeline event
  const jumpToNextEvent = useCallback(() => {
    if (!timelineEvents.length) return;
    const nextEvent = timelineEvents.find(
      (e) => e.frame_index > playback.currentFrame
    );
    if (nextEvent) {
      seek(nextEvent.frame_index);
      setActiveTimelineEvent(nextEvent);
    }
  }, [timelineEvents, playback.currentFrame, seek, setActiveTimelineEvent]);

  // Jump to previous timeline event
  const jumpToPrevEvent = useCallback(() => {
    if (!timelineEvents.length) return;
    const prevEvents = timelineEvents.filter(
      (e) => e.frame_index < playback.currentFrame
    );
    if (prevEvents.length) {
      const prevEvent = prevEvents[prevEvents.length - 1];
      seek(prevEvent.frame_index);
      setActiveTimelineEvent(prevEvent);
    }
  }, [timelineEvents, playback.currentFrame, seek, setActiveTimelineEvent]);

  // Handle event click from timeline
  const handleEventClick = useCallback(
    (event: TimelineEvent) => {
      setActiveTimelineEvent(event);
      pause();
    },
    [setActiveTimelineEvent, pause]
  );

  if (!trajectory) {
    return null;
  }

  return (
    <div
      className={cn(
        "fixed bottom-0 left-0 right-0 h-[var(--playback-bar-height)]",
        "bg-[var(--color-bg-surface-1)] border-t border-white/10",
        "flex items-center px-4 gap-4 z-[var(--z-fixed)]",
        className
      )}
    >
      {/* Left controls */}
      <div className="flex items-center gap-1">
        <IconButton onClick={reset} title="Reset (R)">
          <RotateCcw className="w-5 h-5" />
        </IconButton>

        <IconButton onClick={stepBackward} title="Step back (←)">
          <SkipBack className="w-5 h-5" />
        </IconButton>

        <IconButton
          onClick={togglePlayback}
          title={playback.isPlaying ? "Pause (Space)" : "Play (Space)"}
          className="p-2.5"
        >
          {playback.isPlaying ? (
            <Pause className="w-6 h-6" />
          ) : (
            <Play className="w-6 h-6" />
          )}
        </IconButton>

        <IconButton onClick={stepForward} title="Step forward (→)">
          <SkipForward className="w-5 h-5" />
        </IconButton>
      </div>

      {/* Center - Timeline */}
      <div className="flex-1 flex items-center gap-4">
        {/* Current time */}
        <span className="font-mono text-sm text-white/60 min-w-[50px]">
          {formatTime(currentTime)}
        </span>

        {/* Timeline */}
        <Timeline
          events={timelineEvents}
          duration={playback.duration}
          currentTime={currentTime}
          progress={progress}
          onSeek={seek}
          onEventClick={handleEventClick}
          className="flex-1"
        />

        {/* Total time */}
        <span className="font-mono text-sm text-white/60 min-w-[50px]">
          {formatTime(playback.duration)}
        </span>
      </div>

      {/* Right controls */}
      <div className="flex items-center gap-3">
        <SpeedSelector speed={playback.speed} onSpeedChange={setSpeed} />

        <IconButton
          onClick={() => setFollowMode(!followMode)}
          active={followMode}
          title="Follow robot (F)"
        >
          <Eye className="w-5 h-5" />
        </IconButton>
      </div>
    </div>
  );
}

export default PlaybackControls;
