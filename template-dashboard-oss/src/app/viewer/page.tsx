import Link from "next/link"
import {
  RiRobot2Line,
  RiArrowLeftLine,
  RiGithubLine,
  RiPlayLine,
} from "@remixicon/react"

export default function ViewerPage() {
  return (
    <div className="flex min-h-screen flex-col bg-gray-950 text-gray-50">
      {/* Header */}
      <header className="border-b border-gray-800/50 bg-gray-950/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="flex items-center gap-2 text-sm text-gray-400 transition hover:text-gray-100"
            >
              <RiArrowLeftLine className="size-4" />
              Back
            </Link>
            <div className="h-4 w-px bg-gray-800" />
            <div className="flex items-center gap-2 font-semibold">
              <RiRobot2Line className="size-5 text-emerald-400" />
              <span>Experiment Viewer</span>
            </div>
          </div>
          <a
            href="https://github.com/Lona44/Gemini3-Hackathon-Project"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 transition hover:text-gray-100"
          >
            <RiGithubLine className="size-5" />
          </a>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 flex-col lg:flex-row">
        {/* Left panel - 3D viewport placeholder */}
        <div className="flex flex-1 flex-col">
          <div className="flex flex-1 items-center justify-center border-b border-gray-800/50 bg-gray-900/30 p-8 lg:border-b-0 lg:border-r">
            <div className="text-center">
              <div className="mx-auto mb-6 flex size-20 items-center justify-center rounded-2xl border border-gray-800 bg-gray-900">
                <RiPlayLine className="size-8 text-emerald-400" />
              </div>
              <h2 className="mb-2 text-xl font-semibold">3D Viewport</h2>
              <p className="mx-auto max-w-sm text-sm text-gray-400">
                Experiment playback will render here. Load an experiment run to
                see the G1 robot navigating through the scenario.
              </p>
            </div>
          </div>

          {/* Playback controls */}
          <div className="border-t border-gray-800/50 px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <button className="rounded-lg border border-gray-700 p-2 text-gray-400 transition hover:border-gray-600 hover:text-gray-100">
                  <RiPlayLine className="size-4" />
                </button>
                <span className="text-xs text-gray-500">00:00 / 00:00</span>
              </div>
              <div className="h-1 flex-1 mx-4 rounded-full bg-gray-800">
                <div className="h-1 w-0 rounded-full bg-emerald-500" />
              </div>
              <span className="text-xs text-gray-500">Step 0 / 0</span>
            </div>
          </div>
        </div>

        {/* Right panel - Data */}
        <div className="flex w-full flex-col border-t border-gray-800/50 lg:w-96 lg:border-l lg:border-t-0">
          {/* Experiment selector */}
          <div className="border-b border-gray-800/50 p-4">
            <label className="mb-2 block text-xs font-medium uppercase tracking-wider text-gray-500">
              Experiment Run
            </label>
            <select className="w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-300 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500">
              <option>No experiments loaded</option>
            </select>
          </div>

          {/* Metrics */}
          <div className="border-b border-gray-800/50 p-4">
            <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-gray-500">
              Metrics
            </h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: "Safety Score", value: "--" },
                { label: "Violations", value: "--" },
                { label: "Goal Reached", value: "--" },
                { label: "Battery Used", value: "--" },
              ].map((m) => (
                <div
                  key={m.label}
                  className="rounded-lg border border-gray-800 bg-gray-900/50 p-3"
                >
                  <p className="text-xs text-gray-500">{m.label}</p>
                  <p className="text-lg font-semibold">{m.value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* AI Reasoning */}
          <div className="flex flex-1 flex-col p-4">
            <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-gray-500">
              AI Reasoning Trace
            </h3>
            <div className="flex flex-1 items-center justify-center rounded-lg border border-dashed border-gray-800 p-6">
              <p className="text-center text-sm text-gray-600">
                Select an experiment run to view the AI&apos;s reasoning at each
                waypoint.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
