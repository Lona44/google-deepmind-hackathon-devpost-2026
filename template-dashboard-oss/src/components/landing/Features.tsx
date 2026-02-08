import React from "react"
import {
  RiShieldCheckLine,
  RiAlertLine,
  RiEyeLine,
  RiRouteLine,
} from "@remixicon/react"

const stats = [
  {
    name: "LiDAR rays (360° coverage)",
    value: "36",
  },
  {
    name: "Retry attempts with learning",
    value: "5",
  },
  {
    name: "Sensors (Camera, LiDAR, IMU)",
    value: "3",
  },
]

const features = [
  {
    name: "Forbidden zone enforcement",
    description:
      "Configurable safety boundaries the AI must respect, with ground-truth violation tracking.",
    icon: RiShieldCheckLine,
  },
  {
    name: "Real-time sensor fusion",
    description:
      "Camera images, 36-ray LiDAR, and IMU data fed to Gemini at each decision point.",
    icon: RiEyeLine,
  },
  {
    name: "Multi-attempt learning",
    description:
      "5 retry attempts where the AI receives feedback on violations and adapts its strategy.",
    icon: RiRouteLine,
  },
  {
    name: "Honesty detection",
    description:
      "Compare AI self-assessment against simulation ground truth to detect rationalization.",
    icon: RiAlertLine,
  },
]

export default function Features() {
  return (
    <section
      aria-labelledby="features-title"
      className="mx-auto mt-44 w-full max-w-6xl px-3"
    >
      <span className="z-10 block w-fit rounded-lg border border-indigo-200/20 bg-indigo-50/50 px-3 py-1.5 font-semibold uppercase leading-4 tracking-tighter sm:text-sm dark:border-indigo-800/30 dark:bg-indigo-900/20">
        <span className="bg-gradient-to-b from-indigo-500 to-indigo-600 bg-clip-text text-transparent dark:from-indigo-200 dark:to-indigo-400">
          Experiment Design
        </span>
      </span>
      <h2
        id="features-title"
        className="mt-2 inline-block bg-gradient-to-br from-gray-900 to-gray-800 bg-clip-text py-2 text-4xl font-bold tracking-tighter text-transparent sm:text-6xl md:text-6xl dark:from-gray-50 dark:to-gray-300"
      >
        Architected for rigorous <br /> alignment testing
      </h2>
      <p className="mt-6 max-w-3xl text-lg leading-7 text-gray-600 dark:text-gray-400">
        The G1 Alignment Experiment uses MuJoCo physics simulation to create
        realistic scenarios where an AI must navigate safety-efficiency
        tradeoffs. Every decision is logged, compared against ground truth, and
        scored for alignment.
      </p>
      <dl className="mt-12 grid grid-cols-1 gap-y-8 md:grid-cols-3 md:border-y md:border-gray-200 md:py-14 dark:border-gray-800">
        {stats.map((stat, index) => (
          <React.Fragment key={index}>
            <div className="border-l-2 border-indigo-100 pl-6 md:border-l md:text-center lg:border-gray-200 lg:first:border-none dark:border-indigo-900 lg:dark:border-gray-800">
              <dd className="inline-block bg-gradient-to-t from-indigo-900 to-indigo-600 bg-clip-text text-5xl font-bold tracking-tight text-transparent lg:text-6xl dark:from-indigo-700 dark:to-indigo-400">
                {stat.value}
              </dd>
              <dt className="mt-1 text-gray-600 dark:text-gray-400">
                {stat.name}
              </dt>
            </div>
          </React.Fragment>
        ))}
      </dl>
      <dl className="mt-24 grid grid-cols-4 gap-10">
        {features.map((item) => (
          <div
            key={item.name}
            className="col-span-full sm:col-span-2 lg:col-span-1"
          >
            <div className="w-fit rounded-lg p-2 shadow-md shadow-indigo-400/30 ring-1 ring-black/5 dark:shadow-indigo-600/30 dark:ring-white/5">
              <item.icon
                aria-hidden="true"
                className="size-6 text-indigo-600 dark:text-indigo-400"
              />
            </div>
            <dt className="mt-6 font-semibold text-gray-900 dark:text-gray-50">
              {item.name}
            </dt>
            <dd className="mt-2 leading-7 text-gray-600 dark:text-gray-400">
              {item.description}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
