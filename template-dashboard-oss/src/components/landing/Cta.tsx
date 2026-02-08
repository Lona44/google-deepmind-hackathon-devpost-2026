import Link from "next/link"
import { Button } from "../Button"

export default function Cta() {
  return (
    <section
      aria-labelledby="cta-title"
      className="mx-auto mb-20 mt-32 max-w-6xl p-1 px-2 sm:mt-56"
    >
      <div className="relative flex items-center justify-center">
        <div
          className="mask pointer-events-none absolute -z-10 select-none opacity-70"
          aria-hidden="true"
        >
          <div className="flex size-full flex-col gap-2">
            {Array.from({ length: 20 }, (_, idx) => (
              <div key={`outer-${idx}`}>
                <div className="flex size-full gap-2">
                  {Array.from({ length: 41 }, (_, idx2) => (
                    <div key={`inner-${idx}-${idx2}`}>
                      <div className="size-5 rounded-md shadow shadow-indigo-500/20 ring-1 ring-black/5 dark:shadow-indigo-500/20 dark:ring-white/5"></div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="max-w-4xl">
          <div className="flex flex-col items-center justify-center text-center">
            <div>
              <h3
                id="cta-title"
                className="inline-block bg-gradient-to-t from-gray-900 to-gray-800 bg-clip-text p-2 text-4xl font-bold tracking-tighter text-transparent md:text-6xl dark:from-gray-50 dark:to-gray-300"
              >
                See AI decision-making in action
              </h3>
              <p className="mx-auto mt-4 max-w-2xl text-gray-600 sm:text-lg dark:text-gray-400">
                Explore experiment runs, view AI reasoning traces, and compare
                self-assessment against ground truth metrics.
              </p>
            </div>
            <div className="mt-14 flex flex-col gap-3 sm:flex-row">
              <Button className="h-10 font-semibold" asChild>
                <Link href="/viewer">Launch Experiment Viewer</Link>
              </Button>
              <Button variant="secondary" className="h-10 font-semibold" asChild>
                <Link
                  href="https://github.com/Lona44/Gemini3-Hackathon-Project"
                  target="_blank"
                >
                  View Source Code
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
