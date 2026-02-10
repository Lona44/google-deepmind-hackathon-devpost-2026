import { RiArrowRightUpLine, RiRobot2Line } from "@remixicon/react"
import Link from "next/link"

const navigation = {
  experiment: [
    { name: "Viewer", href: "/viewer", external: false },
    { name: "Dashboard", href: "/overview", external: false },
    { name: "Scenarios", href: "#", external: false },
    { name: "Results", href: "#", external: false },
  ],
  resources: [
    { name: "GitHub", href: "https://github.com/Lona44/google-deepmind-hackathon-devpost-2026", external: true },
    { name: "MuJoCo Docs", href: "https://mujoco.readthedocs.io", external: true },
    { name: "Gemini API", href: "https://ai.google.dev", external: true },
    { name: "Inspect AI", href: "https://inspect.ai-safety-institute.org.uk", external: true },
  ],
  research: [
    { name: "Methodology", href: "#", external: false },
    { name: "Safety Metrics", href: "#", external: false },
    { name: "Model Comparison", href: "#", external: false },
    { name: "Publications", href: "#", external: false },
  ],
  project: [
    { name: "About", href: "#", external: false },
    { name: "Contributing", href: "https://github.com/Lona44/google-deepmind-hackathon-devpost-2026", external: true },
    { name: "License", href: "#", external: false },
    { name: "Contact", href: "https://www.approxiomresearch.com/resume", external: true },
  ],
}

export default function Footer() {
  return (
    <footer id="footer">
      <div className="mx-auto max-w-6xl px-3 pb-8 pt-16 sm:pt-24 lg:pt-32">
        <div className="xl:grid xl:grid-cols-3 xl:gap-20">
          <div className="space-y-8">
            <div className="flex items-center gap-2">
              <RiRobot2Line className="size-8 text-indigo-500" />
              <span className="text-xl font-bold tracking-tight text-gray-900 dark:text-gray-50">
                G1 Alignment
              </span>
            </div>
            <p className="text-sm leading-6 text-gray-600 dark:text-gray-400">
              Studying whether AI prioritizes safety over efficiency in embodied
              settings. Built with MuJoCo + Gemini 3.
            </p>
          </div>
          <div className="mt-16 grid grid-cols-1 gap-14 sm:gap-8 md:grid-cols-2 xl:col-span-2 xl:mt-0">
            <div className="grid grid-cols-2 gap-8">
              <div>
                <h3 className="text-sm font-semibold leading-6 text-gray-900 dark:text-gray-50">
                  Experiment
                </h3>
                <ul
                  role="list"
                  className="mt-6 space-y-4"
                  aria-label="Quick links Experiment"
                >
                  {navigation.experiment.map((item) => (
                    <li key={item.name} className="w-fit">
                      <Link
                        className="flex rounded-md text-sm text-gray-500 transition hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50"
                        href={item.href}
                        target={item.external ? "_blank" : undefined}
                        rel={item.external ? "noopener noreferrer" : undefined}
                      >
                        <span>{item.name}</span>
                        {item.external && (
                          <div className="ml-1 aspect-square size-3 rounded-full bg-gray-100 p-px dark:bg-gray-500/20">
                            <RiArrowRightUpLine
                              aria-hidden="true"
                              className="size-full shrink-0 text-gray-900 dark:text-gray-300"
                            />
                          </div>
                        )}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h3 className="text-sm font-semibold leading-6 text-gray-900 dark:text-gray-50">
                  Resources
                </h3>
                <ul
                  role="list"
                  className="mt-6 space-y-4"
                  aria-label="Quick links Resources"
                >
                  {navigation.resources.map((item) => (
                    <li key={item.name} className="w-fit">
                      <Link
                        className="flex rounded-md text-sm text-gray-500 transition hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50"
                        href={item.href}
                        target={item.external ? "_blank" : undefined}
                        rel={item.external ? "noopener noreferrer" : undefined}
                      >
                        <span>{item.name}</span>
                        {item.external && (
                          <div className="ml-0.5 aspect-square size-3 rounded-full bg-gray-100 p-px dark:bg-gray-500/20">
                            <RiArrowRightUpLine
                              aria-hidden="true"
                              className="size-full shrink-0 text-gray-900 dark:text-gray-300"
                            />
                          </div>
                        )}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-8">
              <div>
                <h3 className="text-sm font-semibold leading-6 text-gray-900 dark:text-gray-50">
                  Research
                </h3>
                <ul
                  role="list"
                  className="mt-6 space-y-4"
                  aria-label="Quick links Research"
                >
                  {navigation.research.map((item) => (
                    <li key={item.name} className="w-fit">
                      <Link
                        className="flex rounded-md text-sm text-gray-500 transition hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50"
                        href={item.href}
                        target={item.external ? "_blank" : undefined}
                        rel={item.external ? "noopener noreferrer" : undefined}
                      >
                        <span>{item.name}</span>
                        {item.external && (
                          <div className="ml-1 aspect-square size-3 rounded-full bg-gray-100 p-px dark:bg-gray-500/20">
                            <RiArrowRightUpLine
                              aria-hidden="true"
                              className="size-full shrink-0 text-gray-900 dark:text-gray-300"
                            />
                          </div>
                        )}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h3 className="text-sm font-semibold leading-6 text-gray-900 dark:text-gray-50">
                  Project
                </h3>
                <ul
                  role="list"
                  className="mt-6 space-y-4"
                  aria-label="Quick links Project"
                >
                  {navigation.project.map((item) => (
                    <li key={item.name} className="w-fit">
                      <Link
                        className="flex rounded-md text-sm text-gray-500 transition hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-50"
                        href={item.href}
                        target={item.external ? "_blank" : undefined}
                        rel={item.external ? "noopener noreferrer" : undefined}
                      >
                        <span>{item.name}</span>
                        {item.external && (
                          <div className="ml-1 aspect-square size-3 rounded-full bg-gray-100 p-px dark:bg-gray-500/20">
                            <RiArrowRightUpLine
                              aria-hidden="true"
                              className="size-full shrink-0 text-gray-900 dark:text-gray-300"
                            />
                          </div>
                        )}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
        <div className="mt-16 flex flex-col items-center justify-between gap-4 border-t border-gray-200 pt-8 sm:mt-20 sm:flex-row lg:mt-24 dark:border-gray-800">
          <p className="text-sm leading-5 text-gray-500 dark:text-gray-400">
            &copy; {new Date().getFullYear()} G1 Alignment Experiment. Built for
            the Gemini 3 Hackathon.
          </p>
          <div className="rounded-full border border-gray-200 py-1 pl-1 pr-2 dark:border-gray-800">
            <div className="flex items-center gap-1.5">
              <div className="relative size-4 shrink-0">
                <div className="absolute inset-[1px] rounded-full bg-emerald-500/20 dark:bg-emerald-600/20" />
                <div className="absolute inset-1 rounded-full bg-emerald-600 dark:bg-emerald-500" />
              </div>
              <span className="text-xs text-gray-700 dark:text-gray-50">
                Experiment running
              </span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}
