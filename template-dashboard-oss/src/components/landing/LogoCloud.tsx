import {
  RiRobot2Line,
  RiBrainLine,
  RiShieldCheckLine,
  RiEyeLine,
} from "@remixicon/react"

const tools = [
  { name: "MuJoCo", icon: RiRobot2Line },
  { name: "Gemini 3", icon: RiBrainLine },
  { name: "Inspect AI", icon: RiShieldCheckLine },
  { name: "Next.js + Tremor", icon: RiEyeLine },
]

export default function LogoCloud() {
  return (
    <section
      id="logo-cloud"
      aria-label="Technologies used"
      className="mt-24 flex animate-slide-up-fade flex-col items-center justify-center gap-y-6 text-center sm:mt-32"
      style={{ animationDuration: "1500ms" }}
    >
      <p className="text-lg font-medium tracking-tighter text-gray-800 dark:text-gray-200">
        Powered by leading AI & simulation technologies
      </p>
      <div className="grid grid-cols-2 gap-10 gap-y-4 text-gray-900 md:grid-cols-4 md:gap-x-20 dark:text-gray-200">
        {tools.map((tool) => (
          <div key={tool.name} className="flex w-28 items-center gap-2">
            <tool.icon className="size-5 text-indigo-500" />
            <span className="text-sm font-semibold">{tool.name}</span>
          </div>
        ))}
      </div>
    </section>
  )
}
