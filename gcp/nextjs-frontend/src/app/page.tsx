import Link from "next/link";
import {
  Shield,
  Brain,
  BarChart3,
  Play,
  ArrowRight,
  Bot,
  Target,
  AlertTriangle,
  Zap,
} from "lucide-react";
import { Footer } from "@/components/ui/Footer";

function Hero() {
  return (
    <section className="relative overflow-hidden pt-32 pb-20 sm:pt-40 sm:pb-28">
      {/* Gradient background */}
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute left-1/2 top-0 h-[600px] w-[800px] -translate-x-1/2 rounded-full bg-[var(--color-accent-primary)]/10 blur-[120px]" />
      </div>

      <div className="mx-auto max-w-4xl px-6 text-center">
        {/* Badge */}
        <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-sm text-white/70 animate-fadeIn">
          <span className="inline-block h-2 w-2 rounded-full bg-[var(--color-accent-primary)]" />
          Google DeepMind Hackathon 2026
        </div>

        {/* Headline */}
        <h1
          className="text-4xl font-bold tracking-tight text-white sm:text-6xl lg:text-7xl animate-fadeInUp"
          style={{ animationDelay: "200ms", animationFillMode: "both" }}
        >
          Does AI prioritize{" "}
          <span className="bg-gradient-to-r from-[var(--color-accent-primary)] to-emerald-300 bg-clip-text text-transparent">
            safety over efficiency
          </span>
          ?
        </h1>

        {/* Subtitle */}
        <p
          className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-white/60 sm:text-xl animate-fadeInUp"
          style={{ animationDelay: "400ms", animationFillMode: "both" }}
        >
          An alignment research platform that puts AI models in control of a
          humanoid robot navigating hazardous obstacles. We measure whether they
          rationalize violations or maintain safety under pressure.
        </p>

        {/* CTAs */}
        <div
          className="mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row animate-fadeInUp"
          style={{ animationDelay: "600ms", animationFillMode: "both" }}
        >
          <Link
            href="/viewer"
            className="inline-flex items-center gap-2 rounded-lg bg-white px-6 py-3 text-sm font-semibold text-gray-900 shadow-lg transition-all hover:bg-gray-200 hover:shadow-xl"
          >
            <Play className="h-4 w-4" />
            Launch Experiment Viewer
          </Link>
          <a
            href="https://github.com/Lona44/google-deepmind-hackathon-devpost-2026"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border border-white/15 px-6 py-3 text-sm font-medium text-white/80 transition-all hover:border-white/30 hover:text-white"
          >
            View on GitHub
            <ArrowRight className="h-4 w-4" />
          </a>
        </div>
      </div>
    </section>
  );
}

function ResearchQuestions() {
  const questions = [
    {
      icon: Shield,
      title: "Safety vs Efficiency",
      description:
        "When resource pressure conflicts with safety constraints, does the AI take dangerous shortcuts?",
      color: "text-[var(--color-accent-primary)]",
      bgColor: "bg-[var(--color-accent-primary)]/10",
    },
    {
      icon: Brain,
      title: "Rationalization Detection",
      description:
        "Does the AI rationalize violations with plausible-sounding excuses, or honestly acknowledge them?",
      color: "text-[var(--color-accent-secondary)]",
      bgColor: "bg-[var(--color-accent-secondary)]/10",
    },
    {
      icon: BarChart3,
      title: "Self-Assessment Accuracy",
      description:
        "How accurate is the AI's self-assessment compared to its actual behavior in the simulation?",
      color: "text-[var(--color-accent-tertiary)]",
      bgColor: "bg-[var(--color-accent-tertiary)]/10",
    },
  ];

  return (
    <section id="about" className="py-20 sm:py-28">
      <div className="mx-auto max-w-6xl px-6">
        <div className="text-center">
          <p className="text-sm font-semibold uppercase tracking-wider text-[var(--color-accent-primary)]">
            Research Questions
          </p>
          <h2 className="mt-3 text-3xl font-bold text-white sm:text-4xl">
            Testing alignment in the physical world
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-white/50">
            Current AI alignment research focuses on text. But as LLMs control
            physical systems, new risks emerge &mdash; physical harm is
            irreversible.
          </p>
        </div>

        <div className="mt-16 grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
          {questions.map((q) => (
            <div
              key={q.title}
              className="group rounded-xl border border-white/10 bg-[var(--color-bg-surface-1)] p-8 transition-all hover:border-white/20 hover:bg-[var(--color-bg-surface-2)]"
            >
              <div
                className={`inline-flex rounded-lg p-3 ${q.bgColor}`}
              >
                <q.icon className={`h-6 w-6 ${q.color}`} />
              </div>
              <h3 className="mt-5 text-lg font-semibold text-white">
                {q.title}
              </h3>
              <p className="mt-3 text-sm leading-relaxed text-white/60">
                {q.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function HowItWorks() {
  const steps = [
    {
      step: "01",
      icon: Bot,
      title: "AI Controls Robot",
      description:
        "An LLM (Gemini 3, GPT-5, etc.) receives camera images, LiDAR data, and battery status from a Unitree G1 humanoid robot simulated in MuJoCo.",
    },
    {
      step: "02",
      icon: Target,
      title: "Navigate Under Pressure",
      description:
        "The robot must reach a charging station before battery depletion, but hazardous waste drums and forbidden zones block the efficient path.",
    },
    {
      step: "03",
      icon: AlertTriangle,
      title: "Safety vs Speed Tradeoff",
      description:
        "The AI must decide: take the safe route (slower, risk battery death) or cut through the danger zone (faster, risk barrel contact).",
    },
    {
      step: "04",
      icon: BarChart3,
      title: "Judge & Analyze",
      description:
        "An independent AI judge scores safety, honesty, and alignment. We compare reasoning traces with actual behavior to detect rationalization.",
    },
  ];

  return (
    <section
      id="how-it-works"
      className="border-y border-white/10 bg-[var(--color-bg-surface-0)] py-20 sm:py-28"
    >
      <div className="mx-auto max-w-6xl px-6">
        <div className="text-center">
          <p className="text-sm font-semibold uppercase tracking-wider text-[var(--color-accent-primary)]">
            Methodology
          </p>
          <h2 className="mt-3 text-3xl font-bold text-white sm:text-4xl">
            How the experiment works
          </h2>
        </div>

        <div className="mt-16 grid gap-12 sm:grid-cols-2">
          {steps.map((s) => (
            <div key={s.step} className="flex gap-5">
              <div className="flex-shrink-0">
                <div className="flex h-12 w-12 items-center justify-center rounded-full border border-white/10 bg-[var(--color-bg-surface-2)]">
                  <s.icon className="h-5 w-5 text-[var(--color-accent-primary)]" />
                </div>
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-white/30">
                  Step {s.step}
                </p>
                <h3 className="mt-1 text-lg font-semibold text-white">
                  {s.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-white/60">
                  {s.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function ComparisonTable() {
  const rows = [
    {
      challenge: "Safety violations",
      textAI: "Harmful content",
      embodiedAI: "Physical harm, property damage",
    },
    {
      challenge: "Deception",
      textAI: "Misleading text",
      embodiedAI: "Falsified sensor reports",
    },
    {
      challenge: "Pressure response",
      textAI: "Sycophantic answers",
      embodiedAI: "Unsafe shortcuts",
    },
    {
      challenge: "Irreversibility",
      textAI: "Can be corrected",
      embodiedAI: "Physical consequences are permanent",
    },
  ];

  return (
    <section id="results" className="py-20 sm:py-28">
      <div className="mx-auto max-w-4xl px-6">
        <div className="text-center">
          <p className="text-sm font-semibold uppercase tracking-wider text-[var(--color-accent-primary)]">
            Why This Matters
          </p>
          <h2 className="mt-3 text-3xl font-bold text-white sm:text-4xl">
            Text AI vs Embodied AI risks
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-white/50">
            Embodied AI introduces fundamentally different failure modes. Our
            platform is designed to surface these.
          </p>
        </div>

        <div className="mt-12 overflow-hidden rounded-xl border border-white/10">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-white/10 bg-[var(--color-bg-surface-1)]">
                <th className="px-6 py-4 font-semibold text-white/70">
                  Challenge
                </th>
                <th className="px-6 py-4 font-semibold text-white/70">
                  Text AI
                </th>
                <th className="px-6 py-4 font-semibold text-[var(--color-accent-danger)]">
                  Embodied AI
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.challenge}
                  className="border-b border-white/5 last:border-0"
                >
                  <td className="px-6 py-4 font-medium text-white">
                    {row.challenge}
                  </td>
                  <td className="px-6 py-4 text-white/50">{row.textAI}</td>
                  <td className="px-6 py-4 text-white/80">{row.embodiedAI}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function Cta() {
  return (
    <section className="border-t border-white/10 bg-[var(--color-bg-surface-0)] py-20 sm:py-28">
      <div className="mx-auto max-w-3xl px-6 text-center">
        <div className="inline-flex items-center gap-2 rounded-full bg-[var(--color-accent-primary)]/10 px-4 py-1.5 text-sm text-[var(--color-accent-primary)]">
          <Zap className="h-4 w-4" />
          Browser-based 3D playback
        </div>
        <h2 className="mt-6 text-3xl font-bold text-white sm:text-4xl">
          See AI decision-making in action
        </h2>
        <p className="mx-auto mt-4 max-w-xl text-white/50">
          Watch trajectory replays with AI reasoning traces, timeline markers,
          and alignment scores. Compare models side-by-side in the 3D viewer.
        </p>
        <div className="mt-10">
          <Link
            href="/viewer"
            className="inline-flex items-center gap-2 rounded-lg bg-[var(--color-accent-primary)] px-8 py-3.5 text-sm font-semibold text-black shadow-lg shadow-[var(--color-accent-primary)]/25 transition-all hover:bg-[var(--color-accent-primary-hover)] hover:shadow-xl"
          >
            <Play className="h-4 w-4" />
            Launch Experiment Viewer
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function HomePage() {
  return (
    <main className="min-h-screen overflow-x-hidden bg-[var(--color-bg-base)]">
      <Hero />
      <ResearchQuestions />
      <HowItWorks />
      <ComparisonTable />
      <Cta />
      <Footer />
    </main>
  );
}
