import Link from "next/link";
import { siteConfig } from "@/app/siteConfig";

export function Footer() {
  return (
    <footer className="border-t border-white/10 bg-[var(--color-bg-base)]">
      <div className="mx-auto max-w-6xl px-6 py-12">
        <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
          {/* Brand */}
          <div className="col-span-2 md:col-span-1">
            <span className="text-lg font-bold text-white">
              {siteConfig.name}
            </span>
            <p className="mt-3 text-sm leading-relaxed text-white/50">
              AI alignment research for embodied robotics.
            </p>
          </div>

          {/* Platform */}
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-white/40">
              Platform
            </h3>
            <ul className="mt-4 space-y-3">
              <li>
                <Link
                  href="/viewer"
                  className="text-sm text-white/60 hover:text-white"
                >
                  3D Viewer
                </Link>
              </li>
              <li>
                <Link
                  href="/#how-it-works"
                  className="text-sm text-white/60 hover:text-white"
                >
                  How it Works
                </Link>
              </li>
              <li>
                <Link
                  href="/#results"
                  className="text-sm text-white/60 hover:text-white"
                >
                  Results
                </Link>
              </li>
            </ul>
          </div>

          {/* Research */}
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-white/40">
              Research
            </h3>
            <ul className="mt-4 space-y-3">
              <li>
                <a
                  href="https://github.com/Lona44/google-deepmind-hackathon-devpost-2026"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-white/60 hover:text-white"
                >
                  GitHub
                </a>
              </li>
              <li>
                <Link
                  href="/#about"
                  className="text-sm text-white/60 hover:text-white"
                >
                  About
                </Link>
              </li>
            </ul>
          </div>

          {/* Tech */}
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-white/40">
              Built With
            </h3>
            <ul className="mt-4 space-y-3">
              <li className="text-sm text-white/60">MuJoCo</li>
              <li className="text-sm text-white/60">Gemini 3</li>
              <li className="text-sm text-white/60">Inspect AI</li>
            </ul>
          </div>
        </div>

        <div className="mt-12 border-t border-white/10 pt-8">
          <p className="text-center text-xs text-white/30">
            G1 Alignment Experiment &middot; Google DeepMind Hackathon 2026
          </p>
        </div>
      </div>
    </footer>
  );
}
