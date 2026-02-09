"use client"

export default function HeroImage() {
  return (
    <section aria-label="Hero demo of the G1 Alignment experiment" className="flow-root">
      <div className="rounded-2xl bg-slate-50/40 p-2 ring-1 ring-inset ring-slate-200/50 dark:bg-gray-900/70 dark:ring-white/10">
        <div className="rounded-xl bg-white ring-1 ring-slate-900/5 dark:bg-slate-950 dark:ring-white/15">
          <video
            autoPlay
            loop
            muted
            playsInline
            className="rounded-xl shadow-2xl dark:shadow-indigo-600/10"
            width={1200}
            height={860}
          >
            <source src="/videos/hero-demo.mp4" type="video/mp4" />
          </video>
        </div>
      </div>
    </section>
  )
}
