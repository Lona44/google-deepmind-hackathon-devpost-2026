"use client"

export default function ViewerPage() {
  return (
    <div className="relative h-screen w-full overflow-hidden bg-gray-950">
      {/* Full-screen viewer iframe */}
      <iframe
        src="http://localhost:5500"
        className="h-full w-full border-0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; cross-origin-isolated"
        title="G1 Alignment Viewer"
      />
    </div>
  )
}
