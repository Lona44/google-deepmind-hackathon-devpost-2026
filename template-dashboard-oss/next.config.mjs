/** @type {import('next').NextConfig} */

const nextConfig = {
  async rewrites() {
    // Only proxy viewer-app in local dev (requires MuJoCo WASM server)
    if (process.env.NODE_ENV === "development") {
      return [
        {
          source: "/viewer-app",
          destination: "http://localhost:5500/",
        },
        {
          source: "/viewer-app/:path*",
          destination: "http://localhost:5500/:path*",
        },
      ];
    }
    return [];
  },
};

export default nextConfig;
