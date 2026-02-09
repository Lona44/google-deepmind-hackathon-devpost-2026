/** @type {import('next').NextConfig} */

const nextConfig = {
  async rewrites() {
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
  },
};

export default nextConfig;
