import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
  // Allow preview subdomain to access dev server assets
  allowedDevOrigins: ["*.space-z.ai"],
};

export default nextConfig;
