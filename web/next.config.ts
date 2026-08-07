import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Fully static: no server, no API routes. `next build` emits `out/`, which
  // deploys identically to Vercel, Netlify or GitHub Pages.
  output: "export",
  // next/image's optimiser requires a server. It is also undesirable here: the
  // gallery PNGs are already exactly 224x224, the model's input size, and
  // resampling them would silently shift pixels off the training distribution.
  images: { unoptimized: true },
};

export default nextConfig;
