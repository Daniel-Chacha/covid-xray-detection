import type { NextConfig } from "next";

import { normalizeBasePath } from "./lib/paths";

const nextConfig: NextConfig = {
  // Fully static: no server, no API routes. `next build` emits `out/`, which
  // deploys identically to Vercel, Netlify or GitHub Pages.
  output: "export",
  // next/image's optimiser requires a server. It is also undesirable here: the
  // gallery PNGs are already exactly 224x224, the model's input size, and
  // resampling them would silently shift pixels off the training distribution.
  images: { unoptimized: true },
  // Sub-path hosting (GitHub Pages under `/<repo>/`) needs BOTH halves:
  // `assetPath()` prefixes everything under `public/`, and this prefixes what
  // Next emits itself — the `/_next/...` script tags, which no application
  // code touches. Without it a sub-path deploy loads no JavaScript at all.
  //
  // Both halves must read the SAME env var through the SAME normaliser, which
  // is why `normalizeBasePath` is exported from `lib/paths.ts` rather than
  // duplicated here: Next hard-errors on a missing leading `/` (E105) and on a
  // trailing `/` (E39), so the two definitions must not be able to drift.
  //
  // The contract is a base *path*, not an origin. `normalizeBasePath` prepends
  // `/` to anything lacking one, so `https://cdn.example.com` would silently
  // become `/https://cdn.example.com`. Pass `/covid-xray-detection`, not a URL.
  basePath: normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH),
};

export default nextConfig;
