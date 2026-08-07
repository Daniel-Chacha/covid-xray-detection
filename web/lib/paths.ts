/**
 * Base-path handling for everything served out of `web/public/`.
 *
 * Next's `basePath` rewrites `next/link` and `next/router` URLs, but it does
 * NOT rewrite string literals pointing at `public/` assets — the Next docs say
 * so explicitly for `next/image` (`api-reference/config/next-config-js/
 * basePath.md`: "you will need to add the `basePath` in front of `src`").
 * `assetPrefix` is no help either: it only covers `/_next/static`, and its own
 * doc states that files in `public/` are excluded and that basePath is the
 * right lever for sub-path hosting.
 *
 * So a sub-path deploy (GitHub Pages under `/<repo>/`) needs the prefix applied
 * by hand, in one place. That place is here.
 *
 * To deploy under a sub-path, set ONE value — the `NEXT_PUBLIC_BASE_PATH`
 * environment variable at build time — and have `next.config.ts` derive
 * `basePath` from the same variable:
 *
 *   const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? ''
 *   const nextConfig: NextConfig = { basePath, output: 'export', ... }
 *
 * `NEXT_PUBLIC_`-prefixed variables are inlined into the client bundle at build
 * time, so this constant is a literal by the time it reaches the browser.
 */
export const BASE_PATH = (process.env.NEXT_PUBLIC_BASE_PATH ?? '').replace(/\/+$/, '')

/** Resolve a root-relative `public/` path against the deploy's base path. */
export function assetPath(path: string): string {
  return `${BASE_PATH}${path.startsWith('/') ? path : `/${path}`}`
}
