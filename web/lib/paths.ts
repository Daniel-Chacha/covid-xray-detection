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
 * `basePath` from it THROUGH `normalizeBasePath`, so both agree exactly:
 *
 *   import { normalizeBasePath } from './lib/paths'
 *   const nextConfig: NextConfig = {
 *     basePath: normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH),
 *     output: 'export',
 *     images: { unoptimized: true },
 *   }
 *
 * Passing the raw env var to `basePath` is NOT safe: Next validates it and
 * hard-errors on a missing leading slash (`next/dist/server/config.js`, error
 * E105) and on a trailing slash (E39). `normalizeBasePath` returns a value that
 * satisfies both rules, so the loose inputs it tolerates here cannot produce a
 * build that fails there.
 *
 * `NEXT_PUBLIC_`-prefixed variables are inlined into the client bundle at build
 * time, so `BASE_PATH` is a literal by the time it reaches the browser.
 */

/**
 * Coerce a base-path env var into the exact form Next's `basePath` requires:
 * either an empty string, or a leading slash with no trailing slash.
 */
export function normalizeBasePath(value: string | undefined): string {
  const trimmed = (value ?? '').trim().replace(/\/+$/, '')
  if (trimmed === '') return ''
  return trimmed.startsWith('/') ? trimmed : `/${trimmed}`
}

export const BASE_PATH = normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH)

/** Resolve a root-relative `public/` path against the deploy's base path. */
export function assetPath(path: string): string {
  return `${BASE_PATH}${path.startsWith('/') ? path : `/${path}`}`
}
