'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

/**
 * `next/link` prefixes `basePath` itself, so these hrefs stay root-relative —
 * unlike `public/` assets, which go through `assetPath()`. `usePathname()`
 * likewise returns the path WITHOUT the base path, so the comparison below
 * holds on a sub-path deploy too.
 */
const LINKS = [
  { href: '/', label: 'Demo' },
  { href: '/about', label: 'About' },
] as const

export default function SiteNav() {
  const pathname = usePathname()

  return (
    <nav aria-label="Site" className="border-b border-neutral-800">
      <ul className="mx-auto flex w-full max-w-5xl gap-1 px-4 py-2 text-sm">
        {LINKS.map(({ href, label }) => {
          const current = pathname === href
          return (
            <li key={href}>
              <Link
                href={href}
                aria-current={current ? 'page' : undefined}
                className={`inline-block rounded px-3 py-1.5 transition ${
                  current
                    ? 'bg-neutral-800 text-neutral-50'
                    : 'text-neutral-400 hover:bg-neutral-900 hover:text-neutral-100'
                }`}
              >
                {label}
              </Link>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}
