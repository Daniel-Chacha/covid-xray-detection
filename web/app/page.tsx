'use client'

import { useEffect, useState } from 'react'
import Gallery from '@/components/Gallery'
import Viewer from '@/components/Viewer'
import { assetPath } from '@/lib/paths'
import type { GalleryItem } from '@/lib/types'

const MANIFEST_URL = assetPath('/gallery/manifest.json')

/**
 * Enough validation to turn a wrong deploy into a message instead of a crash.
 *
 * A sub-path host that serves its 404 page with a 200 would otherwise reach
 * `records[0].reference[variant]` and throw inside render.
 */
function parseManifest(payload: unknown): GalleryItem[] {
  if (!Array.isArray(payload) || payload.length === 0) {
    throw new Error('the manifest did not contain a non-empty array of gallery records.')
  }
  for (const record of payload) {
    const item = record as Partial<GalleryItem>
    if (
      typeof item?.id !== 'string' ||
      !Array.isArray(item?.reference?.raw) ||
      !Array.isArray(item?.reference?.lungs_removed)
    ) {
      throw new Error('a manifest record is missing its id or its reference probabilities.')
    }
  }
  return payload as GalleryItem[]
}

export default function Page() {
  const [items, setItems] = useState<GalleryItem[]>([])
  const [selectedId, setSelectedId] = useState<string>('')
  const [manifestError, setManifestError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    // The plan had neither a `.catch` nor a `response.ok` check here, so a 404
    // — the likeliest outcome of a sub-path deploy — left the page reading
    // "Loading gallery…" forever. Design section 6 requires visible
    // degradation, not a hang.
    fetch(MANIFEST_URL)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`the server answered ${response.status} ${response.statusText}.`)
        }
        return parseManifest(await response.json())
      })
      .then((records) => {
        if (cancelled) return
        setItems(records)
        setSelectedId(records[0].id)
      })
      .catch((cause: unknown) => {
        if (cancelled) return
        setManifestError(cause instanceof Error ? cause.message : String(cause))
      })

    return () => {
      cancelled = true
    }
  }, [])

  const selected = items.find((item) => item.id === selectedId)

  return (
    <main className="mx-auto w-full max-w-5xl space-y-10 px-4 py-10">
      {/*
        Every word of explanation — the headline figures, the training recipe,
        the data caveats — lives on /about. The heading stays for screen-reader
        and landmark navigation only; the hand-picked disclosure design section
        8 requires is still visible, in the gallery's own heading.
      */}
      <h1 className="sr-only">How much of this COVID classifier is real?</h1>

      {items.length > 0 && (
        <Gallery items={items} selectedId={selectedId} onSelect={setSelectedId} />
      )}

      {selected ? (
        <Viewer key={selected.id} item={selected} />
      ) : manifestError ? (
        <div
          role="alert"
          data-testid="manifest-error"
          className="rounded-lg border border-amber-600/40 bg-amber-950/40 px-4 py-3 text-sm text-amber-100"
        >
          <p className="font-semibold">The gallery could not be loaded.</p>
          <p className="mt-1 text-amber-200/90">
            Fetching <code className="font-mono text-xs">{MANIFEST_URL}</code> failed:{' '}
            {manifestError}
          </p>
          <p className="mt-2 text-xs text-amber-200/70">
            If this build is served from a sub-path, it must be built with{' '}
            <code className="font-mono">NEXT_PUBLIC_BASE_PATH</code> set to that prefix. Without the
            manifest there are no images and no reference probabilities, so the page has nothing to
            show.
          </p>
        </div>
      ) : (
        <p className="text-neutral-400">Loading gallery…</p>
      )}
    </main>
  )
}
