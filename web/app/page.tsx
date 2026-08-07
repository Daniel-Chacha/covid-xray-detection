'use client'

import { useEffect, useState } from 'react'
import Gallery from '@/components/Gallery'
import Viewer from '@/components/Viewer'
import { argmax } from '@/lib/infer'
import { activeBackend } from '@/lib/model'
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
  const [backend, setBackend] = useState<string>('')
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

    activeBackend().then(
      (name) => {
        if (!cancelled) setBackend(name)
      },
      () => {
        if (!cancelled) setBackend('unavailable')
      },
    )

    return () => {
      cancelled = true
    }
  }, [])

  const selected = items.find((item) => item.id === selectedId)

  // The prominent number, per design section 8: agreement between the two
  // models, not accuracy. Counted from the manifest rather than hardcoded, so
  // it cannot drift away from the images actually shipped.
  const unchanged = items.filter(
    (item) => argmax(item.reference.raw) === argmax(item.reference.lungs_removed),
  ).length

  return (
    <main className="mx-auto w-full max-w-5xl space-y-10 px-4 py-10">
      <header className="space-y-5">
        <h1 className="text-3xl font-semibold tracking-tight">
          How much of this COVID classifier is real?
        </h1>
        <p className="max-w-2xl text-neutral-300">
          Two DenseNet121 models run in your browser on the same radiograph — one trained on the
          full image, one trained with the lung fields erased to black. Toggle between them. The
          prediction mostly does not move, because the classifier is reading how the image was
          acquired rather than what is inside the chest.
        </p>

        <dl className="grid gap-3 sm:grid-cols-3">
          <div className="rounded-lg border border-sky-400/30 bg-sky-950/20 p-4">
            <dt className="text-xs uppercase tracking-wide text-sky-200/70">
              The two models agree
            </dt>
            <dd className="mt-1 text-3xl font-semibold tabular-nums text-sky-200">
              {items.length > 0 ? `${unchanged} of ${items.length}` : '—'}
            </dd>
            <dd className="mt-1 text-xs text-neutral-400">
              images below where erasing the lungs entirely leaves the top answer unchanged
            </dd>
          </div>
          <div className="rounded-lg border border-neutral-800 p-4">
            <dt className="text-xs uppercase tracking-wide text-neutral-500">
              Held-out test set, 3,142 images
            </dt>
            <dd className="mt-1 text-3xl font-semibold tabular-nums text-neutral-200">97.2%</dd>
            <dd className="mt-1 text-xs text-neutral-400">
              of the full-image model&rsquo;s macro-F1 survives erasing the lungs (0.8288 vs.
              0.8523)
            </dd>
          </div>
          <div className="rounded-lg border border-neutral-800 p-4">
            <dt className="text-xs uppercase tracking-wide text-neutral-500">
              COVID vs. Lung Opacity
            </dt>
            <dd className="mt-1 text-3xl font-semibold tabular-nums text-neutral-200">
              0.9815 / 0.9797
            </dd>
            <dd className="mt-1 text-xs text-neutral-400">
              pair AUC without lungs / with lungs — on the comparison that matters clinically,
              erasing them helps slightly
            </dd>
          </div>
        </dl>

        <div className="max-w-3xl space-y-2 text-xs text-neutral-500">
          <p>
            These twelve test-set images were <strong className="text-neutral-400">hand-picked</strong>{' '}
            to make that argument. They are not a random sample and nothing here is a measure of
            accuracy — whether the model happens to be right about any one of them is beside the
            point. The accuracy figure it does have on the full test set, 0.852 macro-F1, is the
            artefact under examination, not a result to be impressed by.
          </p>
          <p>
            Nothing is uploaded, and there is nothing to upload with: the twelve radiographs ship
            with the page and both models run entirely on your device
            {backend && ` (TensorFlow.js, ${backend} backend)`}.
          </p>
        </div>
      </header>

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
        <p className="text-neutral-500">Loading gallery…</p>
      )}
    </main>
  )
}
