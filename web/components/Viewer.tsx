'use client'

import { useEffect, useRef, useState } from 'react'
import Predictions from './Predictions'
import { argmax, predict } from '@/lib/infer'
import { assetPath } from '@/lib/paths'
import {
  CLASS_NAMES,
  VARIANT_IMAGE,
  VARIANT_LABEL,
  type GalleryItem,
  type Variant,
} from '@/lib/types'

const VARIANTS: Variant[] = ['raw', 'lungs_removed']

/** Probabilities measured in this browser, kept per variant. */
type LiveByVariant = Partial<Record<Variant, number[]>>

/**
 * Outcome of one prediction, tagged with the request it belongs to.
 *
 * Carrying the key means "which request is on screen" is DERIVED during render
 * rather than reset by a second effect: the moment the variant changes, the
 * stored outcome no longer matches and the UI reads as pending, with no
 * cascading render (and no `set-state-in-effect` lint violation).
 */
type Outcome = { key: string; state: 'done' } | { key: string; state: 'error'; message: string }

const PENDING = { key: '', state: 'done' } as const

export default function Viewer({ item }: { item: GalleryItem }) {
  const [variant, setVariant] = useState<Variant>('raw')
  const [showCam, setShowCam] = useState(false)
  const [live, setLive] = useState<LiveByVariant>({})
  const [outcome, setOutcome] = useState<Outcome>(PENDING)
  const imageRef = useRef<HTMLImageElement>(null)

  const requestKey = `${item.id}:${variant}`
  const settled = outcome.key === requestKey ? outcome : null
  const busy = settled === null
  const error = settled !== null && settled.state === 'error' ? settled.message : null

  useEffect(() => {
    let cancelled = false
    const element = imageRef.current
    const scoredAs = variant
    const key = `${item.id}:${scoredAs}`

    // Throwing rather than branching keeps every setState in a promise
    // callback: an async function that throws before its first await still
    // returns a rejected promise, so `failed` runs on a microtask.
    async function run(): Promise<number[]> {
      if (element === null) {
        throw new Error('The radiograph element is missing; cannot run inference.')
      }
      // No decode wait here. `predict` owns it, and does it correctly: an <img>
      // whose load already FAILED reports `complete === true` with
      // `naturalWidth === 0`, so the plan's `if (!element.complete) await …`
      // fell straight through and predicted on a broken image.
      return predict(element, scoredAs)
    }

    run().then(
      (probabilities) => {
        if (cancelled) return
        setLive((previous) => ({ ...previous, [scoredAs]: probabilities }))
        setOutcome({ key, state: 'done' })
      },
      (cause: unknown) => {
        // A SUPERSEDED request must be discarded whether it resolved or
        // rejected. Since Task 6, a variant toggle landing mid-decode makes the
        // outgoing call reject with "Element/variant mismatch" rather than
        // quietly return the other variant's numbers — correct, but it is a
        // symptom of the page working, not failing. Surfacing it would raise an
        // alarming banner on a fast toggle.
        if (cancelled) return
        setOutcome({
          key,
          state: 'error',
          message: cause instanceof Error ? cause.message : String(cause),
        })
      },
    )

    return () => {
      cancelled = true
    }
  }, [item.id, variant])

  function selectVariant(option: Variant) {
    setVariant(option)
    // Belt to the render gate's braces: the heatmap was computed from the raw
    // model on the raw image, so it is meaningless over the erased one.
    if (option !== 'raw') setShowCam(false)
  }

  // BUG the plan shipped: the button was disabled off-raw but the overlay was
  // rendered on `showCam` alone, so turning attention on and then switching
  // variant left the RAW model's heatmap painted over the lungs-erased
  // radiograph — an attribution map for one input displayed over another.
  const camVisible = showCam && variant === 'raw'

  // The headline comparison: does erasing the lungs change the answer?
  // Live numbers are used only once BOTH variants have been measured here, so
  // the two sides of the comparison always come from the same source.
  const bothMeasured = live.raw !== undefined && live.lungs_removed !== undefined
  const rawProbabilities = bothMeasured ? live.raw! : item.reference.raw
  const erasedProbabilities = bothMeasured ? live.lungs_removed! : item.reference.lungs_removed
  const rawTop = argmax(rawProbabilities)
  const erasedTop = argmax(erasedProbabilities)
  const topUnchanged = rawTop === erasedTop

  return (
    // aria-labelledby, not a bare <section>: an unnamed section is not exposed
    // as a landmark at all, so the viewer was unreachable by landmark
    // navigation and the page had exactly one heading in it.
    <section aria-labelledby="viewer-heading" className="space-y-4">
      <h2
        id="viewer-heading"
        className="text-xs font-semibold uppercase tracking-wide text-neutral-400"
      >
        Selected radiograph — {item.id}
      </h2>

      <div className="grid gap-8 md:grid-cols-2">
        <div>
          <div className="relative aspect-square overflow-hidden rounded-lg bg-black">
            {/*
              Displayed FAR larger than 224 CSS pixels, deliberately. tfjs reads
              an <img>'s layout size, not its intrinsic size, so this is exactly
              the case lib/infer.ts's offscreen-canvas rasterisation exists to
              survive. Do not pin the display size — that would hide the bug
              rather than keep it fixed.
            */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              ref={imageRef}
              key={`${item.id}-${variant}`}
              data-testid="viewer-image"
              data-variant={variant}
              src={assetPath(`/gallery/${item.id}/${VARIANT_IMAGE[variant]}`)}
              alt={`${item.true_class.replace('_', ' ')} radiograph — ${VARIANT_LABEL[variant]}`}
              width={224}
              height={224}
              /*
                Load-bearing since lib/infer.ts began rasterising through a
                canvas: a canvas tainted by a non-CORS cross-origin image throws
                SecurityError on getImageData and kills live inference. Kept, so
                the requirement is explicit and fails LOUDLY — a host that does
                not send Access-Control-Allow-Origin breaks the image load
                itself, which is visible on screen and surfaces through the
                banner below, rather than producing an image that renders fine
                and a SecurityError buried inside TF.js. Same-origin (the
                default deploy) is unaffected. Any CDN assetPrefix MUST send
                ACAO.
              */
              crossOrigin="anonymous"
              className="h-full w-full object-contain"
            />
            {camVisible && (
              /*
                Fully opaque on purpose: scripts/build_gallery.py already blends
                the jet heatmap over the grayscale at alpha=0.45 before saving,
                so gradcam.png is a finished image, not a transparent layer.
                z-index deliberately unset — nothing here may cover the z-50
                disclaimer.
              */
              /* eslint-disable-next-line @next/next/no-img-element */
              <img
                key={`${item.id}-gradcam`}
                data-testid="gradcam-overlay"
                src={assetPath(`/gallery/${item.id}/gradcam.png`)}
                alt="Grad-CAM attribution overlay for the full-image model"
                className="pointer-events-none absolute inset-0 h-full w-full object-contain"
              />
            )}
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            {VARIANTS.map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => selectVariant(option)}
                aria-pressed={variant === option}
                data-testid={`variant-${option}`}
                className={`rounded px-3 py-1.5 text-sm transition ${
                  variant === option
                    ? 'bg-sky-500 text-neutral-950'
                    : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
                }`}
              >
                {VARIANT_LABEL[option]}
              </button>
            ))}
            <button
              type="button"
              onClick={() => setShowCam((on) => !on)}
              disabled={variant !== 'raw'}
              aria-pressed={camVisible}
              data-testid="toggle-attention"
              title={
                variant === 'raw'
                  ? 'Grad-CAM for the full-image model'
                  : 'The heatmap belongs to the full-image model; it says nothing about the lungs-erased one'
              }
              className="rounded bg-neutral-800 px-3 py-1.5 text-sm text-neutral-300 transition hover:bg-neutral-700 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-neutral-800"
            >
              {camVisible ? 'Hide' : 'Show'} attention
            </button>
          </div>

          {/*
            neutral-400, not neutral-600. This sentence and the one at the foot
            of the comparison block are the qualifying copy — what the heatmap
            is, and where the numbers came from — and at neutral-600 they
            rendered at 2.53:1 and 2.42:1, roughly a THIRD of the contrast of
            the figures they qualify. Design section 9's failure condition is a
            reader who leaves thinking "impressive COVID detector"; bright
            numbers over near-invisible caveats is how that happens.
          */}
          <p className="mt-3 text-xs text-neutral-400">
            {variant === 'raw'
              ? 'Attention is Grad-CAM from the full-image model, precomputed in Python and blended into the image at 45% opacity.'
              : 'Every pixel inside the segmented lung fields is set to zero. Whatever the model is reading, it is not in there.'}
          </p>
        </div>

        <div className="space-y-6">
          <div>
            <p className="text-xs uppercase tracking-wide text-neutral-400">True label</p>
            <p className="text-lg">{item.true_class.replace('_', ' ')}</p>
            {item.note && <p className="mt-2 text-sm text-neutral-400">{item.note}</p>}
            <p className="mt-2 text-sm text-neutral-400">
              Lung attribution ratio <span className="tabular-nums">{item.lar.toFixed(3)}</span>{' '}
              {/*
                No colour of its own, deliberately. LAR is meaningless without
                its measured floor and ceiling, so the calibration INHERITS the
                number's colour and cannot drift dimmer than it again.
              */}
              <span>(chance 0.238, ceiling 0.376)</span>
            </p>
          </div>

          <div
            data-testid="agreement"
            className="rounded-lg border border-neutral-800 bg-neutral-900/50 p-4"
          >
            <p className="text-xs uppercase tracking-wide text-neutral-400">
              Full image vs. lungs erased
            </p>
            <p className="mt-2 text-sm text-neutral-200">
              {topUnchanged ? (
                <>
                  Erasing the lungs leaves the top answer unchanged:{' '}
                  <span className="text-neutral-50">{CLASS_NAMES[rawTop].replace('_', ' ')}</span>,{' '}
                  <span className="tabular-nums">
                    {(rawProbabilities[rawTop] * 100).toFixed(1)}% →{' '}
                    {(erasedProbabilities[rawTop] * 100).toFixed(1)}%
                  </span>
                  .
                </>
              ) : (
                <>
                  Erasing the lungs changes the top answer:{' '}
                  <span className="text-neutral-50">{CLASS_NAMES[rawTop].replace('_', ' ')}</span>{' '}
                  <span className="tabular-nums">
                    ({(rawProbabilities[rawTop] * 100).toFixed(1)}%)
                  </span>{' '}
                  →{' '}
                  <span className="text-neutral-50">
                    {CLASS_NAMES[erasedTop].replace('_', ' ')}
                  </span>{' '}
                  <span className="tabular-nums">
                    ({(erasedProbabilities[erasedTop] * 100).toFixed(1)}%)
                  </span>
                  .
                </>
              )}
            </p>
            <p className="mt-2 text-xs text-neutral-400">
              {bothMeasured
                ? 'Both sides measured in this browser.'
                : 'Both sides from the Python reference — switch to Lungs erased to measure them here.'}
            </p>
          </div>

          <Predictions
            model={VARIANT_LABEL[variant]}
            live={live[variant] ?? null}
            reference={item.reference[variant]}
            busy={busy}
            failed={error !== null}
          />

          {error && (
            <p
              data-testid="inference-error"
              role="status"
              className="rounded border border-amber-600/40 bg-amber-950/40 px-3 py-2 text-xs text-amber-200"
            >
              <strong className="font-semibold">Live inference unavailable.</strong> Showing the
              reference probabilities computed in Python instead.{' '}
              <span className="text-amber-200/70">({error})</span>
            </p>
          )}
        </div>
      </div>
    </section>
  )
}
