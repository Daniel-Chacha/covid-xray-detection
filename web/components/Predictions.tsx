'use client'

import { argmax, maxDelta } from '@/lib/infer'
import { CLASS_NAMES } from '@/lib/types'

/**
 * Probability bars for one variant, plus the live-vs-Python parity readout.
 *
 * `argmax`/`maxDelta` come from `@/lib/infer`, which pulls in TF.js. That costs
 * nothing here: this component is only ever rendered inside Viewer, which
 * imports `predict` from the same module, so both land in the same chunk.
 * Gallery, which only needs a URL, deliberately imports from `@/lib/paths`.
 */
interface Props {
  /** Probabilities measured in this browser, or null if none yet. */
  live: number[] | null
  /** The manifest's Keras probabilities for the same image and variant. */
  reference: number[]
  /** A prediction is in flight. */
  busy: boolean
  /** Live inference failed; the reference row is all we have. */
  failed: boolean
}

/**
 * Where the displayed numbers came from.
 *
 * Kept explicit because "showing the reference" means something completely
 * different while a prediction is still running than it does after one failed,
 * and the plan's single `live ? … : 'unavailable'` label claimed failure during
 * every normal load.
 */
function sourceLabel(live: number[] | null, busy: boolean, failed: boolean): string {
  if (busy) return live ? 'Live, in your browser — recomputing…' : 'Python reference — measuring in your browser…'
  if (live) return 'Live, measured in your browser'
  if (failed) return 'Python reference (live inference unavailable)'
  return 'Python reference'
}

export default function Predictions({ live, reference, busy, failed }: Props) {
  const shown = live ?? reference
  const top = argmax(shown)

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 text-xs text-neutral-400">
        <span data-testid="prediction-source">{sourceLabel(live, busy, failed)}</span>
        {live && (
          <span
            data-testid="parity-delta"
            className="tabular-nums text-neutral-400"
            title="Largest per-class difference between this browser's output and the Python reference in manifest.json"
          >
            parity Δ {maxDelta(live, reference).toExponential(1)}
          </span>
        )}
      </div>

      {CLASS_NAMES.map((name, index) => (
        <div
          key={name}
          data-testid="prob-row"
          data-class-name={name}
          className={index === top ? 'text-neutral-50' : 'text-neutral-400'}
        >
          <div className="mb-1 flex justify-between text-sm">
            <span>{name.replace('_', ' ')}</span>
            <span className="tabular-nums">{(shown[index] * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded bg-neutral-800">
            <div
              className={`h-full transition-[width] duration-300 ${
                index === top ? 'bg-sky-400' : 'bg-neutral-600'
              }`}
              style={{ width: `${Math.max(shown[index] * 100, 0.5)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}
