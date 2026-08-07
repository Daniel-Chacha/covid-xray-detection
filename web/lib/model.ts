import * as tf from '@tensorflow/tfjs'
import { assetPath } from './paths'
import type { Variant } from './types'

// Routed through assetPath so a sub-path deploy is one env var (see paths.ts),
// not an edit here. Absolute '/models/...' would 404 under GitHub Pages'
// '/<repo>/' prefix.
const MODEL_URL: Record<Variant, string> = {
  raw: assetPath('/models/raw/model.json'),
  lungs_removed: assetPath('/models/lungs_removed/model.json'),
}

// Cache the promise, not the model: concurrent callers during the first load
// then share one download rather than racing two.
const cache = new Map<Variant, Promise<tf.GraphModel>>()

/**
 * Load a converted model, at most once per variant.
 *
 * loadGraphModel, never loadLayersModel — the exported graph is frozen and
 * carries its preprocessing internally.
 */
export function loadModel(variant: Variant): Promise<tf.GraphModel> {
  let pending = cache.get(variant)
  if (!pending) {
    // Settle the backend first so the weights are uploaded straight to the
    // backend that will run inference, instead of being moved afterwards.
    pending = activeBackend().then(() => tf.loadGraphModel(MODEL_URL[variant]))
    // A failed load must not be cached as a permanent failure: drop it so a
    // retry can re-request. (Task 7 reports the error inline.)
    pending.catch(() => cache.delete(variant))
    cache.set(variant, pending)
  }
  return pending
}

/**
 * Backends in preference order.
 *
 * DEVIATION from design section 6, which names "WebGL, WASM or CPU": WASM is a
 * separate package (@tensorflow/tfjs-backend-wasm) that is NOT in package.json,
 * and this task adds no dependencies. `@tensorflow/tfjs` ships CPU and WebGL
 * only, so the real chain is WebGL then CPU. Nothing is lost: total TF.js
 * failure degrades to the manifest's reference numbers (Task 7).
 */
const BACKEND_PREFERENCE = ['webgl', 'cpu'] as const

async function initBackend(): Promise<string> {
  for (const name of BACKEND_PREFERENCE) {
    try {
      // setBackend resolves false when a backend's init fails (no WebGL
      // context, blocklisted driver); some paths throw instead. Both mean
      // "try the next one".
      if (await tf.setBackend(name)) {
        await tf.ready()
        return tf.getBackend()
      }
    } catch {
      // Fall through to the next candidate.
    }
  }
  // Last resort: whatever tfjs selected for itself.
  await tf.ready()
  return tf.getBackend()
}

let backendPromise: Promise<string> | null = null

/**
 * The backend actually running inference — 'webgl' or 'cpu' here.
 *
 * Displayed in the UI because it explains order-of-magnitude latency
 * differences between visitors. Resolved once and cached.
 */
export function activeBackend(): Promise<string> {
  if (backendPromise === null) {
    backendPromise = initBackend()
  }
  return backendPromise
}
