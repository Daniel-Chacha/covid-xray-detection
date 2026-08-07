import * as tf from '@tensorflow/tfjs'
import { loadModel } from './model'
import type { Variant } from './types'

export const INPUT_SIZE = 224

/**
 * RGB. Passed explicitly rather than relying on the default.
 *
 * `tf.browser.fromPixels` defaults to `numChannels = 3` and drops alpha, which
 * matches scripts/verify-parity.mjs stripping the 4th byte of every pixel. The
 * default is stated here so a future tfjs change cannot move it underneath us.
 */
const CHANNELS = 3

/**
 * Wait until the element actually has pixels.
 *
 * `fromPixels` on an <img> that has not finished decoding yields garbage or
 * throws, and garbage would surface as confident wrong probabilities rather
 * than as an error.
 */
async function ensureDecoded(image: HTMLImageElement): Promise<void> {
  const src = image.currentSrc || image.src

  // decode() covers both "not loaded yet" and "loaded but not yet decoded",
  // and rejects (EncodingError) on a corrupt asset instead of half-succeeding.
  if (typeof image.decode === 'function') {
    try {
      await image.decode()
    } catch (cause) {
      throw new Error(`Failed to decode image: ${src}`, { cause })
    }
    return
  }

  // Fallback for engines without HTMLImageElement.decode. `complete` alone is
  // not enough — it is also true after a failed load, hence the naturalWidth
  // check.
  if (image.complete && image.naturalWidth > 0) return

  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      image.removeEventListener('load', onLoad)
      image.removeEventListener('error', onError)
    }
    const onLoad = () => {
      cleanup()
      resolve()
    }
    const onError = () => {
      cleanup()
      reject(new Error(`Failed to load image: ${src}`))
    }
    image.addEventListener('load', onLoad)
    image.addEventListener('error', onError)
  })
}

/**
 * Copy the image into an offscreen canvas sized from its INTRINSIC dimensions.
 *
 * This exists because `tf.browser.fromPixels(imgEl)` reads `imgEl.width` and
 * `imgEl.height`, which for a rendered <img> are the LAYOUT size, not
 * `naturalWidth`/`naturalHeight` (tfjs-core/dist/ops/browser.js, and the WebGL
 * FromPixels kernel, both do `[pixels.width, pixels.height]`; the WebGL kernel
 * then rescales with `drawImage(pixels, 0, 0, width, height)`). Style a gallery
 * image at anything other than 224 CSS pixels and the tensor silently becomes
 * the wrong shape, resampled by the browser's own kernel. No exception is
 * thrown; the predictions just drift. That is precisely the failure mode this
 * project exists to expose.
 *
 * A canvas created with document.createElement and never inserted into the
 * document has no layout box at all, so `canvas.width`/`canvas.height` are the
 * bitmap's real dimensions and no stylesheet can reach them. Three-argument
 * drawImage blits at the source's intrinsic size, 1:1, with no resampling.
 */
function drawToOffscreenCanvas(image: HTMLImageElement): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = image.naturalWidth
  canvas.height = image.naturalHeight

  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (context === null) {
    throw new Error('2D canvas context unavailable; cannot read image pixels')
  }
  context.drawImage(image, 0, 0)
  return canvas
}

/**
 * Probabilities for one gallery image element.
 *
 * The image is fed as raw 0-255 float pixels with no normalisation and no
 * resize, matching scripts/verify-parity.mjs. Preprocessing lives inside the
 * exported graph, and the asset is already exactly 224x224 — resampling here
 * would risk a kernel mismatch with the TensorFlow bilinear used in training,
 * which fails silently. Hence the two assertions below rather than a
 * `resizeBilinear` that would paper over the problem.
 */
export async function predict(image: HTMLImageElement, variant: Variant): Promise<number[]> {
  await ensureDecoded(image)

  const { naturalWidth, naturalHeight } = image
  if (naturalWidth !== INPUT_SIZE || naturalHeight !== INPUT_SIZE) {
    throw new Error(
      `Gallery asset must be exactly ${INPUT_SIZE}x${INPUT_SIZE}, got ` +
        `${naturalWidth}x${naturalHeight} (${image.currentSrc || image.src}). ` +
        'The asset is wrong; do not resize here — the browser\'s resampling ' +
        'kernel differs from the TensorFlow bilinear used in training and the ' +
        'resulting drift is silent.',
    )
  }

  const model = await loadModel(variant)
  const canvas = drawToOffscreenCanvas(image)

  const output = tf.tidy(() => {
    const input = tf.browser.fromPixels(canvas, CHANNELS).toFloat().expandDims(0)

    // Belt and braces: whatever the source turned out to be, the tensor that
    // reaches the graph must have the training shape.
    const shape = input.shape
    const expected = [1, INPUT_SIZE, INPUT_SIZE, CHANNELS]
    if (shape.length !== expected.length || shape.some((d, i) => d !== expected[i])) {
      throw new Error(
        `Expected input tensor [${expected}], got [${shape}]. ` +
          'Refusing to predict on a mis-shaped input.',
      )
    }

    const result = model.predict(input)
    if (!(result instanceof tf.Tensor)) {
      throw new Error('Expected the graph to return a single output tensor.')
    }
    return result
  })

  try {
    return Array.from(await output.data())
  } finally {
    output.dispose()
  }
}

/** Largest absolute per-class difference between live and reference output. */
export function maxDelta(live: number[], reference: number[]): number {
  return Math.max(...live.map((p, i) => Math.abs(p - reference[i])))
}

export function argmax(values: number[]): number {
  return values.indexOf(Math.max(...values))
}
