/**
 * Runs every gallery image through both exported graphs and asserts the result
 * matches the Keras reference probabilities in the manifest.
 *
 * This is the test that matters. A preprocessing mismatch produces confident
 * wrong answers and never an error, so nothing upstream would catch it.
 */
import * as tf from '@tensorflow/tfjs-node'
import { readFileSync } from 'node:fs'
import { PNG } from 'pngjs'
import path from 'node:path'

const ATOL = 2e-3 // matches verify_tfjs.mjs; float16 quantization is lossy
const GALLERY = path.resolve('public/gallery')
const VARIANTS = {
  raw: { model: 'public/models/raw/model.json', image: 'raw.png' },
  lungs_removed: { model: 'public/models/lungs_removed/model.json', image: 'lungs_erased.png' },
}

function loadPixels(file) {
  const png = PNG.sync.read(readFileSync(file))
  if (png.width !== 224 || png.height !== 224) {
    throw new Error(`${file}: expected 224x224, got ${png.width}x${png.height}`)
  }
  const rgb = new Float32Array(224 * 224 * 3)
  for (let i = 0, j = 0; i < png.data.length; i += 4, j += 3) {
    rgb[j] = png.data[i]
    rgb[j + 1] = png.data[i + 1]
    rgb[j + 2] = png.data[i + 2]
  }
  return tf.tensor4d(rgb, [1, 224, 224, 3])
}

const manifest = JSON.parse(readFileSync(path.join(GALLERY, 'manifest.json'), 'utf8'))
let worstError = 0
let flips = 0

for (const [variant, cfg] of Object.entries(VARIANTS)) {
  const model = await tf.loadGraphModel(`file://${path.resolve(cfg.model)}`)
  for (const item of manifest) {
    const input = loadPixels(path.join(GALLERY, item.id, cfg.image))
    const got = Array.from(await model.predict(input).data())
    input.dispose()

    const want = item.reference[variant]
    const error = Math.max(...got.map((g, i) => Math.abs(g - want[i])))
    worstError = Math.max(worstError, error)

    const argmax = (a) => a.indexOf(Math.max(...a))
    if (argmax(got) !== argmax(want)) {
      flips += 1
      console.error(`ARGMAX FLIP  ${variant}  ${item.id}: ${argmax(want)} -> ${argmax(got)}`)
    }
    if (error > ATOL) {
      console.error(`OVER TOLERANCE  ${variant}  ${item.id}: ${error.toExponential(2)}`)
    }
  }
  console.log(`${variant}: checked ${manifest.length} images`)
}

console.log(`max abs probability error: ${worstError.toExponential(2)}  (atol ${ATOL})`)
console.log(`argmax flips: ${flips}`)

if (flips > 0 || worstError > ATOL) {
  console.error('\nFAIL — do not widen ATOL to make this pass. It exists to catch exactly this.')
  process.exit(1)
}
console.log('\nPASS')
