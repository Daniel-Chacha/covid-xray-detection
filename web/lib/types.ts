// Canonical order from covid_xray.config.CLASS_NAMES. The model's output axis
// is index-aligned with this; never re-derive or re-sort it.
export const CLASS_NAMES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'] as const
export type ClassName = (typeof CLASS_NAMES)[number]

export type Variant = 'raw' | 'lungs_removed'

/** One record from public/gallery/manifest.json. */
export interface GalleryItem {
  id: string
  true_class: ClassName
  note: string
  /** Fraction of the raw model's Grad-CAM mass inside the lung mask. */
  lar: number
  /** Keras probabilities, index-aligned with CLASS_NAMES. */
  reference: Record<Variant, number[]>
}

export const VARIANT_LABEL: Record<Variant, string> = {
  raw: 'Full image',
  lungs_removed: 'Lungs erased',
}

/** Which gallery PNG feeds which model. */
export const VARIANT_IMAGE: Record<Variant, string> = {
  raw: 'raw.png',
  lungs_removed: 'lungs_erased.png',
}
