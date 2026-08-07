'use client'

import { useRef, type KeyboardEvent } from 'react'
// assetPath comes from @/lib/paths, NOT @/lib/model — a component that only
// needs a PNG URL must not drag TF.js into its chunk.
import { assetPath } from '@/lib/paths'
import type { GalleryItem } from '@/lib/types'

interface Props {
  items: GalleryItem[]
  selectedId: string
  onSelect: (id: string) => void
}

export default function Gallery({ items, selectedId, onSelect }: Props) {
  // Keyboard navigation focuses tiles through these refs rather than by
  // querying `[data-testid]`. A test hook must not be load-bearing for
  // production behaviour — deleting it should break a test, loudly, not arrow
  // keys, silently.
  const tileRefs = useRef<(HTMLButtonElement | null)[]>([])

  const matchedIndex = items.findIndex((item) => item.id === selectedId)

  // The roving tab stop needs a tile even when nothing matches, or every tile
  // gets tabIndex -1 and the gallery drops out of the tab order entirely. This
  // is deliberately NOT the same thing as being selected: an unmatched id must
  // leave every tile unchecked rather than silently checking tile 0 while the
  // viewer shows something else.
  const tabStopIndex = matchedIndex < 0 ? 0 : matchedIndex

  /**
   * Move selection and focus together.
   *
   * A radio group's contract is that arrow keys move focus AND change the
   * selection, so this is one operation rather than two.
   */
  function focusAt(index: number) {
    const bounded = (index + items.length) % items.length
    onSelect(items[bounded].id)
    tileRefs.current[bounded]?.focus()
  }

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    let target: number
    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        target = tabStopIndex + 1
        break
      case 'ArrowLeft':
      case 'ArrowUp':
        target = tabStopIndex - 1
        break
      case 'Home':
        target = 0
        break
      case 'End':
        target = items.length - 1
        break
      default:
        return
    }
    event.preventDefault()
    focusAt(target)
  }

  return (
    <section aria-labelledby="gallery-heading" className="space-y-3">
      <h2
        id="gallery-heading"
        className="text-xs font-semibold uppercase tracking-wide text-neutral-400"
      >
        {items.length} hand-picked test-set radiographs
      </h2>

      {/*
        `radiogroup`/`radio`, not `group` + `aria-pressed`. The twelve tiles are
        mutually exclusive; a toggle-button group says nothing about
        exclusivity, so assistive tech announced twelve independent
        pressed/unpressed buttons instead of "3 of 12, selected".

        Taking the role means taking its keyboard contract with it (WAI-ARIA
        APG, radio group): ONE tab stop, arrow keys move between radios, moving
        selects. Declaring the role without that would be a downgrade — twelve
        working tab stops traded for a label that lies.
      */}
      <div
        role="radiogroup"
        aria-labelledby="gallery-heading"
        onKeyDown={onKeyDown}
        className="grid grid-cols-4 gap-2 sm:grid-cols-6 lg:grid-cols-12"
      >
        {items.map((item, index) => {
          const label = item.true_class.replace('_', ' ')
          const selected = item.id === selectedId
          return (
            <button
              key={item.id}
              ref={(node) => {
                tileRefs.current[index] = node
              }}
              type="button"
              role="radio"
              onClick={() => onSelect(item.id)}
              aria-checked={selected}
              aria-label={`${label} — ${item.id}`}
              title={`${label} — ${item.id}`}
              tabIndex={index === tabStopIndex ? 0 : -1}
              data-testid="gallery-tile"
              data-item-id={item.id}
              className={`group overflow-hidden rounded border-2 text-left transition ${
                selected ? 'border-sky-400' : 'border-transparent'
              }`}
            >
              {/*
                crossOrigin matches the viewer's, so the two elements share one
                cache entry and a host without Access-Control-Allow-Origin fails
                uniformly and visibly rather than only inside inference.
              */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={assetPath(`/gallery/${item.id}/raw.png`)}
                alt=""
                width={224}
                height={224}
                crossOrigin="anonymous"
                /*
                  The dimming of an unselected tile lives HERE, on the
                  thumbnail, not on the button. It used to be on the button, so
                  it multiplied the class label too and dragged it to 2.20:1 —
                  a third of the contrast of the numbers it qualifies. The
                  thumbnail is nearly all of the tile's visual mass, so moving
                  the opacity keeps the entire selection affordance and hands
                  the label back its legibility.
                */
                className={`aspect-square w-full object-cover transition ${
                  selected ? '' : 'opacity-60 group-hover:opacity-100'
                }`}
              />
              <span
                className={`block truncate px-1 py-0.5 text-[10px] leading-tight ${
                  selected ? 'text-neutral-100' : 'text-neutral-400'
                }`}
              >
                {label}
              </span>
            </button>
          )
        })}
      </div>
    </section>
  )
}
