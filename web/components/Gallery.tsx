'use client'

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
  return (
    <div
      role="group"
      aria-label="Gallery images"
      className="grid grid-cols-4 gap-2 sm:grid-cols-6 lg:grid-cols-12"
    >
      {items.map((item) => {
        const label = item.true_class.replace('_', ' ')
        const selected = item.id === selectedId
        return (
          <button
            key={item.id}
            type="button"
            onClick={() => onSelect(item.id)}
            aria-pressed={selected}
            aria-label={`${label} — ${item.id}`}
            title={`${label} — ${item.id}`}
            data-testid="gallery-tile"
            data-item-id={item.id}
            className={`overflow-hidden rounded border-2 text-left transition ${
              selected ? 'border-sky-400' : 'border-transparent opacity-60 hover:opacity-100'
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
              className="aspect-square w-full object-cover"
            />
            <span className="block truncate px-1 py-0.5 text-[10px] leading-tight text-neutral-500">
              {label}
            </span>
          </button>
        )
      })}
    </div>
  )
}
