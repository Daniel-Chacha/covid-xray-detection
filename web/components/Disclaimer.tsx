/**
 * Persistent, non-dismissible clinical-use disclaimer.
 *
 * Deliberately stateless: no close button, no dismissal state, no
 * localStorage. It is a server component rendered by the root layout, so the
 * text is baked into every emitted HTML file rather than injected on the
 * client — it is present even if JavaScript never runs.
 */
export default function Disclaimer() {
  return (
    <div
      role="note"
      className="sticky top-0 z-50 border-b border-amber-500/40 bg-amber-950/90 px-4 py-2 text-center text-sm text-amber-100 backdrop-blur"
    >
      <strong className="font-semibold">
        Not a medical device. Not for clinical use.
      </strong>{" "}
      This page demonstrates that the model reads acquisition artefacts rather
      than lung pathology. It is a research artefact, not a diagnostic tool.
    </div>
  );
}
