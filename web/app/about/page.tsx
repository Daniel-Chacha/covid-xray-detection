import type { Metadata } from 'next'
import Link from 'next/link'
import type { ReactNode } from 'react'
import type { GalleryItem } from '@/lib/types'
// Read at BUILD time. This is a server component, so the manifest is resolved
// while about.html is generated rather than fetched by the browser, and the
// agreement count below is still derived from the twelve records the demo
// ships instead of being hardcoded next to them.
import manifest from '@/public/gallery/manifest.json'

export const metadata: Metadata = {
  title: 'About — How much of this COVID classifier is real?',
  description:
    'How the two DenseNet121 models were trained, how your browser turns a radiograph into a prediction, and why the dataset behind them cannot support a diagnosis.',
}

const items = manifest as GalleryItem[]

/**
 * `argmax` from lib/infer.ts, restated. Importing that module here would pull
 * TF.js into a page that never runs a model.
 */
function topIndex(values: number[]): number {
  return values.indexOf(Math.max(...values))
}

const unchanged = items.filter(
  (item) => topIndex(item.reference.raw) === topIndex(item.reference.lungs_removed),
).length

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section aria-labelledby={id} className="space-y-4">
      <h2 id={id} className="text-xl font-semibold tracking-tight">
        {title}
      </h2>
      <div className="space-y-4 text-sm leading-relaxed text-neutral-300">{children}</div>
    </section>
  )
}

/** A term/definition row: a label column on wide screens, stacked on narrow ones. */
function Row({ term, children }: { term: string; children: ReactNode }) {
  return (
    <div className="grid gap-1 border-t border-neutral-800 py-3 sm:grid-cols-[10rem_1fr] sm:gap-4">
      <dt className="text-xs uppercase tracking-wide text-neutral-400 sm:pt-0.5">{term}</dt>
      <dd>{children}</dd>
    </div>
  )
}

export default function AboutPage() {
  return (
    <main className="mx-auto w-full max-w-5xl space-y-14 px-4 py-10">
      <header className="space-y-5">
        <p className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
          About this demo
        </p>
        <h1 className="text-3xl font-semibold tracking-tight">
          How much of this COVID classifier is real?
        </h1>
        <p className="max-w-2xl text-neutral-300">
          <Link href="/" className="text-sky-300 underline underline-offset-2 hover:text-sky-200">
            Pick a radiograph
          </Link>
          , then switch between the full image and the lungs erased. Two DenseNet121 models score
          it on your device — and the prediction mostly does not move.
        </p>
        <p className="max-w-2xl text-neutral-300">
          Two DenseNet121 models run in your browser on the same radiograph — one trained on the
          full image, one trained with the lung fields erased to black. The prediction mostly
          does not move between them, because the classifier is reading how the image was
          acquired rather than what is inside the chest.
        </p>

        <dl className="grid gap-3 sm:grid-cols-3">
          <div className="rounded-lg border border-sky-400/30 bg-sky-950/20 p-4">
            <dt className="text-xs uppercase tracking-wide text-sky-200/70">
              The two models agree
            </dt>
            <dd className="mt-1 text-3xl font-semibold tabular-nums text-sky-200">
              {unchanged} of {items.length}
            </dd>
            <dd className="mt-1 text-xs text-neutral-400">
              demo radiographs where erasing the lungs entirely leaves the top answer unchanged
            </dd>
          </div>
          {/*
            The <dt> names the MEASURED QUANTITY, not the dataset. It read
            "Held-out test set, 3,142 images" over a 97.2% in 30px type, so a
            skim produced "97.2% on 3,142 held-out images" — an accuracy
            headline, and precisely the misreading design section 9 warns
            about. The number is a retention ratio and now says so before the
            eye reaches it.
          */}
          <div className="rounded-lg border border-neutral-800 p-4">
            <dt className="text-xs uppercase tracking-wide text-neutral-400">
              Macro-F1 retained without lungs
            </dt>
            <dd className="mt-1 text-3xl font-semibold tabular-nums text-neutral-200">97.2%</dd>
            <dd className="mt-1 text-xs text-neutral-400">
              0.8288 with the lung fields erased against 0.8523 on the full image, over the
              held-out test set of 3,142 radiographs
            </dd>
          </div>
          <div className="rounded-lg border border-neutral-800 p-4">
            <dt className="text-xs uppercase tracking-wide text-neutral-400">
              Pair AUC, lungs erased / full image
            </dt>
            <dd className="mt-1 text-3xl font-semibold tabular-nums text-neutral-200">
              0.9815 / 0.9797
            </dd>
            {/*
              Wording is README.md:41 verbatim, deliberately. This previously
              read "erasing them helps slightly", which reads a DIRECTION off a
              0.0018 gap with no interval — the exact move this project exists
              to criticise, and a contradiction of its own source document,
              which declines to make it.
            */}
            <dd className="mt-1 text-xs text-neutral-400">
              COVID vs. Lung Opacity — both adult, both radiographic opacities, the comparison
              that matters clinically. Erasing the lungs does not hurt: lung parenchyma
              contributes nothing measurable.
            </dd>
          </div>
        </dl>

        {/*
          neutral-400, not neutral-500. Design section 8 requires the
          hand-picked nature of the gallery to be stated plainly; at
          neutral-500 this disclosure rendered at 4.18:1 — below WCAG AA, and
          under a third of the contrast of the three figures it qualifies.
        */}
        <div className="max-w-3xl space-y-2 text-xs text-neutral-400">
          <p>
            The twelve demo radiographs are test-set images,{' '}
            <strong className="text-neutral-100">hand-picked</strong> to make that argument. They
            are not a random sample and nothing on the demo is a measure of accuracy — whether the
            model happens to be right about any one of them is beside the point. The accuracy
            figure it does have on the full test set, 0.852 macro-F1, is the artefact under
            examination, not a result to be impressed by.
          </p>
          <p>
            Nothing is uploaded, and there is nothing to upload with: the twelve radiographs ship
            with the page and both models run entirely on your device through TensorFlow.js.
          </p>
        </div>
      </header>

      <Section id="data-caveats" title="What this data cannot tell you">
        {/*
          Placed BEFORE the method sections, not after them. Design section 9's
          failure condition is a reader who leaves thinking "impressive COVID
          detector"; caveats at the foot of a long page are the ones nobody
          reaches.
        */}
        <div className="rounded-lg border border-amber-600/40 bg-amber-950/30 p-5 text-amber-100">
          <p className="font-semibold">
            Not a medical device. Nothing here is a diagnosis, a screening result or a second
            opinion, and none of it should inform a decision about anyone&apos;s health.
          </p>
          <p className="mt-2 text-amber-200/90">
            The dataset behind both models is a merge of unrelated public collections. It was
            never designed to represent any patient population, and several of its properties
            would mislead a model — and a reader — even if the model were perfect.
          </p>
        </div>

        <dl>
          <Row term="Some are children">
            The Viral Pneumonia images, and part of the Normal class, come from a pediatric
            collection: children aged 1 to 5 at a single hospital in Guangzhou, China. COVID and
            Lung Opacity images are of adults. A model can tell those classes apart from the size
            and shape of a small child&apos;s chest without looking at the lungs — and this one
            does: it identifies Viral Pneumonia almost perfectly (ROC-AUC 0.998) with the lungs
            erased just as with them.
          </Row>
          <Row term="Each class has its own source">
            COVID images were gathered from the Italian SIRM repository and published case
            reports; Lung Opacity and most Normal images come from RSNA. Different hospitals,
            scanners, exposure settings, image processing and burned-in text markers travel with
            each class. The model learns them. That is what the demo measures.
          </Row>
          <Row term="Not representative">
            No age, sex, ethnicity, clinical history or scanner information is available per
            image, so nobody can say whom this data represents. Case-report images lean towards
            cases unusual enough to publish. The class mix — about 17% COVID — reflects how the
            dataset was assembled, not how common any disease is, so a probability on the demo is
            not anyone&apos;s chance of having anything.
          </Row>
          <Row term="Labels are provenance">
            A label records which collection an image came from; no radiologist re-read these
            images for this project. The labels also overlap: COVID is a viral pneumonia, and a
            child&apos;s viral pneumonia is a lung opacity. The four classes are mutually
            exclusive only as bookkeeping.
          </Row>
          <Row term="Reduced images">
            8-bit PNGs at 299×299, downsampled again to 224×224 for the model — far below the
            full-resolution originals a radiologist reads.
          </Row>
          <Row term="Machine-drawn masks">
            The lung outlines used to erase the lungs were generated by the dataset&apos;s authors
            with a segmentation model, not drawn by radiologists. An erased region can miss the
            true lung edge. That is why the study also tested an 8×8 thumbnail, where no lung
            edge exists to leak through.
          </Row>
          <Row term="Patients may repeat">
            The dataset has no patient identifiers, so it was split by image, not by patient. Two
            radiographs of one person can land on both sides of the train/test line, which
            flatters every metric. Read the figures as upper bounds.
          </Row>
          <Row term="It misses COVID">
            The full-image model finds only 63.8% of the COVID cases in the test set — it misses
            more than one in three — while its headline macro-F1 is 0.852. It is reluctant to say
            COVID, which flatters its overall accuracy.
          </Row>
          <Row term="Never tested elsewhere">
            Neither model has seen a radiograph from any other source. On images from a different
            hospital, expect worse performance — possibly much worse — and no warning when it
            happens.
          </Row>
        </dl>
      </Section>

      <Section id="training" title="How the models were trained">
        <p className="max-w-3xl">
          Both models on the demo come from one training script and differ in exactly one thing:
          what the input contains. The full-image model saw the radiographs as published. The
          lungs-erased model saw the same radiographs with every pixel inside the segmented lung
          fields set to zero — the lungs removed, everything else kept.
        </p>

        <dl>
          <Row term="Dataset">
            The COVID-19 Radiography Database (Kaggle, version 5): 21,165 frontal chest
            radiographs, each with a lung segmentation mask. COVID 3,616 · Lung Opacity 6,012 ·
            Normal 10,192 · Viral Pneumonia 1,345.
          </Row>
          <Row term="De-duplication">
            Exact and near-duplicate images were removed before splitting — a hash of the decoded
            pixels plus a perceptual hash at Hamming distance ≤ 1 — so one image cannot sit on
            both sides of the train/test line. 223 were removed; 214 of them were COVID.
          </Row>
          <Row term="Splits">
            Stratified 70 / 15 / 15 on the 20,942 that remained: 14,659 training, 3,141
            validation, 3,142 test. The test set was set aside on the first day and scored once,
            at the end.
          </Row>
          <Row term="Preprocessing">
            Resized to 224×224, bilinear; masks nearest-neighbour. The lungs-erased variant
            zeroes the lung fields in place rather than cropping to them, so the size of a crop
            cannot become a clue of its own.
          </Row>
          <Row term="Augmentation">
            Random horizontal flip, ±14° rotation, ±10% zoom, ±5% shift, and contrast (0.2) and
            brightness (0.15) jitter, since radiograph brightness varies with exposure. No
            vertical flips.
          </Row>
          <Row term="Network">
            DenseNet121 pretrained on ImageNet, with a new four-way softmax head.
          </Row>
          <Row term="Schedule">
            Stage A trains only the head on a frozen backbone (learning rate 10⁻³, up to 8
            epochs). Stage B unfreezes the last 81 layers, from <code>conv5_block1</code> on
            (learning rate 10⁻⁵, up to 15 epochs). Batch size 32. BatchNorm layers stay in
            inference mode throughout; updating their statistics at this batch size quietly
            degrades the model.
          </Row>
          <Row term="Class imbalance">
            Normal outnumbers Viral Pneumonia about 7.6 to 1. Handled with per-sample class
            weights alone — oversampling on top would distort the calibration analysis.
          </Row>
          <Row term="Model selection">
            The checkpoint with the best validation macro-F1 was kept, not the best accuracy,
            which a model can earn by favouring the large classes.
          </Row>
        </dl>

        <p className="max-w-3xl">
          The lungs-erased model&apos;s best checkpoint was its last, epoch 15 of 15: it was still
          improving when training stopped, so its 97.2% of the full-image score is a floor rather
          than a ceiling. The study also trained a lungs-only model and fitted a linear model to
          8×8 thumbnails; neither runs here. The thumbnail model, with no anatomy resolvable at
          all, still separates COVID from Lung Opacity at AUC 0.81.
        </p>
      </Section>

      <Section id="prediction" title="How a prediction is made in your browser">
        <ol className="max-w-3xl list-decimal space-y-3 pl-5 marker:text-neutral-500">
          <li>
            <strong className="text-neutral-100">The images are prepared ahead of time.</strong>{' '}
            Each gallery radiograph, and its lungs-erased copy, was resized to exactly 224×224 in
            Python when the gallery was built. The browser never resizes them: its resampling
            differs from Python&apos;s, and the difference would shift pixels without raising any
            error.
          </li>
          <li>
            <strong className="text-neutral-100">Each model downloads once.</strong> A model is
            the trained Keras network converted to TensorFlow.js — about 27 MB of weights,
            fetched from this site the first time you use it. Nothing travels the other way.
          </li>
          <li>
            <strong className="text-neutral-100">The pixels become a tensor.</strong> The image is
            copied onto an off-screen canvas at its native size and read as red, green and blue
            values from 0 to 255: a 1 × 224 × 224 × 3 array. DenseNet&apos;s input scaling is
            built into the converted model, so the page applies no preprocessing of its own.
          </li>
          <li>
            <strong className="text-neutral-100">The network runs on your device.</strong>{' '}
            TensorFlow.js executes it on your graphics card through WebGL where the browser
            allows, otherwise on the CPU. Out come four probabilities — COVID, Lung Opacity,
            Normal, Viral Pneumonia — that sum to 100%.
          </li>
          <li>
            <strong className="text-neutral-100">The model prediction is the largest of the four.</strong>{' '}
            No threshold and no rules on top. A model made to choose among four classes always
            names one, however unlike its training data the input is.
          </li>
          <li>
            <strong className="text-neutral-100">The browser is checked against Python.</strong>{' '}
            Every gallery image ships with the probabilities the original Keras model produced in
            Python. If live inference fails, the demo shows those instead and says so.
          </li>
        </ol>
      </Section>

      <Section id="reading" title="Reading the demo">
        <dl>
          <Row term="True label">
            The class the dataset assigns — which, as above, records where the image came from as
            much as what it shows.
          </Row>
          <Row term="Model prediction">
            The selected model&apos;s top class and its probability. Toggling between full image
            and lungs erased switches which model this is.
          </Row>
          <Row term="Full image vs. lungs erased">
            Whether erasing the lungs changes the top class, and how far its probability moves.
          </Row>
          <Row term="Parity Δ">
            The largest per-class gap between this browser&apos;s probabilities and Python&apos;s
            for the same image. Values around 10⁻⁶ mean the browser reproduced Python to rounding
            error.
          </Row>
          <Row term="Attention">
            A Grad-CAM heatmap from the full-image model, precomputed in Python — not computed in
            your browser. It exists only for the full image.
          </Row>
          <Row term="Lung attribution ratio">
            The share of that heatmap falling inside the lung mask. 0.238 is what a model ignoring
            the image scores; 0.376 is what a model that can see nothing but lungs scores.
          </Row>
        </dl>
      </Section>

      <p className="text-sm">
        <Link href="/" className="text-sky-300 underline underline-offset-2 hover:text-sky-200">
          ← Back to the demo
        </Link>
      </p>
    </main>
  )
}
