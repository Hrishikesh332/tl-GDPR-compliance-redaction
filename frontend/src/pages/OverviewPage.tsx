import searchIconUrl from '../../strand/icons/search-v2.svg?url'
import analyzeIconUrl from '../../strand/icons/analyze.svg?url'
import entityIconUrl from '../../strand/icons/entity.svg?url'
import neuralNetworkIconUrl from '../../strand/icons/neural-network.svg?url'
import logoMarkUrl from '../../strand/assets/logo-mark.svg?url'

function FeatureCard({ icon, title, description, color }: { icon: string; title: string; description: string; color: string }) {
  return (
    <div className="rounded-xl border border-border bg-surface p-5 flex flex-col gap-3">
      <div
        className="w-10 h-10 rounded-lg flex items-center justify-center"
        style={{ backgroundColor: color }}
      >
        <img src={icon} alt="" className="w-5 h-5 invert" />
      </div>
      <h3 className="text-base font-brand-bold text-text-primary">{title}</h3>
      <p className="text-sm text-text-secondary leading-relaxed">{description}</p>
    </div>
  )
}

function StepCard({ step, title, description }: { step: number; title: string; description: string }) {
  return (
    <div className="flex gap-4">
      <div className="flex flex-col items-center shrink-0">
        <div className="w-8 h-8 rounded-full bg-accent text-brand-charcoal flex items-center justify-center text-sm font-brand-bold">
          {step}
        </div>
        {step < 4 && <div className="w-px flex-1 bg-border mt-2" />}
      </div>
      <div className="pb-6">
        <h4 className="text-sm font-brand-bold text-text-primary">{title}</h4>
        <p className="text-sm text-text-secondary mt-1 leading-relaxed">{description}</p>
      </div>
    </div>
  )
}

export default function OverviewPage() {
  return (
    <div className="w-full max-w-[var(--strand-size-content-max)] mx-auto px-4 sm:px-6 py-6 sm:py-10 space-y-10">

      {/* Hero */}
      <section className="relative rounded-2xl border border-border overflow-hidden">
        <div className="overview-hero-gradient absolute inset-0 opacity-[0.07]" />
        <div className="relative px-6 sm:px-10 py-10 sm:py-14">
          <p className="text-xs font-brand-xbold text-accent uppercase tracking-widest mb-3">
            GDPR Compliance
          </p>
          <h1 className="text-h3 sm:text-h2 font-brand-bold text-text-primary leading-tight max-w-2xl">
            Automated Video Redaction
          </h1>
          <p className="mt-4 text-base sm:text-lg text-text-secondary max-w-xl leading-relaxed">
            Detect, track, and blur faces and objects in video footage to comply with GDPR
            and data privacy regulations. Powered by TwelveLabs video understanding and
            local computer vision.
          </p>
          <div className="flex flex-wrap gap-3 mt-8">
            <a
              href="https://docs.twelvelabs.io"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-accent text-brand-charcoal text-sm font-brand-bold hover:bg-accent-hover transition-colors"
            >
              Docs
            </a>
            <a
              href="https://github.com/twelvelabs-io"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg border border-border text-text-primary text-sm font-brand-bold hover:bg-card transition-colors"
            >
              Code Repo
            </a>
            <a
              href="https://www.twelvelabs.io/contact"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg border border-border text-text-primary text-sm font-brand-bold hover:bg-card transition-colors"
            >
              Talk to Sales
            </a>
          </div>
        </div>
      </section>

      {/* ─── Architecture Diagram (reference style) ───────────────────── */}
      <section className="bg-background border-t border-border -mx-4 sm:-mx-6 px-4 sm:px-6 py-14 sm:py-20">
        <div className="max-w-[1200px] mx-auto">
          <div className="text-center mb-8 sm:mb-10">
            <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider bg-[var(--strand-product-embed-light)] text-[var(--strand-product-embed-dark)] mb-3">
              Architecture
            </span>
            <h2 className="font-brand text-h4 sm:text-h3 font-medium text-text-primary">
              System Architecture
            </h2>
            <p className="mt-3 text-text-secondary max-w-xl mx-auto text-sm">
              End-to-end pipeline from video upload through TwelveLabs indexing, Pegasus analysis,
              and local redaction (face/object detection, tracking, blur, H.264 export).
            </p>
          </div>
          <div className="rounded-2xl border-2 border-dashed border-gray-300 bg-surface flex flex-col items-center justify-center min-h-[320px] sm:min-h-[400px] p-8">
            <div className="w-14 h-14 rounded-xl bg-card flex items-center justify-center mb-4">
              <svg className="w-7 h-7 text-text-tertiary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <path d="M3 9h18M9 21V9" />
              </svg>
            </div>
            <p className="text-text-secondary font-medium text-sm">Architecture Diagram</p>
            <p className="text-text-tertiary text-xs mt-1.5 max-w-xs text-center">
              Drop your architecture diagram image here or replace this placeholder with an
              &lt;img&gt; tag pointing to your diagram asset.
            </p>
          </div>
        </div>
      </section>

      {/* ─── Demo Video (reference style) ─────────────────────────────── */}
      <section id="demo-video" className="bg-surface border-t border-border -mx-4 sm:-mx-6 px-4 sm:px-6 py-14 sm:py-20">
        <div className="max-w-[1200px] mx-auto">
          <div className="text-center mb-8 sm:mb-10">
            <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider bg-[var(--strand-product-generate-light)] text-[var(--strand-product-generate-dark)] mb-3">
              Demo
            </span>
            <h2 className="font-brand text-h4 sm:text-h3 font-medium text-text-primary">
              See It in Action
            </h2>
            <p className="mt-3 text-text-secondary max-w-xl mx-auto text-sm">
              Watch how the GDPR Video Redaction app processes a video from upload to redacted
              export with Pegasus analysis and local blur tracking.
            </p>
          </div>
          <div className="rounded-2xl border-2 border-dashed border-gray-300 bg-brand-charcoal flex flex-col items-center justify-center aspect-video max-w-4xl mx-auto overflow-hidden relative">
            <div className="w-16 h-16 rounded-full bg-white/10 backdrop-blur-sm flex items-center justify-center mb-4 border border-white/20">
              <svg className="w-7 h-7 text-white ml-0.5" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
            </div>
            <p className="text-white/60 font-medium text-sm">Demo Video</p>
            <p className="text-white/40 text-xs mt-1.5 max-w-xs text-center">
              Replace this placeholder with a &lt;video&gt; element or an iframe embed
              pointing to your demo recording.
            </p>
          </div>
        </div>
      </section>

      {/* Core Features */}
      <section>
        <h2 className="text-h5 font-brand-bold text-text-primary mb-1">Core Features</h2>
        <p className="text-sm text-text-secondary mb-6">
          Everything you need for GDPR-compliant video anonymization.
        </p>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <FeatureCard
            icon={logoMarkUrl}
            title="Pegasus Analysis"
            description="Detect the crucial parts of the video with TwelveLabs Pegasus: scene understanding, temporal grounding, and natural language search to find and describe subjects."
            color="var(--strand-product-search-dark)"
          />
          <FeatureCard
            icon={searchIconUrl}
            title="Object Detection (YOLO)"
            description="Apply YOLOv8 for the most optimized process: detect and redact object classes like license plates, phones, and screens with adjustable confidence thresholds."
            color="var(--strand-product-generate-dark)"
          />
          <FeatureCard
            icon={entityIconUrl}
            title="Entity Management"
            description="Build a library of known individuals with face crops. Select which persons to redact or exclude across your video library."
            color="var(--strand-product-generate-dark)"
          />
          <FeatureCard
            icon={logoMarkUrl}
            title="Temporal Optimization"
            description="Use TwelveLabs temporal grounding to skip frames where targets are absent, reducing detection calls by 50-80%."
            color="var(--strand-product-embed-dark)"
          />
        </div>
      </section>

      {/* How It Works */}
      <section className="grid lg:grid-cols-[1fr_1.4fr] gap-8 items-start">
        <div>
          <h2 className="text-h5 font-brand-bold text-text-primary mb-1">How It Works</h2>
          <p className="text-sm text-text-secondary leading-relaxed">
            The pipeline combines cloud-based video intelligence with local real-time
            processing for privacy-safe redaction.
          </p>
        </div>
        <div className="rounded-xl border border-border bg-surface p-5 sm:p-6">
          <StepCard
            step={1}
            title="Upload & Index"
            description="Upload video to TwelveLabs for indexing. The platform generates scene descriptions, person appearances, and temporal segments."
          />
          <StepCard
            step={2}
            title="Detect & Identify"
            description="Run face recognition and object detection locally. Match faces against target encodings and detect objects by class."
          />
          <StepCard
            step={3}
            title="Track & Blur"
            description="Initialize KCF/CSRT trackers on detected bounding boxes. Track across frames and apply Gaussian blur at full resolution."
          />
          <StepCard
            step={4}
            title="Export H.264"
            description="Re-encode the redacted video as H.264 MP4 for universal playback in browsers and media players."
          />
        </div>
      </section>

    </div>
  )
}
