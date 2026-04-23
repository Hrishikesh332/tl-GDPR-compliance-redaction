import analyzeIconSvg from '../../strand/icons/analyze.svg?raw'
import embedIconSvg from '../../strand/icons/embed.svg?raw'
import entityCollectionIconSvg from '../../strand/icons/entity-collection.svg?raw'
import searchIconSvg from '../../strand/icons/search-v2.svg?raw'
import hourglassIconSvg from '../../strand/icons/hourglass.svg?raw'
import rateLimitIconSvg from '../../strand/icons/rate-limit.svg?raw'
import visionDisabledIconSvg from '../../strand/icons/vision-disabled.svg?raw'
import documentListIconSvg from '../../strand/icons/document-list.svg?raw'

const DEMO_VIDEO_EMBED_URL = 'https://www.youtube.com/embed/Uz-WQcANyDg'

function ThemedSvgIcon({
  icon,
  color,
  className = 'w-5 h-5',
}: {
  icon: string
  color: string
  className?: string
}) {
  return (
    <span
      aria-hidden
      className={`block ${className} [&>svg]:block [&>svg]:h-full [&>svg]:w-full`}
      style={{ color }}
      dangerouslySetInnerHTML={{ __html: icon }}
    />
  )
}

function FeatureCard({
  icon,
  title,
  description,
  iconColor,
  iconBackground,
}: {
  icon: string
  title: string
  description: string
  iconColor: string
  iconBackground: string
}) {
  return (
    <div className="rounded-xl border border-border bg-background p-5 flex flex-col gap-3 shadow-sm">
      <div
        className="w-11 h-11 rounded-xl flex items-center justify-center border border-black/5"
        style={{
          backgroundColor: iconBackground,
        }}
      >
        <ThemedSvgIcon icon={icon} color={iconColor} />
      </div>
      <h3 className="text-base font-brand-bold text-text-primary">{title}</h3>
      <p className="text-sm text-text-secondary leading-relaxed">{description}</p>
    </div>
  )
}

function StepCard({ step, title, description }: { step: number; title: string; description: string }) {
  return (
    <article className="group rounded-2xl border border-border bg-surface/95 p-5 sm:p-6 shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md">
      <div className="flex items-start gap-4">
        <div className="w-10 h-10 rounded-full bg-accent text-brand-charcoal flex items-center justify-center text-sm font-brand-bold shadow-sm shrink-0">
          {step}
        </div>
        <div>
          <h4 className="text-base font-brand-bold text-text-primary">{title}</h4>
          <p className="text-sm text-text-secondary mt-1.5 leading-relaxed">{description}</p>
        </div>
      </div>
    </article>
  )
}

function ProblemCard({
  title,
  description,
  icon,
  iconColor,
  iconBackground,
}: {
  title: string
  description: string
  icon: string
  iconColor: string
  iconBackground: string
}) {
  return (
    <article className="rounded-xl border border-border bg-background p-4 sm:p-5 shadow-sm">
      <div
        className="w-10 h-10 rounded-xl border border-black/5 flex items-center justify-center mb-3"
        style={{ backgroundColor: iconBackground }}
      >
        <ThemedSvgIcon icon={icon} color={iconColor} className="w-4 h-4" />
      </div>
      <h3 className="text-sm font-brand-bold text-text-primary">{title}</h3>
      <p className="mt-1.5 text-sm text-text-secondary leading-relaxed">{description}</p>
    </article>
  )
}

export default function OverviewPage() {
  return (
    <div className="w-full max-w-[var(--strand-size-content-max)] mx-auto px-3 sm:px-4 lg:px-5 py-5 sm:py-8">
      <section className="py-6 sm:py-8">
        <div className="max-w-3xl">
          <p className="text-xs font-brand-xbold text-accent uppercase tracking-widest mb-3">
            GDPR Compliance
          </p>
          <h1 className="text-h3 sm:text-h2 font-brand-bold text-text-primary leading-tight">
            Automated Video Redaction
          </h1>
          <p className="mt-4 text-base sm:text-lg text-text-secondary leading-relaxed">
            Detect, track, and blur faces and objects in video footage to comply with GDPR
            and data privacy regulations. Powered by TwelveLabs video understanding and
            local computer vision.
          </p>
          <div className="flex flex-wrap gap-3 mt-7">
            <a
              href="https://docs.twelvelabs.io"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-accent text-brand-charcoal text-sm font-brand-bold hover:bg-accent-hover transition-colors"
            >
              Docs
            </a>
            <a
              href="https://github.com/Hrishikesh332/tl-GDPR-compliance-redaction"
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

      <section className="relative left-1/2 w-screen -translate-x-1/2 border-t border-black/80 bg-surface py-10 sm:py-14">
        <div className="max-w-[1200px] mx-auto px-3 sm:px-4 lg:px-5">
          <div className="max-w-2xl mx-auto text-center">
            <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider bg-[var(--strand-product-search-light)] text-[var(--strand-product-search-dark)] mb-3">
              Problem It Solves
            </span>
            <h2 className="text-h5 sm:text-h4 font-brand-bold text-text-primary mb-2">
              Why Video Redaction Is Hard at Broadcast Scale
            </h2>
            <p className="text-sm sm:text-base text-text-secondary leading-relaxed">
              Manual, frame-by-frame compliance workflows are expensive, slow, and hard to prove
              during audits. This blocks archive monetization and increases regulatory risk.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 gap-4 mt-6">
            <ProblemCard
              title="Manual Review Is Cost-Prohibitive"
              description="Frame-by-frame review does not scale well for long-form footage, high-volume archives, or repeated compliance checks."
              icon={hourglassIconSvg}
              iconColor="var(--strand-product-generate-dark)"
              iconBackground="var(--strand-product-generate-light)"
            />
            <ProblemCard
              title="Compliance Bottlenecks Distribution"
              description="Release workflows slow down when teams need to manually inspect footage before publication, licensing, or reuse."
              icon={rateLimitIconSvg}
              iconColor="var(--strand-product-embed-dark)"
              iconBackground="var(--strand-product-embed-light)"
            />
            <ProblemCard
              title="Manual Blurring Is Inconsistent"
              description="Manual redaction is prone to misses, inconsistency, and rework, especially when subjects move across long scenes."
              icon={visionDisabledIconSvg}
              iconColor="var(--strand-product-search-dark)"
              iconBackground="var(--strand-product-search-light)"
            />
            <ProblemCard
              title="No Reliable Audit Trail"
              description="Manual workflows often fail to produce evidence needed to demonstrate due diligence to regulators."
              icon={documentListIconSvg}
              iconColor="var(--strand-ui-accent-hover)"
              iconBackground="var(--strand-ui-accent-light)"
            />
          </div>
        </div>
      </section>

      <section className="relative left-1/2 w-screen -translate-x-1/2 border-t border-black bg-background py-10 sm:py-14">
        <div className="max-w-[1200px] mx-auto px-3 sm:px-4 lg:px-5">
          <div className="text-center mb-6 sm:mb-8">
            <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider bg-[var(--strand-product-embed-light)] text-[var(--strand-product-embed-dark)] mb-3">
              Architecture
            </span>
            <h2 className="font-brand text-h4 sm:text-h3 font-medium text-text-primary">
              System Architecture
            </h2>
            <p className="mt-3 text-text-secondary max-w-xl mx-auto text-sm">
              End-to-end pipeline from video upload through TwelveLabs indexing and analysis,
              followed by local detection, tracking, blur, and MP4 export.
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

      <section id="demo-video" className="relative left-1/2 w-screen -translate-x-1/2 border-t border-black bg-background py-10 sm:py-14">
        <div className="max-w-[1200px] mx-auto px-3 sm:px-4 lg:px-5">
          <div className="text-center mb-6 sm:mb-8">
            <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider bg-[var(--strand-product-generate-light)] text-[var(--strand-product-generate-dark)] mb-3">
              Demo
            </span>
            <h2 className="font-brand text-h4 sm:text-h3 font-medium text-text-primary">
              See It in Action
            </h2>
            <p className="mt-3 text-text-secondary max-w-xl mx-auto text-sm">
              Watch how the GDPR Video Redaction app processes a video from upload to redacted
              export with TwelveLabs-guided analysis and local blur tracking.
            </p>
          </div>
          <div className="rounded-2xl border border-border bg-brand-charcoal aspect-video max-w-4xl mx-auto overflow-hidden relative shadow-sm">
            <iframe
              src={DEMO_VIDEO_EMBED_URL}
              title="GDPR Video Redaction demo"
              className="absolute inset-0 h-full w-full"
              loading="lazy"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              referrerPolicy="strict-origin-when-cross-origin"
              allowFullScreen
            />
          </div>
        </div>
      </section>

      <section className="relative left-1/2 w-screen -translate-x-1/2 border-t border-black/80 bg-surface py-10 sm:py-14">
        <div className="max-w-[1200px] mx-auto px-3 sm:px-4 lg:px-5">
          <div className="mb-6 sm:mb-8 text-center max-w-2xl mx-auto">
            <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider bg-[var(--strand-ui-accent-light)] text-[var(--strand-ui-accent-hover)] mb-3">
              Features
            </span>
            <h2 className="text-h5 sm:text-h4 font-brand-bold text-text-primary mb-2">Core Features</h2>
            <p className="text-sm sm:text-base text-text-secondary">
              Everything you need for GDPR-compliant video anonymization.
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <FeatureCard
              icon={searchIconSvg}
              title="TwelveLabs Video Search"
              description="Use the TwelveLabs Marengo model to search indexed footage with natural language, images, and saved entities, then jump straight to relevant clips."
              iconColor="var(--strand-product-search-dark)"
              iconBackground="var(--strand-product-search-light)"
            />
            <FeatureCard
              icon={analyzeIconSvg}
              title="TwelveLabs Video Analysis"
              description="Use the TwelveLabs Pegasus model to generate scene, people, and object context that helps guide review and redaction workflows."
              iconColor="var(--strand-product-generate-dark)"
              iconBackground="var(--strand-product-generate-light)"
            />
            <FeatureCard
              icon={entityCollectionIconSvg}
              title="Entity Management"
              description="Build a library of known individuals with face crops. Select which persons to redact or exclude across your video library."
              iconColor="var(--strand-ui-accent-hover)"
              iconBackground="var(--strand-ui-accent-light)"
            />
            <FeatureCard
              icon={embedIconSvg}
              title="Temporal Optimization"
              description="Use available time ranges from selected faces and entities to focus redaction on the parts of the video where targets appear."
              iconColor="var(--strand-product-embed-dark)"
              iconBackground="var(--strand-product-embed-light)"
            />
          </div>
        </div>
      </section>

      <section className="relative left-1/2 w-screen -translate-x-1/2 border-t border-black/80 bg-gradient-to-b from-background via-surface to-card/60 py-10 sm:py-14">
        <div className="max-w-[1200px] mx-auto px-3 sm:px-4 lg:px-5">
          <div className="text-center max-w-2xl mx-auto mb-6 sm:mb-8">
            <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider bg-[var(--strand-product-embed-light)] text-[var(--strand-product-embed-dark)] mb-3">
              Workflow
            </span>
            <h2 className="text-h5 sm:text-h4 font-brand-bold text-text-primary mb-2">How It Works</h2>
            <p className="text-sm sm:text-base text-text-secondary leading-relaxed">
              The app uses TwelveLabs to understand the video, then performs the actual
              redaction flow with local detection, tracking, and export logic.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 lg:gap-5">
            <StepCard
              step={1}
              title="Index & Analyze"
              description="Upload the video to TwelveLabs for indexing, then generate people, object, and scene summaries used throughout the review flow."
            />
            <StepCard
              step={2}
              title="Detect Targets Locally"
              description="Run local face recognition and object detection against selected people, entities, object classes, and manual regions."
            />
            <StepCard
              step={3}
              title="Track Through Motion"
              description="Use the best available OpenCV tracker for the current environment, with optical-flow or static fallback when tracker support is limited."
            />
            <StepCard
              step={4}
              title="Render & Export"
              description="Render the redacted output, then re-encode to H.264 MP4 when ffmpeg is available for broader playback compatibility."
            />
          </div>
        </div>
      </section>

      <section className="relative left-1/2 w-screen -translate-x-1/2 border-t border-black/10 bg-background py-16 sm:py-20">
        <div className="max-w-4xl mx-auto px-3 sm:px-4 lg:px-5 text-center">
          <h2 className="text-[32px] sm:text-[44px] leading-[1.08] font-brand-bold text-text-primary tracking-[-0.02em]">
            Ready to Get Started with Automated GDPR Compliance?
          </h2>
          <p className="mt-4 text-base sm:text-lg text-text-secondary">
            Upload your first video and see results in seconds.
          </p>

          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <a
              href="https://github.com/Hrishikesh332/tl-GDPR-compliance-redaction"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex min-w-[148px] items-center justify-center rounded-[22px] bg-brand-charcoal px-6 py-3 text-base font-brand-bold text-brand-white transition-colors hover:bg-gray-700"
            >
              View Code
            </a>
            <a
              href="https://www.twelvelabs.io/contact"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex min-w-[148px] items-center justify-center rounded-[22px] border border-border bg-background px-6 py-3 text-base font-brand-bold text-text-primary transition-colors hover:bg-card"
            >
              Talk to Sales
            </a>
          </div>
        </div>
      </section>

    </div>
  )
}
