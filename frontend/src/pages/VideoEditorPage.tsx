import React, { useState, useRef, useEffect, useCallback, useMemo, useLayoutEffect, useId } from 'react'
import { useParams, Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import Hls from 'hls.js'
import { useVideoCache } from '../contexts/VideoCache'
import { API_BASE } from '../lib/api'
import { storeLastEditorVideoId, DEMO_EDITOR_VIDEO_ID } from '../lib/editorRouting'
import { EDITOR_LAST_SEARCH_SESSION_KEY } from '../lib/searchSession'
import SnapFaceFromVideoModal, { type SnapFaceResult } from '../components/SnapFaceFromVideoModal'
import visionIconUrl from '../../strand/icons/vision.svg?url'
import searchV2IconUrl from '../../strand/icons/search-v2.svg?url'
import analyzeIconUrl from '../../strand/icons/analyze.svg?url'
import metaInsightsIconUrl from '../../strand/icons/embed.svg?url'

const ANALYZE_SUGGESTIONS: string[] = [
  'Courtroom summary with key timestamps',
  'When does the primary suspect appear?',
  'Suspect and law enforcement interactions',
  'Which faces should be anonymized?',
]

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

function isHlsUrl(url: string): boolean {
  return /\.m3u8(\?|$)/i.test(url) || url.includes('m3u8')
}

function toHlsProxyUrl(hlsUrl: string): string {
  try {
    const u = new URL(hlsUrl)
    return `${API_BASE}/api/hls-proxy/${u.host}${u.pathname}${u.search}`
  } catch {
    return hlsUrl
  }
}

/** Parse "m:ss" or "mm:ss" or "h:mm:ss" to seconds */
function parseTimestampToSeconds(timestamp: string): number {
  const parts = timestamp.trim().split(':').map((s) => parseInt(s, 10))
  if (parts.length === 2) return parts[0] * 60 + parts[1]
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2]
  return 0
}

const TIMESTAMP_REGEX = /(\d{1,2}:\d{2}(?::\d{2})?)/g

function TimestampTag({ time, onSeek, label }: { time: string; onSeek: (seconds: number) => void; label?: string }) {
  const seconds = parseTimestampToSeconds(time)
  return (
    <button
      type="button"
      onClick={() => onSeek(seconds)}
      className="inline-flex items-center px-1.5 py-0.5 rounded text-[11px] font-medium bg-[#00DC82] text-black border border-[#00DC82] hover:bg-[#00c872] hover:border-[#00c872] transition-colors cursor-pointer align-baseline"
      title={`Seek to ${time}`}
    >
      {label ?? time}
    </button>
  )
}

/** Recursively replace timestamp strings (mm:ss or h:mm:ss) in React children with clickable TimestampTags. Skip inside code/pre. */
function withTimestampLinks(children: React.ReactNode, onSeek: (seconds: number) => void, insideCode = false): React.ReactNode {
  return React.Children.map(children, (child) => {
    if (insideCode) return child
    if (typeof child === 'string') {
      const parts = child.split(TIMESTAMP_REGEX)
      if (parts.length <= 1) return child
      return parts.map((part, i) =>
        /^\d{1,2}:\d{2}(?::\d{2})?$/.test(part) ? (
          <TimestampTag key={`${i}-${part}`} time={part} onSeek={onSeek} />
        ) : (
          part
        )
      )
    }
    if (React.isValidElement(child)) {
      const tag = typeof child.type === 'string' ? child.type : ''
      const isCode = tag === 'code' || tag === 'pre'
      return React.cloneElement(child, {
        ...child.props,
        children: withTimestampLinks(child.props.children, onSeek, isCode || insideCode),
      } as Record<string, unknown>)
    }
    return child
  })
}

function normalizeWaveform(values: number[]): number[] {
  if (!values.length) return []
  const nonZero = values.filter((value) => value > 0)
  const referenceValues = nonZero.length > 0 ? nonZero : values
  const sorted = [...referenceValues].sort((a, b) => a - b)
  const peakIndex = Math.max(0, Math.min(sorted.length - 1, Math.floor(sorted.length * 0.96)))
  const referencePeak = Math.max(sorted[peakIndex] || 0, 0.001)
  return values.map((value) => {
    const normalized = Math.min(1, value / referencePeak)
    return Math.pow(normalized, 0.72)
  })
}

function extractWaveformFromAudioBuffer(audioBuffer: AudioBuffer, sampleCount: number): number[] {
  if (sampleCount <= 0 || audioBuffer.length <= 0 || audioBuffer.numberOfChannels <= 0) return []
  const bucketSize = Math.max(1, Math.floor(audioBuffer.length / sampleCount))
  const peakValues: number[] = []

  for (let bucket = 0; bucket < sampleCount; bucket++) {
    const start = bucket * bucketSize
    const end = bucket === sampleCount - 1 ? audioBuffer.length : Math.min(audioBuffer.length, start + bucketSize)
    if (end <= start) {
      peakValues.push(0)
      continue
    }

    const step = Math.max(1, Math.floor((end - start) / 600))
    let bucketPeak = 0
    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
      const channelData = audioBuffer.getChannelData(channel)
      for (let index = start; index < end; index += step) {
        bucketPeak = Math.max(bucketPeak, Math.abs(channelData[index] || 0))
      }
    }
    peakValues.push(bucketPeak)
  }

  return normalizeWaveform(peakValues)
}

function generateWaveformPlaceholder(sampleCount: number): number[] {
  if (sampleCount <= 0) return []
  return Array.from({ length: sampleCount }, (_, index) => {
    const progress = sampleCount === 1 ? 0 : index / (sampleCount - 1)
    const envelope = 0.28 + 0.18 * Math.sin(progress * Math.PI * 1.2)
    const detail = 0.1 * Math.sin(progress * Math.PI * 8 + 0.4) + 0.08 * Math.cos(progress * Math.PI * 17)
    return Math.max(0.08, Math.min(0.72, envelope + detail))
  })
}

function buildWaveformPath(samples: number[], width: number, height: number, paddingY = 8): string {
  if (!samples.length || width <= 0 || height <= 0) return ''
  const centerY = height / 2
  const usableHeight = Math.max(1, height / 2 - paddingY)
  const lastIndex = Math.max(1, samples.length - 1)
  const topPoints = samples.map((sample, index) => {
    const x = (index / lastIndex) * width
    const amplitude = Math.max(0.04, Math.min(1, sample))
    const y = centerY - amplitude * usableHeight
    return `${x.toFixed(2)},${y.toFixed(2)}`
  })
  const bottomPoints = [...samples].reverse().map((sample, reversedIndex) => {
    const index = samples.length - 1 - reversedIndex
    const x = (index / lastIndex) * width
    const amplitude = Math.max(0.04, Math.min(1, sample))
    const y = centerY + amplitude * usableHeight
    return `${x.toFixed(2)},${y.toFixed(2)}`
  })
  return `M ${topPoints.join(' L ')} L ${bottomPoints.join(' L ')} Z`
}

function getSearchRankBucketWeight(rank: number): number {
  if (!Number.isFinite(rank) || rank <= 0) return 0.3

  const bucketIndex = Math.floor((rank - 1) / 10)
  const bucketWeight = Math.max(0.22, 1 - bucketIndex * 0.16)
  const rankInBucket = (rank - 1) % 10
  const withinBucketProgress = 1 - (rankInBucket / 9)
  const withinBucketWeight = 0.94 + withinBucketProgress * 0.06

  return Math.max(0.22, Math.min(1, bucketWeight * withinBucketWeight))
}

function getSearchClipImportance(
  clip: { rank?: number; score?: number },
  maxRank: number,
): number {
  const score = typeof clip.score === 'number' && Number.isFinite(clip.score) ? clip.score : 0
  const scoreWeight = Math.max(0.7, Math.min(1, 0.78 + score * 2.2))

  if (clip.rank != null && maxRank > 0) {
    const rankWeight = getSearchRankBucketWeight(clip.rank)
    return Math.max(0.12, Math.min(1, rankWeight * scoreWeight))
  }

  return Math.max(0.18, Math.min(0.42, scoreWeight * (maxRank > 0 ? 0.55 : 1)))
}

function buildSearchWaveformSamples(
  clips: Array<{ start: number; end: number; rank?: number; score?: number }>,
  duration: number,
  sampleCount: number,
): number[] {
  if (duration <= 0 || sampleCount <= 0 || clips.length === 0) return []

  const rankedClips = clips.filter((clip) => clip.rank != null)
  const maxRank = rankedClips.length > 0
    ? Math.max(...rankedClips.map((clip) => clip.rank as number))
    : 0
  const samples = new Array(sampleCount).fill(0)

  for (const clip of clips) {
    const start = Math.max(0, Math.min(duration, clip.start))
    const end = Math.max(start, Math.min(duration, clip.end))
    if (end <= start) continue

    const startIndex = Math.max(0, Math.floor((start / duration) * sampleCount))
    const endIndex = Math.min(
      sampleCount - 1,
      Math.max(startIndex, Math.ceil((end / duration) * sampleCount) - 1),
    )
    const clipImportance = getSearchClipImportance(clip, maxRank)
    const span = Math.max(1, endIndex - startIndex)

    for (let index = startIndex; index <= endIndex; index += 1) {
      const progress = span <= 1 ? 0.5 : (index - startIndex) / span
      const centerEnvelope = Math.sin(Math.PI * progress)
      const shapedAmplitude = clipImportance * (0.26 + 0.74 * Math.pow(Math.max(0, centerEnvelope), 0.82))
      samples[index] = Math.max(samples[index], shapedAmplitude)
    }
  }

  const smoothed = samples.map((sample, index) => {
    const prev = samples[index - 1] ?? sample
    const next = samples[index + 1] ?? sample
    return prev * 0.18 + sample * 0.64 + next * 0.18
  })
  const peak = Math.max(...smoothed, 0.001)
  return smoothed.map((value) => Math.pow(Math.min(1, value / peak), 1.28))
}

function buildTimelineBarSeries(samples: number[], barCount: number): number[] {
  if (!samples.length || barCount <= 0) return []
  const bars: number[] = []
  for (let barIndex = 0; barIndex < barCount; barIndex += 1) {
    const start = Math.floor((barIndex / barCount) * samples.length)
    const end = Math.max(start + 1, Math.floor(((barIndex + 1) / barCount) * samples.length))
    let peak = 0
    for (let sampleIndex = start; sampleIndex < Math.min(samples.length, end); sampleIndex += 1) {
      peak = Math.max(peak, samples[sampleIndex] || 0)
    }
    bars.push(peak)
  }
  return bars
}

async function seekMediaElement(video: HTMLVideoElement, time: number, timeoutMs: number): Promise<boolean> {
  return new Promise((resolve) => {
    const done = (result: boolean) => {
      video.removeEventListener('seeked', onSeeked)
      video.removeEventListener('error', onError)
      clearTimeout(timeoutId)
      resolve(result)
    }
    const onSeeked = () => done(true)
    const onError = () => done(false)
    const timeoutId = window.setTimeout(() => done(false), timeoutMs)
    video.addEventListener('seeked', onSeeked)
    video.addEventListener('error', onError)
    video.currentTime = Math.max(0, time)
  })
}

async function sampleWaveformFromMediaElement(
  video: HTMLVideoElement,
  duration: number,
  sampleCount: number,
  timeoutMs: number,
): Promise<number[]> {
  if (duration <= 0 || sampleCount <= 0) return []

  const AudioContextCtor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  const audioCtx = new AudioContextCtor()
  const analyser = audioCtx.createAnalyser()
  const gain = audioCtx.createGain()
  analyser.fftSize = 2048
  analyser.smoothingTimeConstant = 0.45
  gain.gain.value = 0

  const source = audioCtx.createMediaElementSource(video)
  source.connect(analyser)
  analyser.connect(gain)
  gain.connect(audioCtx.destination)

  const bufferLength = analyser.frequencyBinCount
  const dataArray = new Uint8Array(bufferLength)
  const sampledWaveform: number[] = []
  const originalMuted = video.muted
  const originalVolume = video.volume
  const originalPlaybackRate = video.playbackRate

  try {
    if (audioCtx.state === 'suspended') await audioCtx.resume()
    video.muted = true
    video.volume = 0
    video.playbackRate = 8

    for (let index = 0; index < sampleCount; index += 1) {
      const t = sampleCount === 1 ? 0 : (index / (sampleCount - 1)) * duration
      const seeked = await seekMediaElement(video, t, timeoutMs)
      if (!seeked) {
        sampledWaveform.push(0)
        continue
      }

      try {
        await video.play()
      } catch {
        /* autoplay can fail in some browsers, continue with current decoded frame */
      }

      await new Promise((resolve) => window.setTimeout(resolve, 90))
      analyser.getByteTimeDomainData(dataArray)
      let peak = 0
      for (let sampleIndex = 0; sampleIndex < bufferLength; sampleIndex += 1) {
        const value = Math.abs((dataArray[sampleIndex] - 128) / 128)
        if (value > peak) peak = value
      }
      sampledWaveform.push(peak)
      video.pause()
    }

    await seekMediaElement(video, 0, timeoutMs).catch(() => false)
    return normalizeWaveform(sampledWaveform)
  } finally {
    video.pause()
    video.muted = originalMuted
    video.volume = originalVolume
    video.playbackRate = originalPlaybackRate
    audioCtx.close().catch(() => {})
  }
}

/* ------------------------------------------------------------------ */
/*  Toolbar config                                                     */
/* ------------------------------------------------------------------ */

type ToolId = 'tracker' | 'search' | 'captions' | 'pegasus'

type SearchClip = {
  start: number
  end: number
  score?: number
  rank?: number
  thumbnailUrl?: string
  thumbnail_url?: string
}

type SearchVideoResult = {
  id: string
  video_id?: string
  title?: string
  clips?: SearchClip[]
  searchScore?: number
}

type SearchSessionEntity = {
  id: string
  name: string
  previewUrl?: string
}

type SearchSessionResult = {
  id?: string
  query: string
  queryText?: string
  entities?: SearchSessionEntity[]
  results: SearchVideoResult[]
}

type PegasusJobStatus = 'idle' | 'queued' | 'processing' | 'ready' | 'failed'
type PegasusSeverity = 'low' | 'medium' | 'high'
type PegasusTimelineCategory = 'person' | 'face' | 'screen' | 'document' | 'text' | 'license_plate' | 'logo' | 'object' | 'scene'
type PegasusActionType = 'select_detected_entity' | 'select_object_class' | 'create_review_bookmark' | 'jump_to_time' | 'draw_custom_region_prompt'

type PegasusMetadata = {
  artifact_id?: string
  job_id?: string
  video_id?: string
  local_job_id?: string | null
  source_fingerprint?: string
  model?: string
  prompt_version?: string
  schema_version?: string
  duration_sec?: number
  created_at?: string
  updated_at?: string
  status?: PegasusJobStatus | string
  source_type?: string
  usage?: Record<string, unknown>
}

type PegasusTimelineEvent = {
  id: string
  start_sec: number
  end_sec: number
  severity: PegasusSeverity
  category: PegasusTimelineCategory
  label: string
  description?: string
  reason?: string
  redaction_target?: string | null
  scene_role?: string | null
  redaction_decision?: string | null
  subject_selection?: string | null
  inclusion_reason?: string | null
  handling_note?: string | null
  confidence?: number
  review_required?: boolean
  recommended_action_ids?: string[]
}

type PegasusRecommendedAction = {
  id: string
  type: PegasusActionType
  label: string
  reason?: string
  confidence?: number
  event_ids?: string[]
  target?: {
    object_class?: string | null
    entity_id?: string | null
    redaction_target?: string | null
    scene_role?: string | null
    redaction_decision?: string | null
    subject_selection?: string | null
    inclusion_reason?: string | null
    start_sec?: number
    end_sec?: number
  }
  apply_mode?: 'automatic_if_matched' | 'review_only'
}

type PegasusResult = {
  metadata: PegasusMetadata
  summary: {
    overall_summary?: string
    privacy_risk_level?: PegasusSeverity | string
    review_priority?: PegasusSeverity | string
  }
  timeline_events: PegasusTimelineEvent[]
  recommended_actions: PegasusRecommendedAction[]
  raw_output?: string
}

type PegasusJobResponse = {
  job_id?: string
  status?: PegasusJobStatus | string
  cached?: boolean
  result?: PegasusResult
  error?: string
}

type PegasusApplyPreviewItem = {
  action_id?: string
  type?: string
  selection_id?: string
  person_id?: string
  object_class?: string
  label?: string
  event_ids?: string[]
  reason?: string
}

type PegasusApplyPreview = {
  can_apply: PegasusApplyPreviewItem[]
  review_only: PegasusApplyPreviewItem[]
  unsupported: PegasusApplyPreviewItem[]
  summary?: {
    selected_faces?: number
    selected_object_classes?: number
    review_bookmarks?: number
  }
  error?: string
}

function getSearchSessionId(session: SearchSessionResult, fallbackIndex = 0): string {
  if (session.id) return session.id
  const entityIds = (session.entities || [])
    .map((entity) => entity?.id)
    .filter(Boolean)
    .join('|')
  if (entityIds) return `entities:${entityIds}`
  const resultIds = (session.results || [])
    .map((result) => result?.id || result?.video_id)
    .filter(Boolean)
    .slice(0, 3)
    .join('|')
  return `search:${session.query || ''}:${session.queryText || ''}:${resultIds}:${fallbackIndex}`
}

function withSearchSessionId(session: SearchSessionResult, fallbackIndex = 0): SearchSessionResult {
  return { ...session, id: getSearchSessionId(session, fallbackIndex) }
}

function normalizeStoredSearchSessions(value: unknown): SearchSessionResult[] {
  const items = Array.isArray(value) ? value : [value]
  return items
    .filter((item): item is SearchSessionResult => (
      !!item &&
      typeof item === 'object' &&
      typeof (item as SearchSessionResult).query === 'string' &&
      Array.isArray((item as SearchSessionResult).results)
    ))
    .map((session, index) => withSearchSessionId(session, index))
}

function persistEditorSearchSessions(sessions: SearchSessionResult[]) {
  try {
    if (sessions.length === 0) {
      sessionStorage.removeItem(EDITOR_LAST_SEARCH_SESSION_KEY)
    } else {
      sessionStorage.setItem(EDITOR_LAST_SEARCH_SESSION_KEY, JSON.stringify(sessions))
    }
  } catch {}
}

function getEntityMonogram(name: string): string {
  const trimmed = name.trim()
  return trimmed ? trimmed.charAt(0).toUpperCase() : '?'
}

function SearchEntityChip({
  entity,
  variant = 'sidebar',
  showPrefix = false,
}: {
  entity: SearchSessionEntity
  variant?: 'sidebar' | 'timeline'
  showPrefix?: boolean
}) {
  const isTimeline = variant === 'timeline'
  const label = showPrefix ? `Entity: ${entity.name}` : entity.name

  return (
    <div
      className={isTimeline
        ? 'inline-flex max-w-[12rem] items-center gap-1.5 rounded-full border border-white/12 bg-black/28 px-1.5 py-1 shadow-[0_1px_0_rgba(255,255,255,0.04)] backdrop-blur-sm'
        : 'inline-flex items-center gap-2 rounded-full border border-border bg-card px-2.5 py-1.5'}
      style={isTimeline ? { opacity: 0.84 } : undefined}
    >
      <div
        className={isTimeline
          ? 'h-5 w-5 shrink-0 overflow-hidden rounded-full border border-white/12 bg-white/10'
          : 'h-8 w-8 shrink-0 overflow-hidden rounded-full border border-border bg-surface'}
      >
        {entity.previewUrl ? (
          <img
            src={entity.previewUrl}
            alt={entity.name}
            className="h-full w-full object-cover"
          />
        ) : (
          <div
            className={isTimeline
              ? 'flex h-full w-full items-center justify-center text-[9px] font-semibold text-white/78'
              : 'flex h-full w-full items-center justify-center text-[11px] font-medium text-text-tertiary'}
          >
            {getEntityMonogram(entity.name)}
          </div>
        )}
      </div>
      <span className={isTimeline ? 'truncate text-[10px] font-medium text-white/88' : 'text-xs text-text-primary'}>
        {label}
      </span>
    </div>
  )
}

type VideoViewport = {
  left: number
  top: number
  width: number
  height: number
}

/** Tracking region in normalized coords (0–1) relative to video viewport */
export type TrackingRegion = {
  id: string
  shape: 'rectangle' | 'ellipse' | 'circle'
  x: number
  y: number
  width: number
  height: number
  effect: 'blur' | 'pixelate' | 'solid'
  anchorTime?: number
  reason?: string
  locked?: boolean
}

type RedactionStyle = 'blur' | 'black'
type ExportQuality = 480 | 720 | 1080

const REDACTION_STYLE_OPTIONS: Array<{ value: RedactionStyle; label: string }> = [
  { value: 'blur', label: 'Blur' },
  { value: 'black', label: 'Mask' },
]

const EXPORT_QUALITY_OPTIONS: Array<{ value: ExportQuality; label: string }> = [
  { value: 480, label: '480p' },
  { value: 720, label: '720p' },
  { value: 1080, label: '1080p' },
]

type TrackingPreviewSample = {
  t: number
  x: number
  y: number
  width: number
  height: number
}

type TrackingPreviewRegion = {
  id: string
  samples: TrackingPreviewSample[]
}

const TOOLS: { id: ToolId; label: string; iconUrl: string }[] = [
  { id: 'tracker', label: 'Live Blur', iconUrl: visionIconUrl },
  { id: 'search', label: 'Search', iconUrl: searchV2IconUrl },
  { id: 'captions', label: 'Analyze/Transcript', iconUrl: analyzeIconUrl },
  { id: 'pegasus', label: 'Meta Insights', iconUrl: metaInsightsIconUrl },
]

type TutorialTargetId = 'player' | ToolId | 'timeline' | 'export'
type TutorialPlacement = 'top' | 'right' | 'bottom' | 'left'

const EDITOR_TUTORIAL_STEPS: Array<{
  title: string
  body: string
  target: TutorialTargetId
  placement: TutorialPlacement
}> = [
  {
    title: 'Editor layout',
    body: 'Use the left toolbar to choose a workflow, the center area to review the video, and the right sidebar to work with the selected tool.',
    target: 'player',
    placement: 'bottom',
  },
  {
    title: 'Detect and blur targets',
    body: 'In Live Blur, run Detect to load saved faces and objects. This is where anonymize mode is managed: use Blur or Unblur beside each item to decide what stays redacted.',
    target: 'tracker',
    placement: 'right',
  },
  {
    title: 'Search inside the video',
    body: 'Search results appear as timeline lanes. Green bars show matches over time, and red bars mark the strongest moments to inspect first.',
    target: 'search',
    placement: 'right',
  },
  {
    title: 'Ask for context',
    body: 'Open Analyze/Transcript to ask questions, summarize the scene, or get topics and categories that help guide review.',
    target: 'captions',
    placement: 'right',
  },
  {
    title: 'Review privacy hotspots',
    body: 'Open Meta Insights to run structured privacy analysis and preview privacy hotspots on the timeline before export.',
    target: 'pegasus',
    placement: 'right',
  },
  {
    title: 'Review the timeline',
    body: 'Use the timeline ruler, zoom controls, and result lanes to jump to exact moments. The playhead shows where preview and export decisions apply.',
    target: 'timeline',
    placement: 'top',
  },
  {
    title: 'Export safely',
    body: 'When the preview looks right, open Export, choose the quality, and download the redacted MP4 with the selected faces, objects, and regions applied.',
    target: 'export',
    placement: 'bottom',
  },
]

const EDITOR_TUTORIAL_POPOVER_WIDTH = 336
const EDITOR_TUTORIAL_POPOVER_HEIGHT = 214

const LIVE_DETECTION_POLL_MS = 180
const LIVE_DETECTION_HOLD_MS = 1400
const LIVE_IDENTIFIED_FACE_HOLD_MS = 1800
const LIVE_FACE_PADDING = 0.06
const LIVE_OBJECT_PADDING = 0.08
const LIVE_DETECTION_SMOOTHING = 0.34
const LIVE_DETECTION_VELOCITY_BLEND = 0.58
const LIVE_GENERIC_STICKY_ALPHA = 0.22
const LIVE_GENERIC_MAJOR_SHIFT_ALPHA = 0.72
const LIVE_GENERIC_MINOR_SHIFT_DISTANCE = 0.028
const LIVE_GENERIC_MAJOR_SHIFT_DISTANCE = 0.16
const LIVE_GENERIC_MAJOR_SHIFT_SIZE_RATIO = 0.5
const LIVE_FACE_STICKY_ALPHA = 0.28
const LIVE_FACE_MAJOR_SHIFT_ALPHA = 0.82
const LIVE_FACE_MINOR_SHIFT_DISTANCE = 0.035
const LIVE_FACE_MAJOR_SHIFT_DISTANCE = 0.18
const LIVE_FACE_MAJOR_SHIFT_SIZE_RATIO = 0.45
const LIVE_FACE_MAX_VELOCITY = 1.45
const LIVE_OBJECT_MAX_VELOCITY = 1.8
const LIVE_SIZE_MAX_VELOCITY = 1.3
const LIVE_FACE_PREDICTION_MAX_LEAD_SEC = 0.18
const LIVE_OBJECT_PREDICTION_MAX_LEAD_SEC = 0.28
const LIVE_FACE_PREDICTION_MAX_LAG_SEC = 0.05
const LIVE_OBJECT_PREDICTION_MAX_LAG_SEC = 0.08
// Hide the audio waveform so the lower lane can focus on entity-search matches.
const SHOW_AUDIO_WAVEFORM = false
// Hide the per-face timeline lane built from local detection metadata
// (appearances / time ranges in detection_metadata.json). Only the
// TwelveLabs entity-search lanes are shown for blurred faces.
const SHOW_DETECTION_METADATA_FACE_LANE = false
const ENTITY_SEARCH_LANE_COLOR = '#00dc82'
const ENTITY_SEARCH_LANE_BORDER = 'rgba(0, 220, 130, 0.22)'
const ENTITY_SEARCH_LANE_BACKGROUND = 'linear-gradient(90deg, rgba(0, 220, 130, 0.07) 0%, rgba(0, 220, 130, 0.025) 50%, rgba(0, 220, 130, 0.07) 100%)'
const ENTITY_SEARCH_LANE_OVERLAY = 'linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0) 58%, rgba(0,220,130,0.02) 100%)'
const ENTITY_SEARCH_LANE_BASELINE = 'rgba(0, 220, 130, 0.18)'
const ENTITY_SEARCH_LANE_BAR_BASE = 'rgba(0, 220, 130, 0.28)'
// Highlight peak bars (where the entity/query match is strongest) in red so
// the most crucial moments stand out from the rest of the green lane.
const ENTITY_SEARCH_LANE_PEAK_COLOR = '#ef4444'
const ENTITY_SEARCH_LANE_PEAK_BAR_BASE = 'rgba(239, 68, 68, 0.32)'
// Bars within this fraction of the lane's max value are treated as peaks and
// turn red. Keep the cutoff forgiving enough that clear match clusters are easy
// to spot even when the lane has one dominant spike.
const ENTITY_SEARCH_LANE_PEAK_THRESHOLD = 0.52
const ENTITY_SEARCH_LANE_SEGMENT_ACTIVE = 'rgba(0, 220, 130, 0.16)'
const ENTITY_SEARCH_LANE_SEGMENT_RING = 'rgba(0, 220, 130, 0.28)'
const ENTITY_SEARCH_LANE_EDGE_LEFT = 'rgba(0, 220, 130, 0.55)'
const ENTITY_SEARCH_LANE_EDGE_RIGHT = 'rgba(0, 220, 130, 0.35)'
const ENTITY_SEARCH_LANE_HEIGHT_PX = 60
const PEGASUS_LANE_HEIGHT_PX = 68
const VIDEO_TIMELINE_LANE_HEIGHT_PX = 76
const FACE_LANE_DEBUG_CACHE_KEY = 'video_redaction_face_lane_debug_cache_v1'
const FACE_LOCK_INTRO_SEEN_KEY = 'video_redaction_face_lock_intro_seen_v1'
const LIVE_REDACTION_OBJECT_CLASSES = [
  'backpack',
  'bicycle',
  'bus',
  'car',
  'cell phone',
  'gun',
  'handbag',
  'knife',
  'laptop',
  'motorcycle',
  'scissors',
  'suitcase',
  'truck',
]

type DetectionItem = {
  id: string
  kind: 'face' | 'object'
  label: string
  tags: string[]
  color: string
  snapBase64?: string
  personId?: string
  objectClass?: string
  timeRanges?: DetectionTimeRange[]
  appearances?: DetectionAppearance[]
  entityId?: string | null
  appearanceCount?: number
  shouldAnonymize?: boolean
  isOfficial?: boolean
  priorityRank?: number
}

type DetectionTimeRange = {
  start_sec?: number
  end_sec?: number
  start?: number
  end?: number
}

type DetectionAppearance = {
  frame_idx?: number
  timestamp?: number
  bbox?: number[]
}

type FaceDetectionApiRecord = {
  person_id?: string
  name?: string
  description?: string
  snap_base64?: string
  time_ranges?: DetectionTimeRange[]
  appearances?: DetectionAppearance[]
  entity_id?: string | null
  tags?: string[]
  should_anonymize?: boolean
  is_official?: boolean
  priority_rank?: number
  appearance_count?: number
}

type ObjectDetectionApiRecord = {
  object_id?: string
  identification?: string
  snap_base64?: string
  appearance_count?: number
  appearances?: DetectionAppearance[]
}

type FaceTimelineSegment = {
  start: number
  end: number
}

type FaceTimelineLaneSource = 'appearances' | 'marengo'

type FaceTimelineLane = {
  personId: string
  item: DetectionItem
  active: boolean
  segments: FaceTimelineSegment[]
  source: FaceTimelineLaneSource
}

type FaceLaneDebugCacheEntry = {
  personId: string
  label: string
  entityId?: string | null
  storedAtMs: number
  localTimeRanges?: DetectionTimeRange[]
  localAppearances?: DetectionAppearance[]
  localSegments?: FaceTimelineSegment[]
  marengoTimeRanges?: DetectionTimeRange[]
  marengoSegments?: FaceTimelineSegment[]
}

type LiveRedactionDetection = {
  id: string
  trackId?: string | null
  kind: 'face' | 'object'
  label: string
  confidence: number
  personId?: string | null
  objectClass?: string | null
  x: number
  y: number
  width: number
  height: number
  sourceTime?: number
  lastSeenAtMs?: number
  velocityX?: number
  velocityY?: number
  velocityWidth?: number
  velocityHeight?: number
}

type FaceLockLaneEntry = {
  f: number
  t: number
  x1: number
  y1: number
  x2: number
  y2: number
  src?: string
  conf?: number
}

type FaceLockLane = {
  build_version: number
  job_id: string
  person_id: string
  video: { width: number; height: number; fps: number; total_frames: number; duration_sec: number }
  segments?: Array<{ start_frame: number; end_frame: number }>
  lane: FaceLockLaneEntry[]
  safety_pad_ratio?: number
  built_at?: string
}

type FaceLockBuildState = {
  status: 'queued' | 'running' | 'ready' | 'failed' | 'missing'
  percent: number
  progress: number
  message?: string | null
}

const FACE_LOCK_LANE_INTERPOLATION_MAX_GAP_SEC = 0.06

function findFaceLockBracketIndices(lane: FaceLockLaneEntry[], targetFrame: number): { lower: number; upper: number } {
  if (!lane.length) return { lower: -1, upper: -1 }
  let lo = 0
  let hi = lane.length - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    const f = lane[mid].f
    if (f === targetFrame) return { lower: mid, upper: mid }
    if (f < targetFrame) lo = mid + 1
    else hi = mid - 1
  }
  return { lower: hi, upper: lo }
}

function interpolateFaceLockLane(
  lane: FaceLockLane,
  timeSec: number,
  maxGapSec: number = FACE_LOCK_LANE_INTERPOLATION_MAX_GAP_SEC,
): { x1: number; y1: number; x2: number; y2: number } | null {
  const entries = lane.lane
  if (!entries || entries.length === 0) return null
  const fps = lane.video?.fps && lane.video.fps > 0 ? lane.video.fps : 25
  const targetFrame = timeSec * fps
  const targetFrameInt = Math.round(targetFrame)
  const { lower, upper } = findFaceLockBracketIndices(entries, targetFrameInt)
  const lowerEntry = lower >= 0 && lower < entries.length ? entries[lower] : null
  const upperEntry = upper >= 0 && upper < entries.length ? entries[upper] : null
  if (lowerEntry && upperEntry && lower === upper) {
    return { x1: lowerEntry.x1, y1: lowerEntry.y1, x2: lowerEntry.x2, y2: lowerEntry.y2 }
  }
  const gapLower = lowerEntry ? Math.abs(targetFrame - lowerEntry.f) / fps : Infinity
  const gapUpper = upperEntry ? Math.abs(upperEntry.f - targetFrame) / fps : Infinity
  if (Math.min(gapLower, gapUpper) > maxGapSec) return null
  if (!lowerEntry && upperEntry) return { x1: upperEntry.x1, y1: upperEntry.y1, x2: upperEntry.x2, y2: upperEntry.y2 }
  if (!upperEntry && lowerEntry) return { x1: lowerEntry.x1, y1: lowerEntry.y1, x2: lowerEntry.x2, y2: lowerEntry.y2 }
  if (!lowerEntry || !upperEntry) return null
  const span = upperEntry.f - lowerEntry.f
  if (span <= 0) return { x1: lowerEntry.x1, y1: lowerEntry.y1, x2: lowerEntry.x2, y2: lowerEntry.y2 }
  if (span > Math.round(maxGapSec * fps) + 1) {
    const nearest = gapLower <= gapUpper ? lowerEntry : upperEntry
    return { x1: nearest.x1, y1: nearest.y1, x2: nearest.x2, y2: nearest.y2 }
  }
  const t = Math.max(0, Math.min(1, (targetFrame - lowerEntry.f) / span))
  return {
    x1: lowerEntry.x1 * (1 - t) + upperEntry.x1 * t,
    y1: lowerEntry.y1 * (1 - t) + upperEntry.y1 * t,
    x2: lowerEntry.x2 * (1 - t) + upperEntry.x2 * t,
    y2: lowerEntry.y2 * (1 - t) + upperEntry.y2 * t,
  }
}

function laneBboxToLiveDetection(
  lane: FaceLockLane,
  bbox: { x1: number; y1: number; x2: number; y2: number },
  personId: string,
  label: string,
): LiveRedactionDetection {
  const videoW = lane.video?.width || 1
  const videoH = lane.video?.height || 1
  const x = Math.max(0, Math.min(1, bbox.x1 / videoW))
  const y = Math.max(0, Math.min(1, bbox.y1 / videoH))
  const width = Math.max(0, Math.min(1 - x, (bbox.x2 - bbox.x1) / videoW))
  const height = Math.max(0, Math.min(1 - y, (bbox.y2 - bbox.y1) / videoH))
  return {
    id: `face-lock-${personId}`,
    trackId: `face-lock-${personId}`,
    kind: 'face',
    label,
    confidence: 1,
    personId,
    x,
    y,
    width,
    height,
    sourceTime: 0,
    lastSeenAtMs: Date.now(),
  }
}

type VideoFrameCallbackCapableElement = HTMLVideoElement & {
  requestVideoFrameCallback?: (callback: VideoFrameRequestCallback) => number
  cancelVideoFrameCallback?: (handle: number) => void
}

let liveBlurScratchCanvas: HTMLCanvasElement | null = null

function getLiveBlurScratchCanvas(): HTMLCanvasElement | null {
  if (typeof document === 'undefined') return null
  if (!liveBlurScratchCanvas) {
    liveBlurScratchCanvas = document.createElement('canvas')
  }
  return liveBlurScratchCanvas
}

function supportsLiveBackdropBlur(): boolean {
  if (typeof window === 'undefined' || typeof CSS === 'undefined' || typeof CSS.supports !== 'function') {
    return false
  }
  return CSS.supports('backdrop-filter: blur(2px)') || CSS.supports('-webkit-backdrop-filter: blur(2px)')
}

function getLiveRedactionRenderBox(detection: LiveRedactionDetection) {
  let renderX = detection.x
  let renderY = detection.y
  let renderWidth = detection.width
  let renderHeight = detection.height

  if (detection.kind === 'face') {
    // The backend already pads face boxes for redaction coverage, so keep the
    // preview expansion subtle here to avoid making motion feel offset.
    const expandX = detection.width * 0.012
    const expandTop = detection.height * 0.03
    const expandBottom = detection.height * 0.02
    renderX = Math.max(0, detection.x - expandX)
    renderY = Math.max(0, detection.y - expandTop)
    renderWidth = Math.min(1 - renderX, detection.width + expandX * 2)
    renderHeight = Math.min(1 - renderY, detection.height + expandTop + expandBottom)
  }

  return {
    x: renderX,
    y: renderY,
    width: renderWidth,
    height: renderHeight,
  }
}

function getLiveRedactionBlurPx(blurStrength: number): number {
  return Math.max(16, Math.round(blurStrength * 0.78))
}

function getLiveRedactionOverlayStyle(
  detection: LiveRedactionDetection,
  style: RedactionStyle,
  blurStrength: number,
): React.CSSProperties {
  const renderBox = getLiveRedactionRenderBox(detection)
  const blurCss = `blur(${getLiveRedactionBlurPx(blurStrength)}px) saturate(0.72)`
  const blurMask = detection.kind === 'face'
    ? 'radial-gradient(ellipse at center, rgba(0,0,0,1) 38%, rgba(0,0,0,0.86) 58%, rgba(0,0,0,0.42) 78%, rgba(0,0,0,0.12) 92%, rgba(0,0,0,0) 100%)'
    : 'radial-gradient(ellipse at center, rgba(0,0,0,1) 44%, rgba(0,0,0,0.82) 68%, rgba(0,0,0,0.34) 88%, rgba(0,0,0,0) 100%)'

  return {
    left: `${renderBox.x * 100}%`,
    top: `${renderBox.y * 100}%`,
    width: `${renderBox.width * 100}%`,
    height: `${renderBox.height * 100}%`,
    borderRadius: detection.kind === 'face' ? '18px' : '10px',
    overflow: 'hidden',
    pointerEvents: 'none',
    transform: 'translateZ(0)',
    willChange: 'left, top, width, height',
    transition: 'left 120ms linear, top 120ms linear, width 120ms linear, height 120ms linear',
    backgroundColor: style === 'black'
      ? 'rgba(5, 6, 8, 0.96)'
      : 'transparent',
    backdropFilter: style === 'blur' ? blurCss : undefined,
    WebkitBackdropFilter: style === 'blur' ? blurCss : undefined,
    maskImage: style === 'blur' ? blurMask : undefined,
    WebkitMaskImage: style === 'blur' ? blurMask : undefined,
    maskMode: style === 'blur' ? 'alpha' : undefined,
    WebkitMaskRepeat: style === 'blur' ? 'no-repeat' : undefined,
    maskRepeat: style === 'blur' ? 'no-repeat' : undefined,
    boxShadow: 'none',
  }
}

function clampUnit(value: number): number {
  return Math.max(0, Math.min(1, value))
}

function normalizeObjectClass(value?: string | null): string | null {
  const normalized = (value || '').trim().toLowerCase()
  return normalized || null
}

function formatPegasusLabel(value?: string | null): string {
  const normalized = (value || '').replace(/_/g, ' ').trim()
  return normalized ? normalized.replace(/\b\w/g, (char) => char.toUpperCase()) : 'Review'
}

function getPegasusSeverityStyle(severity?: string | null): {
  color: string
  soft: string
  border: string
  text: string
} {
  if (severity === 'high') {
    return {
      color: '#f87171',
      soft: 'rgba(248, 113, 113, 0.09)',
      border: 'rgba(248, 113, 113, 0.24)',
      text: 'text-text-secondary',
    }
  }
  if (severity === 'medium') {
    return {
      color: '#fbbf24',
      soft: 'rgba(251, 191, 36, 0.08)',
      border: 'rgba(251, 191, 36, 0.22)',
      text: 'text-text-secondary',
    }
  }
  return {
    color: '#7dd3fc',
    soft: 'rgba(125, 211, 252, 0.07)',
    border: 'rgba(125, 211, 252, 0.2)',
    text: 'text-text-secondary',
  }
}

function formatPegasusDate(value?: string | null): string {
  if (!value) return 'Not generated'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function formatPegasusConfidence(value?: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'N/A'
  return `${Math.round(clampUnit(value) * 100)}%`
}

function ensureMp4Filename(filename?: string | null): string {
  const basename = (filename || '').trim().split(/[\\/]/).pop() || 'redacted-video.mp4'
  const sanitized = basename.replace(/[^\w.-]+/g, '_') || 'redacted-video.mp4'
  return /\.mp4$/i.test(sanitized) ? sanitized : `${sanitized}.mp4`
}

function filenameFromContentDisposition(header: string | null): string | null {
  if (!header) return null
  const utfMatch = header.match(/filename\*=UTF-8''([^;]+)/i)
  if (utfMatch?.[1]) {
    try {
      return decodeURIComponent(utfMatch[1].replace(/"/g, ''))
    } catch {
      return utfMatch[1].replace(/"/g, '')
    }
  }
  const match = header.match(/filename="?([^";]+)"?/i)
  return match?.[1] || null
}

function clickDownloadLink(url: string, filename?: string) {
  if (typeof document === 'undefined' || !url) return
  const link = document.createElement('a')
  link.href = url
  link.download = ensureMp4Filename(filename)
  link.rel = 'noopener'
  link.style.display = 'none'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

async function triggerFileDownload(url: string, filename?: string) {
  if (typeof window === 'undefined' || typeof document === 'undefined' || !url) return

  let objectUrl: string | null = null
  try {
    const response = await fetch(url, {
      method: 'GET',
      cache: 'no-store',
    })
    if (!response.ok) {
      throw new Error(`Download failed (${response.status})`)
    }

    const responseFilename = filenameFromContentDisposition(response.headers.get('Content-Disposition'))
    const resolvedFilename = ensureMp4Filename(filename || responseFilename)
    const blob = await response.blob()
    const responseType = response.headers.get('Content-Type') || blob.type
    if (blob.size <= 0) {
      throw new Error('Download returned an empty MP4')
    }
    if (!/video\/mp4/i.test(responseType)) {
      throw new Error('Download did not return an MP4 file')
    }
    const mp4Blob = blob.type === 'video/mp4' ? blob : new Blob([blob], { type: 'video/mp4' })
    objectUrl = window.URL.createObjectURL(mp4Blob)
    clickDownloadLink(objectUrl, resolvedFilename)
  } catch (error) {
    if (error instanceof Error && (
      /Download failed/i.test(error.message) ||
      /empty MP4/i.test(error.message) ||
      /did not return an MP4/i.test(error.message)
    )) {
      throw error
    }
    // Fall back to direct navigation if the browser blocks the blob fetch.
    clickDownloadLink(url, ensureMp4Filename(filename))
  } finally {
    if (objectUrl) {
      window.setTimeout(() => window.URL.revokeObjectURL(objectUrl as string), 1000)
    }
  }
}

function normalizeDetectionTags(
  rawTags: unknown,
  options: { shouldAnonymize?: boolean; isOfficial?: boolean } = {},
): string[] {
  const tags: string[] = []
  const seen = new Set<string>()
  const source = Array.isArray(rawTags) ? rawTags : []
  source.forEach((value) => {
    const rawTag = String(value || '').trim()
    if (!rawTag) return
    const key = rawTag.toLowerCase()
    let tag = rawTag
    if (key === 'anonymized') {
      if (!options.shouldAnonymize || options.isOfficial) return
      tag = 'Anonymized'
    } else if (key === 'official') {
      if (!options.isOfficial) return
      tag = 'Official'
    }
    if (seen.has(key)) return
    seen.add(key)
    tags.push(tag)
  })
  if (options.shouldAnonymize && !options.isOfficial && !seen.has('anonymized')) {
    seen.add('anonymized')
    tags.unshift('Anonymized')
  }
  if (options.isOfficial && !seen.has('official')) {
    tags.push('Official')
  }
  return tags
}

function sortDetectionItems(items: DetectionItem[]): DetectionItem[] {
  return [...items].sort((a, b) => {
    const groupRank = (item: DetectionItem) => {
      if (item.kind === 'face' && item.shouldAnonymize) return 0
      if (item.kind === 'face') return 1
      return 2
    }
    const groupDifference = groupRank(a) - groupRank(b)
    if (groupDifference !== 0) return groupDifference

    if (a.kind === 'face' && b.kind === 'face') {
      const priorityA = Number.isFinite(a.priorityRank) ? Number(a.priorityRank) : Number.MAX_SAFE_INTEGER
      const priorityB = Number.isFinite(b.priorityRank) ? Number(b.priorityRank) : Number.MAX_SAFE_INTEGER
      if (priorityA !== priorityB) return priorityA - priorityB

      const appearanceDifference = (b.appearanceCount || 0) - (a.appearanceCount || 0)
      if (appearanceDifference !== 0) return appearanceDifference
    }

    return a.label.localeCompare(b.label)
  })
}

function isAnonymizeTarget(item: DetectionItem): boolean {
  return item.kind === 'face' && Boolean(item.shouldAnonymize)
}

function buildDetectionItemsFromApi(
  uniqueFaces: FaceDetectionApiRecord[],
  uniqueObjects: ObjectDetectionApiRecord[],
): DetectionItem[] {
  const items: DetectionItem[] = []
  const faceColor = '#F59E0B'
  const objectColors = ['#3B82F6', '#EF4444', '#10B981', '#8B5CF6']

  uniqueFaces.forEach((face, index) => {
    const personId = (face.person_id || `person_${index}`).toString().trim()
    const shouldAnonymize = Boolean(face.should_anonymize) && !Boolean(face.is_official)
    const isOfficial = Boolean(face.is_official)
    const label = (face.name || face.description || personId || `Person ${index + 1}`).toString().trim()
    items.push({
      id: `face-${personId}`,
      kind: 'face',
      label: label.slice(0, 60),
      tags: normalizeDetectionTags(face.tags, { shouldAnonymize, isOfficial }),
      color: faceColor,
      snapBase64: face.snap_base64,
      personId,
      timeRanges: Array.isArray(face.time_ranges) ? face.time_ranges : [],
      appearances: Array.isArray(face.appearances) ? face.appearances : [],
      entityId: typeof face.entity_id === 'string' && face.entity_id.trim() ? face.entity_id : null,
      appearanceCount: typeof face.appearance_count === 'number' ? face.appearance_count : 0,
      shouldAnonymize,
      isOfficial,
      priorityRank: typeof face.priority_rank === 'number' ? face.priority_rank : undefined,
    })
  })

  const seenClasses = new Set<string>()
  uniqueObjects.forEach((objectItem, index) => {
    const rawObjectClass = (objectItem.identification || objectItem.object_id || `Object ${index + 1}`).toString().trim()
    const objectClass = normalizeObjectClass(rawObjectClass)
    if (!objectClass || seenClasses.has(objectClass)) return
    seenClasses.add(objectClass)
    items.push({
      id: `object-${objectClass}`,
      kind: 'object',
      label: rawObjectClass.slice(0, 60),
      tags: [],
      color: objectColors[index % objectColors.length],
      snapBase64: objectItem.snap_base64,
      objectClass,
      appearances: Array.isArray(objectItem.appearances) ? objectItem.appearances : [],
      appearanceCount: typeof objectItem.appearance_count === 'number' ? objectItem.appearance_count : 0,
    })
  })

  return sortDetectionItems(items)
}

function expandLiveDetection(detection: LiveRedactionDetection): LiveRedactionDetection {
  const padding = detection.kind === 'face' ? LIVE_FACE_PADDING : LIVE_OBJECT_PADDING
  const cx = detection.x + detection.width / 2
  const cy = detection.y + detection.height / 2
  const width = clampUnit(detection.width * (1 + padding * 2))
  const height = clampUnit(detection.height * (1 + padding * 2))
  return {
    ...detection,
    x: clampUnit(cx - width / 2),
    y: clampUnit(cy - height / 2),
    width: Math.min(width, 1),
    height: Math.min(height, 1),
  }
}

function liveDetectionIou(a: LiveRedactionDetection, b: LiveRedactionDetection): number {
  const ax2 = a.x + a.width
  const ay2 = a.y + a.height
  const bx2 = b.x + b.width
  const by2 = b.y + b.height
  const ix1 = Math.max(a.x, b.x)
  const iy1 = Math.max(a.y, b.y)
  const ix2 = Math.min(ax2, bx2)
  const iy2 = Math.min(ay2, by2)
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1)
  const union = a.width * a.height + b.width * b.height - inter
  return union > 0 ? inter / union : 0
}

function liveDetectionCenterDistance(a: LiveRedactionDetection, b: LiveRedactionDetection): number {
  const ax = a.x + a.width / 2
  const ay = a.y + a.height / 2
  const bx = b.x + b.width / 2
  const by = b.y + b.height / 2
  return Math.hypot(ax - bx, ay - by)
}

function liveDetectionTrackKey(detection: LiveRedactionDetection): string | null {
  if (detection.trackId) return `track:${detection.trackId}`
  if (detection.kind === 'face' && detection.personId) return `face:${detection.personId}`
  // Object class is a selection group, not a stable track identity. Multiple
  // objects can share the same label, so matching them by class causes jumps.
  return null
}

function getStableLiveDetectionId(detection: LiveRedactionDetection): string {
  return liveDetectionTrackKey(detection)
    ?? `${detection.kind}:${detection.label}:${detection.id}`
}

function liveDetectionSizeChangeRatio(a: LiveRedactionDetection, b: LiveRedactionDetection): number {
  const widthBase = Math.max(a.width, 0.001)
  const heightBase = Math.max(a.height, 0.001)
  return Math.max(
    Math.abs(b.width - a.width) / widthBase,
    Math.abs(b.height - a.height) / heightBase,
  )
}

function liveDetectionMotionMagnitude(detection: LiveRedactionDetection): number {
  return Math.hypot(detection.velocityX ?? 0, detection.velocityY ?? 0)
}

function getLiveDetectionHoldMs(detection: LiveRedactionDetection): number {
  return detection.kind === 'face' && detection.personId ? LIVE_IDENTIFIED_FACE_HOLD_MS : LIVE_DETECTION_HOLD_MS
}

function clampLiveVelocity(value: number, maxVelocity: number): number {
  return Math.max(-maxVelocity, Math.min(maxVelocity, value))
}

function blendLiveVelocity(previousValue: number | undefined, nextValue: number): number {
  if (!Number.isFinite(previousValue)) return nextValue
  return (previousValue as number) * (1 - LIVE_DETECTION_VELOCITY_BLEND) + nextValue * LIVE_DETECTION_VELOCITY_BLEND
}

function smoothLiveDetection(
  previous: LiveRedactionDetection,
  next: LiveRedactionDetection,
  alpha = LIVE_DETECTION_SMOOTHING,
): LiveRedactionDetection {
  return {
    ...next,
    x: previous.x * (1 - alpha) + next.x * alpha,
    y: previous.y * (1 - alpha) + next.y * alpha,
    width: previous.width * (1 - alpha) + next.width * alpha,
    height: previous.height * (1 - alpha) + next.height * alpha,
  }
}

function withLiveDetectionVelocity(
  previous: LiveRedactionDetection,
  next: LiveRedactionDetection,
): LiveRedactionDetection {
  const sourceDelta = Math.max(
    1 / 120,
    Math.abs((next.sourceTime ?? 0) - (previous.sourceTime ?? next.sourceTime ?? 0)),
  )
  const maxVelocity = next.kind === 'face' ? LIVE_FACE_MAX_VELOCITY : LIVE_OBJECT_MAX_VELOCITY

  return {
    ...next,
    velocityX: blendLiveVelocity(
      previous.velocityX,
      clampLiveVelocity((next.x - previous.x) / sourceDelta, maxVelocity),
    ),
    velocityY: blendLiveVelocity(
      previous.velocityY,
      clampLiveVelocity((next.y - previous.y) / sourceDelta, maxVelocity),
    ),
    velocityWidth: blendLiveVelocity(
      previous.velocityWidth,
      clampLiveVelocity((next.width - previous.width) / sourceDelta, LIVE_SIZE_MAX_VELOCITY),
    ),
    velocityHeight: blendLiveVelocity(
      previous.velocityHeight,
      clampLiveVelocity((next.height - previous.height) / sourceDelta, LIVE_SIZE_MAX_VELOCITY),
    ),
  }
}

function predictLiveDetection(
  detection: LiveRedactionDetection,
  playbackTime: number,
): LiveRedactionDetection {
  if (!Number.isFinite(playbackTime) || !Number.isFinite(detection.sourceTime)) return detection

  const predictionDelta = Math.max(
    -(detection.kind === 'face' ? LIVE_FACE_PREDICTION_MAX_LAG_SEC : LIVE_OBJECT_PREDICTION_MAX_LAG_SEC),
    Math.min(
      detection.kind === 'face' ? LIVE_FACE_PREDICTION_MAX_LEAD_SEC : LIVE_OBJECT_PREDICTION_MAX_LEAD_SEC,
      playbackTime - (detection.sourceTime as number),
    ),
  )

  if (Math.abs(predictionDelta) < 1e-4) return detection

  const width = clampUnit(
    Math.min(1, detection.width + (detection.velocityWidth ?? 0) * predictionDelta),
  )
  const height = clampUnit(
    Math.min(1, detection.height + (detection.velocityHeight ?? 0) * predictionDelta),
  )
  const x = clampUnit(
    Math.min(1 - width, detection.x + (detection.velocityX ?? 0) * predictionDelta),
  )
  const y = clampUnit(
    Math.min(1 - height, detection.y + (detection.velocityY ?? 0) * predictionDelta),
  )

  return {
    ...detection,
    x,
    y,
    width,
    height,
  }
}

function getLiveDetectionMatchScore(
  previousDetection: LiveRedactionDetection,
  anchorDetection: LiveRedactionDetection,
  candidateDetection: LiveRedactionDetection,
): number | null {
  if (candidateDetection.kind !== previousDetection.kind) return null

  const previousTrackKey = liveDetectionTrackKey(previousDetection)
  const candidateTrackKey = liveDetectionTrackKey(candidateDetection)
  if (previousTrackKey && candidateTrackKey && previousTrackKey !== candidateTrackKey) return null
  if (candidateDetection.kind === 'object' && candidateDetection.label !== previousDetection.label) return null
  if (
    candidateDetection.kind === 'face' &&
    previousTrackKey === null &&
    previousDetection.label !== 'Face' &&
    candidateDetection.label !== 'Face' &&
    candidateDetection.label !== previousDetection.label
  ) {
    return null
  }

  const iou = liveDetectionIou(anchorDetection, candidateDetection)
  const distance = liveDetectionCenterDistance(anchorDetection, candidateDetection)
  const sizeChangeRatio = liveDetectionSizeChangeRatio(anchorDetection, candidateDetection)
  const motionAllowance = Math.min(
    previousTrackKey ? 0.14 : 0.08,
    liveDetectionMotionMagnitude(previousDetection) * 0.09,
  )
  const maxDistance = (previousTrackKey ? 0.34 : (candidateDetection.kind === 'face' ? 0.15 : 0.18)) + motionAllowance

  if (iou < 0.04 && distance > maxDistance) return null
  if (sizeChangeRatio > 1.15 && distance > 0.09 && iou < 0.08) return null

  return (
    (previousTrackKey && previousTrackKey === candidateTrackKey ? 10 : 0)
    + iou * 5.2
    - distance * 1.35
    - sizeChangeRatio * 0.35
    + Math.min(1, Math.max(0, candidateDetection.confidence || 0)) * 0.15
  )
}

function stabilizeLiveDetections(
  previous: LiveRedactionDetection[],
  incoming: LiveRedactionDetection[],
  sourceTime: number,
): LiveRedactionDetection[] {
  const now = Date.now()
  const prepared = incoming.map((detection) => expandLiveDetection({
    ...detection,
    sourceTime,
    lastSeenAtMs: now,
    velocityX: 0,
    velocityY: 0,
    velocityWidth: 0,
    velocityHeight: 0,
  }))
  const next: LiveRedactionDetection[] = []
  const predictedPrevious = previous.map((detection) => predictLiveDetection(detection, sourceTime))
  const candidatePairs: Array<{ previousIdx: number; candidateIdx: number; score: number }> = []
  const assignedPrevious = new Set<number>()
  const usedCandidates = new Set<number>()
  const assignedCandidateByPrevious = new Map<number, number>()

  for (let previousIdx = 0; previousIdx < previous.length; previousIdx += 1) {
    for (let candidateIdx = 0; candidateIdx < prepared.length; candidateIdx += 1) {
      const score = getLiveDetectionMatchScore(previous[previousIdx], predictedPrevious[previousIdx], prepared[candidateIdx])
      if (score === null) continue
      candidatePairs.push({ previousIdx, candidateIdx, score })
    }
  }

  candidatePairs.sort((a, b) => b.score - a.score)

  for (const pair of candidatePairs) {
    if (assignedPrevious.has(pair.previousIdx) || usedCandidates.has(pair.candidateIdx)) continue
    assignedPrevious.add(pair.previousIdx)
    usedCandidates.add(pair.candidateIdx)
    assignedCandidateByPrevious.set(pair.previousIdx, pair.candidateIdx)
  }

  for (let previousIdx = 0; previousIdx < previous.length; previousIdx += 1) {
    const previousDetection = previous[previousIdx]
    const anchorDetection = predictedPrevious[previousIdx]
    const matchedIdx = assignedCandidateByPrevious.get(previousIdx)

    if (matchedIdx !== undefined) {
      const matchedDetection = prepared[matchedIdx]
      const distance = liveDetectionCenterDistance(anchorDetection, matchedDetection)
      const sizeChangeRatio = liveDetectionSizeChangeRatio(anchorDetection, matchedDetection)
      const trackKey = liveDetectionTrackKey(previousDetection)

      let alpha = LIVE_DETECTION_SMOOTHING
      if (previousDetection.kind === 'face' && trackKey) {
        alpha = distance <= LIVE_FACE_MINOR_SHIFT_DISTANCE && sizeChangeRatio <= 0.18
          ? LIVE_FACE_STICKY_ALPHA
          : distance >= LIVE_FACE_MAJOR_SHIFT_DISTANCE || sizeChangeRatio >= LIVE_FACE_MAJOR_SHIFT_SIZE_RATIO
            ? LIVE_FACE_MAJOR_SHIFT_ALPHA
            : LIVE_DETECTION_SMOOTHING
      } else {
        alpha = distance <= LIVE_GENERIC_MINOR_SHIFT_DISTANCE && sizeChangeRatio <= 0.16
          ? LIVE_GENERIC_STICKY_ALPHA
          : distance >= LIVE_GENERIC_MAJOR_SHIFT_DISTANCE || sizeChangeRatio >= LIVE_GENERIC_MAJOR_SHIFT_SIZE_RATIO
            ? LIVE_GENERIC_MAJOR_SHIFT_ALPHA
            : LIVE_DETECTION_SMOOTHING
      }

      next.push({
        ...withLiveDetectionVelocity(
          previousDetection,
          smoothLiveDetection(anchorDetection, matchedDetection, alpha),
        ),
        id: previousDetection.id,
      })
      continue
    }

    if (now - (previousDetection.lastSeenAtMs ?? now) <= getLiveDetectionHoldMs(previousDetection)) {
      next.push({
        ...previousDetection,
        velocityX: (previousDetection.velocityX ?? 0) * 0.86,
        velocityY: (previousDetection.velocityY ?? 0) * 0.86,
        velocityWidth: (previousDetection.velocityWidth ?? 0) * 0.82,
        velocityHeight: (previousDetection.velocityHeight ?? 0) * 0.82,
      })
    }
  }

  for (let idx = 0; idx < prepared.length; idx += 1) {
    if (!usedCandidates.has(idx)) {
      next.push({
        ...prepared[idx],
        id: getStableLiveDetectionId(prepared[idx]),
      })
    }
  }

  return next
}

function getSelectionIdForLiveDetection(detection: LiveRedactionDetection): string | null {
  if (detection.kind === 'face' && detection.personId) {
    return `face-${detection.personId}`
  }
  const normalizedObjectClass = normalizeObjectClass(detection.objectClass)
  if (detection.kind === 'object' && normalizedObjectClass) {
    return `object-${normalizedObjectClass}`
  }
  return null
}

function filterLiveDetectionsToSelections(
  detections: LiveRedactionDetection[],
  options: {
    hasRunDetection: boolean
    selectedFacePersonIds: string[]
    selectedObjectClasses: string[]
  },
): LiveRedactionDetection[] {
  if (!options.hasRunDetection) return detections

  const selectedFaces = new Set(options.selectedFacePersonIds)
  const selectedObjects = new Set(options.selectedObjectClasses.map((item) => normalizeObjectClass(item)).filter(Boolean) as string[])

  const hasFaceSelection = selectedFaces.size > 0
  const hasObjectSelection = selectedObjects.size > 0

  return detections.filter((detection) => {
    if (detection.kind === 'face') {
      if (!hasFaceSelection) return false
      if (detection.personId && selectedFaces.has(detection.personId)) return true
      // The backend already filters by person_ids server-side and tracking
      // may carry faces without a personId; keep them so blur stays visible
      // between identity re-lock cycles.
      if (!detection.personId && hasFaceSelection) return true
      return false
    }
    if (detection.kind === 'object') {
      const objectClass = normalizeObjectClass(detection.objectClass)
      if (!objectClass) return false
      if (hasObjectSelection) return selectedObjects.has(objectClass)
      return true
    }
    return false
  })
}

const DUMMY_DETECTIONS: DetectionItem[] = [
  { id: 'object-screen', kind: 'object', label: 'Screen', tags: ['screen', 'display'], color: '#3B82F6', objectClass: 'screen' },
  { id: 'object-plate', kind: 'object', label: 'License Plate', tags: ['license plate', 'object'], color: '#EF4444', objectClass: 'license plate' },
  { id: 'face-person-1', kind: 'face', label: 'Person 1', tags: ['person', 'face'], color: '#F59E0B', personId: 'person_1' },
  { id: 'face-person-2', kind: 'face', label: 'Person 2', tags: ['person', 'face'], color: '#F59E0B', personId: 'person_2' },
]

/* ------------------------------------------------------------------ */
/*  Standard media control icons (inline SVG, no external assets)      */
/* ------------------------------------------------------------------ */

function IconPlay({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>)
}
function IconPause({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /></svg>)
}
const skipForwardPath = 'M6 18V6l6.5 6L6 18zm7.5-12v12l6.5-6-6.5-6z'
function IconReplay5({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <path d={skipForwardPath} transform="translate(24,0) scale(-1,1)" />
    </svg>
  )
}
function IconForward5({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <path d={skipForwardPath} />
    </svg>
  )
}
function IconCameraSnap({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
      <circle cx="12" cy="13" r="4" />
    </svg>
  )
}
function IconVolumeUp({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" /></svg>)
}
function IconVolumeOff({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="currentColor"><path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z" /></svg>)
}
function IconFullscreen({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="currentColor"><path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z" /></svg>)
}
function IconFullscreenExit({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="currentColor"><path d="M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-11V5h-2v5h5V8h-3z" /></svg>)
}
function IconPlus({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14" /></svg>)
}
function IconMinus({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14" /></svg>)
}
function IconChevronRight({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 18l6-6-6-6" /></svg>)
}
function IconAbout({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M12 16v-4M12 8h.01" /></svg>)
}
function IconTopics({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2H2v10l9.29 9.29c.94.94 2.48.94 3.42 0l6.58-6.58c.94-.94.94-2.48 0-3.42L12 2Z" /><path d="M7 7h.01" /></svg>)
}
function IconCategories({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z" /></svg>)
}
function IconChevronLeft({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M15 18l-6-6 6-6" /></svg>)
}
function IconEye({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" /></svg>)
}
function IconEyeOff({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24" /><line x1="1" y1="1" x2="23" y2="23" /></svg>)
}
function IconDownload({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" /></svg>)
}
function IconChevronDown({ className = 'w-4 h-4' }: { className?: string }) {
  return (<svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 9l6 6 6-6" /></svg>)
}
function IconHelp({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M9.1 9a3 3 0 1 1 5.4 1.8c-.8.7-1.5 1.2-1.5 2.7" />
      <path d="M12 17h.01" />
    </svg>
  )
}

function IconFilter({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 6h16M7 12h10M10 18h4" />
    </svg>
  )
}

function IconPegasusCategory({ category, className = 'w-3.5 h-3.5' }: { category: PegasusTimelineCategory; className?: string }) {
  if (category === 'person' || category === 'face') {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 21a8 8 0 0 0-16 0" />
        <circle cx="12" cy="8" r="4" />
      </svg>
    )
  }
  if (category === 'screen' || category === 'document' || category === 'text' || category === 'license_plate') {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="4" width="18" height="14" rx="2" />
        <path d="M8 20h8" />
      </svg>
    )
  }
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20.59 13.41 11 3.83A2 2 0 0 0 9.59 3H4a1 1 0 0 0-1 1v5.59a2 2 0 0 0 .59 1.41l9.58 9.59a2 2 0 0 0 2.83 0l4.59-4.59a2 2 0 0 0 0-2.83Z" />
      <circle cx="7.5" cy="7.5" r="1.25" />
    </svg>
  )
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function fmtTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '00:00.000'
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  const ms = Math.round((sec - Math.floor(sec)) * 1000)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`
}

function fmtShort(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '00:00'
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function asFiniteNumber(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function normalizeFaceTimelineSegment(range: DetectionTimeRange): FaceTimelineSegment | null {
  const start = asFiniteNumber(range.start_sec ?? range.start)
  const endValue = asFiniteNumber(range.end_sec ?? range.end)
  if (start === null) return null
  const end = endValue === null ? start : Math.max(start, endValue)
  return {
    start: Math.max(0, start),
    end: Math.max(start, end),
  }
}

function mergeFaceTimelineSegments(
  segments: FaceTimelineSegment[],
  gapSec = 0.45,
  minDurationSec = 0.55,
): FaceTimelineSegment[] {
  if (!segments.length) return []

  const sorted = [...segments]
    .filter((segment) => Number.isFinite(segment.start) && Number.isFinite(segment.end))
    .sort((a, b) => a.start - b.start)

  if (!sorted.length) return []

  const merged: FaceTimelineSegment[] = [sorted[0]]
  for (const segment of sorted.slice(1)) {
    const previous = merged[merged.length - 1]
    if (segment.start <= previous.end + gapSec) {
      previous.end = Math.max(previous.end, segment.end)
      continue
    }
    merged.push({ ...segment })
  }

  return merged.map((segment) => {
    if (segment.end - segment.start >= minDurationSec) return segment
    const center = (segment.start + segment.end) / 2
    const half = minDurationSec / 2
    return {
      start: Math.max(0, center - half),
      end: Math.max(center + half, segment.start + minDurationSec),
    }
  })
}

function buildFaceTimelineSegmentsFromAppearances(appearances?: DetectionAppearance[]): FaceTimelineSegment[] {
  const timestamps = (appearances || [])
    .map((appearance) => asFiniteNumber(appearance.timestamp))
    .filter((value): value is number => value !== null)
    .sort((a, b) => a - b)

  if (!timestamps.length) return []

  return mergeFaceTimelineSegments(
    timestamps.map((timestamp) => ({
      start: Math.max(0, timestamp - 0.18),
      end: timestamp + 0.28,
    })),
    0.7,
    0.7,
  )
}

function buildFaceTimelineSegmentsFromRanges(ranges?: DetectionTimeRange[]): FaceTimelineSegment[] {
  return mergeFaceTimelineSegments(
    (ranges || [])
      .map((range) => normalizeFaceTimelineSegment(range))
      .filter((segment): segment is FaceTimelineSegment => segment !== null),
  )
}

function buildFaceTimelineSegments(
  item: DetectionItem,
  marengoRanges?: DetectionTimeRange[],
): FaceTimelineSegment[] {
  // Local clustered appearances are the most reliable source for a saved
  // person lane. Broad descriptive ranges and remote entity-search ranges are
  // UI hints only and must not override person-specific appearance anchors.
  const appearanceSegments = buildFaceTimelineSegmentsFromAppearances(item.appearances)
  if (appearanceSegments.length > 0) return appearanceSegments

  const localRangeSegments = buildFaceTimelineSegmentsFromRanges(item.timeRanges)
  if (localRangeSegments.length > 0) return localRangeSegments

  return buildFaceTimelineSegmentsFromRanges(marengoRanges)
}

function getDetectionItemTimelineSegments(item: DetectionItem): FaceTimelineSegment[] {
  if (item.kind === 'face') {
    return buildFaceTimelineSegments(item)
  }

  const rangedSegments = buildFaceTimelineSegmentsFromRanges(item.timeRanges)
  if (rangedSegments.length > 0) return rangedSegments
  return buildFaceTimelineSegmentsFromAppearances(item.appearances)
}

function getDetectionItemNearestGapSec(item: DetectionItem, targetTime: number): number {
  if (!Number.isFinite(targetTime)) return Number.POSITIVE_INFINITY
  const segments = getDetectionItemTimelineSegments(item)
  if (!segments.length) return Number.POSITIVE_INFINITY

  let nearestGap = Number.POSITIVE_INFINITY
  for (const segment of segments) {
    if (targetTime >= segment.start && targetTime <= segment.end) {
      return 0
    }
    const gap = targetTime < segment.start
      ? segment.start - targetTime
      : targetTime - segment.end
    if (gap < nearestGap) {
      nearestGap = gap
    }
  }
  return nearestGap
}

function isDetectionItemLikelyVisibleAtTime(item: DetectionItem, targetTime: number): boolean {
  return getDetectionItemNearestGapSec(item, targetTime) <= 0.18
}

function getDetectionItemSeekTime(item: DetectionItem, referenceTime: number): number | null {
  const timestamps = (item.appearances || [])
    .map((appearance) => asFiniteNumber(appearance.timestamp))
    .filter((value): value is number => value !== null)
    .sort((a, b) => a - b)

  if (timestamps.length > 0) {
    let nearest = timestamps[0]
    let nearestGap = Math.abs(nearest - referenceTime)
    for (const timestamp of timestamps.slice(1)) {
      const gap = Math.abs(timestamp - referenceTime)
      if (gap < nearestGap) {
        nearest = timestamp
        nearestGap = gap
      }
    }
    return nearest
  }

  const segments = getDetectionItemTimelineSegments(item)
  if (segments.length > 0) {
    let bestSegment = segments[0]
    let bestGap = Number.POSITIVE_INFINITY
    for (const segment of segments) {
      const gap = referenceTime < segment.start
        ? segment.start - referenceTime
        : referenceTime > segment.end
          ? referenceTime - segment.end
          : 0
      if (gap < bestGap) {
        bestGap = gap
        bestSegment = segment
      }
    }
    return (bestSegment.start + bestSegment.end) / 2
  }

  return null
}

function loadFaceLaneDebugCache(): Record<string, Record<string, FaceLaneDebugCacheEntry>> {
  if (typeof window === 'undefined') return {}
  try {
    const raw = window.localStorage.getItem(FACE_LANE_DEBUG_CACHE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as Record<string, Record<string, FaceLaneDebugCacheEntry>>
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

function saveFaceLaneDebugCache(cache: Record<string, Record<string, FaceLaneDebugCacheEntry>>) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(FACE_LANE_DEBUG_CACHE_KEY, JSON.stringify(cache))
  } catch {
    /* ignore cache persistence failures */
  }
}

function updateFaceLaneDebugCacheEntry(
  videoId: string,
  personId: string,
  patch: Partial<FaceLaneDebugCacheEntry>,
) {
  if (!videoId || !personId) return
  const cache = loadFaceLaneDebugCache()
  const videoCache = cache[videoId] || {}
  const existing = videoCache[personId]
  cache[videoId] = {
    ...videoCache,
    [personId]: {
      ...existing,
      label: patch.label ?? existing?.label ?? personId,
      ...patch,
      personId,
      storedAtMs: Date.now(),
    },
  }
  saveFaceLaneDebugCache(cache)
}

function drawCanvasBlurRegion(
  ctx: CanvasRenderingContext2D,
  video: HTMLVideoElement,
  detection: LiveRedactionDetection,
  destWidth: number,
  destHeight: number,
  style: RedactionStyle,
  blurStrength: number,
) {
  const renderBox = getLiveRedactionRenderBox(detection)
  const renderX = renderBox.x
  const renderY = renderBox.y
  const renderWidth = renderBox.width
  const renderHeight = renderBox.height

  const sx = renderX * video.videoWidth
  const sy = renderY * video.videoHeight
  const sw = renderWidth * video.videoWidth
  const sh = renderHeight * video.videoHeight
  const dx = renderX * destWidth
  const dy = renderY * destHeight
  const dw = renderWidth * destWidth
  const dh = renderHeight * destHeight

  if (sw <= 1 || sh <= 1 || dw <= 1 || dh <= 1) return

  ctx.save()
  ctx.beginPath()
  ctx.rect(dx, dy, dw, dh)
  ctx.clip()
  if (style === 'black') {
    ctx.fillStyle = 'rgba(5, 6, 8, 0.96)'
    ctx.fillRect(dx, dy, dw, dh)
  } else {
    const scratch = getLiveBlurScratchCanvas()
    const downscaleRatio = Math.max(0.06, 0.32 - (blurStrength / 100) * 0.22)
    const reducedWidth = Math.max(2, Math.round(dw * downscaleRatio))
    const reducedHeight = Math.max(2, Math.round(dh * downscaleRatio))
    const blurPx = Math.max(8, Math.round((blurStrength / 100) * Math.min(dw, dh) * 0.34))

    if (scratch) {
      scratch.width = reducedWidth
      scratch.height = reducedHeight
      const scratchCtx = scratch.getContext('2d')
      if (scratchCtx) {
        scratchCtx.setTransform(1, 0, 0, 1, 0, 0)
        scratchCtx.clearRect(0, 0, reducedWidth, reducedHeight)
        scratchCtx.imageSmoothingEnabled = true
        scratchCtx.drawImage(video, sx, sy, sw, sh, 0, 0, reducedWidth, reducedHeight)

        ctx.imageSmoothingEnabled = true
        ctx.filter = `blur(${blurPx}px)`
        ctx.drawImage(scratch, 0, 0, reducedWidth, reducedHeight, dx, dy, dw, dh)
        ctx.filter = 'none'
      } else {
        ctx.filter = `blur(${blurPx}px)`
        ctx.drawImage(video, sx, sy, sw, sh, dx, dy, dw, dh)
        ctx.filter = 'none'
      }
    } else {
      ctx.filter = `blur(${blurPx}px)`
      ctx.drawImage(video, sx, sy, sw, sh, dx, dy, dw, dh)
      ctx.filter = 'none'
    }

    ctx.fillStyle = 'rgba(0, 0, 0, 0.16)'
    ctx.fillRect(dx, dy, dw, dh)
  }
  ctx.restore()

  const isFace = detection.kind === 'face'
  const accentColor = isFace ? '#ff5252' : '#00dc82'
  const outerLineWidth = Math.max(4, Math.round(Math.min(dw, dh) * 0.05))
  const innerLineWidth = Math.max(2, Math.round(outerLineWidth * 0.45))

  ctx.save()
  ctx.fillStyle = isFace ? 'rgba(255, 82, 82, 0.10)' : 'rgba(0, 220, 130, 0.10)'
  ctx.fillRect(dx, dy, dw, dh)
  ctx.lineWidth = outerLineWidth
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.9)'
  ctx.strokeRect(dx, dy, dw, dh)
  ctx.lineWidth = innerLineWidth
  ctx.strokeStyle = accentColor
  ctx.strokeRect(dx, dy, dw, dh)
  ctx.restore()
}

/* ------------------------------------------------------------------ */
/*  Face-lock progress ring                                             */
/* ------------------------------------------------------------------ */

function FaceLockProgressRing({ percent, active }: { percent: number; active: boolean }) {
  const size = 22
  const strokeWidth = 2.6
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const clamped = Math.max(0, Math.min(100, percent))
  const dashOffset = circumference * (1 - clamped / 100)

  return (
    <span className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className={active ? 'animate-pulse' : ''}
        aria-hidden="true"
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeOpacity={0.15}
          strokeWidth={strokeWidth}
          className="text-text-tertiary"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          className={active ? 'text-emerald-400 transition-[stroke-dashoffset] duration-300' : 'text-emerald-400'}
        />
      </svg>
      <span className="absolute text-[8px] font-semibold tabular-nums text-text-primary leading-none">
        {active ? `${Math.round(clamped)}` : '✓'}
      </span>
    </span>
  )
}

type FaceLockChipEntry = {
  personId: string
  label: string
  snapBase64?: string | null
  color?: string | null
  percent: number
  status: 'ready' | 'running' | 'queued' | 'failed' | 'pending'
}

function FaceLockChip({ entry, variant = 'inline' }: { entry: FaceLockChipEntry; variant?: 'inline' | 'row' }) {
  const isReady = entry.status === 'ready'
  const isFailed = entry.status === 'failed'
  const accent = entry.color || '#34D399'
  const avatarSize = variant === 'row' ? 28 : 22
  const fallbackInitial = entry.label?.trim()?.charAt(0)?.toUpperCase() || '?'
  const ringSize = variant === 'row' ? 18 : 16
  const strokeWidth = 2.2
  const ringRadius = (ringSize - strokeWidth) / 2
  const ringCircumference = 2 * Math.PI * ringRadius
  const ringDashOffset = ringCircumference * (1 - Math.max(0, Math.min(100, entry.percent)) / 100)
  const ringColor = isFailed ? '#f87171' : isReady ? '#34d399' : accent

  return (
    <div
      className={`flex items-center gap-2 ${variant === 'row' ? 'w-full' : 'rounded-full border border-border bg-card px-1.5 py-0.5'}`}
      title={`${entry.label} - ${isFailed ? 'face-lock failed' : isReady ? 'locked' : `${entry.percent}% locked`}`}
    >
      <div
        className="relative shrink-0 overflow-hidden rounded-full border bg-surface"
        style={{
          width: avatarSize,
          height: avatarSize,
          borderColor: isReady ? `${accent}66` : 'rgba(148,163,184,0.32)',
        }}
      >
        {entry.snapBase64 ? (
          <img
            src={`data:image/png;base64,${entry.snapBase64}`}
            alt=""
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-[10px] font-semibold text-text-secondary">
            {fallbackInitial}
          </div>
        )}
      </div>
      <div className="flex min-w-0 flex-1 flex-col leading-tight">
        <span className={`truncate text-[11px] font-medium ${isFailed ? 'text-error' : 'text-text-primary'}`}>
          {entry.label}
        </span>
        <span className="text-[10px] tabular-nums text-text-tertiary">
          {isFailed ? 'Failed' : isReady ? 'Locked' : `${entry.percent}%`}
        </span>
      </div>
      <span
        className="relative inline-flex shrink-0 items-center justify-center"
        style={{ width: ringSize, height: ringSize }}
        aria-hidden
      >
        <svg width={ringSize} height={ringSize} viewBox={`0 0 ${ringSize} ${ringSize}`}>
          <circle
            cx={ringSize / 2}
            cy={ringSize / 2}
            r={ringRadius}
            fill="none"
            stroke="currentColor"
            strokeOpacity={0.18}
            strokeWidth={strokeWidth}
            className="text-text-tertiary"
          />
          <circle
            cx={ringSize / 2}
            cy={ringSize / 2}
            r={ringRadius}
            fill="none"
            stroke={ringColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={ringCircumference}
            strokeDashoffset={isFailed ? 0 : ringDashOffset}
            transform={`rotate(-90 ${ringSize / 2} ${ringSize / 2})`}
            className={!isReady && !isFailed ? 'transition-[stroke-dashoffset] duration-300' : ''}
          />
        </svg>
      </span>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function VideoEditorPage() {
  const { videoId } = useParams<{ videoId: string }>()
  const { getVideo } = useVideoCache()
  const cached = videoId ? getVideo(videoId) : undefined
  const [videoInfo, setVideoInfo] = useState<{
    video_id?: string
    system_metadata?: { filename?: string }
    hls?: { video_url?: string | null; thumbnail_urls?: string[] | string | null; status?: string | null } | null
    overview?: { about?: string; topics?: string[]; categories?: string[] }
  } | null>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const liveBlurCanvasRef = useRef<HTMLCanvasElement>(null)
  const liveBlurOverlayVideoRef = useRef<HTMLVideoElement>(null)
  const timelineRef = useRef<HTMLDivElement>(null)
  const videoContainerRef = useRef<HTMLDivElement>(null)
  const videoStageRef = useRef<HTMLDivElement>(null)
  const editorCenterRef = useRef<HTMLDivElement>(null)
  const hlsRef = useRef<Hls | null>(null)
  const hlsLoadedUrlRef = useRef<string | null>(null)
  const liveBlurOverlayHlsRef = useRef<Hls | null>(null)
  const liveBlurOverlayLoadedUrlRef = useRef<string | null>(null)
  const detectionLoadRequestIdRef = useRef(0)
  const readyRedactionJobIdsRef = useRef<Record<string, true>>({})

  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(1)
  const [isMuted, setIsMuted] = useState(true)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [buffered, setBuffered] = useState(0)
  const [activeTool, setActiveTool] = useState<ToolId>('tracker')
  const [timelineZoom, setTimelineZoom] = useState(1)
  const [hasRunDetection, setHasRunDetection] = useState(false)
  const [detectionFilter, setDetectionFilter] = useState('')
  const [showAnonymizeOnly, setShowAnonymizeOnly] = useState(false)
  const [apiDetections, setApiDetections] = useState<DetectionItem[]>([])
  const [personLaneIds, setPersonLaneIds] = useState<string[]>([])
  const [faceLaneEntityRangesByPersonId, setFaceLaneEntityRangesByPersonId] = useState<Record<string, DetectionTimeRange[]>>({})
  const [detectionLoading, setDetectionLoading] = useState(false)
  const [detectionError, setDetectionError] = useState<string | null>(null)
  const [detectionJobId, setDetectionJobId] = useState<string | null>(null)
  const liveRedactionEnabled = false
  const [liveRedactionDetections, setLiveRedactionDetections] = useState<LiveRedactionDetection[]>([])
  const [liveRedactionLoading, setLiveRedactionLoading] = useState(false)
  const [liveRedactionSeekPending, setLiveRedactionSeekPending] = useState(false)
  const [liveRedactionError, setLiveRedactionError] = useState<string | null>(null)
  const [faceLockLanesByPersonId, setFaceLockLanesByPersonId] = useState<Record<string, FaceLockLane>>({})
  const [faceLockBuildByPersonId, setFaceLockBuildByPersonId] = useState<Record<string, FaceLockBuildState>>({})
  const faceLockLanesByPersonIdRef = useRef<Record<string, FaceLockLane>>({})
  faceLockLanesByPersonIdRef.current = faceLockLanesByPersonId
  const faceLockBuildAttemptsRef = useRef<Record<string, boolean>>({})
  const lockedFacePersonIdsRef = useRef<string[]>([])
  const faceLockLabelByPersonIdRef = useRef<Record<string, string>>({})
  const [showFaceLockIntroPopup, setShowFaceLockIntroPopup] = useState(false)
  const [excludedFromRedactionIds, setExcludedFromRedactionIds] = useState<string[]>([])
  const [redactionStyle, setRedactionStyle] = useState<RedactionStyle>('blur')
  const [blurIntensity, setBlurIntensity] = useState(60)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(true)
  const [rightSidebarOpen, setRightSidebarOpen] = useState(true)
  const [exportMenuOpen, setExportMenuOpen] = useState(false)
  const exportMenuRef = useRef<HTMLDivElement>(null)
  const [tutorialOpen, setTutorialOpen] = useState(false)
  const [tutorialStepIndex, setTutorialStepIndex] = useState(0)
  const [tutorialPosition, setTutorialPosition] = useState<{ left: number; top: number } | null>(null)
  const [tutorialTargetRect, setTutorialTargetRect] = useState<{ left: number; top: number; width: number; height: number } | null>(null)
  const tutorialRef = useRef<HTMLDivElement>(null)
  const tutorialButtonRef = useRef<HTMLButtonElement>(null)
  const toolButtonRefs = useRef<Record<ToolId, HTMLButtonElement | null>>({
    tracker: null,
    search: null,
    captions: null,
    pegasus: null,
  })
  const [faceLockListOpen, setFaceLockListOpen] = useState(false)
  const faceLockListRef = useRef<HTMLDivElement>(null)
  const [exportRedactLoading, setExportRedactLoading] = useState(false)
  const [exportRedactError, setExportRedactError] = useState<string | null>(null)
  const [exportQuality, setExportQuality] = useState<ExportQuality>(720)
  const [exportRedactProgress, setExportRedactProgress] = useState<{ percent: number; message: string } | null>(null)
  const [trackingPreviewByRegion, setTrackingPreviewByRegion] = useState<Record<string, TrackingPreviewSample[]>>({})
  const [trackingPreviewLoading, setTrackingPreviewLoading] = useState(false)
  const [trackingPreviewError, setTrackingPreviewError] = useState<string | null>(null)
  const [isScrubbing, setIsScrubbing] = useState(false)
  const [hoverTime, setHoverTime] = useState<number | null>(null)
  const [snapFaceModalOpen, setSnapFaceModalOpen] = useState(false)
  const [snapFaceFrameDataUrl, setSnapFaceFrameDataUrl] = useState<string | null>(null)
  const [snapFaceCapturedAtSec, setSnapFaceCapturedAtSec] = useState(0)
  const [snapFaceError, setSnapFaceError] = useState<string | null>(null)
  const [snapFaceCapturing, setSnapFaceCapturing] = useState(false)
  const snapFaceCounterRef = useRef(0)
  type RedactionWarningEntry = { person_id?: string; label?: string; reason?: string; fallback?: string }
  type RedactionWarnings = {
    unresolved: RedactionWarningEntry[]
    blurFailures: RedactionWarningEntry[]
    faceLockFailures: RedactionWarningEntry[]
  }
  const [redactionWarnings, setRedactionWarnings] = useState<RedactionWarnings | null>(null)
  const [faceLockBuildAlert, setFaceLockBuildAlert] = useState<{ personId: string; label: string; reason?: string } | null>(null)
  const announcedFaceLockFailuresRef = useRef<Record<string, true>>({})
  const [trackMuted, setTrackMuted] = useState<{ video: boolean; audio: boolean }>({ video: false, audio: false })
  const [trackLocked, setTrackLocked] = useState<{ video: boolean; audio: boolean }>({ video: false, audio: false })
  const [overviewTagsExpanded, setOverviewTagsExpanded] = useState(true)
  const [analyzeQuery, setAnalyzeQuery] = useState('')
  const [analyzeLoading, setAnalyzeLoading] = useState(false)
  const [analyzeSuggestionsOpen, setAnalyzeSuggestionsOpen] = useState(false)
  const [analyzeError, setAnalyzeError] = useState<string | null>(null)
  type AnalyzeMessage = { id: string; role: 'user' | 'assistant'; content: string }
  const [analyzeMessages, setAnalyzeMessages] = useState<AnalyzeMessage[]>([])
  const [summaryText, setSummaryText] = useState<string | null>(null)
  const [summaryTags, setSummaryTags] = useState<{ about?: string; topics?: string[]; categories?: string[] } | null>(null)
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [searchSessionResults, setSearchSessionResults] = useState<SearchSessionResult[]>([])
  const [activeSearchSessionId, setActiveSearchSessionId] = useState<string | null>(null)
  const [pegasusJobId, setPegasusJobId] = useState<string | null>(null)
  const [pegasusStatus, setPegasusStatus] = useState<PegasusJobStatus>('idle')
  const [pegasusResult, setPegasusResult] = useState<PegasusResult | null>(null)
  const [pegasusCached, setPegasusCached] = useState(false)
  const [pegasusLoading, setPegasusLoading] = useState(false)
  const [pegasusError, setPegasusError] = useState<string | null>(null)
  const [pegasusApplyLoading, setPegasusApplyLoading] = useState(false)
  const [pegasusApplyError, setPegasusApplyError] = useState<string | null>(null)
  const [pegasusApplyPreview, setPegasusApplyPreview] = useState<PegasusApplyPreview | null>(null)
  const [pegasusApplyModalOpen, setPegasusApplyModalOpen] = useState(false)
  const [pegasusCategoryFilter, setPegasusCategoryFilter] = useState<'all' | PegasusTimelineCategory>('all')
  const [pegasusSeverityFilter, setPegasusSeverityFilter] = useState<Record<PegasusSeverity, boolean>>({
    high: true,
    medium: true,
    low: true,
  })
  const [pegasusFocusedEventId, setPegasusFocusedEventId] = useState<string | null>(null)
  const [pegasusBookmarkedEventIds, setPegasusBookmarkedEventIds] = useState<string[]>([])
  const pegasusRequestIdRef = useRef(0)
  const searchSessionResult = useMemo(() => {
    if (searchSessionResults.length === 0) return null
    return (
      searchSessionResults.find((session) => getSearchSessionId(session) === activeSearchSessionId) ||
      searchSessionResults[searchSessionResults.length - 1]
    )
  }, [activeSearchSessionId, searchSessionResults])
  const analyzeChatEndRef = useRef<HTMLDivElement>(null)
  const [timelineThumbnails, setTimelineThumbnails] = useState<string[]>([])
  const thumbnailsGeneratedRef = useRef(false)
  const [audioWaveformData, setAudioWaveformData] = useState<number[]>([])
  const [audioWaveformStatus, setAudioWaveformStatus] = useState<'idle' | 'loading' | 'ready' | 'unavailable'>('idle')
  const waveformGeneratedRef = useRef(false)
  const timelinePreviewVideoRef = useRef<HTMLVideoElement | null>(null)
  const searchClipRowRefs = useRef<Array<HTMLButtonElement | null>>([])
  const hlsPreviewRef = useRef<InstanceType<typeof Hls> | null>(null)
  const hlsPreviewLoadedUrlRef = useRef<string | null>(null)
  const [previewVideoReady, setPreviewVideoReady] = useState(false)
  const [demoBannerDismissed, setDemoBannerDismissed] = useState(false)
  const isDemoMode = videoId === DEMO_EDITOR_VIDEO_ID && !demoBannerDismissed
  const [videoViewport, setVideoViewport] = useState<VideoViewport>({ left: 0, top: 0, width: 0, height: 0 })
  const [playerAspectRatio, setPlayerAspectRatio] = useState<number>(16 / 9)
  const [videoStageSize, setVideoStageSize] = useState({ width: 0, height: 0 })
  const liveRedactionInFlightRef = useRef(false)
  const liveRedactionPendingTimeRef = useRef<number | null>(null)
  const liveRedactionRequestIdRef = useRef(0)
  const liveRedactionLastResolvedTimeRef = useRef<number | null>(null)
  const liveBlurAnimationFrameRef = useRef<number | null>(null)
  const liveBlurVideoFrameCallbackRef = useRef<number | null>(null)
  const visibleLiveRedactionDetectionsRef = useRef<LiveRedactionDetection[]>([])
  const requestedFaceLaneEntityIdsRef = useRef<Record<string, boolean>>({})
  const liveBlurClipPathId = useId().replace(/:/g, '')

  const getPlaybackAnchorTime = useCallback(() => videoRef.current?.currentTime ?? currentTime, [currentTime])
  const updateVideoViewport = useCallback(() => {
    const container = videoContainerRef.current
    const video = videoRef.current
    if (!container) return

    const rect = container.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) {
      setVideoViewport({ left: 0, top: 0, width: 0, height: 0 })
      return
    }

    const intrinsicWidth = video?.videoWidth ?? 0
    const intrinsicHeight = video?.videoHeight ?? 0
    if (intrinsicWidth <= 0 || intrinsicHeight <= 0) {
      setVideoViewport({ left: 0, top: 0, width: rect.width, height: rect.height })
      return
    }

    const containerAspect = rect.width / rect.height
    const videoAspect = intrinsicWidth / intrinsicHeight

    if (videoAspect > containerAspect) {
      const width = rect.width
      const height = width / videoAspect
      setVideoViewport({ left: 0, top: (rect.height - height) / 2, width, height })
      return
    }

    const height = rect.height
    const width = height * videoAspect
    setVideoViewport({ left: (rect.width - width) / 2, top: 0, width, height })
  }, [])

  /* Tracker: regions and placement */
  const [trackingRegions, setTrackingRegions] = useState<TrackingRegion[]>([])
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null)
  const [trackerShape, setTrackerShape] = useState<'rectangle' | 'ellipse' | 'circle'>('rectangle')
  const [trackerEffect, setTrackerEffect] = useState<'blur' | 'pixelate' | 'solid'>('blur')
  const [trackerReason, setTrackerReason] = useState('')
  const [smoothEdges, setSmoothEdges] = useState(true)
  const [drawMode, setDrawMode] = useState(false)
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null)
  const [drawCurrent, setDrawCurrent] = useState<{ x: number; y: number } | null>(null)
  const [dragState, setDragState] = useState<{ regionId: string; offsetX: number; offsetY: number } | null>(null)
  const drawStateRef = useRef({ drawStart: null as { x: number; y: number } | null, drawCurrent: null as { x: number; y: number } | null })
  drawStateRef.current = { drawStart, drawCurrent }
  const trackerSettingsRef = useRef<{ trackerShape: 'rectangle' | 'ellipse' | 'circle'; trackerEffect: 'blur' | 'pixelate' | 'solid'; trackerReason: string; addTrackingRegion: (r: Omit<TrackingRegion, 'id'>) => void } | null>(null)

  const streamUrl = cached?.stream_url || videoInfo?.hls?.video_url || undefined
  const title = cached?.metadata?.filename || videoInfo?.system_metadata?.filename || videoId || 'Untitled'


  /* ---- HLS: play m3u8 streams in Chrome/Firefox (TwelveLabs returns HLS) ---- */
  const effectiveStreamUrl = streamUrl && isHlsUrl(streamUrl) ? toHlsProxyUrl(streamUrl) : streamUrl
  const useHls = effectiveStreamUrl && isHlsUrl(effectiveStreamUrl) && Hls.isSupported()
  const selectedDetectionCount = useMemo(
    () => apiDetections.filter((item) => !excludedFromRedactionIds.includes(item.id)).length,
    [apiDetections, excludedFromRedactionIds]
  )
  const liveRedactionActive = liveRedactionEnabled && !!streamUrl
  const selectionPreviewActive = !liveRedactionEnabled && !!effectiveStreamUrl && hasRunDetection && selectedDetectionCount > 0
  const liveRedactionPreviewActive = liveRedactionActive || selectionPreviewActive
  const liveRedactionOverlayVisible = liveRedactionPreviewActive
  const markLiveRedactionSeekPending = useCallback(() => {
    if (!liveRedactionPreviewActive) return
    setLiveRedactionSeekPending(true)
  }, [liveRedactionPreviewActive])

  useEffect(() => {
    detectionLoadRequestIdRef.current += 1
    readyRedactionJobIdsRef.current = {}
    setVideoInfo(null)
    setDetectionJobId(null)
    setHasRunDetection(false)
    setApiDetections([])
    setPersonLaneIds([])
    setFaceLaneEntityRangesByPersonId({})
    setExcludedFromRedactionIds([])
    setDetectionLoading(false)
    setDetectionError(null)
    setLiveRedactionDetections([])
    setLiveRedactionLoading(false)
    setLiveRedactionSeekPending(false)
    setLiveRedactionError(null)
    setExportRedactError(null)
    pegasusRequestIdRef.current += 1
    setPegasusJobId(null)
    setPegasusStatus('idle')
    setPegasusResult(null)
    setPegasusCached(false)
    setPegasusLoading(false)
    setPegasusError(null)
    setPegasusApplyLoading(false)
    setPegasusApplyError(null)
    setPegasusApplyPreview(null)
    setPegasusApplyModalOpen(false)
    setPegasusFocusedEventId(null)
    setPegasusBookmarkedEventIds([])
    liveRedactionPendingTimeRef.current = null
    liveRedactionRequestIdRef.current = 0
    liveRedactionLastResolvedTimeRef.current = null
    requestedFaceLaneEntityIdsRef.current = {}
  }, [videoId])

  useEffect(() => {
    setSummaryText(null)
    setSummaryTags(null)
  }, [videoId])

  useEffect(() => {
    storeLastEditorVideoId(videoId)
  }, [videoId])

  useEffect(() => {
    if (analyzeQuery.trim()) {
      setOverviewTagsExpanded(false)
    }
  }, [analyzeQuery])

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(EDITOR_LAST_SEARCH_SESSION_KEY)
      if (!raw) {
        setSearchSessionResults([])
        setActiveSearchSessionId(null)
        return
      }
      const parsed = JSON.parse(raw)
      const sessions = normalizeStoredSearchSessions(parsed)
      if (sessions.length === 0) {
        setSearchSessionResults([])
        setActiveSearchSessionId(null)
        return
      }
      setSearchSessionResults(sessions)
      setActiveSearchSessionId(getSearchSessionId(sessions[sessions.length - 1]))
    } catch {
      setSearchSessionResults([])
      setActiveSearchSessionId(null)
    }
  }, [videoId])

  /* Load overview from TwelveLabs video user_metadata when opening a video */
  useEffect(() => {
    if (!videoId) return
    let cancelled = false
    fetch(`${API_BASE}/api/videos/${encodeURIComponent(videoId)}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((info: {
        system_metadata?: { filename?: string }
        hls?: { video_url?: string | null; thumbnail_urls?: string[] | string | null; status?: string | null } | null
        overview?: { about?: string; topics?: string[]; categories?: string[] }
      } | null) => {
        if (cancelled || !info) return
        setVideoInfo(info)
        if (!info.overview) return
        const o = info.overview
        setSummaryTags({
          about: o.about,
          topics: o.topics,
          categories: o.categories,
        })
        if (o.about) setSummaryText(o.about)
      })
      .catch(() => { /* ignore */ })
    return () => { cancelled = true }
  }, [videoId])

  useEffect(() => {
    if (!effectiveStreamUrl || !videoRef.current) return
    if (!isHlsUrl(effectiveStreamUrl)) return
    if (!Hls.isSupported()) return

    // Avoid reloading the same URL (e.g. effect re-run from parent re-render) which would restart playback
    if (hlsRef.current && hlsLoadedUrlRef.current === effectiveStreamUrl) return

    const video = videoRef.current
    if (hlsRef.current) {
      hlsRef.current.destroy()
      hlsRef.current = null
      hlsLoadedUrlRef.current = null
    }

    const hls = new Hls()
    hlsRef.current = hls
    hlsLoadedUrlRef.current = effectiveStreamUrl
    hls.loadSource(effectiveStreamUrl)
    hls.attachMedia(video)
    hls.on(Hls.Events.ERROR, (_, data) => {
      if (data.fatal) {
        console.error('[HLS] Fatal error', data.type, data.details, data.reason ?? '')
        hls.destroy()
        hlsRef.current = null
        hlsLoadedUrlRef.current = null
      }
    })
    return () => {
      hls.destroy()
      hlsRef.current = null
      hlsLoadedUrlRef.current = null
    }
  }, [effectiveStreamUrl])

  useEffect(() => {
    if (!effectiveStreamUrl || !liveRedactionOverlayVisible || redactionStyle !== 'blur' || !liveBlurOverlayVideoRef.current) {
      if (liveBlurOverlayHlsRef.current) {
        liveBlurOverlayHlsRef.current.destroy()
        liveBlurOverlayHlsRef.current = null
        liveBlurOverlayLoadedUrlRef.current = null
      }
      return
    }
    if (!isHlsUrl(effectiveStreamUrl)) {
      if (liveBlurOverlayHlsRef.current) {
        liveBlurOverlayHlsRef.current.destroy()
        liveBlurOverlayHlsRef.current = null
        liveBlurOverlayLoadedUrlRef.current = null
      }
      return
    }
    if (!Hls.isSupported()) return
    if (liveBlurOverlayHlsRef.current && liveBlurOverlayLoadedUrlRef.current === effectiveStreamUrl) return

    const video = liveBlurOverlayVideoRef.current
    if (liveBlurOverlayHlsRef.current) {
      liveBlurOverlayHlsRef.current.destroy()
      liveBlurOverlayHlsRef.current = null
      liveBlurOverlayLoadedUrlRef.current = null
    }

    const hls = new Hls()
    liveBlurOverlayHlsRef.current = hls
    liveBlurOverlayLoadedUrlRef.current = effectiveStreamUrl
    hls.loadSource(effectiveStreamUrl)
    hls.attachMedia(video)
    hls.on(Hls.Events.ERROR, (_, data) => {
      if (data.fatal) {
        hls.destroy()
        liveBlurOverlayHlsRef.current = null
        liveBlurOverlayLoadedUrlRef.current = null
      }
    })
    return () => {
      hls.destroy()
      liveBlurOverlayHlsRef.current = null
      liveBlurOverlayLoadedUrlRef.current = null
    }
  }, [effectiveStreamUrl, liveRedactionDetections.length, liveRedactionOverlayVisible, redactionStyle])

  /* Reset timeline thumbnails and waveform when video source changes */
  useEffect(() => {
    setTimelineThumbnails([])
    setAudioWaveformData([])
    setAudioWaveformStatus(SHOW_AUDIO_WAVEFORM ? 'idle' : 'unavailable')
    setPreviewVideoReady(false)
    setPlayerAspectRatio(16 / 9)
    thumbnailsGeneratedRef.current = false
    waveformGeneratedRef.current = !SHOW_AUDIO_WAVEFORM
  }, [effectiveStreamUrl])

  /* Hidden preview video: same stream as main, used only for timeline thumbnails/waveform so main video is never sought on pause */
  useEffect(() => {
    if (!effectiveStreamUrl || !timelinePreviewVideoRef.current || !isHlsUrl(effectiveStreamUrl) || !Hls.isSupported()) return
    if (hlsPreviewRef.current && hlsPreviewLoadedUrlRef.current === effectiveStreamUrl) return

    const video = timelinePreviewVideoRef.current
    if (hlsPreviewRef.current) {
      hlsPreviewRef.current.destroy()
      hlsPreviewRef.current = null
      hlsPreviewLoadedUrlRef.current = null
    }

    const hls = new Hls()
    hlsPreviewRef.current = hls
    hlsPreviewLoadedUrlRef.current = effectiveStreamUrl
    hls.loadSource(effectiveStreamUrl)
    hls.attachMedia(video)
    hls.on(Hls.Events.ERROR, (_, data) => {
      if (data.fatal) {
        hls.destroy()
        hlsPreviewRef.current = null
        hlsPreviewLoadedUrlRef.current = null
      }
    })
    return () => {
      hls.destroy()
      hlsPreviewRef.current = null
      hlsPreviewLoadedUrlRef.current = null
    }
  }, [effectiveStreamUrl])

  useLayoutEffect(() => {
    updateVideoViewport()
    const container = videoContainerRef.current
    if (!container) return

    let observer: ResizeObserver | null = null
    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(() => updateVideoViewport())
      observer.observe(container)
    }

    window.addEventListener('resize', updateVideoViewport)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', updateVideoViewport)
    }
  }, [effectiveStreamUrl, updateVideoViewport])

  useLayoutEffect(() => {
    const stage = videoStageRef.current
    if (!stage) return

    const updateStageSize = () => {
      const rect = stage.getBoundingClientRect()
      const nextWidth = Math.max(0, rect.width)
      const nextHeight = Math.max(0, rect.height)
      setVideoStageSize((current) =>
        current.width === nextWidth && current.height === nextHeight
          ? current
          : { width: nextWidth, height: nextHeight }
      )
    }

    updateStageSize()

    let observer: ResizeObserver | null = null
    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(updateStageSize)
      observer.observe(stage)
    }

    window.addEventListener('resize', updateStageSize)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', updateStageSize)
    }
  }, [])

  /* Preload timeline thumbnails + waveform on a hidden video so pausing the main video does nothing (no traverse) */
  useEffect(() => {
    const video = timelinePreviewVideoRef.current
    if (!effectiveStreamUrl || !video || !Number.isFinite(duration) || duration <= 0 || !previewVideoReady) return
    if (thumbnailsGeneratedRef.current && waveformGeneratedRef.current) return

    const frameCount = 8
    const thumbW = 80
    const thumbH = 45
    const waveformSamples = Math.min(320, Math.max(96, Math.floor(duration * 5)))
    const SEEK_TIMEOUT_MS = 2500

    function captureFrameAt(v: HTMLVideoElement, t: number): Promise<string> {
      return new Promise((resolve) => {
        const done = (result: string) => {
          v.removeEventListener('seeked', onSeeked)
          v.removeEventListener('error', onError)
          clearTimeout(tid)
          resolve(result)
        }
        const onSeeked = () => {
          try {
            const canvas = document.createElement('canvas')
            canvas.width = thumbW
            canvas.height = thumbH
            const ctx = canvas.getContext('2d')
            if (!ctx) { done(''); return }
            ctx.drawImage(v, 0, 0, thumbW, thumbH)
            done(canvas.toDataURL('image/jpeg', 0.75))
          } catch {
            done('')
          }
        }
        const onError = () => done('')
        const tid = setTimeout(() => done(''), SEEK_TIMEOUT_MS)
        v.addEventListener('seeked', onSeeked)
        v.addEventListener('error', onError)
        v.currentTime = t
      })
    }

    let cancelled = false
    ;(async () => {
      const v = timelinePreviewVideoRef.current
      if (!v) return

      if (!thumbnailsGeneratedRef.current) {
        const urls: string[] = []
        for (let i = 0; i < frameCount && !cancelled; i++) {
          const t = i === 0 ? 0 : (i / (frameCount - 1)) * duration
          urls.push(await captureFrameAt(v, t) || '')
        }
        if (!cancelled && urls.length > 0) {
          setTimelineThumbnails(urls)
          thumbnailsGeneratedRef.current = true
        }
      }

      if (!SHOW_AUDIO_WAVEFORM) {
        waveformGeneratedRef.current = true
      } else if (!waveformGeneratedRef.current) {
        if (!cancelled) setAudioWaveformStatus('loading')
        try {
          let waveform: number[] = []

          if (!isHlsUrl(effectiveStreamUrl)) {
            try {
              const response = await fetch(effectiveStreamUrl)
              if (!response.ok) throw new Error('Waveform fetch failed')
              const arrayBuffer = await response.arrayBuffer()
              if (!cancelled && arrayBuffer.byteLength > 0) {
                const AudioContextCtor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
                const decodeCtx = new AudioContextCtor()
                try {
                  const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer.slice(0))
                  waveform = extractWaveformFromAudioBuffer(audioBuffer, waveformSamples)
                } finally {
                  decodeCtx.close().catch(() => {})
                }
              }
            } catch {
              waveform = []
            }
          }

          if (!cancelled && waveform.length === 0) {
            try {
              waveform = await sampleWaveformFromMediaElement(v, duration, waveformSamples, SEEK_TIMEOUT_MS)
            } catch {
              waveform = []
            }
          }

          if (!cancelled && waveform.length > 0) {
            setAudioWaveformData(waveform)
            setAudioWaveformStatus('ready')
          } else if (!cancelled) {
            setAudioWaveformStatus('unavailable')
          }
        } catch {
          if (!cancelled) setAudioWaveformStatus('unavailable')
        }
        waveformGeneratedRef.current = true
      }
    })()
    return () => { cancelled = true }
  }, [duration, effectiveStreamUrl, previewVideoReady])

  /* ---- Video event handlers ---- */

  const onLoadedMetadata = useCallback(() => {
    const v = videoRef.current
    if (!v) return
    setDuration(v.duration)
    v.currentTime = 0
    setCurrentTime(0)
    if (v.videoWidth > 0 && v.videoHeight > 0) {
      setPlayerAspectRatio(v.videoWidth / v.videoHeight)
    }
    v.volume = volume
    v.muted = isMuted
    v.playbackRate = playbackRate
    updateVideoViewport()
    // Start playback so the video is playing in the editor viewport (muted autoplay is allowed)
    v.play().catch(() => {})
  }, [volume, isMuted, playbackRate, updateVideoViewport])

  const onTimeUpdate = useCallback(() => {
    const v = videoRef.current
    if (!v) return
    setCurrentTime(v.currentTime)
    if (v.buffered.length > 0) {
      setBuffered(v.buffered.end(v.buffered.length - 1))
    }
  }, [])

  /* ---- Controls ---- */

  const togglePlay = useCallback(() => {
    const v = videoRef.current
    if (!v) return
    if (v.paused || v.ended) {
      v.play().catch(() => {})
    } else {
      v.pause()
    }
  }, [])

  const skip = useCallback((delta: number) => {
    const v = videoRef.current
    if (!v) return
    markLiveRedactionSeekPending()
    v.currentTime = Math.max(0, Math.min(v.duration, v.currentTime + delta))
  }, [markLiveRedactionSeekPending])

  const seekTo = useCallback((fraction: number) => {
    const v = videoRef.current
    if (!v || !Number.isFinite(v.duration)) return
    markLiveRedactionSeekPending()
    v.currentTime = fraction * v.duration
  }, [markLiveRedactionSeekPending])

  const seekToTime = useCallback((seconds: number, options?: { play?: boolean }) => {
    const v = videoRef.current
    if (!v) return
    const t = Math.max(0, Number.isFinite(v.duration) ? Math.min(v.duration, seconds) : seconds)
    markLiveRedactionSeekPending()
    v.currentTime = t
    setCurrentTime(t)
    if (options?.play ?? true) {
      if (v.paused) v.play().catch(() => {})
    } else if (!v.paused) {
      v.pause()
    }
  }, [markLiveRedactionSeekPending])

  const scrubFromEvent = useCallback((e: MouseEvent | React.MouseEvent) => {
    const el = timelineRef.current
    if (!el || !duration || duration <= 0) return
    const rect = el.getBoundingClientRect()
    const clickX = el.scrollLeft + (e.clientX - rect.left)
    const fraction = Math.max(0, Math.min(1, clickX / el.scrollWidth))
    seekTo(fraction)
  }, [seekTo, duration])

  const handleTimelineMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    setIsScrubbing(true)
    scrubFromEvent(e)
  }, [scrubFromEvent])

  useEffect(() => {
    if (!isScrubbing) return
    const onMove = (e: MouseEvent) => scrubFromEvent(e)
    const onUp = () => setIsScrubbing(false)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
  }, [isScrubbing, scrubFromEvent])

  const handleTimelineHover = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const el = timelineRef.current
    if (!el || !duration || duration <= 0) { setHoverTime(null); return }
    const rect = el.getBoundingClientRect()
    const x = el.scrollLeft + (e.clientX - rect.left)
    const frac = x / el.scrollWidth
    setHoverTime(frac >= 0 && frac <= 1 ? frac * duration : null)
  }, [duration])

  const toggleMute = useCallback(() => {
    const v = videoRef.current
    if (!v) return
    v.muted = !v.muted
    setIsMuted(v.muted)
  }, [])

  const changeRate = useCallback((rate: number) => {
    const v = videoRef.current
    if (!v) return
    v.playbackRate = rate
    setPlaybackRate(rate)
  }, [])

  const changeVolume = useCallback((val: number) => {
    const v = videoRef.current
    if (!v) return
    v.volume = val
    setVolume(val)
    if (val > 0 && v.muted) { v.muted = false; setIsMuted(false) }
  }, [])

  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) {
      document.exitFullscreen()
    } else {
      const el = editorCenterRef.current
      if (el?.requestFullscreen) el.requestFullscreen()
    }
  }, [])

  const handleSnapFaceFromVideo = useCallback(async () => {
    const v = videoRef.current
    if (!v) {
      setSnapFaceError('Video is not ready yet.')
      return
    }
    if (!v.paused) {
      try { v.pause() } catch { /* ignore */ }
    }

    setSnapFaceError(null)
    setSnapFaceCapturing(true)
    try {
      const naturalWidth = v.videoWidth
      const naturalHeight = v.videoHeight
      if (!naturalWidth || !naturalHeight) {
        setSnapFaceError('Video frame is not ready yet. Try again in a moment.')
        return
      }
      // Downscale the captured frame so the upload stays well under any
      // reverse-proxy or backend request-size limits. 1280px on the long
      // side is plenty for face detection (the detector itself runs at
      // 640px) and keeps the JPEG payload at a few hundred KB even for
      // 4K source video, which avoids spurious HTTP 413 errors.
      const MAX_CAPTURE_DIM = 1280
      const longSide = Math.max(naturalWidth, naturalHeight)
      const captureScale = longSide > MAX_CAPTURE_DIM ? MAX_CAPTURE_DIM / longSide : 1
      const captureWidth = Math.max(1, Math.round(naturalWidth * captureScale))
      const captureHeight = Math.max(1, Math.round(naturalHeight * captureScale))
      const canvas = document.createElement('canvas')
      canvas.width = captureWidth
      canvas.height = captureHeight
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        setSnapFaceError('Could not capture the current video frame.')
        return
      }
      ctx.drawImage(v, 0, 0, captureWidth, captureHeight)
      let dataUrl: string
      try {
        dataUrl = canvas.toDataURL('image/jpeg', 0.9)
      } catch {
        setSnapFaceError('The video source blocked frame capture (CORS). Cannot snap from this video.')
        return
      }
      const capturedTime = Number.isFinite(v.currentTime) ? v.currentTime : 0
      setSnapFaceFrameDataUrl(dataUrl)
      setSnapFaceCapturedAtSec(capturedTime)
      setSnapFaceModalOpen(true)
    } finally {
      setSnapFaceCapturing(false)
    }
  }, [])

  const handleSnapFaceAdded = useCallback((result: SnapFaceResult) => {
    snapFaceCounterRef.current += 1
    const personId = `snap_${result.entityId.slice(-8)}_${snapFaceCounterRef.current}`
    const itemId = `face-${personId}`
    const newItem: DetectionItem = {
      id: itemId,
      kind: 'face',
      label: result.name.slice(0, 60),
      tags: normalizeDetectionTags([], { shouldAnonymize: true, isOfficial: false }),
      color: '#F59E0B',
      snapBase64: result.faceBase64,
      personId,
      timeRanges: [],
      appearances: [],
      entityId: result.entityId,
      appearanceCount: 0,
      shouldAnonymize: true,
      isOfficial: false,
    }
    setApiDetections((previous) => sortDetectionItems([...previous, newItem]))
    setExcludedFromRedactionIds((previous) => previous.filter((id) => id !== itemId))
    setHasRunDetection(true)
    setShowAnonymizeOnly(true)
    setRightSidebarOpen(true)
    setSnapFaceModalOpen(false)
    setSnapFaceFrameDataUrl(null)
  }, [])

  useEffect(() => {
    const onFullscreenChange = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', onFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', onFullscreenChange)
  }, [])

  useEffect(() => {
    updateVideoViewport()
  }, [isFullscreen, updateVideoViewport])

  const runAnalyze = useCallback(async () => {
    const prompt = analyzeQuery.trim()
    if (!videoId || !prompt) {
      setAnalyzeError(prompt ? 'Video not loaded.' : 'Enter a question to analyze.')
      return
    }
    setAnalyzeError(null)
    setAnalyzeLoading(true)
    const userMsgId = `u-${Date.now()}`
    const assistantMsgId = `a-${Date.now()}`
    setAnalyzeMessages((prev) => [...prev, { id: userMsgId, role: 'user', content: prompt }])
    setAnalyzeQuery('')
    try {
      const res = await fetch(`${API_BASE}/api/analyze-custom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId, prompt }),
      })
      const json = await res.json().catch(() => ({}))
      if (!res.ok) {
        const err = json?.error || `Request failed (${res.status})`
        setAnalyzeError(err)
        setAnalyzeMessages((prev) => [...prev, { id: assistantMsgId, role: 'assistant', content: `Error: ${err}` }])
        return
      }
      const text = json?.data ?? (typeof json === 'string' ? json : '')
      setAnalyzeMessages((prev) => [...prev, { id: assistantMsgId, role: 'assistant', content: text || 'No response.' }])
    } catch (e) {
      const err = e instanceof Error ? e.message : 'Analysis failed'
      setAnalyzeError(err)
      setAnalyzeMessages((prev) => [...prev, { id: assistantMsgId, role: 'assistant', content: `Error: ${err}` }])
    } finally {
      setAnalyzeLoading(false)
    }
  }, [videoId, analyzeQuery])

  const runGenerateSummary = useCallback(async () => {
    if (!videoId) return
    setSummaryLoading(true)
    setSummaryTags(null)
    try {
      const res = await fetch(`${API_BASE}/api/analyze-custom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: videoId,
          prompt: `Provide a structured overview of this video. Return ONLY valid JSON (no markdown, no extra text) in this exact format:
{"about": "one sentence summary of the video", "topics": ["topic1", "topic2", "topic3"], "categories": ["category1", "category2", "category3"]}
- about: brief one-sentence description
- topics: 3-6 main topics or themes (short phrases)
- categories: 3-6 content types (e.g. Tutorial, Interview, Documentary, Presentation, Demo)`,
        }),
      })
      const json = await res.json().catch(() => ({}))
      if (!res.ok) {
        setSummaryText(`Failed to generate overview: ${json?.error || res.status}`)
        return
      }
      const raw = (json?.data ?? '').trim()
      if (!raw) {
        setSummaryText('No overview generated.')
        return
      }
      // Try to parse JSON (may be wrapped in ```json ... ```)
      let parsed: { about?: string; topics?: string[]; categories?: string[] } | null = null
      const jsonMatch = raw.match(/```(?:json)?\s*([\s\S]*?)```/) || raw.match(/\{[\s\S]*\}/)
      const jsonStr = jsonMatch ? (jsonMatch[1] ?? jsonMatch[0]).trim() : raw
      try {
        parsed = JSON.parse(jsonStr) as { about?: string; topics?: string[]; categories?: string[] }
      } catch {
        // Fallback: use full text as about
        parsed = { about: raw.slice(0, 200), topics: [], categories: [] }
      }
      if (parsed && (parsed.about || (parsed.topics && parsed.topics.length) || (parsed.categories && parsed.categories.length))) {
        const tags = {
          about: typeof parsed.about === 'string' ? parsed.about : undefined,
          topics: Array.isArray(parsed.topics) ? parsed.topics : undefined,
          categories: Array.isArray(parsed.categories) ? parsed.categories : undefined,
        }
        setSummaryTags(tags)
        setSummaryText(parsed.about || raw.slice(0, 300))
        // Persist overview to TwelveLabs video user_metadata (once per video)
        fetch(`${API_BASE}/api/videos/${encodeURIComponent(videoId)}/overview`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            about: tags.about,
            topics: tags.topics ?? [],
            categories: tags.categories ?? [],
          }),
        }).catch(() => { /* ignore save errors */ })
      } else {
        setSummaryText(raw)
      }
    } catch (e) {
      setSummaryText(e instanceof Error ? e.message : 'Failed to generate overview.')
    } finally {
      setSummaryLoading(false)
    }
  }, [videoId])

  const clearDetectionResults = useCallback((errorMessage: string | null, requestId: number) => {
    if (detectionLoadRequestIdRef.current !== requestId) return
    setDetectionJobId(null)
    setApiDetections([])
    setExcludedFromRedactionIds([])
    setHasRunDetection(false)
    setDetectionError(errorMessage)
  }, [])

  const loadDetectionItemsForJob = useCallback(async (
    jobId: string,
    requestId: number,
    interactive: boolean,
  ): Promise<boolean> => {
    const startedAt = Date.now()
    const waitForReadyMs = interactive ? 30000 : 0

    while (true) {
      const [facesRes, objectsRes] = await Promise.all([
        fetch(`${API_BASE}/api/faces/${encodeURIComponent(jobId)}`),
        fetch(`${API_BASE}/api/objects/${encodeURIComponent(jobId)}`),
      ])
      const facesJson = await facesRes.json().catch(() => ({})) as {
        unique_faces?: FaceDetectionApiRecord[]
        error?: string
        message?: string
      }
      const objectsJson = await objectsRes.json().catch(() => ({})) as {
        unique_objects?: ObjectDetectionApiRecord[]
        error?: string
        message?: string
      }

      if (detectionLoadRequestIdRef.current !== requestId) return false

      if (facesRes.status === 202 || objectsRes.status === 202) {
        const shouldKeepPolling = waitForReadyMs > 0 && Date.now() - startedAt < waitForReadyMs
        if (shouldKeepPolling) {
          await delay(1200)
          continue
        }
        if (interactive) {
          clearDetectionResults('Analysis still in progress. Try again in a moment.', requestId)
        }
        return false
      }
      if (facesRes.status === 404 || objectsRes.status === 404) {
        if (interactive) {
          clearDetectionResults('Detection data is not ready for this video yet. Run processing for this video and try again.', requestId)
        }
        return false
      }
      if (!facesRes.ok || !objectsRes.ok) {
        if (interactive) {
          const faceError = facesJson.error || facesJson.message
          const objectError = objectsJson.error || objectsJson.message
          clearDetectionResults(faceError || objectError || 'Detection request failed.', requestId)
        }
        return false
      }

      const items = buildDetectionItemsFromApi(
        Array.isArray(facesJson.unique_faces) ? facesJson.unique_faces : [],
        Array.isArray(objectsJson.unique_objects) ? objectsJson.unique_objects : [],
      )
      if (detectionLoadRequestIdRef.current !== requestId) return false

      setDetectionJobId(jobId)
      readyRedactionJobIdsRef.current[jobId] = true
      setApiDetections(items)
      setExcludedFromRedactionIds(items.map((item) => item.id))
      setHasRunDetection(true)
      if (interactive && items.length === 0) {
        setDetectionError('No faces or objects found for this video.')
      } else {
        setDetectionError(null)
      }
      return true
    }
  }, [clearDetectionResults])

  const hydrateStoredDetections = useCallback(async ({
    ensure = false,
    interactive = false,
  }: {
    ensure?: boolean
    interactive?: boolean
  } = {}): Promise<boolean> => {
    if (!videoId) {
      if (interactive) setDetectionError('Video not loaded.')
      return false
    }

    const requestId = detectionLoadRequestIdRef.current + 1
    detectionLoadRequestIdRef.current = requestId

    if (interactive) {
      setDetectionError(null)
      setDetectionLoading(true)
    }

    try {
      const params = new URLSearchParams()
      if (ensure) {
        params.set('exact', 'true')
        params.set('ensure', 'true')
      }
      const response = await fetch(`${API_BASE}/api/jobs/by-video/${encodeURIComponent(videoId)}?${params.toString()}`)
      const data = await response.json().catch(() => ({})) as {
        job_id?: string
        status?: string
        local_status?: string
        created?: boolean
        error?: string
      }

      if (detectionLoadRequestIdRef.current !== requestId) return false

      const jobId = response.ok && data.job_id ? String(data.job_id) : ''
      if (!jobId) {
        if (interactive) {
          clearDetectionResults(data.error || 'No saved local detection result was found for this video yet.', requestId)
        }
        return false
      }

      setDetectionJobId(jobId)
      if (ensure && (data.created || data.status !== 'ready')) {
        const startedAt = Date.now()
        while (Date.now() - startedAt < 180000) {
          const statusRes = await fetch(`${API_BASE}/api/index/${encodeURIComponent(jobId)}`)
          const statusJson = await statusRes.json().catch(() => ({})) as {
            status?: string
            error?: string
          }
          if (detectionLoadRequestIdRef.current !== requestId) return false
          if (!statusRes.ok) {
            throw new Error(statusJson.error || 'Could not check detection status for this video.')
          }
          if (statusJson.status === 'ready') break
          if (statusJson.status === 'failed') {
            throw new Error(statusJson.error || 'Detection pipeline failed for this video.')
          }
          await delay(1500)
        }
      } else if (data.status !== 'ready') {
        return false
      }

      return loadDetectionItemsForJob(jobId, requestId, interactive)
    } catch (e) {
      if (interactive) {
        clearDetectionResults(e instanceof Error ? e.message : 'Detection request failed', requestId)
      }
      return false
    } finally {
      if (interactive && detectionLoadRequestIdRef.current === requestId) {
        setDetectionLoading(false)
      }
    }
  }, [clearDetectionResults, loadDetectionItemsForJob, videoId])

  useEffect(() => {
    void hydrateStoredDetections()
  }, [hydrateStoredDetections])

  const runDetect = useCallback(async () => {
    await hydrateStoredDetections({ ensure: true, interactive: true })
  }, [hydrateStoredDetections])

  const loadPegasusJob = useCallback(async (
    jobId: string,
    requestId: number,
    options: { poll?: boolean } = {},
  ) => {
    const shouldPoll = options.poll ?? true
    let attempts = 0
    while (attempts < 90) {
      if (pegasusRequestIdRef.current !== requestId) return
      const res = await fetch(`${API_BASE}/api/pegasus/privacy-assist/jobs/${encodeURIComponent(jobId)}`)
      const json = await res.json().catch(() => ({})) as PegasusJobResponse
      if (pegasusRequestIdRef.current !== requestId) return
      if (!res.ok) {
        throw new Error(json.error || `Pegasus request failed (${res.status})`)
      }

      const status = (json.status || 'processing') as PegasusJobStatus
      setPegasusStatus(status)
      setPegasusCached(Boolean(json.cached))
      if (json.result) {
        setPegasusResult(json.result)
        setPegasusFocusedEventId(json.result.timeline_events?.[0]?.id || null)
      }
      if (status === 'ready' || status === 'failed' || !shouldPoll) {
        if (status === 'failed') {
          setPegasusError(json.error || 'Pegasus analysis failed.')
        }
        return
      }
      attempts += 1
      await delay(2000)
    }
    throw new Error('Pegasus analysis is still processing. Try again in a moment.')
  }, [])

  const runPegasusAssist = useCallback(async (force = false) => {
    if (!videoId) {
      setPegasusError('Video not loaded.')
      return
    }
    const requestId = pegasusRequestIdRef.current + 1
    pegasusRequestIdRef.current = requestId
    setPegasusError(null)
    setPegasusLoading(true)
    setPegasusStatus('queued')
    try {
      const res = await fetch(`${API_BASE}/api/pegasus/privacy-assist/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: videoId,
          local_job_id: detectionJobId || undefined,
          force,
        }),
      })
      const json = await res.json().catch(() => ({})) as PegasusJobResponse
      if (pegasusRequestIdRef.current !== requestId) return
      if (!res.ok) {
        throw new Error(json.error || `Pegasus request failed (${res.status})`)
      }
      const jobId = json.job_id || json.result?.metadata?.artifact_id
      if (!jobId) {
        throw new Error('Pegasus did not return a job id.')
      }
      setPegasusJobId(jobId)
      setPegasusCached(Boolean(json.cached))
      setPegasusStatus((json.status || 'processing') as PegasusJobStatus)
      if (json.result) {
        setPegasusResult(json.result)
        setPegasusFocusedEventId(json.result.timeline_events?.[0]?.id || null)
      }
      if ((json.status || '') !== 'ready') {
        await loadPegasusJob(jobId, requestId, { poll: true })
      }
    } catch (e) {
      if (pegasusRequestIdRef.current === requestId) {
        setPegasusStatus('failed')
        setPegasusError(e instanceof Error ? e.message : 'Pegasus analysis failed.')
      }
    } finally {
      if (pegasusRequestIdRef.current === requestId) {
        setPegasusLoading(false)
      }
    }
  }, [detectionJobId, loadPegasusJob, videoId])

  const openPegasusApplyPreview = useCallback(async () => {
    const jobId = pegasusJobId || pegasusResult?.metadata?.job_id || pegasusResult?.metadata?.artifact_id
    if (!jobId || pegasusStatus !== 'ready') {
      setPegasusApplyError('Run Meta Insights before applying recommendations.')
      return
    }
    setPegasusApplyLoading(true)
    setPegasusApplyError(null)
    try {
      const res = await fetch(`${API_BASE}/api/pegasus/privacy-assist/jobs/${encodeURIComponent(jobId)}/apply-preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ local_job_id: detectionJobId || undefined }),
      })
      const json = await res.json().catch(() => ({})) as PegasusApplyPreview
      if (!res.ok) {
        throw new Error(json.error || `Pegasus apply preview failed (${res.status})`)
      }
      setPegasusApplyPreview({
        can_apply: Array.isArray(json.can_apply) ? json.can_apply : [],
        review_only: Array.isArray(json.review_only) ? json.review_only : [],
        unsupported: Array.isArray(json.unsupported) ? json.unsupported : [],
        summary: json.summary || {},
      })
      setPegasusApplyModalOpen(true)
    } catch (e) {
      setPegasusApplyError(e instanceof Error ? e.message : 'Pegasus apply preview failed.')
    } finally {
      setPegasusApplyLoading(false)
    }
  }, [detectionJobId, pegasusJobId, pegasusResult, pegasusStatus])

  const selectedFacePersonIds = useMemo(
    () =>
      apiDetections
        .filter((item) => item.kind === 'face' && item.personId && !excludedFromRedactionIds.includes(item.id))
        .map((item) => item.personId as string),
    [apiDetections, excludedFromRedactionIds]
  )
  const selectedObjectClasses = useMemo(
    () =>
      apiDetections
        .filter((item) => item.kind === 'object' && item.objectClass && !excludedFromRedactionIds.includes(item.id))
        .map((item) => item.objectClass as string),
    [apiDetections, excludedFromRedactionIds]
  )
  const faceDetectionItemsByPersonId = useMemo(() => {
    const map: Record<string, DetectionItem> = {}
    for (const item of apiDetections) {
      if (item.kind === 'face' && item.personId) {
        map[item.personId] = item
      }
    }
    return map
  }, [apiDetections])

  // Surface the first face-lock build failure as a one-shot toast so
  // the user knows the live preview / export will fall back to
  // per-frame relock for that face. We deduplicate per personId so a
  // single failure doesn't keep re-announcing as React re-renders.
  useEffect(() => {
    for (const [personId, state] of Object.entries(faceLockBuildByPersonId)) {
      if (!state || state.status !== 'failed') continue
      if (announcedFaceLockFailuresRef.current[personId]) continue
      announcedFaceLockFailuresRef.current[personId] = true
      const item = faceDetectionItemsByPersonId[personId]
      const label = item?.label || faceLockLabelByPersonIdRef.current[personId] || personId
      setFaceLockBuildAlert({
        personId,
        label,
        reason: state.message || undefined,
      })
      break
    }
  }, [faceDetectionItemsByPersonId, faceLockBuildByPersonId])

  // Locked persons are those whose precomputed face-lock lane is fully
  // loaded. The blur for these persons is drawn from the lane (which was
  // built bidirectionally on the local video using fused tracking +
  // InsightFace anchors + TwelveLabs guidance), never from per-frame
  // InsightFace re-localization. This guarantees the blur stays glued
  // to the face across camera shakes, pans, zooms, and brief misses.
  const lockedFacePersonIds = useMemo(
    () => selectedFacePersonIds.filter((pid) => !!faceLockLanesByPersonId[pid]),
    [faceLockLanesByPersonId, selectedFacePersonIds],
  )
  const lockedFacePersonIdsSet = useMemo(() => new Set(lockedFacePersonIds), [lockedFacePersonIds])

  // Aggregated build status across all selected persons. Each face-lock
  // build runs in its own backend thread, so two clicks on different
  // persons run in parallel — this aggregate is just the average for
  // the small ring shown near the Export button.
  const faceLockBuildSummary = useMemo(() => {
    const buildingPersonIds = selectedFacePersonIds.filter((pid) => {
      const state = faceLockBuildByPersonId[pid]
      return state && (state.status === 'queued' || state.status === 'running')
    })
    const readyCount = lockedFacePersonIds.length
    const totalCount = selectedFacePersonIds.length
    let averagePercent = 0
    if (buildingPersonIds.length > 0) {
      averagePercent = Math.round(
        buildingPersonIds.reduce(
          (acc, pid) => acc + (faceLockBuildByPersonId[pid]?.percent || 0),
          0,
        ) / buildingPersonIds.length,
      )
    } else if (totalCount > 0 && readyCount === totalCount) {
      averagePercent = 100
    }
    return {
      buildingCount: buildingPersonIds.length,
      readyCount,
      totalCount,
      averagePercent,
      hasActive: buildingPersonIds.length > 0,
    }
  }, [faceLockBuildByPersonId, lockedFacePersonIds, selectedFacePersonIds])

  // Per-person entry used to render the export-bar face-lock chips.
  // Each entry has the face snap, label, and the current build percent.
  const faceLockEntries = useMemo(() => {
    const lockedSet = new Set(lockedFacePersonIds)
    const entries: Array<{
      personId: string
      label: string
      snapBase64?: string | null
      color?: string | null
      percent: number
      status: 'ready' | 'running' | 'queued' | 'failed' | 'pending'
    }> = []
    for (const personId of selectedFacePersonIds) {
      const item = faceDetectionItemsByPersonId[personId]
      const state = faceLockBuildByPersonId[personId]
      const isReady = lockedSet.has(personId) || state?.status === 'ready'
      let status: 'ready' | 'running' | 'queued' | 'failed' | 'pending' = 'pending'
      if (isReady) status = 'ready'
      else if (state?.status === 'running') status = 'running'
      else if (state?.status === 'queued') status = 'queued'
      else if (state?.status === 'failed') status = 'failed'
      const percent = isReady ? 100 : Math.max(0, Math.min(100, Math.round(state?.percent || 0)))
      entries.push({
        personId,
        label: item?.label || personId,
        snapBase64: item?.snapBase64 || null,
        color: item?.color || null,
        percent,
        status,
      })
    }
    return entries
  }, [faceDetectionItemsByPersonId, faceLockBuildByPersonId, lockedFacePersonIds, selectedFacePersonIds])

  // Kick off (or resume) face-lock lane builds for any newly selected
  // person. Polls the build status until ready, then caches the lane
  // in component state. Once ready, the live overlay below uses the
  // lane-derived bbox at the current playback time and skips the
  // network roundtrip per frame.
  useEffect(() => {
    if (!detectionJobId || selectedFacePersonIds.length === 0) return
    let cancelled = false

    const buildAndPoll = async (personId: string) => {
      if (cancelled) return
      if (faceLockLanesByPersonIdRef.current[personId]) return
      const attempts = faceLockBuildAttemptsRef.current
      if (attempts[personId]) return
      attempts[personId] = true

      try {
        // First, try to load the lane directly from the cached file
        // on disk. If the backend already has a face-lock lane for
        // this (job, person), there is no point queuing or polling a
        // build — just apply the stored result and we're done.
        const cachedRes = await fetch(
          `${API_BASE}/api/face-lock-track/${encodeURIComponent(detectionJobId)}/${encodeURIComponent(personId)}?include_lane=true`,
        )
        if (cancelled) return
        if (cachedRes.ok) {
          const cachedData = await cachedRes.json().catch(() => ({})) as {
            status?: FaceLockBuildState['status']
            cached?: boolean
            lane?: FaceLockLane
          }
          if (!cancelled && cachedData.status === 'ready' && cachedData.lane) {
            setFaceLockLanesByPersonId((prev) => ({ ...prev, [personId]: cachedData.lane as FaceLockLane }))
            setFaceLockBuildByPersonId((prev) => ({
              ...prev,
              [personId]: { status: 'ready', percent: 100, progress: 1, message: 'cached' },
            }))
            return
          }
        }
      } catch {
        // Ignore cache-fetch errors and fall through to a fresh build.
      }

      setFaceLockBuildByPersonId((prev) => ({
        ...prev,
        [personId]: { status: 'queued', percent: 0, progress: 0, message: 'Locking onto face...' },
      }))

      try {
        const startRes = await fetch(`${API_BASE}/api/face-lock-track/build`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ job_id: detectionJobId, person_id: personId }),
        })
        if (cancelled) return
        if (!startRes.ok) {
          const err = await startRes.json().catch(() => ({}))
          throw new Error((err as { error?: string }).error || `Build start failed (${startRes.status})`)
        }

        let lastPercent = 0
        const startedAt = Date.now()
        // Poll until ready (or failed). The build runs in a backend
        // thread; cap the wait to ~10 minutes for very long videos.
        while (!cancelled && Date.now() - startedAt < 10 * 60 * 1000) {
          await delay(800)
          if (cancelled) return
          const statusRes = await fetch(
            `${API_BASE}/api/face-lock-track/${encodeURIComponent(detectionJobId)}/${encodeURIComponent(personId)}?include_lane=true`,
          )
          if (!statusRes.ok && statusRes.status !== 500) {
            throw new Error(`Status fetch failed (${statusRes.status})`)
          }
          const data = await statusRes.json().catch(() => ({})) as {
            status?: FaceLockBuildState['status']
            percent?: number
            progress?: number
            message?: string | null
            lane?: FaceLockLane
          }
          if (cancelled) return
          const status = (data.status || 'running') as FaceLockBuildState['status']
          const percent = typeof data.percent === 'number' ? data.percent : lastPercent
          lastPercent = percent
          setFaceLockBuildByPersonId((prev) => ({
            ...prev,
            [personId]: {
              status,
              percent,
              progress: typeof data.progress === 'number' ? data.progress : percent / 100,
              message: data.message ?? null,
            },
          }))
          if (status === 'ready' && data.lane) {
            setFaceLockLanesByPersonId((prev) => ({ ...prev, [personId]: data.lane as FaceLockLane }))
            return
          }
          if (status === 'failed') {
            throw new Error(data.message || 'Face-lock build failed')
          }
        }
      } catch (err) {
        if (cancelled) return
        console.warn('[FaceLock] Build failed for', personId, err)
        setFaceLockBuildByPersonId((prev) => ({
          ...prev,
          [personId]: {
            status: 'failed',
            percent: 0,
            progress: 0,
            message: err instanceof Error ? err.message : 'Face-lock build failed',
          },
        }))
        // Allow a manual retry on the next selection toggle.
        delete faceLockBuildAttemptsRef.current[personId]
      }
    }

    for (const personId of selectedFacePersonIds) {
      buildAndPoll(personId)
    }

    return () => {
      cancelled = true
    }
  }, [detectionJobId, selectedFacePersonIds])

  // Drop cached lanes / build state for persons that are no longer
  // selected so the next selection of the same person retries the build.
  useEffect(() => {
    const selected = new Set(selectedFacePersonIds)
    setFaceLockLanesByPersonId((prev) => {
      const next: Record<string, FaceLockLane> = {}
      let changed = false
      for (const [pid, lane] of Object.entries(prev)) {
        if (selected.has(pid)) next[pid] = lane
        else changed = true
      }
      return changed ? next : prev
    })
    setFaceLockBuildByPersonId((prev) => {
      const next: Record<string, FaceLockBuildState> = {}
      let changed = false
      for (const [pid, state] of Object.entries(prev)) {
        if (selected.has(pid)) next[pid] = state
        else changed = true
      }
      return changed ? next : prev
    })
    for (const pid of Object.keys(faceLockBuildAttemptsRef.current)) {
      if (!selected.has(pid)) delete faceLockBuildAttemptsRef.current[pid]
    }
  }, [selectedFacePersonIds])

  useEffect(() => {
    const validPersonIds = new Set(
      apiDetections
        .filter((item) => item.kind === 'face' && item.personId)
        .map((item) => item.personId as string),
    )
    const cachedVideoEntries = videoId ? (loadFaceLaneDebugCache()[videoId] || {}) : {}
    const cachedEntityRangesByPersonId: Record<string, DetectionTimeRange[]> = {}
    const nextRequested: Record<string, boolean> = {}

    for (const item of apiDetections) {
      if (item.kind !== 'face' || !item.personId) continue
      const cachedEntry = cachedVideoEntries[item.personId]
      if (
        cachedEntry?.marengoTimeRanges &&
        cachedEntry.marengoTimeRanges.length > 0 &&
        !!item.entityId &&
        !!cachedEntry.entityId &&
        cachedEntry.entityId === item.entityId
      ) {
        cachedEntityRangesByPersonId[item.personId] = cachedEntry.marengoTimeRanges
        nextRequested[item.personId] = true
      }
    }

    setPersonLaneIds((previous) => previous.filter((personId) => validPersonIds.has(personId)))
    setFaceLaneEntityRangesByPersonId(
      Object.fromEntries(
        Object.entries(cachedEntityRangesByPersonId)
          .filter(([personId, ranges]) => validPersonIds.has(personId) && Array.isArray(ranges) && ranges.length > 0),
      ),
    )

    for (const personId of Object.keys(requestedFaceLaneEntityIdsRef.current)) {
      if (validPersonIds.has(personId)) {
        nextRequested[personId] = true
      }
    }
    requestedFaceLaneEntityIdsRef.current = nextRequested
  }, [apiDetections, videoId])

  useEffect(() => {
    if (!videoId) return
    for (const item of apiDetections) {
      if (item.kind !== 'face' || !item.personId) continue
      updateFaceLaneDebugCacheEntry(videoId, item.personId, {
        label: item.label,
        entityId: item.entityId ?? null,
        localTimeRanges: item.timeRanges || [],
        localAppearances: item.appearances || [],
        localSegments: buildFaceTimelineSegments(item),
      })
    }
  }, [apiDetections, videoId])

  const ensureFaceTimelineLane = useCallback((personId?: string | null) => {
    if (!personId) return
    setPersonLaneIds((previous) => (
      previous.includes(personId) ? previous : [...previous, personId]
    ))
  }, [])

  const confirmPegasusApply = useCallback(() => {
    if (!pegasusApplyPreview) return
    const selectionIds = new Set(
      pegasusApplyPreview.can_apply
        .map((item) => item.selection_id)
        .filter((id): id is string => Boolean(id))
    )
    if (selectionIds.size > 0) {
      setExcludedFromRedactionIds((previous) => previous.filter((id) => !selectionIds.has(id)))
    }
    for (const item of pegasusApplyPreview.can_apply) {
      if (item.person_id) ensureFaceTimelineLane(item.person_id)
    }
    const reviewEventIds = pegasusApplyPreview.review_only.flatMap((item) => item.event_ids || [])
    if (reviewEventIds.length > 0) {
      setPegasusBookmarkedEventIds((previous) => Array.from(new Set([...previous, ...reviewEventIds])))
      setPegasusFocusedEventId((current) => current || reviewEventIds[0] || null)
    }
    setPegasusApplyModalOpen(false)
  }, [ensureFaceTimelineLane, pegasusApplyPreview])

  const removeFaceTimelineLane = useCallback((personId: string) => {
    setPersonLaneIds((previous) => previous.filter((id) => id !== personId))
    setFaceLaneEntityRangesByPersonId((previous) => {
      if (!(personId in previous)) return previous
      const next = { ...previous }
      delete next[personId]
      return next
    })
    if (requestedFaceLaneEntityIdsRef.current[personId]) {
      const nextRequested = { ...requestedFaceLaneEntityIdsRef.current }
      delete nextRequested[personId]
      requestedFaceLaneEntityIdsRef.current = nextRequested
    }
  }, [])

  useEffect(() => {
    if (!videoId || !hasRunDetection || personLaneIds.length === 0) return

    const controllers: AbortController[] = []
    let cancelled = false

    for (const personId of personLaneIds) {
      const item = faceDetectionItemsByPersonId[personId]
      if (!item?.entityId || requestedFaceLaneEntityIdsRef.current[personId]) continue

      requestedFaceLaneEntityIdsRef.current[personId] = true
      const controller = new AbortController()
      controllers.push(controller)

      fetch(`${API_BASE}/api/entities/${encodeURIComponent(item.entityId)}/time-ranges`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId }),
        signal: controller.signal,
      })
        .then(async (res) => {
          const data = await res.json().catch(() => ({})) as { time_ranges?: DetectionTimeRange[] }
          if (!res.ok || cancelled) return
          if (!Array.isArray(data.time_ranges) || data.time_ranges.length === 0) return
          setFaceLaneEntityRangesByPersonId((previous) => ({
            ...previous,
            [personId]: data.time_ranges as DetectionTimeRange[],
          }))
          updateFaceLaneDebugCacheEntry(videoId, personId, {
            label: item.label,
            entityId: item.entityId ?? null,
            marengoTimeRanges: data.time_ranges as DetectionTimeRange[],
            marengoSegments: buildFaceTimelineSegments(item, data.time_ranges as DetectionTimeRange[]),
          })
        })
        .catch(() => {
          /* local timeline lane remains available even when entity search is unavailable */
        })
    }

    return () => {
      cancelled = true
      controllers.forEach((controller) => controller.abort())
    }
  }, [faceDetectionItemsByPersonId, hasRunDetection, personLaneIds, videoId])

  const personTimelineLanes = useMemo((): FaceTimelineLane[] => (
    personLaneIds.flatMap((personId) => {
      const item = faceDetectionItemsByPersonId[personId]
      if (!item) return []

      const marengoRanges = faceLaneEntityRangesByPersonId[personId]
      const localAppearanceSegments = buildFaceTimelineSegmentsFromAppearances(item.appearances)
      const localRangeSegments = buildFaceTimelineSegmentsFromRanges(item.timeRanges)
      return [{
        personId,
        item,
        active: !excludedFromRedactionIds.includes(item.id),
        segments: buildFaceTimelineSegments(item, marengoRanges),
        source: localAppearanceSegments.length > 0 || localRangeSegments.length > 0 || !marengoRanges || marengoRanges.length === 0
          ? 'appearances'
          : 'marengo',
      }]
    })
  ), [excludedFromRedactionIds, faceDetectionItemsByPersonId, faceLaneEntityRangesByPersonId, personLaneIds])

  const waitForRedactionJobReady = useCallback(async (jobId: string, initialStatus?: string | null) => {
    if (!jobId) {
      throw new Error('No local processing job found for this video.')
    }
    if (readyRedactionJobIdsRef.current[jobId]) {
      return jobId
    }
    if (initialStatus === 'ready') {
      readyRedactionJobIdsRef.current[jobId] = true
      return jobId
    }

    const startedAt = Date.now()
    while (Date.now() - startedAt < 180000) {
      const statusRes = await fetch(`${API_BASE}/api/index/${encodeURIComponent(jobId)}`)
      const statusJson = await statusRes.json().catch(() => ({})) as {
        status?: string
        error?: string
      }
      if (!statusRes.ok) {
        throw new Error(statusJson.error || 'Could not check local processing status for this video.')
      }
      if (statusJson.status === 'ready') {
        readyRedactionJobIdsRef.current[jobId] = true
        return jobId
      }
      if (statusJson.status === 'failed') {
        throw new Error(statusJson.error || 'Local processing failed for this video.')
      }
      await delay(1500)
    }

    throw new Error('Local processing is still running. Please try again in a moment.')
  }, [])

  const resolveRedactionJobId = useCallback(async () => {
    if (detectionJobId) {
      return waitForRedactionJobReady(detectionJobId)
    }
    if (!videoId) {
      throw new Error('Video not loaded.')
    }

    const r = await fetch(`${API_BASE}/api/jobs/by-video/${encodeURIComponent(videoId)}?ensure=true&exact=true`)
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error((err as { error?: string }).error || 'No local processing job found for this video.')
    }
    const data = await r.json().catch(() => ({}))
    if (!data.job_id) {
      throw new Error('No local processing job found for this video.')
    }
    const jobId = data.job_id as string
    setDetectionJobId(jobId)
    return waitForRedactionJobReady(jobId, typeof data.status === 'string' ? data.status : null)
  }, [detectionJobId, videoId, waitForRedactionJobReady])

  const requestLiveRedaction = useCallback(async (requestedTime: number, options?: { force?: boolean; resetTracking?: boolean }) => {
    if (!liveRedactionPreviewActive || !effectiveStreamUrl) return
    if (!Number.isFinite(requestedTime) || requestedTime < 0) return

    const lastResolvedTime = liveRedactionLastResolvedTimeRef.current
    if (!options?.force && !isPlaying && lastResolvedTime !== null && Math.abs(lastResolvedTime - requestedTime) < 0.08) {
      return
    }

    if (liveRedactionInFlightRef.current) {
      liveRedactionPendingTimeRef.current = requestedTime
      return
    }

    liveRedactionInFlightRef.current = true
    setLiveRedactionLoading(true)
    const requestId = ++liveRedactionRequestIdRef.current

    try {
      const jobId = await resolveRedactionJobId()
      // [FACE-LOCK LANE BLUR — DISABLED]
      // When re-enabled, exclude lane-locked persons from this request
      // so the backend only spends time on objects and not-yet-locked
      // faces. For now, route every selected person through the
      // /api/live-redaction/detect path:
      // const liveRequestPersonIds = hasRunDetection
      //   ? selectedFacePersonIds.filter((pid) => !lockedFacePersonIdsSet.has(pid))
      //   : undefined
      const liveRequestPersonIds = hasRunDetection ? selectedFacePersonIds : undefined
      const liveIncludeFaces = !hasRunDetection || (liveRequestPersonIds?.length ?? 0) > 0
      const res = await fetch(`${API_BASE}/api/live-redaction/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          time_sec: requestedTime,
          reset_tracking: !!options?.resetTracking,
          include_faces: liveIncludeFaces,
          include_objects: !hasRunDetection || selectedObjectClasses.length > 0,
          person_ids: liveRequestPersonIds,
          object_classes: hasRunDetection ? selectedObjectClasses : undefined,
          forensic_only: true,
          object_confidence: 0.25,
        }),
      })

      const data = await res.json().catch(() => ({})) as {
        detections?: LiveRedactionDetection[]
        error?: string
        time_sec?: number
        object_detection_error?: string | null
      }

      const stale = requestId !== liveRedactionRequestIdRef.current
      const rawDetections = Array.isArray(data.detections) ? data.detections : []
      if (stale) return
      if (!res.ok) {
        throw new Error(data.error || `Live redaction failed (${res.status})`)
      }

      const resolvedTime = typeof data.time_sec === 'number' ? data.time_sec : requestedTime
      const filteredDetections = filterLiveDetectionsToSelections(
        rawDetections,
        {
          hasRunDetection,
          selectedFacePersonIds,
          selectedObjectClasses,
        },
      )
      setLiveRedactionDetections((previous) => {
        return stabilizeLiveDetections(previous, filteredDetections, resolvedTime)
      })
      setLiveRedactionSeekPending(false)
      setLiveRedactionError(data.object_detection_error || null)
      liveRedactionLastResolvedTimeRef.current = resolvedTime
    } catch (e) {
      console.error('[LiveBlur] Error:', e)
      if (requestId !== liveRedactionRequestIdRef.current) return
      setLiveRedactionSeekPending(false)
      setLiveRedactionError(e instanceof Error ? e.message : 'Live redaction failed')
      liveRedactionLastResolvedTimeRef.current = null
    } finally {
      if (requestId === liveRedactionRequestIdRef.current) {
        setLiveRedactionLoading(false)
      }
      liveRedactionInFlightRef.current = false
      const pendingTime = liveRedactionPendingTimeRef.current
      liveRedactionPendingTimeRef.current = null
      if (pendingTime !== null && Number.isFinite(pendingTime)) {
        window.setTimeout(() => {
          requestLiveRedactionRef.current(pendingTime, { force: true })
        }, 0)
      }
    }
  }, [effectiveStreamUrl, hasRunDetection, isPlaying, liveRedactionPreviewActive, lockedFacePersonIdsSet, resolveRedactionJobId, selectedFacePersonIds, selectedObjectClasses])

  const requestLiveRedactionRef = useRef(requestLiveRedaction)
  requestLiveRedactionRef.current = requestLiveRedaction

  const syncLiveRedactionFrame = useCallback((options?: { force?: boolean; clearDetections?: boolean; resetTracking?: boolean; time?: number }) => {
    if (!liveRedactionPreviewActive || !effectiveStreamUrl) return

    const video = videoRef.current
    const targetTime = options?.time ?? video?.currentTime ?? 0
    if (!Number.isFinite(targetTime) || targetTime < 0) return

    liveRedactionLastResolvedTimeRef.current = null

    if (options?.clearDetections) {
      setLiveRedactionDetections([])
      // Only invalidate in-flight requests when explicitly clearing (user
      // jumped timeline), not during normal playback polling.
      if (liveRedactionInFlightRef.current) {
        liveRedactionRequestIdRef.current += 1
      }
    }

    requestLiveRedactionRef.current(targetTime, {
      force: !!options?.force,
      resetTracking: !!options?.resetTracking,
    })
  }, [effectiveStreamUrl, liveRedactionPreviewActive])

  const buildCustomRegionPayload = useCallback((regions: TrackingRegion[]) => (
    regions.map((r) => ({
      id: r.id,
      x: r.x,
      y: r.y,
      width: r.width,
      height: r.height,
      effect: r.effect,
      shape: r.shape,
      anchor_sec: r.anchorTime ?? 0,
      reason: r.reason,
      tracking_mode: /\b(face|person|head)\b/i.test(r.reason || '') ? 'face' : 'generic',
    }))
  ), [])

  const exportRedacted = useCallback(async () => {
    setExportRedactError(null)
    setExportRedactLoading(true)
    setRedactionWarnings(null)
    setExportRedactProgress({ percent: 0, message: 'Starting...' })
    try {
      const jobId = await resolveRedactionJobId()
      const body: Record<string, unknown> = {
        job_id: jobId,
        detect_every_n: 1,
        use_temporal_optimization: false,
        blur_strength: blurIntensity,
        redaction_style: redactionStyle,
        export_quality: `${exportQuality}p`,
        output_height: exportQuality,
      }
      const customRegions = buildCustomRegionPayload(trackingRegions)
      if (customRegions.length > 0) {
        body.custom_regions = customRegions
      }

      // Snap-face entities have a synthetic personId (e.g. "snap_xxx_1")
      // and live only on the client; send their cropped face image so
      // the backend can compute an InsightFace encoding for per-frame
      // matching, instead of failing the export with "person_id not
      // found in unique_faces".
      const snapFaceImages: Array<{ person_id: string; label: string; image_base64: string }> = []
      const detectedFacePersonIds: string[] = []
      const personLabelMap: Record<string, string> = {}
      if (hasRunDetection) {
        for (const item of apiDetections) {
          if (item.kind !== 'face' || !item.personId) continue
          if (excludedFromRedactionIds.includes(item.id)) continue
          personLabelMap[item.personId] = item.label
          if (item.personId.startsWith('snap_')) {
            if (item.snapBase64) {
              snapFaceImages.push({
                person_id: item.personId,
                label: item.label,
                image_base64: item.snapBase64,
              })
            } else {
              // No image cached for this snap entity — keep the
              // person_id so backend reports it as unresolved and the
              // user gets a warning popup.
              detectedFacePersonIds.push(item.personId)
            }
          } else {
            detectedFacePersonIds.push(item.personId)
          }
        }
        if (detectedFacePersonIds.length > 0) {
          body.person_ids = detectedFacePersonIds
        }
        if (snapFaceImages.length > 0) {
          body.face_images = snapFaceImages
        }
        if (Object.keys(personLabelMap).length > 0) {
          body.person_labels = personLabelMap
        }
        if (selectedObjectClasses.length > 0) {
          body.object_classes = selectedObjectClasses
        }
      } else if (customRegions.length === 0) {
        body.face_encodings = ['__ALL__']
        body.object_classes = LIVE_REDACTION_OBJECT_CLASSES
      }

      const startRes = await fetch(`${API_BASE}/api/redact/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!startRes.ok) {
        const err = await startRes.json().catch(() => ({})) as {
          error?: string
          unresolved_person_ids?: RedactionWarningEntry[]
          face_blur_failures?: RedactionWarningEntry[]
        }
        if (Array.isArray(err.unresolved_person_ids) && err.unresolved_person_ids.length > 0) {
          setRedactionWarnings({
            unresolved: err.unresolved_person_ids,
            blurFailures: err.face_blur_failures || [],
            faceLockFailures: [],
          })
        }
        throw new Error(err.error || startRes.statusText)
      }
      const startData = (await startRes.json()) as { redaction_job_id: string }
      const redactionJobId = startData.redaction_job_id

      let resolved = false
      while (!resolved) {
        await new Promise((r) => setTimeout(r, 1200))
        const pollRes = await fetch(`${API_BASE}/api/redact/jobs/${redactionJobId}`)
        if (!pollRes.ok) throw new Error('Failed to check render status')
        const poll = (await pollRes.json()) as {
          status: string; percent?: number; message?: string; error?: string
          result?: {
            download_url?: string
            output_path?: string
            download_filename?: string
            download_ready?: boolean
            unresolved_person_ids?: RedactionWarningEntry[]
            face_blur_failures?: RedactionWarningEntry[]
            face_lock_failures?: RedactionWarningEntry[]
          }
        }
        setExportRedactProgress({ percent: poll.percent ?? 0, message: poll.message ?? 'Rendering...' })

        if (poll.status === 'completed') {
          resolved = true
          const result = poll.result
          if (!result?.download_url || result.download_ready === false) {
            throw new Error('Redaction completed but the MP4 is not ready for download')
          }
          const resolvedDownloadUrl = result.download_url.startsWith('http') ? result.download_url : `${API_BASE}${result.download_url}`
          const resolvedFilename = ensureMp4Filename(
            result?.download_filename || (result?.output_path ? result.output_path.split('/').pop() : null),
          )
          setExportRedactProgress({ percent: 100, message: 'Done!' })

          const unresolved = Array.isArray(result.unresolved_person_ids) ? result.unresolved_person_ids : []
          const blurFailures = Array.isArray(result.face_blur_failures) ? result.face_blur_failures : []
          const lockFailures = Array.isArray(result.face_lock_failures) ? result.face_lock_failures : []
          if (unresolved.length > 0 || blurFailures.length > 0 || lockFailures.length > 0) {
            setRedactionWarnings({
              unresolved,
              blurFailures,
              faceLockFailures: lockFailures,
            })
          }
          if (resolvedDownloadUrl) {
            await triggerFileDownload(resolvedDownloadUrl, resolvedFilename)
            setExportMenuOpen(false)
          }
        } else if (poll.status === 'failed') {
          throw new Error(poll.error || 'Redaction failed on the server')
        }
      }
    } catch (e) {
      setExportRedactError(e instanceof Error ? e.message : 'Export failed')
    } finally {
      setExportRedactLoading(false)
      setTimeout(() => setExportRedactProgress(null), 3000)
    }
  }, [apiDetections, blurIntensity, buildCustomRegionPayload, excludedFromRedactionIds, exportQuality, hasRunDetection, redactionStyle, resolveRedactionJobId, selectedObjectClasses, trackingRegions])

  useEffect(() => {
    if (!liveRedactionPreviewActive || !effectiveStreamUrl) {
      setLiveRedactionDetections([])
      setLiveRedactionLoading(false)
      setLiveRedactionSeekPending(false)
      setLiveRedactionError(null)
      liveRedactionPendingTimeRef.current = null
      liveRedactionLastResolvedTimeRef.current = null
      return
    }

    if (isPlaying) return
    requestLiveRedactionRef.current(videoRef.current?.currentTime ?? currentTime)
  }, [currentTime, effectiveStreamUrl, isPlaying, liveRedactionPreviewActive])

  useEffect(() => {
    if (!liveRedactionPreviewActive || !effectiveStreamUrl || !isPlaying) return

    const pollMs = playbackRate > 1
      ? Math.max(90, Math.round(LIVE_DETECTION_POLL_MS / playbackRate))
      : LIVE_DETECTION_POLL_MS

    requestLiveRedactionRef.current(videoRef.current?.currentTime ?? 0)
    const intervalId = window.setInterval(() => {
      requestLiveRedactionRef.current(videoRef.current?.currentTime ?? 0)
    }, pollMs)

    return () => window.clearInterval(intervalId)
  }, [effectiveStreamUrl, isPlaying, liveRedactionPreviewActive, playbackRate])

  useEffect(() => {
    if (!trackingRegions.length) {
      setTrackingPreviewByRegion({})
      setTrackingPreviewLoading(false)
      setTrackingPreviewError(null)
      return
    }
    if (drawStart || dragState) {
      return
    }

    let cancelled = false
    const controller = new AbortController()
    const timer = window.setTimeout(async () => {
      setTrackingPreviewLoading(true)
      setTrackingPreviewError(null)
      try {
        const jobId = await resolveRedactionJobId()
        if (cancelled) return
        const res = await fetch(`${API_BASE}/api/redact/preview-track`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job_id: jobId,
            preview_fps: 8,
            custom_regions: buildCustomRegionPayload(trackingRegions),
          }),
          signal: controller.signal,
        })
        if (!res.ok) {
          const err = await res.json().catch(() => ({}))
          throw new Error((err as { error?: string }).error || res.statusText)
        }
        const data = await res.json().catch(() => ({})) as { custom_tracks?: TrackingPreviewRegion[] }
        if (cancelled) return
        const next: Record<string, TrackingPreviewSample[]> = {}
        for (const region of data.custom_tracks || []) {
          if (region?.id) next[region.id] = Array.isArray(region.samples) ? region.samples : []
        }
        setTrackingPreviewByRegion(next)
      } catch (e) {
        if (controller.signal.aborted || cancelled) return
        setTrackingPreviewByRegion({})
        setTrackingPreviewError(e instanceof Error ? e.message : 'Tracking preview failed')
      } finally {
        if (!cancelled) setTrackingPreviewLoading(false)
      }
    }, 250)

    return () => {
      cancelled = true
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [buildCustomRegionPayload, dragState, drawStart, resolveRedactionJobId, trackingRegions])

  useEffect(() => {
    analyzeChatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [analyzeMessages, analyzeLoading])

  const markdownComponents = useMemo(
    () => ({
      p: ({ children }: { children?: React.ReactNode }) => (
        <p className="my-1.5 first:mt-0 last:mb-0">{withTimestampLinks(children, seekToTime)}</p>
      ),
      li: ({ children }: { children?: React.ReactNode }) => (
        <li className="my-0.5 ml-4">{withTimestampLinks(children, seekToTime)}</li>
      ),
      ul: ({ children }: { children?: React.ReactNode }) => <ul className="my-1.5 list-disc pl-2">{children}</ul>,
      ol: ({ children }: { children?: React.ReactNode }) => <ol className="my-1.5 list-decimal pl-2">{children}</ol>,
      strong: ({ children }: { children?: React.ReactNode }) => (
        <strong className="font-semibold">{withTimestampLinks(children, seekToTime)}</strong>
      ),
      blockquote: ({ children }: { children?: React.ReactNode }) => (
        <blockquote className="border-l-2 border-border pl-2 my-1.5 text-text-tertiary">
          {withTimestampLinks(children, seekToTime)}
        </blockquote>
      ),
      h1: ({ children }: { children?: React.ReactNode }) => (
        <h1 className="text-base font-semibold mt-2 mb-1">{withTimestampLinks(children, seekToTime)}</h1>
      ),
      h2: ({ children }: { children?: React.ReactNode }) => (
        <h2 className="text-sm font-semibold mt-2 mb-1">{withTimestampLinks(children, seekToTime)}</h2>
      ),
      h3: ({ children }: { children?: React.ReactNode }) => (
        <h3 className="text-sm font-medium mt-1.5 mb-0.5">{withTimestampLinks(children, seekToTime)}</h3>
      ),
      code: ({ children, className }: { children?: React.ReactNode; className?: string }) => (
        <code className={className ?? 'bg-surface px-1 py-0.5 rounded text-xs'}>{children}</code>
      ),
      pre: ({ children }: { children?: React.ReactNode }) => (
        <pre className="bg-surface rounded p-2 my-1.5 overflow-x-auto text-xs whitespace-pre-wrap break-words">
          {children}
        </pre>
      ),
      a: ({ href, children }: { href?: string; children?: React.ReactNode }) => (
        <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent underline hover:no-underline">
          {children}
        </a>
      ),
    }),
    [seekToTime]
  )

  const markdownWrapClass =
    'text-sm text-text-secondary leading-relaxed [&_*]:text-inherit [&_*]:text-sm [&_code]:bg-surface [&_code]:px-1 [&_code]:rounded [&_code]:text-xs [&_pre]:overflow-x-auto [&_pre]:text-xs'

  useEffect(() => {
    if (!exportMenuOpen) return
    const onPointerDown = (e: PointerEvent) => {
      if (exportMenuRef.current?.contains(e.target as Node)) return
      setExportMenuOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    return () => document.removeEventListener('pointerdown', onPointerDown)
  }, [exportMenuOpen])

  useEffect(() => {
    if (!tutorialOpen) return
    const onPointerDown = (e: PointerEvent) => {
      if (tutorialRef.current?.contains(e.target as Node)) return
      setTutorialOpen(false)
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setTutorialOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [tutorialOpen])

  useEffect(() => {
    if (!faceLockListOpen) return
    const onPointerDown = (e: PointerEvent) => {
      if (faceLockListRef.current?.contains(e.target as Node)) return
      setFaceLockListOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    return () => document.removeEventListener('pointerdown', onPointerDown)
  }, [faceLockListOpen])

  /* ---- Tracker: viewport coords (0–1) ---- */
  const getNormFromEvent = useCallback((e: React.MouseEvent | MouseEvent) => {
    const el = videoContainerRef.current
    if (!el) return { x: 0, y: 0, inside: false }
    const rect = el.getBoundingClientRect()
    const viewport = videoViewport.width > 0 && videoViewport.height > 0
      ? videoViewport
      : { left: 0, top: 0, width: rect.width, height: rect.height }
    const relX = e.clientX - rect.left - viewport.left
    const relY = e.clientY - rect.top - viewport.top
    const inside = relX >= 0 && relY >= 0 && relX <= viewport.width && relY <= viewport.height
    const x = viewport.width > 0 ? relX / viewport.width : 0
    const y = viewport.height > 0 ? relY / viewport.height : 0
    return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)), inside }
  }, [videoViewport])

  const resolveTrackingPreviewSampleAtTime = useCallback((samples: TrackingPreviewSample[], t: number): TrackingPreviewSample | null => {
    if (!samples.length) return null
    let lo = 0
    let hi = samples.length - 1
    let best = -1
    while (lo <= hi) {
      const mid = Math.floor((lo + hi) / 2)
      if (samples[mid].t <= t + 1e-4) {
        best = mid
        lo = mid + 1
      } else {
        hi = mid - 1
      }
    }
    if (best < 0) return samples[0] ?? null
    return samples[best] ?? null
  }, [])

  const resolveDisplayedTrackingRegion = useCallback((regionId: string, timeSec: number) => {
    const region = trackingRegions.find((r) => r.id === regionId)
    if (!region) return null
    const anchorTime = region.anchorTime ?? 0
    if (timeSec + 0.05 < anchorTime) return null
    const sample = resolveTrackingPreviewSampleAtTime(trackingPreviewByRegion[regionId] || [], timeSec)
    if (!sample) return region
    return {
      ...region,
      x: sample.x,
      y: sample.y,
      width: sample.width,
      height: sample.height,
    }
  }, [resolveTrackingPreviewSampleAtTime, trackingPreviewByRegion, trackingRegions])

  const addTrackingRegion = useCallback((region: Omit<TrackingRegion, 'id'>) => {
    const id = `region-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    setTrackingRegions((prev) => [...prev, { ...region, id }])
    setSelectedRegionId(id)
    setDrawMode(false)
    setDrawStart(null)
    setDrawCurrent(null)
  }, [])
  trackerSettingsRef.current = { trackerShape, trackerEffect, trackerReason, addTrackingRegion }

  const updateTrackingRegion = useCallback((id: string, patch: Partial<TrackingRegion>) => {
    setTrackingRegions((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)))
  }, [])

  const removeTrackingRegion = useCallback((id: string) => {
    setTrackingRegions((prev) => prev.filter((r) => r.id !== id))
    if (selectedRegionId === id) setSelectedRegionId(null)
  }, [selectedRegionId])

  const startDraw = useCallback(
    (e: React.MouseEvent) => {
      if (activeTool !== 'tracker' || !drawMode || e.button !== 0) return
      e.preventDefault()
      e.stopPropagation()
      const { x, y, inside } = getNormFromEvent(e)
      if (!inside) return
      setDrawStart({ x, y })
      setDrawCurrent({ x, y })
    },
    [activeTool, drawMode, getNormFromEvent]
  )

  const updateDraw = useCallback(
    (e: MouseEvent) => {
      if (!drawStart) return
      const el = videoContainerRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      const viewport = videoViewport.width > 0 && videoViewport.height > 0
        ? videoViewport
        : { left: 0, top: 0, width: rect.width, height: rect.height }
      const x = Math.max(0, Math.min(1, (e.clientX - rect.left - viewport.left) / viewport.width))
      const y = Math.max(0, Math.min(1, (e.clientY - rect.top - viewport.top) / viewport.height))
      setDrawCurrent({ x, y })
    },
    [drawStart, videoViewport]
  )

  const endDraw = useCallback((e?: React.MouseEvent) => {
    if (e && e.button !== 0) return
    const { drawStart: start, drawCurrent: cur } = drawStateRef.current
    if (!start || !cur) {
      setDrawStart(null)
      setDrawCurrent(null)
      return
    }
    let x = Math.min(start.x, cur.x)
    let y = Math.min(start.y, cur.y)
    let width = Math.abs(cur.x - start.x)
    let height = Math.abs(cur.y - start.y)
    const settings = trackerSettingsRef.current
    if (settings?.trackerShape === 'circle' && width >= 0.02 && height >= 0.02) {
      const size = Math.min(width, height)
      x = x + (width - size) / 2
      y = y + (height - size) / 2
      width = size
      height = size
    }
    if (width >= 0.02 && height >= 0.02 && settings) {
      settings.addTrackingRegion({
        shape: settings.trackerShape,
        x,
        y,
        width,
        height,
        effect: settings.trackerEffect,
        anchorTime: getPlaybackAnchorTime(),
        reason: settings.trackerReason || undefined,
        locked: false,
      })
    }
    setDrawStart(null)
    setDrawCurrent(null)
  }, [getPlaybackAnchorTime])

  const startDrag = useCallback(
    (e: React.MouseEvent, regionId: string) => {
      if (e.button !== 0) return
      e.preventDefault()
      e.stopPropagation()
      const region = resolveDisplayedTrackingRegion(regionId, videoRef.current?.currentTime ?? currentTime)
      if (!region) return
      setSelectedRegionId(regionId)
      const { x, y } = getNormFromEvent(e)
      setDragState({ regionId, offsetX: x - region.x, offsetY: y - region.y })
    },
    [currentTime, getNormFromEvent, resolveDisplayedTrackingRegion]
  )

  const updateDrag = useCallback(
    (e: MouseEvent) => {
      if (!dragState) return
      const region = resolveDisplayedTrackingRegion(dragState.regionId, videoRef.current?.currentTime ?? currentTime)
      if (!region || region.locked) return
      const { x, y } = getNormFromEvent(e)
      const newX = Math.max(0, Math.min(1 - region.width, x - dragState.offsetX))
      const newY = Math.max(0, Math.min(1 - region.height, y - dragState.offsetY))
      updateTrackingRegion(dragState.regionId, {
        x: newX,
        y: newY,
        width: region.width,
        height: region.height,
        anchorTime: getPlaybackAnchorTime(),
      })
    },
    [currentTime, dragState, getNormFromEvent, getPlaybackAnchorTime, resolveDisplayedTrackingRegion, updateTrackingRegion]
  )

  const endDrag = useCallback(() => setDragState(null), [])

  useEffect(() => {
    if (!drawStart) return
    const onMouseUp = () => endDraw()
    window.addEventListener('mousemove', updateDraw)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      window.removeEventListener('mousemove', updateDraw)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [drawStart, updateDraw, endDraw])

  useEffect(() => {
    if (!dragState) return
    const onMove = (e: MouseEvent) => updateDrag(e)
    const onUp = () => endDrag()
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [dragState, updateDrag, endDrag])

  /* ---- Keyboard shortcuts ---- */

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const targetTag = (e.target as HTMLElement).tagName
      if (targetTag === 'INPUT' || targetTag === 'SELECT' || targetTag === 'TEXTAREA' || targetTag === 'BUTTON') return
      switch (e.key) {
        case ' ': e.preventDefault(); togglePlay(); break
        case 'ArrowLeft': e.preventDefault(); skip(-5); break
        case 'ArrowRight': e.preventDefault(); skip(5); break
        case 'm': toggleMute(); break
        case 'f': toggleFullscreen(); break
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [togglePlay, skip, toggleMute, toggleFullscreen])

  /* ---- Timeline: progress and ruler ---- */

  const progress = duration > 0 ? currentTime / duration : 0
  const bufferedPct = duration > 0 ? buffered / duration : 0
  const timelineContentWidthPct = timelineZoom * 100
  const audioWaveformDisplayData = useMemo(() => {
    if (!SHOW_AUDIO_WAVEFORM) return []
    if (audioWaveformData.length > 0) return audioWaveformData
    if (audioWaveformStatus === 'loading') {
      return generateWaveformPlaceholder(Math.min(160, Math.max(56, Math.floor(Math.max(duration, 12) * 2.5))))
    }
    return []
  }, [audioWaveformData, audioWaveformStatus, duration])
  const audioWaveformPath = useMemo(
    () => buildWaveformPath(audioWaveformDisplayData, 1200, 84, 8),
    [audioWaveformDisplayData],
  )
  const audioWaveformMeta = useMemo(() => {
    if (!SHOW_AUDIO_WAVEFORM) return 'Audio lane hidden'
    if (audioWaveformStatus === 'ready') return 'Waveform ready'
    if (audioWaveformStatus === 'loading') return 'Building waveform'
    if (audioWaveformStatus === 'unavailable') return 'Waveform unavailable'
    return 'Preparing audio lane'
  }, [audioWaveformStatus])

  const majorStep = useMemo(() => {
    if (duration <= 0) return 30
    const pixelsPerSec = (timelineZoom * 800) / Math.max(duration, 1)
    if (pixelsPerSec > 15) return 5
    if (pixelsPerSec > 6) return 10
    if (pixelsPerSec > 3) return 30
    return 60
  }, [duration, timelineZoom])

  const minorStep = majorStep <= 5 ? 1 : majorStep <= 10 ? 5 : majorStep <= 30 ? 10 : 30

  const majorTicks = useMemo(() => {
    if (duration <= 0) return [0]
    const arr: number[] = []
    for (let s = 0; s <= duration; s += majorStep) arr.push(s)
    return arr
  }, [duration, majorStep])

  const minorTicks = useMemo(() => {
    if (duration <= 0 || minorStep >= majorStep) return []
    const arr: number[] = []
    for (let s = 0; s <= duration; s += minorStep) {
      if (s % majorStep !== 0) arr.push(s)
    }
    return arr
  }, [duration, minorStep, majorStep])

  const detectionList = hasRunDetection ? apiDetections : DUMMY_DETECTIONS
  const anonymizeTargetCount = useMemo(
    () => detectionList.filter(isAnonymizeTarget).length,
    [detectionList],
  )
  const filteredDetections = useMemo((): DetectionItem[] => {
    const q = detectionFilter.trim().toLowerCase()
    const detectionScope = showAnonymizeOnly
      ? detectionList.filter(isAnonymizeTarget)
      : detectionList
    const matchingItems = !q
      ? detectionScope
      : detectionScope.filter(
        (d) =>
          d.label.toLowerCase().includes(q) || d.tags.some((t: string) => t.toLowerCase().includes(q))
      )

    return [...matchingItems].sort((a, b) => {
      const activeDifference = Number(excludedFromRedactionIds.includes(a.id)) - Number(excludedFromRedactionIds.includes(b.id))
      if (activeDifference !== 0) return activeDifference

      const visibleDifference = Number(isDetectionItemLikelyVisibleAtTime(b, currentTime)) - Number(isDetectionItemLikelyVisibleAtTime(a, currentTime))
      if (visibleDifference !== 0) return visibleDifference

      const proximityDifference = getDetectionItemNearestGapSec(a, currentTime) - getDetectionItemNearestGapSec(b, currentTime)
      if (Math.abs(proximityDifference) > 0.001) return proximityDifference

      return a.label.localeCompare(b.label)
    })
  }, [currentTime, detectionFilter, detectionList, excludedFromRedactionIds, showAnonymizeOnly])
  const searchTimelineLanes = useMemo(() => {
    if (!videoId) return []

    return searchSessionResults.flatMap((session, sessionIndex) => {
      const sessionId = getSearchSessionId(session, sessionIndex)
      const result = session.results.find((entry) => entry?.id === videoId || entry?.video_id === videoId)
      if (!result) return []

      const clips = Array.isArray(result.clips)
        ? result.clips
          .filter((clip): clip is SearchClip => (
            !!clip &&
            Number.isFinite(clip.start) &&
            Number.isFinite(clip.end)
          ))
          .map((clip) => ({
            ...clip,
            score: typeof clip.score === 'number' && Number.isFinite(clip.score) ? clip.score : 0,
            rank: typeof clip.rank === 'number' && Number.isFinite(clip.rank) ? clip.rank : undefined,
            thumbnailUrl: clip.thumbnailUrl || clip.thumbnail_url || undefined,
          }))
        : []
      const orderedClips = [...clips].sort((a, b) => {
        if (a.rank != null && b.rank != null && a.rank !== b.rank) return a.rank - b.rank
        if (a.rank != null && b.rank == null) return -1
        if (a.rank == null && b.rank != null) return 1
        if ((b.score ?? 0) !== (a.score ?? 0)) return (b.score ?? 0) - (a.score ?? 0)
        return a.start - b.start
      })
      if (orderedClips.length === 0) return []

      const rankedClips = orderedClips.filter((clip) => clip.rank != null)
      const maxRank = rankedClips.length > 0
        ? Math.max(...rankedClips.map((clip) => clip.rank as number))
        : 0
      const waveformData = duration > 0
        ? buildSearchWaveformSamples(orderedClips, duration, Math.min(320, Math.max(96, Math.floor(duration * 4))))
        : []
      const barCount = Math.min(144, Math.max(72, Math.floor(Math.max(duration, 12) * 1.8)))
      const series = waveformData.length > 0 ? buildTimelineBarSeries(waveformData, barCount) : []
      const maxValue = series.reduce((acc, value) => Math.max(acc, value), 0)
      const peakCutoff = maxValue > 0.05 ? maxValue * ENTITY_SEARCH_LANE_PEAK_THRESHOLD : Infinity
      const bars = series.map((value, index) => ({
        value,
        x: (index / Math.max(1, barCount)) * 1200,
        width: Math.max(4, (1200 / Math.max(1, barCount)) - 1.6),
        isPeak: value >= peakCutoff,
      }))
      const activeClipIndex = orderedClips.findIndex((clip) => currentTime >= clip.start && currentTime <= clip.end)
      const segments = orderedClips.map((clip, index) => {
        const clipStartPct = duration > 0 ? (clip.start / duration) * 100 : 0
        const clipEndPct = duration > 0 ? (clip.end / duration) * 100 : 0
        const clipWidthPct = Math.max(1.5, clipEndPct - clipStartPct)
        return {
          clip,
          index,
          isActive: activeClipIndex === index,
          importance: getSearchClipImportance(clip, maxRank),
          left: `${Math.max(0, Math.min(100, clipStartPct))}%`,
          width: `${Math.max(clipWidthPct, 2)}%`,
        }
      })
      const entities = (session.entities || []).filter((entity) => entity?.id && entity?.name)
      const noteLines = entities.length === 1
        ? [
          `This lane visualizes where the entity search for ${entities[0].name} matches this video over time.`,
          'Taller bars indicate a stronger entity match and usually mean that person is more clearly visible or more prominently present in that moment, while shorter bars suggest a weaker or briefer appearance.',
        ]
        : entities.length > 1
          ? [
            'This lane visualizes where the selected entity search matches this video over time.',
            'Taller bars indicate stronger identity matches and usually mean one or more selected people are more clearly visible in that segment, while shorter bars suggest a weaker or less prominent appearance.',
          ]
          : [
            'This lane visualizes where your normal search matches this video over time.',
            'Taller bars indicate stronger relevance to your search and a more prominent match in that segment, while shorter bars indicate weaker or less certain matches.',
          ]

      return [{
        session,
        sessionId,
        result: { ...result, clips },
        orderedClips,
        bars,
        segments,
        activeClipIndex,
        activeRank: activeClipIndex >= 0 ? (orderedClips[activeClipIndex]?.rank ?? activeClipIndex + 1) : null,
        entities,
        visibleEntities: entities.slice(0, 3),
        hiddenEntityCount: Math.max(0, entities.length - 3),
        noteLines,
      }]
    })
  }, [currentTime, duration, searchSessionResults, videoId])
  const activeSearchTimelineLane = useMemo(() => (
    searchTimelineLanes.find((lane) => lane.sessionId === activeSearchSessionId) ||
    searchTimelineLanes[searchTimelineLanes.length - 1] ||
    null
  ), [activeSearchSessionId, searchTimelineLanes])
  const searchResultForVideo = activeSearchTimelineLane?.result || null
  const orderedSearchClips = activeSearchTimelineLane?.orderedClips || []
  const activeSearchClipIndex = activeSearchTimelineLane?.activeClipIndex ?? -1
  const activeSearchRank = activeSearchTimelineLane?.activeRank ?? null
  const searchEntities = activeSearchTimelineLane?.entities || []
  const pegasusEvents = useMemo(() => (
    (pegasusResult?.timeline_events || [])
      .filter((event) => Number.isFinite(event.start_sec) && Number.isFinite(event.end_sec))
      .map((event) => ({
        ...event,
        start_sec: Math.max(0, event.start_sec),
        end_sec: Math.max(event.start_sec, event.end_sec),
        severity: (['low', 'medium', 'high'].includes(event.severity) ? event.severity : 'medium') as PegasusSeverity,
      }))
      .sort((a, b) => a.start_sec - b.start_sec)
  ), [pegasusResult])
  const pegasusCategories = useMemo(() => {
    const categories = Array.from(new Set(pegasusEvents.map((event) => event.category)))
    return categories.sort((a, b) => formatPegasusLabel(a).localeCompare(formatPegasusLabel(b)))
  }, [pegasusEvents])
  const filteredPegasusEvents = useMemo(() => (
    pegasusEvents.filter((event) => (
      pegasusSeverityFilter[event.severity] &&
      (pegasusCategoryFilter === 'all' || event.category === pegasusCategoryFilter)
    ))
  ), [pegasusCategoryFilter, pegasusEvents, pegasusSeverityFilter])
  const focusedPegasusEvent = useMemo(() => (
    pegasusEvents.find((event) => event.id === pegasusFocusedEventId) || filteredPegasusEvents[0] || null
  ), [filteredPegasusEvents, pegasusEvents, pegasusFocusedEventId])
  const pegasusEventsByGroup = useMemo(() => {
    const groups = {
      high: [] as PegasusTimelineEvent[],
      people: [] as PegasusTimelineEvent[],
      sensitive: [] as PegasusTimelineEvent[],
      objects: [] as PegasusTimelineEvent[],
    }
    for (const event of filteredPegasusEvents) {
      if (event.severity === 'high') groups.high.push(event)
      if (event.category === 'person' || event.category === 'face') {
        groups.people.push(event)
      } else if (['screen', 'document', 'text', 'license_plate'].includes(event.category)) {
        groups.sensitive.push(event)
      } else {
        groups.objects.push(event)
      }
    }
    return groups
  }, [filteredPegasusEvents])
  const pegasusActionById = useMemo(() => {
    const map: Record<string, PegasusRecommendedAction> = {}
    for (const action of pegasusResult?.recommended_actions || []) {
      if (action.id) map[action.id] = action
    }
    return map
  }, [pegasusResult])
  const pegasusRiskLevel = (pegasusResult?.summary?.privacy_risk_level || 'low').toString().toLowerCase()
  const pegasusHighEventCount = useMemo(
    () => pegasusEvents.filter((event) => event.severity === 'high').length,
    [pegasusEvents],
  )
  const pegasusReviewOnlyCount = useMemo(
    () => (pegasusResult?.recommended_actions || []).filter((action) => action.apply_mode === 'review_only').length,
    [pegasusResult],
  )
  const pegasusAutomaticActionCount = useMemo(
    () => (pegasusResult?.recommended_actions || []).filter((action) => action.apply_mode === 'automatic_if_matched').length,
    [pegasusResult],
  )
  const activateSearchSession = useCallback((sessionId: string) => {
    setActiveSearchSessionId(sessionId)
    setActiveTool('search')
    setRightSidebarOpen(true)
  }, [])
  const clearSearchTimelineLane = useCallback((sessionId?: string) => {
    setSearchSessionResults((previous) => {
      const targetId = sessionId || activeSearchSessionId
      if (!targetId) return previous
      const next = previous.filter((session, index) => getSearchSessionId(session, index) !== targetId)
      persistEditorSearchSessions(next)
      if (activeSearchSessionId === targetId) {
        setActiveSearchSessionId(next.length > 0 ? getSearchSessionId(next[next.length - 1]) : null)
      }
      return next
    })
  }, [activeSearchSessionId])
  const clearAllSearchTimelineLanes = useCallback(() => {
    setSearchSessionResults([])
    setActiveSearchSessionId(null)
    persistEditorSearchSessions([])
  }, [])
  // [FACE-LOCK LANE BLUR — TEMPORARILY DISABLED]
  // The face-lock lane is still built and cached on the backend, but
  // the live preview does NOT draw blur from it for now. We fall back
  // to the existing per-frame /api/live-redaction/detect path for all
  // selected persons. Re-enable by uncommenting the blocks below.
  //
  // const faceLockOverlayDetections = useMemo<LiveRedactionDetection[]>(() => {
  //   const lockedIds = lockedFacePersonIds
  //   if (lockedIds.length === 0) return []
  //   const playbackTime = videoRef.current?.currentTime ?? currentTime
  //   const out: LiveRedactionDetection[] = []
  //   for (const personId of lockedIds) {
  //     const lane = faceLockLanesByPersonId[personId]
  //     if (!lane) continue
  //     const bbox = interpolateFaceLockLane(lane, playbackTime)
  //     if (!bbox) continue
  //     const item = faceDetectionItemsByPersonId[personId]
  //     const label = item?.label || personId
  //     out.push(laneBboxToLiveDetection(lane, bbox, personId, label))
  //   }
  //   return out
  // }, [currentTime, faceDetectionItemsByPersonId, faceLockLanesByPersonId, lockedFacePersonIds])

  const visibleLiveRedactionDetections = useMemo(() => {
    const now = Date.now()
    const livePart = liveRedactionDetections.filter((detection) => {
      if (detection.width <= 0 || detection.height <= 0) return false
      if (detection.lastSeenAtMs && now - detection.lastSeenAtMs > getLiveDetectionHoldMs(detection)) return false
      // [FACE-LOCK LANE BLUR — DISABLED] when re-enabled, also drop live
      // face detections for lane-locked persons so the lane bbox is the
      // only source of truth for them:
      // if (detection.kind === 'face' && detection.personId && lockedFacePersonIdsSet.has(detection.personId)) return false
      return true
    })
    // [FACE-LOCK LANE BLUR — DISABLED] when re-enabled, append the
    // lane-derived detections here:
    //   return [...livePart, ...faceLockOverlayDetections]
    return livePart
  }, [liveRedactionDetections])
  const renderedLiveRedactionDetections = useMemo(() => {
    const playbackTime = videoRef.current?.currentTime ?? currentTime
    return visibleLiveRedactionDetections.map((entry) => (
      entry.id.startsWith('face-lock-')
        ? entry
        : predictLiveDetection(entry, playbackTime)
    ))
  }, [currentTime, visibleLiveRedactionDetections])
  const domVideoBlurSupported = supportsLiveBackdropBlur()
  const useDomVideoBlurOverlay = (
    liveRedactionOverlayVisible &&
    redactionStyle === 'blur' &&
    renderedLiveRedactionDetections.length > 0 &&
    domVideoBlurSupported &&
    !useHls
  )
  const liveBlurOverlayVideoStyle = useMemo<React.CSSProperties>(() => ({
    filter: `blur(${getLiveRedactionBlurPx(blurIntensity)}px) saturate(0.72)`,
    clipPath: `url(#${liveBlurClipPathId})`,
    WebkitClipPath: `url(#${liveBlurClipPathId})`,
    transform: 'translateZ(0) scale(1.03)',
    transformOrigin: '50% 50%',
  }), [blurIntensity, liveBlurClipPathId])
  const showLiveRedactionSeekLoader = liveRedactionPreviewActive && liveRedactionSeekPending && (!isPlaying || isScrubbing)
  const livePreviewModeLabel = liveRedactionEnabled ? 'Live blur' : 'Selection preview'
  useEffect(() => {
    visibleLiveRedactionDetectionsRef.current = visibleLiveRedactionDetections
  }, [visibleLiveRedactionDetections])

  useEffect(() => {
    lockedFacePersonIdsRef.current = lockedFacePersonIds
  }, [lockedFacePersonIds])

  useEffect(() => {
    if (!showFaceLockIntroPopup) return
    const timer = window.setTimeout(() => setShowFaceLockIntroPopup(false), 8000)
    return () => window.clearTimeout(timer)
  }, [showFaceLockIntroPopup])

  useEffect(() => {
    const next: Record<string, string> = {}
    for (const item of apiDetections) {
      if (item.kind === 'face' && item.personId) {
        next[item.personId] = item.label || item.personId
      }
    }
    faceLockLabelByPersonIdRef.current = next
  }, [apiDetections])

  useEffect(() => {
    const mainVideo = videoRef.current
    const overlayVideo = liveBlurOverlayVideoRef.current
    if (!mainVideo || !overlayVideo || !useDomVideoBlurOverlay) return

    let cancelled = false
    let frameId: number | null = null

    const syncOverlay = () => {
      if (cancelled) return

      overlayVideo.defaultMuted = true
      overlayVideo.muted = true
      overlayVideo.volume = 0
      overlayVideo.playbackRate = mainVideo.playbackRate || 1

      const mainTime = mainVideo.currentTime
      const driftThreshold = mainVideo.paused || mainVideo.seeking ? 0.02 : 0.08
      if (
        Number.isFinite(mainTime) &&
        (!Number.isFinite(overlayVideo.currentTime) || Math.abs(overlayVideo.currentTime - mainTime) > driftThreshold)
      ) {
        try {
          overlayVideo.currentTime = mainTime
        } catch {
          /* ignore currentTime sync failures while media is still warming up */
        }
      }

      if (mainVideo.paused || mainVideo.ended || mainVideo.seeking) {
        if (!overlayVideo.paused) {
          overlayVideo.pause()
        }
      } else if (overlayVideo.readyState >= 2) {
        overlayVideo.play().catch(() => {})
      }

      frameId = window.requestAnimationFrame(syncOverlay)
    }

    syncOverlay()
    return () => {
      cancelled = true
      if (frameId !== null) {
        window.cancelAnimationFrame(frameId)
      }
      overlayVideo.pause()
    }
  }, [useDomVideoBlurOverlay])

  /* Strand shared classes */
  const btnBase = 'inline-flex items-center justify-center rounded-md border border-border bg-surface text-text-primary hover:bg-card transition-colors'
  const fallbackViewport = {
    left: 0,
    top: 0,
    width: videoContainerRef.current?.clientWidth ?? 0,
    height: videoContainerRef.current?.clientHeight ?? 0,
  }
  const overlayViewport = videoViewport.width > 0 && videoViewport.height > 0
    ? videoViewport
    : fallbackViewport
  const overlayFrameStyle: React.CSSProperties =
    overlayViewport.width > 0 && overlayViewport.height > 0
      ? {
          left: overlayViewport.left,
          top: overlayViewport.top,
          width: overlayViewport.width,
          height: overlayViewport.height,
        }
      : {
          inset: 0,
        }
  const fittedVideoFrameStyle = useMemo<React.CSSProperties>(() => {
    const availableWidth = videoStageSize.width
    const availableHeight = videoStageSize.height
    if (availableWidth <= 0 || availableHeight <= 0 || playerAspectRatio <= 0) {
      return {
        aspectRatio: playerAspectRatio > 0 ? playerAspectRatio : 16 / 9,
        width: '100%',
        maxWidth: '100%',
        maxHeight: '100%',
      }
    }

    const fullWidthHeight = availableWidth / playerAspectRatio
    if (fullWidthHeight <= availableHeight) {
      return {
        width: availableWidth,
        height: fullWidthHeight,
        maxWidth: '100%',
        maxHeight: '100%',
      }
    }

    return {
      width: availableHeight * playerAspectRatio,
      height: availableHeight,
      maxWidth: '100%',
      maxHeight: '100%',
    }
  }, [playerAspectRatio, videoStageSize.height, videoStageSize.width])
  useEffect(() => {
    if (isPlaying) return

    const canvas = liveBlurCanvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = Math.max(0, overlayViewport.width)
    const height = Math.max(0, overlayViewport.height)
    ctx.clearRect(0, 0, width, height)

    if (
      useDomVideoBlurOverlay ||
      !liveRedactionOverlayVisible ||
      width <= 0 ||
      height <= 0 ||
      video.readyState < 2 ||
      visibleLiveRedactionDetections.length === 0
    ) {
      return
    }

    const playbackTime = video.currentTime ?? currentTime
    for (const detection of visibleLiveRedactionDetections.map((entry) => predictLiveDetection(entry, playbackTime))) {
      drawCanvasBlurRegion(ctx, video, detection, width, height, redactionStyle, blurIntensity)
    }
  }, [
    blurIntensity,
    currentTime,
    isPlaying,
    liveRedactionOverlayVisible,
    overlayViewport.height,
    overlayViewport.width,
    redactionStyle,
    useDomVideoBlurOverlay,
    visibleLiveRedactionDetections,
  ])
  useLayoutEffect(() => {
    const canvas = liveBlurCanvasRef.current
    if (!canvas || overlayViewport.width <= 0 || overlayViewport.height <= 0) return
    const dpr = window.devicePixelRatio || 1
    const width = Math.max(1, Math.round(overlayViewport.width))
    const height = Math.max(1, Math.round(overlayViewport.height))
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(height * dpr)
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)
  }, [overlayViewport.height, overlayViewport.width])

  useEffect(() => {
    const canvas = liveBlurCanvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let cancelled = false
    // [FACE-LOCK LANE BLUR — TEMPORARILY DISABLED]
    // When re-enabled, this re-computes lane-derived bboxes every
    // animation frame from the precomputed lane + video.currentTime,
    // and draws them on top of the live-detection blurs.
    //
    // const computeLiveLaneDetections = (playbackTime: number): LiveRedactionDetection[] => {
    //   const ids = lockedFacePersonIdsRef.current
    //   if (!ids || ids.length === 0) return []
    //   const lanes = faceLockLanesByPersonIdRef.current
    //   const labels = faceLockLabelByPersonIdRef.current
    //   const out: LiveRedactionDetection[] = []
    //   for (const personId of ids) {
    //     const lane = lanes[personId]
    //     if (!lane) continue
    //     const bbox = interpolateFaceLockLane(lane, playbackTime)
    //     if (!bbox) continue
    //     out.push(laneBboxToLiveDetection(lane, bbox, personId, labels[personId] || personId))
    //   }
    //   return out
    // }
    const drawFrame = (playbackTime: number) => {
      if (cancelled) return
      const width = Math.max(0, overlayViewport.width)
      const height = Math.max(0, overlayViewport.height)
      ctx.clearRect(0, 0, width, height)
      if (
        !useDomVideoBlurOverlay &&
        liveRedactionOverlayVisible &&
        width > 0 &&
        height > 0 &&
        video.readyState >= 2
      ) {
        // [FACE-LOCK LANE BLUR — DISABLED] only the existing per-frame
        // live-redaction blurs are drawn. To re-enable, restore:
        //   const laneDetections = computeLiveLaneDetections(playbackTime)
        //   ...and draw them after the liveDetections loop below.
        const liveDetections = visibleLiveRedactionDetectionsRef.current
          .filter((entry) => !entry.id.startsWith('face-lock-'))
          .map((entry) => predictLiveDetection(entry, playbackTime))
        if (liveDetections.length === 0) return
        try {
          for (const detection of liveDetections) {
            drawCanvasBlurRegion(ctx, video, detection, width, height, redactionStyle, blurIntensity)
          }
        } catch {
          // Keep the loop alive even if one frame draw fails.
        }
      }
    }

    const scheduleNextFrame = () => {
      if (cancelled) return

      if (isPlaying) {
        liveBlurAnimationFrameRef.current = window.requestAnimationFrame(() => {
          drawFrame(video.currentTime ?? 0)
          scheduleNextFrame()
        })
        return
      }

      drawFrame(video.currentTime ?? 0)
    }

    scheduleNextFrame()
    return () => {
      cancelled = true
      if (liveBlurAnimationFrameRef.current !== null) {
        window.cancelAnimationFrame(liveBlurAnimationFrameRef.current)
        liveBlurAnimationFrameRef.current = null
      }
      liveBlurVideoFrameCallbackRef.current = null
      ctx.clearRect(0, 0, Math.max(0, overlayViewport.width), Math.max(0, overlayViewport.height))
    }
  }, [blurIntensity, isPlaying, liveRedactionOverlayVisible, overlayViewport.height, overlayViewport.width, redactionStyle, useDomVideoBlurOverlay])

  const syncLiveRedactionFrameRef = useRef(syncLiveRedactionFrame)
  syncLiveRedactionFrameRef.current = syncLiveRedactionFrame

  useEffect(() => {
    if (!liveRedactionPreviewActive || !hasRunDetection) return

    setLiveRedactionDetections((previous) =>
      previous.filter((detection) => {
        const selectionId = getSelectionIdForLiveDetection(detection)
        return selectionId ? !excludedFromRedactionIds.includes(selectionId) : true
      })
    )

    const timer = window.setTimeout(() => {
      syncLiveRedactionFrameRef.current({ force: true, clearDetections: true, resetTracking: true })
    }, 0)

    return () => window.clearTimeout(timer)
  }, [excludedFromRedactionIds, hasRunDetection, liveRedactionPreviewActive])

  const toggleDetectionSelectionById = useCallback((selectionId: string, personId?: string | null) => {
    const isExcluded = excludedFromRedactionIds.includes(selectionId)
    const isActivatingFaceBlur = isExcluded && !!personId
    if (isActivatingFaceBlur) {
      ensureFaceTimelineLane(personId)
      try {
        const seen = window.localStorage.getItem(FACE_LOCK_INTRO_SEEN_KEY)
        if (seen !== '1') {
          setShowFaceLockIntroPopup(true)
          window.localStorage.setItem(FACE_LOCK_INTRO_SEEN_KEY, '1')
        }
      } catch {
        // localStorage might be unavailable (private mode); no-op.
      }
    }
    setExcludedFromRedactionIds((previous) => (
      previous.includes(selectionId)
        ? previous.filter((id) => id !== selectionId)
        : [...previous, selectionId]
    ))
  }, [ensureFaceTimelineLane, excludedFromRedactionIds])

  const handleDetectionListToggle = useCallback((item: DetectionItem) => {
    const isExcluded = excludedFromRedactionIds.includes(item.id)
    const isActivatingBlur = isExcluded
    const wasPlaying = !!videoRef.current && !videoRef.current.paused
    if (!isActivatingBlur && item.kind === 'face' && item.personId) {
      removeFaceTimelineLane(item.personId)
    }
    toggleDetectionSelectionById(item.id, item.kind === 'face' ? item.personId : null)

    if (isActivatingBlur && videoId) {
      const showScopedSearchResult = (scopedSearchResult: SearchSessionResult) => {
        const nextSession = withSearchSessionId(scopedSearchResult)
        const nextSessionId = getSearchSessionId(nextSession)
        setSearchSessionResults((previous) => {
          const next = [
            ...previous.filter((session, index) => getSearchSessionId(session, index) !== nextSessionId),
            nextSession,
          ]
          persistEditorSearchSessions(next)
          return next
        })
        setActiveSearchSessionId(nextSessionId)
        setActiveTool('search')
        setRightSidebarOpen(true)
      }

      const fallbackToEntityRanges = () => {
        if (item.kind !== 'face' || !item.entityId) return
        fetch(`${API_BASE}/api/entities/${encodeURIComponent(item.entityId)}/time-ranges`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ video_id: videoId }),
        })
          .then(async (res) => {
            const data = await res.json().catch(() => ({})) as { time_ranges?: DetectionTimeRange[] }
            if (!res.ok) return
            const scopedSegments = buildFaceTimelineSegments(item, data.time_ranges || [])
            if (scopedSegments.length === 0) return
            showScopedSearchResult({
              query: `Entity: ${item.label}`,
              queryText: item.label,
              entities: [{
                id: item.entityId as string,
                name: item.label,
                previewUrl: item.snapBase64 ? `data:image/png;base64,${item.snapBase64}` : undefined,
              }],
              results: [{
                id: videoId,
                video_id: videoId,
                title: item.label,
                clips: scopedSegments.map((segment, index) => ({
                  start: segment.start,
                  end: segment.end,
                  rank: index + 1,
                })),
              }],
            })
          })
          .catch(() => {})
      }

      if (item.snapBase64) {
        fetch(`data:image/png;base64,${item.snapBase64}`)
          .then((response) => response.blob())
          .then((blob) => {
            const formData = new FormData()
            formData.append('query', item.label)
            formData.append('image', blob, `${item.id || 'selection'}.png`)
            return fetch(`${API_BASE}/api/search`, {
              method: 'POST',
              body: formData,
            })
          })
          .then(async (res) => {
            const payload = await res.json().catch(() => ({})) as { results?: SearchVideoResult[] }
            if (!res.ok) {
              fallbackToEntityRanges()
              return
            }
            const videoResult = (payload.results || []).find((entry) => (
              entry?.id === videoId || entry?.video_id === videoId
            ))
            const scopedClips = Array.isArray(videoResult?.clips)
              ? videoResult.clips.filter((clip) => (
                !!clip &&
                Number.isFinite(clip.start) &&
                Number.isFinite(clip.end)
              ))
              : []
            if (!videoResult || scopedClips.length === 0) {
              fallbackToEntityRanges()
              return
            }
            showScopedSearchResult({
              query: `Image search: ${item.label}`,
              queryText: item.label,
              entities: [{
                id: item.entityId || item.id,
                name: item.label,
                previewUrl: `data:image/png;base64,${item.snapBase64}`,
              }],
              results: [{
                ...videoResult,
                id: videoId,
                video_id: videoId,
                title: videoResult.title || item.label,
                clips: scopedClips,
              }],
            })
          })
          .catch(() => {
            fallbackToEntityRanges()
          })
      } else {
        fallbackToEntityRanges()
      }
    }

    if (!isActivatingBlur || !effectiveStreamUrl) return

    setLiveRedactionSeekPending(true)

    if (isDetectionItemLikelyVisibleAtTime(item, currentTime)) return

    const seekTime = getDetectionItemSeekTime(item, currentTime)
    if (seekTime === null || !Number.isFinite(seekTime)) return
    if (Math.abs(seekTime - currentTime) < 0.35) return

    window.setTimeout(() => {
      seekToTime(seekTime, { play: wasPlaying || liveRedactionEnabled })
    }, 0)
  }, [currentTime, effectiveStreamUrl, excludedFromRedactionIds, liveRedactionEnabled, removeFaceTimelineLane, seekToTime, toggleDetectionSelectionById, videoId])

  const previewResolvedRegions = useMemo(() => {
    return trackingRegions.flatMap((region) => {
      const resolved = resolveDisplayedTrackingRegion(region.id, currentTime)
      return resolved ? [resolved] : []
    })
  }, [currentTime, resolveDisplayedTrackingRegion, trackingRegions])
  const regionToOverlayStyle = (region: TrackingRegion): React.CSSProperties => {
    const widthNorm = region.shape === 'circle' ? Math.min(region.width, region.height) : region.width
    const heightNorm = region.shape === 'circle' ? Math.min(region.width, region.height) : region.height
    return {
      left: overlayViewport.left + region.x * overlayViewport.width,
      top: overlayViewport.top + region.y * overlayViewport.height,
      width: widthNorm * overlayViewport.width,
      height: heightNorm * overlayViewport.height,
      borderRadius: region.shape === 'ellipse' || region.shape === 'circle' ? '50%' : '0',
    }
  }
  const regionEffectStyle = (region: TrackingRegion, selected: boolean): React.CSSProperties => {
    if (region.effect === 'solid') {
      return {
        border: selected ? '2px solid rgba(255,255,255,0.98)' : '2px solid rgba(255,255,255,0.65)',
        backgroundColor: 'rgba(10,10,14,0.88)',
        boxShadow: '0 0 0 1px rgba(0,0,0,0.35)',
      }
    }
    if (region.effect === 'pixelate') {
      return {
        border: selected ? '2px solid rgba(255,255,255,0.98)' : '2px solid rgba(255,255,255,0.65)',
        backgroundColor: 'rgba(20,20,26,0.22)',
        backgroundImage: 'linear-gradient(rgba(255,255,255,0.14) 50%, transparent 50%), linear-gradient(90deg, rgba(255,255,255,0.14) 50%, transparent 50%)',
        backgroundSize: '10px 10px',
        backdropFilter: 'saturate(0.85) contrast(1.08)',
        boxShadow: '0 0 0 1px rgba(0,0,0,0.35)',
      }
    }
    return {
      border: selected ? '2px solid rgba(255,255,255,0.98)' : '2px solid rgba(255,255,255,0.72)',
      backgroundColor: 'rgba(255,255,255,0.08)',
      backdropFilter: 'blur(16px) saturate(0.92)',
      WebkitBackdropFilter: 'blur(16px) saturate(0.92)',
      boxShadow: selected
        ? '0 0 0 1px rgba(0,0,0,0.4), inset 0 0 20px rgba(255,255,255,0.08)'
        : '0 0 0 1px rgba(0,0,0,0.3)',
    }
  }
  const drawPreviewStyle: React.CSSProperties | undefined = drawStart && drawCurrent
    ? {
        left: overlayViewport.left + Math.min(drawStart.x, drawCurrent.x) * overlayViewport.width,
        top: overlayViewport.top + Math.min(drawStart.y, drawCurrent.y) * overlayViewport.height,
        width: (trackerShape === 'circle'
          ? Math.min(Math.abs(drawCurrent.x - drawStart.x), Math.abs(drawCurrent.y - drawStart.y))
          : Math.abs(drawCurrent.x - drawStart.x)) * overlayViewport.width,
        height: (trackerShape === 'circle'
          ? Math.min(Math.abs(drawCurrent.x - drawStart.x), Math.abs(drawCurrent.y - drawStart.y))
          : Math.abs(drawCurrent.y - drawStart.y)) * overlayViewport.height,
        borderRadius: trackerShape === 'ellipse' || trackerShape === 'circle' ? '50%' : '0',
      }
    : undefined

  const jumpToActiveSearchRank = useCallback(() => {
    if (activeSearchClipIndex < 0) return
    searchClipRowRefs.current[activeSearchClipIndex]?.scrollIntoView({
      block: 'nearest',
      behavior: 'smooth',
    })
  }, [activeSearchClipIndex])

  useEffect(() => {
    if (searchTimelineLanes.length > 0) {
      setActiveTool('search')
      setRightSidebarOpen(true)
    }
  }, [searchTimelineLanes.length, videoId])

  const tutorialStep = EDITOR_TUTORIAL_STEPS[tutorialStepIndex] ?? EDITOR_TUTORIAL_STEPS[0]
  const isFirstTutorialStep = tutorialStepIndex === 0
  const isLastTutorialStep = tutorialStepIndex === EDITOR_TUTORIAL_STEPS.length - 1
  const getTutorialTargetElement = useCallback((): HTMLElement | null => {
    if (tutorialStep.target === 'player') return videoStageRef.current
    if (tutorialStep.target === 'timeline') return timelineRef.current
    if (tutorialStep.target === 'export') return exportMenuRef.current
    return toolButtonRefs.current[tutorialStep.target] || tutorialButtonRef.current
  }, [tutorialStep.target])

  useEffect(() => {
    if (!tutorialOpen) return
    if (tutorialStep.target === 'tracker' || tutorialStep.target === 'search' || tutorialStep.target === 'captions') {
      setLeftSidebarOpen(true)
    }
  }, [tutorialOpen, tutorialStep.target])

  useLayoutEffect(() => {
    if (!tutorialOpen) {
      setTutorialPosition(null)
      setTutorialTargetRect(null)
      return
    }

    const updateTutorialPosition = () => {
      const target = getTutorialTargetElement()
      if (!target) {
        setTutorialPosition(null)
        setTutorialTargetRect(null)
        return
      }

      const rect = target.getBoundingClientRect()
      const margin = 12
      const gap = 12
      const popoverWidth = Math.min(EDITOR_TUTORIAL_POPOVER_WIDTH, window.innerWidth - margin * 2)
      const popoverHeight = Math.min(EDITOR_TUTORIAL_POPOVER_HEIGHT, window.innerHeight - margin * 2)
      const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value))

      let left = rect.left + rect.width / 2 - popoverWidth / 2
      let top = rect.bottom + gap

      if (tutorialStep.placement === 'right') {
        left = rect.right + gap
        top = rect.top + rect.height / 2 - popoverHeight / 2
        if (left + popoverWidth > window.innerWidth - margin) left = rect.left - popoverWidth - gap
      } else if (tutorialStep.placement === 'left') {
        left = rect.left - popoverWidth - gap
        top = rect.top + rect.height / 2 - popoverHeight / 2
        if (left < margin) left = rect.right + gap
      } else if (tutorialStep.placement === 'top') {
        left = rect.left + rect.width / 2 - popoverWidth / 2
        top = rect.top - popoverHeight - gap
        if (top < margin) top = rect.bottom + gap
      } else {
        left = rect.left + rect.width / 2 - popoverWidth / 2
        top = rect.bottom + gap
        if (top + popoverHeight > window.innerHeight - margin) top = rect.top - popoverHeight - gap
      }

      setTutorialPosition({
        left: clamp(left, margin, window.innerWidth - popoverWidth - margin),
        top: clamp(top, margin, window.innerHeight - popoverHeight - margin),
      })
      setTutorialTargetRect({
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height,
      })
    }

    updateTutorialPosition()
    const frameId = window.requestAnimationFrame(updateTutorialPosition)
    const settledTimer = window.setTimeout(updateTutorialPosition, 240)
    window.addEventListener('resize', updateTutorialPosition)
    window.addEventListener('scroll', updateTutorialPosition, true)
    return () => {
      window.cancelAnimationFrame(frameId)
      window.clearTimeout(settledTimer)
      window.removeEventListener('resize', updateTutorialPosition)
      window.removeEventListener('scroll', updateTutorialPosition, true)
    }
  }, [
    getTutorialTargetElement,
    leftSidebarOpen,
    rightSidebarOpen,
    timelineZoom,
    tutorialOpen,
    tutorialStep.placement,
    tutorialStep.target,
  ])

  const renderPegasusEventButton = (event: PegasusTimelineEvent) => {
    const style = getPegasusSeverityStyle(event.severity)
    const action = (event.recommended_action_ids || [])
      .map((id) => pegasusActionById[id])
      .find(Boolean)
    const bookmarked = pegasusBookmarkedEventIds.includes(event.id)
    return (
      <button
        key={`pegasus-sidebar-event-${event.id}`}
        type="button"
        onClick={() => {
          setPegasusFocusedEventId(event.id)
          seekToTime(event.start_sec, { play: false })
        }}
        className={`w-full rounded-lg border px-3 py-3 text-left transition-colors ${
          focusedPegasusEvent?.id === event.id
            ? 'border-accent/35 bg-accent/10'
            : 'border-border bg-surface/40 hover:border-border/80 hover:bg-card'
        }`}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
              <span
                className={`rounded-full border px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide ${style.text}`}
                style={{ borderColor: style.border, backgroundColor: style.soft }}
              >
                {event.severity}
              </span>
              {bookmarked && (
                <span className="rounded-full border border-white/10 bg-card px-1.5 py-0.5 text-[9px] text-text-tertiary">
                  Bookmarked
                </span>
              )}
            </div>
            <p className="mt-1.5 truncate text-sm font-medium text-text-primary">{event.label}</p>
            <p className="mt-1 text-[11px] text-text-tertiary">
              {fmtShort(event.start_sec)} - {fmtShort(event.end_sec)} · {formatPegasusLabel(event.category)}
            </p>
            {(event.reason || action?.reason) && (
              <p className="mt-1.5 line-clamp-2 text-xs leading-relaxed text-text-secondary">
                {event.reason || action?.reason}
              </p>
            )}
            <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[10px] text-text-tertiary">
              {event.redaction_decision && (
                <span className="rounded-full border border-border bg-card px-1.5 py-0.5 uppercase tracking-wide">
                  {formatPegasusLabel(event.redaction_decision)}
                </span>
              )}
              {event.redaction_target && (
                <span className="rounded-full border border-border bg-card px-1.5 py-0.5 uppercase tracking-wide">
                  {formatPegasusLabel(event.redaction_target)}
                </span>
              )}
              {event.subject_selection && (
                <span className="rounded-full border border-border bg-card px-1.5 py-0.5 uppercase tracking-wide">
                  {formatPegasusLabel(event.subject_selection)}
                </span>
              )}
              {event.scene_role && (
                <span className="rounded-full border border-border bg-card px-1.5 py-0.5 uppercase tracking-wide">
                  {formatPegasusLabel(event.scene_role)}
                </span>
              )}
              <span className="rounded-full border border-border bg-card px-1.5 py-0.5 tabular-nums">
                {formatPegasusConfidence(event.confidence)}
              </span>
            </div>
            {event.handling_note && (
              <p className="mt-1.5 text-[11px] leading-relaxed text-text-tertiary">
                {event.handling_note}
              </p>
            )}
            {event.inclusion_reason && event.inclusion_reason !== event.reason && (
              <p className="mt-1.5 text-[11px] leading-relaxed text-text-tertiary">
                {event.inclusion_reason}
              </p>
            )}
          </div>
          <span className="shrink-0 rounded-md border border-border bg-card px-1.5 py-0.5 text-[10px] text-text-tertiary">
            {fmtShort(event.start_sec)}
          </span>
        </div>
      </button>
    )
  }

  const renderPegasusEventGroup = (label: string, events: PegasusTimelineEvent[]) => {
    if (events.length === 0) return null
    return (
      <div className="border-t border-border/80">
        <div className="sticky top-0 z-[1] flex items-center justify-between border-b border-border/60 bg-surface/85 px-3 py-2 backdrop-blur-sm">
          <span className="text-[10px] font-medium uppercase tracking-wider text-text-tertiary">{label}</span>
          <span className="text-[10px] text-text-tertiary tabular-nums">{events.length}</span>
        </div>
        <div className="space-y-1.5 p-2">
          {events.map(renderPegasusEventButton)}
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-background text-text-primary overflow-hidden">
      <div className="flex flex-1 min-h-0">

        {/* ============ LEFT SIDEBAR (collapsible) ============ */}
        <aside className={`shrink-0 flex flex-col min-w-0 border-r border-border bg-surface overflow-hidden transition-[width] duration-200 ${leftSidebarOpen ? 'w-sidebar' : 'w-10'}`}>
          {leftSidebarOpen ? (
            <>
              <div className="px-4 h-10 flex items-center justify-between border-b border-border shrink-0 gap-1">
                <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">Toolbar</span>
                <button type="button" onClick={() => setLeftSidebarOpen(false)} className={`h-7 w-7 rounded-md ${btnBase}`} aria-label="Collapse toolbar" title="Collapse toolbar">
                  <IconChevronLeft className="w-4 h-4" />
                </button>
              </div>

              <nav className="flex-1 overflow-y-auto py-1 px-2 space-y-0.5">
                {TOOLS.map((t) => (
                  <button
                    key={t.id}
                    type="button"
                    ref={(node) => {
                      toolButtonRefs.current[t.id] = node
                    }}
                    onClick={() => setActiveTool(t.id)}
                    className={`w-full flex items-center gap-2.5 px-2.5 py-2 text-left text-sm rounded-lg transition-colors ${
                      activeTool === t.id
                        ? 'bg-brand-charcoal text-brand-white'
                        : 'text-text-secondary hover:bg-card hover:text-text-primary'
                    }`}
                  >
                    <span className={`w-6 h-6 rounded-md flex items-center justify-center shrink-0 ${
                      activeTool === t.id ? 'bg-white/20' : 'bg-card border border-border'
                    }`}>
                      <img
                        src={t.iconUrl}
                        alt=""
                        className={`w-4 h-4 block object-contain shrink-0 ${activeTool === t.id ? 'brightness-0 invert opacity-95' : 'opacity-60'}`}
                      />
                    </span>
                    {t.label}
                  </button>
                ))}
              </nav>

              <div className="shrink-0 border-t border-border p-3">
                <div className="space-y-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">Redaction</span>
                    <span className="text-[11px] font-mono tabular-nums text-text-tertiary">{blurIntensity}</span>
                  </div>
                  <div className="grid grid-cols-1 gap-1 rounded-lg border border-border bg-card p-1">
                    {REDACTION_STYLE_OPTIONS.map((option) => (
                      <button
                        key={option.value}
                        type="button"
                        onClick={() => setRedactionStyle(option.value)}
                        className={`h-8 rounded-md px-2 text-left text-xs font-medium transition-colors ${
                          redactionStyle === option.value
                            ? 'bg-brand-charcoal text-white shadow-[0_0_0_1px_rgba(15,23,42,0.16)]'
                            : 'text-text-secondary hover:bg-background hover:text-text-primary'
                        }`}
                        aria-pressed={redactionStyle === option.value}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                  <label className={`block space-y-2 text-xs ${redactionStyle === 'black' ? 'text-text-tertiary' : 'text-text-secondary'}`}>
                    <span className="font-medium">Intensity</span>
                    <input
                      type="range"
                      min="15"
                      max="99"
                      step="2"
                      value={blurIntensity}
                      onChange={(e) => setBlurIntensity(Number(e.target.value))}
                      disabled={redactionStyle === 'black'}
                      className="h-2 w-full rounded-full accent-accent cursor-pointer bg-card border border-border disabled:opacity-40 disabled:cursor-not-allowed"
                      aria-label="Redaction intensity"
                    />
                  </label>
                </div>
              </div>

            </>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center py-4 min-h-0">
              <button
                type="button"
                onClick={() => setLeftSidebarOpen(true)}
                className={`h-8 w-8 rounded-md ${btnBase}`}
                aria-label="Expand toolbar"
                title="Expand toolbar"
              >
                <IconChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </aside>

        {/* ============ CENTER (Editor) ============ */}
        <div ref={editorCenterRef} className="flex-1 flex flex-col min-w-0 min-h-0 bg-background overflow-hidden">

          {isDemoMode && (
            <div className="shrink-0 px-5 py-2.5 bg-gradient-to-r from-accent/10 via-accent/5 to-transparent border-b border-accent/20 flex items-center gap-3">
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-accent/15 text-accent border border-accent/25 shrink-0">
                Demo Mode
              </span>
              <p className="text-xs text-text-secondary leading-snug min-w-0">
                You are viewing a pre-loaded demo video. Head to the{' '}
                <Link to="/dashboard" className="font-semibold text-accent hover:underline">Dashboard</Link>{' '}
                to explore your uploaded videos — clicking any video will open it here in the editor for redaction.
              </p>
              <button
                type="button"
                onClick={() => setDemoBannerDismissed(true)}
                className="ml-auto shrink-0 h-6 w-6 rounded-md flex items-center justify-center text-text-tertiary hover:text-text-primary hover:bg-card transition-colors"
                aria-label="Dismiss demo mode notice"
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6L6 18M6 6l12 12" /></svg>
              </button>
            </div>
          )}

          {/* Editor header */}
          <header className="shrink-0 h-14 px-5 flex items-center gap-4 border-b border-border bg-surface">
            <nav className="flex items-center gap-1.5 text-sm font-medium min-w-0" aria-label="Breadcrumb">
              <Link
                to="/dashboard"
                className="py-2 px-1 -ml-1 rounded-md text-text-secondary hover:text-accent hover:bg-card transition-colors truncate"
              >
                Dashboard
              </Link>
              <span className="text-text-tertiary shrink-0" aria-hidden>/</span>
              <span className="text-text-primary truncate" title={title}>{title || 'Untitled'}</span>
            </nav>
            <div className="min-w-0 flex-1" aria-hidden />
            {faceLockEntries.length > 0 && (() => {
              const visibleEntries = faceLockEntries.slice(0, 2)
              const overflowEntries = faceLockEntries.slice(2)
              const overflowCount = overflowEntries.length
              return (
                <div
                  className="shrink-0 flex items-center gap-1.5 mr-2 relative"
                  ref={faceLockListRef}
                  role="status"
                  aria-label={
                    faceLockBuildSummary.hasActive
                      ? `Building face-lock for ${faceLockBuildSummary.buildingCount} faces`
                      : 'Face-lock progress per selected face'
                  }
                >
                  {visibleEntries.map((entry) => (
                    <FaceLockChip key={`face-lock-chip-${entry.personId}`} entry={entry} />
                  ))}
                  {overflowCount > 0 && (
                    <>
                      <button
                        type="button"
                        onClick={() => setFaceLockListOpen((open) => !open)}
                        className={`h-7 px-2 rounded-md text-[11px] font-medium border transition-colors ${
                          faceLockListOpen
                            ? 'border-accent/40 bg-accent/10 text-accent'
                            : 'border-border bg-card text-text-secondary hover:bg-background hover:text-text-primary'
                        }`}
                        aria-haspopup="true"
                        aria-expanded={faceLockListOpen}
                        title={`${overflowCount} more face${overflowCount === 1 ? '' : 's'}`}
                      >
                        +{overflowCount}
                      </button>
                      {faceLockListOpen && (
                        <div className="absolute right-0 top-full mt-1 z-50 w-72 rounded-lg border border-border bg-surface shadow-xl py-1.5">
                          <div className="px-3 py-1.5 border-b border-border flex items-center justify-between">
                            <span className="text-[11px] font-medium uppercase tracking-wider text-text-tertiary">
                              Face-lock progress
                            </span>
                            <span className="text-[11px] tabular-nums text-text-secondary">
                              {faceLockBuildSummary.readyCount}/{faceLockBuildSummary.totalCount}
                            </span>
                          </div>
                          <div className="max-h-72 overflow-y-auto py-1">
                            {faceLockEntries.map((entry) => (
                              <div
                                key={`face-lock-row-${entry.personId}`}
                                className="px-3 py-1.5 flex items-center gap-2.5"
                              >
                                <FaceLockChip entry={entry} variant="row" />
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )
            })()}
            <div className="relative shrink-0" ref={tutorialRef}>
              <button
                type="button"
                ref={tutorialButtonRef}
                onClick={() => {
                  setExportMenuOpen(false)
                  setFaceLockListOpen(false)
                  setTutorialStepIndex(0)
                  setTutorialOpen((open) => !open)
                }}
                className={`h-9 px-3 rounded-lg text-sm font-medium border flex items-center gap-2 ${btnBase}`}
                aria-expanded={tutorialOpen}
                aria-haspopup="dialog"
                aria-controls="editor-tutorial-popover"
              >
                <IconHelp className="w-4 h-4" />
                Tutorial
              </button>
              {tutorialOpen && tutorialTargetRect && (
                <div
                  className="fixed z-[48] pointer-events-none rounded-xl border border-accent/70 shadow-[0_0_0_4px_rgba(0,220,130,0.12),0_10px_30px_rgba(0,0,0,0.18)]"
                  style={{
                    left: tutorialTargetRect.left - 4,
                    top: tutorialTargetRect.top - 4,
                    width: tutorialTargetRect.width + 8,
                    height: tutorialTargetRect.height + 8,
                  }}
                  aria-hidden
                />
              )}
              {tutorialOpen && tutorialPosition && (
                <div
                  id="editor-tutorial-popover"
                  role="dialog"
                  aria-modal="false"
                  aria-labelledby="editor-tutorial-title"
                  className="fixed z-50 w-[21rem] max-w-[calc(100vw-2rem)] rounded-xl border border-border bg-surface shadow-xl"
                  style={tutorialPosition}
                  onKeyDown={(event) => event.stopPropagation()}
                >
                  <div className="border-b border-border px-4 py-3">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="text-[11px] font-medium uppercase tracking-wider text-text-tertiary">
                          Step {tutorialStepIndex + 1} of {EDITOR_TUTORIAL_STEPS.length}
                        </p>
                        <h2 id="editor-tutorial-title" className="mt-1 text-sm font-semibold text-text-primary">
                          {tutorialStep.title}
                        </h2>
                      </div>
                      <button
                        type="button"
                        onClick={() => setTutorialOpen(false)}
                        className="-mr-1 -mt-1 rounded-md p-1.5 text-text-tertiary transition-colors hover:bg-card hover:text-text-primary"
                        aria-label="Close tutorial"
                      >
                        <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                          <line x1="18" y1="6" x2="6" y2="18" />
                          <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                      </button>
                    </div>
                  </div>
                  <div className="px-4 py-3">
                    <p className="text-sm leading-relaxed text-text-secondary">
                      {tutorialStep.body}
                    </p>
                    <div className="mt-4 flex items-center justify-between gap-3">
                      <div className="flex items-center gap-1" aria-hidden>
                        {EDITOR_TUTORIAL_STEPS.map((step, index) => (
                          <span
                            key={step.title}
                            className={`h-1.5 rounded-full transition-all ${
                              index === tutorialStepIndex ? 'w-5 bg-accent' : 'w-1.5 bg-border'
                            }`}
                          />
                        ))}
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => setTutorialStepIndex((index) => Math.max(0, index - 1))}
                          disabled={isFirstTutorialStep}
                          className="h-8 rounded-md border border-border bg-card px-3 text-xs font-medium text-text-secondary transition-colors hover:bg-background hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-45"
                        >
                          Back
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            if (isLastTutorialStep) {
                              setTutorialOpen(false)
                              return
                            }
                            setTutorialStepIndex((index) => Math.min(EDITOR_TUTORIAL_STEPS.length - 1, index + 1))
                          }}
                          className="h-8 rounded-md border border-accent bg-accent px-3 text-xs font-medium text-background transition-colors hover:bg-accent-hover"
                        >
                          {isLastTutorialStep ? 'Done' : 'Next'}
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="relative shrink-0" ref={exportMenuRef}>
              <button
                type="button"
                onClick={() => setExportMenuOpen((o) => !o)}
                className={`h-9 px-3 rounded-lg text-sm font-medium border flex items-center gap-2 ${btnBase}`}
                aria-expanded={exportMenuOpen}
                aria-haspopup="true"
                aria-label="Export options"
              >
                <IconDownload className="w-4 h-4" />
                Export
                <IconChevronDown className={`w-4 h-4 transition-transform ${exportMenuOpen ? 'rotate-180' : ''}`} />
              </button>
              {exportMenuOpen && (
                <div className="absolute right-0 top-full mt-1 py-1 min-w-[14rem] rounded-lg border border-border bg-surface shadow-lg z-50">
                  {exportRedactError && (
                    <p className="px-3 py-2 text-xs text-error border-b border-border">{exportRedactError}</p>
                  )}
                  <div className="px-3 py-2 border-b border-border space-y-2">
                    <div className="text-[11px] font-medium uppercase tracking-wider text-text-tertiary">Quality</div>
                    <div className="grid grid-cols-3 gap-1">
                      {EXPORT_QUALITY_OPTIONS.map((option) => {
                        const active = exportQuality === option.value
                        return (
                          <button
                            key={option.value}
                            type="button"
                            onClick={() => setExportQuality(option.value)}
                            disabled={exportRedactLoading}
                            aria-pressed={active}
                            className={`h-7 rounded-md border text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${
                              active
                                ? 'border-accent bg-accent text-background'
                                : 'border-border text-text-secondary hover:text-text-primary hover:bg-card'
                            }`}
                          >
                            {option.label}
                          </button>
                        )
                      })}
                    </div>
                  </div>
                  <button
                    type="button"
                    className="w-full px-3 py-2 text-left text-sm text-text-primary hover:bg-card transition-colors flex items-center gap-2 border-b border-border whitespace-nowrap disabled:opacity-60 disabled:cursor-not-allowed"
                    onClick={() => { void exportRedacted() }}
                    disabled={exportRedactLoading}
                  >
                    <IconDownload className="w-4 h-4" />
                    {exportRedactLoading ? 'Rendering...' : 'Download Redacted Video'}
                  </button>
                  {exportRedactLoading && exportRedactProgress && (
                    <div className="px-3 py-2 border-b border-border space-y-1">
                      <div className="flex items-center justify-between text-[11px] text-text-secondary">
                        <span className="truncate">{exportRedactProgress.message}</span>
                        <span className="tabular-nums ml-2">{exportRedactProgress.percent}%</span>
                      </div>
                      <div className="h-1 w-full rounded-full bg-border overflow-hidden">
                        <div
                          className="h-full rounded-full bg-accent transition-all duration-500"
                          style={{ width: `${Math.max(0, Math.min(100, exportRedactProgress.percent))}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </header>


          {/* Video viewport: must get enough space to show the player; timeline can shrink when space is tight */}
          <div className="flex-1 min-h-[140px] px-4 pt-4 pb-2 flex items-center justify-center">
            <div ref={videoStageRef} className="size-full max-w-5xl flex items-center justify-center min-w-0 min-h-0">
              <div
                ref={videoContainerRef}
                className="relative flex-none rounded-xl overflow-hidden bg-brand-charcoal border border-border shadow-lg"
                style={fittedVideoFrameStyle}
              >
                {effectiveStreamUrl ? (
                  <>
                  <video
                    ref={videoRef}
                    src={useHls ? undefined : effectiveStreamUrl}
                    className="absolute inset-0 w-full h-full object-contain z-0"
                      playsInline
                      muted={isMuted}
                      loop={false}
                    onLoadedMetadata={onLoadedMetadata}
                    onTimeUpdate={onTimeUpdate}
                    onPlay={() => {
                      setIsPlaying(true)
                      setLiveRedactionSeekPending(false)
                    }}
                    onPause={() => {
                      setIsPlaying(false)
                      window.setTimeout(() => syncLiveRedactionFrame({ force: true }), 0)
                    }}
                    onSeeked={() => {
                      const video = videoRef.current
                      if (!video) return
                      setCurrentTime(video.currentTime)
                      syncLiveRedactionFrame({ force: true, clearDetections: true, resetTracking: true, time: video.currentTime })
                    }}
                    onEnded={() => setIsPlaying(false)}
                    onClick={() => togglePlay()}
                  />
                    {/* Hidden video for timeline preload only; main video is never sought on pause */}
                    <video
                      ref={timelinePreviewVideoRef}
                      src={useHls ? undefined : effectiveStreamUrl}
                      className="absolute opacity-0 pointer-events-none w-0 h-0"
                      muted
                      playsInline
                      onLoadedMetadata={() => setPreviewVideoReady(true)}
                    />
                  </>
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 select-none bg-card/60 rounded-xl">
                    <svg className="w-14 h-14 text-text-tertiary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" aria-hidden>
                      <rect x="2" y="4" width="20" height="16" rx="2" />
                      <path d="M10 9l5 3-5 3V9z" fill="currentColor" stroke="none" />
                    </svg>
                    <p className="text-sm font-medium text-text-primary">No video source</p>
                    <p className="text-xs text-text-secondary">Upload a video from the dashboard</p>
                  </div>
                )}

                {/* Live blur overlay from backend detections */}
                {effectiveStreamUrl && liveRedactionOverlayVisible && (
                  <div className="absolute inset-0 z-10 pointer-events-none">
                    <div className="absolute" style={overlayFrameStyle}>
                      {!showLiveRedactionSeekLoader && useDomVideoBlurOverlay && (
                        <>
                          <svg className="absolute h-0 w-0 pointer-events-none" aria-hidden focusable="false">
                            <defs>
                              <clipPath id={liveBlurClipPathId} clipPathUnits="objectBoundingBox">
                                {renderedLiveRedactionDetections.map((detection, index) => {
                                  const renderBox = getLiveRedactionRenderBox(detection)
                                  return (
                                    <rect
                                      key={`live-clip-${detection.trackId || detection.id || `${detection.kind}-${index}`}`}
                                      x={renderBox.x}
                                      y={renderBox.y}
                                      width={renderBox.width}
                                      height={renderBox.height}
                                    />
                                  )
                                })}
                              </clipPath>
                            </defs>
                          </svg>
                          <video
                            ref={liveBlurOverlayVideoRef}
                            src={useHls ? undefined : effectiveStreamUrl}
                            className="absolute inset-0 h-full w-full pointer-events-none"
                            style={liveBlurOverlayVideoStyle}
                            playsInline
                            muted
                            aria-hidden
                          />
                        </>
                      )}
                      {!showLiveRedactionSeekLoader && renderedLiveRedactionDetections.length > 0 && renderedLiveRedactionDetections.map((detection, index) => (
                        <div
                          key={`live-region-${detection.trackId || detection.id || `${detection.kind}-${index}`}`}
                          className="absolute z-10"
                          style={getLiveRedactionOverlayStyle(detection, redactionStyle, blurIntensity)}
                          aria-hidden
                        />
                      ))}
                      {showLiveRedactionSeekLoader && (
                        <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/18 backdrop-blur-[2px]">
                          <div
                            role="status"
                            aria-live="polite"
                            className="pointer-events-none flex max-w-[18rem] flex-col items-center gap-2 rounded-2xl border border-white/10 bg-brand-charcoal/86 px-5 py-4 text-center shadow-[0_18px_60px_rgba(0,0,0,0.35)]"
                          >
                            <span className="flex h-11 w-11 items-center justify-center rounded-full border border-white/10 bg-white/5">
                              <span className="h-5 w-5 rounded-full border-2 border-white/35 border-t-white animate-spin" aria-hidden />
                            </span>
                            <div className="space-y-1">
                              <p className="text-sm font-medium text-white">
                                {isScrubbing ? 'Updating blur while you scrub' : liveRedactionEnabled ? 'Updating live blur preview' : 'Updating selected blur preview'}
                              </p>
                              <p className="text-[11px] leading-relaxed text-white/68">
                                {isScrubbing
                                  ? 'The blur will lock onto the selected frame as soon as detection finishes.'
                                  : liveRedactionEnabled
                                    ? 'This frame needs a fresh detection pass after the timeline jump.'
                                    : 'This frame needs a fresh identity check for the selected item.'}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                      {/*
                        Active blur regions are intentionally non-interactive on the video surface so
                        clicking the blur (accidentally or on purpose) cannot remove or alter it. To
                        toggle a redaction, use the detections list in the right sidebar.
                      */}
                    </div>
                    <div className="absolute top-3 right-3 rounded-lg bg-brand-charcoal/90 px-2.5 py-1.5 text-[11px] text-white shadow-lg border border-white/10 backdrop-blur-sm">
                      {showLiveRedactionSeekLoader
                        ? isScrubbing
                          ? 'Scrubbing timeline...'
                          : 'Updating blur preview...'
                        : liveRedactionLoading && visibleLiveRedactionDetections.length === 0
                        ? 'Scanning current frame...'
                        : livePreviewModeLabel}
                    </div>
                    {(() => {
                      const buildingPersons = Object.entries(faceLockBuildByPersonId).filter(
                        ([, state]) => state.status === 'queued' || state.status === 'running',
                      )
                      if (buildingPersons.length === 0) return null
                      const avg = Math.round(
                        buildingPersons.reduce((acc, [, state]) => acc + (state.percent || 0), 0) /
                          Math.max(1, buildingPersons.length),
                      )
                      return (
                        <div className="absolute top-12 right-3 max-w-xs rounded-md border border-white/10 bg-brand-charcoal/90 px-2.5 py-1.5 text-[11px] text-white shadow-lg backdrop-blur-sm">
                          <div className="flex items-center gap-2">
                            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                            <span>
                              Locking onto {buildingPersons.length === 1 ? 'face' : `${buildingPersons.length} faces`}... {avg}%
                            </span>
                          </div>
                        </div>
                      )
                    })()}
                    {liveRedactionError && (
                      <div className="absolute top-20 right-3 max-w-xs rounded-md border border-error/40 bg-brand-charcoal/90 px-2.5 py-1.5 text-[11px] text-red-200 shadow-lg">
                        {liveRedactionError}
                      </div>
                    )}
                    {showFaceLockIntroPopup && (
                      <div
                        role="status"
                        className="absolute left-3 top-3 z-30 w-[260px] rounded-xl border border-accent/30 bg-surface/95 p-3 text-text-primary shadow-xl backdrop-blur-md"
                      >
                        <div className="flex items-start gap-2.5">
                          <div className="relative shrink-0 mt-0.5">
                            <span className="absolute inset-0 -m-0.5 rounded-full bg-accent/25 animate-ping" aria-hidden />
                            <span className="relative inline-flex h-6 w-6 items-center justify-center rounded-full bg-accent/15 ring-1 ring-accent/40 text-accent">
                              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden>
                                <path
                                  d="M12 2 4 6v6c0 5 3.4 9.4 8 10 4.6-.6 8-5 8-10V6l-8-4Z"
                                  stroke="currentColor"
                                  strokeWidth="1.8"
                                  strokeLinejoin="round"
                                />
                                <path
                                  d="m9 12 2 2 4-4"
                                  stroke="currentColor"
                                  strokeWidth="1.8"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                              </svg>
                            </span>
                          </div>
                          <div className="min-w-0 flex-1">
                            <p className="text-[12px] font-semibold leading-tight">Locking onto this face</p>
                            <p className="mt-1 text-[11px] leading-snug text-text-secondary">
                              Once the build finishes, the blur snaps precisely to this face for the rest of the clip.
                            </p>
                          </div>
                          <button
                            type="button"
                            aria-label="Dismiss"
                            className="shrink-0 -mt-1 -mr-1 rounded p-1 text-text-tertiary hover:text-text-primary hover:bg-card transition-colors"
                            onClick={() => setShowFaceLockIntroPopup(false)}
                          >
                            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
                              <path d="M2 2L10 10M10 2L2 10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                            </svg>
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Progress: time labels + seek slider */}
          <div className="shrink-0 px-4 pt-3 pb-1 flex items-center gap-3">
            <span className="text-xs font-mono text-text-secondary tabular-nums w-24 shrink-0 text-right">
              {fmtTime(currentTime)}
            </span>
            <input
              type="range"
              min={0}
              max={duration || 100}
              step={0.1}
              value={currentTime}
              onChange={(e) => {
                const t = Number(e.target.value)
                setCurrentTime(t)
                markLiveRedactionSeekPending()
                if (videoRef.current) videoRef.current.currentTime = t
              }}
              className="flex-1 h-2 rounded-full accent-accent cursor-pointer bg-card border border-border"
              aria-label="Seek"
            />
            <span className="text-xs font-mono text-text-secondary tabular-nums w-24 shrink-0">
              {fmtTime(duration)}
            </span>
          </div>

          {/* Controls bar */}
          <div className="shrink-0 min-h-12 px-4 py-3 border-t border-border bg-surface flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-1">
              <button type="button" className={`h-9 w-9 ${btnBase}`} onClick={() => skip(-5)} aria-label="Rewind 5s" title="Rewind 5s">
                <IconReplay5 className="w-4 h-4" />
              </button>
              <button
                type="button"
                className={`h-10 w-10 ${btnBase} bg-brand-charcoal text-brand-white border-brand-charcoal hover:bg-brand-grey hover:text-brand-charcoal`}
                onClick={togglePlay}
                aria-label={isPlaying ? 'Pause' : 'Play'}
                title={isPlaying ? 'Pause' : 'Play'}
              >
                {isPlaying ? <IconPause className="w-5 h-5" /> : <IconPlay className="w-5 h-5" />}
              </button>
              <button type="button" className={`h-9 w-9 ${btnBase}`} onClick={() => skip(5)} aria-label="Forward 5s" title="Forward 5s">
                <IconForward5 className="w-4 h-4" />
              </button>
            </div>

            <button
              type="button"
              onClick={handleSnapFaceFromVideo}
              disabled={!effectiveStreamUrl || snapFaceCapturing}
              className="inline-flex items-center gap-1.5 h-9 px-3 rounded-md border border-accent/30 bg-accent/10 text-accent text-xs font-medium hover:bg-accent/15 hover:border-accent/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Snap face from video and add to anonymize list"
              title="Pause and snap a face from this frame to anonymize"
            >
              {snapFaceCapturing ? (
                <span className="w-3.5 h-3.5 border-2 border-accent/30 border-t-accent rounded-full animate-spin" aria-hidden />
              ) : (
                <IconCameraSnap className="w-4 h-4" />
              )}
              <span>Snap face</span>
            </button>

            <div className="flex items-center gap-2 text-xs">
              <span className="font-mono text-text-secondary tabular-nums">{fmtTime(currentTime)}</span>
              <span className="text-text-tertiary">/</span>
              <span className="font-mono text-text-secondary tabular-nums">{fmtTime(duration)}</span>
            </div>

            {snapFaceError && (
              <p className="text-[11px] text-red-400 max-w-[16rem] truncate" title={snapFaceError}>
                {snapFaceError}
              </p>
            )}

            <div className="flex items-center gap-2">
              <label className="text-xs text-text-tertiary whitespace-nowrap">Speed</label>
              <select
                className="h-8 min-w-[4rem] rounded-md bg-surface border border-border px-2 text-xs text-text-primary cursor-pointer"
                value={playbackRate}
                onChange={(e) => changeRate(Number(e.target.value))}
              >
                {[0.25, 0.5, 0.75, 1, 1.25, 1.5, 2].map((r) => (
                  <option key={r} value={r}>{r}x</option>
                ))}
              </select>
            </div>

            <div className="flex-1 min-w-4" />

            <div className="flex items-center gap-2">
              <button type="button" className={`h-9 w-9 ${btnBase}`} onClick={toggleMute} aria-label={isMuted ? 'Unmute' : 'Mute'} title={isMuted ? 'Unmute' : 'Mute'}>
                {isMuted ? <IconVolumeOff className="w-4 h-4" /> : <IconVolumeUp className="w-4 h-4" />}
              </button>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={isMuted ? 0 : volume}
                onChange={(e) => changeVolume(Number(e.target.value))}
                className="w-20 h-1.5 rounded-full accent-accent cursor-pointer bg-card border border-border"
                aria-label="Volume"
              />
            </div>

            <button type="button" className={`h-9 w-9 ${btnBase}`} onClick={toggleFullscreen} aria-label={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'} title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}>
              {isFullscreen ? <IconFullscreenExit className="w-4 h-4" /> : <IconFullscreen className="w-4 h-4" />}
            </button>
          </div>

          {/* ============ TIMELINE ============ */}
          <div className="flex-1 min-h-0 flex flex-col border-t border-border bg-surface select-none overflow-hidden" style={{ minHeight: 100 }}>
            <div className="h-10 px-3 flex items-center justify-between border-b border-border shrink-0">
              <div className="flex items-center gap-3 min-w-0">
                <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">Timeline</span>
                <span className="text-xs font-mono text-text-tertiary tabular-nums">{fmtShort(currentTime)} / {fmtShort(duration)}</span>
              </div>
              <div className="flex items-center gap-1">
                <button type="button" className={`h-6 w-6 ${btnBase} border-0`} onClick={() => setTimelineZoom((z) => Math.max(z / 1.5, 0.5))} aria-label="Zoom out" title="Zoom out">
                  <IconMinus className="w-3 h-3" />
                </button>
                <div className="w-20 h-1.5 rounded-full bg-border mx-1 relative overflow-hidden">
                  <div className="absolute inset-y-0 left-0 rounded-full bg-accent" style={{ width: `${Math.min(100, ((timelineZoom - 0.5) / 9.5) * 100)}%` }} />
                </div>
                <button type="button" className={`h-6 w-6 ${btnBase} border-0`} onClick={() => setTimelineZoom((z) => Math.min(z * 1.5, 10))} aria-label="Zoom in" title="Zoom in">
                  <IconPlus className="w-3 h-3" />
                </button>
              </div>
            </div>

            <div className="flex flex-1 min-h-0">
              <div
                ref={timelineRef}
                className="flex-1 min-h-0 overflow-x-auto overflow-y-auto bg-background"
                onMouseDown={handleTimelineMouseDown}
                onMouseMove={handleTimelineHover}
                onMouseLeave={() => setHoverTime(null)}
              >
                <div className="min-w-full flex flex-col relative" style={{ width: `${timelineContentWidthPct}%`, minHeight: '100%' }}>
                  <div className="h-9 shrink-0 relative border-b border-border bg-card/90 cursor-col-resize overflow-hidden">
                    <div className="absolute inset-0 bg-gradient-to-b from-white/[0.03] via-transparent to-transparent pointer-events-none" />
                    {minorTicks.map((sec) => (
                      <div key={`m-${sec}`} className="absolute bottom-0 w-px bg-border/80" style={{ left: `${(sec / Math.max(duration, 1)) * 100}%`, height: 8 }} />
                    ))}
                    {majorTicks.map((sec) => (
                      <div key={`M-${sec}`} className="absolute top-0 bottom-0" style={{ left: `${duration > 0 ? (sec / duration) * 100 : 0}%` }}>
                        <div className="absolute bottom-0 w-px bg-text-tertiary" style={{ height: 14 }} />
                        <span className="absolute left-1.5 top-1.5 text-[10px] font-mono text-text-tertiary tabular-nums whitespace-nowrap leading-none">{fmtShort(sec)}</span>
                      </div>
                    ))}
                    {hoverTime !== null && !isScrubbing && (
                      <div className="absolute top-0 bottom-0 pointer-events-none z-10" style={{ left: `${(hoverTime / Math.max(duration, 1)) * 100}%` }}>
                        <div className="absolute bottom-0 w-px h-full bg-border" />
                        <span className="absolute top-1 left-1/2 -translate-x-1/2 px-1.5 py-0.5 rounded text-[9px] font-mono bg-surface text-text-primary border border-border tabular-nums whitespace-nowrap shadow-sm">{fmtShort(hoverTime)}</span>
                      </div>
                    )}
                  </div>

                  <div
                    className={`shrink-0 relative border-b border-border ${trackMuted.video ? 'opacity-40' : ''}`}
                    style={{ height: `${VIDEO_TIMELINE_LANE_HEIGHT_PX}px` }}
                  >
                    <div className="absolute inset-x-0 inset-y-1.5 rounded-lg overflow-hidden border border-accent/25 bg-gradient-to-r from-brand-charcoal via-[#123228] to-brand-charcoal shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                      <div className="absolute inset-0 flex">
                        {timelineThumbnails.length > 0 ? (
                          timelineThumbnails.map((src, i) => (
                            <div key={i} className="flex-1 min-w-0 h-full relative" style={{ flexBasis: 0 }}>
                              <img src={src} alt="" className="w-full h-full object-cover block" />
                            </div>
                          ))
                        ) : (
                          (() => {
                            const frameCount = Math.min(16, Math.max(8, Math.floor(Math.max(duration, 8) / 2)))
                            return Array.from({ length: frameCount }, (_, i) => (
                              <div key={i} className="relative h-full flex-1 min-w-0 border-r border-white/10" style={{ flexBasis: 0 }}>
                                <div className="absolute inset-0 bg-gradient-to-br from-white/[0.04] via-transparent to-black/30" />
                                <div className="absolute inset-x-0 bottom-0 h-4 bg-gradient-to-t from-black/40 to-transparent" />
                                <div className="absolute top-2 left-2 right-2 h-px bg-white/10" />
                                {i === frameCount - 1 ? null : <div className="absolute top-0 right-0 bottom-0 w-px bg-accent/15" />}
                              </div>
                            ))
                          })()
                        )}
                      </div>
                      <div className="absolute inset-0 bg-gradient-to-r from-black/60 via-transparent to-black/40 pointer-events-none" />
                      <div className="absolute inset-x-2.5 top-2 z-[2] flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <p className="text-xs font-medium text-white truncate" title={title}>{title || 'Untitled'}</p>
                        </div>
                        <span className="rounded-full border border-white/10 bg-black/30 px-1.5 py-0.5 text-[9px] font-mono tabular-nums text-white/80">{fmtShort(duration)}</span>
                      </div>
                      <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-accent/80" />
                      <div className="absolute right-0 top-0 bottom-0 w-1.5 bg-accent/60" />
                      <div className="absolute left-0 bottom-0 h-1 bg-accent/80" style={{ width: `${Math.max(0, Math.min(100, progress * 100))}%` }} />
                    </div>
                    {trackLocked.video && (
                      <div className="absolute inset-0 bg-background/40 rounded-lg flex items-center justify-center">
                        <svg className="w-4 h-4 text-text-tertiary" viewBox="0 0 24 24" fill="currentColor"><path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zM9 8V6c0-1.66 1.34-3 3-3s3 1.34 3 3v2H9z" /></svg>
                      </div>
                    )}
                  </div>

                  {searchTimelineLanes.length > 0 && (
                    <div
                      className="shrink-0 border-b border-border"
                      style={{ maxHeight: searchTimelineLanes.length > 3 ? 3 * (ENTITY_SEARCH_LANE_HEIGHT_PX + 50) : undefined }}
                    >
                      <div
                        className={searchTimelineLanes.length > 3 ? 'overflow-y-auto overflow-x-hidden lane-scroller' : ''}
                        style={searchTimelineLanes.length > 3 ? { maxHeight: 3 * (ENTITY_SEARCH_LANE_HEIGHT_PX + 50) } : undefined}
                      >
                        {searchTimelineLanes.map((lane) => (
                          <div
                            key={`search-lane-${lane.sessionId}`}
                            className="border-b border-border last:border-b-0"
                          >
                            <div className="relative" style={{ height: `${ENTITY_SEARCH_LANE_HEIGHT_PX}px` }}>
                              <div
                                className={`absolute inset-x-0 inset-y-1.5 rounded-lg overflow-hidden shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] ${activeSearchTimelineLane?.sessionId === lane.sessionId ? 'ring-1 ring-accent/45' : ''}`}
                                style={{
                                  border: `1px solid ${ENTITY_SEARCH_LANE_BORDER}`,
                                  background: ENTITY_SEARCH_LANE_BACKGROUND,
                                }}
                                onClick={() => activateSearchSession(lane.sessionId)}
                                role="button"
                                tabIndex={0}
                                onKeyDown={(event) => {
                                  if (event.key === 'Enter' || event.key === ' ') {
                                    event.preventDefault()
                                    activateSearchSession(lane.sessionId)
                                  }
                                }}
                              >
                                <div
                                  className="absolute inset-0 pointer-events-none"
                                  style={{ background: ENTITY_SEARCH_LANE_OVERLAY }}
                                />
                                <div
                                  className="absolute inset-x-3 bottom-[9px] h-px"
                                  style={{ backgroundColor: ENTITY_SEARCH_LANE_BASELINE }}
                                />
                                {lane.visibleEntities.length > 0 && (
                                  <div className="absolute left-3 top-2 right-12 z-[4] flex items-center gap-1.5 overflow-hidden pointer-events-none">
                                    {lane.visibleEntities.map((entity) => (
                                      <SearchEntityChip
                                        key={`timeline-search-entity-${lane.sessionId}-${entity.id}`}
                                        entity={entity}
                                        variant="timeline"
                                      />
                                    ))}
                                    {lane.hiddenEntityCount > 0 && (
                                      <div
                                        className="inline-flex items-center rounded-full border border-white/10 bg-black/22 px-2 py-1 text-[10px] font-medium text-white/72 shadow-[0_1px_0_rgba(255,255,255,0.04)] backdrop-blur-sm"
                                        style={{ opacity: 0.8 }}
                                      >
                                        +{lane.hiddenEntityCount} more
                                      </div>
                                    )}
                                  </div>
                                )}
                                <button
                                  type="button"
                                  className="absolute right-3 top-2 z-[5] inline-flex h-5 w-5 items-center justify-center rounded-md border border-white/18 bg-black/20 text-white/80 transition hover:border-white/32 hover:bg-black/35 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/45"
                                  onClick={(event) => {
                                    event.stopPropagation()
                                    clearSearchTimelineLane(lane.sessionId)
                                  }}
                                  title="Remove entity lane"
                                  aria-label="Remove entity lane"
                                >
                                  <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" aria-hidden>
                                    <path d="M6 6L18 18M18 6L6 18" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                                  </svg>
                                </button>
                                {lane.bars.length > 0 ? (
                                  <>
                                    <svg className="absolute inset-0 h-full w-full z-[1]" viewBox="0 0 1200 84" preserveAspectRatio="none" aria-hidden>
                                      {lane.bars.map((bar, index) => {
                                        const barHeight = 8 + bar.value * 56
                                        const y = 76 - barHeight
                                        return (
                                          <rect
                                            key={`search-bar-base-${lane.sessionId}-${index}`}
                                            x={bar.x.toFixed(2)}
                                            y={y.toFixed(2)}
                                            width={bar.width.toFixed(2)}
                                            height={barHeight.toFixed(2)}
                                            rx="1.8"
                                            fill={bar.isPeak ? ENTITY_SEARCH_LANE_PEAK_BAR_BASE : ENTITY_SEARCH_LANE_BAR_BASE}
                                          />
                                        )
                                      })}
                                    </svg>
                                    <svg
                                      className="absolute inset-0 h-full w-full z-[2]"
                                      viewBox="0 0 1200 84"
                                      preserveAspectRatio="none"
                                      aria-hidden
                                      style={{ clipPath: `inset(0 ${Math.max(0, 100 - progress * 100)}% 0 0)` }}
                                    >
                                      {lane.bars.map((bar, index) => {
                                        const barHeight = 8 + bar.value * 56
                                        const y = 76 - barHeight
                                        return (
                                          <rect
                                            key={`search-bar-active-${lane.sessionId}-${index}`}
                                            x={bar.x.toFixed(2)}
                                            y={y.toFixed(2)}
                                            width={bar.width.toFixed(2)}
                                            height={barHeight.toFixed(2)}
                                            rx="1.8"
                                            fill={bar.isPeak ? ENTITY_SEARCH_LANE_PEAK_COLOR : ENTITY_SEARCH_LANE_COLOR}
                                          />
                                        )
                                      })}
                                    </svg>
                                  </>
                                ) : null}
                                {lane.segments.map(({ clip, index, isActive, importance, left, width }) => (
                                  <button
                                    key={`lane-hit-${lane.sessionId}-${clip.start}-${clip.end}-${index}`}
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation()
                                      activateSearchSession(lane.sessionId)
                                      seekToTime(clip.start)
                                    }}
                                    className="absolute inset-y-0 z-[3] bg-transparent"
                                    style={{
                                      left,
                                      width,
                                    }}
                                    title={`${fmtShort(clip.start)} - ${fmtShort(clip.end)}`}
                                  >
                                    <span
                                      className={`absolute inset-y-1 rounded-full transition-all ${isActive ? '' : 'hover:bg-[rgba(0,220,130,0.10)]'}`}
                                      style={{
                                        left: 0,
                                        right: 0,
                                        backgroundColor: isActive ? ENTITY_SEARCH_LANE_SEGMENT_ACTIVE : undefined,
                                        boxShadow: isActive ? `0 0 0 1px ${ENTITY_SEARCH_LANE_SEGMENT_RING}` : undefined,
                                        opacity: isActive ? 0.95 : 0.15 + importance * 0.42,
                                      }}
                                    />
                                  </button>
                                ))}
                                <div className="absolute left-0 top-0 bottom-0 w-1.5" style={{ backgroundColor: ENTITY_SEARCH_LANE_EDGE_LEFT }} />
                                <div className="absolute right-0 top-0 bottom-0 w-1.5" style={{ backgroundColor: ENTITY_SEARCH_LANE_EDGE_RIGHT }} />
                              </div>
                            </div>
                            <div className="px-3 pb-2.5 pt-1.5">
                              <p className="text-[11px] leading-4 text-text-tertiary">
                                {lane.noteLines[0]}
                              </p>
                              <p className="text-[11px] leading-4 text-text-tertiary">
                                {lane.noteLines[1]}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {pegasusStatus === 'ready' && filteredPegasusEvents.length > 0 && (
                    <div className="shrink-0 border-b border-border">
                      <div className="relative" style={{ height: `${PEGASUS_LANE_HEIGHT_PX}px` }}>
                        <div
                          className="absolute inset-x-0 inset-y-1.5 rounded-lg overflow-hidden border border-border bg-card shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]"
                          onClick={() => {
                            setActiveTool('pegasus')
                            setRightSidebarOpen(true)
                          }}
                          role="button"
                          tabIndex={0}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter' || event.key === ' ') {
                              event.preventDefault()
                              setActiveTool('pegasus')
                              setRightSidebarOpen(true)
                            }
                          }}
                        >
                          <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-white/[0.025] via-transparent to-black/[0.05]" />
                          <div className="absolute inset-x-3 bottom-[10px] h-px bg-border/70" />
                          <div className="absolute left-3 top-2 right-3 z-[4] flex items-center justify-between gap-2 pointer-events-none">
                            <div className="inline-flex min-w-0 items-center gap-2 rounded-full border border-border bg-surface/80 px-2 py-1 shadow-[0_1px_0_rgba(255,255,255,0.03)] backdrop-blur-sm">
                              <img src={metaInsightsIconUrl} alt="" className="h-3.5 w-3.5 opacity-80" />
                              <span className="truncate text-[10px] font-medium text-text-secondary">Meta Insights</span>
                            </div>
                            <div className="flex shrink-0 items-center gap-1.5">
                              <span className="rounded-full border border-border bg-surface/80 px-2 py-1 text-[9px] font-medium uppercase tracking-wide text-text-tertiary">
                                {filteredPegasusEvents.length} hotspot{filteredPegasusEvents.length === 1 ? '' : 's'}
                              </span>
                            </div>
                          </div>
                          {filteredPegasusEvents.map((event) => {
                            const style = getPegasusSeverityStyle(event.severity)
                            const startPct = duration > 0 ? (event.start_sec / duration) * 100 : 0
                            const endPct = duration > 0 ? (event.end_sec / duration) * 100 : startPct
                            const widthPct = Math.max(1.4, endPct - startPct)
                            const isActive = currentTime >= event.start_sec && currentTime <= event.end_sec
                            const isFocused = focusedPegasusEvent?.id === event.id
                            const bookmarked = pegasusBookmarkedEventIds.includes(event.id)
                            const action = (event.recommended_action_ids || [])
                              .map((id) => pegasusActionById[id])
                              .find(Boolean)
                            return (
                              <button
                                key={`pegasus-event-${event.id}`}
                                type="button"
                                onClick={(clickEvent) => {
                                  clickEvent.stopPropagation()
                                  setActiveTool('pegasus')
                                  setRightSidebarOpen(true)
                                  setPegasusFocusedEventId(event.id)
                                  seekToTime(event.start_sec, { play: false })
                                }}
                                className="absolute inset-y-0 z-[3] bg-transparent"
                                style={{
                                  left: `${Math.max(0, Math.min(100, startPct))}%`,
                                  width: `${Math.min(100, Math.max(widthPct, 2))}%`,
                                }}
                                title={`${formatPegasusLabel(event.category)} - ${event.label} (${fmtShort(event.start_sec)} - ${fmtShort(event.end_sec)}): ${event.reason || action?.reason || 'Review this moment.'}`}
                              >
                                <span
                                  className="absolute inset-y-2 rounded-full transition-all"
                                  style={{
                                    left: 0,
                                    right: 0,
                                    backgroundColor: isActive || isFocused ? style.soft : `${style.color}20`,
                                    border: `1px solid ${bookmarked ? 'rgba(255,255,255,0.66)' : style.border}`,
                                    boxShadow: isActive || isFocused ? `0 0 0 1px ${style.color}55, 0 0 18px ${style.color}28` : undefined,
                                  }}
                                />
                              </button>
                            )
                          })}
                          {focusedPegasusEvent && (
                            <div className="absolute bottom-2 right-3 z-[5] max-w-[18rem] rounded-lg border border-border bg-surface/90 px-2.5 py-1.5 text-left shadow-lg backdrop-blur-sm pointer-events-none">
                              <p className="truncate text-[10px] font-semibold text-text-primary">{focusedPegasusEvent.label}</p>
                              <p className="mt-0.5 truncate text-[9px] text-text-tertiary">
                                {formatPegasusLabel(focusedPegasusEvent.category)} · {fmtShort(focusedPegasusEvent.start_sec)} - {fmtShort(focusedPegasusEvent.end_sec)}
                              </p>
                            </div>
                          )}
                          <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-accent/45" />
                          <div className="absolute right-0 top-0 bottom-0 w-1.5 bg-border" />
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Per-face "Blur on / Blur off" lane built from detection metadata
                      is intentionally hidden. The TwelveLabs entity search lane
                      above is the only timeline shown for blurred faces. */}
                  {SHOW_DETECTION_METADATA_FACE_LANE && personTimelineLanes.length > 0 && (
                    <div
                      className="shrink-0 relative"
                      style={{ maxHeight: personTimelineLanes.length > 3 ? 3 * 52 + 8 : undefined }}
                    >
                      <div
                        className={personTimelineLanes.length > 3 ? 'overflow-y-auto overflow-x-hidden lane-scroller' : ''}
                        style={personTimelineLanes.length > 3 ? { maxHeight: 3 * 52 + 8 } : undefined}
                      >
                        {personTimelineLanes.map((lane) => {
                          const accentColor = lane.item.color || '#F59E0B'
                          const clampedSegments = lane.segments.map((segment) => ({
                            start: Math.max(0, duration > 0 ? Math.min(duration, segment.start) : segment.start),
                            end: Math.max(0, duration > 0 ? Math.min(duration, segment.end) : segment.end),
                          })).filter((segment) => segment.end >= segment.start)
                          const activeSegmentIndex = clampedSegments.findIndex(
                            (segment) => currentTime >= segment.start && currentTime <= segment.end,
                          )

                          return (
                            <div
                              key={`face-lane-${lane.personId}`}
                              className={`h-[52px] shrink-0 relative border-b border-border ${lane.active ? '' : 'opacity-70'}`}
                            >
                              <div
                                className="absolute inset-x-0 inset-y-1.5 rounded-lg overflow-hidden shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]"
                                style={{
                                  border: `1px solid ${lane.active ? `${accentColor}42` : 'rgba(148,163,184,0.22)'}`,
                                  background: lane.active
                                    ? `linear-gradient(90deg, ${accentColor}14 0%, rgba(255,255,255,0.02) 48%, ${accentColor}10 100%)`
                                    : 'linear-gradient(90deg, rgba(148,163,184,0.08) 0%, rgba(148,163,184,0.04) 100%)',
                                }}
                              >
                                <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-white/[0.03] via-transparent to-black/[0.08]" />
                                <div
                                  className="absolute inset-x-3 bottom-[9px] h-px"
                                  style={{ backgroundColor: lane.active ? `${accentColor}30` : 'rgba(148,163,184,0.18)' }}
                                />
                                <div className="absolute left-3 top-2 right-3 z-[4] flex items-center justify-between gap-2 pointer-events-none">
                                  <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-black/22 px-2 py-1 shadow-[0_1px_0_rgba(255,255,255,0.04)] backdrop-blur-sm">
                                    <div className="h-6 w-6 shrink-0 overflow-hidden rounded-full border border-white/12 bg-white/10">
                                      {lane.item.snapBase64 ? (
                                        <img
                                          src={`data:image/png;base64,${lane.item.snapBase64}`}
                                          alt=""
                                          className="h-full w-full object-cover"
                                        />
                                      ) : (
                                        <div className="flex h-full w-full items-center justify-center text-[10px] font-semibold text-white/82">
                                          {lane.item.label.charAt(0).toUpperCase()}
                                        </div>
                                      )}
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-1.5 pointer-events-auto">
                                    <span className={`rounded-full border px-2 py-1 text-[9px] font-medium uppercase tracking-[0.12em] ${
                                      lane.active
                                        ? 'border-white/10 bg-black/24 text-white/78'
                                        : 'border-border bg-background/70 text-text-tertiary'
                                    }`}>
                                      {lane.active ? 'Blur on' : 'Blur off'}
                                    </span>
                                    <button
                                      type="button"
                                      onClick={(event) => {
                                        event.stopPropagation()
                                        removeFaceTimelineLane(lane.personId)
                                      }}
                                      className="rounded-full border border-white/12 bg-black/24 px-2 py-1 text-[9px] font-medium uppercase tracking-[0.12em] text-white/78 hover:bg-black/38"
                                      title="Remove lane"
                                    >
                                      Remove
                                    </button>
                                  </div>
                                </div>
                                {clampedSegments.length > 0 ? clampedSegments.map((segment, index) => {
                                  const startPct = duration > 0 ? (segment.start / duration) * 100 : 0
                                  const endPct = duration > 0 ? (segment.end / duration) * 100 : 0
                                  const widthPct = Math.max(1.8, endPct - startPct)
                                  const isActive = activeSegmentIndex === index
                                  return (
                                    <button
                                      key={`face-lane-segment-${lane.personId}-${segment.start}-${segment.end}-${index}`}
                                      type="button"
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        seekToTime(segment.start)
                                      }}
                                      className="absolute inset-y-0 z-[3] bg-transparent"
                                      style={{
                                        left: `${Math.max(0, Math.min(100, startPct))}%`,
                                        width: `${Math.max(widthPct, 2)}%`,
                                      }}
                                      title={`${fmtShort(segment.start)} - ${fmtShort(segment.end)}`}
                                    >
                                      <span
                                        className="absolute inset-y-2 rounded-full transition-all"
                                        style={{
                                          left: 0,
                                          right: 0,
                                          backgroundColor: isActive ? `${accentColor}34` : `${accentColor}1f`,
                                          boxShadow: isActive ? `0 0 0 1px ${accentColor}55` : undefined,
                                          opacity: isActive ? 1 : lane.active ? 0.94 : 0.58,
                                        }}
                                      />
                                    </button>
                                  )
                                }) : (
                                  <div className="absolute inset-0 z-[1] flex items-center justify-center pt-2">
                                    <span className="rounded-full border border-white/10 bg-black/24 px-2.5 py-1 text-[10px] font-medium text-white/68">
                                      Waiting for saved face ranges
                                    </span>
                                  </div>
                                )}
                                <div
                                  className="absolute left-0 top-0 bottom-0 w-1.5"
                                  style={{ backgroundColor: lane.active ? `${accentColor}cc` : 'rgba(148,163,184,0.44)' }}
                                />
                                <div
                                  className="absolute right-0 top-0 bottom-0 w-1.5"
                                  style={{ backgroundColor: lane.active ? `${accentColor}70` : 'rgba(148,163,184,0.22)' }}
                                />
                              </div>
                            </div>
                          )
                        })}
                      </div>
                      {personTimelineLanes.length > 3 && (
                        <div className="absolute bottom-0 inset-x-0 h-5 pointer-events-none bg-gradient-to-t from-background/80 to-transparent z-10" />
                      )}
                    </div>
                  )}

                  {SHOW_AUDIO_WAVEFORM && (
                    <div className={`h-[48px] shrink-0 relative border-b border-border ${trackMuted.audio ? 'opacity-40' : ''}`}>
                      <div className="absolute inset-x-0 inset-y-1.5 rounded-lg overflow-hidden border border-highlight/25 bg-gradient-to-r from-highlight/[0.14] via-surface to-highlight/[0.14] shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                        <div className="absolute inset-0 bg-gradient-to-b from-white/[0.03] via-transparent to-highlight/[0.05] pointer-events-none" />
                        <div className="absolute inset-x-3 top-1/2 h-px -translate-y-1/2 bg-highlight/15" />
                        {audioWaveformPath ? (
                          <>
                            <svg className="absolute inset-0 h-full w-full z-[1]" viewBox="0 0 1200 84" preserveAspectRatio="none" aria-hidden>
                              <path d={audioWaveformPath} fill="rgba(0, 220, 130, 0.16)" stroke="rgba(64, 230, 164, 0.42)" strokeWidth="1.2" />
                            </svg>
                            <svg
                              className="absolute inset-0 h-full w-full z-[2]"
                              viewBox="0 0 1200 84"
                              preserveAspectRatio="none"
                              aria-hidden
                              style={{ clipPath: `inset(0 ${Math.max(0, 100 - progress * 100)}% 0 0)` }}
                            >
                              <path d={audioWaveformPath} fill="rgba(117, 255, 194, 0.34)" stroke="rgba(133, 255, 204, 0.98)" strokeWidth="1.4" />
                            </svg>
                          </>
                        ) : (
                          <div className="absolute inset-0 z-[1] flex items-center justify-center">
                            <span className="rounded-full border border-highlight/20 bg-surface/80 px-2.5 py-1 text-[10px] font-medium text-text-tertiary">
                              No waveform available for this source yet
                            </span>
                          </div>
                        )}
                        <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-highlight/70" />
                        <div className="absolute right-0 top-0 bottom-0 w-1.5 bg-highlight/40" />
                      </div>
                      {trackLocked.audio && (
                        <div className="absolute inset-0 bg-background/40 rounded-lg flex items-center justify-center">
                          <svg className="w-4 h-4 text-text-tertiary" viewBox="0 0 24 24" fill="currentColor"><path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zM9 8V6c0-1.66 1.34-3 3-3s3 1.34 3 3v2H9z" /></svg>
                        </div>
                      )}
                    </div>
                  )}

                  {majorTicks.map((sec) => (
                    <div key={`g-${sec}`} className="absolute pointer-events-none bg-border/30" style={{ left: `${duration > 0 ? (sec / duration) * 100 : 0}%`, top: 36, bottom: 0, width: 1 }} />
                  ))}

                  <div className="absolute top-0 h-9 pointer-events-none bg-border/20 rounded-sm" style={{ left: 0, width: `${bufferedPct * 100}%` }} />

                  <div className="absolute top-0 bottom-0 pointer-events-none z-30" style={{ left: `${progress * 100}%` }}>
                    <svg className="absolute top-0 left-1/2 -translate-x-1/2" width="14" height="14" viewBox="0 0 14 14" fill="none">
                      <path d="M7 0L11 5.5H8V14H6V5.5H3L7 0Z" fill="#00DC82" />
                    </svg>
                    <div className="absolute top-[12px] bottom-0 left-1/2 -translate-x-1/2 w-px bg-accent shadow-[0_0_12px_rgba(0,220,130,0.45)]" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ============ RIGHT SIDEBAR (collapsible) ============ */}
        <aside className={`shrink-0 flex flex-col min-h-0 border-l border-border bg-surface overflow-hidden transition-[width] duration-200 ${rightSidebarOpen ? 'w-80' : 'w-10'}`}>
          {rightSidebarOpen ? (
            <>
              {activeTool === 'captions' ? (
                /* Analyze sidebar — TwelveLabs UI */
                <>
                  <div className="px-3 h-10 flex items-center justify-between border-b border-border shrink-0">
                    <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">Analyze</span>
                    <button type="button" onClick={() => setRightSidebarOpen(false)} className={`h-7 w-7 rounded-md ${btnBase}`} aria-label="Collapse sidebar" title="Collapse sidebar">
                      <IconChevronRight className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Overview tags — collapsible; only when overview has been generated */}
                  {!summaryLoading && summaryTags && (
                    <div className="border-b border-border shrink-0">
                      <button
                        type="button"
                        onClick={() => setOverviewTagsExpanded((e) => !e)}
                        className="w-full px-3 pt-3 pb-2 flex items-center justify-between text-left hover:bg-surface/50 transition-colors rounded-t-lg"
                      >
                        <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">Overview</span>
                        <span className="text-text-tertiary transition-transform shrink-0" style={{ transform: overviewTagsExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}>
                          <IconChevronRight className="w-4 h-4" />
                        </span>
                      </button>
                      {overviewTagsExpanded && (
                        <div className="px-3 pb-3 pt-0 space-y-3">
                          {summaryTags.about && (
                            <div className="space-y-1.5">
                              <p className="text-xs font-medium text-text-tertiary uppercase tracking-wider flex items-center gap-1.5">
                                <IconAbout className="w-3.5 h-3.5 opacity-70" />
                                About
                              </p>
                              <p className="text-sm text-text-secondary leading-relaxed">{summaryTags.about}</p>
                            </div>
                          )}
                          {summaryTags.topics && summaryTags.topics.length > 0 && (
                            <div className="space-y-1.5">
                              <p className="text-xs font-medium text-text-tertiary uppercase tracking-wider flex items-center gap-1.5">
                                <IconTopics className="w-3.5 h-3.5 opacity-70" />
                                Topics
                              </p>
                              <div className="flex flex-wrap gap-1.5">
                                {summaryTags.topics.map((tag) => (
                                  <span key={tag} className="inline-flex items-center px-2 py-1 rounded-md text-sm font-medium bg-accent/15 text-accent border border-accent/30">
                                    {tag}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          {summaryTags.categories && summaryTags.categories.length > 0 && (
                            <div className="space-y-1.5">
                              <p className="text-xs font-medium text-text-tertiary uppercase tracking-wider flex items-center gap-1.5">
                                <IconCategories className="w-3.5 h-3.5 opacity-70" />
                                Categories
                              </p>
                              <div className="flex flex-wrap gap-1.5">
                                {summaryTags.categories.map((tag) => (
                                  <span key={tag} className="inline-flex items-center px-2 py-1 rounded-md text-sm font-medium bg-card border border-border text-text-secondary">
                                    {tag}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Overview CTA — only when overview not yet generated (About is shown above once done) */}
                  {!summaryTags && (
                    <div className="shrink-0 p-3">
                      <div className="rounded-lg border border-border bg-card overflow-hidden">
                        {summaryLoading ? (
                          <div className="px-3 py-2.5">
                            <p className="text-sm text-text-tertiary">Generating overview…</p>
                          </div>
                        ) : (
                          <div className="px-3 py-2.5 space-y-2">
                            <p className="text-sm text-text-tertiary">Generate a short overview of this video.</p>
                            <button
                              type="button"
                              onClick={runGenerateSummary}
                              disabled={!videoId}
                              className="w-full h-8 rounded-md text-sm font-medium bg-accent text-white border border-accent hover:bg-accent-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                              Generate overview
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Chat — analyze Q&A with markdown responses */}
                  <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
                    <div className="flex-1 overflow-y-auto p-3 space-y-3">
                      {analyzeMessages.length === 0 && !analyzeLoading && (
                        <p className="text-sm text-text-tertiary">Ask a question about this video below.</p>
                      )}
                      {analyzeMessages.map((msg) => (
                        <div key={msg.id} className={msg.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
                          <div
                            className={`max-w-[92%] rounded-lg px-3 py-2 text-sm ${
                              msg.role === 'user'
                                ? 'bg-accent text-white'
                                : 'bg-card border border-border text-text-secondary'
                            }`}
                          >
                            {msg.role === 'user' ? (
                              <p className="whitespace-pre-wrap">{msg.content}</p>
                            ) : (
                              <div className={markdownWrapClass}>
                                <ReactMarkdown components={markdownComponents}>{msg.content}</ReactMarkdown>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                      {analyzeLoading && (
                        <div className="flex justify-start">
                          <div className="rounded-lg px-3 py-2 bg-card border border-border text-text-tertiary text-sm">
                            Analyzing…
                          </div>
                        </div>
                      )}
                      <div ref={analyzeChatEndRef} />
                    </div>
                    {analyzeError && (
                      <p className="px-3 text-sm text-error shrink-0">{analyzeError}</p>
                    )}
                    <div className="p-3 border-t border-border shrink-0">
                      <div className="flex gap-1.5">
                        <input
                          type="text"
                          value={analyzeQuery}
                          onChange={(e) => setAnalyzeQuery(e.target.value)}
                          onFocus={() => setOverviewTagsExpanded(false)}
                          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && runAnalyze()}
                          placeholder="Ask about this video..."
                          disabled={!videoId}
                          className="flex-1 h-9 rounded-md bg-surface border border-border px-3 text-sm text-text-primary placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent disabled:opacity-60 disabled:cursor-not-allowed"
                        />
                        <div className="relative shrink-0 z-20">
                          <button
                            type="button"
                            onClick={() => setAnalyzeSuggestionsOpen((open) => !open)}
                            className="h-9 px-2.5 inline-flex items-center gap-1 rounded-md bg-surface border border-border text-xs font-medium text-text-secondary hover:bg-card hover:text-text-primary transition-colors"
                            aria-haspopup="true"
                            aria-expanded={analyzeSuggestionsOpen}
                          >
                            Suggestions
                            <IconChevronDown className={`w-3 h-3 transition-transform ${analyzeSuggestionsOpen ? 'rotate-180' : ''}`} />
                          </button>
                          {analyzeSuggestionsOpen && (
                            <div className="absolute right-0 bottom-full mb-1 py-1 min-w-[14rem] max-w-xs rounded-lg border border-border bg-surface shadow-lg z-30">
                              {ANALYZE_SUGGESTIONS.map((suggestion) => (
                                <button
                                  key={suggestion}
                                  type="button"
                                  onClick={() => {
                                    setAnalyzeQuery(suggestion)
                                    setAnalyzeSuggestionsOpen(false)
                                  }}
                                  className="w-full px-3 py-1.5 text-left text-sm text-text-secondary hover:bg-card hover:text-text-primary transition-colors"
                                >
                                  {suggestion}
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                        <button
                          type="button"
                          onClick={runAnalyze}
                          disabled={!videoId || analyzeLoading || !analyzeQuery.trim()}
                          className="h-9 w-9 shrink-0 rounded-md inline-flex items-center justify-center bg-accent text-white border border-accent hover:bg-accent-hover transition-colors focus:outline-none focus:ring-2 focus:ring-accent/30 disabled:opacity-50 disabled:cursor-not-allowed"
                          aria-label="Analyze"
                          title="Analyze"
                        >
                          {analyzeLoading ? (
                            <span className="w-4 h-4 border-2 border-white/60 border-t-white rounded-full animate-spin" aria-hidden />
                          ) : (
                            <img src={metaInsightsIconUrl} alt="" className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                </>
              ) : activeTool === 'search' ? (
                <>
                  <div className="px-3 h-10 flex items-center justify-between border-b border-border shrink-0">
                    <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">Search</span>
                    <div className="flex items-center gap-1.5">
                      {searchTimelineLanes.length > 0 && (
                        <button
                          type="button"
                          onClick={clearAllSearchTimelineLanes}
                          className="h-7 rounded-md border border-border bg-card px-2 text-[10px] font-medium uppercase tracking-wide text-text-tertiary transition-colors hover:bg-background hover:text-text-primary"
                        >
                          Clear all
                        </button>
                      )}
                      <button type="button" onClick={() => setRightSidebarOpen(false)} className={`h-7 w-7 shrink-0 rounded-md ${btnBase}`} aria-label="Collapse sidebar" title="Collapse sidebar">
                        <IconChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  {searchResultForVideo ? (
                    <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
                      <div className="p-3 border-b border-border shrink-0 space-y-3">
                        {searchTimelineLanes.length > 1 && (
                          <div>
                            <p className="text-[10px] font-medium text-text-tertiary uppercase tracking-wider">Stacked searches</p>
                            <div className="mt-2 flex flex-col gap-1.5">
                              {searchTimelineLanes.map((lane) => {
                                const isSelected = activeSearchTimelineLane?.sessionId === lane.sessionId
                                const label = lane.entities.map((entity) => entity.name).join(', ') || lane.session.queryText || lane.session.query || 'Search'
                                const previewEntities = lane.entities.slice(0, 3)
                                const extraEntityCount = Math.max(0, lane.entities.length - previewEntities.length)
                                return (
                                  <button
                                    key={`sidebar-search-stack-${lane.sessionId}`}
                                    type="button"
                                    onClick={() => activateSearchSession(lane.sessionId)}
                                    className={`flex items-center justify-between gap-2 rounded-lg border px-2.5 py-2 text-left text-xs transition-colors ${
                                      isSelected
                                        ? 'border-accent/35 bg-accent/10 text-accent'
                                        : 'border-border bg-card text-text-secondary hover:bg-background hover:text-text-primary'
                                    }`}
                                  >
                                    <div className="flex min-w-0 items-center gap-2">
                                      {previewEntities.length > 0 && (
                                        <div className="flex shrink-0 -space-x-1.5">
                                          {previewEntities.map((entity) => (
                                            <div
                                              key={`sidebar-search-stack-${lane.sessionId}-entity-${entity.id}`}
                                              className={`h-6 w-6 shrink-0 overflow-hidden rounded-full border bg-surface ring-1 ${
                                                isSelected ? 'border-accent/50 ring-accent/20' : 'border-border ring-background'
                                              }`}
                                              title={entity.name}
                                            >
                                              {entity.previewUrl ? (
                                                <img
                                                  src={entity.previewUrl}
                                                  alt={entity.name}
                                                  className="h-full w-full object-cover"
                                                />
                                              ) : (
                                                <div className="flex h-full w-full items-center justify-center text-[10px] font-semibold text-text-tertiary">
                                                  {getEntityMonogram(entity.name)}
                                                </div>
                                              )}
                                            </div>
                                          ))}
                                          {extraEntityCount > 0 && (
                                            <div
                                              className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full border bg-surface ring-1 text-[9px] font-semibold ${
                                                isSelected
                                                  ? 'border-accent/50 ring-accent/20 text-accent'
                                                  : 'border-border ring-background text-text-tertiary'
                                              }`}
                                              title={`${extraEntityCount} more entit${extraEntityCount === 1 ? 'y' : 'ies'}`}
                                            >
                                              +{extraEntityCount}
                                            </div>
                                          )}
                                        </div>
                                      )}
                                      <span className="min-w-0 truncate">{label}</span>
                                    </div>
                                    <span className="shrink-0 rounded-full border border-current/20 px-1.5 py-0.5 text-[10px]">
                                      {lane.orderedClips.length}
                                    </span>
                                  </button>
                                )
                              })}
                            </div>
                          </div>
                        )}
                        {searchEntities.length > 0 && (
                          <div>
                            <p className="text-[10px] font-medium text-text-tertiary uppercase tracking-wider">Entity</p>
                            <div className="mt-2 flex flex-wrap gap-2">
                              {searchEntities.map((entity) => (
                                <SearchEntityChip
                                  key={entity.id}
                                  entity={entity}
                                />
                              ))}
                            </div>
                          </div>
                        )}
                        {searchSessionResult?.queryText && (
                          <div>
                            <p className="text-[10px] font-medium text-text-tertiary uppercase tracking-wider">Query</p>
                            <p className="mt-1 text-xs leading-relaxed text-text-primary">
                              {searchSessionResult.queryText}
                            </p>
                          </div>
                        )}
                        {!searchEntities.length && !searchSessionResult?.queryText && (
                          <div>
                            <p className="text-[10px] font-medium text-text-tertiary uppercase tracking-wider">Search</p>
                            <p className="mt-1 text-xs leading-relaxed text-text-primary">
                              {searchSessionResult?.query || 'Latest search'}
                            </p>
                          </div>
                        )}
                        <div className="flex items-center justify-between gap-2 text-xs">
                          <span className="text-text-tertiary">Matched clips</span>
                          <span className="text-text-primary">
                            {orderedSearchClips.length} clip{orderedSearchClips.length === 1 ? '' : 's'}
                          </span>
                        </div>
                        {activeSearchRank != null && (
                          <button
                            type="button"
                            onClick={jumpToActiveSearchRank}
                            className="inline-flex w-full items-center justify-between rounded-lg border border-border bg-card px-3 py-2 text-left text-xs text-text-primary transition-colors hover:bg-background"
                          >
                            <span>Jump to current rank</span>
                            <span className="rounded-full border border-accent/30 bg-accent/10 px-2 py-0.5 text-[10px] font-semibold text-accent">
                              Rank {activeSearchRank}
                            </span>
                          </button>
                        )}
                      </div>
                      <div className="flex-1 min-h-0 overflow-y-auto">
                        {orderedSearchClips.length > 0 ? orderedSearchClips.map((clip, index) => {
                          const rank = clip.rank ?? index + 1
                          const isActive = activeSearchClipIndex === index
                          return (
                            <button
                              ref={(node) => {
                                searchClipRowRefs.current[index] = node
                              }}
                              key={`${clip.start}-${clip.end}-${rank}-${index}`}
                              type="button"
                              onClick={() => seekToTime(clip.start)}
                              className={`w-full border-b border-l-2 px-3 py-3 text-left transition-colors ${isActive ? 'border-l-accent bg-accent/10' : 'border-l-transparent hover:bg-card'}`}
                            >
                              <div className="flex items-start gap-3">
                                <div className={`relative h-12 w-20 shrink-0 overflow-hidden rounded-md border bg-card ${isActive ? 'border-accent/50 shadow-[0_0_0_1px_rgba(0,220,130,0.14)]' : 'border-border'}`}>
                                  {clip.thumbnailUrl ? (
                                    <img
                                      src={clip.thumbnailUrl}
                                      alt=""
                                      className="h-full w-full object-cover"
                                    />
                                  ) : (
                                    <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-card to-surface text-[10px] font-medium text-text-tertiary">
                                      {fmtShort(clip.start)}
                                    </div>
                                  )}
                                </div>
                                <div className="min-w-0 flex-1">
                                  <div className="flex items-center justify-between gap-2">
                                    <p className={`truncate text-sm ${isActive ? 'text-accent' : 'text-text-primary'}`}>
                                      {fmtShort(clip.start)} - {fmtShort(clip.end)}
                                    </p>
                                    <div className="flex shrink-0 items-center gap-2">
                                      <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${isActive ? 'border-accent/40 bg-accent/15 text-accent' : 'border-border bg-card text-text-secondary'}`}>
                                        Rank {rank}
                                      </span>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </button>
                          )
                        }) : (
                          <div className="px-3 py-6">
                            <p className="text-xs text-text-tertiary">No clips were returned for this search result.</p>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="flex-1 min-h-0 flex items-center justify-center p-4">
                      <p className="max-w-[220px] text-center text-xs leading-relaxed text-text-tertiary">
                        Run a search from the dashboard, then open a result video to see matched clips and rank here.
                      </p>
                    </div>
                  )}
                </>
              ) : activeTool === 'pegasus' ? (
                <>
                  <div className="px-3 h-10 flex items-center justify-between border-b border-border shrink-0">
                    <div className="min-w-0 flex items-center gap-2">
                      <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">Meta Insights</span>
                      {pegasusResult && (
                        <span className="rounded-full border border-border bg-card px-1.5 py-0.5 text-[10px] tabular-nums text-text-tertiary">
                          {filteredPegasusEvents.length}
                        </span>
                      )}
                    </div>
                    <button type="button" onClick={() => setRightSidebarOpen(false)} className={`h-7 w-7 rounded-md ${btnBase}`} aria-label="Collapse sidebar" title="Collapse sidebar">
                      <IconChevronRight className="w-4 h-4" />
                    </button>
                  </div>

                  <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
                    <div className="shrink-0 border-b border-border p-3 space-y-3 bg-surface/40">
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => runPegasusAssist(false)}
                          disabled={!videoId || pegasusLoading}
                          className="h-9 flex-1 rounded-md border border-accent bg-accent px-3 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-60"
                        >
                          {pegasusLoading ? 'Analyzing...' : pegasusResult ? 'Re-run Meta Insights' : 'Run Meta Insights'}
                        </button>
                        <button
                          type="button"
                          onClick={() => runPegasusAssist(true)}
                          disabled={!videoId || pegasusLoading}
                          className="h-9 rounded-md border border-border bg-card px-3 text-xs font-medium text-text-secondary transition-colors hover:bg-surface hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-60"
                        >
                          Refresh
                        </button>
                      </div>

                      {pegasusError && (
                        <p className="rounded-md border border-red-400/25 bg-red-400/10 px-2.5 py-2 text-xs leading-relaxed text-red-300">
                          {pegasusError}
                        </p>
                      )}

                      {pegasusStatus !== 'ready' && !pegasusResult && (
                        <div className="rounded-lg border border-border bg-card/80 p-3">
                          <p className="text-xs leading-relaxed text-text-tertiary">
                            Finds only the verdict subject, protected people, and sensitive details that need redaction review.
                          </p>
                        </div>
                      )}

                      {pegasusResult && (
                        <div className="space-y-3">
                          <div className="grid grid-cols-3 gap-2">
                            <div className="rounded-lg border border-border bg-card px-2.5 py-2 text-center">
                              <p className="text-[10px] uppercase tracking-wide text-text-tertiary">High risk</p>
                              <p className="mt-1 text-base font-semibold tabular-nums text-text-primary">{pegasusHighEventCount}</p>
                            </div>
                            <div className="rounded-lg border border-border bg-card px-2.5 py-2 text-center">
                              <p className="text-[10px] uppercase tracking-wide text-text-tertiary">Review only</p>
                              <p className="mt-1 text-base font-semibold tabular-nums text-text-primary">{pegasusReviewOnlyCount}</p>
                            </div>
                            <div className="rounded-lg border border-border bg-card px-2.5 py-2 text-center">
                              <p className="text-[10px] uppercase tracking-wide text-text-tertiary">Auto-match</p>
                              <p className="mt-1 text-base font-semibold tabular-nums text-text-primary">{pegasusAutomaticActionCount}</p>
                            </div>
                          </div>

                          <div className="space-y-2 rounded-lg border border-border bg-card/80 p-2.5">
                            <div className="flex items-center justify-between">
                              <span className="text-[10px] font-medium uppercase tracking-wider text-text-tertiary">Filters</span>
                              <span className="text-[10px] text-text-tertiary tabular-nums">{filteredPegasusEvents.length}/{pegasusEvents.length}</span>
                            </div>
                            <div className="flex flex-wrap gap-1.5">
                              <button
                                type="button"
                                onClick={() => setPegasusCategoryFilter('all')}
                                aria-pressed={pegasusCategoryFilter === 'all'}
                                className={`inline-flex h-7 items-center gap-1.5 rounded-full border px-2.5 text-[10px] font-semibold uppercase tracking-wide transition-colors ${
                                  pegasusCategoryFilter === 'all'
                                    ? 'border-accent/40 bg-accent/10 text-accent'
                                    : 'border-border bg-card text-text-secondary hover:border-border/80 hover:text-text-primary'
                                }`}
                              >
                                <IconFilter className="h-3.5 w-3.5" />
                                <span>All</span>
                              </button>
                              {pegasusCategories.map((category) => (
                                <button
                                  key={category}
                                  type="button"
                                  onClick={() => setPegasusCategoryFilter(category)}
                                  aria-pressed={pegasusCategoryFilter === category}
                                  className={`inline-flex h-7 items-center gap-1.5 rounded-full border px-2.5 text-[10px] font-semibold uppercase tracking-wide transition-colors ${
                                    pegasusCategoryFilter === category
                                      ? 'border-accent/40 bg-accent/10 text-accent'
                                      : 'border-border bg-card text-text-secondary hover:border-border/80 hover:text-text-primary'
                                  }`}
                                  title={formatPegasusLabel(category)}
                                >
                                  <IconPegasusCategory category={category} className="h-3.5 w-3.5" />
                                  <span>{formatPegasusLabel(category)}</span>
                                </button>
                              ))}
                            </div>
                            <div className="grid grid-cols-3 gap-1.5">
                              {(['high', 'medium', 'low'] as PegasusSeverity[]).map((severity) => {
                                const style = getPegasusSeverityStyle(severity)
                                return (
                                  <button
                                    key={severity}
                                    type="button"
                                    onClick={() => setPegasusSeverityFilter((previous) => ({
                                      ...previous,
                                      [severity]: !previous[severity],
                                    }))}
                                    className={`h-7 rounded-md border text-[10px] font-semibold uppercase tracking-wide transition-opacity ${
                                      pegasusSeverityFilter[severity] ? '' : 'opacity-50'
                                    } ${pegasusSeverityFilter[severity] ? '' : 'border-border bg-card'}`}
                                    style={pegasusSeverityFilter[severity]
                                      ? { borderColor: style.border, backgroundColor: style.soft }
                                      : undefined}
                                  >
                                    <span className={style.text}>{severity}</span>
                                  </button>
                                )
                              })}
                            </div>
                          </div>

                        </div>
                      )}
                    </div>

                    {pegasusResult ? (
                      <div className="flex-1 min-h-0 overflow-y-auto">
                        {renderPegasusEventGroup('High priority', pegasusEventsByGroup.high)}
                        {renderPegasusEventGroup('People / faces', pegasusEventsByGroup.people)}
                        {renderPegasusEventGroup('Screens / documents / text', pegasusEventsByGroup.sensitive)}
                        {renderPegasusEventGroup('Objects / logos / scenes', pegasusEventsByGroup.objects)}
                        {filteredPegasusEvents.length === 0 && (
                          <div className="flex min-h-[9rem] items-center justify-center px-4 py-8">
                            <p className="max-w-[220px] text-center text-xs leading-relaxed text-text-tertiary">
                              No Pegasus hotspots match the current filters.
                            </p>
                          </div>
                        )}
                      </div>
                    ) : (
	                      <div className="flex-1 min-h-0 flex items-center justify-center p-4">
	                        <p className="max-w-[220px] text-center text-xs leading-relaxed text-text-tertiary">
	                          Run Meta Insights to show only redaction-worthy moments on the timeline.
	                        </p>
	                      </div>
                    )}
                  </div>
                </>
              ) : (
            /* Detection sidebar (Tracker or Detection selected) — same flex pattern as Analyze so video player middle behaves identically */
            <>
              <div className="px-3 h-10 flex items-center justify-between border-b border-border shrink-0 gap-1">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider truncate min-w-0">Detections</span>
                  {hasRunDetection && (
                    <span className="text-xs text-text-tertiary tabular-nums shrink-0">({filteredDetections.length})</span>
                  )}
                </div>
                <div className="ml-auto flex shrink-0 items-center gap-1.5">
                  {hasRunDetection && (
                    <button
                      type="button"
                      role="switch"
                      aria-checked={showAnonymizeOnly}
                      aria-label={showAnonymizeOnly ? 'Show all detections' : 'Show anonymize tagged detections only'}
                      title={showAnonymizeOnly ? 'Show all detections' : 'Show anonymize tagged detections only'}
                      onClick={() => setShowAnonymizeOnly((current) => !current)}
                      className={`inline-flex h-7 max-w-[7.75rem] shrink-0 items-center gap-1.5 overflow-hidden rounded-md border px-1.5 pr-2 text-[10px] font-medium transition-colors ${
                        showAnonymizeOnly
                          ? 'border-accent/35 bg-accent/10 text-accent shadow-[0_0_0_1px_rgba(0,220,130,0.08)]'
                          : 'border-border bg-card text-text-tertiary hover:border-accent/30 hover:bg-background hover:text-text-secondary'
                      }`}
                    >
                      <span
                        aria-hidden
                        className={`relative h-3.5 w-6 shrink-0 rounded-full border transition-colors ${
                          showAnonymizeOnly ? 'border-accent/35 bg-accent/20' : 'border-border bg-surface'
                        }`}
                      >
                        <span
                          className={`absolute left-0.5 top-1/2 h-2.5 w-2.5 -translate-y-1/2 rounded-full bg-current shadow-sm transition-transform ${
                            showAnonymizeOnly ? 'translate-x-2.5' : 'translate-x-0'
                          }`}
                        />
                      </span>
                      <span className="min-w-0 truncate">Anon only</span>
                      <span className={`rounded-sm px-1 tabular-nums ${
                        showAnonymizeOnly ? 'bg-accent/15 text-accent' : 'bg-surface text-text-tertiary'
                      }`}>
                        {anonymizeTargetCount}
                      </span>
                    </button>
                  )}
                  <button type="button" onClick={() => setRightSidebarOpen(false)} className={`h-7 w-7 shrink-0 rounded-md ${btnBase}`} aria-label="Collapse sidebar" title="Collapse sidebar">
                    <IconChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
              {hasRunDetection ? (
                <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
                  <div className="p-2 border-b border-border shrink-0">
                    <input
                      type="search"
                      placeholder="Filter detections..."
                      value={detectionFilter}
                      onChange={(e) => setDetectionFilter(e.target.value)}
                      className="w-full h-8 rounded-md bg-surface border border-border px-3 text-xs text-text-primary placeholder:text-text-tertiary"
                    />
                    <p className="mt-2 text-[10px] leading-relaxed text-text-tertiary">
                      Use Blur or Unblur to decide exactly which saved faces and objects should stay redacted for this video. The preview follows the selected saved targets, and if a saved target is not on the current frame the editor can jump to a nearby occurrence so you can verify the blur.
                    </p>
                  </div>
                  <div className="flex-1 min-h-0 overflow-y-auto">
                    {filteredDetections.length > 0 ? filteredDetections.map((d) => {
                      const excluded = excludedFromRedactionIds.includes(d.id)
                      const visibleNow = isDetectionItemLikelyVisibleAtTime(d, currentTime)
                      const nearestSeekTime = visibleNow ? null : getDetectionItemSeekTime(d, currentTime)
                      return (
                        <div
                          key={d.id}
                          className={`flex items-center gap-2.5 px-3 py-2 hover:bg-card border-b border-border-light transition-colors ${excluded ? 'opacity-60' : ''}`}
                        >
                          <div
                            className="w-8 h-8 rounded-md shrink-0 flex items-center justify-center overflow-hidden text-[9px] font-medium text-white bg-card border border-border"
                            style={d.snapBase64 ? undefined : { backgroundColor: d.color + '30', border: `1px solid ${d.color}50` }}
                          >
                            {d.snapBase64 ? (
                              <img src={`data:image/png;base64,${d.snapBase64}`} alt="" className="w-full h-full object-cover" />
                            ) : (
                              <span style={{ color: d.color }}>{d.label.charAt(0)}</span>
                            )}
                          </div>
                          <div className="min-w-0 flex-1">
                            <p className="text-sm text-text-primary truncate">{d.label}</p>
                            {(() => {
                              const personId = d.kind === 'face' ? d.personId || null : null
                              if (!personId || excluded) return null
                              const lane = faceLockLanesByPersonId[personId]
                              const buildState = faceLockBuildByPersonId[personId]
                              if (lane) {
                                return null
                              }
                              if (buildState && (buildState.status === 'queued' || buildState.status === 'running')) {
                                return (
                                  <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                                    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-medium bg-accent/10 border border-accent/30 text-accent">
                                      <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
                                      Locking onto face... {buildState.percent || 0}%
                                    </span>
                                  </div>
                                )
                              }
                              if (buildState && buildState.status === 'failed') {
                                return (
                                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium bg-amber-400/10 border border-amber-400/30 text-amber-300" title={buildState.message || ''}>
                                      Face-lock fallback active
                                    </span>
                                  </div>
                                )
                              }
                              return null
                            })()}
                            {(visibleNow || nearestSeekTime !== null) && (
                              <div className="mt-1.5 flex flex-wrap gap-1.5">
                                {visibleNow && (
                                  <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-accent/10 border border-accent/20 text-accent">
                                    On screen now
                                  </span>
                                )}
                                {!visibleNow && nearestSeekTime !== null && (
                                  <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-surface border border-border-light text-text-tertiary">
                                    Nearest at {fmtShort(nearestSeekTime)}
                                  </span>
                                )}
                              </div>
                            )}
                            {d.tags && d.tags.length > 0 && (
                              <div className="flex flex-wrap gap-1.5 mt-1.5">
                                {d.tags.map((tag) => (
                                  <span
                                    key={tag}
                                    className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium bg-surface border border-border-light text-text-tertiary tracking-wide"
                                  >
                                    {tag}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDetectionListToggle(d)
                            }}
                            className={`shrink-0 inline-flex items-center gap-1.5 h-8 px-2.5 rounded-md border text-xs font-medium transition-colors ${
                              excluded
                                ? 'border-border bg-background text-text-secondary hover:border-accent/40 hover:text-accent'
                                : 'border-accent/25 bg-accent-light text-accent hover:border-accent/40'
                            }`}
                            title={excluded ? 'Blur this item over the video again' : 'Stop blurring this item over the video'}
                            aria-label={excluded ? 'Blur this item over the video again' : 'Stop blurring this item over the video'}
                          >
                            {excluded ? <IconEyeOff className="w-3.5 h-3.5" /> : <IconEye className="w-3.5 h-3.5" />}
                            <span>{excluded ? 'Blur' : 'Unblur'}</span>
                          </button>
                        </div>
                      )
                    }) : (
                      <div className="flex h-full min-h-[8rem] items-center justify-center px-4 py-8">
                        <p className="max-w-[220px] text-center text-xs leading-relaxed text-text-tertiary">
                          {showAnonymizeOnly
                            ? 'No anonymize-tagged detections match this view.'
                            : 'No detections match this filter.'}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
                  <div className="flex-1 min-h-0 overflow-y-auto flex flex-col items-center justify-center min-h-[8rem] gap-3 p-4">
                  {detectionError && (
                    <p className="text-xs text-error text-center">{detectionError}</p>
                  )}
                  <button
                    type="button"
                    onClick={runDetect}
                    disabled={detectionLoading}
                    className="px-6 py-3 rounded-lg text-sm font-medium bg-accent text-white border border-accent hover:opacity-90 transition-opacity focus:outline-none focus:ring-2 focus:ring-accent/30 focus:ring-offset-2 disabled:opacity-60 disabled:cursor-not-allowed"
                  >
                    {detectionLoading ? 'Loading…' : 'Detect'}
                  </button>
                  <p className="text-[10px] text-text-tertiary text-center max-w-[180px]">
                    Loads the saved faces and objects for this video. Selection preview can still run from saved detection data.
                  </p>
                </div>
                </div>
              )}
            </>
          )}
            </>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center py-4 min-h-0">
              <button
                type="button"
                onClick={() => setRightSidebarOpen(true)}
                className={`h-8 w-8 rounded-md ${btnBase}`}
                aria-label="Expand sidebar"
                title="Expand sidebar"
              >
                <IconChevronLeft className="w-4 h-4" />
              </button>
            </div>
          )}
        </aside>
      </div>
      <SnapFaceFromVideoModal
        open={snapFaceModalOpen}
        onClose={() => {
          setSnapFaceModalOpen(false)
          setSnapFaceFrameDataUrl(null)
        }}
        frameDataUrl={snapFaceFrameDataUrl}
        capturedAtSec={snapFaceCapturedAtSec}
        onFaceAdded={handleSnapFaceAdded}
      />

      {redactionWarnings && (
        <RedactionWarningsModal
          warnings={redactionWarnings}
          onClose={() => setRedactionWarnings(null)}
        />
      )}

      {pegasusApplyModalOpen && pegasusApplyPreview && (
        <PegasusApplyPreviewModal
          preview={pegasusApplyPreview}
          onClose={() => setPegasusApplyModalOpen(false)}
          onConfirm={confirmPegasusApply}
        />
      )}

      {faceLockBuildAlert && (
        <FaceLockFailureToast
          label={faceLockBuildAlert.label}
          reason={faceLockBuildAlert.reason}
          onDismiss={() => setFaceLockBuildAlert(null)}
        />
      )}
    </div>
  )
}

interface RedactionWarningsModalProps {
  warnings: {
    unresolved: { person_id?: string; label?: string; reason?: string; fallback?: string }[]
    blurFailures: { person_id?: string; label?: string; reason?: string; fallback?: string }[]
    faceLockFailures: { person_id?: string; label?: string; reason?: string; fallback?: string }[]
  }
  onClose: () => void
}

function RedactionWarningsModal({ warnings, onClose }: RedactionWarningsModalProps) {
  type WarningSection = {
    title: string
    description: string
    entries: typeof warnings.unresolved
    tone: 'amber' | 'red'
  }
  const sections: WarningSection[] = ([
    {
      title: 'Faces that could not be redacted',
      description:
        'These faces appeared in the detection list but the server could not resolve a face encoding for them. They are NOT blurred in the downloaded video.',
      entries: [...warnings.unresolved, ...warnings.blurFailures],
      tone: 'red' as const,
    },
    {
      title: 'Face-lock fell back to per-frame match',
      description:
        'For these faces, the face-lock track could not be built so the redactor used the per-frame InsightFace path instead. The blur is still applied, but it may shift slightly under fast motion or sudden cuts.',
      entries: warnings.faceLockFailures,
      tone: 'amber' as const,
    },
  ] as WarningSection[]).filter((section) => section.entries.length > 0)

  return (
    <div className="fixed inset-0 z-[210] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-brand-charcoal/45 backdrop-blur-sm" onClick={onClose} aria-hidden />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="redaction-warnings-title"
        className="relative w-full max-w-md rounded-xl border border-gray-200 bg-surface shadow-xl"
      >
        <div className="flex items-center justify-between border-b border-gray-200 px-5 py-4">
          <div className="flex items-center gap-2">
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-amber-500/15 text-amber-600">
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
            </span>
            <h2 id="redaction-warnings-title" className="text-lg font-semibold text-gray-900">
              Redaction completed with warnings
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 hover:text-gray-700 transition-colors"
            aria-label="Close"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="p-5 space-y-4 max-h-[70vh] overflow-y-auto">
          {sections.length === 0 ? (
            <p className="text-sm text-gray-600">No issues to report.</p>
          ) : (
            sections.map((section) => (
              <div key={section.title} className={`rounded-xl border p-4 ${section.tone === 'red' ? 'border-red-200 bg-red-50' : 'border-amber-200 bg-amber-50'}`}>
                <p className={`text-sm font-medium ${section.tone === 'red' ? 'text-red-800' : 'text-amber-800'}`}>{section.title}</p>
                <p className={`mt-1 text-xs ${section.tone === 'red' ? 'text-red-700/85' : 'text-amber-700/85'}`}>{section.description}</p>
                <ul className="mt-3 space-y-2">
                  {section.entries.map((entry, idx) => {
                    const display = entry.label || entry.person_id || `Face ${idx + 1}`
                    return (
                      <li
                        key={`${entry.person_id || 'face'}-${idx}`}
                        className={`flex items-start gap-2 rounded-md border bg-white/70 px-3 py-2 ${section.tone === 'red' ? 'border-red-200/70' : 'border-amber-200/70'}`}
                      >
                        <span className={`mt-0.5 inline-flex h-1.5 w-1.5 shrink-0 rounded-full ${section.tone === 'red' ? 'bg-red-500' : 'bg-amber-500'}`} aria-hidden />
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-medium text-gray-900 truncate">{display}</p>
                          {entry.reason && (
                            <p className="mt-0.5 text-xs text-gray-600">{entry.reason}</p>
                          )}
                        </div>
                      </li>
                    )
                  })}
                </ul>
              </div>
            ))
          )}
        </div>

        <div className="flex justify-end border-t border-gray-200 px-5 py-3">
          <button
            type="button"
            onClick={onClose}
            className="h-8 px-3 rounded-[9.6px] text-sm font-medium bg-brand-charcoal text-brand-white hover:bg-gray-700 transition-colors"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  )
}

interface PegasusApplyPreviewModalProps {
  preview: PegasusApplyPreview
  onClose: () => void
  onConfirm: () => void
}

function PegasusApplyPreviewModal({ preview, onClose, onConfirm }: PegasusApplyPreviewModalProps) {
  const canApplyCount = preview.can_apply.length
  const reviewOnlyCount = preview.review_only.length
  const unsupportedCount = preview.unsupported.length
  const renderItems = (items: PegasusApplyPreviewItem[], emptyText: string) => (
    items.length > 0 ? (
      <div className="space-y-2">
        {items.slice(0, 5).map((item, index) => (
          <div key={`${item.action_id || item.selection_id || item.label || 'item'}-${index}`} className="rounded-lg border border-border bg-surface/70 px-3 py-2">
            <p className="truncate text-sm font-medium text-text-primary">
              {item.label || formatPegasusLabel(item.type)}
            </p>
            <p className="mt-0.5 text-[11px] text-text-tertiary">
              {formatPegasusLabel(item.type)}{item.event_ids?.length ? ` · ${item.event_ids.length} timeline item${item.event_ids.length === 1 ? '' : 's'}` : ''}
            </p>
            {item.reason && (
              <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-text-secondary">
                {item.reason}
              </p>
            )}
          </div>
        ))}
        {items.length > 5 && (
          <p className="text-[11px] text-text-tertiary">+{items.length - 5} more</p>
        )}
      </div>
    ) : (
      <p className="rounded-lg border border-border bg-surface/60 px-3 py-2 text-xs text-text-tertiary">
        {emptyText}
      </p>
    )
  )

  return (
    <div className="fixed inset-0 z-[220] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-brand-charcoal/45 backdrop-blur-sm" onClick={onClose} aria-hidden />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="pegasus-apply-preview-title"
        className="relative flex max-h-[82vh] w-full max-w-lg flex-col overflow-hidden rounded-xl border border-border bg-card shadow-xl"
      >
        <div className="flex items-start justify-between gap-3 border-b border-border px-5 py-4">
          <div>
            <p className="text-[10px] font-medium uppercase tracking-wider text-text-tertiary">Meta Insights</p>
            <h2 id="pegasus-apply-preview-title" className="mt-1 text-lg font-semibold text-text-primary">
              Review recommendations
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-2 text-text-tertiary transition-colors hover:bg-surface hover:text-text-primary"
            aria-label="Close"
          >
            <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 py-4">
          <div className="grid grid-cols-3 gap-2">
            <div className="rounded-lg border border-border bg-surface/60 px-3 py-2 text-center">
              <p className="text-base font-semibold tabular-nums text-text-primary">{canApplyCount}</p>
              <p className="text-[10px] uppercase tracking-wide text-text-tertiary">Auto-match</p>
            </div>
            <div className="rounded-lg border border-border bg-surface/60 px-3 py-2 text-center">
              <p className="text-base font-semibold tabular-nums text-text-primary">{reviewOnlyCount}</p>
              <p className="text-[10px] uppercase tracking-wide text-text-tertiary">Bookmarks</p>
            </div>
            <div className="rounded-lg border border-border bg-surface/60 px-3 py-2 text-center">
              <p className="text-base font-semibold tabular-nums text-text-primary">{unsupportedCount}</p>
              <p className="text-[10px] uppercase tracking-wide text-text-tertiary">Unsupported</p>
            </div>
          </div>

          <section>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-tertiary">Will apply</h3>
            {renderItems(preview.can_apply, 'No saved faces or object classes matched automatically.')}
          </section>

          <section>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-tertiary">Review only</h3>
            {renderItems(preview.review_only, 'No review bookmarks will be added.')}
          </section>

          {unsupportedCount > 0 && (
            <section>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-tertiary">Unsupported</h3>
              {renderItems(preview.unsupported, 'No unsupported recommendations.')}
            </section>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-border px-5 py-3">
          <button
            type="button"
            onClick={onClose}
            className="h-8 rounded-md border border-border bg-surface px-3 text-sm font-medium text-text-secondary transition-colors hover:text-text-primary"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className="h-8 rounded-md bg-accent px-3 text-sm font-semibold text-white transition-colors hover:bg-accent/90"
          >
            Confirm apply
          </button>
        </div>
      </div>
    </div>
  )
}

interface FaceLockFailureToastProps {
  label: string
  reason?: string
  onDismiss: () => void
}

function FaceLockFailureToast({ label, reason, onDismiss }: FaceLockFailureToastProps) {
  useEffect(() => {
    const t = window.setTimeout(onDismiss, 8000)
    return () => window.clearTimeout(t)
  }, [onDismiss])

  return (
    <div className="fixed bottom-6 right-6 z-[180] w-[320px] rounded-xl border border-amber-300/60 bg-amber-50 shadow-xl">
      <div className="flex items-start gap-3 p-4">
        <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-amber-500/20 text-amber-700 shrink-0">
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-amber-900">Face-lock could not be built</p>
          <p className="mt-0.5 text-xs text-amber-800 truncate" title={label}>
            {label}
          </p>
          <p className="mt-1 text-[11px] leading-snug text-amber-800/85">
            {reason || 'The redactor will fall back to per-frame face matching for this person.'}
          </p>
        </div>
        <button
          type="button"
          onClick={onDismiss}
          className="-mr-1 -mt-1 rounded p-1 text-amber-700 hover:text-amber-900 hover:bg-amber-100/60 transition-colors shrink-0"
          aria-label="Dismiss"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>
    </div>
  )
}
