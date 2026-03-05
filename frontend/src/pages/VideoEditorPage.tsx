import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import Hls from 'hls.js'
import { useVideoCache } from '../contexts/VideoCache'
import { API_BASE } from '../lib/api'
import visionIconUrl from '../../strand/icons/vision.svg?url'
import searchV2IconUrl from '../../strand/icons/search-v2.svg?url'
import analyzeIconUrl from '../../strand/icons/analyze.svg?url'

function isHlsUrl(url: string): boolean {
  return /\.m3u8(\?|$)/i.test(url) || url.includes('m3u8')
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
      className="inline-flex items-center px-1.5 py-0.5 rounded text-[11px] font-medium bg-accent/20 text-accent border border-accent/40 hover:bg-accent/30 hover:border-accent/60 transition-colors cursor-pointer align-baseline"
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

/* ------------------------------------------------------------------ */
/*  Toolbar config                                                     */
/* ------------------------------------------------------------------ */

type ToolId = 'tracker' | 'search-list' | 'captions'

/** Tracking region in normalized coords (0–1) relative to video viewport */
export type TrackingRegion = {
  id: string
  shape: 'rectangle' | 'ellipse' | 'circle'
  x: number
  y: number
  width: number
  height: number
  effect: 'blur' | 'pixelate' | 'solid'
  reason?: string
  locked?: boolean
}

const TOOLS: { id: ToolId; label: string; iconUrl: string }[] = [
  { id: 'tracker', label: 'Tracker', iconUrl: visionIconUrl },
  { id: 'search-list', label: 'Detection', iconUrl: searchV2IconUrl },
  { id: 'captions', label: 'Analyze/Transcript', iconUrl: analyzeIconUrl },
]

const DEFAULT_DETECTION_JOB_ID = '0456d15f-f83'

type DetectionItem = {
  id: string
  label: string
  tags: string[]
  color: string
  snapBase64?: string
}

const DUMMY_DETECTIONS: DetectionItem[] = [
  { id: '1', label: 'Screen 1', tags: ['screen', 'display'], color: '#3B82F6' },
  { id: '2', label: 'Plate 13', tags: ['license plate', 'object'], color: '#EF4444' },
  { id: '3', label: 'Head 16', tags: ['person', 'face'], color: '#F59E0B' },
  { id: '4', label: 'Head 17', tags: ['person', 'face'], color: '#F59E0B' },
  { id: '5', label: 'Head 22', tags: ['person', 'face'], color: '#F59E0B' },
  { id: '6', label: 'Head 27', tags: ['person', 'face'], color: '#F59E0B' },
  { id: '7', label: 'Head 28', tags: ['person', 'face'], color: '#F59E0B' },
  { id: '8', label: 'Screen 31', tags: ['screen', 'display'], color: '#3B82F6' },
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

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function VideoEditorPage() {
  const { videoId } = useParams<{ videoId: string }>()
  const { getVideo } = useVideoCache()
  const cached = videoId ? getVideo(videoId) : undefined

  const videoRef = useRef<HTMLVideoElement>(null)
  const timelineRef = useRef<HTMLDivElement>(null)
  const videoContainerRef = useRef<HTMLDivElement>(null)
  const editorCenterRef = useRef<HTMLDivElement>(null)
  const hlsRef = useRef<Hls | null>(null)
  const hlsLoadedUrlRef = useRef<string | null>(null)

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
  const [apiDetections, setApiDetections] = useState<DetectionItem[]>([])
  const [detectionLoading, setDetectionLoading] = useState(false)
  const [detectionError, setDetectionError] = useState<string | null>(null)
  const [detectionJobId, setDetectionJobId] = useState<string | null>(null)
  const [excludedFromRedactionIds, setExcludedFromRedactionIds] = useState<string[]>([])
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [rightSidebarOpen, setRightSidebarOpen] = useState(true)
  const [exportMenuOpen, setExportMenuOpen] = useState(false)
  const exportMenuRef = useRef<HTMLDivElement>(null)
  const [exportRedactLoading, setExportRedactLoading] = useState(false)
  const [exportRedactError, setExportRedactError] = useState<string | null>(null)
  const [exportRedactDownloadUrl, setExportRedactDownloadUrl] = useState<string | null>(null)
  const [isScrubbing, setIsScrubbing] = useState(false)
  const [hoverTime, setHoverTime] = useState<number | null>(null)
  const [trackMuted, setTrackMuted] = useState<{ video: boolean; audio: boolean }>({ video: false, audio: false })
  const [trackLocked, setTrackLocked] = useState<{ video: boolean; audio: boolean }>({ video: false, audio: false })
  const [analyzeSummaryExpanded, setAnalyzeSummaryExpanded] = useState(true)
  const [analyzeQuery, setAnalyzeQuery] = useState('')
  const [analyzeLoading, setAnalyzeLoading] = useState(false)
  const [analyzeError, setAnalyzeError] = useState<string | null>(null)
  type AnalyzeMessage = { id: string; role: 'user' | 'assistant'; content: string }
  const [analyzeMessages, setAnalyzeMessages] = useState<AnalyzeMessage[]>([])
  const [summaryText, setSummaryText] = useState<string | null>(null)
  const [summaryLoading, setSummaryLoading] = useState(false)
  const analyzeChatEndRef = useRef<HTMLDivElement>(null)
  const [timelineThumbnails, setTimelineThumbnails] = useState<string[]>([])
  const thumbnailsGeneratedRef = useRef(false)
  const [audioWaveformData, setAudioWaveformData] = useState<number[]>([])
  const waveformGeneratedRef = useRef(false)
  const timelinePreviewVideoRef = useRef<HTMLVideoElement | null>(null)
  const hlsPreviewRef = useRef<InstanceType<typeof Hls> | null>(null)
  const hlsPreviewLoadedUrlRef = useRef<string | null>(null)
  const [previewVideoReady, setPreviewVideoReady] = useState(false)

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

  const streamUrl = cached?.stream_url
  const title = cached?.metadata?.filename || videoId || 'Untitled'

  /* ---- HLS: play m3u8 streams in Chrome/Firefox (TwelveLabs returns HLS) ---- */
  const useHls = streamUrl && isHlsUrl(streamUrl) && Hls.isSupported()

  useEffect(() => {
    if (!streamUrl || !videoRef.current) return
    if (!isHlsUrl(streamUrl)) return
    if (!Hls.isSupported()) return

    // Avoid reloading the same URL (e.g. effect re-run from parent re-render) which would restart playback
    if (hlsRef.current && hlsLoadedUrlRef.current === streamUrl) return

    const video = videoRef.current
    if (hlsRef.current) {
      hlsRef.current.destroy()
      hlsRef.current = null
      hlsLoadedUrlRef.current = null
    }

    const hls = new Hls()
    hlsRef.current = hls
    hlsLoadedUrlRef.current = streamUrl
    hls.loadSource(streamUrl)
    hls.attachMedia(video)
    hls.on(Hls.Events.ERROR, (_, data) => {
      if (data.fatal) {
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
  }, [streamUrl])

  /* Reset timeline thumbnails and waveform when video source changes */
  useEffect(() => {
    setTimelineThumbnails([])
    setAudioWaveformData([])
    setPreviewVideoReady(false)
    thumbnailsGeneratedRef.current = false
    waveformGeneratedRef.current = false
  }, [streamUrl])

  /* Hidden preview video: same stream as main, used only for timeline thumbnails/waveform so main video is never sought on pause */
  useEffect(() => {
    if (!streamUrl || !timelinePreviewVideoRef.current || !isHlsUrl(streamUrl) || !Hls.isSupported()) return
    if (hlsPreviewRef.current && hlsPreviewLoadedUrlRef.current === streamUrl) return

    const video = timelinePreviewVideoRef.current
    if (hlsPreviewRef.current) {
      hlsPreviewRef.current.destroy()
      hlsPreviewRef.current = null
      hlsPreviewLoadedUrlRef.current = null
    }

    const hls = new Hls()
    hlsPreviewRef.current = hls
    hlsPreviewLoadedUrlRef.current = streamUrl
    hls.loadSource(streamUrl)
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
  }, [streamUrl])

  /* Preload timeline thumbnails + waveform on a hidden video so pausing the main video does nothing (no traverse) */
  useEffect(() => {
    const video = timelinePreviewVideoRef.current
    if (!streamUrl || !video || !Number.isFinite(duration) || duration <= 0 || !previewVideoReady) return
    if (thumbnailsGeneratedRef.current && waveformGeneratedRef.current) return

    const frameCount = 8
    const thumbW = 80
    const thumbH = 45
    const waveformSamples = Math.min(24, Math.max(12, Math.floor(duration)))
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

      if (!waveformGeneratedRef.current) {
        try {
          const audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)()
          if (audioCtx.state === 'suspended') await audioCtx.resume()
          const source = audioCtx.createMediaElementSource(v)
          const analyser = audioCtx.createAnalyser()
          analyser.fftSize = 2048
          analyser.smoothingTimeConstant = 0.6
          source.connect(analyser)
          analyser.connect(audioCtx.destination)
          const bufferLength = analyser.frequencyBinCount
          const dataArray = new Uint8Array(bufferLength)
          const waveform: number[] = []
          for (let i = 0; i < waveformSamples && !cancelled; i++) {
            const t = (i / (waveformSamples - 1)) * duration
            await new Promise<void>((resolve) => {
              const onSeeked = () => {
                v.removeEventListener('seeked', onSeeked)
                clearTimeout(tid)
                analyser.getByteTimeDomainData(dataArray)
                let sum = 0
                for (let j = 0; j < bufferLength; j++) {
                  const n = (dataArray[j] - 128) / 128
                  sum += n * n
                }
                waveform.push(Math.sqrt(sum / bufferLength))
                resolve()
              }
              const tid = setTimeout(() => { v.removeEventListener('seeked', onSeeked); waveform.push(0); resolve() }, SEEK_TIMEOUT_MS)
              v.addEventListener('seeked', onSeeked)
              v.currentTime = t
            })
          }
          if (!cancelled && waveform.length > 0) {
            const max = Math.max(...waveform, 0.001)
            setAudioWaveformData(waveform.map((x) => x / max))
          }
        } catch {
          /* ignore */
        }
        waveformGeneratedRef.current = true
      }
    })()
    return () => { cancelled = true }
  }, [streamUrl, duration, previewVideoReady])

  /* ---- Video event handlers ---- */

  const onLoadedMetadata = useCallback(() => {
    const v = videoRef.current
    if (!v) return
    setDuration(v.duration)
    v.currentTime = 0
    setCurrentTime(0)
    v.volume = volume
    v.muted = isMuted
    v.playbackRate = playbackRate
    // Start playback so the video is playing in the editor viewport (muted autoplay is allowed)
    v.play().catch(() => {})
  }, [volume, isMuted, playbackRate])

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
      v.play()
    } else {
      v.pause()
    }
  }, [])

  const skip = useCallback((delta: number) => {
    const v = videoRef.current
    if (!v) return
    v.currentTime = Math.max(0, Math.min(v.duration, v.currentTime + delta))
  }, [])

  const seekTo = useCallback((fraction: number) => {
    const v = videoRef.current
    if (!v || !Number.isFinite(v.duration)) return
    v.currentTime = fraction * v.duration
  }, [])

  const seekToTime = useCallback((seconds: number) => {
    const v = videoRef.current
    if (!v) return
    const t = Math.max(0, Number.isFinite(v.duration) ? Math.min(v.duration, seconds) : seconds)
    v.currentTime = t
    setCurrentTime(t)
    if (v.paused) v.play().catch(() => {})
  }, [])

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

  useEffect(() => {
    const onFullscreenChange = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', onFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', onFullscreenChange)
  }, [])

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
    setAnalyzeSummaryExpanded(true)
    try {
      const res = await fetch(`${API_BASE}/api/analyze-custom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: videoId,
          prompt: 'Provide a concise summary of this video. Include main topics, key moments, and any important details.',
        }),
      })
      const json = await res.json().catch(() => ({}))
      if (!res.ok) {
        setSummaryText(`Failed to generate summary: ${json?.error || res.status}`)
        return
      }
      const text = json?.data ?? ''
      setSummaryText(text || 'No summary generated.')
    } catch (e) {
      setSummaryText(e instanceof Error ? e.message : 'Failed to generate summary.')
    } finally {
      setSummaryLoading(false)
    }
  }, [videoId])

  const runDetect = useCallback(async () => {
    setDetectionError(null)
    setDetectionLoading(true)
    let jobId = DEFAULT_DETECTION_JOB_ID
    try {
      if (videoId) {
        const r = await fetch(`${API_BASE}/api/jobs/by-video/${encodeURIComponent(videoId)}`)
        if (r.ok) {
          const data = await r.json().catch(() => ({}))
          if (data.job_id) jobId = data.job_id
        }
      }
      setDetectionJobId(jobId)
      const [facesRes, objectsRes] = await Promise.all([
        fetch(`${API_BASE}/api/faces/${encodeURIComponent(jobId)}`),
        fetch(`${API_BASE}/api/objects/${encodeURIComponent(jobId)}`),
      ])
      const facesJson = await facesRes.json().catch(() => ({}))
      const objectsJson = await objectsRes.json().catch(() => ({}))
      if (facesRes.status === 202 || objectsRes.status === 202) {
        setDetectionError('Analysis still in progress. Try again in a moment.')
        setDetectionLoading(false)
        return
      }
      if (facesRes.status === 404 || objectsRes.status === 404) {
        setDetectionError('Job not found. Use a job ID that has completed detection (e.g. 0456d15f-f83).')
        setDetectionLoading(false)
        return
      }
      const items: DetectionItem[] = []
      const faceColor = '#F59E0B'
      const objectColors = ['#3B82F6', '#EF4444', '#10B981', '#8B5CF6']
      if (facesRes.ok && Array.isArray(facesJson.unique_faces)) {
        facesJson.unique_faces.forEach((f: { person_id?: string; description?: string; snap_base64?: string }, i: number) => {
          const name = (f.description || f.person_id || `Person ${i}`).toString().trim()
          items.push({
            id: `face-${f.person_id ?? i}`,
            label: name.slice(0, 60),
            tags: [],
            color: faceColor,
            snapBase64: f.snap_base64,
          })
        })
      }
      if (objectsRes.ok && Array.isArray(objectsJson.unique_objects)) {
        objectsJson.unique_objects.forEach((o: { object_id?: string; identification?: string; snap_base64?: string }, i: number) => {
          const name = (o.identification || o.object_id || `Object ${i}`).toString().trim()
          items.push({
            id: `object-${o.object_id ?? i}`,
            label: name.slice(0, 60),
            tags: [],
            color: objectColors[i % objectColors.length],
            snapBase64: o.snap_base64,
          })
        })
      }
      if (items.length === 0) {
        if (facesRes.status === 404 && objectsRes.status === 404) {
          setDetectionError('Job not found. Ensure the job has run and snaps exist (e.g. snaps/0456d15f-f83).')
        } else if (facesRes.ok || objectsRes.ok) {
          setDetectionError('No faces or objects found for this job.')
        }
      } else {
        setDetectionError(null)
      }
      setApiDetections(items)
      setHasRunDetection(true)
    } catch (e) {
      setDetectionError(e instanceof Error ? e.message : 'Detection request failed')
    } finally {
      setDetectionLoading(false)
    }
  }, [videoId])

  const exportRedacted = useCallback(async () => {
    setExportRedactError(null)
    setExportRedactDownloadUrl(null)
    if (!trackingRegions.length) {
      setExportRedactError('Add at least one redaction region by drawing on the video.')
      return
    }
    setExportRedactLoading(true)
    try {
      let jobId = DEFAULT_DETECTION_JOB_ID
      if (videoId) {
        const r = await fetch(`${API_BASE}/api/jobs/by-video/${encodeURIComponent(videoId)}`)
        if (r.ok) {
          const data = await r.json().catch(() => ({}))
          if (data.job_id) jobId = data.job_id
        }
      }
      const custom_regions = trackingRegions.map((r) => ({
        x: r.x,
        y: r.y,
        width: r.width,
        height: r.height,
        effect: r.effect,
      }))
      const res = await fetch(`${API_BASE}/api/redact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId, custom_regions }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error((err as { error?: string }).error || res.statusText)
      }
      const result = (await res.json()) as { download_url?: string }
      if (result.download_url) {
        setExportRedactDownloadUrl(result.download_url.startsWith('http') ? result.download_url : `${API_BASE}${result.download_url}`)
      }
      // Keep menu open so user can click "Download redacted video"
    } catch (e) {
      setExportRedactError(e instanceof Error ? e.message : 'Export failed')
    } finally {
      setExportRedactLoading(false)
    }
  }, [videoId, trackingRegions])

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
        <h1 className="text-sm font-semibold mt-2 mb-1">{withTimestampLinks(children, seekToTime)}</h1>
      ),
      h2: ({ children }: { children?: React.ReactNode }) => (
        <h2 className="text-xs font-semibold mt-2 mb-1">{withTimestampLinks(children, seekToTime)}</h2>
      ),
      h3: ({ children }: { children?: React.ReactNode }) => (
        <h3 className="text-xs font-medium mt-1.5 mb-0.5">{withTimestampLinks(children, seekToTime)}</h3>
      ),
      code: ({ children, className }: { children?: React.ReactNode; className?: string }) => (
        <code className={className ?? 'bg-surface px-1 py-0.5 rounded text-[11px]'}>{children}</code>
      ),
      pre: ({ children }: { children?: React.ReactNode }) => (
        <pre className="bg-surface rounded p-2 my-1.5 overflow-x-auto text-[11px] whitespace-pre-wrap break-words">
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
    'text-xs text-text-secondary leading-relaxed [&_*]:text-inherit [&_*]:text-xs [&_code]:bg-surface [&_code]:px-1 [&_code]:rounded [&_pre]:overflow-x-auto'

  useEffect(() => {
    if (!exportMenuOpen) return
    const onPointerDown = (e: PointerEvent) => {
      if (exportMenuRef.current?.contains(e.target as Node)) return
      setExportMenuOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    return () => document.removeEventListener('pointerdown', onPointerDown)
  }, [exportMenuOpen])

  /* ---- Tracker: viewport coords (0–1) ---- */
  const getNormFromEvent = useCallback((e: React.MouseEvent | MouseEvent) => {
    const el = videoContainerRef.current
    if (!el) return { x: 0, y: 0 }
    const rect = el.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height
    return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) }
  }, [])

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
      const { x, y } = getNormFromEvent(e)
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
      const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
      const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))
      setDrawCurrent({ x, y })
    },
    [drawStart]
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
        reason: settings.trackerReason || undefined,
        locked: false,
      })
    }
    setDrawStart(null)
    setDrawCurrent(null)
  }, [])

  const startDrag = useCallback(
    (e: React.MouseEvent, regionId: string) => {
      if (e.button !== 0) return
      e.preventDefault()
      e.stopPropagation()
      const region = trackingRegions.find((r) => r.id === regionId)
      if (!region) return
      const { x, y } = getNormFromEvent(e)
      setDragState({ regionId, offsetX: x - region.x, offsetY: y - region.y })
    },
    [trackingRegions, getNormFromEvent]
  )

  const updateDrag = useCallback(
    (e: MouseEvent) => {
      if (!dragState) return
      const region = trackingRegions.find((r) => r.id === dragState.regionId)
      if (!region || region.locked) return
      const { x, y } = getNormFromEvent(e)
      const newX = Math.max(0, Math.min(1 - region.width, x - dragState.offsetX))
      const newY = Math.max(0, Math.min(1 - region.height, y - dragState.offsetY))
      updateTrackingRegion(dragState.regionId, { x: newX, y: newY })
    },
    [dragState, trackingRegions, getNormFromEvent, updateTrackingRegion]
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
      if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'SELECT') return
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
  const filteredDetections = useMemo((): DetectionItem[] => {
    const q = detectionFilter.trim().toLowerCase()
    if (!q) return detectionList
    return detectionList.filter(
      (d) =>
        d.label.toLowerCase().includes(q) || d.tags.some((t: string) => t.toLowerCase().includes(q))
    )
  }, [detectionFilter, detectionList])

  /* Strand shared classes */
  const btnBase = 'inline-flex items-center justify-center rounded-md border border-border bg-surface text-text-primary hover:bg-card transition-colors'

  return (
    <div className="flex flex-col h-[calc(100vh-theme(spacing.header))] bg-background text-text-primary overflow-hidden">
      <div className="flex flex-1 min-h-0">

        {/* ============ LEFT SIDEBAR ============ */}
        <aside className="w-sidebar shrink-0 flex flex-col border-r border-border bg-surface overflow-hidden">
          <div className="px-4 h-10 flex items-center border-b border-border shrink-0">
            <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">Toolbar</span>
          </div>

          <nav className="flex-1 overflow-y-auto py-1 px-2 space-y-0.5">
            {TOOLS.map((t) => (
              <button
                key={t.id}
                type="button"
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

          {/* Tracker panel: motion tracking utilities */}
          {activeTool === 'tracker' && (
            <div className="border-t border-border p-3 space-y-3 shrink-0 overflow-y-auto max-h-[50vh]">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">Motion Tracker</span>
              </div>

              <div>
                <label className="block text-xs text-text-tertiary mb-1">Region shape</label>
                <select
                  className="w-full h-8 rounded-md bg-surface border border-border px-2 text-sm text-text-primary"
                  value={trackerShape}
                  onChange={(e) => setTrackerShape(e.target.value as 'rectangle' | 'ellipse' | 'circle')}
                >
                  <option value="rectangle">Rectangle</option>
                  <option value="ellipse">Ellipse</option>
                  <option value="circle">Circle</option>
                </select>
              </div>

              <div>
                <label className="block text-xs text-text-tertiary mb-1">Redaction effect</label>
                <select
                  className="w-full h-8 rounded-md bg-surface border border-border px-2 text-sm text-text-primary"
                  value={trackerEffect}
                  onChange={(e) => setTrackerEffect(e.target.value as 'blur' | 'pixelate' | 'solid')}
                >
                  <option value="blur">Blur</option>
                  <option value="pixelate">Pixelate</option>
                  <option value="solid">Solid mask</option>
                </select>
                <p className="text-[10px] text-text-tertiary mt-0.5">Applied live on the video overlay.</p>
              </div>

              <div>
                <label className="block text-xs text-text-tertiary mb-1">Reason (optional)</label>
                <input
                  type="text"
                  className="w-full h-8 rounded-md bg-surface border border-border px-2 text-sm text-text-primary placeholder:text-text-tertiary"
                  placeholder="e.g. PII, face"
                  value={trackerReason}
                  onChange={(e) => setTrackerReason(e.target.value)}
                />
              </div>

              <p className="text-[11px] text-text-tertiary leading-snug">
                Draw on the video to add redaction regions. Blur is applied live. Each drawn region is motion-tracked on export (KCF tracker) so the blur follows the content.
              </p>

              <button
                type="button"
                className={`w-full h-9 text-sm ${btnBase} flex items-center justify-center gap-2 ${drawMode ? 'ring-2 ring-accent ring-offset-2 ring-offset-surface' : ''}`}
                onClick={() => setDrawMode((prev) => !prev)}
              >
                <IconPlus className="w-4 h-4" />
                {drawMode ? 'Click on video to draw region' : 'Add tracking region'}
              </button>

              {trackingRegions.length > 0 && (
                <>
                  <div className="pt-2 border-t border-border">
                    <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider block mb-2">Regions ({trackingRegions.length})</span>
                    <ul className="space-y-1.5 max-h-32 overflow-y-auto">
                      {trackingRegions.map((r) => (
                        <li
                          key={r.id}
                          className={`flex items-center justify-between gap-2 px-2 py-1.5 rounded-md border text-xs cursor-pointer ${selectedRegionId === r.id ? 'bg-card border-accent' : 'bg-surface border-border hover:bg-card'}`}
                          onClick={() => setSelectedRegionId(r.id)}
                        >
                          <span className="truncate text-text-primary capitalize">{r.shape} · {r.effect}</span>
                          <div className="flex items-center gap-1 shrink-0">
                            <button
                              type="button"
                              className="p-1 rounded hover:bg-error/20 text-text-tertiary hover:text-error"
                              onClick={(e) => { e.stopPropagation(); removeTrackingRegion(r.id) }}
                              aria-label="Remove region"
                            >
                              <IconMinus className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {selectedRegionId && (() => {
                    const r = trackingRegions.find((x) => x.id === selectedRegionId)
                    if (!r) return null
                    return (
                      <div className="pt-2 border-t border-border space-y-2">
                        <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">Selected region</span>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <label className="text-text-tertiary block mb-0.5">X %</label>
                            <input
                              type="number"
                              min={0}
                              max={100}
                              step={1}
                              className="w-full h-7 rounded-md bg-surface border border-border px-2 text-text-primary"
                              value={Math.round(r.x * 100)}
                              onChange={(e) => updateTrackingRegion(r.id, { x: Number(e.target.value) / 100 })}
                            />
                          </div>
                          <div>
                            <label className="text-text-tertiary block mb-0.5">Y %</label>
                            <input
                              type="number"
                              min={0}
                              max={100}
                              step={1}
                              className="w-full h-7 rounded-md bg-surface border border-border px-2 text-text-primary"
                              value={Math.round(r.y * 100)}
                              onChange={(e) => updateTrackingRegion(r.id, { y: Number(e.target.value) / 100 })}
                            />
                          </div>
                          <div>
                            <label className="text-text-tertiary block mb-0.5">Width %</label>
                            <input
                              type="number"
                              min={1}
                              max={100}
                              step={1}
                              className="w-full h-7 rounded-md bg-surface border border-border px-2 text-text-primary"
                              value={Math.round(r.width * 100)}
                              onChange={(e) => updateTrackingRegion(r.id, { width: Number(e.target.value) / 100 })}
                            />
                          </div>
                          <div>
                            <label className="text-text-tertiary block mb-0.5">Height %</label>
                            <input
                              type="number"
                              min={1}
                              max={100}
                              step={1}
                              className="w-full h-7 rounded-md bg-surface border border-border px-2 text-text-primary"
                              value={Math.round(r.height * 100)}
                              onChange={(e) => updateTrackingRegion(r.id, { height: Number(e.target.value) / 100 })}
                            />
                          </div>
                        </div>
                        <label className="flex items-center gap-2 text-xs text-text-secondary cursor-pointer select-none">
                          <input
                            type="checkbox"
                            className="rounded border-border text-accent w-3.5 h-3.5"
                            checked={r.locked ?? false}
                            onChange={(e) => updateTrackingRegion(r.id, { locked: e.target.checked })}
                          />
                          Lock position
                        </label>
                      </div>
                    )
                  })()}
                </>
              )}

              <div className="pt-2 border-t border-border">
                <label className="flex items-center gap-2 text-xs text-text-secondary cursor-pointer select-none">
                  <input
                    type="checkbox"
                    className="rounded border-border text-accent w-3.5 h-3.5"
                    checked={smoothEdges}
                    onChange={(e) => setSmoothEdges(e.target.checked)}
                  />
                  Smooth edges
                </label>
                <p className="text-xs text-text-tertiary mt-1 leading-relaxed">
                  Softer mask borders when effect is applied by backend.
                </p>
              </div>

            </div>
          )}
        </aside>

        {/* ============ CENTER (Editor) ============ */}
        <div ref={editorCenterRef} className="flex-1 flex flex-col min-w-0 bg-background overflow-hidden">

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
                <div className="absolute right-0 top-full mt-1 py-1 min-w-[10rem] rounded-lg border border-border bg-surface shadow-lg z-50">
                  <button
                    type="button"
                    disabled={exportRedactLoading}
                    className="w-full px-3 py-2 text-left text-sm text-text-primary hover:bg-card transition-colors flex items-center gap-2 disabled:opacity-60 disabled:cursor-not-allowed"
                    onClick={() => exportRedacted()}
                  >
                    {exportRedactLoading ? (
                      <>
                        <span className="w-4 h-4 border-2 border-accent/60 border-t-accent rounded-full animate-spin shrink-0" aria-hidden />
                        Exporting…
                      </>
                    ) : (
                      'Redacted'
                    )}
                  </button>
                  {exportRedactError && (
                    <p className="px-3 py-2 text-xs text-error border-t border-border">{exportRedactError}</p>
                  )}
                  {exportRedactDownloadUrl && (
                    <a
                      href={exportRedactDownloadUrl}
                      download
                      className="block w-full px-3 py-2 text-left text-sm text-accent hover:bg-card transition-colors flex items-center gap-2 border-t border-border"
                      onClick={() => setExportMenuOpen(false)}
                    >
                      <IconDownload className="w-4 h-4" />
                      Download redacted video
                    </a>
                  )}
                  <button
                    type="button"
                    className="w-full px-3 py-2 text-left text-sm text-text-primary hover:bg-card transition-colors flex items-center gap-2"
                    onClick={() => { setExportMenuOpen(false); /* TODO: download */ }}
                  >
                    Download
                  </button>
                  <button
                    type="button"
                    className="w-full px-3 py-2 text-left text-sm text-text-primary hover:bg-card transition-colors flex items-center gap-2"
                    onClick={() => { setExportMenuOpen(false); /* TODO: report */ }}
                  >
                    Report
                  </button>
                </div>
              )}
            </div>
          </header>

          {/* Video viewport: fixed aspect 16:9, centered */}
          <div className="shrink-0 px-4 pt-4">
            <div
              ref={videoContainerRef}
              className="relative w-full max-w-4xl mx-auto aspect-video rounded-xl overflow-hidden bg-brand-charcoal border border-border shadow-lg"
            >
              {/* App name watermark: top-left corner */}
              <div
                className="absolute top-3 left-3 z-20 pointer-events-none select-none px-2.5 py-1.5 rounded bg-black/60 backdrop-blur-sm border border-white/10"
                aria-hidden
              >
                <span className="text-xs font-semibold tracking-wide text-white/95">
                  GDPR Compliance [Video REDACTION]
                </span>
              </div>

              {streamUrl ? (
                <>
                  <video
                    ref={videoRef}
                    src={useHls ? undefined : streamUrl}
                    className="absolute inset-0 w-full h-full object-contain z-0"
                    playsInline
                    muted={isMuted}
                    loop={false}
                    onLoadedMetadata={onLoadedMetadata}
                    onTimeUpdate={onTimeUpdate}
                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                    onEnded={() => setIsPlaying(false)}
                    onClick={(e) => { if (activeTool !== 'tracker' || !drawMode) togglePlay(); else e.stopPropagation() }}
                  />
                  {/* Hidden video for timeline preload only; main video is never sought on pause */}
                  <video
                    ref={timelinePreviewVideoRef}
                    src={useHls ? undefined : streamUrl}
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

              {/* Tracker overlay: draw + region boxes (only when Tracker tool active); z-10 so video (z-0) stays visible behind */}
              {activeTool === 'tracker' && streamUrl && (
                <div
                  className={`absolute inset-0 z-10 pointer-events-auto ${drawMode ? 'cursor-crosshair' : 'cursor-default'}`}
                  onMouseDown={(e) => {
                    if (drawMode) startDraw(e)
                    else if (e.target === e.currentTarget) {
                      setSelectedRegionId(null)
                      togglePlay()
                    }
                  }}
                  onMouseUp={(e) => { if (drawStart) endDraw(e) }}
                >
                  {/* Floating tracker toolbar */}
                  <div className="absolute top-3 right-3 z-30 pointer-events-auto flex items-center gap-1 bg-brand-charcoal/90 backdrop-blur-sm rounded-lg px-2 py-1.5 shadow-lg border border-white/10">
                    <select
                      className="h-7 rounded-md bg-white/10 border-0 px-2 text-xs text-brand-white cursor-pointer focus:ring-1 focus:ring-accent"
                      value={trackerShape}
                      onChange={(e) => setTrackerShape(e.target.value as 'rectangle' | 'ellipse' | 'circle')}
                    >
                      <option value="rectangle">Rect</option>
                      <option value="ellipse">Ellipse</option>
                      <option value="circle">Circle</option>
                    </select>
                    <select
                      className="h-7 rounded-md bg-white/10 border-0 px-2 text-xs text-brand-white cursor-pointer focus:ring-1 focus:ring-accent"
                      value={trackerEffect}
                      onChange={(e) => setTrackerEffect(e.target.value as 'blur' | 'pixelate' | 'solid')}
                    >
                      <option value="blur">Blur</option>
                      <option value="pixelate">Pixelate</option>
                      <option value="solid">Solid</option>
                    </select>
                    <div className="w-px h-5 bg-white/20 mx-0.5" />
                    <button
                      type="button"
                      className={`h-7 px-3 rounded-md text-xs font-medium transition-colors flex items-center gap-1.5 ${
                        drawMode
                          ? 'bg-accent text-brand-charcoal'
                          : 'bg-white/10 text-brand-white hover:bg-white/20'
                      }`}
                      onClick={() => setDrawMode((prev) => !prev)}
                    >
                      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 5v14M5 12h14" /></svg>
                      {drawMode ? 'Drawing...' : 'Draw'}
                    </button>
                    {selectedRegionId && (
                      <button
                        type="button"
                        className="h-7 px-2 rounded-md text-xs font-medium bg-white/10 text-brand-white hover:bg-error/80 transition-colors"
                        onClick={() => removeTrackingRegion(selectedRegionId)}
                      >
                        Delete
                      </button>
                    )}
                    {trackingRegions.length > 0 && (
                      <span className="text-xs text-white/50 ml-1">{trackingRegions.length} region{trackingRegions.length !== 1 ? 's' : ''}</span>
                    )}
                  </div>
                  {/* Existing regions: draggable boxes with effect (blur / pixelate / solid) */}
                  {trackingRegions.map((r) => {
                    const isBlur = r.effect === 'blur'
                    const isPixelate = r.effect === 'pixelate'
                    const isSolid = r.effect === 'solid'
                    const effectStyle: React.CSSProperties = isSolid
                      ? { background: 'var(--strand-brand-charcoal)' }
                      : isBlur
                        ? { backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)', background: 'rgba(0,0,0,0.05)' }
                        : { backdropFilter: 'blur(4px)', WebkitBackdropFilter: 'blur(4px)', background: 'rgba(0,0,0,0.25)' }
                    return (
                      <div
                        key={r.id}
                        className={`absolute border-2 pointer-events-auto cursor-move transition-colors overflow-hidden ${selectedRegionId === r.id ? 'border-accent z-10 ring-2 ring-accent/50' : 'border-brand-white/70 hover:border-brand-white'}`}
                        style={{
                          left: `${r.x * 100}%`,
                          top: `${r.y * 100}%`,
                          width: `${(r.shape === 'circle' ? Math.min(r.width, r.height) : r.width) * 100}%`,
                          height: `${(r.shape === 'circle' ? Math.min(r.width, r.height) : r.height) * 100}%`,
                          borderRadius: r.shape === 'ellipse' || r.shape === 'circle' ? '50%' : '0',
                          pointerEvents: r.locked ? 'none' : 'auto',
                          ...effectStyle,
                        }}
                        onMouseDown={(e) => { e.stopPropagation(); setSelectedRegionId(r.id); if (!r.locked) startDrag(e, r.id) }}
                        title={`${r.effect}${r.reason ? ` — ${r.reason}` : ''}`}
                      />
                    )
                  })}
                  {/* Draw preview */}
                  {drawStart && drawCurrent && (
                    <div
                      className="absolute border-2 border-dashed border-accent bg-accent/10 pointer-events-none"
                      style={{
                        left: `${Math.min(drawStart.x, drawCurrent.x) * 100}%`,
                        top: `${Math.min(drawStart.y, drawCurrent.y) * 100}%`,
                        width: `${(trackerShape === 'circle' ? Math.min(Math.abs(drawCurrent.x - drawStart.x), Math.abs(drawCurrent.y - drawStart.y)) : Math.abs(drawCurrent.x - drawStart.x)) * 100}%`,
                        height: `${(trackerShape === 'circle' ? Math.min(Math.abs(drawCurrent.x - drawStart.x), Math.abs(drawCurrent.y - drawStart.y)) : Math.abs(drawCurrent.y - drawStart.y)) * 100}%`,
                        borderRadius: trackerShape === 'ellipse' || trackerShape === 'circle' ? '50%' : '0',
                      }}
                    />
                  )}
                </div>
              )}
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

            <div className="flex items-center gap-2 text-xs">
              <span className="font-mono text-text-secondary tabular-nums">{fmtTime(currentTime)}</span>
              <span className="text-text-tertiary">/</span>
              <span className="font-mono text-text-secondary tabular-nums">{fmtTime(duration)}</span>
            </div>

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
          <div className="shrink-0 flex flex-col border-t border-border bg-surface select-none" style={{ minHeight: 170 }}>
            {/* Timeline header */}
            <div className="h-8 px-3 flex items-center justify-between border-b border-border shrink-0">
              <div className="flex items-center gap-3">
                <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">Timeline</span>
                <span className="text-xs font-mono text-text-tertiary tabular-nums">{fmtShort(currentTime)} / {fmtShort(duration)}</span>
              </div>
              <div className="flex items-center gap-1">
                <button type="button" className={`h-6 w-6 ${btnBase} border-0`} onClick={() => setTimelineZoom((z) => Math.max(z / 1.5, 0.5))} aria-label="Zoom out" title="Zoom out">
                  <IconMinus className="w-3 h-3" />
                </button>
                <div className="w-16 h-1 rounded-full bg-border mx-1 relative">
                  <div className="absolute inset-y-0 left-0 rounded-full bg-accent" style={{ width: `${Math.min(100, ((timelineZoom - 0.5) / 9.5) * 100)}%` }} />
                </div>
                <button type="button" className={`h-6 w-6 ${btnBase} border-0`} onClick={() => setTimelineZoom((z) => Math.min(z * 1.5, 10))} aria-label="Zoom in" title="Zoom in">
                  <IconPlus className="w-3 h-3" />
                </button>
              </div>
            </div>

            <div className="flex flex-1 min-h-0">
              {/* Track handles */}
              <div className="w-32 shrink-0 border-r border-border flex flex-col bg-card">
                <div className="h-7 border-b border-border flex items-center px-2">
                  <span className="text-[10px] text-text-tertiary uppercase tracking-wider">Tracks</span>
                </div>
                {/* Video handle */}
                <div className="h-11 flex items-center gap-1.5 px-2 border-b border-border group">
                  <span className="w-1.5 h-6 rounded-full bg-accent shrink-0" />
                  <span className="text-xs text-text-secondary truncate flex-1 leading-none">Video</span>
                  <button type="button" onClick={() => setTrackMuted(p => ({ ...p, video: !p.video }))} className={`h-5 w-5 rounded inline-flex items-center justify-center transition-colors ${trackMuted.video ? 'text-text-tertiary bg-card' : 'text-text-tertiary opacity-0 group-hover:opacity-100 hover:text-text-primary hover:bg-card'}`} aria-label={trackMuted.video ? 'Unmute' : 'Mute'} title={trackMuted.video ? 'Unmute' : 'Mute'}>
                    <svg className="w-3 h-3" viewBox="0 0 24 24" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3z" /></svg>
                  </button>
                  <button type="button" onClick={() => setTrackLocked(p => ({ ...p, video: !p.video }))} className={`h-5 w-5 rounded inline-flex items-center justify-center transition-colors ${trackLocked.video ? 'text-text-tertiary bg-card' : 'text-text-tertiary opacity-0 group-hover:opacity-100 hover:text-text-primary hover:bg-card'}`} aria-label={trackLocked.video ? 'Unlock' : 'Lock'} title={trackLocked.video ? 'Unlock' : 'Lock'}>
                    <svg className="w-3 h-3" viewBox="0 0 24 24" fill="currentColor">{trackLocked.video ? <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zM9 8V6c0-1.66 1.34-3 3-3s3 1.34 3 3v2H9z" /> : <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6h2c0-1.66 1.34-3 3-3s3 1.34 3 3v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm0 12H6V10h12v10zm-6-3c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2z" />}</svg>
                  </button>
                </div>
                {/* Audio handle */}
                <div className="h-9 flex items-center gap-1.5 px-2 border-b border-border group">
                  <span className="w-1.5 h-5 rounded-full bg-highlight shrink-0" />
                  <span className="text-xs text-text-tertiary truncate flex-1 leading-none">Audio</span>
                  <button type="button" onClick={() => setTrackMuted(p => ({ ...p, audio: !p.audio }))} className={`h-5 w-5 rounded inline-flex items-center justify-center transition-colors ${trackMuted.audio ? 'text-text-tertiary bg-card' : 'text-text-tertiary opacity-0 group-hover:opacity-100 hover:text-text-primary hover:bg-card'}`} aria-label={trackMuted.audio ? 'Unmute' : 'Mute'} title={trackMuted.audio ? 'Unmute' : 'Mute'}>
                    <svg className="w-3 h-3" viewBox="0 0 24 24" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3z" /></svg>
                  </button>
                  <button type="button" onClick={() => setTrackLocked(p => ({ ...p, audio: !p.audio }))} className={`h-5 w-5 rounded inline-flex items-center justify-center transition-colors ${trackLocked.audio ? 'text-text-tertiary bg-card' : 'text-text-tertiary opacity-0 group-hover:opacity-100 hover:text-text-primary hover:bg-card'}`} aria-label={trackLocked.audio ? 'Unlock' : 'Lock'} title={trackLocked.audio ? 'Unlock' : 'Lock'}>
                    <svg className="w-3 h-3" viewBox="0 0 24 24" fill="currentColor">{trackLocked.audio ? <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zM9 8V6c0-1.66 1.34-3 3-3s3 1.34 3 3v2H9z" /> : <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6h2c0-1.66 1.34-3 3-3s3 1.34 3 3v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm0 12H6V10h12v10zm-6-3c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2z" />}</svg>
                  </button>
                </div>
              </div>

              {/* Scrollable timeline content */}
              <div
                ref={timelineRef}
                className="flex-1 min-h-0 overflow-x-auto overflow-y-hidden bg-background"
                onMouseDown={handleTimelineMouseDown}
                onMouseMove={handleTimelineHover}
                onMouseLeave={() => setHoverTime(null)}
              >
                <div className="min-w-full flex flex-col relative" style={{ width: `${timelineContentWidthPct}%`, minHeight: '100%' }}>

                  {/* Ruler */}
                  <div className="h-7 shrink-0 relative border-b border-border bg-card cursor-col-resize">
                    {minorTicks.map((sec) => (
                      <div key={`m-${sec}`} className="absolute bottom-0 w-px bg-border" style={{ left: `${(sec / Math.max(duration, 1)) * 100}%`, height: 5 }} />
                    ))}
                    {majorTicks.map((sec) => (
                      <div key={`M-${sec}`} className="absolute top-0 bottom-0" style={{ left: `${duration > 0 ? (sec / duration) * 100 : 0}%` }}>
                        <div className="absolute bottom-0 w-px bg-text-tertiary" style={{ height: 10 }} />
                        <span className="absolute left-1.5 top-1 text-[10px] font-mono text-text-tertiary tabular-nums whitespace-nowrap leading-none">{fmtShort(sec)}</span>
                      </div>
                    ))}
                    {hoverTime !== null && !isScrubbing && (
                      <div className="absolute top-0 bottom-0 pointer-events-none z-10" style={{ left: `${(hoverTime / Math.max(duration, 1)) * 100}%` }}>
                        <div className="absolute bottom-0 w-px h-full bg-border" />
                        <span className="absolute -top-0.5 left-1/2 -translate-x-1/2 px-1 py-px rounded text-[9px] font-mono bg-surface text-text-primary border border-border tabular-nums whitespace-nowrap shadow-sm">{fmtShort(hoverTime)}</span>
                      </div>
                    )}
                  </div>

                  {/* Video track */}
                  <div className={`h-11 shrink-0 relative border-b border-border ${trackMuted.video ? 'opacity-40' : ''}`}>
                    <div className="absolute inset-y-[3px] left-0 right-0 rounded-lg overflow-hidden border border-accent/30 bg-accent-light flex">
                      {/* Filmstrip: actual video frames or placeholder dividers */}
                      {timelineThumbnails.length > 0 ? (
                        timelineThumbnails.map((src, i) => (
                          <div key={i} className="flex-1 min-w-0 h-full relative" style={{ flexBasis: 0 }}>
                            <img src={src} alt="" className="w-full h-full object-cover block" />
                          </div>
                        ))
                      ) : (
                        duration > 0 && (() => {
                          const frameCount = Math.min(24, Math.max(8, Math.floor(duration / 1.5)))
                          return (
                            <>
                              {Array.from({ length: frameCount - 1 }, (_, i) => (
                                <div
                                  key={i}
                                  className="absolute top-0 bottom-0 w-px bg-accent/20 pointer-events-none"
                                  style={{ left: `${((i + 1) / frameCount) * 100}%` }}
                                />
                              ))}
                            </>
                          )
                        })()
                      )}
                      <div className="absolute left-0 right-0 inset-y-0 bg-gradient-to-r from-black/50 via-transparent to-black/40 pointer-events-none rounded-lg z-[1]" />
                      <span className="absolute inset-y-0 left-3 flex items-center text-xs font-medium text-text-primary truncate pr-6 z-[2] drop-shadow-sm" title={title}>{title || 'Untitled'}</span>
                      <span className="absolute inset-y-0 right-3 flex items-center text-[10px] font-mono text-text-tertiary tabular-nums z-[2] drop-shadow-sm">{fmtShort(duration)}</span>
                      <div className="absolute left-0 inset-y-0 w-1.5 bg-accent/60 hover:bg-accent cursor-col-resize rounded-l-lg transition-colors z-[2]" />
                      <div className="absolute right-0 inset-y-0 w-1.5 bg-accent/60 hover:bg-accent cursor-col-resize rounded-r-lg transition-colors z-[2]" />
                    </div>
                    {trackLocked.video && (
                      <div className="absolute inset-0 bg-background/50 rounded-lg flex items-center justify-center">
                        <svg className="w-3.5 h-3.5 text-text-tertiary" viewBox="0 0 24 24" fill="currentColor"><path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zM9 8V6c0-1.66 1.34-3 3-3s3 1.34 3 3v2H9z" /></svg>
                      </div>
                    )}
                  </div>

                  {/* Audio track */}
                  <div className={`h-9 shrink-0 relative border-b border-border ${trackMuted.audio ? 'opacity-40' : ''}`}>
                    <div className="absolute inset-y-[3px] left-0 right-0 rounded-lg overflow-hidden border border-highlight/20 bg-highlight/10 flex items-center">
                      {/* Waveform: draw on top with solid bars so it’s always visible */}
                      {audioWaveformData.length > 0 ? (
                        <div className="absolute inset-0 flex items-center justify-center gap-[3px] px-3 z-[1]" style={{ paddingTop: 6, paddingBottom: 6 }}>
                          {audioWaveformData.map((amp, i) => {
                            const normalized = Math.min(1, amp)
                            const barHeightPx = Math.max(8, Math.round(normalized * 18))
                            return (
                              <div
                                key={i}
                                className="flex-1 min-w-0 h-full flex items-center justify-center"
                                style={{ minWidth: 3 }}
                              >
                                <div
                                  className="rounded-sm w-full shrink-0"
                                  style={{
                                    height: barHeightPx,
                                    minHeight: 8,
                                    maxWidth: 6,
                                    backgroundColor: 'var(--strand-accent, #00DC82)',
                                  }}
                                  title={`${duration > 0 ? ((i / (audioWaveformData.length - 1)) * duration).toFixed(1) : 0}s`}
                                />
                              </div>
                            )
                          })}
                        </div>
                      ) : null}
                      <div className="absolute left-0 right-0 inset-y-0 bg-gradient-to-r from-surface/50 via-transparent to-surface/50 pointer-events-none rounded-lg z-0" />
                      <span className="absolute inset-y-0 left-3 flex items-center text-xs font-medium text-text-secondary truncate pr-2 z-[2] drop-shadow-sm">Audio</span>
                      <span className="absolute inset-y-0 right-3 flex items-center text-[10px] font-mono text-text-tertiary tabular-nums z-[2] drop-shadow-sm">{fmtShort(duration)}</span>
                      <div className="absolute left-0 inset-y-0 w-1.5 bg-highlight/40 hover:bg-highlight cursor-col-resize rounded-l-lg transition-colors z-[2]" />
                      <div className="absolute right-0 inset-y-0 w-1.5 bg-highlight/40 hover:bg-highlight cursor-col-resize rounded-r-lg transition-colors z-[2]" />
                    </div>
                    {trackLocked.audio && (
                      <div className="absolute inset-0 bg-background/50 rounded-lg flex items-center justify-center">
                        <svg className="w-3.5 h-3.5 text-text-tertiary" viewBox="0 0 24 24" fill="currentColor"><path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zM9 8V6c0-1.66 1.34-3 3-3s3 1.34 3 3v2H9z" /></svg>
                      </div>
                    )}
                  </div>

                  {/* Gridlines */}
                  {majorTicks.map((sec) => (
                    <div key={`g-${sec}`} className="absolute pointer-events-none bg-border/30" style={{ left: `${duration > 0 ? (sec / duration) * 100 : 0}%`, top: 28, bottom: 0, width: 1 }} />
                  ))}

                  {/* Buffered indicator */}
                  <div className="absolute top-0 h-7 pointer-events-none bg-border/20 rounded-sm" style={{ left: 0, width: `${bufferedPct * 100}%` }} />

                  {/* Playhead */}
                  <div className="absolute top-0 bottom-0 pointer-events-none z-30" style={{ left: `${progress * 100}%` }}>
                    <svg className="absolute top-0 left-1/2 -translate-x-1/2" width="12" height="12" viewBox="0 0 12 12" fill="none">
                      <path d="M6 0L9 5H6.5V12H5.5V5H3L6 0Z" fill="#00DC82" />
                    </svg>
                    <div className="absolute top-[10px] bottom-0 left-1/2 -translate-x-1/2 w-px bg-accent" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ============ RIGHT SIDEBAR (collapsible) ============ */}
        <aside className={`shrink-0 flex flex-col border-l border-border bg-surface overflow-hidden transition-[width] duration-200 ${rightSidebarOpen ? 'w-80' : 'w-10'}`}>
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

                  {/* Category tags */}
                  <div className="px-3 pt-3 pb-2 border-b border-border shrink-0 space-y-2">
                    <p className="text-[10px] font-medium text-text-tertiary uppercase tracking-wider">Media</p>
                    <div className="flex flex-wrap gap-1.5">
                      {['Video', 'Audio'].map((tag) => (
                        <span key={tag} className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-card border border-border text-text-secondary">
                          {tag}
                        </span>
                      ))}
                    </div>
                    <p className="text-[10px] font-medium text-text-tertiary uppercase tracking-wider pt-1">Content</p>
                    <div className="flex flex-wrap gap-1.5">
                      {['Text-in-video', 'Conversation'].map((tag) => (
                        <span key={tag} className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-card border border-border text-text-secondary">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Summary — collapsible; separate from chat */}
                  <div className="shrink-0 p-3">
                    <div className="rounded-lg border border-border bg-card overflow-hidden">
                      <button
                        type="button"
                        onClick={() => setAnalyzeSummaryExpanded((e) => !e)}
                        className="w-full px-3 py-2.5 flex items-center justify-between text-left bg-surface hover:bg-card transition-colors"
                      >
                        <span className="text-xs font-medium text-text-secondary">Summary</span>
                        <span className="text-text-tertiary transition-transform" style={{ transform: analyzeSummaryExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}>
                          <IconChevronRight className="w-4 h-4" />
                        </span>
                      </button>
                      {analyzeSummaryExpanded && (
                        <div className="px-3 pb-3 pt-0 border-t border-border">
                          {summaryLoading && (
                            <p className="text-xs text-text-tertiary py-2">Generating summary…</p>
                          )}
                          {!summaryLoading && summaryText && (
                            <div className={`pt-2 ${markdownWrapClass}`}>
                              <ReactMarkdown components={markdownComponents}>{summaryText}</ReactMarkdown>
                            </div>
                          )}
                          {!summaryLoading && !summaryText && (
                            <div className="pt-2 space-y-2">
                              <p className="text-xs text-text-tertiary">Generate a short summary of this video.</p>
                              <button
                                type="button"
                                onClick={runGenerateSummary}
                                disabled={!videoId}
                                className="w-full h-8 rounded-md text-xs font-medium bg-accent text-white border border-accent hover:bg-accent-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                              >
                                Generate summary
                              </button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Chat — analyze Q&A with markdown responses */}
                  <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
                    <div className="flex-1 overflow-y-auto p-3 space-y-3">
                      {analyzeMessages.length === 0 && !analyzeLoading && (
                        <p className="text-xs text-text-tertiary">Ask a question about this video below.</p>
                      )}
                      {analyzeMessages.map((msg) => (
                        <div key={msg.id} className={msg.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
                          <div
                            className={`max-w-[92%] rounded-lg px-3 py-2 text-xs ${
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
                          <div className="rounded-lg px-3 py-2 bg-card border border-border text-text-tertiary text-xs">
                            Analyzing…
                          </div>
                        </div>
                      )}
                      <div ref={analyzeChatEndRef} />
                    </div>
                    {analyzeError && (
                      <p className="px-3 text-xs text-error shrink-0">{analyzeError}</p>
                    )}
                    <div className="p-3 border-t border-border shrink-0">
                      <div className="flex gap-1.5">
                        <input
                          type="text"
                          value={analyzeQuery}
                          onChange={(e) => setAnalyzeQuery(e.target.value)}
                          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && runAnalyze()}
                          placeholder="Ask about this video..."
                          disabled={!videoId}
                          className="flex-1 h-9 rounded-md bg-surface border border-border px-3 text-xs text-text-primary placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent disabled:opacity-60 disabled:cursor-not-allowed"
                        />
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
                            <img src={analyzeIconUrl} alt="" className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
            /* Detection sidebar (Tracker or Detection selected) */
            <>
              <div className="px-3 h-10 flex items-center justify-between border-b border-border shrink-0 gap-1">
                <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider truncate min-w-0">Detections</span>
                {hasRunDetection && (
                  <span className="text-xs text-text-tertiary tabular-nums shrink-0">({filteredDetections.length})</span>
                )}
                <button type="button" onClick={() => setRightSidebarOpen(false)} className={`h-7 w-7 shrink-0 rounded-md ${btnBase}`} aria-label="Collapse sidebar" title="Collapse sidebar">
                  <IconChevronRight className="w-4 h-4" />
                </button>
              </div>
              {hasRunDetection ? (
                <>
                  <div className="p-2 border-b border-border shrink-0">
                    <input
                      type="search"
                      placeholder="Filter detections..."
                      value={detectionFilter}
                      onChange={(e) => setDetectionFilter(e.target.value)}
                      className="w-full h-8 rounded-md bg-surface border border-border px-3 text-xs text-text-primary placeholder:text-text-tertiary"
                    />
                  </div>
                  <div className="flex-1 overflow-y-auto">
                    {filteredDetections.map((d) => {
                      const excluded = excludedFromRedactionIds.includes(d.id)
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
                              setExcludedFromRedactionIds((prev) =>
                                prev.includes(d.id) ? prev.filter((id) => id !== d.id) : [...prev, d.id]
                              )
                            }}
                            className={`shrink-0 p-1.5 rounded-md transition-colors ${excluded ? 'text-text-tertiary hover:bg-card hover:text-accent' : 'text-accent hover:bg-accent-light'}`}
                            title={excluded ? 'Include in redaction (blur this)' : 'Exclude from redaction (do not blur)'}
                            aria-label={excluded ? 'Include in redaction' : 'Exclude from redaction'}
                          >
                            {excluded ? <IconEyeOff className="w-4 h-4" /> : <IconEye className="w-4 h-4" />}
                          </button>
                        </div>
                      )
                    })}
                  </div>
                </>
              ) : (
                <div className="flex-1 overflow-y-auto flex flex-col items-center justify-center min-h-[8rem] gap-3 p-4">
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
                    Loads faces & objects from job (e.g. {DEFAULT_DETECTION_JOB_ID})
                  </p>
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
    </div>
  )
}
