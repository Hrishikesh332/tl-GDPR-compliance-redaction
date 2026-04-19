import { useState, useMemo, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import AddImageModal from '../components/AddImageModal'
import AddEntityModal from '../components/AddEntityModal'
import { useVideoCache, type CachedVideo } from '../contexts/VideoCache'
import { API_BASE } from '../lib/api'
import searchIconUrl from '../../strand/icons/search.svg?url'
import arrowBoxUpIconUrl from '../../strand/icons/arrow-box-up.svg?url'

type SearchAttachment = {
  id: string
  type: 'image' | 'entity'
  name: string
  previewUrl: string
  file?: File
}

type SearchSessionAttachment = {
  id: string
  type: 'image' | 'entity'
  name: string
  previewUrl: string
}

type SearchSessionEntity = {
  id: string
  name: string
  previewUrl: string
}

type SearchSessionResult = {
  query: string
  queryText?: string
  attachments?: SearchSessionAttachment[]
  entities?: SearchSessionEntity[]
  results: VideoItem[]
}

const SEARCH_SUGGESTIONS: string[] = [
  'Primary suspect in courtroom',
  'Judge announcing verdict or ruling',
  'Witness testimony',
  'Faces or people to anonymize',
]

type EntityOption = {
  id: string
  name: string
  previewUrl: string
}

function IconAddImage({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 16 16"
      fill="none"
      preserveAspectRatio="xMidYMid meet"
      aria-hidden
    >
      <path
        fill="currentColor"
        fillRule="evenodd"
        clipRule="evenodd"
        d="M8.834 5.334a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0m1.5-.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1M9.94 7.623a.767.767 0 0 0-1.212.02l-1.447 1.93-.738-.738c-.299-.3-.784-.3-1.084 0l-1.356 1.356a.767.767 0 0 0 .542 1.31h6.802c.642 0 1-.744.598-1.246zm-2.02 2.764 1.427-1.904 1.614 2.017H7.816a1 1 0 0 0 .103-.113m-1.156.082.032.031H5.208l.793-.793z"
      />
      <path
        fill="currentColor"
        fillRule="evenodd"
        clipRule="evenodd"
        d="M5.6 2A3.6 3.6 0 0 0 2 5.6v4.8A3.6 3.6 0 0 0 5.6 14h4.8a3.6 3.6 0 0 0 3.6-3.6V5.6A3.6 3.6 0 0 0 10.4 2zm4.8 1H5.6A2.6 2.6 0 0 0 3 5.6v4.8A2.6 2.6 0 0 0 5.6 13h4.8a2.6 2.6 0 0 0 2.6-2.6V5.6A2.6 2.6 0 0 0 10.4 3"
      />
    </svg>
  )
}

function IconEntity({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 12 12" fill="currentColor">
      <path fillRule="evenodd" clipRule="evenodd" d="M6 2C7.10457 2 8 2.89543 8 4C8 5.10457 7.10457 6 6 6C4.89543 6 4 5.10457 4 4C4 2.89543 4.89543 2 6 2ZM6 3C5.44772 3 5 3.44772 5 4C5 4.55228 5.44772 5 6 5C6.55228 5 7 4.55228 7 4C7 3.44772 6.55228 3 6 3Z" />
      <path fillRule="evenodd" clipRule="evenodd" d="M8.40039 0C10.3883 0.000211285 11.9998 1.61169 12 3.59961V8.40039C11.9998 10.3883 10.3883 11.9998 8.40039 12H3.59961C1.61169 11.9998 0.000211285 10.3883 0 8.40039V3.59961C0.000211156 1.61169 1.61169 0.000211157 3.59961 0H8.40039ZM4.50098 7.5C3.16242 7.5 1.96779 8.54749 1.60938 10.0713C2.08624 10.6387 2.80047 10.9999 3.59961 11H8.40039C9.19957 10.9999 9.91279 10.6377 10.3896 10.0703C10.0309 8.54742 8.83897 7.50019 7.50098 7.5H4.50098ZM3.59961 1C2.16396 1.00018 1.00018 2.16396 1 3.59961V8.40039C1.00002 8.5262 1.01201 8.64948 1.0293 8.77051C1.70868 7.43439 2.98545 6.5 4.50098 6.5H7.50098C9.01528 6.50015 10.291 7.43307 10.9707 8.76758C10.9877 8.64746 11 8.5252 11 8.40039V3.59961C10.9998 2.16398 9.83602 1.00021 8.40039 1H3.59961Z" />
    </svg>
  )
}

function IconPlay({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 9 11" fill="currentColor">
      <path fillRule="evenodd" clipRule="evenodd" d="M1.03927 1.03269V9.96731L7.91655 5.5L1.03927 1.03269ZM0 0.928981C0 0.182271 0.886347 -0.25826 1.5376 0.164775L8.57453 4.73579C9.14182 5.10429 9.14182 5.89571 8.57453 6.2642L1.5376 10.8352C0.88635 11.2583 0 10.8177 0 10.071V0.928981Z" />
    </svg>
  )
}

function IconChevronDown({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
    </svg>
  )
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result)
        return
      }
      reject(new Error('Unable to read image preview'))
    }
    reader.onerror = () => reject(reader.error ?? new Error('Unable to read image preview'))
    reader.readAsDataURL(file)
  })
}

function fileFromDataUrl(dataUrl: string, fileName: string): File | undefined {
  try {
    const [header, payload] = dataUrl.split(',', 2)
    if (!header || !payload || !header.startsWith('data:')) return undefined
    const mimeType = header.match(/data:([^;]+)/)?.[1] || 'image/png'
    const binary = atob(payload)
    const bytes = new Uint8Array(binary.length)
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index)
    }
    return new File([bytes], fileName || 'search-image', { type: mimeType })
  } catch {
    return undefined
  }
}

function normalizePersistedSearchAttachments(parsed: SearchSessionResult): SearchAttachment[] {
  if (Array.isArray(parsed.attachments) && parsed.attachments.length > 0) {
    return parsed.attachments.map((attachment) => ({
      ...attachment,
      file: attachment.type === 'image' && attachment.previewUrl.startsWith('data:')
        ? fileFromDataUrl(attachment.previewUrl, attachment.name)
        : undefined,
    }))
  }

  if (!Array.isArray(parsed.entities)) return []
  return parsed.entities.map((entity) => ({
    id: entity.id,
    type: 'entity' as const,
    name: entity.name,
    previewUrl: entity.previewUrl,
  }))
}

function SearchAttachmentChip({
  attachment,
  onRemove,
  compact = false,
}: {
  attachment: SearchAttachment | SearchSessionAttachment
  onRemove?: () => void
  compact?: boolean
}) {
  const isEntity = attachment.type === 'entity'
  const initials = attachment.name.trim().slice(0, 2).toUpperCase() || (isEntity ? 'EN' : 'IM')

  return (
    <span className="inline-flex items-center gap-1.5 pl-1 pr-2 py-0.5 rounded-full border border-border bg-card text-text-secondary">
      {attachment.previewUrl ? (
        <img
          src={attachment.previewUrl}
          alt={attachment.name}
          className={`${compact ? 'w-5 h-5' : 'w-6 h-6'} object-cover ${isEntity ? 'rounded-full' : 'rounded'}`}
        />
      ) : (
        <span
          className={`${compact ? 'w-5 h-5 text-[10px]' : 'w-6 h-6 text-[10px]'} ${
            isEntity ? 'rounded-full' : 'rounded'
          } bg-gray-200 text-gray-600 flex items-center justify-center font-medium shrink-0`}
          aria-hidden
        >
          {initials}
        </span>
      )}
      <span className={`max-w-[120px] truncate font-medium ${compact ? 'text-[11px]' : 'text-xs'}`}>{attachment.name}</span>
      {onRemove ? (
        <button
          type="button"
          onClick={onRemove}
          className="ml-0.5 p-0.5 rounded-full hover:bg-gray-200 text-gray-400 hover:text-gray-600 transition-colors"
          aria-label={`Remove ${attachment.name}`}
        >
          <svg className="w-3 h-3" viewBox="0 0 12 12" fill="currentColor">
            <path d="M6.02 5.31L8.97 2.37l.71.7L6.73 6.02l2.93 2.93-.71.71L6.02 6.73 3.07 9.67l-.7-.7L5.31 6.02 2.35 3.05l.7-.7L6.02 5.31Z" />
          </svg>
        </button>
      ) : null}
    </span>
  )
}

function SearchQueryChip({ queryText }: { queryText: string }) {
  return (
    <span className="inline-flex items-center px-3 py-1 rounded-full border border-border bg-card text-sm text-text-primary max-w-full">
      <span className="truncate">{queryText}</span>
    </span>
  )
}

function EntityAvatars({ entities }: { entities: TableEntity[] }) {
  if (!entities?.length) return <span className="text-sm text-gray-400">—</span>
  const maxCircles = 4
  const show = entities.slice(0, maxCircles)
  const rest = entities.length - show.length
  return (
    <div className="flex items-center gap-0.5">
      <div className="flex items-center -space-x-2.5">
        {show.map((e) => (
          <div
            key={e.id}
            className="relative w-7 h-7 rounded-full border-2 border-surface overflow-hidden bg-card shrink-0 flex items-center justify-center ring-1 ring-white"
            title={e.name}
          >
            {e.imageUrl ? (
              <img src={e.imageUrl} alt="" className="w-full h-full object-cover" />
            ) : (
              <span className="text-[10px] font-medium text-gray-600">{e.initials}</span>
            )}
          </div>
        ))}
      </div>
      {rest > 0 && (
        <span
          className="ml-1.5 min-w-[1.5rem] h-6 px-1.5 rounded-full bg-gray-100 text-gray-600 flex items-center justify-center text-xs font-medium shrink-0"
          title={`${rest} more`}
        >
          +{rest}
        </span>
      )}
    </div>
  )
}

function TagPills({ tags }: { tags: string[] }) {
  if (!tags?.length) return <span className="text-sm text-gray-400">—</span>
  return (
    <div className="flex flex-wrap gap-1.5">
      {tags.map((tag) => (
        <span
          key={tag}
          className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-gray-100 text-gray-700 border border-gray-200/80"
        >
          {tag}
        </span>
      ))}
    </div>
  )
}

type TableEntity = { id: string; name: string; imageUrl?: string; initials: string }

type ClipMatch = {
  start: number
  end: number
  score?: number
  type: string
  rank?: number
  thumbnailUrl?: string
}

type VideoItem = {
  id: string
  title: string
  uploadDate: string
  duration: string
  totalMinutes: number
  category: string
  tags?: string[]
  entities?: TableEntity[]
  streamUrl?: string
  thumbnailUrl?: string
  clips?: ClipMatch[]
  searchScore?: number
  bestRank?: number
}

function relevanceLabel(clips: ClipMatch[] | undefined): { label: string; color: string } {
  if (!clips || clips.length === 0) return { label: '', color: '' }
  const rankedClips = clips.filter((clip) => typeof clip.rank === 'number' && Number.isFinite(clip.rank))
  if (rankedClips.length > 0) {
    const bestRank = Math.min(...rankedClips.map((clip) => clip.rank as number))
    if (bestRank <= 1) return { label: 'Rank 1', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' }
    if (bestRank <= 3) return { label: `Rank ${bestRank}`, color: 'bg-green-100 text-green-800 border-green-300' }
    if (bestRank <= 5) return { label: `Rank ${bestRank}`, color: 'bg-yellow-100 text-yellow-800 border-yellow-300' }
    return { label: `Rank ${bestRank}`, color: 'bg-slate-100 text-slate-700 border-slate-300' }
  }

  const scoredClips = clips.filter((clip) => typeof clip.score === 'number' && Number.isFinite(clip.score))
  if (scoredClips.length === 0) return { label: 'Match', color: 'bg-slate-100 text-slate-700 border-slate-300' }

  const bestScore = Math.max(...scoredClips.map((clip) => clip.score as number))
  if (bestScore >= 0.10) return { label: 'Highest', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' }
  if (bestScore >= 0.08) return { label: 'High', color: 'bg-green-100 text-green-800 border-green-300' }
  if (bestScore >= 0.06) return { label: 'Medium', color: 'bg-yellow-100 text-yellow-800 border-yellow-300' }
  return { label: 'Match', color: 'bg-slate-100 text-slate-700 border-slate-300' }
}

function formatTotalDuration(videos: VideoItem[], videoDurations?: Record<string, number>) {
  const totalSeconds = videos.reduce((sum, v) => {
    const sec = videoDurations?.[v.id]
    if (sec != null && Number.isFinite(sec)) return sum + sec
    return sum + v.totalMinutes * 60
  }, 0)
  if (totalSeconds <= 0) return '0 min'
  const h = Math.floor(totalSeconds / 3600)
  const m = Math.floor((totalSeconds % 3600) / 60)
  if (h > 0) return `${h} h ${m} min`
  return `${m} min`
}

function formatDurationHHMMSS(duration: string): string {
  const parts = duration.split(':').map(Number)
  if (parts.length >= 3) {
    const [h = 0, m = 0, s = 0] = parts
    const totalM = h * 60 + m
    return `${totalM.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }
  return duration
}

function formatDurationShort(duration: string): string {
  const parts = duration.split(':').map(Number)
  if (parts.length >= 3) {
    const [h, m, s] = parts
    if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
    return `${m}:${s.toString().padStart(2, '0')}`
  }
  return duration
}

function formatSecondsToTimestamp(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '—'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  return `${m}:${s.toString().padStart(2, '0')}`
}

function normalizeEntityOption(entity: any): EntityOption | null {
  const id = entity?.id || entity?.entity_id
  if (!id) return null
  const metadata = entity?.metadata || {}
  const faceB64 = metadata.face_snap_base64
  const previewUrl = faceB64 ? `data:image/png;base64,${faceB64}` : ''
  return {
    id,
    name: metadata.name || entity?.name || id,
    previewUrl,
  }
}

type DashboardProps = { onOpenUpload?: () => void }

export default function Dashboard({ onOpenUpload }: DashboardProps) {
  const [addImageModalOpen, setAddImageModalOpen] = useState(false)
  const [addEntityModalOpen, setAddEntityModalOpen] = useState(false)
  const [searchAttachments, setSearchAttachments] = useState<SearchAttachment[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy] = useState<'recent' | 'name'>('recent')
  const [activeCategory] = useState<string>('All')
  const { videos: cachedVideos } = useVideoCache()
  const apiVideos = useMemo<VideoItem[]>(() => {
    return cachedVideos.map((v: CachedVideo) => {
      const meta = v.metadata || {}
      const dateStr = meta.created_at || meta.uploaded_at || ''
      let uploadDate = ''
      if (dateStr) {
        try {
          const d = new Date(dateStr)
          uploadDate = `${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}-${d.getFullYear()}`
        } catch { uploadDate = dateStr }
      }
      const dur = meta.duration
      return {
        id: v.id,
        title: meta.filename || v.id,
        uploadDate,
        duration: dur != null && Number.isFinite(dur) ? formatSecondsToTimestamp(dur) : '—',
        totalMinutes: dur != null && Number.isFinite(dur) ? dur / 60 : 0,
        category: 'Uploaded',
        tags: [
          ...(Array.isArray(meta.tags) ? meta.tags : []),
          meta.indexed_at ? 'Indexed' : 'Uploaded',
        ],
        entities: [],
        streamUrl: v.stream_url || undefined,
        thumbnailUrl: v.thumbnail_url || undefined,
      }
    })
  }, [cachedVideos])

  const [videoDurations, setVideoDurations] = useState<Record<string, number>>({})
  const [videoLoadFailed, setVideoLoadFailed] = useState<Record<string, boolean>>({})
  const [videoLoaded, setVideoLoaded] = useState<Record<string, boolean>>({})
  const videoLoadedRef = useRef<Record<string, boolean>>({})
  videoLoadedRef.current = videoLoaded
  const SEARCH_PLACEHOLDERS = [
    'Search actions, objects, sounds and logos',
    "Search with entities (@ + name)",
    'Search with image and text across videos',
  ]
  const [placeholderIdx, setPlaceholderIdx] = useState(0)
  useEffect(() => {
    if (searchQuery || searchAttachments.length) return
    const timer = setInterval(() => {
      setPlaceholderIdx((i) => (i + 1) % SEARCH_PLACEHOLDERS.length)
    }, 6000)
    return () => clearInterval(timer)
  }, [searchQuery, searchAttachments.length])

  const [searchSuggestionsOpen, setSearchSuggestionsOpen] = useState(false)
  const [searchResults, setSearchResults] = useState<SearchSessionResult | null>(null)
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [entitiesList, setEntitiesList] = useState<EntityOption[]>([])
  const entityDropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    fetch(`${API_BASE}/api/entities`)
      .then((res) => res.json())
      .then((data: { entities?: any[]; unavailable?: boolean }) => {
        if (cancelled || data.unavailable) { setEntitiesList([]); return }
        const list: EntityOption[] = (data.entities || [])
          .map((e: any) => normalizeEntityOption(e))
          .filter(Boolean) as EntityOption[]
        setEntitiesList(list)
      })
      .catch(() => { if (!cancelled) setEntitiesList([]) })
    return () => { cancelled = true }
  }, [])

  const { entityMentionQuery, entityDropdownVisible, filteredEntities, queryBeforeMention } = useMemo(() => {
    const lastAt = searchQuery.lastIndexOf('@')
    if (lastAt < 0) {
      return {
        entityMentionQuery: '',
        entityDropdownVisible: false,
        filteredEntities: [] as EntityOption[],
        queryBeforeMention: searchQuery,
      }
    }
    const afterAt = searchQuery.slice(lastAt + 1)
    const hasSpace = afterAt.includes(' ')
    const mentionQuery = hasSpace ? afterAt.split(/\s/)[0] || '' : afterAt
    const filter = mentionQuery.toLowerCase()
    const filtered = filter
      ? entitiesList.filter((e) => e.name.toLowerCase().includes(filter))
      : entitiesList
    return {
      entityMentionQuery: mentionQuery,
      entityDropdownVisible: true,
      filteredEntities: filtered.slice(0, 8),
      queryBeforeMention: searchQuery.slice(0, lastAt),
    }
  }, [searchQuery, entitiesList])

  useEffect(() => {
    if (!entityDropdownVisible) return
    function handleClickOutside(e: MouseEvent) {
      if (entityDropdownRef.current && !entityDropdownRef.current.contains(e.target as Node)) {
        setSearchQuery((q) => {
          const lastAt = q.lastIndexOf('@')
          if (lastAt < 0) return q
          return q.slice(0, lastAt)
        })
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [entityDropdownVisible])

  const allVideos = useMemo(() => [...apiVideos], [apiVideos])

  const PINNED_VIDEO_IDS = ['699fa0975ce336753bf904c2', '69b677662fc4a03916fba00e']
  const filteredVideos = useMemo(() => {
    if (searchResults) return searchResults.results
    let list = allVideos
    if (activeCategory !== 'All') {
      list = list.filter((v) => v.category === activeCategory)
    }
    if (sortBy === 'name') {
      list = [...list].sort((a, b) => a.title.localeCompare(b.title))
    }
    const pinnedSet = new Set(PINNED_VIDEO_IDS)
    const pinned = PINNED_VIDEO_IDS.map((id) => list.find((v) => v.id === id)).filter(Boolean) as VideoItem[]
    const rest = list.filter((v) => !pinnedSet.has(v.id))
    return [...pinned, ...rest]
  }, [allVideos, sortBy, activeCategory, searchResults])

  async function handleSearch() {
    const query = searchQuery.trim()
    const imageAttachments = searchAttachments.filter(
      (a): a is SearchAttachment & { type: 'image'; file: File } => a.type === 'image' && !!a.file,
    )
    const entityAttachments = searchAttachments.filter(
      (a): a is SearchAttachment & { type: 'entity' } => a.type === 'entity',
    )
    const hasQuery = query.length > 0
    const hasImages = imageAttachments.length > 0
    const hasEntities = entityAttachments.length > 0
    if (!hasQuery && !hasImages && !hasEntities) return
    setSearchError(null)
    setSearchLoading(true)
    try {
      type RawClip = { start: number; end: number; score?: number | null; rank?: number; thumbnail_url?: string }
      type RawResult = { video_id: string; score?: number | null; clips?: RawClip[] }
      const dedupeRawClips = (clips: RawClip[]) => {
        const seen = new Set<string>()
        return clips.filter((clip) => {
          const key = [
            clip.start,
            clip.end,
            clip.rank ?? '',
            clip.score ?? '',
            clip.thumbnail_url ?? '',
          ].join('|')
          if (seen.has(key)) return false
          seen.add(key)
          return true
        })
      }
      const composedQuery = [
        ...entityAttachments.map((attachment) => `<@${attachment.id}>`),
        query,
      ].filter(Boolean).join(' ').trim()
      const operator = hasImages && (imageAttachments.length > 1 || composedQuery.length > 0)
        ? 'and'
        : undefined

      const res = hasImages
        ? await (async () => {
          const formData = new FormData()
          for (const attachment of imageAttachments) {
            formData.append('image', attachment.file)
          }
          if (composedQuery) formData.append('query', composedQuery)
          if (operator) formData.append('operator', operator)
          return fetch(`${API_BASE}/api/search`, {
            method: 'POST',
            body: formData,
          })
        })()
        : await fetch(`${API_BASE}/api/search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: composedQuery }),
        })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        setSearchError(data.error || 'Search failed')
        setSearchResults(null)
        return
      }

      const videoLookup = new Map(cachedVideos.map((v) => [v.id, v]))

      const rawResults = Array.isArray(data.results) ? (data.results as RawResult[]) : []
      const results: VideoItem[] = rawResults.map((r) => {
        const cached = videoLookup.get(r.video_id)
        const meta = cached?.metadata || {}
        let uploadDate = ''
        try {
          const u = meta.created_at || ''
          if (u) {
            const d = new Date(u)
            uploadDate = `${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}-${d.getFullYear()}`
          }
        } catch {}
        const normalizedClips = dedupeRawClips(r.clips || []).map((c) => ({
          start: c.start,
          end: c.end,
          score: typeof c.score === 'number' && Number.isFinite(c.score) ? c.score : undefined,
          rank: typeof c.rank === 'number' && Number.isFinite(c.rank) ? c.rank : undefined,
          thumbnailUrl: c.thumbnail_url || undefined,
        }))
        const rankedClips = normalizedClips.filter((c) => typeof c.rank === 'number')
        const bestRank = rankedClips.length > 0
          ? Math.min(...rankedClips.map((c) => c.rank as number))
          : undefined
        const scoredClips = normalizedClips.filter((c) => typeof c.score === 'number')
        const bestScore = scoredClips.length > 0
          ? Math.max(...scoredClips.map((c) => c.score as number))
          : (typeof r.score === 'number' && Number.isFinite(r.score) ? r.score : undefined)
        return {
          id: r.video_id,
          title: meta.filename || r.video_id,
          uploadDate,
          duration: meta.duration ? formatSecondsToTimestamp(meta.duration) : '—',
          totalMinutes: meta.duration ? meta.duration / 60 : 0,
          category: 'Uploaded',
          tags: [],
          entities: [],
          streamUrl: cached?.stream_url || undefined,
          thumbnailUrl: cached?.thumbnail_url || undefined,
          clips: normalizedClips.map((c) => ({
            start: c.start,
            end: c.end,
            score: c.score,
            type: 'visual',
            rank: c.rank,
            thumbnailUrl: c.thumbnailUrl,
          })),
          bestRank,
          searchScore: bestScore,
        }
      })

      results.sort((a, b) => {
        const rankA = a.bestRank ?? Number.POSITIVE_INFINITY
        const rankB = b.bestRank ?? Number.POSITIVE_INFINITY
        if (rankA !== rankB) return rankA - rankB

        const scoreA = a.searchScore ?? Number.NEGATIVE_INFINITY
        const scoreB = b.searchScore ?? Number.NEGATIVE_INFINITY
        if (scoreA !== scoreB) return scoreB - scoreA

        return a.title.localeCompare(b.title)
      })

      const displayParts: string[] = []
      if (hasQuery) displayParts.push(query)
      if (hasImages) displayParts.push(`${imageAttachments.length} image${imageAttachments.length === 1 ? '' : 's'}`)
      if (hasEntities) {
        displayParts.push(`Entity: ${entityAttachments.map((a) => a.name).join(', ')}`)
      }
      const displayQuery = displayParts.join(' + ')
      const persistedAttachments: SearchSessionAttachment[] = searchAttachments.map((attachment) => ({
        id: attachment.id,
        type: attachment.type,
        name: attachment.name,
        previewUrl: attachment.previewUrl,
      }))
      const entities = persistedAttachments
        .filter((attachment): attachment is SearchSessionAttachment & { type: 'entity' } => attachment.type === 'entity')
        .map((attachment) => ({
          id: attachment.id,
          name: attachment.name,
          previewUrl: attachment.previewUrl,
        }))
      setSearchResults({
        query: displayQuery,
        queryText: hasQuery ? query : '',
        attachments: persistedAttachments,
        entities,
        results,
      })
      try {
        const sessionPayload: SearchSessionResult = {
          query: displayQuery,
          queryText: hasQuery ? query : '',
          attachments: persistedAttachments,
          entities,
          results,
        }
        sessionStorage.setItem('video_redaction_last_search', JSON.stringify(sessionPayload))
      } catch {}
    } catch (e) {
      setSearchError('Search request failed')
      setSearchResults(null)
    } finally {
      setSearchLoading(false)
    }
  }

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem('video_redaction_last_search')
      if (raw) {
        const parsed = JSON.parse(raw) as SearchSessionResult
        if (parsed?.query != null && Array.isArray(parsed.results)) {
          const restoredAttachments = normalizePersistedSearchAttachments(parsed)
          const restoredQuery = typeof parsed.queryText === 'string'
            ? parsed.queryText
            : restoredAttachments.length > 0
              ? ''
              : parsed.query
          setSearchQuery(restoredQuery)
          setSearchAttachments(restoredAttachments)
          setSearchResults({
            query: parsed.query,
            queryText: typeof parsed.queryText === 'string' ? parsed.queryText : '',
            attachments: restoredAttachments.map((attachment) => ({
              id: attachment.id,
              type: attachment.type,
              name: attachment.name,
              previewUrl: attachment.previewUrl,
            })),
            entities: Array.isArray(parsed.entities) ? parsed.entities : restoredAttachments
              .filter((attachment): attachment is SearchAttachment & { type: 'entity' } => attachment.type === 'entity')
              .map((attachment) => ({
                id: attachment.id,
                name: attachment.name,
                previewUrl: attachment.previewUrl,
              })),
            results: parsed.results,
          })
        }
      }
    } catch {}
  }, [])

  function clearSearch() {
    setSearchResults(null)
    setSearchError(null)
    try {
      sessionStorage.removeItem('video_redaction_last_search')
    } catch {}
  }

  function removeSearchAttachment(attachmentId: string) {
    setSearchAttachments((prev) => {
      const attachment = prev.find((item) => item.id === attachmentId)
      if (attachment?.type === 'image' && attachment.previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(attachment.previewUrl)
      }
      return prev.filter((item) => item.id !== attachmentId)
    })
  }

  function selectEntity(entity: EntityOption) {
    setSearchAttachments((prev) => {
      if (prev.some((a) => a.type === 'entity' && a.id === entity.id)) return prev
      return [...prev, { id: entity.id, type: 'entity' as const, name: entity.name, previewUrl: entity.previewUrl || '' }]
    })
    setSearchQuery((q) => {
      const lastAt = q.lastIndexOf('@')
      if (lastAt < 0) return q
      const before = q.slice(0, lastAt).trimEnd()
      return before ? `${before} ` : ''
    })
  }

  return (
    <div className="w-full min-w-0">
      <div className="search-bar-gradient-outer mb-4 shadow-sm w-full min-w-0">
        <div className="search-bar-gradient-border-wrap">
          <div className="search-bar-gradient-border" aria-hidden />
        </div>
        <div className="search-bar-gradient-inner w-full">
          <div className="px-3 sm:px-4 pt-4 pb-2 min-w-0 w-full">
            <div className="flex flex-wrap items-center gap-2 min-w-0">
              {searchAttachments.map((att) => (
                <SearchAttachmentChip
                  key={att.id}
                  attachment={att}
                  onRemove={() => removeSearchAttachment(att.id)}
                />
              ))}
              <div ref={entityDropdownRef} className="relative flex-1 min-w-[100px] sm:min-w-[120px] overflow-visible">
                {!searchQuery && !searchAttachments.length && (
                  <div className="pointer-events-none absolute inset-0 flex items-center overflow-hidden">
                    <div
                      key={placeholderIdx}
                      className="text-base sm:text-lg text-text-tertiary whitespace-nowrap animate-placeholder-slide"
                    >
                      {SEARCH_PLACEHOLDERS[placeholderIdx]}
                    </div>
                  </div>
                )}
                <input
                  type="text"
                  placeholder={searchAttachments.length ? 'Add search terms...' : ''}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (entityDropdownVisible) {
                      if (e.key === 'Escape') {
                        setSearchQuery((q) => (q.lastIndexOf('@') >= 0 ? q.slice(0, q.lastIndexOf('@')) : q))
                        e.preventDefault()
                        return
                      }
                      if (e.key === 'Enter' && filteredEntities.length > 0) {
                        selectEntity(filteredEntities[0])
                        e.preventDefault()
                        return
                      }
                    }
                    if (e.key === 'Enter') { e.preventDefault(); handleSearch(); }
                  }}
                  className="relative w-full text-base sm:text-lg text-text-primary placeholder:text-text-tertiary bg-transparent focus:outline-none z-[1]"
                  aria-label="Search videos"
                  aria-autocomplete="list"
                  aria-expanded={entityDropdownVisible}
                  aria-controls="entity-mention-list"
                />
                {entityDropdownVisible && (
                  <ul
                    id="entity-mention-list"
                    role="listbox"
                    className="absolute left-0 right-0 top-full mt-1 z-[100] max-h-56 overflow-auto rounded-lg border border-border bg-surface shadow-lg py-1 min-w-[200px]"
                  >
                    {filteredEntities.length === 0 ? (
                      <li className="px-3 py-2 text-sm text-text-tertiary">No matching entities</li>
                    ) : (
                      filteredEntities.map((entity) => (
                        <li key={entity.id} role="option">
                          <button
                            type="button"
                            onClick={() => selectEntity(entity)}
                            className="w-full flex items-center gap-3 px-3 py-2 text-left text-sm text-text-primary hover:bg-card transition-colors"
                          >
                            {entity.previewUrl ? (
                              <img src={entity.previewUrl} alt="" className="w-8 h-8 rounded-full object-cover shrink-0" />
                            ) : (
                              <span className="w-8 h-8 rounded-full bg-gray-200 text-gray-600 flex items-center justify-center text-xs font-medium shrink-0">
                                {entity.name.slice(0, 2).toUpperCase()}
                              </span>
                            )}
                            <span className="font-medium truncate">{entity.name}</span>
                          </button>
                        </li>
                      ))
                    )}
                  </ul>
                )}
              </div>
            </div>
          </div>
          <div className="px-3 sm:px-4 pb-4 flex flex-wrap items-center justify-between gap-2 w-full">
            <div className="flex flex-wrap items-center gap-2 min-w-0">
              <button
                type="button"
                onClick={() => setAddImageModalOpen(true)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-border bg-surface text-sm text-text-secondary hover:bg-card transition-colors"
              >
                <IconAddImage className="w-4 h-4 shrink-0" />
                <span className="hidden sm:inline">Add Image</span>
              </button>
              <button
                type="button"
                onClick={() => setAddEntityModalOpen(true)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-border bg-surface text-sm text-text-secondary hover:bg-card transition-colors"
              >
                <IconEntity className="w-3.5 h-3.5 shrink-0" />
                <span className="hidden sm:inline">Add Entity</span>
                <span className="hidden sm:inline text-[10px] font-medium bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded">BETA</span>
              </button>
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setSearchSuggestionsOpen((open) => !open)}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-border bg-surface text-sm text-text-secondary hover:bg-card hover:text-text-primary transition-colors"
                  aria-haspopup="true"
                  aria-expanded={searchSuggestionsOpen}
                >
                  SUGGESTIONS
                  <IconChevronDown className={`w-3 h-3 shrink-0 transition-transform ${searchSuggestionsOpen ? 'rotate-180' : ''}`} />
                </button>
                {searchSuggestionsOpen && (
                  <div className="absolute left-0 mt-1 w-72 max-w-xs rounded-xl border border-border bg-surface shadow-xl z-50 py-1">
                    {SEARCH_SUGGESTIONS.map((suggestion) => (
                      <button
                        key={suggestion}
                        type="button"
                        onClick={() => {
                          setSearchQuery(suggestion)
                          setSearchSuggestionsOpen(false)
                        }}
                        className="w-full px-3 py-1.5 text-left text-[11px] text-text-secondary hover:bg-card hover:text-text-primary transition-colors"
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
            <button
              type="button"
              disabled={(!searchQuery.trim() && searchAttachments.length === 0) || searchLoading}
              onClick={() => handleSearch()}
              className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors shrink-0 ${
                (searchQuery.trim() || searchAttachments.length > 0) && !searchLoading
                  ? 'bg-brand-charcoal text-brand-white hover:bg-gray-600 cursor-pointer'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
              aria-label="Search"
            >
              {searchLoading ? (
                <span className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden />
              ) : (
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="7" />
                  <line x1="16.65" y1="16.65" x2="21" y2="21" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {searchError && (
        <p className="mb-3 text-sm text-red-600" role="alert">
          {searchError}
        </p>
      )}
      {searchResults && (
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <span className="text-sm text-text-secondary">
            {searchResults.results.length === 0
              ? 'No videos found for'
              : `${searchResults.results.length} result${searchResults.results.length !== 1 ? 's' : ''} for`}
          </span>
          {searchResults.queryText ? <SearchQueryChip queryText={searchResults.queryText} /> : null}
          {(searchResults.attachments || []).map((attachment) => (
            <SearchAttachmentChip
              key={`summary-${attachment.id}`}
              attachment={attachment}
              compact
            />
          ))}
          {!searchResults.queryText && !(searchResults.attachments || []).length ? (
            <span className="text-sm text-text-secondary">“{searchResults.query}”</span>
          ) : null}
          <button
            type="button"
            onClick={clearSearch}
            className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-brand-charcoal bg-card border border-border rounded-lg hover:bg-gray-200/80 hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-brand-charcoal/30 focus:ring-offset-1 transition-colors"
          >
            Clear search
          </button>
        </div>
      )}

      <div className="dashboard-video-grid">
            {onOpenUpload && (
              <button
                type="button"
                onClick={onOpenUpload}
                className="group flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-border bg-card hover:border-gray-400 hover:bg-gray-200/80 transition-all duration-200 text-left focus:outline-none focus:ring-2 focus:ring-accent/30 focus:ring-offset-2 aspect-video min-w-0 py-4 px-3"
              >
                <span className="flex items-center justify-center w-11 h-11 text-text-tertiary group-hover:text-text-secondary transition-colors mb-3">
                  <img src={arrowBoxUpIconUrl} alt="" className="w-5 h-5" aria-hidden />
                </span>
                <p className="text-sm font-semibold text-text-primary">Drop videos or browse files</p>
                <div className="flex flex-wrap items-center justify-center gap-2 mt-2">
                  <span className="inline-flex px-2.5 py-0.5 rounded-full text-xs font-medium text-text-secondary border border-border">
                    MP4, MOV, AVI
                  </span>
                  <span className="inline-flex px-2.5 py-0.5 rounded-full text-xs font-medium text-text-secondary border border-border">
                    Max 300 MB
                  </span>
                </div>
                <p className="mt-2.5 text-[11px] sm:text-xs text-text-tertiary max-w-[200px] text-center leading-snug">
                  Processing takes ~2–3 min (indexing, tool detection, analysis)
                </p>
              </button>
            )}
            {filteredVideos.map((v) => (
                <Link
                  key={v.id}
                  to={`/editor/${v.id}`}
                  className="group block focus:outline-none focus:ring-2 focus:ring-accent/30 rounded-xl min-w-0"
                >
                  <div className="relative aspect-video rounded-xl overflow-hidden bg-brand-charcoal">
                    <img
                      src={`/generated-thumbnails/${v.id}.jpg`}
                      alt={v.title}
                      loading="eager"
                      decoding="sync"
                      className="absolute inset-0 w-full h-full object-cover"
                      style={{ zIndex: 5 }}
                      onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none' }}
                    />
                    {v.streamUrl && (
                      <>
                        <div
                          className={`absolute inset-0 flex items-center justify-center bg-brand-charcoal z-[1] transition-opacity duration-200 ${videoLoaded[v.id] && !videoLoadFailed[v.id] ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}
                          aria-hidden
                        >
                          <div className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center">
                            <IconPlay className="w-5 h-5 text-white ml-0.5" />
                          </div>
                        </div>
                        <video
                          src={v.streamUrl}
                          className={`absolute inset-0 w-full h-full object-cover brightness-105 z-0 transition-opacity duration-200 ${videoLoaded[v.id] ? 'opacity-100' : 'opacity-0'}`}
                          muted
                          loop
                          playsInline
                          preload="metadata"
                          aria-label={v.title}
                          data-video-id={v.id}
                          onLoadedMetadata={(e) => {
                            const el = e.currentTarget
                            const id = el.dataset.videoId
                            if (id && Number.isFinite(el.duration)) {
                              setVideoDurations((prev) => (prev[id] === el.duration ? prev : { ...prev, [id]: el.duration }))
                            }
                          }}
                          onLoadedData={(e) => {
                            const id = (e.currentTarget as HTMLVideoElement).dataset.videoId
                            if (id) setVideoLoaded((prev) => ({ ...prev, [id]: true }))
                          }}
                          onError={() => {
                            setVideoLoadFailed((prev) => ({ ...prev, [v.id]: true }))
                          }}
                        />
                      </>
                    )}
                    <div className="absolute inset-0 flex items-center justify-center bg-transparent group-hover:bg-brand-charcoal/40 pointer-events-none transition-colors duration-200" style={{ zIndex: 10 }}>
                      <div className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                        <IconPlay className="w-5 h-5 text-white ml-0.5" />
                      </div>
                    </div>
                    {(() => {
                      const rel = relevanceLabel(v.clips)
                      return rel.label ? (
                        <span className={`absolute top-2 left-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold border ${rel.color}`} style={{ zIndex: 11 }}>
                          {rel.label}
                        </span>
                      ) : null
                    })()}
                    <span className="absolute left-1/2 -translate-x-1/2 bottom-1.5 px-3 py-1 text-sm font-mono font-medium text-white rounded border border-white/90 tabular-nums bg-black/50 [text-shadow:0_0_2px_rgba(0,0,0,0.9)]" style={{ zIndex: 11 }}>
                      {videoDurations[v.id] != null ? formatSecondsToTimestamp(videoDurations[v.id]) : v.duration !== '—' ? formatDurationHHMMSS(v.duration) : '—'}
                    </span>
                  </div>
                  <p className="mt-2.5 text-sm text-text-secondary truncate group-hover:text-text-primary transition-colors">
                    {v.title}
                  </p>
                </Link>
              ))}
            </div>

      <AddImageModal
        open={addImageModalOpen}
        onClose={() => setAddImageModalOpen(false)}
        onImageAdded={async (file) => {
          let previewUrl = ''
          try {
            previewUrl = await readFileAsDataUrl(file)
          } catch {
            previewUrl = URL.createObjectURL(file)
          }
          setSearchAttachments((prev) => [
            ...prev,
            { id: `img-${Date.now()}`, type: 'image', name: file.name, previewUrl, file },
          ])
          setAddImageModalOpen(false)
        }}
      />
      <AddEntityModal
        open={addEntityModalOpen}
        onClose={() => setAddEntityModalOpen(false)}
        onEntityAdded={(selection) => {
          const entityId = selection.id
          if (!entityId) return
          setSearchAttachments((prev) => [
            ...prev,
            { id: entityId, type: 'entity', name: selection.name?.trim() || 'Entity', previewUrl: selection.previewUrl },
          ])
          setAddEntityModalOpen(false)
        }}
      />
    </div>
  )
}
