import { createContext, useContext, useState, useCallback, useRef, useEffect, type ReactNode } from 'react'
import { API_BASE } from '../lib/api'

const CACHE_KEY = 'video_redaction_cache'
const CACHE_STALE_MS = 60_000

export type CachedVideo = {
  id: string
  stream_url?: string
  thumbnail_url?: string
  metadata: Record<string, any>
}

type VideoCacheState = {
  videos: CachedVideo[]
  loading: boolean
  error: string | null
  lastFetchedAt: number
  getVideo: (id: string) => CachedVideo | undefined
  refresh: (force?: boolean) => Promise<void>
}

const VideoCacheContext = createContext<VideoCacheState | null>(null)

function loadFromStorage(): CachedVideo[] {
  try {
    const raw = localStorage.getItem(CACHE_KEY)
    if (raw) return JSON.parse(raw) as CachedVideo[]
  } catch { /* ignore */ }
  return []
}

function saveToStorage(videos: CachedVideo[]) {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(videos))
  } catch { /* ignore */ }
}

export function VideoCacheProvider({ children }: { children: ReactNode }) {
  const [videos, setVideos] = useState<CachedVideo[]>(loadFromStorage)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastFetchedAt, setLastFetchedAt] = useState(0)
  const fetchingRef = useRef(false)
  const mountedRef = useRef(true)

  const refresh = useCallback(async (force = false) => {
    if (fetchingRef.current) return
    if (!force && lastFetchedAt && Date.now() - lastFetchedAt < CACHE_STALE_MS) return

    fetchingRef.current = true
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/api/videos`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      const list: CachedVideo[] = (data.videos || []).map((v: any) => ({
        id: v.video_id || v.id,
        stream_url: v.hls_url || v.stream_url || undefined,
        thumbnail_url: v.thumbnail_url || undefined,
        metadata: {
          filename: v.system_metadata?.filename ?? v.metadata?.filename,
          duration: v.system_metadata?.duration ?? v.metadata?.duration,
          fps: v.system_metadata?.fps,
          width: v.system_metadata?.width,
          height: v.system_metadata?.height,
          size: v.system_metadata?.size,
          created_at: v.created_at,
          indexed_at: v.indexed_at,
          ...(v.metadata || {}),
        },
      }))
      if (mountedRef.current) {
        setVideos(list)
        saveToStorage(list)
        setLastFetchedAt(Date.now())
      }
    } catch (e: any) {
      if (mountedRef.current) setError(e.message || 'Failed to fetch videos')
    } finally {
      fetchingRef.current = false
      if (mountedRef.current) setLoading(false)
    }
  }, [lastFetchedAt])

  useEffect(() => {
    mountedRef.current = true
    refresh(true)
    return () => {
      mountedRef.current = false
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const getVideo = useCallback(
    (id: string) => videos.find((v) => v.id === id),
    [videos],
  )

  return (
    <VideoCacheContext.Provider value={{ videos, loading, error, lastFetchedAt, getVideo, refresh }}>
      {children}
    </VideoCacheContext.Provider>
  )
}

export function useVideoCache() {
  const ctx = useContext(VideoCacheContext)
  if (!ctx) throw new Error('useVideoCache must be used within VideoCacheProvider')
  return ctx
}
