export const DEMO_EDITOR_VIDEO_ID = '69b677662fc4a03916fba00e'
export const LAST_EDITOR_VIDEO_STORAGE_KEY = 'video_redaction_last_editor_video_id'

type VideoLike = {
  id: string
}

export function isEditorExperiencePath(pathname: string): boolean {
  return pathname === '/editor' || pathname.startsWith('/editor/') || pathname.startsWith('/video/')
}

export function readLastEditorVideoId(): string | null {
  if (typeof window === 'undefined') return null
  try {
    const stored = window.localStorage.getItem(LAST_EDITOR_VIDEO_STORAGE_KEY)
    const normalized = (stored || '').trim()
    return normalized || null
  } catch {
    return null
  }
}

export function storeLastEditorVideoId(videoId?: string | null) {
  const normalized = (videoId || '').trim()
  if (typeof window === 'undefined' || !normalized) return
  try {
    window.localStorage.setItem(LAST_EDITOR_VIDEO_STORAGE_KEY, normalized)
  } catch {
    /* ignore storage failures */
  }
}

export function pickEditorVideoId(
  videos: VideoLike[],
  options: { preferredId?: string | null } = {},
): string | null {
  const normalizedVideos = videos
    .map((video) => ({
      ...video,
      id: String(video.id || '').trim(),
    }))
    .filter((video) => video.id)

  const videoIds = new Set(normalizedVideos.map((video) => video.id))
  const preferredId = (options.preferredId || '').trim()
  if (preferredId && videoIds.has(preferredId)) {
    return preferredId
  }
  if (normalizedVideos.length > 0) {
    if (videoIds.has(DEMO_EDITOR_VIDEO_ID)) {
      return DEMO_EDITOR_VIDEO_ID
    }
    return normalizedVideos[0].id
  }
  return DEMO_EDITOR_VIDEO_ID
}
