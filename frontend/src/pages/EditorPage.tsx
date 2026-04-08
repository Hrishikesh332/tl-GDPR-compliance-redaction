import { useMemo } from 'react'
import { Link, Navigate } from 'react-router-dom'
import { useVideoCache } from '../contexts/VideoCache'
import { pickEditorVideoId, readLastEditorVideoId } from '../lib/editorRouting'

export default function EditorPage() {
  const { videos, loading, error } = useVideoCache()
  const lastEditorVideoId = useMemo(() => readLastEditorVideoId(), [])
  const targetVideoId = useMemo(
    () => pickEditorVideoId(videos, { preferredId: lastEditorVideoId }),
    [lastEditorVideoId, videos],
  )

  if (targetVideoId) {
    return <Navigate to={`/editor/${encodeURIComponent(targetVideoId)}`} replace />
  }

  return (
    <div className="w-full min-w-0 px-3 sm:px-4 py-4 sm:py-6">
      <div className="max-w-xl rounded-2xl border border-border bg-card px-5 py-6 shadow-sm">
        <p className="text-xs font-brand-xbold uppercase tracking-[0.14em] text-text-tertiary">
          Editor
        </p>
        <h2 className="mt-2 text-2xl font-brand-bold text-text-primary">
          {loading ? 'Opening the editor...' : 'No video is ready for the editor yet'}
        </h2>
        <p className="mt-3 text-sm leading-6 text-text-secondary">
          {loading
            ? 'We are loading the best available video for the live editor experience.'
            : error || 'Upload or index a video from the dashboard, then come back here to open the full redaction editor.'}
        </p>
        {!loading && (
          <div className="mt-5">
            <Link
              to="/dashboard"
              className="inline-flex items-center justify-center rounded-lg bg-brand-charcoal px-4 py-2 text-sm font-medium text-brand-white transition-colors hover:bg-gray-700"
            >
              Go to dashboard
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}
