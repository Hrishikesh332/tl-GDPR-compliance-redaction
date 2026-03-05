import { useState, useMemo, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import AddImageModal from '../components/AddImageModal'
import AddEntityModal, { type EntitySelection } from '../components/AddEntityModal'
import { useVideoCache, type CachedVideo } from '../contexts/VideoCache'
import { API_BASE } from '../lib/api'
import searchIconUrl from '../../strand/icons/search.svg?url'
import arrowBoxUpIconUrl from '../../strand/icons/arrow-box-up.svg?url'

type SearchAttachment = {
  id: string
  type: 'image' | 'entity'
  name: string
  previewUrl: string
}

type EntityOption = {
  id: string
  name: string
  previewUrl: string
}

function IconFilter({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 16 16"
      fill="none"
      preserveAspectRatio="xMidYMid meet"
    >
      <path
        fill="currentColor"
        fillRule="evenodd"
        clipRule="evenodd"
        d="M8 4.25H2v-1h6zm6 0h-2v-1h2zM6.667 12.25H2v-1h4.667zm7.333 0h-3.333v-1H14zM8 7.25h6v1H8zm-6 0h2v1H2zM9.668 3.417v.666h.667v-.666zm-.2-1a.8.8 0 0 0-.8.8v1.066a.8.8 0 0 0 .8.8h1.067a.8.8 0 0 0 .8-.8V3.217a.8.8 0 0 0-.8-.8zM5.668 7.417v.666h.667v-.666zm-.2-1a.8.8 0 0 0-.8.8v1.066a.8.8 0 0 0 .8.8h1.067a.8.8 0 0 0 .8-.8V7.217a.8.8 0 0 0-.8-.8zM8.334 11.417v.666h.667v-.666zm-.2-1a.8.8 0 0 0-.8.8v1.066a.8.8 0 0 0 .8.8h1.067a.8.8 0 0 0 .8-.8v-1.066a.8.8 0 0 0-.8-.8z"
      />
    </svg>
  )
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

function IconVision({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 12.9122 9.33301" fill="currentColor">
      <path d="M6.45506 3.66699C7.00735 3.66699 7.45506 4.11471 7.45506 4.66699C7.4548 5.21905 7.00718 5.66699 6.45506 5.66699C5.90302 5.6669 5.45533 5.21899 5.45506 4.66699C5.45506 4.11477 5.90286 3.66709 6.45506 3.66699Z" />
      <path fillRule="evenodd" clipRule="evenodd" d="M6.45604 2C7.92873 2.00008 9.12303 3.19428 9.12303 4.66699C9.12286 6.13955 7.92862 7.33293 6.45604 7.33301C4.98339 7.33301 3.78922 6.1396 3.78905 4.66699C3.78905 3.19423 4.98328 2 6.45604 2ZM6.45604 3C5.53556 3 4.78905 3.74652 4.78905 4.66699C4.78922 5.58732 5.53567 6.33301 6.45604 6.33301C7.37634 6.33293 8.12286 5.58727 8.12303 4.66699C8.12303 3.74657 7.37644 3.00008 6.45604 3Z" />
      <path fillRule="evenodd" clipRule="evenodd" d="M6.45604 0C9.52829 0 11.814 2.75324 12.709 4.03027C12.9799 4.41688 12.9799 4.91711 12.709 5.30371C11.8138 6.58091 9.52798 9.33301 6.45604 9.33301C3.38403 9.33284 1.09814 6.58075 0.203109 5.30371C-0.067777 4.9172 -0.0676288 4.41685 0.203109 4.03027C1.09808 2.75328 3.38393 0.000169362 6.45604 0ZM6.45604 1C5.22672 1.00008 4.10185 1.55189 3.13573 2.31934C2.17222 3.08473 1.44025 4.00742 1.02244 4.60352C1.00473 4.62879 0.999984 4.65085 0.999984 4.66699C1.00004 4.6831 1.00485 4.70439 1.02244 4.72949C1.44022 5.32558 2.1721 6.24916 3.13573 7.01465C4.10178 7.78197 5.22685 8.33292 6.45604 8.33301C7.68544 8.33301 8.81115 7.78213 9.77733 7.01465C10.741 6.2491 11.4728 5.32558 11.8906 4.72949C11.9082 4.70442 11.912 4.68308 11.9121 4.66699C11.9121 4.65085 11.9083 4.62879 11.8906 4.60352C11.4728 4.00746 10.7409 3.08479 9.77733 2.31934C8.81111 1.55179 7.68551 1 6.45604 1Z" />
    </svg>
  )
}

function IconTranscription({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 13.5 13" fill="currentColor">
      <path d="M10 0C11.933 0 13.5 1.567 13.5 3.5V9.5C13.5 11.433 11.933 13 10 13H7.04199V12H10C11.3807 12 12.5 10.8807 12.5 9.5V3.5C12.5 2.11929 11.3807 1 10 1H5C3.61929 1 2.5 2.11929 2.5 3.5V6.43848H1.5V3.5C1.5 1.567 3.067 1.28851e-07 5 0H10ZM4.09863 5.72461C4.86348 5.16443 5.99984 5.69804 6 6.7002V11.7344C6 12.8035 4.70713 13.339 3.95117 12.583L2.58594 11.2178H1.2002C0.53746 11.2178 1.06091e-05 10.6803 0 10.0176V8.41797C0 7.75523 0.537453 7.21777 1.2002 7.21777H2.58594L3.95117 5.85156L4.09863 5.72461ZM5 6.7002C4.99983 6.52215 4.78415 6.43266 4.6582 6.55859L3.05859 8.15918L3.02832 8.18359C3.00641 8.19827 2.98192 8.20874 2.95605 8.21387L2.91699 8.21777H1.2002L1.16016 8.22168C1.0688 8.24016 1 8.32117 1 8.41797V10.0176L1.00391 10.0576C1.01984 10.136 1.0817 10.198 1.16016 10.2139L1.2002 10.2178H2.91699C2.97004 10.2178 3.02109 10.2389 3.05859 10.2764L4.6582 11.876C4.7842 12.002 5 11.9126 5 11.7344V6.7002ZM6.50098 7.5C7.6051 7.50053 8.5 8.39576 8.5 9.5C8.5 10.6042 7.6051 11.4985 6.50098 11.499V10.499C7.05281 10.4985 7.5 10.052 7.5 9.5C7.5 8.94804 7.05281 8.50053 6.50098 8.5V7.5ZM11 9C11.2761 9 11.5 9.22386 11.5 9.5C11.5 9.77614 11.2761 10 11 10H9.5C9.22386 10 9 9.77614 9 9.5C9 9.22386 9.22386 9 9.5 9H11ZM11 6C11.2761 6 11.5 6.22386 11.5 6.5C11.5 6.77614 11.2761 7 11 7H8C7.72386 7 7.5 6.77614 7.5 6.5C7.5 6.22386 7.72386 6 8 6H11ZM11 3C11.2761 3 11.5 3.22386 11.5 3.5C11.5 3.77614 11.2761 4 11 4H5C4.72386 4 4.5 3.77614 4.5 3.5C4.5 3.22386 4.72386 3 5 3H11Z" />
    </svg>
  )
}

function IconSpeech({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 12 12.0376" fill="currentColor">
      <path fillRule="evenodd" clipRule="evenodd" d="M8.8 1H3.2C1.98497 1 1 1.98497 1 3.2V6.13333C1 7.34836 1.98497 8.33333 3.2 8.33333H3.92451C4.32261 8.33333 4.69151 8.54221 4.89633 8.88357L6 10.723L7.10367 8.88357C7.30849 8.5422 7.6774 8.33333 8.07549 8.33333H8.8C10.015 8.33333 11 7.34836 11 6.13333V3.2C11 1.98497 10.015 1 8.8 1ZM3.2 0C1.43269 0 0 1.43269 0 3.2V6.13333C0 7.90065 1.43269 9.33333 3.2 9.33333H3.92451C3.97134 9.33333 4.01474 9.35791 4.03884 9.39807L5.42834 11.7139C5.68727 12.1455 6.31273 12.1455 6.57166 11.7139L7.96116 9.39807C7.98526 9.35791 8.02866 9.33333 8.07549 9.33333H8.8C10.5673 9.33333 12 7.90064 12 6.13333V3.2C12 1.43269 10.5673 0 8.8 0H3.2Z" />
    </svg>
  )
}

function IconCheckbox({ checked, className = 'w-5 h-5' }: { checked: boolean; className?: string }) {
  if (checked) {
    return (
      <svg className={className} viewBox="0 0 12 12">
        {/* Black filled checkbox */}
        <path fill="#1D1C1B" fillRule="evenodd" clipRule="evenodd" d="M8.39941 0C10.3875 0 11.9998 1.61157 12 3.59961V8.39941C12 10.3876 10.3876 12 8.39941 12H3.59961C1.61157 11.9998 0 10.3875 0 8.39941V3.59961C0.000177149 1.61168 1.61168 0.000179596 3.59961 0H8.39941ZM3.59961 1C2.16396 1.00018 1.00018 2.16396 1 3.59961V8.39941C1 9.83521 2.16385 10.9998 3.59961 11H8.39941C9.83532 11 11 9.83532 11 8.39941V3.59961C10.9998 2.16385 9.83521 1 8.39941 1H3.59961Z" />
        {/* White tick */}
        <path fill="#fff" d="M9.09961 3.48535L6.32715 8.47656C5.89108 9.26149 4.76862 9.27973 4.30664 8.50977L2.90039 6.16699L3.75781 5.65234L5.16406 7.99609C5.23009 8.10563 5.38975 8.10282 5.45215 7.99121L8.22559 3L9.09961 3.48535Z" />
      </svg>
    )
  }
  return (
    <svg className={className} viewBox="0 0 12 12" fill="currentColor">
      <path fillRule="evenodd" clipRule="evenodd" d="M8.40039 0C10.3883 0.000211285 11.9998 1.61169 12 3.59961V8.40039C11.9998 10.3883 10.3883 11.9998 8.40039 12H3.59961C1.61169 11.9998 0.000211285 10.3883 0 8.40039V3.59961C0.000211156 1.61169 1.61169 0.000211157 3.59961 0H8.40039ZM3.59961 1C2.16398 1.00021 1.00021 2.16398 1 3.59961V8.40039C1.00021 9.83602 2.16398 10.9998 3.59961 11H8.40039C9.83602 10.9998 10.9998 9.83602 11 8.40039V3.59961C10.9998 2.16398 9.83602 1.00021 8.40039 1H3.59961Z" />
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

function IconInfo({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 12 12" fill="currentColor">
      <path d="M6.66699 9.33301H5.33301V5.33301H6.66699V9.33301Z" />
      <path d="M6.66699 4H5.33301V2.66699H6.66699V4Z" />
      <path fillRule="evenodd" clipRule="evenodd" d="M8.40039 0C10.3883 0.000211285 11.9998 1.61169 12 3.59961V8.40039C11.9998 10.3883 10.3883 11.9998 8.40039 12H3.59961C1.61169 11.9998 0.000211285 10.3883 0 8.40039V3.59961C0.000211156 1.61169 1.61169 0.000211157 3.59961 0H8.40039ZM3.59961 1C2.16398 1.00021 1.00021 2.16398 1 3.59961V8.40039C1.00021 9.83602 2.16398 10.9998 3.59961 11H8.40039C9.83602 10.9998 10.9998 9.83602 11 8.40039V3.59961C10.9998 2.16398 9.83602 1.00021 8.40039 1H3.59961Z" />
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

function IconVideo({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 9 11" fill="currentColor">
      <path fillRule="evenodd" clipRule="evenodd" d="M1.03927 1.03269V9.96731L7.91655 5.5L1.03927 1.03269ZM0 0.928981C0 0.182271 0.886347 -0.25826 1.5376 0.164775L8.57453 4.73579C9.14182 5.10429 9.14182 5.89571 8.57453 6.2642L1.5376 10.8352C0.88635 11.2583 0 10.8177 0 10.071V0.928981Z" />
    </svg>
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

/* ------------------------------------------------------------------ */
/*  Dummy data                                                         */
/* ------------------------------------------------------------------ */

type TableEntity = { id: string; name: string; imageUrl?: string; initials: string }

type ClipMatch = {
  start: number
  end: number
  score: number
  type: string
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
}

function relevanceLabel(clips: ClipMatch[] | undefined): { label: string; color: string } {
  if (!clips || clips.length === 0) return { label: '', color: '' }
  const best = Math.max(...clips.map((c) => c.score))
  if (best >= 0.10) return { label: 'Highest', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' }
  if (best >= 0.08) return { label: 'High', color: 'bg-green-100 text-green-800 border-green-300' }
  if (best >= 0.06) return { label: 'Medium', color: 'bg-yellow-100 text-yellow-800 border-yellow-300' }
  return { label: 'Low', color: 'bg-red-50 text-red-700 border-red-200' }
}

function clipScoreColor(score: number): string {
  if (score >= 0.10) return 'bg-emerald-50 text-emerald-700 border-emerald-200'
  if (score >= 0.08) return 'bg-green-50 text-green-700 border-green-200'
  if (score >= 0.06) return 'bg-yellow-50 text-yellow-700 border-yellow-200'
  return 'bg-red-50 text-red-600 border-red-200'
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

/** Format as mm:ss for thumbnail badge (hours omitted). */
function formatDurationHHMMSS(duration: string): string {
  const parts = duration.split(':').map(Number)
  if (parts.length >= 3) {
    const [h = 0, m = 0, s = 0] = parts
    const totalM = h * 60 + m
    return `${totalM.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }
  return duration
}

/** Short form for caption under title (e.g. 2:19). */
function formatDurationShort(duration: string): string {
  const parts = duration.split(':').map(Number)
  if (parts.length >= 3) {
    const [h, m, s] = parts
    if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
    return `${m}:${s.toString().padStart(2, '0')}`
  }
  return duration
}

/** Format seconds to badge (mm:ss) or (h:mm:ss). */
function formatSecondsToTimestamp(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '—'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  return `${m}:${s.toString().padStart(2, '0')}`
}

/* ------------------------------------------------------------------ */
/*  Advanced Parameters Dropdown                                       */
/* ------------------------------------------------------------------ */

interface SearchOptions {
  visual: boolean
  audio: boolean
  transcription: boolean
  lexical: boolean
  semantic: boolean
}

function AdvancedParamsDropdown({
  options,
  onChange,
  onApply,
}: {
  options: SearchOptions
  onChange: (opts: SearchOptions) => void
  onApply: () => void
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  function toggle(key: keyof SearchOptions) {
    const next = { ...options, [key]: !options[key] }
    if (key === 'transcription') {
      next.lexical = next.transcription
      next.semantic = next.transcription
    }
    if ((key === 'lexical' || key === 'semantic') && !next.lexical && !next.semantic) {
      next.transcription = false
    } else if ((key === 'lexical' || key === 'semantic') && (next.lexical || next.semantic)) {
      next.transcription = true
    }
    onChange(next)
  }

  const hasActiveOptions = options.visual || options.audio || options.transcription

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-sm font-medium transition-colors ${
          open || hasActiveOptions
            ? 'border-border bg-card text-text-primary'
            : 'border-border bg-surface text-text-secondary hover:bg-card'
        }`}
        aria-label="Advanced parameters"
        aria-expanded={open}
      >
        <IconFilter className="w-3.5 h-3.5 shrink-0" />
        <span className="hidden sm:inline">Filter</span>
        <IconChevronDown className={`w-3.5 h-3.5 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-72 rounded-xl border border-border bg-surface p-5 shadow-xl z-[100]">
          <p className="text-xs text-text-tertiary uppercase tracking-wider mb-1.5">Advanced Parameters</p>
          <div className="flex items-center gap-1.5 mb-4">
            <h4 className="text-sm font-medium text-text-primary">Select search options</h4>
            <IconInfo className="w-3.5 h-3.5 text-text-tertiary" />
          </div>

          <div className="space-y-3">
            {/* Visual */}
            <button type="button" onClick={() => toggle('visual')} className="flex items-center gap-2.5 w-full text-left group">
              <IconCheckbox checked={options.visual} className="w-5 h-5 text-brand-charcoal" />
              <span className="text-sm font-medium text-text-secondary group-hover:text-text-primary">Visual</span>
            </button>

            {/* Audio */}
            <button type="button" onClick={() => toggle('audio')} className="flex items-center gap-2.5 w-full text-left group">
              <IconCheckbox checked={options.audio} className="w-5 h-5 text-brand-charcoal" />
              <span className="text-sm font-medium text-text-secondary group-hover:text-text-primary">Audio</span>
            </button>

            {/* Transcription */}
            <div>
              <button type="button" onClick={() => toggle('transcription')} className="flex items-center gap-2.5 w-full text-left group">
                <IconCheckbox checked={options.transcription} className="w-5 h-5 text-brand-charcoal" />
                <span className="text-sm font-medium text-text-secondary group-hover:text-text-primary">Transcription</span>
              </button>
              {/* Sub-options */}
              <div className="ml-8 mt-2 space-y-2">
                <button type="button" onClick={() => toggle('lexical')} className="flex items-center gap-2.5 w-full text-left group">
                  <IconCheckbox checked={options.lexical} className="w-5 h-5 text-brand-charcoal" />
                  <span className="text-sm text-gray-600 group-hover:text-gray-800">Lexical</span>
                </button>
                <button type="button" onClick={() => toggle('semantic')} className="flex items-center gap-2.5 w-full text-left group">
                  <IconCheckbox checked={options.semantic} className="w-5 h-5 text-brand-charcoal" />
                  <span className="text-sm text-gray-600 group-hover:text-gray-800">Semantic</span>
                </button>
              </div>
            </div>
          </div>

          {/* Apply button */}
          <button
            type="button"
            onClick={() => {
              onApply()
              setOpen(false)
            }}
            className="w-full mt-5 h-10 rounded-full bg-brand-charcoal text-brand-white text-sm font-medium hover:bg-gray-700 transition-colors"
          >
            Apply
          </button>
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Dashboard                                                          */
/* ------------------------------------------------------------------ */

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

  const [searchOptions, setSearchOptions] = useState<SearchOptions>({
    visual: true,
    audio: true,
    transcription: true,
    lexical: true,
    semantic: true,
  })
  const [searchResults, setSearchResults] = useState<{ query: string; results: VideoItem[] } | null>(null)
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
        const list: EntityOption[] = (data.entities || []).map((e: any) => {
          const meta = e.metadata || {}
          const name = meta.name || e.name || e.id
          const b64 = meta.face_snap_base64
          const previewUrl = b64 ? `data:image/png;base64,${b64}` : ''
          return { id: e.id, name, previewUrl }
        })
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

  const filteredVideos = useMemo(() => {
    if (searchResults) return searchResults.results
    let list = allVideos
    if (activeCategory !== 'All') {
      list = list.filter((v) => v.category === activeCategory)
    }
    if (sortBy === 'name') {
      list = [...list].sort((a, b) => a.title.localeCompare(b.title))
    }
    return list
  }, [allVideos, sortBy, activeCategory, searchResults])

  async function handleSearch() {
    const query = searchQuery.trim()
    const entityIds = searchAttachments.filter((a) => a.type === 'entity').map((a) => a.id)
    const hasQuery = query.length > 0
    const hasEntities = entityIds.length > 0
    if (!hasQuery && !hasEntities) return
    setSearchError(null)
    setSearchLoading(true)
    try {
      type RawClip = { start: number; end: number; score: number; rank?: number; thumbnail_url?: string }
      type RawResult = { video_id: string; clips: RawClip[] }
      let allResults: RawResult[] = []

      if (hasQuery) {
        const res = await fetch(`${API_BASE}/api/search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query }),
        })
        const data = await res.json().catch(() => ({}))
        if (!res.ok) { setSearchError(data.error || 'Search failed'); setSearchResults(null); return }
        allResults = data.results || []
      }

      if (hasEntities) {
        for (const entityId of entityIds) {
          const res = await fetch(`${API_BASE}/api/entities/${entityId}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: hasQuery ? query : '' }),
          })
          const data = await res.json().catch(() => ({}))
          if (res.ok && data.results) {
            for (const r of data.results as RawResult[]) {
              const existing = allResults.find((e) => e.video_id === r.video_id)
              if (existing) {
                existing.clips = [...existing.clips, ...r.clips]
              } else {
                allResults.push(r)
              }
            }
          }
        }
      }

      const videoLookup = new Map(cachedVideos.map((v) => [v.id, v]))

      const results: VideoItem[] = allResults.map((r) => {
        const cached = videoLookup.get(r.video_id)
        const meta = cached?.metadata || {}
        let uploadDate = ''
        try {
          const u = meta.created_at || ''
          if (u) {
            const d = new Date(u)
            uploadDate = `${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}-${d.getFullYear()}`
          }
        } catch { /* ignore */ }
        const bestScore = r.clips.length > 0 ? Math.max(...r.clips.map((c) => c.score)) : 0
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
          clips: r.clips.map((c) => ({ start: c.start, end: c.end, score: c.score, type: 'visual' })),
          searchScore: bestScore,
        }
      })

      results.sort((a, b) => (b.searchScore ?? 0) - (a.searchScore ?? 0))

      const displayQuery = hasQuery ? query : `Entity: ${searchAttachments.filter((a) => a.type === 'entity').map((a) => a.name).join(', ')}`
      setSearchResults({ query: displayQuery, results })
      try {
        sessionStorage.setItem('video_redaction_last_search', JSON.stringify({ query: displayQuery, results }))
      } catch { /* ignore */ }
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
        const parsed = JSON.parse(raw) as { query: string; results: VideoItem[] }
        if (parsed?.query != null && Array.isArray(parsed.results)) {
          setSearchQuery(parsed.query)
          setSearchResults({ query: parsed.query, results: parsed.results })
        }
      }
    } catch {
      // ignore
    }
  }, [])

  function clearSearch() {
    setSearchResults(null)
    setSearchError(null)
    try {
      sessionStorage.removeItem('video_redaction_last_search')
    } catch {
      // ignore
    }
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
      {/* Search bar — vibrant gradient border, curved corners, rotating effect */}
      <div className="search-bar-gradient-outer mb-4 shadow-sm w-full min-w-0">
        <div className="search-bar-gradient-border-wrap">
          <div className="search-bar-gradient-border" aria-hidden />
        </div>
        <div className="search-bar-gradient-inner w-full">
          <div className="px-3 sm:px-4 pt-4 pb-2 min-w-0 w-full">
            {/* Attachment chips + input on one line */}
            <div className="flex flex-wrap items-center gap-2 min-w-0">
              {searchAttachments.map((att) => (
                <span
                  key={att.id}
                  className="inline-flex items-center gap-1.5 pl-1 pr-2 py-0.5 rounded-full border border-border bg-card text-sm text-text-secondary"
                >
                  <img
                    src={att.previewUrl}
                    alt={att.name}
                    className={`w-6 h-6 object-cover ${att.type === 'entity' ? 'rounded-full' : 'rounded'}`}
                  />
                  <span className="max-w-[120px] truncate text-xs font-medium">{att.name}</span>
                  <button
                    type="button"
                    onClick={() =>
                      setSearchAttachments((prev) => prev.filter((a) => a.id !== att.id))
                    }
                    className="ml-0.5 p-0.5 rounded-full hover:bg-gray-200 text-gray-400 hover:text-gray-600 transition-colors"
                    aria-label={`Remove ${att.name}`}
                  >
                    <svg className="w-3 h-3" viewBox="0 0 12 12" fill="currentColor">
                      <path d="M6.02 5.31L8.97 2.37l.71.7L6.73 6.02l2.93 2.93-.71.71L6.02 6.73 3.07 9.67l-.7-.7L5.31 6.02 2.35 3.05l.7-.7L6.02 5.31Z" />
                    </svg>
                  </button>
                </span>
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
              {/* AdvancedParamsDropdown (Filter) commented out per request
              <AdvancedParamsDropdown
                options={searchOptions}
                onChange={setSearchOptions}
                onApply={() => {}}
              />
              */}
            </div>
            <button
              type="button"
              disabled={(!searchQuery.trim() && !searchAttachments.some((a) => a.type === 'entity')) || searchLoading}
              onClick={() => handleSearch()}
              className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors shrink-0 ${
                (searchQuery.trim() || searchAttachments.some((a) => a.type === 'entity')) && !searchLoading
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

      {/* Search result summary + error */}
      {searchError && (
        <p className="mb-3 text-sm text-red-600" role="alert">
          {searchError}
        </p>
      )}
      {searchResults && (
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <span className="text-sm text-text-secondary">
            {searchResults.results.length === 0
              ? `No videos found for “${searchResults.query}”.`
              : `${searchResults.results.length} result${searchResults.results.length !== 1 ? 's' : ''} for “${searchResults.query}”`}
          </span>
          <button
            type="button"
            onClick={clearSearch}
            className="text-sm font-medium text-brand-charcoal hover:underline focus:outline-none focus:ring-2 focus:ring-brand-charcoal/30 rounded"
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
                  to={`/video/${v.id}`}
                  className="group block focus:outline-none focus:ring-2 focus:ring-accent/30 rounded-xl min-w-0"
                >
                  <div className="relative aspect-video rounded-xl overflow-hidden bg-brand-charcoal">
                    {v.thumbnailUrl ? (
                      <img
                        src={v.thumbnailUrl}
                        alt=""
                        className="absolute inset-0 w-full h-full object-cover z-0"
                      />
                    ) : v.streamUrl ? (
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
                    ) : null}
                    <div className="absolute inset-0 flex items-center justify-center bg-transparent group-hover:bg-brand-charcoal/40 z-[2] pointer-events-none transition-colors duration-200">
                      <div className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                        <IconPlay className="w-5 h-5 text-white ml-0.5" />
                      </div>
                    </div>
                    {(() => {
                      const rel = relevanceLabel(v.clips)
                      return rel.label ? (
                        <span className={`absolute top-2 left-2 z-[3] inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold border ${rel.color}`}>
                          {rel.label}
                        </span>
                      ) : null
                    })()}
                    <span className="absolute left-1/2 -translate-x-1/2 bottom-1.5 px-3 py-1 text-sm font-mono font-medium text-white rounded border border-white/90 tabular-nums bg-black/50 [text-shadow:0_0_2px_rgba(0,0,0,0.9)]">
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
        onImageAdded={(file) => {
          const url = URL.createObjectURL(file)
          setSearchAttachments((prev) => [
            ...prev,
            { id: `img-${Date.now()}`, type: 'image', name: file.name, previewUrl: url },
          ])
          setAddImageModalOpen(false)
        }}
      />
      <AddEntityModal
        open={addEntityModalOpen}
        onClose={() => setAddEntityModalOpen(false)}
        onEntityAdded={(selection) => {
          setSearchAttachments((prev) => [
            ...prev,
            { id: selection.id ?? `ent-${Date.now()}`, type: 'entity', name: selection.name?.trim() || selection.file?.name || 'Entity', previewUrl: selection.previewUrl },
          ])
          setAddEntityModalOpen(false)
        }}
      />
    </div>
  )
}
