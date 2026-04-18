import { useState, useEffect, useRef } from 'react'
import { BrowserRouter, Routes, Route, NavLink, Link, useLocation, useNavigate } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import OverviewPage from './pages/OverviewPage'
import EditorPage from './pages/EditorPage'
import VideoEditorPage from './pages/VideoEditorPage'
import UploadVideosModal from './components/UploadVideosModal'
import { VideoCacheProvider, useVideoCache } from './contexts/VideoCache'
import { isEditorExperiencePath } from './lib/editorRouting'
import logoFullUrl from '../strand/assets/logo-full.svg?url'
import logoMarkUrl from '../strand/assets/logo-mark.svg?url'
import devicesIconUrl from '../strand/icons/devices.svg?url'

const UPLOAD_NOTIFICATION_MS = 6000
const MOBILE_EDITOR_BREAKPOINT = '(max-width: 767px)'

const navItems = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/overview', label: 'Overview' },
  { to: '/editor', label: 'Editor' },
]

function NavLinks({
  items = navItems,
  mobile = false,
  onNavigate,
}: {
  items?: Array<{ to: string; label: string }>
  mobile?: boolean
  onNavigate?: () => void
}) {
  const location = useLocation()
  const base = 'font-brand-xbold px-3 py-2 rounded-lg text-sm font-medium transition-colors border'
  const active = 'text-text-primary bg-card border-border'
  const inactive = 'border-transparent text-text-secondary hover:bg-card hover:text-text-primary'
  const isItemActive = (to: string, routeActive: boolean) => {
    if (to === '/editor') return isEditorExperiencePath(location.pathname)
    if (to === '/overview') return location.pathname === '/' || routeActive
    return routeActive
  }
  const linkClass = ({ isActive, isPending }: { isActive: boolean; isPending: boolean }) => {
    const resolvedActive = isPending ? false : isActive
    return `${base} ${resolvedActive ? active : inactive} ${mobile ? 'block w-full text-left' : ''}`
  }

  return (
    <>
      {items.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.to !== '/editor'}
          className={({ isActive, isPending }) => linkClass({
            isActive: isItemActive(item.to, isActive),
            isPending,
          })}
          onClick={onNavigate}
          aria-current={isItemActive(item.to, location.pathname === item.to) ? 'page' : undefined}
        >
          {item.label}
        </NavLink>
      ))}
    </>
  )
}

function DesktopOnlyAccessNotice({ section }: { section: 'dashboard' | 'editor' | 'video' }) {
  const title = section === 'video'
    ? 'Open on desktop to edit this video'
    : section === 'dashboard'
      ? 'Open on desktop to use the dashboard'
      : 'Open on desktop to use the editor'
  const description = section === 'dashboard'
    ? 'Search, indexing, and redaction tools are optimized for larger screens.'
    : 'The redaction editor is optimized for larger screens.'
  const eyebrow = section === 'dashboard' ? 'Dashboard' : 'Editor'

  return (
    <div className="min-h-full bg-[linear-gradient(180deg,rgba(255,255,255,0.78),rgba(244,243,243,0.96))]">
      <div className="w-full max-w-2xl mx-auto px-5 py-12 sm:px-6 sm:py-16">
        <div className="inline-flex items-center gap-2 rounded-full border border-border bg-surface/80 px-3 py-1.5 text-[11px] font-brand-xbold uppercase tracking-[0.14em] text-text-tertiary">
          <img src={devicesIconUrl} alt="" className="h-3.5 w-3.5 opacity-70" />
          Desktop only
        </div>

        <div className="mt-10">
          <p className="text-xs font-brand-xbold uppercase tracking-[0.14em] text-text-tertiary">
            {eyebrow}
          </p>
          <h2 className="mt-3 max-w-xl text-[30px] leading-[1.05] font-brand-bold text-text-primary">
            {title}
          </h2>
          <p className="mt-4 max-w-lg text-sm leading-7 text-text-secondary">
            {description}
          </p>
        </div>

        <div className="mt-8 flex items-start gap-3 rounded-xl border border-border bg-surface/75 px-4 py-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-card border border-border">
            <img src={devicesIconUrl} alt="" className="h-4.5 w-4.5 opacity-80" />
          </div>
          <div className="min-w-0">
            <p className="text-sm font-medium text-text-primary">
              Use desktop mode to continue.
            </p>
            <p className="mt-1 text-sm text-text-secondary">
              Overview stays available on mobile.
            </p>
          </div>
        </div>

        <div className="mt-8 flex items-center gap-3">
          <Link
            to="/overview"
            className="inline-flex items-center justify-center rounded-lg bg-brand-charcoal px-4 py-2 text-sm font-medium text-brand-white hover:bg-gray-700 transition-colors"
          >
            Back to overview
          </Link>
        </div>
      </div>
    </div>
  )
}

function Shell() {
  const navigate = useNavigate()
  const { refresh: refreshVideoCache } = useVideoCache()
  const [uploadModalOpen, setUploadModalOpen] = useState(false)
  const [uploadNotification, setUploadNotification] = useState<string | null>(null)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [isMobileEditorViewport, setIsMobileEditorViewport] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const location = useLocation()

  useEffect(() => {
    if (!uploadNotification) return
    const t = setTimeout(() => setUploadNotification(null), UPLOAD_NOTIFICATION_MS)
    return () => clearTimeout(t)
  }, [uploadNotification])

  useEffect(() => {
    const media = window.matchMedia(MOBILE_EDITOR_BREAKPOINT)
    const updateViewport = () => setIsMobileEditorViewport(media.matches)
    updateViewport()
    if (typeof media.addEventListener === 'function') {
      media.addEventListener('change', updateViewport)
      return () => media.removeEventListener('change', updateViewport)
    }
    media.addListener(updateViewport)
    return () => media.removeListener(updateViewport)
  }, [])

  useEffect(() => {
    setMobileMenuOpen(false)
  }, [location.pathname])

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMobileMenuOpen(false)
      }
    }
    if (mobileMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside)
      document.body.style.overflow = 'hidden'
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.body.style.overflow = ''
    }
  }, [mobileMenuOpen])

  const isOverviewRoute = location.pathname === '/' || location.pathname === '/overview'
  const mobileRestrictedSection =
    isMobileEditorViewport && !isOverviewRoute
      ? (location.pathname.startsWith('/video/') || location.pathname.startsWith('/editor/')
          ? 'video'
          : location.pathname === '/editor'
            ? 'editor'
            : 'dashboard')
      : null
  const homeLinkTarget = isMobileEditorViewport ? '/overview' : '/dashboard'

  return (
    <div className="min-h-screen h-screen max-h-screen bg-background text-text-primary flex flex-col overflow-hidden">
      <div className="relative" ref={menuRef}>
        <header className="bg-background px-4 py-3 flex items-center justify-between shrink-0 border-b border-border">
          <div className="flex items-center gap-3 min-w-0 flex-1 md:flex-initial">
            <div className="flex items-center gap-2 mr-2 md:mr-6 min-w-0">
              <Link
                to={homeLinkTarget}
                className="font-brand text-text-primary hover:opacity-80 transition-opacity cursor-pointer shrink-0 text-left bg-transparent border-0 p-0 no-underline block"
                aria-label={isMobileEditorViewport ? 'Go to overview' : 'Go to dashboard'}
              >
                <h1 className="text-base md:text-h5 font-medium truncate">GDPR Compliance [Video REDACTION]</h1>
              </Link>
              <span className="hidden md:inline-flex items-center px-2 py-1 rounded-sm border border-accent/25 bg-accent/10 text-accent text-xs font-medium shrink-0 uppercase tracking-wide pointer-events-none select-none">
                DEMO
              </span>
            </div>

            <nav className="hidden md:flex items-center gap-0.5">
              <NavLinks />
            </nav>
          </div>

          <button
            type="button"
            onClick={() => setMobileMenuOpen((o) => !o)}
            className={`md:hidden p-2 rounded-lg text-text-primary hover:bg-card border transition-colors ${
              mobileMenuOpen ? 'border-border bg-card' : 'border-transparent'
            }`}
            aria-expanded={mobileMenuOpen}
            aria-label={mobileMenuOpen ? 'Close menu' : 'Open menu'}
          >
            {mobileMenuOpen ? (
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            ) : (
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            )}
          </button>

          <div className={`flex items-center gap-2 shrink-0 ${mobileMenuOpen ? 'max-md:hidden' : ''}`}>
            <img src={logoMarkUrl} alt="" className="h-7 w-auto md:hidden" />
            <img src={logoFullUrl} alt="" className="h-7 w-auto hidden md:block" />
          </div>
        </header>

        <div
          className={`md:hidden absolute left-0 right-0 top-full z-50 bg-background border-b border-border shadow-lg transition-all duration-200 ease-out ${
            mobileMenuOpen ? 'opacity-100 visible' : 'opacity-0 invisible pointer-events-none'
          }`}
        >
          <nav className="flex flex-col p-3 gap-1">
            <NavLinks items={navItems} mobile onNavigate={() => setMobileMenuOpen(false)} />
          </nav>
        </div>
      </div>

      {uploadNotification && (
        <div
          role="status"
          aria-live="polite"
          className="fixed top-4 left-1/2 -translate-x-1/2 z-[300] max-w-md px-4 py-3 rounded-lg bg-brand-charcoal text-brand-white shadow-lg border border-border"
        >
          <p className="text-sm font-medium">{uploadNotification}</p>
        </div>
      )}

      <div className="flex flex-1 min-h-0 overflow-hidden">
        <main className="flex-1 min-h-0 overflow-auto min-w-0" key={location.pathname}>
          {mobileRestrictedSection ? (
            <DesktopOnlyAccessNotice section={mobileRestrictedSection} />
          ) : (
            <Routes location={location}>
              <Route path="/" element={<OverviewPage />} />
              <Route
                path="/dashboard"
                element={
                  <div className="w-full min-w-0 px-3 sm:px-4 py-4 sm:py-6">
                    <Dashboard onOpenUpload={() => setUploadModalOpen(true)} />
                  </div>
                }
              />
              <Route path="/overview" element={<OverviewPage />} />
              <Route path="/editor" element={<EditorPage />} />
              <Route path="/editor/:videoId" element={<VideoEditorPage />} />
              <Route path="/video/:videoId" element={<VideoEditorPage />} />
            </Routes>
          )}
        </main>
      </div>
      <UploadVideosModal
        open={uploadModalOpen}
        onClose={() => setUploadModalOpen(false)}
        onUploadSubmitted={() => {
          setUploadNotification(
            'Video is sent for indexing and will be updated on the dashboard as soon as indexing is done.'
          )
          setUploadModalOpen(false)
          navigate('/dashboard')
        }}
        onUploadSuccess={refreshVideoCache}
      />
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <VideoCacheProvider>
        <Shell />
      </VideoCacheProvider>
    </BrowserRouter>
  )
}

export default App
