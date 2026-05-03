import { useState, useRef, useCallback, useEffect, useMemo } from 'react'
import { API_BASE } from '../lib/api'

function IconClose({ className = 'w-5 h-5' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 12 12" fill="currentColor">
      <path d="M6.02051 5.31348L8.9668 2.36719L9.67383 3.07422L6.72754 6.02051L9.65332 8.94629L8.94629 9.65332L6.02051 6.72754L3.07422 9.67383L2.36719 8.9668L5.31348 6.02051L2.34668 3.05371L3.05371 2.34668L6.02051 5.31348Z" />
      <path fillRule="evenodd" clipRule="evenodd" d="M8.40039 0C10.3883 0.000211285 11.9998 1.61169 12 3.59961V8.40039C11.9998 10.3883 10.3883 11.9998 8.40039 12H3.59961C1.61169 11.9998 0.000211285 10.3883 0 8.40039V3.59961C0.000211156 1.61169 1.61169 0.000211157 3.59961 0H8.40039ZM3.59961 1C2.16398 1.00021 1.00021 2.16398 1 3.59961V8.40039C1.00021 9.83602 2.16398 10.9998 3.59961 11H8.40039C9.83602 10.9998 10.9998 9.83602 11 8.40039V3.59961C10.9998 2.16398 9.83602 1.00021 8.40039 1H3.59961Z" />
    </svg>
  )
}

function IconCamera({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
      <circle cx="12" cy="13" r="4" />
    </svg>
  )
}

interface DetectedFace {
  index: number
  confidence: number
  bbox: { x: number; y: number; w: number; h: number }
  image_base64: string
}

interface CropRect {
  x: number
  y: number
  width: number
  height: number
}

export interface SnapFaceResult {
  entityId: string
  name: string
  faceBase64: string
  bbox?: { x: number; y: number; w: number; h: number }
  capturedAtSec: number
}

interface SnapFaceFromVideoModalProps {
  open: boolean
  onClose: () => void
  frameDataUrl: string | null
  capturedAtSec: number
  defaultName?: string
  onFaceAdded?: (result: SnapFaceResult) => void
}

const MIN_CROP_SIZE = 60
const MAX_CROP_SIZE = 280
const CROP_CONTAINER_W = 320
const CROP_CONTAINER_H = 240

function dataUrlToFile(dataUrl: string, filename: string): Promise<File> {
  return fetch(dataUrl)
    .then((res) => res.blob())
    .then((blob) => new File([blob], filename, { type: blob.type || 'image/png' }))
}

function fmtTimeShort(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00'
  const totalSeconds = Math.floor(seconds)
  const m = Math.floor(totalSeconds / 60)
  const s = totalSeconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Could not load image.'))
    img.src = src
  })
}

export default function SnapFaceFromVideoModal({
  open,
  onClose,
  frameDataUrl,
  capturedAtSec,
  defaultName,
  onFaceAdded,
}: SnapFaceFromVideoModalProps) {
  const [mode, setMode] = useState<'detect' | 'manual'>('detect')
  const [entityName, setEntityName] = useState('')
  const [detectedFaces, setDetectedFaces] = useState<DetectedFace[]>([])
  const [isDetecting, setIsDetecting] = useState(false)
  const [detectError, setDetectError] = useState<string | null>(null)
  const [selectedFaceIdx, setSelectedFaceIdx] = useState<number | null>(null)
  const [isRegistering, setIsRegistering] = useState(false)
  const [registrationError, setRegistrationError] = useState<string | null>(null)

  const [crop, setCrop] = useState<CropRect>({ x: 100, y: 60, width: 120, height: 120 })
  const [isDragging, setIsDragging] = useState(false)
  const [isResizing, setIsResizing] = useState(false)
  const dragStart = useRef({ x: 0, y: 0, cropX: 0, cropY: 0 })
  const resizeStart = useRef({ x: 0, y: 0, w: 0, h: 0 })
  const cropContainerRef = useRef<HTMLDivElement>(null)

  const detectionRunIdRef = useRef(0)

  const suggestedName = useMemo(() => {
    if (defaultName && defaultName.trim()) return defaultName.trim()
    return `Person at ${fmtTimeShort(capturedAtSec)}`
  }, [defaultName, capturedAtSec])

  useEffect(() => {
    if (!open) {
      setMode('detect')
      setEntityName('')
      setDetectedFaces([])
      setIsDetecting(false)
      setDetectError(null)
      setSelectedFaceIdx(null)
      setIsRegistering(false)
      setRegistrationError(null)
      setCrop({ x: 100, y: 60, width: 120, height: 120 })
      return
    }
    setEntityName(suggestedName)
  }, [open, suggestedName])

  useEffect(() => {
    if (!open || !frameDataUrl) return

    setDetectedFaces([])
    setDetectError(null)
    setSelectedFaceIdx(null)
    setIsDetecting(true)

    const runId = detectionRunIdRef.current + 1
    detectionRunIdRef.current = runId

    let cancelled = false

    ;(async () => {
      try {
        const ext = frameDataUrl.startsWith('data:image/jpeg') ? 'jpg' : 'png'
        const file = await dataUrlToFile(frameDataUrl, `snap-${Date.now()}.${ext}`)
        const formData = new FormData()
        formData.append('image', file)
        const res = await fetch(`${API_BASE}/api/detect-faces`, {
          method: 'POST',
          body: formData,
        })
        if (cancelled || detectionRunIdRef.current !== runId) return
        if (!res.ok) {
          if (res.status === 413) {
            throw new Error(
              'Captured frame is too large for the server (HTTP 413). You can still mark the face area manually below.',
            )
          }
          throw new Error(`HTTP ${res.status}`)
        }
        const data = await res.json()
        if (cancelled || detectionRunIdRef.current !== runId) return
        const faces: DetectedFace[] = Array.isArray(data.faces) ? data.faces : []
        if (faces.length === 0) {
          setDetectError('No face detected in this frame.')
          setDetectedFaces([])
          setSelectedFaceIdx(null)
        } else {
          const sorted = [...faces].sort((a, b) => {
            const areaA = (a.bbox?.w || 0) * (a.bbox?.h || 0)
            const areaB = (b.bbox?.w || 0) * (b.bbox?.h || 0)
            if (areaA !== areaB) return areaB - areaA
            return (b.confidence || 0) - (a.confidence || 0)
          })
          setDetectedFaces(sorted)
          setSelectedFaceIdx(0)
          setDetectError(null)
        }
      } catch (err) {
        if (cancelled || detectionRunIdRef.current !== runId) return
        setDetectError(
          err instanceof Error
            ? err.message
            : 'Could not run face detection on the captured frame.',
        )
        setDetectedFaces([])
        setSelectedFaceIdx(null)
      } finally {
        if (!cancelled && detectionRunIdRef.current === runId) {
          setIsDetecting(false)
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [open, frameDataUrl])

  /* ── Manual crop drag/resize handlers ───────────────────────────── */

  const clampCrop = useCallback((c: CropRect, maxW: number, maxH: number): CropRect => {
    const w = Math.max(MIN_CROP_SIZE, Math.min(MAX_CROP_SIZE, c.width))
    const h = Math.max(MIN_CROP_SIZE, Math.min(MAX_CROP_SIZE, c.height))
    const x = Math.max(0, Math.min(maxW - w, c.x))
    const y = Math.max(0, Math.min(maxH - h, c.y))
    return { x, y, width: w, height: h }
  }, [])

  const onMouseDownCrop = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      setIsDragging(true)
      dragStart.current = { x: e.clientX, y: e.clientY, cropX: crop.x, cropY: crop.y }
    },
    [crop.x, crop.y],
  )

  const onMouseDownResize = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      e.stopPropagation()
      setIsResizing(true)
      resizeStart.current = { x: e.clientX, y: e.clientY, w: crop.width, h: crop.height }
    },
    [crop.width, crop.height],
  )

  useEffect(() => {
    if (mode !== 'manual' || !cropContainerRef.current || (!isDragging && !isResizing)) return
    const rect = cropContainerRef.current.getBoundingClientRect()
    const maxW = rect.width
    const maxH = rect.height

    const onMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        const dx = e.clientX - dragStart.current.x
        const dy = e.clientY - dragStart.current.y
        setCrop((prev) =>
          clampCrop(
            {
              ...prev,
              x: dragStart.current.cropX + dx,
              y: dragStart.current.cropY + dy,
            },
            maxW,
            maxH,
          ),
        )
      }
      if (isResizing) {
        const dx = e.clientX - resizeStart.current.x
        const dy = e.clientY - resizeStart.current.y
        const size = Math.max(
          MIN_CROP_SIZE,
          Math.min(MAX_CROP_SIZE, resizeStart.current.w + dx, resizeStart.current.h + dy),
        )
        setCrop((prev) => clampCrop({ ...prev, width: size, height: size }, maxW, maxH))
      }
    }
    const onMouseUp = () => {
      setIsDragging(false)
      setIsResizing(false)
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [mode, isDragging, isResizing, clampCrop])

  /* ── Crop the captured frame to a base64 PNG ───────────────────── */

  const cropFrameToBase64 = useCallback(async (): Promise<{
    base64: string
    mimeType: string
    bbox: { x: number; y: number; w: number; h: number }
  }> => {
    if (!frameDataUrl) throw new Error('No frame captured.')

    const image = await loadImage(frameDataUrl)
    const containerW = CROP_CONTAINER_W
    const containerH = CROP_CONTAINER_H
    // The frame is rendered with object-cover at this fixed container size, so
    // we mirror that math here to convert the on-screen crop coords into the
    // natural image coordinate space.
    const scale = Math.max(containerW / image.naturalWidth, containerH / image.naturalHeight)
    const renderedW = image.naturalWidth * scale
    const renderedH = image.naturalHeight * scale
    const offsetX = (containerW - renderedW) / 2
    const offsetY = (containerH - renderedH) / 2

    const sourceX = Math.max(0, Math.min(image.naturalWidth, (crop.x - offsetX) / scale))
    const sourceY = Math.max(0, Math.min(image.naturalHeight, (crop.y - offsetY) / scale))
    const sourceW = Math.max(1, Math.min(image.naturalWidth - sourceX, crop.width / scale))
    const sourceH = Math.max(1, Math.min(image.naturalHeight - sourceY, crop.height / scale))

    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, Math.round(sourceW))
    canvas.height = Math.max(1, Math.round(sourceH))
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Could not prepare crop canvas.')
    ctx.drawImage(image, sourceX, sourceY, sourceW, sourceH, 0, 0, canvas.width, canvas.height)
    // JPEG keeps the upload small enough to clear typical reverse-proxy
    // request limits (which is what HTTP 413 from "Add to anonymize"
    // usually indicates).
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9)
    const base64 = dataUrl.includes(',') ? dataUrl.split(',', 2)[1] : dataUrl
    return {
      base64,
      mimeType: 'image/jpeg',
      bbox: {
        x: Math.round(sourceX),
        y: Math.round(sourceY),
        w: Math.round(sourceW),
        h: Math.round(sourceH),
      },
    }
  }, [crop, frameDataUrl])

  /* ── Registration: shared between detect and manual modes ──────── */

  const registerFromBase64 = useCallback(
    async (
      faceBase64: string,
      mimeType: string,
      bbox?: { x: number; y: number; w: number; h: number },
    ) => {
      const name = entityName.trim()
      if (!name) throw new Error('Name is required.')

      const facePreviewUrl = `data:${mimeType};base64,${faceBase64}`
      const safeName = name.replace(/\s+/g, '-').toLowerCase() || 'entity'
      const ext = mimeType === 'image/jpeg' ? 'jpg' : 'png'
      const faceFile = await dataUrlToFile(facePreviewUrl, `${safeName}.${ext}`)

      const formData = new FormData()
      formData.append('image', faceFile)
      formData.append('name', name)
      formData.append('preview_base64', faceBase64)

      const res = await fetch(`${API_BASE}/api/entities/upload-face`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        if (res.status === 413) {
          throw new Error(
            'Selected face image is too large for the server (HTTP 413). Try a smaller / tighter crop.',
          )
        }
        throw new Error(data.error || `HTTP ${res.status}`)
      }
      const entity = data.entity || {}
      const entityId = entity.id || entity.entity_id
      if (!entityId) {
        throw new Error('Entity was created without an ID.')
      }

      onFaceAdded?.({
        entityId,
        name: entity.name || name,
        faceBase64,
        bbox,
        capturedAtSec,
      })
      onClose()
    },
    [capturedAtSec, entityName, onClose, onFaceAdded],
  )

  const reencodeAsJpeg = useCallback(async (sourceBase64: string, sourceMime: string) => {
    try {
      const image = await loadImage(`data:${sourceMime};base64,${sourceBase64}`)
      const MAX_FACE_DIM = 512
      const longSide = Math.max(image.naturalWidth, image.naturalHeight)
      const scale = longSide > MAX_FACE_DIM ? MAX_FACE_DIM / longSide : 1
      const w = Math.max(1, Math.round(image.naturalWidth * scale))
      const h = Math.max(1, Math.round(image.naturalHeight * scale))
      const canvas = document.createElement('canvas')
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext('2d')
      if (!ctx) return { base64: sourceBase64, mimeType: sourceMime }
      ctx.drawImage(image, 0, 0, w, h)
      const dataUrl = canvas.toDataURL('image/jpeg', 0.9)
      const base64 = dataUrl.includes(',') ? dataUrl.split(',', 2)[1] : dataUrl
      return { base64, mimeType: 'image/jpeg' }
    } catch {
      return { base64: sourceBase64, mimeType: sourceMime }
    }
  }, [])

  const handleConfirmDetect = useCallback(async () => {
    if (selectedFaceIdx === null) return
    const face = detectedFaces[selectedFaceIdx]
    if (!face) return
    if (!entityName.trim()) return

    setIsRegistering(true)
    setRegistrationError(null)
    try {
      // The /api/detect-faces endpoint returns face crops as PNG. Re-encode
      // to JPEG so the upload-face request stays small and never trips a
      // reverse-proxy / CDN HTTP 413 limit.
      const { base64, mimeType } = await reencodeAsJpeg(face.image_base64, 'image/png')
      await registerFromBase64(base64, mimeType, face.bbox)
    } catch (e) {
      setRegistrationError(e instanceof Error ? e.message : 'Could not register face.')
    } finally {
      setIsRegistering(false)
    }
  }, [detectedFaces, entityName, reencodeAsJpeg, registerFromBase64, selectedFaceIdx])

  const handleConfirmManual = useCallback(async () => {
    if (!entityName.trim()) return
    setIsRegistering(true)
    setRegistrationError(null)
    try {
      const { base64, mimeType, bbox } = await cropFrameToBase64()
      await registerFromBase64(base64, mimeType, bbox)
    } catch (e) {
      setRegistrationError(e instanceof Error ? e.message : 'Could not register face.')
    } finally {
      setIsRegistering(false)
    }
  }, [cropFrameToBase64, entityName, registerFromBase64])

  const switchToManual = useCallback(() => {
    setRegistrationError(null)
    setMode('manual')
  }, [])

  const switchToDetect = useCallback(() => {
    setRegistrationError(null)
    setMode('detect')
  }, [])

  const canConfirmDetect =
    !isRegistering &&
    !isDetecting &&
    entityName.trim().length > 0 &&
    selectedFaceIdx !== null &&
    detectedFaces[selectedFaceIdx] !== undefined

  const canConfirmManual = !isRegistering && entityName.trim().length > 0

  if (!open) return null

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-brand-charcoal/40 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden
      />
      <div
        className="relative w-full max-w-md rounded-xl border border-gray-200 bg-surface shadow-xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby="snap-face-modal-title"
      >
        <div className="flex items-center justify-between border-b border-gray-200 px-5 py-4">
          <div className="flex items-center gap-2">
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-accent/10 text-accent">
              <IconCamera className="w-4 h-4" />
            </span>
            <h2 id="snap-face-modal-title" className="text-lg font-semibold text-gray-900">
              {mode === 'manual' ? 'Mark face area manually' : 'Snap face from video'}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 hover:text-gray-700 transition-colors"
            aria-label="Close"
          >
            <IconClose />
          </button>
        </div>

        <div className="p-5">
          {mode === 'detect' && (
            <div className="flex items-center gap-4 mb-4 p-3 rounded-xl bg-gray-50 border border-gray-200">
              <div className="w-20 h-14 rounded-lg overflow-hidden bg-gray-200 shrink-0 border border-gray-200">
                {frameDataUrl ? (
                  <img
                    src={frameDataUrl}
                    alt="Captured video frame"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-[10px] text-gray-400">
                    No frame
                  </div>
                )}
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-gray-900">Captured frame</p>
                <p className="text-xs text-gray-500">
                  Paused at {fmtTimeShort(capturedAtSec)}. We detect every face we can in this frame—tap one to add that
                  person (use Snap again from this or another moment for more people).
                </p>
              </div>
            </div>
          )}

          <div className="mb-4">
            <label htmlFor="snap-face-name" className="block text-sm font-medium text-gray-700 mb-1.5">
              Person name
            </label>
            <input
              id="snap-face-name"
              type="text"
              value={entityName}
              onChange={(e) => setEntityName(e.target.value)}
              placeholder="e.g. Suspect, Witness, John Smith"
              className="w-full h-10 px-3 rounded-xl border border-gray-200 bg-white text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
            />
          </div>

          {mode === 'detect' ? (
            <>
              <div className="mb-4">
                {isDetecting && (
                  <div className="flex flex-col items-center py-8 gap-3">
                    <div className="w-8 h-8 border-2 border-gray-300 border-t-gray-800 rounded-full animate-spin" />
                    <p className="text-sm text-gray-500">Detecting faces in the frame...</p>
                  </div>
                )}

                {!isDetecting && detectError && (
                  <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
                    <p className="text-sm text-amber-800">{detectError}</p>
                    <p className="mt-1 text-xs text-amber-700/80">
                      You can still continue: mark the face area manually on the frame.
                    </p>
                    <button
                      type="button"
                      onClick={switchToManual}
                      className="mt-3 inline-flex items-center gap-1.5 h-8 px-3 rounded-md text-xs font-medium bg-amber-600 text-white hover:bg-amber-700 transition-colors"
                    >
                      Mark area manually
                    </button>
                  </div>
                )}

                {!isDetecting && !detectError && detectedFaces.length > 0 && (
                  <div>
                    <p className="text-sm text-gray-600 mb-3">
                      {detectedFaces.length === 1
                        ? '1 face detected in this frame.'
                        : `${detectedFaces.length} faces detected in this frame.`}{' '}
                      Tap one to anonymize—run Snap face again if you need another person.
                    </p>
                    <div className="flex flex-wrap gap-3">
                      {detectedFaces.map((face, idx) => (
                        <button
                          key={`${face.index}-${idx}`}
                          type="button"
                          onClick={() => setSelectedFaceIdx(idx)}
                          className={`relative w-20 h-20 rounded-full overflow-hidden border-[2.5px] transition-all duration-150 ${
                            selectedFaceIdx === idx
                              ? 'border-accent ring-2 ring-accent/40 scale-105 shadow-[0_0_14px_rgba(0,220,130,0.45)]'
                              : 'border-gray-200 hover:border-gray-400'
                          }`}
                        >
                          <img
                            src={`data:image/png;base64,${face.image_base64}`}
                            alt={`Detected face ${idx + 1}`}
                            className="w-full h-full object-cover"
                          />
                          <span className="absolute bottom-0 left-1/2 -translate-x-1/2 text-[10px] font-medium text-white bg-black/60 px-1.5 py-px rounded-t">
                            {Math.round((face.confidence || 0) * 100)}%
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {!isDetecting && !detectError && detectedFaces.length > 0 && (
                <div className="mb-4 text-center">
                  <button
                    type="button"
                    onClick={switchToManual}
                    className="text-xs text-gray-500 hover:text-gray-700 underline underline-offset-2 transition-colors"
                  >
                    None of these? Mark the area manually
                  </button>
                </div>
              )}

              {registrationError && (
                <p className="mb-4 text-sm text-red-600">{registrationError}</p>
              )}

              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={onClose}
                  className="h-8 px-3 rounded-[9.6px] text-sm font-medium text-gray-700 bg-gray-200 hover:bg-gray-300 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleConfirmDetect}
                  disabled={!canConfirmDetect}
                  className="h-8 px-3 rounded-[9.6px] text-sm font-medium bg-brand-charcoal text-brand-white hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-2"
                >
                  {isRegistering && (
                    <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  )}
                  {isRegistering ? 'Adding...' : 'Add to anonymize list'}
                </button>
              </div>
            </>
          ) : (
            <>
              <p className="text-xs text-gray-600 mb-3">
                Drag the circle to position it over the face, or grab the corner handle to resize. The
                cropped area will be added to the anonymize list.
              </p>

              <div
                ref={cropContainerRef}
                className="relative mx-auto rounded-xl overflow-hidden bg-gray-100 border border-gray-200"
                style={{ width: CROP_CONTAINER_W, height: CROP_CONTAINER_H }}
              >
                {frameDataUrl ? (
                  <img
                    src={frameDataUrl}
                    alt="Captured video frame"
                    className="absolute inset-0 w-full h-full object-cover select-none pointer-events-none"
                    draggable={false}
                    style={{ objectPosition: 'center' }}
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center text-xs text-gray-400">
                    No frame captured
                  </div>
                )}

                <div className="absolute inset-0 pointer-events-none">
                  <svg
                    className="absolute inset-0 w-full h-full"
                    viewBox={`0 0 ${CROP_CONTAINER_W} ${CROP_CONTAINER_H}`}
                    preserveAspectRatio="none"
                  >
                    <defs>
                      <mask id="snap-face-crop-mask">
                        <rect width={CROP_CONTAINER_W} height={CROP_CONTAINER_H} fill="white" />
                        <ellipse
                          cx={crop.x + crop.width / 2}
                          cy={crop.y + crop.height / 2}
                          rx={crop.width / 2}
                          ry={crop.height / 2}
                          fill="black"
                        />
                      </mask>
                    </defs>
                    <rect
                      width={CROP_CONTAINER_W}
                      height={CROP_CONTAINER_H}
                      fill="rgba(0,0,0,0.5)"
                      mask="url(#snap-face-crop-mask)"
                    />
                  </svg>
                </div>

                <div
                  className="absolute border-[2.5px] border-white rounded-full shadow-lg cursor-move"
                  style={{
                    left: crop.x,
                    top: crop.y,
                    width: crop.width,
                    height: crop.height,
                    boxShadow: '0 0 0 1px rgba(0,0,0,0.3), inset 0 0 0 1px rgba(255,255,255,0.2)',
                  }}
                  onMouseDown={onMouseDownCrop}
                >
                  <div
                    className="absolute -bottom-2 -right-2 w-5 h-5 bg-white border-2 border-gray-400 rounded-full cursor-se-resize flex items-center justify-center shadow-md hover:border-brand-charcoal hover:scale-110 transition-all"
                    onMouseDown={onMouseDownResize}
                    aria-hidden
                  >
                    <svg
                      className="w-3 h-3 text-gray-600"
                      viewBox="0 0 12 12"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.75"
                      strokeLinecap="round"
                    >
                      <path d="M2 10L10 2" />
                      <path d="M6 10L10 6" />
                    </svg>
                  </div>
                </div>
              </div>

              {registrationError && (
                <p className="mt-4 text-sm text-red-600">{registrationError}</p>
              )}

              <div className="mt-5 flex items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={switchToDetect}
                  className="h-8 px-3 rounded-[9.6px] text-xs font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 transition-colors"
                >
                  Back to auto-detect
                </button>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={onClose}
                    className="h-8 px-3 rounded-[9.6px] text-sm font-medium text-gray-700 bg-gray-200 hover:bg-gray-300 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={handleConfirmManual}
                    disabled={!canConfirmManual}
                    className="h-8 px-3 rounded-[9.6px] text-sm font-medium bg-brand-charcoal text-brand-white hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-2"
                  >
                    {isRegistering && (
                      <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    )}
                    {isRegistering ? 'Adding...' : 'Add to anonymize list'}
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
