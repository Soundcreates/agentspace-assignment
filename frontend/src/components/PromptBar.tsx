import { useRef, useState, type DragEvent, type FormEvent, type KeyboardEvent } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import { LoadingIndicator } from './LoadingIndicator'

gsap.registerPlugin(useGSAP)

const ACCEPT = '.txt,.md,.pdf,text/plain,text/markdown,application/pdf'

type PromptBarProps = {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  onAttach: (files: File[]) => void
  indexing: boolean
  asking: boolean
  disabled?: boolean
}

export function PromptBar({
  value,
  onChange,
  onSubmit,
  onAttach,
  indexing,
  asking,
  disabled,
}: PromptBarProps) {
  const barRef = useRef<HTMLFormElement>(null)
  const attachRef = useRef<HTMLButtonElement>(null)
  const submitRef = useRef<HTMLButtonElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const dragDepth = useRef(0)
  const [dragging, setDragging] = useState(false)

  useGSAP(
    (_ctx, contextSafe) => {
      const scaleOn = contextSafe?.((el: HTMLElement) => {
        gsap.to(el, { scale: 1.08, duration: 0.22, ease: 'power2.out' })
      })
      const scaleOff = contextSafe?.((el: HTMLElement) => {
        gsap.to(el, { scale: 1, duration: 0.28, ease: 'power2.out' })
      })
      const attach = attachRef.current
      const submit = submitRef.current
      if (!attach || !submit || !scaleOn || !scaleOff) return

      const enterAttach = () => scaleOn(attach)
      const leaveAttach = () => scaleOff(attach)
      const enterSubmit = () => scaleOn(submit)
      const leaveSubmit = () => scaleOff(submit)
      attach.addEventListener('mouseenter', enterAttach)
      attach.addEventListener('mouseleave', leaveAttach)
      submit.addEventListener('mouseenter', enterSubmit)
      submit.addEventListener('mouseleave', leaveSubmit)
      return () => {
        attach.removeEventListener('mouseenter', enterAttach)
        attach.removeEventListener('mouseleave', leaveAttach)
        submit.removeEventListener('mouseenter', enterSubmit)
        submit.removeEventListener('mouseleave', leaveSubmit)
      }
    },
    { scope: barRef },
  )

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    onSubmit()
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      onSubmit()
    }
  }

  const busy = indexing || asking

  function hasFiles(event: DragEvent) {
    return Array.from(event.dataTransfer.types).includes('Files')
  }

  function handleDragEnter(event: DragEvent) {
    if (busy || !hasFiles(event)) return
    event.preventDefault()
    dragDepth.current += 1
    setDragging(true)
  }

  function handleDragOver(event: DragEvent) {
    if (busy || !hasFiles(event)) return
    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
  }

  function handleDragLeave(event: DragEvent) {
    if (!hasFiles(event)) return
    event.preventDefault()
    dragDepth.current = Math.max(0, dragDepth.current - 1)
    if (dragDepth.current === 0) setDragging(false)
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    if (busy) return
    const files = Array.from(event.dataTransfer.files)
    if (files.length) onAttach(files)
  }

  return (
    <form
      ref={barRef}
      onSubmit={handleSubmit}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`prompt-bar relative w-full max-w-2xl rounded-[1.75rem] border p-2 shadow-[0_24px_80px_rgba(12,18,24,0.35)] backdrop-blur-2xl transition-[border-color,background-color,box-shadow] duration-200 ${
        dragging
          ? 'border-gold/70 bg-gold/15 shadow-[0_0_0_1px_rgba(232,201,160,0.35),0_24px_80px_rgba(12,18,24,0.35)]'
          : 'border-white/25 bg-white/12'
      }`}
    >
      {dragging && (
        <div
          className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center rounded-[1.75rem] bg-ink/35"
          aria-hidden
        >
          <p className="font-display text-lg italic text-cream">Drop .txt, .md, or .pdf files</p>
        </div>
      )}
      <div className="flex items-end gap-1">
        <button
          ref={attachRef}
          type="button"
          onClick={() => fileRef.current?.click()}
          disabled={busy}
          className="m-1 flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl text-cream/90 transition hover:bg-white/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold disabled:opacity-50"
          aria-label="Attach files"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
            <path
              d="M21 12.5v3.25A4.25 4.25 0 0 1 16.75 20H9.5A5.5 5.5 0 0 1 4 14.5V8.25A4.25 4.25 0 0 1 8.25 4H14"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
            <path
              d="M14 4l6 6v6.25A2.25 2.25 0 0 1 17.75 18.5H10A3.5 3.5 0 0 1 6.5 15V9.25A2.25 2.25 0 0 1 8.75 7H16"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
          </svg>
        </button>
        <input
          ref={fileRef}
          type="file"
          multiple
          accept={ACCEPT}
          className="hidden"
          onChange={(event) => {
            if (event.target.files?.length) onAttach(Array.from(event.target.files))
            event.target.value = ''
          }}
        />
        <textarea
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={disabled}
          placeholder="Ask about the files you attached… or drop files here"
          className="max-h-40 min-h-12 flex-1 resize-none bg-transparent px-2 py-3.5 text-[1.05rem] leading-snug text-cream placeholder:text-cream/45 focus:outline-none"
        />
        <button
          ref={submitRef}
          type="submit"
          disabled={busy || !value.trim()}
          className="m-1 flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-cream text-ink shadow-sm transition hover:bg-gold focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold disabled:cursor-not-allowed disabled:opacity-40"
          aria-label="Ask"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
            <path
              d="M5 12h12M13 6l6 6-6 6"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
      {indexing && (
        <div className="px-4 pb-3 pt-1">
          <LoadingIndicator label="Reading and indexing your files…" />
        </div>
      )}
    </form>
  )
}
