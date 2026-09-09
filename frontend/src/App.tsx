import { useRef, useState } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import { AttachedFiles } from './components/AttachedFiles'
import { Background } from './components/Background'
import { ChatThread, type ChatMessage } from './components/ChatThread'
import { Greeting } from './components/Greeting'
import { PromptBar } from './components/PromptBar'
import { useAsk } from './hooks/useAsk'
import { useSession } from './hooks/useSession'
import type { RetrievedChunk } from './lib/api'

gsap.registerPlugin(useGSAP)

const SUPPORTED_EXTENSIONS = new Set(['.txt', '.md', '.pdf'])

function fileExtension(name: string): string {
  const i = name.lastIndexOf('.')
  return i >= 0 ? name.slice(i).toLowerCase() : ''
}

function newId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

export default function App() {
  const pageRef = useRef<HTMLDivElement>(null)
  const [question, setQuestion] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const { session, indexing, error: sessionError, setError: setSessionError, upload, clearSession } =
    useSession()
  const { asking, error: askError, setError: setAskError, ask } = useAsk()
  const chatStarted = messages.length > 0

  useGSAP(
    () => {
      if (chatStarted) return
      const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      if (reduce) {
        gsap.set(['.greeting-kicker', '.greeting-title', '.greeting-line', '.prompt-bar'], {
          opacity: 1,
          y: 0,
        })
        return
      }
      gsap.set(['.greeting-kicker', '.greeting-title', '.greeting-line', '.prompt-bar'], {
        opacity: 0,
        y: 22,
      })
      const tl = gsap.timeline({ defaults: { ease: 'power3.out' } })
      tl.to('.greeting-kicker', { opacity: 1, y: 0, duration: 0.7 }, 0.15)
        .to('.greeting-title', { opacity: 1, y: 0, duration: 0.9 }, 0.28)
        .to('.greeting-line', { opacity: 1, y: 0, duration: 0.8 }, 0.42)
        .to('.prompt-bar', { opacity: 1, y: 0, duration: 0.85 }, 0.55)
    },
    { scope: pageRef, dependencies: [chatStarted] },
  )

  async function handleAttach(incoming: File[]) {
    const supported = incoming.filter((file) =>
      SUPPORTED_EXTENSIONS.has(fileExtension(file.name)),
    )
    if (!supported.length) {
      setSessionError('Only .txt, .md, or .pdf files are supported.')
      return
    }
    if (supported.length < incoming.length) {
      setSessionError('Some files were skipped — only .txt, .md, or .pdf are supported.')
    } else {
      setSessionError(null)
    }
    const merged = [...files]
    for (const file of supported) {
      if (!merged.some((existing) => existing.name === file.name && existing.size === file.size)) {
        merged.push(file)
      }
    }
    setFiles(merged)
    setAskError(null)
    await upload(merged)
  }

  function handleRemove(name: string) {
    const next = files.filter((file) => file.name !== name)
    setFiles(next)
    if (next.length) void upload(next)
    else clearSession()
  }

  async function handleSubmit() {
    const trimmed = question.trim()
    if (!trimmed || indexing || asking) return
    if (!files.length || !session) {
      setSessionError('Attach a .txt, .md, or .pdf file first.')
      return
    }

    const assistantId = newId()
    setMessages((prev) => [
      ...prev,
      { id: newId(), role: 'user', content: trimmed },
      { id: assistantId, role: 'assistant', content: '' },
    ])
    setQuestion('')
    setAskError(null)

    const handlers = {
      onToken: (text: string) => {
        setMessages((prev) =>
          prev.map((message) =>
            message.id === assistantId
              ? { ...message, content: message.content + text }
              : message,
          ),
        )
      },
      onDone: (answer: string, chunks: RetrievedChunk[]) => {
        setMessages((prev) =>
          prev.map((message) =>
            message.id === assistantId ? { ...message, content: answer, chunks } : message,
          ),
        )
      },
    }

    let failure = await ask(session.session_id, trimmed, handlers)

    // Server reloads wipe in-memory sessions; re-index attached files and retry once.
    if (failure && /session not found/i.test(failure) && files.length) {
      clearSession()
      const next = await upload(files)
      if (next) {
        setMessages((prev) =>
          prev.map((message) =>
            message.id === assistantId ? { ...message, content: '' } : message,
          ),
        )
        failure = await ask(next.session_id, trimmed, handlers)
      } else {
        failure = 'Session expired after a server reload. Re-attach your files and try again.'
      }
    }

    if (failure) {
      setMessages((prev) =>
        prev.map((message) =>
          message.id === assistantId ? { ...message, content: failure } : message,
        ),
      )
    }
  }

  function handleLeaveSession() {
    setMessages([])
    setQuestion('')
    setFiles([])
    clearSession()
    setSessionError(null)
    setAskError(null)
  }

  const notice = sessionError || askError

  return (
    <div
      ref={pageRef}
      className={`relative ${chatStarted ? 'h-svh overflow-hidden' : 'min-h-svh'}`}
    >
      <Background />
      {chatStarted && (
        <button
          type="button"
          onClick={handleLeaveSession}
          className="fixed left-4 top-4 z-20 flex h-11 w-11 items-center justify-center rounded-full border border-white/20 bg-ink/45 text-cream shadow-[0_12px_40px_rgba(10,16,22,0.35)] backdrop-blur-xl transition hover:border-gold/50 hover:bg-ink/60 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold"
          aria-label="Leave session and return home"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
            <path
              d="M6 6l12 12M18 6L6 18"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
            />
          </svg>
        </button>
      )}
      <main
        className={`relative z-10 mx-auto flex w-full max-w-3xl flex-col px-5 ${
          chatStarted
            ? 'h-full justify-between pb-[max(1rem,env(safe-area-inset-bottom))] pt-6'
            : 'min-h-svh items-center justify-center py-16'
        }`}
      >
        {!chatStarted && <Greeting className="mb-10" />}

        {chatStarted && (
          <div className="chat-scroll min-h-0 flex-1 overflow-y-auto overscroll-contain pb-4 pt-2">
            <ChatThread messages={messages} streaming={asking} />
          </div>
        )}

        <div
          className={`flex w-full flex-col items-center ${
            chatStarted ? 'shrink-0 pt-3' : ''
          }`}
        >
          <PromptBar
            value={question}
            onChange={setQuestion}
            onSubmit={() => void handleSubmit()}
            onAttach={(list) => void handleAttach(list)}
            indexing={indexing}
            asking={asking}
          />
          {!chatStarted && (
            <p className="mt-3 text-center text-xs tracking-wide text-mist/70">
              Drag and drop files onto the prompt, or use the paperclip
            </p>
          )}
          <AttachedFiles files={files} onRemove={handleRemove} />
          {session && !indexing && (
            <p className="mt-3 text-xs tracking-wide text-mist/80">
              Indexed {session.chunk_count} passages from {session.source_count} file
              {session.source_count === 1 ? '' : 's'}
              {session.partial ? ' (partial)' : ''}
            </p>
          )}
          {notice && (
            <p className="mt-3 text-sm text-gold" role="alert">
              {notice}
            </p>
          )}
        </div>
      </main>
    </div>
  )
}
