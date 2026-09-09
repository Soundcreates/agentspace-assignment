import { useLayoutEffect, useRef, useState } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import type { RetrievedChunk } from '../lib/api'
import { LoadingIndicator } from './LoadingIndicator'

gsap.registerPlugin(useGSAP)

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  chunks?: RetrievedChunk[]
}

type ChatThreadProps = {
  messages: ChatMessage[]
  streaming: boolean
}

function Sources({ chunks }: { chunks: RetrievedChunk[] }) {
  const [open, setOpen] = useState(false)
  if (!chunks.length) return null
  return (
    <div className="mt-3 border-t border-white/10 pt-3">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="text-xs font-medium text-mist underline-offset-4 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold"
      >
        {open ? 'Hide sources' : `Show sources (${chunks.length})`}
      </button>
      {open && (
        <ul className="mt-2 space-y-2">
          {chunks.map((chunk) => (
            <li key={`${chunk.id}-${chunk.rank}`} className="rounded-xl bg-white/5 p-2.5 text-xs">
              <p className="mb-1 uppercase tracking-wider text-gold/80">
                {chunk.filename} · rank {chunk.rank}
              </p>
              <p className="line-clamp-5 text-cream/75">{chunk.text}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export function ChatThread({ messages, streaming }: ChatThreadProps) {
  const threadRef = useRef<HTMLDivElement>(null)
  const endRef = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, streaming])

  useGSAP(
    () => {
      const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      if (reduce) return
      const bubbles = threadRef.current?.querySelectorAll('[data-bubble]:not([data-animated])')
      if (!bubbles?.length) return
      bubbles.forEach((node) => {
        node.setAttribute('data-animated', 'true')
        const isUser = node.getAttribute('data-role') === 'user'
        gsap.fromTo(
          node,
          { opacity: 0, y: 14, x: isUser ? 12 : -12 },
          { opacity: 1, y: 0, x: 0, duration: 0.45, ease: 'power3.out' },
        )
      })
    },
    { scope: threadRef, dependencies: [messages.length] },
  )

  return (
    <div ref={threadRef} className="mx-auto flex w-full max-w-2xl flex-col gap-4" role="log" aria-live="polite">
      {messages.map((message, index) => {
        const isUser = message.role === 'user'
        const isLastAssistant =
          message.role === 'assistant' && index === messages.length - 1 && streaming
        return (
          <div
            key={message.id}
            data-bubble
            data-role={message.role}
            className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}
          >
            <article
              className={`max-w-[85%] rounded-3xl px-4 py-3 text-[1.02rem] leading-relaxed shadow-[0_12px_40px_rgba(10,16,22,0.28)] backdrop-blur-xl ${
                isUser
                  ? 'rounded-br-lg border border-gold/35 bg-cream/90 text-ink'
                  : 'rounded-bl-lg border border-white/15 bg-ink/50 text-cream'
              }`}
            >
              <p className="mb-1 text-[10px] font-medium uppercase tracking-[0.24em] opacity-70">
                {isUser ? 'You' : 'Agent'}
              </p>
              <p className="whitespace-pre-wrap">
                {message.content || (isLastAssistant ? '' : '…')}
                {isLastAssistant && !message.content && (
                  <span className="mt-1 inline-block">
                    <LoadingIndicator label="Thinking…" />
                  </span>
                )}
                {isLastAssistant && message.content ? (
                  <span className="ml-0.5 inline-block h-4 w-0.5 animate-pulse bg-gold align-middle" />
                ) : null}
              </p>
              {!isUser && message.chunks ? <Sources chunks={message.chunks} /> : null}
            </article>
          </div>
        )
      })}
      <div ref={endRef} />
    </div>
  )
}
