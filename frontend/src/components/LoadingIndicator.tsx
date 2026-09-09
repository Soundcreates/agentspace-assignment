import { useRef } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'

gsap.registerPlugin(useGSAP)

type LoadingIndicatorProps = {
  label: string
}

export function LoadingIndicator({ label }: LoadingIndicatorProps) {
  const rootRef = useRef<HTMLDivElement>(null)

  useGSAP(
    () => {
      gsap.to('.load-dot', {
        opacity: 0.25,
        y: -3,
        duration: 0.45,
        stagger: { each: 0.12, yoyo: true, repeat: -1 },
        ease: 'sine.inOut',
      })
    },
    { scope: rootRef },
  )

  return (
    <div ref={rootRef} className="flex items-center gap-2 text-sm text-cream/80" role="status">
      <span className="flex gap-1" aria-hidden>
        <span className="load-dot h-1.5 w-1.5 rounded-full bg-gold" />
        <span className="load-dot h-1.5 w-1.5 rounded-full bg-gold" />
        <span className="load-dot h-1.5 w-1.5 rounded-full bg-gold" />
      </span>
      <span>{label}</span>
    </div>
  )
}
