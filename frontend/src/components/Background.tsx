import { useRef } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import background from '../assets/background.png'

gsap.registerPlugin(useGSAP)

export function Background() {
  const imageRef = useRef<HTMLImageElement>(null)

  useGSAP(
    () => {
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
      gsap.to(imageRef.current, {
        scale: 1.08,
        x: 18,
        y: -12,
        duration: 28,
        ease: 'sine.inOut',
        yoyo: true,
        repeat: -1,
      })
    },
    { scope: imageRef },
  )

  return (
    <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden" aria-hidden>
      <img
        ref={imageRef}
        src={background}
        alt=""
        className="h-full w-full origin-center object-cover will-change-transform"
      />
      <div className="absolute inset-0 bg-gradient-to-b from-ink/35 via-ink/25 to-ink/70" />
    </div>
  )
}
