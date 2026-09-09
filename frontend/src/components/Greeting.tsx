type GreetingProps = {
  className?: string
}

function greetingForHour(hour: number): { title: string; line: string } {
  if (hour < 12) {
    return { title: 'Good morning.', line: 'What would you like to explore?' }
  }
  if (hour < 17) {
    return { title: 'Good afternoon.', line: 'What would you like to explore?' }
  }
  return { title: 'Good evening.', line: 'What would you like to explore?' }
}

export function Greeting({ className = '' }: GreetingProps) {
  const { title, line } = greetingForHour(new Date().getHours())
  return (
    <header className={`text-center ${className}`}>
      <p className="greeting-kicker mb-3 text-[11px] font-medium uppercase tracking-[0.38em] text-gold/90">
        Your local papers, in the valley mist
      </p>
      <h1 className="greeting-title font-display text-[clamp(2.4rem,6vw,4.6rem)] font-medium leading-[1.05] tracking-[-0.03em] text-cream">
        {title}
      </h1>
      <p className="greeting-line mt-3 font-display text-[clamp(1.2rem,2.6vw,1.85rem)] italic text-cream/85">
        {line}
      </p>
    </header>
  )
}
