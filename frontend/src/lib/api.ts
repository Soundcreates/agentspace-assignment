export type SessionResponse = {
  session_id: string
  chunk_count: number
  source_count: number
  partial: boolean
}

export type RetrievedChunk = {
  id: string
  filename: string
  rank: number
  text: string
}

export type AskDoneEvent = {
  type: 'done'
  answer: string
  retrieved_chunks: RetrievedChunk[]
  partial: boolean
  source_ids: string[]
}

export type AskTokenEvent = { type: 'token'; text: string }
export type AskErrorEvent = { type: 'error'; message: string }
export type AskEvent = AskTokenEvent | AskDoneEvent | AskErrorEvent

function apiBaseUrl(): string {
  const raw = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? ''
  return raw.trim().replace(/\/$/, '')
}

function apiUrl(path: string): string {
  const base = apiBaseUrl()
  const normalized = path.startsWith('/') ? path : `/${path}`
  return `${base}${normalized}`
}

async function readError(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as { detail?: unknown }
    if (typeof data.detail === 'string') return data.detail
    if (Array.isArray(data.detail)) return JSON.stringify(data.detail)
  } catch {
    /* ignore */
  }
  return response.statusText || 'Request failed'
}

export async function createSession(files: File[]): Promise<SessionResponse> {
  const form = new FormData()
  for (const file of files) form.append('files', file)
  const response = await fetch(apiUrl('/api/session'), {
    method: 'POST',
    body: form,
  })
  if (!response.ok) throw new Error(await readError(response))
  return (await response.json()) as SessionResponse
}

export async function askQuestion(
  sessionId: string,
  question: string,
  onEvent: (event: AskEvent) => void,
  topK = 6,
): Promise<void> {
  const response = await fetch(apiUrl('/api/ask'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ session_id: sessionId, question, top_k: topK }),
  })
  if (!response.ok) throw new Error(await readError(response))
  if (!response.body) throw new Error('No response stream')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() ?? ''
    for (const part of parts) {
      const line = part
        .split('\n')
        .find((entry) => entry.startsWith('data: '))
      if (!line) continue
      const payload = JSON.parse(line.slice(6)) as AskEvent
      onEvent(payload)
    }
  }
}
