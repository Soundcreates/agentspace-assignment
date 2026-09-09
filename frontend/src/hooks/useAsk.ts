import { useCallback, useState } from 'react'
import { askQuestion, type RetrievedChunk } from '../lib/api'

type AskHandlers = {
  onToken: (text: string) => void
  onDone: (answer: string, chunks: RetrievedChunk[]) => void
}

export function useAsk() {
  const [asking, setAsking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const ask = useCallback(async (sessionId: string, question: string, handlers: AskHandlers) => {
    setAsking(true)
    setError(null)
    let failure: string | null = null
    try {
      await askQuestion(sessionId, question, (event) => {
        if (event.type === 'token') {
          handlers.onToken(event.text)
        } else if (event.type === 'done') {
          handlers.onDone(event.answer, event.retrieved_chunks)
        } else {
          failure = event.message
          setError(event.message)
        }
      })
    } catch (err) {
      failure = err instanceof Error ? err.message : 'Could not ask the question'
      setError(failure)
    } finally {
      setAsking(false)
    }
    return failure
  }, [])

  return { asking, error, setError, ask }
}
