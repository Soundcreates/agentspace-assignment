import { useCallback, useState } from 'react'
import { createSession, type SessionResponse } from '../lib/api'

export function useSession() {
  const [session, setSession] = useState<SessionResponse | null>(null)
  const [indexing, setIndexing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const upload = useCallback(async (files: File[]) => {
    if (!files.length) {
      setError('Attach a .txt, .md, or .pdf file first.')
      return null
    }
    setIndexing(true)
    setError(null)
    try {
      const next = await createSession(files)
      setSession(next)
      return next
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Could not index files'
      setError(message)
      setSession(null)
      return null
    } finally {
      setIndexing(false)
    }
  }, [])

  const clearSession = useCallback(() => {
    setSession(null)
  }, [])

  return { session, indexing, error, setError, upload, clearSession }
}
