import { useEffect, useState } from 'react'
import { fetchHealth } from '../api/client'

type Status = 'checking' | 'up' | 'down'

type UseApiHealth = {
  status: Status
  message: string
  refresh: () => void
}

export function useApiHealth(): UseApiHealth {
  const [status, setStatus] = useState<Status>('checking')
  const [message, setMessage] = useState('Checking API…')

  const check = async () => {
    try {
      setStatus('checking')
      setMessage('Checking API…')
      await fetchHealth()
      setStatus('up')
      setMessage('API reachable')
    } catch (e: any) {
      setStatus('down')
      setMessage(e?.message || 'API unreachable')
    }
  }

  useEffect(() => {
    check()
    // Recheck occasionally to keep status fresh
    const id = setInterval(check, 60_000)
    return () => clearInterval(id)
  }, [])

  return { status, message, refresh: check }
}
