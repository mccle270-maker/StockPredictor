import { useState, useCallback } from 'react'
import { fetchPredict, PredictPayload, PredictResponse } from '../api/client'

export function usePredictions() {
  const [data, setData] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const runPredict = useCallback(async (payload: PredictPayload) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetchPredict(payload)
      setData(res)
    } catch (e: any) {
      setError(e.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }, [])

  return { data, loading, error, runPredict }
}
