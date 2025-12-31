export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export type PredictPayload = {
  ticker: string
  horizon?: number
  model_type?: string
}

export type PredictResponse = {
  ticker: string
  model_type: string
  horizon: number
  pred_next_ret: number
  pred_next_price: number
  confidence_score?: number | null
  prob_up?: number | null
  prob_down?: number | null
  prob_up_gaf?: number | null
  last_close?: number | null
  vol_20d?: number | null
  walk_forward?: Record<string, unknown> | null
  feature_importance?: { feature: string; importance: number | null }[] | null
}

export async function fetchPredict(payload: PredictPayload): Promise<PredictResponse> {
  const res = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Predict failed: ${res.status} ${text}`)
  }
  return res.json()
}

export async function fetchHealth(): Promise<'ok'> {
  const res = await fetch(`${API_URL}/health`)
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`)
  return 'ok'
}
