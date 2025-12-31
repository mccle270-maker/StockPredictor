import type { PredictResponse } from '../api/client'

type Props = { data: PredictResponse | null; loading: boolean }

type FeatureRow = { feature: string; importance: number }

export function FeatureImportance({ data, loading }: Props) {
  const features = normalize(data?.feature_importance)
  return (
    <div className="rounded-3xl border border-white/5 bg-white/5 p-4 shadow-inner shadow-black/20">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Top Features</h2>
        <span className="text-xs text-indigo-100/70">Model-driven</span>
      </div>
      <div className="mt-3 grid gap-2 md:grid-cols-2">
        {features.length === 0 && (
          <p className="text-sm text-indigo-100/70">{loading ? 'Loading…' : 'Waiting for prediction…'}</p>
        )}
        {features.map((f) => (
          <div key={f.feature} className="flex items-center justify-between rounded-2xl bg-black/30 px-3 py-2">
            <span className="text-sm text-indigo-100">{f.feature}</span>
            <span className="text-sm font-semibold text-white">{f.importance.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function normalize(list: PredictResponse['feature_importance']): FeatureRow[] {
  if (!Array.isArray(list)) return []
  return list
    .filter((f): f is { feature: string; importance: number } => !!f.feature && typeof f.importance === 'number')
    .sort((a, b) => (b.importance ?? 0) - (a.importance ?? 0))
    .slice(0, 10)
}
