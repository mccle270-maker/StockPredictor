import type { PredictResponse } from '../api/client'

type Props = { data: PredictResponse | null; loading: boolean }

type MetricItem = { label: string; value: string }

type FoldRow = { label: string; value: string }

export function Metrics({ data, loading }: Props) {
  const wf = data?.walk_forward as Record<string, unknown> | null | undefined
  const metrics: MetricItem[] = [
    { label: 'Sharpe', value: formatNumber(wf?.sharpe) },
    { label: 'Win Rate', value: formatPercent(wf?.win_rate) },
    { label: 'Max Drawdown', value: formatPercent(wf?.max_drawdown, true) },
    { label: 'Accuracy', value: formatPercent(wf?.accuracy) },
  ]

  const folds: FoldRow[] = extractFolds(wf)

  return (
    <div className="rounded-3xl border border-white/5 bg-white/5 p-4 shadow-inner shadow-black/20">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Walk-Forward Metrics</h2>
        <span className="text-xs text-indigo-100/70">Retrained per fold</span>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-4">
        {metrics.map((m) => (
          <div key={m.label} className="rounded-2xl border border-white/5 bg-black/30 px-3 py-3 text-sm text-indigo-100/80">
            <p className="text-[11px] uppercase tracking-[0.18em] text-indigo-100/60">{m.label}</p>
            <p className="mt-1 text-lg font-semibold text-white">{loading ? '…' : m.value}</p>
          </div>
        ))}
      </div>

      <div className="mt-4">
        <p className="text-xs uppercase tracking-[0.15em] text-indigo-100/70">Folds</p>
        <div className="mt-2 overflow-hidden rounded-2xl border border-white/5 bg-black/30">
          <div className="grid grid-cols-2 gap-px bg-white/5 text-xs text-indigo-100/80">
            {folds.length === 0 && <div className="col-span-2 p-3">{loading ? 'Loading…' : 'Waiting for prediction…'}</div>}
            {folds.map((f) => (
              <div key={f.label} className="bg-slate-900/50 px-3 py-2">
                <p className="text-[11px] uppercase tracking-[0.16em] text-indigo-100/60">{f.label}</p>
                <p className="mt-1 text-white">{f.value}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function formatNumber(val: unknown) {
  const n = typeof val === 'number' ? val : Number(val)
  return Number.isFinite(n) ? n.toFixed(2) : '—'
}

function formatPercent(val: unknown, signed = false) {
  const n = typeof val === 'number' ? val : Number(val)
  if (!Number.isFinite(n)) return '—'
  const pct = (n * 100).toFixed(1)
  return signed ? `${n >= 0 ? '' : ''}${pct}%` : `${pct}%`
}

function extractFolds(wf: Record<string, unknown> | null | undefined): FoldRow[] {
  if (!wf) return []
  // If backend returns a list of folds, flatten; else show key metrics
  if (Array.isArray(wf)) {
    return wf.slice(0, 6).map((entry, idx) => ({ label: `Fold ${idx + 1}`, value: formatFold(entry) }))
  }
  // Otherwise map known keys
  const keys = Object.keys(wf)
  return keys.slice(0, 6).map((k) => ({ label: k, value: formatFold(wf[k]) }))
}

function formatFold(entry: unknown) {
  if (entry == null) return '—'
  if (typeof entry === 'number') return formatPercent(entry)
  if (typeof entry === 'object') {
    const obj = entry as Record<string, unknown>
    const acc = obj.accuracy ?? obj.sharpe ?? obj.win_rate ?? obj.ret
    if (acc != null && typeof acc === 'number') return formatPercent(acc)
    return Object.entries(obj)
      .slice(0, 2)
      .map(([k, v]) => `${k}: ${formatPercent(v)}`)
      .join('  ')
  }
  return String(entry)
}
