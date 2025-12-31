import { FormEvent, useState } from 'react'

export type ControlsProps = {
  loading: boolean
  onSubmit: (opts: { ticker: string; horizon: number; model_type: string }) => void
}

const MODEL_OPTIONS = [
  { value: 'rf', label: 'Random Forest' },
  { value: 'xgb', label: 'XGBoost' },
  { value: 'gbrt', label: 'Grad Boost' },
  { value: 'linreg', label: 'Linear' },
]

export function Controls({ loading, onSubmit }: ControlsProps) {
  const [ticker, setTicker] = useState('AAPL')
  const [horizon, setHorizon] = useState(1)
  const [modelType, setModelType] = useState('rf')

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    onSubmit({ ticker: ticker.trim().toUpperCase(), horizon, model_type: modelType })
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-3xl border border-white/5 bg-white/5 p-4 shadow-inner shadow-black/20 backdrop-blur"
    >
      <div className="grid gap-4 md:grid-cols-5 md:items-end">
        <div className="md:col-span-2">
          <label className="text-xs uppercase tracking-[0.15em] text-indigo-100/70">Ticker</label>
          <input
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="mt-1 w-full rounded-xl border border-white/10 bg-white/10 px-3 py-2 text-white placeholder:text-indigo-100/50 focus:border-indigo-400 focus:outline-none"
            placeholder="AAPL"
            autoComplete="off"
          />
        </div>
        <div>
          <label className="text-xs uppercase tracking-[0.15em] text-indigo-100/70">Horizon (days)</label>
          <input
            type="number"
            min={1}
            max={20}
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
            className="mt-1 w-full rounded-xl border border-white/10 bg-white/10 px-3 py-2 text-white focus:border-indigo-400 focus:outline-none"
          />
        </div>
        <div>
          <label className="text-xs uppercase tracking-[0.15em] text-indigo-100/70">Model</label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="mt-1 w-full rounded-xl border border-white/10 bg-white/10 px-3 py-2 text-white focus:border-indigo-400 focus:outline-none"
          >
            {MODEL_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value} className="bg-slate-900 text-white">
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <button
          type="submit"
          disabled={loading}
          className="h-full rounded-xl bg-indigo-400/90 px-4 py-3 font-semibold text-slate-900 shadow-lg shadow-indigo-500/30 transition hover:-translate-y-[1px] hover:shadow-xl disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? 'Running…' : 'Run Prediction'}
        </button>
      </div>
    </form>
  )
}
