import { motion } from 'framer-motion'

type Props = {
  onGetStarted: () => void
  loading: boolean
}

export function Hero({ onGetStarted, loading }: Props) {
  return (
    <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-indigo-500/30 via-fuchsia-500/20 to-sky-400/30 p-8 shadow-2xl border border-white/10">
      <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.2em] text-indigo-100/80">AI Stock Signals</p>
          <h1 className="text-3xl md:text-4xl font-bold text-white drop-shadow-sm">Predict the next move</h1>
          <p className="mt-3 max-w-xl text-indigo-50/80">
            Live ML predictions, walk-forward metrics, and feature importance—ready to act.
          </p>
        </div>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          disabled={loading}
          onClick={onGetStarted}
          className="inline-flex items-center gap-2 rounded-full bg-white/90 px-6 py-3 text-slate-900 font-semibold shadow-lg shadow-indigo-500/30 disabled:opacity-50"
        >
          {loading ? 'Loading…' : 'Get Started'}
        </motion.button>
      </div>
    </div>
  )
}
