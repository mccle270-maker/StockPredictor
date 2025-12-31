import { motion } from 'framer-motion'
import type { PredictResponse } from '../api/client'

type Props = { data: PredictResponse | null; loading: boolean }

export function Stats({ data, loading }: Props) {
  const items = [
    { label: 'Predicted Return', value: data ? `${(data.pred_next_ret * 100).toFixed(2)}%` : '—' },
    { label: 'Prob Up', value: data?.prob_up != null ? `${(data.prob_up * 100).toFixed(1)}%` : '—' },
    { label: 'Next Price', value: data?.pred_next_price ? `$${data.pred_next_price.toFixed(2)}` : '—' },
    { label: 'Last Close', value: data?.last_close ? `$${data.last_close.toFixed(2)}` : '—' },
  ]

  return (
    <div className="grid gap-4 md:grid-cols-4">
      {items.map((item) => (
        <motion.div
          key={item.label}
          whileHover={{ y: -2 }}
          className="rounded-2xl border border-white/5 bg-white/5 p-4 shadow-inner shadow-black/20"
        >
          <p className="text-xs uppercase tracking-[0.15em] text-indigo-100/70">{item.label}</p>
          <p className="mt-2 text-xl font-semibold text-white">{loading ? '…' : item.value}</p>
        </motion.div>
      ))}
    </div>
  )
}
