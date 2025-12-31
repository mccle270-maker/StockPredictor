import { API_URL } from '../api/client'

export type ApiStatusProps = {
  status: 'checking' | 'up' | 'down'
  message: string
  onRefresh: () => void
}

export function ApiStatus({ status, message, onRefresh }: ApiStatusProps) {
  const color =
    status === 'up' ? 'bg-emerald-500/20 text-emerald-100 border-emerald-400/50' :
    status === 'checking' ? 'bg-amber-500/20 text-amber-100 border-amber-400/50' :
    'bg-red-500/20 text-red-100 border-red-400/50'

  const dot = status === 'up' ? 'bg-emerald-400' : status === 'checking' ? 'bg-amber-400' : 'bg-red-400'

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-2xl border px-4 py-3 text-sm shadow-inner shadow-black/20 backdrop-blur"
         aria-live="polite">
      <div className={`flex items-center gap-2 rounded-full px-3 py-1 ${color}`}>
        <span className={`h-2 w-2 rounded-full ${dot}`} />
        <span className="font-semibold capitalize">{status}</span>
      </div>
      <span className="text-indigo-100/80">{message}</span>
      <span className="text-xs text-indigo-200/70">{API_URL}</span>
      <button
        type="button"
        onClick={onRefresh}
        className="ml-auto rounded-full border border-white/10 bg-white/10 px-3 py-1 text-xs font-semibold text-white transition hover:bg-white/20"
      >
        Retry
      </button>
    </div>
  )
}
