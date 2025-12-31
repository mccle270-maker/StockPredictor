import { useEffect } from 'react'
import { Hero } from './components/Hero'
import { Stats } from './components/Stats'
import { Controls } from './components/Controls'
import { Metrics } from './components/Metrics'
import { FeatureImportance } from './components/FeatureImportance'
import { ApiStatus } from './components/ApiStatus'
import { usePredictions } from './hooks/usePredictions'
import { useApiHealth } from './hooks/useApiHealth'

function App() {
  const { data, loading, error, runPredict } = usePredictions()
  const { status, message, refresh } = useApiHealth()

  useEffect(() => {
    runPredict({ ticker: 'AAPL', horizon: 1, model_type: 'rf' })
  }, [runPredict])

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-930 to-slate-950 text-slate-50">
      <div className="mx-auto flex max-w-5xl flex-col gap-8 px-4 py-10">
        <Hero onGetStarted={() => runPredict({ ticker: 'AAPL', horizon: 1, model_type: 'rf' })} loading={loading} />

        <ApiStatus status={status} message={message} onRefresh={refresh} />

        <Controls loading={loading} onSubmit={runPredict} />

        {error && (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-100">
            {error}
          </div>
        )}

        <Stats data={data} loading={loading} />

        <div className="grid gap-4 md:grid-cols-2">
          <Metrics data={data} loading={loading} />
          <FeatureImportance data={data} loading={loading} />
        </div>
      </div>
    </div>
  )
}

export default App
