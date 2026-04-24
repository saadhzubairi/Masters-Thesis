"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { RunCard } from "./components/run-card"
import { listRuns, getRun } from "@/lib/api"
import type { RunSummary } from "@/lib/types"
import { toast } from "sonner"

export default function Dashboard() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadRuns()
    const interval = setInterval(loadRuns, 3000) // Poll for status updates
    return () => clearInterval(interval)
  }, [])

  async function loadRuns() {
    try {
      const data = await listRuns()
      setRuns(data)
    } catch {
      // Backend not running yet -- that is fine
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold">Dashboard</h2>
          <p className="text-sm text-muted-foreground">
            {runs.length} experiment{runs.length !== 1 ? "s" : ""}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={async () => {
              try {
                toast.info("Fetching all run data...")
                const details = await Promise.all(runs.map(r => getRun(r.id)))
                const payload = details.map(run => {
                  const epochs = run.metrics?.epochs || []
                  const last = epochs[epochs.length - 1]
                  const totalEpochs = run.config?.stages
                    ? run.config.stages.reduce((s: number, st: { epochs: number }) => s + st.epochs, 0)
                    : 0
                  const elapsed = last?.elapsed_s || 0
                  const avgPerEpoch = epochs.length > 0 ? elapsed / epochs.length : 0
                  return {
                    id: run.id,
                    name: run.name,
                    status: run.status,
                    config: run.config,
                    summary: {
                      total_epochs: totalEpochs,
                      completed_epochs: epochs.length,
                      train_loss: last?.train_loss ?? null,
                      test_loss: last?.test_loss ?? null,
                      stage: last?.stage ?? null,
                      elapsed_s: elapsed || null,
                      avg_per_epoch_s: avgPerEpoch > 0 ? +avgPerEpoch.toFixed(2) : null,
                      loss_components: last?.components ?? null,
                    },
                    metrics: run.metrics?.summary ?? null,
                    epochs: epochs.map(e => ({
                      epoch: e.epoch,
                      stage: e.stage,
                      train_loss: e.train_loss,
                      test_loss: e.test_loss ?? null,
                      components: e.components,
                      learned_params: e.learned_params ?? null,
                      elapsed_s: e.elapsed_s ?? null,
                    })),
                    errors: run.metrics?.errors ?? null,
                  }
                })
                navigator.clipboard.writeText(JSON.stringify(payload, null, 2))
                toast.success(`Copied data for ${payload.length} experiments`)
              } catch {
                toast.error("Failed to fetch run data")
              }
            }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            Copy All
          </Button>
          <Button asChild>
            <Link href="/runs/new">+ New Run</Link>
          </Button>
        </div>
      </div>
      {loading ? (
        <p className="text-sm text-muted-foreground">Loading...</p>
      ) : runs.length === 0 ? (
        <p className="text-sm text-muted-foreground">No experiments yet. Start a new run!</p>
      ) : (
        <div className="space-y-2">
          {runs.map((run) => (
            <RunCard key={run.id} run={run} onDeleted={loadRuns} />
          ))}
        </div>
      )}
    </div>
  )
}
