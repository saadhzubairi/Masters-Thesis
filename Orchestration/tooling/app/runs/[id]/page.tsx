"use client"

import { useEffect, useState, useCallback, use } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { getRun, stopRun, streamRun } from "@/lib/api"
import { EpochTable } from "@/app/components/epoch-table"
import { LossChart } from "@/app/components/loss-chart"
import { ResultsGallery } from "@/app/components/results-gallery"
import type { RunDetail, EpochData, SSEEvent } from "@/lib/types"

const statusVariant: Record<string, string> = {
  running: "bg-amber-100 text-amber-800 border-amber-200",
  complete: "bg-green-100 text-green-800 border-green-200",
  failed: "bg-red-100 text-red-800 border-red-200",
  stopped: "bg-zinc-100 text-zinc-800 border-zinc-200",
  pending: "bg-zinc-100 text-zinc-600 border-zinc-200",
}

export default function RunDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const [run, setRun] = useState<RunDetail | null>(null)
  const [liveEpochs, setLiveEpochs] = useState<EpochData[]>([])
  const [currentStage, setCurrentStage] = useState<string>("")
  const [totalEpochs, setTotalEpochs] = useState(0)
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    getRun(id).then((data) => {
      setRun(data)
      if (data.metrics?.epochs) {
        setLiveEpochs(data.metrics.epochs)
      }
      if (data.config?.stages) {
        setTotalEpochs(data.config.stages.reduce((s: number, st: { epochs: number }) => s + st.epochs, 0))
      }
    })
  }, [id])

  useEffect(() => {
    if (!run || run.status !== "running") return

    const cleanup = streamRun(id, (event: SSEEvent) => {
      if (event.type === "epoch") {
        const ep = event as unknown as EpochData
        setLiveEpochs((prev) => [...prev, ep])
        setCurrentStage(ep.stage || "")
        if (ep.elapsed_s) setElapsed(ep.elapsed_s)
      } else if (event.type === "started") {
        setTotalEpochs((event.total_epochs as number) || 0)
      } else if (event.type === "complete") {
        getRun(id).then(setRun)
      } else if (event.type === "error") {
        getRun(id).then(setRun)
      }
    })

    return cleanup
  }, [id, run?.status])

  const handleStop = useCallback(async () => {
    await stopRun(id)
    const data = await getRun(id)
    setRun(data)
  }, [id])

  if (!run) return <div className="p-6 text-sm text-muted-foreground">Loading...</div>

  const progress = totalEpochs > 0 ? (liveEpochs.length / totalEpochs) * 100 : 0
  const lastEpoch = liveEpochs[liveEpochs.length - 1]

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">
            Run #{id.slice(-6)} — {run.name}
          </h2>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className={statusVariant[run.status] || ""}>
            {run.status.toUpperCase()}
          </Badge>
          {run.status === "running" && (
            <Button variant="destructive" size="sm" onClick={handleStop}>
              Stop
            </Button>
          )}
        </div>
      </div>

      {/* Stats bar */}
      <div className="flex gap-4">
        {[
          { label: "PROGRESS", value: `${liveEpochs.length} / ${totalEpochs}` },
          { label: "TRAIN LOSS", value: lastEpoch?.train_loss?.toFixed(6) || "—" },
          { label: "TEST LOSS", value: lastEpoch?.test_loss?.toFixed(6) || "—" },
          { label: "STAGE", value: currentStage || "—" },
          { label: "ELAPSED", value: elapsed > 0 ? `${Math.floor(elapsed / 60)}m ${Math.floor(elapsed % 60)}s` : "—" },
        ].map(({ label, value }) => (
          <div key={label} className="border px-4 py-2">
            <div className="text-[10px] text-muted-foreground">{label}</div>
            <div className="text-sm font-semibold font-mono">{value}</div>
          </div>
        ))}
      </div>

      {/* Progress bar */}
      {run.status === "running" && <Progress value={progress} className="h-2" />}

      {/* Loss chart */}
      {liveEpochs.length > 0 && <LossChart epochs={liveEpochs} />}

      {/* Epoch table */}
      {liveEpochs.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold mb-2">Epoch Breakdown</h3>
          <EpochTable epochs={liveEpochs} />
        </div>
      )}

      {/* Results gallery */}
      {run.files && run.files.length > 0 && (
        <ResultsGallery runId={id} files={run.files} />
      )}
    </div>
  )
}
