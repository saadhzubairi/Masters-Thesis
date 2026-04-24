"use client"

import { useEffect, useState, useCallback, useRef, use } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useRouter } from "next/navigation"
import { toast } from "sonner"
import { getRun, stopRun, retryRun, runDemos, softDeleteRun, restoreRun, permanentlyDeleteRun, streamRun } from "@/lib/api"
import type { RunConfig } from "@/lib/types"
import { EpochTable } from "@/app/components/epoch-table"
import { LossChart } from "@/app/components/loss-chart"
import { ResultsGallery } from "@/app/components/results-gallery"
import type { RunDetail, SSEEvent } from "@/lib/types"

const statusVariant: Record<string, string> = {
  running: "bg-amber-100 text-amber-800 border-amber-200",
  complete: "bg-green-100 text-green-800 border-green-200",
  failed: "bg-red-100 text-red-800 border-red-200",
  stopped: "bg-zinc-100 text-zinc-800 border-zinc-200",
  pending: "bg-zinc-100 text-zinc-600 border-zinc-200",
  deleted: "bg-red-50 text-red-600 border-red-200",
}

function formatElapsed(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

/** Live-ticking elapsed timer that resets to the latest epoch's elapsed_s on each data update. */
function useRollingElapsed(run: RunDetail | null): number {
  const [display, setDisplay] = useState(0)
  const baseRef = useRef(0)
  const baseTimeRef = useRef(0)

  // Sync to latest epoch data
  useEffect(() => {
    if (!run) return
    const epochs = run.metrics?.epochs || []
    const last = epochs[epochs.length - 1]
    const knownElapsed = last?.elapsed_s || 0
    baseRef.current = knownElapsed
    baseTimeRef.current = Date.now() / 1000
    setDisplay(knownElapsed)
  }, [run?.metrics?.epochs?.length])

  // Tick every second while running
  useEffect(() => {
    if (!run || (run.status !== "running" && run.status !== "pending")) return
    const interval = setInterval(() => {
      const now = Date.now() / 1000
      const delta = now - baseTimeRef.current
      setDisplay(baseRef.current + delta)
    }, 1000)
    return () => clearInterval(interval)
  }, [run?.status])

  return display
}

/** Pulsing dot for active status */
function PulsingDot() {
  return (
    <span className="relative flex h-2.5 w-2.5">
      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-75" />
      <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-amber-500" />
    </span>
  )
}

/** Animated progress bar with stripes */
function AnimatedProgress({ value }: { value: number }) {
  return (
    <div className="relative h-2.5 w-full overflow-hidden bg-zinc-100 rounded-sm">
      <div
        className="h-full bg-amber-500 transition-all duration-700 ease-out relative overflow-hidden"
        style={{ width: `${Math.min(value, 100)}%` }}
      >
        <div
          className="absolute inset-0 opacity-30"
          style={{
            backgroundImage:
              "linear-gradient(45deg, rgba(255,255,255,.3) 25%, transparent 25%, transparent 50%, rgba(255,255,255,.3) 50%, rgba(255,255,255,.3) 75%, transparent 75%)",
            backgroundSize: "16px 16px",
            animation: "stripe-scroll 0.6s linear infinite",
          }}
        />
      </div>
    </div>
  )
}

/** Stat card with value animation on change */
function StatCard({
  label,
  value,
  highlight,
}: {
  label: string
  value: string
  highlight?: boolean
}) {
  const [flash, setFlash] = useState(false)
  const prevRef = useRef(value)

  useEffect(() => {
    if (prevRef.current !== value && value !== "—") {
      setFlash(true)
      const t = setTimeout(() => setFlash(false), 600)
      prevRef.current = value
      return () => clearTimeout(t)
    }
  }, [value])

  return (
    <div
      className={`border px-4 py-2 transition-colors duration-300 ${
        highlight ? "border-amber-200 bg-amber-50/50" : ""
      } ${flash ? "bg-amber-50" : ""}`}
    >
      <div className="text-[10px] text-muted-foreground">{label}</div>
      <div
        className={`text-sm font-semibold font-mono transition-colors duration-300 ${
          flash ? "text-amber-700" : ""
        }`}
      >
        {value}
      </div>
    </div>
  )
}

export default function RunDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const router = useRouter()
  const [run, setRun] = useState<RunDetail | null>(null)
  const [retryOpen, setRetryOpen] = useState(false)
  const [demosRunning, setDemosRunning] = useState(false)
  const [batchProgress, setBatchProgress] = useState<{ batch: number; total: number; loss: number; gradNorm: number } | null>(null)
  const rollingElapsed = useRollingElapsed(run)

  const loadRun = useCallback(async () => {
    try {
      const data = await getRun(id)
      setRun(data)
    } catch {
      // ignore fetch errors
    }
  }, [id])

  useEffect(() => {
    loadRun()
  }, [loadRun])

  // Poll while running
  useEffect(() => {
    if (!run || (run.status !== "running" && run.status !== "pending")) return
    const interval = setInterval(loadRun, 2000)
    return () => clearInterval(interval)
  }, [run?.status, loadRun])

  // SSE stream for real-time toast notifications
  useEffect(() => {
    if (!run || (run.status !== "running" && run.status !== "pending")) return
    const cleanup = streamRun(id, (event: SSEEvent) => {
      if (event.type === "batch_progress") {
        setBatchProgress({
          batch: event.batch as number,
          total: event.total_batches as number,
          loss: event.batch_loss as number,
          gradNorm: event.grad_norm as number,
        })
      } else if (event.type === "epoch") {
        setBatchProgress(null)
      } else if (event.type === "demo_started") {
        toast.info(`Running ${event.demo as string}...`, { id: `demo-${event.demo}`, duration: Infinity })
      } else if (event.type === "demo_done") {
        toast.success(`${event.demo as string} completed`, { id: `demo-${event.demo}` })
      } else if (event.type === "demo_error") {
        toast.error(`${event.demo as string} failed: ${event.error as string}`, { id: `demo-${event.demo}` })
      } else if (event.type === "training_done") {
        setBatchProgress(null)
        toast.success("Training complete — running demos...")
      } else if (event.type === "complete") {
        setBatchProgress(null)
        toast.success("Run finished!")
        loadRun()
      }
    })
    return cleanup
  }, [id, run?.status, loadRun])

  const handleStop = useCallback(async () => {
    await stopRun(id)
    loadRun()
  }, [id, loadRun])

  const handleDelete = useCallback(async () => {
    await softDeleteRun(id)
    loadRun()
  }, [id, loadRun])

  const handleRestore = useCallback(async () => {
    await restoreRun(id)
    loadRun()
  }, [id, loadRun])

  const handlePermanentDelete = useCallback(async () => {
    if (!confirm("Permanently delete this run? This cannot be undone.")) return
    await permanentlyDeleteRun(id)
    window.location.href = "/runs/deleted"
  }, [id])

  const handleRetry = useCallback(async () => {
    const { run_id } = await retryRun(id)
    router.push(`/runs/${run_id}`)
  }, [id, router])

  const handleRunDemos = useCallback(async () => {
    setDemosRunning(true)
    toast.info("Running demos...", { id: "demos-running", duration: Infinity })
    try {
      const result = await runDemos(id)
      if (result.errors.length > 0) {
        toast.error(`Demos finished with ${result.errors.length} error(s)`, { id: "demos-running" })
      } else {
        toast.success("Demos completed!", { id: "demos-running" })
      }
      loadRun()
    } catch (err) {
      toast.error(`Demo failed: ${err}`, { id: "demos-running" })
    } finally {
      setDemosRunning(false)
    }
  }, [id, loadRun])

  if (!run) {
    return (
      <div className="p-6 flex items-center gap-2 text-sm text-muted-foreground">
        <span className="inline-block h-4 w-4 border-2 border-zinc-300 border-t-zinc-600 rounded-full animate-spin" />
        Loading run...
      </div>
    )
  }

  const isActive = run.status === "running" || run.status === "pending"
  const epochs = run.metrics?.epochs || []
  const totalEpochs = run.config?.stages
    ? run.config.stages.reduce((s: number, st: { epochs: number }) => s + st.epochs, 0)
    : 0
  const progress = totalEpochs > 0 ? (epochs.length / totalEpochs) * 100 : 0
  const lastEpoch = epochs[epochs.length - 1]
  const currentStage = lastEpoch?.stage || "—"
  const staticElapsed = lastEpoch?.elapsed_s || 0
  const displayElapsed = isActive ? rollingElapsed : staticElapsed

  // Derived timing stats
  const epochCount = epochs.length
  const avgPerEpoch = epochCount > 0 ? displayElapsed / epochCount : 0
  const remainingEpochs = totalEpochs - epochCount
  const eta = epochCount > 0 && remainingEpochs > 0 ? avgPerEpoch * remainingEpochs : 0

  return (
    <div className="p-6 space-y-6">
      {/* Back link */}
      <Link href="/" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors">
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m15 18-6-6 6-6"/></svg>
        Dashboard
      </Link>

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {isActive && <PulsingDot />}
          <h2 className="text-lg font-semibold">
            Run #{id.slice(-6)} — {run.name}
          </h2>
          <span className="text-[10px] font-mono px-1.5 py-0.5 border bg-zinc-50 text-muted-foreground">
            {run.config?.model_type === "lbeads_fast" ? "LBEADS-NET Fast" : run.config?.model_type === "lbeads_v5" ? "LBEADS-NET v5" : "LBEADS-NET"}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            className="text-muted-foreground"
            onClick={() => {
              const lastEp = epochs[epochs.length - 1]
              const payload = {
                id: run.id,
                name: run.name,
                status: run.status,
                config: run.config,
                summary: {
                  total_epochs: totalEpochs,
                  completed_epochs: epochCount,
                  train_loss: lastEp?.train_loss ?? null,
                  test_loss: lastEp?.test_loss ?? null,
                  stage: lastEp?.stage ?? null,
                  elapsed_s: lastEp?.elapsed_s ?? null,
                  elapsed_formatted: displayElapsed > 0 ? formatElapsed(displayElapsed) : null,
                  avg_per_epoch_s: avgPerEpoch > 0 ? +avgPerEpoch.toFixed(2) : null,
                  epochs_per_min: avgPerEpoch > 0 ? +(60 / avgPerEpoch).toFixed(2) : null,
                  loss_components: lastEp?.components ?? null,
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
              navigator.clipboard.writeText(JSON.stringify(payload, null, 2))
              toast.success("Run data copied to clipboard")
            }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            Copy
          </Button>
          <Badge
            variant="outline"
            className={`${statusVariant[run.status] || ""} ${
              run.status === "running" ? "animate-pulse" : ""
            }`}
          >
            {run.status.toUpperCase()}
          </Badge>
          {run.status === "running" && (
            <Button variant="destructive" size="sm" onClick={handleStop}>
              Stop
            </Button>
          )}
          {run.status === "deleted" ? (
            <>
              <Button variant="outline" size="sm" onClick={handleRestore}>
                Restore
              </Button>
              <Button variant="destructive" size="sm" onClick={handlePermanentDelete}>
                Delete Forever
              </Button>
            </>
          ) : run.status !== "running" && run.status !== "pending" ? (
            <>
              <Button
                variant="outline"
                size="sm"
                disabled={demosRunning}
                onClick={handleRunDemos}
              >
                {demosRunning ? (
                  <>
                    <span className="inline-block h-3 w-3 border-2 border-zinc-300 border-t-zinc-600 rounded-full animate-spin mr-1.5" />
                    Running...
                  </>
                ) : (
                  "Run Demos"
                )}
              </Button>
              <div className="relative">
                <Button variant="outline" size="sm" onClick={() => setRetryOpen(!retryOpen)}>
                  Retry
                  <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="ml-1"><path d="m6 9 6 6 6-6"/></svg>
                </Button>
                {retryOpen && (
                  <>
                    <div className="fixed inset-0 z-40" onClick={() => setRetryOpen(false)} />
                    <div className="absolute right-0 top-full mt-1 z-50 border bg-white shadow-lg min-w-[200px]">
                      <button
                        className="w-full text-left px-3 py-2 text-xs hover:bg-zinc-50 transition-colors"
                        onClick={async () => {
                          setRetryOpen(false)
                          const { run_id } = await retryRun(id)
                          router.push(`/runs/${run_id}`)
                        }}
                      >
                        <span className="font-medium">Retry same config</span>
                        <span className="block text-muted-foreground mt-0.5">Re-run with identical parameters</span>
                      </button>
                      <button
                        className="w-full text-left px-3 py-2 text-xs hover:bg-zinc-50 transition-colors border-t"
                        onClick={() => {
                          setRetryOpen(false)
                          const cfg = run.config as RunConfig | undefined
                          if (cfg) {
                            sessionStorage.setItem("clone_config", JSON.stringify(cfg))
                          }
                          router.push("/runs/new?clone_from_session=1")
                        }}
                      >
                        <span className="font-medium">Retry with different config</span>
                        <span className="block text-muted-foreground mt-0.5">Edit parameters before starting</span>
                      </button>
                    </div>
                  </>
                )}
              </div>
              <Button variant="outline" size="sm" onClick={handleDelete} className="text-muted-foreground hover:text-red-600">
                Delete
              </Button>
            </>
          ) : null}
        </div>
      </div>

      {/* Stats bar */}
      <div className="grid grid-cols-4 gap-3">
        <StatCard label="PROGRESS" value={`${epochCount} / ${totalEpochs}`} highlight={isActive} />
        <StatCard label="TRAIN LOSS" value={lastEpoch?.train_loss != null ? lastEpoch.train_loss.toExponential(4) : "—"} />
        <StatCard label="TEST LOSS" value={lastEpoch?.test_loss != null ? lastEpoch.test_loss.toExponential(4) : "—"} />
        <StatCard label="STAGE" value={currentStage} />
        <StatCard
          label="ELAPSED"
          value={displayElapsed > 0 ? formatElapsed(displayElapsed) : "—"}
          highlight={isActive}
        />
        <StatCard
          label="AVG / EPOCH"
          value={avgPerEpoch > 0 ? formatElapsed(avgPerEpoch) : "—"}
        />
        <StatCard
          label="ETA"
          value={eta > 0 ? formatElapsed(eta) : epochCount >= totalEpochs && totalEpochs > 0 ? "Done" : "—"}
          highlight={isActive && eta > 0}
        />
        <StatCard
          label="EPOCHS / MIN"
          value={avgPerEpoch > 0 ? (60 / avgPerEpoch).toFixed(1) : "—"}
        />
      </div>

      {/* Animated progress bar */}
      {isActive && <AnimatedProgress value={progress} />}

      {/* Batch progress bar — resets each epoch */}
      {isActive && batchProgress && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[10px] text-muted-foreground font-mono">
            <span>batch {batchProgress.batch}/{batchProgress.total}</span>
            <span>loss {batchProgress.loss.toFixed(5)} · grad {batchProgress.gradNorm.toFixed(4)}</span>
          </div>
          <div className="relative h-1.5 w-full overflow-hidden bg-zinc-100 rounded-sm">
            <div
              className="h-full bg-zinc-400 transition-all duration-300 ease-out"
              style={{ width: `${(batchProgress.batch / batchProgress.total) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Waiting indicator when no epochs yet */}
      {isActive && epochs.length === 0 && (
        <div className="flex items-center gap-3 py-8 justify-center text-muted-foreground">
          <span className="inline-block h-5 w-5 border-2 border-zinc-300 border-t-amber-500 rounded-full animate-spin" />
          <span className="text-sm">Waiting for first epoch...</span>
        </div>
      )}

      {/* Loss chart */}
      {epochs.length > 0 && <LossChart epochs={epochs} />}

      {/* Epoch table */}
      {epochs.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold mb-2">Epoch Breakdown</h3>
          <EpochTable epochs={epochs} />
        </div>
      )}

      {/* Process logs (stderr) — collapsible */}
      {run.logs && run.logs.length > 0 && (
        <details className="group">
          <summary className="flex items-center justify-between cursor-pointer list-none">
            <div className="flex items-center gap-2">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="transition-transform duration-200 group-open:rotate-90"
              >
                <path d="m9 18 6-6-6-6" />
              </svg>
              <h3 className="text-sm font-semibold">Logs</h3>
              <span className="text-xs text-muted-foreground">({run.logs.length} lines)</span>
            </div>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                className="text-xs h-7"
                onClick={(e) => {
                  e.preventDefault()
                  const mainLogs = run.logs!.filter(l => !l.startsWith("## "))
                  const prompt = `My model config: ${JSON.stringify(run.config, null, 2)}\n\nLogs:\n${mainLogs.join("\n")}`
                  navigator.clipboard.writeText(prompt)
                  toast.success("Prompt copied to clipboard")
                }}
              >
                Copy Prompt
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="text-xs h-7"
                onClick={(e) => {
                  e.preventDefault()
                  navigator.clipboard.writeText(run.logs!.filter(l => !l.startsWith("## ")).join("\n"))
                  toast.success("Logs copied to clipboard")
                }}
              >
                Copy
              </Button>
            </div>
          </summary>
          <pre className="mt-2 border border-zinc-200 bg-zinc-950 text-zinc-200 text-xs p-4 overflow-x-auto max-h-80 overflow-y-auto whitespace-pre-wrap">
            {run.logs.map((line, i) => (
              <span key={i} className={line.startsWith("## ") ? "text-zinc-600" : ""}>
                {line}
                {"\n"}
              </span>
            ))}
          </pre>
        </details>
      )}

      {/* Demo errors */}
      {run.metrics?.errors && run.metrics.errors.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold">Demo Errors</h3>
          {run.metrics.errors.map((err, i) => (
            <div key={i} className="border border-red-200 bg-red-50 px-3 py-2 text-sm">
              <span className="font-medium text-red-800">{err.source}:</span>{" "}
              <span className="text-red-700">{err.message}</span>
            </div>
          ))}
        </div>
      )}

      {/* Results gallery */}
      {run.files && run.files.length > 0 && (
        <ResultsGallery runId={id} files={run.files} />
      )}
    </div>
  )
}
