"use client"

import Link from "next/link"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { softDeleteRun } from "@/lib/api"
import type { RunSummary } from "@/lib/types"

const statusVariant: Record<string, string> = {
  running: "bg-amber-100 text-amber-800 border-amber-200",
  complete: "bg-green-100 text-green-800 border-green-200",
  failed: "bg-red-100 text-red-800 border-red-200",
  stopped: "bg-zinc-100 text-zinc-800 border-zinc-200",
  pending: "bg-zinc-100 text-zinc-600 border-zinc-200",
}

interface RunCardProps {
  run: RunSummary
  onDeleted?: () => void
}

export function RunCard({ run, onDeleted }: RunCardProps) {
  const progress = run.total_epochs > 0 ? (run.epoch_count / run.total_epochs) * 100 : 0

  const handleDelete = async (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    try {
      await softDeleteRun(run.id)
      onDeleted?.()
    } catch {
      // ignore
    }
  }

  const isActive = run.status === "running" || run.status === "pending"

  return (
    <Link
      href={`/runs/${run.id}`}
      className={`block border p-4 hover:bg-zinc-50 transition-all duration-200 ${
        isActive ? "border-l-4 border-l-amber-400 bg-amber-50/30" : ""
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {isActive && (
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-amber-500" />
            </span>
          )}
          <span className="font-semibold text-sm">Run #{run.id.slice(-6)}</span>
          <span className="text-[10px] font-mono px-1.5 py-0.5 border bg-zinc-50 text-muted-foreground">
            {run.model_type === "lbeads_fast" ? "FAST" : "STD"}
          </span>
          <span className="text-muted-foreground text-sm">{run.name}</span>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className={statusVariant[run.status] || ""}>
            {run.status.toUpperCase()}
          </Badge>
          {!isActive && (
            <button
              onClick={handleDelete}
              className="p-1 text-muted-foreground hover:text-red-600 transition-colors"
              title="Move to recycle bin"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            </button>
          )}
        </div>
      </div>
      {run.status === "running" && (
        <Progress value={progress} className="h-1.5 mb-2" />
      )}
      <div className="flex gap-6 text-xs text-muted-foreground">
        <span>Epochs <span className="text-foreground font-medium">{run.epoch_count}{run.total_epochs > 0 ? ` / ${run.total_epochs}` : ""}</span></span>
        {run.summary?.test_mse != null && (
          <span>Test MSE <span className="text-foreground font-medium">{run.summary.test_mse.toFixed(5)}</span></span>
        )}
        {run.summary?.test_mae != null && (
          <span>MAE <span className="text-foreground font-medium">{run.summary.test_mae.toFixed(4)}</span></span>
        )}
        <span className="ml-auto">
          {new Date(run.created_at * 1000).toLocaleString()}
        </span>
      </div>
    </Link>
  )
}
