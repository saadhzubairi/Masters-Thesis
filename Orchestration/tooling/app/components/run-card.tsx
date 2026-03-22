"use client"

import Link from "next/link"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
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
  totalEpochs?: number
}

export function RunCard({ run, totalEpochs }: RunCardProps) {
  const progress = totalEpochs ? (run.epoch_count / totalEpochs) * 100 : 0

  return (
    <Link href={`/runs/${run.id}`} className="block border p-4 hover:bg-zinc-50 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <div>
          <span className="font-semibold text-sm">Run #{run.id.slice(-6)}</span>
          <span className="text-muted-foreground text-sm ml-2">{run.name}</span>
        </div>
        <Badge variant="outline" className={statusVariant[run.status] || ""}>
          {run.status.toUpperCase()}
        </Badge>
      </div>
      {run.status === "running" && (
        <Progress value={progress} className="h-1.5 mb-2" />
      )}
      <div className="flex gap-6 text-xs text-muted-foreground">
        <span>Epochs <span className="text-foreground font-medium">{run.epoch_count}</span></span>
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
