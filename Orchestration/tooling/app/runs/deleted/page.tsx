"use client"

import { useEffect, useState, useCallback } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { listDeletedRuns, restoreRun, permanentlyDeleteRun } from "@/lib/api"
import type { RunSummary } from "@/lib/types"

export default function RecycleBinPage() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [loading, setLoading] = useState(true)

  const loadRuns = useCallback(async () => {
    try {
      const data = await listDeletedRuns()
      setRuns(data)
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadRuns()
  }, [loadRuns])

  const handleRestore = async (id: string) => {
    await restoreRun(id)
    loadRuns()
  }

  const handlePermanentDelete = async (id: string) => {
    if (!confirm("Permanently delete this run? This cannot be undone.")) return
    await permanentlyDeleteRun(id)
    loadRuns()
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <h2 className="text-lg font-semibold">Recycle Bin</h2>
        <p className="text-sm text-muted-foreground">
          {runs.length} deleted run{runs.length !== 1 ? "s" : ""}
        </p>
      </div>
      {loading ? (
        <p className="text-sm text-muted-foreground">Loading...</p>
      ) : runs.length === 0 ? (
        <p className="text-sm text-muted-foreground">Recycle bin is empty.</p>
      ) : (
        <div className="space-y-2">
          {runs.map((run) => (
            <div key={run.id} className="border p-4 flex items-center justify-between">
              <Link href={`/runs/${run.id}`} className="flex-1 hover:underline">
                <span className="font-semibold text-sm">Run #{run.id.slice(-6)}</span>
                <span className="text-muted-foreground text-sm ml-2">{run.name}</span>
                <Badge variant="outline" className="ml-3 bg-red-50 text-red-600 border-red-200">
                  DELETED
                </Badge>
              </Link>
              <div className="flex items-center gap-2 ml-4">
                <span className="text-xs text-muted-foreground">
                  {new Date(run.created_at * 1000).toLocaleString()}
                </span>
                <Button variant="outline" size="sm" onClick={() => handleRestore(run.id)}>
                  Restore
                </Button>
                <Button variant="destructive" size="sm" onClick={() => handlePermanentDelete(run.id)}>
                  Delete Forever
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
