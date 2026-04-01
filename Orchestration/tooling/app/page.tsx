"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { RunCard } from "./components/run-card"
import { listRuns } from "@/lib/api"
import type { RunSummary } from "@/lib/types"

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
        <Button asChild>
          <Link href="/runs/new">+ New Run</Link>
        </Button>
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
