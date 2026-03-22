// lib/api.ts
import type { RunConfig, RunSummary, RunDetail, SSEEvent } from "./types"

const API_BASE = "/api"

export async function createRun(
  config: RunConfig
): Promise<{ run_id: string }> {
  const res = await fetch(`${API_BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function listRuns(): Promise<RunSummary[]> {
  const res = await fetch(`${API_BASE}/runs`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getRun(id: string): Promise<RunDetail> {
  const res = await fetch(`${API_BASE}/runs/${id}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function stopRun(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/runs/${id}/stop`, { method: "POST" })
  if (!res.ok) throw new Error(await res.text())
}

export function streamRun(
  id: string,
  onEvent: (event: SSEEvent) => void
): () => void {
  const eventSource = new EventSource(`${API_BASE}/runs/${id}/stream`)
  eventSource.onmessage = (e) => {
    const event: SSEEvent = JSON.parse(e.data)
    onEvent(event)
  }
  eventSource.onerror = () => {
    eventSource.close()
  }
  return () => eventSource.close()
}

export function getFileUrl(runId: string, filePath: string): string {
  return `${API_BASE}/runs/${runId}/files/${filePath}`
}
