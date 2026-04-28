// lib/api.ts
import type { RunConfig, RunSummary, RunDetail, SSEEvent, DataConfig } from "./types"

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

export async function retryRun(id: string): Promise<{ run_id: string }> {
  const res = await fetch(`${API_BASE}/runs/${id}/retry`, { method: "POST" })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function runDemos(id: string): Promise<{ demo: string[] | null; demo_chromatogram: string[] | null; errors: { source: string; message: string }[] }> {
  const res = await fetch(`${API_BASE}/runs/${id}/demos`, { method: "POST" })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function softDeleteRun(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/runs/${id}/delete`, { method: "POST" })
  if (!res.ok) throw new Error(await res.text())
}

export async function restoreRun(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/runs/${id}/restore`, { method: "POST" })
  if (!res.ok) throw new Error(await res.text())
}

export async function permanentlyDeleteRun(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/runs/${id}`, { method: "DELETE" })
  if (!res.ok) throw new Error(await res.text())
}

export async function listDeletedRuns(): Promise<RunSummary[]> {
  const res = await fetch(`${API_BASE}/runs?deleted=true`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export function getFileUrl(runId: string, filePath: string): string {
  return `${API_BASE}/runs/${runId}/files/${filePath}`
}

export interface GeneratePreviewConfig {
  n_signals?: number
  N?: number
  seed?: number | null
  baseline?: {
    smooth_sigma?: number
    sine_amp?: number
    sine_freq_min?: number
    sine_freq_max?: number
    baseline_amp_min?: number
    baseline_amp_max?: number
  }
  peak_layers?: {
    num_peaks_min?: number
    num_peaks_max?: number
    amplitude_min?: number
    amplitude_max?: number
    rise_width_min?: number
    rise_width_max?: number
    decay_width_min?: number
    decay_width_max?: number
    plateau_width_min?: number
    plateau_width_max?: number
    peak_shape_mode?: string
  }[]
  noise?: {
    noise_level?: number
  }
}

export interface SignalData {
  y: number[]
  x_true: number[]
  f_true: number[]
  noise: number[]
}

export async function generatePreview(
  config: GeneratePreviewConfig
): Promise<{ signals: SignalData[] }> {
  const res = await fetch(`${API_BASE}/generate-preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export interface BeadsConfig {
  d: number
  fc: number
  r: number
  lam0: number
  lam1: number
  lam2: number
  Nit: number
  label: string
}

export interface BeadsResult {
  label: string
  x_pred?: number[]
  f_pred?: number[]
  peak_mse?: number
  baseline_mse?: number
  error?: string
}

export async function runBeadsTest(
  signal: number[], x_true: number[], f_true: number[], configs: BeadsConfig[]
): Promise<{ results: BeadsResult[] }> {
  const res = await fetch(`${API_BASE}/beads-test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ signal, x_true, f_true, configs }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export interface RealDataRecord {
  name: string
  n_channels: number
  fs: number
  sig_len: number
}

export interface RealDataInferResult {
  y: number[]
  x_pred: number[]
  f_pred: number[]
  N: number
  fs: number
  channel: string
  model_type: string
}

export async function listRealDataRecords(): Promise<{ records: RealDataRecord[] }> {
  const res = await fetch(`${API_BASE}/real-data/records`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function inferRealData(
  run_id: string, record: string, channel: number, start: number, length: number
): Promise<RealDataInferResult> {
  const res = await fetch(`${API_BASE}/real-data/infer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id, record, channel, start, length }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getDataConfig(): Promise<DataConfig> {
  const res = await fetch(`${API_BASE}/data-config`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function saveDataConfig(config: DataConfig): Promise<void> {
  const res = await fetch(`${API_BASE}/data-config`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  })
  if (!res.ok) throw new Error(await res.text())
}
