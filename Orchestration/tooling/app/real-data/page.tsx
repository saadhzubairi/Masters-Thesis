"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import { listRuns, listRealDataRecords, inferRealData } from "@/lib/api"
import type { RealDataRecord, RealDataInferResult } from "@/lib/api"
import type { RunSummary } from "@/lib/types"

function drawChart(
  el: HTMLCanvasElement,
  traces: { arr: number[]; color: string; width: number; alpha: number }[],
  title: string,
  legend: { color: string; label: string }[],
  xLen: number,
  fs: number,
  startSample: number,
) {
  const ctx = el.getContext("2d")
  if (!ctx) return
  const dpr = window.devicePixelRatio || 1
  const w = el.clientWidth
  const h = el.clientHeight
  el.width = w * dpr
  el.height = h * dpr
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, w, h)

  const allVals = traces.flatMap((t) => t.arr)
  const maxVal = Math.max(...allVals) * 1.05 || 1
  const minVal = Math.min(...allVals) * 1.05
  const padL = 50, padR = 10, padT = 24, padB = 22
  const plotW = w - padL - padR
  const plotH = h - padT - padB
  const len = traces[0]?.arr.length || 0

  function xPos(i: number) { return padL + (i / Math.max(len - 1, 1)) * plotW }
  function yPos(v: number) { return padT + (1 - (v - minVal) / (maxVal - minVal || 1)) * plotH }

  // Grid
  ctx.strokeStyle = "#f0f0f0"
  ctx.lineWidth = 1
  for (let i = 0; i <= 4; i++) {
    const yy = padT + (i / 4) * plotH
    ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke()
  }
  // X-axis grid
  const nXTicks = 8
  for (let i = 0; i <= nXTicks; i++) {
    const xx = padL + (i / nXTicks) * plotW
    ctx.beginPath(); ctx.moveTo(xx, padT); ctx.lineTo(xx, padT + plotH); ctx.stroke()
  }

  // Y labels
  ctx.fillStyle = "#a1a1aa"
  ctx.font = "10px monospace"
  ctx.textAlign = "right"
  for (let i = 0; i <= 4; i++) {
    const val = maxVal - (i / 4) * (maxVal - minVal)
    ctx.fillText(val.toFixed(1), padL - 4, padT + (i / 4) * plotH + 4)
  }

  // X labels (sample index or time)
  ctx.textAlign = "center"
  for (let i = 0; i <= nXTicks; i++) {
    const sampleIdx = startSample + Math.round((i / nXTicks) * xLen)
    const timeSec = sampleIdx / fs
    ctx.fillText(timeSec.toFixed(2) + "s", padL + (i / nXTicks) * plotW, padT + plotH + 14)
  }
  ctx.textAlign = "left"

  // Title
  ctx.fillStyle = "#18181b"
  ctx.font = "bold 12px sans-serif"
  ctx.fillText(title, padL, 16)

  // Legend (top right)
  let legendX = w - padR
  ctx.font = "10px sans-serif"
  for (let i = legend.length - 1; i >= 0; i--) {
    const l = legend[i]
    const textW = ctx.measureText(l.label).width
    legendX -= textW + 16
    ctx.fillStyle = l.color
    ctx.fillRect(legendX, 8, 10, 3)
    ctx.fillStyle = "#71717a"
    ctx.fillText(l.label, legendX + 13, 14)
  }

  // Traces
  for (const t of traces) {
    ctx.strokeStyle = t.color
    ctx.lineWidth = t.width
    ctx.globalAlpha = t.alpha
    ctx.beginPath()
    t.arr.forEach((v, i) => {
      i === 0 ? ctx.moveTo(xPos(i), yPos(v)) : ctx.lineTo(xPos(i), yPos(v))
    })
    ctx.stroke()
  }
  ctx.globalAlpha = 1
}

function RealDataCharts({ result, startSample }: { result: RealDataInferResult; startSample: number }) {
  return (
    <div className="space-y-3">
      <div className="border bg-white">
        <canvas
          className="w-full h-48"
          ref={(el) => {
            if (!el || result.y.length === 0) return
            drawChart(
              el,
              [{ arr: result.y, color: "#71717a", width: 1.5, alpha: 1 }],
              "Original Signal (y)",
              [{ color: "#71717a", label: "Observed" }],
              result.N, result.fs, startSample,
            )
          }}
        />
      </div>
      <div className="border bg-white">
        <canvas
          className="w-full h-48"
          ref={(el) => {
            if (!el || result.y.length === 0) return
            drawChart(
              el,
              [
                { arr: result.y, color: "#71717a", width: 1, alpha: 0.35 },
                { arr: result.f_pred, color: "#2563eb", width: 1.5, alpha: 1 },
                { arr: result.x_pred, color: "#dc2626", width: 1.5, alpha: 1 },
              ],
              `Decomposition (${result.model_type})`,
              [
                { color: "#71717a", label: "y" },
                { color: "#dc2626", label: "x_pred (peaks)" },
                { color: "#2563eb", label: "f_pred (baseline)" },
              ],
              result.N, result.fs, startSample,
            )
          }}
        />
      </div>
      <div className="border bg-white">
        <canvas
          className="w-full h-48"
          ref={(el) => {
            if (!el || result.y.length === 0) return
            const reconstructed = result.x_pred.map((v, i) => v + result.f_pred[i])
            drawChart(
              el,
              [
                { arr: result.y, color: "#71717a", width: 1, alpha: 0.35 },
                { arr: reconstructed, color: "#7c3aed", width: 1.5, alpha: 1 },
              ],
              "Reconstructed Signal (x_pred + f_pred)",
              [
                { color: "#71717a", label: "Original (y)" },
                { color: "#7c3aed", label: "x_pred + f_pred" },
              ],
              result.N, result.fs, startSample,
            )
          }}
        />
      </div>
      <p className="text-[10px] text-muted-foreground text-right">
        {result.channel} @ {result.fs}Hz | N={result.N} | {result.model_type}
      </p>
    </div>
  )
}

export default function RealDataPage() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [records, setRecords] = useState<RealDataRecord[]>([])
  const [selectedRun, setSelectedRun] = useState("")
  const [selectedRecord, setSelectedRecord] = useState("")
  const [channel, setChannel] = useState(0)
  const [start, setStart] = useState(0)
  const [length, setLength] = useState(1024)
  const [result, setResult] = useState<RealDataInferResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listRuns().then((r) => setRuns(r.filter((x) => x.status === "complete"))).catch(() => {})
    listRealDataRecords().then((r) => setRecords(r.records)).catch(() => {})
  }, [])

  const selectedRecordInfo = records.find((r) => r.name === selectedRecord)
  const maxStart = selectedRecordInfo ? Math.max(0, selectedRecordInfo.sig_len - length) : 0

  async function handleInfer() {
    if (!selectedRun || !selectedRecord) return
    setLoading(true)
    setError(null)
    try {
      const res = await inferRealData(selectedRun, selectedRecord, channel, start, length)
      setResult(res)
    } catch (err) {
      setError(String(err))
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-4">
      <div>
        <h1 className="text-lg font-bold">Real Data Test</h1>
        <p className="text-xs text-muted-foreground mt-1">
          Run a trained LBEADS-NET model on real ECG signals from the MIT-BIH Noise Stress Test Database.
        </p>
      </div>

      {/* Run selector */}
      <div className="border bg-zinc-50/50 p-4 space-y-3">
        <p className="text-sm font-semibold">Select Model</p>
        <Select value={selectedRun} onValueChange={setSelectedRun}>
          <SelectTrigger className="h-9 text-xs">
            <SelectValue placeholder="Select a completed run..." />
          </SelectTrigger>
          <SelectContent>
            {runs.map((r) => (
              <SelectItem key={r.id} value={r.id} className="text-xs">
                <span className="font-mono">#{r.id.slice(-6)}</span>
                {" — "}
                {r.name}
                <span className="text-muted-foreground ml-2">
                  ({r.model_type || "lbeads"})
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Record + segment config */}
      <div className="border bg-zinc-50/50 p-4 space-y-3">
        <p className="text-sm font-semibold">Select Signal</p>
        <div className="grid grid-cols-[1fr_auto_auto_auto_auto] gap-3 items-end">
          <div>
            <Label className="text-[10px]">Record</Label>
            <Select value={selectedRecord} onValueChange={setSelectedRecord}>
              <SelectTrigger className="mt-0.5 h-8 text-xs">
                <SelectValue placeholder="Select record..." />
              </SelectTrigger>
              <SelectContent>
                {records.map((r) => (
                  <SelectItem key={r.name} value={r.name} className="text-xs">
                    {r.name}
                    <span className="text-muted-foreground ml-2">
                      ({r.n_channels}ch, {r.fs}Hz, {(r.sig_len / r.fs).toFixed(0)}s)
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label className="text-[10px]">Channel</Label>
            <Select value={String(channel)} onValueChange={(v) => setChannel(parseInt(v))}>
              <SelectTrigger className="mt-0.5 h-8 text-xs w-20">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Array.from({ length: selectedRecordInfo?.n_channels || 2 }, (_, i) => (
                  <SelectItem key={i} value={String(i)} className="text-xs">Ch {i}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label className="text-[10px]">Start sample</Label>
            <Input
              type="number"
              value={start}
              onChange={(e) => setStart(Math.min(parseInt(e.target.value) || 0, maxStart))}
              className="mt-0.5 h-8 text-xs w-28"
              min={0}
              max={maxStart}
              step={length}
            />
          </div>
          <div>
            <Label className="text-[10px]">Length</Label>
            <Input
              type="number"
              value={length}
              onChange={(e) => setLength(parseInt(e.target.value) || 1024)}
              className="mt-0.5 h-8 text-xs w-24"
              min={256}
              step={256}
            />
          </div>
          <Button size="sm" onClick={handleInfer} disabled={loading || !selectedRun || !selectedRecord}>
            {loading ? (
              <>
                <span className="inline-block h-3 w-3 border-2 border-zinc-300 border-t-zinc-600 rounded-full animate-spin mr-1.5" />
                Running...
              </>
            ) : (
              "Run Inference"
            )}
          </Button>
        </div>
        {selectedRecordInfo && (
          <p className="text-[10px] text-muted-foreground">
            Record: {selectedRecordInfo.sig_len.toLocaleString()} samples
            ({(selectedRecordInfo.sig_len / selectedRecordInfo.fs).toFixed(1)}s at {selectedRecordInfo.fs}Hz)
            | Showing samples {start.toLocaleString()}–{Math.min(start + length, selectedRecordInfo.sig_len).toLocaleString()}
          </p>
        )}
      </div>

      {error && (
        <div className="border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{error}</div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-2">
          <p className="text-sm font-semibold">Inference Result</p>
          <RealDataCharts result={result} startSample={start} />
        </div>
      )}
    </div>
  )
}
