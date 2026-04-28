"use client"

import { useRef, useEffect, useCallback } from "react"

interface SignalChartProps {
  y: number[]
  x_true: number[]
  f_true: number[]
  noise: number[]
  label?: string
}

const TRACES = [
  { key: "y" as const, color: "#71717a", label: "Observed" },
  { key: "f_true" as const, color: "#2563eb", label: "Baseline" },
  { key: "x_true" as const, color: "#dc2626", label: "Peaks" },
  { key: "noise" as const, color: "#16a34a", label: "Noise" },
]

/** Draw a single trace into a sub-region of a canvas context */
function drawPane(
  ctx: CanvasRenderingContext2D,
  arr: number[],
  color: string,
  title: string,
  ox: number, oy: number, pw: number, ph: number,
) {
  const padL = 50, padR = 10, padT = 28, padB = 10
  const plotW = pw - padL - padR
  const plotH = ph - padT - padB
  const len = arr.length
  const maxVal = Math.max(...arr) * 1.05 || 1
  const minVal = Math.min(...arr) * 1.05

  function xPos(i: number) { return ox + padL + (i / Math.max(len - 1, 1)) * plotW }
  function yPos(v: number) { return oy + padT + (1 - (v - minVal) / (maxVal - minVal || 1)) * plotH }

  // Background
  ctx.fillStyle = "#ffffff"
  ctx.fillRect(ox, oy, pw, ph)

  // Border
  ctx.strokeStyle = "#e4e4e7"
  ctx.lineWidth = 1
  ctx.strokeRect(ox + padL, oy + padT, plotW, plotH)

  // Grid
  ctx.strokeStyle = "#f0f0f0"
  for (let i = 1; i < 4; i++) {
    const yy = oy + padT + (i / 4) * plotH
    ctx.beginPath(); ctx.moveTo(ox + padL, yy); ctx.lineTo(ox + padL + plotW, yy); ctx.stroke()
  }

  // Y-axis labels
  ctx.fillStyle = "#a1a1aa"
  ctx.font = "11px monospace"
  ctx.textAlign = "right"
  for (let i = 0; i <= 4; i++) {
    const val = maxVal - (i / 4) * (maxVal - minVal)
    ctx.fillText(val.toFixed(2), ox + padL - 4, oy + padT + (i / 4) * plotH + 4)
  }
  ctx.textAlign = "left"

  // Title
  ctx.fillStyle = "#18181b"
  ctx.font = "bold 13px sans-serif"
  ctx.fillText(title, ox + padL, oy + 18)

  // Trace
  ctx.strokeStyle = color
  ctx.lineWidth = 1.5
  ctx.beginPath()
  arr.forEach((v, i) => {
    i === 0 ? ctx.moveTo(xPos(i), yPos(v)) : ctx.lineTo(xPos(i), yPos(v))
  })
  ctx.stroke()
}

export function SignalChart({ y, x_true, f_true, noise, label }: SignalChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || y.length === 0) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, w, h)

    const data: Record<string, number[]> = { y, x_true, f_true, noise }
    const all = [...y, ...x_true, ...f_true, ...noise]
    const maxVal = Math.max(...all) * 1.05
    const minVal = Math.min(...all) * 1.05

    const padL = 40, padR = 6, padT = 6, padB = 4
    const plotW = w - padL - padR
    const plotH = h - padT - padB
    const len = y.length

    function xPos(i: number) { return padL + (i / Math.max(len - 1, 1)) * plotW }
    function yPos(v: number) { return padT + (1 - (v - minVal) / (maxVal - minVal || 1)) * plotH }

    // Grid lines
    ctx.strokeStyle = "#f0f0f0"
    ctx.lineWidth = 1
    for (let i = 0; i <= 3; i++) {
      const yy = padT + (i / 3) * plotH
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke()
    }

    // Y-axis labels
    ctx.fillStyle = "#a1a1aa"
    ctx.font = "9px monospace"
    for (let i = 0; i <= 3; i++) {
      const val = maxVal - (i / 3) * (maxVal - minVal)
      ctx.fillText(val.toFixed(2), 1, padT + (i / 3) * plotH + 3)
    }

    // Draw traces (noise first so it's behind, observed last)
    const order: (keyof typeof data)[] = ["noise", "f_true", "x_true", "y"]
    for (const key of order) {
      const trace = TRACES.find((t) => t.key === key)!
      const arr = data[key]
      ctx.strokeStyle = trace.color
      ctx.lineWidth = key === "y" ? 1.5 : 1
      ctx.globalAlpha = key === "noise" ? 0.5 : 1
      ctx.beginPath()
      arr.forEach((v, i) => {
        i === 0 ? ctx.moveTo(xPos(i), yPos(v)) : ctx.lineTo(xPos(i), yPos(v))
      })
      ctx.stroke()
      ctx.globalAlpha = 1
    }
  }, [y, x_true, f_true, noise])

  const handleDownload = useCallback(() => {
    if (y.length === 0) return
    const W = 1200, paneH = 220, gap = 8, titleH = 36
    const H = titleH + paneH * 4 + gap * 3
    const offscreen = document.createElement("canvas")
    offscreen.width = W
    offscreen.height = H
    const ctx = offscreen.getContext("2d")
    if (!ctx) return

    // Background
    ctx.fillStyle = "#fafafa"
    ctx.fillRect(0, 0, W, H)

    // Title
    ctx.fillStyle = "#18181b"
    ctx.font = "bold 16px sans-serif"
    ctx.fillText("Signal Decomposition:  y = x + f + w", 16, 24)

    const panes: { arr: number[]; color: string; title: string }[] = [
      { arr: y, color: "#71717a", title: "y  (Observed Signal)" },
      { arr: x_true, color: "#dc2626", title: "x  (Peaks)" },
      { arr: f_true, color: "#2563eb", title: "f  (Baseline)" },
      { arr: noise, color: "#16a34a", title: "w  (Noise)" },
    ]

    panes.forEach((p, i) => {
      const oy = titleH + i * (paneH + gap)
      drawPane(ctx, p.arr, p.color, p.title, 0, oy, W, paneH)
    })

    const link = document.createElement("a")
    link.download = `${label?.replace(/\s+/g, "_") || "signal"}_decomposition.png`
    link.href = offscreen.toDataURL("image/png")
    link.click()
  }, [y, x_true, f_true, noise, label])

  return (
    <div className="border bg-white">
      {label && (
        <div className="px-3 py-1.5 border-b bg-zinc-50 flex items-center justify-between">
          <span className="text-xs font-medium text-muted-foreground">{label}</span>
          <div className="flex items-center gap-3">
            {TRACES.map((t) => (
              <span key={t.key} className="flex items-center gap-1 text-[10px] text-muted-foreground">
                <span className="inline-block w-2.5 h-0.5 rounded-full" style={{ backgroundColor: t.color }} />
                {t.label}
              </span>
            ))}
            <button
              onClick={handleDownload}
              className="text-[10px] text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1 ml-2"
              title="Download 4-pane decomposition PNG"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>
              PNG
            </button>
          </div>
        </div>
      )}
      <canvas ref={canvasRef} className="w-full h-36" />
    </div>
  )
}
