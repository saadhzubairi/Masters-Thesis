"use client"

import { useRef, useEffect } from "react"

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
          </div>
        </div>
      )}
      <canvas ref={canvasRef} className="w-full h-36" />
    </div>
  )
}
