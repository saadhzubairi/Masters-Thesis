"use client"

import { useRef, useEffect } from "react"
import type { EpochData } from "@/lib/types"

interface LossChartProps {
  epochs: EpochData[]
}

export function LossChart({ epochs }: LossChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || epochs.length === 0) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx.scale(dpr, dpr)

    ctx.clearRect(0, 0, w, h)

    const trainLosses = epochs.map((e) => e.train_loss)
    const testLosses = epochs.map((e) => e.test_loss).filter((v): v is number => v != null)
    const allValues = [...trainLosses, ...testLosses]
    const maxVal = Math.max(...allValues) * 1.1
    const minVal = 0

    const padL = 50, padR = 10, padT = 10, padB = 30
    const plotW = w - padL - padR
    const plotH = h - padT - padB

    function x(i: number) { return padL + (i / Math.max(epochs.length - 1, 1)) * plotW }
    function y(v: number) { return padT + (1 - (v - minVal) / (maxVal - minVal)) * plotH }

    // Grid
    ctx.strokeStyle = "#e5e5e5"
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const yy = padT + (i / 4) * plotH
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke()
      ctx.fillStyle = "#71717a"; ctx.font = "10px monospace"
      ctx.fillText(((maxVal - minVal) * (1 - i / 4) + minVal).toFixed(4), 2, yy + 3)
    }

    // Train loss
    ctx.strokeStyle = "#18181b"
    ctx.lineWidth = 2
    ctx.beginPath()
    trainLosses.forEach((v, i) => { i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v)) })
    ctx.stroke()

    // Test loss
    if (testLosses.length > 0) {
      ctx.strokeStyle = "#dc2626"
      ctx.lineWidth = 2
      ctx.beginPath()
      epochs.forEach((e, i) => {
        if (e.test_loss != null) {
          i === 0 || epochs[i - 1].test_loss == null ? ctx.moveTo(x(i), y(e.test_loss)) : ctx.lineTo(x(i), y(e.test_loss))
        }
      })
      ctx.stroke()
    }

    // Stage boundary markers
    ctx.strokeStyle = "#a1a1aa"
    ctx.lineWidth = 1
    ctx.setLineDash([4, 4])
    for (let i = 1; i < epochs.length; i++) {
      if (epochs[i].stage !== epochs[i - 1].stage) {
        ctx.beginPath(); ctx.moveTo(x(i), padT); ctx.lineTo(x(i), padT + plotH); ctx.stroke()
        ctx.fillStyle = "#71717a"; ctx.font = "9px monospace"
        ctx.fillText(epochs[i].stage, x(i) + 2, padT + 10)
      }
    }
    ctx.setLineDash([])

    // X axis labels
    ctx.fillStyle = "#71717a"; ctx.font = "10px monospace"
    const step = Math.max(1, Math.floor(epochs.length / 10))
    for (let i = 0; i < epochs.length; i += step) {
      ctx.fillText(String(epochs[i].epoch), x(i) - 4, h - 5)
    }
  }, [epochs])

  return (
    <div className="border p-4">
      <div className="flex items-center gap-4 mb-2">
        <span className="text-sm font-semibold">Loss Curves</span>
        <span className="text-xs"><span className="inline-block w-3 h-0.5 bg-foreground mr-1 align-middle" /> Train</span>
        <span className="text-xs"><span className="inline-block w-3 h-0.5 bg-red-600 mr-1 align-middle" /> Test</span>
      </div>
      <canvas ref={canvasRef} className="w-full h-48" />
    </div>
  )
}
