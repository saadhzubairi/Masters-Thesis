"use client"

import { useState, useEffect } from "react"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface AlphaSliderProps {
  name: string
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  description?: string
}

export function AlphaSlider({ name, value, onChange, min = 0, max = 10, step = 0.01, description }: AlphaSliderProps) {
  const [raw, setRaw] = useState(String(value))
  const [focused, setFocused] = useState(false)

  useEffect(() => {
    if (!focused) setRaw(String(value))
  }, [value, focused])

  const parsed = parseFloat(raw)
  const invalid = raw === "" || isNaN(parsed) || parsed < 0

  return (
    <div className="flex items-center gap-3">
      <Label className="w-40 text-xs font-mono shrink-0 flex items-center">
        {name}
        {description && (
          <span className="relative group ml-1 inline-flex cursor-help" title={description}>
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground/60 group-hover:text-foreground transition-colors">
              <circle cx="12" cy="12" r="10" />
              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
              <path d="M12 17h.01" />
            </svg>
            <span className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 w-56 rounded bg-zinc-900 px-2.5 py-1.5 text-[11px] leading-snug text-zinc-100 font-sans font-normal opacity-0 group-hover:opacity-100 transition-opacity z-50 shadow-lg">
              {description}
            </span>
          </span>
        )}
      </Label>
      <Slider
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={min}
        max={max}
        step={step}
        className="flex-1"
      />
      <Input
        type="number"
        value={focused ? raw : value}
        onChange={(e) => {
          const v = e.target.value
          setRaw(v)
          const n = parseFloat(v)
          if (!isNaN(n) && n >= 0) onChange(Math.min(n, max))
        }}
        onFocus={() => {
          setFocused(true)
          setRaw(String(value))
        }}
        onBlur={() => {
          setFocused(false)
          if (raw === "" || isNaN(parsed) || parsed < 0) setRaw(String(value))
        }}
        className={`w-20 text-xs font-mono h-8 ${invalid ? "border-red-400 focus-visible:ring-red-300" : ""}`}
        min={min}
        max={max}
        step={step}
      />
    </div>
  )
}
