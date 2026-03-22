"use client"

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
  return (
    <div className="flex items-center gap-3">
      <Label className="w-40 text-xs font-mono shrink-0">{name}</Label>
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
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="w-20 text-xs font-mono h-8"
        min={min}
        max={max}
        step={step}
      />
    </div>
  )
}
