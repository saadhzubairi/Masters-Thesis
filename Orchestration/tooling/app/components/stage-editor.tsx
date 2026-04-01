"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { AlphaSlider } from "./alpha-slider"
import type { StageConfig, LossConfig } from "@/lib/types"
import { DEFAULT_LOSS_CONFIG } from "@/lib/types"

interface StageEditorProps {
  stages: StageConfig[]
  onChange: (stages: StageConfig[]) => void
}

const ALPHA_KEYS = [
  "alpha_mse", "alpha_l1", "alpha_tv", "alpha_smooth", "alpha_neg",
  "alpha_baseline", "alpha_leakage", "alpha_ortho", "alpha_baseline_tv",
] as const

export function StageEditor({ stages, onChange }: StageEditorProps) {
  const [openIndex, setOpenIndex] = useState<number | null>(stages.length - 1)

  function updateStage(index: number, updates: Partial<StageConfig>) {
    const next = stages.map((s, i) => (i === index ? { ...s, ...updates } : s))
    onChange(next)
  }

  function removeStage(index: number) {
    onChange(stages.filter((_, i) => i !== index))
    if (openIndex === index) setOpenIndex(null)
  }

  function addStage() {
    onChange([...stages, { name: `Stage ${String.fromCharCode(65 + stages.length)}`, epochs: 10, loss_config: {} }])
    setOpenIndex(stages.length)
  }

  function toggleAlpha(stageIndex: number, key: string, enabled: boolean) {
    const stage = stages[stageIndex]
    const nextConfig = { ...stage.loss_config }
    if (enabled) {
      nextConfig[key as keyof LossConfig] = DEFAULT_LOSS_CONFIG[key as keyof LossConfig] as never
    } else {
      delete nextConfig[key as keyof LossConfig]
    }
    updateStage(stageIndex, { loss_config: nextConfig })
  }

  return (
    <div className="space-y-3">
      {stages.map((stage, i) => {
        const isOpen = openIndex === i
        return (
          <div key={i} className="border bg-zinc-50/50">
            <Button
              variant="ghost"
              onClick={() => setOpenIndex(isOpen ? null : i)}
              className="w-full flex items-center justify-between px-3 py-2.5 h-auto hover:bg-zinc-100"
            >
              <div className="flex items-center gap-2">
                <span className="bg-foreground text-background px-2 py-0.5 text-xs font-bold">
                  {String.fromCharCode(65 + i)}
                </span>
                <span className="text-xs">{stage.name}</span>
                <span className="text-xs text-muted-foreground">{stage.epochs} epochs</span>
              </div>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className={`transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}
              >
                <path d="m6 9 6 6 6-6" />
              </svg>
            </Button>
            {isOpen && (
              <div className="px-3 pb-3 pt-2 border-t bg-white space-y-3">
                <div className="flex items-center gap-3">
                  <div className="flex-1">
                    <Label className="text-xs">Stage Name</Label>
                    <Input
                      value={stage.name}
                      onChange={(e) => updateStage(i, { name: e.target.value })}
                      className="h-8 text-xs mt-1"
                    />
                  </div>
                  <div className="w-24">
                    <Label className="text-xs">Epochs</Label>
                    <Input
                      type="number"
                      value={stage.epochs}
                      onChange={(e) => updateStage(i, { epochs: parseInt(e.target.value) || 1 })}
                      className="h-8 text-xs mt-1"
                    />
                  </div>
                  {stages.length > 1 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 text-xs text-destructive mt-5"
                      onClick={() => removeStage(i)}
                    >
                      Remove
                    </Button>
                  )}
                </div>
                <div className="space-y-1">
                  <Label className="text-xs text-muted-foreground">Loss Overrides</Label>
                  {ALPHA_KEYS.map((key) => {
                    const active = key in stage.loss_config
                    const value = active ? (stage.loss_config[key] as number) : 0
                    return (
                      <div
                        key={key}
                        className="flex items-center gap-2 px-2 py-1.5 hover:bg-zinc-50 transition-colors"
                      >
                        <label className="cursor-pointer flex items-center" onClick={(e) => e.stopPropagation()}>
                          <input
                            type="checkbox"
                            checked={active}
                            onChange={(e) => toggleAlpha(i, key, e.target.checked)}
                            className="accent-foreground cursor-pointer"
                          />
                        </label>
                        <div
                          className={active ? "flex-1" : "flex-1 opacity-40 cursor-pointer"}
                          onClick={() => { if (!active) toggleAlpha(i, key, true) }}
                        >
                          <AlphaSlider
                            name={key}
                            value={value}
                            onChange={(v) => {
                              const nextConfig = { ...stage.loss_config, [key]: v }
                              updateStage(i, { loss_config: nextConfig })
                            }}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )
      })}
      <Button variant="outline" size="sm" onClick={addStage} className="text-xs">
        + Add Stage
      </Button>
    </div>
  )
}
