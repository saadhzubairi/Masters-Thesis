"use client"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { AlphaSlider } from "./alpha-slider"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
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
  function updateStage(index: number, updates: Partial<StageConfig>) {
    const next = stages.map((s, i) => (i === index ? { ...s, ...updates } : s))
    onChange(next)
  }

  function removeStage(index: number) {
    onChange(stages.filter((_, i) => i !== index))
  }

  function addStage() {
    onChange([...stages, { name: `Stage ${String.fromCharCode(65 + stages.length)}`, epochs: 10, loss_config: {} }])
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
      {stages.map((stage, i) => (
        <Collapsible key={i} defaultOpen={i === stages.length - 1}>
          <div className="border p-3">
            <CollapsibleTrigger className="flex items-center justify-between w-full">
              <div className="flex items-center gap-2">
                <span className="bg-foreground text-background px-2 py-0.5 text-xs font-bold">
                  {String.fromCharCode(65 + i)}
                </span>
                <Input
                  value={stage.name}
                  onChange={(e) => updateStage(i, { name: e.target.value })}
                  className="h-7 text-xs w-40"
                  onClick={(e) => e.stopPropagation()}
                />
              </div>
              <div className="flex items-center gap-2">
                <Label className="text-xs text-muted-foreground">Epochs:</Label>
                <Input
                  type="number"
                  value={stage.epochs}
                  onChange={(e) => updateStage(i, { epochs: parseInt(e.target.value) || 1 })}
                  className="h-7 text-xs w-16"
                  onClick={(e) => e.stopPropagation()}
                />
                {stages.length > 1 && (
                  <Button variant="ghost" size="sm" className="h-7 text-xs text-destructive" onClick={(e) => { e.stopPropagation(); removeStage(i) }}>
                    Remove
                  </Button>
                )}
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3 space-y-2">
              {ALPHA_KEYS.map((key) => {
                const active = key in stage.loss_config
                const value = active ? (stage.loss_config[key] as number) : 0
                return (
                  <div key={key} className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={active}
                      onChange={(e) => toggleAlpha(i, key, e.target.checked)}
                      className="accent-foreground"
                    />
                    <div className={active ? "flex-1" : "flex-1 opacity-40"}>
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
            </CollapsibleContent>
          </div>
        </Collapsible>
      ))}
      <Button variant="outline" size="sm" onClick={addStage} className="text-xs">
        + Add Stage
      </Button>
    </div>
  )
}
