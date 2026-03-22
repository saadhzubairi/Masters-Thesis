"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Separator } from "@/components/ui/separator"
import { AlphaSlider } from "./alpha-slider"
import { StageEditor } from "./stage-editor"
import { createRun, listRuns, getRun } from "@/lib/api"
import type { RunConfig, RunSummary, ModelConfig, TrainingConfig, LossConfig, StageConfig } from "@/lib/types"
import {
  DEFAULT_MODEL_CONFIG,
  DEFAULT_TRAINING_CONFIG,
  DEFAULT_LOSS_CONFIG,
  DEFAULT_STAGES,
} from "@/lib/types"

export function ConfigForm() {
  const router = useRouter()
  const [name, setName] = useState("")
  const [model, setModel] = useState<ModelConfig>({ ...DEFAULT_MODEL_CONFIG })
  const [training, setTraining] = useState<TrainingConfig>({ ...DEFAULT_TRAINING_CONFIG })
  const [loss, setLoss] = useState<LossConfig>({ ...DEFAULT_LOSS_CONFIG })
  const [stages, setStages] = useState<StageConfig[]>(structuredClone(DEFAULT_STAGES))
  const [submitting, setSubmitting] = useState(false)
  const [previousRuns, setPreviousRuns] = useState<RunSummary[]>([])

  useEffect(() => {
    listRuns().then(setPreviousRuns).catch(() => {})
  }, [])

  async function handleClone(runId: string) {
    try {
      const run = await getRun(runId)
      if (run.config) {
        setModel(run.config.model || DEFAULT_MODEL_CONFIG)
        setTraining(run.config.training || DEFAULT_TRAINING_CONFIG)
        setLoss(run.config.loss || DEFAULT_LOSS_CONFIG)
        setStages(run.config.stages || DEFAULT_STAGES)
        setName(`${run.name} (clone)`)
      }
    } catch { /* ignore */ }
  }

  async function handleSubmit() {
    setSubmitting(true)
    try {
      const config: RunConfig = { name: name || "Untitled Run", model, training, loss, stages }
      const { run_id } = await createRun(config)
      router.push(`/runs/${run_id}`)
    } catch (err) {
      alert(`Failed to start run: ${err}`)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Clone Source */}
      <div>
        <Label className="text-xs font-semibold">Clone From</Label>
        <div className="flex gap-2 mt-1">
          <Select onValueChange={handleClone}>
            <SelectTrigger className="h-9 text-xs">
              <SelectValue placeholder="Select a previous run..." />
            </SelectTrigger>
            <SelectContent>
              {previousRuns.map((r) => (
                <SelectItem key={r.id} value={r.id} className="text-xs">
                  Run #{r.id.slice(-6)} — {r.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Run Name */}
      <div>
        <Label className="text-xs font-semibold">Run Name</Label>
        <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. high ortho experiment" className="mt-1 h-9 text-sm" />
      </div>

      <Separator />

      {/* Model Architecture */}
      <Collapsible defaultOpen>
        <CollapsibleTrigger className="flex items-center justify-between w-full">
          <span className="text-sm font-semibold">Model Architecture</span>
          <span className="text-xs text-muted-foreground">toggle</span>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 grid grid-cols-3 gap-3">
          {Object.entries(model).map(([key, val]) => (
            <div key={key}>
              <Label className="text-xs font-mono">{key}</Label>
              {typeof val === "boolean" ? (
                <div className="mt-1">
                  <input type="checkbox" checked={val} onChange={(e) => setModel({ ...model, [key]: e.target.checked })} className="accent-foreground" />
                </div>
              ) : (
                <Input
                  type="number"
                  value={val}
                  onChange={(e) => setModel({ ...model, [key]: parseFloat(e.target.value) || 0 })}
                  className="mt-1 h-8 text-xs font-mono"
                  step={key === "fc" ? 0.001 : 1}
                />
              )}
            </div>
          ))}
        </CollapsibleContent>
      </Collapsible>

      <Separator />

      {/* Training Parameters */}
      <Collapsible defaultOpen>
        <CollapsibleTrigger className="flex items-center justify-between w-full">
          <span className="text-sm font-semibold">Training Parameters</span>
          <span className="text-xs text-muted-foreground">toggle</span>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 grid grid-cols-3 gap-3">
          {Object.entries(training).map(([key, val]) => (
            <div key={key}>
              <Label className="text-xs font-mono">{key}</Label>
              <Input
                type="number"
                value={val}
                onChange={(e) => setTraining({ ...training, [key]: parseFloat(e.target.value) || 0 })}
                className="mt-1 h-8 text-xs font-mono"
                step={key === "learning_rate" ? 0.0001 : key === "noise_level" ? 0.001 : 1}
              />
            </div>
          ))}
        </CollapsibleContent>
      </Collapsible>

      <Separator />

      {/* Loss Weights */}
      <Collapsible defaultOpen>
        <CollapsibleTrigger className="flex items-center justify-between w-full">
          <span className="text-sm font-semibold">Loss Weights</span>
          <span className="text-xs text-muted-foreground">toggle</span>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 space-y-2">
          {Object.entries(loss).map(([key, val]) => {
            if (typeof val === "boolean") {
              return (
                <div key={key} className="flex items-center gap-3">
                  <Label className="w-40 text-xs font-mono">{key}</Label>
                  <input type="checkbox" checked={val} onChange={(e) => setLoss({ ...loss, [key]: e.target.checked })} className="accent-foreground" />
                </div>
              )
            }
            return (
              <AlphaSlider
                key={key}
                name={key}
                value={val as number}
                onChange={(v) => setLoss({ ...loss, [key]: v })}
                max={key.includes("threshold") ? 1 : key.includes("abs_min") ? 0.1 : key === "huber_delta" ? 1 : 10}
              />
            )
          })}
        </CollapsibleContent>
      </Collapsible>

      <Separator />

      {/* Training Stages */}
      <div>
        <span className="text-sm font-semibold">Training Stages</span>
        <div className="mt-3">
          <StageEditor stages={stages} onChange={setStages} />
        </div>
      </div>

      <Separator />

      {/* Submit */}
      <div className="flex justify-end">
        <Button onClick={handleSubmit} disabled={submitting} className="px-8">
          {submitting ? "Starting..." : "Start Training"}
        </Button>
      </div>
    </div>
  )
}
