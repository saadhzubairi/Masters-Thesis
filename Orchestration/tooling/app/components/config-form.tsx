"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { AlphaSlider } from "./alpha-slider"
import { StageEditor } from "./stage-editor"
import { LlmConfigDialog } from "./llm-config-dialog"
import { createRun, listRuns, getRun, getDataConfig } from "@/lib/api"
import type {
  RunConfig, RunSummary, ModelConfig, FastModelConfig, TrainingConfig,
  LossConfig, StageConfig, DataConfig, ModelType, DeviceType,
} from "@/lib/types"
import {
  DEFAULT_MODEL_CONFIG,
  DEFAULT_FAST_MODEL_CONFIG,
  DEFAULT_V5_MODEL_CONFIG,
  DEFAULT_TRAINING_CONFIG,
  DEFAULT_LOSS_CONFIG,
  DEFAULT_STAGES,
} from "@/lib/types"

const TOOLTIPS: Record<string, string> = {
  // Shared
  N: "Signal length in samples. Must match your synthetic data length. Optimal: 4096. Range: 512–8192.",
  d: "Derivative order for the BEADS asymmetric penalty. d=1 penalizes first differences (piecewise-constant peaks); d=2 penalizes second differences (piecewise-linear). Optimal: 1. Range: 1–2.",
  fc: "Normalized cutoff frequency for the high-pass/low-pass filter pair. Lower = smoother baseline separation. Optimal: 0.006. Range: 0.003–0.01.",
  num_layers: "Number of unrolled BEADS iterations (network depth). More layers = more refinement but slower. Optimal: 5–8. Range: 3–15.",
  // LBEADS only
  solve_cg_iters: "Conjugate gradient iterations for the peak-estimation solve step per layer. Higher = more accurate but slower. Optimal: 5. Range: 3–12.",
  lowpass_cg_iters: "Conjugate gradient iterations for the baseline low-pass filter per layer. Higher = smoother baseline. Optimal: 24. Range: 8–48.",
  shared_params: "If true, all layers share parameters (recurrent). If false, each layer is independent. Optimal: false for most cases. Shared can help with fewer layers.",
  // Fast only
  lowpass_iterations: "Number of iterated low-pass filter applications. More = smoother baseline. Optimal: 3. Range: 1–5.",
  // Init params (shared labels)
  init_lam0: "Initial λ₀ (asymmetric penalty weight). Controls sparsity.",
  init_lam1: "Initial λ₁ (first-derivative penalty).",
  init_lam2: "Initial λ₂ (second-derivative penalty).",
  init_r: "Initial asymmetry ratio r. Higher = stronger negative suppression. Optimal: 6.0. Range: 2–12.",
  init_step: "Initial layer relaxation step size. Optimal: 1.0. Range: 0.05–2.0.",
  init_step_size: "Initial proximal gradient step size. Optimal: 0.1. Range: 0.01–1.0.",
  init_output_gain: "Initial output gain multiplier. Optimal: 1.0. Range: 0.1–10.0.",
  // Training Parameters
  learning_rate: "Adam optimizer learning rate. Optimal: 1e-3. Range: 5e-4–3e-3. Lower for fine-tuning, higher for quick exploration.",
  batch_size: "Signals per training batch. Larger = more stable gradients but more memory. Optimal: 4–8. Range: 2–32.",
  num_samples: "Total synthetic signals for training + validation. More = better generalization but slower. Optimal: 500. Range: 200–2000.",
  noise_level: "Std dev of additive Gaussian noise in synthetic data. Optimal: 0.01. Range: 0.001–0.05. Higher makes the model more noise-robust.",
  train_ratio: "Fraction used for training (rest = validation). Optimal: 0.8. Range: 0.7–0.9.",
  seed: "Random seed for reproducibility. Any integer. Default: 42.",
  // Loss Weights
  alpha_mse: "Peak reconstruction MSE. Primary fidelity term. Optimal: 1.0. Range: 0.5–2.0. Keep this as the anchor.",
  alpha_l1: "L1 sparsity on peaks. Promotes sparse (mostly-zero) peaks. Optimal: 0.01. Range: 0–0.1. Too high = suppressed small peaks.",
  alpha_tv: "Total Variation on peaks. Promotes piecewise-constant shapes. Optimal: 0.01. Range: 0–0.1. Too high = over-smoothed peaks.",
  alpha_smooth: "Baseline smoothness (2nd differences). Optimal: 0.2. Range: 0.05–1.0. Higher = smoother baseline.",
  alpha_neg: "Negativity penalty on peaks. Discourages negative peak values. Optimal: 0.5–2.0. Range: 0–5.0. With softplus constraint, 0.5 suffices.",
  alpha_baseline: "Baseline reconstruction MSE. Optimal: 0.5. Range: 0.1–2.0. Important for good baseline separation.",
  alpha_leakage: "Baseline leakage penalty. Penalizes peak energy absorbed into baseline. Optimal: 0.3–0.5. Range: 0–2.0.",
  alpha_ortho: "Orthogonality between peaks and baseline. Enforces mutual exclusivity. Optimal: 0.1–0.2. Range: 0–1.0. Too high = training instability.",
  alpha_baseline_tv: "Total Variation on baseline. Extra smoothness constraint. Optimal: 0 (usually covered by alpha_smooth). Range: 0–0.5.",
  peak_mask_rel_threshold: "Relative threshold for peak activity mask. Optimal: 0.02. Range: 0.01–0.1. Lower = more sensitive peak detection.",
  peak_mask_abs_min: "Absolute minimum for peak mask. Prevents masking at tiny amplitudes. Optimal: 1e-4. Range: 1e-5–1e-3.",
  use_huber: "Use Huber loss instead of MSE for peaks. Less sensitive to outliers. Optimal: false for clean data, true if noisy.",
  huber_delta: "Huber loss delta. Below = quadratic (MSE), above = linear (MAE). Optimal: 0.1. Range: 0.01–1.0. Only matters if use_huber is on.",
}

function stepFor(key: string): number {
  if (key === "fc") return 0.001
  if (key.startsWith("init_lam")) return 0.001
  if (key === "init_step_size") return 0.01
  if (key === "init_r" || key === "init_step" || key === "init_output_gain") return 0.1
  if (key === "learning_rate") return 0.0001
  if (key === "noise_level") return 0.001
  return 1
}

function Tip({ text }: { text: string }) {
  return (
    <span className="relative group ml-1 inline-flex cursor-help" title={text}>
      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground/60 group-hover:text-foreground transition-colors">
        <circle cx="12" cy="12" r="10" />
        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
        <path d="M12 17h.01" />
      </svg>
      <span className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 w-56 rounded bg-zinc-900 px-2.5 py-1.5 text-[11px] leading-snug text-zinc-100 opacity-0 group-hover:opacity-100 transition-opacity z-50 shadow-lg">
        {text}
      </span>
    </span>
  )
}

function timeAgo(epochSeconds: number): string {
  const diff = Math.floor(Date.now() / 1000 - epochSeconds)
  if (diff < 60) return "just now"
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function Section({
  title,
  defaultOpen = true,
  children,
  className = "",
}: {
  title: string
  defaultOpen?: boolean
  children: React.ReactNode
  className?: string
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={`border bg-zinc-50/50 ${className}`}>
      <Button
        variant="ghost"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 h-auto hover:bg-zinc-100"
      >
        <span className="text-sm font-semibold">{title}</span>
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
          className={`transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </Button>
      {open && <div className="px-4 pb-4 pt-2 border-t bg-white">{children}</div>}
    </div>
  )
}

function NumberField({ label, value, onChange, step, tooltip, allowZero = true }: {
  label: string; value: number; onChange: (v: number) => void; step: number; tooltip?: string; allowZero?: boolean
}) {
  const [raw, setRaw] = useState(String(value))
  const [focused, setFocused] = useState(false)

  // Sync from parent when not focused (e.g. clone/reset)
  useEffect(() => {
    if (!focused) setRaw(String(value))
  }, [value, focused])

  const parsed = parseFloat(raw)
  const invalid = raw === "" || isNaN(parsed) || parsed < 0 || (!allowZero && parsed === 0)

  return (
    <div>
      <Label className="text-xs">
        {label}
        {tooltip && <Tip text={tooltip} />}
      </Label>
      <Input
        type="number"
        value={focused ? raw : value}
        onChange={(e) => {
          const v = e.target.value
          setRaw(v)
          const n = parseFloat(v)
          if (!isNaN(n) && n >= 0) onChange(n)
        }}
        onFocus={() => {
          setFocused(true)
          setRaw(String(value))
        }}
        onBlur={() => {
          setFocused(false)
          // Restore last valid value if the field is empty/invalid
          if (raw === "" || isNaN(parsed) || parsed < 0) {
            setRaw(String(value))
          }
        }}
        onKeyDown={(e) => {
          // Prevent arrow keys from going negative
          if (e.key === "ArrowDown" && value - step < 0) {
            e.preventDefault()
          }
        }}
        className={`mt-1 h-8 text-xs ${invalid ? "border-red-400 focus-visible:ring-red-300" : ""}`}
        step={step}
        min={0}
      />
    </div>
  )
}

function BoolField({ label, checked, onChange, tooltip }: {
  label: string; checked: boolean; onChange: (v: boolean) => void; tooltip?: string
}) {
  return (
    <div>
      <Label className="text-xs">
        {label}
        {tooltip && <Tip text={tooltip} />}
      </Label>
      <div className="mt-1">
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="accent-foreground" />
      </div>
    </div>
  )
}

/** Renders a grid of fields from an object, splitting init_ keys into a separate section when requested */
function ObjectFields({
  obj, setObj, filter,
}: {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  obj: Record<string, any>; setObj: (next: any) => void; filter?: (key: string) => boolean
}) {
  const entries = Object.entries(obj).filter(([k]) => !filter || filter(k))
  return (
    <div className="grid grid-cols-2 gap-3">
      {entries.map(([key, val]) => {
        if (typeof val === "boolean") {
          return <BoolField key={key} label={key} checked={val} onChange={(v) => setObj({ ...obj, [key]: v })} tooltip={TOOLTIPS[key]} />
        }
        const displayLabel = key.startsWith("init_") ? key.replace("init_", "") : key
        return (
          <NumberField
            key={key}
            label={displayLabel}
            value={val as number}
            onChange={(v) => setObj({ ...obj, [key]: v })}
            step={stepFor(key)}
            tooltip={TOOLTIPS[key]}
          />
        )
      })}
    </div>
  )
}

export function ConfigForm() {
  const router = useRouter()
  const [modelType, setModelType] = useState<ModelType | null>(null)
  const [device, setDevice] = useState<DeviceType>("cpu")
  const [name, setName] = useState("")
  const [model, setModel] = useState<ModelConfig>({ ...DEFAULT_MODEL_CONFIG })
  const [modelFast, setModelFast] = useState<FastModelConfig>({ ...DEFAULT_FAST_MODEL_CONFIG })
  const [modelV5, setModelV5] = useState<FastModelConfig>({ ...DEFAULT_V5_MODEL_CONFIG })
  const [training, setTraining] = useState<TrainingConfig>({ ...DEFAULT_TRAINING_CONFIG })
  const [loss, setLoss] = useState<LossConfig>({ ...DEFAULT_LOSS_CONFIG })
  const [stages, setStages] = useState<StageConfig[]>(structuredClone(DEFAULT_STAGES))
  const [submitting, setSubmitting] = useState(false)
  const [previousRuns, setPreviousRuns] = useState<RunSummary[]>([])

  useEffect(() => {
    listRuns().then(setPreviousRuns).catch(() => {})
    // Check for config passed via sessionStorage (from "retry with different config")
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search)
      if (params.get("clone_from_session") === "1") {
        try {
          const raw = sessionStorage.getItem("clone_config")
          if (raw) {
            sessionStorage.removeItem("clone_config")
            const cfg = JSON.parse(raw)
            const mt = cfg.model_type || "lbeads"
            setModelType(mt)
            if (cfg.device) setDevice(cfg.device)
            setModel(cfg.model || DEFAULT_MODEL_CONFIG)
            if (cfg.model_fast) setModelFast(cfg.model_fast)
            if (cfg.model_fast) setModelV5(cfg.model_fast)
            setTraining(cfg.training || DEFAULT_TRAINING_CONFIG)
            setLoss(cfg.loss || DEFAULT_LOSS_CONFIG)
            setStages(cfg.stages || DEFAULT_STAGES)
            setName(`${cfg.name || "Untitled"} (retry)`)
          }
        } catch { /* ignore parse errors */ }
      }
    }
  }, [])

  async function handleClone(runId: string) {
    try {
      const run = await getRun(runId)
      if (run.config) {
        const mt = run.config.model_type || "lbeads"
        setModelType(mt)
        if (run.config.device) setDevice(run.config.device)
        setModel(run.config.model || DEFAULT_MODEL_CONFIG)
        if (run.config.model_fast) setModelFast(run.config.model_fast)
        if (run.config.model_fast) setModelV5(run.config.model_fast)
        setTraining(run.config.training || DEFAULT_TRAINING_CONFIG)
        setLoss(run.config.loss || DEFAULT_LOSS_CONFIG)
        setStages(run.config.stages || DEFAULT_STAGES)
        setName(`${run.name} (clone)`)
      }
    } catch { /* ignore */ }
  }

  async function handleSubmit() {
    if (!modelType) return
    setSubmitting(true)
    try {
      let data: DataConfig | undefined
      try {
        data = await getDataConfig()
      } catch {
        // No saved data config — omit from run
      }
      const config: RunConfig = {
        name: name || "Untitled Run",
        model_type: modelType,
        device,
        model,
        ...(modelType === "lbeads_fast" ? { model_fast: modelFast } : modelType === "lbeads_v5" ? { model_fast: modelV5 } : {}),
        training, loss, stages, data,
      }
      const { run_id } = await createRun(config)
      router.push(`/runs/${run_id}`)
    } catch (err) {
      alert(`Failed to start run: ${err}`)
    } finally {
      setSubmitting(false)
    }
  }

  const isStandard = modelType === "lbeads"
  const isV5 = modelType === "lbeads_v5"

  const llmDialog = (
    <LlmConfigDialog
      getCurrentConfig={() => ({
        name: name || "Untitled Run",
        model_type: modelType || "lbeads",
        device,
        model,
        ...(modelType === "lbeads_fast" ? { model_fast: modelFast } : modelType === "lbeads_v5" ? { model_fast: modelV5 } : {}),
        training, loss, stages,
      })}
      onApply={(cfg) => {
        if (cfg.model_type) setModelType(cfg.model_type)
        if (cfg.device) setDevice(cfg.device)
        if (cfg.name !== undefined) setName(cfg.name)
        if (cfg.model) setModel({ ...DEFAULT_MODEL_CONFIG, ...cfg.model })
        if (cfg.model_fast) setModelFast({ ...DEFAULT_FAST_MODEL_CONFIG, ...cfg.model_fast })
        if (cfg.model_fast) setModelV5({ ...DEFAULT_V5_MODEL_CONFIG, ...cfg.model_fast })
        if (cfg.training) setTraining({ ...DEFAULT_TRAINING_CONFIG, ...cfg.training })
        if (cfg.loss) setLoss({ ...DEFAULT_LOSS_CONFIG, ...cfg.loss })
        if (cfg.stages?.length) setStages(cfg.stages)
      }}
    />
  )

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-lg font-semibold mb-1">New Training Run</h2>
          <p className="text-sm text-muted-foreground">Configure hyperparameters and start training.</p>
        </div>
        {llmDialog}
      </div>

      {modelType === null ? (
        <>
          {/* Clone from previous */}
          <div className="max-w-sm">
            <Label className="text-xs font-semibold">Clone From Previous Run</Label>
            <Select onValueChange={handleClone}>
              <SelectTrigger className="mt-1 h-9 text-xs">
                <SelectValue placeholder="Select a previous run..." />
              </SelectTrigger>
              <SelectContent>
                {previousRuns.map((r) => (
                  <SelectItem key={r.id} value={r.id} className="text-xs">
                    Run #{r.id.slice(-6)} — {r.name} · {timeAgo(r.created_at)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="border-t pt-6">
            <p className="text-sm font-semibold mb-1">Select Model Type</p>
            <p className="text-xs text-muted-foreground mb-4">Choose the architecture for this training run.</p>
            <div className="grid grid-cols-3 gap-4 max-w-3xl">
              <button
                onClick={() => setModelType("lbeads")}
                className="border bg-white p-5 text-left hover:border-foreground transition-colors group"
              >
                <p className="text-sm font-bold font-mono group-hover:underline">LBEADS-NET</p>
                <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed">
                  Full unrolled BEADS with CG solver per layer. More accurate, slower to train.
                  Learnable per-layer λ₀, λ₁, λ₂, r, step, output gain.
                </p>
                <p className="text-[10px] text-muted-foreground/60 mt-2 font-mono">shared_params · solve_cg_iters · lowpass_cg_iters</p>
              </button>
              <button
                onClick={() => setModelType("lbeads_fast")}
                className="border bg-white p-5 text-left hover:border-foreground transition-colors group"
              >
                <p className="text-sm font-bold font-mono group-hover:underline">LBEADS-NET Fast</p>
                <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed">
                  Proximal gradient with asymmetric soft thresholding. Faster to train, fully differentiable.
                  Per-layer λ₀, λ₁, λ₂, r, step size.
                </p>
                <p className="text-[10px] text-muted-foreground/60 mt-2 font-mono">lowpass_iterations · init_step_size</p>
              </button>
              <button
                onClick={() => setModelType("lbeads_v5")}
                className="border bg-white p-5 text-left hover:border-foreground transition-colors group"
              >
                <p className="text-sm font-bold font-mono group-hover:underline">LBEADS-NET v5</p>
                <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed">
                  Proximal gradient (same as Fast) with v5 defaults: 20 layers, smaller lambdas,
                  single-pass lowpass. Tuned for stable refinement from high-pass init.
                </p>
                <p className="text-[10px] text-muted-foreground/60 mt-2 font-mono">lowpass_iterations=1 · 20 layers · smaller λ</p>
              </button>
            </div>
          </div>
        </>
      ) : (
        <div className="space-y-4">
          {/* Row 1: Model Type badge + Device + Clone From + Run Name */}
          <div className="grid grid-cols-[auto_auto_1fr_1fr] gap-4 items-end">
            <div>
              <Label className="text-xs font-semibold">Model</Label>
              <button
                onClick={() => setModelType(null)}
                className="mt-1 flex items-center gap-1.5 border px-3 h-9 text-xs font-mono font-bold hover:bg-zinc-50 transition-colors"
                title="Click to change model type"
              >
                {isStandard ? "LBEADS-NET" : isV5 ? "LBEADS-NET v5" : "LBEADS-NET Fast"}
                <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground">
                  <path d="m6 9 6 6 6-6" />
                </svg>
              </button>
            </div>
            <div>
              <Label className="text-xs font-semibold">Device</Label>
              <Select value={device} onValueChange={(v) => setDevice(v as DeviceType)}>
                <SelectTrigger className="mt-1 h-9 text-xs w-28">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="cpu" className="text-xs">CPU</SelectItem>
                  <SelectItem value="mps" className="text-xs">MPS (Apple Silicon)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs font-semibold">Clone From</Label>
              <Select onValueChange={handleClone}>
                <SelectTrigger className="mt-1 h-9 text-xs">
                  <SelectValue placeholder="Select a previous run..." />
                </SelectTrigger>
                <SelectContent>
                  {previousRuns.map((r) => (
                    <SelectItem key={r.id} value={r.id} className="text-xs">
                      Run #{r.id.slice(-6)} — {r.name} · {timeAgo(r.created_at)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs font-semibold">Run Name</Label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. high ortho experiment"
                className="mt-1 h-9 text-sm"
              />
            </div>
          </div>

          {/* Row 2: Architecture + Training + Init Params */}
          <div className="grid grid-cols-3 gap-4">
            <Section title={isStandard ? "Model Architecture" : isV5 ? "Model Architecture (v5)" : "Model Architecture (Fast)"}>
              {isStandard ? (
                <ObjectFields obj={model} setObj={setModel} filter={(k) => !k.startsWith("init_")} />
              ) : isV5 ? (
                <ObjectFields obj={modelV5} setObj={setModelV5} filter={(k) => !k.startsWith("init_")} />
              ) : (
                <ObjectFields obj={modelFast} setObj={setModelFast} filter={(k) => !k.startsWith("init_")} />
              )}
            </Section>

            <Section title="Training Parameters">
              <ObjectFields obj={training} setObj={setTraining} />
            </Section>

            <Section title="Initial Parameters">
              {isStandard ? (
                <ObjectFields obj={model} setObj={setModel} filter={(k) => k.startsWith("init_")} />
              ) : isV5 ? (
                <ObjectFields obj={modelV5} setObj={setModelV5} filter={(k) => k.startsWith("init_")} />
              ) : (
                <ObjectFields obj={modelFast} setObj={setModelFast} filter={(k) => k.startsWith("init_")} />
              )}
            </Section>
          </div>

          {/* Loss Weights */}
          <Section title="Loss Weights" defaultOpen={false}>
            <div className="space-y-2">
              {Object.entries(loss).map(([key, val]) => {
                if (typeof val === "boolean") {
                  return (
                    <label key={key} className="flex items-center gap-3 px-2 py-1.5 cursor-pointer hover:bg-zinc-50 transition-colors">
                      <input
                        type="checkbox"
                        checked={val}
                        onChange={(e) => setLoss({ ...loss, [key]: e.target.checked })}
                        className="accent-foreground cursor-pointer"
                      />
                      <span className="text-xs">
                        {key}
                        {TOOLTIPS[key] && <Tip text={TOOLTIPS[key]} />}
                      </span>
                    </label>
                  )
                }
                return (
                  <AlphaSlider
                    key={key}
                    name={key}
                    value={val as number}
                    onChange={(v) => setLoss({ ...loss, [key]: v })}
                    max={key.includes("threshold") ? 1 : key.includes("abs_min") ? 0.1 : key === "huber_delta" ? 1 : 10}
                    description={TOOLTIPS[key]}
                  />
                )
              })}
            </div>
          </Section>

          {/* Training Stages */}
          <Section title="Training Stages">
            <StageEditor stages={stages} onChange={setStages} />
          </Section>

          {/* Submit */}
          <div className="flex justify-end pt-2">
            <Button onClick={handleSubmit} disabled={submitting} className="px-8">
              {submitting ? "Starting..." : "Start Training"}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
