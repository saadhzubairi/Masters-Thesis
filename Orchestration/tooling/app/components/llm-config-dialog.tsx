"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog, DialogContent, DialogDescription, DialogFooter,
  DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog"
import type { RunConfig } from "@/lib/types"

const FIELD_DESCRIPTIONS = `### model_type
- "lbeads": Full unrolled BEADS with CG solver per layer. More accurate, slower to train.
- "lbeads_fast": Proximal gradient with asymmetric soft thresholding. Faster, fully differentiable.

### model (for model_type "lbeads")
- N: Signal length in samples (optimal: 4096, range: 512-8192)
- d: Derivative order for BEADS asymmetric penalty (1 or 2)
- fc: Normalized cutoff frequency for high-pass/low-pass filter (optimal: 0.006, range: 0.003-0.01)
- num_layers: Number of unrolled BEADS iterations (optimal: 5-8, range: 3-15)
- solve_cg_iters: CG iterations for peak-estimation solve step (optimal: 5, range: 3-12)
- lowpass_cg_iters: CG iterations for baseline low-pass filter (optimal: 24, range: 8-48)
- shared_params: If true, all layers share parameters (boolean)
- init_lam0: Initial lambda_0, asymmetric penalty weight
- init_lam1: Initial lambda_1, first-derivative penalty
- init_lam2: Initial lambda_2, second-derivative penalty
- init_r: Initial asymmetry ratio (optimal: 6.0, range: 2-12)
- init_step: Initial layer relaxation step size (optimal: 1.0, range: 0.05-2.0)
- init_output_gain: Initial output gain multiplier (optimal: 1.0, range: 0.1-10.0)

### model_fast (for model_type "lbeads_fast")
- N, d, fc, num_layers: Same as above
- lowpass_iterations: Number of iterated low-pass filter applications (optimal: 3, range: 1-5)
- lowpass_cg_iters: CG iterations for low-pass filter (optimal: 12, range: 8-48)
- init_lam0, init_lam1, init_lam2, init_r: Same as above
- init_step_size: Initial proximal gradient step size (optimal: 0.1, range: 0.01-1.0)

### training
- learning_rate: Adam optimizer LR (optimal: 1e-3, range: 5e-4 to 3e-3)
- batch_size: Signals per batch (optimal: 4-8, range: 2-32)
- num_samples: Total synthetic signals (optimal: 500, range: 200-2000)
- noise_level: Additive Gaussian noise std (optimal: 0.01, range: 0.001-0.05)
- train_ratio: Fraction for training (optimal: 0.8, range: 0.7-0.9)
- seed: Random seed (any integer)

### loss
- alpha_mse: Peak reconstruction MSE weight (anchor, optimal: 1.0)
- alpha_l1: L1 sparsity on peaks (optimal: 0.01, range: 0-0.1)
- alpha_tv: Total Variation on peaks (optimal: 0.01, range: 0-0.1)
- alpha_smooth: Baseline smoothness (optimal: 0.2, range: 0.05-1.0)
- alpha_neg: Negativity penalty on peaks (optimal: 0.5-2.0)
- alpha_baseline: Baseline reconstruction MSE (optimal: 0.5, range: 0.1-2.0)
- alpha_leakage: Baseline leakage penalty (optimal: 0.3-0.5, range: 0-2.0)
- alpha_ortho: Peak/baseline orthogonality (optimal: 0.1-0.2, range: 0-1.0)
- alpha_baseline_tv: TV on baseline (optimal: 0, range: 0-0.5)
- peak_mask_rel_threshold: Relative threshold for peak mask (optimal: 0.02)
- peak_mask_abs_min: Absolute minimum for peak mask (optimal: 1e-4)
- use_huber: Use Huber loss instead of MSE (boolean)
- huber_delta: Huber loss delta (optimal: 0.1, range: 0.01-1.0)

### stages
Array of training curriculum stages. Each stage has:
- name: Stage identifier string
- epochs: Number of epochs (positive integer)
- loss_config: Partial loss config overrides (omitted keys use top-level loss values)`

function buildPrompt(config: RunConfig): string {
  return `You are helping configure a training run for LBEADS-NET, a deep unrolled neural network for chromatographic baseline correction. It separates chromatographic signals into peak and baseline components.

Modify the configuration below based on my requirements. Return ONLY valid JSON — no markdown fences, no explanation, no commentary.

Current configuration:
${JSON.stringify(config, null, 2)}

Field descriptions:
${FIELD_DESCRIPTIONS}

Rules:
- Return the COMPLETE JSON object with all fields
- model_type must be "lbeads" or "lbeads_fast"
- Include "model" key always. Include "model_fast" only if model_type is "lbeads_fast"
- All numeric values must be non-negative
- stages must be a non-empty array`
}

function validateConfig(obj: unknown): { valid: true; config: Partial<RunConfig> } | { valid: false; error: string } {
  if (typeof obj !== "object" || obj === null || Array.isArray(obj)) {
    return { valid: false, error: "JSON must be an object" }
  }
  const cfg = obj as Record<string, unknown>

  if (cfg.model_type !== undefined && cfg.model_type !== "lbeads" && cfg.model_type !== "lbeads_fast") {
    return { valid: false, error: `Invalid model_type: "${cfg.model_type}". Must be "lbeads" or "lbeads_fast".` }
  }

  if (cfg.stages !== undefined) {
    if (!Array.isArray(cfg.stages) || cfg.stages.length === 0) {
      return { valid: false, error: "stages must be a non-empty array" }
    }
    for (let i = 0; i < cfg.stages.length; i++) {
      const s = cfg.stages[i]
      if (typeof s !== "object" || !s) {
        return { valid: false, error: `stages[${i}] must be an object` }
      }
      if (typeof s.epochs !== "number" || s.epochs <= 0) {
        return { valid: false, error: `stages[${i}].epochs must be a positive number` }
      }
    }
  }

  for (const section of ["model", "model_fast", "training", "loss"]) {
    if (cfg[section] && typeof cfg[section] === "object") {
      for (const [key, val] of Object.entries(cfg[section] as Record<string, unknown>)) {
        if (typeof val === "number" && val < 0) {
          return { valid: false, error: `${section}.${key} must be non-negative (got ${val})` }
        }
      }
    }
  }

  return { valid: true, config: cfg as Partial<RunConfig> }
}

interface LlmConfigDialogProps {
  getCurrentConfig: () => RunConfig
  onApply: (cfg: Partial<RunConfig>) => void
}

export function LlmConfigDialog({ getCurrentConfig, onApply }: LlmConfigDialogProps) {
  const [open, setOpen] = useState(false)
  const [jsonText, setJsonText] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  function handleCopyPrompt() {
    const prompt = buildPrompt(getCurrentConfig())
    navigator.clipboard.writeText(prompt)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  function handleApply() {
    setError(null)

    // Strip markdown fences if LLM wrapped the JSON
    let cleaned = jsonText.trim()
    if (cleaned.startsWith("```")) {
      cleaned = cleaned.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "")
    }

    let parsed: unknown
    try {
      parsed = JSON.parse(cleaned)
    } catch (e) {
      setError(`Invalid JSON: ${(e as Error).message}`)
      return
    }

    const result = validateConfig(parsed)
    if (!result.valid) {
      setError(result.error)
      return
    }

    onApply(result.config)
    setJsonText("")
    setOpen(false)
  }

  return (
    <Dialog open={open} onOpenChange={(v) => { setOpen(v); if (!v) { setError(null); setJsonText("") } }}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="gap-1.5 shrink-0">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
            <path d="M5 3v4" />
            <path d="M19 17v4" />
            <path d="M3 5h4" />
            <path d="M17 19h4" />
          </svg>
          Ask LLM
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Configure via LLM</DialogTitle>
          <DialogDescription>
            Copy the prompt below, give it to an LLM with your requirements, then paste the JSON response back here.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Step 1: Copy prompt */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <p className="text-sm font-medium">1. Copy prompt</p>
              <Button variant="outline" size="sm" onClick={handleCopyPrompt} className="h-7 text-xs">
                {copied ? "Copied!" : "Copy to Clipboard"}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Copies your current config + field descriptions as a prompt. Paste into ChatGPT, Claude, etc. with what you want to change.
            </p>
          </div>

          {/* Step 2: Paste JSON */}
          <div>
            <p className="text-sm font-medium mb-1.5">2. Paste JSON response</p>
            <textarea
              value={jsonText}
              onChange={(e) => { setJsonText(e.target.value); setError(null) }}
              placeholder="Paste the JSON configuration from your LLM here..."
              className="w-full h-48 rounded-md border bg-zinc-50 px-3 py-2 text-xs font-mono resize-y focus:outline-none focus:ring-2 focus:ring-ring"
              spellCheck={false}
            />
            {error && (
              <p className="mt-1.5 text-xs text-red-600 font-medium">{error}</p>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
          <Button onClick={handleApply} disabled={!jsonText.trim()}>Apply Configuration</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
