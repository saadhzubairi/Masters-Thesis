"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { SignalChart } from "@/app/components/signal-chart"
import { generatePreview, getDataConfig, saveDataConfig } from "@/lib/api"
import type { SignalData } from "@/lib/api"
import type { PeakLayerConfig, DataConfig } from "@/lib/types"
import { DEFAULT_PEAK_LAYER, DEFAULT_DATA_CONFIG } from "@/lib/types"

function Section({
  title,
  defaultOpen = true,
  actions,
  children,
}: {
  title: string
  defaultOpen?: boolean
  actions?: React.ReactNode
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border bg-zinc-50/50">
      <div className="flex items-center justify-between px-4 py-3">
        <Button
          variant="ghost"
          onClick={() => setOpen(!open)}
          className="flex items-center gap-2 p-0 h-auto hover:bg-transparent"
        >
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
          <span className="text-sm font-semibold">{title}</span>
        </Button>
        {actions && <div className="flex items-center gap-1">{actions}</div>}
      </div>
      {open && (
        <div className="px-4 pb-4 pt-2 border-t bg-white">{children}</div>
      )}
    </div>
  )
}

function RangeInput({
  label,
  min,
  max,
  onMinChange,
  onMaxChange,
  step,
}: {
  label: string
  min: number
  max: number
  onMinChange: (v: number) => void
  onMaxChange: (v: number) => void
  step?: number
}) {
  return (
    <div>
      <Label className="text-xs">{label}</Label>
      <div className="flex gap-1 mt-1">
        <Input
          type="number"
          value={min}
          onChange={(e) => onMinChange(parseFloat(e.target.value) || 0)}
          className="h-8 text-xs"
          step={step}
          placeholder="min"
        />
        <Input
          type="number"
          value={max}
          onChange={(e) => onMaxChange(parseFloat(e.target.value) || 0)}
          className="h-8 text-xs"
          step={step}
          placeholder="max"
        />
      </div>
    </div>
  )
}

function NumInput({
  label,
  value,
  onChange,
  step,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  step?: number
}) {
  return (
    <div>
      <Label className="text-xs">{label}</Label>
      <Input
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="mt-1 h-8 text-xs"
        step={step}
      />
    </div>
  )
}

function PeakLayerEditor({
  index,
  layer,
  onChange,
  onRemove,
  canRemove,
}: {
  index: number
  layer: PeakLayerConfig
  onChange: (updated: PeakLayerConfig) => void
  onRemove: () => void
  canRemove: boolean
}) {
  function update<K extends keyof PeakLayerConfig>(key: K, val: PeakLayerConfig[K]) {
    onChange({ ...layer, [key]: val })
  }

  return (
    <Section
      title={`Peak Layer ${index + 1}`}
      actions={
        canRemove ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={onRemove}
            className="h-6 px-2 text-xs text-red-600 hover:text-red-700 hover:bg-red-50"
          >
            Remove
          </Button>
        ) : undefined
      }
    >
      <div className="grid grid-cols-2 gap-3">
        <RangeInput
          label="num_peaks_range"
          min={layer.num_peaks_min}
          max={layer.num_peaks_max}
          onMinChange={(v) => update("num_peaks_min", v)}
          onMaxChange={(v) => update("num_peaks_max", v)}
          step={1}
        />
        <RangeInput
          label="amplitude_range"
          min={layer.amplitude_min}
          max={layer.amplitude_max}
          onMinChange={(v) => update("amplitude_min", v)}
          onMaxChange={(v) => update("amplitude_max", v)}
          step={0.05}
        />
        <RangeInput
          label="rise_width_range"
          min={layer.rise_width_min}
          max={layer.rise_width_max}
          onMinChange={(v) => update("rise_width_min", v)}
          onMaxChange={(v) => update("rise_width_max", v)}
          step={5}
        />
        <RangeInput
          label="decay_width_range"
          min={layer.decay_width_min}
          max={layer.decay_width_max}
          onMinChange={(v) => update("decay_width_min", v)}
          onMaxChange={(v) => update("decay_width_max", v)}
          step={5}
        />
        <RangeInput
          label="plateau_width_range"
          min={layer.plateau_width_min}
          max={layer.plateau_width_max}
          onMinChange={(v) => update("plateau_width_min", v)}
          onMaxChange={(v) => update("plateau_width_max", v)}
          step={1}
        />
        <div>
          <Label className="text-xs">peak_shape_mode</Label>
          <Select
            value={layer.peak_shape_mode}
            onValueChange={(v) => update("peak_shape_mode", v as PeakLayerConfig["peak_shape_mode"])}
          >
            <SelectTrigger className="mt-1 h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="linear">linear</SelectItem>
              <SelectItem value="exp">exp</SelectItem>
              <SelectItem value="mixed">mixed</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
    </Section>
  )
}

export default function DataGeneratorPage() {
  const [baseline, setBaseline] = useState({ ...DEFAULT_DATA_CONFIG.baseline })
  const [peakLayers, setPeakLayers] = useState<PeakLayerConfig[]>(
    structuredClone(DEFAULT_DATA_CONFIG.peak_layers)
  )
  const [noise, setNoise] = useState({ ...DEFAULT_DATA_CONFIG.noise })
  const [N, setN] = useState(4096)
  const [seed, setSeed] = useState("")
  const [signals, setSignals] = useState<SignalData[]>([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saveMsg, setSaveMsg] = useState<string | null>(null)

  useEffect(() => {
    getDataConfig()
      .then((cfg: DataConfig) => {
        if (cfg.baseline) setBaseline(cfg.baseline)
        if (cfg.peak_layers?.length) setPeakLayers(cfg.peak_layers)
        if (cfg.noise) setNoise(cfg.noise)
      })
      .catch(() => {})
  }, [])

  async function handleGenerate() {
    setLoading(true)
    setError(null)
    try {
      const result = await generatePreview({
        n_signals: 5,
        N,
        seed: seed ? parseInt(seed, 10) : null,
        baseline,
        peak_layers: peakLayers,
        noise,
      })
      setSignals(result.signals)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  async function handleSave() {
    setSaving(true)
    setSaveMsg(null)
    try {
      await saveDataConfig({ baseline, peak_layers: peakLayers, noise })
      setSaveMsg("Saved")
      setTimeout(() => setSaveMsg(null), 2000)
    } catch (err) {
      setSaveMsg(`Error: ${err}`)
    } finally {
      setSaving(false)
    }
  }

  function addPeakLayer() {
    setPeakLayers((prev) => [...prev, { ...DEFAULT_PEAK_LAYER }])
  }

  function removePeakLayer(index: number) {
    setPeakLayers((prev) => prev.filter((_, i) => i !== index))
  }

  function updatePeakLayer(index: number, updated: PeakLayerConfig) {
    setPeakLayers((prev) => prev.map((l, i) => (i === index ? updated : l)))
  }

  function updateBaseline<K extends keyof typeof baseline>(key: K, val: (typeof baseline)[K]) {
    setBaseline((prev) => ({ ...prev, [key]: val }))
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold">Synthetic Data Generator</h1>
          <p className="text-xs text-muted-foreground mt-1">
            Fine-tune signal generation parameters and preview samples.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {saveMsg && (
            <span className={`text-xs ${saveMsg.startsWith("Error") ? "text-red-600" : "text-green-600"}`}>
              {saveMsg}
            </span>
          )}
          <Button variant="outline" size="sm" onClick={handleSave} disabled={saving}>
            {saving ? "Saving..." : "Save Config"}
          </Button>
        </div>
      </div>

      {/* Baseline + Noise side by side */}
      <div className="grid grid-cols-2 gap-4">
        <Section title="Baseline">
          <div className="grid grid-cols-2 gap-3">
            <NumInput
              label="smooth_sigma"
              value={baseline.smooth_sigma}
              onChange={(v) => updateBaseline("smooth_sigma", v)}
              step={10}
            />
            <NumInput
              label="sine_amp"
              value={baseline.sine_amp}
              onChange={(v) => updateBaseline("sine_amp", v)}
              step={0.01}
            />
            <RangeInput
              label="sine_freq_range"
              min={baseline.sine_freq_min}
              max={baseline.sine_freq_max}
              onMinChange={(v) => updateBaseline("sine_freq_min", v)}
              onMaxChange={(v) => updateBaseline("sine_freq_max", v)}
              step={0.1}
            />
            <RangeInput
              label="baseline_amp_range"
              min={baseline.baseline_amp_min}
              max={baseline.baseline_amp_max}
              onMinChange={(v) => updateBaseline("baseline_amp_min", v)}
              onMaxChange={(v) => updateBaseline("baseline_amp_max", v)}
              step={0.01}
            />
          </div>
        </Section>

        <div className="space-y-4">
          <Section title="Noise">
            <div>
              <Label className="text-xs">noise_level</Label>
              <div className="flex items-center gap-3 mt-1">
                <input
                  type="range"
                  min={0}
                  max={0.1}
                  step={0.001}
                  value={noise.noise_level}
                  onChange={(e) => setNoise({ noise_level: parseFloat(e.target.value) })}
                  className="flex-1 accent-foreground"
                />
                <span className="text-xs font-mono w-12 text-right">
                  {noise.noise_level.toFixed(3)}
                </span>
              </div>
            </div>
          </Section>

          <Section title="Global">
            <div className="grid grid-cols-2 gap-3">
              <NumInput label="N (signal length)" value={N} onChange={setN} step={512} />
              <div>
                <Label className="text-xs">seed (optional)</Label>
                <Input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                  placeholder="random"
                  className="mt-1 h-8 text-xs"
                />
              </div>
            </div>
          </Section>
        </div>
      </div>

      {/* Peak layers */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Peak Layers</h2>
          <Button variant="outline" size="sm" onClick={addPeakLayer}>
            + Add Layer
          </Button>
        </div>
        <div className="grid grid-cols-2 gap-4">
          {peakLayers.map((layer, i) => (
            <PeakLayerEditor
              key={i}
              index={i}
              layer={layer}
              onChange={(updated) => updatePeakLayer(i, updated)}
              onRemove={() => removePeakLayer(i)}
              canRemove={peakLayers.length > 1}
            />
          ))}
        </div>
      </div>

      {/* Generate button */}
      <div className="flex items-center gap-4">
        <Button onClick={handleGenerate} disabled={loading} className="px-8">
          {loading ? "Generating..." : "Generate Preview"}
        </Button>
        {error && <span className="text-xs text-red-600">{error}</span>}
      </div>

      {/* Preview charts */}
      {signals.length > 0 && (
        <div className="space-y-2">
          {signals.map((sig, i) => (
            <SignalChart
              key={i}
              y={sig.y}
              x_true={sig.x_true}
              f_true={sig.f_true}
              noise={sig.noise}
              label={`Signal ${i + 1}`}
            />
          ))}
        </div>
      )}
    </div>
  )
}
