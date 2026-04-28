"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { generatePreview, getDataConfig, runBeadsTest } from "@/lib/api"
import type { SignalData, BeadsConfig, BeadsResult } from "@/lib/api"
import type { DataConfig } from "@/lib/types"
import { DEFAULT_DATA_CONFIG } from "@/lib/types"
import { SignalChart } from "@/app/components/signal-chart"

const DEFAULT_CONFIGS: BeadsConfig[] = [
  { d: 1, fc: 0.006, r: 6.0, lam0: 0.4, lam1: 4.0, lam2: 3.2, Nit: 30, label: "Default" },
  { d: 1, fc: 0.006, r: 6.0, lam0: 0.01, lam1: 0.5, lam2: 0.5, Nit: 30, label: "Low lambdas" },
  { d: 1, fc: 0.01, r: 6.0, lam0: 0.1, lam1: 1.0, lam2: 1.0, Nit: 50, label: "Higher fc" },
]

function NumField({ label, value, onChange, step }: {
  label: string; value: number; onChange: (v: number) => void; step?: number
}) {
  return (
    <div>
      <Label className="text-[10px]">{label}</Label>
      <Input
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="mt-0.5 h-7 text-xs"
        step={step}
      />
    </div>
  )
}

function BeadsConfigEditor({ config, onChange, index }: {
  config: BeadsConfig; onChange: (c: BeadsConfig) => void; index: number
}) {
  function set<K extends keyof BeadsConfig>(k: K, v: BeadsConfig[K]) {
    onChange({ ...config, [k]: v })
  }
  return (
    <div className="border bg-white p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold">Config {index + 1}</span>
        <Input
          value={config.label}
          onChange={(e) => set("label", e.target.value)}
          placeholder="Label"
          className="h-6 text-xs w-36"
        />
      </div>
      <div className="grid grid-cols-4 gap-2">
        <NumField label="d" value={config.d} onChange={(v) => set("d", v)} step={1} />
        <NumField label="fc" value={config.fc} onChange={(v) => set("fc", v)} step={0.001} />
        <NumField label="r" value={config.r} onChange={(v) => set("r", v)} step={0.5} />
        <NumField label="Nit" value={config.Nit} onChange={(v) => set("Nit", v)} step={5} />
        <NumField label="lam0" value={config.lam0} onChange={(v) => set("lam0", v)} step={0.01} />
        <NumField label="lam1" value={config.lam1} onChange={(v) => set("lam1", v)} step={0.1} />
        <NumField label="lam2" value={config.lam2} onChange={(v) => set("lam2", v)} step={0.1} />
      </div>
    </div>
  )
}

function ResultPane({ result, signal }: { result: BeadsResult; signal: SignalData }) {
  if (result.error) {
    return (
      <div className="border border-red-200 bg-red-50 p-3">
        <p className="text-xs font-semibold text-red-800">{result.label || "Config"}</p>
        <p className="text-xs text-red-700 mt-1">{result.error}</p>
      </div>
    )
  }
  return (
    <div className="border bg-white">
      <div className="px-3 py-2 border-b bg-zinc-50 flex items-center justify-between">
        <span className="text-xs font-semibold">{result.label || "Result"}</span>
        <div className="flex items-center gap-4 text-[10px] font-mono text-muted-foreground">
          <span>Peak MSE: {result.peak_mse?.toExponential(3)}</span>
          <span>Baseline MSE: {result.baseline_mse?.toExponential(3)}</span>
        </div>
      </div>
      <SignalChart
        y={signal.y}
        x_true={result.x_pred || []}
        f_true={result.f_pred || []}
        noise={signal.noise}
        label={`${result.label}: BEADS output (x_pred=red, f_pred=blue)`}
      />
    </div>
  )
}

export default function BeadsTestPage() {
  const [configs, setConfigs] = useState<BeadsConfig[]>(structuredClone(DEFAULT_CONFIGS))
  const [signal, setSignal] = useState<SignalData | null>(null)
  const [results, setResults] = useState<BeadsResult[]>([])
  const [generating, setGenerating] = useState(false)
  const [testing, setTesting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [N, setN] = useState(1024)
  const [seed, setSeed] = useState("42")
  const [dataConfig, setDataConfig] = useState<DataConfig>(structuredClone(DEFAULT_DATA_CONFIG))

  useEffect(() => {
    getDataConfig().then(setDataConfig).catch(() => {})
  }, [])

  async function handleGenerateSignal() {
    setGenerating(true)
    setError(null)
    setResults([])
    try {
      const result = await generatePreview({
        n_signals: 1,
        N,
        seed: seed ? parseInt(seed, 10) : null,
        baseline: dataConfig.baseline,
        peak_layers: dataConfig.peak_layers,
        noise: dataConfig.noise,
      })
      setSignal(result.signals[0])
    } catch (err) {
      setError(String(err))
    } finally {
      setGenerating(false)
    }
  }

  async function handleRunBeads() {
    if (!signal) return
    setTesting(true)
    setError(null)
    try {
      const res = await runBeadsTest(signal.y, signal.x_true, signal.f_true, configs)
      setResults(res.results)
    } catch (err) {
      setError(String(err))
    } finally {
      setTesting(false)
    }
  }

  function updateConfig(index: number, c: BeadsConfig) {
    setConfigs((prev) => prev.map((p, i) => (i === index ? c : p)))
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-4">
      <div>
        <h1 className="text-lg font-bold">BEADS Classic Test</h1>
        <p className="text-xs text-muted-foreground mt-1">
          Generate a synthetic signal, then test 3 classic BEADS configurations against it.
        </p>
      </div>

      {/* Signal generation controls */}
      <div className="border bg-zinc-50/50 p-4 space-y-3">
        <p className="text-sm font-semibold">Signal</p>
        <p className="text-xs text-muted-foreground">
          Uses the saved data generator config. Change it on the Data Generator page.
        </p>
        <div className="flex items-end gap-3">
          <NumField label="N" value={N} onChange={setN} step={512} />
          <div>
            <Label className="text-[10px]">seed</Label>
            <Input
              type="number"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="random"
              className="mt-0.5 h-7 text-xs w-24"
            />
          </div>
          <Button size="sm" onClick={handleGenerateSignal} disabled={generating}>
            {generating ? "Generating..." : "Generate Signal"}
          </Button>
        </div>
      </div>

      {/* Signal preview */}
      {signal && (
        <div>
          <p className="text-xs font-semibold text-muted-foreground mb-1">Input Signal (ground truth)</p>
          <SignalChart y={signal.y} x_true={signal.x_true} f_true={signal.f_true} noise={signal.noise} label="y = x + f + w" />
        </div>
      )}

      {/* BEADS configs */}
      <div className="space-y-2">
        <p className="text-sm font-semibold">BEADS Configurations</p>
        <div className="grid grid-cols-3 gap-3">
          {configs.map((c, i) => (
            <BeadsConfigEditor key={i} config={c} onChange={(v) => updateConfig(i, v)} index={i} />
          ))}
        </div>
      </div>

      {/* Run button */}
      <div className="flex items-center gap-4">
        <Button onClick={handleRunBeads} disabled={!signal || testing} className="px-8">
          {testing ? (
            <>
              <span className="inline-block h-3 w-3 border-2 border-zinc-300 border-t-zinc-600 rounded-full animate-spin mr-1.5" />
              Running BEADS...
            </>
          ) : (
            "Run BEADS on Signal"
          )}
        </Button>
        {error && <span className="text-xs text-red-600">{error}</span>}
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-3">
          <p className="text-sm font-semibold">Results</p>
          {results.map((r, i) => (
            <ResultPane key={i} result={r} signal={signal!} />
          ))}
        </div>
      )}
    </div>
  )
}
