"use client"

import { useState, Fragment } from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import type { EpochData } from "@/lib/types"

interface EpochTableProps {
  epochs: EpochData[]
}

const COMPONENT_KEYS = [
  "reconstruction", "l1_sparsity", "total_variation", "baseline_smooth",
  "baseline_recon", "baseline_leakage", "peak_baseline_ortho", "non_negativity", "baseline_tv",
]

const PARAM_NAMES = ["lam0", "lam1", "lam2", "r", "step", "step_size"]

/** Group flat learned_params into rows per layer + globals. */
function groupParamsByLayer(params: Record<string, number> | undefined) {
  if (!params) return { layers: [], globals: {} as Record<string, number> }
  const layers: { index: number; values: Record<string, number> }[] = []
  const globals: Record<string, number> = {}
  const layerMap = new Map<number, Record<string, number>>()

  for (const [key, val] of Object.entries(params)) {
    const m = key.match(/^layer_(\d+)_(.+)$/)
    if (m) {
      const idx = parseInt(m[1])
      if (!layerMap.has(idx)) layerMap.set(idx, {})
      layerMap.get(idx)![m[2]] = val
    } else {
      globals[key] = val
    }
  }

  for (const [idx, values] of [...layerMap.entries()].sort((a, b) => a[0] - b[0])) {
    layers.push({ index: idx, values })
  }
  return { layers, globals }
}

export function EpochTable({ epochs }: EpochTableProps) {
  const [expanded, setExpanded] = useState<number | null>(null)
  const sorted = [...epochs].reverse()

  return (
    <div className="border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="text-xs w-16">Epoch</TableHead>
            <TableHead className="text-xs w-16">Stage</TableHead>
            <TableHead className="text-xs">Train Loss</TableHead>
            <TableHead className="text-xs">Test Loss</TableHead>
            <TableHead className="text-xs">Recon</TableHead>
            <TableHead className="text-xs">Baseline</TableHead>
            <TableHead className="text-xs">Leakage</TableHead>
            <TableHead className="text-xs">Ortho</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((ep) => (
            <Fragment key={ep.epoch}>
              <TableRow
                className={`cursor-pointer hover:bg-zinc-50 ${ep.epoch === sorted[0]?.epoch ? "bg-green-50" : ""}`}
                onClick={() => setExpanded(expanded === ep.epoch ? null : ep.epoch)}
              >
                <TableCell className="text-xs font-medium">{ep.epoch}</TableCell>
                <TableCell className="text-xs">{ep.stage}</TableCell>
                <TableCell className="text-xs font-mono">{ep.train_loss?.toFixed(6)}</TableCell>
                <TableCell className="text-xs font-mono">{ep.test_loss?.toFixed(6) ?? "—"}</TableCell>
                <TableCell className="text-xs font-mono">{ep.components?.reconstruction?.toFixed(6) ?? "—"}</TableCell>
                <TableCell className="text-xs font-mono">{ep.components?.baseline_recon?.toFixed(6) ?? "—"}</TableCell>
                <TableCell className="text-xs font-mono">{ep.components?.baseline_leakage?.toFixed(6) ?? "—"}</TableCell>
                <TableCell className="text-xs font-mono">{ep.components?.peak_baseline_ortho?.toFixed(6) ?? "—"}</TableCell>
              </TableRow>
              {expanded === ep.epoch && (
                <TableRow>
                  <TableCell colSpan={8} className="bg-zinc-50 p-4">
                    <div className="grid grid-cols-3 gap-2 text-xs mb-3">
                      <div className="font-semibold col-span-3 text-muted-foreground">All Loss Components</div>
                      {COMPONENT_KEYS.map((k) => (
                        <div key={k}>
                          <span className="text-muted-foreground">{k}:</span>{" "}
                          <span className="font-mono">{ep.components?.[k]?.toFixed(6) ?? "—"}</span>
                        </div>
                      ))}
                    </div>
                    {ep.learned_params && Object.keys(ep.learned_params).length > 0 && (() => {
                      const { layers, globals } = groupParamsByLayer(ep.learned_params)
                      // Collect all param names across layers for consistent columns
                      const colNames = PARAM_NAMES.filter(n =>
                        layers.some(l => n in l.values)
                      )
                      return (
                        <div className="border-t pt-3 text-xs">
                          <div className="font-semibold text-muted-foreground mb-2">Learned Parameters</div>
                          {Object.keys(globals).length > 0 && (
                            <div className="flex gap-4 mb-2">
                              {Object.entries(globals).map(([k, v]) => (
                                <span key={k}>
                                  <span className="text-muted-foreground">{k}:</span>{" "}
                                  <span className="font-mono font-medium">{v.toFixed(4)}</span>
                                </span>
                              ))}
                            </div>
                          )}
                          {layers.length > 0 && (
                            <div className="overflow-x-auto">
                              <table className="w-full text-xs">
                                <thead>
                                  <tr className="border-b">
                                    <th className="text-left py-1 pr-3 text-muted-foreground font-medium">Layer</th>
                                    {colNames.map(n => (
                                      <th key={n} className="text-right py-1 px-2 text-muted-foreground font-medium">{n}</th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {layers.map(l => (
                                    <tr key={l.index} className="border-b border-zinc-100">
                                      <td className="py-1 pr-3 font-medium">{l.index}</td>
                                      {colNames.map(n => (
                                        <td key={n} className="text-right py-1 px-2 font-mono">
                                          {l.values[n]?.toFixed(4) ?? "—"}
                                        </td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </div>
                      )
                    })()}
                  </TableCell>
                </TableRow>
              )}
            </Fragment>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
