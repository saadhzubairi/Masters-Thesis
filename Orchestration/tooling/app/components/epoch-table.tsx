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

const PARAM_KEYS = ["lam0", "lam1", "lam2", "r", "step", "output_gain"]

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
                    {ep.learned_params && (
                      <div className="grid grid-cols-3 gap-2 text-xs border-t pt-3">
                        <div className="font-semibold col-span-3 text-muted-foreground">Learned Parameters</div>
                        {PARAM_KEYS.map((k) => (
                          <div key={k}>
                            <span className="text-muted-foreground">{k}:</span>{" "}
                            <span className="font-mono">{ep.learned_params?.[k]?.toFixed(4) ?? "—"}</span>
                          </div>
                        ))}
                      </div>
                    )}
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
