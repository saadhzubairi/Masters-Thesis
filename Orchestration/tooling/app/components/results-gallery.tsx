"use client"

import { useState } from "react"
import { getFileUrl } from "@/lib/api"

interface ResultsGalleryProps {
  runId: string
  files: string[]
}

interface Section {
  title: string
  description: string
  images: string[]
}

function categorizeImages(files: string[]): Section[] {
  const images = files.filter((f) => f.endsWith(".png"))
  const sections: Section[] = []

  // Training plot
  const trainingPlots = images.filter((f) => f === "training_plot.png")
  if (trainingPlots.length > 0) {
    sections.push({
      title: "Training",
      description: "Loss curves and training metrics",
      images: trainingPlots,
    })
  }

  // Demo sections in a specific order with nice labels
  const demoSections: { prefix: string; title: string; description: string }[] = [
    {
      prefix: "demo/hybrid-snr/",
      title: "Hybrid Inference (SNR Threshold)",
      description: "Hybrid BEADS + LBEADS-NET with SNR-based stage selection",
    },
    {
      prefix: "demo/hybrid/",
      title: "Hybrid Inference (No Threshold)",
      description: "Hybrid BEADS + LBEADS-NET without noise thresholding",
    },
    {
      prefix: "demo/raw/",
      title: "Raw Net Output",
      description: "Direct LBEADS-NET forward pass (no classical refinement)",
    },
  ]

  for (const { prefix, title, description } of demoSections) {
    const sectionImages = images
      .filter((f) => f.startsWith(prefix))
      .sort((a, b) => {
        const order = (f: string) => {
          if (f.includes("before_after")) return 0
          if (f.includes("peak_candidates")) return 1
          if (f.includes("baseline_candidates")) return 2
          if (f.includes("metrics")) return 3
          if (f.includes("selected_output")) return 4
          if (f.includes("peaks")) return 5
          if (f.includes("baselines")) return 6
          return 7
        }
        return order(a) - order(b)
      })
    if (sectionImages.length > 0) {
      sections.push({ title, description, images: sectionImages })
    }
  }

  // Chromatogram comparison — split real vs synthetic
  const chromImages = images.filter((f) => f.startsWith("demo_chrom/"))
  const chromReal = chromImages.filter((f) => f.includes("demo_chromatogram"))
  const chromSynth = chromImages
    .filter((f) => f.includes("synthetic_beads_vs_lbeads"))
    .sort()
  if (chromReal.length > 0) {
    sections.push({
      title: "Real Chromatogram: BEADS vs LBEADS-NET",
      description: "Classical BEADS vs LBEADS-NET on legacy chromatogram data",
      images: chromReal,
    })
  }
  if (chromSynth.length > 0) {
    sections.push({
      title: "Synthetic: BEADS vs LBEADS-NET",
      description: "Classical BEADS vs LBEADS-NET on synthetic signals (with ground truth)",
      images: chromSynth,
    })
  }

  // Catch-all for anything else
  const categorized = new Set(sections.flatMap((s) => s.images))
  const uncategorized = images.filter((f) => !categorized.has(f))
  if (uncategorized.length > 0) {
    sections.push({
      title: "Other",
      description: "",
      images: uncategorized,
    })
  }

  return sections
}

function formatFileName(file: string): string {
  // Extract just the filename, strip extension, convert underscores to spaces
  const name = file.split("/").pop()?.replace(".png", "") ?? file
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

export function ResultsGallery({ runId, files }: ResultsGalleryProps) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const sections = categorizeImages(files)

  if (sections.length === 0) return null

  return (
    <div className="space-y-6">
      <h3 className="text-sm font-semibold">Results</h3>
      {sections.map((section) => (
        <div key={section.title}>
          <div className="mb-2">
            <h4 className="text-xs font-semibold text-foreground">{section.title}</h4>
            {section.description && (
              <p className="text-xs text-muted-foreground">{section.description}</p>
            )}
          </div>
          <div
            className={
              section.images.length === 1
                ? "grid grid-cols-1 max-w-lg gap-3"
                : section.images.length === 2
                  ? "grid grid-cols-2 gap-3"
                  : section.images.length <= 4
                    ? "grid grid-cols-2 lg:grid-cols-4 gap-3"
                    : "grid grid-cols-3 gap-3"
            }
          >
            {section.images.map((file) => (
              <div
                key={file}
                className="border rounded p-2 cursor-pointer hover:bg-zinc-50 transition-colors"
                onClick={() => setExpanded(file)}
              >
                <img
                  src={getFileUrl(runId, file)}
                  alt={file}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground mt-1 truncate">
                  {formatFileName(file)}
                </p>
              </div>
            ))}
          </div>
        </div>
      ))}
      {expanded && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setExpanded(null)}
        >
          <div
            className="bg-white p-4 rounded-lg max-w-5xl max-h-[90vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={getFileUrl(runId, expanded)}
              alt={expanded}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground mt-2">{expanded}</p>
          </div>
        </div>
      )}
    </div>
  )
}
