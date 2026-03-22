"use client"

import { useState } from "react"
import { getFileUrl } from "@/lib/api"

interface ResultsGalleryProps {
  runId: string
  files: string[]
}

export function ResultsGallery({ runId, files }: ResultsGalleryProps) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const images = files.filter((f) => f.endsWith(".png"))

  if (images.length === 0) return null

  return (
    <div>
      <h3 className="text-sm font-semibold mb-3">Results</h3>
      <div className="grid grid-cols-3 gap-3">
        {images.map((file) => (
          <div key={file} className="border p-2 cursor-pointer hover:bg-zinc-50" onClick={() => setExpanded(file)}>
            <img src={getFileUrl(runId, file)} alt={file} className="w-full" />
            <p className="text-xs text-muted-foreground mt-1 truncate">{file}</p>
          </div>
        ))}
      </div>
      {expanded && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setExpanded(null)}>
          <div className="bg-white p-4 max-w-4xl max-h-[90vh] overflow-auto" onClick={(e) => e.stopPropagation()}>
            <img src={getFileUrl(runId, expanded)} alt={expanded} className="w-full" />
            <p className="text-xs text-muted-foreground mt-2">{expanded}</p>
          </div>
        </div>
      )}
    </div>
  )
}
