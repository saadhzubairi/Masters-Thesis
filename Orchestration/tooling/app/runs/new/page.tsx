import { ConfigForm } from "@/app/components/config-form"

export default function NewRunPage() {
  return (
    <div className="p-6 max-w-3xl">
      <h2 className="text-lg font-semibold mb-1">New Training Run</h2>
      <p className="text-sm text-muted-foreground mb-6">Configure hyperparameters and start training.</p>
      <ConfigForm />
    </div>
  )
}
