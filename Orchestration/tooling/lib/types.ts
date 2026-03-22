// lib/types.ts
export interface RunConfig {
  name: string
  model: ModelConfig
  training: TrainingConfig
  loss: LossConfig
  stages: StageConfig[]
}

export interface ModelConfig {
  N: number
  d: number
  fc: number
  num_layers: number
  solve_cg_iters: number
  lowpass_cg_iters: number
  shared_params: boolean
}

export interface TrainingConfig {
  learning_rate: number
  batch_size: number
  num_samples: number
  noise_level: number
  train_ratio: number
  seed: number
}

export interface LossConfig {
  alpha_mse: number
  alpha_l1: number
  alpha_tv: number
  alpha_smooth: number
  alpha_neg: number
  alpha_baseline: number
  alpha_leakage: number
  alpha_ortho: number
  alpha_baseline_tv: number
  peak_mask_rel_threshold: number
  peak_mask_abs_min: number
  use_huber: boolean
  huber_delta: number
}

export interface StageConfig {
  name: string
  epochs: number
  loss_config: Partial<LossConfig>
}

export interface RunSummary {
  id: string
  name: string
  status: "pending" | "running" | "complete" | "failed" | "stopped"
  created_at: number
  epoch_count: number
  summary: Record<string, number>
}

export interface RunDetail extends RunSummary {
  config: RunConfig
  metrics: {
    epochs: EpochData[]
    summary: Record<string, number>
  }
  files: string[]
}

export interface EpochData {
  epoch: number
  stage: string
  train_loss: number
  test_loss?: number
  components: Record<string, number>
  learned_params?: Record<string, number>
  elapsed_s?: number
}

export interface SSEEvent {
  type:
    | "started"
    | "epoch"
    | "stage_change"
    | "training_done"
    | "demo_started"
    | "demo_done"
    | "demo_error"
    | "error"
    | "complete"
  [key: string]: unknown
}

export const DEFAULT_MODEL_CONFIG: ModelConfig = {
  N: 4096,
  d: 1,
  fc: 0.006,
  num_layers: 5,
  solve_cg_iters: 5,
  lowpass_cg_iters: 24,
  shared_params: false,
}

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  learning_rate: 1e-3,
  batch_size: 4,
  num_samples: 500,
  noise_level: 0.01,
  train_ratio: 0.8,
  seed: 42,
}

export const DEFAULT_LOSS_CONFIG: LossConfig = {
  alpha_mse: 1.0,
  alpha_l1: 0.01,
  alpha_tv: 0.01,
  alpha_smooth: 0.2,
  alpha_neg: 2.0,
  alpha_baseline: 0.5,
  alpha_leakage: 0.5,
  alpha_ortho: 0.2,
  alpha_baseline_tv: 0.0,
  peak_mask_rel_threshold: 0.02,
  peak_mask_abs_min: 1e-4,
  use_huber: false,
  huber_delta: 0.1,
}

export const DEFAULT_STAGES: StageConfig[] = [
  {
    name: "A_peak_recon",
    epochs: 5,
    loss_config: {
      alpha_mse: 1.0,
      alpha_l1: 0,
      alpha_tv: 0,
      alpha_smooth: 0,
      alpha_neg: 0,
      alpha_baseline: 0,
      alpha_leakage: 0,
      alpha_ortho: 0,
      alpha_baseline_tv: 0,
    },
  },
  {
    name: "B_full_loss",
    epochs: 20,
    loss_config: {},
  },
]
