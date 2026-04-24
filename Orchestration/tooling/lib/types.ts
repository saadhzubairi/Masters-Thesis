// lib/types.ts
export interface PeakLayerConfig {
  num_peaks_min: number
  num_peaks_max: number
  amplitude_min: number
  amplitude_max: number
  rise_width_min: number
  rise_width_max: number
  decay_width_min: number
  decay_width_max: number
  plateau_width_min: number
  plateau_width_max: number
  peak_shape_mode: "linear" | "exp" | "mixed"
}

export interface DataConfig {
  baseline: {
    smooth_sigma: number
    sine_amp: number
    sine_freq_min: number
    sine_freq_max: number
    baseline_amp_min: number
    baseline_amp_max: number
  }
  peak_layers: PeakLayerConfig[]
  noise: { noise_level: number }
}

export type ModelType = "lbeads" | "lbeads_fast" | "lbeads_v5"

export type DeviceType = "cpu" | "mps"

export interface RunConfig {
  name: string
  model_type: ModelType
  device?: DeviceType
  model: ModelConfig
  model_fast?: FastModelConfig
  training: TrainingConfig
  loss: LossConfig
  stages: StageConfig[]
  data?: DataConfig
}

export interface ModelConfig {
  N: number
  d: number
  fc: number
  num_layers: number
  solve_cg_iters: number
  lowpass_cg_iters: number
  shared_params: boolean
  init_lam0: number
  init_lam1: number
  init_lam2: number
  init_r: number
  init_step: number
  init_output_gain: number
}

export interface FastModelConfig {
  N: number
  d: number
  fc: number
  num_layers: number
  lowpass_iterations: number
  lowpass_cg_iters: number
  init_lam0: number
  init_lam1: number
  init_lam2: number
  init_r: number
  init_step_size: number
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
  model_type?: ModelType
  status: "pending" | "running" | "complete" | "failed" | "stopped" | "deleted"
  created_at: number
  epoch_count: number
  total_epochs: number
  summary: Record<string, number>
}

export interface DemoError {
  source: string
  message: string
}

export interface RunDetail extends RunSummary {
  config: RunConfig
  metrics: {
    epochs: EpochData[]
    summary: Record<string, number>
    errors?: DemoError[]
  }
  files: string[]
  logs?: string[]
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
    | "batch_progress"
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
  init_lam0: 0.002,
  init_lam1: 0.3,
  init_lam2: 0.3,
  init_r: 6.0,
  init_step: 1.0,
  init_output_gain: 1.0,
}

export const DEFAULT_FAST_MODEL_CONFIG: FastModelConfig = {
  N: 4096,
  d: 1,
  fc: 0.006,
  num_layers: 10,
  lowpass_iterations: 3,
  lowpass_cg_iters: 12,
  init_lam0: 0.4,
  init_lam1: 4.0,
  init_lam2: 3.2,
  init_r: 6.0,
  init_step_size: 0.1,
}

export const DEFAULT_V5_MODEL_CONFIG: FastModelConfig = {
  N: 4096,
  d: 1,
  fc: 0.006,
  num_layers: 20,
  lowpass_iterations: 1,
  lowpass_cg_iters: 12,
  init_lam0: 0.01,
  init_lam1: 0.5,
  init_lam2: 0.5,
  init_r: 6.0,
  init_step_size: 0.05,
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

export const DEFAULT_PEAK_LAYER: PeakLayerConfig = {
  num_peaks_min: 2,
  num_peaks_max: 6,
  amplitude_min: 0.2,
  amplitude_max: 1.0,
  rise_width_min: 10,
  rise_width_max: 80,
  decay_width_min: 20,
  decay_width_max: 200,
  plateau_width_min: 0,
  plateau_width_max: 10,
  peak_shape_mode: "mixed",
}

export const DEFAULT_DATA_CONFIG: DataConfig = {
  baseline: {
    smooth_sigma: 100,
    sine_amp: 0.1,
    sine_freq_min: 0.5,
    sine_freq_max: 2.0,
    baseline_amp_min: 0.08,
    baseline_amp_max: 0.35,
  },
  peak_layers: [{ ...DEFAULT_PEAK_LAYER }],
  noise: { noise_level: 0.01 },
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
