"""
Synthetic Data Test Package

Experimental framework for comparing BEADS and LBEADS-NET
on synthetic signals with known ground truth.
"""

from .synthetic_data_generator import (
    SyntheticSignal,
    SyntheticDataGenerator,
    save_dataset,
    load_dataset
)

from .metrics import (
    EvaluationResult,
    compute_mse,
    compute_snr_input,
    compute_snr_output,
    compute_delta_snr,
    evaluate_single,
    aggregate_results,
    print_table1,
    generate_latex_table
)

__all__ = [
    'SyntheticSignal',
    'SyntheticDataGenerator',
    'save_dataset',
    'load_dataset',
    'EvaluationResult',
    'compute_mse',
    'compute_snr_input',
    'compute_snr_output',
    'compute_delta_snr',
    'evaluate_single',
    'aggregate_results',
    'print_table1',
    'generate_latex_table'
]
