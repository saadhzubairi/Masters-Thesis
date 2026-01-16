"""
Run Experiments Script
======================

Simple entry point to run experiment grids.

Usage:
    # Run quick test (1 experiment)
    python run_experiments.py --grid quick
    
    # Run Gaussian kernel tuning (42 experiments)
    python run_experiments.py --grid gaussian
    
    # Run first 5 experiments from depth grid
    python run_experiments.py --grid depth --max-experiments 5
    
    # Resume from experiment 10
    python run_experiments.py --grid gaussian --start-from 10
    
    # List all available grids
    python run_experiments.py --list-grids
"""

from worker import main

if __name__ == "__main__":
    main()
