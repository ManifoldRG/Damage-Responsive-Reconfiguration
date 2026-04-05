#!/usr/bin/env python3
"""
Reconstruction Method Comparison Runner

Compares token-based (sidh-test) vs displacement-based (paper-code)
Phase 2 reconstruction using shared Phase 1 state for fair comparison.

Usage:
    python run_reconstruction_comparison.py
    python run_reconstruction_comparison.py --n-min 10 --n-max 50 --trials 100
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from src.monte_carlo import (
    run_comparison_trial, CONFIG_MODE_RANDOM, CONFIG_MODE_TREE,
    FAULT_MODE_RANDOM,
)


SUMMARY_HEADERS = [
    'n_modules', 'n_faults', 'n_trials', 'n_meaningful_trials',
    'reconstruction_method',
    'mean_shape_difference', 'std_shape_difference',
    'mean_shape_difference_phase1', 'std_shape_difference_phase1',
    'reconnection_rate',
    'mean_phase1_moves', 'mean_phase2_moves',
    'mean_total_moves', 'std_total_moves',
]

TRIALS_HEADERS = [
    'trial_id', 'n_modules', 'n_faults', 'seed',
    'reconstruction_method',
    'restored', 'phase1_moves', 'phase2_moves',
    'shape_difference', 'shape_difference_phase1',
    'phase1_iterations', 'total_moves',
]


def main():
    parser = argparse.ArgumentParser(
        description="Compare token-based vs displacement-based reconstruction"
    )
    parser.add_argument("--n-min", type=int, default=10)
    parser.add_argument("--n-max", type=int, default=50)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--faults", type=int, default=1)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="monte_carlo_results")
    parser.add_argument("--mode-2d", action="store_true")
    parser.add_argument("--tree", action="store_true")
    parser.add_argument("--dynamic-faults", action="store_true")
    parser.add_argument("--jobs", "-j", type=int, default=-1)
    parser.add_argument("--no-graphs", action="store_true")

    args = parser.parse_args()

    n_min = args.n_min
    n_max = args.n_max
    n_step = args.n_step
    n_trials = args.trials
    base_seed = args.seed
    mode_2d = args.mode_2d
    config_mode = CONFIG_MODE_TREE if args.tree else CONFIG_MODE_RANDOM
    dynamic_faults = args.dynamic_faults
    n_faults = args.faults
    n_jobs = args.jobs

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "timestamp": timestamp,
        "type": "reconstruction_comparison",
        "n_min": n_min, "n_max": n_max, "n_step": n_step,
        "n_faults": n_faults if not dynamic_faults else "dynamic (n/10)",
        "dynamic_faults": dynamic_faults,
        "n_trials": n_trials, "seed": base_seed,
        "mode_2d": mode_2d, "config_mode": config_mode,
        "n_jobs": n_jobs,
    }
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    all_n_values = list(range(n_min, n_max + 1, n_step))

    print("=" * 70)
    print("RECONSTRUCTION METHOD COMPARISON")
    print("=" * 70)
    print("Comparing: token-based (sidh-test) vs displacement-based (paper-code)")
    print(f"  Module range: n = {n_min} to {n_max} (step {n_step})")
    if dynamic_faults:
        print(f"  Faults: f = floor(n/10) [dynamic]")
    else:
        print(f"  Faults: f = {n_faults}")
    print(f"  Trials per n: {n_trials}")
    print(f"  Seed: {base_seed}")
    print(f"  Output: {output_dir}")
    print(f"  Jobs: {n_jobs}")
    print(f"  Phase 1 is shared — only Phase 2 differs between methods")
    print("=" * 70)
    print()

    total_trials = len(all_n_values) * n_trials
    print(f"Running {len(all_n_values)} n-values x {n_trials} trials = "
          f"{total_trials:,} trials (x2 Phase 2 methods each)")
    print()

    # Open CSV files
    summary_csv_path = os.path.join(output_dir, "sweep_summary.csv")
    trials_csv_path = os.path.join(output_dir, "trials.csv")

    summary_file = open(summary_csv_path, 'w', newline='')
    trials_file = open(trials_csv_path, 'w', newline='')

    try:
        summary_writer = csv.writer(summary_file)
        trials_writer = csv.writer(trials_file)
        summary_writer.writerow(SUMMARY_HEADERS)
        trials_writer.writerow(TRIALS_HEADERS)

        for n in tqdm(all_n_values, desc="Parameter sweep"):
            f = max(1, n // 10) if dynamic_faults else n_faults
            config_seed = base_seed + (n - n_min) * n_trials

            trial_args = [
                (n, f, config_seed + i, i, mode_2d, True, config_mode,
                 FAULT_MODE_RANDOM, "furthest", 2)
                for i in range(n_trials)
            ]

            if n_jobs == 1:
                pair_results = []
                for a in trial_args:
                    pair_results.append(run_comparison_trial(*a))
            else:
                pair_results = Parallel(n_jobs=n_jobs)(
                    delayed(run_comparison_trial)(*a) for a in trial_args
                )

            # Split into method-specific lists
            token_trials = [p[0] for p in pair_results]
            disp_trials = [p[1] for p in pair_results]

            for method, trials in [("token", token_trials), ("displacement", disp_trials)]:
                meaningful = [t for t in trials if t.phase1_moves > 0]
                successful = [t for t in meaningful if t.restored]
                n_meaningful = len(meaningful)

                if successful:
                    shape_diffs = [t.shape_difference for t in successful]
                    mean_sd = float(np.mean(shape_diffs))
                    std_sd = float(np.std(shape_diffs))
                    sd_p1 = [t.shape_difference_phase1 for t in successful
                             if t.shape_difference_phase1 is not None]
                    mean_sd_p1 = float(np.mean(sd_p1)) if sd_p1 else float('nan')
                    std_sd_p1 = float(np.std(sd_p1)) if sd_p1 else float('nan')
                else:
                    mean_sd = std_sd = float('nan')
                    mean_sd_p1 = std_sd_p1 = float('nan')

                if n_meaningful > 0:
                    reconn_rate = len(successful) / n_meaningful
                    mean_p1 = float(np.mean([t.phase1_moves for t in meaningful]))
                    mean_p2 = float(np.mean([t.phase2_moves for t in meaningful]))
                    mean_total = float(np.mean([t.total_moves for t in meaningful]))
                    std_total = float(np.std([t.total_moves for t in meaningful]))
                else:
                    reconn_rate = float('nan')
                    mean_p1 = mean_p2 = mean_total = std_total = float('nan')

                summary_writer.writerow([
                    n, f, n_trials, n_meaningful, method,
                    f'{mean_sd:.6f}', f'{std_sd:.6f}',
                    f'{mean_sd_p1:.6f}', f'{std_sd_p1:.6f}',
                    f'{reconn_rate:.4f}',
                    f'{mean_p1:.2f}', f'{mean_p2:.2f}',
                    f'{mean_total:.2f}', f'{std_total:.2f}',
                ])
                summary_file.flush()

                for trial in trials:
                    trials_writer.writerow([
                        trial.trial_id, trial.n_modules, trial.n_faults,
                        trial.seed, method,
                        trial.restored, trial.phase1_moves, trial.phase2_moves,
                        trial.shape_difference if trial.shape_difference is not None else '',
                        trial.shape_difference_phase1 if trial.shape_difference_phase1 is not None else '',
                        trial.phase1_iterations, trial.total_moves,
                    ])
                trials_file.flush()

    finally:
        summary_file.close()
        trials_file.close()

    print(f"\nSweep complete!")
    print(f"Saved: {summary_csv_path}")
    print(f"Saved: {trials_csv_path}")

    if not args.no_graphs:
        generate_comparison_graphs(summary_csv_path, output_dir, n_faults,
                                   n_trials, dynamic_faults)

    print(f"\nAll results saved to: {output_dir}/")


def generate_comparison_graphs(summary_csv_path, output_dir, n_faults,
                               n_trials, dynamic_faults):
    """Generate side-by-side comparison plots."""
    # Read data, split by method
    data = {"token": {}, "displacement": {}}

    with open(summary_csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                method = row['reconstruction_method']
                n = int(row['n_modules'])
                if method not in data:
                    continue
                data[method][n] = {
                    'shape_diff': float(row['mean_shape_difference']),
                    'std_shape_diff': float(row['std_shape_difference']),
                    'shape_diff_p1': float(row['mean_shape_difference_phase1']),
                    'reconn': float(row['reconnection_rate']),
                    'p1_moves': float(row['mean_phase1_moves']),
                    'p2_moves': float(row['mean_phase2_moves']),
                    'total_moves': float(row['mean_total_moves']),
                    'std_total': float(row['std_total_moves']),
                }
            except (ValueError, KeyError):
                continue

    # Get common n values
    common_n = sorted(set(data["token"].keys()) & set(data["displacement"].keys()))
    if not common_n:
        print("No common n values found, skipping graphs.")
        return

    tok = data["token"]
    disp = data["displacement"]

    fault_desc = "f=n/10" if dynamic_faults else f"f={n_faults}"
    sigma = 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Reconstruction Comparison: Token vs Displacement\n'
        f'({fault_desc}, {n_trials} trials per n)',
        fontsize=14, fontweight='bold'
    )

    # 1. Shape Difference comparison
    ax = axes[0, 0]
    tok_sd = [tok[n]['shape_diff'] for n in common_n]
    disp_sd = [disp[n]['shape_diff'] for n in common_n]
    ax.plot(common_n, gaussian_filter1d(tok_sd, sigma), color='blue',
            linewidth=2, label='Token-based')
    ax.plot(common_n, gaussian_filter1d(disp_sd, sigma), color='red',
            linewidth=2, label='Displacement-based')
    ax.scatter(common_n, tok_sd, alpha=0.2, color='blue', s=15)
    ax.scatter(common_n, disp_sd, alpha=0.2, color='red', s=15)
    ax.set_xlabel('Number of Modules (n)')
    ax.set_ylabel('Shape Difference')
    ax.set_title('Shape Difference After Both Phases')
    all_sd = tok_sd + disp_sd
    sd_max = max(all_sd) if all_sd else 1.0
    ax.set_ylim(0, min(1.0, sd_max * 1.3 + 0.02))
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Phase 2 moves comparison
    ax = axes[0, 1]
    tok_p2 = [tok[n]['p2_moves'] for n in common_n]
    disp_p2 = [disp[n]['p2_moves'] for n in common_n]
    ax.plot(common_n, gaussian_filter1d(tok_p2, sigma), color='blue',
            linewidth=2, label='Token-based')
    ax.plot(common_n, gaussian_filter1d(disp_p2, sigma), color='red',
            linewidth=2, label='Displacement-based')
    ax.scatter(common_n, tok_p2, alpha=0.2, color='blue', s=15)
    ax.scatter(common_n, disp_p2, alpha=0.2, color='red', s=15)
    ax.set_xlabel('Number of Modules (n)')
    ax.set_ylabel('Mean Phase 2 Moves')
    ax.set_title('Phase 2 Moves (Reconstruction Only)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Total moves comparison
    ax = axes[1, 0]
    tok_total = [tok[n]['total_moves'] for n in common_n]
    disp_total = [disp[n]['total_moves'] for n in common_n]
    ax.plot(common_n, gaussian_filter1d(tok_total, sigma), color='blue',
            linewidth=2, label='Token-based')
    ax.plot(common_n, gaussian_filter1d(disp_total, sigma), color='red',
            linewidth=2, label='Displacement-based')
    ax.scatter(common_n, tok_total, alpha=0.2, color='blue', s=15)
    ax.scatter(common_n, disp_total, alpha=0.2, color='red', s=15)
    ax.set_xlabel('Number of Modules (n)')
    ax.set_ylabel('Mean Total Moves')
    ax.set_title('Total Moves (Phase 1 + Phase 2)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Shape difference delta (token - displacement)
    ax = axes[1, 1]
    delta_sd = [tok[n]['shape_diff'] - disp[n]['shape_diff'] for n in common_n]
    delta_p2 = [tok[n]['p2_moves'] - disp[n]['p2_moves'] for n in common_n]
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.plot(common_n, gaussian_filter1d(delta_sd, sigma), color='purple',
            linewidth=2, label='Δ Shape Diff (tok-disp)')
    ax.scatter(common_n, delta_sd, alpha=0.2, color='purple', s=15)
    ax.set_xlabel('Number of Modules (n)')
    ax.set_ylabel('Δ (Token - Displacement)')
    ax.set_title('Difference: Negative = Token Better')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    graph_path = os.path.join(output_dir, "comparison_plots.png")
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {graph_path}")

    # Print summary table
    print("\n" + "=" * 85)
    print("COMPARISON SUMMARY")
    print("=" * 85)
    print(f"{'n':>4} {'Token SD':>10} {'Disp SD':>10} {'Δ SD':>8} "
          f"{'Token P2':>9} {'Disp P2':>9} {'Δ P2':>7}")
    print("-" * 85)

    step = max(1, len(common_n) // 15)
    for i, n in enumerate(common_n):
        if i % step == 0 or i == len(common_n) - 1:
            t = tok[n]
            d = disp[n]
            delta_s = t['shape_diff'] - d['shape_diff']
            delta_m = t['p2_moves'] - d['p2_moves']
            print(f"{n:>4} {t['shape_diff']:>10.4f} {d['shape_diff']:>10.4f} "
                  f"{delta_s:>+8.4f} "
                  f"{t['p2_moves']:>9.1f} {d['p2_moves']:>9.1f} "
                  f"{delta_m:>+7.1f}")
    print("=" * 85)

    # Overall averages
    avg_tok_sd = np.mean([tok[n]['shape_diff'] for n in common_n])
    avg_disp_sd = np.mean([disp[n]['shape_diff'] for n in common_n])
    avg_tok_p2 = np.mean([tok[n]['p2_moves'] for n in common_n])
    avg_disp_p2 = np.mean([disp[n]['p2_moves'] for n in common_n])
    print(f"\nOverall averages:")
    print(f"  Token:        SD={avg_tok_sd:.4f}, P2 moves={avg_tok_p2:.1f}")
    print(f"  Displacement: SD={avg_disp_sd:.4f}, P2 moves={avg_disp_p2:.1f}")
    winner_sd = "Token" if avg_tok_sd < avg_disp_sd else "Displacement"
    winner_p2 = "Token" if avg_tok_p2 < avg_disp_p2 else "Displacement"
    print(f"  Better shape:  {winner_sd}")
    print(f"  Fewer moves:   {winner_p2}")


if __name__ == "__main__":
    main()
