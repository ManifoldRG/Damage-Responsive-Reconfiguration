#!/usr/bin/env python3
"""
Monte Carlo Simulation Sweep Runner

Runs parameter sweeps for damage response algorithms and generates
CSV results and visualization graphs.

Usage:
    python run_monte_carlo_sweep.py [options]

Examples:
    python run_monte_carlo_sweep.py                          # Default: n=5-50, 100 trials
    python run_monte_carlo_sweep.py --n-max 100 --trials 1000
    python run_monte_carlo_sweep.py --n-min 10 --n-max 30 --trials 500
"""

import argparse
import csv
import json
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from src.monte_carlo import run_parameter_sweep, CONFIG_MODE_RANDOM, CONFIG_MODE_TREE


def main():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulation sweep for damage response algorithms"
    )
    parser.add_argument(
        "--n-min", type=int, default=5,
        help="Minimum number of modules (default: 5)"
    )
    parser.add_argument(
        "--n-max", type=int, default=50,
        help="Maximum number of modules (default: 50)"
    )
    parser.add_argument(
        "--faults", type=int, default=1,
        help="Number of faults per trial (default: 1)"
    )
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Number of trials per configuration (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="monte_carlo_results",
        help="Output directory (default: monte_carlo_results)"
    )
    parser.add_argument(
        "--no-graphs", action="store_true",
        help="Skip graph generation"
    )
    parser.add_argument(
        "--mode-2d", action="store_true",
        help="Use 2D mode instead of 3D"
    )
    parser.add_argument(
        "--fully-connected", action="store_true",
        help="Connect new modules to ALL adjacent modules (more branches, fewer moves). "
             "This is now the default behavior."
    )
    parser.add_argument(
        "--chain-like", action="store_true",
        help="Connect new modules to only ONE adjacent module (chain-like, more moves). "
             "Overrides the default fully-connected behavior."
    )
    parser.add_argument(
        "--tree", action="store_true",
        help="Use tree-based configuration generation (no cycles, each module has one parent)."
    )
    parser.add_argument(
        "--dynamic-faults", action="store_true",
        help="Use dynamic fault count: faults = floor(n/10) for each n. "
             "Ignores --faults when enabled."
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=-1,
        help="Number of parallel jobs (-1 for all cores, 1 for sequential). Default: -1"
    )

    args = parser.parse_args()

    # Determine fully_connected setting (default is True now)
    fully_connected = not args.chain_like

    # Determine config mode
    config_mode = CONFIG_MODE_TREE if args.tree else CONFIG_MODE_RANDOM

    # Generate timestamp and create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration for reproducibility
    config = {
        "timestamp": timestamp,
        "n_min": args.n_min,
        "n_max": args.n_max,
        "n_faults": args.faults if not args.dynamic_faults else "dynamic (n/10)",
        "dynamic_faults": args.dynamic_faults,
        "n_trials": args.trials,
        "seed": args.seed,
        "mode_2d": args.mode_2d,
        "fully_connected": fully_connected,
        "config_mode": config_mode,
        "n_jobs": args.jobs,
    }
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Determine connectivity description
    if config_mode == CONFIG_MODE_TREE:
        connectivity_desc = "tree (no cycles)"
    elif fully_connected:
        connectivity_desc = "fully-connected (default)"
    else:
        connectivity_desc = "chain-like"

    print("=" * 70)
    print("MONTE CARLO SIMULATION SWEEP")
    print("=" * 70)
    print("Parameters:")
    print(f"  Module range: n = {args.n_min} to {args.n_max}")
    if args.dynamic_faults:
        print(f"  Faults per trial: f = floor(n/10) [dynamic]")
    else:
        print(f"  Faults per trial: f = {args.faults}")
    print(f"  Trials per config: {args.trials}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {output_dir}")
    print(f"  Mode: {'2D' if args.mode_2d else '3D'}")
    print(f"  Config type: {config_mode}")
    print(f"  Connectivity: {connectivity_desc}")
    print(f"  Parallel jobs: {args.jobs} {'(all cores)' if args.jobs == -1 else ''}")
    print("=" * 70)
    print()

    # Run parameter sweep
    total_configs = args.n_max - args.n_min + 1
    total_trials = total_configs * args.trials
    print(f"Running {total_configs} configurations × {args.trials} trials = {total_trials:,} total trials")
    print("This may take a while...\n")

    sweep_results = run_parameter_sweep(
        n_range=(args.n_min, args.n_max),
        f_range=(args.faults, args.faults),
        n_trials=args.trials,
        seed=args.seed,
        mode_2d=args.mode_2d,
        fully_connected=fully_connected,
        verbose=True,
        config_mode=config_mode,
        dynamic_faults=args.dynamic_faults,
        n_jobs=args.jobs
    )

    print("\nSweep complete! Saving results...")

    # Extract data - handle both fixed and dynamic faults
    n_values = sorted(set(n for (n, f) in sweep_results.keys()))

    # Build lookup for results (handles dynamic faults where f varies with n)
    def get_result(n):
        """Get result for n, finding the matching (n, f) key."""
        for (n_key, f_key), res in sweep_results.items():
            if n_key == n:
                return res
        return None

    reconnection_rates = [get_result(n).reconnection_rate for n in n_values]
    shape_diffs_mean = [get_result(n).mean_shape_difference for n in n_values]
    shape_diffs_std = [get_result(n).std_shape_difference for n in n_values]

    # Save aggregated results to CSV
    csv_filename = os.path.join(output_dir, "sweep_summary.csv")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'n_modules', 'n_faults', 'n_trials', 'n_meaningful_trials',
            'mean_shape_difference', 'std_shape_difference',
            'reconnection_rate', 'full_restoration_rate',
            'mean_phase1_moves', 'mean_phase2_moves'
        ])
        for n in n_values:
            res = get_result(n)
            writer.writerow([
                res.n_modules, res.n_faults, res.n_trials, res.n_meaningful_trials,
                f'{res.mean_shape_difference:.6f}', f'{res.std_shape_difference:.6f}',
                f'{res.reconnection_rate:.4f}', f'{res.full_restoration_rate:.4f}',
                f'{res.mean_phase1_moves:.2f}', f'{res.mean_phase2_moves:.2f}'
            ])
    print(f"Saved: {csv_filename}")

    # Save full trial results to CSV for detailed analysis
    trials_csv_filename = os.path.join(output_dir, "trials.csv")
    with open(trials_csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'trial_id', 'n_modules', 'n_faults', 'seed',
            'restored', 'phase1_moves', 'phase2_moves',
            'shape_difference'
        ])
        for n in n_values:
            res = get_result(n)
            for trial in res.trials:
                writer.writerow([
                    trial.trial_id, trial.n_modules, trial.n_faults, trial.seed,
                    trial.restored, trial.phase1_moves, trial.phase2_moves,
                    trial.shape_difference if trial.shape_difference is not None else ''
                ])
    print(f"Saved: {trials_csv_filename}")

    # Generate graphs unless disabled
    if not args.no_graphs:
        generate_graphs(
            n_values, reconnection_rates,
            shape_diffs_mean, shape_diffs_std,
            args, output_dir,
            dynamic_faults=args.dynamic_faults
        )

    # Print summary table
    print_summary_table(n_values, sweep_results, args.faults, args.dynamic_faults, get_result)

    print(f"\nTotal trials run: {total_trials:,}")
    print(f"All results saved to: {output_dir}/")


def generate_graphs(n_values, reconnection_rates, shape_diffs_mean, shape_diffs_std,
                    args, output_dir, dynamic_faults=False):
    """Generate and save visualization graphs."""

    # Smooth the data
    sigma = 2
    reconnection_smooth = gaussian_filter1d(reconnection_rates, sigma=sigma)
    shape_diff_smooth = gaussian_filter1d(shape_diffs_mean, sigma=sigma)

    x_max = max(n_values) + 5

    # Determine fault description for titles
    if dynamic_faults:
        fault_desc = "f=n/10"
    else:
        fault_desc = f"f={args.faults}"

    # Create combined figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Monte Carlo Simulation Results ({fault_desc} faults, {args.trials} trials per n)',
        fontsize=14, fontweight='bold'
    )

    # Plot 1: Reconnection Rate
    ax1 = axes[0]
    ax1.scatter(n_values, reconnection_rates, alpha=0.4, color='blue', s=20, label='Raw data')
    ax1.plot(n_values, reconnection_smooth, color='blue', linewidth=2.5, label='Smoothed')
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='100%')
    ax1.set_xlabel('Number of Modules (n)', fontsize=11)
    ax1.set_ylabel('Reconnection Rate', fontsize=11)
    ax1.set_title('Reconnection Rate vs Structure Size', fontsize=12)
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')

    # Plot 2: Shape Difference
    ax2 = axes[1]
    ax2.fill_between(n_values,
                     np.maximum(0, np.array(shape_diffs_mean) - np.array(shape_diffs_std)),
                     np.minimum(1, np.array(shape_diffs_mean) + np.array(shape_diffs_std)),
                     alpha=0.2, color='orange')
    ax2.scatter(n_values, shape_diffs_mean, alpha=0.4, color='orange', s=20, label='Raw data')
    ax2.plot(n_values, shape_diff_smooth, color='darkorange', linewidth=2.5, label='Smoothed')
    ax2.set_xlabel('Number of Modules (n)', fontsize=11)
    ax2.set_ylabel('Shape Difference', fontsize=11)
    ax2.set_title('Shape Difference diff(P, Q) vs Structure Size', fontsize=12)
    ax2.set_xlim(0, x_max)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()

    # Save combined figure
    graph_filename = os.path.join(output_dir, "graphs_combined.png")
    plt.savefig(graph_filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {graph_filename}")

    # Save individual graphs
    metrics = [
        ('reconnection_rate', reconnection_rates, reconnection_smooth, None, 'blue', 'blue'),
        ('shape_difference', shape_diffs_mean, shape_diff_smooth, shape_diffs_std, 'orange', 'darkorange'),
    ]

    for metric_name, data, smooth_data, std_data, color, dark_color in metrics:
        fig2, ax = plt.subplots(figsize=(10, 7))
        if std_data:
            ax.fill_between(n_values,
                            np.maximum(0, np.array(data) - np.array(std_data)),
                            np.minimum(1, np.array(data) + np.array(std_data)),
                            alpha=0.2, color=color, label='±1 std')
        ax.scatter(n_values, data, alpha=0.4, color=color, s=30, label='Raw data')
        ax.plot(n_values, smooth_data, color=dark_color, linewidth=2.5, label='Smoothed (σ=2)')
        ax.set_xlabel('Number of Modules (n)', fontsize=13)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=13)
        ax.set_title(
            f'{metric_name.replace("_", " ").title()} vs Structure Size\n'
            f'({fault_desc} faults, {args.trials} trials per n)',
            fontsize=14
        )
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        if metric_name == 'reconnection_rate':
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)

        individual_filename = os.path.join(output_dir, f"graph_{metric_name}.png")
        plt.savefig(individual_filename, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved: {individual_filename}")

    plt.close('all')


def print_summary_table(n_values, sweep_results, n_faults, dynamic_faults=False, get_result=None):
    """Print a summary table of results."""
    print("\n" + "=" * 90)
    if dynamic_faults:
        print(f"SUMMARY TABLE (n={n_values[0]} to n={n_values[-1]}, f=n/10 dynamic)")
    else:
        print(f"SUMMARY TABLE (n={n_values[0]} to n={n_values[-1]}, f={n_faults})")
    print("=" * 90)
    print(f"{'n':>4} {'f':>3} {'Meaningful':>10} {'Reconn%':>8} {'ShapeDiff (mean±std)':>22}")
    print("-" * 90)

    # Print every 5th value (or adjust based on range)
    step = max(1, len(n_values) // 20)
    for i, n in enumerate(n_values):
        if i % step == 0 or n == n_values[-1]:
            if get_result:
                res = get_result(n)
            else:
                res = sweep_results[(n, n_faults)]
            print(
                f"{n:>4} {res.n_faults:>3} {res.n_meaningful_trials:>10} {res.reconnection_rate*100:>7.1f}% "
                f"{res.mean_shape_difference:>10.4f} ± {res.std_shape_difference:<8.4f}"
            )
    print("=" * 90)


if __name__ == "__main__":
    main()
