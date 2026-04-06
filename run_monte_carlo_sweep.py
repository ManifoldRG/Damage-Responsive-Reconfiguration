#!/usr/bin/env python3
"""
Monte Carlo Simulation Sweep Runner

Runs parameter sweeps for damage response algorithms and generates
CSV results and visualization graphs. Results are saved incrementally
after each n-value completes, enabling crash recovery via --resume.

Usage:
    python run_monte_carlo_sweep.py [options]

Examples:
    python run_monte_carlo_sweep.py                          # Default: n=5-50, 100 trials
    python run_monte_carlo_sweep.py --n-max 100 --trials 1000
    python run_monte_carlo_sweep.py --n-min 10 --n-max 30 --trials 500
    python run_monte_carlo_sweep.py --resume monte_carlo_results/20260301_143000/
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from src.monte_carlo import (
    run_monte_carlo, CONFIG_MODE_RANDOM, CONFIG_MODE_TREE,
    FAULT_MODE_RANDOM, FAULT_MODE_CLUSTER, FAULT_MODE_RANDOM_CLUSTERS, FAULT_MODE_LOCALIZED
)


SUMMARY_HEADERS = [
    'n_modules', 'n_faults', 'n_trials', 'n_meaningful_trials',
    'mean_shape_difference', 'std_shape_difference',
    'mean_shape_difference_phase1', 'std_shape_difference_phase1',
    'reconnection_rate', 'std_reconnection_rate',
    'full_restoration_rate',
    'mean_phase1_moves', 'mean_phase2_moves',
    'mean_steps_to_reconnection', 'std_steps_to_reconnection',
    'mean_total_moves', 'std_total_moves',
    'mean_token_transmissions', 'std_token_transmissions',
    'fault_mode', 'fault_pct', 'token_strategy', 'safety_radius'
]

TRIALS_HEADERS = [
    'trial_id', 'n_modules', 'n_faults', 'seed',
    'restored', 'phase1_moves', 'phase2_moves',
    'shape_difference', 'shape_difference_phase1',
    'phase1_iterations', 'total_moves',
    'token_transmissions', 'fault_mode', 'token_strategy', 'safety_radius'
]

TOKEN_STRATEGIES = ["furthest", "nearest", "random"]
SAFETY_RADII = [2, 3, 4]


def load_completed_n_values(output_dir):
    """Read sweep_summary.csv and return set of completed n values."""
    csv_path = os.path.join(output_dir, "sweep_summary.csv")
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(int(row['n_modules']))
            except (ValueError, KeyError):
                continue
    return completed


def load_resume_config(resume_dir):
    """Load config.json from a previous run directory."""
    config_path = os.path.join(resume_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: No config.json found in {resume_dir}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, 'r') as f:
        return json.load(f)


def warn_arg_conflicts(args, config):
    """Print warnings if non-default CLI args differ from resumed config."""
    checks = [
        ('n_min', '--n-min', args.n_min, config['n_min'], 5),
        ('n_max', '--n-max', args.n_max, config['n_max'], 50),
        ('n_trials', '--trials', args.trials, config['n_trials'], 100),
        ('seed', '--seed', args.seed, config['seed'], 42),
    ]
    conflicts = []
    for key, flag, cli_val, cfg_val, default in checks:
        if cli_val != default and cli_val != cfg_val:
            conflicts.append(f"  {flag}: CLI={cli_val}, config={cfg_val}")
    if conflicts:
        print("Warning: CLI args differ from resumed config (using config values):")
        for c in conflicts:
            print(c)
        print()


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
        "--n-step", type=int, default=1,
        help="Step size between n values (default: 1)"
    )
    parser.add_argument(
        "--n-values", type=int, nargs="+", default=None,
        help="Explicit list of n values to sweep (overrides --n-min/--n-max/--n-step)"
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
    parser.add_argument(
        "--resume", type=str, default=None,
        metavar="PATH",
        help="Resume from an existing output directory, skipping n values "
             "already present in sweep_summary.csv"
    )
    parser.add_argument(
        "--cluster-faults", action="store_true",
        help="Run cluster failure sweep: for each n, test cluster sizes 2,3,4,5"
    )
    parser.add_argument(
        "--dynamic-pct", action="store_true",
        help="Run dynamic percentage fault sweep: 10%%, 20%%, 30%% x 3 spatial patterns"
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run token selection strategy ablation: sweep furthest, nearest, random for each config"
    )
    parser.add_argument(
        "--ablation-hops", action="store_true",
        help="Run safety radius ablation: sweep hop radii 2, 3, 4 for is_movable() check"
    )
    parser.add_argument(
        "--reconstruction", type=str, default="displacement",
        choices=["token", "displacement"],
        help="Phase 2 reconstruction method (default: displacement)"
    )

    args = parser.parse_args()

    # --- Resume mode: load config from previous run ---
    if args.resume:
        resume_dir = args.resume
        if not os.path.isdir(resume_dir):
            print(f"Error: Resume directory does not exist: {resume_dir}", file=sys.stderr)
            sys.exit(1)

        config = load_resume_config(resume_dir)
        warn_arg_conflicts(args, config)

        n_min = config['n_min']
        n_max = config['n_max']
        n_trials = config['n_trials']
        base_seed = config['seed']
        mode_2d = config.get('mode_2d', False)
        fully_connected = config.get('fully_connected', True)
        config_mode = config.get('config_mode', CONFIG_MODE_RANDOM)
        dynamic_faults = config.get('dynamic_faults', False)
        n_faults_cfg = config.get('n_faults', 1)
        n_faults = n_faults_cfg if isinstance(n_faults_cfg, int) else 1
        n_jobs = config.get('n_jobs', -1)
        cluster_faults = config.get('cluster_faults', False)
        dynamic_pct = config.get('dynamic_pct', False)
        ablation = config.get('ablation', False)
        ablation_hops = config.get('ablation_hops', False)
        n_step = config.get('n_step', 1)
        reconstruction_method = config.get('reconstruction_method', 'displacement')
        output_dir = resume_dir

        completed = load_completed_n_values(output_dir)
        print(f"Resuming from: {output_dir}")
        if completed:
            sorted_done = sorted(completed)
            preview = sorted_done[:5]
            suffix = f"...{sorted_done[-1]}" if len(sorted_done) > 5 else ""
            print(f"Already completed: {len(completed)} n-values "
                  f"({', '.join(map(str, preview))}{suffix})")
        print()
    else:
        # --- Fresh run ---
        n_min = args.n_min
        n_max = args.n_max
        n_trials = args.trials
        base_seed = args.seed
        mode_2d = args.mode_2d
        fully_connected = not args.chain_like
        config_mode = CONFIG_MODE_TREE if args.tree else CONFIG_MODE_RANDOM
        dynamic_faults = args.dynamic_faults
        n_faults = args.faults
        n_jobs = args.jobs
        cluster_faults = args.cluster_faults
        dynamic_pct = args.dynamic_pct
        ablation = args.ablation
        ablation_hops = args.ablation_hops
        n_step = args.n_step
        reconstruction_method = args.reconstruction

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)

        config = {
            "timestamp": timestamp,
            "n_min": n_min,
            "n_max": n_max,
            "n_faults": n_faults if not dynamic_faults else "dynamic (n/10)",
            "dynamic_faults": dynamic_faults,
            "n_trials": n_trials,
            "seed": base_seed,
            "mode_2d": mode_2d,
            "fully_connected": fully_connected,
            "config_mode": config_mode,
            "n_jobs": n_jobs,
            "cluster_faults": cluster_faults,
            "dynamic_pct": dynamic_pct,
            "ablation": ablation,
            "ablation_hops": ablation_hops,
            "n_step": n_step,
            "reconstruction_method": reconstruction_method,
        }
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        completed = set()

    # Determine connectivity description
    if config_mode == CONFIG_MODE_TREE:
        connectivity_desc = "tree (no cycles)"
    elif fully_connected:
        connectivity_desc = "fully-connected (default)"
    else:
        connectivity_desc = "chain-like"

    no_graphs = args.no_graphs
    if args.n_values:
        all_n_values = sorted(args.n_values)
    else:
        all_n_values = list(range(n_min, n_max + 1, n_step))
    remaining = [n for n in all_n_values if n not in completed]

    print("=" * 70)
    print("MONTE CARLO SIMULATION SWEEP")
    print("=" * 70)
    print("Parameters:")
    print(f"  Module range: n = {n_min} to {n_max}")
    if dynamic_faults:
        print(f"  Faults per trial: f = floor(n/10) [dynamic]")
    else:
        print(f"  Faults per trial: f = {n_faults}")
    print(f"  Trials per config: {n_trials}")
    print(f"  Random seed: {base_seed}")
    print(f"  Output directory: {output_dir}")
    print(f"  Mode: {'2D' if mode_2d else '3D'}")
    print(f"  Config type: {config_mode}")
    print(f"  Connectivity: {connectivity_desc}")
    print(f"  Parallel jobs: {n_jobs} {'(all cores)' if n_jobs == -1 else ''}")
    print(f"  Reconstruction: {reconstruction_method}")
    if cluster_faults:
        print(f"  Cluster faults: enabled (cluster sizes 2,3,4,5)")
    if dynamic_pct:
        print(f"  Dynamic pct: enabled (10%,20%,30% x random,random_clusters,localized)")
    if ablation:
        print(f"  Ablation: enabled (token strategies: furthest, nearest, random)")
    if ablation_hops:
        print(f"  Ablation-hops: enabled (safety radii: {SAFETY_RADII})")
    if completed:
        print(f"  Resuming: {len(completed)}/{len(all_n_values)} n-values already done")
    print("=" * 70)
    print()

    # Calculate multiplier for new modes
    configs_per_n = 1
    if cluster_faults:
        configs_per_n = 4  # cluster sizes 2,3,4,5
    elif dynamic_pct:
        configs_per_n = 9  # 3 pcts x 3 patterns
    if ablation:
        configs_per_n *= len(TOKEN_STRATEGIES)  # multiply by 3 strategies
    if ablation_hops:
        configs_per_n *= len(SAFETY_RADII)  # multiply by 3 radii
    total_remaining = len(remaining) * n_trials * configs_per_n
    print(f"Running {len(remaining)} n-values x {configs_per_n} configs x {n_trials} trials = {total_remaining:,} total trials")
    if completed:
        print(f"  (skipping {len(completed)} already-completed configurations)")
    print("This may take a while...\n")

    # --- Build list of (n, f, fault_mode, fault_pct_label, token_strategy, safety_radius) sweep jobs ---
    strategies = TOKEN_STRATEGIES if ablation else ["furthest"]
    radii = SAFETY_RADII if ablation_hops else [2]
    sweep_jobs = []
    for n in remaining:
        if cluster_faults:
            for cluster_size in [2, 3, 4, 5]:
                for strat in strategies:
                    for radius in radii:
                        sweep_jobs.append((n, cluster_size, FAULT_MODE_CLUSTER, '', strat, radius))
        elif dynamic_pct:
            for pct in [0.10, 0.20, 0.30]:
                f = max(1, math.ceil(n * pct))
                pct_label = f'{int(pct*100)}%'
                for fm in [FAULT_MODE_RANDOM, FAULT_MODE_RANDOM_CLUSTERS, FAULT_MODE_LOCALIZED]:
                    for strat in strategies:
                        for radius in radii:
                            sweep_jobs.append((n, f, fm, pct_label, strat, radius))
        else:
            f = max(1, n // 10) if dynamic_faults else n_faults
            for strat in strategies:
                for radius in radii:
                    sweep_jobs.append((n, f, FAULT_MODE_RANDOM, '', strat, radius))

    # --- Streaming sweep loop ---
    if sweep_jobs:
        summary_csv_path = os.path.join(output_dir, "sweep_summary.csv")
        trials_csv_path = os.path.join(output_dir, "trials.csv")

        # Append if resuming, write fresh otherwise
        if completed:
            summary_mode = 'a'
            trials_mode = 'a'
        else:
            summary_mode = 'w'
            trials_mode = 'w'

        summary_file = open(summary_csv_path, summary_mode, newline='')
        trials_file = open(trials_csv_path, trials_mode, newline='')

        try:
            summary_writer = csv.writer(summary_file)
            trials_writer = csv.writer(trials_file)

            if not completed:
                summary_writer.writerow(SUMMARY_HEADERS)
                trials_writer.writerow(TRIALS_HEADERS)

            job_iterator = tqdm(sweep_jobs, desc="Parameter sweep")
            for job_idx, (n, f, fault_mode, fault_pct_label, tok_strat, radius) in enumerate(job_iterator):
                job_iterator.set_postfix(n=n, f=f, mode=fault_mode, strat=tok_strat, radius=radius)

                # Deterministic seed: offset by n position and job sub-index within that n
                config_seed = base_seed + (n - n_min) * n_trials * configs_per_n + job_idx * 7

                result = run_monte_carlo(
                    n_modules=n,
                    n_faults=f,
                    n_trials=n_trials,
                    seed=config_seed,
                    mode_2d=mode_2d,
                    fully_connected=fully_connected,
                    verbose=False,
                    config_mode=config_mode,
                    n_jobs=n_jobs,
                    fault_mode=fault_mode,
                    token_strategy=tok_strat,
                    safety_radius=radius,
                    reconstruction_method=reconstruction_method
                )

                # Stream summary row
                summary_writer.writerow([
                    result.n_modules, result.n_faults, result.n_trials,
                    result.n_meaningful_trials,
                    f'{result.mean_shape_difference:.6f}',
                    f'{result.std_shape_difference:.6f}',
                    f'{result.mean_shape_difference_phase1:.6f}',
                    f'{result.std_shape_difference_phase1:.6f}',
                    f'{result.reconnection_rate:.4f}',
                    f'{result.std_reconnection_rate:.4f}',
                    f'{result.full_restoration_rate:.4f}',
                    f'{result.mean_phase1_moves:.2f}',
                    f'{result.mean_phase2_moves:.2f}',
                    f'{result.mean_steps_to_reconnection:.2f}',
                    f'{result.std_steps_to_reconnection:.2f}',
                    f'{result.mean_total_moves:.2f}',
                    f'{result.std_total_moves:.2f}',
                    f'{result.mean_token_transmissions:.2f}',
                    f'{result.std_token_transmissions:.2f}',
                    fault_mode,
                    fault_pct_label,
                    tok_strat,
                    radius
                ])
                summary_file.flush()

                # Stream trial rows
                for trial in result.trials:
                    trials_writer.writerow([
                        trial.trial_id, trial.n_modules, trial.n_faults,
                        trial.seed, trial.restored, trial.phase1_moves,
                        trial.phase2_moves,
                        trial.shape_difference if trial.shape_difference is not None else '',
                        trial.shape_difference_phase1 if trial.shape_difference_phase1 is not None else '',
                        trial.phase1_iterations, trial.total_moves,
                        trial.token_transmissions,
                        trial.fault_mode,
                        trial.token_strategy,
                        trial.safety_radius
                    ])
                trials_file.flush()

        finally:
            summary_file.close()
            trials_file.close()

        print(f"\nSweep complete!")
        print(f"Saved: {summary_csv_path}")
        print(f"Saved: {trials_csv_path}")
    else:
        print("All n-values already completed! Regenerating graphs...\n")

    # --- Read back full CSV for graphs and summary ---
    summary_csv_path = os.path.join(output_dir, "sweep_summary.csv")
    if not os.path.exists(summary_csv_path):
        print("No sweep_summary.csv found, skipping graphs and summary.")
        return

    csv_n = []
    csv_f = []
    csv_trials = []
    csv_meaningful = []
    csv_reconn = []
    csv_shape_mean = []
    csv_shape_std = []
    csv_full_restore = []
    csv_p1_moves = []
    csv_p2_moves = []

    with open(summary_csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                csv_n.append(int(row['n_modules']))
                csv_f.append(int(row['n_faults']))
                csv_trials.append(int(row['n_trials']))
                csv_meaningful.append(int(row['n_meaningful_trials']))
                csv_reconn.append(float(row['reconnection_rate']))
                csv_shape_mean.append(float(row['mean_shape_difference']))
                csv_shape_std.append(float(row['std_shape_difference']))
                csv_full_restore.append(float(row['full_restoration_rate']))
                csv_p1_moves.append(float(row['mean_phase1_moves']))
                csv_p2_moves.append(float(row['mean_phase2_moves']))
            except (ValueError, KeyError):
                continue

    if not csv_n:
        print("No valid data in sweep_summary.csv, skipping graphs and summary.")
        return

    # Sort by n
    order = sorted(range(len(csv_n)), key=lambda i: csv_n[i])
    csv_n = [csv_n[i] for i in order]
    csv_f = [csv_f[i] for i in order]
    csv_trials = [csv_trials[i] for i in order]
    csv_meaningful = [csv_meaningful[i] for i in order]
    csv_reconn = [csv_reconn[i] for i in order]
    csv_shape_mean = [csv_shape_mean[i] for i in order]
    csv_shape_std = [csv_shape_std[i] for i in order]

    if not no_graphs:
        generate_graphs(
            csv_n, csv_reconn, csv_shape_mean, csv_shape_std,
            n_faults=n_faults, n_trials=n_trials,
            output_dir=output_dir,
            dynamic_faults=dynamic_faults
        )

    print_summary_table(csv_n, csv_f, csv_meaningful, csv_reconn,
                        csv_shape_mean, csv_shape_std, dynamic_faults)

    print(f"\nTotal configurations: {len(csv_n)}")
    print(f"All results saved to: {output_dir}/")


def generate_graphs(n_values, reconnection_rates, shape_diffs_mean, shape_diffs_std,
                    n_faults, n_trials, output_dir, dynamic_faults=False):
    """Generate and save visualization graphs."""

    sigma = 2
    reconnection_smooth = gaussian_filter1d(reconnection_rates, sigma=sigma)
    shape_diff_smooth = gaussian_filter1d(shape_diffs_mean, sigma=sigma)

    x_max = max(n_values) + 5

    if dynamic_faults:
        fault_desc = "f=n/10"
    else:
        fault_desc = f"f={n_faults}"

    # Create combined figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Monte Carlo Simulation Results ({fault_desc} faults, {n_trials} trials per n)',
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
            f'({fault_desc} faults, {n_trials} trials per n)',
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


def print_summary_table(n_values, f_values, meaningful_values,
                        reconn_values, shape_mean_values, shape_std_values,
                        dynamic_faults=False):
    """Print a summary table of results."""
    print("\n" + "=" * 90)
    if dynamic_faults:
        print(f"SUMMARY TABLE (n={n_values[0]} to n={n_values[-1]}, f=n/10 dynamic)")
    else:
        print(f"SUMMARY TABLE (n={n_values[0]} to n={n_values[-1]}, f={f_values[0]})")
    print("=" * 90)
    print(f"{'n':>4} {'f':>3} {'Meaningful':>10} {'Reconn%':>8} {'ShapeDiff (mean+/-std)':>22}")
    print("-" * 90)

    step = max(1, len(n_values) // 20)
    for i in range(len(n_values)):
        if i % step == 0 or i == len(n_values) - 1:
            print(
                f"{n_values[i]:>4} {f_values[i]:>3} {meaningful_values[i]:>10} "
                f"{reconn_values[i]*100:>7.1f}% "
                f"{shape_mean_values[i]:>10.4f} +/- {shape_std_values[i]:<8.4f}"
            )
    print("=" * 90)


if __name__ == "__main__":
    main()
