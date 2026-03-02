#!/usr/bin/env python3
"""
Generate quad comparison plots and summary table for Monte Carlo simulation results.
Compares fully-connected vs tree topologies under single and dynamic fault scenarios.

Auto-detects the most recent date directory under monte_carlo_results/.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from pathlib import Path

# Data paths
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / 'monte_carlo_results'

CONFIG_NAMES = {
    'FC Single': 'fully_connected_single_fault',
    'FC Dynamic': 'fully_connected_dynamic_faults',
    'Tree Single': 'tree_single_fault',
    'Tree Dynamic': 'tree_dynamic_faults',
}

# Plot styling
COLORS = {
    'FC Single': '#2196F3',      # Blue
    'FC Dynamic': '#1565C0',     # Dark Blue
    'Tree Single': '#4CAF50',    # Green
    'Tree Dynamic': '#2E7D32',   # Dark Green
}

MARKERS = {
    'FC Single': 'o',
    'FC Dynamic': 's',
    'Tree Single': '^',
    'Tree Dynamic': 'D',
}

LINESTYLES = {
    'FC Single': '-',
    'FC Dynamic': '--',
    'Tree Single': '-',
    'Tree Dynamic': '--',
}

LOWESS_FRAC = {
    'FC Single': 0.25,
    'FC Dynamic': 0.3,
    'Tree Single': 0.4,
    'Tree Dynamic': 0.3,
}


def find_latest_date_dir():
    """Find the most recent date directory (YYYYMMDD) under monte_carlo_results/."""
    date_dirs = sorted(
        [d for d in RESULTS_DIR.iterdir()
         if d.is_dir() and d.name.isdigit() and len(d.name) == 8],
        reverse=True
    )
    if not date_dirs:
        raise FileNotFoundError(
            f"No date directories (YYYYMMDD) found in {RESULTS_DIR}"
        )
    return date_dirs[0]


def find_sweep_csv(config_dir: Path) -> Path:
    """Find the most recent sweep_summary.csv within a config directory.

    Handles both layouts:
      - config_dir/sweep_summary.csv  (flat)
      - config_dir/<timestamp>/sweep_summary.csv  (timestamped subdirectory)
    """
    # Check flat layout first
    flat = config_dir / 'sweep_summary.csv'
    if flat.exists():
        return flat

    # Look for timestamped subdirectories
    subdirs = sorted(
        [d for d in config_dir.iterdir() if d.is_dir()],
        reverse=True
    )
    for subdir in subdirs:
        candidate = subdir / 'sweep_summary.csv'
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No sweep_summary.csv found in {config_dir}"
    )


def resolve_data_paths(date_dir: Path) -> dict:
    """Build data paths for all 4 configs from a date directory."""
    paths = {}
    for display_name, dir_name in CONFIG_NAMES.items():
        config_dir = date_dir / dir_name
        if not config_dir.exists():
            raise FileNotFoundError(
                f"Config directory not found: {config_dir}"
            )
        paths[display_name] = find_sweep_csv(config_dir)
    return paths


def smooth_data(x, y, config_name):
    """Apply LOWESS smoothing with config-dependent bandwidth."""
    frac = LOWESS_FRAC.get(config_name, 0.2)
    result = lowess(y, x, frac=frac, it=3, return_sorted=True)
    return result[:, 0], result[:, 1]


def load_data(data_paths):
    """Load all CSV data files."""
    data = {}
    for name, path in data_paths.items():
        print(f"  {name}: {path}")
        df = pd.read_csv(path)
        data[name] = df
    return data


def create_quad_plot(data, metric, ylabel, title, filename, output_dir, legend_loc='upper right'):
    """Create a 2x2 quad plot comparing all configurations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    subplot_titles = [
        'Fully-Connected Single Fault',
        'Fully-Connected Dynamic Faults',
        'Tree Single Fault',
        'Tree Dynamic Faults'
    ]

    configs = ['FC Single', 'FC Dynamic', 'Tree Single', 'Tree Dynamic']

    x_max = max(data[c]['n_modules'].max() for c in configs) + 5

    for idx, (ax, config, subplot_title) in enumerate(zip(axes.flat, configs, subplot_titles)):
        df = data[config]
        x = df['n_modules'].values
        y = df[metric].values

        mask = ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]

        if len(y_clean) > 0:
            ax.scatter(x_clean, y_clean, c=COLORS[config], alpha=0.4, s=30,
                      marker=MARKERS[config], label='Raw data')

            if len(y_clean) > 3:
                x_s, y_smooth = smooth_data(x_clean, y_clean, config)
                ax.plot(x_s, y_smooth, color=COLORS[config], linewidth=2.5,
                       linestyle=LINESTYLES[config], label='Smoothed')

            std_col = f'std_{metric.replace("mean_", "")}'
            if std_col in df.columns:
                std = df[std_col].values[mask]
                y_upper = y_clean + std
                y_lower = np.maximum(y_clean - std, 0)
                if len(y_clean) > 3:
                    _, y_upper_smooth = smooth_data(x_clean, y_upper, config)
                    _, y_lower_smooth = smooth_data(x_clean, y_lower, config)
                    ax.fill_between(x_clean, y_lower_smooth, y_upper_smooth,
                                   color=COLORS[config], alpha=0.15)

        ax.set_title(subplot_title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Modules (n)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc=legend_loc, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / filename}")


def create_overlay_quad_plot(data, metric, ylabel, title, filename, output_dir):
    """Create a single plot with all 4 configurations overlaid."""
    fig, ax = plt.subplots(figsize=(12, 8))

    configs = ['FC Single', 'FC Dynamic', 'Tree Single', 'Tree Dynamic']

    x_max = max(data[c]['n_modules'].max() for c in configs) + 5

    for config in configs:
        df = data[config]
        x = df['n_modules'].values
        y = df[metric].values

        mask = ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]

        if len(y_clean) > 0:
            ax.scatter(x_clean, y_clean, c=COLORS[config], alpha=0.2, s=20,
                      marker=MARKERS[config])

            if len(y_clean) > 3:
                x_s, y_smooth = smooth_data(x_clean, y_clean, config)
                ax.plot(x_s, y_smooth, color=COLORS[config], linewidth=2.5,
                       linestyle=LINESTYLES[config], marker=MARKERS[config],
                       markevery=10, markersize=8, label=config)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Modules (n)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / filename}")


def generate_summary_table(data, output_file):
    """Generate summary table for every 10 modules."""
    configs = ['FC Single', 'FC Dynamic', 'Tree Single', 'Tree Dynamic']
    max_n = max(data[c]['n_modules'].max() for c in configs)
    n_values = list(range(10, int(max_n) + 1, 10))

    metrics = [
        ('reconnection_rate', 'Reconn. Rate', '{:.1%}'),
        ('mean_shape_difference', 'Shape Diff.', '{:.3f}'),
        ('mean_phase1_moves', 'Phase 1 Moves', '{:.1f}'),
        ('mean_phase2_moves', 'Phase 2 Moves', '{:.1f}'),
        ('n_meaningful_trials', 'Meaningful Trials', '{:.0f}'),
    ]

    rows = []
    for n in n_values:
        row = {'n': n}
        for config in configs:
            df = data[config]
            df_n = df[df['n_modules'] == n]
            if len(df_n) > 0:
                for metric, _, fmt in metrics:
                    val = df_n[metric].values[0]
                    key = f'{config}_{metric}'
                    row[key] = val
            else:
                for metric, _, _ in metrics:
                    key = f'{config}_{metric}'
                    row[key] = np.nan
        rows.append(row)

    with open(output_file, 'w') as f:
        f.write("# Monte Carlo Simulation Results Summary\n\n")
        f.write("Comparison of Fully-Connected (FC) vs Tree topologies under Single and Dynamic fault scenarios.\n\n")

        for metric, label, fmt in metrics:
            f.write(f"\n## {label}\n\n")
            f.write("| n | FC Single | FC Dynamic | Tree Single | Tree Dynamic |\n")
            f.write("|---:|---:|---:|---:|---:|\n")

            for row in rows:
                n = row['n']
                vals = []
                for config in configs:
                    key = f'{config}_{metric}'
                    val = row.get(key, np.nan)
                    if pd.isna(val):
                        vals.append('-')
                    else:
                        vals.append(fmt.format(val))
                f.write(f"| {n} | {' | '.join(vals)} |\n")

    print(f"Saved: {output_file}")

    csv_rows = []
    for row in rows:
        csv_row = {'n_modules': row['n']}
        for config in configs:
            for metric, label, _ in metrics:
                key = f'{config}_{metric}'
                csv_key = f'{config.replace(" ", "_")}_{metric}'
                csv_row[csv_key] = row.get(key, np.nan)
        csv_rows.append(csv_row)

    csv_df = pd.DataFrame(csv_rows)
    csv_file = str(output_file).replace('.md', '.csv')
    csv_df.to_csv(csv_file, index=False)
    print(f"Saved: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from Monte Carlo results"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date directory to use (YYYYMMDD). Default: most recent."
    )
    args = parser.parse_args()

    if args.date:
        date_dir = RESULTS_DIR / args.date
        if not date_dir.exists():
            raise FileNotFoundError(f"Date directory not found: {date_dir}")
    else:
        date_dir = find_latest_date_dir()

    run_date = date_dir.name
    output_dir = date_dir / f'comparison_{run_date}'
    output_dir.mkdir(exist_ok=True)

    print(f"Date directory: {date_dir}")
    print(f"Output directory: {output_dir}")

    print("\nLoading data...")
    data_paths = resolve_data_paths(date_dir)
    data = load_data(data_paths)

    print("\nGenerating quad plots...")

    create_quad_plot(
        data, 'reconnection_rate', 'Reconnection Rate',
        'Reconnection Rate Comparison: FC vs Tree, Single vs Dynamic Faults',
        'quad_reconnection_rate.png', output_dir, legend_loc='lower right'
    )

    create_quad_plot(
        data, 'mean_shape_difference', 'Shape Difference',
        'Shape Difference Comparison: FC vs Tree, Single vs Dynamic Faults',
        'quad_shape_difference.png', output_dir
    )

    print("\nGenerating overlay plots...")

    create_overlay_quad_plot(
        data, 'reconnection_rate', 'Reconnection Rate',
        'Reconnection Rate: All Configurations',
        'overlay_reconnection_rate.png', output_dir
    )

    create_overlay_quad_plot(
        data, 'mean_shape_difference', 'Shape Difference',
        'Shape Difference: All Configurations',
        'overlay_shape_difference.png', output_dir
    )

    print("\nGenerating summary table...")
    generate_summary_table(data, str(output_dir / 'comparison_summary.md'))

    print("\nDone!")


if __name__ == '__main__':
    main()
