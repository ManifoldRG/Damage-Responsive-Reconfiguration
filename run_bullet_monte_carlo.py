#!/usr/bin/env python3
"""
PyBullet Monte Carlo Simulation Sweep Runner

Mirrors ``run_monte_carlo_sweep.py`` but runs each trial through the
BulletSimulator + DecentralizedCoagulation/Restructuring agent pipeline
instead of the discrete UDQDGSystem.

Trials run sequentially by default. PyBullet is not thread-safe but it is
process-safe — pass ``--workers N`` to dispatch trials across N
subprocesses (each subprocess gets its own DIRECT physics client).

Usage:
    python run_bullet_monte_carlo.py [options]

Examples:
    python run_bullet_monte_carlo.py --n-max 10 --trials 5
    python run_bullet_monte_carlo.py --n-min 5 --n-max 20 --trials 20 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from src.agent_policy import (
    DecentralizedCoagulation,
    DecentralizedRestructuring,
    DisplacementRestructuring,
    ModuleAgent,
    ModuleState,
)
from src.bullet_sim import BulletSimulator
from src.configurations import create_random_configuration, create_random_tree_configuration
from src.mc_runner import (
    IDLE_OR_CAPPED_MIN_SIM_TIME,
    Scenario as BulletScenario,
    _MCCoagulation,
    _MCDisplacementRestructuring,
    _MCRestructuring,
    _all_idle_or_capped,
    _phase1_done,
    _phase2_done,
    _run_phase,
    run_trial,
    udqdg_to_scenario as udqdg_to_bullet_scenario,
)
from src.monte_carlo import (
    TrialResult,
    MonteCarloResults,
    calculate_shape_difference,
    select_faulty_modules,
    CONFIG_MODE_RANDOM,
    CONFIG_MODE_TREE,
    FAULT_MODE_RANDOM,
    FAULT_MODE_CLUSTER,
    FAULT_MODE_RANDOM_CLUSTERS,
    FAULT_MODE_LOCALIZED,
)


TOKEN_STRATEGIES = ["furthest", "nearest", "random"]
SAFETY_RADII = [2, 3, 4]


def warn_arg_conflicts(args, config):
    """Print warnings if non-default CLI args differ from resumed config."""
    checks = [
        ("n_min", "--n-min", args.n_min, config["n_min"], 5),
        ("n_max", "--n-max", args.n_max, config["n_max"], 20),
        ("n_trials", "--trials", args.trials, config["n_trials"], 100),
        ("seed", "--seed", args.seed, config["seed"], 42),
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


def _connected_components_from_bonds(
    bond_matrix: np.ndarray,
    active_body_indices: List[int],
) -> List[List[int]]:
    """BFS over the bond graph restricted to ``active_body_indices``.

    Used to characterize the final structural state at phase end. Faulty
    body indices should NOT be in active_body_indices; they're physical
    obstacles in the sim but excluded from the policy's connectivity check.
    """
    active = set(active_body_indices)
    seen: Set[int] = set()
    components: List[List[int]] = []
    for start in active_body_indices:
        if start in seen:
            continue
        comp: List[int] = []
        stack = [start]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.append(v)
            for j in np.where(bond_matrix[v])[0]:
                jj = int(j)
                if jj in active and jj not in seen:
                    stack.append(jj)
        components.append(sorted(comp))
    return components


def _write_trial_diagnostics(
    *,
    dump_diagnostics_dir: str,
    trial_id: int,
    seed: int,
    n_modules: int,
    n_faults: int,
    fault_mode: str,
    safety_radius: int,
    use_flood_echo: bool,
    token_gen_interval: float,
    sim,
    coag,
    restruct,
    scenario,
    phase1_connected: bool,
    total_phase1_moves: int,
    total_phase2_moves: int,
    total_phase1_ticks: int,
) -> None:
    """Write one JSON blob with per-trial diagnostic state.

    Called just before ``sim.disconnect()`` so the bullet world is still
    queryable. Inverts the monkey-patch pattern from
    ``debug_pivot_outcomes.py`` — same data, but written to disk for
    offline analysis instead of printed to stdout.
    """
    os.makedirs(dump_diagnostics_dir, exist_ok=True)
    bond_matrix = sim.get_bond_matrix()
    active_idxs = [scenario.body_indices[mid] for mid in scenario.module_ids]
    components = _connected_components_from_bonds(bond_matrix, active_idxs)

    final_pos = sim.get_positions()
    bond_pairs = [
        [int(i), int(j)]
        for i in range(bond_matrix.shape[0])
        for j in range(i + 1, bond_matrix.shape[1])
        if bond_matrix[i, j]
    ]

    payload = {
        "trial_id": int(trial_id),
        "seed": int(seed),
        "n_modules": int(n_modules),
        "n_faults": int(n_faults),
        "fault_mode": fault_mode,
        "safety_radius": int(safety_radius),
        "use_flood_echo": bool(use_flood_echo),
        "token_gen_interval": float(token_gen_interval),
        "connected": bool(phase1_connected),
        "total_phase1_moves": int(total_phase1_moves),
        "total_phase2_moves": int(total_phase2_moves),
        "total_phase1_ticks": int(total_phase1_ticks),
        "reversal_count_coag": int(getattr(coag, "reversal_count", 0)),
        "reversal_count_restruct": int(getattr(restruct, "reversal_count", 0))
        if restruct is not None else 0,
        "pivot_log": sim.get_pivot_diagnostic_log(),
        "auto_bond_stats": sim.get_auto_bond_stats(),
        "attitude_drift_log": [list(t) for t in sim.get_attitude_drift_log()],
        "final_pos": final_pos.tolist(),
        "final_bonds": bond_pairs,
        "fault_body_idxs": [int(i) for i in scenario.fault_body_idxs],
        "fault_ids": list(scenario.fault_ids),
        "module_ids": list(scenario.module_ids),
        "body_indices": {mid: int(idx)
                          for mid, idx in scenario.body_indices.items()},
        "final_components": [[int(i) for i in c] for c in components],
        "n_active_modules": len(scenario.module_ids),
    }
    out_path = os.path.join(
        dump_diagnostics_dir,
        f"trial_{trial_id:04d}_n{n_modules}_f{n_faults}"
        f"_{fault_mode}_seed{seed}.json",
    )
    with open(out_path, "w") as f:
        json.dump(payload, f)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_single_bullet_trial(
    n_modules: int,
    n_faults: int,
    seed: int,
    trial_id: int = 0,
    mode_2d: bool = False,
    fully_connected: bool = True,
    config_mode: str = CONFIG_MODE_RANDOM,
    fault_mode: str = FAULT_MODE_RANDOM,
    *,
    temperature: float = 0.1,
    pivot_exclusion_radius: int = 4,
    max_phase_time: float = 180.0,
    stall_interval: float = 10.0,
    stall_patience: int = 16,
    dt: float = 0.1,
    restructuring_method: str = "rendezvous",
    token_strategy: str = "furthest",
    safety_radius: int = 2,
    module_shape: str = "sphere",
    max_pivot_time: Optional[float] = None,
    use_flood_echo: bool = True,
    token_gen_interval: float = 0.1,
    dump_diagnostics_dir: Optional[str] = None,
    max_moves_per_module: int = 10,
    use_position_history: bool = True,
) -> TrialResult:
    """Execute one PyBullet-based Monte Carlo trial.

    Generates a structure, finds a fault set that disconnects it, builds a
    ``BulletSimulator``, and delegates the phase loop to
    ``src.mc_runner.run_trial``. Returns a ``TrialResult`` compatible with
    the existing MC framework.

    ``max_phase_time`` / ``stall_interval`` / ``stall_patience`` are kept in
    the signature for back-compat with sweep CLI plumbing but unused — the
    per-agent action-point cap bounds total work.
    """
    del max_phase_time, stall_interval, stall_patience
    from src.monte_carlo import _causes_disconnection

    for structure_attempt in range(500):
        gen_seed = seed + structure_attempt * 9973
        if config_mode == CONFIG_MODE_TREE:
            system = create_random_tree_configuration(
                n_modules, seed=gen_seed, mode_2d=mode_2d, balanced=False)
        else:
            system = create_random_configuration(
                n_modules, seed=gen_seed, mode_2d=mode_2d,
                fully_connected=fully_connected)

        faulty_module_ids = select_faulty_modules(
            system, n_faults, gen_seed + 1000, fault_mode)

        if _causes_disconnection(system, faulty_module_ids):
            break

    original_positions = {
        mid: module.position.copy()
        for mid, module in system.modules.items()
    }
    faulty_modules_set = set(faulty_module_ids)

    # Simultaneous fault injection: mark every selected fault before any
    # bullet world is built. The policy reconfigures around all faults
    # collectively in a single coag+restruct, instead of repairing one
    # fault at a time. Faults remain as full-mass collision-on bodies in
    # the bullet world (just excluded from the policy's module set).
    for fid in faulty_module_ids:
        if system.modules[fid].is_active:
            system.mark_fault(fid)

    scenario = udqdg_to_bullet_scenario(system, faulty_module_ids)

    if module_shape == "cube":
        BulletSimulator.USE_ROLLING_SPHERE_PIVOT = False
        BulletSimulator.MAX_PIVOT_TIME = 40.0
    else:
        BulletSimulator.USE_ROLLING_SPHERE_PIVOT = True
        BulletSimulator.MAX_PIVOT_TIME = 20.0
    if max_pivot_time is not None:
        BulletSimulator.MAX_PIVOT_TIME = float(max_pivot_time)
    sim = BulletSimulator(
        scenario.n_total, scenario.pos0, scenario.bonded0, gui=False,
        module_shape=module_shape)

    def _diag_cb(*, sim, coag, restruct, scenario, phase1_connected,
                 total_phase1_moves, total_phase2_moves, total_phase1_ticks):
        if dump_diagnostics_dir is None:
            return
        _write_trial_diagnostics(
            dump_diagnostics_dir=dump_diagnostics_dir,
            trial_id=trial_id,
            seed=seed,
            n_modules=n_modules,
            n_faults=n_faults,
            fault_mode=fault_mode,
            safety_radius=safety_radius,
            use_flood_echo=use_flood_echo,
            token_gen_interval=token_gen_interval,
            sim=sim,
            coag=coag,
            restruct=restruct,
            scenario=scenario,
            phase1_connected=phase1_connected,
            total_phase1_moves=total_phase1_moves,
            total_phase2_moves=total_phase2_moves,
            total_phase1_ticks=total_phase1_ticks,
        )

    try:
        result = run_trial(
            sim=sim,
            scenario=scenario,
            faulty_modules_set=faulty_modules_set,
            original_positions=original_positions,
            trial_id=trial_id,
            n_modules=n_modules,
            n_faults=n_faults,
            temperature=temperature,
            pivot_exclusion_radius=pivot_exclusion_radius,
            dt=dt,
            restructuring_method=restructuring_method,
            token_strategy=token_strategy,
            safety_radius=safety_radius,
            use_flood_echo=use_flood_echo,
            token_gen_interval=token_gen_interval,
            max_moves_per_module=max_moves_per_module,
            use_position_history=use_position_history,
            forward_ap_cost=0.1,
            fault_mode=fault_mode,
            diagnostics_callback=_diag_cb,
        )
    finally:
        sim.disconnect()

    # ``run_trial`` does not know the structure-gen seed; stamp it here so
    # the TrialResult is fully reproducible from CSV.
    result.seed = seed
    return result


# ---------------------------------------------------------------------------
# Aggregation loop (reuses MonteCarloResults)
# ---------------------------------------------------------------------------

def _run_trial_dispatch(kwargs: Dict) -> TrialResult:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    return run_single_bullet_trial(**kwargs)


def run_bullet_monte_carlo(
    n_modules: int,
    n_faults: int = 1,
    n_trials: int = 100,
    seed: Optional[int] = None,
    mode_2d: bool = False,
    fully_connected: bool = True,
    verbose: bool = False,
    config_mode: str = CONFIG_MODE_RANDOM,
    fault_mode: str = FAULT_MODE_RANDOM,
    workers: int = 1,
    **trial_kwargs,
) -> MonteCarloResults:
    """Run Monte Carlo simulation via PyBullet.

    ``workers``: 1 = sequential (default); >1 = ProcessPoolExecutor with that
    many subprocess workers. Each worker holds its own PyBullet DIRECT
    client, so trials are fully isolated.
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    trial_kwarg_list: List[Dict] = [
        dict(
            n_modules=n_modules,
            n_faults=n_faults,
            seed=seed + i,
            trial_id=i,
            mode_2d=mode_2d,
            fully_connected=fully_connected,
            config_mode=config_mode,
            fault_mode=fault_mode,
            **trial_kwargs,
        )
        for i in range(n_trials)
    ]

    trials: List[TrialResult] = []
    if workers <= 1:
        iterator = tqdm(trial_kwarg_list, desc=f"n={n_modules}", disable=not verbose)
        for kw in iterator:
            trials.append(run_single_bullet_trial(**kw))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = tqdm(
                executor.map(_run_trial_dispatch, trial_kwarg_list),
                total=n_trials,
                desc=f"n={n_modules} (x{workers})",
                disable=not verbose,
            )
            for result in iterator:
                trials.append(result)

    meaningful_trials = [t for t in trials if t.phase1_moves > 0]
    successful_trials = [t for t in meaningful_trials if t.restored]
    shape_diffs = [t.shape_difference for t in successful_trials]

    n_meaningful = len(meaningful_trials)
    restored_count = len(successful_trials)
    full_restored_count = sum(
        1 for t in successful_trials
        if t.shape_difference is not None and t.shape_difference == 0.0
    )

    if successful_trials:
        mean_shape_diff = float(np.mean(shape_diffs))
        std_shape_diff = float(np.std(shape_diffs))
        shape_diffs_p1 = [
            t.shape_difference_phase1 for t in successful_trials
            if t.shape_difference_phase1 is not None
        ]
        if shape_diffs_p1:
            mean_shape_diff_p1 = float(np.mean(shape_diffs_p1))
            std_shape_diff_p1 = float(np.std(shape_diffs_p1))
        else:
            mean_shape_diff_p1 = float("nan")
            std_shape_diff_p1 = float("nan")
    else:
        mean_shape_diff = float("nan")
        std_shape_diff = float("nan")
        mean_shape_diff_p1 = float("nan")
        std_shape_diff_p1 = float("nan")

    if n_meaningful > 0:
        reconnection_rate = restored_count / n_meaningful
        full_restoration_rate = full_restored_count / n_meaningful
        mean_p1_moves = float(np.mean([t.phase1_moves for t in meaningful_trials]))
        mean_p2_moves = float(np.mean([t.phase2_moves for t in meaningful_trials]))
        steps = [t.phase1_iterations for t in meaningful_trials]
        total_moves_list = [t.total_moves for t in meaningful_trials]
        tokens_list = [t.token_transmissions for t in meaningful_trials]
        mean_steps = float(np.mean(steps))
        std_steps = float(np.std(steps))
        mean_total_moves = float(np.mean(total_moves_list))
        std_total_moves = float(np.std(total_moves_list))
        mean_tokens = float(np.mean(tokens_list))
        std_tokens = float(np.std(tokens_list))
        std_reconn = float(np.sqrt(reconnection_rate * (1 - reconnection_rate) / n_meaningful))
    else:
        reconnection_rate = float("nan")
        full_restoration_rate = float("nan")
        mean_p1_moves = float("nan")
        mean_p2_moves = float("nan")
        mean_steps = float("nan")
        std_steps = float("nan")
        mean_total_moves = float("nan")
        std_total_moves = float("nan")
        mean_tokens = float("nan")
        std_tokens = float("nan")
        std_reconn = float("nan")

    from src.monte_carlo import _aggregate_component_metrics
    comp_metrics = _aggregate_component_metrics(meaningful_trials)

    return MonteCarloResults(
        n_modules=n_modules,
        n_faults=n_faults,
        n_trials=n_trials,
        n_meaningful_trials=n_meaningful,
        mean_shape_difference=mean_shape_diff,
        std_shape_difference=std_shape_diff,
        mean_shape_difference_phase1=mean_shape_diff_p1,
        std_shape_difference_phase1=std_shape_diff_p1,
        reconnection_rate=reconnection_rate,
        full_restoration_rate=full_restoration_rate,
        mean_phase1_moves=mean_p1_moves,
        mean_phase2_moves=mean_p2_moves,
        mean_steps_to_reconnection=mean_steps,
        std_steps_to_reconnection=std_steps,
        mean_total_moves=mean_total_moves,
        std_total_moves=std_total_moves,
        mean_token_transmissions=mean_tokens,
        std_token_transmissions=std_tokens,
        std_reconnection_rate=std_reconn,
        trials=meaningful_trials,
        **comp_metrics,
    )


# ---------------------------------------------------------------------------
# CSV / graph helpers
# ---------------------------------------------------------------------------

SUMMARY_HEADERS = [
    "n_modules", "n_faults", "n_trials", "n_meaningful_trials",
    "mean_shape_difference", "std_shape_difference",
    "mean_shape_difference_phase1", "std_shape_difference_phase1",
    "reconnection_rate", "std_reconnection_rate",
    "full_restoration_rate",
    "mean_phase1_moves", "mean_phase2_moves",
    "mean_steps_to_reconnection", "std_steps_to_reconnection",
    "mean_total_moves", "std_total_moves",
    "mean_token_transmissions", "std_token_transmissions",
    "mean_n_components_post_damage",
    "mean_n_components_post_phase1",
    "mean_n_components_post_phase2",
    "mean_rejoined_frac_phase1",
    "mean_rejoined_frac_total",
    "fault_mode", "fault_pct",
    "token_strategy", "safety_radius", "restructuring_method",
]

TRIALS_HEADERS = [
    "trial_id", "n_modules", "n_faults", "seed",
    "restored", "phase1_moves", "phase2_moves",
    "shape_difference", "shape_difference_phase1",
    "phase1_iterations", "total_moves",
    "token_transmissions",
    "n_components_post_damage", "n_components_post_phase1",
    "n_components_post_phase2",
    "largest_component_frac_post_damage",
    "largest_component_frac_post_phase1",
    "largest_component_frac_post_phase2",
    "fault_mode",
    "token_strategy", "safety_radius", "restructuring_method",
]


def generate_graphs(
    n_values, reconnection_rates, shape_diffs_mean, shape_diffs_std,
    n_faults, n_trials, output_dir, dynamic_faults=False,
):
    sigma = max(1, len(n_values) // 10)
    reconnection_smooth = gaussian_filter1d(reconnection_rates, sigma=sigma)
    shape_diff_smooth = gaussian_filter1d(shape_diffs_mean, sigma=sigma)

    x_max = max(n_values) + 5
    fault_desc = "f=n/10" if dynamic_faults else f"f={n_faults}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"PyBullet MC Results ({fault_desc} faults, {n_trials} trials/n)",
        fontsize=14, fontweight="bold",
    )

    ax1 = axes[0]
    ax1.scatter(n_values, reconnection_rates, alpha=0.4, color="blue", s=20, label="Raw")
    ax1.plot(n_values, reconnection_smooth, color="blue", linewidth=2.5, label="Smoothed")
    ax1.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="100%")
    ax1.set_xlabel("Number of Modules (n)")
    ax1.set_ylabel("Reconnection Rate")
    ax1.set_title("Reconnection Rate vs Structure Size")
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    ax2 = axes[1]
    ax2.fill_between(
        n_values,
        np.maximum(0, np.array(shape_diffs_mean) - np.array(shape_diffs_std)),
        np.minimum(1, np.array(shape_diffs_mean) + np.array(shape_diffs_std)),
        alpha=0.2, color="orange",
    )
    ax2.scatter(n_values, shape_diffs_mean, alpha=0.4, color="orange", s=20, label="Raw")
    ax2.plot(n_values, shape_diff_smooth, color="darkorange", linewidth=2.5, label="Smoothed")
    ax2.set_xlabel("Number of Modules (n)")
    ax2.set_ylabel("Shape Difference")
    ax2.set_title("Shape Difference diff(P, Q) vs Structure Size")
    ax2.set_xlim(0, x_max)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    graph_path = os.path.join(output_dir, "graphs_combined.png")
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {graph_path}")

    for metric_name, data, smooth_data, std_data, color, dark_color in [
        ("reconnection_rate", reconnection_rates, reconnection_smooth, None, "blue", "blue"),
        ("shape_difference", shape_diffs_mean, shape_diff_smooth, shape_diffs_std, "orange", "darkorange"),
    ]:
        fig2, ax = plt.subplots(figsize=(10, 7))
        if std_data:
            ax.fill_between(
                n_values,
                np.maximum(0, np.array(data) - np.array(std_data)),
                np.minimum(1, np.array(data) + np.array(std_data)),
                alpha=0.2, color=color, label="\u00b11 std",
            )
        ax.scatter(n_values, data, alpha=0.4, color=color, s=30, label="Raw")
        ax.plot(n_values, smooth_data, color=dark_color, linewidth=2.5, label=f"Smoothed (\u03c3={sigma})")
        ax.set_xlabel("Number of Modules (n)", fontsize=13)
        ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=13)
        ax.set_title(
            f"{metric_name.replace('_', ' ').title()} vs Structure Size\n"
            f"({fault_desc} faults, {n_trials} trials/n)",
            fontsize=14,
        )
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        if metric_name == "reconnection_rate":
            ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)

        ind_path = os.path.join(output_dir, f"graph_{metric_name}.png")
        plt.savefig(ind_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {ind_path}")

    plt.close("all")


def print_summary_table(
    n_values, f_values, meaningful_values,
    reconn_values, shape_mean_values, shape_std_values,
    dynamic_faults=False,
):
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PyBullet Monte Carlo simulation sweep for damage response"
    )
    parser.add_argument("--n-min", type=int, default=5,
                        help="Minimum number of modules (default: 5)")
    parser.add_argument("--n-max", type=int, default=20,
                        help="Maximum number of modules (default: 20)")
    parser.add_argument("--n-step", type=int, default=1,
                        help="Step size between n values (default: 1)")
    parser.add_argument("--faults", type=int, default=1,
                        help="Number of faults per trial (default: 1)")
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of trials per configuration (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="bullet_mc_results",
                        help="Output directory (default: bullet_mc_results)")
    parser.add_argument("--no-graphs", action="store_true",
                        help="Skip graph generation")
    parser.add_argument("--mode-2d", action="store_true",
                        help="Use 2D mode")
    parser.add_argument("--n-values", type=int, nargs="+", default=None,
                        help="Explicit list of n values to sweep (overrides --n-min/--n-max/--n-step)")
    parser.add_argument("--fully-connected", action="store_true",
                        help="Connect to ALL adjacent modules (default behavior)")
    parser.add_argument("--chain-like", action="store_true",
                        help="Chain-like connectivity (overrides default fully-connected)")
    parser.add_argument("--tree", action="store_true",
                        help="Tree-based configuration (no cycles)")
    parser.add_argument("--dynamic-faults", action="store_true",
                        help="Faults = floor(n/10) for each n")
    parser.add_argument("--cluster-faults", action="store_true",
                        help="Cluster failure sweep: cluster sizes 2,3,4,5")
    parser.add_argument("--dynamic-pct", action="store_true",
                        help="Dynamic pct fault sweep: 10%%,20%%,30%% x 3 patterns")
    parser.add_argument("--fault-modes", type=str, nargs="+", default=None,
                        choices=["random", "cluster", "random_clusters", "localized"],
                        help="Explicit fault modes to sweep (overrides --cluster-faults/--dynamic-pct)")
    parser.add_argument("--ablation", action="store_true",
                        help="Sweep token selection strategies: furthest, nearest, random")
    parser.add_argument("--ablation-hops", action="store_true",
                        help="Sweep safety radii: 2, 3, 4 for is_movable() check")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Coagulation temperature (default: 0.01)")
    parser.add_argument("--pivot-radius", type=int, default=4,
                        help="Pivot exclusion radius (default: 4)")
    parser.add_argument("--max-phase-time", type=float, default=180.0,
                        help="Max sim-seconds per phase (default: 180)")
    parser.add_argument("--stall-interval", type=float, default=10.0,
                        help="Seconds between stall checks (default: 10)")
    parser.add_argument("--max-moves-per-module", type=int, default=10,
                        help="Per-module cap on successful pivots per phase. "
                             "Once a module reaches this count it stops "
                             "initiating new pivots for the remainder of the "
                             "phase. Replaces the centralized stall timer "
                             "with a decentralized bound: total work per "
                             "phase is at most N × this value.")
    parser.add_argument("--stall-patience", type=int, default=16,
                        help="Stall windows before phase exit (default: 16)")
    parser.add_argument("--resume", type=str, default=None, metavar="PATH",
                        help="Resume from existing output directory")
    parser.add_argument("--restructuring-method", type=str, default="rendezvous",
                        choices=["rendezvous", "displacement"],
                        help="Phase 2 method: rendezvous tokens or displacement-guided (default: rendezvous)")
    parser.add_argument("--module-shape", type=str, default="sphere",
                        choices=["sphere", "cube"],
                        help="Module geometry: sphere (rolling pivots) or cube "
                             "(edge-lever pivots; only laterals are eligible) "
                             "(default: sphere)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel subprocess workers per (n,config) batch "
                             "(default: 1=sequential)")
    parser.add_argument("--safety-radius", type=int, default=2,
                        help="Single safety radius for is_movable() check "
                             "(default: 2). Ignored if --ablation-hops set.")
    parser.add_argument("--max-pivot-time", type=float, default=None,
                        help="Override BulletSimulator.MAX_PIVOT_TIME in sec "
                             "(default: 20s sphere / 40s cube)")
    parser.add_argument("--no-position-history", action="store_true",
                        help="Disable per-agent position-history dedup in "
                             "pick_target. Allows the agent to re-target "
                             "lattice cells it has previously visited.")
    parser.add_argument("--no-flood-echo", action="store_true",
                        help="Disable flood/echo component-discovery protocol. "
                             "Fault-adjacent modules emit tokens unconditionally "
                             "(matches non-bullet MC behavior). Diagnostic.")
    parser.add_argument("--token-gen-interval", type=float, default=0.1,
                        help="Seconds of sim time between consecutive token "
                             "emissions per fault-adjacent module (default: 0.1, "
                             "matching the policy tick dt). Lower = more "
                             "aggressive token flow.")
    parser.add_argument("--dump-diagnostics", type=str, default=None,
                        metavar="DIR",
                        help="If set, write one JSON per trial under DIR with "
                             "pivot log, auto-bond stats, attitude drift, "
                             "final structural state. Diagnostic only.")
    args = parser.parse_args()

    # --- Resume mode ---
    if args.resume:
        resume_dir = args.resume
        if not os.path.isdir(resume_dir):
            print(f"Error: Resume directory does not exist: {resume_dir}",
                  file=sys.stderr)
            sys.exit(1)
        config_path = os.path.join(resume_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"Error: No config.json in {resume_dir}", file=sys.stderr)
            sys.exit(1)
        with open(config_path, "r") as f:
            config = json.load(f)

        warn_arg_conflicts(args, config)

        n_min = config["n_min"]
        n_max = config["n_max"]
        n_trials = config["n_trials"]
        base_seed = config["seed"]
        mode_2d = config.get("mode_2d", False)
        fully_connected = config.get("fully_connected", True)
        config_mode = config.get("config_mode", CONFIG_MODE_RANDOM)
        dynamic_faults = config.get("dynamic_faults", False)
        n_faults = config.get("n_faults", 1)
        if not isinstance(n_faults, int):
            n_faults = 1
        cluster_faults = config.get("cluster_faults", False)
        dynamic_pct = config.get("dynamic_pct", False)
        n_step = config.get("n_step", 1)
        temperature = config.get("temperature", 0.01)
        pivot_radius = config.get("pivot_radius", 4)
        max_phase_time = config.get("max_phase_time", 300.0)
        stall_interval = config.get("stall_interval", 20.0)
        stall_patience = config.get("stall_patience", 8)
        restructuring_method = config.get("restructuring_method", "rendezvous")
        module_shape = config.get("module_shape", "sphere")
        ablation = config.get("ablation", False)
        ablation_hops = config.get("ablation_hops", False)
        n_values_explicit = config.get("n_values", None)
        fault_modes_explicit = config.get("fault_modes", None)
        output_dir = resume_dir

        csv_path = os.path.join(output_dir, "sweep_summary.csv")
        completed: set = set()
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        completed.add(int(row["n_modules"]))
                    except (ValueError, KeyError):
                        continue
        print(f"Resuming from: {output_dir}")
        if completed:
            print(f"Already completed: {sorted(completed)}")
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
        cluster_faults = args.cluster_faults
        dynamic_pct = args.dynamic_pct
        n_step = args.n_step
        temperature = args.temperature
        pivot_radius = args.pivot_radius
        max_phase_time = args.max_phase_time
        stall_interval = args.stall_interval
        stall_patience = args.stall_patience
        restructuring_method = args.restructuring_method
        module_shape = args.module_shape
        ablation = args.ablation
        ablation_hops = args.ablation_hops
        n_values_explicit = args.n_values
        fault_modes_explicit = args.fault_modes

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            "cluster_faults": cluster_faults,
            "dynamic_pct": dynamic_pct,
            "n_step": n_step,
            "temperature": temperature,
            "pivot_radius": pivot_radius,
            "max_phase_time": max_phase_time,
            "stall_interval": stall_interval,
            "stall_patience": stall_patience,
            "restructuring_method": restructuring_method,
            "module_shape": module_shape,
            "ablation": ablation,
            "ablation_hops": ablation_hops,
            "n_values": n_values_explicit,
            "fault_modes": fault_modes_explicit,
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        completed = set()

    no_graphs = args.no_graphs

    if n_values_explicit:
        all_n_values = sorted(n_values_explicit)
    else:
        all_n_values = list(range(n_min, n_max + 1, n_step))
    remaining = [n for n in all_n_values if n not in completed]

    strategies = TOKEN_STRATEGIES if ablation else ["furthest"]
    radii = SAFETY_RADII if ablation_hops else [args.safety_radius]

    if config_mode == CONFIG_MODE_TREE:
        connectivity_desc = "tree (no cycles)"
    elif fully_connected:
        connectivity_desc = "fully-connected (default)"
    else:
        connectivity_desc = "chain-like"

    configs_per_n = 1
    if fault_modes_explicit:
        configs_per_n = len(fault_modes_explicit)
    elif cluster_faults:
        configs_per_n = 4
    elif dynamic_pct:
        configs_per_n = 6  # 3 pcts x 2 patterns (random_clusters disabled)
    configs_per_n *= len(strategies) * len(radii)

    print("=" * 70)
    print("PYBULLET MONTE CARLO SIMULATION SWEEP")
    print("=" * 70)
    print("Parameters:")
    if n_values_explicit:
        print(f"  n values: {all_n_values}")
    else:
        print(f"  Module range: n = {n_min} to {n_max} (step {n_step})")
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
    print(f"  Temperature: {temperature}")
    print(f"  Pivot exclusion radius: {pivot_radius}")
    print(f"  Max phase time: {max_phase_time}s")
    print(f"  Restructuring: {restructuring_method}")
    print(f"  Module shape: {module_shape}")
    if ablation:
        print(f"  Token strategy ablation: {strategies}")
    else:
        print(f"  Token strategy: furthest")
    if ablation_hops:
        print(f"  Safety radius ablation: {radii}")
    else:
        print(f"  Safety radius: 2")
    if args.workers > 1:
        print(f"  Workers: {args.workers} subprocesses (PyBullet process-safe)")
    else:
        print(f"  Sequential (use --workers N for subprocess parallelism)")
    if fault_modes_explicit:
        print(f"  Fault modes: {fault_modes_explicit}")
    elif cluster_faults:
        print(f"  Cluster faults: enabled (sizes 2,3,4,5)")
    elif dynamic_pct:
        print(f"  Dynamic pct: enabled (10%,20%,30% x random,random_clusters,localized)")
    else:
        print(f"  Fault mode: random")
    if completed:
        print(f"  Resuming: {len(completed)}/{len(all_n_values)} n-values done")
    print("=" * 70)

    total_remaining = len(remaining) * n_trials * configs_per_n
    print(f"\nRunning {len(remaining)} n-values x {configs_per_n} configs x {n_trials} trials = {total_remaining:,} total trials")
    if completed:
        print(f"  (skipping {len(completed)} already-completed configurations)")
    print("This may take a while...\n")

    # --- Build sweep jobs (6-tuples) ---
    sweep_jobs: List[Tuple] = []
    for n in remaining:
        for strat in strategies:
            for rad in radii:
                if fault_modes_explicit:
                    for fm in fault_modes_explicit:
                        f = max(1, n // 10) if dynamic_faults else n_faults
                        sweep_jobs.append((n, f, fm, "", strat, rad))
                elif cluster_faults:
                    for cluster_size in [2, 3, 4, 5]:
                        sweep_jobs.append((n, cluster_size, FAULT_MODE_CLUSTER, "", strat, rad))
                elif dynamic_pct:
                    for pct in [0.10, 0.20, 0.30]:
                        f = max(1, math.ceil(n * pct))
                        pct_label = f"{int(pct * 100)}%"
                        for fm in [FAULT_MODE_RANDOM, FAULT_MODE_LOCALIZED]:  # FAULT_MODE_RANDOM_CLUSTERS disabled
                            sweep_jobs.append((n, f, fm, pct_label, strat, rad))
                else:
                    f = max(1, n // 10) if dynamic_faults else n_faults
                    sweep_jobs.append((n, f, FAULT_MODE_RANDOM, "", strat, rad))

    if sweep_jobs:
        summary_csv_path = os.path.join(output_dir, "sweep_summary.csv")
        trials_csv_path = os.path.join(output_dir, "trials.csv")

        if completed:
            summary_mode, trials_mode = "a", "a"
        else:
            summary_mode, trials_mode = "w", "w"

        summary_file = open(summary_csv_path, summary_mode, newline="")
        trials_file = open(trials_csv_path, trials_mode, newline="")

        try:
            summary_writer = csv.writer(summary_file)
            trials_writer = csv.writer(trials_file)

            if not completed:
                summary_writer.writerow(SUMMARY_HEADERS)
                trials_writer.writerow(TRIALS_HEADERS)

            job_iter = tqdm(sweep_jobs, desc="Parameter sweep")
            for job_idx, (n, f, fault_mode, fault_pct_label, strat, rad) in enumerate(job_iter):
                job_iter.set_postfix(n=n, f=f, mode=fault_mode, strat=strat, rad=rad)

                config_seed = base_seed + (n - n_min) * n_trials * configs_per_n + job_idx * 7

                result = run_bullet_monte_carlo(
                    n_modules=n,
                    n_faults=f,
                    n_trials=n_trials,
                    seed=config_seed,
                    mode_2d=mode_2d,
                    fully_connected=fully_connected,
                    verbose=False,
                    config_mode=config_mode,
                    fault_mode=fault_mode,
                    workers=args.workers,
                    temperature=temperature,
                    pivot_exclusion_radius=pivot_radius,
                    max_phase_time=max_phase_time,
                    stall_interval=stall_interval,
                    stall_patience=stall_patience,
                    restructuring_method=restructuring_method,
                    token_strategy=strat,
                    safety_radius=rad,
                    module_shape=module_shape,
                    max_pivot_time=args.max_pivot_time,
                    use_flood_echo=not args.no_flood_echo,
                    token_gen_interval=args.token_gen_interval,
                    dump_diagnostics_dir=args.dump_diagnostics,
                    max_moves_per_module=args.max_moves_per_module,
                    use_position_history=not args.no_position_history,
                )

                summary_writer.writerow([
                    result.n_modules, result.n_faults, result.n_trials,
                    result.n_meaningful_trials,
                    f"{result.mean_shape_difference:.6f}",
                    f"{result.std_shape_difference:.6f}",
                    f"{result.mean_shape_difference_phase1:.6f}",
                    f"{result.std_shape_difference_phase1:.6f}",
                    f"{result.reconnection_rate:.4f}",
                    f"{result.std_reconnection_rate:.4f}",
                    f"{result.full_restoration_rate:.4f}",
                    f"{result.mean_phase1_moves:.2f}",
                    f"{result.mean_phase2_moves:.2f}",
                    f"{result.mean_steps_to_reconnection:.2f}",
                    f"{result.std_steps_to_reconnection:.2f}",
                    f"{result.mean_total_moves:.2f}",
                    f"{result.std_total_moves:.2f}",
                    f"{result.mean_token_transmissions:.2f}",
                    f"{result.std_token_transmissions:.2f}",
                    f"{result.mean_n_components_post_damage:.4f}",
                    f"{result.mean_n_components_post_phase1:.4f}",
                    f"{result.mean_n_components_post_phase2:.4f}",
                    f"{result.mean_rejoined_frac_phase1:.4f}",
                    f"{result.mean_rejoined_frac_total:.4f}",
                    fault_mode,
                    fault_pct_label,
                    strat,
                    rad,
                    restructuring_method,
                ])
                summary_file.flush()

                for trial in result.trials:
                    trials_writer.writerow([
                        trial.trial_id, trial.n_modules, trial.n_faults,
                        trial.seed, trial.restored, trial.phase1_moves,
                        trial.phase2_moves,
                        trial.shape_difference if trial.shape_difference is not None else "",
                        trial.shape_difference_phase1 if trial.shape_difference_phase1 is not None else "",
                        trial.phase1_iterations, trial.total_moves,
                        trial.token_transmissions,
                        trial.n_components_post_damage,
                        trial.n_components_post_phase1,
                        trial.n_components_post_phase2
                        if trial.n_components_post_phase2 is not None else "",
                        f"{trial.largest_component_frac_post_damage:.4f}",
                        f"{trial.largest_component_frac_post_phase1:.4f}",
                        f"{trial.largest_component_frac_post_phase2:.4f}"
                        if trial.largest_component_frac_post_phase2 is not None else "",
                        trial.fault_mode,
                        strat,
                        rad,
                        restructuring_method,
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

    # --- Read back CSV for graphs + summary ---
    summary_csv_path = os.path.join(output_dir, "sweep_summary.csv")
    if not os.path.exists(summary_csv_path):
        print("No sweep_summary.csv found, skipping graphs.")
        return

    csv_n, csv_f, csv_meaningful = [], [], []
    csv_reconn, csv_shape_mean, csv_shape_std = [], [], []

    with open(summary_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                csv_n.append(int(row["n_modules"]))
                csv_f.append(int(row["n_faults"]))
                csv_meaningful.append(int(row["n_meaningful_trials"]))
                csv_reconn.append(float(row["reconnection_rate"]))
                csv_shape_mean.append(float(row["mean_shape_difference"]))
                csv_shape_std.append(float(row["std_shape_difference"]))
            except (ValueError, KeyError):
                continue

    if not csv_n:
        print("No valid data, skipping graphs.")
        return

    order = sorted(range(len(csv_n)), key=lambda i: csv_n[i])
    csv_n = [csv_n[i] for i in order]
    csv_f = [csv_f[i] for i in order]
    csv_meaningful = [csv_meaningful[i] for i in order]
    csv_reconn = [csv_reconn[i] for i in order]
    csv_shape_mean = [csv_shape_mean[i] for i in order]
    csv_shape_std = [csv_shape_std[i] for i in order]

    if not no_graphs and len(csv_n) >= 2:
        generate_graphs(
            csv_n, csv_reconn, csv_shape_mean, csv_shape_std,
            n_faults=n_faults, n_trials=n_trials,
            output_dir=output_dir, dynamic_faults=dynamic_faults,
        )

    print_summary_table(
        csv_n, csv_f, csv_meaningful, csv_reconn,
        csv_shape_mean, csv_shape_std, dynamic_faults,
    )

    print(f"\nTotal configurations: {len(csv_n)}")
    print(f"All results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
