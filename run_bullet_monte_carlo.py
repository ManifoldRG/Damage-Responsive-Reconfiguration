#!/usr/bin/env python3
"""
PyBullet Monte Carlo Simulation Sweep Runner

Mirrors ``run_monte_carlo_sweep.py`` but runs each trial through the
BulletSimulator + DecentralizedCoagulation/Restructuring agent pipeline
instead of the discrete UDQDGSystem.

Sequential execution only (PyBullet is not thread-safe).

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
from dataclasses import dataclass
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
    ModuleAgent,
    ModuleState,
)
from src.bullet_sim import BulletSimulator
from src.configurations import create_random_configuration, create_random_tree_configuration
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


# ---------------------------------------------------------------------------
# Translation layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BulletScenario:
    """All inputs needed to run a single-fault PyBullet trial."""
    n_total: int
    pos0: np.ndarray            # (n_total, 3)
    bonded0: np.ndarray         # (n_total, n_total) bool
    fault_id: str
    fault_body_idx: int
    fault_adjacent: List[str]
    module_ids: List[str]       # active (non-faulty) module IDs
    body_indices: Dict[str, int]
    pre_damage_neighbor_slots: Dict[str, List[np.ndarray]]
    original_positions: Dict[str, np.ndarray]


def udqdg_to_bullet_scenario(
    system,
    fault_id: str,
) -> BulletScenario:
    """Convert a UDQDGSystem + single fault ID into BulletSimulator inputs.

    Maps module IDs to body indices 0..N-1 (sorted by ID for determinism).
    The fault module becomes a real body in the physics world (collision on,
    full mass) just like the line-fault demo.
    """
    all_mids = sorted(system.modules.keys())
    n_total = len(all_mids)
    mid_to_idx: Dict[str, int] = {mid: i for i, mid in enumerate(all_mids)}

    pos0 = np.zeros((n_total, 3))
    for mid, idx in mid_to_idx.items():
        pos0[idx] = system.modules[mid].position

    bonded0 = np.zeros((n_total, n_total), dtype=bool)
    for (a, b) in system.edges:
        ia, ib = mid_to_idx[a], mid_to_idx[b]
        bonded0[ia, ib] = True
        bonded0[ib, ia] = True

    fault_body_idx = mid_to_idx[fault_id]
    module_ids = [mid for mid in all_mids if mid != fault_id]
    body_indices = {mid: mid_to_idx[mid] for mid in module_ids}

    fault_adjacent: List[str] = []
    for (a, b) in system.edges:
        if a == fault_id and b != fault_id:
            fault_adjacent.append(b)
        elif b == fault_id and a != fault_id:
            if a not in fault_adjacent:
                fault_adjacent.append(a)

    pre_damage_neighbor_slots: Dict[str, List[np.ndarray]] = {}
    fault_pos = system.modules[fault_id].position
    for mid in module_ids:
        mid_pos = system.modules[mid].position
        neighbors = system.get_neighbors(mid)
        dirs: List[np.ndarray] = []
        for nbr in neighbors:
            nbr_pos = system.modules[nbr].position
            direction = nbr_pos - mid_pos
            norm = np.linalg.norm(direction)
            if norm > 1e-9:
                direction = direction / norm
            dirs.append(direction)
        pre_damage_neighbor_slots[mid] = dirs

    original_positions = {
        mid: system.modules[mid].position.copy()
        for mid in all_mids
    }

    return BulletScenario(
        n_total=n_total,
        pos0=pos0,
        bonded0=bonded0,
        fault_id=fault_id,
        fault_body_idx=fault_body_idx,
        fault_adjacent=fault_adjacent,
        module_ids=module_ids,
        body_indices=body_indices,
        pre_damage_neighbor_slots=pre_damage_neighbor_slots,
        original_positions=original_positions,
    )


# ---------------------------------------------------------------------------
# Coagulation / Restructuring subclasses (body-frame token origin)
# ---------------------------------------------------------------------------

class _MCCoagulation(DecentralizedCoagulation):
    """Token origin stored in holder's body frame (drift-invariant)."""

    def _origin_world_for_pick_target(
        self, agent: ModuleAgent, pos: np.ndarray
    ) -> Optional[np.ndarray]:
        if agent.token is None:
            return None
        R = self.sim.body_rotation_matrix(agent.body_idx)
        return pos[agent.body_idx] + R @ agent.token.direction


class _MCRestructuring(DecentralizedRestructuring):
    """Token origin stored in holder's body frame (drift-invariant)."""

    def _origin_world_for_pick_target(
        self, agent: ModuleAgent, pos: np.ndarray
    ) -> Optional[np.ndarray]:
        if agent.token is None:
            return None
        R = self.sim.body_rotation_matrix(agent.body_idx)
        return pos[agent.body_idx] + R @ agent.token.direction


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

def _phase1_done(policy, sim: BulletSimulator) -> bool:
    return policy.is_connected() and not sim.has_active_pivots()


def _phase2_done(policy, sim: BulletSimulator) -> bool:
    if sim.has_active_pivots():
        return False
    for a in policy.agents.values():
        if a.state != ModuleState.IDLE:
            return False
        if a.incoming_tokens or a.token is not None:
            return False
    return True


def _run_phase(
    sim: BulletSimulator,
    policy,
    done_fn,
    dt: float,
    max_time: float,
    stall_interval: float,
    stall_patience: int,
) -> int:
    """Run a simulation phase, returns tick count."""
    stall_check_time = sim.sim_time
    last_move_count = 0
    stall_count = 0
    phase_start = sim.sim_time
    ticks = 0

    while sim.sim_time - phase_start < max_time:
        sim.step(dt)
        policy.tick()
        ticks += 1

        if done_fn(policy, sim):
            break

        if sim.sim_time - stall_check_time > stall_interval:
            if policy.successful_moves == last_move_count:
                stall_count += 1
                if stall_count >= stall_patience:
                    break
            else:
                stall_count = 0
                last_move_count = policy.successful_moves
            stall_check_time = sim.sim_time

    return ticks


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
    temperature: float = 0.01,
    pivot_exclusion_radius: int = 4,
    max_phase_time: float = 300.0,
    stall_interval: float = 20.0,
    stall_patience: int = 8,
    dt: float = 0.1,
) -> TrialResult:
    """Execute one PyBullet-based Monte Carlo trial.

    Generates a structure, injects faults one at a time, runs coagulation
    and restructuring through BulletSimulator + agent policies, and
    returns a TrialResult compatible with the existing MC framework.
    """
    if config_mode == CONFIG_MODE_TREE:
        system = create_random_tree_configuration(
            n_modules, seed=seed, mode_2d=mode_2d, balanced=False)
    else:
        system = create_random_configuration(
            n_modules, seed=seed, mode_2d=mode_2d,
            fully_connected=fully_connected)

    original_positions = {
        mid: module.position.copy()
        for mid, module in system.modules.items()
    }

    faulty_module_ids = select_faulty_modules(
        system, n_faults, seed + 1000, fault_mode)
    faulty_modules_set = set(faulty_module_ids)

    total_phase1_moves = 0
    total_phase2_moves = 0
    total_phase1_ticks = 0
    restored = True
    post_phase1_positions: Optional[Dict[str, np.ndarray]] = None

    for fault_id in faulty_module_ids:
        if not system.modules[fault_id].is_active:
            continue

        scenario = udqdg_to_bullet_scenario(system, fault_id)

        BulletSimulator.USE_ROLLING_SPHERE_PIVOT = True
        BulletSimulator.MAX_PIVOT_TIME = 30.0
        sim = BulletSimulator(
            scenario.n_total, scenario.pos0, scenario.bonded0, gui=False)

        try:
            coag = _MCCoagulation(
                sim,
                fault_id=scenario.fault_id,
                module_ids=scenario.module_ids,
                body_indices=scenario.body_indices,
            )
            coag.PIVOT_EXCLUSION_RADIUS = pivot_exclusion_radius
            coag.ALLOW_FAULT_AS_PIVOT_NEIGHBOR = True
            coag.TEMPERATURE = temperature
            coag.TOKEN_GEN_INTERVAL = 1.0
            coag.set_fault_adjacent(
                scenario.fault_adjacent, scenario.fault_body_idx)

            phase1_ticks = _run_phase(
                sim, coag, _phase1_done, dt,
                max_phase_time, stall_interval, stall_patience)

            phase1_connected = coag.is_connected()
            total_phase1_moves += coag.total_moves
            total_phase1_ticks += phase1_ticks

            if not phase1_connected:
                restored = False

            pos_snap = sim.get_positions()
            post_phase1_positions = {
                mid: pos_snap[scenario.body_indices[mid]].copy()
                for mid in scenario.module_ids
            }

            if phase1_connected:
                coag_moved: Set[str] = {m["module"] for m in coag.move_log}
                restruct = _MCRestructuring(
                    sim=sim,
                    module_ids=scenario.module_ids,
                    body_indices=scenario.body_indices,
                    coag_moved=coag_moved,
                    pre_damage_neighbor_slots=scenario.pre_damage_neighbor_slots,
                )
                restruct.PIVOT_EXCLUSION_RADIUS = pivot_exclusion_radius
                restruct.ALLOW_FAULT_AS_PIVOT_NEIGHBOR = True
                restruct.generate_initial_tokens()

                _run_phase(
                    sim, restruct, _phase2_done, dt,
                    max_phase_time, stall_interval, stall_patience)

                total_phase2_moves += restruct.total_moves
        finally:
            sim.disconnect()

        system.modules[fault_id].is_active = False

    if restored and post_phase1_positions is not None:
        shape_diff_phase1 = calculate_shape_difference(
            original_positions, post_phase1_positions, faulty_modules_set)

        final_positions = post_phase1_positions
        shape_diff = calculate_shape_difference(
            original_positions, final_positions, faulty_modules_set)
    else:
        shape_diff = None
        shape_diff_phase1 = None

    return TrialResult(
        trial_id=trial_id,
        n_modules=n_modules,
        n_faults=n_faults,
        seed=seed,
        restored=restored,
        phase1_moves=total_phase1_moves,
        phase2_moves=total_phase2_moves,
        shape_difference=shape_diff,
        shape_difference_phase1=shape_diff_phase1,
        phase1_iterations=total_phase1_ticks,
        total_moves=total_phase1_moves + total_phase2_moves,
        token_transmissions=0,
        fault_mode=fault_mode,
    )


# ---------------------------------------------------------------------------
# Aggregation loop (reuses MonteCarloResults)
# ---------------------------------------------------------------------------

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
    **trial_kwargs,
) -> MonteCarloResults:
    """Run Monte Carlo simulation via PyBullet (sequential only)."""
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    trials: List[TrialResult] = []
    iterator = tqdm(range(n_trials), desc=f"n={n_modules}", disable=not verbose)
    for i in iterator:
        trial_seed = seed + i
        result = run_single_bullet_trial(
            n_modules=n_modules,
            n_faults=n_faults,
            seed=trial_seed,
            trial_id=i,
            mode_2d=mode_2d,
            fully_connected=fully_connected,
            config_mode=config_mode,
            fault_mode=fault_mode,
            **trial_kwargs,
        )
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
    "fault_mode", "fault_pct",
]

TRIALS_HEADERS = [
    "trial_id", "n_modules", "n_faults", "seed",
    "restored", "phase1_moves", "phase2_moves",
    "shape_difference", "shape_difference_phase1",
    "phase1_iterations", "total_moves",
    "token_transmissions", "fault_mode",
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
    parser.add_argument("--chain-like", action="store_true",
                        help="Chain-like connectivity (default: fully-connected)")
    parser.add_argument("--tree", action="store_true",
                        help="Tree-based configuration (no cycles)")
    parser.add_argument("--dynamic-faults", action="store_true",
                        help="Faults = floor(n/10) for each n")
    parser.add_argument("--cluster-faults", action="store_true",
                        help="Cluster failure sweep: cluster sizes 2,3,4,5")
    parser.add_argument("--dynamic-pct", action="store_true",
                        help="Dynamic pct fault sweep: 10%%,20%%,30%% x 3 patterns")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="Coagulation temperature (default: 0.01)")
    parser.add_argument("--pivot-radius", type=int, default=4,
                        help="Pivot exclusion radius (default: 4)")
    parser.add_argument("--max-phase-time", type=float, default=300.0,
                        help="Max sim-seconds per phase (default: 300)")
    parser.add_argument("--stall-interval", type=float, default=20.0,
                        help="Seconds between stall checks (default: 20)")
    parser.add_argument("--stall-patience", type=int, default=8,
                        help="Stall windows before phase exit (default: 8)")
    parser.add_argument("--resume", type=str, default=None, metavar="PATH",
                        help="Resume from existing output directory")
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
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        completed = set()

    no_graphs = args.no_graphs
    all_n_values = list(range(n_min, n_max + 1, n_step))
    remaining = [n for n in all_n_values if n not in completed]

    if config_mode == CONFIG_MODE_TREE:
        connectivity_desc = "tree (no cycles)"
    elif fully_connected:
        connectivity_desc = "fully-connected (default)"
    else:
        connectivity_desc = "chain-like"

    configs_per_n = 1
    if cluster_faults:
        configs_per_n = 4
    elif dynamic_pct:
        configs_per_n = 9

    print("=" * 70)
    print("PYBULLET MONTE CARLO SIMULATION SWEEP")
    print("=" * 70)
    print("Parameters:")
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
    print(f"  Sequential (PyBullet not thread-safe)")
    if cluster_faults:
        print(f"  Cluster faults: enabled (sizes 2,3,4,5)")
    if dynamic_pct:
        print(f"  Dynamic pct: enabled (10%,20%,30% x random,random_clusters,localized)")
    if completed:
        print(f"  Resuming: {len(completed)}/{len(all_n_values)} n-values done")
    print("=" * 70)

    total_remaining = len(remaining) * n_trials * configs_per_n
    print(f"\nRunning {len(remaining)} n-values x {configs_per_n} configs x {n_trials} trials = {total_remaining:,} total trials")
    if completed:
        print(f"  (skipping {len(completed)} already-completed configurations)")
    print("This may take a while...\n")

    # --- Build sweep jobs ---
    sweep_jobs: List[Tuple] = []
    for n in remaining:
        if cluster_faults:
            for cluster_size in [2, 3, 4, 5]:
                sweep_jobs.append((n, cluster_size, FAULT_MODE_CLUSTER, ""))
        elif dynamic_pct:
            for pct in [0.10, 0.20, 0.30]:
                f = max(1, math.ceil(n * pct))
                pct_label = f"{int(pct * 100)}%"
                for fm in [FAULT_MODE_RANDOM, FAULT_MODE_RANDOM_CLUSTERS, FAULT_MODE_LOCALIZED]:
                    sweep_jobs.append((n, f, fm, pct_label))
        else:
            f = max(1, n // 10) if dynamic_faults else n_faults
            sweep_jobs.append((n, f, FAULT_MODE_RANDOM, ""))

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
            for job_idx, (n, f, fault_mode, fault_pct_label) in enumerate(job_iter):
                job_iter.set_postfix(n=n, f=f, mode=fault_mode)

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
                    temperature=temperature,
                    pivot_exclusion_radius=pivot_radius,
                    max_phase_time=max_phase_time,
                    stall_interval=stall_interval,
                    stall_patience=stall_patience,
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
                    fault_mode,
                    fault_pct_label,
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
                        trial.fault_mode,
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
