"""Render one MC bullet trial with SIMULTANEOUS multi-fault injection.

The non-bullet MC nominally injects all faults at once, but in practice the
runners (both run_comparison_trial and run_single_bullet_trial) loop fault
by fault. The policy itself supports the simultaneous path via
DecentralizedCoagulation.set_multi_fault_adjacent. This renderer uses that
path: all faults are marked at once, one bullet world is built with every
fault as a real obstacle body, and a single coag+restruct phase pair runs
over the entire damaged structure.

Adaptive zoom: per-frame bbox over all module positions with EMA smoothing.

Output: Media/mc_n50_fault_simul.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Set

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import imageio.v3 as iio
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from examples.media_paths import media_path
from run_bullet_monte_carlo import (
    _MCCoagulation,
    _MCDisplacementRestructuring,
    _MCRestructuring,
    _phase1_done,
    _phase2_done,
)
from src.agent_policy import ModuleState
from src.bullet_sim import BulletSimulator
from src.configurations import create_random_configuration
from src.monte_carlo import (
    FAULT_MODE_RANDOM,
    _causes_disconnection,
    select_faulty_modules,
)


def build_multi_fault_scenario(system, fault_ids: List[str]):
    """Build a bullet scenario with multiple faults injected at once.

    All fault modules become real bodies in the physics world (full mass,
    collision-on) — same treatment as the single-fault scenario builder.
    Returns a dict with everything the policy and sim need.
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

    fault_set = set(fault_ids)
    fault_body_idxs = [mid_to_idx[fid] for fid in fault_ids]
    module_ids = [mid for mid in all_mids if mid not in fault_set]
    body_indices = {mid: mid_to_idx[mid] for mid in module_ids}

    # Adjacency: any non-fault module that neighbors at least one fault.
    # set_multi_fault_adjacent wants Dict[active_module_id -> fault_body_idx],
    # one fault per adjacent module (pick first encountered).
    adjacent_map: Dict[str, int] = {}
    for (a, b) in system.edges:
        if a in fault_set and b not in fault_set:
            adjacent_map.setdefault(b, mid_to_idx[a])
        elif b in fault_set and a not in fault_set:
            adjacent_map.setdefault(a, mid_to_idx[b])

    pre_damage_neighbor_slots: Dict[str, List[np.ndarray]] = {}
    for mid in module_ids:
        mid_pos = system.modules[mid].position
        dirs: List[np.ndarray] = []
        for nbr in system.get_neighbors(mid):
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

    return dict(
        n_total=n_total,
        pos0=pos0,
        bonded0=bonded0,
        fault_ids=list(fault_ids),
        fault_body_idxs=fault_body_idxs,
        adjacent_map=adjacent_map,
        module_ids=module_ids,
        body_indices=body_indices,
        pre_damage_neighbor_slots=pre_damage_neighbor_slots,
        original_positions=original_positions,
    )


def _pivoting_body_indices(policy):
    return [a.body_idx for a in policy.agents.values()
            if a.state in (ModuleState.PIVOTING, ModuleState.REVERSING)]


def _draw_frame(
    ax,
    sim: BulletSimulator,
    positions: np.ndarray,
    pivoting_idx: List[int],
    fault_idxs: Set[int],
    title: str,
    view_bbox,
):
    ax.clear()
    R = float(sim.MODULE_RADIUS)
    N = len(positions)

    bm = sim.get_bond_matrix()
    for i in range(N):
        for j in range(i + 1, N):
            if bm[i, j]:
                ax.plot(
                    [positions[i][0], positions[j][0]],
                    [positions[i][1], positions[j][1]],
                    "-", color="#888", linewidth=1.2, zorder=1,
                )

    piv_set = set(pivoting_idx)
    for idx in range(N):
        if idx in fault_idxs:
            color = "#c084fc"
            alpha = 0.45
        elif idx in piv_set:
            color = "#ff8c00"
            alpha = 0.95
        else:
            color = "#00d9ff"
            alpha = 0.85
        c = mpatches.Circle(
            (positions[idx][0], positions[idx][1]),
            R * 0.9, color=color, alpha=alpha, zorder=2,
        )
        ax.add_patch(c)

    xmin, xmax, ymin, ymax = view_bbox
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_facecolor("#14141e")
    ax.set_title(title, color="white", fontsize=9)
    ax.tick_params(colors="#555", labelsize=5)
    for spine in ax.spines.values():
        spine.set_color("#333")


def _append_frame(fig, frames: list) -> None:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frames.append(buf.reshape(h, w, 4)[:, :, :3].copy())


def _compute_bbox(positions: np.ndarray, pad: float):
    xs = positions[:, 0]
    ys = positions[:, 1]
    xmin, xmax = float(xs.min()) - pad, float(xs.max()) + pad
    ymin, ymax = float(ys.min()) - pad, float(ys.max()) + pad
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin)
    return cx - half, cx + half, cy - half, cy + half


def _ema_bbox(prev, new, alpha: float):
    if prev is None:
        return new
    return tuple(alpha * n + (1 - alpha) * p for p, n in zip(prev, new))


def run_phase_with_capture(
    sim, policy, done_fn, dt, stall_interval, stall_patience,
    fig, ax_fig, frames, fault_body_idxs,
    capture_state, frame_interval, phase_label,
):
    stall_check_time = sim.sim_time
    last_move_count = 0
    stall_count = 0
    last_heartbeat = sim.sim_time

    while True:
        sim.step(dt)
        policy.tick()

        if sim.sim_time - last_heartbeat > 10.0:
            print(f"  [{phase_label}] t={sim.sim_time:.1f}s "
                  f"successful={policy.successful_moves} "
                  f"total={policy.total_moves}", flush=True)
            last_heartbeat = sim.sim_time

        if done_fn(policy, sim):
            break

        if sim.sim_time - capture_state["last_capture"] >= frame_interval:
            capture_state["last_capture"] = sim.sim_time
            pos = sim.get_positions()
            new_bbox = _compute_bbox(pos, pad=2.0)
            capture_state["view_bbox"] = _ema_bbox(
                capture_state["view_bbox"], new_bbox, alpha=0.15)
            piv = _pivoting_body_indices(policy)
            title = (
                f"phase={phase_label}  t={sim.sim_time:.1f}s  "
                f"moves={policy.total_moves}  pivoting={len(piv)}"
            )
            _draw_frame(
                ax_fig, sim, pos, piv, fault_body_idxs,
                title, capture_state["view_bbox"])
            _append_frame(fig, frames)

        if sim.sim_time - stall_check_time > stall_interval:
            if policy.successful_moves == last_move_count:
                stall_count += 1
                if stall_count >= stall_patience:
                    print(f"  [{phase_label}] stalled at t={sim.sim_time:.1f}s",
                          flush=True)
                    break
            else:
                stall_count = 0
                last_move_count = policy.successful_moves
            stall_check_time = sim.sim_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--n-faults", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--safety-radius", type=int, default=4)
    parser.add_argument("--pivot-exclusion-radius", type=int, default=4)
    parser.add_argument("--restructuring-method", type=str, default="displacement",
                        choices=["displacement", "rendezvous"])
    parser.add_argument("--frame-interval", type=float, default=0.4)
    parser.add_argument("--stall-interval", type=float, default=10.0)
    parser.add_argument("--stall-patience", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-pivot-time", type=float, default=20.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    print(f"Building n={args.n} structure (seed={args.seed})", flush=True)

    system = None
    faulty_module_ids: List[str] = []
    for attempt in range(50):
        gen_seed = args.seed + attempt * 9973
        system = create_random_configuration(
            args.n, seed=gen_seed, mode_2d=False, fully_connected=True)
        faulty_module_ids = select_faulty_modules(
            system, args.n_faults, gen_seed + 1000, FAULT_MODE_RANDOM)
        if _causes_disconnection(system, faulty_module_ids):
            print(f"  gen_seed={gen_seed} faults={faulty_module_ids}", flush=True)
            break

    # Mark every fault at once so the system reflects simultaneous injection.
    for fid in faulty_module_ids:
        system.mark_fault(fid)

    scenario = build_multi_fault_scenario(system, faulty_module_ids)
    print(f"Built scenario: {scenario['n_total']} bodies, "
          f"{len(scenario['fault_ids'])} faults, "
          f"{len(scenario['adjacent_map'])} fault-adjacent modules",
          flush=True)

    BulletSimulator.USE_ROLLING_SPHERE_PIVOT = True
    BulletSimulator.MAX_PIVOT_TIME = float(args.max_pivot_time)

    sim = BulletSimulator(
        scenario["n_total"], scenario["pos0"], scenario["bonded0"],
        gui=False, module_shape="sphere")

    fig, ax_fig = plt.subplots(1, 1, figsize=(7.0, 7.0), facecolor="#14141e")
    frames: list = []
    fault_idx_set: Set[int] = set(scenario["fault_body_idxs"])
    capture_state = {"last_capture": -1e9, "view_bbox": None}

    try:
        # _MCCoagulation expects a single fault_id; we pass the first and
        # then override the multi-fault state immediately after.
        coag = _MCCoagulation(
            sim,
            fault_id=scenario["fault_ids"][0],
            module_ids=scenario["module_ids"],
            body_indices=scenario["body_indices"],
        )
        coag.PIVOT_EXCLUSION_RADIUS = args.pivot_exclusion_radius
        coag._safety_radius = args.safety_radius
        coag.ALLOW_FAULT_AS_PIVOT_NEIGHBOR = True
        coag.TEMPERATURE = 0.01
        coag.TOKEN_GEN_INTERVAL = 1.0
        coag.set_multi_fault_adjacent(
            fault_ids=scenario["fault_ids"],
            fault_body_idxs=scenario["fault_body_idxs"],
            adjacent_map=scenario["adjacent_map"],
        )

        # Capture initial frame.
        pos0 = sim.get_positions()
        capture_state["view_bbox"] = _compute_bbox(pos0, pad=2.0)
        _draw_frame(ax_fig, sim, pos0, [], fault_idx_set,
                    f"phase=coag-init  faults={len(scenario['fault_ids'])}",
                    capture_state["view_bbox"])
        _append_frame(fig, frames)
        capture_state["last_capture"] = sim.sim_time

        print("Phase 1: coagulation", flush=True)
        run_phase_with_capture(
            sim, coag, _phase1_done, args.dt,
            args.stall_interval, args.stall_patience,
            fig, ax_fig, frames, fault_idx_set,
            capture_state, args.frame_interval, phase_label="coag",
        )
        phase1_connected = coag.is_connected()
        print(f"Phase 1 done: connected={phase1_connected} "
              f"moves={coag.total_moves}", flush=True)

        if phase1_connected:
            coag_moved: Set[str] = {m["module"] for m in coag.move_log}
            if args.restructuring_method == "displacement":
                restruct = _MCDisplacementRestructuring(
                    sim=sim,
                    module_ids=scenario["module_ids"],
                    body_indices=scenario["body_indices"],
                    coag_moved=coag_moved,
                    original_positions=scenario["original_positions"],
                )
            else:
                restruct = _MCRestructuring(
                    sim=sim,
                    module_ids=scenario["module_ids"],
                    body_indices=scenario["body_indices"],
                    coag_moved=coag_moved,
                    pre_damage_neighbor_slots=scenario["pre_damage_neighbor_slots"],
                    token_strategy="furthest",
                )
            restruct.PIVOT_EXCLUSION_RADIUS = args.pivot_exclusion_radius
            restruct._safety_radius = args.safety_radius
            restruct.ALLOW_FAULT_AS_PIVOT_NEIGHBOR = True
            restruct.generate_initial_tokens()

            print("Phase 2: restructuring", flush=True)
            run_phase_with_capture(
                sim, restruct, _phase2_done, args.dt,
                args.stall_interval, args.stall_patience,
                fig, ax_fig, frames, fault_idx_set,
                capture_state, args.frame_interval, phase_label="restruct",
            )
            print(f"Phase 2 done: moves={restruct.total_moves}", flush=True)
    finally:
        sim.disconnect()
        plt.close(fig)

    default_name = (f"mc_n{args.n}_f{args.n_faults}_sr{args.safety_radius}"
                    f"_simul.mp4")
    out = args.output or media_path(default_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if frames:
        iio.imwrite(out, frames, fps=args.fps, codec="libx264")
        print(f"Wrote {len(frames)} frames to {os.path.abspath(out)}",
              flush=True)
    else:
        print("No frames captured.", flush=True)


if __name__ == "__main__":
    main()
