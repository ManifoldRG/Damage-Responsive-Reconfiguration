"""
Vicsek-fractal multi-fault demo.

A level-2 4-arm 2D Vicsek fractal (25 modules, 5 fault centres) runs
through the same coagulation + restructuring pipeline as the line/star
demos.  This is the first multi-fault scenario for the async PyBullet
agent system: all 5 sub-star centres are simultaneously passive fault
bodies and the 20 remaining modules must reconnect around them.

Output: ``Media/vicsek_fault_agent.mp4`` + optional 3D viewer.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Set

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import imageio.v3 as iio
import matplotlib.pyplot as plt

from examples.vicsek_fault_scenario import build_vicsek_fault_scenario
from examples.media_paths import media_path
from examples.render_five_module_pivot_chain import (
    _append_frame,
    _pivot_debug_overlay,
    draw_frame,
)
from examples.render_line_fault_agent import launch_3d_viewer
from src.agent_policy import (
    DecentralizedCoagulation,
    DecentralizedRestructuring,
    ModuleAgent,
    ModuleState,
)
from src.bullet_sim import BulletSimulator


class VicsekCoagulation(DecentralizedCoagulation):
    """Body-frame token origin for the Vicsek scenario."""

    def _origin_world_for_pick_target(
        self, agent: ModuleAgent, pos: np.ndarray
    ) -> Optional[np.ndarray]:
        if agent.token is None:
            return None
        R = self.sim.body_rotation_matrix(agent.body_idx)
        return pos[agent.body_idx] + R @ agent.token.direction


class VicsekRestructuring(DecentralizedRestructuring):
    """Body-frame token origin for the Vicsek scenario."""

    def _origin_world_for_pick_target(
        self, agent: ModuleAgent, pos: np.ndarray
    ) -> Optional[np.ndarray]:
        if agent.token is None:
            return None
        R = self.sim.body_rotation_matrix(agent.body_idx)
        return pos[agent.body_idx] + R @ agent.token.direction


def _first_pivoting_body_idx(policy) -> Optional[int]:
    for a in policy.agents.values():
        if a.state in (ModuleState.PIVOTING, ModuleState.REVERSING):
            return a.body_idx
    return None


def _pivoting_mids(policy) -> List[str]:
    return [mid for mid, a in policy.agents.items()
            if a.state in (ModuleState.PIVOTING, ModuleState.REVERSING)]


def _highlight_indices(policy, fault_body_idxs: List[int]):
    for a in policy.agents.values():
        if a.state in (ModuleState.PIVOTING, ModuleState.REVERSING):
            ax = a.pivot_axis_idx if a.pivot_axis_idx is not None else a.body_idx
            ho = a.handoff_idx if a.pivot_type == "lateral" else None
            return a.body_idx, ax, ho
    fb = fault_body_idxs[0] if fault_body_idxs else 0
    return fb, fb, None


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


def main():
    parser = argparse.ArgumentParser(
        description="Agent-driven Vicsek-fractal multi-fault demo.",
    )
    parser.add_argument(
        "--max-time", type=float, default=600.0,
        help="Wall sim time limit per phase (seconds).",
    )
    parser.add_argument(
        "--stall-interval", type=float, default=20.0,
        help="Seconds between stall checks.",
    )
    parser.add_argument(
        "--stall-patience", type=int, default=8,
        help="Consecutive stall windows before exit.",
    )
    parser.add_argument(
        "--frame-interval", type=float, default=0.5,
        help="Seconds between video frames.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="MP4 path (default: Media/vicsek_fault_agent.mp4).",
    )
    parser.add_argument(
        "--no-viewer", action="store_true",
        help="Skip the interactive 3D viewer.",
    )
    args = parser.parse_args()

    scenario = build_vicsek_fault_scenario()
    print(f"Vicsek: {scenario.n} modules, "
          f"faults={scenario.fault_ids} (body {scenario.fault_body_idxs}), "
          f"active={len(scenario.module_ids)}")

    BulletSimulator.USE_ROLLING_SPHERE_PIVOT = True
    BulletSimulator.MAX_PIVOT_TIME = 30.0
    sim = BulletSimulator(
        scenario.n, scenario.pos0, scenario.bonded0, gui=False)

    # Use the first fault as the nominal fault_id for the constructor
    coag = VicsekCoagulation(
        sim,
        fault_id=scenario.fault_ids[0],
        module_ids=scenario.module_ids,
        body_indices=scenario.body_indices,
    )
    coag.PIVOT_EXCLUSION_RADIUS = 4
    coag.ALLOW_FAULT_AS_PIVOT_NEIGHBOR = True
    coag.TOKEN_GEN_INTERVAL = 1.0
    coag.set_multi_fault_adjacent(
        fault_ids=scenario.fault_ids,
        fault_body_idxs=scenario.fault_body_idxs,
        adjacent_map=scenario.adjacent_map,
    )

    dt = 0.1
    frames: list = []
    snapshots: list = []
    history: list = []
    last_capture = -1.0

    fig, ax_fig = plt.subplots(1, 1, figsize=(8.0, 8.0), facecolor="#14141e")

    def capture_frame(phase: str, policy) -> None:
        nonlocal last_capture
        if sim.sim_time - last_capture < args.frame_interval:
            return
        last_capture = sim.sim_time
        pos = sim.get_positions()
        p_idx, ax_idx, ho_idx = _highlight_indices(
            policy, scenario.fault_body_idxs)
        angle_err_str, attract_pt, repel_pt, goal_com = _pivot_debug_overlay(
            sim, p_idx, pos)
        active_movers = _pivoting_mids(policy)
        mov_str = "+".join(active_movers) if active_movers else "-"
        title = (
            f"t={sim.sim_time:.1f}s  phase={phase}  "
            f"moves={policy.total_moves}  "
            f"pivoting=[{mov_str}]{angle_err_str}"
        )
        draw_frame(
            ax_fig, sim, pos,
            p_idx, ax_idx, ho_idx,
            title=title,
            history=history,
            attract_pt=attract_pt,
            repel_pt=repel_pt,
            goal_com=goal_com,
            ghost_goal_idx=scenario.fault_body_idxs[0],
        )
        _append_frame(fig, frames)
        snapshots.append({
            "pos": pos.copy(),
            "bonds": sim.get_bond_matrix().copy(),
            "pivot_idx": p_idx,
            "axis_idx": ax_idx,
            "fault_idx": scenario.fault_body_idxs[0],
            "t": sim.sim_time,
            "phase": phase,
        })

    def run_phase(phase: str, policy) -> None:
        stall_check_time = sim.sim_time
        last_move_count = 0
        stall_count = 0
        phase_start = sim.sim_time
        done_fn = _phase1_done if phase == "coag" else _phase2_done
        last_heartbeat = sim.sim_time

        while sim.sim_time - phase_start < args.max_time:
            sim.step(dt)
            policy.tick()

            if sim.sim_time - last_heartbeat > 10.0:
                print(f"[heartbeat] t={sim.sim_time:.1f}s  "
                      f"successful={policy.successful_moves}  "
                      f"total={policy.total_moves}", flush=True)
                last_heartbeat = sim.sim_time
            if done_fn(policy, sim):
                break

            trail_idx = _first_pivoting_body_idx(policy)
            if trail_idx is not None:
                history.append(sim.get_positions()[trail_idx][:2].copy())

            capture_frame(phase, policy)

            if sim.sim_time - stall_check_time > args.stall_interval:
                if policy.successful_moves == last_move_count:
                    stall_count += 1
                    print(f"[stall] no new successful moves for "
                          f"{stall_count}/{args.stall_patience} intervals "
                          f"(successful={policy.successful_moves}, "
                          f"total={policy.total_moves}, t={sim.sim_time:.1f}s)",
                          flush=True)
                    if stall_count >= args.stall_patience:
                        print(f"Phase {phase!r} stalled; exiting.", flush=True)
                        break
                else:
                    stall_count = 0
                    last_move_count = policy.successful_moves
                    print(f"[stall] reset — successful_moves="
                          f"{policy.successful_moves}, "
                          f"total={policy.total_moves}, t={sim.sim_time:.1f}s",
                          flush=True)
                stall_check_time = sim.sim_time

    module_radius = float(sim.MODULE_RADIUS)

    try:
        # Phase 1: Coagulation
        run_phase("coag", coag)
        phase1_connected = coag.is_connected()
        coag_moved: Set[str] = {m["module"] for m in coag.move_log}
        print(
            f"Phase 1: connected={phase1_connected}  "
            f"moves={coag.total_moves}  movers={sorted(coag_moved)}")

        # Phase 2: Restructuring
        if phase1_connected:
            restruct = VicsekRestructuring(
                sim=sim,
                module_ids=scenario.module_ids,
                body_indices=scenario.body_indices,
                coag_moved=coag_moved,
                pre_damage_neighbor_slots=scenario.pre_damage_neighbor_slots,
            )
            restruct.PIVOT_EXCLUSION_RADIUS = coag.PIVOT_EXCLUSION_RADIUS
            restruct.ALLOW_FAULT_AS_PIVOT_NEIGHBOR = True
            restruct.generate_initial_tokens()
            run_phase("restruct", restruct)
            phase2_done = coag.is_connected()
            print(
                f"Phase 2: done={phase2_done}  "
                f"moves={restruct.total_moves}  "
                f"movers={sorted({m['module'] for m in restruct.move_log})}")

    finally:
        sim.disconnect()
        plt.close(fig)

    out = args.output or media_path("vicsek_fault_agent.mp4")
    if frames:
        iio.imwrite(out, frames, fps=12, codec="libx264")
        print(f"Wrote {len(frames)} frames to {os.path.abspath(out)}")
    else:
        print("No frames captured.")

    if not args.no_viewer and snapshots:
        launch_3d_viewer(snapshots, module_radius)


if __name__ == "__main__":
    main()
