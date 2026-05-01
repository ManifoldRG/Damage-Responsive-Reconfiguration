"""
Star-fault demo driven by ``DecentralizedCoagulation`` + ``DecentralizedRestructuring``.

A 19-module star (6 arms of 3, central fault at M0) uses the same agent
pipeline as the line-fault demo.  The fault module is a real, full-mass,
collision-on body.  Fault-adjacent modules (the 6 inner-arm modules) emit
tokens; endpoints fold inward under the same ``is_movable`` / pivot logic.

Output: ``Media/star_fault_agent.mp4`` (matplotlib Agg) + optional 3D viewer.
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

from examples.star_fault_scenario import build_star_fault_scenario
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
    DisplacementRestructuring,
    ModuleAgent,
    ModuleState,
)
from src.bullet_sim import BulletSimulator


class StarFaultCoagulation(DecentralizedCoagulation):
    """Body-frame token origin for the star scenario."""

    def _origin_world_for_pick_target(
        self, agent: ModuleAgent, pos: np.ndarray
    ) -> Optional[np.ndarray]:
        if agent.token is None:
            return None
        R = self.sim.body_rotation_matrix(agent.body_idx)
        return pos[agent.body_idx] + R @ agent.token.direction


class StarFaultRestructuring(DecentralizedRestructuring):
    """Body-frame token origin for the star scenario."""

    def _origin_world_for_pick_target(
        self, agent: ModuleAgent, pos: np.ndarray
    ) -> Optional[np.ndarray]:
        if agent.token is None:
            return None
        R = self.sim.body_rotation_matrix(agent.body_idx)
        return pos[agent.body_idx] + R @ agent.token.direction


class StarFaultDisplacementRestructuring(DisplacementRestructuring):
    """Displacement-guided restructuring for the star-fault scenario."""
    pass


def _first_pivoting_body_idx(policy) -> Optional[int]:
    for a in policy.agents.values():
        if a.state in (ModuleState.PIVOTING, ModuleState.REVERSING):
            return a.body_idx
    return None


def _pivoting_mids(policy) -> List[str]:
    return [mid for mid, a in policy.agents.items()
            if a.state in (ModuleState.PIVOTING, ModuleState.REVERSING)]


def _highlight_indices(policy, fault_body_idx: int):
    for a in policy.agents.values():
        if a.state in (ModuleState.PIVOTING, ModuleState.REVERSING):
            ax = a.pivot_axis_idx if a.pivot_axis_idx is not None else a.body_idx
            ho = a.handoff_idx if a.pivot_type == "lateral" else None
            return a.body_idx, ax, ho
    return fault_body_idx, fault_body_idx, None


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
        description="Agent-driven star-fault demo: coagulation + restructuring.",
    )
    parser.add_argument(
        "--arm-len", type=int, default=3,
        help="Modules per arm (default 3 → 19 total).",
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
        help="MP4 path (default: Media/star_fault_agent.mp4).",
    )
    parser.add_argument(
        "--no-viewer", action="store_true",
        help="Skip the interactive 3D viewer.",
    )
    parser.add_argument(
        "--restructuring-method", type=str, default="rendezvous",
        choices=["rendezvous", "displacement"],
        help="Phase 2 method: rendezvous tokens or displacement-guided.",
    )
    args = parser.parse_args()

    scenario = build_star_fault_scenario(arm_len=args.arm_len)
    print(f"Star: {scenario.n} modules, arm_len={scenario.arm_len}, "
          f"fault=M0, fault_adjacent={scenario.fault_adjacent}")

    BulletSimulator.USE_ROLLING_SPHERE_PIVOT = True
    BulletSimulator.MAX_PIVOT_TIME = 30.0
    sim = BulletSimulator(
        scenario.n, scenario.pos0, scenario.bonded0, gui=False)

    coag = StarFaultCoagulation(
        sim,
        fault_id=scenario.fault_id,
        module_ids=scenario.module_ids,
        body_indices=scenario.body_indices,
    )
    coag.PIVOT_EXCLUSION_RADIUS = 4
    coag.ALLOW_FAULT_AS_PIVOT_NEIGHBOR = True
    coag.TOKEN_GEN_INTERVAL = 10.0
    coag.set_fault_adjacent(scenario.fault_adjacent, scenario.fault_body_idx)

    dt = 0.1
    frames: list = []
    snapshots: list = []
    history: list = []
    last_capture = -1.0

    fig, ax_fig = plt.subplots(1, 1, figsize=(6.0, 6.0), facecolor="#14141e")

    def capture_frame(phase: str, policy) -> None:
        nonlocal last_capture
        if sim.sim_time - last_capture < args.frame_interval:
            return
        last_capture = sim.sim_time
        pos = sim.get_positions()
        p_idx, ax_idx, ho_idx = _highlight_indices(
            policy, scenario.fault_body_idx)
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
            ghost_goal_idx=scenario.fault_body_idx,
        )
        _append_frame(fig, frames)
        snapshots.append({
            "pos": pos.copy(),
            "bonds": sim.get_bond_matrix().copy(),
            "pivot_idx": p_idx,
            "axis_idx": ax_idx,
            "fault_idx": scenario.fault_body_idx,
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

    original_positions: dict = {}
    pos0 = sim.get_positions()
    for mid in scenario.module_ids:
        original_positions[mid] = pos0[scenario.body_indices[mid]].copy()

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
            if args.restructuring_method == "displacement":
                restruct = StarFaultDisplacementRestructuring(
                    sim=sim,
                    module_ids=scenario.module_ids,
                    body_indices=scenario.body_indices,
                    coag_moved=coag_moved,
                    original_positions=original_positions,
                )
            else:
                restruct = StarFaultRestructuring(
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
                f"Phase 2 ({args.restructuring_method}): done={phase2_done}  "
                f"moves={restruct.total_moves}  "
                f"movers={sorted({m['module'] for m in restruct.move_log})}")

    finally:
        sim.disconnect()
        plt.close(fig)

    default_name = f"star_fault_agent_{args.restructuring_method}.mp4"
    out = args.output or media_path(default_name)
    if frames:
        iio.imwrite(out, frames, fps=12, codec="libx264")
        print(f"Wrote {len(frames)} frames to {os.path.abspath(out)}")
    else:
        print("No frames captured.")

    if not args.no_viewer and snapshots:
        launch_3d_viewer(snapshots, module_radius)


if __name__ == "__main__":
    main()
