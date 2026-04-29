"""
Line-fault demo driven by ``DecentralizedCoagulation`` + ``DecentralizedRestructuring``.

An ``n``-module Y-line has a fault at the center (default M5 for n=11). The
fault module is a **real, full-mass, collision-on** body — a genuine obstacle,
not a ghost. Fault-adjacent modules emit tokens pointing at the fault. Tokens
propagate outward, and because ``is_movable`` rejects articulation points,
**only the two endpoint modules are initially movable** (M0 and M10 for the
default line); the chain folds inward as endpoints retract into new slots.

**Concurrent pivots** are allowed under a **2-hop local exclusion**
(``PIVOT_EXCLUSION_RADIUS = 2``): a module may start a pivot only if no body
within 2 bonded hops is currently PIVOTING. Modules ≥3 hops apart (e.g., the
two ends of the 11-line) may therefore pivot simultaneously without the
physics neighborhoods colliding. The legacy global lock is still the default
for every other caller.

**Phase 1 — Coagulation**: runs until the active modules form a single
connected component (graph is reconnected around the fault) or stall.

**Phase 2 — Restructuring**: ``DecentralizedRestructuring`` takes over; the
phase-1 movers continue pivoting to fill empty lattice slots announced by
non-movers' tokens. Terminates when all tokens are drained and no pivot is
active, or on stall.

Output: ``Media/line_fault_agent.mp4`` (matplotlib Agg).
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

from examples.line_fault_scenario import build_y_line_fault_scenario
from examples.media_paths import media_path
from examples.render_five_module_pivot_chain import (
    _append_frame,
    _pivot_debug_overlay,
    draw_frame,
)
from src.agent_policy import (
    DecentralizedCoagulation,
    DecentralizedRestructuring,
    ModuleAgent,
    ModuleState,
)
from src.bullet_sim import BulletSimulator


def launch_3d_viewer(snapshots: list, radius: float) -> None:
    """Interactive 3D post-hoc viewer using pyvista."""
    import pyvista as pv

    if not snapshots:
        print("No snapshots to display.")
        return

    n_modules = len(snapshots[0]["pos"])
    n_frames = len(snapshots)

    pl = pv.Plotter()
    pl.set_background("#14141e")

    color_default = "#00d9ff"
    color_pivot = "#ff8c00"
    color_axis = "#00ffaa"
    color_fault = "#c084fc"

    snap = snapshots[0]
    sphere_actors = []
    for i in range(n_modules):
        s = pv.Sphere(radius=radius * 0.9, center=snap["pos"][i])
        actor = pl.add_mesh(s, color=color_default, opacity=0.85)
        sphere_actors.append(actor)

    label_actor = pl.add_point_labels(
        snap["pos"], [f"M{i}" for i in range(n_modules)],
        font_size=10, text_color="white",
        shape=None, render_points_as_spheres=False,
        always_visible=True,
    )

    bond_lines = []
    bonds = snap["bonds"]
    for i in range(n_modules):
        for j in range(i + 1, n_modules):
            if bonds[i, j]:
                line = pv.Line(snap["pos"][i], snap["pos"][j])
                actor = pl.add_mesh(line, color="#aaaaaa", line_width=2)
                bond_lines.append((i, j, actor))

    title_actor = pl.add_text(
        f"t={snap['t']:.1f}s  phase={snap['phase']}  frame=0/{n_frames - 1}",
        position="upper_left", font_size=10, color="white",
    )

    def update_frame(value):
        nonlocal label_actor
        idx = int(round(value))
        idx = max(0, min(idx, n_frames - 1))
        snap = snapshots[idx]
        pos = snap["pos"]
        p_idx = snap["pivot_idx"]
        ax_idx = snap["axis_idx"]
        f_idx = snap["fault_idx"]

        for i in range(n_modules):
            new_sphere = pv.Sphere(radius=radius * 0.9, center=pos[i])
            if i == p_idx:
                c = color_pivot
            elif i == ax_idx:
                c = color_axis
            elif i == f_idx:
                c = color_fault
            else:
                c = color_default
            sphere_actors[i].mapper.SetInputData(new_sphere)
            sphere_actors[i].GetProperty().SetColor(
                pv.Color(c).float_rgb)

        pl.remove_actor(label_actor)
        label_actor = pl.add_point_labels(
            pos, [f"M{i}" for i in range(n_modules)],
            font_size=10, text_color="white",
            shape=None, render_points_as_spheres=False,
            always_visible=True,
        )

        for i_b, j_b, actor in bond_lines:
            actor.SetVisibility(False)
        bond_lines.clear()

        bonds = snap["bonds"]
        for i in range(n_modules):
            for j in range(i + 1, n_modules):
                if bonds[i, j]:
                    line = pv.Line(pos[i], pos[j])
                    actor = pl.add_mesh(line, color="#aaaaaa", line_width=2)
                    bond_lines.append((i, j, actor))

        title_actor.SetText(
            0,
            f"t={snap['t']:.1f}s  phase={snap['phase']}  "
            f"frame={idx}/{n_frames - 1}",
        )
        pl.render()

    pl.add_slider_widget(
        update_frame,
        rng=[0, n_frames - 1],
        value=0,
        title="Frame",
        pointa=(0.1, 0.05),
        pointb=(0.9, 0.05),
        style="modern",
    )

    pl.show()


class LineFaultCoagulation(DecentralizedCoagulation):
    """Line-fault coagulation: parent behavior + body-frame token origin.

    The fault module is a real, full-mass, collision-on body, so the parent
    ``pick_target``/``is_movable`` logic suffices without ghost-goal hooks.

    Token directions are stored in each holder's body frame, making
    propagation drift-invariant.  To reconstruct the world-frame scoring
    origin we rotate the body-frame direction back: ``pos + R @ direction``.
    """

    def _origin_world_for_pick_target(
        self, agent: ModuleAgent, pos: np.ndarray
    ) -> Optional[np.ndarray]:
        if agent.token is None:
            return None
        R = self.sim.body_rotation_matrix(agent.body_idx)
        return pos[agent.body_idx] + R @ agent.token.direction


class LineFaultRestructuring(DecentralizedRestructuring):
    """Line-fault restructuring with the same token-origin correction.

    Token directions are stored in the holder's body frame.  To reconstruct
    the world-frame scoring origin we rotate back via ``R @ direction``.
    """

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


def _highlight_indices(policy, fault_body_idx: int):
    """Pick one module to highlight as the 'pivot', plus its axis and handoff.

    If any module is pivoting, use the first one; otherwise fall back to the
    fault body so the frame is still informative.
    """
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
        description=(
            "Agent-driven line-fault demo: coagulation + restructuring, "
            "matplotlib 2D, local 2-hop pivot exclusion."
        ),
    )
    parser.add_argument(
        "--n", type=int, default=11,
        help="Number of modules including the fault (default 11).",
    )
    parser.add_argument(
        "--fault-index", type=int, default=None,
        help="Body index of the fault module (default: n // 2).",
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
        help="MP4 path (default: Media/line_fault_agent.mp4).",
    )
    parser.add_argument(
        "--no-viewer", action="store_true",
        help="Skip the interactive 3D viewer after the simulation.",
    )
    args = parser.parse_args()

    scenario = build_y_line_fault_scenario(
        n=args.n, fault_body_idx=args.fault_index)

    BulletSimulator.USE_ROLLING_SPHERE_PIVOT = True
    BulletSimulator.MAX_PIVOT_TIME = 30.0
    sim = BulletSimulator(
        scenario.n, scenario.pos0, scenario.bonded0, gui=False)
    # NOTE: fault body kept as full-mass, collision-on. No set_body_mass or
    # set_body_collision_with_all overrides (cf. five_module_pivot_chain_agent).

    coag = LineFaultCoagulation(
        sim,
        fault_id=scenario.fault_id,
        module_ids=scenario.module_ids,
        body_indices=scenario.body_indices,
    )
    coag.PIVOT_EXCLUSION_RADIUS = 4
    coag.ALLOW_FAULT_AS_PIVOT_NEIGHBOR = True
    coag.TOKEN_GEN_INTERVAL = 1.0
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
                print(f"[heartbeat] t={sim.sim_time:.1f}s  successful={policy.successful_moves}  total={policy.total_moves}", flush=True)
                last_heartbeat = sim.sim_time
            if done_fn(policy, sim):
                break

            # Trail: first currently-pivoting module.
            trail_idx = _first_pivoting_body_idx(policy)
            if trail_idx is not None:
                history.append(sim.get_positions()[trail_idx][:2].copy())

            capture_frame(phase, policy)

            if sim.sim_time - stall_check_time > args.stall_interval:
                if policy.successful_moves == last_move_count:
                    stall_count += 1
                    print(f"[stall] no new successful moves for {stall_count}/{args.stall_patience} intervals "
                          f"(successful={policy.successful_moves}, total={policy.total_moves}, t={sim.sim_time:.1f}s)", flush=True)
                    if stall_count >= args.stall_patience:
                        print(f"Phase {phase!r} stalled; exiting.", flush=True)
                        break
                else:
                    stall_count = 0
                    last_move_count = policy.successful_moves
                    print(f"[stall] reset — successful_moves={policy.successful_moves}, total={policy.total_moves}, t={sim.sim_time:.1f}s", flush=True)
                stall_check_time = sim.sim_time

    module_radius = float(sim.MODULE_RADIUS)

    try:
        # --- Phase 1: Coagulation ---
        run_phase("coag", coag)
        phase1_connected = coag.is_connected()
        coag_moved: Set[str] = {m["module"] for m in coag.move_log}
        print(
            f"Phase 1: connected={phase1_connected}  "
            f"moves={coag.total_moves}  movers={sorted(coag_moved)}")

        # --- Phase 2: Restructuring ---
        if phase1_connected:
            restruct = LineFaultRestructuring(
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

    out = args.output or media_path("line_fault_agent.mp4")
    if frames:
        iio.imwrite(out, frames, fps=12, codec="libx264")
        print(f"Wrote {len(frames)} frames to {os.path.abspath(out)}")
    else:
        print("No frames captured.")

    if not args.no_viewer and snapshots:
        launch_3d_viewer(snapshots, module_radius)


if __name__ == "__main__":
    main()
