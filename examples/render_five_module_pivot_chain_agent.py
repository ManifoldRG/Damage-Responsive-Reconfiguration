"""
Five-module line along +Y driven by ``DecentralizedCoagulation`` (agent policy).

**M5 ghost goal (collision-off)**

A sixth PyBullet body **M5** sits one ``NOMINAL_DIST`` past the current chain
tip, with **no collision** against other modules and **low mass**. Tokens use
the standard fault pipeline: ``fault_id="M5"``, ``_fault_body_idx`` = M5's
index, and ``set_fault_adjacent`` lists the **single** spine module that should
emit toward M5 for the current turn (**M4**, then **M0**–**M3**, then **M4**
again per ``_TIP_BODY_PER_TURN``). All other modules only **forward** tokens.

When the active mover's turn ends (no strictly improving ``pick_target``), the
turn counter advances, the **M4–M5** (or current) bond is removed, **M5** is
teleported one step past the new tip, rebonded, and ``set_fault_adjacent`` is
updated so the correct neighbor emits.

**Sequential movers**

Only the spine module index ``_TIP_BODY_PER_TURN[turn]`` may pivot (``is_movable``). While it is their turn, they keep
pivoting while some eligible pivot strictly moves their COM closer to **M5**
(see ``require_closer_to_origin``). When none do, they forward the token, the
turn advances, and M5 relocates as above.

Output: ``Media/five_module_pivot_chain_agent.mp4`` (matplotlib Agg).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Set

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import imageio.v3 as iio
import matplotlib.pyplot as plt

from examples.media_paths import media_path
from examples.render_five_module_pivot_chain import (
    _append_frame,
    _pivot_debug_overlay,
    draw_frame,
)
from src.agent_policy import DecentralizedCoagulation, ModuleAgent, ModuleState, Token
from src.bullet_sim import BulletSimulator


class SequentialAssemblyFrameCoagulation(DecentralizedCoagulation):
    """
    M5 as real goal body; fault-adjacent emitter rotates with chain tip per turn.
    """

    # Where M5 is bonded per turn (goal placement); last turn places M5 past M3.
    _TIP_BODY_PER_TURN = (4, 0, 1, 2, 3, 3)
    # self.turn is used directly as the module number for movability (0→M0, ..., 4→M4).
    # On the last turn (5), M4 pivots toward M5 past M3, so we map turn 5 → M4.
    _MOVER_OVERRIDE = {5: 4}

    def __init__(
        self,
        sim: BulletSimulator,
        fault_id: str,
        module_ids: list[str],
        body_indices: dict[str, int],
        *,
        goal_body_idx: int = 5,
    ):
        super().__init__(sim, fault_id, module_ids, body_indices)
        self.turn = 0
        self.goal_body_idx = int(goal_body_idx)
        self.set_fault_adjacent(
            [f"M{self._TIP_BODY_PER_TURN[0]}"], self.goal_body_idx)

    def _pick_target_occupancy_skip_body_indices(self) -> Set[int]:
        """M5 occupies the nominal empty slot; allow pivot targets into that cell."""
        return {self.goal_body_idx}

    def _ghost_goal_world_pos_for_axial_corner(
            self, pos: np.ndarray) -> Optional[np.ndarray]:
        """Allow colinear lattice step into M5's cell (parallel pivot-axis arm)."""
        return pos[self.goal_body_idx].copy()

    def _active_body_indices(self) -> list[int]:
        return sorted(self._idx_to_mid.keys())

    def _spine_outward_unit(self, pos: np.ndarray, tip_idx: int) -> np.ndarray:
        bm = self.sim.get_bond_matrix()
        g = self.goal_body_idx
        nbrs = [j for j in range(self.sim.N) if bm[tip_idx, j] and j != g]
        if not nbrs:
            return np.array([0.0, 1.0, 0.0], dtype=float)
        if len(nbrs) == 1:
            j = nbrs[0]
            u = pos[tip_idx] - pos[j]
            n = float(np.linalg.norm(u))
            return u / n if n > 1e-12 else np.array([0.0, 1.0, 0.0], dtype=float)
        idxs = self._active_body_indices()
        com = pos[idxs].mean(axis=0)
        interior = min(
            nbrs,
            key=lambda jj: float(np.linalg.norm(pos[jj] - com)),
        )
        u = pos[tip_idx] - pos[interior]
        n = float(np.linalg.norm(u))
        return u / n if n > 1e-12 else np.array([0.0, 1.0, 0.0], dtype=float)

    def _relocate_goal_module(self) -> None:
        """Rebond and teleport M5 for the current ``self.turn`` (after increment)."""
        turn = self.turn
        if turn <= 0 or turn >= len(self._TIP_BODY_PER_TURN):
            return
        prev_tip = int(self._TIP_BODY_PER_TURN[turn - 1])
        new_tip = int(self._TIP_BODY_PER_TURN[turn])
        g = self.goal_body_idx
        if self.sim.is_bonded(prev_tip, g):
            self.sim.remove_bond(prev_tip, g)
        pos = self.sim.get_positions()
        nom = float(self.sim.NOMINAL_DIST)
        u_hat = self._spine_outward_unit(pos, new_tip)
        new_pos = pos[new_tip] + nom * u_hat
        self.sim.reset_module_world_pose(g, new_pos)
        self.sim.create_bond(new_tip, g)
        self.set_fault_adjacent([f"M{new_tip}"], g)

    def pick_target(self, agent: ModuleAgent, *, require_closer_to_origin: bool = True):
        return super().pick_target(
            agent, require_closer_to_origin=require_closer_to_origin)


    def _on_no_pick_target(self, agent: ModuleAgent) -> None:
        mid = self._idx_to_mid.get(agent.body_idx)
        mover = self._MOVER_OVERRIDE.get(self.turn, self.turn)
        if mid != f"M{mover}":
            return
        if self.turn >= len(self._TIP_BODY_PER_TURN):
            return
        self.turn += 1
        if self.turn < len(self._TIP_BODY_PER_TURN):
            self._relocate_goal_module()

    def is_movable(self, body_idx: int, safety_radius: int = 2) -> bool:
        if self.turn >= len(self._TIP_BODY_PER_TURN):
            return False
        mover = self._MOVER_OVERRIDE.get(self.turn, self.turn)
        mid = self._idx_to_mid.get(body_idx)
        if mid != f"M{mover}":
            return False
        return super().is_movable(body_idx, safety_radius)

    def _forward_token(self, agent: ModuleAgent):
        if agent.token is None:
            return
        pos = self.sim.get_positions()
        neighbors = self.get_physical_neighbors(agent.body_idx)
        src = agent.token.source_id
        for n_idx in neighbors:
            n_mid = self._idx_to_mid.get(n_idx)
            if n_mid is None or n_mid not in self.agents:
                continue
            n_agent = self.agents[n_mid]
            if n_agent.state == ModuleState.PIVOTING:
                continue
            my_pos = pos[agent.body_idx]
            nbr_pos = pos[n_idx]
            R_me = self.sim.body_rotation_matrix(agent.body_idx)
            R_nbr = self.sim.body_rotation_matrix(n_idx)
            world_dir = R_me @ agent.token.direction + (my_pos - nbr_pos)
            propagated_dir = R_nbr.T @ world_dir
            n_agent.incoming_tokens.append(
                Token(direction=propagated_dir, source_id=src))


def _frame_pivot_indices(
    policy: SequentialAssemblyFrameCoagulation, pos: np.ndarray
):
    for agent in policy.agents.values():
        if agent.state == ModuleState.PIVOTING:
            ax = agent.pivot_axis_idx if agent.pivot_axis_idx is not None else 0
            ho = (
                agent.handoff_idx
                if agent.pivot_type == "lateral"
                else None
            )
            return agent.body_idx, ax, ho
    if policy.turn < len(policy._TIP_BODY_PER_TURN):
        mover = policy._MOVER_OVERRIDE.get(policy.turn, policy.turn)
        k = policy.agents[f"M{mover}"].body_idx
        return k, k, None
    return 0, 0, None


def _demo_complete(policy: SequentialAssemblyFrameCoagulation, sim: BulletSimulator) -> bool:
    if policy.turn < len(policy._TIP_BODY_PER_TURN):
        return False
    return not sim.has_active_pivots()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Five-module line: sequential movers M0..M4 with M5 ghost goal "
            "and fault-adjacent token emission."
        ),
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=2400.0,
        help="Wall sim time limit (seconds)",
    )
    parser.add_argument(
        "--stall-interval",
        type=float,
        default=20.0,
        help="Seconds between stall checks (no new moves)",
    )
    parser.add_argument(
        "--stall-patience",
        type=int,
        default=8,
        help="Consecutive stall windows before exit",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="MP4 path (default: Media/five_module_pivot_chain_agent.mp4)",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=0.12,
        help="Seconds between video frames",
    )
    args = parser.parse_args()

    n_mod = 5
    goal_idx = 5
    nom = float(BulletSimulator.NOMINAL_DIST)
    half = (n_mod - 1) / 2.0
    pos0 = np.zeros((goal_idx + 1, 3), dtype=float)
    for i in range(n_mod):
        pos0[i] = (0.0, (i - half) * nom, 0.0)
    pos0[goal_idx] = pos0[n_mod - 1].copy()
    pos0[goal_idx][1] += nom

    bonded = np.zeros((goal_idx + 1, goal_idx + 1), dtype=bool)
    for i in range(n_mod - 1):
        bonded[i, i + 1] = bonded[i + 1, i] = True
    bonded[n_mod - 1, goal_idx] = bonded[goal_idx, n_mod - 1] = True

    BulletSimulator.USE_ROLLING_SPHERE_PIVOT = True
    BulletSimulator.MAX_PIVOT_TIME = 100.0
    n_sim = goal_idx + 1
    sim = BulletSimulator(n_sim, pos0, bonded, gui=False)
    sim.set_body_collision_with_all(goal_idx, False)
    sim.set_body_mass(goal_idx, 1e-3)

    module_ids = [f"M{i}" for i in range(n_mod)]
    body_indices = {f"M{i}": i for i in range(n_mod)}
    policy = SequentialAssemblyFrameCoagulation(
        sim,
        fault_id="M5",
        module_ids=module_ids,
        body_indices=body_indices,
        goal_body_idx=goal_idx,
    )

    dt = float(BulletSimulator.PHYSICS_DT)
    frames: list = []
    history: list = []
    last_capture = -1.0
    stall_check_time = sim.sim_time
    last_move_count = 0
    stall_count = 0

    fig, ax_fig = plt.subplots(1, 1, figsize=(5.5, 5.5), facecolor="#14141e")

    spine_indices = list(range(n_mod))
    init_pos = sim.get_positions()
    init_vel = sim.get_velocities()
    init_com = np.mean(init_pos[spine_indices], axis=0)
    init_com_vel = np.mean(init_vel[spine_indices], axis=0)

    try:
        while sim.sim_time < args.max_time:
            sim.step(dt)
            pos = sim.get_positions()
            policy.tick()

            if _demo_complete(policy, sim):
                break

            p_idx, ax_idx, ho_idx = _frame_pivot_indices(policy, pos)
            history.append(pos[p_idx][:2].copy())

            if sim.sim_time - last_capture >= args.frame_interval:
                last_capture = sim.sim_time
                angle_err_str, attract_pt, repel_pt, goal_com = _pivot_debug_overlay(
                    sim, p_idx, pos)
                tip_i = (
                    policy._TIP_BODY_PER_TURN[policy.turn]
                    if policy.turn < len(policy._TIP_BODY_PER_TURN)
                    else -1
                )
                title = (
                    f"t={sim.sim_time:.1f}s  moves={policy.total_moves}  "
                    f"step={policy.turn}  mover=M{tip_i}  "
                    f"highlight=M{p_idx}{angle_err_str}"
                )
                draw_frame(
                    ax_fig,
                    sim,
                    pos,
                    p_idx,
                    ax_idx,
                    ho_idx,
                    title=title,
                    history=history,
                    attract_pt=attract_pt,
                    repel_pt=repel_pt,
                    goal_com=goal_com,
                    ghost_goal_idx=policy.goal_body_idx,
                )
                _append_frame(fig, frames)

            if sim.sim_time - stall_check_time > args.stall_interval:
                if policy.total_moves == last_move_count:
                    stall_count += 1
                    if stall_count >= args.stall_patience:
                        break
                else:
                    stall_count = 0
                    last_move_count = policy.total_moves
                stall_check_time = sim.sim_time

    finally:
        final_pos = sim.get_positions()
        final_vel = sim.get_velocities()
        final_com = np.mean(final_pos[spine_indices], axis=0)
        final_com_vel = np.mean(final_vel[spine_indices], axis=0)
        delta_com = final_com - init_com
        delta_com_vel = final_com_vel - init_com_vel
        print("\n=== Assembly CoM report (M0-M4) ===")
        print(f"  Initial CoM pos:  {init_com}")
        print(f"  Final   CoM pos:  {final_com}")
        print(f"  Delta   CoM pos:  {delta_com}  (norm={np.linalg.norm(delta_com):.6f} m)")
        print(f"  Initial CoM vel:  {init_com_vel}")
        print(f"  Final   CoM vel:  {final_com_vel}")
        print(f"  Delta   CoM vel:  {delta_com_vel}  (norm={np.linalg.norm(delta_com_vel):.6f} m/s)")

        sim.disconnect()
        plt.close(fig)

    out = args.output or media_path("five_module_pivot_chain_agent.mp4")
    if frames:
        iio.imwrite(out, frames, fps=12, codec="libx264")
        print(f"Wrote {len(frames)} frames to {os.path.abspath(out)}")
    else:
        print("No frames captured.")
    print(f"Total policy moves: {policy.total_moves}  final_turn={policy.turn}")
    if policy.move_log:
        print("Last move:", policy.move_log[-1])


if __name__ == "__main__":
    main()
