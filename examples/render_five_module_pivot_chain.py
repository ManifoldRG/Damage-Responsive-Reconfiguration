"""
Five-module line along +Y: sequential "pivot down the chain" demos.

Each round, the lowest-Y module (traveler) runs **one** opening corner, **three**
sequential lateral handoffs up the spine, then **one** closing corner to the end:

  1) **Corner** around the neighbor above (opens L) — axis ``round_order[1]``.
  2) **Lateral** axis ``round_order[1]`` → handoff ``round_order[2]``.
  3) **Lateral** axis ``round_order[2]`` → handoff ``round_order[3]``.
  4) **Lateral** axis ``round_order[3]`` → handoff ``round_order[4]``.
  5) **Corner** around ``round_order[4]`` (top), COM one nominal step past the
     top along the **current** spine (spine neighbor = bonded neighbor of top
     that is not the traveler). This step is **skipped** if that extension site
     is already too close to another module's COM (bent spine vs. lattice goal).

``round_order`` is body indices sorted by Y at the **start** of the round
(bottom → top). Five rounds cycle each module as traveler.

Output: ``Media/five_module_pivot_chain.mp4`` (matplotlib Agg, 12 fps).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.media_paths import media_path
from src.bullet_sim import BulletSimulator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio.v3 as iio


def colinear_target_past_axis(
    sim: BulletSimulator,
    axis_idx: int,
    neighbor_toward_chain_start: int,
    pos: np.ndarray,
    nom: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One nominal step past ``axis_idx`` along the spine (same as 3-module demo).

    With N>3, that cell may coincide with the next module up the line; the
    controller still matches the proven 3-body choreography. (A goal past the
    global top is often unreachable before pivot timeout when the pivot axis is
    still the third body from the bottom.)
    """
    u = pos[axis_idx] - pos[neighbor_toward_chain_start]
    un = float(np.linalg.norm(u))
    if un < 1e-9:
        raise ValueError("degenerate spine")
    u = u / un
    target_w = pos[axis_idx] + nom * u
    R = sim.body_rotation_matrix(axis_idx)
    target_local = R.T @ (target_w - pos[axis_idx])
    return target_w, target_local


def spine_neighbor_below_top(
    sim: BulletSimulator, top_idx: int, pivot_idx: int
) -> int:
    """Bonded neighbor of ``top_idx`` that is not the traveler (interior spine)."""
    bm = sim.get_bond_matrix()
    nbrs = [j for j in range(sim.N) if bm[top_idx, j] and j != pivot_idx]
    if not nbrs:
        raise RuntimeError(
            f"top body {top_idx} has no bond neighbor other than pivot {pivot_idx}"
        )
    if len(nbrs) == 1:
        return int(nbrs[0])
    pos = sim.get_positions()
    return int(max(nbrs, key=lambda j: float(np.linalg.norm(pos[pivot_idx] - pos[j]))))


def closing_extension_target_if_clear(
    sim: BulletSimulator,
    top_idx: int,
    pivot_idx: int,
    pos: np.ndarray,
    nom: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """COM goal one nominal step past ``top`` along top–spine, or None if invalid.

    After long lateral chains, a naïve colinear goal can land inside another
    module or match a redundant pose; skip the closing corner in those cases.
    """
    below = spine_neighbor_below_top(sim, top_idx, pivot_idx)
    target_w, target_local = colinear_target_past_axis(
        sim, top_idx, below, pos, nom
    )
    # COM separation for touching spheres: ~NOMINAL_DIST; keep margin.
    min_others = float(nom * 0.88)
    top_d = float(np.linalg.norm(pos[top_idx] - target_w))
    if abs(top_d - nom) > 0.34 * nom:
        return None
    pv_d = float(np.linalg.norm(pos[pivot_idx] - target_w))
    if pv_d < float(nom * 0.22):
        return None
    for j in range(sim.N):
        if j in (pivot_idx, top_idx):
            continue
        if float(np.linalg.norm(pos[j] - target_w)) < min_others:
            return None
    return target_w, target_local


def corner1_target_perpendicular_to_arm(
    sim: BulletSimulator,
    axis_idx: int,
    pivot_idx: int,
    pos: np.ndarray,
    nom: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First corner: axis COM + nom * perp(arm) in XY; prefer +world X when tied."""
    axis_pos = pos[axis_idx].copy()
    traveler_pos = pos[pivot_idx]
    arm = traveler_pos - axis_pos
    arm_xy = arm[:2]
    nxy = float(np.linalg.norm(arm_xy))
    if nxy < 1e-9:
        raise ValueError("degenerate arm (need side-by-side chain segment)")
    perp_xy = np.array([-arm_xy[1], arm_xy[0]], dtype=float)
    perp_xy = perp_xy / float(np.linalg.norm(perp_xy))
    if perp_xy[0] < 0:
        perp_xy = -perp_xy
    target_w = np.array(
        [
            axis_pos[0] + nom * perp_xy[0],
            axis_pos[1] + nom * perp_xy[1],
            axis_pos[2],
        ],
        dtype=float,
    )
    R = sim.body_rotation_matrix(axis_idx)
    target_local = R.T @ (target_w - pos[axis_idx])
    return target_w, target_local


def start_corner(
    sim: BulletSimulator,
    pivot: int,
    axis: int,
    lattice_ref: int,
    target_local: np.ndarray,
    attract_body: int,
    attract_conn: int,
) -> None:
    pos = sim.get_positions()
    tw = sim.target_world_from_local(lattice_ref, target_local)
    r_vec = pos[pivot] - pos[axis]
    rot_axis = sim.get_rotation_axis(pos[pivot], pos[axis], tw)
    r_target = tw - pos[axis]
    cos_a = np.clip(
        np.dot(r_vec, r_target)
        / (np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12),
        -1,
        1,
    )
    angle = float(np.arccos(cos_a))
    kp, kd = sim.compute_pd_gains(r_vec, duration=12.0)
    sim.start_pivot(
        pivot,
        axis,
        rot_axis,
        angle,
        kp,
        kd,
        duration=12.0,
        lattice_ref_body_idx=lattice_ref,
        target_pos_local=target_local,
        attract_body_idx=attract_body,
        attract_connector=attract_conn,
        pivot_type="corner",
    )


def start_lateral(
    sim: BulletSimulator,
    pivot: int,
    axis: int,
    lattice_ref: int,
    target_local: np.ndarray,
    attract_body: int,
    attract_conn: int,
    repel_idx: int | None = None,
) -> None:
    pos = sim.get_positions()
    tw = sim.target_world_from_local(lattice_ref, target_local)
    r_vec = pos[pivot] - pos[axis]
    rot_axis = sim.get_rotation_axis(pos[pivot], pos[axis], tw)
    r_target = tw - pos[axis]
    cos_a = np.clip(
        np.dot(r_vec, r_target)
        / (np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12),
        -1,
        1,
    )
    angle = float(np.arccos(cos_a))
    kp, kd = sim.compute_pd_gains(r_vec, duration=12.0)
    sim.start_pivot(
        pivot,
        axis,
        rot_axis,
        angle,
        kp,
        kd,
        duration=12.0,
        lattice_ref_body_idx=lattice_ref,
        target_pos_local=target_local,
        attract_body_idx=attract_body,
        attract_connector=attract_conn,
        pivot_type="lateral",
        repel_idx=repel_idx,
    )


def draw_frame(
    ax,
    sim: BulletSimulator,
    positions: np.ndarray,
    pivot_idx: int,
    cur_axis_idx: int,
    handoff_idx: int | None,
    title: str = "",
    history: list | None = None,
    attract_pt: np.ndarray | None = None,
    repel_pt: np.ndarray | None = None,
    goal_com: np.ndarray | None = None,
    *,
    ghost_goal_idx: int | None = None,
):
    ax.clear()
    R = float(sim.MODULE_RADIUS)
    colors = {pivot_idx: "#ff8c00", cur_axis_idx: "#00d9ff"}
    if handoff_idx is not None:
        colors[handoff_idx] = "#00d9ff"
    if ghost_goal_idx is not None:
        colors[ghost_goal_idx] = "#c084fc"
    labels_map = {i: f"M{i}" for i in range(len(positions))}

    if history and len(history) > 1:
        hx = [p[0] for p in history]
        hy = [p[1] for p in history]
        ax.plot(hx, hy, "-", color="#ff8c0066", linewidth=1.5)

    bm = sim.get_bond_matrix()
    N = len(positions)
    for i in range(N):
        for j in range(i + 1, N):
            if bm[i, j]:
                ax.plot(
                    [positions[i][0], positions[j][0]],
                    [positions[i][1], positions[j][1]],
                    "-",
                    color="#aaa",
                    linewidth=2,
                    zorder=1,
                )

    for idx in range(N):
        alpha = 0.55 if ghost_goal_idx is not None and idx == ghost_goal_idx else 0.85
        c = mpatches.Circle(
            (positions[idx][0], positions[idx][1]),
            R * 0.9,
            color=colors.get(idx, "#00d9ff"),
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(c)
        ax.text(
            positions[idx][0],
            positions[idx][1],
            labels_map.get(idx, ""),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            zorder=3,
        )
        for ci in range(6):
            cp = sim.get_connector_world_pos(idx, ci)
            ax.plot(cp[0], cp[1], ".", color="#ffffff", markersize=2, zorder=4)

    if goal_com is not None:
        ax.plot(goal_com[0], goal_com[1], "x", color="red", markersize=8, mew=2, zorder=5)
    if attract_pt is not None:
        ax.plot(
            attract_pt[0],
            attract_pt[1],
            "o",
            color="#00ff00",
            markersize=6,
            mew=0,
            zorder=5,
        )
    if repel_pt is not None:
        ax.plot(
            repel_pt[0],
            repel_pt[1],
            "o",
            color="#ff3333",
            markersize=6,
            mew=0,
            zorder=5,
        )

    pad = 3.5
    cx = float(np.mean([p[0] for p in positions]))
    cy = float(np.mean([p[1] for p in positions]))
    ax.set_xlim(cx - pad, cx + pad)
    ax.set_ylim(cy - pad, cy + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#14141e")
    ax.set_title(title, color="white", fontsize=8)
    ax.tick_params(colors="#555", labelsize=5)
    for spine in ax.spines.values():
        spine.set_color("#333")


def _append_frame(fig, frames: list) -> None:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frames.append(buf.reshape(h, w, 4)[:, :, :3].copy())


def _pivot_debug_overlay(sim: BulletSimulator, pivot_idx: int, pos: np.ndarray):
    angle_err_str = ""
    attract_pt = repel_pt = goal_com = None
    if pivot_idx in sim._active_pivots:
        ps = sim._active_pivots[pivot_idx]
        rp = pos[pivot_idx] - pos[ps.axis_idx]
        tc = sim._measure_angle(rp, ps.r0, ps.rot_axis)
        ae = abs(tc - ps.target_angle)
        if ae > np.pi:
            ae = 2 * np.pi - ae
        angle_err_str = f"  err={ae:.2f}rad"
        attract_pt = sim.get_connector_world_pos(
            ps.attract_body_idx, ps.attract_connector
        )[:2]
        goal_com = sim.resolve_pivot_target_world(ps)[:2]
        rep = ps.repel_idx if ps.repel_idx is not None else ps.axis_idx
        if rep >= 0 and ps.repel_connector is not None:
            repel_pt = sim.get_connector_world_pos(rep, ps.repel_connector)[:2]
    return angle_err_str, attract_pt, repel_pt, goal_com


def _chain_order_by_y(pos: np.ndarray) -> np.ndarray:
    return np.argsort(pos[:, 1])


def main():
    N = 5
    nom = float(BulletSimulator.NOMINAL_DIST)
    # Line along +Y: M0 bottom ... M4 top (centered on origin in y)
    half = (N - 1) / 2.0
    pos0 = np.array(
        [[0.0, (i - half) * nom, 0.0] for i in range(N)],
        dtype=float,
    )
    bonded = np.zeros((N, N), dtype=bool)
    for i in range(N - 1):
        bonded[i, i + 1] = bonded[i + 1, i] = True

    BulletSimulator.USE_ROLLING_SPHERE_PIVOT = True
    # Rolling pivots often need >300 s sim time to reach COM tolerance; 100 s forces timeouts.
    BulletSimulator.MAX_PIVOT_TIME = 450.0
    sim = BulletSimulator(N, pos0, bonded, gui=False)

    pos = sim.get_positions()
    round_order = _chain_order_by_y(pos)
    pivot_idx = int(round_order[0])
    axis_corner = int(round_order[1])  # M1 in initial line

    target_w_c1, target_local_c1 = corner1_target_perpendicular_to_arm(
        sim, axis_corner, pivot_idx, pos, nom
    )
    attract_c1 = axis_corner
    attract_conn_c1 = sim.nearest_connector(
        attract_c1, target_w_c1 - pos[attract_c1]
    )

    start_corner(
        sim,
        pivot_idx,
        axis_corner,
        axis_corner,
        target_local_c1,
        attract_c1,
        attract_conn_c1,
    )

    round_idx = 0
    phase = "corner1"
    lateral_idx = 0
    axis_cur = int(round_order[1])
    handoff_cur = int(round_order[2])
    handoff_done = False
    lattice_ref_lat = axis_cur
    target_local_lat: np.ndarray | None = None
    attract_lat = handoff_cur
    attract_conn_lat = 0

    frames: list = []
    history: list = []
    dt = 0.01
    max_t = 2400.0
    frame_interval = 0.12
    last_capture = -1.0

    fig, ax_fig = plt.subplots(1, 1, figsize=(5.5, 5.5), facecolor="#14141e")

    def begin_round_corner1() -> None:
        nonlocal pos, pivot_idx, axis_corner, phase, handoff_done
        nonlocal round_order, lateral_idx, axis_cur, handoff_cur
        nonlocal lattice_ref_lat, target_local_lat, attract_lat, attract_conn_lat
        pos = sim.get_positions()
        round_order = _chain_order_by_y(pos)
        pivot_idx = int(round_order[0])
        axis_corner = int(round_order[1])
        lateral_idx = 0
        axis_cur = int(round_order[1])
        handoff_cur = int(round_order[2])
        tw1, tl1 = corner1_target_perpendicular_to_arm(
            sim, axis_corner, pivot_idx, pos, nom
        )
        ac1 = axis_corner
        acn1 = sim.nearest_connector(ac1, tw1 - pos[ac1])
        start_corner(sim, pivot_idx, axis_corner, axis_corner, tl1, ac1, acn1)
        phase = "corner1"
        handoff_done = False
        lattice_ref_lat = axis_cur
        target_local_lat = None

    def finish_traveler_round() -> bool:
        """Advance to next round or end video. Returns True if the main loop should break."""
        nonlocal round_idx, pivot_idx, pos, history
        if round_idx < N - 1:
            round_idx += 1
            begin_round_corner1()
            return False
        draw_frame(
            ax_fig,
            sim,
            pos,
            pivot_idx,
            int(round_order[4]),
            None,
            title=(
                f"t={sim.sim_time:.1f}s ALL ROUNDS COMPLETE "
                f"(order Y: {list(_chain_order_by_y(pos))})"
            ),
            history=history,
        )
        _append_frame(fig, frames)
        return True

    try:
        while sim.sim_time < max_t:
            sim.step(dt)
            pos = sim.get_positions()
            history.append(pos[pivot_idx][:2].copy())

            if phase == "lateral" and not handoff_done:
                if pivot_idx in sim._active_pivots and sim._active_pivots[
                    pivot_idx
                ].pivot_type == "lateral":
                    d = float(np.linalg.norm(pos[pivot_idx] - pos[handoff_cur]))
                    if d < sim.handoff_contact_distance():
                        sim.create_bond(pivot_idx, handoff_cur)
                        sim.remove_bond(pivot_idx, axis_cur)
                        sim.stop_pivot(pivot_idx)
                        pos = sim.get_positions()
                        new_ax = handoff_cur
                        tw = sim.target_world_from_local(
                            lattice_ref_lat, target_local_lat
                        )
                        r_vec = pos[pivot_idx] - pos[new_ax]
                        rot_axis = sim.get_rotation_axis(
                            pos[pivot_idx], pos[new_ax], tw
                        )
                        r_target = tw - pos[new_ax]
                        cos_a = np.clip(
                            np.dot(r_vec, r_target)
                            / (
                                np.linalg.norm(r_vec)
                                * np.linalg.norm(r_target)
                                + 1e-12
                            ),
                            -1,
                            1,
                        )
                        angle = float(np.arccos(cos_a))
                        kp, kd = sim.compute_pd_gains(r_vec, duration=12.0)
                        sim.start_pivot(
                            pivot_idx,
                            new_ax,
                            rot_axis,
                            angle,
                            kp,
                            kd,
                            duration=12.0,
                            lattice_ref_body_idx=lattice_ref_lat,
                            target_pos_local=target_local_lat,
                            attract_body_idx=attract_lat,
                            attract_connector=attract_conn_lat,
                            pivot_type="lateral",
                            repel_idx=-1,
                        )
                        handoff_done = True

            if sim.sim_time - last_capture >= frame_interval:
                last_capture = sim.sim_time
                angle_err_str, attract_pt, repel_pt, goal_com = _pivot_debug_overlay(
                    sim, pivot_idx, pos
                )
                if phase == "corner1":
                    cur_ax = axis_corner
                    sub = f"[r{round_idx} corner1]"
                    title = (
                        f"t={sim.sim_time:.1f}s {sub} "
                        f"tr=M{pivot_idx} ax=M{axis_corner}{angle_err_str}"
                    )
                elif phase == "lateral":
                    status = "leg1" if not handoff_done else "leg2"
                    cur_ax = axis_cur if not handoff_done else handoff_cur
                    sub = f"[r{round_idx} lat{lateral_idx} {status}]"
                    title = (
                        f"t={sim.sim_time:.1f}s {sub} "
                        f"tr=M{pivot_idx} ax=M{axis_cur} ho=M{handoff_cur}"
                        f"{angle_err_str}"
                    )
                else:
                    cur_ax = int(round_order[4])
                    sub = f"[r{round_idx} corner_end]"
                    title = (
                        f"t={sim.sim_time:.1f}s {sub} "
                        f"tr=M{pivot_idx} top=M{int(round_order[4])}"
                        f"{angle_err_str}"
                    )
                draw_frame(
                    ax_fig,
                    sim,
                    pos,
                    pivot_idx,
                    cur_ax,
                    handoff_cur if phase == "lateral" else None,
                    title=title,
                    history=history,
                    attract_pt=attract_pt,
                    repel_pt=repel_pt,
                    goal_com=goal_com,
                )
                _append_frame(fig, frames)

            pivot_done = (
                sim.has_active_pivots()
                and sim.is_pivot_complete(pivot_idx)
                and not (phase == "lateral" and not handoff_done)
            )
            if pivot_done:
                sim.stop_pivot(pivot_idx)
                pos = sim.get_positions()

                if phase == "corner1":
                    lateral_idx = 0
                    axis_cur = int(round_order[1])
                    handoff_cur = int(round_order[2])
                    lattice_ref_lat = axis_cur
                    R0 = sim.body_rotation_matrix(lattice_ref_lat)
                    target_w = sim.lateral_pivot_target_world(
                        axis_cur, pivot_idx, handoff_cur
                    )
                    target_local_lat = R0.T @ (target_w - pos[lattice_ref_lat])
                    attract_lat = handoff_cur
                    attract_conn_lat = sim.lateral_handoff_attract_connector(
                        axis_cur, pivot_idx, handoff_cur
                    )
                    start_lateral(
                        sim,
                        pivot_idx,
                        axis_cur,
                        lattice_ref_lat,
                        target_local_lat,
                        attract_lat,
                        attract_conn_lat,
                    )
                    handoff_done = False
                    phase = "lateral"
                elif phase == "lateral":
                    d_ph = float(np.linalg.norm(pos[pivot_idx] - pos[handoff_cur]))
                    if d_ph < 1.12:
                        sim.create_bond(pivot_idx, handoff_cur)
                    if lateral_idx < N - 3:
                        lateral_idx += 1
                        axis_cur = int(round_order[lateral_idx + 1])
                        handoff_cur = int(round_order[lateral_idx + 2])
                        lattice_ref_lat = axis_cur
                        R0 = sim.body_rotation_matrix(lattice_ref_lat)
                        target_w = sim.lateral_pivot_target_world(
                            axis_cur, pivot_idx, handoff_cur
                        )
                        target_local_lat = R0.T @ (target_w - pos[lattice_ref_lat])
                        attract_lat = handoff_cur
                        attract_conn_lat = sim.lateral_handoff_attract_connector(
                            axis_cur, pivot_idx, handoff_cur
                        )
                        start_lateral(
                            sim,
                            pivot_idx,
                            axis_cur,
                            lattice_ref_lat,
                            target_local_lat,
                            attract_lat,
                            attract_conn_lat,
                        )
                        handoff_done = False
                    else:
                        top = int(round_order[4])
                        goal = closing_extension_target_if_clear(
                            sim, top, pivot_idx, pos, nom
                        )
                        if goal is not None:
                            target_w_c2, target_local_c2 = goal
                            attract_c2 = top
                            attract_conn_c2 = sim.nearest_connector(
                                attract_c2, target_w_c2 - pos[attract_c2]
                            )
                            start_corner(
                                sim,
                                pivot_idx,
                                top,
                                top,
                                target_local_c2,
                                attract_c2,
                                attract_conn_c2,
                            )
                            phase = "corner_end"
                        else:
                            if finish_traveler_round():
                                break
                elif phase == "corner_end":
                    if finish_traveler_round():
                        break

    finally:
        sim.disconnect()
        plt.close(fig)

    pos_final = pos
    order_f = _chain_order_by_y(pos_final)
    print("Final Y-order (bottom -> top body indices):", list(order_f))
    print("Expected [0,1,2,3,4] for restored line order.")

    out = media_path("five_module_pivot_chain.mp4")
    iio.imwrite(out, frames, fps=12, codec="libx264")
    print(f"Wrote {len(frames)} frames to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
