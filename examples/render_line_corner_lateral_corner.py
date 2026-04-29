"""
Three-phase scripted demo: 3-module line along +Y, one moving module (M0).

1) Corner pivot: M0 around M1 toward +X (opens an L, same geometry as the
   lateral pivot MP4 expects).
2) Lateral pivot: M0 transfers from M1 to M2 (same start_pivot + handoff logic
   as render_lateral_pivot.py).
3) Corner pivot: M0 around M2 so the final COM sits **one nominal step past M2**
   along the **M1→M2** line — i.e. M1, M2, M0 end **colinear** (straight chain).

Uses the same BulletSimulator helpers and capture cadence as render_corner_pivot.py
and render_lateral_pivot.py (matplotlib Agg, 5\"\" fig, 12 fps libx264).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
    """COM goal one nominal spacing past ``axis_idx`` along M_start … M_axis.

    ``neighbor_toward_chain_start`` is the neighbor of ``axis_idx`` on the side
    opposite the free end (here M1 when axis is M2), so
    ``pos[axis] + nom * normalize(pos[axis]-pos[neighbor])`` extends the chain.
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
):
    ax.clear()
    R = float(sim.MODULE_RADIUS)
    colors = {pivot_idx: "#ff8c00", cur_axis_idx: "#00d9ff"}
    if handoff_idx is not None:
        colors[handoff_idx] = "#00d9ff"
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
        c = mpatches.Circle(
            (positions[idx][0], positions[idx][1]),
            R * 0.9,
            color=colors.get(idx, "#00d9ff"),
            alpha=0.85,
            zorder=2,
        )
        ax.add_patch(c)
        ax.text(
            positions[idx][0],
            positions[idx][1],
            labels_map.get(idx, ""),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            zorder=3,
        )
        for ci in range(6):
            cp = sim.get_connector_world_pos(idx, ci)
            ax.plot(cp[0], cp[1], ".", color="#ffffff", markersize=3, zorder=4)

    if goal_com is not None:
        ax.plot(goal_com[0], goal_com[1], "x", color="red", markersize=10, mew=2, zorder=5)
    if attract_pt is not None:
        ax.plot(
            attract_pt[0],
            attract_pt[1],
            "o",
            color="#00ff00",
            markersize=7,
            mew=0,
            zorder=5,
        )
    if repel_pt is not None:
        ax.plot(
            repel_pt[0],
            repel_pt[1],
            "o",
            color="#ff3333",
            markersize=7,
            mew=0,
            zorder=5,
        )

    pad = 2.8
    cx = float(np.mean([p[0] for p in positions]))
    cy = float(np.mean([p[1] for p in positions]))
    ax.set_xlim(cx - pad, cx + pad)
    ax.set_ylim(cy - pad, cy + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#14141e")
    ax.set_title(title, color="white", fontsize=9)
    ax.tick_params(colors="#555", labelsize=6)
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


def main():
    N = 3
    nom = float(BulletSimulator.NOMINAL_DIST)
    # Line along +Y: M0 bottom, M1 mid, M2 top
    pos0 = np.array(
        [
            [0.0, -nom, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, nom, 0.0],
        ],
        dtype=float,
    )
    bonded = np.zeros((N, N), dtype=bool)
    bonded[0, 1] = bonded[1, 0] = True
    bonded[1, 2] = bonded[2, 1] = True

    pivot_idx, axis_m1, handoff_m2 = 0, 1, 2

    BulletSimulator.USE_ROLLING_SPHERE_PIVOT = True
    sim = BulletSimulator(N, pos0, bonded, gui=False)

    pos = sim.get_positions()
    # Phase 1: corner — arm M1->M0 is -Y; perpendicular lattice step +X
    lattice_ref_c1 = axis_m1
    R1 = sim.body_rotation_matrix(lattice_ref_c1)
    target_w_c1 = pos[axis_m1] + np.array([nom, 0.0, 0.0], dtype=float)
    target_local_c1 = R1.T @ (target_w_c1 - pos[lattice_ref_c1])
    attract_c1 = axis_m1
    attract_conn_c1 = sim.nearest_connector(
        attract_c1, target_w_c1 - pos[attract_c1]
    )

    start_corner(
        sim,
        pivot_idx,
        axis_m1,
        lattice_ref_c1,
        target_local_c1,
        attract_c1,
        attract_conn_c1,
    )

    phase = "corner1"
    handoff_done = False
    lattice_ref_lat = axis_m1
    target_local_lat: np.ndarray | None = None
    attract_lat = handoff_m2
    attract_conn_lat = 0

    frames: list = []
    history: list = []
    dt = 0.01
    max_t = 200.0
    frame_interval = 0.1
    last_capture = -1.0

    fig, ax_fig = plt.subplots(1, 1, figsize=(5, 5), facecolor="#14141e")

    try:
        while sim.sim_time < max_t:
            sim.step(dt)
            pos = sim.get_positions()
            history.append(pos[pivot_idx][:2].copy())

            if phase == "lateral" and not handoff_done:
                if pivot_idx in sim._active_pivots and sim._active_pivots[
                    pivot_idx
                ].pivot_type == "lateral":
                    d = float(np.linalg.norm(pos[pivot_idx] - pos[handoff_m2]))
                    if d < sim.handoff_contact_distance():
                        sim.create_bond(pivot_idx, handoff_m2)
                        sim.remove_bond(pivot_idx, axis_m1)
                        sim.stop_pivot(pivot_idx)
                        pos = sim.get_positions()
                        new_ax = handoff_m2
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
                    cur_ax = axis_m1
                    title = f"t={sim.sim_time:.1f}s [corner1]{angle_err_str}"
                elif phase == "lateral":
                    status = "leg1" if not handoff_done else "leg2"
                    cur_ax = axis_m1 if not handoff_done else handoff_m2
                    title = f"t={sim.sim_time:.1f}s [lateral {status}]{angle_err_str}"
                else:
                    cur_ax = handoff_m2
                    title = f"t={sim.sim_time:.1f}s [corner2]{angle_err_str}"

                draw_frame(
                    ax_fig,
                    sim,
                    pos,
                    pivot_idx,
                    cur_ax,
                    handoff_m2 if phase == "lateral" else None,
                    title=title,
                    history=history,
                    attract_pt=attract_pt,
                    repel_pt=repel_pt,
                    goal_com=goal_com,
                )
                _append_frame(fig, frames)

            # Transitions on pivot completion (lateral: ignore spurious complete
            # before bond handoff — second leg must run with handoff_done True).
            pivot_done = (
                sim.has_active_pivots()
                and sim.is_pivot_complete(pivot_idx)
                and not (phase == "lateral" and not handoff_done)
            )
            if pivot_done:
                sim.stop_pivot(pivot_idx)
                pos = sim.get_positions()

                if phase == "corner1":
                    # Same L-shaped layout as render_lateral_pivot.py — begin lateral
                    lattice_ref_lat = axis_m1
                    R0 = sim.body_rotation_matrix(lattice_ref_lat)
                    target_w = sim.lateral_pivot_target_world(
                        axis_m1, pivot_idx, handoff_m2
                    )
                    target_local_lat = R0.T @ (
                        target_w - pos[lattice_ref_lat]
                    )
                    attract_lat = handoff_m2
                    attract_conn_lat = sim.lateral_handoff_attract_connector(
                        axis_m1, pivot_idx, handoff_m2
                    )
                    start_lateral(
                        sim,
                        pivot_idx,
                        axis_m1,
                        lattice_ref_lat,
                        target_local_lat,
                        attract_lat,
                        attract_conn_lat,
                    )
                    handoff_done = False
                    phase = "lateral"
                elif phase == "lateral":
                    d_02 = float(np.linalg.norm(pos[pivot_idx] - pos[handoff_m2]))
                    if d_02 < 1.12:
                        sim.create_bond(pivot_idx, handoff_m2)
                    # Final corner: M0 around M2 — goal COM past M2 colinear with M1–M2.
                    lattice_ref_c2 = handoff_m2
                    target_w_c2, target_local_c2 = colinear_target_past_axis(
                        sim,
                        handoff_m2,
                        axis_m1,
                        pos,
                        nom,
                    )
                    attract_c2 = handoff_m2
                    attract_conn_c2 = sim.nearest_connector(
                        attract_c2, target_w_c2 - pos[attract_c2]
                    )
                    start_corner(
                        sim,
                        pivot_idx,
                        handoff_m2,
                        lattice_ref_c2,
                        target_local_c2,
                        attract_c2,
                        attract_conn_c2,
                    )
                    phase = "corner2"
                else:
                    draw_frame(
                        ax_fig,
                        sim,
                        pos,
                        pivot_idx,
                        handoff_m2,
                        None,
                        title=f"t={sim.sim_time:.1f}s COMPLETE",
                        history=history,
                    )
                    _append_frame(fig, frames)
                    break

    finally:
        sim.disconnect()
        plt.close(fig)

    out = os.path.join(
        os.path.dirname(__file__), "..", "line_corner_lateral_corner.mp4"
    )
    iio.imwrite(out, frames, fps=12, codec="libx264")
    print(f"Wrote {len(frames)} frames to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
