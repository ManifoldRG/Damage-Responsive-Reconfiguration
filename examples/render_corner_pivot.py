"""Render an MP4 of a 2-body 90° corner pivot (Bullet, rigid bonds + P2P)."""
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


def start_corner(
    sim,
    pivot,
    axis,
    lattice_ref,
    target_local,
    attract_body,
    attract_conn,
):
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


def draw_frame(
    ax,
    sim,
    positions,
    target,
    pivot_idx,
    axis_idx,
    title="",
    history=None,
    attract_pt=None,
    repel_pt=None,
    goal_com=None,
):
    ax.clear()
    R = float(sim.MODULE_RADIUS)
    colors = {pivot_idx: "#ff8c00", axis_idx: "#00d9ff"}
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

    pad = 2.2
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


def main():
    N = 2
    nom = float(BulletSimulator.NOMINAL_DIST)
    pos0 = np.array([[-nom, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    bonded = np.zeros((N, N), dtype=bool)
    bonded[0, 1] = bonded[1, 0] = True

    pivot_idx, axis_idx = 0, 1
    BulletSimulator.USE_ROLLING_SPHERE_PIVOT = True
    sim = BulletSimulator(N, pos0, bonded, gui=False)

    pos = sim.get_positions()
    lattice_ref = axis_idx
    R0 = sim.body_rotation_matrix(lattice_ref)
    # Corner lattice neighbor of axis, ⟂ initial arm (−X): +Y cell for pivot COM
    target = pos[axis_idx] + np.array([0.0, nom, 0.0], dtype=float)
    target_local = R0.T @ (target - pos[lattice_ref])
    attract_body = axis_idx
    attract_conn = sim.nearest_connector(attract_body, target - pos[attract_body])

    start_corner(
        sim,
        pivot_idx,
        axis_idx,
        lattice_ref,
        target_local,
        attract_body,
        attract_conn,
    )

    frames = []
    history = []
    dt = 0.01
    max_t = 70.0
    frame_interval = 0.1

    fig, ax_fig = plt.subplots(1, 1, figsize=(5, 5), facecolor="#14141e")
    last_capture = -1.0

    try:
        while sim.sim_time < max_t:
            sim.step(dt)
            pos = sim.get_positions()
            history.append(pos[pivot_idx][:2].copy())

            if sim.sim_time - last_capture >= frame_interval:
                last_capture = sim.sim_time
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
                        repel_pt = sim.get_connector_world_pos(rep, ps.repel_connector)[
                            :2
                        ]

                draw_frame(
                    ax_fig,
                    sim,
                    pos,
                    target,
                    pivot_idx,
                    axis_idx,
                    title=f"t={sim.sim_time:.1f}s corner{angle_err_str}",
                    history=history,
                    attract_pt=attract_pt,
                    repel_pt=repel_pt,
                    goal_com=goal_com,
                )
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                frames.append(buf.reshape(h, w, 4)[:, :, :3].copy())

            if sim.has_active_pivots() and sim.is_pivot_complete(pivot_idx):
                sim.stop_pivot(pivot_idx)
                draw_frame(
                    ax_fig,
                    sim,
                    pos,
                    target,
                    pivot_idx,
                    axis_idx,
                    title=f"t={sim.sim_time:.1f}s COMPLETE",
                    history=history,
                )
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                frames.append(buf.reshape(h, w, 4)[:, :, :3].copy())
                break
    finally:
        sim.disconnect()
        plt.close(fig)

    out = os.path.join(os.path.dirname(__file__), "..", "corner_pivot.mp4")
    iio.imwrite(out, frames, fps=12, codec="libx264")
    print(f"Wrote {len(frames)} frames to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
