"""
Render MP4 of an 11-module line with a center fault + damage response.

Uses the PyBullet stack (BulletSimulator + AsyncSimRunner).
Line runs along Y axis. Camera orbits so pivot motions are clearly visible.

Video export uses Plotly + Kaleido (headless Chromium): each frame is slow.
Defaults favor speed (800x400, scale=1, 60 frames). Use --hq for the old
120-frame / 2x-scale look, or tune --frames / --width / --height / --scale.
"""

import argparse
import json
import sys
import os
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import plotly.graph_objects as go
import imageio.v3 as iio

from line_fault_scenario import build_y_line_fault_scenario
from src.bullet_sim import BulletSimulator
from src.bullet_bridge import AsyncSimRunner, print_diagnostic_summary

# ── Colors ────────────────────────────────────────────────────────────

ACTIVE_COLOR = "rgb(0, 217, 255)"
PIVOT_COLOR = "rgb(255, 140, 0)"
FAULT_COLOR = "rgb(200, 0, 0)"
GHOST_COLOR = "rgba(100, 100, 100, 0.15)"
BOND_COLOR = "rgb(200, 200, 200)"
PIVOT_BOND_COLOR = "rgb(255, 180, 80)"
BG_COLOR = "rgb(20, 20, 30)"
TRAIL_COLOR = "rgba(255, 140, 0, 0.5)"


def make_sphere_mesh(cx, cy, cz, r=0.45, color=ACTIVE_COLOR, n=14):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        showscale=False, opacity=0.92,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.4, roughness=0.3),
        hoverinfo="skip",
    )


def make_bond_line(p1, p2, color=BOND_COLOR):
    p1, p2 = np.asarray(p1), np.asarray(p2)
    return go.Scatter3d(
        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
        mode="lines", line=dict(color=color, width=8),
        hoverinfo="skip", showlegend=False,
    )


def make_trail(positions, color=TRAIL_COLOR):
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines", line=dict(color=color, width=3),
        hoverinfo="skip", showlegend=False,
    )


def make_label(pos, text, offset_z=0.65):
    return go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2] + offset_z],
        mode="text", text=[text],
        textfont=dict(size=9, color="white"),
        hoverinfo="skip", showlegend=False,
    )


def compute_bonds(positions, max_dist=1.05):
    mids = list(positions.keys())
    bonds = []
    for i in range(len(mids)):
        for j in range(i + 1, len(mids)):
            d = np.linalg.norm(positions[mids[i]] - positions[mids[j]])
            if d < max_dist:
                bonds.append((mids[i], mids[j]))
    return bonds


def render_frame(positions, labels=None, ghosts=None, ghost_bonds=None,
                 trails=None, pivot_ids=None, fault_id=None,
                 active_bonds=None,
                 title="", bounds_y=7.0, bounds_xz=4.0,
                 camera_eye=None, fig_width=800, fig_height=400,
                 sphere_n=10):
    fig = go.Figure()
    pivot_set = set(pivot_ids or [])

    if ghosts:
        for mid, gpos in ghosts.items():
            fig.add_trace(make_sphere_mesh(
                gpos[0], gpos[1], gpos[2], r=0.38, color=GHOST_COLOR, n=8))
    if ghost_bonds:
        for (a, b) in ghost_bonds:
            if a in ghosts and b in ghosts:
                fig.add_trace(make_bond_line(
                    ghosts[a], ghosts[b], color="rgba(80,80,80,0.15)"))

    if trails:
        for mid, traj in trails.items():
            if len(traj) > 2:
                fig.add_trace(make_trail(traj))

    bonds = active_bonds if active_bonds is not None else compute_bonds(positions)
    for (a, b) in bonds:
        if a in pivot_set or b in pivot_set:
            bond_color = PIVOT_BOND_COLOR
        else:
            bond_color = BOND_COLOR
        fig.add_trace(make_bond_line(positions[a], positions[b], color=bond_color))

    for mid, pos in positions.items():
        if mid == fault_id:
            color = FAULT_COLOR
        elif mid in pivot_set:
            color = PIVOT_COLOR
        else:
            color = ACTIVE_COLOR
        fig.add_trace(make_sphere_mesh(
            pos[0], pos[1], pos[2], color=color, n=sphere_n))

    if labels:
        for mid, pos in positions.items():
            if mid in labels:
                fig.add_trace(make_label(pos, labels[mid]))

    center_y = 5.0
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-bounds_xz, bounds_xz], showgrid=False,
                       zeroline=False, showticklabels=False, title="",
                       showbackground=False),
            yaxis=dict(range=[center_y - bounds_y, center_y + bounds_y],
                       showgrid=False, zeroline=False, showticklabels=False,
                       title="", showbackground=False),
            zaxis=dict(range=[-bounds_xz, bounds_xz], showgrid=False,
                       zeroline=False, showticklabels=False, title="",
                       showbackground=False),
            bgcolor=BG_COLOR,
            aspectmode="cube",
            camera=dict(
                eye=camera_eye or dict(x=1.8, y=-1.4, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor=BG_COLOR,
        margin=dict(l=0, r=0, t=45, b=0),
        title=dict(text=title, font=dict(color="white", size=13), x=0.5),
        width=fig_width, height=fig_height,
    )
    return fig


def fig_to_image(fig, scale: int = 1):
    img_bytes = fig.to_image(format="png", scale=scale)
    return iio.imread(img_bytes)


def subsample_trajectories(trajectories, n_frames):
    max_len = max(len(t) for t in trajectories.values())
    if max_len <= n_frames:
        indices = list(range(max_len))
    else:
        indices = np.linspace(0, max_len - 1, n_frames, dtype=int).tolist()
    result = {}
    for mid, traj in trajectories.items():
        result[mid] = [traj[min(i, len(traj) - 1)] for i in indices]
    return result, indices


def write_mp4(frames, output_path, fps=15):
    import imageio
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        writer.append_data(frame)
    writer.close()


# ── Simulation save/load ─────────────────────────────────────────────

SIM_CACHE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "line_fault_pybullet_sim_cache.json"))
OUTPUT_MP4_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "line_fault_pybullet.mp4"))


def save_sim_state(trajectories, ghost_pos, ghost_bonds, pivot_mids,
                   fault_id, diagnostic_report, algo_summary,
                   bond_snapshots=None):
    """Save simulation results so we can re-render without re-running PyBullet."""
    data = {
        "trajectories": {mid: [pos.tolist() for pos in traj]
                         for mid, traj in trajectories.items()},
        "ghost_pos": {mid: pos.tolist() for mid, pos in ghost_pos.items()},
        "ghost_bonds": ghost_bonds,
        "pivot_mids": sorted(pivot_mids),
        "fault_id": fault_id,
        "diagnostic_report": diagnostic_report,
        "algo_summary": algo_summary,
        "bond_snapshots": bond_snapshots or [],
    }
    with open(SIM_CACHE_PATH, "w") as f:
        json.dump(data, f)
    print(f"Simulation state saved: {SIM_CACHE_PATH}")


def load_sim_state():
    """Load previously saved simulation results."""
    with open(SIM_CACHE_PATH, "r") as f:
        data = json.load(f)
    trajectories = {mid: [np.array(pos) for pos in traj]
                    for mid, traj in data["trajectories"].items()}
    ghost_pos = {mid: np.array(pos) for mid, pos in data["ghost_pos"].items()}
    ghost_bonds = [tuple(bond) for bond in data["ghost_bonds"]]
    pivot_mids = set(data["pivot_mids"])
    fault_id = data["fault_id"]
    diagnostic_report = data.get("diagnostic_report")
    algo_summary = data.get("algo_summary")
    bond_snapshots = [
        [tuple(b) for b in frame_bonds]
        for frame_bonds in data.get("bond_snapshots", [])
    ]
    return (trajectories, ghost_pos, ghost_bonds, pivot_mids, fault_id,
            diagnostic_report, algo_summary, bond_snapshots)


# ── Main ──────────────────────────────────────────────────────────────


def run_simulation(
    n: int = 11,
    fault_body_idx: int | None = None,
    *,
    pivot_attract_scale: float = 0.1,
):
    """Run PyBullet damage response and return all data needed for rendering."""
    scenario = build_y_line_fault_scenario(n, fault_body_idx)

    print(f"Building {scenario.n}-module line (fault={scenario.fault_id})...")
    print("Running damage response (algorithm + PyBullet physics)...")
    BulletSimulator.USE_ROLLING_SPHERE_PIVOT = True
    sim = BulletSimulator(
        scenario.n,
        scenario.pos0,
        scenario.bonded0,
        gui=False,
        pivot_attract_scale=float(pivot_attract_scale),
    )
    try:
        bridge = AsyncSimRunner(
            sim,
            scenario.module_ids,
            scenario.body_indices,
            stall_interval=20.0,
            stall_patience=18,
        )
        result = bridge.run_damage_response(
            fault_id=scenario.fault_id,
            fault_adjacent=scenario.fault_adjacent,
            fault_body_idx=scenario.fault_body_idx,
            pre_damage_neighbor_slots=scenario.pre_damage_neighbor_slots,
            token_strategy="furthest",
        )
    finally:
        sim.disconnect()

    trajectories = {
        mid: [np.array(p, dtype=float) for p in pts]
        for mid, pts in result["trajectories"].items()
    }
    bond_snapshots = result.get("bond_snapshots", [])
    pivot_mids: Set[str] = {entry["module"] for entry in result["move_log"]}

    algo_summary = {
        "n_modules": scenario.n,
        "fault_id": scenario.fault_id,
        "reconnected": result["phase1"].get("connected", False),
        "phase1_moves": result["phase1"].get("total_moves", 0),
        "phase2_moves": result["phase2"].get("total_moves", 0),
        "total_moves": result["total_moves"],
        "pivot_modules": sorted(pivot_mids),
        "mean_error": 0.0,
        "max_error": 0.0,
        "com_drift": result["com_drift"],
        "rotation_deg": 0.0,
    }

    print(f"\nAlgorithm result:")
    print(f"  Phase 1: reconnected={algo_summary['reconnected']}, "
          f"moves={algo_summary['phase1_moves']}")
    print(f"  Phase 2: moves={algo_summary['phase2_moves']}")
    print(f"  Total moves: {algo_summary['total_moves']}")
    print(f"  Pivot modules: {algo_summary['pivot_modules']}")
    print(f"  COM drift: {algo_summary['com_drift']:.4f}")

    print_diagnostic_summary(result)

    diagnostic_report = None

    save_sim_state(trajectories, scenario.ghost_pos, scenario.ghost_bonds,
                   pivot_mids, scenario.fault_id, diagnostic_report, algo_summary,
                   bond_snapshots)

    return (trajectories, scenario.ghost_pos, scenario.ghost_bonds, pivot_mids,
            scenario.fault_id, diagnostic_report, algo_summary, bond_snapshots)


def render_video(trajectories, ghost_pos, ghost_bonds, pivot_mids, fault_id,
                 n_modules: int = 11,
                 n_frames=None, fps=30, fig_width=800, fig_height=400,
                 image_scale=1, sphere_n=10, bond_snapshots=None):
    """Render MP4 from cached or freshly run simulation trajectories.

    By default renders all trajectory samples at 30 fps (real-time).
    Pass ``n_frames`` to subsample instead.
    """
    # Add fault module trajectory (stationary at its original position)
    if fault_id not in trajectories:
        fault_pos = ghost_pos[fault_id]
        max_len = max(len(t) for t in trajectories.values())
        trajectories[fault_id] = [fault_pos.copy()] * max_len

    if n_frames is not None:
        sampled, indices = subsample_trajectories(trajectories, n_frames)
    else:
        # Use all frames (real-time at the given fps)
        max_len = max(len(t) for t in trajectories.values())
        indices = list(range(max_len))
        sampled = {mid: traj[:] for mid, traj in trajectories.items()}

    n_total = len(indices)
    print(
        f"\nRendering {n_total} frames at {fig_width}x{fig_height}, "
        f"fps={fps}, Kaleido scale={image_scale} (each frame is a full "
        f"Chromium rasterize; expect ~1-15 s/frame depending on CPU)."
    )

    # Subsample bond snapshots to match trajectory frames
    sampled_bonds = None
    if bond_snapshots:
        sampled_bonds = [
            bond_snapshots[min(i, len(bond_snapshots) - 1)]
            for i in indices
        ]

    labels = {mid: mid for mid in ghost_pos}

    # Orbiting camera: full 360 orbit over all frames
    orbit_radius = 2.5
    orbit_elevation = 0.7

    frames = []
    for i in range(len(indices)):
        positions = {mid: sampled[mid][i] for mid in sampled}
        trails = {mid: sampled[mid][:i + 1] for mid in pivot_mids
                  if mid in sampled}

        # Compute orbiting camera eye position
        angle = 2 * np.pi * i / len(indices)
        cam_eye = dict(
            x=orbit_radius * np.cos(angle),
            y=orbit_radius * np.sin(angle),
            z=orbit_elevation,
        )

        pct = int(100 * i / len(indices))
        frame_bonds = sampled_bonds[i] if sampled_bonds else None
        fig = render_frame(
            positions, labels=labels,
            ghosts=ghost_pos, ghost_bonds=ghost_bonds,
            trails=trails,
            pivot_ids=list(pivot_mids),
            fault_id=fault_id,
            active_bonds=frame_bonds,
            title=(f"{n_modules}-Module Line  |  PyBullet physics  |  fault ({fault_id})  |  "
                   f"{pct}%"),
            bounds_y=7.0,
            bounds_xz=4.0,
            camera_eye=cam_eye,
            fig_width=fig_width, fig_height=fig_height,
            sphere_n=sphere_n,
        )
        frames.append(fig_to_image(fig, scale=image_scale))
        print(f"  Frame {i + 1}/{len(indices)}", end="\r")

    for _ in range(20):
        frames.append(frames[-1])

    write_mp4(frames, OUTPUT_MP4_PATH, fps=fps)
    output_path = OUTPUT_MP4_PATH
    print(f"\nSaved: {output_path}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="PyBullet line-fault sim + MP4 (Kaleido; use --hq for slow high-res).")
    p.add_argument("--render-only", action="store_true",
                   help="skip sim; render from line_fault_pybullet_sim_cache.json")
    p.add_argument("--frames", type=int, default=80,
                   help="number of uniformly sampled frames along the trajectory")
    p.add_argument("--fps", type=int, default=12, help="MP4 frame rate")
    p.add_argument("--width", type=int, default=800, help="figure width in px")
    p.add_argument("--height", type=int, default=400, help="figure height in px")
    p.add_argument("--scale", type=int, default=1,
                   help="Kaleido raster scale (1=fast, 2=much slower, sharper)")
    p.add_argument("--sphere-n", type=int, default=10,
                   help="sphere mesh resolution (10 fast, 14 smoother)")
    p.add_argument("--hq", action="store_true",
                   help="slow preset: 120 frames, 1200x600, scale 2, sphere-n 14")
    p.add_argument("--n", type=int, default=11, help="modules in Y-line")
    p.add_argument("--fault-index", type=int, default=None,
                   help="fault body index (default: n//2)")
    p.add_argument(
        "--pivot-attract-scale",
        type=float,
        default=0.1,
        help="scale rigid pivot attract gain (default 0.1 = 10× weaker than 20 N·m²)",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    if args.hq:
        args.frames = 120
        args.width = 1200
        args.height = 600
        args.scale = 2
        args.sphere_n = 14

    if args.render_only:
        if not os.path.exists(SIM_CACHE_PATH):
            print(f"No cached simulation found at {SIM_CACHE_PATH}")
            print("Run without --render-only first to generate simulation data.")
            sys.exit(1)
        print("Loading cached simulation data...")
        (trajectories, ghost_pos, ghost_bonds, pivot_mids, fault_id,
         diagnostic_report, algo_summary, bond_snapshots) = load_sim_state()
        print(f"Loaded: {algo_summary['total_moves']} moves, "
              f"{len(trajectories)} modules")
        if diagnostic_report is not None:
            print_diagnostic_summary(diagnostic_report)
    else:
        (trajectories, ghost_pos, ghost_bonds, pivot_mids, fault_id,
         diagnostic_report, algo_summary, bond_snapshots) = run_simulation(
            n=args.n,
            fault_body_idx=args.fault_index,
            pivot_attract_scale=args.pivot_attract_scale,
        )

    render_video(
        trajectories, ghost_pos, ghost_bonds, pivot_mids, fault_id,
        n_modules=args.n,
        n_frames=args.frames,
        fps=args.fps,
        fig_width=args.width,
        fig_height=args.height,
        image_scale=args.scale,
        sphere_n=args.sphere_n,
        bond_snapshots=bond_snapshots,
    )


if __name__ == "__main__":
    main()
