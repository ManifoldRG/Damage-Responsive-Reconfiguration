"""
Generate a GIF animation of a cluster fault scenario with 40 modules.

Creates a random 40-module tree structure, injects cluster faults,
and animates the sequential damage response for each fault.
"""

import sys
import os
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(__file__))

from src.configurations import create_random_tree_configuration
from src.monte_carlo import select_faulty_modules, FAULT_MODE_RANDOM_CLUSTERS
from src.visualizer import UDQDGVisualizer
from examples.visualize_pivots import (
    EXPORT_FRAMES,
    PAUSE_BETWEEN,
    _export_corner_pivot,
    _export_lateral_pivot,
    _export_restoration_pivot,
    _export_parallel_pivotsteps,
    _render_all_static_modules,
)

N_MODULES = 40
N_FAULTS = 10
SEED = 42
OUTPUT_PATH = "gifs/cluster_fault_40_10f.gif"


def write_pause(viz, n_frames, frame_counter, total_frames,
                initial_az, initial_el, rot_az, rot_el):
    """Write pause frames with camera rotation."""
    for i in range(n_frames):
        t = min(1.0, (frame_counter + i) / max(1, total_frames))
        viz.plotter.camera.azimuth = initial_az + t * rot_az
        viz.plotter.camera.elevation = initial_el + t * rot_el
        viz.plotter.write_frame()
    return frame_counter + n_frames


def main():
    os.makedirs("gifs", exist_ok=True)

    print(f"=== Cluster Fault GIF: {N_MODULES} modules, {N_FAULTS} faults ===")

    # --- Pre-compute all results ---
    system_data = create_random_tree_configuration(N_MODULES, seed=SEED)
    fault_ids = select_faulty_modules(system_data, N_FAULTS, SEED + 1000, FAULT_MODE_RANDOM_CLUSTERS)
    print(f"Fault modules: {fault_ids}")

    fault_results = []
    for fid in fault_ids:
        if not system_data.modules[fid].is_active:
            fault_results.append((fid, None))
            print(f"  {fid}: already inactive")
            continue
        result = system_data.full_damage_response(
            fault_module_id=fid,
            restore_positions=True,
            max_phase1_iterations=1000,
            max_phase2_iterations=100,
            record_steps=True,
        )
        fault_results.append((fid, result))
        p1 = result.get("phase1") or {}
        p2 = result.get("phase2") or {}
        print(f"  {fid}: P1={p1.get('total_moves', 0)} moves, P2={p2.get('restoration_moves', 0)} moves")

    # --- Count total operations for camera budget ---
    pause_frames = int(PAUSE_BETWEEN * 30)
    phase_pause = int(1.5 * 30)  # shorter pauses to keep GIF manageable
    fault_pause = int(1.0 * 30)

    total_ops = 0
    for _, result in fault_results:
        if result is None:
            continue
        p1 = result.get("phase1") or {}
        p2 = result.get("phase2") or {}
        pgroups = p1.get("parallel_steps", [])
        total_ops += len(pgroups) + len(p2.get("steps", []))

    total_animation_frames = max(1, total_ops * (EXPORT_FRAMES + pause_frames))

    # --- Create fresh system for visualization ---
    system = create_random_tree_configuration(N_MODULES, seed=SEED)

    viz = UDQDGVisualizer(system)
    viz.setup_scene()
    viz.plotter.open_gif(OUTPUT_PATH, fps=30)
    viz.render_system()

    # Initial pause
    for _ in range(15):
        viz.plotter.write_frame()

    fc = 0  # frame counter
    az0 = viz.plotter.camera.azimuth
    el0 = viz.plotter.camera.elevation
    rot_az = 180.0
    rot_el = 30.0

    # --- Animate each fault ---
    for fault_idx, (fid, result) in enumerate(fault_results):
        print(f"\n--- Fault {fault_idx + 1}/{len(fault_results)}: {fid} ---")

        # Mark fault (turns module red)
        system.mark_fault(fid)
        _render_all_static_modules(viz, set())
        viz.plotter.write_frame()

        if result is None:
            # Already inactive, just show red and continue
            fc = write_pause(viz, fault_pause, fc, total_animation_frames,
                             az0, el0, rot_az, rot_el)
            continue

        p1 = result.get("phase1") or {}
        p2 = result.get("phase2") or {}
        histories = result.get("movement_histories", {})
        pgroups = p1.get("parallel_steps", [])
        p2_steps = p2.get("steps", [])

        if not pgroups and not p2_steps:
            # No movement needed — just pause briefly
            fc = write_pause(viz, fault_pause, fc, total_animation_frames,
                             az0, el0, rot_az, rot_el)
            continue

        # Add ghost markers
        ghost_ids = []
        ghost_sphere = pv.Sphere(radius=viz.sphere_radius * 0.9)
        for mid, history in histories.items():
            if history.has_moved():
                pos = history.get_original_position()
                ghost = ghost_sphere.copy()
                ghost.points += pos
                name = f"ghost_{mid}_{fault_idx}"
                viz.plotter.add_mesh(
                    ghost, color=viz.colors["ghost"], opacity=0.3,
                    name=name, reset_camera=False,
                )
                ghost_ids.append(name)
        viz.plotter.write_frame()

        # Phase 1: Coagulation
        if pgroups:
            print(f"  Phase 1: {p1.get('total_moves', 0)} moves in {len(pgroups)} groups")
            for idx, group in enumerate(pgroups):
                t = fc / max(1, total_animation_frames)
                viz.plotter.camera.azimuth = az0 + t * rot_az
                viz.plotter.camera.elevation = el0 + t * rot_el

                _export_parallel_pivotsteps(
                    viz, group, EXPORT_FRAMES,
                    fc, total_animation_frames,
                    az0, el0, rot_az, rot_el,
                )
                fc += EXPORT_FRAMES

                if idx < len(pgroups) - 1:
                    fc = write_pause(viz, pause_frames, fc, total_animation_frames,
                                     az0, el0, rot_az, rot_el)

        # Pause between phases
        if pgroups and p2_steps:
            fc = write_pause(viz, phase_pause, fc, total_animation_frames,
                             az0, el0, rot_az, rot_el)

        # Phase 2: Restructuring
        if p2_steps:
            print(f"  Phase 2: {len(p2_steps)} moves")
            for idx, step in enumerate(p2_steps):
                t = fc / max(1, total_animation_frames)
                viz.plotter.camera.azimuth = az0 + t * rot_az
                viz.plotter.camera.elevation = el0 + t * rot_el

                _export_restoration_pivot(
                    viz, step.module_id, step.param1, step.param2,
                    step.pivot_type, EXPORT_FRAMES,
                    fc, total_animation_frames,
                    az0, el0, rot_az, rot_el,
                    from_pos=step.from_pos, to_pos=step.to_pos,
                )
                fc += EXPORT_FRAMES

                if idx < len(p2_steps) - 1:
                    fc = write_pause(viz, pause_frames, fc, total_animation_frames,
                                     az0, el0, rot_az, rot_el)

        # Clean up ghost markers to free VTK actors
        for name in ghost_ids:
            try:
                viz.plotter.remove_actor(name, reset_camera=False)
            except Exception:
                pass

        # Pause between faults
        if fault_idx < len(fault_results) - 1:
            fc = write_pause(viz, fault_pause, fc, total_animation_frames,
                             az0, el0, rot_az, rot_el)

    # Final pause
    for i in range(15):
        t = min(1.0, (fc + i) / max(1, total_animation_frames))
        viz.plotter.camera.azimuth = az0 + t * rot_az
        viz.plotter.camera.elevation = el0 + t * rot_el
        viz.plotter.write_frame()

    viz.plotter.close()
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
