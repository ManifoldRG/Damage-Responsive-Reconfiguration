#!/usr/bin/env python3
"""
Interactive 3D visualization of UDQDG system pivots.

DEPRECATED: The ``full_damage_response`` / ``get_damage_response_summary``
calls in this script were removed from ``UDQDGSystem`` in the Graph
Parity refactor. Pivot animations are now driven by replaying the
``move_log`` produced by ``DecentralizedCoagulation`` /
``DecentralizedRestructuring`` running on top of ``GraphSimulator``
(see ``src/agent_policy.py`` and ``src/graph_sim.py``). Migrate this
script's data-collection section to that pipeline before running.

Usage:
    python examples/visualize_pivots.py [config_type]

Config types: star, tree, line, cross, helix, l_shape
"""

import sys
import os
import numpy as np
import pyvista as pv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import system components
# from src.udqdg_system import UDQDGSystem
from src.visualizer import UDQDGVisualizer
from src.configurations import (
    create_star_configuration,
    create_tree_configuration,
    create_line_configuration,
    create_cross_configuration,
    create_helix_configuration,
    create_l_shape_configuration,
    create_grid_configuration,
    create_lattice_plane_configuration,
    create_t_shape_configuration,
    create_ring_configuration,
    create_dual_star_bridge_configuration
)

# ============================================
# ANIMATION SPEED CONTROL - SMOOTH AND FAST
# ============================================
TOTAL_FRAMES = 12            # Interpolation frames for interactive (12 = fast, still smooth)
EXPORT_FRAMES = 12           # Same frames as interactive for matching speed
PAUSE_BETWEEN = 0.25          # Pause between pivots (brief pause for visual separation)
# ============================================


def demo_corner_pivots():
    """Demonstrate corner pivot animations."""
    print("=== Corner Pivot Demo ===")
    print("Creating L-shaped configuration...")

    system = create_l_shape_configuration()
    viz = UDQDGVisualizer(system)

    print("Initial configuration loaded.")
    print("Performing corner pivot sequence...")

    # Initialize window first
    viz.show_window()

    # Pause to show initial configuration
    import time
    time.sleep(0.5)

    # Define pivot sequence (avoiding collisions with M2)
    pivot_sequence = [
        ('corner', 'M4', 'M3', 'POS_Z'),   # Up
        ('corner', 'M4', 'M3', 'POS_Y'),   # Forward
        ('corner', 'M4', 'M3', 'NEG_Z'),   # Down
        ('corner', 'M4', 'M3', 'POS_X'),   # Back to original
    ]

    print("Animating sequence with new fast system...")
    viz.animate_pivot_sequence(pivot_sequence, n_frames=TOTAL_FRAMES, pause_between=PAUSE_BETWEEN)

    # Pause at end
    time.sleep(0.5)

    viz.plotter.close()


def demo_lateral_pivots():
    """Demonstrate lateral pivot animations."""
    print("=== Lateral Pivot Demo ===")
    print("Creating line configuration for lateral pivots...")

    # Create a simple line along X axis
    system = create_line_configuration(length=6, axis='X')

    # Add a branch module above the line to demonstrate rolling
    system.add_module("M6", np.array([2, 0, 1]))
    system.connect_modules("M2", "M6")

    viz = UDQDGVisualizer(system)

    print("Initial configuration loaded.")
    print("Performing lateral pivot sequence...")

    # Initialize window
    viz.show_window()

    # Pause to show initial configuration
    import time
    time.sleep(0.5)

    # Branch module (above M2) can roll along the line
    # It maintains its vertical offset while moving horizontally
    pivot_sequence = [ # TODO need to update connections upon lateral pivot?
        ('lateral', 'M6', 'M2', 'M3'),   # Roll from M2 to M3
        ('lateral', 'M6', 'M3', 'M4'),   # Continue rolling to M4
        ('lateral', 'M6', 'M4', 'M5'),   # Roll to M5
        ('lateral', 'M6', 'M5', 'M4'),   # Roll back to M4
    ]

    print("Animating lateral pivot sequence (rolling along line)...")
    viz.animate_pivot_sequence(pivot_sequence, n_frames=TOTAL_FRAMES, pause_between=PAUSE_BETWEEN)

    viz.plotter.close()


def demo_parallel_pivots():
    """Demonstrate parallel pivot animations."""
    print("=== Parallel Pivot Demo ===")
    print("Creating star configuration...")

    system = create_star_configuration(size=3)
    viz = UDQDGVisualizer(system)

    print("Initial configuration loaded.")
    print("Performing parallel Z-axis rotation...")

    # Initialize window
    viz.show_window()

    # Pause to show initial configuration
    import time
    time.sleep(0.5)

    # Z-axis rotation: outer modules in XY plane rotate together
    # Each pivots around their middle neighbor (2-step from center)
    pivot_sequence = [
        ('parallel', [
            ('corner', 'PX3', 'PX2', 'POS_Z'),  # +X arm bends up
            ('corner', 'PY3', 'PY2', 'POS_Z'),  # +Y arm bends up
            ('corner', 'NX3', 'NX2', 'POS_Z'),  # -X arm bends up
            ('corner', 'NY3', 'NY2', 'POS_Z'),  # -Y arm bends up
        ]),
        ('parallel', [
            ('corner', 'PX3', 'PX2', 'POS_Y'),  # +X arm bends to the right
            ('corner', 'NX3', 'NX2', 'NEG_Y'),  # -X arm bends to the left
        ]),
    ]

    print("Executing parallel movements...")
    viz.animate_pivot_sequence(pivot_sequence, n_frames=TOTAL_FRAMES, pause_between=PAUSE_BETWEEN)

    viz.plotter.close()


def demo_spiral_motion():
    """Demonstrate spiral motion - rotating and rolling down a line."""
    print("=== Spiral Motion Demo ===")
    print("Creating line configuration...")

    # Create a line along X axis
    system = create_line_configuration(length=6, axis='X')
    viz = UDQDGVisualizer(system)

    print("Initial line configuration loaded.")
    print("Performing spiral motion sequence...")

    # Initialize window
    viz.show_window()

    # Pause to show initial configuration
    import time
    time.sleep(0.5)

    # Sequence: Create L-shape, then spiral down by rotating and laterally moving
    pivot_sequence = [
        # Create the L-shape: rotate M5 upward (around M4)
        ('corner', 'M5', 'M4', 'POS_Z'),

        # Now spiral down: rotate, lateral move, rotate, lateral move, repeat
        # M5 starts at [4, 0, 1] connected to M4
        ('corner', 'M5', 'M4', 'POS_Y'), 
        ('lateral', 'M5', 'M4', 'M3'),

        ('corner', 'M5', 'M3', 'NEG_Z'),
        ('lateral', 'M5', 'M3', 'M2'),   

        ('corner', 'M5', 'M2', 'NEG_Y'),  # Error here, there is a weird connection that appears # TODO
        ('lateral', 'M5', 'M2', 'M1'),      # Lateral move to M1

        ('corner', 'M5', 'M1', 'POS_Z'),  # Rotate to horizontal
        ('lateral', 'M5', 'M1', 'M0'),      # Lateral move to M0

        ('corner', 'M5', 'M0', 'NEG_X'),  # Final rotation to align with line
    ]

    print("Animating spiral motion (L-shape then spiral down line)...")
    viz.animate_pivot_sequence(pivot_sequence, n_frames=TOTAL_FRAMES, pause_between=PAUSE_BETWEEN)

    viz.plotter.close()


def demo_reconfiguration_sequence():
    """Demonstrate a sequence of pivots to reconfigure the structure."""
    print("=== Reconfiguration Sequence Demo ===")
    print("Creating tree configuration...")

    system = create_tree_configuration(depth=3, branching=2)
    viz = UDQDGVisualizer(system)

    print("Initial tree configuration loaded.")
    print("Reconfiguring to more compact form...")

    # Initialize window
    viz.show_window()

    # Pause to show initial configuration
    import time
    time.sleep(0.5)

    # Perform a series of pivots to reorganize
    # (This is illustrative - actual pivot sequence would depend on connectivity)
    print("Step 1: Corner pivot on branch")
    if viz.animate_corner_pivot("R_0_0", "R_0", "NEG_X", n_frames=30):
        print("Success!")

    print("Step 2: Another corner pivot")
    if viz.animate_corner_pivot("R_1_0", "R_1", "POS_Z", n_frames=30):
        print("Success!")

    viz.plotter.close()


def interactive_visualization(config_type: str = 'star'):
    """
    Launch interactive visualization of specified configuration.

    Args:
        config_type: Type of configuration to visualize
    """
    print(f"=== Interactive {config_type.upper()} Configuration ===")

    # Create system based on type
    config_map = {
        'star': lambda: create_star_configuration(size=3),
        'tree': lambda: create_tree_configuration(depth=3, branching=2),
        'line': lambda: create_line_configuration(length=7, axis='X'),
        'cross': lambda: create_cross_configuration(arm_length=3),
        'helix': lambda: create_helix_configuration(turns=2, modules_per_turn=6),
        'l_shape': lambda: create_l_shape_configuration()
    }

    if config_type not in config_map:
        print(f"Unknown configuration: {config_type}")
        print(f"Available: {', '.join(config_map.keys())}")
        return

    system = config_map[config_type]()
    viz = UDQDGVisualizer(system)

    print(f"Modules: {len(system.modules)}")
    print(f"Edges: {len(system.get_all_edges())}")
    print("\nControls:")
    print("  - Left click + drag: Rotate view")
    print("  - Right click + drag: Pan view")
    print("  - Scroll: Zoom in/out")
    print("  - 'q': Quit")

    viz.show()


def demo_damage_response():
    """Demonstrate damage response and reconfiguration."""
    print("=== Damage Response Demo ===")
    print("Creating star configuration...")

    system = create_star_configuration(size=3)
    viz = UDQDGVisualizer(system)

    # Simulate damage
    print("Simulating damage to module PX2...")
    system.modules['PX2'].is_active = False

    # Now PX3 cannot pivot (neighbor inactive)
    print("Module PX3 is now non-pivotable (has inactive neighbor)")

    print("Damaged module shown in red.")
    print("System will need reconfiguration to restore connectivity...")

    viz.show()


def demo_ego_fault_response(parallel: bool = True):
    """
    Demonstrate the ego-based fault response algorithm with step-by-step animation.
    Shows how modules autonomously respond to a fault and reconnect.

    Args:
        parallel: If True, animate parallel subgraph moves simultaneously
    """
    print("=== Ego Fault Response Demo ===")
    print("Creating 3D star configuration...")

    import time

    # Step 1: Create system and run algorithm to get steps
    # We run on a separate system to capture the steps
    system_for_steps = create_star_configuration(size=2)
    system_for_steps.mark_fault('C')  # Fault at center disconnects all arms

    print(f"\nConfiguration: {len(system_for_steps.modules)} modules")
    print("Fault location: Center (C)")
    print("This disconnects all 6 arms into separate components\n")

    print("Running ego_fault_response algorithm...")
    stats = system_for_steps.ego_fault_response(
        'C',
        record_steps=True,
        parallel_subgraphs=parallel
    )

    print(f"Algorithm complete:")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  Reconnected: {stats['reconnected']}")
    print(f"  Collisions resolved: {stats['collisions_resolved']}")
    print(f"  Steps recorded: {len(stats['steps'])}")
    if parallel and stats.get('parallel_steps'):
        print(f"  Parallel groups: {len(stats['parallel_steps'])}")

    # Step 2: Create a fresh system for visualization
    system = create_star_configuration(size=2)
    system.mark_fault('C')  # Mark fault (visual indicator)

    viz = UDQDGVisualizer(system)

    print("\nStarting visualization...")
    print("Controls:")
    print("  - Left click + drag: Rotate view")
    print("  - Right click + drag: Pan view")
    print("  - Scroll: Zoom in/out")
    print("  - 'q': Quit\n")

    # Initialize window
    viz.show_window()

    # Pause to show initial state with fault
    print("Showing initial state with fault (red module)...")
    time.sleep(1.0)

    # Animate the recorded steps
    if parallel and stats.get('parallel_steps'):
        print("\nAnimating ego fault response steps (parallel mode)...")
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False,
            parallel_groups=stats['parallel_steps']
        )
    else:
        print("\nAnimating ego fault response steps (sequential mode)...")
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False
        )

    # Pause at end
    print("\nAnimation complete! System reconnected.")
    time.sleep(1.0)

    viz.plotter.close()


def demo_ego_large_star():
    """
    Demonstrate ego fault response on a larger 3D star (size=3).
    More modules means more parallel moves and potential collisions.
    """
    print("=== Ego Fault Response - Large Star Demo ===")
    print("Creating large 3D star configuration (size=3)...")

    import time

    # Run algorithm
    system_for_steps = create_star_configuration(size=3)
    system_for_steps.mark_fault('C')

    print(f"\nConfiguration: {len(system_for_steps.modules)} modules")
    print("Fault location: Center (C)")
    print("This disconnects all 6 arms (18 modules total)\n")

    print("Running ego_fault_response algorithm...")
    stats = system_for_steps.ego_fault_response('C', record_steps=True, parallel_subgraphs=True)

    print(f"Algorithm complete:")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  Reconnected: {stats['reconnected']}")
    print(f"  Collisions resolved: {stats['collisions_resolved']}")
    if stats.get('parallel_steps'):
        print(f"  Parallel groups: {len(stats['parallel_steps'])}")

    # Fresh system for visualization
    system = create_star_configuration(size=3)
    system.mark_fault('C')

    viz = UDQDGVisualizer(system)

    print("\nStarting visualization...")
    viz.show_window()

    time.sleep(1.0)

    if stats.get('parallel_steps'):
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False,
            parallel_groups=stats['parallel_steps']
        )
    else:
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False
        )

    print("\nAnimation complete!")
    time.sleep(1.0)
    viz.plotter.close()


def demo_ego_t_shape():
    """
    Demonstrate ego fault response on a T-shaped configuration.
    Center fault creates 3 disconnected arms that must reconnect.
    """
    print("=== Ego Fault Response - T-Shape Demo ===")
    print("Creating T-shaped configuration...")

    import time

    # Run algorithm
    system_for_steps = create_t_shape_configuration(arm_length=3)
    system_for_steps.mark_fault('C')

    print(f"\nConfiguration: {len(system_for_steps.modules)} modules")
    print("Fault location: Center (C)")
    print("This disconnects 3 arms\n")

    print("Running ego_fault_response algorithm...")
    stats = system_for_steps.ego_fault_response('C', record_steps=True, parallel_subgraphs=True)

    print(f"Algorithm complete:")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  Reconnected: {stats['reconnected']}")
    print(f"  Collisions resolved: {stats['collisions_resolved']}")

    # Fresh system for visualization
    system = create_t_shape_configuration(arm_length=3)
    system.mark_fault('C')

    viz = UDQDGVisualizer(system)

    print("\nStarting visualization...")
    viz.show_window()

    time.sleep(1.0)

    if stats.get('parallel_steps'):
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False,
            parallel_groups=stats['parallel_steps']
        )
    else:
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False
        )

    print("\nAnimation complete!")
    time.sleep(1.0)
    viz.plotter.close()


def demo_ego_cross():
    """
    Demonstrate ego fault response on a cross/+ configuration.
    Center fault creates 5 disconnected arms (4 in XY plane + 1 in Z).
    """
    print("=== Ego Fault Response - Cross Demo ===")
    print("Creating cross configuration...")

    import time

    # Run algorithm
    system_for_steps = create_cross_configuration(arm_length=2)
    system_for_steps.mark_fault('C')

    print(f"\nConfiguration: {len(system_for_steps.modules)} modules")
    print("Fault location: Center (C)")
    print("This disconnects 5 arms\n")

    print("Running ego_fault_response algorithm...")
    stats = system_for_steps.ego_fault_response('C', record_steps=True, parallel_subgraphs=True)

    print(f"Algorithm complete:")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  Reconnected: {stats['reconnected']}")
    print(f"  Collisions resolved: {stats['collisions_resolved']}")

    # Fresh system for visualization
    system = create_cross_configuration(arm_length=2)
    system.mark_fault('C')

    viz = UDQDGVisualizer(system)

    print("\nStarting visualization...")
    viz.show_window()

    time.sleep(1.0)

    if stats.get('parallel_steps'):
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False,
            parallel_groups=stats['parallel_steps']
        )
    else:
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False
        )

    print("\nAnimation complete!")
    time.sleep(1.0)
    viz.plotter.close()


def demo_ego_line():
    """
    Demonstrate ego fault response on a 9-module line.
    Middle fault splits the line into two halves that must reconnect.
    """
    print("=== Ego Fault Response - Line Demo ===")
    print("Creating 9-module line configuration...")

    import time

    # Run algorithm
    system_for_steps = create_line_configuration(length=9, axis='X')
    # Fault in the middle (M4)
    system_for_steps.mark_fault('M4')

    print(f"\nConfiguration: {len(system_for_steps.modules)} modules in a line")
    print("Fault location: M4 (middle)")
    print("This splits into two halves: [M0-M3] and [M5-M8]\n")

    print("Running ego_fault_response algorithm...")
    stats = system_for_steps.ego_fault_response('M4', record_steps=True, parallel_subgraphs=True)

    print(f"Algorithm complete:")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  Reconnected: {stats['reconnected']}")
    print(f"  Collisions resolved: {stats['collisions_resolved']}")
    if stats.get('parallel_steps'):
        print(f"  Parallel groups: {len(stats['parallel_steps'])}")

    # Fresh system for visualization
    system = create_line_configuration(length=9, axis='X')
    system.mark_fault('M4')

    viz = UDQDGVisualizer(system)

    print("\nStarting visualization...")
    viz.show_window()

    time.sleep(1.0)

    if stats.get('parallel_steps'):
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False,
            parallel_groups=stats['parallel_steps']
        )
    else:
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False
        )

    print("\nAnimation complete!")
    time.sleep(1.0)
    viz.plotter.close()


def demo_ego_grid():
    """
    Demonstrate ego fault response on a 3x3x3 grid.
    Center fault tests reconnection in a dense 3D structure.
    """
    print("=== Ego Fault Response - 3D Grid Demo ===")
    print("Creating 3x3x3 grid configuration...")

    import time

    # Run algorithm
    system_for_steps = create_grid_configuration(size=3)
    # Fault in the center
    system_for_steps.mark_fault('M1_1_1')

    print(f"\nConfiguration: {len(system_for_steps.modules)} modules in 3x3x3 grid")
    print("Fault location: M1_1_1 (center)")

    print("\nRunning ego_fault_response algorithm...")
    stats = system_for_steps.ego_fault_response('M1_1_1', record_steps=True, parallel_subgraphs=True)

    print(f"Algorithm complete:")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  Reconnected: {stats['reconnected']}")
    print(f"  Collisions resolved: {stats['collisions_resolved']}")

    # Fresh system for visualization
    system = create_grid_configuration(size=3)
    system.mark_fault('M1_1_1')

    viz = UDQDGVisualizer(system)

    print("\nStarting visualization...")
    viz.show_window()

    time.sleep(1.0)

    if stats.get('parallel_steps'):
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False,
            parallel_groups=stats['parallel_steps']
        )
    else:
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False
        )

    print("\nAnimation complete!")
    time.sleep(1.0)
    viz.plotter.close()


def demo_ego_ring():
    """
    Demonstrate ego fault response on a ring/loop configuration.
    Breaking one module in a ring should still leave it connected.
    """
    print("=== Ego Fault Response - Ring Demo ===")
    print("Creating ring configuration...")

    import time

    # Run algorithm
    system_for_steps = create_ring_configuration(radius=2)
    # Find a module to fault - pick one on the top edge
    system_for_steps.mark_fault('T0')

    print(f"\nConfiguration: {len(system_for_steps.modules)} modules in a ring")
    print("Fault location: T0 (top center)")
    print("Ring should remain connected since it's a loop!\n")

    print("Running ego_fault_response algorithm...")
    stats = system_for_steps.ego_fault_response('T0', record_steps=True, parallel_subgraphs=True)

    print(f"Algorithm complete:")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  Reconnected: {stats['reconnected']}")

    # Fresh system for visualization
    system = create_ring_configuration(radius=2)
    system.mark_fault('T0')

    viz = UDQDGVisualizer(system)

    print("\nStarting visualization...")
    viz.show_window()

    time.sleep(1.0)

    if stats.get('parallel_steps') and stats['parallel_steps']:
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False,
            parallel_groups=stats['parallel_steps']
        )
    elif stats['steps']:
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False
        )
    else:
        print("No moves needed - ring remained connected!")

    print("\nAnimation complete!")
    time.sleep(1.0)
    viz.plotter.close()


def demo_ego_dual_star_bridge():
    """
    Demonstrate ego fault response on dual star bridge configuration.
    Bridge damage disconnects the two stars.
    """
    print("=== Ego Fault Response - Dual Star Bridge Demo ===")
    print("Creating dual star bridge configuration...")

    import time

    # Run algorithm
    bridge_length = 9
    center_module = f'BR{bridge_length // 2 + 1}'  # BR5 for 9-module bridge

    system_for_steps = create_dual_star_bridge_configuration(star_size=3, bridge_length=bridge_length)
    system_for_steps.mark_fault(center_module)

    print(f"\nConfiguration: {len(system_for_steps.modules)} modules")
    print(f"Fault location: {center_module} (center of {bridge_length}-module bridge)")
    print("This disconnects the two stars\n")

    print("Running ego_fault_response algorithm...")
    stats = system_for_steps.ego_fault_response(center_module, record_steps=True, parallel_subgraphs=True)

    print(f"Algorithm complete:")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  Reconnected: {stats['reconnected']}")
    print(f"  Collisions resolved: {stats['collisions_resolved']}")
    if stats.get('parallel_steps'):
        print(f"  Parallel groups: {len(stats['parallel_steps'])}")

    # Fresh system for visualization
    system = create_dual_star_bridge_configuration(star_size=3, bridge_length=3)
    system.mark_fault('BR2')

    viz = UDQDGVisualizer(system)

    print("\nStarting visualization...")
    viz.show_window()

    time.sleep(1.0)

    if stats.get('parallel_steps'):
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False,
            parallel_groups=stats['parallel_steps']
        )
    else:
        viz.animate_ego_response(
            stats['steps'],
            n_frames=TOTAL_FRAMES,
            pause_between=PAUSE_BETWEEN,
            camera_mode='fixed',
            reset_positions=False
        )

    print("\nAnimation complete!")
    time.sleep(1.0)
    viz.plotter.close()


def demo_general_sequence():
    """Demonstrate the general-purpose pivot sequence animator."""
    print("=== General Pivot Sequence Demo ===")
    print("Creating star configuration...")

    system = create_star_configuration(size=3)
    viz = UDQDGVisualizer(system)

    print("Demonstrating all pivot types and directions...")
    viz.show_window()

    # Pause to show initial configuration
    import time
    time.sleep(0.5)

    # Comprehensive sequence: demonstrate freedom of direction, then parallel capabilities
    sequence = [
        # Demonstrate freedom of direction - single pivots in various directions
        ('corner', 'PX3', 'PX2', 'POS_Z'),      # +Z direction
        ('corner', 'PY3', 'PY2', 'NEG_Z'),      # -Z direction
        ('corner', 'NX3', 'NX2', 'POS_Y'),      # +Y direction
        ('corner', 'NY3', 'NY2', 'POS_X'),      # +X direction
        ('corner', 'PZ3', 'PZ2', 'POS_X'),      # +X direction
        ('corner', 'NZ3', 'NZ2', 'NEG_X'),      # -X direction

        # 2-parallel movement (corner + corner)
        ('parallel', [
            ('corner', 'PX3', 'PX2', 'POS_Y'),
            ('corner', 'PY3', 'PY2', 'POS_X'),
        ]),

        # 3-parallel movement (corner + corner + corner)
        ('parallel', [
            ('corner', 'NX3', 'NX2', 'POS_Z'),
            ('corner', 'NY3', 'NY2', 'POS_Z'),
            ('corner', 'PZ3', 'PZ2', 'NEG_Y'),
        ]),

        # 4-parallel movement (all corners)
        ('parallel', [
            ('corner', 'PX3', 'PX2', 'NEG_Z'),
            ('corner', 'PY3', 'PY2', 'NEG_Z'),
            ('corner', 'NZ3', 'NZ2', 'POS_Y'),
            ('corner', 'NX3', 'NX2', 'NEG_Y'),
        ]),
    ]

    print("Animating comprehensive sequence...")
    viz.animate_pivot_sequence(sequence, n_frames=TOTAL_FRAMES, pause_between=PAUSE_BETWEEN)

    # Pause at end
    time.sleep(0.5)

    viz.plotter.close()

def export_full_damage_response_gif(
    config_func,
    config_kwargs: dict,
    fault_module: str,
    output_path: str = None,
    rotate_camera: bool = True
):
    """
    Export a full damage response animation (Phase 1 + Phase 2) as a GIF.

    Args:
        config_func: Configuration creation function (e.g., create_star_configuration)
        config_kwargs: Keyword arguments for the configuration function
        fault_module: ID of the module to fault
        output_path: Output file path (defaults to gifs/full_damage_response.gif)
        rotate_camera: If True, slowly rotate camera during animation. If False, keep fixed.
    """
    import os

    if output_path is None:
        os.makedirs('gifs', exist_ok=True)
        output_path = f'gifs/full_damage_response.gif'

    print(f"=== Exporting Full Damage Response to GIF ===")
    print(f"Output: {output_path}")

    # Run the full damage response algorithm to get all data
    system_for_data = config_func(**config_kwargs)
    result = system_for_data.full_damage_response(
        fault_module,
        restore_positions=True,
        max_phase2_iterations=100,
        record_steps=True
    )

    if not result.get('overall_success') and not result.get('phase1', {}).get('reconnected'):
        print(f"Damage response failed: {result.get('error', 'Phase 1 failed')}")
        return

    phase1 = result['phase1']
    phase2 = result.get('phase2')
    histories = result['movement_histories']

    print(f"\nPhase 1: {phase1.get('total_moves', 0)} moves")
    if phase2:
        print(f"Phase 2: {phase2.get('restoration_moves', 0)} moves")

    # Create fresh system for visualization
    system = config_func(**config_kwargs)
    system.mark_fault(fault_module)

    viz = UDQDGVisualizer(system)
    viz.setup_scene()

    # Open GIF writer
    viz.plotter.open_gif(output_path, fps=30)

    # Render initial state
    viz.render_system()
    viz.plotter.write_frame()

    # Initial pause (0.5 sec at 30fps = 15 frames)
    for _ in range(14):
        viz.plotter.write_frame()

    # Calculate total frames for camera rotation
    phase1_steps = phase1.get('steps', [])
    parallel_groups = phase1.get('parallel_steps', None)
    phase2_steps = phase2.get('steps', []) if phase2 else []

    # Count Phase 1 operations
    if parallel_groups:
        phase1_ops = sum(len(g) for g in parallel_groups)
        phase1_groups = len(parallel_groups)
    else:
        phase1_ops = len(phase1_steps)
        phase1_groups = len(phase1_steps)

    phase2_ops = len(phase2_steps)
    pause_frames = int(PAUSE_BETWEEN * 30)
    phase_pause_frames = int(2.0 * 30)  # 2 second pause between phases

    total_animation_frames = (
        phase1_groups * EXPORT_FRAMES +
        (phase1_groups - 1) * pause_frames +
        phase_pause_frames +
        phase2_ops * EXPORT_FRAMES +
        max(0, phase2_ops - 1) * pause_frames
    )

    frame_counter = 0
    initial_azimuth = viz.plotter.camera.azimuth
    initial_elevation = viz.plotter.camera.elevation
    total_rotation_azimuth = 120.0 if rotate_camera else 0.0
    total_rotation_elevation = 30.0 if rotate_camera else 0.0

    # ========== ADD GHOST MARKERS FOR ALL DISPLACED MODULES ==========
    # Show ghost markers at original positions before Phase 1 starts
    ghost_sphere = pv.Sphere(radius=viz.sphere_radius * 0.9)
    ghost_positions = {}

    for module_id, history in histories.items():
        if history.has_moved():
            original_pos = history.get_original_position()
            ghost_positions[module_id] = original_pos

            ghost = ghost_sphere.copy()
            ghost.points += original_pos
            viz.plotter.add_mesh(
                ghost,
                color=viz.colors['ghost'],
                opacity=0.3,
                name=f"ghost_{module_id}",
                reset_camera=False
            )

    # Write frame with ghosts visible
    viz.plotter.write_frame()

    # ========== PHASE 1: Reconnection ==========
    print("\n--- Phase 1: Connectivity Restoration ---")

    if parallel_groups:
        for idx, group in enumerate(parallel_groups):
            print(f"  [Iteration {group[0].iteration if group else 0}] {len(group)} parallel moves")

            # Update camera
            global_t = frame_counter / max(1, total_animation_frames)
            viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
            viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation

            # Animate parallel group
            _export_parallel_pivotsteps(
                viz, group, EXPORT_FRAMES,
                frame_counter, total_animation_frames,
                initial_azimuth, initial_elevation,
                total_rotation_azimuth, total_rotation_elevation
            )
            frame_counter += EXPORT_FRAMES

            # Pause between groups
            if idx < len(parallel_groups) - 1:
                for i in range(pause_frames):
                    # Use frame_counter + i to smoothly rotate camera during pause
                    global_t = (frame_counter + i) / max(1, total_animation_frames)
                    viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
                    viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation
                    viz.plotter.write_frame()
                frame_counter += pause_frames
    else:
        for idx, step in enumerate(phase1_steps):
            print(f"  {step.pivot_type} {step.module_id}")

            if step.pivot_type == 'corner':
                _export_corner_pivot(
                    viz, step.module_id, step.param1, step.param2, EXPORT_FRAMES,
                    frame_counter, total_animation_frames,
                    initial_azimuth, initial_elevation,
                    total_rotation_azimuth, total_rotation_elevation,
                    from_pos=step.from_pos, to_pos=step.to_pos
                )
            elif step.pivot_type == 'lateral':
                _export_lateral_pivot(
                    viz, step.module_id, step.param1, step.param2, EXPORT_FRAMES,
                    frame_counter, total_animation_frames,
                    initial_azimuth, initial_elevation,
                    total_rotation_azimuth, total_rotation_elevation,
                    from_pos=step.from_pos, to_pos=step.to_pos
                )
            frame_counter += EXPORT_FRAMES

            if idx < len(phase1_steps) - 1:
                for i in range(pause_frames):
                    # Use frame_counter + i to smoothly rotate camera during pause
                    global_t = (frame_counter + i) / max(1, total_animation_frames)
                    viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
                    viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation
                    viz.plotter.write_frame()
                frame_counter += pause_frames

    # ========== PAUSE BETWEEN PHASES ==========
    print("\n--- Pause between phases ---")
    for i in range(phase_pause_frames):
        # Use frame_counter + i to smoothly rotate camera during pause
        global_t = (frame_counter + i) / max(1, total_animation_frames)
        viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
        viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation
        viz.plotter.write_frame()
    frame_counter += phase_pause_frames

    # ========== PHASE 2: Position Restoration ==========
    if phase2_steps:
        print("\n--- Phase 2: Position Restoration ---")

        # Ghost markers already added before Phase 1
        # Animate Phase 2 steps
        for idx, step in enumerate(phase2_steps):
            dist_improvement = step.distance_before - step.distance_after
            print(f"  {step.pivot_type} {step.module_id}: {step.distance_before:.2f} → {step.distance_after:.2f}")

            # Update camera
            global_t = frame_counter / max(1, total_animation_frames)
            viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
            viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation

            # Animate restoration pivot (with green color)
            _export_restoration_pivot(
                viz, step.module_id, step.param1, step.param2,
                step.pivot_type, EXPORT_FRAMES,
                frame_counter, total_animation_frames,
                initial_azimuth, initial_elevation,
                total_rotation_azimuth, total_rotation_elevation,
                from_pos=step.from_pos, to_pos=step.to_pos
            )
            frame_counter += EXPORT_FRAMES

            # Remove ghost if module reached original position
            if step.is_restoration_complete and step.module_id in ghost_positions:
                viz.plotter.remove_actor(f"ghost_{step.module_id}", reset_camera=False)
                print(f"    ✓ {step.module_id} restored!")

            # Pause between steps
            if idx < len(phase2_steps) - 1:
                for i in range(pause_frames):
                    # Use frame_counter + i to smoothly rotate camera during pause
                    global_t = (frame_counter + i) / max(1, total_animation_frames)
                    viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
                    viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation
                    viz.plotter.write_frame()
                frame_counter += pause_frames

        # Keep ghost markers visible at end (don't remove them)

    # Final pause - continue smooth camera rotation
    for i in range(15):
        global_t = min(1.0, (frame_counter + i) / max(1, total_animation_frames))
        viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
        viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation
        viz.plotter.write_frame()

    viz.plotter.close()
    print(f"\n✓ Saved to {output_path}")


def _export_restoration_pivot(viz, pivot_module, param1, param2, pivot_type, n_frames,
                               frame_offset=0, total_frames=None, initial_azimuth=0, initial_elevation=0,
                               total_rotation_azimuth=90.0, total_rotation_elevation=30.0,
                               from_pos=None, to_pos=None):
    """Helper to export a restoration pivot with green color."""
    if from_pos is None or to_pos is None:
        return

    initial_pos = from_pos.copy()
    final_pos = to_pos.copy()

    sphere = pv.Sphere(radius=viz.sphere_radius)

    if pivot_type == 'corner':
        axis_pos = viz.system.modules[param1].position.copy()
        v1 = initial_pos - axis_pos
        v2 = final_pos - axis_pos

        for i in range(n_frames + 1):
            t = i / n_frames
            angle = t * np.pi / 2
            interp_v = np.cos(angle) * v1 + np.sin(angle) * v2
            interp_v = interp_v / np.linalg.norm(interp_v) * np.linalg.norm(v1)
            interp_pos = axis_pos + interp_v

            viz.system.modules[pivot_module].position = interp_pos

            # Update camera
            if total_frames:
                global_t = (frame_offset + i) / total_frames
                viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
                viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation

            # Re-render static modules
            _render_all_static_modules(viz, {pivot_module})

            # Render moving module in green
            sphere_moved = sphere.copy()
            sphere_moved.points += interp_pos
            viz.plotter.add_mesh(
                sphere_moved,
                color=viz.colors['restoring'],  # Green!
                opacity=viz.sphere_opacity,
                name=f"module_{pivot_module}",
                reset_camera=False
            )
            viz.plotter.write_frame()

    else:  # lateral
        for i in range(n_frames + 1):
            t = i / n_frames
            t_smooth = t * t * (3 - 2 * t)
            interp_pos = (1 - t_smooth) * initial_pos + t_smooth * final_pos

            viz.system.modules[pivot_module].position = interp_pos

            if total_frames:
                global_t = (frame_offset + i) / total_frames
                viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
                viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation

            _render_all_static_modules(viz, {pivot_module})

            sphere_moved = sphere.copy()
            sphere_moved.points += interp_pos
            viz.plotter.add_mesh(
                sphere_moved,
                color=viz.colors['restoring'],
                opacity=viz.sphere_opacity,
                name=f"module_{pivot_module}",
                reset_camera=False
            )
            viz.plotter.write_frame()

    # Final frame with active color
    _render_all_static_modules(viz, {pivot_module})
    sphere_final = sphere.copy()
    sphere_final.points += final_pos
    viz.plotter.add_mesh(
        sphere_final,
        color=viz.colors['active'],
        opacity=viz.sphere_opacity,
        name=f"module_{pivot_module}",
        reset_camera=False
    )
    viz.plotter.write_frame()


def export_demo_gif(demo_name: str, output_path: str = None, rotate_camera: bool = True):
    """
    Export a demo animation as a GIF.

    Args:
        demo_name: Name of demo ('corner', 'lateral', 'spiral', 'parallel', 'general')
        output_path: Optional custom output path. Defaults to 'output/{demo_name}.gif'
        rotate_camera: If True, slowly rotate camera during animation. If False, keep fixed.
    """
    import os

    if output_path is None:
        os.makedirs('gifs', exist_ok=True)
        output_path = f'gifs/{demo_name}_pivot_demo.gif'

    print(f"=== Exporting {demo_name.upper()} Demo to GIF ===")
    print(f"Output: {output_path}")

    # Create system and get sequence based on demo type
    if demo_name == 'corner':
        system = create_l_shape_configuration()
        sequence = [
            ('corner', 'M4', 'M3', 'POS_Z'),
            ('corner', 'M4', 'M3', 'POS_Y'),
            ('corner', 'M4', 'M3', 'NEG_Z'),
            ('corner', 'M4', 'M3', 'POS_X'),
        ]
    elif demo_name == 'lateral':
        system = create_line_configuration(length=6, axis='X')
        system.add_module("M6", np.array([2, 0, 1]))
        system.connect_modules("M2", "M6")
        sequence = [
            ('lateral', 'M6', 'M2', 'M3'),
            ('lateral', 'M6', 'M3', 'M4'),
            ('lateral', 'M6', 'M4', 'M5'),
            ('lateral', 'M6', 'M5', 'M4'),
        ]
    elif demo_name == 'spiral':
        system = create_line_configuration(length=6, axis='X')
        sequence = [
            ('corner', 'M5', 'M4', 'POS_Z'),
            ('corner', 'M5', 'M4', 'POS_Y'),
            ('lateral', 'M5', 'M4', 'M3'),
            ('corner', 'M5', 'M3', 'NEG_Z'),
            ('lateral', 'M5', 'M3', 'M2'),
            ('corner', 'M5', 'M2', 'NEG_Y'),
            ('lateral', 'M5', 'M2', 'M1'),
            ('corner', 'M5', 'M1', 'POS_Z'),
            ('lateral', 'M5', 'M1', 'M0'),
            ('corner', 'M5', 'M0', 'NEG_X'),
        ]
    elif demo_name == 'parallel':
        system = create_star_configuration(size=3)
        sequence = [
            ('parallel', [
                ('corner', 'PX3', 'PX2', 'POS_Z'),
                ('corner', 'PY3', 'PY2', 'POS_Z'),
                ('corner', 'NX3', 'NX2', 'POS_Z'),
                ('corner', 'NY3', 'NY2', 'POS_Z'),
            ]),
            ('parallel', [
                ('corner', 'PX3', 'PX2', 'POS_Y'),
                ('corner', 'NX3', 'NX2', 'NEG_Y'),
            ]),
        ]
    elif demo_name == 'general':
        system = create_star_configuration(size=3)
        sequence = [
            ('corner', 'PX3', 'PX2', 'POS_Z'),
            ('corner', 'PY3', 'PY2', 'NEG_Z'),
            ('corner', 'NX3', 'NX2', 'POS_Y'),
            ('corner', 'NY3', 'NY2', 'POS_X'),
            ('corner', 'PZ3', 'PZ2', 'POS_X'),
            ('corner', 'NZ3', 'NZ2', 'NEG_X'),
            ('parallel', [
                ('corner', 'PX3', 'PX2', 'POS_Y'),
                ('corner', 'PY3', 'PY2', 'POS_X'),
            ]),
            ('parallel', [
                ('corner', 'NX3', 'NX2', 'POS_Z'),
                ('corner', 'NY3', 'NY2', 'POS_Z'),
                ('corner', 'PZ3', 'PZ2', 'NEG_Y'),
            ]),
            ('parallel', [
                ('corner', 'PX3', 'PX2', 'NEG_Z'),
                ('corner', 'PY3', 'PY2', 'NEG_Z'),
                ('corner', 'NZ3', 'NZ2', 'POS_Y'),
                ('corner', 'NX3', 'NX2', 'NEG_Y'),
            ]),
        ]
    elif demo_name == 'ego':
        # Ego fault response - run algorithm and convert steps to sequence
        system_for_steps = create_star_configuration(size=2)
        system_for_steps.mark_fault('C')
        stats = system_for_steps.ego_fault_response('C', record_steps=True, parallel_subgraphs=True)

        # Create fresh system for visualization
        system = create_star_configuration(size=2)
        system.mark_fault('C')

        # Convert parallel groups to sequence format with parallel wrappers
        if stats.get('parallel_steps'):
            sequence = []
            for group in stats['parallel_steps']:
                if len(group) == 1:
                    # Single step, no need for parallel wrapper
                    sequence.append(group[0].to_tuple())
                else:
                    # Multiple steps, wrap in parallel tuple
                    parallel_ops = [step.to_tuple() for step in group]
                    sequence.append(('parallel', parallel_ops))
            print(f"Ego algorithm: {stats['total_moves']} moves in {len(sequence)} groups (parallel)")
            print(f"  Collisions resolved: {stats['collisions_resolved']}")
        else:
            # Fallback to sequential
            sequence = [step.to_tuple() for step in stats['steps']]
            print(f"Ego algorithm: {len(sequence)} steps to export (sequential)")
    elif demo_name == 'ego-large':
        # Larger star fault response
        system_for_steps = create_star_configuration(size=3)
        system_for_steps.mark_fault('C')
        stats = system_for_steps.ego_fault_response('C', record_steps=True, parallel_subgraphs=True)

        system = create_star_configuration(size=3)
        system.mark_fault('C')

        if stats.get('parallel_steps'):
            sequence = []
            for group in stats['parallel_steps']:
                if len(group) == 1:
                    sequence.append(group[0].to_tuple())
                else:
                    parallel_ops = [step.to_tuple() for step in group]
                    sequence.append(('parallel', parallel_ops))
            print(f"Ego-large algorithm: {stats['total_moves']} moves in {len(sequence)} groups (parallel)")
            print(f"  Collisions resolved: {stats['collisions_resolved']}")
        else:
            sequence = [step.to_tuple() for step in stats['steps']]
            print(f"Ego-large algorithm: {len(sequence)} steps to export (sequential)")
    elif demo_name == 'ego-t':
        # T-shape fault response
        system_for_steps = create_t_shape_configuration(arm_length=3)
        system_for_steps.mark_fault('C')
        stats = system_for_steps.ego_fault_response('C', record_steps=True, parallel_subgraphs=True)

        system = create_t_shape_configuration(arm_length=3)
        system.mark_fault('C')

        if stats.get('parallel_steps'):
            sequence = []
            for group in stats['parallel_steps']:
                if len(group) == 1:
                    sequence.append(group[0].to_tuple())
                else:
                    parallel_ops = [step.to_tuple() for step in group]
                    sequence.append(('parallel', parallel_ops))
            print(f"Ego-T algorithm: {stats['total_moves']} moves in {len(sequence)} groups (parallel)")
            print(f"  Collisions resolved: {stats['collisions_resolved']}")
        else:
            sequence = [step.to_tuple() for step in stats['steps']]
            print(f"Ego-T algorithm: {len(sequence)} steps to export (sequential)")
    elif demo_name == 'ego-cross':
        # Cross fault response
        system_for_steps = create_cross_configuration(arm_length=2)
        system_for_steps.mark_fault('C')
        stats = system_for_steps.ego_fault_response('C', record_steps=True, parallel_subgraphs=True)

        system = create_cross_configuration(arm_length=2)
        system.mark_fault('C')

        if stats.get('parallel_steps'):
            sequence = []
            for group in stats['parallel_steps']:
                if len(group) == 1:
                    sequence.append(group[0].to_tuple())
                else:
                    parallel_ops = [step.to_tuple() for step in group]
                    sequence.append(('parallel', parallel_ops))
            print(f"Ego-cross algorithm: {stats['total_moves']} moves in {len(sequence)} groups (parallel)")
            print(f"  Collisions resolved: {stats['collisions_resolved']}")
        else:
            sequence = [step.to_tuple() for step in stats['steps']]
            print(f"Ego-cross algorithm: {len(sequence)} steps to export (sequential)")
    elif demo_name == 'ego-line':
        # 9-module line fault response
        system_for_steps = create_line_configuration(length=9, axis='X')
        system_for_steps.mark_fault('M4')
        stats = system_for_steps.ego_fault_response('M4', record_steps=True, parallel_subgraphs=True)

        system = create_line_configuration(length=9, axis='X')
        system.mark_fault('M4')

        if stats.get('parallel_steps'):
            sequence = []
            for group in stats['parallel_steps']:
                if len(group) == 1:
                    sequence.append(group[0].to_tuple())
                else:
                    parallel_ops = [step.to_tuple() for step in group]
                    sequence.append(('parallel', parallel_ops))
            print(f"Ego-line algorithm: {stats['total_moves']} moves in {len(sequence)} groups (parallel)")
            print(f"  Collisions resolved: {stats['collisions_resolved']}")
        else:
            sequence = [step.to_tuple() for step in stats['steps']]
            print(f"Ego-line algorithm: {len(sequence)} steps to export (sequential)")
    elif demo_name == 'ego-grid':
        # 3x3x3 grid fault response
        system_for_steps = create_grid_configuration(size=3)
        system_for_steps.mark_fault('M1_1_1')
        stats = system_for_steps.ego_fault_response('M1_1_1', record_steps=True, parallel_subgraphs=True)

        system = create_grid_configuration(size=3)
        system.mark_fault('M1_1_1')

        if stats.get('parallel_steps'):
            sequence = []
            for group in stats['parallel_steps']:
                if len(group) == 1:
                    sequence.append(group[0].to_tuple())
                else:
                    parallel_ops = [step.to_tuple() for step in group]
                    sequence.append(('parallel', parallel_ops))
            print(f"Ego-grid algorithm: {stats['total_moves']} moves in {len(sequence)} groups (parallel)")
            print(f"  Collisions resolved: {stats['collisions_resolved']}")
        else:
            sequence = [step.to_tuple() for step in stats['steps']]
            print(f"Ego-grid algorithm: {len(sequence)} steps to export (sequential)")
    elif demo_name == 'ego-ring':
        # Ring fault response
        system_for_steps = create_ring_configuration(radius=2)
        system_for_steps.mark_fault('T0')
        stats = system_for_steps.ego_fault_response('T0', record_steps=True, parallel_subgraphs=True)

        system = create_ring_configuration(radius=2)
        system.mark_fault('T0')

        if stats.get('parallel_steps') and stats['parallel_steps']:
            sequence = []
            for group in stats['parallel_steps']:
                if len(group) == 1:
                    sequence.append(group[0].to_tuple())
                else:
                    parallel_ops = [step.to_tuple() for step in group]
                    sequence.append(('parallel', parallel_ops))
            print(f"Ego-ring algorithm: {stats['total_moves']} moves in {len(sequence)} groups (parallel)")
            print(f"  Collisions resolved: {stats['collisions_resolved']}")
        elif stats.get('steps'):
            sequence = [step.to_tuple() for step in stats['steps']]
            print(f"Ego-ring algorithm: {len(sequence)} steps to export (sequential)")
        else:
            sequence = []
            print("Ego-ring algorithm: No moves needed - ring remained connected!")
    elif demo_name == 'ego-dual-star':
        # Dual star bridge fault response - fault center of bridge
        bridge_length = 9
        center_module = f'BR{bridge_length // 2 + 1}'  # BR5 for 9-module bridge

        system_for_steps = create_dual_star_bridge_configuration(star_size=3, bridge_length=bridge_length)
        system_for_steps.mark_fault(center_module)
        stats = system_for_steps.ego_fault_response(center_module, record_steps=True, parallel_subgraphs=True)

        # Create fresh system for visualization
        system = create_dual_star_bridge_configuration(star_size=3, bridge_length=bridge_length)
        system.mark_fault(center_module)

        # Keep the actual PivotStep objects instead of converting to tuples
        # This preserves from_pos and to_pos
        if stats.get('parallel_steps'):
            sequence = []
            for group in stats['parallel_steps']:
                if len(group) == 1:
                    sequence.append(group[0])  # Keep PivotStep object
                else:
                    sequence.append(('parallel', group))  # Keep PivotStep objects in list
            print(f"Ego-dual-star algorithm: {stats['total_moves']} moves in {len(sequence)} groups (parallel)")
            print(f"  Collisions resolved: {stats['collisions_resolved']}")
        else:
            sequence = stats['steps']  # Keep PivotStep objects
            print(f"Ego-dual-star algorithm: {len(sequence)} steps to export (sequential)")
    elif demo_name == 'full':
        # Full damage response (Phase 1 + Phase 2) - star size=2
        return export_full_damage_response_gif(
            create_star_configuration, {'size': 2}, 'C', output_path,
            rotate_camera=rotate_camera
        )
    elif demo_name == 'full-large':
        # Full damage response - large star size=3
        return export_full_damage_response_gif(
            create_star_configuration, {'size': 3}, 'C', output_path,
            rotate_camera=rotate_camera
        )
    elif demo_name == 'full-dual-star':
        # Full damage response - dual star bridge
        bridge_length = 9
        center_module = f'BR{bridge_length // 2 + 1}'
        return export_full_damage_response_gif(
            create_dual_star_bridge_configuration,
            {'star_size': 3, 'bridge_length': bridge_length},
            center_module,
            output_path,
            rotate_camera=rotate_camera
        )
    else:
        print(f"Unknown demo: {demo_name}")
        print("Available: corner, lateral, spiral, parallel, general, ego, ego-large, ego-t, ego-cross, ego-line, ego-grid, ego-ring, ego-dual-star, full, full-large, full-dual-star")
        return

    # Create visualizer and export
    viz = UDQDGVisualizer(system)
    viz.setup_scene()

    # Open GIF writer BEFORE rendering system
    viz.plotter.open_gif(output_path, fps=30)

    # Now render the system (including inactive/red BR2)
    viz.render_system()

    # IMPORTANT: Write the initial frame immediately to capture BR2 in red
    viz.plotter.write_frame()

    # Pause at beginning (0.5 seconds at 30fps = 15 frames)
    initial_pause_frames = 14  # Reduced by 1 since we already wrote one frame
    for _ in range(initial_pause_frames):
        viz.plotter.write_frame()

    # Animate the sequence frame by frame
    print(f"Rendering {len(sequence)} operations...")

    # Calculate total frames for camera rotation (including pauses)
    total_operations = 0
    for op in sequence:
        if hasattr(op, 'pivot_type'):
            # PivotStep object - single operation
            total_operations += 1
        elif op[0] == 'parallel':
            # Parallel tuple - multiple operations
            total_operations += len(op[1])
        else:
            # Regular tuple - single operation
            total_operations += 1
    pause_frames = int(PAUSE_BETWEEN * 30)
    total_animation_frames = total_operations * EXPORT_FRAMES + (len(sequence) - 1) * pause_frames
    frame_counter = 0

    # Initial camera position
    initial_azimuth = viz.plotter.camera.azimuth
    initial_elevation = viz.plotter.camera.elevation

    # Camera rotation totals
    total_rotation_azimuth = 90.0
    total_rotation_elevation = 30.0

    for idx, operation in enumerate(sequence):
        # Check if operation is a PivotStep object (has pivot_type attribute) or tuple
        if hasattr(operation, 'pivot_type'):
            # It's a PivotStep object - use recorded positions
            if operation.pivot_type == 'corner':
                _export_corner_pivot(viz, operation.module_id, operation.param1, operation.param2, EXPORT_FRAMES,
                                   frame_counter, total_animation_frames, initial_azimuth, initial_elevation,
                                   total_rotation_azimuth, total_rotation_elevation,
                                   from_pos=operation.from_pos, to_pos=operation.to_pos)
            elif operation.pivot_type == 'lateral':
                _export_lateral_pivot(viz, operation.module_id, operation.param1, operation.param2, EXPORT_FRAMES,
                                    frame_counter, total_animation_frames, initial_azimuth, initial_elevation,
                                    total_rotation_azimuth, total_rotation_elevation,
                                    from_pos=operation.from_pos, to_pos=operation.to_pos)
            frame_counter += EXPORT_FRAMES
        elif operation[0] == 'corner':
            # Legacy tuple format
            _, pivot_module, axis_module, new_direction = operation
            _export_corner_pivot(viz, pivot_module, axis_module, new_direction, EXPORT_FRAMES,
                               frame_counter, total_animation_frames, initial_azimuth, initial_elevation,
                               total_rotation_azimuth, total_rotation_elevation)
            frame_counter += EXPORT_FRAMES
        elif operation[0] == 'lateral':
            # Legacy tuple format
            _, pivot_module, old_neighbor, new_neighbor = operation
            _export_lateral_pivot(viz, pivot_module, old_neighbor, new_neighbor, EXPORT_FRAMES,
                                frame_counter, total_animation_frames, initial_azimuth, initial_elevation,
                                total_rotation_azimuth, total_rotation_elevation)
            frame_counter += EXPORT_FRAMES
        elif operation[0] == 'parallel':
            # Parallel operations - check if they're PivotSteps or tuples
            _, ops = operation
            # Convert PivotSteps to tuples with positions for parallel export
            if ops and hasattr(ops[0], 'pivot_type'):
                # PivotStep objects - need to handle specially
                _export_parallel_pivotsteps(viz, ops, EXPORT_FRAMES,
                                          frame_counter, total_animation_frames, initial_azimuth, initial_elevation,
                                          total_rotation_azimuth, total_rotation_elevation)
            else:
                # Legacy tuples
                _export_parallel_pivots(viz, ops, EXPORT_FRAMES,
                                      frame_counter, total_animation_frames, initial_azimuth, initial_elevation,
                                      total_rotation_azimuth, total_rotation_elevation)
            frame_counter += EXPORT_FRAMES

        # Add pause frames between operations (but not after the last one)
        # Continue camera rotation during pause
        if idx < len(sequence) - 1:
            for pause_i in range(pause_frames):
                # Update camera during pause
                global_t = (frame_counter + pause_i) / total_animation_frames
                viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
                viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation
                viz.plotter.write_frame()
            frame_counter += pause_frames

    # Pause at end (0.5 seconds at 30fps = 15 frames)
    end_pause_frames = 15
    for _ in range(end_pause_frames):
        viz.plotter.write_frame()

    # Close and save
    viz.plotter.close()
    print(f"✓ Saved to {output_path}")


def _render_all_static_modules(viz, exclude_modules=set()):
    """Re-render all modules except the ones being animated."""
    sphere = pv.Sphere(radius=viz.sphere_radius)
    for module_id, module in viz.system.modules.items():
        if module_id not in exclude_modules:
            sphere_at_pos = sphere.copy()
            sphere_at_pos.points += module.position
            color = viz.colors['active'] if module.is_active else viz.colors['inactive']
            viz.plotter.add_mesh(
                sphere_at_pos,
                color=color,
                opacity=viz.sphere_opacity,
                name=f"module_{module_id}",
                reset_camera=False
            )


def _export_corner_pivot(viz, pivot_module, axis_module, new_direction, n_frames,
                         frame_offset=0, total_frames=None, initial_azimuth=0, initial_elevation=0,
                         total_rotation_azimuth=90.0, total_rotation_elevation=30.0,
                         from_pos=None, to_pos=None):
    """Helper to export a corner pivot without interactive updates."""
    if from_pos is not None and to_pos is not None:
        # Use recorded positions (doesn't modify graph)
        initial_pos = from_pos.copy()
        final_pos = to_pos.copy()
        axis_pos = viz.system.modules[axis_module].position.copy()
    else:
        # Legacy mode: actually execute pivot
        initial_pos = viz.system.modules[pivot_module].position.copy()
        success = viz.system.corner_pivot(pivot_module, axis_module, new_direction)

        if not success:
            return

        final_pos = viz.system.modules[pivot_module].position.copy()
        axis_pos = viz.system.modules[axis_module].position.copy()
        viz.system.modules[pivot_module].position = initial_pos.copy()

    v1 = initial_pos - axis_pos
    v2 = final_pos - axis_pos
    sphere = pv.Sphere(radius=viz.sphere_radius)

    for i in range(n_frames + 1):
        t = i / n_frames
        angle = t * np.pi / 2
        interp_v = np.cos(angle) * v1 + np.sin(angle) * v2
        interp_v = interp_v / np.linalg.norm(interp_v) * np.linalg.norm(v1)
        interp_pos = axis_pos + interp_v
        viz.system.modules[pivot_module].position = interp_pos

        # Update camera position (diagonal rotation)
        if total_frames:
            global_t = (frame_offset + i) / total_frames
            viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
            viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation

        # Re-render all static modules first
        _render_all_static_modules(viz, {pivot_module})

        # Then render the moving module on top
        sphere_moved = sphere.copy()
        sphere_moved.points += interp_pos
        viz.plotter.add_mesh(sphere_moved, color=viz.colors['pivoting'], opacity=viz.sphere_opacity,
                            name=f"module_{pivot_module}", reset_camera=False)
        viz.plotter.write_frame()

    # Final frame with correct color - maintain camera rotation
    if total_frames:
        global_t = (frame_offset + n_frames + 1) / total_frames
        viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
        viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation

    _render_all_static_modules(viz, {pivot_module})
    sphere_final = sphere.copy()
    sphere_final.points += final_pos
    color = viz.colors['active'] if viz.system.modules[pivot_module].is_active else viz.colors['inactive']
    viz.plotter.add_mesh(sphere_final, color=color, opacity=viz.sphere_opacity,
                        name=f"module_{pivot_module}", reset_camera=False)
    viz.plotter.write_frame()


def _export_lateral_pivot(viz, pivot_module, old_neighbor, new_neighbor, n_frames,
                           frame_offset=0, total_frames=None, initial_azimuth=0, initial_elevation=0,
                           total_rotation_azimuth=90.0, total_rotation_elevation=30.0,
                           from_pos=None, to_pos=None):
    """Helper to export a lateral pivot without interactive updates."""
    old_neighbor_pos = viz.system.modules[old_neighbor].position.copy()
    new_neighbor_pos = viz.system.modules[new_neighbor].position.copy()

    if from_pos is not None and to_pos is not None:
        # Use recorded positions (doesn't modify graph)
        initial_pos = from_pos.copy()
        final_pos = to_pos.copy()
    else:
        # Legacy mode: actually execute pivot
        initial_pos = viz.system.modules[pivot_module].position.copy()
        success = viz.system.lateral_pivot(pivot_module, old_neighbor, new_neighbor)

        if not success:
            return

        final_pos = viz.system.modules[pivot_module].position.copy()
        viz.system.modules[pivot_module].position = initial_pos.copy()

    # Calculate arc midpoint (same logic as interactive visualizer)
    neighbor_midpoint = (old_neighbor_pos + new_neighbor_pos) / 2.0
    neighbor_distance = np.linalg.norm(new_neighbor_pos - old_neighbor_pos)
    r = np.linalg.norm(initial_pos - old_neighbor_pos)  # Actual radius from neighbor

    h_squared = r**2 - (neighbor_distance / 2.0)**2

    if h_squared < 0:
        mid_pos = (initial_pos + final_pos) / 2.0
    else:
        h = np.sqrt(h_squared)
        neighbor_direction = new_neighbor_pos - old_neighbor_pos
        neighbor_direction_norm = neighbor_direction / np.linalg.norm(neighbor_direction)
        v_to_initial = initial_pos - neighbor_midpoint
        perp_component = v_to_initial - np.dot(v_to_initial, neighbor_direction_norm) * neighbor_direction_norm

        if np.linalg.norm(perp_component) > 0.001:
            perp_direction = perp_component / np.linalg.norm(perp_component)
            mid_pos = neighbor_midpoint + h * perp_direction
        else:
            mid_pos = (initial_pos + final_pos) / 2.0

    sphere = pv.Sphere(radius=viz.sphere_radius)

    for i in range(n_frames + 1):
        t = i / n_frames

        if t <= 0.5:
            # First half: arc around old_neighbor
            local_t = t * 2
            v_start = initial_pos - old_neighbor_pos
            v_target = mid_pos - old_neighbor_pos

            cos_theta = np.dot(v_start, v_target) / (np.linalg.norm(v_start) * np.linalg.norm(v_target))
            cos_theta = np.clip(cos_theta, -1, 1)
            theta = np.arccos(cos_theta)

            if abs(theta) < 0.001:
                interp_v = v_start
            else:
                interp_v = (np.sin((1 - local_t) * theta) / np.sin(theta)) * v_start + \
                          (np.sin(local_t * theta) / np.sin(theta)) * v_target

            interp_pos = old_neighbor_pos + interp_v
        else:
            # Second half: arc around new_neighbor
            local_t = (t - 0.5) * 2
            v_start = mid_pos - new_neighbor_pos
            v_target = final_pos - new_neighbor_pos

            cos_theta = np.dot(v_start, v_target) / (np.linalg.norm(v_start) * np.linalg.norm(v_target))
            cos_theta = np.clip(cos_theta, -1, 1)
            theta = np.arccos(cos_theta)

            if abs(theta) < 0.001:
                interp_v = v_start
            else:
                interp_v = (np.sin((1 - local_t) * theta) / np.sin(theta)) * v_start + \
                          (np.sin(local_t * theta) / np.sin(theta)) * v_target

            interp_pos = new_neighbor_pos + interp_v

        viz.system.modules[pivot_module].position = interp_pos

        # Update camera position (diagonal rotation)
        if total_frames:
            global_t = (frame_offset + i) / total_frames
            viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
            viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation

        sphere_moved = sphere.copy()
        sphere_moved.points += interp_pos
        viz.plotter.add_mesh(sphere_moved, color=viz.colors['pivoting'], opacity=viz.sphere_opacity,
                            name=f"module_{pivot_module}", reset_camera=False)
        viz.plotter.write_frame()

    # Final frame with correct color
    sphere_final = sphere.copy()
    sphere_final.points += final_pos
    color = viz.colors['active'] if viz.system.modules[pivot_module].is_active else viz.colors['inactive']
    viz.plotter.add_mesh(sphere_final, color=color, opacity=viz.sphere_opacity,
                        name=f"module_{pivot_module}", reset_camera=False)
    viz.plotter.write_frame()


def _export_parallel_pivotsteps(viz, pivot_steps, n_frames,
                                frame_offset=0, total_frames=None, initial_azimuth=0, initial_elevation=0,
                                total_rotation_azimuth=90.0, total_rotation_elevation=30.0):
    """Export parallel pivots using PivotStep objects with recorded positions."""
    # Store initial positions and prepare data for each pivot
    pivot_data = []
    for step in pivot_steps:
        data = {
            'module_id': step.module_id,
            'pivot_type': step.pivot_type,
            'param1': step.param1,
            'param2': step.param2,
            'from_pos': step.from_pos.copy(),
            'to_pos': step.to_pos.copy(),
            'sphere': pv.Sphere(radius=viz.sphere_radius)
        }

        if step.pivot_type == 'corner':
            data['axis_pos'] = viz.system.modules[step.param1].position.copy()
            v1 = data['from_pos'] - data['axis_pos']
            v2 = data['to_pos'] - data['axis_pos']
            data['v1'] = v1
            data['v2'] = v2
        elif step.pivot_type == 'lateral':
            data['old_neighbor_pos'] = viz.system.modules[step.param1].position.copy()
            data['new_neighbor_pos'] = viz.system.modules[step.param2].position.copy()

        pivot_data.append(data)

    # Animate all pivots in parallel
    for i in range(n_frames + 1):
        t = i / n_frames

        # Update camera
        if total_frames:
            global_t = (frame_offset + i) / total_frames
            viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
            viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation

        # Update each pivoting module
        for data in pivot_data:
            if data['pivot_type'] == 'corner':
                # Corner pivot arc
                angle = t * np.pi / 2
                interp_v = np.cos(angle) * data['v1'] + np.sin(angle) * data['v2']
                interp_v = interp_v / np.linalg.norm(interp_v) * np.linalg.norm(data['v1'])
                interp_pos = data['axis_pos'] + interp_v
            elif data['pivot_type'] == 'lateral':
                # Simplified lateral path (linear interpolation for parallel export)
                interp_pos = (1 - t) * data['from_pos'] + t * data['to_pos']

            viz.system.modules[data['module_id']].position = interp_pos

            # Render sphere
            sphere_moved = data['sphere'].copy()
            sphere_moved.points += interp_pos
            viz.plotter.add_mesh(
                sphere_moved,
                color=viz.colors['pivoting'],
                opacity=viz.sphere_opacity,
                name=f"module_{data['module_id']}",
                reset_camera=False
            )

        viz.plotter.write_frame()

    # Final frame with correct colors - maintain camera rotation
    if total_frames:
        global_t = (frame_offset + n_frames + 1) / total_frames
        viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
        viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation

    for data in pivot_data:
        sphere_final = data['sphere'].copy()
        sphere_final.points += data['to_pos']
        color = viz.colors['active'] if viz.system.modules[data['module_id']].is_active else viz.colors['inactive']
        viz.plotter.add_mesh(
            sphere_final,
            color=color,
            opacity=viz.sphere_opacity,
            name=f"module_{data['module_id']}",
            reset_camera=False
        )
    viz.plotter.write_frame()


# ============================================
# PHASE 2: POSITION RESTORATION DEMOS
# ============================================

def demo_restoration():
    """
    Demonstrate Phase 2 position restoration after a fault response.
    Shows modules returning to their original positions with ghost markers.
    """
    print("=== Phase 2: Position Restoration Demo ===")
    print("Creating star configuration...")

    import time

    # Step 1: Run Phase 1 to get movement histories
    system = create_star_configuration(size=2)
    system.mark_fault('C')

    print(f"\nConfiguration: {len(system.modules)} modules")
    print("Fault location: Center (C)")

    print("\nRunning Phase 1 (ego_fault_response_with_history)...")
    phase1_stats, histories = system.ego_fault_response_with_history('C')

    if not phase1_stats.get('reconnected'):
        print("Phase 1 failed to reconnect - cannot demonstrate restoration")
        return

    print(f"Phase 1 complete:")
    print(f"  Iterations: {phase1_stats['iterations']}")
    print(f"  Total moves: {phase1_stats['total_moves']}")
    print(f"  Modules displaced: {phase1_stats['modules_displaced']}")

    # Step 2: Run Phase 2 restoration
    print("\nRunning Phase 2 (position_restoration)...")
    fault_positions = {(0, 0, 0)}  # Center position

    phase2_stats = system.position_restoration(
        histories,
        fault_positions,
        max_iterations=50,
        record_steps=True
    )

    print(f"Phase 2 complete:")
    print(f"  Restoration moves: {phase2_stats['restoration_moves']}")
    print(f"  Fully restored: {len(phase2_stats['fully_restored'])}")
    print(f"  Initial displacement: {phase2_stats['initial_displacement']:.2f}")
    print(f"  Final displacement: {phase2_stats['final_displacement']:.2f}")

    if not phase2_stats.get('steps'):
        print("\nNo restoration steps to animate (modules may not have moved)")
        return

    # Step 3: Create fresh system for visualization
    viz_system = create_star_configuration(size=2)
    viz_system.mark_fault('C')

    # Run Phase 1 again on visualization system to get it to post-Phase-1 state
    viz_phase1_stats, viz_histories = viz_system.ego_fault_response_with_history('C')

    viz = UDQDGVisualizer(viz_system)

    print("\nStarting restoration animation...")
    viz.show_window()

    time.sleep(0.5)

    # Animate restoration with ghost markers
    viz.animate_restoration(
        steps=phase2_stats['steps'],
        movement_histories=viz_histories,
        n_frames=TOTAL_FRAMES,
        pause_between=PAUSE_BETWEEN,
        camera_mode='fixed',
        show_ghosts=True,
        reset_positions=False  # Don't reset - we're already in post-Phase-1 state
    )

    print("\nRestoration animation complete!")
    time.sleep(1.0)
    viz.plotter.close()


def demo_full_damage_response():
    """
    Demonstrate complete damage response: Phase 1 + Phase 2.
    Shows the full cycle from fault detection to position restoration.
    """
    print("=== Full Damage Response Demo (Phase 1 + Phase 2) ===")
    print("Creating star configuration...")

    import time

    # Run full damage response
    system_for_data = create_star_configuration(size=2)
    result = system_for_data.full_damage_response('C', restore_positions=True, max_phase2_iterations=50)

    if not result.get('overall_success'):
        print(f"Damage response failed: {result.get('error', 'Unknown error')}")
        return

    print(f"\n{system_for_data.get_damage_response_summary(result)}")

    phase1 = result['phase1']
    phase2 = result['phase2']
    histories = result['movement_histories']

    # Check if we have animation data
    if not phase1.get('parallel_steps') and not phase1.get('steps'):
        print("No Phase 1 steps recorded")
        return

    # Create fresh system for visualization
    viz_system = create_star_configuration(size=2)
    viz_system.mark_fault('C')

    viz = UDQDGVisualizer(viz_system)

    print("\nStarting full damage response animation...")
    viz.show_window()

    time.sleep(0.5)

    # Animate full damage response
    phase1_steps = phase1.get('steps', [])
    parallel_groups = phase1.get('parallel_steps', None)
    phase2_steps = phase2.get('steps', []) if phase2 else []

    viz.animate_full_damage_response(
        phase1_steps=phase1_steps,
        phase2_steps=phase2_steps,
        movement_histories=histories,
        n_frames=TOTAL_FRAMES,
        pause_between=PAUSE_BETWEEN,
        pause_between_phases=2.0,
        camera_mode='fixed',
        parallel_groups=parallel_groups,
        show_ghosts=True
    )

    print("\nFull damage response animation complete!")
    time.sleep(1.0)
    viz.plotter.close()


def demo_full_large_star():
    """
    Demonstrate full damage response on a larger star (size=3).
    More modules = more dramatic restoration.
    """
    print("=== Full Damage Response - Large Star Demo ===")
    print("Creating large star configuration (size=3)...")

    import time

    # Run full damage response
    system_for_data = create_star_configuration(size=3)
    result = system_for_data.full_damage_response('C', restore_positions=True, max_phase2_iterations=100)

    if not result.get('overall_success'):
        print(f"Damage response failed: {result.get('error', 'Unknown error')}")
        return

    print(f"\n{system_for_data.get_damage_response_summary(result)}")

    phase1 = result['phase1']
    phase2 = result['phase2']
    histories = result['movement_histories']

    # Create fresh system for visualization
    viz_system = create_star_configuration(size=3)
    viz_system.mark_fault('C')

    viz = UDQDGVisualizer(viz_system)

    print("\nStarting animation...")
    viz.show_window()

    time.sleep(0.5)

    phase1_steps = phase1.get('steps', [])
    parallel_groups = phase1.get('parallel_steps', None)
    phase2_steps = phase2.get('steps', []) if phase2 else []

    viz.animate_full_damage_response(
        phase1_steps=phase1_steps,
        phase2_steps=phase2_steps,
        movement_histories=histories,
        n_frames=TOTAL_FRAMES,
        pause_between=PAUSE_BETWEEN,
        pause_between_phases=2.0,
        camera_mode='fixed',
        parallel_groups=parallel_groups,
        show_ghosts=True
    )

    time.sleep(1.0)
    viz.plotter.close()


def demo_full_dual_star():
    """
    Demonstrate full damage response on dual star bridge configuration.
    Bridge damage disconnects the two stars, then restoration moves modules back.
    """
    print("=== Full Damage Response - Dual Star Bridge Demo ===")
    print("Creating dual star bridge configuration...")

    import time

    bridge_length = 9
    center_module = f'BR{bridge_length // 2 + 1}'  # BR5 for 9-module bridge

    # Run full damage response
    system_for_data = create_dual_star_bridge_configuration(star_size=3, bridge_length=bridge_length)
    result = system_for_data.full_damage_response(
        center_module,
        restore_positions=True,
        max_phase2_iterations=100
    )

    if not result.get('overall_success'):
        print(f"Damage response failed: {result.get('error', 'Unknown error')}")
        return

    print(f"\n{system_for_data.get_damage_response_summary(result)}")

    phase1 = result['phase1']
    phase2 = result['phase2']
    histories = result['movement_histories']

    # Create fresh system for visualization
    viz_system = create_dual_star_bridge_configuration(star_size=3, bridge_length=bridge_length)
    viz_system.mark_fault(center_module)

    viz = UDQDGVisualizer(viz_system)

    print("\nStarting animation...")
    viz.show_window()

    time.sleep(0.5)

    phase1_steps = phase1.get('steps', [])
    parallel_groups = phase1.get('parallel_steps', None)
    phase2_steps = phase2.get('steps', []) if phase2 else []

    viz.animate_full_damage_response(
        phase1_steps=phase1_steps,
        phase2_steps=phase2_steps,
        movement_histories=histories,
        n_frames=TOTAL_FRAMES,
        pause_between=PAUSE_BETWEEN,
        pause_between_phases=2.0,
        camera_mode='fixed',
        parallel_groups=parallel_groups,
        show_ghosts=True
    )

    time.sleep(1.0)
    viz.plotter.close()


def _export_parallel_pivots(viz, pivot_operations, n_frames,
                             frame_offset=0, total_frames=None, initial_azimuth=0, initial_elevation=0,
                             total_rotation_azimuth=90.0, total_rotation_elevation=30.0):
    """Helper to export parallel pivots without interactive updates."""
    pivot_data = {}

    for op in pivot_operations:
        if op[0] == 'corner':
            _, pivot_module, axis_module, new_direction = op
            initial_pos = viz.system.modules[pivot_module].position.copy()
            axis_pos = viz.system.modules[axis_module].position.copy()
            viz.system.corner_pivot(pivot_module, axis_module, new_direction)
            final_pos = viz.system.modules[pivot_module].position.copy()

            pivot_data[pivot_module] = {
                'type': 'corner',
                'initial': initial_pos,
                'final': final_pos,
                'axis': axis_pos,
                'v1': initial_pos - axis_pos,
                'v2': final_pos - axis_pos
            }
        elif op[0] == 'lateral':
            _, pivot_module, old_neighbor, new_neighbor = op
            initial_pos = viz.system.modules[pivot_module].position.copy()
            old_neighbor_pos = viz.system.modules[old_neighbor].position.copy()
            new_neighbor_pos = viz.system.modules[new_neighbor].position.copy()

            viz.system.lateral_pivot(pivot_module, old_neighbor, new_neighbor)
            final_pos = viz.system.modules[pivot_module].position.copy()

            # Calculate arc midpoint
            neighbor_midpoint = (old_neighbor_pos + new_neighbor_pos) / 2.0
            neighbor_distance = np.linalg.norm(new_neighbor_pos - old_neighbor_pos)
            r = np.linalg.norm(initial_pos - old_neighbor_pos)  # Actual radius from neighbor

            h_squared = r**2 - (neighbor_distance / 2.0)**2

            if h_squared < 0:
                mid_pos = (initial_pos + final_pos) / 2.0
            else:
                h = np.sqrt(h_squared)
                neighbor_direction = new_neighbor_pos - old_neighbor_pos
                neighbor_direction_norm = neighbor_direction / np.linalg.norm(neighbor_direction)
                v_to_initial = initial_pos - neighbor_midpoint
                perp_component = v_to_initial - np.dot(v_to_initial, neighbor_direction_norm) * neighbor_direction_norm

                if np.linalg.norm(perp_component) > 0.001:
                    perp_direction = perp_component / np.linalg.norm(perp_component)
                    mid_pos = neighbor_midpoint + h * perp_direction
                else:
                    mid_pos = (initial_pos + final_pos) / 2.0

            pivot_data[pivot_module] = {
                'type': 'lateral',
                'initial': initial_pos,
                'final': final_pos,
                'mid_pos': mid_pos,
                'old_neighbor_pos': old_neighbor_pos,
                'new_neighbor_pos': new_neighbor_pos
            }

    sphere = pv.Sphere(radius=viz.sphere_radius)

    for i in range(n_frames + 1):
        t = i / n_frames

        for module_id, data in pivot_data.items():
            if data['type'] == 'corner':
                angle = t * np.pi / 2
                interp_v = np.cos(angle) * data['v1'] + np.sin(angle) * data['v2']
                interp_v = interp_v / np.linalg.norm(interp_v) * np.linalg.norm(data['v1'])
                interp_pos = data['axis'] + interp_v
            else:  # lateral
                if t <= 0.5:
                    # First half: arc around old_neighbor
                    local_t = t * 2
                    v_start = data['initial'] - data['old_neighbor_pos']
                    v_target = data['mid_pos'] - data['old_neighbor_pos']

                    cos_theta = np.dot(v_start, v_target) / (np.linalg.norm(v_start) * np.linalg.norm(v_target))
                    cos_theta = np.clip(cos_theta, -1, 1)
                    theta = np.arccos(cos_theta)

                    if abs(theta) < 0.001:
                        interp_v = v_start
                    else:
                        interp_v = (np.sin((1 - local_t) * theta) / np.sin(theta)) * v_start + \
                                  (np.sin(local_t * theta) / np.sin(theta)) * v_target

                    interp_pos = data['old_neighbor_pos'] + interp_v
                else:
                    # Second half: arc around new_neighbor
                    local_t = (t - 0.5) * 2
                    v_start = data['mid_pos'] - data['new_neighbor_pos']
                    v_target = data['final'] - data['new_neighbor_pos']

                    cos_theta = np.dot(v_start, v_target) / (np.linalg.norm(v_start) * np.linalg.norm(v_target))
                    cos_theta = np.clip(cos_theta, -1, 1)
                    theta = np.arccos(cos_theta)

                    if abs(theta) < 0.001:
                        interp_v = v_start
                    else:
                        interp_v = (np.sin((1 - local_t) * theta) / np.sin(theta)) * v_start + \
                                  (np.sin(local_t * theta) / np.sin(theta)) * v_target

                    interp_pos = data['new_neighbor_pos'] + interp_v

            viz.system.modules[module_id].position = interp_pos
            sphere_moved = sphere.copy()
            sphere_moved.points += interp_pos
            viz.plotter.add_mesh(sphere_moved, color=viz.colors['pivoting'], opacity=viz.sphere_opacity,
                                name=f"module_{module_id}", reset_camera=False)

        # Update camera position (diagonal rotation)
        if total_frames:
            global_t = (frame_offset + i) / total_frames
            viz.plotter.camera.azimuth = initial_azimuth + global_t * total_rotation_azimuth
            viz.plotter.camera.elevation = initial_elevation + global_t * total_rotation_elevation

        viz.plotter.write_frame()

    # Final frame
    for module_id, data in pivot_data.items():
        sphere_final = sphere.copy()
        sphere_final.points += data['final']
        viz.plotter.add_mesh(sphere_final, color=viz.colors['active'], opacity=viz.sphere_opacity,
                            name=f"module_{module_id}", reset_camera=False)
    viz.plotter.write_frame()


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == 'corner':
            demo_corner_pivots()
        elif command == 'lateral':
            demo_lateral_pivots()
        elif command == 'parallel':
            demo_parallel_pivots()
        elif command == 'spiral':
            demo_spiral_motion()
        elif command == 'sequence':
            demo_reconfiguration_sequence()
        elif command == 'damage':
            demo_damage_response()
        elif command == 'ego':
            demo_ego_fault_response()
        elif command == 'ego-large':
            demo_ego_large_star()
        elif command == 'ego-t':
            demo_ego_t_shape()
        elif command == 'ego-cross':
            demo_ego_cross()
        elif command == 'ego-line':
            demo_ego_line()
        elif command == 'ego-grid':
            demo_ego_grid()
        elif command == 'ego-ring':
            demo_ego_ring()
        elif command == 'ego-dual-star':
            demo_ego_dual_star_bridge()
        elif command == 'general':
            demo_general_sequence()
        # Phase 2 restoration demos
        elif command == 'restore':
            demo_restoration()
        elif command == 'full':
            demo_full_damage_response()
        elif command == 'full-large':
            demo_full_large_star()
        elif command == 'full-dual-star':
            demo_full_dual_star()
        elif command == 'export':
            # Export mode: python examples/visualize_pivots.py export <demo_name> [output_path] [--no-rotate]
            if len(sys.argv) > 2:
                demo_name = sys.argv[2].lower()
                remaining_args = sys.argv[3:]
                no_rotate = '--no-rotate' in remaining_args
                positional = [a for a in remaining_args if not a.startswith('--')]
                output_path = positional[0] if positional else None
                export_demo_gif(demo_name, output_path, rotate_camera=not no_rotate)
            else:
                print("Usage: python examples/visualize_pivots.py export <demo_name> [output_path] [--no-rotate]")
                print("Available demos: corner, lateral, spiral, parallel, general, ego, ego-large, ego-t, ego-cross, ego-line, ego-grid, ego-ring")
        else:
            # Interactive mode with specified configuration
            interactive_visualization(command)
    else:
        # Default: show all demos
        print("UDQDG System Visualization Demos")
        print("=" * 40)
        print("\nAvailable demos:")
        print("  python examples/visualize_pivots.py corner     - Corner pivot demo")
        print("  python examples/visualize_pivots.py lateral    - Lateral pivot demo")
        print("  python examples/visualize_pivots.py parallel   - Parallel pivots demo")
        print("  python examples/visualize_pivots.py spiral     - Spiral motion demo")
        print("  python examples/visualize_pivots.py general    - General sequence demo")
        print("  python examples/visualize_pivots.py sequence   - Reconfiguration sequence")
        print("  python examples/visualize_pivots.py damage     - Damage response demo")
        print("\nPhase 1: Ego fault response demos (parallel subgraph movement):")
        print("  python examples/visualize_pivots.py ego            - Star (size=2) fault response")
        print("  python examples/visualize_pivots.py ego-large      - Large star (size=3) fault response")
        print("  python examples/visualize_pivots.py ego-t          - T-shape fault response")
        print("  python examples/visualize_pivots.py ego-cross      - Cross fault response")
        print("  python examples/visualize_pivots.py ego-line       - 9-module line fault response")
        print("  python examples/visualize_pivots.py ego-grid       - 3x3x3 grid fault response")
        print("  python examples/visualize_pivots.py ego-ring       - Ring/loop fault response")
        print("  python examples/visualize_pivots.py ego-dual-star  - Dual star bridge fault response")
        print("\nPhase 2: Position restoration demos (modules return to original positions):")
        print("  python examples/visualize_pivots.py restore        - Phase 2 restoration only")
        print("  python examples/visualize_pivots.py full           - Full Phase 1 + Phase 2 (star)")
        print("  python examples/visualize_pivots.py full-large     - Full Phase 1 + Phase 2 (large star)")
        print("  python examples/visualize_pivots.py full-dual-star - Full Phase 1 + Phase 2 (dual star)")
        print("\nExport animations:")
        print("  python examples/visualize_pivots.py export <demo> [path]")
        print("    Available: corner, lateral, spiral, parallel, general, ego, ego-large, ego-t, ego-cross, ego-line, ego-grid, ego-ring, ego-dual-star")
        print("    Example: python examples/visualize_pivots.py export ego gifs/ego_response.gif")
        print("\nInteractive configurations:")
        print("  python examples/visualize_pivots.py star       - Star configuration")
        print("  python examples/visualize_pivots.py tree       - Tree configuration")
        print("  python examples/visualize_pivots.py line       - Line configuration")
        print("  python examples/visualize_pivots.py cross      - Cross configuration")
        print("  python examples/visualize_pivots.py l_shape    - L-shape configuration")
        print("\n" + "=" * 40)

        # Run default interactive demo
        interactive_visualization('star')


if __name__ == '__main__':
    main()
