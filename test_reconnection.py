#!/usr/bin/env python3
"""
Test script for reconnection scenarios.

Tests various configurations where faults break connectivity
and the algorithm should be able to restore it.
"""

import numpy as np
from src.udqdg_system import UDQDGSystem

def print_array(array, title="System State"):
    """Print 2D array with legend."""
    print(f"\n{title}")
    print("=" * 40)
    print("Legend: 0=empty, 1=active, -1=faulty")
    print("-" * 40)
    for row in array:
        print(" ".join(f"{val:2d}" for val in row))
    print("-" * 40)
    module_count = np.sum(np.abs(array))
    print(f"Modules in array: {module_count}")
    print("=" * 40)

def test_t_shape():
    """Test T-shape where center fault disconnects arms."""
    print("\n" + "="*60)
    print("TEST: T-Shape with Center Fault (Extended Arms)")
    print("="*60)

    system = UDQDGSystem(mode_2d=True)

    # Create extended T shape:
    #         M14
    #          |
    #         M13
    #          |
    #         M12
    #          |
    # M-2,1-M-1,1-M01-M11-M21-M31
    #          |
    #         M00
    #          |
    #         M0-1

    # Horizontal arm (left to right)
    system.add_module("M-2,1", np.array([-2, 1, 0]))
    system.add_module("M-1,1", np.array([-1, 1, 0]))
    system.add_module("M01", np.array([0, 1, 0]))  # Center
    system.add_module("M11", np.array([1, 1, 0]))
    system.add_module("M21", np.array([2, 1, 0]))
    system.add_module("M31", np.array([3, 1, 0]))

    # Vertical arm bottom
    system.add_module("M00", np.array([0, 0, 0]))
    system.add_module("M0-1", np.array([0, -1, 0]))

    # Vertical arm top
    system.add_module("M02", np.array([0, 2, 0]))
    system.add_module("M03", np.array([0, 3, 0]))
    system.add_module("M04", np.array([0, 4, 0]))

    # Connect horizontal arm
    system.connect_modules("M-2,1", "M-1,1")
    system.connect_modules("M-1,1", "M01")
    system.connect_modules("M01", "M11")
    system.connect_modules("M11", "M21")
    system.connect_modules("M21", "M31")

    # Connect vertical bottom
    system.connect_modules("M01", "M00")
    system.connect_modules("M00", "M0-1")

    # Connect vertical top
    system.connect_modules("M01", "M02")
    system.connect_modules("M02", "M03")
    system.connect_modules("M03", "M04")

    initial_count = len(system.modules)
    initial_connected = system.is_connected(active_only=True)
    print(f"\nInitial system: {initial_count} modules")
    print(f"Initial connectivity: {'CONNECTED' if initial_connected else 'DISCONNECTED'}")
    print_array(system.to_2d_array(), "Initial Configuration")

    # Mark center as faulty (disconnects all 4 arms)
    center = "M01"
    system.mark_fault(center)
    after_fault_connected = system.is_connected(active_only=True)
    components = system.get_connected_components(active_only=True)

    print(f"\n[FAULT] Marked module {center} as faulty")
    print(f"After fault connectivity: {'CONNECTED' if after_fault_connected else f'DISCONNECTED ({len(components)} components)'}")
    if not after_fault_connected:
        for i, comp in enumerate(components):
            print(f"  Component {i+1}: {sorted(comp)}")
    print_array(system.to_2d_array(), "After Fault")

    # Run ego-based fault response
    print("\n[RUNNING] Ego-based fault response algorithm...")
    stats = system.ego_fault_response(center)

    # Print results
    final_count = len(system.modules)
    final_connected = system.is_connected(active_only=True)
    final_components = system.get_connected_components(active_only=True)

    print(f"\n[COMPLETE] Algorithm completed")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  New connections formed: {stats.get('new_connections_formed', 0)}")
    print(f"  Modules responded: {len(stats['modules_responded'])}")
    if stats['modules_responded']:
        print(f"  Which modules: {stats['modules_responded']}")
    print(f"  Module count: {initial_count} -> {final_count} ({'PRESERVED' if initial_count == final_count else 'LOST ' + str(initial_count - final_count)})")
    print(f"  Connectivity: {'CONNECTED' if final_connected else f'DISCONNECTED ({len(final_components)} components)'}")
    print(f"  Reconnected: {'YES' if stats.get('reconnected', False) else 'NO'}")

    print_array(system.to_2d_array(), "Final Configuration")

def test_square_with_center_fault():
    """Test 3x3 square where center fault should allow reconnection."""
    print("\n" + "="*60)
    print("TEST: 3x3 Square with Center Fault")
    print("="*60)

    system = UDQDGSystem(mode_2d=True)
    system.create_grid(size=3)

    initial_count = len(system.modules)
    initial_connected = system.is_connected(active_only=True)
    print(f"\nInitial system: {initial_count} modules")
    print(f"Initial connectivity: {'CONNECTED' if initial_connected else 'DISCONNECTED'}")
    print_array(system.to_2d_array(), "Initial Configuration")

    # Mark center as faulty
    center = "M11"
    system.mark_fault(center)
    after_fault_connected = system.is_connected(active_only=True)

    print(f"\n[FAULT] Marked module {center} as faulty")
    print(f"After fault connectivity: {'CONNECTED' if after_fault_connected else 'DISCONNECTED'}")
    print_array(system.to_2d_array(), "After Fault")

    # Run ego-based fault response
    print("\n[RUNNING] Ego-based fault response algorithm...")
    stats = system.ego_fault_response(center)

    # Print results
    final_count = len(system.modules)
    final_connected = system.is_connected(active_only=True)

    print(f"\n[COMPLETE] Algorithm completed")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  Modules responded: {len(stats['modules_responded'])}")
    if stats['modules_responded']:
        print(f"  Which modules: {stats['modules_responded']}")
    print(f"  Module count: {initial_count} -> {final_count} ({'PRESERVED' if initial_count == final_count else 'LOST ' + str(initial_count - final_count)})")
    print(f"  Connectivity: {'CONNECTED' if final_connected else 'DISCONNECTED'}")
    print(f"  Reconnected: {'YES' if stats.get('reconnected', False) else 'NO'}")

    print_array(system.to_2d_array(), "Final Configuration")

def test_l_shape():
    """Test L-shape where corner fault disconnects."""
    print("\n" + "="*60)
    print("TEST: L-Shape with Corner Fault (Extended)")
    print("="*60)

    system = UDQDGSystem(mode_2d=True)

    # Create extended L shape:
    # M-2,3-M-1,3-M03-M13-M23-M33-M43
    #                  |
    #                 M22
    #                  |
    #                 M21
    #                  |
    #                 M20
    #                  |
    #                 M2-1

    # Horizontal arm (top)
    system.add_module("M-2,3", np.array([-2, 3, 0]))
    system.add_module("M-1,3", np.array([-1, 3, 0]))
    system.add_module("M03", np.array([0, 3, 0]))
    system.add_module("M13", np.array([1, 3, 0]))
    system.add_module("M23", np.array([2, 3, 0]))  # Corner
    system.add_module("M33", np.array([3, 3, 0]))
    system.add_module("M43", np.array([4, 3, 0]))

    # Vertical arm (going down)
    system.add_module("M22", np.array([2, 2, 0]))
    system.add_module("M21", np.array([2, 1, 0]))
    system.add_module("M20", np.array([2, 0, 0]))
    system.add_module("M2-1", np.array([2, -1, 0]))

    # Connect horizontal arm
    system.connect_modules("M-2,3", "M-1,3")
    system.connect_modules("M-1,3", "M03")
    system.connect_modules("M03", "M13")
    system.connect_modules("M13", "M23")
    system.connect_modules("M23", "M33")
    system.connect_modules("M33", "M43")

    # Connect vertical arm
    system.connect_modules("M23", "M22")
    system.connect_modules("M22", "M21")
    system.connect_modules("M21", "M20")
    system.connect_modules("M20", "M2-1")

    initial_count = len(system.modules)
    initial_connected = system.is_connected(active_only=True)
    print(f"\nInitial system: {initial_count} modules")
    print(f"Initial connectivity: {'CONNECTED' if initial_connected else 'DISCONNECTED'}")
    print_array(system.to_2d_array(), "Initial Configuration")

    # Mark corner as faulty (disconnects horizontal and vertical arms)
    corner = "M23"
    system.mark_fault(corner)
    after_fault_connected = system.is_connected(active_only=True)
    components = system.get_connected_components(active_only=True)

    print(f"\n[FAULT] Marked module {corner} as faulty")
    print(f"After fault connectivity: {'CONNECTED' if after_fault_connected else f'DISCONNECTED ({len(components)} components)'}")
    if not after_fault_connected:
        for i, comp in enumerate(components):
            print(f"  Component {i+1}: {sorted(comp)}")
    print_array(system.to_2d_array(), "After Fault")

    # Run ego-based fault response
    print("\n[RUNNING] Ego-based fault response algorithm...")
    stats = system.ego_fault_response(corner)

    # Print results
    final_count = len(system.modules)
    final_connected = system.is_connected(active_only=True)
    final_components = system.get_connected_components(active_only=True)

    print(f"\n[COMPLETE] Algorithm completed")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  New connections formed: {stats.get('new_connections_formed', 0)}")
    print(f"  Modules responded: {len(stats['modules_responded'])}")
    if stats['modules_responded']:
        print(f"  Which modules: {stats['modules_responded']}")
    print(f"  Module count: {initial_count} -> {final_count} ({'PRESERVED' if initial_count == final_count else 'LOST ' + str(initial_count - final_count)})")
    print(f"  Connectivity: {'CONNECTED' if final_connected else f'DISCONNECTED ({len(final_components)} components)'}")
    print(f"  Reconnected: {'YES' if stats.get('reconnected', False) else 'NO'}")

    print_array(system.to_2d_array(), "Final Configuration")

def test_star_shape():
    """Test star shape where center fault disconnects all arms."""
    print("\n" + "="*60)
    print("TEST: Star Shape with Center Fault (Extended)")
    print("="*60)

    system = UDQDGSystem(mode_2d=True)

    # Create extended star/plus shape:
    #           M15
    #            |
    #           M14
    #            |
    #           M13
    #            |
    #           M12
    #            |
    # M-3,1-M-2,1-M-1,1-M01-M11-M21-M31-M41
    #            |
    #           M00
    #            |
    #           M0-1
    #            |
    #           M0-2

    # Center
    system.add_module("M01", np.array([0, 1, 0]))  # Center

    # Left arm (west)
    system.add_module("M-1,1", np.array([-1, 1, 0]))
    system.add_module("M-2,1", np.array([-2, 1, 0]))
    system.add_module("M-3,1", np.array([-3, 1, 0]))

    # Right arm (east)
    system.add_module("M11", np.array([1, 1, 0]))
    system.add_module("M21", np.array([2, 1, 0]))
    system.add_module("M31", np.array([3, 1, 0]))
    system.add_module("M41", np.array([4, 1, 0]))

    # Top arm (north)
    system.add_module("M02", np.array([0, 2, 0]))
    system.add_module("M03", np.array([0, 3, 0]))
    system.add_module("M04", np.array([0, 4, 0]))
    system.add_module("M05", np.array([0, 5, 0]))

    # Bottom arm (south)
    system.add_module("M00", np.array([0, 0, 0]))
    system.add_module("M0-1", np.array([0, -1, 0]))
    system.add_module("M0-2", np.array([0, -2, 0]))

    # Connect left arm
    system.connect_modules("M01", "M-1,1")
    system.connect_modules("M-1,1", "M-2,1")
    system.connect_modules("M-2,1", "M-3,1")

    # Connect right arm
    system.connect_modules("M01", "M11")
    system.connect_modules("M11", "M21")
    system.connect_modules("M21", "M31")
    system.connect_modules("M31", "M41")

    # Connect top arm
    system.connect_modules("M01", "M02")
    system.connect_modules("M02", "M03")
    system.connect_modules("M03", "M04")
    system.connect_modules("M04", "M05")

    # Connect bottom arm
    system.connect_modules("M01", "M00")
    system.connect_modules("M00", "M0-1")
    system.connect_modules("M0-1", "M0-2")

    initial_count = len(system.modules)
    initial_connected = system.is_connected(active_only=True)
    print(f"\nInitial system: {initial_count} modules")
    print(f"Initial connectivity: {'CONNECTED' if initial_connected else 'DISCONNECTED'}")
    print_array(system.to_2d_array(), "Initial Configuration")

    # Mark center as faulty
    center = "M01"
    system.mark_fault(center)
    after_fault_connected = system.is_connected(active_only=True)
    components = system.get_connected_components(active_only=True)

    print(f"\n[FAULT] Marked module {center} as faulty")
    print(f"After fault connectivity: {'CONNECTED' if after_fault_connected else f'DISCONNECTED ({len(components)} components)'}")
    if not after_fault_connected:
        for i, comp in enumerate(components):
            print(f"  Component {i+1}: {sorted(comp)}")
    print_array(system.to_2d_array(), "After Fault")

    # Run ego-based fault response
    print("\n[RUNNING] Ego-based fault response algorithm...")
    stats = system.ego_fault_response(center)

    # Print results
    final_count = len(system.modules)
    final_connected = system.is_connected(active_only=True)
    final_components = system.get_connected_components(active_only=True)

    print(f"\n[COMPLETE] Algorithm completed")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  New connections formed: {stats.get('new_connections_formed', 0)}")
    print(f"  Modules responded: {len(stats['modules_responded'])}")
    if stats['modules_responded']:
        print(f"  Which modules: {stats['modules_responded']}")
    print(f"  Module count: {initial_count} -> {final_count} ({'PRESERVED' if initial_count == final_count else 'LOST ' + str(initial_count - final_count)})")
    print(f"  Connectivity: {'CONNECTED' if final_connected else f'DISCONNECTED ({len(final_components)} components)'}")
    print(f"  Reconnected: {'YES' if stats.get('reconnected', False) else 'NO'}")

    print_array(system.to_2d_array(), "Final Configuration")

def test_dense_grid():
    """Test dense grid with multiple fault tolerance."""
    print("\n" + "="*60)
    print("TEST: Dense 5x5 Grid with Center Fault")
    print("="*60)

    system = UDQDGSystem(mode_2d=True)
    system.create_grid(size=5)

    initial_count = len(system.modules)
    initial_connected = system.is_connected(active_only=True)
    print(f"\nInitial system: {initial_count} modules")
    print(f"Initial connectivity: {'CONNECTED' if initial_connected else 'DISCONNECTED'}")
    print_array(system.to_2d_array(), "Initial Configuration")

    # Mark center as faulty
    center = "M22"
    system.mark_fault(center)
    after_fault_connected = system.is_connected(active_only=True)
    components = system.get_connected_components(active_only=True)

    print(f"\n[FAULT] Marked module {center} as faulty")
    print(f"After fault connectivity: {'CONNECTED' if after_fault_connected else f'DISCONNECTED ({len(components)} components)'}")
    if not after_fault_connected:
        for i, comp in enumerate(components):
            print(f"  Component {i+1}: {len(comp)} modules")
    print_array(system.to_2d_array(), "After Fault")

    # Run ego-based fault response
    print("\n[RUNNING] Ego-based fault response algorithm...")
    stats = system.ego_fault_response(center)

    # Print results
    final_count = len(system.modules)
    final_connected = system.is_connected(active_only=True)
    final_components = system.get_connected_components(active_only=True)

    print(f"\n[COMPLETE] Algorithm completed")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  New connections formed: {stats.get('new_connections_formed', 0)}")
    print(f"  Modules responded: {len(stats['modules_responded'])}")
    if stats['modules_responded']:
        print(f"  Which modules: {stats['modules_responded']}")
    print(f"  Module count: {initial_count} -> {final_count} ({'PRESERVED' if initial_count == final_count else 'LOST ' + str(initial_count - final_count)})")
    print(f"  Connectivity: {'CONNECTED' if final_connected else f'DISCONNECTED ({len(final_components)} components)'}")
    print(f"  Reconnected: {'YES' if stats.get('reconnected', False) else 'NO'}")

    print_array(system.to_2d_array(), "Final Configuration")

def test_plus_with_mass():
    """Test plus shape with added mass around center."""
    print("\n" + "="*60)
    print("TEST: Plus Shape with Dense Center")
    print("="*60)

    system = UDQDGSystem(mode_2d=True)

    # Create a plus with 3x3 dense center
    # Center 3x3 block
    for x in range(-1, 2):
        for y in range(-1, 2):
            system.add_module(f"M{x},{y}", np.array([x, y, 0]))

    # Add arms extending from the center block
    # North arm
    for y in range(2, 5):
        system.add_module(f"M0,{y}", np.array([0, y, 0]))

    # South arm
    for y in range(-2, -5, -1):
        system.add_module(f"M0,{y}", np.array([0, y, 0]))

    # East arm
    for x in range(2, 5):
        system.add_module(f"M{x},0", np.array([x, 0, 0]))

    # West arm
    for x in range(-2, -5, -1):
        system.add_module(f"M{x},0", np.array([x, 0, 0]))

    # Connect center 3x3 grid
    for x in range(-1, 2):
        for y in range(-1, 2):
            current = f"M{x},{y}"
            if x < 1:
                right = f"M{x+1},{y}"
                if right in system.modules:
                    system.connect_modules(current, right)
            if y < 1:
                up = f"M{x},{y+1}"
                if up in system.modules:
                    system.connect_modules(current, up)

    # Connect north arm
    for y in range(1, 4):
        system.connect_modules(f"M0,{y}", f"M0,{y+1}")

    # Connect south arm
    for y in range(-1, -4, -1):
        system.connect_modules(f"M0,{y}", f"M0,{y-1}")

    # Connect east arm
    for x in range(1, 4):
        system.connect_modules(f"M{x},0", f"M{x+1},0")

    # Connect west arm
    for x in range(-1, -4, -1):
        system.connect_modules(f"M{x},0", f"M{x-1},0")

    initial_count = len(system.modules)
    initial_connected = system.is_connected(active_only=True)
    print(f"\nInitial system: {initial_count} modules")
    print(f"Initial connectivity: {'CONNECTED' if initial_connected else 'DISCONNECTED'}")
    print_array(system.to_2d_array(), "Initial Configuration")

    # Mark exact center as faulty
    center = "M0,0"
    system.mark_fault(center)
    after_fault_connected = system.is_connected(active_only=True)
    components = system.get_connected_components(active_only=True)

    print(f"\n[FAULT] Marked module {center} as faulty")
    print(f"After fault connectivity: {'CONNECTED' if after_fault_connected else f'DISCONNECTED ({len(components)} components)'}")
    if not after_fault_connected:
        for i, comp in enumerate(components):
            print(f"  Component {i+1}: {len(comp)} modules")
    print_array(system.to_2d_array(), "After Fault")

    # Run ego-based fault response
    print("\n[RUNNING] Ego-based fault response algorithm...")
    stats = system.ego_fault_response(center)

    # Print results
    final_count = len(system.modules)
    final_connected = system.is_connected(active_only=True)
    final_components = system.get_connected_components(active_only=True)

    print(f"\n[COMPLETE] Algorithm completed")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  New connections formed: {stats.get('new_connections_formed', 0)}")
    print(f"  Modules responded: {len(stats['modules_responded'])}")
    if stats['modules_responded']:
        print(f"  Which modules: {stats['modules_responded']}")
    print(f"  Module count: {initial_count} -> {final_count} ({'PRESERVED' if initial_count == final_count else 'LOST ' + str(initial_count - final_count)})")
    print(f"  Connectivity: {'CONNECTED' if final_connected else f'DISCONNECTED ({len(final_components)} components)'}")
    print(f"  Reconnected: {'YES' if stats.get('reconnected', False) else 'NO'}")

    print_array(system.to_2d_array(), "Final Configuration")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("RECONNECTION ALGORITHM TESTS")
    print("="*60)

    # Run tests
    test_square_with_center_fault()
    test_dense_grid()
    test_plus_with_mass()
    test_t_shape()
    test_l_shape()
    test_star_shape()

    print("\n" + "="*60)
    print("ALL RECONNECTION TESTS COMPLETED")
    print("="*60 + "\n")
