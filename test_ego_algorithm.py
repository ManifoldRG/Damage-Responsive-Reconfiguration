#!/usr/bin/env python3
"""
Test script for ego-based fault response algorithm.

Demonstrates cellular migration-like behavior where modules independently
decide to move towards a fault using local information only.
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
    # Count modules in array
    module_count = np.sum(np.abs(array))
    print(f"Modules in array: {module_count}")
    print("=" * 40)

def test_simple_grid():
    """Test ego algorithm on a simple 3x3 grid."""
    print("\n" + "="*60)
    print("TEST: Simple 3x3 Grid with Center Fault")
    print("="*60)

    # Create 3x3 grid
    system = UDQDGSystem(mode_2d=True)
    system.create_grid(size=3)

    initial_count = len(system.modules)
    print(f"\nInitial system: {initial_count} modules, {len(system.get_all_edges())} edges")
    print_array(system.to_2d_array(), "Initial Configuration")

    # Mark center module as faulty
    center_module = "M11"  # Center of 3x3 grid
    system.mark_fault(center_module)
    print(f"\n[FAULT] Marked module {center_module} as faulty")
    print_array(system.to_2d_array(), "After Fault")

    # Run ego-based fault response
    print("\n[RUNNING] Ego-based fault response algorithm...")
    stats = system.ego_fault_response(center_module)

    # Print results
    final_count = len(system.modules)
    is_connected = system.is_connected(active_only=True)
    num_components = len(system.get_connected_components(active_only=True))

    print(f"\n[COMPLETE] Algorithm completed")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  Modules responded: {len(stats['modules_responded'])}")
    if stats['modules_responded']:
        print(f"  Which modules: {stats['modules_responded']}")
    print(f"  Module count: {initial_count} -> {final_count} ({'PRESERVED' if initial_count == final_count else 'LOST ' + str(initial_count - final_count)})")
    print(f"  Connectivity: {'CONNECTED' if is_connected else f'DISCONNECTED ({num_components} components)'}")

    print_array(system.to_2d_array(), "Final Configuration")

def test_edge_fault():
    """Test ego algorithm with fault at edge of grid."""
    print("\n" + "="*60)
    print("TEST: 4x4 Grid with Edge Fault")
    print("="*60)

    # Create 4x4 grid in 2D mode
    system = UDQDGSystem(mode_2d=True)
    system.create_grid(size=4)

    initial_count = len(system.modules)
    print(f"\nInitial system: {initial_count} modules, {len(system.get_all_edges())} edges")
    print_array(system.to_2d_array(), "Initial Configuration")

    # Mark edge module as faulty
    edge_module = "M01"  # Edge module
    system.mark_fault(edge_module)
    print(f"\n[FAULT] Marked module {edge_module} as faulty")
    print_array(system.to_2d_array(), "After Fault")

    # Run ego-based fault response
    print("\n[RUNNING] Ego-based fault response algorithm...")
    stats = system.ego_fault_response(edge_module)

    # Print results
    final_count = len(system.modules)
    print(f"\n[COMPLETE] Algorithm completed")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  Modules responded: {len(stats['modules_responded'])}")
    if stats['modules_responded']:
        print(f"  Which modules: {stats['modules_responded']}")
    print(f"  Module count: {initial_count} -> {final_count} ({'PRESERVED' if initial_count == final_count else 'LOST ' + str(initial_count - final_count)})")

    print_array(system.to_2d_array(), "Final Configuration")

def test_corner_fault():
    """Test ego algorithm with fault at corner of grid."""
    print("\n" + "="*60)
    print("TEST: 3x3 Grid with Corner Fault")
    print("="*60)

    # Create 3x3 grid
    system = UDQDGSystem(mode_2d=True)
    system.create_grid(size=3)

    initial_count = len(system.modules)
    print(f"\nInitial system: {initial_count} modules, {len(system.get_all_edges())} edges")
    print_array(system.to_2d_array(), "Initial Configuration")

    # Mark corner module as faulty
    corner_module = "M00"  # Top-left corner
    system.mark_fault(corner_module)
    print(f"\n[FAULT] Marked module {corner_module} as faulty")
    print_array(system.to_2d_array(), "After Fault")

    # Run ego-based fault response
    print("\n[RUNNING] Ego-based fault response algorithm...")
    stats = system.ego_fault_response(corner_module)

    # Print results
    final_count = len(system.modules)
    print(f"\n[COMPLETE] Algorithm completed")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  Modules responded: {len(stats['modules_responded'])}")
    if stats['modules_responded']:
        print(f"  Which modules: {stats['modules_responded']}")
    print(f"  Module count: {initial_count} -> {final_count} ({'PRESERVED' if initial_count == final_count else 'LOST ' + str(initial_count - final_count)})")

    print_array(system.to_2d_array(), "Final Configuration")

def test_ego_checks():
    """Test individual ego decision checks."""
    print("\n" + "="*60)
    print("TEST: Ego Decision Checks")
    print("="*60)

    # Create 3x3 grid
    system = UDQDGSystem(mode_2d=True)
    system.create_grid(size=3)

    print("\nTesting ego checks on various modules:")
    print("-" * 40)

    # Test corner (should be leaf)
    corner = "M00"
    print(f"\n{corner} (corner):")
    print(f"  Is leaf? {system.is_leaf_node(corner)}")
    print(f"  1-hop neighbors: {system.get_neighbors(corner)}")
    print(f"  3-hop neighbors: {len(system.get_k_hop_neighbors(corner, 3))} modules")

    # Test edge (not leaf, has 3 neighbors)
    edge = "M10"
    print(f"\n{edge} (edge):")
    print(f"  Is leaf? {system.is_leaf_node(edge)}")
    print(f"  1-hop neighbors: {system.get_neighbors(edge)}")
    print(f"  3-hop neighbors: {len(system.get_k_hop_neighbors(edge, 3))} modules")

    # Test center (has 4 neighbors)
    center = "M11"
    print(f"\n{center} (center):")
    print(f"  Is leaf? {system.is_leaf_node(center)}")
    print(f"  1-hop neighbors: {system.get_neighbors(center)}")
    print(f"  3-hop neighbors: {len(system.get_k_hop_neighbors(center, 3))} modules")

    # Mark center as fault and test response capability
    system.mark_fault(center)
    print(f"\n[FAULT] Marked {center} as faulty")
    print("\nChecking which modules can respond:")
    print("-" * 40)

    for module_id in sorted(system.modules.keys()):
        if module_id != center:
            can_respond = system.can_respond_to_fault(module_id)
            print(f"  {module_id}: {'[YES] Can respond' if can_respond else '[NO] Cannot respond'}")

def test_disconnected_fault():
    """Test with a fault that disconnects the graph."""
    print("\n" + "="*60)
    print("TEST: Disconnecting Fault")
    print("="*60)

    # Create a line of modules
    system = UDQDGSystem(mode_2d=True)

    # Create a 5-module line: M0-M1-M2-M3-M4
    for i in range(5):
        system.add_module(f"M{i}", np.array([i, 0, 0]))

    # Connect them in a line
    for i in range(4):
        system.connect_modules(f"M{i}", f"M{i+1}")

    initial_count = len(system.modules)
    initial_connected = system.is_connected(active_only=True)
    print(f"\nInitial system: {initial_count} modules")
    print(f"Initial connectivity: {'CONNECTED' if initial_connected else 'DISCONNECTED'}")
    print_array(system.to_2d_array(), "Initial Configuration")

    # Mark middle module as faulty (this will disconnect the line)
    middle_module = "M2"
    system.mark_fault(middle_module)
    after_fault_connected = system.is_connected(active_only=True)
    components = system.get_connected_components(active_only=True)

    print(f"\n[FAULT] Marked module {middle_module} as faulty")
    print(f"After fault connectivity: {'CONNECTED' if after_fault_connected else f'DISCONNECTED ({len(components)} components)'}")
    if not after_fault_connected:
        for i, comp in enumerate(components):
            print(f"  Component {i+1}: {sorted(comp)}")
    print_array(system.to_2d_array(), "After Fault")

    # Run ego-based fault response
    print("\n[RUNNING] Ego-based fault response algorithm...")
    stats = system.ego_fault_response(middle_module)

    # Print results
    final_count = len(system.modules)
    final_connected = system.is_connected(active_only=True)
    final_components = system.get_connected_components(active_only=True)

    print(f"\n[COMPLETE] Algorithm completed")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total moves: {stats['total_moves']}")
    print(f"  Modules responded: {len(stats['modules_responded'])}")
    if stats['modules_responded']:
        print(f"  Which modules: {stats['modules_responded']}")
    print(f"  Module count: {initial_count} -> {final_count} ({'PRESERVED' if initial_count == final_count else 'LOST ' + str(initial_count - final_count)})")
    print(f"  Connectivity: {'CONNECTED' if final_connected else f'DISCONNECTED ({len(final_components)} components)'}")
    print(f"  Reconnected: {'YES' if stats.get('reconnected', False) else 'NO'}")
    if not final_connected and not stats.get('reconnected', False):
        print(f"  [NOTE] Could not reconnect - no valid moves available")

    print_array(system.to_2d_array(), "Final Configuration")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EGO-BASED FAULT RESPONSE ALGORITHM TESTS")
    print("="*60)

    # Run all tests
    test_ego_checks()
    test_simple_grid()
    test_corner_fault()
    test_edge_fault()
    test_disconnected_fault()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60 + "\n")
