#!/usr/bin/env python3

import sys
import numpy as np
from pyprojroot import here
from loguru import logger

sys.path.insert(0, str(here()))

from src.modular_system import ModularSystem
from src.test_bench import TestBench


def demonstrate_basic_system():
    """Demonstrate basic modular system functionality."""
    logger.info("=== Basic System Demonstration ===")
    
    # Create a modular system
    system = ModularSystem()
    
    # Add modules in a simple 2x2 grid
    positions = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    
    for i, (x, y, z) in enumerate(positions):
        module_id = f"M{i}"
        system.add_module(module_id, np.array([x, y, z], dtype=float))
    
    # Connect modules to form a 2x2 grid
    connections = [("M0", "M1"), ("M0", "M2"), ("M1", "M3"), ("M2", "M3")]
    
    for mod_a, mod_b in connections:
        system.connect_modules(mod_a, mod_b)
    
    # Test initial connectivity
    logger.info(f"Initial: Connected={system.is_connected()}, Components={system.number_connected_components()}")
    
    # Simulate damage to module M1
    logger.info("Simulating damage to module M1...")
    system.simulate_damage(["M1"])
    
    # Check connectivity after damage
    logger.info(f"After damage: Connected={system.is_connected()}, Components={system.number_connected_components()}")
    
    # Show components
    components = system.get_connected_components()
    for i, comp in enumerate(components):
        logger.info(f"Component {i}: {sorted(comp)}")
    
    return system


def demonstrate_algorithm_comparison():
    """Demonstrate algorithm comparison using the test bench."""
    logger.info("\n=== Algorithm Comparison Demonstration ===")
    
    # Create test bench
    bench = TestBench()
    
    # Add a custom algorithm
    class SimpleConnectivityAlgorithm:
        """Simple algorithm that just reports connectivity status."""
        
        def __init__(self):
            self.name = "SimpleConnectivity"
        
        def __call__(self, system, max_steps=100):
            """Check connectivity and return immediately."""
            pivot_operations = []
            
            if system.is_connected():
                return True, pivot_operations, "System already connected"
            
            # Try one "virtual" pivot operation
            pivotable = system.get_pivotable_modules()
            if pivotable:
                operation = {
                    'step': 0,
                    'pivot_module': pivotable[0],
                    'type': 'connectivity_check'
                }
                pivot_operations.append(operation)
            
            # Check connectivity again (in real implementation, this would change)
            success = system.is_connected()
            message = "Connectivity restored" if success else "Failed to restore connectivity"
            
            return success, pivot_operations, message
    
    # Register the custom algorithm
    bench.register_algorithm("simple", SimpleConnectivityAlgorithm())
    
    # Create test scenarios
    scenarios = [
        bench.create_simple_grid_scenario(3, True),   # 3x3 grid with center damage
        bench.create_line_scenario(5, 2),             # Linear chain with middle damage
        bench.create_t_shape_scenario(6, 1)           # T-shape with single damage
    ]
    
    for scenario in scenarios:
        bench.add_scenario(scenario)
    
    # Run benchmark comparing algorithms
    logger.info("Running benchmark suite...")
    results = bench.run_benchmark_suite(
        algorithms=["dummy", "simple"],
        scenarios=None  # Use all registered scenarios
    )
    
    # Print results
    bench.print_summary(results)
    
    # Save detailed report
    report_path = bench.save_report()
    logger.info(f"Detailed report saved to: {report_path}")
    
    return results


def demonstrate_networkx_integration():
    """Demonstrate NetworkX integration features."""
    logger.info("\n=== NetworkX Integration Demonstration ===")
    
    # Create a 3x3 grid system
    system = ModularSystem()
    
    # Create grid
    for x in range(3):
        for y in range(3):
            system.add_module(f"G{x}{y}", np.array([x, y, 0], dtype=float))
    
    # Connect grid
    for x in range(3):
        for y in range(3):
            current_id = f"G{x}{y}"
            if x < 2:  # Connect to right neighbor
                system.connect_modules(current_id, f"G{x+1}{y}")
            if y < 2:  # Connect to top neighbor
                system.connect_modules(current_id, f"G{x}{y+1}")
    
    logger.info("Created 3x3 grid system")
    
    # Show NetworkX analysis
    metrics = system.get_system_metrics()
    logger.info(f"Nodes: {metrics['active_modules']}, Edges: {metrics['active_connections']}")
    logger.info(f"Density: {metrics['density']:.3f}, Clustering: {metrics['average_clustering']:.3f}")
    
    # Test shortest path
    path = system.shortest_path("G00", "G22")
    if path:
        logger.info(f"Shortest path G00 to G22: {' -> '.join(path)}")
    
    # Simulate damage
    system.simulate_damage(["G11"])  # Damage center module
    logger.info(f"After center damage: Connected={system.is_connected()}, Components={system.number_connected_components()}")
    
    # Find critical elements
    bridges = system.find_bridges()
    articulation_points = system.find_articulation_points()
    logger.info(f"Bridges: {len(bridges)}, Articulation points: {len(articulation_points)}")
    
    return system


def main():
    """Main demonstration function."""
    logger.info("Starting Damage-Responsive Reconfiguration Demo")
    
    # Run demonstrations
    system1 = demonstrate_basic_system()
    results = demonstrate_algorithm_comparison()
    system2 = demonstrate_networkx_integration()
    
    logger.info("\n=== Demo Summary ===")
    logger.info(f"Basic system connected: {system1.is_connected()}")
    logger.info(f"Algorithm tests: {len(results)}")
    logger.info(f"NetworkX system connected: {system2.is_connected()}")
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()