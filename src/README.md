# Damage-Responsive Reconfiguration Test Bench

A comprehensive test bench for evaluating algorithms that solve the damage-responsive reconfiguration problem on modular robot systems, implemented with NetworkX for robust graph analysis.

## Overview

This project implements a formal mathematical model of modular spacecraft systems and provides a flexible test bench for evaluating different reconfiguration algorithms. The system represents modular robots as graphs with lattice constraints and provides tools for simulating damage events and testing connectivity recovery algorithms.

### Key Features

- **Formal Graph Representation**: Implementation of the mathematical model G_t = (V_t, E_t, m_t, Q_t, g_t)
- **NetworkX Integration**: Leverages NetworkX for efficient graph analysis and connectivity algorithms
- **Cubic Lattice Constraints**: Enforces cubic lattice geometry for module connections
- **Algorithm Test Bench**: Pluggable framework for testing different reconfiguration algorithms
- **Comprehensive Analysis**: Built-in metrics, reporting, and visualization capabilities
- **Damage Simulation**: Realistic damage scenarios with connectivity impact analysis

## Problem Formulation

The system models a modular spacecraft as a graph:

```
𝒢_t = (V_t, E_t, m_t, Q_t, g_t)
```

Where:
- **V_t**: Set of modules (vertices)
- **E_t**: Set of physical connections (edges) 
- **m_t**: Activity map (active=1, inactive=0)
- **Q_t**: Absolute attitude (quaternions) per module
- **g_t**: Edge gains (unit vectors for displacement directions)

The active subgraph **𝒢^A_t** contains only active modules and their connections. The goal is to restore connectivity after damage events through valid pivot operations.

## Installation

### Requirements

- Python ≥3.13
- NetworkX ≥3.5
- NumPy
- Loguru

### Setup with uv (Recommended)

```bash
git clone <repository-url>
cd Damage-Responsive-Reconfiguration
uv sync
```

## Quick Start

### Basic System Usage

```python
from src.modular_system import ModularSystem
import numpy as np

# Create a modular system with cubic lattice
system = ModularSystem()

# Add modules
system.add_module("A", np.array([0, 0, 0]))
system.add_module("B", np.array([1, 0, 0]))
system.add_module("C", np.array([0, 1, 0]))

# Connect modules (must satisfy cubic lattice constraints)
system.connect_modules("A", "B")
system.connect_modules("A", "C")

# Check connectivity
print(f"Connected: {system.is_connected()}")
print(f"Components: {system.number_connected_components()}")

# Simulate damage
system.simulate_damage(["B"])
print(f"After damage: {system.is_connected()}")
```

### Test Bench Usage

```python
from src.test_bench import TestBench

# Create test bench
bench = TestBench()

# Add scenarios
bench.add_scenario(bench.create_simple_grid_scenario(3, True))
bench.add_scenario(bench.create_line_scenario(5, 2))

# Run benchmark
results = bench.run_benchmark_suite()

# Print results
bench.print_summary()
```

## Architecture

### Core Components

1. **ModularSystem** (`src/modular_system.py`)
   - Main system class implementing the formal graph model
   - NetworkX integration for connectivity analysis
   - Cubic lattice constraint validation
   - Damage simulation and state tracking

2. **TestBench** (`src/test_bench.py`)
   - Algorithm evaluation framework
   - Scenario generation and management
   - Performance metrics and reporting
   - Result analysis and visualization

3. **Module** (`src/module.py`)
   - Individual module representation
   - Connection port management
   - Message passing system
   - Activity state tracking

## Built-in Algorithm

### Dummy Algorithm
- **Strategy**: Reports system state without performing actual reconfiguration
- **Purpose**: Framework testing and as a template for custom implementations
- **Complexity**: O(1) per step
- **Best for**: Testing the framework and understanding the algorithm interface

## NetworkX Integration

The system leverages NetworkX for powerful graph analysis:

### Connectivity Analysis
```python
# Basic connectivity
system.is_connected()
system.number_connected_components()
system.get_connected_components()

# Advanced analysis
system.find_bridges()                    # Critical connections
system.find_articulation_points()       # Critical nodes
system.node_connectivity()              # Minimum node cut
system.edge_connectivity()              # Minimum edge cut
```

### Path Analysis
```python
# Shortest paths
path = system.shortest_path("A", "B")

# Position computation using BFS
positions = system.compute_positions(root="A")
```

## Running Examples

### Basic Usage (`examples/test_bench_example.py`)
Demonstrates the basic functionality of the damage-responsive reconfiguration system:
- Basic modular system creation and configuration
- Module connections with cubic lattice constraints
- Damage simulation and connectivity analysis
- NetworkX integration features
- Algorithm comparison using the test bench

```bash
uv run examples/test_bench_example.py
```

### Test Bench Demo (`src/test_bench.py`)
Runs the main test bench demonstration:
- Dummy algorithm testing
- Standard scenario testing
- Comprehensive reporting

```bash
uv run src/test_bench.py
```

## Repository Structure

```
├── src/                 # Core package modules
│   ├── __init__.py     # Package initialization
│   ├── module.py       # Individual module representation
│   ├── modular_system.py  # Main system class with NetworkX
│   └── test_bench.py   # Algorithm testing framework
├── examples/           # Usage examples and demos  
│   ├── basic_usage.py  # Basic functionality demo
│   └── advanced_testing.py # Advanced features demo
├── tests/              # Test suite (coming soon)
├── pyproject.toml      # Project configuration
└── README.md          # This file
```

## Extending the Framework

### Custom Algorithms

Implement the `ReconfigurationAlgorithm` protocol or extend `BaseReconfigurationAlgorithm`:

```python
from src.test_bench import BaseReconfigurationAlgorithm

class MyAlgorithm(BaseReconfigurationAlgorithm):
    def __init__(self):
        super().__init__("MyAlgorithm")
    
    def execute(self, system, max_steps=100):
        pivot_operations = []
        
        for step in range(max_steps):
            if system.is_connected():
                return True, pivot_operations, "Success"
            
            # Algorithm logic here...
            
        return False, pivot_operations, "Failed"
```

## License

This project is part of research into damage-responsive reconfiguration for modular robotic systems.

## Citation

If you use this work in research, please cite:

```bibtex
@software{damage_responsive_reconfig,
  title = {Damage-Responsive Reconfiguration Test Bench},
  year = {2025},
  note = {NetworkX-based implementation}
}
```
