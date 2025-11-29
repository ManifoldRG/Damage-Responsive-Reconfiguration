# Greedy Ego-Based Fault Response Algorithm

**Version**: 1.0
**Last Updated**: 2025-11-29

---

## Overview

This document describes the greedy, decentralized fault response algorithm implemented in `src/udqdg_system.py`. The algorithm enables a modular robotic system to autonomously reconnect after a module failure using only local information (3-hop neighborhood).

### Key Properties

- **Decentralized**: Each module makes decisions based on local 3-hop information
- **Greedy**: Modules always move strictly closer to their target
- **Parallel**: Multiple subgraphs can move simultaneously with collision detection
- **Deterministic**: Tie-breaking rules ensure reproducible behavior

---

## Algorithm Structure

```
ego_fault_response(fault_id, max_iterations=1000):
    mark fault_id as faulty and inactive

    for iteration in 1..max_iterations:
        if system is connected:
            return SUCCESS

        components = get_connected_components()
        candidate_moves = []

        for each component:
            module = select_responding_module(component)
            if module has valid pivots:
                candidate_moves.append(module's best pivot)

        resolve_collisions(candidate_moves)
        execute_moves(candidate_moves)
        form_new_connections()

        if no moves made and no new connections:
            return STUCK

    return MAX_ITERATIONS_REACHED
```

---

## Phase 1: Module Eligibility (`can_respond_to_fault`)

A module can respond to a fault signal if ALL of the following conditions are met:

### Check 1: Leaf Node Exception
```
if module has exactly 1 neighbor:
    return CAN_MOVE  # Leaf nodes can always move
```

### Check 2: Pivot Availability
```
if no corner pivots AND no lateral pivots are mechanically possible:
    return CANNOT_MOVE
```

### Check 3: Connectivity Preservation
```
for each pair of neighbors (A, B):
    if A cannot reach B within 3 hops WITHOUT going through me:
        return CANNOT_MOVE  # I'm a critical bridge
return CAN_MOVE
```

**Rationale**: This check ensures that if a module moves, its neighbors remain connected to each other through alternative paths. The 3-hop limit models the local information available to each module.

---

## Phase 2: Module Selection (`select_responding_module`)

From each connected component, exactly ONE module is selected to move per iteration.

### Selection Criteria (in order of priority):

1. **Must pass eligibility** (`can_respond_to_fault` returns true)
2. **Must have valid pivot moves** toward the target
3. **Primary sort**: Distance to fault position (ascending)
   - *Rationale*: Models damage signal propagation - closer modules "hear" the distress call first
4. **Tiebreaker**: Module ID (lexicographic order)
   - *Rationale*: Ensures deterministic behavior

### Algorithm:
```python
candidates = [(distance_to_fault, module_id)
              for module in component
              if can_respond_to_fault(module)]

candidates.sort(key=lambda x: (x[0], x[1]))

for (dist, module_id) in candidates:
    if get_available_pivots(module_id, target) is not empty:
        return module_id

return None  # No valid responder in this component
```

---

## Phase 3: Target Position Calculation (`_get_target_position`)

The target position determines which direction a module should move.

### Rules:

1. **If system is disconnected** (multiple components):
   - Target = position of the closest module in ANY other component
   - *Rationale*: Move toward the nearest "island" to bridge the gap

2. **If system is connected** (single component):
   - Target = fault position
   - *Rationale*: Fill the hole left by the faulty module

---

## Phase 4: Pivot Selection (`get_available_pivots`)

For each eligible module, find all valid pivot moves that decrease distance to target.

### Corner Pivot Validation:
```
for each neighbor as axis_module:
    for each direction in {+X, -X, +Y, -Y, +Z, -Z}:
        new_position = axis_module.position + direction

        if direction is not orthogonal to current attachment:
            continue  # Corner pivots require 90-degree turns
        if new_position is occupied:
            continue  # Collision
        if new_position == target_position:
            continue  # Can't move onto fault (it's still there physically)
        if distance(new_position, target) >= distance(current, target):
            continue  # Must move STRICTLY closer (greedy)

        add to available_pivots
```

### Lateral Pivot Validation:
```
for each neighbor as old_neighbor:
    for each of old_neighbor's neighbors as new_neighbor:
        new_position = new_neighbor.position + same_offset_direction

        if new_position is occupied:
            continue
        if new_position == target_position:
            continue
        if distance(new_position, target) >= distance(current, target):
            continue

        add to available_pivots
```

### Greedy Constraint

**Critical**: Only moves that STRICTLY decrease distance to target are considered. This prevents oscillation but may cause the algorithm to get stuck in certain configurations (see Limitations).

---

## Phase 5: Collision Detection and Resolution

When multiple components each select a module to move, they may target the same destination position.

### Detection:
```python
position_groups = group_moves_by_target_position(candidate_moves)

for position, moves in position_groups:
    if len(moves) > 1:
        # COLLISION DETECTED
```

### Resolution (Tiebreaker):
```python
# Sort colliding modules by:
# 1. Distance to fault (ascending - closer wins)
# 2. Module ID (lexicographic - lower wins)
moves.sort(key=lambda m: (m.distance_to_fault, m.module_id))

winner = moves[0]
losers = moves[1:]  # Wait until next iteration
```

---

## Phase 6: Move Execution

After collision resolution, all winning moves execute simultaneously.

### Pivot Execution:
1. Disconnect ALL current neighbors (prevents phantom connections)
2. Update module position
3. Reconnect to axis module (for corner) or new neighbor (for lateral)

### Post-Move Connection Formation:
```python
for each pair of modules (A, B):
    if distance(A, B) == 1 AND not already connected:
        form_connection(A, B)
```

This allows modules from different components to reconnect when they become adjacent.

---

## Termination Conditions

The algorithm terminates when any of these conditions is met:

| Condition | Result |
|-----------|--------|
| All active modules form single connected component | **SUCCESS** |
| No moves possible AND no new connections formed | **STUCK** |
| Maximum iterations reached | **TIMEOUT** |

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 1000 | Maximum algorithm iterations |
| `one_per_subgraph` | True | Limit to one move per component per iteration |
| `parallel_subgraphs` | True | Execute moves from different components simultaneously |
| `record_steps` | False | Record each pivot for visualization/replay |

---

## Complexity Analysis

- **Per iteration**: O(V * E) for connectivity checks, O(V * D) for pivot enumeration
- **Typical iterations**: Proportional to maximum distance between components
- **Space**: O(V + E) for graph storage, O(V) for step recording

Where V = number of modules, E = number of connections, D = direction set size (6 for 3D).

---

## Known Limitations

### 1. Strictly Greedy Movement

The algorithm only accepts moves that strictly decrease distance to target. This can cause issues when:

- A "setup move" (temporarily moving away) would enable reconnection
- The optimal path requires non-monotonic distance changes

**Example that WORKS**: Line with middle fault - modules form a 3-module bridge above the fault.

### 2. Faulty Modules Occupy Space

Faulty modules remain physically present and cannot be moved through. This is physically accurate but limits some reconnection strategies.

### 3. No Global Planning

Each module only has 3-hop local information. There's no global coordination to find optimal solutions. The algorithm finds *a* solution, not necessarily the *optimal* solution.

### 4. Deterministic but Not Optimal

Tie-breaking rules ensure reproducibility but may not select the globally optimal responder.

---

## Configuration Results

| Configuration | Fault | Moves | Groups | Collisions | Result |
|--------------|-------|-------|--------|------------|--------|
| Star (size=2) | Center | 5 | 1 | 1 | Reconnects |
| Star (size=3) | Center | 11 | 4 | 9 | Reconnects |
| T-shape | Junction | 6 | 3 | 2 | Reconnects |
| Cross | Center | 4 | 1 | 1 | Reconnects |
| Line (9 modules) | Middle | 13 | 7 | 1 | Reconnects |
| Grid (3x3x3) | Center | 0 | 0 | 0 | Already connected |
| Ring | Corner | 0 | 0 | 0 | Already connected |

---

## Implementation Reference

| Component | Location |
|-----------|----------|
| Main algorithm | `src/udqdg_system.py:ego_fault_response()` |
| Eligibility check | `src/udqdg_system.py:can_respond_to_fault()` |
| Module selection | `src/udqdg_system.py:select_responding_module()` |
| Pivot enumeration | `src/udqdg_system.py:get_available_pivots()` |
| Target calculation | `src/udqdg_system.py:_get_target_position()` |
| Collision detection | `src/udqdg_system.py:ego_fault_response()` (parallel mode) |
| Connectivity check | `src/udqdg_system.py:_get_k_hop_neighbors_excluding()` |

---

## Visualization

Demo commands:
```bash
# Interactive demos
python examples/visualize_pivots.py ego        # Star (size=2)
python examples/visualize_pivots.py ego-large  # Star (size=3)
python examples/visualize_pivots.py ego-line   # Line (9 modules)
python examples/visualize_pivots.py ego-t      # T-shape
python examples/visualize_pivots.py ego-cross  # Cross

# GIF export
python examples/visualize_pivots.py export ego gifs/ego_response.gif
python examples/visualize_pivots.py export ego-line gifs/ego_line.gif
```

---

## References

- Problem formulation: `docs/problem_statement.md`
- Test cases: `test_reconnection.py`
