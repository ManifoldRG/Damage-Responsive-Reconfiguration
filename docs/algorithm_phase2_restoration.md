# Phase 2: Position Restoration Algorithm

## Overview

After the **Phase 1 ego-based fault response** successfully reconnects disconnected modules around a damaged node, **Phase 2** focuses on **restoring modules to their original positions** (or as close as possible) while maintaining system connectivity.

The key insight is that the UDQDG framework's vector-based translations are additive—by tracking cumulative displacement vectors, modules retain knowledge of their original positions and can attempt to return.

---

## Problem Statement

### Given
- A connected modular system that has undergone Phase 1 reconnection
- Modules that have moved from their original positions during coagulation
- A record of cumulative displacement vectors for each moved module

### Goal
Restore modules toward their original positions such that:
1. **Connectivity is preserved** at all times
2. **Modules return to original positions** where possible
3. **Minimize total displacement** from original configuration
4. **Maintain fault isolation** (don't occupy the faulty module's position)

### Constraints
- Same pivot constraints as Phase 1 (corner/lateral pivots only)
- Same connectivity preservation (3-hop reachability)
- Same leaf node eligibility rules
- Original position may be blocked → find closest available position

---

## Mathematical Foundation

### Cumulative Displacement Vector

For each module $m$, track:

$$\vec{d}_m = \sum_{i=1}^{n} \vec{\delta}_i$$

Where:
- $\vec{\delta}_i$ is the translation vector for move $i$
- $n$ is the total number of moves module $m$ has made
- $\vec{d}_m$ is the cumulative displacement from original position

### Original Position Recovery

$$\vec{p}_{original} = \vec{p}_{current} - \vec{d}_m$$

### Restoration Target Priority

For module $m$ seeking to restore:

1. **Primary target**: $\vec{p}_{original}$ (if unoccupied and not fault position)
2. **Secondary targets**: Adjacent lattice points to $\vec{p}_{original}$
3. **Tertiary targets**: 2-hop lattice points from $\vec{p}_{original}$

---

## Data Structures

### MovementHistory

```python
@dataclass
class MovementHistory:
    """Tracks cumulative movement for a module"""
    module_id: str
    original_position: np.ndarray      # Position before any Phase 1 moves
    cumulative_displacement: np.ndarray # Sum of all move vectors
    move_count: int                      # Number of moves made
    move_sequence: List[np.ndarray]      # Individual move vectors (optional)

    @property
    def current_position(self) -> np.ndarray:
        """Calculate current position from original + displacement"""
        return self.original_position + self.cumulative_displacement

    def add_move(self, delta: np.ndarray) -> None:
        """Record a new move"""
        self.cumulative_displacement += delta
        self.move_count += 1
        self.move_sequence.append(delta.copy())

    def get_original_position(self) -> np.ndarray:
        """Return the original position before all moves"""
        return self.original_position.copy()

    def get_restoration_distance(self, current_pos: np.ndarray) -> float:
        """Distance from current position to original position"""
        return np.linalg.norm(current_pos - self.original_position)
```

### RestorationStep

```python
@dataclass
class RestorationStep:
    """Records a restoration move"""
    iteration: int
    pivot_type: str                     # 'corner' or 'lateral'
    module_id: str
    param1: str                         # axis_module or old_neighbor
    param2: str                         # new_direction or new_neighbor
    from_pos: np.ndarray
    to_pos: np.ndarray
    distance_before: float              # Distance to original before move
    distance_after: float               # Distance to original after move
    is_restoration_complete: bool       # True if reached original position
```

---

## Algorithm Design

### Phase 2 Entry Conditions

Phase 2 activates when:
1. Phase 1 completes successfully (`stats['reconnected'] == True`)
2. At least one module has non-zero cumulative displacement
3. System is in stable, connected state

### Main Algorithm: `position_restoration()`

```python
def position_restoration(
    self,
    movement_histories: Dict[str, MovementHistory],
    fault_positions: Set[Tuple[int, int, int]],
    max_iterations: int = 100,
    record_steps: bool = True
) -> Dict[str, Any]:
    """
    Phase 2: Restore modules toward original positions after reconnection.

    Args:
        movement_histories: Cumulative displacement data from Phase 1
        fault_positions: Positions to avoid (damaged module locations)
        max_iterations: Maximum restoration iterations
        record_steps: Whether to record step-by-step history

    Returns:
        Dictionary with restoration statistics and steps
    """
```

### Restoration Eligibility Check

A module can attempt restoration if:

```python
def can_restore_position(self, module_id: str, fault_positions: Set) -> bool:
    """
    Check if module can attempt position restoration.
    Uses same eligibility checks as Phase 1 + restoration-specific checks.
    """
    # Check 1: Must have displacement (actually moved in Phase 1)
    if module_id not in self.movement_histories:
        return False
    if np.allclose(self.movement_histories[module_id].cumulative_displacement, 0):
        return False

    # Check 2: Standard eligibility (leaf node exception)
    neighbors = self.get_neighbors(module_id)
    if len(neighbors) == 1:
        return True  # Leaf nodes can always try to restore

    # Check 3: Has valid pivot options toward original position
    if not self._has_restoration_pivot(module_id):
        return False

    # Check 4: 3-hop connectivity preservation
    # (Same as Phase 1 - neighbors must remain connected)
    if not self._preserves_connectivity(module_id):
        return False

    return True
```

### Restoration Pivot Selection

```python
def get_restoration_pivots(
    self,
    module_id: str,
    fault_positions: Set[Tuple]
) -> List[Dict]:
    """
    Get valid pivots that move module toward original position.

    Returns list of moves sorted by:
    1. Distance reduction to original position (primary)
    2. Pivot type preference (corner > lateral for larger moves)
    """
    history = self.movement_histories[module_id]
    original_pos = history.get_original_position()
    current_pos = self.modules[module_id].position
    current_distance = np.linalg.norm(current_pos - original_pos)

    valid_pivots = []

    # Enumerate corner pivots
    for neighbor in self.get_neighbors(module_id):
        for direction_name, direction in self.directions.items():
            new_pos = self.modules[neighbor].position + np.array(direction)

            if self._is_valid_restoration_pivot(
                module_id, neighbor, direction, new_pos,
                original_pos, fault_positions
            ):
                new_distance = np.linalg.norm(new_pos - original_pos)
                if new_distance < current_distance:  # Greedy toward original
                    valid_pivots.append({
                        'type': 'corner',
                        'axis': neighbor,
                        'direction': direction_name,
                        'new_pos': new_pos,
                        'distance_reduction': current_distance - new_distance,
                        'reaches_original': np.allclose(new_pos, original_pos)
                    })

    # Enumerate lateral pivots
    for old_neighbor in self.get_neighbors(module_id):
        for new_neighbor in self.get_neighbors(old_neighbor):
            if new_neighbor == module_id:
                continue
            # Calculate lateral pivot destination
            # ... (similar logic)

    # Sort by distance reduction (descending), prefer reaching original
    valid_pivots.sort(
        key=lambda p: (-p['reaches_original'], -p['distance_reduction'])
    )

    return valid_pivots
```

### Module Selection for Restoration

```python
def select_restoration_module(
    self,
    candidates: Set[str],
    fault_positions: Set[Tuple]
) -> Optional[str]:
    """
    Select which module should attempt restoration.

    Priority:
    1. Modules that can reach their exact original position
    2. Modules with largest displacement (moved furthest)
    3. Tiebreaker: module ID
    """
    scored_candidates = []

    for module_id in candidates:
        if not self.can_restore_position(module_id, fault_positions):
            continue

        history = self.movement_histories[module_id]
        displacement = np.linalg.norm(history.cumulative_displacement)

        pivots = self.get_restoration_pivots(module_id, fault_positions)
        if not pivots:
            continue

        can_reach_original = any(p['reaches_original'] for p in pivots)

        scored_candidates.append({
            'module_id': module_id,
            'can_reach_original': can_reach_original,
            'displacement': displacement,
            'best_pivot': pivots[0]
        })

    if not scored_candidates:
        return None

    # Sort: prefer reaching original, then by displacement, then ID
    scored_candidates.sort(
        key=lambda c: (
            -c['can_reach_original'],
            -c['displacement'],
            c['module_id']
        )
    )

    return scored_candidates[0]['module_id']
```

### Main Restoration Loop

```python
def position_restoration(self, ...) -> Dict[str, Any]:
    stats = {
        'iterations': 0,
        'restoration_moves': 0,
        'fully_restored': [],      # Modules that reached original position
        'partially_restored': [],  # Modules that moved closer but not complete
        'could_not_restore': [],   # Modules that couldn't move at all
        'total_displacement_before': 0.0,
        'total_displacement_after': 0.0,
        'steps': [] if record_steps else None
    }

    # Calculate initial total displacement
    for module_id, history in movement_histories.items():
        current_pos = self.modules[module_id].position
        stats['total_displacement_before'] += history.get_restoration_distance(current_pos)

    for iteration in range(max_iterations):
        stats['iterations'] = iteration + 1

        # Get all modules that can attempt restoration
        candidates = {
            m_id for m_id in movement_histories.keys()
            if self.can_restore_position(m_id, fault_positions)
        }

        if not candidates:
            break  # No more restoration possible

        # Select module and perform restoration move
        selected = self.select_restoration_module(candidates, fault_positions)

        if not selected:
            break

        # Execute the best restoration pivot
        history = movement_histories[selected]
        pivots = self.get_restoration_pivots(selected, fault_positions)
        best_pivot = pivots[0]

        from_pos = self.modules[selected].position.copy()

        # Perform pivot
        if best_pivot['type'] == 'corner':
            success = self.corner_pivot(
                selected,
                best_pivot['axis'],
                best_pivot['direction']
            )
        else:
            success = self.lateral_pivot(
                selected,
                best_pivot['old_neighbor'],
                best_pivot['new_neighbor']
            )

        if success:
            to_pos = self.modules[selected].position.copy()
            delta = to_pos - from_pos

            # Update movement history (displacement vector changes)
            history.add_move(delta)

            stats['restoration_moves'] += 1

            if best_pivot['reaches_original']:
                stats['fully_restored'].append(selected)

            # Record step
            if record_steps:
                stats['steps'].append(RestorationStep(
                    iteration=iteration,
                    pivot_type=best_pivot['type'],
                    module_id=selected,
                    param1=best_pivot.get('axis') or best_pivot.get('old_neighbor'),
                    param2=best_pivot.get('direction') or best_pivot.get('new_neighbor'),
                    from_pos=from_pos,
                    to_pos=to_pos,
                    distance_before=np.linalg.norm(from_pos - history.original_position),
                    distance_after=np.linalg.norm(to_pos - history.original_position),
                    is_restoration_complete=best_pivot['reaches_original']
                ))

        # Check termination: all displaced modules restored
        if all(
            np.allclose(
                self.modules[m_id].position,
                movement_histories[m_id].original_position
            )
            for m_id in movement_histories.keys()
        ):
            break

    # Calculate final displacement
    for module_id, history in movement_histories.items():
        current_pos = self.modules[module_id].position
        distance = history.get_restoration_distance(current_pos)
        stats['total_displacement_after'] += distance

        if module_id not in stats['fully_restored']:
            if distance < np.linalg.norm(history.cumulative_displacement):
                stats['partially_restored'].append(module_id)
            else:
                stats['could_not_restore'].append(module_id)

    stats['displacement_reduction'] = (
        stats['total_displacement_before'] - stats['total_displacement_after']
    )
    stats['success'] = len(stats['could_not_restore']) == 0

    return stats
```

---

## Integration with Phase 1

### Modified Phase 1 to Track History

```python
def ego_fault_response_with_history(
    self,
    fault_module_id: str,
    **kwargs
) -> Tuple[Dict[str, Any], Dict[str, MovementHistory]]:
    """
    Extended ego_fault_response that returns movement histories.
    """
    # Initialize movement histories for all modules
    movement_histories = {}
    for module_id, module in self.modules.items():
        if module.is_active and not module.is_faulty:
            movement_histories[module_id] = MovementHistory(
                module_id=module_id,
                original_position=module.position.copy(),
                cumulative_displacement=np.zeros(3),
                move_count=0,
                move_sequence=[]
            )

    # Run Phase 1 with move tracking callback
    def on_move(module_id: str, from_pos: np.ndarray, to_pos: np.ndarray):
        delta = to_pos - from_pos
        movement_histories[module_id].add_move(delta)

    stats = self._ego_fault_response_internal(
        fault_module_id,
        on_move_callback=on_move,
        **kwargs
    )

    return stats, movement_histories
```

### Combined Execution

```python
def full_damage_response(
    self,
    fault_module_id: str,
    restore_positions: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Complete damage response: Phase 1 (reconnect) + Phase 2 (restore).
    """
    # Phase 1: Reconnection
    phase1_stats, movement_histories = self.ego_fault_response_with_history(
        fault_module_id,
        **kwargs
    )

    result = {
        'phase1': phase1_stats,
        'phase2': None,
        'overall_success': phase1_stats['reconnected']
    }

    if not phase1_stats['reconnected']:
        return result  # Phase 1 failed, can't proceed

    if not restore_positions:
        return result  # User opted out of restoration

    # Phase 2: Position Restoration
    fault_pos = tuple(self.modules[fault_module_id].position.astype(int))

    phase2_stats = self.position_restoration(
        movement_histories=movement_histories,
        fault_positions={fault_pos},
        **kwargs
    )

    result['phase2'] = phase2_stats
    result['overall_success'] = (
        phase1_stats['reconnected'] and
        phase2_stats['success']
    )

    return result
```

---

## Edge Cases and Special Handling

### Case 1: Original Position Occupied

When a module's original position is now occupied by another module:

```python
def find_closest_available_position(
    self,
    target_pos: np.ndarray,
    fault_positions: Set[Tuple],
    max_distance: int = 3
) -> Optional[np.ndarray]:
    """
    Find closest unoccupied lattice position to target.
    Uses BFS from target position.
    """
    occupied = self._get_all_occupied_positions()

    for distance in range(1, max_distance + 1):
        # Check all lattice points at this distance
        for pos in self._get_lattice_points_at_distance(target_pos, distance):
            pos_tuple = tuple(pos.astype(int))
            if pos_tuple not in occupied and pos_tuple not in fault_positions:
                return pos

    return None  # No available position found
```

### Case 2: Restoration Blocked by Connectivity

When a module would disconnect the system if it moved:

- Module remains in current position
- Marked as `could_not_restore` with reason `connectivity_critical`
- Consider alternative: wait for other modules to restore first

### Case 3: Cascading Restoration

When module A restoring allows module B to restore:

```python
def cascading_restoration(self, ...):
    """
    Iteratively attempt restoration until no progress.
    After each move, re-evaluate all candidates.
    """
    while True:
        moved_any = False
        for module_id in candidates_by_priority:
            if self.can_restore_position(module_id, ...):
                if self.attempt_restoration_move(module_id, ...):
                    moved_any = True
                    break  # Re-evaluate from start

        if not moved_any:
            break
```

### Case 4: Multiple Faults

When multiple modules are faulty:

```python
fault_positions = {
    tuple(self.modules[f_id].position.astype(int))
    for f_id in faulty_module_ids
}
```

---

## Termination Conditions

| Condition | Result |
|-----------|--------|
| All displaced modules at original positions | **FULL SUCCESS** |
| No more valid restoration moves available | **PARTIAL SUCCESS** (some restored) |
| Max iterations reached | **TIMEOUT** |
| Connectivity would be broken | **BLOCKED** (module stays) |

---

## Metrics and Reporting

### Restoration Quality Metrics

```python
@dataclass
class RestorationMetrics:
    """Summary metrics for restoration quality"""

    # Count metrics
    total_displaced_modules: int
    fully_restored_count: int
    partially_restored_count: int
    unrestored_count: int

    # Distance metrics
    total_initial_displacement: float
    total_final_displacement: float
    displacement_reduction_percent: float

    # Move metrics
    total_restoration_moves: int
    average_moves_per_module: float

    # Efficiency
    restoration_rate: float  # fully_restored / total_displaced

    def to_dict(self) -> Dict:
        return asdict(self)
```

---

## Visualization Integration

### Animation Support

```python
def visualize_restoration(
    self,
    restoration_steps: List[RestorationStep],
    show_original_positions: bool = True,
    highlight_displacement: bool = True
):
    """
    Visualize restoration process with:
    - Ghost spheres at original positions
    - Displacement vectors showing current -> original
    - Animated restoration moves
    - Progress indicators
    """
```

### Color Coding

| State | Color |
|-------|-------|
| Fully restored | Green |
| Partially restored | Yellow |
| Cannot restore | Orange |
| Never moved | Blue (original) |
| Faulty | Red |

---

## Testing Strategy

### Unit Tests

1. **Single module restoration** - One module moved and restored
2. **Chain restoration** - Multiple modules in sequence
3. **Blocked restoration** - Module cannot move without breaking connectivity
4. **Original position occupied** - Falls back to closest available
5. **Cascading restoration** - Restoration enables further restoration

### Integration Tests

1. **Star configuration center fault** - Full Phase 1 + Phase 2
2. **Grid configuration** - Multiple modules restore
3. **Line configuration** - End-to-end restoration
4. **Complex tree** - Hierarchical restoration

### Validation Criteria

- [ ] All moved modules tracked in history
- [ ] Cumulative displacement equals position difference
- [ ] Connectivity preserved at every step
- [ ] Fault positions never occupied
- [ ] Restoration reduces total displacement

---

## Implementation Roadmap

### Stage 1: Movement History Infrastructure
- [ ] Implement `MovementHistory` dataclass
- [ ] Add `RestorationStep` dataclass
- [ ] Modify Phase 1 to track move vectors
- [ ] Unit tests for history tracking

### Stage 2: Core Restoration Logic
- [ ] Implement `can_restore_position()` eligibility check
- [ ] Implement `get_restoration_pivots()` enumeration
- [ ] Implement `select_restoration_module()` selection
- [ ] Unit tests for restoration logic

### Stage 3: Main Algorithm
- [ ] Implement `position_restoration()` main loop
- [ ] Add termination conditions
- [ ] Implement collision handling (if parallel)
- [ ] Integration tests

### Stage 4: Edge Cases
- [ ] Handle blocked original positions
- [ ] Implement cascading restoration
- [ ] Handle multiple faults
- [ ] Edge case tests

### Stage 5: Integration
- [ ] Create `full_damage_response()` combined function
- [ ] Update visualization for Phase 2
- [ ] Add restoration metrics/reporting
- [ ] End-to-end tests

### Stage 6: Optimization
- [ ] Profile performance
- [ ] Optimize pivot enumeration
- [ ] Consider parallel restoration
- [ ] Benchmark tests

---

## Open Questions

1. **Parallel vs Sequential Restoration**: Should multiple modules restore simultaneously (with collision detection) or one at a time?

2. **Restoration Order**: Priority by displacement magnitude vs. by ability to reach original position vs. by enabling others?

3. **Partial Restoration**: If exact original position unreachable, how close is "good enough"?

4. **Fault Expansion**: If restoration creates a hole elsewhere, is that acceptable?

5. **Oscillation Prevention**: How to prevent modules from moving back and forth?

---

## Summary

Phase 2 leverages the additive nature of UDQDG translation vectors to track cumulative displacement and guide modules back toward their original positions. The algorithm:

1. **Tracks** displacement vectors during Phase 1 reconnection
2. **Evaluates** which modules can safely attempt restoration
3. **Prioritizes** modules that can reach their exact original position
4. **Executes** greedy pivots that reduce distance to original
5. **Terminates** when no further restoration is possible

This creates a complete damage-response system:
- **Phase 1**: Coagulate around fault to restore connectivity
- **Phase 2**: Disperse back toward original configuration

The result is a self-healing modular system that minimizes disruption from component failures.
