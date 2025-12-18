import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import random
from dataclasses import dataclass
import networkx as nx
try:
    from .dual_quaternion import UnitDualQuaternion, LATTICE_DIRECTIONS, LATTICE_DIRECTIONS_2D
except ImportError:
    from dual_quaternion import UnitDualQuaternion, LATTICE_DIRECTIONS, LATTICE_DIRECTIONS_2D

@dataclass
class SphericalModule:
    """A spherical module in the UDQDG system."""
    id: str
    position: np.ndarray
    is_active: bool = True
    is_faulty: bool = False  # Damaged/faulty module flag
    radius: float = 0.5  # Half of unit lattice step for perfect touching
    color: Tuple[float, float, float] = (0.7, 0.7, 0.9)


@dataclass
class PivotStep:
    """Records a single pivot operation for visualization/replay."""
    iteration: int
    pivot_type: str  # 'corner' or 'lateral'
    module_id: str
    param1: str      # axis_module (corner) or old_neighbor (lateral)
    param2: str      # new_direction (corner) or new_neighbor (lateral)
    from_pos: np.ndarray
    to_pos: np.ndarray

    def to_tuple(self) -> Tuple:
        """Convert to animation-compatible tuple format."""
        return (self.pivot_type, self.module_id, self.param1, self.param2)


class UDQDGSystem:
    """
    Simple Unit Dual Quaternion Directed Graph system.
    
    Implements G_t = (V_t, E_t, m_t, Q_t, q̂_t) with:
    - Spherical modules
    - Dual quaternion edge gains for pure translations  
    - Corner and lateral pivot operations
    """
    
    def __init__(self, mode_2d: bool = False):
        self.modules: Dict[str, SphericalModule] = {}
        self.edges: Dict[Tuple[str, str], UnitDualQuaternion] = {}
        self.time = 0
        self.mode_2d = mode_2d
        self.directions = LATTICE_DIRECTIONS_2D if mode_2d else LATTICE_DIRECTIONS
        
    def add_module(self, module_id: str, position: np.ndarray, active: bool = True) -> SphericalModule:
        """Add a spherical module to the system."""
        module = SphericalModule(module_id, np.array(position, dtype=float), active)
        self.modules[module_id] = module
        return module

    def validate_no_overlaps(self) -> None:
        """
        Validate that no two modules occupy the same position.
        Raises ValueError if overlapping modules are found.
        """
        positions = {}
        for module_id, module in self.modules.items():
            pos_tuple = tuple(module.position.round(6))  # Round to avoid floating point issues
            if pos_tuple in positions:
                raise ValueError(
                    f"Configuration error: Module '{module_id}' at position {module.position} "
                    f"overlaps with module '{positions[pos_tuple]}' at the same position!"
                )
            positions[pos_tuple] = module_id

    def connect_modules(self, module_a: str, module_b: str) -> bool:
        """Connect two modules with dual quaternion edge gain."""
        if module_a not in self.modules or module_b not in self.modules:
            return False
            
        pos_a = self.modules[module_a].position
        pos_b = self.modules[module_b].position
        
        # Calculate translation vector
        translation = pos_b - pos_a
        
        # Verify it's a unit lattice step
        if not self._is_unit_lattice_step(translation):
            return False
            
        # Create dual quaternion edge gains
        edge_gain_ab = UnitDualQuaternion(translation)
        edge_gain_ba = edge_gain_ab.inverse
        
        self.edges[(module_a, module_b)] = edge_gain_ab
        self.edges[(module_b, module_a)] = edge_gain_ba
        
        return True
    
    def _is_unit_lattice_step(self, translation: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if translation is a valid unit lattice step."""
        # Check if magnitude is approximately 1.0
        if not np.isclose(np.linalg.norm(translation), 1.0, atol=tolerance):
            return False
            
        # Check if direction is in allowed lattice directions
        direction = tuple(np.round(translation).astype(int))
        return direction in self.directions.values()
    
    def get_neighbors(self, module_id: str) -> List[str]:
        """Get all connected neighbors of a module."""
        neighbors = []
        for (a, b) in self.edges:
            if a == module_id:
                neighbors.append(b)
        return neighbors
    
    def corner_pivot(self, pivot_module: str, axis_module: str, new_direction: str) -> bool:
        """
        Perform corner pivot: move pivot_module around corner of axis_module.

        Args:
            pivot_module: Module to pivot
            axis_module: Module to pivot around
            new_direction: New lattice direction (e.g., 'POS_X')
        """
        if not self._can_corner_pivot(pivot_module, axis_module, new_direction):
            return False

        # Get all current neighbors before moving
        old_neighbors = self.get_neighbors(pivot_module)

        # Remove ALL connections (we'll reconnect valid ones after move)
        for neighbor in old_neighbors:
            if (pivot_module, neighbor) in self.edges:
                del self.edges[(pivot_module, neighbor)]
            if (neighbor, pivot_module) in self.edges:
                del self.edges[(neighbor, pivot_module)]

        # Calculate new position
        axis_pos = self.modules[axis_module].position
        new_translation = np.array(self.directions[new_direction])
        new_position = axis_pos + new_translation

        # Update module position
        self.modules[pivot_module].position = new_position

        # Reconnect to axis module (guaranteed to be adjacent after corner pivot)
        self.connect_modules(axis_module, pivot_module)

        return True
    
    def lateral_pivot(self, pivot_module: str, old_neighbor: str, new_neighbor: str) -> bool:
        """
        Perform lateral pivot: roll pivot_module from old_neighbor to new_neighbor.
        """
        if not self._can_lateral_pivot(pivot_module, old_neighbor, new_neighbor):
            return False

        # Get current direction
        old_edge = self.edges.get((old_neighbor, pivot_module))
        if not old_edge:
            return False

        direction = old_edge.translation

        # Get all current neighbors before moving
        old_neighbors = self.get_neighbors(pivot_module)

        # Remove ALL connections (we'll reconnect valid ones after move)
        for neighbor in old_neighbors:
            if (pivot_module, neighbor) in self.edges:
                del self.edges[(pivot_module, neighbor)]
            if (neighbor, pivot_module) in self.edges:
                del self.edges[(neighbor, pivot_module)]

        # Calculate new position
        new_neighbor_pos = self.modules[new_neighbor].position
        new_position = new_neighbor_pos + direction

        # Update module position
        self.modules[pivot_module].position = new_position

        # Reconnect to new_neighbor (guaranteed to be adjacent after lateral pivot)
        self.connect_modules(new_neighbor, pivot_module)

        return True
    
    def _is_pivotable(self, module_id: str) -> bool:
        """Check if module can pivot (active with all active neighbors)."""
        if module_id not in self.modules:
            return False
        if not self.modules[module_id].is_active:
            return False

        neighbors = self.get_neighbors(module_id)
        return all(self.modules[n].is_active for n in neighbors)

    def _are_orthogonal(self, dir1: np.ndarray, dir2: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if two directions are orthogonal."""
        return np.abs(np.dot(dir1, dir2)) < tolerance

    def _get_occupied_ports(self, module_id: str) -> Set[Tuple[int, int, int]]:
        """Get all port directions currently occupied by edges."""
        occupied = set()
        for neighbor in self.get_neighbors(module_id):
            edge_gain = self.edges.get((module_id, neighbor))
            if edge_gain:
                direction = tuple(edge_gain.translation.astype(int))
                occupied.add(direction)
        return occupied

    def _is_port_available(self, module_id: str, direction: Tuple[int, int, int]) -> bool:
        """Check if a port in given direction is available."""
        occupied = self._get_occupied_ports(module_id)
        return direction not in occupied

    def _can_corner_pivot(self, pivot_module: str, axis_module: str, new_direction: str) -> bool:
        """Check if corner pivot is valid."""
        if not (pivot_module in self.modules and axis_module in self.modules):
            return False
        if not (self.modules[pivot_module].is_active and self.modules[axis_module].is_active):
            return False
        if new_direction not in self.directions:
            return False

        # Check if pivot module is pivotable (all neighbors active)
        if not self._is_pivotable(pivot_module):
            return False

        # Get current edge and check orthogonality
        current_edge = self.edges.get((pivot_module, axis_module))
        if current_edge:
            current_translation = current_edge.translation
            new_translation = np.array(self.directions[new_direction])

            if not self._are_orthogonal(current_translation, new_translation):
                return False

        # Check port exclusivity
        new_dir_tuple = self.directions[new_direction]
        neg_new_dir = tuple(-x for x in new_dir_tuple)

        # Axis module must have free port in new_direction (where pivot module will be)
        if not self._is_port_available(axis_module, new_dir_tuple):
            return False

        # Pivot module must have free port at -new_direction (toward axis)
        # Exception: if already connected in this direction, port will be reused
        current_dir = tuple(current_edge.translation.astype(int)) if current_edge else None
        if current_dir != neg_new_dir and not self._is_port_available(pivot_module, neg_new_dir):
            return False

        return True
    
    def _can_lateral_pivot(self, pivot_module: str, old_neighbor: str, new_neighbor: str) -> bool:
        """Check if lateral pivot is valid."""
        if not all(m in self.modules for m in [pivot_module, old_neighbor, new_neighbor]):
            return False
        if not all(self.modules[m].is_active for m in [pivot_module, old_neighbor, new_neighbor]):
            return False

        # Check if pivot module is pivotable (all neighbors active)
        if not self._is_pivotable(pivot_module):
            return False

        # Check if old_neighbor and new_neighbor are connected
        if not ((old_neighbor, new_neighbor) in self.edges or (new_neighbor, old_neighbor) in self.edges):
            return False

        # Get the direction to maintain
        old_edge = self.edges.get((old_neighbor, pivot_module))
        if not old_edge:
            return False

        direction = tuple(old_edge.translation.astype(int))
        neg_direction = tuple(-x for x in direction)

        # New neighbor must have free port at -direction
        if not self._is_port_available(new_neighbor, neg_direction):
            return False

        # Pivot module keeps same direction, so port will be reused
        return True
    
    def random_pivot(self) -> bool:
        """Perform a random valid pivot operation."""
        active_modules = [m for m in self.modules.values() if m.is_active]
        if len(active_modules) < 2:
            return False
            
        # Randomly choose corner or lateral pivot
        pivot_type = random.choice(['corner', 'lateral'])
        
        if pivot_type == 'corner':
            return self._random_corner_pivot()
        else:
            return self._random_lateral_pivot()
    
    def _random_corner_pivot(self) -> bool:
        """Perform random corner pivot."""
        # Find modules with neighbors  
        candidates = []
        for module_id in self.modules:
            neighbors = self.get_neighbors(module_id)
            if neighbors and self.modules[module_id].is_active:
                candidates.extend([(module_id, n) for n in neighbors if self.modules[n].is_active])
        
        if not candidates:
            return False
            
        pivot_module, axis_module = random.choice(candidates)
        new_direction = random.choice(list(self.directions.keys()))
        
        return self.corner_pivot(pivot_module, axis_module, new_direction)
    
    def _random_lateral_pivot(self) -> bool:
        """Perform random lateral pivot."""
        # Find pivot candidates with multiple connection options
        for pivot_module in self.modules:
            if not self.modules[pivot_module].is_active:
                continue
                
            neighbors = self.get_neighbors(pivot_module)
            if len(neighbors) < 1:
                continue
                
            old_neighbor = random.choice(neighbors)
            
            # Find neighbors of the old_neighbor that could be new targets
            potential_new = self.get_neighbors(old_neighbor)
            potential_new = [n for n in potential_new if n != pivot_module and self.modules[n].is_active]
            
            if potential_new:
                new_neighbor = random.choice(potential_new)
                return self.lateral_pivot(pivot_module, old_neighbor, new_neighbor)
                
        return False
    
    def create_grid(self, size: int = 3) -> None:
        """Create a simple grid configuration for testing."""
        for x in range(size):
            for y in range(size):
                module_id = f"M{x}{y}"
                position = np.array([x, y, 0], dtype=float)
                self.add_module(module_id, position)
                
        # Connect grid
        for x in range(size):
            for y in range(size):
                current = f"M{x}{y}"
                if x < size - 1:
                    self.connect_modules(current, f"M{x+1}{y}")
                if y < size - 1:
                    self.connect_modules(current, f"M{x}{y+1}")
    
    def get_all_positions(self) -> Dict[str, np.ndarray]:
        """Get positions of all modules."""
        return {mid: module.position.copy() for mid, module in self.modules.items()}
    
    def get_all_edges(self) -> List[Tuple[str, str]]:
        """Get all edges (undirected)."""
        edges = set()
        for (a, b) in self.edges:
            edge = tuple(sorted([a, b]))
            edges.add(edge)
        return list(edges)

    # Ego-based Fault Response Algorithm Methods

    def mark_fault(self, module_id: str) -> bool:
        """Mark a module as faulty (damaged)."""
        if module_id not in self.modules:
            return False
        self.modules[module_id].is_faulty = True
        self.modules[module_id].is_active = False
        return True

    def get_k_hop_neighbors(self, module_id: str, k: int) -> Set[str]:
        """Get all neighbors within k hops using BFS."""
        if module_id not in self.modules:
            return set()

        visited = set()
        current_level = {module_id}

        for _ in range(k):
            next_level = set()
            for node in current_level:
                if node not in visited:
                    visited.add(node)
                    neighbors = self.get_neighbors(node)
                    next_level.update(neighbors)
            current_level = next_level

        # Remove the starting module itself
        visited.discard(module_id)
        return visited

    def is_leaf_node(self, module_id: str) -> bool:
        """Check if module is a leaf (only one neighbor)."""
        if module_id not in self.modules:
            return False
        neighbors = self.get_neighbors(module_id)
        return len(neighbors) == 1

    def can_respond_to_fault(self, module_id: str) -> bool:
        """
        Ego-based decision: Can this module respond to a fault signal?

        Checks:
        1. Am I a leaf node? (only one neighbor) → can move
        2. Are pivot options available to me?
        3. Can my neighbors reach each other WITHOUT going through me?
           (ensures I'm not critical for connectivity)
        """
        if module_id not in self.modules:
            return False

        module = self.modules[module_id]
        if not module.is_active or module.is_faulty:
            return False

        # Check 1: Leaf nodes can always move
        if self.is_leaf_node(module_id):
            return True

        # Check 2: Are pivot options available?
        neighbors = self.get_neighbors(module_id)
        has_pivot_option = False

        for neighbor in neighbors:
            # Check corner pivots
            for direction in self.directions.keys():
                if self._can_corner_pivot(module_id, neighbor, direction):
                    has_pivot_option = True
                    break

            # Check lateral pivots
            neighbor_neighbors = self.get_neighbors(neighbor)
            for new_neighbor in neighbor_neighbors:
                if new_neighbor != module_id and self._can_lateral_pivot(module_id, neighbor, new_neighbor):
                    has_pivot_option = True
                    break

            if has_pivot_option:
                break

        if not has_pivot_option:
            return False

        # Check 3: Can my neighbors reach each other without me?
        # Use 3-hop information but exclude paths through self
        neighbors = self.get_neighbors(module_id)
        if len(neighbors) <= 1:
            return True  # Only one neighbor, can't break connectivity

        # For each neighbor, check if it can reach all other neighbors
        # through paths that don't go through me (using 3-hop limit)
        for i, neighbor_a in enumerate(neighbors):
            # Get 3-hop neighbors of neighbor_a, excluding paths through module_id
            reachable = self._get_k_hop_neighbors_excluding(neighbor_a, 3, exclude={module_id})

            # Check if all other neighbors are reachable
            for neighbor_b in neighbors[i+1:]:
                if neighbor_b not in reachable:
                    # neighbor_a can't reach neighbor_b without going through me
                    # So I'm critical for connectivity - can't move
                    return False

        return True

    def _get_k_hop_neighbors_excluding(
        self,
        module_id: str,
        k: int,
        exclude: Set[str]
    ) -> Set[str]:
        """
        Get all modules reachable within k hops, excluding certain modules from paths.

        Args:
            module_id: Starting module
            k: Maximum number of hops
            exclude: Set of module IDs to exclude from paths

        Returns:
            Set of reachable module IDs (excluding self and excluded modules)
        """
        if k <= 0:
            return set()

        visited = {module_id} | exclude  # Don't revisit self or excluded
        current_frontier = {module_id}
        all_reachable = set()

        for _ in range(k):
            next_frontier = set()
            for node in current_frontier:
                for neighbor in self.get_neighbors(node):
                    if neighbor not in visited:
                        # Only include active, non-faulty modules
                        if self.modules[neighbor].is_active and not self.modules[neighbor].is_faulty:
                            next_frontier.add(neighbor)
                            all_reachable.add(neighbor)
                            visited.add(neighbor)
            current_frontier = next_frontier
            if not current_frontier:
                break

        return all_reachable

    def select_responding_module(
        self,
        component: Set[str],
        fault_pos: np.ndarray,
        all_components: Optional[List[Set[str]]] = None
    ) -> Optional[str]:
        """
        Select ONE module from a component to respond to fault.

        Selection criteria (models damage signal propagation):
        1. Must be able to respond (can_respond_to_fault)
        2. Must have at least one valid pivot move available
        3. Primary sort: distance to fault (ascending - closer modules respond first)
        4. Tiebreaker: module ID (lexicographic for determinism)

        Args:
            component: Set of module IDs in this connected component
            fault_pos: Position of the fault
            all_components: All connected components (for target calculation)

        Returns:
            Module ID of selected responder, or None if no valid candidates
        """
        candidates = []
        for module_id in component:
            if self.can_respond_to_fault(module_id):
                dist = np.linalg.norm(self.modules[module_id].position - fault_pos)
                candidates.append((dist, module_id))

        if not candidates:
            return None

        # Sort by distance (ascending), then by ID (lexicographic)
        candidates.sort(key=lambda x: (x[0], x[1]))

        # Find first candidate that actually has valid pivots
        for dist, module_id in candidates:
            target_pos = self._get_target_position(
                module_id, component, all_components or [component], fault_pos
            )
            pivots = self.get_available_pivots(module_id, target_pos)
            if pivots:
                return module_id

        # No candidate has valid pivots
        return None

    def _position_is_occupied(self, position: np.ndarray, exclude_module: Optional[str] = None) -> bool:
        """Check if a position is occupied by any module."""
        for module_id, module in self.modules.items():
            if exclude_module and module_id == exclude_module:
                continue
            if np.allclose(module.position, position):
                return True
        return False

    def get_available_pivots(self, module_id: str, target_pos: np.ndarray) -> List[Tuple[str, str, str, Optional[str]]]:
        """
        Get available pivot operations that move towards target position.

        Returns list of tuples: (pivot_type, pivot_module, axis/old_neighbor, new_direction/new_neighbor)
        - For corner: ('corner', module_id, axis_module, new_direction)
        - For lateral: ('lateral', module_id, old_neighbor, new_neighbor)

        Note: Only returns moves that strictly decrease distance to target.
        This is a greedy algorithm that may get stuck in some configurations
        (e.g., line split in middle where leaves can't bridge the gap).
        """
        if module_id not in self.modules:
            return []

        current_pos = self.modules[module_id].position
        current_distance = np.linalg.norm(current_pos - target_pos)

        available_pivots = []
        neighbors = self.get_neighbors(module_id)

        # Check corner pivots
        for neighbor in neighbors:
            for direction in self.directions.keys():
                if self._can_corner_pivot(module_id, neighbor, direction):
                    # Calculate new position after corner pivot
                    neighbor_pos = self.modules[neighbor].position
                    new_translation = np.array(self.directions[direction])
                    new_pos = neighbor_pos + new_translation

                    # Don't move onto the fault location itself
                    if np.allclose(new_pos, target_pos):
                        continue

                    # Don't move to an occupied position (collision detection)
                    if self._position_is_occupied(new_pos, exclude_module=module_id):
                        continue

                    new_distance = np.linalg.norm(new_pos - target_pos)

                    # Only include if it moves us closer
                    if new_distance < current_distance:
                        available_pivots.append(('corner', module_id, neighbor, direction))

        # Check lateral pivots
        for old_neighbor in neighbors:
            neighbor_neighbors = self.get_neighbors(old_neighbor)
            for new_neighbor in neighbor_neighbors:
                if new_neighbor != module_id and self._can_lateral_pivot(module_id, old_neighbor, new_neighbor):
                    # Calculate new position after lateral pivot
                    old_edge = self.edges.get((old_neighbor, module_id))
                    if old_edge:
                        direction = old_edge.translation
                        new_neighbor_pos = self.modules[new_neighbor].position
                        new_pos = new_neighbor_pos + direction

                        # Don't move onto the fault location itself
                        if np.allclose(new_pos, target_pos):
                            continue

                        # Don't move to an occupied position (collision detection)
                        if self._position_is_occupied(new_pos, exclude_module=module_id):
                            continue

                        new_distance = np.linalg.norm(new_pos - target_pos)

                        # Only include if it moves us closer
                        if new_distance < current_distance:
                            available_pivots.append(('lateral', module_id, old_neighbor, new_neighbor))

        return available_pivots

    def _form_new_connections(self) -> int:
        """
        Scan for and form new connections between adjacent modules.

        Returns:
            Number of new connections formed
        """
        new_connections = 0
        active_modules = [mid for mid, m in self.modules.items() if m.is_active]

        for module_id in active_modules:
            module_pos = self.modules[module_id].position

            # Check all other active modules
            for other_id in active_modules:
                if module_id == other_id:
                    continue

                # Skip if already connected
                if (module_id, other_id) in self.edges:
                    continue

                other_pos = self.modules[other_id].position
                translation = other_pos - module_pos

                # Check if they're at unit lattice distance
                if self._is_unit_lattice_step(translation):
                    # Check if both modules have free ports for connection
                    direction = tuple(translation.astype(int))
                    neg_direction = tuple((-translation).astype(int))

                    if self._is_port_available(module_id, direction) and \
                       self._is_port_available(other_id, neg_direction):
                        # Form the connection
                        if self.connect_modules(module_id, other_id):
                            new_connections += 1

        return new_connections

    def _calculate_pivot_destination(
        self,
        module_id: str,
        pivot: Tuple[str, str, str, str]
    ) -> np.ndarray:
        """
        Calculate where a module will end up after executing a pivot.

        Args:
            module_id: The module being pivoted
            pivot: Tuple of (pivot_type, module_id, param1, param2)

        Returns:
            The destination position as numpy array
        """
        pivot_type, _, param1, param2 = pivot
        module_pos = self.modules[module_id].position

        if pivot_type == 'corner':
            # Corner pivot: rotate 90 degrees around axis_module
            axis_module = param1
            new_direction = param2  # String like 'POS_X', 'NEG_Y', etc.
            axis_pos = self.modules[axis_module].position

            # Get direction vector from LATTICE_DIRECTIONS or self.directions
            if new_direction in self.directions:
                direction_vec = np.array(self.directions[new_direction], dtype=float)
            else:
                # Fallback: return axis position
                return axis_pos.copy()

            # New position is axis position + direction
            return axis_pos + direction_vec

        elif pivot_type == 'lateral':
            # Lateral pivot: roll from old_neighbor to new_neighbor
            old_neighbor = param1
            new_neighbor = param2
            new_neighbor_pos = self.modules[new_neighbor].position

            # Calculate direction from new_neighbor to module's destination
            # The module ends up on the opposite side of new_neighbor from old_neighbor
            old_neighbor_pos = self.modules[old_neighbor].position

            # Module moves to position that's adjacent to new_neighbor,
            # continuing in same direction as old_neighbor -> new_neighbor
            direction = new_neighbor_pos - old_neighbor_pos
            direction = direction / np.linalg.norm(direction)  # Normalize
            return new_neighbor_pos + direction

        # Fallback: return current position
        return module_pos.copy()

    def ego_fault_response(
        self,
        fault_id: str,
        max_iterations: int = 1000,
        one_per_subgraph: bool = True,
        parallel_subgraphs: bool = True,
        record_steps: bool = False
    ) -> Dict[str, any]:
        """
        Execute ego-based fault response algorithm.

        Models damage signal propagation where modules closer to the fault
        respond first. When one_per_subgraph=True (default), only ONE module
        per connected component moves per iteration, selected by distance to fault.

        When parallel_subgraphs=True (default), all subgraphs move their selected
        module simultaneously. Collisions are resolved using distance+ID tiebreaker.

        The algorithm continues until either:
        1. Active modules are reconnected (single connected component)
        2. No more moves are possible
        3. Max iterations reached

        Args:
            fault_id: ID of the faulty module
            max_iterations: Maximum iterations to run
            one_per_subgraph: If True, only one module per component moves per iteration
                             (models signal propagation). If False, all responsive
                             modules can move (legacy behavior).
            parallel_subgraphs: If True, all subgraphs move simultaneously with
                               collision detection. If False, sequential execution.
            record_steps: If True, record each pivot operation for visualization/replay.

        Returns:
            Dictionary with statistics about the response.
            If record_steps=True, includes 'steps' (list of PivotStep) and
            'failed_attempts' (list of failed pivot info for debugging).
        """
        if fault_id not in self.modules:
            return {"success": False, "reason": "Fault module not found"}

        if not self.modules[fault_id].is_faulty:
            return {"success": False, "reason": "Module is not marked as faulty"}

        fault_pos = self.modules[fault_id].position
        stats = {
            "iterations": 0,
            "total_moves": 0,
            "modules_responded": set(),
            "wave_fronts": [],
            "reconnected": False,
            "new_connections_formed": 0,
            "collisions_resolved": 0,
            "success": True
        }

        # Step recording for visualization
        if record_steps:
            stats["steps"] = []
            stats["parallel_steps"] = []  # Groups of simultaneous moves
            stats["failed_attempts"] = []

        for iteration in range(max_iterations):
            stats["iterations"] = iteration + 1

            # Check if we've reconnected
            if self.is_connected(active_only=True):
                stats["reconnected"] = True
                break

            moves_this_iteration = 0

            # Get all connected components
            components = self.get_connected_components(active_only=True)

            if one_per_subgraph:
                # One module per subgraph mode
                # Step 1: Collect candidate moves from all components
                candidate_moves = []  # List of (module_id, pivot, target_pos, from_pos, dist_to_fault)

                for component in components:
                    module_id = self.select_responding_module(
                        component, fault_pos, all_components=components
                    )

                    if module_id is None:
                        continue

                    target_pos = self._get_target_position(
                        module_id, component, components, fault_pos
                    )

                    pivots = self.get_available_pivots(module_id, target_pos)

                    if pivots:
                        pivot = pivots[0]
                        from_pos = self.modules[module_id].position.copy()
                        dist_to_fault = np.linalg.norm(from_pos - fault_pos)

                        # Calculate where module will end up after pivot
                        pivot_target = self._calculate_pivot_destination(
                            module_id, pivot
                        )

                        candidate_moves.append({
                            "module_id": module_id,
                            "pivot": pivot,
                            "target_pos": pivot_target,
                            "from_pos": from_pos,
                            "dist_to_fault": dist_to_fault
                        })

                if parallel_subgraphs and len(candidate_moves) > 1:
                    # Step 2: Detect and resolve collisions
                    # Group moves by target position
                    position_groups = {}
                    for move in candidate_moves:
                        pos_key = tuple(move["target_pos"].astype(int))
                        if pos_key not in position_groups:
                            position_groups[pos_key] = []
                        position_groups[pos_key].append(move)

                    # Resolve collisions using distance + ID tiebreaker
                    moves_to_execute = []
                    for pos_key, moves in position_groups.items():
                        if len(moves) == 1:
                            moves_to_execute.append(moves[0])
                        else:
                            # Collision! Sort by distance to fault, then by ID
                            moves.sort(key=lambda m: (m["dist_to_fault"], m["module_id"]))
                            winner = moves[0]
                            moves_to_execute.append(winner)
                            stats["collisions_resolved"] += len(moves) - 1

                            if record_steps:
                                # Log collision losers
                                for loser in moves[1:]:
                                    stats["failed_attempts"].append({
                                        "iteration": iteration + 1,
                                        "module_id": loser["module_id"],
                                        "pivot_type": loser["pivot"][0],
                                        "reason": f"collision with {winner['module_id']} at {pos_key}"
                                    })

                    # Step 3: Execute all non-colliding moves (parallel)
                    parallel_group = []  # For step recording
                    for move in moves_to_execute:
                        pivot_type, pivot_module, param1, param2 = move["pivot"]
                        module_id = move["module_id"]
                        from_pos = move["from_pos"]

                        success = False
                        if pivot_type == 'corner':
                            success = self.corner_pivot(pivot_module, param1, param2)
                        elif pivot_type == 'lateral':
                            success = self.lateral_pivot(pivot_module, param1, param2)

                        if success:
                            moves_this_iteration += 1
                            stats["modules_responded"].add(module_id)
                            stats["total_moves"] += 1

                            if record_steps:
                                to_pos = self.modules[module_id].position.copy()
                                step = PivotStep(
                                    iteration=iteration + 1,
                                    pivot_type=pivot_type,
                                    module_id=module_id,
                                    param1=param1,
                                    param2=param2,
                                    from_pos=from_pos,
                                    to_pos=to_pos
                                )
                                stats["steps"].append(step)
                                parallel_group.append(step)
                        elif record_steps:
                            stats["failed_attempts"].append({
                                "iteration": iteration + 1,
                                "module_id": module_id,
                                "pivot_type": pivot_type,
                                "param1": param1,
                                "param2": param2
                            })

                    # Record parallel group for visualization
                    if record_steps and parallel_group:
                        stats["parallel_steps"].append(parallel_group)

                else:
                    # Sequential execution (single component or parallel_subgraphs=False)
                    for move in candidate_moves:
                        pivot_type, pivot_module, param1, param2 = move["pivot"]
                        module_id = move["module_id"]
                        from_pos = move["from_pos"]

                        success = False
                        if pivot_type == 'corner':
                            success = self.corner_pivot(pivot_module, param1, param2)
                        elif pivot_type == 'lateral':
                            success = self.lateral_pivot(pivot_module, param1, param2)

                        if success:
                            moves_this_iteration += 1
                            stats["modules_responded"].add(module_id)
                            stats["total_moves"] += 1

                            if record_steps:
                                to_pos = self.modules[module_id].position.copy()
                                step = PivotStep(
                                    iteration=iteration + 1,
                                    pivot_type=pivot_type,
                                    module_id=module_id,
                                    param1=param1,
                                    param2=param2,
                                    from_pos=from_pos,
                                    to_pos=to_pos
                                )
                                stats["steps"].append(step)
                        elif record_steps:
                            stats["failed_attempts"].append({
                                "iteration": iteration + 1,
                                "module_id": module_id,
                                "pivot_type": pivot_type,
                                "param1": param1,
                                "param2": param2
                            })
            else:
                # Legacy mode: all responsive modules can move
                for module_id in sorted(self.modules.keys()):
                    module = self.modules[module_id]

                    if not module.is_active or module.is_faulty:
                        continue

                    if not self.can_respond_to_fault(module_id):
                        continue

                    my_component = None
                    for comp in components:
                        if module_id in comp:
                            my_component = comp
                            break

                    target_pos = self._get_target_position(
                        module_id, my_component, components, fault_pos
                    )

                    pivots = self.get_available_pivots(module_id, target_pos)

                    if pivots:
                        pivot_type, pivot_module, param1, param2 = pivots[0]

                        # Record position before pivot
                        from_pos = self.modules[module_id].position.copy()

                        success = False
                        if pivot_type == 'corner':
                            success = self.corner_pivot(pivot_module, param1, param2)
                        elif pivot_type == 'lateral':
                            success = self.lateral_pivot(pivot_module, param1, param2)

                        if success:
                            moves_this_iteration += 1
                            stats["modules_responded"].add(module_id)
                            stats["total_moves"] += 1

                            # Record step for visualization
                            if record_steps:
                                to_pos = self.modules[module_id].position.copy()
                                step = PivotStep(
                                    iteration=iteration + 1,
                                    pivot_type=pivot_type,
                                    module_id=module_id,
                                    param1=param1,
                                    param2=param2,
                                    from_pos=from_pos,
                                    to_pos=to_pos
                                )
                                stats["steps"].append(step)
                        elif record_steps:
                            stats["failed_attempts"].append({
                                "iteration": iteration + 1,
                                "module_id": module_id,
                                "pivot_type": pivot_type,
                                "param1": param1,
                                "param2": param2
                            })

            # After all moves, scan for and form new connections
            new_conn = self._form_new_connections()
            stats["new_connections_formed"] += new_conn

            # Stop if no moves were made this iteration AND no new connections formed
            if moves_this_iteration == 0 and new_conn == 0:
                break

        stats["modules_responded"] = list(stats["modules_responded"])
        return stats

    def _get_target_position(
        self,
        module_id: str,
        my_component: Optional[Set[str]],
        all_components: List[Set[str]],
        fault_pos: np.ndarray
    ) -> np.ndarray:
        """
        Determine target position for a module to move towards.

        If disconnected (multiple components), targets closest module in other component.
        Otherwise, targets the fault position.
        """
        if len(all_components) > 1 and my_component is not None:
            # Find closest module in other components
            module_pos = self.modules[module_id].position
            min_distance = float('inf')
            target_pos = fault_pos

            for comp in all_components:
                if comp == my_component:
                    continue
                for other_id in comp:
                    other_pos = self.modules[other_id].position
                    distance = np.linalg.norm(module_pos - other_pos)
                    if distance < min_distance:
                        min_distance = distance
                        target_pos = other_pos

            return target_pos
        else:
            return fault_pos

    def is_connected(self, active_only: bool = True) -> bool:
        """
        Check if the graph is connected.

        Args:
            active_only: If True, only check connectivity of active modules

        Returns:
            True if connected, False otherwise
        """
        if not self.modules:
            return True

        # Build networkx graph
        G = nx.Graph()

        # Add nodes
        for module_id, module in self.modules.items():
            if not active_only or module.is_active:
                G.add_node(module_id)

        # Add edges
        for (src, dst) in self.edges:
            if src in G.nodes and dst in G.nodes:
                G.add_edge(src, dst)

        # Check connectivity
        return nx.is_connected(G) if G.number_of_nodes() > 0 else True

    def get_connected_components(self, active_only: bool = True) -> List[Set[str]]:
        """
        Get all connected components.

        Args:
            active_only: If True, only consider active modules

        Returns:
            List of sets, each set containing module IDs in a component
        """
        if not self.modules:
            return []

        # Build networkx graph
        G = nx.Graph()

        # Add nodes
        for module_id, module in self.modules.items():
            if not active_only or module.is_active:
                G.add_node(module_id)

        # Add edges
        for (src, dst) in self.edges:
            if src in G.nodes and dst in G.nodes:
                G.add_edge(src, dst)

        # Get connected components
        return [set(component) for component in nx.connected_components(G)]

    def to_2d_array(self, padding: int = 1) -> np.ndarray:
        """
        Convert system to 2D array for visualization (assumes z=0 plane).

        Args:
            padding: Number of empty cells to pad around the modules (default: 1)

        Returns:
            2D numpy array where:
            0 = empty space
            1 = active module
            -1 = faulty module
        """
        if not self.modules:
            return np.array([[]])

        # Find bounds
        positions = [m.position for m in self.modules.values()]
        x_coords = [int(p[0]) for p in positions]
        y_coords = [int(p[1]) for p in positions]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Create array with padding (note: y-axis is inverted for visualization)
        width = max_x - min_x + 1 + 2 * padding
        height = max_y - min_y + 1 + 2 * padding
        array = np.zeros((height, width), dtype=int)

        # Fill array (offset by padding)
        for module in self.modules.values():
            x = int(module.position[0]) - min_x + padding
            y = int(module.position[1]) - min_y + padding

            if module.is_faulty:
                array[y, x] = -1
            elif module.is_active:
                array[y, x] = 1
            else:
                array[y, x] = 0

        return array