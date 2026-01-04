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


@dataclass
class MovementHistory:
    """
    Tracks cumulative movement for a module during fault response.

    The UDQDG framework uses additive translation vectors, so we can
    recover the original position by: original = current - cumulative_displacement
    """
    module_id: str
    original_position: np.ndarray      # Position before any Phase 1 moves
    cumulative_displacement: np.ndarray  # Sum of all move vectors (Σδᵢ)
    move_count: int = 0                  # Number of moves made
    move_sequence: List[np.ndarray] = None  # Individual move vectors (optional)

    def __post_init__(self):
        """Initialize move_sequence if not provided."""
        if self.move_sequence is None:
            self.move_sequence = []
        # Ensure arrays are copies to avoid reference issues
        self.original_position = np.array(self.original_position, dtype=float)
        self.cumulative_displacement = np.array(self.cumulative_displacement, dtype=float)

    def add_move(self, delta: np.ndarray) -> None:
        """
        Record a new move.

        Args:
            delta: The translation vector for this move (to_pos - from_pos)
        """
        delta = np.array(delta, dtype=float)
        self.cumulative_displacement += delta
        self.move_count += 1
        self.move_sequence.append(delta.copy())

    def get_original_position(self) -> np.ndarray:
        """Return the original position before all moves."""
        return self.original_position.copy()

    def get_restoration_distance(self, current_pos: np.ndarray) -> float:
        """
        Calculate distance from current position to original position.

        Args:
            current_pos: The module's current position

        Returns:
            Euclidean distance to original position
        """
        return float(np.linalg.norm(current_pos - self.original_position))

    def get_displacement_magnitude(self) -> float:
        """Return the magnitude of cumulative displacement."""
        return float(np.linalg.norm(self.cumulative_displacement))

    def has_moved(self) -> bool:
        """Check if the module has moved at all."""
        return self.move_count > 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "module_id": self.module_id,
            "original_position": self.original_position.tolist(),
            "cumulative_displacement": self.cumulative_displacement.tolist(),
            "move_count": self.move_count,
            "displacement_magnitude": self.get_displacement_magnitude()
        }


@dataclass
class RestorationStep:
    """Records a single restoration move during Phase 2."""
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

    def to_tuple(self) -> Tuple:
        """Convert to animation-compatible tuple format."""
        return (self.pivot_type, self.module_id, self.param1, self.param2)

    def distance_improvement(self) -> float:
        """Return how much closer to original position after this move."""
        return self.distance_before - self.distance_after


@dataclass
class RestorationMetrics:
    """Summary metrics for Phase 2 restoration quality."""

    # Count metrics
    total_displaced_modules: int = 0
    fully_restored_count: int = 0
    partially_restored_count: int = 0
    unrestored_count: int = 0

    # Distance metrics
    total_initial_displacement: float = 0.0
    total_final_displacement: float = 0.0
    displacement_reduction: float = 0.0
    displacement_reduction_percent: float = 0.0

    # Move metrics
    total_restoration_moves: int = 0
    iterations_used: int = 0

    # Success indicator
    all_restored: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "total_displaced_modules": self.total_displaced_modules,
            "fully_restored_count": self.fully_restored_count,
            "partially_restored_count": self.partially_restored_count,
            "unrestored_count": self.unrestored_count,
            "total_initial_displacement": self.total_initial_displacement,
            "total_final_displacement": self.total_final_displacement,
            "displacement_reduction": self.displacement_reduction,
            "displacement_reduction_percent": self.displacement_reduction_percent,
            "total_restoration_moves": self.total_restoration_moves,
            "iterations_used": self.iterations_used,
            "all_restored": self.all_restored,
        }


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
        # Exception 1: if already connected to axis in this direction, port will be reused
        # Exception 2: if the blocking neighbor will no longer be adjacent after pivot
        current_dir = tuple(current_edge.translation.astype(int)) if current_edge else None
        if current_dir != neg_new_dir:
            if not self._is_port_available(pivot_module, neg_new_dir):
                # Port is occupied - check if the pivot will break that connection
                # Find which neighbor occupies that port
                blocking_neighbor = None
                for neighbor in self.get_neighbors(pivot_module):
                    edge = self.edges.get((pivot_module, neighbor))
                    if edge:
                        edge_dir = tuple(edge.translation.astype(int))
                        if edge_dir == neg_new_dir:
                            blocking_neighbor = neighbor
                            break

                if blocking_neighbor:
                    # Calculate new position after pivot
                    axis_pos = self.modules[axis_module].position
                    new_pos = axis_pos + np.array(new_dir_tuple)
                    blocking_pos = self.modules[blocking_neighbor].position

                    # Check if still adjacent after pivot (distance <= 1)
                    if np.linalg.norm(new_pos - blocking_pos) <= 1.0 + 1e-6:
                        # Still adjacent - port would remain occupied, block pivot
                        return False
                    # Not adjacent - pivot will break connection, port will be freed
                else:
                    # No blocking neighbor found but port is occupied - shouldn't happen
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

    # Phase 2: Position Restoration Methods

    def ego_fault_response_with_history(
        self,
        fault_module_id: str,
        max_iterations: int = 1000,
        one_per_subgraph: bool = True,
        parallel_subgraphs: bool = True,
        record_steps: bool = True
    ) -> Tuple[Dict[str, any], Dict[str, 'MovementHistory']]:
        """
        Execute ego-based fault response with movement history tracking.

        This is an extended version of ego_fault_response that also returns
        movement histories for each module, enabling Phase 2 restoration.

        The movement history tracks:
        - Original position before any moves
        - Cumulative displacement vector (sum of all move deltas)
        - Individual move vectors for detailed analysis

        Args:
            fault_module_id: ID of the faulty module
            max_iterations: Maximum iterations to run
            one_per_subgraph: If True, only one module per component moves per iteration
            parallel_subgraphs: If True, all subgraphs move simultaneously
            record_steps: If True, record each pivot operation (default True for history)

        Returns:
            Tuple of:
            - stats: Dictionary with fault response statistics (same as ego_fault_response)
            - movement_histories: Dict mapping module_id to MovementHistory objects
        """
        if fault_module_id not in self.modules:
            return (
                {"success": False, "reason": "Fault module not found"},
                {}
            )

        if not self.modules[fault_module_id].is_faulty:
            return (
                {"success": False, "reason": "Module is not marked as faulty"},
                {}
            )

        # Initialize movement histories for all active, non-faulty modules
        movement_histories: Dict[str, MovementHistory] = {}
        for module_id, module in self.modules.items():
            if module.is_active and not module.is_faulty:
                movement_histories[module_id] = MovementHistory(
                    module_id=module_id,
                    original_position=module.position.copy(),
                    cumulative_displacement=np.zeros(3),
                    move_count=0,
                    move_sequence=[]
                )

        # Run Phase 1 with step recording enabled
        stats = self.ego_fault_response(
            fault_id=fault_module_id,
            max_iterations=max_iterations,
            one_per_subgraph=one_per_subgraph,
            parallel_subgraphs=parallel_subgraphs,
            record_steps=True  # Always record for history extraction
        )

        # Extract movement deltas from recorded steps
        if "steps" in stats:
            for step in stats["steps"]:
                module_id = step.module_id
                if module_id in movement_histories:
                    delta = step.to_pos - step.from_pos
                    movement_histories[module_id].add_move(delta)

        # If caller didn't want steps, remove them from stats
        if not record_steps:
            stats.pop("steps", None)
            stats.pop("parallel_steps", None)
            stats.pop("failed_attempts", None)

        # Add summary of movement to stats
        moved_modules = [
            mid for mid, hist in movement_histories.items()
            if hist.has_moved()
        ]
        stats["modules_with_history"] = len(movement_histories)
        stats["modules_displaced"] = len(moved_modules)
        stats["total_displacement"] = sum(
            hist.get_displacement_magnitude()
            for hist in movement_histories.values()
        )

        return stats, movement_histories

    def get_movement_summary(
        self,
        movement_histories: Dict[str, 'MovementHistory']
    ) -> Dict[str, any]:
        """
        Generate a summary of movement histories.

        Args:
            movement_histories: Dictionary of MovementHistory objects

        Returns:
            Summary dictionary with aggregate statistics
        """
        if not movement_histories:
            return {
                "total_modules": 0,
                "modules_moved": 0,
                "total_moves": 0,
                "total_displacement": 0.0,
                "max_displacement": 0.0,
                "avg_displacement": 0.0,
                "modules_by_move_count": {}
            }

        moved = [h for h in movement_histories.values() if h.has_moved()]
        displacements = [h.get_displacement_magnitude() for h in moved]

        # Group by move count
        move_counts = {}
        for hist in movement_histories.values():
            count = hist.move_count
            if count not in move_counts:
                move_counts[count] = []
            move_counts[count].append(hist.module_id)

        return {
            "total_modules": len(movement_histories),
            "modules_moved": len(moved),
            "total_moves": sum(h.move_count for h in movement_histories.values()),
            "total_displacement": sum(displacements),
            "max_displacement": max(displacements) if displacements else 0.0,
            "avg_displacement": (
                sum(displacements) / len(displacements) if displacements else 0.0
            ),
            "modules_by_move_count": move_counts
        }

    def verify_position_consistency(
        self,
        movement_histories: Dict[str, 'MovementHistory']
    ) -> Dict[str, any]:
        """
        Verify that movement histories are consistent with current positions.

        This validates that: current_pos = original_pos + cumulative_displacement

        Args:
            movement_histories: Dictionary of MovementHistory objects

        Returns:
            Verification results with any inconsistencies found
        """
        results = {
            "consistent": True,
            "checked": 0,
            "inconsistencies": []
        }

        for module_id, history in movement_histories.items():
            if module_id not in self.modules:
                continue

            results["checked"] += 1
            current_pos = self.modules[module_id].position
            expected_pos = history.original_position + history.cumulative_displacement

            if not np.allclose(current_pos, expected_pos, atol=1e-6):
                results["consistent"] = False
                results["inconsistencies"].append({
                    "module_id": module_id,
                    "current_position": current_pos.tolist(),
                    "expected_position": expected_pos.tolist(),
                    "difference": (current_pos - expected_pos).tolist()
                })

        return results

    # Phase 2: Core Restoration Logic

    def can_restore_position(
        self,
        module_id: str,
        movement_histories: Dict[str, 'MovementHistory'],
        fault_positions: Set[Tuple[int, int, int]]
    ) -> bool:
        """
        Check if a module can attempt position restoration.

        A module can restore if:
        1. It has non-zero displacement (actually moved in Phase 1)
        2. It's a leaf node OR passes connectivity preservation check
        3. It has at least one valid pivot toward original position

        Args:
            module_id: Module to check
            movement_histories: Movement histories from Phase 1
            fault_positions: Set of fault positions to avoid

        Returns:
            True if module can attempt restoration
        """
        # Check 1: Module must exist and be active
        if module_id not in self.modules:
            return False

        module = self.modules[module_id]
        if not module.is_active or module.is_faulty:
            return False

        # Check 2: Must have displacement (actually moved in Phase 1)
        if module_id not in movement_histories:
            return False

        history = movement_histories[module_id]
        if not history.has_moved():
            return False

        # Check 3: Leaf node exception - can always try to restore
        neighbors = self.get_neighbors(module_id)
        if len(neighbors) == 1:
            # Still need to verify has valid pivots
            pivots = self.get_restoration_pivots(
                module_id, movement_histories, fault_positions
            )
            return len(pivots) > 0

        # Check 4: Has valid pivot options toward original position
        pivots = self.get_restoration_pivots(
            module_id, movement_histories, fault_positions
        )
        if not pivots:
            return False

        # Check 5: 3-hop connectivity preservation (same as Phase 1)
        # Neighbors must be able to reach each other without going through me
        if len(neighbors) > 1:
            for i, neighbor_a in enumerate(neighbors):
                reachable = self._get_k_hop_neighbors_excluding(
                    neighbor_a, 3, exclude={module_id}
                )
                for neighbor_b in neighbors[i+1:]:
                    if neighbor_b not in reachable:
                        # Can't move - would disconnect neighbors
                        return False

        return True

    def get_restoration_pivots(
        self,
        module_id: str,
        movement_histories: Dict[str, 'MovementHistory'],
        fault_positions: Set[Tuple[int, int, int]]
    ) -> List[Dict]:
        """
        Get valid pivot operations that move module toward its original position.

        Only returns moves that strictly decrease distance to original position.
        Prioritizes moves that reach the exact original position.

        Args:
            module_id: Module to get pivots for
            movement_histories: Movement histories from Phase 1
            fault_positions: Set of fault positions to avoid

        Returns:
            List of pivot dictionaries sorted by priority:
            - reaches_original (descending)
            - distance_reduction (descending)

            Each dict contains:
            - type: 'corner' or 'lateral'
            - axis/old_neighbor: pivot axis module
            - direction/new_neighbor: target direction or neighbor
            - new_pos: resulting position
            - distance_reduction: how much closer to original
            - reaches_original: True if this move reaches exact original
        """
        if module_id not in self.modules:
            return []

        if module_id not in movement_histories:
            return []

        history = movement_histories[module_id]
        original_pos = history.get_original_position()
        current_pos = self.modules[module_id].position
        current_distance = np.linalg.norm(current_pos - original_pos)

        # If already at original position, no restoration needed
        if np.allclose(current_pos, original_pos, atol=1e-6):
            return []

        valid_pivots = []
        neighbors = self.get_neighbors(module_id)

        # Check corner pivots
        for neighbor in neighbors:
            if not self.modules[neighbor].is_active:
                continue

            for direction_name, direction in self.directions.items():
                if self._can_corner_pivot(module_id, neighbor, direction_name):
                    # Calculate new position after corner pivot
                    neighbor_pos = self.modules[neighbor].position
                    new_translation = np.array(direction)
                    new_pos = neighbor_pos + new_translation

                    # Skip if this is a fault position
                    pos_tuple = tuple(new_pos.astype(int))
                    if pos_tuple in fault_positions:
                        continue

                    # Skip if position is occupied
                    if self._position_is_occupied(new_pos, exclude_module=module_id):
                        continue

                    new_distance = np.linalg.norm(new_pos - original_pos)

                    # Only include if it moves us closer to original
                    if new_distance < current_distance - 1e-6:
                        reaches_original = np.allclose(new_pos, original_pos, atol=1e-6)
                        valid_pivots.append({
                            'type': 'corner',
                            'axis': neighbor,
                            'direction': direction_name,
                            'new_pos': new_pos.copy(),
                            'distance_reduction': current_distance - new_distance,
                            'reaches_original': reaches_original
                        })

        # Check lateral pivots
        for old_neighbor in neighbors:
            if not self.modules[old_neighbor].is_active:
                continue

            neighbor_neighbors = self.get_neighbors(old_neighbor)
            for new_neighbor in neighbor_neighbors:
                if new_neighbor == module_id:
                    continue
                if not self.modules[new_neighbor].is_active:
                    continue

                if self._can_lateral_pivot(module_id, old_neighbor, new_neighbor):
                    # Calculate new position after lateral pivot
                    old_edge = self.edges.get((old_neighbor, module_id))
                    if old_edge:
                        direction = old_edge.translation
                        new_neighbor_pos = self.modules[new_neighbor].position
                        new_pos = new_neighbor_pos + direction

                        # Skip if this is a fault position
                        pos_tuple = tuple(new_pos.astype(int))
                        if pos_tuple in fault_positions:
                            continue

                        # Skip if position is occupied
                        if self._position_is_occupied(new_pos, exclude_module=module_id):
                            continue

                        new_distance = np.linalg.norm(new_pos - original_pos)

                        # Only include if it moves us closer to original
                        if new_distance < current_distance - 1e-6:
                            reaches_original = np.allclose(new_pos, original_pos, atol=1e-6)
                            valid_pivots.append({
                                'type': 'lateral',
                                'old_neighbor': old_neighbor,
                                'new_neighbor': new_neighbor,
                                'new_pos': new_pos.copy(),
                                'distance_reduction': current_distance - new_distance,
                                'reaches_original': reaches_original
                            })

        # Sort by priority: reaches_original first, then by distance reduction
        valid_pivots.sort(
            key=lambda p: (-p['reaches_original'], -p['distance_reduction'])
        )

        return valid_pivots

    def select_restoration_module(
        self,
        movement_histories: Dict[str, 'MovementHistory'],
        fault_positions: Set[Tuple[int, int, int]]
    ) -> Optional[str]:
        """
        Select which module should attempt restoration next.

        Selection priority:
        1. Modules that can reach their exact original position
        2. Modules with largest displacement (moved furthest from original)
        3. Tiebreaker: module ID (lexicographic for determinism)

        Args:
            movement_histories: Movement histories from Phase 1
            fault_positions: Set of fault positions to avoid

        Returns:
            Module ID of selected module, or None if no modules can restore
        """
        scored_candidates = []

        for module_id, history in movement_histories.items():
            if not history.has_moved():
                continue

            if not self.can_restore_position(module_id, movement_histories, fault_positions):
                continue

            pivots = self.get_restoration_pivots(
                module_id, movement_histories, fault_positions
            )
            if not pivots:
                continue

            displacement = history.get_displacement_magnitude()
            can_reach_original = any(p['reaches_original'] for p in pivots)
            best_pivot = pivots[0]

            scored_candidates.append({
                'module_id': module_id,
                'can_reach_original': can_reach_original,
                'displacement': displacement,
                'best_pivot': best_pivot
            })

        if not scored_candidates:
            return None

        # Sort: prefer reaching original, then by displacement (descending), then by ID
        scored_candidates.sort(
            key=lambda c: (
                -c['can_reach_original'],  # True > False, so negate
                -c['displacement'],         # Larger displacement first
                c['module_id']              # Alphabetical tiebreaker
            )
        )

        return scored_candidates[0]['module_id']

    def get_restoration_candidates(
        self,
        movement_histories: Dict[str, 'MovementHistory'],
        fault_positions: Set[Tuple[int, int, int]]
    ) -> List[Dict]:
        """
        Get all modules that can attempt restoration with their best pivots.

        Returns a sorted list of candidates with restoration information.

        Args:
            movement_histories: Movement histories from Phase 1
            fault_positions: Set of fault positions to avoid

        Returns:
            List of candidate dictionaries sorted by restoration priority
        """
        candidates = []

        for module_id, history in movement_histories.items():
            if not history.has_moved():
                continue

            if not self.can_restore_position(module_id, movement_histories, fault_positions):
                continue

            pivots = self.get_restoration_pivots(
                module_id, movement_histories, fault_positions
            )
            if not pivots:
                continue

            current_pos = self.modules[module_id].position
            original_pos = history.get_original_position()

            candidates.append({
                'module_id': module_id,
                'current_position': current_pos.copy(),
                'original_position': original_pos.copy(),
                'displacement': history.get_displacement_magnitude(),
                'distance_to_original': np.linalg.norm(current_pos - original_pos),
                'can_reach_original': any(p['reaches_original'] for p in pivots),
                'num_valid_pivots': len(pivots),
                'best_pivot': pivots[0]
            })

        # Sort by restoration priority
        candidates.sort(
            key=lambda c: (
                -c['can_reach_original'],
                -c['displacement'],
                c['module_id']
            )
        )

        return candidates

    def execute_restoration_pivot(
        self,
        module_id: str,
        pivot: Dict,
        movement_histories: Dict[str, 'MovementHistory']
    ) -> bool:
        """
        Execute a single restoration pivot and update movement history.

        Args:
            module_id: Module to move
            pivot: Pivot dictionary from get_restoration_pivots()
            movement_histories: Movement histories to update

        Returns:
            True if pivot was successful
        """
        if module_id not in self.modules:
            return False

        if module_id not in movement_histories:
            return False

        # Record position before pivot
        from_pos = self.modules[module_id].position.copy()

        # Execute the pivot
        success = False
        if pivot['type'] == 'corner':
            success = self.corner_pivot(
                module_id,
                pivot['axis'],
                pivot['direction']
            )
        elif pivot['type'] == 'lateral':
            success = self.lateral_pivot(
                module_id,
                pivot['old_neighbor'],
                pivot['new_neighbor']
            )

        if success:
            # Update movement history with the restoration move
            to_pos = self.modules[module_id].position.copy()
            delta = to_pos - from_pos
            movement_histories[module_id].add_move(delta)

            # Form any new connections after the move
            self._form_new_connections()

        return success

    # Phase 2: Main Restoration Algorithm

    def position_restoration(
        self,
        movement_histories: Dict[str, 'MovementHistory'],
        fault_positions: Set[Tuple[int, int, int]],
        max_iterations: int = 100,
        record_steps: bool = True
    ) -> Dict[str, any]:
        """
        Phase 2: Restore modules toward their original positions after reconnection.

        This algorithm iteratively moves modules back toward their original
        positions while maintaining connectivity. It uses the same eligibility
        checks as Phase 1 (leaf node exception, 3-hop connectivity).

        The algorithm terminates when:
        1. All displaced modules are at their original positions
        2. No more valid restoration moves are available
        3. Maximum iterations reached

        Args:
            movement_histories: Movement histories from Phase 1 (will be updated)
            fault_positions: Set of fault positions to avoid
            max_iterations: Maximum restoration iterations
            record_steps: Whether to record step-by-step history

        Returns:
            Dictionary with restoration statistics:
            - iterations: Number of iterations executed
            - restoration_moves: Total moves made
            - fully_restored: List of module IDs that reached original position
            - partially_restored: List of module IDs that moved closer
            - could_not_restore: List of module IDs that couldn't move
            - metrics: RestorationMetrics object with summary statistics
            - steps: List of RestorationStep objects (if record_steps=True)
            - success: True if all displaced modules were restored
        """
        stats = {
            "iterations": 0,
            "restoration_moves": 0,
            "fully_restored": [],
            "partially_restored": [],
            "could_not_restore": [],
            "initial_displacement": 0.0,
            "final_displacement": 0.0,
            "success": False
        }

        if record_steps:
            stats["steps"] = []

        # Identify modules that need restoration (have non-zero displacement)
        displaced_modules = {
            module_id: history
            for module_id, history in movement_histories.items()
            if history.has_moved()
        }

        if not displaced_modules:
            stats["success"] = True
            stats["metrics"] = RestorationMetrics(all_restored=True)
            return stats

        # Calculate initial total displacement
        stats["initial_displacement"] = sum(
            history.get_restoration_distance(self.modules[module_id].position)
            for module_id, history in displaced_modules.items()
            if module_id in self.modules
        )

        # Track which modules have been fully restored
        restored_modules = set()

        for iteration in range(max_iterations):
            stats["iterations"] = iteration + 1

            # Check termination: all displaced modules restored
            all_at_original = True
            for module_id, history in displaced_modules.items():
                if module_id not in self.modules:
                    continue
                current_pos = self.modules[module_id].position
                original_pos = history.get_original_position()
                if not np.allclose(current_pos, original_pos, atol=1e-6):
                    all_at_original = False
                    break

            if all_at_original:
                stats["success"] = True
                break

            # Select module to restore
            selected = self.select_restoration_module(
                movement_histories, fault_positions
            )

            if selected is None:
                # No more valid restoration moves
                break

            # Get best pivot for selected module
            pivots = self.get_restoration_pivots(
                selected, movement_histories, fault_positions
            )

            if not pivots:
                # This shouldn't happen if select_restoration_module is correct
                break

            best_pivot = pivots[0]
            history = movement_histories[selected]
            original_pos = history.get_original_position()

            # Record position before pivot
            from_pos = self.modules[selected].position.copy()
            distance_before = np.linalg.norm(from_pos - original_pos)

            # Execute the restoration pivot
            success = self.execute_restoration_pivot(
                selected, best_pivot, movement_histories
            )

            if success:
                stats["restoration_moves"] += 1

                to_pos = self.modules[selected].position.copy()
                distance_after = np.linalg.norm(to_pos - original_pos)
                is_complete = np.allclose(to_pos, original_pos, atol=1e-6)

                if is_complete:
                    restored_modules.add(selected)

                # Record step
                if record_steps:
                    step = RestorationStep(
                        iteration=iteration + 1,
                        pivot_type=best_pivot['type'],
                        module_id=selected,
                        param1=best_pivot.get('axis') or best_pivot.get('old_neighbor'),
                        param2=best_pivot.get('direction') or best_pivot.get('new_neighbor'),
                        from_pos=from_pos,
                        to_pos=to_pos,
                        distance_before=distance_before,
                        distance_after=distance_after,
                        is_restoration_complete=is_complete
                    )
                    stats["steps"].append(step)

        # Calculate final displacement and categorize modules
        stats["final_displacement"] = 0.0

        for module_id, history in displaced_modules.items():
            if module_id not in self.modules:
                continue

            current_pos = self.modules[module_id].position
            original_pos = history.get_original_position()
            distance = np.linalg.norm(current_pos - original_pos)
            stats["final_displacement"] += distance

            if np.allclose(current_pos, original_pos, atol=1e-6):
                stats["fully_restored"].append(module_id)
            elif distance < history.get_displacement_magnitude():
                # Closer to original than initial displacement
                stats["partially_restored"].append(module_id)
            else:
                stats["could_not_restore"].append(module_id)

        # Check if all restored
        stats["success"] = len(stats["could_not_restore"]) == 0 and \
                          len(stats["partially_restored"]) == 0

        # Generate metrics
        initial_disp = stats["initial_displacement"]
        final_disp = stats["final_displacement"]
        reduction = initial_disp - final_disp

        stats["metrics"] = RestorationMetrics(
            total_displaced_modules=len(displaced_modules),
            fully_restored_count=len(stats["fully_restored"]),
            partially_restored_count=len(stats["partially_restored"]),
            unrestored_count=len(stats["could_not_restore"]),
            total_initial_displacement=initial_disp,
            total_final_displacement=final_disp,
            displacement_reduction=reduction,
            displacement_reduction_percent=(reduction / initial_disp * 100) if initial_disp > 0 else 0.0,
            total_restoration_moves=stats["restoration_moves"],
            iterations_used=stats["iterations"],
            all_restored=stats["success"]
        )

        return stats

    def full_damage_response(
        self,
        fault_module_id: str,
        restore_positions: bool = True,
        max_phase1_iterations: int = 1000,
        max_phase2_iterations: int = 100,
        one_per_subgraph: bool = True,
        parallel_subgraphs: bool = True,
        record_steps: bool = True
    ) -> Dict[str, any]:
        """
        Complete damage response: Phase 1 (reconnect) + Phase 2 (restore).

        This is the main entry point for the full damage-responsive
        reconfiguration algorithm. It:
        1. Marks the specified module as faulty
        2. Runs Phase 1 ego-based fault response to reconnect
        3. If successful and restore_positions=True, runs Phase 2 to restore

        Args:
            fault_module_id: ID of the faulty module
            restore_positions: Whether to run Phase 2 restoration (default True)
            max_phase1_iterations: Max iterations for Phase 1
            max_phase2_iterations: Max iterations for Phase 2
            one_per_subgraph: Phase 1 option - one module per component per iteration
            parallel_subgraphs: Phase 1 option - parallel subgraph movement
            record_steps: Whether to record step history for both phases

        Returns:
            Dictionary with:
            - phase1: Phase 1 statistics
            - phase2: Phase 2 statistics (or None if not run)
            - movement_histories: Final movement histories
            - overall_success: True if both phases succeeded
            - total_moves: Combined moves from both phases
            - fault_position: Position of the faulty module
        """
        result = {
            "phase1": None,
            "phase2": None,
            "movement_histories": {},
            "overall_success": False,
            "total_moves": 0,
            "fault_position": None
        }

        # Validate fault module
        if fault_module_id not in self.modules:
            result["error"] = "Fault module not found"
            return result

        # Record fault position before marking
        result["fault_position"] = tuple(
            self.modules[fault_module_id].position.astype(int)
        )

        # Mark the module as faulty (if not already)
        if not self.modules[fault_module_id].is_faulty:
            self.mark_fault(fault_module_id)

        # Phase 1: Reconnection with history tracking
        phase1_stats, movement_histories = self.ego_fault_response_with_history(
            fault_module_id=fault_module_id,
            max_iterations=max_phase1_iterations,
            one_per_subgraph=one_per_subgraph,
            parallel_subgraphs=parallel_subgraphs,
            record_steps=record_steps
        )

        result["phase1"] = phase1_stats
        result["movement_histories"] = movement_histories
        result["total_moves"] = phase1_stats.get("total_moves", 0)

        # Check Phase 1 success
        if not phase1_stats.get("reconnected", False):
            result["overall_success"] = False
            return result

        # Phase 2: Position restoration (if requested)
        if restore_positions:
            fault_positions = {result["fault_position"]}

            phase2_stats = self.position_restoration(
                movement_histories=movement_histories,
                fault_positions=fault_positions,
                max_iterations=max_phase2_iterations,
                record_steps=record_steps
            )

            result["phase2"] = phase2_stats
            result["total_moves"] += phase2_stats.get("restoration_moves", 0)

            # Overall success requires both phases
            result["overall_success"] = phase2_stats.get("success", False)
        else:
            # Only Phase 1 was requested
            result["overall_success"] = True

        return result

    def get_damage_response_summary(
        self,
        response_result: Dict[str, any]
    ) -> str:
        """
        Generate a human-readable summary of a damage response.

        Args:
            response_result: Result from full_damage_response()

        Returns:
            Formatted string summary
        """
        lines = []
        lines.append("=" * 60)
        lines.append("DAMAGE RESPONSE SUMMARY")
        lines.append("=" * 60)

        # Fault info
        fault_pos = response_result.get("fault_position")
        if fault_pos:
            lines.append(f"Fault Position: {fault_pos}")

        # Phase 1 summary
        p1 = response_result.get("phase1", {})
        if p1:
            lines.append("")
            lines.append("PHASE 1: Reconnection")
            lines.append("-" * 40)
            lines.append(f"  Iterations: {p1.get('iterations', 0)}")
            lines.append(f"  Moves: {p1.get('total_moves', 0)}")
            lines.append(f"  Modules responded: {len(p1.get('modules_responded', []))}")
            lines.append(f"  Reconnected: {'YES' if p1.get('reconnected') else 'NO'}")

        # Phase 2 summary
        p2 = response_result.get("phase2")
        if p2:
            lines.append("")
            lines.append("PHASE 2: Position Restoration")
            lines.append("-" * 40)
            lines.append(f"  Iterations: {p2.get('iterations', 0)}")
            lines.append(f"  Moves: {p2.get('restoration_moves', 0)}")
            lines.append(f"  Fully restored: {len(p2.get('fully_restored', []))}")
            lines.append(f"  Partially restored: {len(p2.get('partially_restored', []))}")
            lines.append(f"  Could not restore: {len(p2.get('could_not_restore', []))}")

            metrics = p2.get("metrics")
            if metrics:
                lines.append(f"  Displacement reduction: {metrics.displacement_reduction:.2f}")
                lines.append(f"  Reduction percent: {metrics.displacement_reduction_percent:.1f}%")

        # Overall result
        lines.append("")
        lines.append("OVERALL RESULT")
        lines.append("-" * 40)
        lines.append(f"  Total moves: {response_result.get('total_moves', 0)}")
        success = response_result.get("overall_success", False)
        lines.append(f"  Success: {'YES' if success else 'NO'}")
        lines.append("=" * 60)

        return "\n".join(lines)