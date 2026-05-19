import numpy as np
from typing import Any, Dict, List, Set, Tuple, Optional
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
    is_moving: bool = False  # Set True during concurrent movement phase
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
        """Check if module can pivot (must be active)."""
        if module_id not in self.modules:
            return False
        return self.modules[module_id].is_active

    def _are_orthogonal(self, dir1: np.ndarray, dir2: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if two directions are orthogonal."""
        return np.abs(np.dot(dir1, dir2)) < tolerance

    def _get_occupied_ports(self, module_id: str) -> Set[Tuple[int, int, int]]:
        """Get port directions occupied by edges to active modules."""
        occupied = set()
        for neighbor in self.get_neighbors(module_id):
            if not self.modules[neighbor].is_active:
                continue
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

    def mark_fault(self, module_id: str) -> bool:
        """Mark a module as faulty (damaged)."""
        if module_id not in self.modules:
            return False
        self.modules[module_id].is_faulty = True
        self.modules[module_id].is_active = False
        return True

    def get_active_neighbors(self, module_id: str) -> List[str]:
        """Get active, non-faulty neighbors of a module."""
        return [n for n in self.get_neighbors(module_id)
                if self.modules[n].is_active and not self.modules[n].is_faulty]

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

