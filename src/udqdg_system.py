import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import random
from dataclasses import dataclass
try:
    from .dual_quaternion import UnitDualQuaternion, LATTICE_DIRECTIONS
except ImportError:
    from dual_quaternion import UnitDualQuaternion, LATTICE_DIRECTIONS

@dataclass
class SphericalModule:
    """A spherical module in the UDQDG system."""
    id: str
    position: np.ndarray
    is_active: bool = True
    radius: float = 0.5  # Half of unit lattice step for perfect touching
    color: Tuple[float, float, float] = (0.7, 0.7, 0.9)

class UDQDGSystem:
    """
    Simple Unit Dual Quaternion Directed Graph system.
    
    Implements G_t = (V_t, E_t, m_t, Q_t, q̂_t) with:
    - Spherical modules
    - Dual quaternion edge gains for pure translations  
    - Corner and lateral pivot operations
    """
    
    def __init__(self):
        self.modules: Dict[str, SphericalModule] = {}
        self.edges: Dict[Tuple[str, str], UnitDualQuaternion] = {}
        self.time = 0
        
    def add_module(self, module_id: str, position: np.ndarray, active: bool = True) -> SphericalModule:
        """Add a spherical module to the system."""
        module = SphericalModule(module_id, np.array(position, dtype=float), active)
        self.modules[module_id] = module
        return module
        
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
        return direction in LATTICE_DIRECTIONS.values()
    
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
            
        # Remove old connection
        if (pivot_module, axis_module) in self.edges:
            del self.edges[(pivot_module, axis_module)]
            del self.edges[(axis_module, pivot_module)]
        
        # Calculate new position
        axis_pos = self.modules[axis_module].position
        new_translation = np.array(LATTICE_DIRECTIONS[new_direction])
        new_position = axis_pos + new_translation
        
        # Update module position
        self.modules[pivot_module].position = new_position
        
        # Create new connection
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

        # Remove old connection (edges are bidirectional)
        if (pivot_module, old_neighbor) in self.edges:
            del self.edges[(pivot_module, old_neighbor)]
        if (old_neighbor, pivot_module) in self.edges:
            del self.edges[(old_neighbor, pivot_module)]
        
        # Calculate new position
        new_neighbor_pos = self.modules[new_neighbor].position
        new_position = new_neighbor_pos + direction
        
        # Update module position
        self.modules[pivot_module].position = new_position
        
        # Create new connection # TODO fix this connection I think?
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
        if new_direction not in LATTICE_DIRECTIONS:
            return False

        # Check if pivot module is pivotable (all neighbors active)
        if not self._is_pivotable(pivot_module):
            return False

        # Get current edge and check orthogonality
        current_edge = self.edges.get((pivot_module, axis_module))
        if current_edge:
            current_translation = current_edge.translation
            new_translation = np.array(LATTICE_DIRECTIONS[new_direction])

            if not self._are_orthogonal(current_translation, new_translation):
                return False

        # Check port exclusivity
        new_dir_tuple = LATTICE_DIRECTIONS[new_direction]
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
        new_direction = random.choice(list(LATTICE_DIRECTIONS.keys()))
        
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