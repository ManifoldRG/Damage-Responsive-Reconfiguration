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
            
        # Remove old connection
        if (pivot_module, axis_module) in self.edges:
            del self.edges[(pivot_module, axis_module)]
            del self.edges[(axis_module, pivot_module)]
        
        # Calculate new position
        axis_pos = self.modules[axis_module].position
        new_translation = np.array(self.directions[new_direction])
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
        3. Are my 1-hop connections a subset of my 3-hop connections?
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

        # Check 3: Are 1-hop ⊆ 3-hop? (connectivity check)
        one_hop = set(self.get_neighbors(module_id))
        three_hop = self.get_k_hop_neighbors(module_id, 3)

        # If all my direct neighbors can still reach each other through 3-hop paths,
        # then I'm not critical for connectivity
        return one_hop.issubset(three_hop)

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

    def ego_fault_response(self, fault_id: str, max_iterations: int = 1000) -> Dict[str, any]:
        """
        Execute ego-based fault response algorithm with wave propagation.

        The fault signal propagates outward one hop per timestep.
        At each timestep, only modules that have received the signal (neighbors of
        previously signaled modules) can make decisions and move.

        The algorithm continues until either:
        1. Active modules are reconnected (single connected component)
        2. No more moves are possible
        3. Max iterations reached

        Returns:
            Dictionary with statistics about the response
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
            "success": True
        }

        for iteration in range(max_iterations):
            stats["iterations"] = iteration + 1

            # Check if we've reconnected
            if self.is_connected(active_only=True):
                stats["reconnected"] = True
                break

            moves_this_iteration = 0

            # Propagate signal to all active modules
            # In reconnection mode, all modules can respond each iteration
            for module_id in sorted(self.modules.keys()):
                module = self.modules[module_id]

                # Skip if not active
                if not module.is_active or module.is_faulty:
                    continue

                # Ego decision: Should I respond?
                if not self.can_respond_to_fault(module_id):
                    continue

                # If disconnected, try to move toward other components for reconnection
                # Otherwise move toward fault
                if not self.is_connected(active_only=True):
                    # Find target position: closest module in a different component
                    components = self.get_connected_components(active_only=True)
                    my_component = None
                    for comp in components:
                        if module_id in comp:
                            my_component = comp
                            break

                    if my_component:
                        # Find closest module in other components
                        min_distance = float('inf')
                        target_pos = fault_pos
                        for comp in components:
                            if comp == my_component:
                                continue
                            for other_id in comp:
                                other_pos = self.modules[other_id].position
                                distance = np.linalg.norm(module.position - other_pos)
                                if distance < min_distance:
                                    min_distance = distance
                                    target_pos = other_pos
                    else:
                        target_pos = fault_pos
                else:
                    target_pos = fault_pos

                # Get available pivots towards target
                pivots = self.get_available_pivots(module_id, target_pos)

                if pivots:
                    # Execute first available pivot
                    pivot_type, pivot_module, param1, param2 = pivots[0]

                    success = False
                    if pivot_type == 'corner':
                        success = self.corner_pivot(pivot_module, param1, param2)
                    elif pivot_type == 'lateral':
                        success = self.lateral_pivot(pivot_module, param1, param2)

                    if success:
                        moves_this_iteration += 1
                        stats["modules_responded"].add(module_id)
                        stats["total_moves"] += 1

            # After all moves, scan for and form new connections
            new_conn = self._form_new_connections()
            stats["new_connections_formed"] += new_conn

            # Stop if no moves were made this iteration AND no new connections formed
            if moves_this_iteration == 0 and new_conn == 0:
                break

        stats["modules_responded"] = list(stats["modules_responded"])
        return stats

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