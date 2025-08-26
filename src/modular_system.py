import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import uuid
from loguru import logger

from .module import Module, Direction, ConnectionPort, Message


@dataclass
class GraphState:
    """Represents the state of the modular system at time t."""

    time: int
    vertices: Set[str]  # V_t
    edges: Set[Tuple[str, str]]  # E_t (undirected)
    activity_map: Dict[str, int]  # m_t: V_t -> {0,1}
    attitudes: Dict[str, np.ndarray]  # Q_t: V_t -> S^3 (quaternions)
    edge_gains: Dict[Tuple[str, str], np.ndarray]  # g_t: directed edges -> S^2


class ModularSystem:
    """
    Represents the modular spacecraft as a graph G_t = (V_t, E_t, m_t, Q_t, g_t).

    This class implements the formal mathematical model from the problem formulation,
    including lattice constraints, connectivity analysis, and damage simulation using NetworkX.
    """

    def __init__(self, step_length: float = 1.0):
        self.step_length = step_length
        self.current_time = 0

        # Core graph components using NetworkX
        self.modules: Dict[str, Module] = {}  # Module objects
        self.graph = nx.Graph()  # NetworkX undirected graph for connectivity analysis
        self.directed_graph = nx.DiGraph()  # For edge gains and directed operations

        # State tracking
        self.history: List[GraphState] = []
        self.damage_events: List[Tuple[int, List[str]]] = []  # (time, damaged_modules)

        # Cubic lattice direction constraints
        self.direction_set = self._get_cubic_direction_set()

        logger.info("Initialized ModularSystem with cubic lattice")

    def _get_cubic_direction_set(self) -> Set[Tuple[float, float, float]]:
        """Get the allowable direction set D for the cubic lattice."""
        return {
            (1, 0, 0),
            (-1, 0, 0),  # ±x
            (0, 1, 0),
            (0, -1, 0),  # ±y
            (0, 0, 1),
            (0, 0, -1),  # ±z
        }

    def add_module(
        self, module_id: str = None, position: np.ndarray = None, active: bool = True
    ) -> str:
        """Add a module to the system."""
        if module_id is None:
            module_id = str(uuid.uuid4())[:8]

        if position is None:
            position = np.zeros(3)

        # Create module object
        module = Module(module_id, position)
        module.is_active = active

        # Add to system
        self.modules[module_id] = module
        self.graph.add_node(module_id, position=position, active=active)
        self.directed_graph.add_node(module_id, position=position, active=active)

        logger.debug(f"Added module {module_id} at position {position}")
        return module_id

    def connect_modules(
        self, module_a: str, module_b: str, direction_a_to_b: Direction = None
    ) -> bool:
        """
        Connect two modules with lattice constraints.

        Args:
            module_a: ID of first module
            module_b: ID of second module
            direction_a_to_b: Direction from A to B (auto-computed if None)

        Returns:
            bool: True if connection successful
        """
        if module_a not in self.modules or module_b not in self.modules:
            return False

        pos_a = self.modules[module_a].position
        pos_b = self.modules[module_b].position

        # Compute displacement vector
        displacement = pos_b - pos_a
        distance = np.linalg.norm(displacement)

        # Verify unit step length
        if not np.isclose(distance, self.step_length, rtol=1e-6):
            logger.warning(
                f"Connection {module_a}-{module_b} violates unit step constraint: {distance}"
            )
            return False

        # Normalize to get edge gain
        if distance > 0:
            edge_gain = displacement / distance
        else:
            logger.error(f"Zero distance between modules {module_a} and {module_b}")
            return False

        # Verify lattice direction constraint
        if not self._is_valid_direction(edge_gain):
            logger.warning(
                f"Connection {module_a}-{module_b} violates lattice direction constraint"
            )
            return False

        # Add edges to NetworkX graphs
        self.graph.add_edge(module_a, module_b, gain=edge_gain)
        self.directed_graph.add_edge(module_a, module_b, gain=edge_gain)
        self.directed_graph.add_edge(module_b, module_a, gain=-edge_gain)

        # Connect at module level
        if direction_a_to_b is None:
            direction_a_to_b = self._vector_to_direction(edge_gain)

        direction_b_to_a = Direction(tuple(-np.array(direction_a_to_b.value)))

        self.modules[module_a].connect_to_module(direction_a_to_b, module_b)
        self.modules[module_b].connect_to_module(direction_b_to_a, module_a)

        logger.debug(f"Connected modules {module_a}-{module_b} with gain {edge_gain}")
        return True

    def _is_valid_direction(self, vector: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if a unit vector is in the allowable direction set."""
        for direction in self.direction_set:
            if np.allclose(vector, direction, atol=tolerance):
                return True
        return False

    def _vector_to_direction(self, vector: np.ndarray) -> Direction:
        """Convert a vector to the corresponding Direction enum."""
        vector = vector / np.linalg.norm(vector)  # Normalize

        # Find closest direction
        for direction in Direction:
            if np.allclose(vector, direction.vector, atol=1e-6):
                return direction

        raise ValueError(f"Vector {vector} does not correspond to a valid Direction")

    def simulate_damage(self, module_ids: List[str]) -> bool:
        """
        Simulate damage event by deactivating specified modules.

        Args:
            module_ids: List of module IDs to damage

        Returns:
            bool: True if damage applied successfully
        """
        self.current_time += 1
        damaged = []

        for module_id in module_ids:
            if module_id in self.modules and self.modules[module_id].is_active:
                self.modules[module_id].set_damage(True)
                self.graph.nodes[module_id]["active"] = False
                self.directed_graph.nodes[module_id]["active"] = False
                damaged.append(module_id)

        if damaged:
            self.damage_events.append((self.current_time, damaged))
            logger.warning(f"Damage event at t={self.current_time}: {damaged}")

        return len(damaged) > 0

    def get_active_subgraph(self) -> nx.Graph:
        """
        Get the active subgraph G^A_t = (V^A_t, E^A_t) using NetworkX.

        Returns:
            NetworkX graph containing only active nodes and edges
        """
        # Get active nodes
        active_nodes = [
            node
            for node, data in self.graph.nodes(data=True)
            if data.get("active", True)
        ]

        # Create subgraph with only active nodes
        active_subgraph = self.graph.subgraph(active_nodes).copy()

        return active_subgraph

    def is_connected(self) -> bool:
        """Check if the active subgraph is connected using NetworkX."""
        active_graph = self.get_active_subgraph()
        return nx.is_connected(active_graph) if len(active_graph.nodes) > 0 else True

    def get_connected_components(self) -> List[Set[str]]:
        """Get connected components of the active subgraph using NetworkX."""
        active_graph = self.get_active_subgraph()
        return [set(component) for component in nx.connected_components(active_graph)]

    def number_connected_components(self) -> int:
        """Get number of connected components using NetworkX."""
        active_graph = self.get_active_subgraph()
        return nx.number_connected_components(active_graph)

    def verify_lattice_constraints(self) -> Tuple[bool, List[str]]:
        """
        Verify that all edge gains satisfy lattice direction constraints
        and cycle closure conditions.

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []

        # Check direction constraints
        for u, v, data in self.directed_graph.edges(data=True):
            gain = data.get("gain")
            if gain is not None and not self._is_valid_direction(gain):
                violations.append(f"Edge {u}->{v} has invalid direction {gain}")

        # Check cycle closure using NetworkX cycle detection
        active_graph = self.get_active_subgraph()
        try:
            # Find cycles in the undirected graph
            cycles = nx.cycle_basis(active_graph)

            for cycle in cycles:
                if len(cycle) > 2:  # Ignore trivial cycles
                    cycle_sum = np.zeros(3)
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        if self.directed_graph.has_edge(u, v):
                            gain = self.directed_graph[u][v].get("gain", np.zeros(3))
                            cycle_sum += gain
                        elif self.directed_graph.has_edge(v, u):
                            gain = self.directed_graph[v][u].get("gain", np.zeros(3))
                            cycle_sum -= gain  # Reverse direction

                    if not np.allclose(cycle_sum, 0, atol=1e-6):
                        violations.append(
                            f"Cycle {cycle} does not satisfy closure: sum = {cycle_sum}"
                        )

        except nx.NetworkXError as e:
            violations.append(f"Error in cycle analysis: {e}")

        return len(violations) == 0, violations

    def get_pivotable_modules(self) -> List[str]:
        """
        Get list of modules that can perform pivot operations.

        A module can pivot if:
        1. It is active (m_t(u) = 1)
        2. All its neighbors are active
        """
        pivotable = []
        active_graph = self.get_active_subgraph()

        for module_id in active_graph.nodes():
            if not self.modules[module_id].is_active:
                continue

            # Check if all neighbors are active using NetworkX
            neighbors = list(active_graph.neighbors(module_id))
            all_neighbors_active = all(
                self.modules[neighbor_id].is_active for neighbor_id in neighbors
            )

            if all_neighbors_active:
                pivotable.append(module_id)

        return pivotable

    def get_pivot_axis_modules(self) -> List[str]:
        """
        Get list of modules that can serve as pivot axes.

        A module can be pivoted on if it is active (m_t(u) = 1).
        """
        return [
            module_id for module_id, module in self.modules.items() if module.is_active
        ]

    def compute_positions(self, root: str = None) -> Dict[str, np.ndarray]:
        """
        Compute positions of all vertices using lattice geometry and NetworkX BFS.

        Args:
            root: Root vertex (defaults to first active vertex)

        Returns:
            Dictionary mapping module_id to 3D position
        """
        active_graph = self.get_active_subgraph()

        if len(active_graph.nodes) == 0:
            return {}

        if root is None:
            root = next(iter(active_graph.nodes))
        elif root not in active_graph.nodes:
            logger.error(f"Root {root} is not in active graph")
            return {}

        positions = {root: np.zeros(3)}

        # Use NetworkX BFS for traversal
        for u, v in nx.bfs_edges(active_graph, root):
            if u in positions and v not in positions:
                # Get edge gain from directed graph
                if self.directed_graph.has_edge(u, v):
                    gain = self.directed_graph[u][v].get("gain", np.zeros(3))
                    positions[v] = positions[u] + gain * self.step_length
                elif self.directed_graph.has_edge(v, u):
                    gain = self.directed_graph[v][u].get("gain", np.zeros(3))
                    positions[v] = positions[u] - gain * self.step_length
                else:
                    logger.warning(f"No directed edge between {u} and {v}")
                    positions[v] = positions[u]

        return positions

    def find_bridges(self) -> List[Tuple[str, str]]:
        """Find bridge edges (critical connections) using NetworkX."""
        active_graph = self.get_active_subgraph()
        return list(nx.bridges(active_graph))

    def find_articulation_points(self) -> List[str]:
        """Find articulation points (critical nodes) using NetworkX."""
        active_graph = self.get_active_subgraph()
        return list(nx.articulation_points(active_graph))

    def shortest_path(self, source: str, target: str) -> List[str]:
        """Find shortest path between two modules using NetworkX."""
        active_graph = self.get_active_subgraph()
        try:
            return nx.shortest_path(active_graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def node_connectivity(self, source: str = None, target: str = None) -> int:
        """
        Compute node connectivity using NetworkX.

        Args:
            source: Source node (if None, computes overall connectivity)
            target: Target node (if None, computes overall connectivity)

        Returns:
            Node connectivity value
        """
        active_graph = self.get_active_subgraph()
        try:
            if source is None and target is None:
                return nx.node_connectivity(active_graph)
            elif source is not None and target is not None:
                return nx.node_connectivity(active_graph, source, target)
            else:
                raise ValueError(
                    "Both source and target must be specified or both None"
                )
        except nx.NetworkXError:
            return 0

    def edge_connectivity(self, source: str = None, target: str = None) -> int:
        """
        Compute edge connectivity using NetworkX.

        Args:
            source: Source node (if None, computes overall connectivity)
            target: Target node (if None, computes overall connectivity)

        Returns:
            Edge connectivity value
        """
        active_graph = self.get_active_subgraph()
        try:
            if source is None and target is None:
                return nx.edge_connectivity(active_graph)
            elif source is not None and target is not None:
                return nx.edge_connectivity(active_graph, source, target)
            else:
                raise ValueError(
                    "Both source and target must be specified or both None"
                )
        except nx.NetworkXError:
            return 0

    def save_state(self) -> GraphState:
        """Save current system state."""
        edges = set((u, v) if u < v else (v, u) for u, v in self.graph.edges())

        activity_map = {
            module_id: int(module.is_active)
            for module_id, module in self.modules.items()
        }

        attitudes = {
            module_id: module.orientation.copy()
            for module_id, module in self.modules.items()
        }

        edge_gains = {}
        for u, v, data in self.directed_graph.edges(data=True):
            if "gain" in data:
                edge_gains[(u, v)] = data["gain"].copy()

        state = GraphState(
            time=self.current_time,
            vertices=set(self.modules.keys()),
            edges=edges,
            activity_map=activity_map,
            attitudes=attitudes,
            edge_gains=edge_gains,
        )

        self.history.append(state)
        return state

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics for analysis."""
        active_graph = self.get_active_subgraph()
        components = self.get_connected_components()

        # Additional NetworkX metrics
        bridges = self.find_bridges()
        articulation_points = self.find_articulation_points()

        metrics = {
            "total_modules": len(self.modules),
            "active_modules": len(active_graph.nodes),
            "inactive_modules": len(self.modules) - len(active_graph.nodes),
            "total_connections": self.graph.number_of_edges(),
            "active_connections": active_graph.number_of_edges(),
            "connected_components": len(components),
            "largest_component_size": max(len(c) for c in components)
            if components
            else 0,
            "is_connected": self.is_connected(),
            "pivotable_modules": len(self.get_pivotable_modules()),
            "pivot_axis_capable": len(self.get_pivot_axis_modules()),
            "bridges": len(bridges),
            "articulation_points": len(articulation_points),
            "node_connectivity": self.node_connectivity(),
            "edge_connectivity": self.edge_connectivity(),
            "current_time": self.current_time,
            "damage_events": len(self.damage_events),
        }

        # Add density and clustering if graph has nodes
        if len(active_graph.nodes) > 0:
            metrics["density"] = nx.density(active_graph)
            if len(active_graph.nodes) > 2:
                try:
                    metrics["average_clustering"] = nx.average_clustering(active_graph)
                except:
                    metrics["average_clustering"] = 0.0
            else:
                metrics["average_clustering"] = 0.0
        else:
            metrics["density"] = 0.0
            metrics["average_clustering"] = 0.0

        return metrics

    def __str__(self) -> str:
        metrics = self.get_system_metrics()
        return (
            f"ModularSystem(t={metrics['current_time']}, "
            f"modules={metrics['active_modules']}/{metrics['total_modules']}, "
            f"connected={metrics['is_connected']}, "
            f"components={metrics['connected_components']})"
        )
