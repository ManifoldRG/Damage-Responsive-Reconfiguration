import numpy as np
import pyvista as pv
from typing import Dict, List, Tuple, Optional
try:
    from .udqdg_system import UDQDGSystem, SphericalModule
except ImportError:
    from udqdg_system import UDQDGSystem, SphericalModule


class UDQDGVisualizer:
    """Interactive 3D visualizer for UDQDG modular systems using PyVista."""

    def __init__(self, system: UDQDGSystem, window_size: Tuple[int, int] = (1200, 800)):
        self.system = system
        self.plotter = pv.Plotter(window_size=window_size)

        # Visual settings
        self.sphere_actors = {}
        self.edge_actors = {}

        # Color scheme (sleek modern palette)
        self.colors = {
            'active': '#00D9FF',      # Vibrant cyan
            'inactive': '#8B0000',    # Dark red
            'pivoting': '#FF8C00',    # Orange highlight
            'edge': '#E0E0E0',        # Light gray
            'background': '#1A1A1A'   # Dark background
        }

        self.sphere_radius = 0.5  # Half of unit lattice step for perfect touching
        self.edge_radius = 0.08
        self.sphere_opacity = 1.0  # Opaque spheres
        self.camera_position = None  # Store camera for animation

    def setup_scene(self):
        """Configure the 3D scene with lighting and camera."""
        # Dark modern background
        self.plotter.set_background(self.colors['background'])

        # Add multiple lights for better depth perception
        self.plotter.add_light(pv.Light(position=(5, 5, 5), intensity=0.8))
        self.plotter.add_light(pv.Light(position=(-5, -5, 5), intensity=0.4))

        # Add axes for orientation
        self.plotter.add_axes(color='white', line_width=3)

        # Camera setup for good initial view
        self.plotter.camera.position = (8, 8, 8)
        self.plotter.camera.focal_point = (0, 0, 0)

    def render_system(self, highlight_module: Optional[str] = None):
        """Render the current state of the modular system."""
        # Don't clear on initial render - just add all meshes with names
        self.sphere_actors.clear()
        self.edge_actors.clear()

        # Edges disabled - only showing spheres
        # for (module_a, module_b) in self.system.get_all_edges():
        #     self._render_edge(module_a, module_b, reset_camera=False)

        # Render modules
        for module_id, module in self.system.modules.items():
            is_highlighted = (module_id == highlight_module)
            self._render_module(module, is_highlighted, reset_camera=False)

        self.plotter.reset_camera()

    def _render_module(self, module: SphericalModule, highlight: bool = False, reset_camera: bool = True):
        """Render a single spherical module."""
        sphere = pv.Sphere(radius=self.sphere_radius, center=module.position)

        # Determine color
        if highlight:
            color = self.colors['pivoting']
        elif module.is_active:
            color = self.colors['active']
        else:
            color = self.colors['inactive']

        # Add with smooth shading, metallic finish, and transparency - use name for updates
        actor = self.plotter.add_mesh(
            sphere,
            color=color,
            smooth_shading=True,
            specular=0.5,
            specular_power=20,
            metallic=0.1,
            opacity=self.sphere_opacity,
            name=f"module_{module.id}",
            reset_camera=reset_camera
        )

        self.sphere_actors[module.id] = actor

    def _render_edge(self, module_a: str, module_b: str, reset_camera: bool = True):
        """Render an edge between two modules as a line."""
        pos_a = self.system.modules[module_a].position
        pos_b = self.system.modules[module_b].position

        # Create line between the two points
        line = pv.Line(pos_a, pos_b)

        edge_key = tuple(sorted([module_a, module_b]))
        actor = self.plotter.add_mesh(
            line,
            color=self.colors['edge'],
            line_width=3,
            name=f"edge_{edge_key[0]}_{edge_key[1]}",
            reset_camera=reset_camera
        )

        self.edge_actors[edge_key] = actor

    def animate_pivot_sequence(self, pivot_sequence: List[Tuple], n_frames: int = None, pause_between: float = None):
        """
        Animate any sequence of pivots from the UDQDG framework.

        Args:
            pivot_sequence: List of pivot operations:
                - ('corner', pivot_module, axis_module, new_direction)
                - ('lateral', pivot_module, old_neighbor, new_neighbor)
                - ('parallel', [(op1), (op2), ...]) for simultaneous pivots
            n_frames: Number of interpolation frames per pivot (uses default if None)
            pause_between: Pause duration between sequential pivots (uses 0.1 if None)
        """
        import time

        # Use instance settings if not provided
        pause_between = pause_between if pause_between is not None else 0.5

        for operation in pivot_sequence:
            if operation[0] == 'corner':
                _, pivot_module, axis_module, new_direction = operation
                self._animate_single_corner_pivot(pivot_module, axis_module, new_direction, n_frames)
                time.sleep(pause_between)

            elif operation[0] == 'lateral':
                _, pivot_module, old_neighbor, new_neighbor = operation
                self._animate_single_lateral_pivot(pivot_module, old_neighbor, new_neighbor, n_frames)
                time.sleep(pause_between)

            elif operation[0] == 'parallel':
                _, ops = operation
                self._animate_parallel_pivots(ops, n_frames)
                time.sleep(pause_between)

    def _animate_single_corner_pivot(self, pivot_module: str, axis_module: str,
                                      new_direction: str, n_frames: int = None):
        """Internal method for animating a single corner pivot."""
        # Default frame count from example config (TOTAL_FRAMES)
        if n_frames is None:
            n_frames = 30  # Will be overridden by example config

        initial_pos = self.system.modules[pivot_module].position.copy()
        success = self.system.corner_pivot(pivot_module, axis_module, new_direction)

        if not success:
            return False

        final_pos = self.system.modules[pivot_module].position.copy()
        axis_pos = self.system.modules[axis_module].position.copy()

        # Restore initial position for smooth animation
        self.system.modules[pivot_module].position = initial_pos.copy()

        v1 = initial_pos - axis_pos
        v2 = final_pos - axis_pos

        # Pre-create sphere mesh once
        sphere = pv.Sphere(radius=self.sphere_radius)

        # Get neighbors once (won't change during animation)
        neighbors = list(self.system.get_neighbors(pivot_module))

        for i in range(n_frames + 1):
            t = i / n_frames
            angle = t * np.pi / 2

            interp_v = np.cos(angle) * v1 + np.sin(angle) * v2
            interp_v = interp_v / np.linalg.norm(interp_v) * np.linalg.norm(v1)
            interp_pos = axis_pos + interp_v

            # Update module position in system
            self.system.modules[pivot_module].position = interp_pos

            # Update pivoting module mesh - translate existing mesh instead of recreating
            sphere_moved = sphere.copy()
            sphere_moved.points += interp_pos
            self.plotter.add_mesh(
                sphere_moved,
                color=self.colors['pivoting'],
                opacity=self.sphere_opacity,
                name=f"module_{pivot_module}",
                reset_camera=False
            )

            # Edges disabled - only showing spheres
            # for neighbor in neighbors:
            #     edge_key = tuple(sorted([pivot_module, neighbor]))
            #     neighbor_pos = self.system.modules[neighbor].position
            #     line = pv.Line(interp_pos, neighbor_pos)
            #     self.plotter.add_mesh(
            #         line,
            #         color=self.colors['edge'],
            #         line_width=3,
            #         name=f"edge_{edge_key[0]}_{edge_key[1]}",
            #         reset_camera=False
            #     )

            # Render frame and process events for interactivity
            self.plotter.update()

        # Restore default color after pivoting completes
        sphere_final = sphere.copy()
        sphere_final.points += final_pos
        self.plotter.add_mesh(
            sphere_final,
            color=self.colors['active'],
            opacity=self.sphere_opacity,
            name=f"module_{pivot_module}",
            reset_camera=False
        )
        self.plotter.update()

        return True

    def _animate_single_lateral_pivot(self, pivot_module: str, old_neighbor: str,
                                     new_neighbor: str, n_frames: int = None):
        """Internal method for animating a single lateral pivot."""
        # Default frame count from example config (TOTAL_FRAMES)
        if n_frames is None:
            n_frames = 30  # Will be overridden by example config

        initial_pos = self.system.modules[pivot_module].position.copy()

        # Store edge state before pivot
        old_edge_key = (pivot_module, old_neighbor)
        old_edge_key_rev = (old_neighbor, pivot_module)
        saved_old_edge = self.system.edges.get(old_edge_key)
        saved_old_edge_rev = self.system.edges.get(old_edge_key_rev)

        # Get initial neighbors before pivot modifies edges
        initial_neighbors = list(self.system.get_neighbors(pivot_module))

        success = self.system.lateral_pivot(pivot_module, old_neighbor, new_neighbor)

        if not success:
            return False

        final_pos = self.system.modules[pivot_module].position.copy()

        # Store new edge state after pivot
        new_edge_key = (pivot_module, new_neighbor)
        new_edge_key_rev = (new_neighbor, pivot_module)
        saved_new_edge = self.system.edges.get(new_edge_key)
        saved_new_edge_rev = self.system.edges.get(new_edge_key_rev)

        # Restore initial state for animation
        self.system.modules[pivot_module].position = initial_pos.copy()

        # Restore old edge temporarily
        if saved_old_edge is not None:
            self.system.edges[old_edge_key] = saved_old_edge
            self.system.edges[old_edge_key_rev] = saved_old_edge_rev

        # Remove new edge temporarily
        self.system.edges.pop(new_edge_key, None)
        self.system.edges.pop(new_edge_key_rev, None)

        # Pre-create sphere mesh once
        sphere = pv.Sphere(radius=self.sphere_radius)

        # Get neighbor positions
        old_neighbor_pos = self.system.modules[old_neighbor].position
        new_neighbor_pos = self.system.modules[new_neighbor].position

        # Get other neighbors (excluding old and new)
        other_neighbors = [n for n in initial_neighbors if n not in [old_neighbor, new_neighbor]]

        old_neighbor_pos = self.system.modules[old_neighbor].position
        new_neighbor_pos = self.system.modules[new_neighbor].position

        # Calculate the midpoint position (where transition happens)
        # The midpoint is equidistant from both old and new neighbors
        # It's at the intersection of two spheres centered at the neighbors

        # For a lateral pivot, the path maintains distance from the rotation center
        # Midpoint is where the module is at distance r from BOTH neighbors
        r = np.linalg.norm(initial_pos - old_neighbor_pos)  # Radius from neighbors

        # The midpoint is on a circle equidistant from both neighbors
        # In the plane perpendicular to the line connecting the neighbors
        neighbor_midpoint = (old_neighbor_pos + new_neighbor_pos) / 2.0
        neighbor_distance = np.linalg.norm(new_neighbor_pos - old_neighbor_pos)

        # Height above the line between neighbors
        # Using Pythagorean theorem: r^2 = (neighbor_distance/2)^2 + h^2
        h_squared = r**2 - (neighbor_distance / 2.0)**2

        if h_squared < 0:
            # Neighbors too far apart, use simple midpoint
            mid_pos = (initial_pos + final_pos) / 2.0
        else:
            h = np.sqrt(h_squared)

            # Direction perpendicular to neighbor line, in the same direction as initial offset
            neighbor_direction = new_neighbor_pos - old_neighbor_pos
            neighbor_direction_norm = neighbor_direction / np.linalg.norm(neighbor_direction)

            # Find perpendicular direction (use initial position as guide)
            v_to_initial = initial_pos - neighbor_midpoint
            # Remove component parallel to neighbor direction
            perp_component = v_to_initial - np.dot(v_to_initial, neighbor_direction_norm) * neighbor_direction_norm

            if np.linalg.norm(perp_component) > 0.001:
                perp_direction = perp_component / np.linalg.norm(perp_component)
                mid_pos = neighbor_midpoint + h * perp_direction
            else:
                # Fallback if perpendicular direction unclear
                mid_pos = (initial_pos + final_pos) / 2.0

        for i in range(n_frames + 1):
            t = i / n_frames

            if t <= 0.5:
                # First half: 60-degree arc rotation around old_neighbor
                local_t = t * 2  # Scale to 0-1 for first half

                # Slerp (spherical linear interpolation) for smooth arc
                v_start = initial_pos - old_neighbor_pos
                v_target = mid_pos - old_neighbor_pos

                # Calculate angle between vectors
                cos_theta = np.dot(v_start, v_target) / (np.linalg.norm(v_start) * np.linalg.norm(v_target))
                cos_theta = np.clip(cos_theta, -1, 1)  # Numerical safety
                theta = np.arccos(cos_theta)

                # Slerp formula
                if abs(theta) < 0.001:
                    interp_v = v_start
                else:
                    interp_v = (np.sin((1 - local_t) * theta) / np.sin(theta)) * v_start + \
                              (np.sin(local_t * theta) / np.sin(theta)) * v_target

                interp_pos = old_neighbor_pos + interp_v
            else:
                # Second half: 60-degree arc rotation around new_neighbor
                local_t = (t - 0.5) * 2  # Scale to 0-1 for second half

                # Slerp for smooth arc
                v_start = mid_pos - new_neighbor_pos
                v_target = final_pos - new_neighbor_pos

                # Calculate angle between vectors
                cos_theta = np.dot(v_start, v_target) / (np.linalg.norm(v_start) * np.linalg.norm(v_target))
                cos_theta = np.clip(cos_theta, -1, 1)
                theta = np.arccos(cos_theta)

                # Slerp formula
                if abs(theta) < 0.001:
                    interp_v = v_start
                else:
                    interp_v = (np.sin((1 - local_t) * theta) / np.sin(theta)) * v_start + \
                              (np.sin(local_t * theta) / np.sin(theta)) * v_target

                interp_pos = new_neighbor_pos + interp_v

            # Update module position in system
            self.system.modules[pivot_module].position = interp_pos

            # Update pivoting module mesh - translate existing mesh instead of recreating
            sphere_moved = sphere.copy()
            sphere_moved.points += interp_pos
            self.plotter.add_mesh(
                sphere_moved,
                color=self.colors['pivoting'],
                opacity=self.sphere_opacity,
                name=f"module_{pivot_module}",
                reset_camera=False
            )

            # Edges disabled - only showing spheres
            # old_edge_key = tuple(sorted([pivot_module, old_neighbor]))
            # new_edge_key = tuple(sorted([pivot_module, new_neighbor]))
            # if t <= 0.5:
            #     line = pv.Line(interp_pos, old_neighbor_pos)
            #     self.plotter.add_mesh(line, color=self.colors['edge'], line_width=3,
            #                          name=f"edge_{old_edge_key[0]}_{old_edge_key[1]}", reset_camera=False)
            # else:
            #     line = pv.Line(interp_pos, new_neighbor_pos)
            #     self.plotter.add_mesh(line, color=self.colors['edge'], line_width=3,
            #                          name=f"edge_{new_edge_key[0]}_{new_edge_key[1]}", reset_camera=False)

            # Render frame and process events for interactivity
            self.plotter.update()

        # Restore final state
        self.system.modules[pivot_module].position = final_pos.copy()

        # Remove old edge (should have been removed by pivot)
        self.system.edges.pop(old_edge_key, None)
        self.system.edges.pop(old_edge_key_rev, None)

        # Restore new edge
        if saved_new_edge is not None:
            self.system.edges[new_edge_key] = saved_new_edge
            self.system.edges[new_edge_key_rev] = saved_new_edge_rev

        # Restore default color after pivoting completes
        sphere_final = sphere.copy()
        sphere_final.points += final_pos
        self.plotter.add_mesh(
            sphere_final,
            color=self.colors['active'],
            opacity=self.sphere_opacity,
            name=f"module_{pivot_module}",
            reset_camera=False
        )
        self.plotter.update()

        return True

    def _animate_parallel_pivots(self, pivot_operations: List[Tuple], n_frames: int = None):
        """Internal method for animating parallel pivots."""
        # Default frame count from example config (TOTAL_FRAMES)
        if n_frames is None:
            n_frames = 30  # Will be overridden by example config

        # Keep same frame count as single pivots for matching duration
        # The parallel nature means more work per frame, but duration stays consistent

        # Store initial/final positions and pivot metadata
        pivot_data = {}

        for op in pivot_operations:
            if op[0] == 'corner':
                _, pivot_module, axis_module, new_direction = op
                initial_pos = self.system.modules[pivot_module].position.copy()
                axis_pos = self.system.modules[axis_module].position.copy()
                self.system.corner_pivot(pivot_module, axis_module, new_direction)
                final_pos = self.system.modules[pivot_module].position.copy()

                # Store corner pivot data for arc interpolation
                pivot_data[pivot_module] = {
                    'type': 'corner',
                    'initial': initial_pos,
                    'final': final_pos,
                    'axis': axis_pos,
                    'v1': initial_pos - axis_pos,
                    'v2': final_pos - axis_pos
                }

            elif op[0] == 'lateral':
                _, pivot_module, old_neighbor, new_neighbor = op
                initial_pos = self.system.modules[pivot_module].position.copy()
                self.system.lateral_pivot(pivot_module, old_neighbor, new_neighbor)
                final_pos = self.system.modules[pivot_module].position.copy()

                # Store lateral pivot data for linear interpolation
                pivot_data[pivot_module] = {
                    'type': 'lateral',
                    'initial': initial_pos,
                    'final': final_pos
                }

        # Pre-create sphere mesh once for all modules
        sphere = pv.Sphere(radius=self.sphere_radius)

        for i in range(n_frames + 1):
            t = i / n_frames

            # Update all pivoting module positions and meshes
            for module_id, data in pivot_data.items():
                if data['type'] == 'corner':
                    # Arc interpolation for corner pivots
                    angle = t * np.pi / 2
                    interp_v = np.cos(angle) * data['v1'] + np.sin(angle) * data['v2']
                    interp_v = interp_v / np.linalg.norm(interp_v) * np.linalg.norm(data['v1'])
                    interp_pos = data['axis'] + interp_v
                else:
                    # Linear interpolation for lateral pivots (with easing)
                    t_smooth = t * t * (3 - 2 * t)
                    interp_pos = (1 - t_smooth) * data['initial'] + t_smooth * data['final']

                self.system.modules[module_id].position = interp_pos

                # Update pivoting module mesh - translate existing mesh instead of recreating
                sphere_moved = sphere.copy()
                sphere_moved.points += interp_pos
                self.plotter.add_mesh(
                    sphere_moved,
                    color=self.colors['pivoting'],
                    opacity=self.sphere_opacity,
                    name=f"module_{module_id}",
                    reset_camera=False
                )

            # Edges disabled - only showing spheres
            # updated_edges = set()
            # for module_id in pivot_data.keys():
            #     for neighbor in self.system.get_neighbors(module_id):
            #         edge_key = tuple(sorted([module_id, neighbor]))
            #         if edge_key in updated_edges:
            #             continue
            #         updated_edges.add(edge_key)
            #         module_pos = self.system.modules[module_id].position
            #         neighbor_pos = self.system.modules[neighbor].position
            #         line = pv.Line(module_pos, neighbor_pos)
            #         self.plotter.add_mesh(line, color=self.colors['edge'], line_width=3,
            #                              name=f"edge_{edge_key[0]}_{edge_key[1]}", reset_camera=False)

            # Render frame and process events for interactivity
            self.plotter.update()

        # Restore default colors for all pivoted modules after animation completes
        for module_id, data in pivot_data.items():
            sphere_final = sphere.copy()
            sphere_final.points += data['final']
            self.plotter.add_mesh(
                sphere_final,
                color=self.colors['active'],
                opacity=self.sphere_opacity,
                name=f"module_{module_id}",
                reset_camera=False
            )
        self.plotter.update()

    def animate_corner_pivot(self, pivot_module: str, axis_module: str,
                               new_direction: str, n_frames: int = 30, pause_after: float = 0.3):
        """Animate a corner pivot with smooth 90-degree arc motion."""
        import time

        # Store initial state
        initial_pos = self.system.modules[pivot_module].position.copy()

        # Perform the pivot to get final position
        success = self.system.corner_pivot(pivot_module, axis_module, new_direction)
        if not success:
            print(f"Corner pivot failed: {pivot_module} around {axis_module}")
            return False

        final_pos = self.system.modules[pivot_module].position.copy()
        axis_pos = self.system.modules[axis_module].position.copy()

        # Get the actor for the pivoting module
        pivot_actor = self.sphere_actors.get(pivot_module)
        if not pivot_actor:
            return False

        # Calculate arc path (90-degree rotation around axis)
        v1 = initial_pos - axis_pos
        v2 = final_pos - axis_pos

        # Interpolate along the arc
        for i in range(n_frames + 1):
            t = i / n_frames
            angle = t * np.pi / 2  # 90 degrees

            # Interpolate between v1 and v2 along the arc
            interp_v = np.cos(angle) * v1 + np.sin(angle) * v2
            interp_v = interp_v / np.linalg.norm(interp_v) * np.linalg.norm(v1)

            interp_pos = axis_pos + interp_v

            # Update actor position directly (no clearing/re-rendering!)
            pivot_actor.position = interp_pos

            # Update connected edges
            self._update_edges_for_module(pivot_module, interp_pos)

            self.plotter.render()
            time.sleep(0.016)  # ~60 fps

        # Pause after animation
        time.sleep(pause_after)

        return True

    def _update_edges_for_module(self, module_id: str, position: np.ndarray):
        """Update edge positions for a module by updating mesh coordinates."""
        neighbors = self.system.get_neighbors(module_id)

        for neighbor in neighbors:
            edge_key = tuple(sorted([module_id, neighbor]))
            edge_name = f"edge_{edge_key[0]}_{edge_key[1]}"

            # Get neighbor position - use actual position from system, not actor
            # (actor position might be stale during animation)
            neighbor_actor = self.sphere_actors.get(neighbor)
            if neighbor_actor:
                # Get the actual current position from the actor
                neighbor_pos = np.array(neighbor_actor.position)
            else:
                neighbor_pos = self.system.modules[neighbor].position

            # Remove old edge
            self.plotter.remove_actor(edge_name, reset_camera=False)

            # Create updated edge line
            line = pv.Line(position, neighbor_pos)

            new_actor = self.plotter.add_mesh(
                line,
                color=self.colors['edge'],
                line_width=3,
                name=edge_name,
                reset_camera=False
            )

            self.edge_actors[edge_key] = new_actor

    def animate_lateral_pivot(self, pivot_module: str, old_neighbor: str,
                             new_neighbor: str, n_frames: int = 30, pause_after: float = 0.3):
        """Animate a lateral pivot with rolling/sliding motion."""
        import time

        # Store initial state
        initial_pos = self.system.modules[pivot_module].position.copy()

        # Perform the pivot to get final position
        success = self.system.lateral_pivot(pivot_module, old_neighbor, new_neighbor)
        if not success:
            print(f"Lateral pivot failed: {pivot_module} from {old_neighbor} to {new_neighbor}")
            return False

        final_pos = self.system.modules[pivot_module].position.copy()

        # Get the actor for the pivoting module
        pivot_actor = self.sphere_actors.get(pivot_module)
        if not pivot_actor:
            return False

        # Linear interpolation for lateral pivot (rolling along surface)
        for i in range(n_frames + 1):
            t = i / n_frames
            # Smooth easing function (ease-in-out)
            t_smooth = t * t * (3 - 2 * t)

            interp_pos = (1 - t_smooth) * initial_pos + t_smooth * final_pos

            # Update actor position directly
            pivot_actor.position = interp_pos

            # Update connected edges
            self._update_edges_for_module(pivot_module, interp_pos)

            self.plotter.render()
            time.sleep(0.016)  # ~60 fps

        # Pause after animation
        time.sleep(pause_after)

        return True

    def animate_parallel_pivots(self, pivot_operations: List[Tuple], n_frames: int = 30, pause_after: float = 0.5):
        """
        Animate multiple pivots simultaneously.

        Args:
            pivot_operations: List of (type, pivot_module, *args) tuples
                - ('corner', pivot_module, axis_module, new_direction)
                - ('lateral', pivot_module, old_neighbor, new_neighbor)
        """
        import time

        # Store initial positions
        initial_positions = {}
        final_positions = {}
        pivot_actors = {}

        # Execute all pivots and record positions
        for op in pivot_operations:
            if op[0] == 'corner':
                _, pivot_module, axis_module, new_direction = op
                initial_positions[pivot_module] = self.system.modules[pivot_module].position.copy()
                self.system.corner_pivot(pivot_module, axis_module, new_direction)
                final_positions[pivot_module] = self.system.modules[pivot_module].position.copy()
                pivot_actors[pivot_module] = self.sphere_actors.get(pivot_module)

            elif op[0] == 'lateral':
                _, pivot_module, old_neighbor, new_neighbor = op
                initial_positions[pivot_module] = self.system.modules[pivot_module].position.copy()
                self.system.lateral_pivot(pivot_module, old_neighbor, new_neighbor)
                final_positions[pivot_module] = self.system.modules[pivot_module].position.copy()
                pivot_actors[pivot_module] = self.sphere_actors.get(pivot_module)

        # Animate all movements simultaneously
        for i in range(n_frames + 1):
            t = i / n_frames
            t_smooth = t * t * (3 - 2 * t)  # Ease-in-out

            # Update all pivoting module positions
            for module_id in initial_positions:
                interp_pos = (1 - t_smooth) * initial_positions[module_id] + \
                           t_smooth * final_positions[module_id]

                # Update actor position directly
                actor = pivot_actors.get(module_id)
                if actor:
                    actor.position = interp_pos

                # Update edges for this module
                self._update_edges_for_module(module_id, interp_pos)

            self.plotter.render()
            time.sleep(0.016)  # ~60 fps

        # Pause after animation
        time.sleep(pause_after)

    def show(self, interactive: bool = True):
        """Display the visualization."""
        self.setup_scene()
        self.render_system()

        if interactive:
            self.plotter.show()
        else:
            self.plotter.show(auto_close=False)

    def show_window(self):
        """Initialize the interactive window without blocking."""
        self.setup_scene()
        self.render_system()
        self.plotter.show(interactive_update=True, auto_close=False)

    def export_animation(self, filename: str, operations: List[Tuple]):
        """Export animation sequence to file (GIF or MP4)."""
        self.plotter.open_gif(filename) if filename.endswith('.gif') else self.plotter.open_movie(filename)

        for op in operations:
            if op[0] == 'corner':
                self.animate_corner_pivot(*op[1:])
            elif op[0] == 'lateral':
                self.animate_lateral_pivot(*op[1:])

        self.plotter.close()
