import numpy as np
from boids.agent import Boid
import time
from boids.polygon_room import PolygonRoom
from boids.geometry import normalize_obstacle, point_in_rotated_rect
from boids.spatial_grid import SpatialGrid

DEBUG_PERF = False

class PolygonSimulation:
    """Version paralel i Simulation, që përdor PolygonRoom (formë e lirë) në vend të Environment."""

    def __init__(self, num_boids, vertices, doors, obstacles=None,
                 circle_obstacles=None, exit_width=15.0,
                 separation_weight=1.5, alignment_weight=1.0,
                 cohesion_weight_open=0.05, cohesion_weight_near_obstacle=1.0,
                 obstacle_proximity_threshold=45.0,
                 exit_weight=1.2, obstacle_weight=5.0):
        self.room = PolygonRoom(vertices, doors, obstacles, circle_obstacles, exit_width)
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight_open = cohesion_weight_open
        self.cohesion_weight_near_obstacle = cohesion_weight_near_obstacle
        self.obstacle_proximity_threshold = obstacle_proximity_threshold
        self.exit_weight = exit_weight
        self.obstacle_weight = obstacle_weight

        self.boids = [Boid(self._random_free_position(),
                            [np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
                      for _ in range(num_boids)]

        self.evacuation_times = []
        self.time_elapsed = 0

        self._perf_timers = {"neighbors": 0.0, "sep_align": 0.0, "pathfinding": 0.0,
                             "seek_exit": 0.0, "obstacle_dist": 0.0, "cohesion": 0.0,
                             "obstacle_avoid": 0.0, "other_forces": 0.0,
                             "collisions": 0.0, "boundaries": 0.0}
        self._perf_frame_count = 0
        self.spatial_grid = SpatialGrid(cell_size=50.0)

    def _random_free_position(self, max_attempts=100):
        for _ in range(max_attempts):
            x = np.random.uniform(self.room.min_x, self.room.max_x)
            y = np.random.uniform(self.room.min_y, self.room.max_y)
            if self.room.contains_point((x, y)) and not self._is_inside_any_obstacle((x, y)):
                return [x, y]
        return [(self.room.min_x + self.room.max_x) / 2, (self.room.min_y + self.room.max_y) / 2]

    def _is_inside_any_obstacle(self, position, padding=10.0):
        x, y = position
        for obstacle in self.room.obstacles:
            ox, oy, ow, oh, angle = normalize_obstacle(obstacle)
            if point_in_rotated_rect(np.array([x, y]), ox, oy, ow, oh, angle, inflate=padding):
                return True

        for (cx, cy, radius) in self.room.circle_obstacles:
            if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < radius + padding:
                return True

        return False

    def get_neighbors(self, boid, radius=50.0):
        return self.spatial_grid.get_neighbors(boid, radius)

    def _get_exit_target(self, position, final_approach_radius=70.0):
        nearest = self.room.nearest_exit(position)
        if np.linalg.norm(position - nearest) < final_approach_radius:
            return nearest
        return self.room.get_next_waypoint(position)

    def step(self):
        self.time_elapsed += 1
        evacuated = []
        self.spatial_grid.build(self.boids)

        for boid in self.boids:
            t0 = time.perf_counter()
            neighbors = self.get_neighbors(boid)
            t1 = time.perf_counter()

            separation_force = boid.separation(neighbors) * self.separation_weight
            alignment_force = boid.align(neighbors) * self.alignment_weight
            t1b = time.perf_counter()

            exit_target = self._get_exit_target(boid.position)  # kjo thërret flow field
            t2 = time.perf_counter()

            exit_force = boid.seek_exit(exit_target) * self.exit_weight
            t2a = time.perf_counter()

            obstacle_distance = self.room.distance_to_nearest_obstacle(boid.position)
            near_obstacle = obstacle_distance < self.obstacle_proximity_threshold
            t2b = time.perf_counter()

            effective_cohesion_weight = (self.cohesion_weight_near_obstacle
                                         if near_obstacle
                                         else self.cohesion_weight_open)
            cohesion_force = boid.cohesion(neighbors) * effective_cohesion_weight
            t2b2 = time.perf_counter()

            obstacle_force = self.room.obstacle_avoidance_force(
                boid.position) * self.obstacle_weight
            t2c = time.perf_counter()

            boundary_info = self.room.closest_boundary_point(boid.position)  # NJË herë
            bounds_force = self.room.keep_within_bounds(boid, boundary_info=boundary_info)

            noise = np.random.uniform(-0.05, 0.05, size=2)

            acceleration = (separation_force + alignment_force + cohesion_force +
                            exit_force + obstacle_force + bounds_force + noise)

            force_magnitude = np.linalg.norm(acceleration)
            if force_magnitude > boid.max_force:
                acceleration = (acceleration / force_magnitude) * boid.max_force

            t3 = time.perf_counter()

            boid.update(acceleration)
            self.room.resolve_collisions(boid)
            t4 = time.perf_counter()

            self.room.enforce_boundaries(boid, boundary_info=boundary_info)
            t5 = time.perf_counter()

            stuck_threshold = 15 if near_obstacle else 40
            boid.check_and_escape_if_stuck(stuck_frames_threshold=stuck_threshold)

            if self.room.has_reached_exit(boid.position, boundary_info=boundary_info):
                evacuated.append(boid)
                self.evacuation_times.append(self.time_elapsed)

            if DEBUG_PERF:
                self._perf_timers["neighbors"] += (t1 - t0)
                self._perf_timers["sep_align"] += (t1b - t1)  # ri-emërtuar
                self._perf_timers["pathfinding"] += (t2 - t1b)  # tani vetëm waypoint lookup
                self._perf_timers["seek_exit"] += (t2a - t2)  # NEW
                self._perf_timers["obstacle_dist"] += (t2b - t2a)  # tani vetëm distance_to_nearest_obstacle
                self._perf_timers["cohesion"] += (t2b2 - t2b)  # NEW
                self._perf_timers["obstacle_avoid"] += (t2c - t2b2)  # tani vetëm obstacle_avoidance_force
                self._perf_timers["other_forces"] += (t3 - t2c)
                self._perf_timers["collisions"] += (t4 - t3)
                self._perf_timers["boundaries"] += (t5 - t4)

        for boid in evacuated:
            self.boids.remove(boid)

        if DEBUG_PERF:
            self._perf_frame_count += 1
            if self._perf_frame_count >= 60:
                total = sum(self._perf_timers.values())
                # print(f"[PERF] boids={len(self.boids)} total={total * 1000:.1f}ms/60frames "
                #       f"neighbors={self._perf_timers['neighbors'] * 1000:.1f}ms "
                #       f"sep_align={self._perf_timers['sep_align'] * 1000:.1f}ms "
                #       f"pathfinding={self._perf_timers['pathfinding'] * 1000:.1f}ms "
                #       f"seek_exit={self._perf_timers['seek_exit'] * 1000:.1f}ms "
                #       f"obstacle_dist={self._perf_timers['obstacle_dist'] * 1000:.1f}ms "
                #       f"cohesion={self._perf_timers['cohesion'] * 1000:.1f}ms "
                #       f"obstacle_avoid={self._perf_timers['obstacle_avoid'] * 1000:.1f}ms "
                #       f"other_forces={self._perf_timers['other_forces'] * 1000:.1f}ms "
                #       f"collisions={self._perf_timers['collisions'] * 1000:.1f}ms "
                #       f"boundaries={self._perf_timers['boundaries'] * 1000:.1f}ms")
                self._perf_frame_count = 0
                for k in self._perf_timers:
                    self._perf_timers[k] = 0.0

    def is_finished(self):
        return len(self.boids) == 0