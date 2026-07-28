import numpy as np
from boids.agent import Boid
from boids.polygon_room import PolygonRoom


class PolygonSimulation:
    """Version paralel i Simulation, që përdor PolygonRoom (formë e lirë) në vend të Environment."""

    def __init__(self, num_boids, vertices, doors, obstacles=None,
                 circle_obstacles=None, exit_width=15.0,
                 separation_weight=1.5, alignment_weight=1.0,
                 cohesion_weight=1.0, exit_weight=1.2, obstacle_weight=3.5):
        self.room = PolygonRoom(vertices, doors, obstacles, circle_obstacles, exit_width)
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.exit_weight = exit_weight
        self.obstacle_weight = obstacle_weight

        self.boids = [Boid(self._random_free_position(),
                            [np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
                      for _ in range(num_boids)]

        self.evacuation_times = []
        self.time_elapsed = 0

    def _random_free_position(self, max_attempts=100):
        for _ in range(max_attempts):
            x = np.random.uniform(self.room.min_x, self.room.max_x)
            y = np.random.uniform(self.room.min_y, self.room.max_y)
            if self.room.contains_point((x, y)) and not self._is_inside_any_obstacle((x, y)):
                return [x, y]
        return [(self.room.min_x + self.room.max_x) / 2, (self.room.min_y + self.room.max_y) / 2]

    def _is_inside_any_obstacle(self, position, padding=10.0):
        x, y = position
        for (ox, oy, ow, oh) in self.room.obstacles:
            if (ox - padding) < x < (ox + ow + padding) and (oy - padding) < y < (oy + oh + padding):
                return True
        for (cx, cy, radius) in self.room.circle_obstacles:
            if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < radius + padding:
                return True
        return False

    def get_neighbors(self, boid, radius=50.0):
        return [o for o in self.boids if o is not boid and
                np.linalg.norm(boid.position - o.position) < radius]

    def _get_exit_target(self, position, final_approach_radius=70.0):
        nearest = self.room.nearest_exit(position)
        if np.linalg.norm(position - nearest) < final_approach_radius:
            return nearest
        return self.room.get_next_waypoint(position)

    def step(self):
        self.time_elapsed += 1
        evacuated = []
        for boid in self.boids:
            neighbors = self.get_neighbors(boid)
            acceleration = (
                boid.separation(neighbors) * self.separation_weight +
                boid.align(neighbors) * self.alignment_weight +
                boid.cohesion(neighbors) * self.cohesion_weight +
                boid.seek_exit(self._get_exit_target(boid.position)) * self.exit_weight +
                self.room.obstacle_avoidance_force(boid.position) * self.obstacle_weight +
                self.room.keep_within_bounds(boid) +
                np.random.uniform(-0.05, 0.05, size=2)
            )
            force_magnitude = np.linalg.norm(acceleration)
            if force_magnitude > boid.max_force:
                acceleration = (acceleration / force_magnitude) * boid.max_force

            boid.update(acceleration)
            self.room.resolve_collisions(boid)
            self.room.enforce_boundaries(boid)

            if self.room.has_reached_exit(boid.position):
                evacuated.append(boid)
                self.evacuation_times.append(self.time_elapsed)

        for boid in evacuated:
            self.boids.remove(boid)

    def is_finished(self):
        return len(self.boids) == 0