import numpy as np
import time
from boids.agent import Boid
from boids.environment import Environment
from boids.geometry import normalize_obstacle, point_in_rotated_rect
from boids.spatial_grid import SpatialGrid

DEBUG_PERF = False # Vendos True për diagnostikim performance, shih [PERF] print (i komentuar poshtë).

class Simulation:
    """
    Menaxhon një grup (flock) boid-esh dhe simulimin e tyre në kohë,
    përfshirë tërheqjen drejt daljeve dhe evakuimin nga hapësira.
    """

    def __init__(self, num_boids, width, height, exits, obstacles=None,
                 exit_width=15.0, circle_obstacles=None,
                 separation_weight=1.5,
                 alignment_weight=1.0,
                 cohesion_weight_open=0.03,
                 cohesion_weight_near_obstacle=1.0,
                 obstacle_proximity_threshold=45.0,
                 exit_weight=1.2,
                 obstacle_weight=5.0):
        self.width = width
        self.height = height

        self.spatial_grid = SpatialGrid(cell_size=50.0)
        self.environment = Environment(width, height, exits, obstacles, exit_width,
                                       circle_obstacles=circle_obstacles)
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight_open = cohesion_weight_open
        self.cohesion_weight_near_obstacle = cohesion_weight_near_obstacle
        self.obstacle_proximity_threshold = obstacle_proximity_threshold
        self.exit_weight = exit_weight
        self.obstacle_weight = obstacle_weight
        self._perf_timers = {"neighbors": 0.0, "sep_align": 0.0, "pathfinding": 0.0,
                             "seek_exit": 0.0, "obstacle_dist": 0.0, "cohesion": 0.0,
                             "obstacle_avoid": 0.0, "other_forces": 0.0,
                             "collisions": 0.0, "boundaries": 0.0}
        self._perf_frame_count = 0

        # Krijojmë N boid-e me pozicione dhe shpejtësi fillestare random,
        # duke shmangur pozicionet që bien brenda ndonjë pengese (do të
        # krijonin boid të "ngujuar" që nga vetë fillimi).
        self.boids = []
        for _ in range(num_boids):
            position = self._random_free_position(width, height)
            velocity = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
            self.boids.append(Boid(position, velocity))

        # Regjistrimi i kohës së evakuimit për çdo boid (për analizë).
        self.evacuation_times = []
        self.time_elapsed = 0  # numërues i "frame"-ve/hapave kohorë.

    def _random_free_position(self, width, height, max_attempts=50):
        """
        Gjeneron një pozicion random që NUK bie brenda asnjë pengese.
        Provon deri në 'max_attempts' herë; nëse s'gjen dot (rast
        ekstrem, hapësirë shumë e mbushur me pengesa), kthen pozicionin
        e fundit të provuar si fallback, për të mos ngecur pafundësisht.
        """
        for _ in range(max_attempts):
            position = [np.random.uniform(0, width), np.random.uniform(0, height)]
            if not self._is_inside_any_obstacle(position):
                return position
        return position

    def _is_inside_any_obstacle(self, position, padding=10.0):
        x, y = position
        for obstacle in self.environment.obstacles:
            ox, oy, ow, oh, angle = normalize_obstacle(obstacle)
            if point_in_rotated_rect(np.array([x, y]), ox, oy, ow, oh, angle, inflate=padding):
                return True

        for (cx, cy, radius) in self.environment.circle_obstacles:
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist < radius + padding:
                return True

        return False

    def get_neighbors(self, boid, radius=50.0):
        return self.spatial_grid.get_neighbors(boid, radius)

    def keep_within_bounds(self, boid, margin=50, turn_force=0.5):
        """
        Forcë e BUTË (jo kufi absolut - ai është enforce_boundaries)
        që shtyn boid-in larg çdo muri kur i afrohet - tani e njëjtë
        për të 4 anët, pa nevojë të "dijë" për dyert specifikisht
        (enforce_boundaries e trajton lejimin real të kalimit te dyert).
        """
        steer = np.zeros(2)

        if boid.position[0] < margin:
            steer[0] = turn_force
        elif boid.position[0] > self.width - margin:
            steer[0] = -turn_force

        if boid.position[1] < margin:
            steer[1] = turn_force
        elif boid.position[1] > self.height - margin:
            steer[1] = -turn_force

        return steer

    def _get_exit_target(self, position, final_approach_radius=70.0):
        """
        Nëse boid-i është mjaftueshëm afër (distancë Euklidiane) me
        pikën e saktë të një dere, e ndjek atë pikë DIREKT, duke
        anashkaluar flow field-in - pranë vetë derës s'ka pengesa nga
        vetë definicioni, kështu tërheqja direkte është gjithmonë e
        sigurt. Kjo eliminon lëkundjet/zigzag që mund të shkaktohen
        nga gradienti i flow field pranë vetë qëllimit final.
        """
        nearest = self.environment.nearest_exit(position)
        distance_to_exit = np.linalg.norm(position - nearest)

        if distance_to_exit < final_approach_radius:
            return nearest

        return self.environment.get_next_waypoint(position)

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

            obstacle_distance = self.environment.distance_to_nearest_obstacle(boid.position)
            near_obstacle = obstacle_distance < self.obstacle_proximity_threshold
            t2b = time.perf_counter()
            effective_cohesion_weight = (self.cohesion_weight_near_obstacle
                                         if near_obstacle
                                         else self.cohesion_weight_open)
            cohesion_force = boid.cohesion(neighbors) * effective_cohesion_weight
            t2b2 = time.perf_counter()

            obstacle_force = self.environment.obstacle_avoidance_force(
                boid.position) * self.obstacle_weight
            t2c = time.perf_counter()

            bounds_force = self.keep_within_bounds(boid)

            noise = np.random.uniform(-0.05, 0.05, size=2)

            acceleration = (separation_force + alignment_force + cohesion_force +
                            exit_force + obstacle_force + bounds_force + noise)

            force_magnitude = np.linalg.norm(acceleration)
            if force_magnitude > boid.max_force:
                acceleration = (acceleration / force_magnitude) * boid.max_force

            t3 = time.perf_counter()

            boid.update(acceleration)
            self.environment.resolve_collisions(boid)
            t4 = time.perf_counter()

            self.environment.enforce_boundaries(boid)
            t5 = time.perf_counter()

            stuck_threshold = 25 if near_obstacle else 40
            boid.check_and_escape_if_stuck(stuck_frames_threshold=stuck_threshold)

            if self.environment.has_reached_exit(boid.position):
                evacuated.append(boid)
                self.evacuation_times.append(self.time_elapsed)

            if DEBUG_PERF:
                self._perf_timers["neighbors"] += (t1 - t0)
                self._perf_timers["sep_align"] += (t1b - t1)
                self._perf_timers["pathfinding"] += (t2 - t1b)
                self._perf_timers["seek_exit"] += (t2a - t2)
                self._perf_timers["obstacle_dist"] += (t2b - t2a)
                self._perf_timers["cohesion"] += (t2b2 - t2b)  # NEW
                self._perf_timers["obstacle_avoid"] += (t2c - t2b2)
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
        """Kthen True nëse të gjithë boid-et kanë evakuuar."""
        return len(self.boids) == 0