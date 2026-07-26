import numpy as np
from boids.agent import Boid
from boids.environment import Environment

class Simulation:
    """
    Menaxhon një grup (flock) boid-esh dhe simulimin e tyre në kohë,
    përfshirë tërheqjen drejt daljeve dhe evakuimin nga hapësira.
    """

    def __init__(self, num_boids, width, height, exits, obstacles=None,
                 exit_width=15.0,
                 separation_weight=1.5,
                 alignment_weight=1.0,
                 cohesion_weight=1.0,
                 exit_weight=1.2,
                 obstacle_weight=3.5):
        self.width = width
        self.height = height

        self.environment = Environment(width, height, exits, obstacles, exit_width)

        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.exit_weight = exit_weight
        self.obstacle_weight = obstacle_weight

        # Krijojmë N boid-e me pozicione dhe shpejtësi fillestare random,
        # duke shmangur pozicionet që bien brenda ndonjë pengese (do të
        # krijonin boid të "ngujuar" që nga vetë fillimi)
        self.boids = []
        for _ in range(num_boids):
            position = self._random_free_position(width, height)
            velocity = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
            self.boids.append(Boid(position, velocity))

        # Regjistrimi i kohës së evakuimit për çdo boid (për analizë)
        self.evacuation_times = []
        self.time_elapsed = 0  # numërues i "frame"-ve/hapave kohorë

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
        for (ox, oy, ow, oh) in self.environment.obstacles:
            if (ox - padding) < x < (ox + ow + padding) and \
                    (oy - padding) < y < (oy + oh + padding):
                return True
        return False

    def get_neighbors(self, boid, radius=50.0):
        neighbors = []
        for other in self.boids:
            if other is boid:
                continue
            distance = np.linalg.norm(boid.position - other.position)
            if distance < radius:
                neighbors.append(other)
        return neighbors

    def keep_within_bounds(self, boid, margin=50, turn_force=0.5):
        """
        Mban boid-in brenda kufijve. Muri i sipërm mbetet i PLOTË (i
        ngurtë) KUDO përveç saktësisht brenda gjerësisë reale të derës
        (self.environment.exit_width) - kështu boid-et NUK mund të
        'rrëshqasin jashtë' anash derës, vetëm saktësisht nëpër të.
        """
        steer = np.zeros(2)

        if boid.position[0] < margin:
            steer[0] = turn_force
        elif boid.position[0] > self.width - margin:
            steer[0] = -turn_force

        # Zona e "kalimit të lirë" = saktësisht gjysma e gjerësisë së
        # derës (jo më shumë) - kështu vetëm brenda vetë derës lejohet
        # kalimi, çdo gjë tjetër pranë saj mbetet mur i ngurtë
        half_exit_width = self.environment.exit_width / 2

        near_an_exit_x = any(
            abs(boid.position[0] - exit_pos[0]) < half_exit_width
            for exit_pos in self.environment.exits
        )

        if boid.position[1] < margin and not near_an_exit_x:
            steer[1] = turn_force
        elif boid.position[1] > self.height - margin:
            steer[1] = -turn_force

        return steer

    def step(self):
        """
        Ekzekuton një hap kohor: përditëson çdo boid, dhe heq nga
        simulimi ata që kanë arritur te dalja (evakuuar).
        """
        self.time_elapsed += 1
        evacuated = []

        for boid in self.boids:
            neighbors = self.get_neighbors(boid)

            separation_force = boid.separation(neighbors) * self.separation_weight
            alignment_force = boid.align(neighbors) * self.alignment_weight
            cohesion_force = boid.cohesion(neighbors) * self.cohesion_weight

            exit_target = self.environment.get_next_waypoint(boid.position)
            exit_force = boid.seek_exit(exit_target) * self.exit_weight

            obstacle_force = self.environment.obstacle_avoidance_force(
                boid.position) * self.obstacle_weight

            bounds_force = self.keep_within_bounds(boid)

            # Pak "zhurmë"/paparashikueshmëri e vogël - thyen simetritë e
            # përkryera që mund të shkaktojnë bllokim te pengesat, dhe njëkohësisht
            # e bën lëvizjen më realiste (njerëzit s'lëvizin në linja perfekte)
            noise = np.random.uniform(-0.05, 0.05, size=2)

            acceleration = (separation_force + alignment_force + cohesion_force +
                            exit_force + obstacle_force + bounds_force + noise)

            force_magnitude = np.linalg.norm(acceleration)
            if force_magnitude > boid.max_force:
                acceleration = (acceleration / force_magnitude) * boid.max_force

            boid.update(acceleration)
            self.environment.resolve_collisions(boid)
            self.environment.enforce_boundaries(boid)

            # Kontrollojmë nëse ka arritur te dalja
            if self.environment.has_reached_exit(boid.position):
                evacuated.append(boid)
                self.evacuation_times.append(self.time_elapsed)

        # Heqim boid-et e evakuuar nga simulimi
        for boid in evacuated:
            self.boids.remove(boid)

    def is_finished(self):
        """Kthen True nëse të gjithë boid-et kanë evakuuar."""
        return len(self.boids) == 0