import numpy as np
import heapq
from collections import deque


class FlowField:
    """
    Ndërton një 'hartë rrjedhe' (flow field): për çdo qelizë të lirë,
    njehson drejtimin optimal drejt daljes më të afërt duke shmangur
    pengesat. Përdor Dijkstra multi-burim me KOSTO SHTESË pranë
    mureve (jo bllokim absolut) - kështu pathfinding-u PREFERON rrugë
    larg mureve kur ka mundësi, por gjithmonë gjen rrugë edhe në
    korridore të ngushta, pa krijuar konflikt me forcën fizike të
    shmangies (obstacle_avoidance_force).
    """

    def __init__(self, width, height, obstacles, exits, exit_width,
                 cell_size=20, blocked_inflate=8.0,
                 wall_avoid_radius=40.0, wall_penalty_weight=3.0):
        self.cell_size = cell_size
        self.cols = int(np.ceil(width / cell_size))
        self.rows = int(np.ceil(height / cell_size))

        # Bllokim absolut vetëm për vetë pengesën + një margjinë e vogël
        # (mjafton për të penguar 'prerjen e cepit' diagonal, jo krejt
        # zonën e ndikimit të forcës fizike)
        self.blocked = self._build_blocked_grid(obstacles, blocked_inflate)

        # Distanca (në px) e çdo qelize deri te muri/pengesa më e afërt -
        # përdoret për të llogaritur 'kosto shtesë' pranë mureve
        self.wall_distance = self._compute_wall_distance()

        self.wall_avoid_radius = wall_avoid_radius
        self.wall_penalty_weight = wall_penalty_weight

        source_cells = self._find_exit_cells(exits, exit_width)
        self.distance = self._weighted_dijkstra(source_cells)
        self.direction = self._compute_directions()

    def _build_blocked_grid(self, obstacles, inflate):
        blocked = np.zeros((self.rows, self.cols), dtype=bool)
        for (ox, oy, ow, oh) in obstacles:
            x0, y0 = ox - inflate, oy - inflate
            x1, y1 = ox + ow + inflate, oy + oh + inflate
            col0 = max(0, int(x0 // self.cell_size))
            col1 = min(self.cols - 1, int(x1 // self.cell_size))
            row0 = max(0, int(y0 // self.cell_size))
            row1 = min(self.rows - 1, int(y1 // self.cell_size))
            blocked[row0:row1 + 1, col0:col1 + 1] = True
        return blocked

    def _find_exit_cells(self, exits, exit_width):
        sources = []
        half = exit_width / 2
        row = 0
        for (ex, ey) in exits:
            col0 = max(0, int((ex - half) // self.cell_size))
            col1 = min(self.cols - 1, int((ex + half) // self.cell_size))
            for col in range(col0, col1 + 1):
                if not self.blocked[row, col]:
                    sources.append((row, col))
        return sources

    def _neighbors(self, row, col):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.blocked[nr, nc]:
                    continue

                # Ndalo prerjen diagonale të cepit (corner-cutting)
                if dr != 0 and dc != 0:
                    if self.blocked[row + dr, col] or self.blocked[row, col + dc]:
                        continue
                    base_cost = np.sqrt(2)
                else:
                    base_cost = 1.0

                yield nr, nc, base_cost

    def _compute_wall_distance(self):
        """
        BFS multi-burim nga çdo qelizë e bllokuar - jep distancën (në
        numër qelizash, e shndërruar në px) deri te pengesa/muri më i
        afërt, për çdo qelizë të lirë.
        """
        dist = np.full((self.rows, self.cols), np.inf)
        q = deque()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.blocked[r, c]:
                    dist[r, c] = 0.0
                    q.append((r, c))

        while q:
            r, c = q.popleft()
            d = dist[r, c]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        step = np.sqrt(2) if dr != 0 and dc != 0 else 1.0
                        nd = d + step
                        if nd < dist[nr, nc]:
                            dist[nr, nc] = nd
                            q.append((nr, nc))

        return dist * self.cell_size

    def _wall_penalty(self, row, col):
        """
        Kosto shtesë proporcionale me afërsinë te muri - sa më afër,
        aq më e shtrenjtë (jo e ndaluar) rruga aty. Kjo e bën
        pathfinding-un të PREFERONTE rrugë larg mureve kur mundet,
        pa e refuzuar plotësisht kalimin pranë tyre kur nevojitet.
        """
        wd = self.wall_distance[row, col]
        if wd >= self.wall_avoid_radius or np.isinf(wd):
            return 0.0
        closeness = (self.wall_avoid_radius - wd) / self.wall_avoid_radius
        return closeness * self.wall_penalty_weight

    def _weighted_dijkstra(self, source_cells):
        dist = np.full((self.rows, self.cols), np.inf)
        pq = []
        for (row, col) in source_cells:
            dist[row, col] = 0.0
            heapq.heappush(pq, (0.0, row, col))

        while pq:
            d, row, col = heapq.heappop(pq)
            if d > dist[row, col]:
                continue
            for nr, nc, base_cost in self._neighbors(row, col):
                penalty = self._wall_penalty(nr, nc)
                cost = base_cost * (1.0 + penalty)
                nd = d + cost
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(pq, (nd, nr, nc))

        return dist

    def _compute_directions(self):
        direction = np.zeros((self.rows, self.cols, 2))
        for row in range(self.rows):
            for col in range(self.cols):
                if self.blocked[row, col] or np.isinf(self.distance[row, col]):
                    continue
                best_dist = self.distance[row, col]
                best_delta = None
                for nr, nc, _ in self._neighbors(row, col):
                    if self.distance[nr, nc] < best_dist:
                        best_dist = self.distance[nr, nc]
                        best_delta = (nr - row, nc - col)
                if best_delta is not None:
                    dr, dc = best_delta
                    vec = np.array([dc, dr], dtype=float)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        direction[row, col] = vec / norm
        return direction

    def _cell_of(self, position):
        col = int(position[0] // self.cell_size)
        row = int(position[1] // self.cell_size)
        col = min(max(col, 0), self.cols - 1)
        row = min(max(row, 0), self.rows - 1)
        return row, col

    def _nearest_reachable_cell(self, row, col, max_radius=6):
        """
        Rrjetë sigurie: nëse qeliza aktuale është (rrallë) e bllokuar
        ose e paarritshme, kërkon spirale poshtë/lart/anash për
        qelizën e lirë/të arritshme më të afërt.
        """
        if not self.blocked[row, col] and not np.isinf(self.distance[row, col]):
            return row, col

        for radius in range(1, max_radius + 1):
            best = None
            best_dist = np.inf
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if not self.blocked[nr, nc] and not np.isinf(self.distance[nr, nc]):
                            if self.distance[nr, nc] < best_dist:
                                best_dist = self.distance[nr, nc]
                                best = (nr, nc)
            if best is not None:
                return best
        return None

    def get_waypoint(self, position, steps=3):
        row, col = self._cell_of(position)
        cell = self._nearest_reachable_cell(row, col)
        if cell is None:
            return None
        row, col = cell

        for _ in range(steps):
            vec = self.direction[row, col]
            if np.linalg.norm(vec) == 0:
                break
            next_col = col + int(round(vec[0]))
            next_row = row + int(round(vec[1]))
            if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
                break
            if self.blocked[next_row, next_col]:
                break
            row, col = next_row, next_col

        world_x = (col + 0.5) * self.cell_size
        world_y = (row + 0.5) * self.cell_size
        return np.array([world_x, world_y])