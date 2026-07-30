import numpy as np
import heapq
from collections import deque


def point_in_polygon(point, vertices):
    x, y = point
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def closest_point_on_segment(point, a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    p = np.array(point, dtype=float)
    ab = b - a
    length_sq = np.dot(ab, ab)
    if length_sq == 0:
        return a, 0.0
    t = np.clip(np.dot(p - a, ab) / length_sq, 0.0, 1.0)
    return a + t * ab, t


class PolygonRoom:
    """
    Dhomë me formë të lirë poligoni (jo domosdoshmërisht drejtkëndëshe),
    definuar nga lista kulmesh që formojnë kufi të mbyllur. Dyert janë
    'gaps' mbi segmente specifike. Veçori PARALELE me Environment - nuk
    e prek fare sistemin ekzistues të dhomës drejtkëndëshe.
    """

    def __init__(self, vertices, doors=None, obstacles=None,
                 circle_obstacles=None, exit_width=15.0):
        self.vertices = [tuple(v) for v in vertices]
        self.segments = [(self.vertices[i], self.vertices[(i + 1) % len(self.vertices)])
                          for i in range(len(self.vertices))]
        self.doors = doors if doors is not None else []
        self.obstacles = obstacles if obstacles is not None else []
        self.circle_obstacles = circle_obstacles if circle_obstacles is not None else []
        self.exit_width = exit_width

        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)

        self.flow_field = PolygonFlowField(self, cell_size=20)

    def contains_point(self, position):
        return point_in_polygon(position, self.vertices)

    def closest_boundary_point(self, position):
        best_dist = np.inf
        best_seg, best_point, best_t = -1, None, 0.0
        for i, (a, b) in enumerate(self.segments):
            cp, t = closest_point_on_segment(position, a, b)
            d = np.linalg.norm(np.array(position) - cp)
            if d < best_dist:
                best_dist, best_seg, best_point, best_t = d, i, cp, t
        return best_dist, best_seg, best_point, best_t

    def _doors_on_segment(self, seg_idx):
        a, b = self.segments[seg_idx]
        a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
        result = []
        for (door_seg, t_center, half_width) in self.doors:
            if door_seg == seg_idx:
                result.append((a_arr + t_center * (b_arr - a_arr), half_width))
        return result

    def is_in_door_gap(self, seg_idx, point):
        for (center_point, half_width) in self._doors_on_segment(seg_idx):
            if np.linalg.norm(np.array(point) - center_point) < half_width:
                return True
        return False

    def has_reached_exit(self, position, buffer=10.0):
        """
        Kontrollon nëse pozicioni ka arritur te dera - buffer i vogël
        e fiksuar (10px), i pavarur nga exit_width (gjerësia e derës
        përdoret vetëm për të përcaktuar SA GJERË është gap-i, jo sa
        LARG nga muri konsiderohet i evakuuar - këto janë dy koncepte
        të ndryshme, njësoj si te sistemi me 4 mure).
        """
        dist, seg_idx, closest_point, _ = self.closest_boundary_point(position)
        return dist < buffer and self.is_in_door_gap(seg_idx, closest_point)

    def nearest_exit(self, position):
        best_dist, best_point = np.inf, np.array(position)
        for seg_idx in range(len(self.segments)):
            for (center_point, _) in self._doors_on_segment(seg_idx):
                d = np.linalg.norm(np.array(position) - center_point)
                if d < best_dist:
                    best_dist, best_point = d, center_point
        return best_point

    def get_next_waypoint(self, position):
        waypoint = self.flow_field.get_waypoint(position, steps=3)
        return waypoint if waypoint is not None else self.nearest_exit(position)

    def keep_within_bounds(self, boid, margin=50, turn_force=0.5):
        dist, seg_idx, closest_point, _ = self.closest_boundary_point(boid.position)
        if dist < margin and not self.is_in_door_gap(seg_idx, closest_point):
            direction = np.array(boid.position) - np.array(closest_point)
            norm = np.linalg.norm(direction)
            if norm > 0:
                return (direction / norm) * turn_force * ((margin - dist) / margin)
        return np.zeros(2)

    def enforce_boundaries(self, boid):
        if not self.contains_point(boid.position):
            dist, seg_idx, closest_point, _ = self.closest_boundary_point(boid.position)
            if self.is_in_door_gap(seg_idx, closest_point):
                return
            a, b = self.segments[seg_idx]
            edge = np.array(b) - np.array(a)
            normal = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal = normal / norm_len
                center = np.array([(self.min_x + self.max_x) / 2,
                                    (self.min_y + self.max_y) / 2])
                if np.dot(normal, center - np.array(closest_point)) < 0:
                    normal = -normal
                boid.position = np.array(closest_point) + normal * 2
                inward_speed = np.dot(boid.velocity, normal)
                if inward_speed < 0:
                    boid.velocity -= inward_speed * normal

    def obstacle_avoidance_force(self, position, avoid_radius=40.0):
        steer = np.zeros(2)
        for (ox, oy, ow, oh) in self.obstacles:
            closest_x = np.clip(position[0], ox, ox + ow)
            closest_y = np.clip(position[1], oy, oy + oh)
            closest_point = np.array([closest_x, closest_y])
            direction = position - closest_point
            distance = np.linalg.norm(direction)
            if distance < avoid_radius:
                if distance > 0:
                    steer += (direction / distance) * (avoid_radius - distance) / avoid_radius
                else:
                    steer += np.array([1.0, 0.0])
        for (cx, cy, radius) in self.circle_obstacles:
            center = np.array([cx, cy])
            direction = position - center
            dist_to_center = np.linalg.norm(direction)
            dist_to_edge = dist_to_center - radius
            if dist_to_edge < avoid_radius:
                if dist_to_center > 0:
                    unit_dir = direction / dist_to_center
                    strength = max(0.0, (avoid_radius - dist_to_edge) / avoid_radius)
                    steer += unit_dir * strength
                else:
                    steer += np.array([1.0, 0.0])
        return steer

    def resolve_collisions(self, boid):
        for (ox, oy, ow, oh) in self.obstacles:
            if ox < boid.position[0] < ox + ow and oy < boid.position[1] < oy + oh:
                dist_left = boid.position[0] - ox
                dist_right = (ox + ow) - boid.position[0]
                dist_top = boid.position[1] - oy
                dist_bottom = (oy + oh) - boid.position[1]
                min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
                obstacle_center_x = ox + ow / 2
                if min_dist == dist_left:
                    boid.position[0] = ox - 1
                    boid.velocity[0] = -abs(boid.velocity[0]) - 1.0
                elif min_dist == dist_right:
                    boid.position[0] = ox + ow + 1
                    boid.velocity[0] = abs(boid.velocity[0]) + 1.0
                elif min_dist == dist_top:
                    boid.position[1] = oy - 1
                    boid.velocity[1] = -abs(boid.velocity[1])
                    boid.velocity[0] += -2.0 if boid.position[0] < obstacle_center_x else 2.0
                else:
                    boid.position[1] = oy + oh + 1
                    boid.velocity[1] = abs(boid.velocity[1])
                    boid.velocity[0] += -2.0 if boid.position[0] < obstacle_center_x else 2.0

        for (cx, cy, radius) in self.circle_obstacles:
            center = np.array([cx, cy])
            direction = boid.position - center
            dist = np.linalg.norm(direction)
            if dist < radius:
                unit_dir = direction / dist if dist > 0 else np.array([1.0, 0.0])
                boid.position = center + unit_dir * (radius + 1)
                radial_velocity = np.dot(boid.velocity, unit_dir)
                if radial_velocity < 0:
                    boid.velocity -= 2 * radial_velocity * unit_dir

    def distance_to_nearest_obstacle(self, position):
        return self.flow_field.distance_to_nearest_obstacle(position)


class PolygonFlowField:
    def __init__(self, room, cell_size=20, wall_avoid_radius=40.0, wall_penalty_weight=3.0):
        self.room = room
        self.cell_size = cell_size
        self.cols = int(np.ceil((room.max_x - room.min_x) / cell_size)) + 1
        self.rows = int(np.ceil((room.max_y - room.min_y) / cell_size)) + 1
        self.origin_x = room.min_x
        self.origin_y = room.min_y
        self.wall_avoid_radius = wall_avoid_radius
        self.wall_penalty_weight = wall_penalty_weight

        self.blocked = self._build_blocked_grid()
        self.wall_distance = self._compute_wall_distance()
        source_cells = self._find_exit_cells()
        self.distance = self._weighted_dijkstra(source_cells)
        self.direction = self._compute_directions()

    def _cell_center(self, row, col):
        return (self.origin_x + (col + 0.5) * self.cell_size,
                self.origin_y + (row + 0.5) * self.cell_size)

    def _build_blocked_grid(self):
        blocked = np.zeros((self.rows, self.cols), dtype=bool)
        for row in range(self.rows):
            for col in range(self.cols):
                cx, cy = self._cell_center(row, col)
                if not self.room.contains_point((cx, cy)):
                    blocked[row, col] = True
                    continue
                for (ox, oy, ow, oh) in self.room.obstacles:
                    if ox - 8 < cx < ox + ow + 8 and oy - 8 < cy < oy + oh + 8:
                        blocked[row, col] = True
                        break
                if not blocked[row, col]:
                    for (ccx, ccy, radius) in self.room.circle_obstacles:
                        if np.sqrt((cx - ccx) ** 2 + (cy - ccy) ** 2) < radius + 8:
                            blocked[row, col] = True
                            break
        return blocked

    def _find_exit_cells(self):
        sources = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.blocked[row, col]:
                    continue
                cx, cy = self._cell_center(row, col)
                dist, seg_idx, closest_point, _ = self.room.closest_boundary_point((cx, cy))
                if dist < self.cell_size and self.room.is_in_door_gap(seg_idx, closest_point):
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
                if dr != 0 and dc != 0:
                    if self.blocked[row + dr, col] or self.blocked[row, col + dc]:
                        continue
                    cost = np.sqrt(2)
                else:
                    cost = 1.0
                yield nr, nc, cost

    def _compute_wall_distance(self):
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
        wd = self.wall_distance[row, col]
        if wd >= self.wall_avoid_radius or np.isinf(wd):
            return 0.0
        return ((self.wall_avoid_radius - wd) / self.wall_avoid_radius) * self.wall_penalty_weight

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
                cost = base_cost * (1.0 + self._wall_penalty(nr, nc))
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
        col = int((position[0] - self.origin_x) // self.cell_size)
        row = int((position[1] - self.origin_y) // self.cell_size)
        return min(max(row, 0), self.rows - 1), min(max(col, 0), self.cols - 1)

    def _nearest_reachable_cell(self, row, col, max_radius=6):
        if not self.blocked[row, col] and not np.isinf(self.distance[row, col]):
            return row, col
        for radius in range(1, max_radius + 1):
            best, best_dist = None, np.inf
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if not self.blocked[nr, nc] and not np.isinf(self.distance[nr, nc]):
                            if self.distance[nr, nc] < best_dist:
                                best_dist, best = self.distance[nr, nc], (nr, nc)
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
            next_row, next_col = row + int(round(vec[1])), col + int(round(vec[0]))
            if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
                break
            if self.blocked[next_row, next_col]:
                break
            row, col = next_row, next_col
        return np.array(self._cell_center(row, col))

    def distance_to_nearest_obstacle(self, position):
        row, col = self._cell_of(position)
        return self.wall_distance[row, col]