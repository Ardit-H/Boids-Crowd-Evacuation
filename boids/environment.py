import numpy as np
from boids.pathfinding import FlowField
from boids.geometry import (normalize_obstacle, rect_center, to_local_space,
                             to_world_space, rotate_vector,
                             resolve_axis_aligned_collision_local,
                             closest_point_on_rotated_rect)

DEBUG_COLLISIONS = False

def _exit_point(side, pos, width, height):
    if side == "top":
        return (pos, 0.0)
    elif side == "bottom":
        return (pos, height)
    elif side == "left":
        return (0.0, pos)
    else:  # "right"
        return (width, pos)


class Environment:
    """
    Përfaqëson hapësirën fizike të evakuimit: kufijtë e dhomës, daljet
    (dyert - tani në çdo nga 4 anët) dhe pengesat brenda saj.
    """

    def __init__(self, width, height, exits, obstacles=None, exit_width=15.0,
                 circle_obstacles=None):
        self.width = width
        self.height = height
        self.obstacles = obstacles if obstacles is not None else []
        self.circle_obstacles = circle_obstacles if circle_obstacles is not None else []
        self.exit_width = exit_width

        # Normalizon çdo dalje në format (side, pos) - mban pajtueshmëri
        # me formatin e vjetër (x, 0), i cili supozohej gjithmonë te
        # muri i sipërm
        self.exit_specs = []
        for item in exits:
            side, pos = item
            if isinstance(side, str):
                self.exit_specs.append((side, float(pos)))
            else:
                # Format i vjetër: (x, y) -> supozo murin e sipërm
                self.exit_specs.append(("top", float(side)))

        # Pikat aktuale (x, y) të çdo dere - përdoren për distanca/fallback
        self.exits = [
            np.array(_exit_point(side, pos, width, height), dtype=float)
            for (side, pos) in self.exit_specs
        ]

        self.flow_field = FlowField(width, height, self.obstacles,
                                    self.exit_specs, exit_width,
                                    circle_obstacles=self.circle_obstacles)

    def nearest_exit(self, position):
        distances = [np.linalg.norm(position - exit_pos) for exit_pos in self.exits]
        nearest_index = np.argmin(distances)
        return self.exits[nearest_index]

    def distance_to_nearest_exit(self, position):
        nearest = self.nearest_exit(position)
        return np.linalg.norm(position - nearest)

    def get_next_waypoint(self, position):
        waypoint = self.flow_field.get_waypoint(position, steps=3)
        if waypoint is None:
            return self.nearest_exit(position)
        return waypoint

    def has_reached_exit(self, position):
        """
        Kontrollon nëse pozicioni ka arritur te ndonjë dalje. Buffer-i
        prej 10px (jo 20px si më parë) e bën evakuimin të duket më
        pranë vijës vizuale të derës - kjo tani është e sigurt pa
        krijuar 'overshoot' të dukshëm, sepse seek_exit ngadalëson
        boid-in ndërsa afrohet (arrival behavior).
        """
        x, y = position
        half = self.exit_width / 2 + 6

        for (side, pos) in self.exit_specs:
            if side == "top" and abs(x - pos) < half and y < 10:
                return True
            elif side == "bottom" and abs(x - pos) < half and y > self.height - 10:
                return True
            elif side == "left" and abs(y - pos) < half and x < 10:
                return True
            elif side == "right" and abs(y - pos) < half and x > self.width - 10:
                return True
        return False

    def obstacle_avoidance_force(self, position, avoid_radius=40.0):
        steer = np.zeros(2)

        for obstacle in self.obstacles:
            ox, oy, ow, oh, angle = normalize_obstacle(obstacle)

            if angle == 0.0:
                # --- KODI ORIGJINAL, PA ASNJË NDRYSHIM ---
                closest_x = np.clip(position[0], ox, ox + ow)
                closest_y = np.clip(position[1], oy, oy + oh)
                closest_point = np.array([closest_x, closest_y])
            else:
                # --- RASTI I RROTULLUAR ---
                closest_point, _ = closest_point_on_rotated_rect(
                    position, ox, oy, ow, oh, angle)

            direction = position - closest_point
            distance = np.linalg.norm(direction)
            if distance < avoid_radius:
                if distance > 0:
                    steer += (direction / distance) * (avoid_radius - distance) / avoid_radius
                else:
                    steer += np.array([1.0, 0.0])

        # Pengesat rrethore - pika më e afërt është gjithmonë në vetë
        # rrethin (qendër + rreze në drejtim të pozicionit), ndryshe
        # nga drejtkëndëshi ku pika më e afërt varet nga cepi/skaji
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
        for obstacle in self.obstacles:
            ox, oy, ow, oh, angle = normalize_obstacle(obstacle)

            if angle == 0.0:
                # --- KODI ORIGJINAL, PA ASNJË NDRYSHIM ---
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
                        if boid.position[0] < obstacle_center_x:
                            boid.velocity[0] -= 2.0
                        else:
                            boid.velocity[0] += 2.0
                    else:
                        boid.position[1] = oy + oh + 1
                        boid.velocity[1] = abs(boid.velocity[1])
                        if boid.position[0] < obstacle_center_x:
                            boid.velocity[0] -= 2.0
                        else:
                            boid.velocity[0] += 2.0

                    side = ("left" if min_dist == dist_left else
                            "right" if min_dist == dist_right else
                            "top" if min_dist == dist_top else "bottom")
                    if DEBUG_COLLISIONS:
                        print(f"[COLLISION axis=0] side={side} min_dist={min_dist:.1f} "
                             f"pos={boid.position} vel_after={boid.velocity}")

            else:
                # --- RASTI I RROTULLUAR - ripërdor TË NJËJTIN algoritëm,
                # ekzekutuar në hapësirën lokale të pengesës ---
                center = rect_center(ox, oy, ow, oh)
                local_pos = to_local_space(boid.position, center, angle)
                half_w, half_h = ow / 2.0, oh / 2.0

                if abs(local_pos[0]) < half_w and abs(local_pos[1]) < half_h:
                    local_vel = to_local_space(boid.velocity, np.zeros(2), angle)
                    new_local_pos, new_local_vel = resolve_axis_aligned_collision_local(
                        local_pos, local_vel, ow, oh)

                    boid.position = to_world_space(new_local_pos, center, angle)
                    boid.velocity = rotate_vector(new_local_vel, angle)

                    if DEBUG_COLLISIONS:
                        print(f"[COLLISION rotated angle={np.degrees(angle):.0f}] "
                            f"pos={boid.position} vel_after={boid.velocity}")

        # Pengesat rrethore - MBETET E PANDRYSHUAR
        for (cx, cy, radius) in self.circle_obstacles:
            center = np.array([cx, cy])
            direction = boid.position - center
            dist = np.linalg.norm(direction)

            if dist < radius:
                if dist > 0:
                    unit_dir = direction / dist
                else:
                    unit_dir = np.array([1.0, 0.0])
                    dist = 0.001

                boid.position = center + unit_dir * (radius + 1)

                radial_velocity = np.dot(boid.velocity, unit_dir)
                if radial_velocity < 0:
                    boid.velocity -= 2 * radial_velocity * unit_dir

    def enforce_boundaries(self, boid):
        """
        Kufi i FORTË i dhomës në të gjitha 4 anët. Çdo anë mbetet e
        fortë KUDO përveç saktësisht brenda gjerësisë së një dere të
        vendosur në atë anë specifike.
        """
        half = self.exit_width / 2
        x, y = boid.position

        if x < 0:
            near_door = any(side == "left" and abs(y - pos) < half
                             for side, pos in self.exit_specs)
            if not near_door:
                boid.position[0] = 0
                boid.velocity[0] = abs(boid.velocity[0])
        elif x > self.width:
            near_door = any(side == "right" and abs(y - pos) < half
                             for side, pos in self.exit_specs)
            if not near_door:
                boid.position[0] = self.width
                boid.velocity[0] = -abs(boid.velocity[0])

        if y < 0:
            near_door = any(side == "top" and abs(x - pos) < half
                             for side, pos in self.exit_specs)
            if not near_door:
                boid.position[1] = 0
                if boid.velocity[1] < 0:
                    boid.velocity[1] = 0
        elif y > self.height:
            near_door = any(side == "bottom" and abs(x - pos) < half
                             for side, pos in self.exit_specs)
            if not near_door:
                boid.position[1] = self.height
                if boid.velocity[1] > 0:
                    boid.velocity[1] = 0

    def distance_to_nearest_obstacle(self, position):
        return self.flow_field.distance_to_nearest_obstacle(position)