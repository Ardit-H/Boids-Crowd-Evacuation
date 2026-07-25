import numpy as np


class Environment:
    """
    Përfaqëson hapësirën fizike të evakuimit: kufijtë e dhomës,
    daljet (dyert) dhe pengesat brenda saj.
    """

    def __init__(self, width, height, exits, obstacles=None, exit_width=15.0):
        self.width = width
        self.height = height

        self.exits = [np.array(exit_point, dtype=float) for exit_point in exits]
        self.obstacles = obstacles if obstacles is not None else []

        # Gjerësia e derës - sa larg pikës qendrore konsiderohet "brenda"
        # derës. Vlerë më e madhe = derë më e gjerë = evakuim potencialisht
        # më i shpejtë (më lehtë të arrihet, më pak grumbullim te pika qendrore)
        self.exit_width = exit_width

    def nearest_exit(self, position):
        """
        Gjen daljen më të afërt me pozicionin e dhënë.
        Kthen pozicionin e daljes më të afërt.
        """
        distances = [np.linalg.norm(position - exit_pos) for exit_pos in self.exits]
        nearest_index = np.argmin(distances)
        return self.exits[nearest_index]

    def distance_to_nearest_exit(self, position):
        """Distanca deri te dalja më e afërt (përdoret për matje evakuimi)."""
        nearest = self.nearest_exit(position)
        return np.linalg.norm(position - nearest)

    def has_reached_exit(self, position):
        """
        Kontrollon nëse pozicioni ka arritur te ndonjë dalje - konsideron
        derën si një SEGMENT (jo pikë), kështu boid-et që kalojnë afër
        y=0 brenda gjerësisë së derës evakuojnë menjëherë, pa pasur
        nevojë të arrijnë saktësisht qendrën e derës.
        """
        for exit_pos in self.exits:
            within_width = abs(position[0] - exit_pos[0]) < (self.exit_width / 2 + 10)
            near_door_y = position[1] < 20
            if within_width and near_door_y:
                return True
        return False

    def obstacle_avoidance_force(self, position, avoid_radius=40.0):
        """
        Llogarit forcën shtytëse larg pengesave. Në vend të qendrës së
        drejtkëndëshit, gjen pikën më të afërt TË VETË DREJTKËNDËSHIT
        (edge/kënd), që e bën shmangien saktë edhe kur boid-i është
        pranë një skaji, jo vetëm pranë qendrës.
        """
        steer = np.zeros(2)

        for (ox, oy, ow, oh) in self.obstacles:
            # Gjen pikën më të afërt brenda drejtkëndëshit ndaj pozicionit
            closest_x = np.clip(position[0], ox, ox + ow)
            closest_y = np.clip(position[1], oy, oy + oh)
            closest_point = np.array([closest_x, closest_y])

            direction = position - closest_point
            distance = np.linalg.norm(direction)

            if distance < avoid_radius:
                if distance > 0:
                    # Sa më afër skajit, aq më fortë forca (1/distance)
                    steer += (direction / distance) * (avoid_radius - distance) / avoid_radius
                else:
                    # Boid-i është SAKTË brenda drejtkëndëshit (rast ekstrem)
                    # - shtyje larg me forcë maksimale në një drejtim arbitrar
                    steer += np.array([1.0, 0.0])

        return steer

    def resolve_collisions(self, boid):
        """
        Kontroll i fortë: nëse boid-i bie brenda një pengese, e nxjerr
        jashtë saj, ia anulon shpejtësinë në atë drejtim, DHE i shton
        një shtytje horizontale drejt anës më të afërt të pengesës -
        kjo e detyron të "rrëshqasë" anash në vend që të mbetet i
        ngrirë duke u kërcyer vetëm vertikalisht.
        """
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
                    # Shtytje horizontale drejt anës më të afërt
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

    def enforce_boundaries(self, boid):
        """
        Kufi i FORTË i dhomës: pavarësisht forcave (që janë vetëm
        'sugjerime' të buta), boid-i KURRË nuk mund të kalojë jashtë
        mureve përveç saktësisht brenda gjerësisë së një dere. Kjo
        eliminon çdo mundësi kalimi 'anash' ose zhdukjeje përtej
        ekranit - i njëjti parim si resolve_collisions për pengesat.
        """
        # Muri i majtë / djathtë - gjithmonë i fortë
        if boid.position[0] < 0:
            boid.position[0] = 0
            boid.velocity[0] = abs(boid.velocity[0])
        elif boid.position[0] > self.width:
            boid.position[0] = self.width
            boid.velocity[0] = -abs(boid.velocity[0])

        # Muri i poshtëm - gjithmonë i fortë
        if boid.position[1] > self.height:
            boid.position[1] = self.height
            boid.velocity[1] = -abs(boid.velocity[1])

        # Muri i sipërm - i fortë KUDO përveç saktësisht brenda
        # gjerësisë së një dere (aty lejohet kalimi, dhe do të
        # evakuohet menjëherë nga has_reached_exit)
        if boid.position[1] < 0:
            near_door = any(
                abs(boid.position[0] - exit_pos[0]) < (self.exit_width / 2)
                for exit_pos in self.exits
            )
            if not near_door:
                boid.position[1] = 0
                if boid.velocity[1] < 0:
                    boid.velocity[1] = 0