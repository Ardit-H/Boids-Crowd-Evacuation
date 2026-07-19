import numpy as np


class Boid:
    """
    Përfaqëson një agjent individual (person) në simulimin e turmës.
    Çdo Boid ka pozicion dhe shpejtësi, dhe sillet sipas rregullave
    të separation, alignment dhe cohesion (Craig Reynolds, 1986).
    """

    def __init__(self, position, velocity, max_speed=4.0, max_force=0.1):
        # Pozicioni aktual i agjentit në hapësirë (vektor 2D: [x, y])
        self.position = np.array(position, dtype=float)

        # Shpejtësia aktuale (vektor 2D: [vx, vy])
        self.velocity = np.array(velocity, dtype=float)

        # Shpejtësia maksimale e lejuar (kufizim realist - njeriu s'vrapon pafundësisht shpejt)
        self.max_speed = max_speed

        # Forca maksimale që mund të aplikohet në një hap kohor
        # (kufizon sa shpejt mund të ndryshojë drejtimin - "manovrueshmëria")
        self.max_force = max_force

    def separation(self, neighbors, desired_separation=25.0):
        """
        Rregulli 1: Largohu nga fqinjët shumë të afërt për të shmangur përplasjen.
        Kthen një vektor "force" që e shtyn boid-in larg nga fqinjët e afërt.
        """
        steer = np.zeros(2)
        count = 0

        for other in neighbors:
            distance = np.linalg.norm(self.position - other.position)
            if 0 < distance < desired_separation:
                # Sa më afër është fqinji, aq më fort e shtyn larg (pesha 1/distance)
                diff = (self.position - other.position) / distance
                steer += diff
                count += 1

        if count > 0:
            steer /= count  # mesatarja e forcave shtytëse nga të gjithë fqinjët e afërt

        return steer

    def align(self, neighbors, neighbor_radius=50.0):
        """
        Rregulli 2: Lëviz në drejtim të ngjashëm me fqinjët përreth.
        Kthen vektorin e mesatarizuar të shpejtësisë së fqinjëve.
        """
        avg_velocity = np.zeros(2)
        count = 0

        for other in neighbors:
            distance = np.linalg.norm(self.position - other.position)
            if 0 < distance < neighbor_radius:
                avg_velocity += other.velocity
                count += 1

        if count > 0:
            avg_velocity /= count
            steer = avg_velocity - self.velocity
            return steer

        return np.zeros(2)

    def cohesion(self, neighbors, neighbor_radius=50.0):
        """
        Rregulli 3: Lëviz drejt qendrës mesatare (qendra e masës) të fqinjëve.
        Kthen vektorin drejt qendrës së grupit lokal.
        """
        center = np.zeros(2)
        count = 0

        for other in neighbors:
            distance = np.linalg.norm(self.position - other.position)
            if 0 < distance < neighbor_radius:
                center += other.position
                count += 1

        if count > 0:
            center /= count
            direction_to_center = center - self.position
            return direction_to_center

        return np.zeros(2)

    def seek_exit(self, exit_position):
        """
        Rregull shtesë (specifik për evakuim): tërhiqet drejt pikës së daljes.
        Ky është 'goal-seeking behavior' - ndryshe nga separation/alignment/
        cohesion, kjo forcë e drejton agjentin drejt një qëllimi specifik,
        jo thjesht ndaj sjelljes së fqinjëve.
        """
        desired = exit_position - self.position
        distance = np.linalg.norm(desired)

        if distance > 0:
            desired = (desired / distance) * self.max_speed
            steer = desired - self.velocity
            return steer

        return np.zeros(2)

    def update(self, acceleration):
        """
        Përditëson shpejtësinë dhe pozicionin e boid-it bazuar në forcën/nxitimin
        e llogaritur nga rregullat (separation + alignment + cohesion + forca të tjera).
        """
        self.velocity += acceleration

        # Kufizojmë shpejtësinë maksimale (agjenti s'mund të lëvizë më shpejt se max_speed)
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

        self.position += self.velocity