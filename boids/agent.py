import numpy as np


class Boid:
    """
    Përfaqëson një agjent individual (person) në simulimin e turmës.
    Çdo Boid ka pozicion dhe shpejtësi, dhe sillet sipas rregullave
    të separation, alignment dhe cohesion (Craig Reynolds, 1986).
    """

    def __init__(self, position, velocity, max_speed=4.0, max_force=0.1):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.max_speed = max_speed
        self.max_force = max_force

        # Gjendje për zbulimin e "ngërçit" (deadlock) - regjistron pozicionin
        # çdo N frame dhe kontrollon nëse ka pasur lëvizje reale që atëherë
        self.stuck_check_position = self.position.copy()
        self.stuck_frame_counter = 0

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

    def seek_exit(self, exit_position, slow_radius=18.0, min_speed_ratio=0.6):
        """
        Tërhiqet drejt pikës së daljes, me ngadalësim gradual pranë
        qëllimit ('arrival behavior') që parandalon overshoot. Ka një
        'dysheme' minimale shpejtësie (min_speed_ratio) - kështu forca
        drejt derës MOS bëhet kurrë aq e dobët sa të mposhtet lehtësisht
        nga separation/cohesion e boid-eve të tjerë të grumbulluar pranë
        derës, gjë që do të shkaktonte lëkundje/rrotullim pikërisht atje.
        """
        desired = exit_position - self.position
        distance = np.linalg.norm(desired)

        if distance > 0:
            if distance < slow_radius:
                ratio = max(min_speed_ratio, distance / slow_radius)
                speed = self.max_speed * ratio
            else:
                speed = self.max_speed

            desired = (desired / distance) * speed
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

    def check_and_escape_if_stuck(self, stuck_frames_threshold=40, min_displacement=10.0,
                                  escape_strength=0.4):
        """
        Nëse boid-i s'ka lëvizur mjaftueshëm (min_displacement px) brenda
        stuck_frames_threshold frame-ve të fundit, i jep një impuls
        shpejtësie në drejtim RANDOM për ta zhbllokuar nga një ekuilibër
        i ngërçuar forcash (rasti tipik: 2-3 boid të mbetur që orbitojnë
        njëri-tjetrin pa avancuar kurrë drejt daljes).
        escape_strength: fraksion i max_speed që i shtohet shpejtësisë
        ekzistuese (jo zëvendësim total) - kështu efekti është një shtytje
        korrigjuese, jo një ndryshim i papritur drejtimi 180°.
        """
        self.stuck_frame_counter += 1

        if self.stuck_frame_counter >= stuck_frames_threshold:
            displacement = np.linalg.norm(self.position - self.stuck_check_position)

            if displacement < min_displacement:
                angle = np.random.uniform(0, 2 * np.pi)
                impulse = np.array([np.cos(angle), np.sin(angle)]) * self.max_speed * escape_strength
                self.velocity += impulse
                speed = np.linalg.norm(self.velocity)
                if speed > self.max_speed:
                    self.velocity = (self.velocity / speed) * self.max_speed

            self.stuck_check_position = self.position.copy()
            self.stuck_frame_counter = 0