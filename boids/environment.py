import numpy as np


class Environment:
    """
    Përfaqëson hapësirën fizike të evakuimit: kufijtë e dhomës,
    daljet (dyert) dhe pengesat brenda saj.
    """

    def __init__(self, width, height, exits, obstacles=None):
        self.width = width
        self.height = height

        # Daljet: listë e pikave (x, y) - qendra e çdo dere
        # p.sh. [(450, 0)] për një derë të vetme lart mesit
        self.exits = [np.array(exit_point, dtype=float) for exit_point in exits]

        # Pengesat: listë e drejtkëndëshave (x, y, gjerësi, lartësi)
        self.obstacles = obstacles if obstacles is not None else []

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

    def has_reached_exit(self, position, threshold=15.0):
        """
        Kontrollon nëse pozicioni i dhënë ka arritur te ndonjë dalje
        (brenda një distance të vogël - threshold).
        """
        return self.distance_to_nearest_exit(position) < threshold

    def obstacle_avoidance_force(self, position, avoid_radius=30.0):
        """
        Llogarit forcën shtytëse larg pengesave (mure/kolona brenda dhomës).
        Çdo pengesë trajtohet si drejtkëndësh me qendër e cila shtyn larg.
        """
        steer = np.zeros(2)

        for (ox, oy, ow, oh) in self.obstacles:
            # Qendra e pengesës
            obstacle_center = np.array([ox + ow / 2, oy + oh / 2])
            distance = np.linalg.norm(position - obstacle_center)

            if distance < avoid_radius:
                direction = (position - obstacle_center)
                dist_norm = np.linalg.norm(direction)
                if dist_norm > 0:
                    steer += direction / dist_norm

        return steer