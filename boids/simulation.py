import numpy as np
from boids.agent import Boid


class Simulation:
    """
    Menaxhon një grup (flock) boid-esh dhe simulimin e tyre në kohë.
    Kombinon rregullat separation, alignment, cohesion me pesha të
    ndryshme, dhe i mban boid-et brenda kufijve të hapësirës.
    """

    def __init__(self, num_boids, width, height,
                 separation_weight=1.5,
                 alignment_weight=1.0,
                 cohesion_weight=1.0):
        self.width = width
        self.height = height

        # Peshat e secilit rregull - këto janë parametra që mund t'i ndryshojmë
        # për të parë si ndryshon sjellja e turmës
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight

        # Krijojmë N boid-e me pozicione dhe shpejtësi fillestare random
        self.boids = []
        for _ in range(num_boids):
            position = [np.random.uniform(0, width), np.random.uniform(0, height)]
            velocity = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
            self.boids.append(Boid(position, velocity))

    def get_neighbors(self, boid, radius=50.0):
        """
        Gjen të gjithë boid-et brenda një rrezeje të caktuar nga boid-i i dhënë.
        Kjo simulon "shikimin"/perceptimin lokal të një personi në turmë -
        njerëzit nuk reagojnë ndaj gjithë turmës, vetëm ndaj atyre që janë më afër.
        """
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
        Nëse boid-i i afrohet skajit të hapësirës, i shton një forcë
        që e kthen drejt qendrës - kështu evitohet dalja jashtë "dhomës".
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

    def step(self):
        """
        Ekzekuton një hap kohor ("frame") të simulimit:
        për çdo boid, llogarit fqinjët, kombinon forcat, dhe e përditëson.
        """
        for boid in self.boids:
            neighbors = self.get_neighbors(boid)

            # Llogarit tre forcat bazë
            separation_force = boid.separation(neighbors) * self.separation_weight
            alignment_force = boid.align(neighbors) * self.alignment_weight
            cohesion_force = boid.cohesion(neighbors) * self.cohesion_weight

            # Forca shtesë për me mos dalë jashtë kufijve
            bounds_force = self.keep_within_bounds(boid)

            # Kombinon të gjitha forcat në një "acceleration" total
            acceleration = separation_force + alignment_force + cohesion_force + bounds_force

            # Kufizon forcën totale (realizëm - njeriu s'ndryshon drejtim papritmas/pafundësisht)
            force_magnitude = np.linalg.norm(acceleration)
            if force_magnitude > boid.max_force:
                acceleration = (acceleration / force_magnitude) * boid.max_force

            boid.update(acceleration)