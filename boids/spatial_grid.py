import numpy as np
from collections import defaultdict


class SpatialGrid:
    """
    Ndan hapësirën në qeliza (bucket) sipas pozicionit, për të gjetur
    shpejt fqinjët e një boid-i pa kontrolluar TË GJITHË boid-et e
    tjerë (O(n) mesatarisht, në vend të O(n²)). Ribuildon në fillim
    të çdo frame-i.
    """

    def __init__(self, cell_size=50.0):
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    def _cell_of(self, position):
        return (int(position[0] // self.cell_size),
                int(position[1] // self.cell_size))

    def build(self, boids):
        self.grid.clear()
        for boid in boids:
            self.grid[self._cell_of(boid.position)].append(boid)

    def get_neighbors(self, boid, radius=50.0):
        cx, cy = self._cell_of(boid.position)
        neighbors = []

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in self.grid.get((cx + dx, cy + dy), []):
                    if other is boid:
                        continue
                    distance = np.linalg.norm(boid.position - other.position)
                    if distance < radius:
                        neighbors.append(other)

        return neighbors