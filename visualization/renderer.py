import pygame
import numpy as np
import sys
import os

# Kjo linjë siguron që Python e gjen follderin "boids" edhe kur e ekzekutojmë
# këtë file direkt nga brenda follderit "visualization"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from boids.simulation import Simulation

# --- Konfigurimi bazë i dritares ---
WIDTH, HEIGHT = 900, 700
NUM_BOIDS = 80
FPS = 60

# Ngjyrat (RGB)
BACKGROUND_COLOR = (20, 20, 30)
BOID_COLOR = (100, 200, 255)


def draw_boid(screen, boid, size=6):
    """
    Vizaton një boid si trekëndësh të vogël, i rrotulluar sipas
    drejtimit të lëvizjes (velocity) - kështu shohim vizualisht
    ku po "shikon"/lëviz çdo agjent, jo vetëm pozicionin e tij.
    """
    angle = np.arctan2(boid.velocity[1], boid.velocity[0])

    # Tre kulmet e trekëndëshit, relative ndaj pozicionit qendror
    points = [
        (size * 2, 0),
        (-size, size),
        (-size, -size)
    ]

    rotated_points = []
    for px, py in points:
        rx = px * np.cos(angle) - py * np.sin(angle)
        ry = px * np.sin(angle) + py * np.cos(angle)
        rotated_points.append((boid.position[0] + rx, boid.position[1] + ry))

    pygame.draw.polygon(screen, BOID_COLOR, rotated_points)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids Crowd Simulation")
    clock = pygame.time.Clock()

    simulation = Simulation(NUM_BOIDS, WIDTH, HEIGHT)

    running = True
    while running:
        # Kontrollojmë nëse përdoruesi ka mbyllur dritaren
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Përditësojmë simulimin (një hap kohor)
        simulation.step()

        # Vizatojmë gjithçka nga fillimi (pastrojmë ekranin dhe rivizatojmë)
        screen.fill(BACKGROUND_COLOR)
        for boid in simulation.boids:
            draw_boid(screen, boid)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()