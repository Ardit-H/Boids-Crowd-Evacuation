import pygame
import numpy as np
import sys
import os
from analysis.metrics import print_summary, plot_evacuation_histogram
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
EXIT_COLOR = (80, 220, 100)


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

def draw_exits(screen, environment, exit_width=60, thickness=6):
    """
    Vizaton daljet si segmente jeshile mbi murin përkatës,
    që të shihet qartë ku ndodhet dera në hapësirë.
    """
    for exit_pos in environment.exits:
        x, y = exit_pos
        # Meqë dera jonë është në murin e sipërm (y=0), e vizatojmë
        # si një vijë horizontale të trashë mbi atë pozicion
        start = (x - exit_width / 2, y)
        end = (x + exit_width / 2, y)
        pygame.draw.line(screen, EXIT_COLOR, start, end, thickness)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids Crowd Simulation")
    clock = pygame.time.Clock()

    exits = [(WIDTH / 4, 0), (3 * WIDTH / 4, 0)]
    simulation = Simulation(NUM_BOIDS, WIDTH, HEIGHT, exits)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not simulation.is_finished():
            simulation.step()
        else:
            # Të gjithë boid-et kanë evakuuar - ndalojmë automatikisht
            running = False

        screen.fill(BACKGROUND_COLOR)
        draw_exits(screen, simulation.environment)
        for boid in simulation.boids:
            draw_boid(screen, boid)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

    # Shfaq statistikat dhe grafikun pasi të mbyllet dritarja e pygame
    print_summary(simulation.evacuation_times)
    plot_evacuation_histogram(simulation.evacuation_times)

    sys.exit()


if __name__ == "__main__":
    main()