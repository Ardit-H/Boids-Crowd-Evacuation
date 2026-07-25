import pygame
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from boids.simulation import Simulation
from analysis.metrics import print_summary, plot_evacuation_histogram

# --- Konfigurimi bazë i dritares ---
SIM_WIDTH, SIM_HEIGHT = 900, 700
PANEL_WIDTH = 260
WIDTH, HEIGHT = SIM_WIDTH + PANEL_WIDTH, SIM_HEIGHT
FPS = 60

# Ngjyrat (RGB)
BACKGROUND_COLOR = (20, 20, 30)
PANEL_COLOR = (35, 35, 48)
BOID_COLOR = (100, 200, 255)
EXIT_COLOR = (80, 220, 100)
OBSTACLE_COLOR = (200, 80, 80)
BUTTON_COLOR = (60, 60, 80)
BUTTON_HOVER_COLOR = (80, 80, 105)
BUTTON_SELECTED_COLOR = (70, 130, 180)
TEXT_COLOR = (230, 230, 230)

# Pengesa fikse (e njëjta si në versionin origjinal), përdoret vetëm
# kur përdoruesi zgjedh "Me pengesë" te paneli
DEFAULT_OBSTACLE = [(SIM_WIDTH / 2 - 40, 150, 80, 40)]


class Button:
    def __init__(self, x, y, w, h, label, value):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.value = value
        self.selected = False

    def draw(self, screen, font, mouse_pos):
        if self.selected:
            color = BUTTON_SELECTED_COLOR
        elif self.rect.collidepoint(mouse_pos):
            color = BUTTON_HOVER_COLOR
        else:
            color = BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        text_surface = font.render(self.label, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


def make_button_group(x, y_start, w, h, gap, items):
    buttons = []
    for i, (label, value) in enumerate(items):
        buttons.append(Button(x, y_start + i * (h + gap), w, h, label, value))
    return buttons


def draw_boid(screen, boid, size=6):
    angle = np.arctan2(boid.velocity[1], boid.velocity[0])
    points = [(size * 2, 0), (-size, size), (-size, -size)]
    rotated_points = []
    for px, py in points:
        rx = px * np.cos(angle) - py * np.sin(angle)
        ry = px * np.sin(angle) + py * np.cos(angle)
        rotated_points.append((boid.position[0] + rx, boid.position[1] + ry))
    pygame.draw.polygon(screen, BOID_COLOR, rotated_points)


def draw_exits(screen, environment, thickness=6):
    """
    Vizaton daljet me gjerësinë REALE (environment.exit_width) - jo
    një vlerë fikse - kështu vija jeshile përkon saktësisht me zonën
    ku boid-et realisht mund të kalojnë, pa krijuar përshtypjen e
    gabuar se dera është më e gjerë se ç'është funksionalisht.
    """
    for exit_pos in environment.exits:
        x, y = exit_pos
        half_width = environment.exit_width / 2
        start = (x - half_width, y)
        end = (x + half_width, y)
        pygame.draw.line(screen, EXIT_COLOR, start, end, thickness)


def draw_obstacles(screen, environment):
    for (ox, oy, ow, oh) in environment.obstacles:
        pygame.draw.rect(screen, OBSTACLE_COLOR, (ox, oy, ow, oh))


def build_exits(num_doors, width):
    if num_doors == 1:
        return [(width / 2, 0)]
    elif num_doors == 2:
        return [(width / 4, 0), (3 * width / 4, 0)]
    else:
        return [(width / 5, 0), (width / 2, 0), (4 * width / 5, 0)]


def create_simulation(num_boids, num_doors, exit_width, use_obstacle):
    exits = build_exits(num_doors, SIM_WIDTH)
    obstacles = DEFAULT_OBSTACLE if use_obstacle else []
    return Simulation(num_boids, SIM_WIDTH, SIM_HEIGHT, exits,
                       obstacles=obstacles, exit_width=exit_width)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids Crowd Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 15)
    title_font = pygame.font.SysFont("Arial", 17, bold=True)

    # --- Gjendja aktuale e zgjedhur (default) ---
    selected_doors = 1
    selected_width = 15.0
    selected_density = 80
    selected_obstacle = False

    # --- Krijo grupet e butonave brenda panelit ---
    px = SIM_WIDTH + 20
    bw, bh, gap = PANEL_WIDTH - 40, 30, 8

    door_buttons = make_button_group(px, 55, bw, bh, gap,
        [("1 Derë", 1), ("2 Dyer", 2), ("3 Dyer", 3)])
    door_buttons[0].selected = True

    width_buttons = make_button_group(px, 205, bw, bh, gap,
        [("E ngushtë (15px)", 15.0), ("Mesatare (30px)", 30.0), ("E gjerë (60px)", 60.0)])
    width_buttons[0].selected = True

    density_buttons = make_button_group(px, 355, bw, bh, gap,
        [("40 boid (ulët)", 40), ("80 boid (mesatar)", 80), ("150 boid (lartë)", 150)])
    density_buttons[1].selected = True

    obstacle_buttons = make_button_group(px, 505, bw, bh, gap,
        [("Pa pengesë", False), ("Me pengesë", True)])
    obstacle_buttons[0].selected = True

    start_button = Button(px, 600, bw, 40, "▶  Fillo Simulimin", None)
    graph_button = Button(px, 650, bw, 40, "📊  Shfaq Grafikun", None)

    all_groups = [door_buttons, width_buttons, density_buttons, obstacle_buttons]

    simulation = None
    running_sim = False
    stats_printed = False

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                for group in all_groups:
                    for btn in group:
                        if btn.is_clicked(mouse_pos):
                            for other in group:
                                other.selected = False
                            btn.selected = True

                for btn in door_buttons:
                    if btn.selected:
                        selected_doors = btn.value
                for btn in width_buttons:
                    if btn.selected:
                        selected_width = btn.value
                for btn in density_buttons:
                    if btn.selected:
                        selected_density = btn.value
                for btn in obstacle_buttons:
                    if btn.selected:
                        selected_obstacle = btn.value

                if start_button.is_clicked(mouse_pos):
                    simulation = create_simulation(selected_density, selected_doors,
                                                    selected_width, selected_obstacle)
                    running_sim = True
                    stats_printed = False

                if graph_button.is_clicked(mouse_pos) and simulation is not None \
                        and simulation.is_finished():
                    histogram_path = os.path.join(RESULTS_DIR, "evacuation_histogram.png")
                    plot_evacuation_histogram(simulation.evacuation_times,
                                              save_path=histogram_path)

        # --- Përditëso simulimin ---
        if running_sim and simulation is not None:
            if not simulation.is_finished():
                simulation.step()
            else:
                running_sim = False
                if not stats_printed:
                    print_summary(simulation.evacuation_times)
                    stats_printed = True

        # --- Vizato ---
        screen.fill(BACKGROUND_COLOR)

        if simulation is not None:
            draw_exits(screen, simulation.environment)
            draw_obstacles(screen, simulation.environment)
            for boid in simulation.boids:
                draw_boid(screen, boid)

        pygame.draw.rect(screen, PANEL_COLOR, (SIM_WIDTH, 0, PANEL_WIDTH, HEIGHT))

        screen.blit(title_font.render("Numri i Dyerve", True, TEXT_COLOR), (px, 25))
        for btn in door_buttons:
            btn.draw(screen, font, mouse_pos)

        screen.blit(title_font.render("Gjerësia e Derës", True, TEXT_COLOR), (px, 175))
        for btn in width_buttons:
            btn.draw(screen, font, mouse_pos)

        screen.blit(title_font.render("Densiteti (Boid-e)", True, TEXT_COLOR), (px, 325))
        for btn in density_buttons:
            btn.draw(screen, font, mouse_pos)

        screen.blit(title_font.render("Pengesa", True, TEXT_COLOR), (px, 475))
        for btn in obstacle_buttons:
            btn.draw(screen, font, mouse_pos)

        start_button.draw(screen, font, mouse_pos)

        if simulation is not None and simulation.is_finished():
            graph_button.draw(screen, font, mouse_pos)

        if simulation is not None:
            remaining = len(simulation.boids)
            status_text = f"Aktivë: {remaining}   Koha: {simulation.time_elapsed}"
            screen.blit(font.render(status_text, True, TEXT_COLOR), (px, 705))

            if simulation.is_finished():
                screen.blit(font.render("✓ Evakuimi përfundoi!", True, EXIT_COLOR), (px, 728))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()