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

SIM_WIDTH, SIM_HEIGHT = 900, 700
PANEL_WIDTH = 260
WIDTH, HEIGHT = SIM_WIDTH + PANEL_WIDTH, SIM_HEIGHT
FPS = 60

BACKGROUND_COLOR = (20, 20, 30)
PANEL_COLOR = (35, 35, 48)
BOID_COLOR = (100, 200, 255)
EXIT_COLOR = (80, 220, 100)
OBSTACLE_COLOR = (200, 80, 80)
CIRCLE_OBSTACLE_COLOR = (200, 120, 80)
PREVIEW_COLOR = (200, 80, 80, 120)
BUTTON_COLOR = (60, 60, 80)
BUTTON_HOVER_COLOR = (80, 80, 105)
BUTTON_SELECTED_COLOR = (70, 130, 180)
MODE_ACTIVE_COLOR = (200, 150, 60)
TEXT_COLOR = (230, 230, 230)


class Button:
    def __init__(self, x, y, w, h, label, value):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.value = value
        self.selected = False

    def draw(self, screen, font, mouse_pos, color_override=None):
        if color_override:
            color = color_override
        elif self.selected:
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
    return [Button(x, y_start + i * (h + gap), w, h, label, value)
            for i, (label, value) in enumerate(items)]


def draw_boid(screen, boid, size=6):
    angle = np.arctan2(boid.velocity[1], boid.velocity[0])
    points = [(size * 2, 0), (-size, size), (-size, -size)]
    rotated_points = []
    for px, py in points:
        rx = px * np.cos(angle) - py * np.sin(angle)
        ry = px * np.sin(angle) + py * np.cos(angle)
        rotated_points.append((boid.position[0] + rx, boid.position[1] + ry))
    pygame.draw.polygon(screen, BOID_COLOR, rotated_points)


def exit_spec_to_point(side, pos, width, height):
    if side == "top":
        return (pos, 0.0)
    elif side == "bottom":
        return (pos, height)
    elif side == "left":
        return (0.0, pos)
    else:
        return (width, pos)


def draw_exits_from_specs(screen, exit_specs, exit_width, sim_width, sim_height, thickness=6):
    """Vizaton daljet duke përshtatur orientimin (horizontal/vertikal) sipas anës."""
    half = exit_width / 2
    for (side, pos) in exit_specs:
        if side in ("top", "bottom"):
            y = 0 if side == "top" else sim_height
            pygame.draw.line(screen, EXIT_COLOR, (pos - half, y), (pos + half, y), thickness)
        else:
            x = 0 if side == "left" else sim_width
            pygame.draw.line(screen, EXIT_COLOR, (x, pos - half), (x, pos + half), thickness)


def determine_wall_side(mouse_pos, sim_width, sim_height, edge_margin=30):
    """
    Përcakton në cilin nga 4 muret ka klikuar përdoruesi (bazuar te
    ana më e afërt), dhe kthen (side, pos) - None nëse klikimi është
    shumë larg nga çdo mur.
    """
    x, y = mouse_pos
    dist_top, dist_bottom = y, sim_height - y
    dist_left, dist_right = x, sim_width - x
    min_dist = min(dist_top, dist_bottom, dist_left, dist_right)

    if min_dist > edge_margin:
        return None

    if min_dist == dist_top:
        return ("top", x)
    elif min_dist == dist_bottom:
        return ("bottom", x)
    elif min_dist == dist_left:
        return ("left", y)
    else:
        return ("right", y)


def draw_obstacles_from_list(screen, obstacles):
    for (ox, oy, ow, oh) in obstacles:
        pygame.draw.rect(screen, OBSTACLE_COLOR, (ox, oy, ow, oh))

def draw_circle_obstacles_from_list(screen, circles):
    for (cx, cy, radius) in circles:
        pygame.draw.circle(screen, CIRCLE_OBSTACLE_COLOR, (int(cx), int(cy)), int(radius))

def sync_from_simulation_if_active(simulation, custom_exits, custom_obstacles,
                                    custom_circle_obstacles):
    if simulation is not None:
        custom_exits[:] = list(simulation.environment.exit_specs)
        custom_obstacles[:] = list(simulation.environment.obstacles)
        custom_circle_obstacles[:] = list(simulation.environment.circle_obstacles)
        return None, False
    return simulation, None

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids Crowd Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 15)
    small_font = pygame.font.SysFont("Arial", 12)
    title_font = pygame.font.SysFont("Arial", 17, bold=True)

    # --- Gjendja e konfigurimit ---
    selected_width = 15.0
    selected_density = 80
    custom_exits = [("top", SIM_WIDTH / 2)]   # default: 1 derë në mes të murit sipër
    custom_obstacles = []                      # default: pa pengesa
    custom_circle_obstacles = []  # default: pa pengesa rrethore

    # --- Mënyra e vendosjes (placement mode) ---
    # None = normal (kontrollon simulimin), "door" = vendos dyer,
    # "obstacle" = vendos pengesa (drag për madhësi)
    placement_mode = None
    dragging_from = None  # pika e fillimit të drag-ut për pengesë

    px = SIM_WIDTH + 20
    bw, bh, gap = PANEL_WIDTH - 40, 28, 6

    width_buttons = make_button_group(px, 30, bw, bh, gap,
        [("E ngushtë (15px)", 15.0), ("Mesatare (30px)", 30.0), ("E gjerë (60px)", 60.0)])
    width_buttons[0].selected = True

    density_buttons = make_button_group(px, 165, bw, bh, gap,
        [("40 boid (ulët)", 40), ("80 boid (mesatar)", 80), ("150 boid (lartë)", 150)])
    density_buttons[1].selected = True

    door_mode_btn = Button(px, 300, bw, 32, "🚪 Vendos Dyer (klikim)", "door")
    obstacle_mode_btn = Button(px, 336, bw, 32, "▦ Pengesë Drejtk. (zvarrit)", "obstacle")
    circle_mode_btn = Button(px, 372, bw, 32, "● Pengesë Rrethore (zvarrit)", "circle")
    clear_doors_btn = Button(px, 412, bw, 26, "Pastro Dyert", "clear_doors")
    clear_obstacles_btn = Button(px, 442, bw, 26, "Pastro Pengesat", "clear_obstacles")

    start_button = Button(px, 490, bw, 38, "▶  Fillo Simulimin", None)
    graph_button = Button(px, 536, bw, 38, "📊  Shfaq Grafikun", None)
    selection_groups = [width_buttons, density_buttons]

    simulation = None
    running_sim = False
    stats_printed = False

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        in_sim_area = mouse_pos[0] < SIM_WIDTH and mouse_pos[1] < SIM_HEIGHT

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                # --- Klikime brenda zonës së simulimit (placement) ---
                if in_sim_area and placement_mode == "door":
                    result = determine_wall_side(mouse_pos, SIM_WIDTH, SIM_HEIGHT)
                    if result is not None:
                        custom_exits.append(result)

                elif in_sim_area and placement_mode == "obstacle":
                    dragging_from = mouse_pos

                elif in_sim_area and placement_mode == "circle":
                    dragging_from = mouse_pos

                elif not in_sim_area:
                    # --- Klikime te paneli (butona) ---
                    for group in selection_groups:
                        for btn in group:
                            if btn.is_clicked(mouse_pos):
                                for other in group:
                                    other.selected = False
                                btn.selected = True

                    for btn in width_buttons:
                        if btn.selected:
                            selected_width = btn.value
                    for btn in density_buttons:
                        if btn.selected:
                            selected_density = btn.value

                    if door_mode_btn.is_clicked(mouse_pos):
                        simulation, running_sim = sync_from_simulation_if_active(
                            simulation, custom_exits, custom_obstacles, custom_circle_obstacles)
                        placement_mode = None if placement_mode == "door" else "door"
                    if obstacle_mode_btn.is_clicked(mouse_pos):
                        simulation, running_sim = sync_from_simulation_if_active(
                            simulation, custom_exits, custom_obstacles, custom_circle_obstacles)
                        placement_mode = None if placement_mode == "obstacle" else "obstacle"
                    if circle_mode_btn.is_clicked(mouse_pos):
                        simulation, running_sim = sync_from_simulation_if_active(
                            simulation, custom_exits, custom_obstacles, custom_circle_obstacles)
                        placement_mode = None if placement_mode == "circle" else "circle"
                    if clear_doors_btn.is_clicked(mouse_pos):
                        simulation, running_sim = sync_from_simulation_if_active(
                            simulation, custom_exits, custom_obstacles, custom_circle_obstacles)
                        custom_exits = []
                    if clear_obstacles_btn.is_clicked(mouse_pos):
                        simulation, running_sim = sync_from_simulation_if_active(
                            simulation, custom_exits, custom_obstacles, custom_circle_obstacles)
                        custom_obstacles = []
                        custom_circle_obstacles = []

                    if start_button.is_clicked(mouse_pos) and len(custom_exits) > 0:
                        simulation = Simulation(selected_density, SIM_WIDTH, SIM_HEIGHT,
                                                custom_exits, obstacles=custom_obstacles,
                                                exit_width=selected_width,
                                                circle_obstacles=custom_circle_obstacles)
                        running_sim = True
                        stats_printed = False

                    if graph_button.is_clicked(mouse_pos) and simulation is not None \
                            and simulation.is_finished():
                        histogram_path = os.path.join(RESULTS_DIR, "evacuation_histogram.png")
                        plot_evacuation_histogram(simulation.evacuation_times,
                                                   save_path=histogram_path)

            if event.type == pygame.MOUSEBUTTONUP:
                if dragging_from is not None and in_sim_area:
                    if placement_mode == "obstacle":
                        x1, y1 = dragging_from
                        x2, y2 = mouse_pos
                        ox, oy = min(x1, x2), min(y1, y2)
                        ow, oh = abs(x2 - x1), abs(y2 - y1)
                        if ow > 5 and oh > 5:
                            custom_obstacles.append((ox, oy, ow, oh))

                    elif placement_mode == "circle":
                        cx, cy = dragging_from
                        radius = np.linalg.norm(np.array(mouse_pos) - np.array(dragging_from))
                        if radius > 5:
                            custom_circle_obstacles.append((cx, cy, radius))

                dragging_from = None

        # --- Përditëso simulimin (vetëm jashtë placement mode) ---
        if running_sim and simulation is not None and placement_mode is None:
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
            draw_exits_from_specs(screen, simulation.environment.exit_specs,
                                  simulation.environment.exit_width, SIM_WIDTH, SIM_HEIGHT)
            draw_obstacles_from_list(screen, simulation.environment.obstacles)
            draw_circle_obstacles_from_list(screen, simulation.environment.circle_obstacles)
            for boid in simulation.boids:
                draw_boid(screen, boid)
        else:
            draw_exits_from_specs(screen, custom_exits, selected_width, SIM_WIDTH, SIM_HEIGHT)
            draw_obstacles_from_list(screen, custom_obstacles)
            draw_circle_obstacles_from_list(screen, custom_circle_obstacles)

        # Preview i pengesës gjatë "drag"
        if dragging_from is not None:
            if placement_mode == "obstacle":
                x1, y1 = dragging_from
                x2, y2 = mouse_pos
                ox, oy = min(x1, x2), min(y1, y2)
                ow, oh = abs(x2 - x1), abs(y2 - y1)
                preview_surface = pygame.Surface((max(ow, 1), max(oh, 1)), pygame.SRCALPHA)
                preview_surface.fill((200, 80, 80, 120))
                screen.blit(preview_surface, (ox, oy))

            elif placement_mode == "circle":
                cx, cy = dragging_from
                radius = int(np.linalg.norm(np.array(mouse_pos) - np.array(dragging_from)))
                preview_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(preview_surface, (200, 120, 80, 120),
                                   (radius, radius), radius)
                screen.blit(preview_surface, (cx - radius, cy - radius))

        pygame.draw.rect(screen, PANEL_COLOR, (SIM_WIDTH, 0, PANEL_WIDTH, HEIGHT))

        screen.blit(title_font.render("Gjerësia e Derës", True, TEXT_COLOR), (px, 5))
        for btn in width_buttons:
            btn.draw(screen, font, mouse_pos)

        screen.blit(title_font.render("Densiteti (Boid-e)", True, TEXT_COLOR), (px, 140))
        for btn in density_buttons:
            btn.draw(screen, font, mouse_pos)

        door_mode_btn.draw(screen, small_font, mouse_pos,
                           MODE_ACTIVE_COLOR if placement_mode == "door" else None)
        obstacle_mode_btn.draw(screen, small_font, mouse_pos,
                               MODE_ACTIVE_COLOR if placement_mode == "obstacle" else None)
        circle_mode_btn.draw(screen, small_font, mouse_pos,
                             MODE_ACTIVE_COLOR if placement_mode == "circle" else None)
        clear_doors_btn.draw(screen, small_font, mouse_pos)
        clear_obstacles_btn.draw(screen, small_font, mouse_pos)

        start_button.draw(screen, font, mouse_pos)
        if simulation is not None and simulation.is_finished():
            graph_button.draw(screen, font, mouse_pos)

        info_y = 570
        screen.blit(small_font.render(f"Dyer: {len(custom_exits)}", True, TEXT_COLOR), (px, info_y))
        screen.blit(small_font.render(f"Pengesa: {len(custom_obstacles)}", True, TEXT_COLOR), (px, info_y + 20))

        if placement_mode == "door":
            hint = small_font.render("Klikoni mbi hapësirën për derë", True, MODE_ACTIVE_COLOR)
            screen.blit(hint, (px, info_y + 45))
        elif placement_mode == "obstacle":
            hint = small_font.render("Zvarritni për pengesë", True, MODE_ACTIVE_COLOR)
            screen.blit(hint, (px, info_y + 45))

        if simulation is not None:
            remaining = len(simulation.boids)
            status_text = f"Aktivë: {remaining}   Koha: {simulation.time_elapsed}"
            screen.blit(font.render(status_text, True, TEXT_COLOR), (px, 660))
            if simulation.is_finished():
                screen.blit(font.render("✓ Evakuimi përfundoi!", True, EXIT_COLOR), (px, 682))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()