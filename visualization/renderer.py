import pygame
import numpy as np
import sys
import os
import ctypes
import ctypes.wintypes
import pygame.gfxdraw
import time
from boids.geometry import normalize_obstacle, get_rect_corners

DEBUG_COLLISIONS = False
DEBUG_PERF = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from boids.simulation import Simulation
from boids.polygon_simulation import PolygonSimulation
from analysis.metrics import print_summary, plot_evacuation_histogram

SIM_WIDTH, SIM_HEIGHT = 900, 700
PANEL_WIDTH = 260
WIDTH, HEIGHT = SIM_WIDTH + PANEL_WIDTH, SIM_HEIGHT
FPS = 60

BACKGROUND_COLOR = (20, 20, 30)
PANEL_COLOR = (35, 35, 48)
BOID_COLOR = (100, 200, 255)
EXIT_COLOR = (80, 220, 100)
WALL_COLOR = (230, 230, 230)
WALL_PREVIEW_COLOR = (150, 150, 150)
OBSTACLE_COLOR = (200, 80, 80)
CIRCLE_OBSTACLE_COLOR = (200, 120, 80)
PREVIEW_COLOR = (200, 80, 80, 120)
BUTTON_COLOR = (60, 60, 80)
BUTTON_HOVER_COLOR = (80, 80, 105)
BUTTON_SELECTED_COLOR = (70, 130, 180)
MODE_ACTIVE_COLOR = (200, 150, 60)
TEXT_COLOR = (230, 230, 230)


class Button:
    """
    Buton i thjeshtë, i klikueshëm për panelin e UI-it, me gjendje
    hover/selected dhe një 'value' arbitrar që identifikon veprimin
    që kryen kur klikohet.
    """

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
    """Krijon një kolonë vertikale butonash nga lista 'items'
    (label, value), të pozicionuar automatikisht sipas 'gap'."""
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


def draw_thick_line(screen, color, p1, p2, thickness=3):
    """
    Vizaton një vijë të trashë si poligon ME ANTI-ALIASING (pygame.
    gfxdraw), sepse pygame.draw.polygon standarde nuk zbut skajet
    (aliasing i dukshëm/efekt 'shkallë' në kënde diagonale, jo
    problem i gjeometrisë vetë).
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length == 0:
        return
    unit = direction / length
    perp = np.array([-unit[1], unit[0]]) * (thickness / 2)

    points = [
        (int(round((p1 + perp)[0])), int(round((p1 + perp)[1]))),
        (int(round((p2 + perp)[0])), int(round((p2 + perp)[1]))),
        (int(round((p2 - perp)[0])), int(round((p2 - perp)[1]))),
        (int(round((p1 - perp)[0])), int(round((p1 - perp)[1]))),
    ]

    pygame.gfxdraw.aapolygon(screen, points, color)
    pygame.gfxdraw.filled_polygon(screen, points, color)


def draw_thick_polyline(screen, color, points, thickness=3, closed=True):
    """Vizaton një seri vijash të trasha (mur) duke përdorur draw_thick_line për çdo segment."""
    n = len(points)
    edges = n if closed else n - 1
    for i in range(edges):
        draw_thick_line(screen, color, points[i], points[(i + 1) % n], thickness)


def exit_spec_to_point(side, pos, width, height):
    if side == "top":
        return (pos, 0.0)
    elif side == "bottom":
        return (pos, height)
    elif side == "left":
        return (0.0, pos)
    else:
        return (width, pos)


def draw_exits_from_specs(screen, exit_specs, sim_width, sim_height, thickness=6):
    """Vizaton daljet duke përshtatur orientimin (horizontal/vertikal) sipas anës."""
    for (side, pos, width) in exit_specs:
        half = width / 2
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
    for obstacle in obstacles:
        ox, oy, ow, oh, angle = normalize_obstacle(obstacle)
        if angle == 0.0:
            pygame.draw.rect(screen, OBSTACLE_COLOR, (ox, oy, ow, oh))
        else:
            corners = get_rect_corners(ox, oy, ow, oh, angle)
            points = [(int(x), int(y)) for x, y in corners]
            pygame.draw.polygon(screen, OBSTACLE_COLOR, points)

def draw_circle_obstacles_from_list(screen, circles):
    for (cx, cy, radius) in circles:
        pygame.draw.circle(screen, CIRCLE_OBSTACLE_COLOR, (int(cx), int(cy)), int(radius))

# Kur simulimi është aktiv dhe përdoruesi ndryshon placement mode
# (p.sh. hap "Vendos Dyer" ndërkohë që simulimi po ndodh), duhet të
# kapim gjendjen aktuale (dyer/pengesa) mbrapa te variablat "custom_*"
# dhe të ndalim simulimin - kështu placement-i i ri shtohet mbi
# gjendjen ekzistuese, jo mbi një listë bosh që do fshinte çdo gjë
# që ishte vendosur më parë.
def sync_from_simulation_if_active(simulation, custom_exits, custom_obstacles,
                                    custom_circle_obstacles):
    if simulation is not None:
        # Kap gjendjen aktuale mbrapa te "custom_*" pavarësisht cilit
        # lloj simulimi po vraponte (dhomë 4-mure ose poligon).
        if isinstance(simulation, PolygonSimulation):
            custom_obstacles[:] = list(simulation.room.obstacles)
            custom_circle_obstacles[:] = list(simulation.room.circle_obstacles)
            # Dyert e poligonit s'i shtojmë te custom_exits (ai është
            # vetëm për dhomën me 4 mure) - i lëmë siç janë.
        else:
            custom_exits[:] = list(simulation.environment.exit_specs)
            custom_obstacles[:] = list(simulation.environment.obstacles)
            custom_circle_obstacles[:] = list(simulation.environment.circle_obstacles)
        return None, False
    return simulation, None

def main():
    global SIM_WIDTH, SIM_HEIGHT, WIDTH, HEIGHT

    pygame.init()

    # Krijo fillimisht një dritare të vogël, të ripërmasueshme - do ta
    # maksimizojmë menjëherë përmes Windows API, i cili e llogarit vetë
    # saktë zonën e disponueshme (duke përjashtuar taskbar-in), pa
    # nevojë për hamendje manuale të title bar-it/kufijve.
    screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
    pygame.display.set_caption("Boids Crowd Simulation")

    try:
        hwnd = pygame.display.get_wm_info()["window"]
        SW_MAXIMIZE = 3
        ctypes.windll.user32.ShowWindow(hwnd, SW_MAXIMIZE)

        # Lexo madhësinë REALE të zonës së brendshme (client area) pas
        # maksimizimit - kjo është saktësisht ajo çka mund ta shohim,
        # pa llogaritur vetë title bar/border.
        client_rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(client_rect))
        actual_width = client_rect.right - client_rect.left
        actual_height = client_rect.bottom - client_rect.top

        WIDTH, HEIGHT = actual_width, actual_height
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
    except Exception:
        # Fallback nëse s'jemi në Windows - përdor rezolucionin e ekranit.
        display_info = pygame.display.Info()
        WIDTH = display_info.current_w
        HEIGHT = display_info.current_h - 70
        screen = pygame.display.set_mode((WIDTH, HEIGHT))

    SIM_WIDTH = WIDTH - PANEL_WIDTH
    SIM_HEIGHT = HEIGHT
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 15)
    small_font = pygame.font.SysFont("Arial", 12)
    title_font = pygame.font.SysFont("Arial", 17, bold=True)

    # --- Gjendja e konfigurimit. ---
    selected_width = 15.0
    selected_density = 80
    custom_exits = [("top", SIM_WIDTH / 2, 15.0)]   # default: 1 derë në mes të murit sipër.
    custom_obstacles = []                      # default: pa pengesa.
    custom_circle_obstacles = []  # default: pa pengesa rrethore.

    # --- Gjendja për dhomë me formë të lirë (poligon). ---
    wall_vertices = []  # pikat e klikuar për të ndërtuar poligonin.
    wall_doors = []  # (seg_idx, t_center, half_width).
    use_custom_room = False  # nëse True, Fillo Simulimin përdor PolygonSimulation.
    room_closed = False  # nëse poligoni është mbyllur (kthyer te pika e parë).

    # --- Mënyra e vendosjes (placement mode). ---
    # None = normal (kontrollon simulimin), "door" = vendos dyer,
    # "obstacle" = vendos pengesa (drag për madhësi).
    placement_mode = None
    dragging_from = None  # pika e fillimit të drag-ut për pengesë.

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
    delete_door_btn = Button(px, 472, bw, 26, "Fshij Derë (klikim)", "delete_door")
    delete_obstacle_btn = Button(px, 502, bw, 26, "Fshij Pengesë (klikim)", "delete_obstacle")
    start_button = Button(px, 534, bw, 38, "▶  Fillo Simulimin", None)
    graph_button = Button(px, 580, bw, 38, "📊  Shfaq Grafikun", None)

    wall_mode_btn = Button(px, 582, bw, 28, "✏️ Vizato Mur (klikim)", "wall")
    wall_door_mode_btn = Button(px, 612, bw, 28, "🚪 Dyer në Mur (klikim)", "wall_door")
    clear_walls_btn = Button(px, 642, bw, 24, "Pastro Murin e Personalizuar", "clear_walls")

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
                # Klikime brenda zonës së simulimit (placement).
                if in_sim_area and placement_mode == "door" and not (use_custom_room and room_closed):
                    result = determine_wall_side(mouse_pos, SIM_WIDTH, SIM_HEIGHT)
                    if result is not None:
                        custom_exits.append((result[0], result[1], selected_width))

                # Pengesë drejtkëndëshe - regjistron vetëm pikën e fillimit,
                # madhësia përcaktohet në MOUSEBUTTONUP (drag).
                elif in_sim_area and placement_mode == "obstacle":
                    dragging_from = mouse_pos

                # Pengesë rrethore - njësoj, radius llogaritet në MOUSEBUTTONUP.
                elif in_sim_area and placement_mode == "circle":
                    dragging_from = mouse_pos

                # Ndërtim poligoni: çdo klikim shton kulm; klikimi pranë kulmit
                # të parë (< 15px) e mbyll formën dhe aktivizon PolygonSimulation.
                elif in_sim_area and placement_mode == "wall" and not room_closed:
                    if len(wall_vertices) >= 3 and \
                            np.linalg.norm(np.array(mouse_pos) - np.array(wall_vertices[0])) < 15:
                        room_closed = True
                        use_custom_room = True
                    else:
                        wall_vertices.append(mouse_pos)

                # Vendos derë mbi murin e personalizuar: gjen segmentin më të
                # afërt me klikimin dhe regjistron pozicionin relativ (t) mbi të.
                elif in_sim_area and placement_mode == "wall_door" and room_closed:
                    n = len(wall_vertices)
                    best_seg, best_dist, best_t = -1, np.inf, 0.0
                    for i in range(n):
                        a = np.array(wall_vertices[i])
                        b = np.array(wall_vertices[(i + 1) % n])
                        ab = b - a
                        length_sq = np.dot(ab, ab)
                        t = 0.0 if length_sq == 0 else \
                            np.clip(np.dot(np.array(mouse_pos) - a, ab) / length_sq, 0.0, 1.0)
                        closest = a + t * ab
                        d = np.linalg.norm(np.array(mouse_pos) - closest)
                        if d < best_dist:
                            best_dist, best_seg, best_t = d, i, t
                    if best_dist < 20:
                        half_width_t = (selected_width / 2) / \
                                       np.linalg.norm(np.array(wall_vertices[(best_seg + 1) % n]) -
                                                      np.array(wall_vertices[best_seg]))
                        wall_doors.append((best_seg, best_t, selected_width / 2))

                elif in_sim_area and placement_mode == "delete_obstacle":
                    mx, my = mouse_pos
                    removed = False
                    for i, obstacle in enumerate(custom_obstacles):
                        ox, oy, ow, oh = obstacle[:4]
                        if ox <= mx <= ox + ow and oy <= my <= oy + oh:
                            del custom_obstacles[i]
                            removed = True
                            break
                    if not removed:
                        for i, (cx, cy, radius) in enumerate(custom_circle_obstacles):
                            if (mx - cx) ** 2 + (my - cy) ** 2 <= radius ** 2:
                                del custom_circle_obstacles[i]
                                break

                elif in_sim_area and placement_mode == "delete_door":
                    mx, my = mouse_pos
                    if use_custom_room and room_closed:
                        for i, (seg_idx, t_center, half_width) in enumerate(wall_doors):
                            n = len(wall_vertices)
                            a = np.array(wall_vertices[seg_idx])
                            b = np.array(wall_vertices[(seg_idx + 1) % n])
                            center = a + t_center * (b - a)
                            if np.linalg.norm(np.array(mouse_pos) - center) < half_width + 10:
                                del wall_doors[i]
                                break
                    else:
                        for i, (side, pos, width) in enumerate(custom_exits):
                            px_, py_ = exit_spec_to_point(side, pos, SIM_WIDTH, SIM_HEIGHT)
                            if np.linalg.norm(np.array(mouse_pos) - np.array([px_, py_])) < width / 2 + 10:
                                del custom_exits[i]
                                break

                elif not in_sim_area:
                    # Klikime te paneli (butona).
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

                    if door_mode_btn.is_clicked(mouse_pos) and not (use_custom_room and room_closed):
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
                        wall_doors = []
                    if clear_obstacles_btn.is_clicked(mouse_pos):
                        simulation, running_sim = sync_from_simulation_if_active(
                            simulation, custom_exits, custom_obstacles, custom_circle_obstacles)
                        custom_obstacles = []
                        custom_circle_obstacles = []
                    if delete_door_btn.is_clicked(mouse_pos):
                        simulation, running_sim = sync_from_simulation_if_active(
                            simulation, custom_exits, custom_obstacles, custom_circle_obstacles)
                        placement_mode = None if placement_mode == "delete_door" else "delete_door"
                    if delete_obstacle_btn.is_clicked(mouse_pos):
                        simulation, running_sim = sync_from_simulation_if_active(
                            simulation, custom_exits, custom_obstacles, custom_circle_obstacles)
                        placement_mode = None if placement_mode == "delete_obstacle" else "delete_obstacle"

                    # Fillon simulimin - degëzohet sipas asaj nëse përdoruesi ka
                    # ndërtuar mur të personalizuar (PolygonSimulation) apo përdor
                    # dhomën standarde me 4 mure (Simulation).
                    if use_custom_room and room_closed and len(wall_doors) > 0:
                        if start_button.is_clicked(mouse_pos):
                            simulation = PolygonSimulation(
                                selected_density, wall_vertices, wall_doors,
                                obstacles=custom_obstacles,
                                circle_obstacles=custom_circle_obstacles,
                                exit_width=selected_width)
                            running_sim = True
                            stats_printed = False
                    elif start_button.is_clicked(mouse_pos) and len(custom_exits) > 0:
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
                    if wall_mode_btn.is_clicked(mouse_pos):
                        placement_mode = None if placement_mode == "wall" else "wall"
                    if wall_door_mode_btn.is_clicked(mouse_pos):
                        placement_mode = None if placement_mode == "wall_door" else "wall_door"
                    if clear_walls_btn.is_clicked(mouse_pos):
                        wall_vertices = []
                        wall_doors = []
                        use_custom_room = False
                        room_closed = False
                        # Nëse ka simulim aktiv me murin e personalizuar, fshije
                        # gjithashtu - përndryshe vizatimi/logjika do të përpiqej
                        # të përdorte wall_vertices (tani bosh) me seg_idx të
                        # vjetër, duke shkaktuar IndexError
                        if simulation is not None and isinstance(simulation, PolygonSimulation):
                            simulation = None
                            running_sim = False

            if event.type == pygame.KEYDOWN:
                if placement_mode == "obstacle" and len(custom_obstacles) > 0:
                    simulation, running_sim = sync_from_simulation_if_active(
                        simulation, custom_exits, custom_obstacles, custom_circle_obstacles)
                    ox, oy, ow, oh, angle = normalize_obstacle(custom_obstacles[-1])
                    if event.key == pygame.K_q:
                        custom_obstacles[-1] = (ox, oy, ow, oh, angle - np.radians(5))
                        if DEBUG_COLLISIONS:
                            pass # print(f"[ROTATE] pengesa e fundit -> {np.degrees(custom_obstacles[-1][4]):.0f}°")
                    elif event.key == pygame.K_e:
                        custom_obstacles[-1] = (ox, oy, ow, oh, angle + np.radians(5))
                        if DEBUG_COLLISIONS:
                            pass # print(f"[ROTATE] pengesa e fundit -> {np.degrees(custom_obstacles[-1][4]):.0f}°")

            if event.type == pygame.MOUSEBUTTONUP:
                if dragging_from is not None and in_sim_area:
                    if placement_mode == "obstacle":
                        x1, y1 = dragging_from
                        x2, y2 = mouse_pos
                        ox, oy = min(x1, x2), min(y1, y2)
                        ow, oh = abs(x2 - x1), abs(y2 - y1)
                        if ow > 5 and oh > 5:
                            custom_obstacles.append((ox, oy, ow, oh, 0.0))

                    elif placement_mode == "circle":
                        cx, cy = dragging_from
                        radius = np.linalg.norm(np.array(mouse_pos) - np.array(dragging_from))
                        if radius > 5:
                            custom_circle_obstacles.append((cx, cy, radius))

                dragging_from = None

        step_start = time.perf_counter()
        step_time = 0.0
        # --- Përditëso simulimin (vetëm jashtë placement mode). ---
        if running_sim and simulation is not None and placement_mode is None:
            if not simulation.is_finished():
                simulation.step()
                step_time = time.perf_counter() - step_start
            else:
                running_sim = False
                if not stats_printed:
                    print_summary(simulation.evacuation_times)
                    stats_printed = True

        # --- Vizato. ---
        # Tre gjendje të mundshme: simulim poligon aktiv, simulim drejtkëndësh
        # aktiv, ose asnjë simulim (ende në modalitetin e konfigurimit/preview).
        draw_start = time.perf_counter()
        screen.fill(BACKGROUND_COLOR)

        if simulation is not None and isinstance(simulation, PolygonSimulation):
            draw_thick_polyline(screen, WALL_COLOR, wall_vertices, thickness=3, closed=True)
            draw_obstacles_from_list(screen, simulation.room.obstacles)
            draw_circle_obstacles_from_list(screen, simulation.room.circle_obstacles)

            n = len(wall_vertices)
            for (seg_idx, t_center, half_width) in simulation.room.doors:
                a = np.array(wall_vertices[seg_idx])
                b = np.array(wall_vertices[(seg_idx + 1) % n])
                center = a + t_center * (b - a)
                edge = b - a
                edge_len = np.linalg.norm(edge)
                if edge_len > 0:
                    unit = edge / edge_len
                    p1 = center - unit * half_width
                    p2 = center + unit * half_width
                    draw_thick_line(screen, EXIT_COLOR, p1, p2, thickness=6)

            for boid in simulation.boids:
                draw_boid(screen, boid)
        elif simulation is not None:
            draw_exits_from_specs(screen, simulation.environment.exit_specs, SIM_WIDTH, SIM_HEIGHT)
            draw_obstacles_from_list(screen, simulation.environment.obstacles)
            draw_circle_obstacles_from_list(screen, simulation.environment.circle_obstacles)
            for boid in simulation.boids:
                draw_boid(screen, boid)
        else:
            draw_exits_from_specs(screen, custom_exits, SIM_WIDTH, SIM_HEIGHT)
            draw_obstacles_from_list(screen, custom_obstacles)
            draw_circle_obstacles_from_list(screen, custom_circle_obstacles)

            # Vizato murin e personalizuar gjatë ndërtimit.
            if len(wall_vertices) > 0:
                if len(wall_vertices) > 1:
                    draw_thick_polyline(screen, WALL_PREVIEW_COLOR, wall_vertices,
                                        thickness=3, closed=room_closed)
                for v in wall_vertices:
                    pygame.draw.circle(screen, WALL_COLOR, v, 4)

                if room_closed:
                    n = len(wall_vertices)
                    for (seg_idx, t_center, half_width) in wall_doors:
                        a = np.array(wall_vertices[seg_idx])
                        b = np.array(wall_vertices[(seg_idx + 1) % n])
                        center = a + t_center * (b - a)
                        edge = b - a
                        edge_len = np.linalg.norm(edge)
                        if edge_len > 0:
                            unit = edge / edge_len
                            p1 = center - unit * half_width
                            p2 = center + unit * half_width
                            draw_thick_line(screen, EXIT_COLOR, p1, p2, thickness=6)

        # Preview i pengesës gjatë "drag".
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
        delete_door_btn.draw(screen, small_font, mouse_pos,
                             MODE_ACTIVE_COLOR if placement_mode == "delete_door" else None)
        delete_obstacle_btn.draw(screen, small_font, mouse_pos,
                                 MODE_ACTIVE_COLOR if placement_mode == "delete_obstacle" else None)
        wall_mode_btn.draw(screen, small_font, mouse_pos,
                           MODE_ACTIVE_COLOR if placement_mode == "wall" else None)
        wall_door_mode_btn.draw(screen, small_font, mouse_pos,
                                MODE_ACTIVE_COLOR if placement_mode == "wall_door" else None)
        clear_walls_btn.draw(screen, small_font, mouse_pos)

        start_button.draw(screen, font, mouse_pos)
        if simulation is not None and simulation.is_finished():
            graph_button.draw(screen, font, mouse_pos)

        info_y = 690
        if use_custom_room and room_closed:
            door_count = len(wall_doors)
        else:
            door_count = len(custom_exits)
        screen.blit(small_font.render(f"Dyer: {door_count}", True, TEXT_COLOR), (px, info_y))

        total_obstacles = len(custom_obstacles) + len(custom_circle_obstacles)
        screen.blit(small_font.render(f"Pengesa: {total_obstacles}", True, TEXT_COLOR), (px, info_y + 20))

        if placement_mode == "door":
            hint = small_font.render("Klikoni mbi hapësirën për derë", True, MODE_ACTIVE_COLOR)
            screen.blit(hint, (px, info_y + 45))
        elif placement_mode == "obstacle":
            hint = small_font.render("Zvarritni | Q/E: rrotullo të fundit",
                                     True, MODE_ACTIVE_COLOR)
            screen.blit(hint, (px, info_y + 45))

        if simulation is not None:
            remaining = len(simulation.boids)
            status_text = f"Aktivë: {remaining}   Koha: {simulation.time_elapsed}"
            screen.blit(font.render(status_text, True, TEXT_COLOR), (px, 770))
            if simulation.is_finished():
                screen.blit(font.render("✓ Evakuimi përfundoi!", True, EXIT_COLOR), (px, 682))

        draw_time = time.perf_counter() - draw_start
        if DEBUG_PERF and simulation is not None:
            pass # print(f"[PERF-RENDER] step={step_time * 1000:.1f}ms draw={draw_time * 1000:.1f}ms")
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()