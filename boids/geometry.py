import numpy as np


def normalize_obstacle(obstacle):
    """(ox,oy,ow,oh) -> (ox,oy,ow,oh,0.0); (ox,oy,ow,oh,angle) mbetet i pandryshuar."""
    if len(obstacle) == 5:
        return obstacle
    ox, oy, ow, oh = obstacle
    return (ox, oy, ow, oh, 0.0)


def rect_center(ox, oy, ow, oh):
    return np.array([ox + ow / 2.0, oy + oh / 2.0])


def to_local_space(point, center, angle):
    dx, dy = point[0] - center[0], point[1] - center[1]
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    return np.array([dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a])


def to_world_space(local_point, center, angle):
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x = local_point[0] * cos_a - local_point[1] * sin_a
    y = local_point[0] * sin_a + local_point[1] * cos_a
    return np.array([x + center[0], y + center[1]])


def closest_point_on_rotated_rect(position, ox, oy, ow, oh, angle):
    center = rect_center(ox, oy, ow, oh)
    local = to_local_space(position, center, angle)
    half_w, half_h = ow / 2.0, oh / 2.0

    inside = abs(local[0]) < half_w and abs(local[1]) < half_h

    if inside:
        # Brenda pengesës - gjej SKAJIN MË TË AFËRT (jo clamp, që s'ndryshon
        # asgjë kur je brenda) - njësoj si llogaritja origjinale
        # dist_left/right/top/bottom, por e përgjithësuar për çdo kënd
        dist_right = half_w - local[0]
        dist_left = local[0] + half_w
        dist_top = half_h - local[1]
        dist_bottom = local[1] + half_h

        min_dist = min(dist_right, dist_left, dist_top, dist_bottom)

        if min_dist == dist_right:
            closest_local = np.array([half_w, local[1]])
        elif min_dist == dist_left:
            closest_local = np.array([-half_w, local[1]])
        elif min_dist == dist_top:
            closest_local = np.array([local[0], half_h])
        else:
            closest_local = np.array([local[0], -half_h])
    else:
        clamped_x = np.clip(local[0], -half_w, half_w)
        clamped_y = np.clip(local[1], -half_h, half_h)
        closest_local = np.array([clamped_x, clamped_y])

    closest_world = to_world_space(closest_local, center, angle)
    return closest_world, inside


def point_in_rotated_rect(point, ox, oy, ow, oh, angle, inflate=0.0):
    center = rect_center(ox, oy, ow, oh)
    local = to_local_space(point, center, angle)
    return abs(local[0]) < ow / 2.0 + inflate and abs(local[1]) < oh / 2.0 + inflate


def get_rect_corners(ox, oy, ow, oh, angle):
    center = rect_center(ox, oy, ow, oh)
    half_w, half_h = ow / 2.0, oh / 2.0
    local_corners = [(-half_w, -half_h), (half_w, -half_h),
                      (half_w, half_h), (-half_w, half_h)]
    return [to_world_space(np.array(c), center, angle) for c in local_corners]


def rotated_rect_bounding_box(ox, oy, ow, oh, angle, inflate=0.0):
    """Kthen (min_x, min_y, max_x, max_y) - kutia AABB që përmban të gjitha
    4 qoshet e rrotulluara. Përdoret nga grid rasterization (pathfinding.py,
    polygon_room.py) për të ditur cilat qeliza fare duhen kontrolluar,
    para se të bëhet testi i saktë point_in_rotated_rect për secilën."""
    corners = get_rect_corners(ox, oy, ow, oh, angle)
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs) - inflate, min(ys) - inflate,
            max(xs) + inflate, max(ys) + inflate)



def rotate_vector(vec, angle):
    """Rrotullon VETËM drejtimin e një vektori (shpejtësi) - pa translim,
    sepse shpejtësia s'ka pozicion, vetëm drejtim/madhësi."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([vec[0] * cos_a - vec[1] * sin_a,
                      vec[0] * sin_a + vec[1] * cos_a])


def resolve_axis_aligned_collision_local(local_pos, local_vel, ow, oh):
    """
    EKZAKTËSISHT algoritmi origjinal i resolve_collisions (dist_left/
    right/top/bottom + kicks +1.0/+2.0), por duke punuar në koordinata
    LOKALE (origjina = qendra e pengesës, pa rrotullim). Për angle=0,
    hapësira lokale ËSHTË hapësira botërore e zhvendosur vetëm nga
    qendra - kështu formula prodhon saktësisht të njëjtin rezultat
    numerik si origjinali (mund të verifikohet duke krahasuar këtë
    funksion me degën "if angle == 0.0" te resolve_collisions).
    Kthen (new_local_pos, new_local_vel).
    """
    half_w, half_h = ow / 2.0, oh / 2.0
    lx, ly = local_pos[0], local_pos[1]

    dist_left = lx - (-half_w)
    dist_right = half_w - lx
    dist_top = ly - (-half_h)
    dist_bottom = half_h - ly

    min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

    new_pos = np.array([lx, ly])
    new_vel = np.array(local_vel, dtype=float)

    if min_dist == dist_left:
        new_pos[0] = -half_w - 1
        new_vel[0] = -abs(new_vel[0]) - 1.0
    elif min_dist == dist_right:
        new_pos[0] = half_w + 1
        new_vel[0] = abs(new_vel[0]) + 1.0
    elif min_dist == dist_top:
        new_pos[1] = -half_h - 1
        new_vel[1] = -abs(new_vel[1])
        if lx < 0:
            new_vel[0] -= 2.0
        else:
            new_vel[0] += 2.0
    else:
        new_pos[1] = half_h + 1
        new_vel[1] = abs(new_vel[1])
        if lx < 0:
            new_vel[0] -= 2.0
        else:
            new_vel[0] += 2.0

    return new_pos, new_vel