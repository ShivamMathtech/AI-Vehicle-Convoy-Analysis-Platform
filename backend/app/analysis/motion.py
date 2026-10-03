import math

# Image-plane directions only (y grows downward). Not geographic without camera calibration.
DIRS = ["East", "South-East", "South", "South-West", "West", "North-West", "North", "North-East"]


def direction(p0, p1, min_px=5.0):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if math.hypot(dx, dy) < min_px:
        return "Stationary"
    ang = math.degrees(math.atan2(dy, dx)) % 360
    return DIRS[int(((ang + 22.5) % 360) // 45)]
