"""Golf ball flight: gravity, drag and Magnus lift on a spinning ball, RK4. Frame: x right, y target, z up."""

import numpy as np

G = 9.81
MASS = 0.04593                      # [kg] USGA max
RADIUS = 0.021335                   # [m] USGA min diameter 42.67 mm
AREA = np.pi * RADIUS**2
RHO = 1.184                         # [kg/m^3] sea level, 25 C (TrackMan's normalised conditions)

# aerodynamics vs spin ratio S = r*omega/v, fitted to TrackMan's 2023 PGA averages (scripts/fit_flight.py)
CD0, CD1 = 0.2607, 0.2816           # drag: CD0 + CD1*S
CL_MAX, CL_S = 0.4360, 0.1598       # lift: CL_MAX * (1 - exp(-S/CL_S))
SPIN_TAU = 25.0                     # [s] spin decay time constant, assumed (~4% per second; carry barely depends on it)


def launch_vector(speed, launch_deg, azimuth_deg=None):
    """Ball velocity from speed [m/s], vertical launch and horizontal start angle (+ right) [deg]."""
    el = np.radians(launch_deg)
    az = np.radians(np.zeros_like(el) if azimuth_deg is None else azimuth_deg)
    return speed[:, None] * np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], -1)


def spin_vector(rpm, axis_deg, vel):
    """Backspin about the horizontal axis across the ball's path, tilted by axis_deg (+ curves right) [rad/s]."""
    h = vel[:, :2] / np.linalg.norm(vel[:, :2], axis=-1, keepdims=True)
    right = np.stack([h[:, 1], -h[:, 0], np.zeros(len(h))], -1)
    t = np.radians(axis_deg)[:, None]
    axis = np.cos(t) * right - np.sin(t) * np.array([0.0, 0.0, 1.0])
    return axis * (rpm * 2 * np.pi / 60)[:, None]


def _accel(v, w_hat, w_mag, rho):
    speed = np.sqrt((v * v).sum(-1, keepdims=True)).clip(1e-6)
    s = RADIUS * w_mag / speed
    cd = CD0 + CD1 * s
    cl = CL_MAX * (1.0 - np.exp(-s / CL_S))
    k = 0.5 * rho * AREA / MASS * speed
    # Magnus along spin axis x velocity, written out: np.cross costs more than the rest of the step
    wx, wy, wz = w_hat[:, 0:1], w_hat[:, 1:2], w_hat[:, 2:3]
    vx, vy, vz = v[:, 0:1], v[:, 1:2], v[:, 2:3]
    magnus = np.concatenate([wy * vz - wz * vy, wz * vx - wx * vz, wx * vy - wy * vx], -1)
    a = k * (cl * magnus - cd * v)
    a[:, 2] -= G
    return a


def fly(vel, spin, z0=0.0, dt=0.2, t_max=15.0, rho=RHO, path=False):
    """Fly each ball from height z0 above the ground (x = y = 0) until it lands (z = 0).

    Returns carry (horizontal distance), side (+ right of the y axis), curve (+ right of the start line),
    apex, flight time, landing angle, all [m]/[s]/[deg]; path=True adds positions every step.
    """
    n = len(vel)
    p, v = np.zeros((n, 3)), np.array(vel, dtype=float)
    p[:, 2] = z0
    w0 = np.linalg.norm(spin, axis=-1, keepdims=True)
    w_hat = spin / w0.clip(1e-9)
    land_p, land_v, land_t = np.zeros((n, 3)), np.zeros((n, 3)), np.full(n, t_max)
    landed = np.zeros(n, dtype=bool)
    apex = p[:, 2].copy()
    track = [p.copy()]
    t = 0.0
    while t < t_max and not landed.all():
        wm = lambda tt: w0 * np.exp(-tt / SPIN_TAU)
        a1 = _accel(v, w_hat, wm(t), rho)
        a2 = _accel(v + 0.5 * dt * a1, w_hat, wm(t + 0.5 * dt), rho)
        a3 = _accel(v + 0.5 * dt * a2, w_hat, wm(t + 0.5 * dt), rho)
        a4 = _accel(v + dt * a3, w_hat, wm(t + dt), rho)
        p_new = p + dt * v + dt * dt / 6.0 * (a1 + a2 + a3)
        v_new = v + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        # landing
        cross = ~landed & (p_new[:, 2] < 0.0) & (v_new[:, 2] < 0.0)
        if cross.any():
            f = (p[cross, 2] / (p[cross, 2] - p_new[cross, 2]))[:, None]
            land_p[cross] = p[cross] + f * (p_new[cross] - p[cross])
            land_v[cross] = v[cross] + f * (v_new[cross] - v[cross])
            land_t[cross] = t + f[:, 0] * dt
            landed |= cross
        p = np.where(landed[:, None], land_p, p_new)
        v = np.where(landed[:, None], land_v, v_new)
        apex = np.maximum(apex, p[:, 2])
        t += dt
        if path:
            track.append(p.copy())
    # unlanded balls
    land_p[~landed], land_v[~landed] = p[~landed], v[~landed]
    az = np.arctan2(vel[:, 0], vel[:, 1])
    out = {
        "carry": np.linalg.norm(land_p[:, :2], axis=-1),
        "side": land_p[:, 0],
        "curve": land_p[:, 0] * np.cos(az) - land_p[:, 1] * np.sin(az),
        "apex": apex,
        "time": land_t,
        "land_deg": np.degrees(np.arctan2(-land_v[:, 2], np.linalg.norm(land_v[:, :2], axis=-1))),
    }
    if path:
        out["path"] = np.stack(track, 1)
        out["dt"] = dt
    return out
