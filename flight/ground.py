"""Bounce and roll on turf after landing. Frame: x right, y target, z up (flight/ball.py).

Penner's tilted-plane impact (Can. J. Phys. 80, 2002), generalised to 3D, with the fairway fitted to the USGA/R&A 2021 run line.
"""

from dataclasses import dataclass

import numpy as np

from .ball import G, RADIUS, fly

MAX_BOUNCES = 12
HOP_APEX = 0.03  # [m] a lower hop is rolling


@dataclass(frozen=True)
class Surface:
    name: str
    mu: float       # kinetic friction
    e_scale: float  # on Penner's restitution e(v)
    theta0: float   # [deg] turf tilt at 13.3 m/s down, 44.4 deg off vertical
    rho: float      # rolling resistance


# fitted by scripts/fit_ground.py to run [yd] = 79.1 - 1.6 * landing angle [deg], 32-45 deg (rho is effective, not a measured roll)
FAIRWAY = Surface("fairway", 0.40, 0.889, 11.47, 0.765)


def tilt_deg(vel, surface):
    """Effective turf tilt; quadratic in the descent, so the tilt fades with the bounces (Penner's linear form does not)."""
    vz, vh = np.abs(vel[:, 2]), np.linalg.norm(vel[:, :2], axis=-1)
    phi = np.degrees(np.arctan2(vh, vz))
    return np.minimum(surface.theta0 * (vz / 13.3) ** 2 * (phi / 44.4), 30.0)


def _restitution(vn, surface):
    e = np.where(vn <= 20.0, 0.510 - 0.0375 * vn + 0.000903 * vn**2, 0.120)
    return surface.e_scale * e


def impact(vel, spin, surface):
    """One bounce: velocity and spin after, all (N, 3)."""
    vh = np.linalg.norm(vel[:, :2], axis=-1, keepdims=True)
    h = np.concatenate([vel[:, :2] / vh.clip(1e-9), np.zeros_like(vh)], -1)
    th = np.radians(tilt_deg(vel, surface))[:, None]
    n = np.cos(th) * np.array([0.0, 0.0, 1.0]) - np.sin(th) * h
    vn = (vel * n).sum(-1, keepdims=True)
    jn = (1.0 + _restitution(-vn, surface)) * -vn
    # contact point velocity; friction stops it (grip) or is capped at mu (slip), sphere inertia 2/5 m r^2
    u = vel - vn * n + np.cross(spin, -RADIUS * n)
    un = np.linalg.norm(u, axis=-1, keepdims=True)
    jt = np.minimum(surface.mu * jn, 2.0 / 7.0 * un)
    uh = u / un.clip(1e-9)
    return vel + jn * n - jt * uh, spin + 5.0 / (2.0 * RADIUS) * np.cross(n, uh) * jt


def run(land_vel, land_spin, surface=FAIRWAY, log=None):
    """Bounce and roll from the landing spot: run (+ = forward along the landing heading), rest offset (x, y), bounces, time.

    Balls not descending don't run. log (a list) collects the hops and rolls as dicts, for trajectory().
    """
    v, w = np.array(land_vel, dtype=float), np.array(land_spin, dtype=float)
    n = len(v)
    pos, t = np.zeros((n, 2)), np.zeros(n)
    bounces, hop_s = np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
    moving = v[:, 2] < 0.0
    active = moving.copy()
    h0 = v[:, :2] / np.linalg.norm(v[:, :2], axis=-1, keepdims=True).clip(1e-9)
    for _ in range(MAX_BOUNCES):
        if not active.any():
            break
        v[active], w[active] = impact(v[active], w[active], surface)
        bounces[active] += 1
        # no bounce worth a hop, or the plane kicked it down: it rolls from here
        hop_s = active & (v[:, 2] ** 2 / (2.0 * G) > HOP_APEX) & (v[:, 2] > 0.0)
        th = 2.0 * v[:, 2] / G
        if log is not None:
            for i in np.nonzero(hop_s)[0]:
                log.append({"i": i, "kind": "hop", "t0": t[i], "p0": pos[i].copy(), "v": v[i].copy(), "T": th[i]})
        pos[hop_s] += v[hop_s, :2] * th[hop_s, None]
        t[hop_s] += th[hop_s]
        v[hop_s, 2] *= -1.0
        active = hop_s
    # the balls that stopped bouncing: the slip ends at once, then constant deceleration
    u = v[:, :2] + np.cross(w, np.array([0.0, 0.0, -RADIUS]))[:, :2]
    vr = (v[:, :2] - 2.0 / 7.0 * u) * moving[:, None]
    s = np.linalg.norm(vr, axis=-1)
    a = 5.0 / 7.0 * surface.rho * G
    if log is not None:
        for i in np.nonzero(moving)[0]:
            log.append({"i": i, "kind": "roll", "t0": t[i], "p0": pos[i].copy(), "v": vr[i].copy(), "T": s[i] / a, "a": a})
    pos = pos + vr / s.clip(1e-9)[:, None] * (s**2 / (2.0 * a))[:, None]
    t = t + s / a
    return {"run": (pos * h0).sum(-1) * moving, "rest": pos, "bounces": bounces, "time": t}


def trajectory(vel, spin, z0, surface=FAIRWAY, dt=0.05):
    """One shot from launch to rest: positions (T, 3) every dt, dt, and the landing and rest times [s].

    vel and spin are (1, 3); the flight is stepped finely, so the landing spot is the analytic one to a few tens of cm.
    """
    step = 0.02
    out = fly(vel, spin, z0=z0, dt=step, path=True)
    t_land = float(out["time"][0])
    k = int(np.ceil(t_land / step))
    ts = np.append(np.arange(k) * step, t_land)
    ps = np.concatenate([out["path"][0][:k], out["land_pos"][0][None]])
    events: list = []
    if out["land_vel"][0, 2] < 0.0 and t_land < 15.0:
        g = run(out["land_vel"], out["land_spin"], surface, events)
        t_rest = t_land + float(g["time"][0])
    else:
        t_rest = t_land
    # ground segments, sampled on the flight's fine step
    for ev in events:
        m = max(2, int(np.ceil(ev["T"] / step)))
        s = np.linspace(0.0, ev["T"], m + 1)[1:]
        p0 = out["land_pos"][0, :2] + ev["p0"]
        if ev["kind"] == "hop":
            xy = p0 + np.outer(s, ev["v"][:2])
            z = ev["v"][2] * s - 0.5 * G * s**2
        else:
            speed = np.linalg.norm(ev["v"])
            d = speed * s - 0.5 * ev["a"] * s**2
            xy = p0 + np.outer(d, ev["v"] / max(speed, 1e-9))
            z = np.zeros_like(s)
        ts = np.append(ts, t_land + ev["t0"] + s)
        ps = np.concatenate([ps, np.column_stack([xy, np.maximum(z, 0.0)])])
    grid = np.arange(0.0, t_rest + dt, dt)
    grid[-1] = t_rest
    path = np.column_stack([np.interp(grid, ts, ps[:, i]) for i in range(3)])
    return path, dt, {"land": t_land, "rest": t_rest}
