"""Launch and spin from the club at impact calibrated per club on TrackMan's averages"""

import numpy as np

from .data import LPGA, PGA, row

MPH = 0.44704
W_FACE = 0.85                       # start direction: 85% face normal, 15% club path (GolfEnvCfg.face_launch_weight)


def spin_loft_from_launch(beta_deg, w=W_FACE):
    """Spin loft [deg] whose blended start direction sits beta_deg above the club path."""
    b = np.radians(beta_deg)
    return np.degrees(b + np.arcsin((1.0 - w) / w * np.sin(b)))


def _rpm_per_mph():
    # spin ~ k * club speed * sin(spin loft): the face sliding across the ball; k per club from both tours
    k = {}
    for club in set(PGA) & set(LPGA):
        logs = []
        for table in (PGA, LPGA):
            r = row(table, club)
            sl = spin_loft_from_launch(r["launch_deg"] - r["attack_deg"])
            logs.append(np.log(r["spin_rpm"] / (r["club_mph"] * np.sin(np.radians(sl)))))
        k[club] = float(np.exp(np.mean(logs)))
    return k


K = _rpm_per_mph()
CLUBS = tuple(sorted(K))


def launch(club_vel, normal, ball_speed, club, w=W_FACE):
    """Ball launch from club velocity [m/s] and the lofted face normal at impact, (N, 3) in x right, y target, z up"""
    club_speed = np.linalg.norm(club_vel, axis=-1)
    v_hat = club_vel / club_speed[:, None].clip(1e-6)
    n = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    d = w * n + (1.0 - w) * v_hat
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    spin_loft = np.arccos(np.clip((n * v_hat).sum(-1), -1.0, 1.0))
    rpm = K[club] * club_speed / MPH * np.sin(spin_loft)
    axis = np.cross(v_hat, n)
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True).clip(1e-9)
    deg = lambda x: np.degrees(np.arcsin(np.clip(x, -1.0, 1.0)))
    return {
        "vel": ball_speed[:, None] * d,
        "spin": axis * (rpm * 2 * np.pi / 60)[:, None],
        "spin_rpm": rpm,
        "axis_deg": deg(-axis[:, 2]),
        "launch_deg": deg(d[:, 2]),
        "attack_deg": deg(v_hat[:, 2]),
        "dynamic_loft_deg": deg(n[:, 2]),
        "spin_loft_deg": np.degrees(spin_loft),
    }
