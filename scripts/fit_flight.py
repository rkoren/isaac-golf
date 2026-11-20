"""Fit the ball's drag and lift to TrackMan's 2023 PGA averages; LPGA stays out as the check.

uv run python scripts/fit_flight.py   (paste the printed constants into flight/ball.py)
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flight import ball, spin  # noqa: E402
from flight.data import LPGA, PGA, row  # noqa: E402

MPH, YD = 0.44704, 0.9144
NAMES = ("CD0", "CD1", "CL_MAX", "CL_S")


def flown(table):
    r = [row(table, c) for c in table]
    vel = ball.launch_vector(np.array([x["ball_mph"] for x in r]) * MPH, np.array([x["launch_deg"] for x in r]))
    out = ball.fly(vel, ball.spin_vector(np.array([x["spin_rpm"] for x in r]), np.zeros(len(r)), vel))
    want = np.array([[x["carry_yd"], x["height_yd"], x["land_deg"]] for x in r])
    return np.stack([out["carry"] / YD, out["apex"] / YD, out["land_deg"]], -1), want


def residuals(params):
    for name, value in zip(NAMES, params):
        setattr(ball, name, value)
    got, want = flown(PGA)
    # carry counts most; apex and landing angle shape the arc
    return np.concatenate([(got[:, 0] / want[:, 0] - 1) * 3, got[:, 1] / want[:, 1] - 1, (got[:, 2] - want[:, 2]) / 10])


fit = least_squares(
    residuals, [getattr(ball, n) for n in NAMES], bounds=([0.05, 0, 0.05, 0.01], [0.5, 2, 1.0, 2])
)
residuals(fit.x)
print("fitted:", "  ".join(f"{n} = {v:.4f}" for n, v in zip(NAMES, fit.x)))
for label, table in (("PGA (fitted)", PGA), ("LPGA (held out)", LPGA)):
    got, want = flown(table)
    print(f"\n{label}: carry yd, apex yd, land deg (model / TrackMan)")
    for c, g, w in zip(table, got, want):
        print(f"  {c:7s} {g[0]:5.0f} / {w[0]:3.0f} ({g[0] / w[0] - 1:+5.1%})   {g[1]:4.0f} / {w[1]:2.0f}   {g[2]:4.0f} / {w[2]:2.0f}")
    err = got[:, 0] / want[:, 0] - 1
    print(f"  carry error: mean {err.mean():+.1%}, worst {np.abs(err).max():.1%}")
print("\ncoefficients over the flight's range:")
for s in (0.05, 0.1, 0.2, 0.3, 0.45):
    print(f"  S {s}: CD {ball.CD0 + ball.CD1 * s:.3f}  CL {ball.CL_MAX * (1 - np.exp(-s / ball.CL_S)):.3f}")
print("\nspin, rpm per mph of club speed per sin(spin loft):", "  ".join(f"{c} {k:.0f}" for c, k in sorted(spin.K.items())))
