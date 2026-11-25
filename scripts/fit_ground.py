"""Fit the fairway's bounce and roll to the USGA/R&A 2021 run line, run [yd] = 79.1 - 1.6 * landing angle [deg].

uv run python scripts/fit_ground.py   (paste the printed constants into flight/ground.py)

mu stays at 0.4: these landings grip, so the line can't see it.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flight import ball, ground  # noqa: E402
from flight.data import LPGA, PGA, row  # noqa: E402

MPH, YD = 0.44704, 0.9144
ANGLES = np.arange(32.0, 45.5, 0.5)
SPEED, RPM = 26.6, 2300.0           # tour-driver landing: ball.fly on the PGA row lands at 27 m/s
START = np.array([11.5, 0.90, 0.75])  # theta0, e_scale, rho


def runs(params, speed=SPEED, rpm=RPM, angles=ANGLES):
    s = ground.Surface("fit", 0.40, params[1], params[0], params[2])
    a = np.radians(angles)
    v = np.stack([np.zeros_like(a), speed * np.cos(a), -speed * np.sin(a)], -1)
    return ground.run(v, ball.spin_vector(np.full(len(a), rpm), np.zeros(len(a)), v), s)["run"] / YD


def residuals(params):
    line = 79.1 - 1.6 * ANGLES
    # a weak pull to the start keeps the three parameters from trading off freely
    return np.concatenate([runs(params) - line, 0.5 * (params - START) / START])


fit = least_squares(residuals, START, bounds=([2.0, 0.3, 0.05], [30.0, 1.2, 1.5]))
theta0, e_scale, rho = fit.x
print(f"FAIRWAY = Surface(\"fairway\", 0.40, {e_scale:.3f}, {theta0:.2f}, {rho:.3f})")
err = runs(fit.x) - (79.1 - 1.6 * ANGLES)
print(f"run against the USGA line over 32-45 deg: worst {np.abs(err).max():.2f} yd, mean {err.mean():+.2f} yd")
for a in (32, 36, 39, 42, 45):
    print(f"  {a} deg: model {runs(fit.x, angles=np.array([float(a)]))[0]:5.1f} yd, line {79.1 - 1.6 * a:5.1f} yd")
print("\nrun against landing speed and spin (not in the fit):")
for sp, rpm in ((22.0, 2300), (30.0, 2300), (26.6, 0), (26.6, 4500)):
    r = runs(fit.x, sp, rpm, np.array([32.0, 38.0, 45.0]))
    print(f"  {sp} m/s, {rpm:.0f} rpm: " + ", ".join(f"{x:.1f} yd at {a}" for x, a in zip(r, ("32", "38", "45"))))
print("\ntour rows run out from ball.fly (carry + run, yd):")
s = ground.Surface("fit", 0.40, e_scale, theta0, rho)
for name, table in (("PGA", PGA), ("LPGA", LPGA)):
    for c in table:
        r = row(table, c)
        vel = ball.launch_vector(np.array([r["ball_mph"] * MPH]), np.array([r["launch_deg"]]))
        o = ball.fly(vel, ball.spin_vector(np.array([float(r["spin_rpm"])]), np.zeros(1), vel))
        g = ground.run(o["land_vel"], o["land_spin"], s)
        print(f"  {name:4s} {c:7s} carry {o['carry'][0] / YD:4.0f}  lands {o['land_deg'][0]:3.0f} deg  run {g['run'][0] / YD:5.1f}")
