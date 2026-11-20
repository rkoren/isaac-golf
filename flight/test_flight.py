"""Flight model against TrackMan's 2023 tour averages: uv run python -m unittest flight.test_flight -v"""

import math
import unittest

import numpy as np

from flight import ball, spin
from flight.data import LPGA, PGA, row

MPH, YD = 0.44704, 0.9144


def table_flight(table, clubs=None):
    """Each row's published launch (speed, launch angle, spin) flown straight down the target line."""
    clubs = clubs or list(table)
    r = [row(table, c) for c in clubs]
    vel = ball.launch_vector(np.array([x["ball_mph"] for x in r]) * MPH, np.array([x["launch_deg"] for x in r]))
    w = ball.spin_vector(np.array([x["spin_rpm"] for x in r]), np.zeros(len(r)), vel)
    return clubs, r, ball.fly(vel, w)


def delivery(club_mph, attack, spin_loft, face_to_path=0.0, path=0.0):
    """Club velocity and lofted face normal (x right, y target, z up) from angles in degrees."""
    a, p = math.radians(attack), math.radians(path)
    v = np.array([[math.sin(p) * math.cos(a), math.cos(p) * math.cos(a), math.sin(a)]]) * club_mph * MPH
    dl, f = math.radians(attack + spin_loft), math.radians(path + face_to_path)
    n = np.array([[math.sin(f) * math.cos(dl), math.cos(f) * math.cos(dl), math.sin(dl)]])
    return v, n


class Integrator(unittest.TestCase):
    def test_vacuum_matches_projectile(self):
        vel = ball.launch_vector(np.array([40.0]), np.array([30.0]))
        out = ball.fly(vel, np.zeros((1, 3)), rho=0.0, dt=0.01)
        self.assertAlmostEqual(out["carry"][0], 40.0**2 * math.sin(math.radians(60)) / ball.G, delta=0.05)

    def test_time_step_converged(self):
        r = row(PGA, "Driver")
        vel = ball.launch_vector(np.array([r["ball_mph"] * MPH]), np.array([r["launch_deg"]]))
        w = ball.spin_vector(np.array([r["spin_rpm"]]), np.zeros(1), vel)
        coarse, fine = ball.fly(vel, w), ball.fly(vel, w, dt=0.002)
        self.assertLess(abs(coarse["carry"][0] - fine["carry"][0]), 0.5)

    def test_backspin_lifts(self):
        vel = ball.launch_vector(np.array([76.0, 76.0]), np.array([10.4, 10.4]))
        w = ball.spin_vector(np.array([0.0, 2545.0]), np.zeros(2), vel)
        out = ball.fly(vel, w)
        self.assertGreater(out["apex"][1], out["apex"][0] + 5.0)
        self.assertGreater(out["carry"][1], out["carry"][0])


class Tours(unittest.TestCase):
    # tolerances are what the 4-coefficient model achieves (docs/experiments.md), so a change that loses accuracy fails
    def check(self, table, carry_tol, height_yd, land_deg):
        clubs, r, out = table_flight(table)
        for i, c in enumerate(clubs):
            with self.subTest(club=c):
                self.assertLess(abs(out["carry"][i] / YD / r[i]["carry_yd"] - 1.0), carry_tol)
                self.assertLess(abs(out["apex"][i] / YD - r[i]["height_yd"]), height_yd)
                self.assertLess(abs(out["land_deg"][i] - r[i]["land_deg"]), land_deg)

    def test_pga_fit(self):
        self.check(PGA, carry_tol=0.11, height_yd=9.0, land_deg=4.0)

    def test_lpga_held_out(self):
        self.check(LPGA, carry_tol=0.05, height_yd=8.0, land_deg=4.0)


class Spin(unittest.TestCase):
    def test_tour_spin_rates(self):
        for table in (PGA, LPGA):
            for c in spin.CLUBS:
                r = row(table, c)
                sl = spin.spin_loft_from_launch(r["launch_deg"] - r["attack_deg"])
                v, n = delivery(r["club_mph"], r["attack_deg"], sl)
                out = spin.launch(v, n, np.array([r["ball_mph"] * MPH]), c)
                with self.subTest(table=table is PGA and "PGA" or "LPGA", club=c):
                    self.assertLess(abs(out["spin_rpm"][0] / r["spin_rpm"] - 1.0), 0.20)
                    self.assertAlmostEqual(out["launch_deg"][0], r["launch_deg"], delta=0.05)
                    self.assertAlmostEqual(out["axis_deg"][0], 0.0, delta=1e-6)

    def test_open_face_slices_right(self):
        results = {}
        for ftp in (-6.0, 0.0, 6.0):
            v, n = delivery(68.0, -1.0, 13.0, face_to_path=ftp)
            out = spin.launch(v, n, np.array([68.0 * 1.45 * MPH]), "Driver")
            results[ftp] = (out["axis_deg"][0], ball.fly(out["vel"], out["spin"])["curve"][0])
        self.assertGreater(results[6.0][0], 5.0)
        self.assertGreater(results[6.0][1], 5.0)
        self.assertLess(results[-6.0][1], -5.0)
        self.assertAlmostEqual(results[6.0][1], -results[-6.0][1], delta=0.5)
        self.assertLess(abs(results[0.0][1]), 0.5)

    def test_more_spin_loft_more_spin(self):
        rpm = []
        for sl in (5.0, 13.0, 25.0):
            v, n = delivery(68.0, -1.0, sl)
            rpm.append(spin.launch(v, n, np.array([90.0 * MPH]), "Driver")["spin_rpm"][0])
        self.assertLess(rpm[0], rpm[1])
        self.assertLess(rpm[1], rpm[2])


if __name__ == "__main__":
    unittest.main()
