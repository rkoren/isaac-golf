"""Bounce and roll after landing: uv run python -m unittest flight.test_ground -v"""

import math
import unittest

import numpy as np

from flight import ball, ground
from flight.ground import FAIRWAY

YD = 0.9144


def landing(speed, angle_deg, rpm, axis_deg=0.0, heading_deg=0.0):
    """Landing velocity (descending at angle_deg from horizontal) and spin of rpm (+ axis_deg curves right)."""
    a, h = math.radians(angle_deg), math.radians(heading_deg)
    v = np.array([[math.sin(h) * math.cos(a), math.cos(h) * math.cos(a), -math.sin(a)]]) * speed
    return v, ball.spin_vector(np.array([float(rpm)]), np.array([float(axis_deg)]), v)


def ke(v, w):
    return 0.5 * (v * v).sum(-1) + 0.5 * 0.4 * ball.RADIUS**2 * (w * w).sum(-1)


# speed, angle, rpm: a mix of gripping and slipping landings
CASES = ((22.0, 55.0, 4000), (30.0, 25.0, 2500), (28.0, 30.0, 500), (20.0, 70.0, 9000), (40.0, 8.0, 4000))


def in_tilted_frame(v):
    th = math.radians(ground.tilt_deg(v, FAIRWAY)[0])
    vy, vx = -v[0, 2], v[0, 1]
    return th, vx * math.cos(th) - vy * math.sin(th), vx * math.sin(th) + vy * math.cos(th)


class Impact(unittest.TestCase):
    def penner_2d(self, v, w):
        """Penner's equations 2, 3, 5 to 7 for travel along +y and spin w (backspin +)."""
        th, vxp, vyp = in_tilted_frame(v)
        om, r = w[0, 0], ball.RADIUS
        e = FAIRWAY.e_scale * (0.510 - 0.0375 * vyp + 0.000903 * vyp**2 if vyp <= 20 else 0.120)
        muc = 2 * (vxp + r * om) / (7 * (1 + e) * vyp)
        if FAIRWAY.mu < muc:
            rx = vxp - FAIRWAY.mu * vyp * (1 + e)
            rom = om - 5 * FAIRWAY.mu / (2 * r) * vyp * (1 + e)
        else:
            rx = 5 / 7 * vxp - 2 / 7 * r * om
            rom = -rx / r
        ry = e * vyp
        return rx * math.cos(th) - ry * math.sin(th), rx * math.sin(th) + ry * math.cos(th), rom

    def test_matches_penner_in_plane(self):
        for case in CASES:
            v, w = landing(*case)
            vo, wo = ground.impact(v, w, FAIRWAY)
            ey, ez, om = self.penner_2d(v, w)
            with self.subTest(case=case):
                self.assertAlmostEqual(vo[0, 1], ey, places=6)
                self.assertAlmostEqual(vo[0, 2], ez, places=6)
                self.assertAlmostEqual(wo[0, 0], om, places=4)
                self.assertAlmostEqual(vo[0, 0], 0.0, places=9)

    def test_grip_and_slip_both_occur(self):
        kinds = set()
        for case in CASES:
            v, w = landing(*case)
            _, vxp, vyp = in_tilted_frame(v)
            e = FAIRWAY.e_scale * (0.510 - 0.0375 * vyp + 0.000903 * vyp**2)
            kinds.add(bool(FAIRWAY.mu < 2 * (vxp + ball.RADIUS * w[0, 0]) / (7 * (1 + e) * vyp)))
        self.assertEqual(kinds, {True, False})

    def test_energy_never_rises(self):
        rng = np.random.default_rng(0)
        n = 4000
        v = np.stack([rng.uniform(-10, 10, n), rng.uniform(2, 45, n), -rng.uniform(0.5, 30, n)], -1)
        w = rng.uniform(-1100, 1100, (n, 3))
        vo, wo = ground.impact(v, w, FAIRWAY)
        self.assertTrue((ke(vo, wo) <= ke(v, w) * (1 + 1e-9)).all())

    def test_sidespin_kicks_sideways_and_mirrors(self):
        v, w = landing(26.0, 35.0, 3000, axis_deg=12.0)
        vo, _ = ground.impact(v, w, FAIRWAY)
        v2, w2 = landing(26.0, 35.0, 3000, axis_deg=-12.0)
        vo2, _ = ground.impact(v2, w2, FAIRWAY)
        self.assertGreater(abs(vo[0, 0]), 0.05)
        self.assertAlmostEqual(vo[0, 0], -vo2[0, 0], places=9)
        self.assertAlmostEqual(vo[0, 1], vo2[0, 1], places=9)


class Run(unittest.TestCase):
    def one(self, speed, ang, rpm, **kw):
        v, w = landing(speed, ang, rpm, **kw)
        return ground.run(v, w, FAIRWAY)

    def test_zero_spin_runs_forward_and_stops(self):
        out = self.one(26.6, 38.0, 0)
        self.assertGreater(out["run"][0], 3.0)
        self.assertLessEqual(out["bounces"][0], ground.MAX_BOUNCES)
        self.assertTrue(np.isfinite(out["time"][0]))

    def test_run_falls_with_landing_angle(self):
        runs = [self.one(26.6, a, 2300)["run"][0] for a in range(30, 50, 2)]
        self.assertTrue(all(a > b for a, b in zip(runs, runs[1:])))

    def test_usga_line(self):
        # USGA/R&A 2021 fairway model (yd, landing angle in degrees): run = 79.1 - 1.6 * angle
        for a in range(32, 46):
            got = self.one(26.6, a, 2300)["run"][0] / YD
            with self.subTest(angle=a):
                self.assertLess(abs(got - (79.1 - 1.6 * a)), 2.0)

    def test_steep_wedge_stops_quickly(self):
        for rpm in (7000, 9000, 11000):
            out = self.one(22.0, 52.0, rpm)
            with self.subTest(rpm=rpm):
                self.assertLess(abs(out["run"][0] / YD), 3.0)

    def test_spin_back_is_bounded(self):
        rng = np.random.default_rng(1)
        n = 2000
        a, s = rng.uniform(math.radians(20), math.radians(75), n), rng.uniform(8, 30, n)
        v = np.stack([np.zeros(n), s * np.cos(a), -s * np.sin(a)], -1)
        w = ball.spin_vector(rng.uniform(0, 12000, n), np.zeros(n), v)
        out = ground.run(v, w, FAIRWAY)
        self.assertLess(-out["run"].min() / YD, 12.0)
        self.assertTrue((out["bounces"] <= ground.MAX_BOUNCES).all())
        self.assertTrue(np.isfinite(out["time"]).all())

    def test_skull_does_not_pop_up(self):
        # 40 m/s, 6 degrees, little spin: leaves slower than it arrived (Penner's linear tilt sent it up at 11 m/s)
        v, w = landing(40.0, 6.0, 600)
        vo, _ = ground.impact(v, w, FAIRWAY)
        self.assertLess(vo[0, 2], -v[0, 2])
        self.assertTrue(np.isfinite(ground.run(v, w, FAIRWAY)["run"][0]))

    def test_rising_ball_is_skipped(self):
        v = np.array([[0.0, 20.0, 1.0], [0.0, 20.0, -5.0]])
        out = ground.run(v, np.zeros((2, 3)), FAIRWAY)
        self.assertEqual(out["run"][0], 0.0)
        self.assertEqual(out["bounces"][0], 0)
        self.assertGreater(out["run"][1], 0.0)

    def test_mirror_symmetry(self):
        a, b = self.one(26.6, 38.0, 2300, axis_deg=10.0), self.one(26.6, 38.0, 2300, axis_deg=-10.0)
        self.assertAlmostEqual(a["rest"][0, 0], -b["rest"][0, 0], places=9)
        self.assertAlmostEqual(a["rest"][0, 1], b["rest"][0, 1], places=9)


class Trajectory(unittest.TestCase):
    def setUp(self):
        self.vel = ball.launch_vector(np.array([58.0]), np.array([25.0]))
        self.spin = ball.spin_vector(np.array([7000.0]), np.zeros(1), self.vel)

    def test_path_uniform_and_ends_at_rest(self):
        path, dt, times = ground.trajectory(self.vel, self.spin, 0.0, FAIRWAY)
        self.assertEqual(path.shape[1], 3)
        out = ball.fly(self.vel, self.spin)
        g = ground.run(out["land_vel"], out["land_spin"], FAIRWAY)
        total = out["land_pos"][0, :2] + g["rest"][0]
        self.assertAlmostEqual(path[-1, 2], 0.0, delta=1e-6)
        self.assertLess(np.linalg.norm(path[-1, :2] - total), 0.8)
        speeds = np.linalg.norm(np.diff(path, axis=0), axis=-1) / dt
        self.assertLess(speeds[-1], 1.0)
        self.assertAlmostEqual(times["land"], out["time"][0], delta=0.1)
        self.assertGreater(times["rest"], times["land"])

    def test_fly_reports_the_landing_state(self):
        out = ball.fly(self.vel, self.spin)
        self.assertEqual(out["land_vel"].shape, (1, 3))
        self.assertLess(out["land_vel"][0, 2], 0.0)
        w0 = np.linalg.norm(self.spin)
        self.assertAlmostEqual(np.linalg.norm(out["land_spin"]) / w0, math.exp(-out["time"][0] / ball.SPIN_TAU), places=9)
        ang = math.degrees(math.atan2(-out["land_vel"][0, 2], np.linalg.norm(out["land_vel"][0, :2])))
        self.assertAlmostEqual(ang, out["land_deg"][0], places=6)

    def test_carry_is_unchanged_by_the_extra_outputs(self):
        out = ball.fly(self.vel, self.spin)
        self.assertAlmostEqual(np.linalg.norm(out["land_pos"][0, :2]), out["carry"][0], places=9)


if __name__ == "__main__":
    unittest.main()
