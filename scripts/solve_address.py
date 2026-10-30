"""Solve the address pose using the sim itself as the forward-kinematics engine.

torso is the ROOT, so root pitch tips the whole body, legs included. A real spine tilt needs
root pitch forward AND lower_waist bending back so the legs stay vertical under the hips.

The body's rigid translation is free (we choose root pos at reset), so we do NOT search it:
we drop the pose until the feet touch the ground and recentre it on the feet. The ball is an
OUTPUT too -- a golfer addresses the ball wherever the club naturally sits.

Prints root pose, joint angles and ball position to paste into envs/golf/.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys

import torch

from isaaclab.app import add_launcher_args, launch_simulation

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--task", default="Isaac-Golf-Swing-Direct")
ap.add_argument("--num_envs", type=int, default=4096)
ap.add_argument("--iters", type=int, default=25)
ap.add_argument("--elite", type=float, default=0.03)
ap.add_argument("--ball_z", type=float, default=0.05, help="teed ball centre height")
ap.add_argument("--hand_x", type=float, default=0.25, help="lead hand in front of the feet")
ap.add_argument("--hand_z", type=float, default=0.72, help="lead hand height at address")
ap.add_argument("--shaft_deg", type=float, default=42.0, help="shaft angle from vertical")
add_launcher_args(ap)
args, extra = ap.parse_known_args()
sys.argv = [sys.argv[0]] + [a for a in extra if "=" in a and not a.startswith("-")]

import gymnasium as gym  # noqa: E402

torch.manual_seed(0)   # reproducible: the printed pose and weld must stay in sync

import envs.golf  # noqa: F401, E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

# root orientation is searched as a full rotation vector (axis * angle), not a pitch about
# world y: the torso link frame is rotated 90 deg from the prim frame, so "pitch" reasoning
# kept pinning the solver against its own bound. let CEM find the orientation.
SEARCH = [
    ("rot_x", -1.6, 1.6), ("rot_y", -1.6, 1.6), ("rot_z", -1.6, 1.6),
    ("lower_waist:0", -0.5, 0.5),
    ("lower_waist:1", -1.309, 0.524),
    ("pelvis", -0.6, 0.6),
    ("right_thigh:1", -1.0, 0.3),
    ("left_thigh:1", -1.0, 0.3),
    ("right_shin", -0.9, 0.0),
    ("left_shin", -0.9, 0.0),
    ("left_upper_arm:0", -1.571, 1.222),
    ("left_upper_arm:2", -1.571, 1.222),
    ("left_lower_arm", -1.571, 0.873),
    ("right_upper_arm:0", -1.571, 1.222),
    ("right_upper_arm:2", -1.571, 1.222),
    ("right_lower_arm", -1.571, 0.873),
]
# evaluate airborne with gravity off so the single step is pure forward kinematics, with no
# chance of physx depenetrating a limb out of the floor and handing back a pose we did not
# write. (measured: it changed the result by <1%, so it was not the bug it looked like --
# kept because a correct measurement is worth more than a cheap one.)
EVAL_Z = 5.0
FOOT_Z = 0.045      # foot capsule radius ~0.027, so this is about resting on the plane

env_cfg, _ = resolve_task_config(args.task, "")
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.gravity = (0.0, 0.0, 0.0)
args.device = env_cfg.sim.device

with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
    env = gym.make(args.task, cfg=env_cfg)
    cleanup.callback(env.close)
    env.reset()
    u = env.unwrapped
    dev, n = u.device, u.num_envs

    ROT_DIMS = 3
    names = [s[0] for s in SEARCH[ROT_DIMS:]]
    jidx, _ = u.robot.find_joints(names, preserve_order=True)
    jidx = torch.tensor(jidx, dtype=torch.long, device=dev)
    club_i, lf_i, rf_i, head_i, torso_i, pelvis_i = [
        u.robot.find_bodies([b])[0][0]
        for b in ("club", "left_foot", "right_foot", "head", "torso", "pelvis")
    ]

    lo = torch.tensor([s[1] for s in SEARCH], device=dev)
    hi = torch.tensor([s[2] for s in SEARCH], device=dev)
    mean, std = (lo + hi) / 2, (hi - lo) / 4
    origins = u.scene.env_origins
    hand_target = torch.tensor([args.hand_x, 0.0, args.hand_z], device=dev)
    GRIP_Z = 0.85           # club-frame z held by the hand, see make_golfer_asset.py
    TARGET_DIR = torch.tensor([0.0, 1.0, 0.0], device=dev)   # right-hander hits to its left

    def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
        ang = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-9)
        axis = v / ang
        return torch.cat([torch.cos(ang / 2), axis * torch.sin(ang / 2)], dim=-1)

    def pose_of(cand: torch.Tensor):
        """Write one candidate per env, step once, return body positions (env-local)."""
        root = torch.zeros(n, 7, device=dev)
        root[:, :3] = origins + torch.tensor([0.0, 0.0, EVAL_Z], device=dev)
        root[:, 3:] = quat_from_rotvec(cand[:, :ROT_DIMS])
        u.robot.write_root_pose_to_sim(root)
        u.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        jpos = u.robot.data.default_joint_pos.torch.clone()
        jpos[:, jidx] = cand[:, ROT_DIMS:]
        u.robot.write_joint_state_to_sim(position=jpos, velocity=torch.zeros_like(jpos))
        # one step so physx recomputes link transforms. 1/120 s of gravity is ~0.3 mm.
        u.sim.step(render=False)
        u.scene.update(u.physics_dt)
        return u.robot.data.body_link_pos_w.torch - origins.unsqueeze(1)

    def score(pos: torch.Tensor):
        """Drop onto the ground, recentre on the feet, then judge the POSE (not the club).

        The club direction is not searched here: it is set by the weld rotation, which we
        solve in closed form afterwards. All this has to do is produce a natural address
        whose lead hand sits exactly one club length from the ball.
        """
        shift = torch.zeros(n, 3, device=dev)
        shift[:, 2] = FOOT_Z - pos[:, :, 2].min(dim=1).values
        shift[:, :2] = -(pos[:, lf_i, :2] + pos[:, rf_i, :2]) / 2
        p = pos + shift.unsqueeze(1)
        lead, trail = p[:, u._lead_idx], p[:, u._trail_idx]
        lf, rf, torso, pelvis = p[:, lf_i], p[:, rf_i], p[:, torso_i], p[:, pelvis_i]

        # put the hands where a golfer holds them. the ball is DERIVED from this afterwards
        # (hand + one club length down the shaft), so nothing here fights the club geometry.
        cost = 20.0 * torch.sum((lead - hand_target) ** 2, dim=-1)
        # feet down, both of them
        cost += 8.0 * ((lf[:, 2] - FOOT_Z) ** 2 + (rf[:, 2] - FOOT_Z) ** 2)
        # stand up: do not fold in half. torso well above pelvis, pelvis well above feet.
        # (frame-independent -- the torso link frame is rotated 90 deg from the prim frame,
        # so never reason about its local axes.)
        cost += 8.0 * torch.relu(0.35 - (torso[:, 2] - pelvis[:, 2])) ** 2
        cost += 8.0 * torch.relu(1.00 - torso[:, 2]) ** 2
        cost += 8.0 * torch.relu(0.70 - pelvis[:, 2]) ** 2
        # trail hand on the grip
        cost += 2.0 * torch.sum((trail - lead) ** 2, dim=-1)
        return cost, p, shift

    for it in range(args.iters):
        cand = (mean + std * torch.randn(n, len(SEARCH), device=dev)).clamp_(lo, hi)
        cost, _, _ = score(pose_of(cand))
        k = max(8, int(args.elite * n))
        elite = cand[torch.topk(-cost, k).indices]
        mean, std = elite.mean(0), elite.std(0).clamp_min(0.005)
        if it % 4 == 0 or it == args.iters - 1:
            print(f"iter {it:2d}  best {cost.min().item():.5f}  elite mean {cost.topk(k, largest=False).values.mean():.5f}")

    cost, p, shift = score(pose_of(mean.repeat(n, 1)))
    lead = p[0, u._lead_idx]
    root_quat = quat_from_rotvec(mean[:ROT_DIMS].unsqueeze(0))[0]
    hand_q = u.robot.data.body_link_quat_w.torch[0, u._lead_idx]

    # ball is an OUTPUT: one club length down a shaft at the chosen angle
    th = math.radians(args.shaft_deg)
    shaft_dir = torch.tensor([math.sin(th), 0.0, -math.cos(th)], device=dev)
    ball = lead + GRIP_Z * shaft_dir

    # solve the weld: we need the shaft to run from the hand to the ball, so the club's
    # +z (grip direction) must point from ball back to hand, expressed in the HAND frame.
    # the remaining roll is free, and we spend it aiming the clubface at the target.
    from isaaclab.utils.math import matrix_from_quat

    R_hand = matrix_from_quat(hand_q.unsqueeze(0))[0]
    d = lead - ball
    a = (R_hand.T @ d) / torch.linalg.norm(d)                 # club +z, in hand frame
    f = R_hand.T @ TARGET_DIR                                  # wanted face normal
    f = f - (f @ a) * a
    f = f / torch.linalg.norm(f)                               # club +x, orthogonal to +z
    # usd fixed joint: T_body0 * (p0,q0) == T_body1 * (p1,q1), so with q0 = identity
    #   R_club = R_hand @ R_weld^T  and  club_pos = hand_pos - R_club @ (0,0,GRIP_Z).
    # the weld must therefore map the shaft direction TO z, not from it -> rows, not columns.
    R_weld = torch.stack([f, torch.linalg.cross(a, f), a], dim=0)
    w = torch.sqrt(torch.clamp(1 + R_weld.trace(), min=1e-9)) / 2
    q = torch.tensor([
        w,
        (R_weld[2, 1] - R_weld[1, 2]) / (4 * w),
        (R_weld[0, 2] - R_weld[2, 0]) / (4 * w),
        (R_weld[1, 0] - R_weld[0, 1]) / (4 * w),
    ])

    print()
    print(f"final cost {cost[0].item():.5f}")
    print(f"hand->ball distance {torch.linalg.norm(d).item():.4f} m   (club grip length {GRIP_Z})")
    print()
    print("# --- paste into envs/golf/golfer.py ---")
    # the full shift, x/y included: the score() recentred the feet on the origin, so the
    # root has to carry that offset too or the body lands somewhere else entirely.
    print(f"    pos=({shift[0, 0].item():.3f}, {shift[0, 1].item():.3f}, {EVAL_Z + shift[0, 2].item():.3f}),")
    print(f"    rot=({root_quat[0]:.4f}, {root_quat[1]:.4f}, {root_quat[2]:.4f}, {root_quat[3]:.4f}),")
    print("ADDRESS_JOINT_POS = {")
    for name, val in zip(names, mean[ROT_DIMS:].tolist()):
        print(f'    "{name}": {val:.3f},')
    print("}")
    print()
    print("# --- paste into scripts/make_golfer_asset.py ---")
    print(f"WELD_ROT = ({q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f})   # (w,x,y,z)")
    print()
    print("# --- paste into envs/golf/golf_env_cfg.py ---")
    print(f"    ball_pos = ({ball[0]:.3f}, {ball[1]:.3f}, {ball[2]:.3f})")
    print()
    print("resulting pose (env-local, after dropping to the ground):")
    for label, idx in (("lead hand", u._lead_idx), ("trail hand", u._trail_idx),
                       ("torso", torso_i), ("pelvis", pelvis_i), ("head", head_i),
                       ("left_foot", lf_i), ("right_foot", rf_i)):
        q2 = p[0, idx]
        print(f"  {label:11} ({q2[0]:+.3f}, {q2[1]:+.3f}, {q2[2]:+.3f})")
    print(f"  hands apart {torch.linalg.norm(p[0, u._trail_idx] - p[0, u._lead_idx]).item():.3f} m")
    print(f"  lowest body z {p[0, :, 2].min().item():+.3f}   torso-pelvis {(p[0, torso_i, 2] - p[0, pelvis_i, 2]).item():+.3f} m")
