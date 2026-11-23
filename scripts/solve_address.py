"""Solve the address pose using the sim itself as the forward-kinematics engine.

torso is the ROOT, so root pitch tips the whole body, legs included. A real spine tilt needs
root pitch forward AND lower_waist bending back so the legs stay vertical under the hips.

The body's rigid translation is free (we choose root pos at reset), so we do NOT search it:
we drop the pose until the feet touch the ground and recentre it on the feet. The ball is an
OUTPUT too -- a golfer addresses the ball wherever the club naturally sits.

Writes envs/golf/address/<club>.json: root pose, joints, ball position, weld.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import torch

from isaaclab.app import add_launcher_args, launch_simulation

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--task", default="Isaac-Golf-Swing-Direct")
ap.add_argument("--num_envs", type=int, default=4096)
ap.add_argument("--iters", type=int, default=40)
ap.add_argument("--elite", type=float, default=0.03)
ap.add_argument("--club", default="driver", help="key into envs/golf/clubs.py")
ap.add_argument("--ball_gap", type=float, default=0.02, help="air between clubface and ball at address [m]")
ap.add_argument("--knee_deg", type=float, default=25.0, help="knee flex at address")
ap.add_argument("--lead_elbow_deg", type=float, default=-1.0, help="lead elbow bend at address; below 0 free")
ap.add_argument("--arms", default="stock", choices=("stock", "full"), help="body: full = twisting shoulders, wrists")
ap.add_argument("--trail_on_shaft", type=float, default=0.085, help="full arms: trail hand this far down the grip [m]")
ap.add_argument("--out", default=None, help="address json path (default envs/golf/address/<club>[_full].json)")
add_launcher_args(ap)
args, extra = ap.parse_known_args()
sys.argv = [sys.argv[0]] + [a for a in extra if "=" in a and not a.startswith("-")]

import gymnasium as gym

torch.manual_seed(0)

import envs.golf
from envs.golf.golf_env_cfg import BALL_R
from envs.golf.clubs import CLUBS, address_path, golfer_usd
from envs.golf.shapes import capsule_table, collision_pairs, penetrations
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import resolve_task_config

# root orientation
SEARCH = [
    ("rot_x", -1.6, 1.6), ("rot_y", -1.6, 1.6), ("rot_z", -1.6, 1.6),
    ("lower_waist:0", -0.5, 0.5),
    ("lower_waist:1", -1.309, 0.524),
    ("pelvis", -0.6, 0.6),
    ("right_thigh:0", -0.785, 0.262),
    ("left_thigh:0", -0.785, 0.262),
    ("right_thigh:1", -1.0, 0.3),
    ("left_thigh:1", -1.0, 0.3),
    ("right_foot:0", -0.873, 0.873),
    ("right_foot:1", -0.873, 0.873),
    ("left_foot:0", -0.873, 0.873),
    ("left_foot:1", -0.873, 0.873),
    ("right_shin", -0.9, 0.0),
    ("left_shin", -0.9, 0.0),
    ("left_upper_arm:0", -1.571, 1.222),
    ("left_upper_arm:2", -1.571, 1.222),
    ("left_lower_arm", -1.571, 0.873),
    ("right_upper_arm:0", -1.571, 1.222),
    ("right_upper_arm:2", -1.571, 1.222),
    ("right_lower_arm", -1.571, 0.873),
]
if args.arms == "full":
    # elbows straighten to the new limit; the shoulders' third axis; the trail wrist (the lead stays neutral:
    # its axes are built from the address hand)
    SEARCH = [(nm, -1.90, hi) if nm.endswith("_lower_arm") else (nm, lo, hi) for nm, lo, hi in SEARCH]
    SEARCH += [("left_upper_arm:1", -1.3, 1.3), ("right_upper_arm:1", -1.3, 1.3),
               ("right_wrist:1", -0.8, 0.8), ("right_wrist:2", -0.8, 0.8)]
 
EVAL_Z = 5.0
club = CLUBS[args.club]
GOLFER_USD = golfer_usd(args.club, arms=args.arms)

env_cfg, _ = resolve_task_config(args.task, "")
env_cfg.club = args.club
env_cfg.arms = args.arms
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.gravity = (0.0, 0.0, 0.0)

for props in env_cfg.scene.robot.spawn.articulation_props:
    if hasattr(props, "enabled_self_collisions"):
        props.enabled_self_collisions = False
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
    club_i, lf_i, rf_i, head_i, torso_i, pelvis_i, lsh_i, rsh_i, lhip_i, rhip_i, lkn_i, rkn_i = [
        u.robot.find_bodies([b])[0][0]
        for b in ("club", "left_foot", "right_foot", "head", "torso", "pelvis",
                  "left_upper_arm", "right_upper_arm", "left_thigh", "right_thigh", "left_shin", "right_shin")
    ]

    # limbs must not start inside each other
    cap_local, cap_owner, caps = capsule_table(GOLFER_USD, u.robot.body_names)
    cap_pairs = collision_pairs(GOLFER_USD, caps)
    cap_local = torch.tensor(cap_local, dtype=torch.float32, device=dev)
    cap_owner = torch.tensor(cap_owner, dtype=torch.long, device=dev)
    # foot capsule endpoints, for flat feet on the ground
    foot_pts = {f: [i for c in caps if c[3] == f for i in c[:2]] for f in ("left_foot", "right_foot")}
    foot_r = next(c[2] for c in caps if c[3] == "left_foot")
    mass = u.robot.data.body_mass.torch[0]
    knee_target = math.radians(args.knee_deg)
    lua_i, lla_i = (u.robot.find_bodies([b])[0][0] for b in ("left_upper_arm", "left_lower_arm"))

    lo = torch.tensor([s[1] for s in SEARCH], device=dev)
    hi = torch.tensor([s[2] for s in SEARCH], device=dev)
    mean, std = (lo + hi) / 2, (hi - lo) / 4
    origins = u.scene.env_origins
    # hands high enough that the shaft angle puts the head at its address height
    th = math.radians(club.shaft_deg)
    hand_target = torch.tensor([club.hand_x, 0.0, club.head_z + club.grip * math.cos(th)], device=dev)
    GRIP_Z = club.grip
    TARGET_DIR = torch.tensor([0.0, 1.0, 0.0], device=dev)   # right-hander hits to its left
    SHAFT_DIR = torch.tensor([math.sin(th), 0.0, -math.cos(th)], device=dev)  # lead hand -> clubhead

    def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
        """(x, y, z, w), the order isaac lab 3.0 writes and reads."""
        ang = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-9)
        axis = v / ang
        return torch.cat([axis * torch.sin(ang / 2), torch.cos(ang / 2)], dim=-1)

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
        # one step so physx recomputes link transforms
        u.sim.step(render=False)
        u.scene.update(u.physics_dt)
        pos = u.robot.data.body_link_pos_w.torch - origins.unsqueeze(1)
        return pos, u.robot.data.body_link_quat_w.torch.clone()

    def score(pose: tuple[torch.Tensor, torch.Tensor]):
        """Stand the pose on its feet, centred on the ankles, and judge it (the club comes later)."""
        pos, quat = pose
        world = pos[:, cap_owner] + quat_apply(quat[:, cap_owner], cap_local.expand(n, -1, -1))
        fl, fr = world[:, foot_pts["left_foot"]], world[:, foot_pts["right_foot"]]
        shift = torch.zeros(n, 3, device=dev)
        shift[:, 2] = foot_r - torch.cat([fl, fr], 1)[..., 2].min(dim=1).values
        shift[:, :2] = -(pos[:, lf_i, :2] + pos[:, rf_i, :2]) / 2
        p = pos + shift.unsqueeze(1)
        fl, fr = fl + shift.unsqueeze(1), fr + shift.unsqueeze(1)
        lead, trail = p[:, u._lead_idx], p[:, u._trail_idx]
        lf, rf, torso, pelvis = p[:, lf_i], p[:, rf_i], p[:, torso_i], p[:, pelvis_i]

        # hands where a golfer holds them; the ball is derived from this
        cost = 20.0 * torch.sum((lead - hand_target) ** 2, dim=-1)
        if not club.tee:  # ball on the ground: the sole rests on it, so hold the head height to the millimetre
            cost += 400.0 * (lead[:, 2] - hand_target[2]) ** 2
        # both feet flat on the ground, toes forward
        for f in (fl, fr):
            cost += 8.0 * (f[..., 2].min(dim=1).values - foot_r) ** 2
            cost += 50.0 * ((f[..., 2] - f[..., 2].mean(1, keepdim=True)) ** 2).sum(-1)
            toe = f[:, 1::2, :2] - f[:, 0::2, :2]
            cost += 2.0 * (1.0 - (toe[..., 0].abs() / torch.linalg.norm(toe, dim=-1).clamp_min(1e-6))).sum(-1)
        # athletic: feet wide and square, knees soft, weight over the middle of the feet
        cost += 20.0 * ((lf[:, 1] - rf[:, 1]) - club.stance) ** 2 + 10.0 * (lf[:, 0] - rf[:, 0]) ** 2
        for hip, knee, foot in ((lhip_i, lkn_i, lf_i), (rhip_i, rkn_i, rf_i)):
            flex = torch.arccos(torch.nn.functional.cosine_similarity(
                p[:, knee] - p[:, hip], p[:, foot] - p[:, knee], dim=-1).clamp(-1.0, 1.0))
            cost += 4.0 * (flex - knee_target) ** 2
        if args.lead_elbow_deg >= 0:
            # a straight lead arm, so the swing's pull doesn't lengthen it into the ground at impact
            bend = torch.arccos(torch.nn.functional.cosine_similarity(
                p[:, lla_i] - p[:, lua_i], lead - p[:, lla_i], dim=-1).clamp(-1.0, 1.0))
            cost += 8.0 * (bend - math.radians(args.lead_elbow_deg)) ** 2
        com = (p * mass.view(1, -1, 1)).sum(1) / mass.sum()
        support = torch.cat([fl, fr], 1)[..., :2].mean(1)
        cost += 20.0 * torch.sum((com[:, :2] - support) ** 2, dim=-1)
        # stand up: do not fold in half. torso well above pelvis, pelvis well above feet.
        cost += 8.0 * torch.relu(0.35 - (torso[:, 2] - pelvis[:, 2])) ** 2
        cost += 8.0 * torch.relu(1.00 - torso[:, 2]) ** 2
        cost += 8.0 * torch.relu(0.70 - pelvis[:, 2]) ** 2
        # trail hand on the grip: full arms can reach its real spot, just below the lead hand
        if args.arms == "full":
            cost += 20.0 * torch.sum((trail - lead - args.trail_on_shaft * SHAFT_DIR) ** 2, dim=-1)
        else:
            cost += 2.0 * torch.sum((trail - lead) ** 2, dim=-1)
        # face the ball
        for l_i, r_i, w in ((lsh_i, rsh_i, 2.0), (lhip_i, rhip_i, 4.0)):
            lat = p[:, l_i] - p[:, r_i]
            lat = lat / torch.linalg.norm(lat, dim=-1, keepdim=True).clamp_min(1e-6)
            cost += w * (1.0 - lat @ TARGET_DIR)
        # no limb inside another
        cost += 200.0 * (torch.relu(penetrations(world, caps, cap_pairs)) ** 2).sum(-1)
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

    shaft_dir = torch.tensor([math.sin(th), 0.0, -math.cos(th)], device=dev)
    clubhead = lead + GRIP_Z * shaft_dir
    ball = clubhead + (club.head_size[0] / 2 + BALL_R + args.ball_gap) * TARGET_DIR

    from isaaclab.utils.math import matrix_from_quat

    R_hand = matrix_from_quat(hand_q.unsqueeze(0))[0]
    d = lead - clubhead
    a = (R_hand.T @ d) / torch.linalg.norm(d)                 # club +z, in hand frame
    f = R_hand.T @ TARGET_DIR                                  # wanted face normal
    f = f - (f @ a) * a
    f = f / torch.linalg.norm(f)                               # club +x, orthogonal to +z
    # weld maps the shaft direction to club z (rows, not columns)
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
    print(f"hand->clubhead distance {torch.linalg.norm(d).item():.4f} m   (club grip length {GRIP_Z})")
    print(f"clubhead->ball {torch.linalg.norm(ball - clubhead).item():.4f} m   (face gap {args.ball_gap} m)")
    # full shift incl. x/y: score() recentred the feet on the origin
    out = Path(args.out) if args.out else address_path(args.club, args.arms)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "root_pos": [round(shift[0, 0].item(), 3), round(shift[0, 1].item(), 3), round(EVAL_Z + shift[0, 2].item(), 3)],
        "root_rot": [round(v, 4) for v in root_quat.tolist()],
        "joint_pos": {n: round(v, 3) for n, v in zip(names, mean[ROT_DIMS:].tolist())},
        "ball_pos": [round(v, 3) for v in ball.tolist()],
        "weld_rot": [round(v, 5) for v in q.tolist()],
    }, indent=2) + "\n")
    print(f"\nwrote {out}; now rebuild the asset: uv run python scripts/make_golfer_asset.py --club {args.club} --arms {args.arms}")
    print("resulting pose (env-local, after dropping to the ground):")
    for l_i, r_i, label in ((lsh_i, rsh_i, "shoulder"), (lhip_i, rhip_i, "hip")):
        lat = p[0, l_i] - p[0, r_i]
        print(f"  {label} line (left - right) ({lat[0]:+.3f}, {lat[1]:+.3f}, {lat[2]:+.3f})   want +y")
    for label, idx in (("lead hand", u._lead_idx), ("trail hand", u._trail_idx),
                       ("torso", torso_i), ("pelvis", pelvis_i), ("head", head_i),
                       ("left_foot", lf_i), ("right_foot", rf_i)):
        q2 = p[0, idx]
        print(f"  {label:11} ({q2[0]:+.3f}, {q2[1]:+.3f}, {q2[2]:+.3f})")
    print(f"  hands apart {torch.linalg.norm(p[0, u._trail_idx] - p[0, u._lead_idx]).item():.3f} m")
    sh, el = p[0, lua_i], p[0, lla_i]
    bend = torch.arccos(torch.nn.functional.cosine_similarity(el - sh, lead - el, dim=0).clamp(-1.0, 1.0))
    print(f"  lead elbow bend {math.degrees(bend.item()):.1f} deg")
    print(f"  lowest body z {p[0, :, 2].min().item():+.3f}   torso-pelvis {(p[0, torso_i, 2] - p[0, pelvis_i, 2]).item():+.3f} m")
