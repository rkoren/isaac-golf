"""Check + print where club, hands and ball are at address"""

from __future__ import annotations

import argparse
import contextlib
import sys

import torch

from isaaclab.app import add_launcher_args, launch_simulation

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--task", default="Isaac-Golf-Swing-Direct")
ap.add_argument("--num_envs", type=int, default=4)
ap.add_argument("--steps", type=int, default=60, help="zero-action settle steps")
ap.add_argument("--jitter", action="store_true", help="keep the reset joint jitter (default: exact address)")
add_launcher_args(ap)
args, extra = ap.parse_known_args()
sys.argv = [sys.argv[0]] + [a for a in extra if "=" in a and not a.startswith("-")]

import gymnasium as gym

import envs.golf
from envs.golf.golfer import GOLFER_USD
from envs.golf.shapes import capsule_table, collision_pairs, penetrations
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import resolve_task_config

env_cfg, _ = resolve_task_config(args.task, "")
env_cfg.scene.num_envs = args.num_envs
if not args.jitter:
    env_cfg.initial_joint_pos_range = (0.0, 0.0)
    env_cfg.initial_joint_vel_range = (0.0, 0.0)
args.device = env_cfg.sim.device


def report(u, label: str) -> None:
    org = u.scene.env_origins
    p = lambda t: "(" + ", ".join(f"{v:+.3f}" for v in t) + ")"
    print(f"\n--- {label} (env 0, relative to env origin) ---")
    print(f"  torso      {p((u.torso_pos - org)[0])}")
    print(f"  lead hand  {p((u.lead_pos - org)[0])}")
    print(f"  trail hand {p((u.trail_pos - org)[0])}")
    print(f"  clubhead   {p((u.club_pos - org)[0])}")
    print(f"  ball       {p((u.ball_pos - org)[0])}")
    print(f"  clubhead->ball dist {u.club_ball_dist[0]:.3f} m   (mean {u.club_ball_dist.mean():.3f})")
    print(f"  hands apart         {torch.linalg.norm(u.trail_pos - u.lead_pos, dim=-1)[0]:.3f} m")
    print(f"  clubhead height     {(u.club_pos - org)[0, 2]:+.3f} m")


def overlaps(u) -> list[tuple[float, str, str]]:
    local, owner, caps = capsule_table(GOLFER_USD, u.robot.body_names)
    pairs = collision_pairs(GOLFER_USD, caps)
    owner_t = torch.tensor(owner, device=u.device)
    local_t = torch.tensor(local, dtype=torch.float32, device=u.device)
    world = u.robot.data.body_link_pos_w.torch[0, owner_t] + quat_apply(
        u.robot.data.body_link_quat_w.torch[0, owner_t], local_t
    )
    depth = penetrations(world, caps, pairs)
    return sorted(
        ((d.item(), caps[i][3], caps[j][3]) for d, (i, j) in zip(depth, pairs) if d > 0), reverse=True
    )


with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
    env = gym.make(args.task, cfg=env_cfg)
    cleanup.callback(env.close)
    print(f"[INFO] obs space {env.observation_space}  action space {env.action_space}")
    env.reset()
    u = env.unwrapped
    print(f"[INFO] bodies ({len(u.robot.body_names)}): {u.robot.body_names}")
    print(f"[INFO] joints ({len(u.robot.joint_names)})")
    report(u, "at reset")
    found = overlaps(u)
    print(f"\n--- overlapping limbs at reset that physx will push apart: {len(found)} ---")
    for depth, n1, n2 in found:
        print(f"  {n1:16} x {n2:16} {100 * depth:.1f} cm")

    zero = torch.zeros(u.num_envs, u.cfg.action_space, device=u.device)
    with torch.inference_mode():
        for _ in range(args.steps):
            env.step(zero)
    report(u, f"after {args.steps} zero-action steps")
