"""Check impact and ballspeed"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys

import torch

from isaaclab.app import add_launcher_args, launch_simulation

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--task", default="Isaac-Golf-Swing-Direct")
ap.add_argument("--num_envs", type=int, default=480)
ap.add_argument("--v_min", type=float, default=2.0, help="slowest clubhead speed [m/s]")
ap.add_argument("--v_max", type=float, default=50.0, help="fastest clubhead speed [m/s]")
ap.add_argument("--start", type=float, default=0.4, help="head starts this far behind the ball [m]")
ap.add_argument("--dt", type=float, default=None, help="physics step override [s]")
ap.add_argument("--decimation", type=int, default=None, help="physics steps per env step override")
ap.add_argument("--ball_contact_offset", type=float, default=None, help="ball contact offset override [m]")
ap.add_argument("--miss_by", type=float, default=0.0,
                help="shift the ball off the head's path along its toe-heel axis [m]: anything 'hit' is a ghost")
add_launcher_args(ap)
args, extra = ap.parse_known_args()
sys.argv = [sys.argv[0]] + [a for a in extra if "=" in a and not a.startswith("-")]

import gymnasium as gym

import envs.golf
from isaaclab_tasks.utils import resolve_task_config

env_cfg, _ = resolve_task_config(args.task, "")
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.gravity = (0.0, 0.0, 0.0)
env_cfg.initial_joint_pos_range = (0.0, 0.0)
env_cfg.initial_joint_vel_range = (0.0, 0.0)
env_cfg.post_impact_steps = 10_000
if args.dt:
    env_cfg.sim.dt = args.dt
if args.ball_contact_offset is not None:
    env_cfg.scene.ball.spawn.collision_props.contact_offset = args.ball_contact_offset
if args.decimation:
    env_cfg.decimation = args.decimation
    env_cfg.sim.render_interval = args.decimation
args.device = env_cfg.sim.device

with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
    env = gym.make(args.task, cfg=env_cfg)
    cleanup.callback(env.close)
    u = env.unwrapped
    n, dev = u.num_envs, u.device
    speeds = torch.linspace(args.v_min, args.v_max, n, device=dev)
    d = torch.tensor(u.cfg.target_dir, dtype=torch.float32, device=dev)
    d = d / torch.linalg.norm(d)
    ids = torch.arange(n, device=dev)

    with torch.inference_mode():
        env.reset()
        u.robot.actuators.target_command.set_position_index(value=u.robot.data.default_joint_pos.torch)
        root = u.robot.data.root_link_pose_w.torch.clone()
        root[:, :3] += u.ball_pos - u.club_pos - args.start * d
        u.robot.write_root_pose_to_sim_index(root_pose=root, env_ids=ids)
        vel = torch.zeros(n, 6, device=dev)
        vel[:, :3] = speeds.unsqueeze(-1) * d
        u.robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
        if args.miss_by:
            from isaaclab.utils.math import quat_apply

            club_q = u.robot.data.body_link_quat_w.torch[:, u._club_idx]
            toe = quat_apply(club_q, torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3))
            toe = torch.where(toe[:, 2:3] < 0, -toe, toe)
            ball = u.ball.data.root_link_pose_w.torch.clone()
            ball[:, :3] += args.miss_by * toe
            u.ball.write_root_pose_to_sim_index(root_pose=ball, env_ids=ids)
        u._compute_intermediate_values()
        u.prev_club_pos.copy_(u.club_pos)
        u.prev_club_quat.copy_(u.club_quat)
        u.prev_ball_pos.copy_(u.ball_pos)
        u.armed[:] = True  # head starts 0.4 m back

        zero = torch.zeros(n, u.cfg.action_space, device=dev)
        max_ball = torch.zeros(n, device=dev)
        steps = math.ceil((args.start + 0.6) / (args.v_min * u.step_dt))
        for k in range(steps):
            env.step(zero)
            max_ball = torch.maximum(max_ball, torch.linalg.norm(u.ball.data.root_lin_vel_w.torch, dim=-1))
            if k == 0:
                rigid = u.club_speed / speeds

    struck = max_ball > speeds / 3
    seen = u.impacted
    physics_dt = env_cfg.sim.dt
    print(f"\nphysics dt {physics_dt:.5f} s ({1 / physics_dt:.0f} Hz), decimation {env_cfg.decimation}, "
          f"ball contact offset {env_cfg.scene.ball.spawn.collision_props.contact_offset}, miss by {args.miss_by} m, "
          f"{n} envs, centre strike, face first")
    print(f"rigidity check: head speed / commanded after 1 step, mean {rigid.mean():.3f}, "
          f"range {rigid.min():.3f}-{rigid.max():.3f} (should be ~1)")
    print(f"{'club m/s':>10} {'travel/substep':>15} {'physics hit':>12} {'detector':>9} {'ball/club':>10}")
    edges = torch.arange(0.0, args.v_max + 5.0, 5.0, device=dev)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (speeds >= lo) & (speeds < hi)
        if not m.any():
            continue
        mid = speeds[m].mean().item()
        print(f"{lo:>4.0f}-{hi:<5.0f} {100 * mid * physics_dt:>12.1f} cm {struck[m].float().mean():>11.0%} "
              f"{seen[m].float().mean():>9.0%} {(max_ball[m] / speeds[m]).mean():>10.2f}")
