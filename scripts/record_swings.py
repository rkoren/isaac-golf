"""Record and view our swing attempts

  uv run --env-file omni.env python scripts/record_swings.py logs/rsl_rl/golf_swing/<run> physics=isaacsim_physx
  uv run --env-file omni.env python scripts/record_swings.py <run>/model_0.pt <run>/model_199.pt physics=isaacsim_physx

"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

VIEWER = Path(__file__).with_name("swing_viewer.html")
MOVED = 0.3  # [m/s] ball speed that means something touched it

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("checkpoints", nargs="+", help="model_*.pt files, or run dirs (every checkpoint in them)")
ap.add_argument("--task", default="Isaac-Golf-Swing-Direct")
ap.add_argument("--num_envs", type=int, default=64, help="episodes rolled out per checkpoint")
ap.add_argument("--keep", type=int, default=12, help="episodes saved per checkpoint, spread best to worst")
ap.add_argument("--stochastic", action="store_true", help="sample actions like training does, not the mean")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default=None, help="output .html (default: <run>/swings/swings_<iters>.html)")
add_launcher_args(ap)

hydra = [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
args = ap.parse_args([a for a in sys.argv[1:] if a not in hydra])
sys.argv = [sys.argv[0]] + hydra


def iteration(path: Path) -> int:
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else -1


ckpts: list[Path] = []
for c in map(Path, args.checkpoints):
    ckpts += sorted(c.glob("model_*.pt"), key=iteration) if c.is_dir() else [c]
if not ckpts:
    sys.exit(f"no checkpoints found in {args.checkpoints}")

import gymnasium as gym

import envs.golf
from envs.golf.golf_env_cfg import MPH
from envs.golf.golfer import CLUB_BODY
from envs.golf.shapes import body_shapes
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import resolve_task_config

env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
args.device = env_cfg.sim.device


def part(body: str) -> str:
    # left arm leads for a right-hander
    return "club" if body == CLUB_BODY else "lead" if body.startswith("left_") else (
        "trail" if body.startswith("right_") else "core"
    )


def surface_dist(ball: np.ndarray, pts: np.ndarray, prim: dict) -> float:
    """Ball centre to a shape's surface [m]. Only used to vote on what struck the ball."""
    if prim["k"] == "cap":
        a, b = pts[prim["a"]], pts[prim["b"]]
        s = np.clip(np.dot(ball - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0.0, 1.0)
        return float(np.linalg.norm(ball - (a + s * (b - a))) - prim["r"])
    if prim["k"] == "sph":
        return float(np.linalg.norm(ball - pts[prim["c"]]) - prim["r"])
    c = pts[prim["c"]]
    axes = [pts[prim[k]] - c for k in "xyz"]
    half = np.array([np.linalg.norm(e) for e in axes])
    local = np.array([np.dot(ball - c, e / h) for e, h in zip(axes, half)])
    return float(np.linalg.norm(local - np.clip(local, -half, half)))


def striker(prim: dict) -> str:
    if prim["body"] == CLUB_BODY:
        return "clubhead" if prim["k"] == "box" else "shaft"
    return prim["body"]


def pack(a: torch.Tensor) -> str:
    return base64.b64encode(a.contiguous().cpu().numpy().tobytes()).decode("ascii")


with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
    env = gym.make(args.task, cfg=env_cfg)
    cleanup.callback(env.close)
    u = env.unwrapped
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(env, agent_cfg)

    # every shape as body-local points; the viewer only gets world points
    shapes = body_shapes(u.cfg.scene.robot.spawn.usd_path)
    names = u.robot.body_names
    local, owner, prims = [], [], []

    def add(body: int, pt: tuple) -> int:
        local.append(pt)
        owner.append(body)
        return len(local) - 1

    for b, name in enumerate(names):
        for s in shapes.get(name, []):
            if s[0] == "cap":
                prims.append({"k": "cap", "a": add(b, s[1]), "b": add(b, s[2]), "r": s[3]})
            elif s[0] == "sph":
                prims.append({"k": "sph", "c": add(b, s[1]), "r": s[2]})
            else:
                c = add(b, s[1])
                prims.append({"k": "box", "c": c, "x": add(b, s[2]), "y": add(b, s[3]), "z": add(b, s[4])})
            prims[-1].update(part=part(name), body=name)
    missing = [n for n in names if n not in shapes]
    if missing:
        print(f"[WARN] no collision shapes for bodies {missing}; they won't be drawn")
    local_t = torch.tensor(local, dtype=torch.float32, device=u.device)
    owner_t = torch.tensor(owner, dtype=torch.long, device=u.device)
    K = len(local)  # ball goes in slot K
    club_idx = names.index(CLUB_BODY)

    N, T = u.num_envs, u.max_episode_length + 1
    out_ckpts = []
    for ckpt in ckpts:
        runner.load(str(ckpt))
        policy = runner.get_inference_policy(device=u.device)
        dist = policy.distribution
        std = torch.zeros(1) if dist is None else (
            dist.std_param if hasattr(dist, "std_param") else dist.log_std_param.exp()
        )

        pts = torch.zeros(N, T, K + 1, 3, device=u.device)
        # club speed, ball speed, club-ball dist, torso height, armed, impacted, step reward
        sc = torch.zeros(N, T, 7, device=u.device)
        length = torch.zeros(N, dtype=torch.long, device=u.device)
        alive = torch.ones(N, dtype=torch.bool, device=u.device)
        smash = torch.zeros(N, device=u.device)  # 0 if no strike
        backwards = torch.zeros(N, dtype=torch.bool, device=u.device)
        impact = torch.zeros(N, 3, device=u.device)  # club speed, lead hand speed, time since address
        offline = torch.zeros(N, device=u.device)  # launch angle off the target line [deg], + = right
        strike_off = torch.zeros(N, 2, device=u.device)  # ball on the face: heel +, high + [m]
        face_path = torch.zeros(N, 2, device=u.device)  # face and path angles off the target line [deg], + = right
        lead_moved = torch.zeros(N, device=u.device)  # lead foot's distance from its address spot [m]

        # reset inside inference mode too, or the next checkpoint's reset fails
        with torch.inference_mode():
            obs, _ = env.reset()
            org = u.scene.env_origins
            for t in range(T):
                # read the sim directly: env caches are stale at frame 0
                d = u.robot.data
                world = d.body_link_pos_w.torch[:, owner_t] + quat_apply(
                    d.body_link_quat_w.torch[:, owner_t], local_t.expand(N, K, 3)
                )
                ball = u.ball.data.root_pos_w.torch
                frame = torch.cat([world, ball[:, None]], dim=1) - org[:, None]
                club_pos = d.body_link_pos_w.torch[:, club_idx]
                row = torch.stack(
                    [
                        torch.linalg.norm(d.body_link_lin_vel_w.torch[:, club_idx], dim=-1),
                        torch.linalg.norm(u.ball.data.root_lin_vel_w.torch, dim=-1),
                        torch.linalg.norm(ball - club_pos, dim=-1),
                        d.root_pos_w.torch[:, 2],
                        u.armed.float(),
                        u.impacted.float(),
                    ],
                    dim=-1,
                )
                pts[alive, t] = frame[alive]
                if hasattr(u, "smash"):
                    smash = torch.where(alive, torch.maximum(smash, u.smash), smash)
                if hasattr(u, "backwards"):
                    backwards |= alive & u.backwards
                now = torch.stack([u.impact_speed, u.impact_hand_speed, u.impact_time], dim=-1)
                impact = torch.where(alive[:, None], now, impact)
                if hasattr(u, "offline"):
                    offline = torch.where(alive, u.offline, offline)
                if hasattr(u, "strike_off"):
                    strike_off = torch.where(alive[:, None], u.strike_off, strike_off)
                if hasattr(u, "face_deg"):
                    face_path = torch.where(alive[:, None], torch.stack([u.face_deg, u.path_deg], -1), face_path)
                    moved_now = torch.linalg.norm(u.feet_xy[:, 0] - u.feet_home[:, 0], dim=-1)
                    lead_moved = torch.where(alive, moved_now, lead_moved)
                sc[alive, t, :6] = row[alive]
                length += alive.long()
                if t == T - 1:
                    break
                obs, rew, dones, _ = env.step(policy(obs, stochastic_output=args.stochastic))
                sc[alive, t, 6] = rew[alive]
                # envs auto-reset inside step: stop recording an env at its first episode end
                alive &= ~dones.bool()
                if not alive.any():
                    break

        # per-episode stats over ALL rollouts; only --keep of them get saved
        idx = torch.arange(T, device=u.device)[None].expand(N, T)
        valid = idx < length[:, None]
        max_ball = (sc[..., 1] * valid).amax(1)
        max_club = (sc[..., 0] * valid).amax(1)
        min_dist = torch.where(valid, sc[..., 2], 1e3).amin(1)
        min_torso = torch.where(valid, sc[..., 3], 1e3).amin(1)
        hit = (sc[..., 5] > 0.5).any(1)
        hit_frame = torch.where(hit, (sc[..., 5] > 0.5).float().argmax(1), -1)
        ret = sc[..., 6].sum(1)
        fell = min_torso < 0.6
        # first frame below 0.6 m after impact, for how long the finish holds
        down = valid & (sc[..., 3] < 0.6) & (idx >= hit_frame[:, None]) & hit[:, None]
        fall_after = torch.where(down.any(1), (down.float().argmax(1) - hit_frame).float() * u.step_dt, float("nan"))
        finish_torso = sc[torch.arange(N, device=u.device), (length - 1).clamp_min(0), 3]  # kneeling shows here, not in fell
        # top of the backswing: farthest the head gets from the ball before impact
        before = valid & ((idx < hit_frame[:, None]) | ~hit[:, None])
        top = torch.where(before, sc[..., 2], 0.0).max(1)
        backswing, top_time = top.values, top.indices * u.step_dt
        # what the reward pays: armed gain over the running max
        ball_v = sc[..., 1] * valid
        prev_max = torch.cat([torch.zeros_like(ball_v[:, :1]), ball_v.cummax(1).values[:, :-1]], 1)
        paid = (torch.relu(ball_v - prev_max) * (sc[..., 4] > 0.5) * valid).sum(1)

        # first toucher: nearest shape when the ball started moving
        moved = (sc[..., 1] > MOVED) & valid
        first = moved.float().argmax(1)
        pts_np, sc_np = pts.cpu().numpy(), sc.cpu().numpy()
        struck: list[dict | None] = []
        for i in range(N):
            if not moved[i].any():
                struck.append(None)
                continue
            t1 = int(first[i]); t0 = max(t1 - 1, 0)
            ball = pts_np[i, t0, K]
            near = min(prims, key=lambda pr: min(surface_dist(ball, pts_np[i, f], pr) for f in (t0, t1)))
            struck.append({"by": striker(near), "frame": t1, "armed": bool(sc_np[i, t1, 4] > 0.5)})
        # touched before arming
        early = [s is not None and not s["armed"] for s in struck]
        swung = [s["by"] for s in struck if s is not None and s["armed"]]

        # best first by what the reward pays; closest approach breaks ties among misses
        order = sorted(range(N), key=lambda i: (-paid[i].item(), min_dist[i].item()))
        n = min(args.keep, N)
        picks = sorted({order[round(j * (N - 1) / max(n - 1, 1))] for j in range(n)}, key=order.index)

        # shot shape for a right-hander from face-to-path
        ftp = face_path[hit, 0] - face_path[hit, 1]
        bands = {"hook": ftp < -6, "draw": (ftp >= -6) & (ftp < -2), "straight": ftp.abs() <= 2,
                 "fade": (ftp > 2) & (ftp <= 6), "slice": ftp > 6}
        downswing = impact[hit, 2] - top_time[hit]
        summary = {
            "contact_rate": hit.float().mean().item(),
            "moved_rate": moved.any(1).float().mean().item(),
            "pre_arm_touch_rate": sum(early) / N,
            "clubhead_strike_rate": swung.count("clubhead") / N,
            "body_strike_rate": sum(b not in ("clubhead", "shaft") for b in swung) / N,
            "struck_by": {b: swung.count(b) for b in sorted(set(swung), key=swung.count, reverse=True)},
            "mean_paid_ball_speed": paid.mean().item(),
            "mean_smash": smash[smash > 0].mean().item() if (smash > 0).any() else 0.0,
            "mean_impact_club_speed": impact[hit, 0].mean().item() if hit.any() else 0.0,
            "mean_impact_hand_speed": impact[hit, 1].mean().item() if hit.any() else 0.0,
            "mean_impact_time": impact[hit, 2].mean().item() if hit.any() else 0.0,
            "backwards_rate": backwards.float().mean().item(),
            "mean_backswing": backswing[hit].mean().item() if hit.any() else 0.0,
            "mean_top_time": top_time[hit].mean().item() if hit.any() else 0.0,
            "mean_offline_deg": offline[hit].abs().mean().item() if hit.any() else 0.0,
            "offline_bias_deg": offline[hit].mean().item() if hit.any() else 0.0,
            "offline_p90_deg": offline[hit].abs().quantile(0.9).item() if hit.any() else 0.0,
            "smash_p10": smash[smash > 0].quantile(0.1).item() if (smash > 0).any() else 0.0,
            "strike_heel_m": strike_off[hit, 0].mean().item() if hit.any() else 0.0,
            "strike_high_m": strike_off[hit, 1].mean().item() if hit.any() else 0.0,
            "strike_heel_abs_m": strike_off[hit, 0].abs().mean().item() if hit.any() else 0.0,
            "strike_high_abs_m": strike_off[hit, 1].abs().mean().item() if hit.any() else 0.0,
            "mean_face_deg": face_path[hit, 0].mean().item() if hit.any() else 0.0,
            "mean_path_deg": face_path[hit, 1].mean().item() if hit.any() else 0.0,
            "mean_face_to_path_deg": ftp.mean().item() if hit.any() else 0.0,
            "face_to_path_abs_deg": ftp.abs().mean().item() if hit.any() else 0.0,
            "shot_shapes": {k: v.float().mean().item() for k, v in bands.items()} if hit.any() else {},
            "mean_downswing_time": downswing.mean().item() if hit.any() else 0.0,
            "tempo_ratio": (top_time[hit].mean() / downswing.mean().clamp_min(1e-6)).item() if hit.any() else 0.0,
            "mean_lead_foot_moved": lead_moved.mean().item(),
            "fell_rate": fell.float().mean().item(),
            "median_fall_after_impact": fall_after.nanmedian().item() if down.any() else 0.0,
            "mean_finish_torso": finish_torso.mean().item(),
            "tall_finish_rate": (finish_torso >= 0.9).float().mean().item(),
            "mean_max_ball_speed": max_ball.mean().item(),
            "best_ball_speed": max_ball.max().item(),
            "mean_max_club_speed": max_club.mean().item(),
            "mean_min_dist": min_dist.mean().item(),
            "mean_return": ret.mean().item(),
            "action_std": std.mean().item(),
        }
        print(
            f"[{ckpt.stem}] contact {summary['contact_rate']:.0%}  fell {summary['fell_rate']:.0%}  "
            f"torso at finish {summary['mean_finish_torso']:.2f} m  "
            f"ball {MPH * summary['mean_max_ball_speed']:.1f} mph (best {MPH * summary['best_ball_speed']:.1f})  "
            f"club {MPH * summary['mean_max_club_speed']:.1f} mph  closest {summary['mean_min_dist']:.3f} m  "
            f"return {summary['mean_return']:.1f}  action std {summary['action_std']:.2f}"
        )
        print(
            f"    paid ball speed {MPH * summary['mean_paid_ball_speed']:.1f} mph  smash {summary['mean_smash']:.2f}  "
            f"touched before backswing "
            f"{summary['pre_arm_touch_rate']:.0%}  struck after backswing by: {summary['struck_by'] or 'nothing'}"
        )
        print(
            f"    backswing {summary['mean_backswing']:.2f} m at {summary['mean_top_time']:.2f} s  "
            f"at impact: club {MPH * summary['mean_impact_club_speed']:.1f} mph  "
            f"lead hand {MPH * summary['mean_impact_hand_speed']:.1f} mph  "
            f"{summary['mean_impact_time']:.2f} s after address  "
            f"offline {summary['mean_offline_deg']:.0f} deg (mean {summary['offline_bias_deg']:+.0f}, + = right)"
        )
        print(
            f"    spread: offline p90 {summary['offline_p90_deg']:.1f} deg  smash p10 {summary['smash_p10']:.2f}  "
            f"finish torso >= 0.9 m {summary['tall_finish_rate']:.0%}  "
            f"median fall {summary['median_fall_after_impact']:.2f} s after impact"
        )
        print(
            f"    strike on the face: heel {100 * summary['strike_heel_m']:+.1f} cm "
            f"(|{100 * summary['strike_heel_abs_m']:.1f}|), high {100 * summary['strike_high_m']:+.1f} cm "
            f"(|{100 * summary['strike_high_abs_m']:.1f}|)"
        )
        shapes = "  ".join(f"{k} {v:.0%}" for k, v in summary["shot_shapes"].items())
        print(
            f"    face {summary['mean_face_deg']:+.1f} deg  path {summary['mean_path_deg']:+.1f} deg  "
            f"face-to-path {summary['mean_face_to_path_deg']:+.1f} (|{summary['face_to_path_abs_deg']:.1f}|)  {shapes}"
        )
        print(
            f"    tempo: top {summary['mean_top_time']:.2f} s, downswing {summary['mean_downswing_time']:.2f} s "
            f"({summary['tempo_ratio']:.1f}:1)  lead foot moved {summary['mean_lead_foot_moved']:.2f} m"
        )

        episodes = []
        for i in picks:
            L = length[i].item()
            mm = (pts[i, :L] * 1000).round().clamp(-32767, 32767).to(torch.int16)
            episodes.append(
                {
                    "rank": order.index(i) + 1, "env": i, "frames": L, "hit": bool(hit[i]),
                    "hitFrame": hit_frame[i].item(), "maxBall": max_ball[i].item(),
                    "maxClub": max_club[i].item(), "minDist": min_dist[i].item(),
                    "fell": bool(fell[i]), "ret": ret[i].item(), "paid": paid[i].item(), "struck": struck[i],
                    "smash": smash[i].item(), "backwards": bool(backwards[i]),
                    "impactClub": impact[i, 0].item(), "impactHand": impact[i, 1].item(), "impactTime": impact[i, 2].item(),
                    "offline": offline[i].item(), "backswing": backswing[i].item(), "topTime": top_time[i].item(),
                    "pts": pack(mm), "sc": pack(sc[i, :L].float()),
                }
            )
        out_ckpts.append({"name": ckpt.stem, "iter": iteration(ckpt), "path": str(ckpt),
                          "summary": summary, "episodes": episodes})

    data = {
        "meta": {
            "task": args.task,
            "physics": type(env_cfg.sim.physics).__name__,
            "stochastic": args.stochastic,
            "numEnvs": N,
            "dt": u.step_dt,
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "run": ckpts[0].parent.name,
            "ballR": u.scene["ball"].cfg.spawn.radius,
            "teeH": u.cfg.ball_pos[2] - u.scene["ball"].cfg.spawn.radius,
            "targetDir": list(u.cfg.target_dir),
            "impactRadius": u.cfg.impact_radius,
            "armDist": u.cfg.impact_arm_dist,
        },
        "nPts": K + 1,
        "prims": prims,
        "checkpoints": out_ckpts,
    }
    # written inside the launch block: closing the app can hard-exit the process
    out = Path(args.out) if args.out else (
        ckpts[0].parent / "swings" / ("swings_" + "_".join(str(iteration(c)) for c in ckpts)
                                      + ("_stoch" if args.stochastic else "") + ".html")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out.write_text(VIEWER.read_text(encoding="utf-8").replace("__SWING_DATA__", blob), encoding="utf-8")
    print(f"[INFO] wrote {out} ({out.stat().st_size / 1e6:.1f} MB) - open it in a browser")
