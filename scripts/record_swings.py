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
# hydra overrides (physics=...) look positional to argparse, so split them off by hand
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
from envs.golf.golfer import CLUB_BODY, GOLFER_USD
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import resolve_task_config

env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
args.device = env_cfg.sim.device


def body_shapes(usd_path: str) -> dict[str, list[tuple]]:
    """Collision shapes per body, as points in that body's frame. Read from the asset, so a
    resized club or swapped humanoid shows up in the viewer without touching it."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    shapes: dict[str, list[tuple]] = {}
    seen = set()
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = prim.GetParent()
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body:
            continue
        # geom -> body. usd matrices are row-vector, hence geom * inverse(body)
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0) * (
            UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0).GetInverse()
        )
        p = lambda *v: tuple(round(c, 5) for c in m.Transform(Gf.Vec3d(*v)))
        kind = prim.GetTypeName()
        if kind == "Capsule":
            g = UsdGeom.Capsule(prim)
            h, r = g.GetHeightAttr().Get() / 2, g.GetRadiusAttr().Get()
            axis = [0.0, 0.0, 0.0]
            axis["XYZ".index(g.GetAxisAttr().Get())] = h
            shape = ("cap", p(*axis), p(*(-a for a in axis)), round(r, 4))
        elif kind == "Sphere":
            shape = ("sph", p(0, 0, 0), round(UsdGeom.Sphere(prim).GetRadiusAttr().Get(), 4))
        elif kind == "Cube":
            s = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
            shape = ("box", p(0, 0, 0), p(s, 0, 0), p(0, s, 0), p(0, 0, s))
        else:
            continue
        key = (body.GetName(), shape)
        if key not in seen:  # vendored asset authors some shapes twice
            seen.add(key)
            shapes.setdefault(body.GetName(), []).append(shape)
    return shapes


def part(body: str) -> str:
    # lead/trail for a right-hander: left arm leads. colours in the viewer key off this
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

    # flatten every shape into one point list: (body index, local point). the viewer only ever
    # sees world points, so it never has to know this sim's quaternion order (xyzw in 3.0)
    shapes = body_shapes(GOLFER_USD)
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

        # reset inside inference mode too: stepping in it turns env buffers into inference
        # tensors, and the next checkpoint's reset would then fail writing to them
        with torch.inference_mode():
            obs, _ = env.reset()
            org = u.scene.env_origins
            for t in range(T):
                # read the sim, not the env's cached values: those are computed before the
                # post-reset kinematics refresh, so frame 0 would be stale
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
        # what the ball-speed reward actually pays: gain over the running max, once armed.
        # raw max ball speed also counts a ball knocked off the tee at reset, which pays nothing
        ball_v = sc[..., 1] * valid
        prev_max = torch.cat([torch.zeros_like(ball_v[:, :1]), ball_v.cummax(1).values[:, :-1]], 1)
        paid = (torch.relu(ball_v - prev_max) * (sc[..., 4] > 0.5) * valid).sum(1)

        # who touched the ball first: nearest shape to where the ball sat, over the frames
        # either side of the step it started moving
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
        knocked = [s is not None and not s["armed"] for s in struck]
        swung = [s["by"] for s in struck if s is not None and s["armed"]]

        # best first by what the reward pays; closest approach breaks ties among misses
        order = sorted(range(N), key=lambda i: (-paid[i].item(), min_dist[i].item()))
        n = min(args.keep, N)
        picks = sorted({order[round(j * (N - 1) / max(n - 1, 1))] for j in range(n)}, key=order.index)

        summary = {
            "contact_rate": hit.float().mean().item(),
            "moved_rate": moved.any(1).float().mean().item(),
            "reset_knock_rate": sum(knocked) / N,
            "clubhead_strike_rate": swung.count("clubhead") / N,
            "body_strike_rate": sum(b not in ("clubhead", "shaft") for b in swung) / N,
            "struck_by": {b: swung.count(b) for b in sorted(set(swung), key=swung.count, reverse=True)},
            "mean_paid_ball_speed": paid.mean().item(),
            "fell_rate": fell.float().mean().item(),
            "mean_max_ball_speed": max_ball.mean().item(),
            "best_ball_speed": max_ball.max().item(),
            "mean_max_club_speed": max_club.mean().item(),
            "mean_min_dist": min_dist.mean().item(),
            "mean_return": ret.mean().item(),
            "action_std": std.mean().item(),
        }
        print(
            f"[{ckpt.stem}] contact {summary['contact_rate']:.0%}  fell {summary['fell_rate']:.0%}  "
            f"ball {summary['mean_max_ball_speed']:.2f} m/s (best {summary['best_ball_speed']:.2f})  "
            f"club {summary['mean_max_club_speed']:.2f} m/s  closest {summary['mean_min_dist']:.3f} m  "
            f"return {summary['mean_return']:.1f}  action std {summary['action_std']:.2f}"
        )
        print(
            f"    paid ball speed {summary['mean_paid_ball_speed']:.2f} m/s  knocked at reset "
            f"{summary['reset_knock_rate']:.0%}  struck after backswing by: {summary['struck_by'] or 'nothing'}"
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
