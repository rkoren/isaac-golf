"""Archive a run's swings into media/: progression.mp4 (checkpoints side by side) and the swing viewer

Several run dirs (a chain, each resumed from the one before) film as one progression; the viewer is the last run's.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MEDIA = REPO / "media"
UV = ["uv", "run", "--env-file", "omni.env"]
MPH = 2.23694  # m/s -> mph
YD = 1.09361  # m -> yards

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run", nargs="+", help="run dir(s) in training order, e.g. logs/rsl_rl/golf_swing/2026-10-29_23-27-19_fixes_120hz")
ap.add_argument("--stage", required=True, help="curriculum stage label, e.g. stage0")
ap.add_argument("--note", default="", help="one line for the index")
ap.add_argument("--max_ckpts", type=int, default=4, help="checkpoints to film, spread first to last")
ap.add_argument("--video_length", type=int, default=300, help="env steps per clip (60 per second)")
ap.add_argument("--existing", action="store_true", help="copy clips/viewer already in the run dir")
args = ap.parse_args()

runs = [Path(r).resolve() for r in args.run]
run = runs[-1]
iteration = lambda p: int(re.search(r"(\d+)$", p.stem).group(1))
ckpts = sorted((c for r in runs for c in r.glob("model_*.pt")), key=iteration)
if not ckpts:
    sys.exit(f"no checkpoints in {args.run}")
n = min(args.max_ckpts, len(ckpts))
ckpts = sorted({ckpts[round(i * (len(ckpts) - 1) / max(n - 1, 1))] for i in range(n)}, key=iteration)

out = MEDIA / f"{time.strftime('%Y-%m-%d')}_{args.stage}_{run.name}"
out.mkdir(parents=True, exist_ok=True)
REPLAY = ("club", "wrist", "muscle", "muscle_k", "damping_scale", "leg_damping_scale", "impact_arm_dist",
          "max_depenetration_velocity", "motor_noise", "motor_noise_tau", "ball_forward", "wrist_gear", "flight_obs",
          "turf_area")


def replay_overrides(r: Path) -> list[str]:
    """the run's club and body from its env.yaml (reward-only settings don't change the swing)"""
    env_yaml = (r / "params" / "env.yaml").read_text(encoding="utf-8") if (r / "params" / "env.yaml").exists() else ""
    ov = [f"env.{k}={m.group(1)}" for k in REPLAY if (m := re.search(rf"^{k}: (\S+)", env_yaml, re.M))]
    # runs before the knob only have the robot's nested value
    if not any("max_depenetration_velocity" in o for o in ov) and (
        m := re.search(r"^\s+max_depenetration_velocity: ([\d.]+)", env_yaml, re.M)
    ):
        ov.append(f"env.max_depenetration_velocity={m.group(1)}")
    print(f"[archive] replay overrides for {r.name}: {ov}")
    return ov


def sh(cmd: list[str]) -> None:
    print("[archive] " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


clips = []
for c in ckpts:
    if not args.existing:
        sh(UV + ["isaaclab", "play", "--rl_library", "rsl_rl", "--task", "Isaac-Golf-Swing-Direct",
                 "--num_envs", "1", "--checkpoint", str(c), "--video", "--video_length", str(args.video_length),
                 "physics=isaacsim_physx", *replay_overrides(c.parent)])
    made = sorted((c.parent / "videos" / "play").glob(f"clip_{c.stem}_*.mp4"))
    if not made:
        print(f"[archive] no clip for {c.stem}, skipping")
        continue

    # only the stacked progression is kept; single clips stay in the run's videos/play/
    clips.append((iteration(c), made[0] if args.existing else made[-1]))

if clips:
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    font = "fontfile='C\\:/Windows/Fonts/arial.ttf':fontsize=36:fontcolor=black:x=20:y=20"
    inputs = sum((["-i", str(p)] for _, p in clips), [])
    chains = "".join(f"[{i}]scale=640:-2,drawtext=text='iter {it}':{font}[v{i}];" for i, (it, _) in enumerate(clips))
    stack = "".join(f"[v{i}]" for i in range(len(clips)))
    # more than 4 clips wrap onto two rows
    cols = len(clips) if len(clips) <= 4 else -(-len(clips) // 2)
    pos = lambda k, d: "+".join([d + "0"] * k) or "0"
    layout = "|".join(f"{pos(i % cols, 'w')}_{pos(i // cols, 'h')}" for i in range(len(clips)))
    graph = chains + (f"{stack}xstack=inputs={len(clips)}:layout={layout}:fill=white" if len(clips) > 1 else "[v0]null")
    subprocess.run([ff, "-loglevel", "error", "-y", *inputs, "-filter_complex", graph,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out / "progression.mp4")], check=True)

if not args.existing:
    sh(UV + ["python", "scripts/record_swings.py", str(run), "physics=isaacsim_physx", *replay_overrides(run)])
viewers = sorted((run / "swings").glob("*.html"), key=lambda p: p.stat().st_mtime)
summary = ""
if viewers:
    shutil.copy2(viewers[-1], out / "swings.html")
    html = viewers[-1].read_text(encoding="utf-8")
    blob = re.search(r'id="swing-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    last = json.loads(blob.replace("<\\/", "</"))["checkpoints"][-1]
    s = last["summary"]
    club = s.get("clubhead_strike_rate")
    summary = (f"{last['name']}: contact {s['contact_rate']:.0%}"
               + (f", clubhead strikes {club:.0%}, body {s['body_strike_rate']:.0%}" if club is not None else "")
               + f", ball {MPH * s['mean_max_ball_speed']:.1f} mph, club {MPH * s['mean_max_club_speed']:.1f} mph"
               + (f", hands {MPH * s['mean_impact_hand_speed']:.1f} mph, impact {s['mean_impact_time']:.2f} s"
                  if s.get("mean_impact_time") else "")
               + (f", backswing {s['mean_backswing']:.2f} m" if s.get("mean_backswing") else "")
               + (f", smash {s['mean_smash']:.2f}" if s.get("mean_smash") else "")
               + (f", offline {s['mean_offline_deg']:.0f}°" if s.get("mean_offline_deg") is not None else "")
               + (f", carry {YD * s['mean_carry']:.0f} yd, curve {YD * s['mean_curve']:+.0f} yd"
                  if s.get("mean_carry") is not None else "")
               + f", fell {s['fell_rate']:.0%}"
               + (f", finish torso {s['mean_finish_torso']:.2f} m" if s.get("mean_finish_torso") else ""))

index = MEDIA / "index.md"
if not index.exists():
    index.write_text("# Swing archive\n\nOne row per archived run. Open `swings.html` in a browser.\n\n"
                     "| date | stage | run | note | last checkpoint | files |\n|---|---|---|---|---|---|\n",
                     encoding="utf-8")
rel = out.relative_to(MEDIA).as_posix()
files = " ".join(f"[{p.name}]({rel}/{p.name})" for p in sorted(out.iterdir()))
with index.open("a", encoding="utf-8") as f:
    name = run.name if len(runs) == 1 else f"{run.name} (chain from {runs[0].name})"
    f.write(f"| {time.strftime('%Y-%m-%d')} | {args.stage} | {name} | {args.note} | {summary} | {files} |\n")
print(f"[archive] {out}")
