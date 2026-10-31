"""Archive a run's swings into media/
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

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run", help="run dir, e.g. logs/rsl_rl/golf_swing/2026-10-29_23-27-19_fixes_120hz")
ap.add_argument("--stage", required=True, help="curriculum stage label, e.g. stage0")
ap.add_argument("--note", default="", help="one line for the index")
ap.add_argument("--max_ckpts", type=int, default=4, help="checkpoints to film, spread first to last")
ap.add_argument("--video_length", type=int, default=300, help="env steps per clip (60 per second)")
ap.add_argument("--existing", action="store_true", help="copy clips/viewer already in the run dir")
args = ap.parse_args()

run = Path(args.run).resolve()
iteration = lambda p: int(re.search(r"(\d+)$", p.stem).group(1))
ckpts = sorted(run.glob("model_*.pt"), key=iteration)
if not ckpts:
    sys.exit(f"no checkpoints in {run}")
n = min(args.max_ckpts, len(ckpts))
ckpts = sorted({ckpts[round(i * (len(ckpts) - 1) / max(n - 1, 1))] for i in range(n)}, key=iteration)

out = MEDIA / f"{time.strftime('%Y-%m-%d')}_{args.stage}_{run.name}"
out.mkdir(parents=True, exist_ok=True)
play_dir = run / "videos" / "play"
# replay with the club the run trained with
env_yaml = (run / "params" / "env.yaml").read_text(encoding="utf-8") if (run / "params" / "env.yaml").exists() else ""
club = re.search(r"^club: (\S+)", env_yaml, re.M)
OVERRIDES = [f"env.club={club.group(1)}"] if club else []


def sh(cmd: list[str]) -> None:
    print("[archive] " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


clips = []
for c in ckpts:
    if not args.existing:
        sh(UV + ["isaaclab", "play", "--rl_library", "rsl_rl", "--task", "Isaac-Golf-Swing-Direct",
                 "--num_envs", "1", "--checkpoint", str(c), "--video", "--video_length", str(args.video_length),
                 "physics=isaacsim_physx", *OVERRIDES])
    made = sorted(play_dir.glob(f"clip_{c.stem}_*.mp4"))
    if not made:
        print(f"[archive] no clip for {c.stem}, skipping")
        continue

    src = made[0] if args.existing else made[-1]
    dst = out / f"{c.stem}.mp4"
    shutil.copy2(src, dst)
    clips.append((iteration(c), dst))

if clips:
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    font = "fontfile='C\\:/Windows/Fonts/arial.ttf':fontsize=36:fontcolor=black:x=20:y=20"
    inputs = sum((["-i", str(p)] for _, p in clips), [])
    chains = "".join(f"[{i}]scale=640:-2,drawtext=text='iter {it}':{font}[v{i}];" for i, (it, _) in enumerate(clips))
    stack = "".join(f"[v{i}]" for i in range(len(clips)))
    graph = chains + (f"{stack}hstack={len(clips)}" if len(clips) > 1 else "[v0]null")
    subprocess.run([ff, "-loglevel", "error", "-y", *inputs, "-filter_complex", graph,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out / "progression.mp4")], check=True)

if not args.existing:
    sh(UV + ["python", "scripts/record_swings.py", str(run), "physics=isaacsim_physx", *OVERRIDES])
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
               + (f", smash {s['mean_smash']:.2f}" if s.get("mean_smash") else "")
               + f", fell {s['fell_rate']:.0%}")

index = MEDIA / "index.md"
if not index.exists():
    index.write_text("# Swing archive\n\nOne row per archived run. Open `swings.html` in a browser.\n\n"
                     "| date | stage | run | note | last checkpoint | files |\n|---|---|---|---|---|---|\n",
                     encoding="utf-8")
rel = out.relative_to(MEDIA).as_posix()
files = " ".join(f"[{p.name}]({rel}/{p.name})" for p in sorted(out.iterdir()))
with index.open("a", encoding="utf-8") as f:
    f.write(f"| {time.strftime('%Y-%m-%d')} | {args.stage} | {run.name} | {args.note} | {summary} | {files} |\n")
print(f"[archive] {out}")
