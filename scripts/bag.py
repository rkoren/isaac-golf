"""Carry, total and dispersion of each club, from 512 swings of its checkpoint, for the caddie: envs/golf/bag.json [m]

uv run python scripts/bag.py 7iron=logs/rsl_rl/golf_swing/<run>/model_N.pt pw=<...> driver@driver_fa=<...>   (label@club key)

Each club swings with its own checkpoint; the club key is what it is called in the bag and the club the checkpoint swings, the body
settings come from the checkpoint's env.yaml. A club's row replaces its old one; the other rows stay.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

from replay import replay_overrides, viewer_data

REPO = Path(__file__).resolve().parents[1]
BAG = REPO / "envs" / "golf" / "bag.json"
UV = ["uv", "run", "--env-file", "omni.env"]


def stats(x: list[float]) -> tuple[float, float]:
    m = sum(x) / len(x)
    return m, math.sqrt(sum((v - m) ** 2 for v in x) / max(len(x) - 1, 1))


def club_row(club: str, checkpoint: Path, num_envs: int) -> dict:
    out = Path(checkpoint).parent / "swings" / f"bag_{club}_{checkpoint.stem}.html"
    cmd = UV + ["python", "scripts/record_swings.py", str(checkpoint), "--num_envs", str(num_envs), "--keep", "1", "--out", str(out),
                "physics=isaacsim_physx", *(o for o in replay_overrides(checkpoint.parent) if not o.startswith("env.club=")), f"env.club={club}"]
    subprocess.run(cmd, cwd=REPO, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return summarise(viewer_data(out)["checkpoints"][-1], checkpoint, num_envs)


def summarise(ck: dict, checkpoint: Path, num_envs: int) -> dict:
    s, sh = ck["summary"], ck["shots"]
    carry, carry_sd = stats(sh["carry"])
    total, total_sd = stats(sh["total"])
    side, side_sd = stats(sh["total_side"])
    return {
        "carry": round(carry, 2), "carry_sd": round(carry_sd, 2), "total": round(total, 2), "total_sd": round(total_sd, 2),
        "side": round(side, 2), "side_sd": round(side_sd, 2), "run": round(stats(sh["run"])[0], 2),
        "apex": round(stats(sh["apex"])[0], 2), "land_deg": round(stats(sh["land_deg"])[0], 1),
        "ball_speed": round(s["mean_max_ball_speed"], 2), "fell_rate": round(s["fell_rate"], 3), "contact_rate": round(s["contact_rate"], 3),
        "swings": num_envs, "checkpoint": checkpoint.resolve().relative_to(REPO).as_posix(), "measured": time.strftime("%Y-%m-%d"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clubs", nargs="+", help="club=checkpoint, e.g. 7iron=logs/rsl_rl/golf_swing/<run>/model_4491.pt")
    ap.add_argument("--num_envs", type=int, default=512)
    args = ap.parse_args()
    bag = json.loads(BAG.read_text()) if BAG.exists() else {}
    for item in args.clubs:
        name, ckpt = item.split("=", 1)
        name, _, key = name.partition("@")
        bag[name] = club_row(key or name, Path(ckpt), args.num_envs)
        yd = 1.09361
        r = bag[name]
        print(f"{name}: carry {yd * r['carry']:.0f} +- {yd * r['carry_sd']:.0f} yd, total {yd * r['total']:.0f}, "
              f"side {yd * r['side']:+.1f} +- {yd * r['side_sd']:.1f} yd, fell {r['fell_rate']:.0%}")
    BAG.write_text(json.dumps(bag, indent=1) + "\n")


if __name__ == "__main__":
    main()
