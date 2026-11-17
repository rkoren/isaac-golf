"""Copy a checkpoint with its action noise rescaled, to resume training at a chosen exploration std

  uv run python scripts/set_action_std.py <run>/model_N.pt 0.25 <out dir>/model_N.pt
  isaaclab train ... --checkpoint <out dir>/model_N.pt

rsl_rl learns the std (entropy_coef 0), but it barely moves: set it here instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src", help="model_*.pt to copy")
ap.add_argument("std", type=float, help="new mean action std (per-joint ratios are kept)")
ap.add_argument("out", help="output .pt; keep the model_N name so the iteration count reads right")
args = ap.parse_args()

ck = torch.load(args.src, map_location="cpu", weights_only=False)
p = ck["actor_state_dict"]["distribution.std_param"]
ck["actor_state_dict"]["distribution.std_param"] = p * (args.std / p.mean())
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(ck, args.out)
print(f"action std {p.mean():.3f} -> {args.std:.3f}: {args.out}")
