"""Copy a checkpoint with extra observation inputs appended (zero weights), to resume after adding observations

  uv run python scripts/widen_obs.py <run>/model_N.pt 3 <out dir>/model_N.pt
  isaaclab train ... --checkpoint <out dir>/model_N.pt env.flight_obs=True

The new inputs start ignored (policy and value unchanged); training learns to use them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src", help="model_*.pt to copy")
ap.add_argument("n", type=int, help="observations to append")
ap.add_argument("out", help="output .pt; keep the model_N name so the iteration count reads right")
args = ap.parse_args()

ck = torch.load(args.src, map_location="cpu", weights_only=False)
old = ck["actor_state_dict"]["mlp.0.weight"].shape[1]


def widen(t: torch.Tensor, fill: float) -> torch.Tensor:
    return torch.cat((t, torch.full((*t.shape[:-1], args.n), fill, dtype=t.dtype)), dim=-1)


for sd in (ck["actor_state_dict"], ck["critic_state_dict"]):
    sd["mlp.0.weight"] = widen(sd["mlp.0.weight"], 0.0)
    # new inputs normalized as mean 0, std 1 (they are scaled to about unit size in the env)
    sd["obs_normalizer._mean"] = widen(sd["obs_normalizer._mean"], 0.0)
    sd["obs_normalizer._var"] = widen(sd["obs_normalizer._var"], 1.0)
    sd["obs_normalizer._std"] = widen(sd["obs_normalizer._std"], 1.0)
# Adam moments of the two input layers
for st in ck["optimizer_state_dict"]["state"].values():
    for key in ("exp_avg", "exp_avg_sq"):
        if key in st and st[key].dim() == 2 and st[key].shape[1] == old:
            st[key] = widen(st[key], 0.0)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(ck, args.out)
print(f"observations {old} -> {old + args.n}: {args.out}")
