# Experiments log

One row per run. No logged config/result = didn't happen.

## Phase 0 — setup sanity (2026-10-24)

Stock Isaac Lab tasks, no golf code yet. Purpose: prove the stack trains headless on this box.
All runs: `uv run --env-file omni.env isaaclab train --rl_library rsl_rl --task <T> --num_envs <N> --max_iterations 50 physics=<P>`.
Headless (3.0 default). rsl_rl PPO, `num_steps_per_env=32`.

| Task | Envs | Backend | Iters | s/iter | env-steps/s | Peak VRAM | Verdict |
|---|---|---|---|---|---|---|---|
| Isaac-Cartpole | 4096 (default) | isaacsim_physx | 20 | 0.38 | — | — | PPO loop closes, reward 1.10 |
| Isaac-Humanoid-Direct | 1024 | isaacsim_physx | 50 | 0.70 | ~47k | 3.6 GB | **Phase 0 milestone.** reward 1.12 -> 1.25 climbing |
| Isaac-Humanoid-Direct | 4096 | isaacsim_physx | 50 | 0.82 | ~160k | 4.9 GB | 4096 fits easily; reward 1.84 at iter 50 |
| Isaac-Humanoid-Direct | 4096 | newton_mjwarp | 50 | 0.56 | ~234k | 2.0 GB | fallback works, and it's *faster* |

VRAM figures are whole-GPU (includes ~0.65 GB desktop) out of 16 GB.

**Takeaways**
- 4096 envs costs ~4.2 GB of training VRAM — nowhere near the 16 GB ceiling. The CLAUDE.md
  worry about scaling to 4096 was unfounded; headroom exists for a club + ball per env.
- Scaling 1024 -> 4096 is 3.4x throughput for 1.17x iteration time. Train at 4096.
- 50 iterations proves nothing about learning, only that the loop runs. Real curves start in Phase 1.
- **Newton beat PhysX on the same task**: 1.5x throughput (234k vs 160k steps/s) and 2.4x less VRAM
  (2.0 vs 4.9 GB). Its 66 s wall time vs PhysX's 43 s is one-time Warp kernel compilation, not steady state
  (compare s/iter, not total). Decision unchanged for now — Phase 1 starts on PhysX because Phase 3's USD
  course and GUI replay live on the Isaac Sim side — but Newton is no longer just a fallback, it's the
  faster option. Revisit at the Stage 2 gate, where impact fidelity starts to matter.

## Phase 1 Stage 0 — swing at all (2026-10-24)

Golf env: stock humanoid + 0.87 m club welded to `left_hand`, gravity-free ball teed at
(0.819, 0.001, 0.089). Address pose and weld rotation both solved by `scripts/solve_address.py`.
All runs 1024 envs, rsl_rl PPO, `physics=isaacsim_physx`.

| # | Iters | s/iter | env-steps/s | Reward first -> last | contact_rate | ball speed | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 1.08 | ~30k | **-1.2 -> -6.0** | 1.0 (meaningless) | 0 | **failed by design** — see below |
| 2 | 150 | 1.11 | ~30k | **+2.6 -> +6.7** | 0.0 -> 0.17 (spiky) | 1.0 -> 2.22 m/s | climbs, but by *poking* |
| 3 | 200 | 1.11 | ~30k | **+0.06 -> +6.05** (x100) | max 0.50, mean ~0.04 | 1.14 -> 2.42 m/s | ball reward gated on arming |

**Run 1 — the instructive failure.** Stage 0 as originally written in CLAUDE.md (potential-based
clubhead-to-ball distance) *trains the policy to stand still*. The club starts at address ~6 cm
from the ball, so there is no distance to close, and any motion is punished twice: the approach
term goes negative and the energy penalty bites. Measured: `Reward/approach` pinned at -0.99,
`Metrics/clubhead_speed` fell 5.9 -> 3.2 m/s, reward fell to -6.0. The optimum of that reward
function is a statue. `contact_rate` also read 1.0 from iteration 0 — proximity alone, with the
head already inside the impact radius, measures nothing.

**Run 2 — restructured.** Dropped the approach term entirely. Pay for (a) **ball speed gained**
(the real objective — the ball is gravity-free and starts at rest, so any speed is a real strike),
(b) a **capped backswing** term for winding up, (c) clubhead speed near the ball. Contact now
requires **arming** (head must travel >0.25 m from the ball first). Reward climbs and armed
strikes appear within ~30 iterations.

**Run 3 — arming gate on the payout.** Run 2 climbed to +6.7 while `contact_rate` fell back to ~0:
the policy had found that *poking the ball from address* pays, and never swung. The ball-speed
reward is now multiplied by `armed`, so only a strike that backswung first earns anything.
Reward restarts near zero (the easy exploit is worth nothing) and then climbs 100x. Where it
comes from matters: `Reward/ball_speed` (the gated term) went 0.43 -> 2.83 and `Metrics/impact_speed`
0.055 -> 0.267 with a 3.67 m/s peak, so genuine armed strikes are happening and getting faster.
Clubhead speed now *rises* (5.20 -> 5.51 m/s) where in run 1 it fell to 3.2.

| run 3 term | first 10 | last 10 |
|---|---|---|
| `Reward/ball_speed` (gated) | +0.433 | **+2.829** |
| `Reward/backswing` | +0.083 | +0.079 |
| `Reward/clubhead_speed` | +0.052 | +0.089 |
| `Reward/trail_hand` | -0.142 | -0.093 (hand staying on the grip) |

200 iterations is ~6.5M samples -- early. `contact_rate` is still spiky (max 0.5, mean ~0.04),
so "reliably contacts the ball" is not claimed yet; that is the Stage 0-1 gate.

**Takeaways**
- Shaping toward a target the agent already occupies rewards doing nothing. Distance-based
  shaping needs the agent to start away from the goal.
- Any "contact" metric whose trigger condition is already true at reset is worthless. Arm it.
- The club + ball + CCD + 8 solver iterations cost ~35% throughput vs the bare humanoid
  (~30k vs ~47k env-steps/s at 1024 envs). Cheap for what it buys.
- **Reward exploits show up as a metric going the wrong way while reward climbs.** Run 2's reward
  rose while contact rate fell -- that divergence was the tell. Log the terms separately or you
  cannot see it.
- Gate a sparse payout on the behaviour you actually want, not just on the outcome: paying for
  ball speed alone bought a dribble; paying for ball speed *after a backswing* buys a swing.

### Run 3 replayed — it learned to kick (2026-10-29)

Watched run 3 (`logs/rsl_rl/golf_swing/2026-10-24_22-47-19`) with `scripts/record_swings.py`:
64 rollouts per checkpoint, policy-mean actions, PhysX. For each rollout it finds the shape nearest
the ball on the step the ball first moved. "Paid" is ball speed gained after arming, i.e. exactly
what `Reward/ball_speed` pays.

| ckpt | detector contact | struck by clubhead | struck by body | knocked at reset | paid ball speed | max ball speed | fell |
|---|---|---|---|---|---|---|---|
| model_0 | 3% | 1/64 | 0/64 | 17/64 | 0.15 m/s | 1.05 m/s | 100% |
| model_100 | 2% | 1/64 | 27/64 | 12/64 | 1.29 m/s | 1.88 m/s | 100% |
| model_199 | 2% | 1/64 | **47/64** | 8/64 | 3.90 m/s | 4.59 m/s | 100% |

At model_199 the armed strikes were left shin ×32, left foot ×13, shaft ×4, clubhead ×1, right leg ×2.
The run 3 write-up read the climbing `Reward/ball_speed` as "genuine armed strikes getting faster":
they are armed, and they are getting faster, but they are **kicks**. The golfer winds up (arming the
contact gate), falls, and sweeps the ball with its lead leg from the ground.

Also found while watching:
- **Reset knock.** In 1 of every 4–8 resets the clubhead starts touching the ball (address leaves ~4 cm,
  ±0.05 rad joint jitter closes it). PhysX pushes them apart and launches the ball, up to 22 m/s, on
  frame 1. It pays nothing (unarmed), but it inflates `Metrics/ball_speed` and wastes the episode:
  the ball is gone and `max_ball_speed` starts high.
- **Address faces −Y, not +X.** Shoulder and hip lines run along X, head leans −Y of the pelvis, ball
  off the golfer's left side. `solve_address.py` never constrained heading.
- **CCD is off.** PhysX logs `CCD disabled when GPU dynamics is enabled`, so `enable_ccd=True` in
  `GolfPhysicsCfg` does nothing on GPU, and the throughput note above credits CCD for a cost it isn't paying.

**Takeaways**
- Gating *when* a payout counts (after a backswing) is not gating *what* earns it. Ball speed after
  arming bought a kick. The payout has to know what struck the ball.
- Reward and its own metric can agree and both mislead. Only watching the swings showed this.
