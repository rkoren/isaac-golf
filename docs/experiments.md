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

## Env fixes after the replay (2026-10-30)

Four problems from the replay: wrong facing, reset knock, CCD off, clubhead half-size. All
measured with new checks: `scripts/check_address.py` (now also reports limbs that start
overlapping) and `scripts/check_impact.py` (drives the clubhead through the ball at set speeds).

**Clubhead.** USD `Cube.size` is the edge length, so the old `HEAD_SIZE / 2` scale built a
2 × 5 × 2.5 cm head. Now 4 × 10 × 5 cm. `impact_radius` 0.05 → 0.07 to cover it: a face-centre
strike now happens with ball centre 4.1 cm from head centre, a toe strike out to ~7 cm.

**Facing.** Root cause was not a missing heading term. `solve_address.py` built quaternions as
(w, x, y, z) and Isaac Lab 3.0 reads (x, y, z, w): rotation vector 0 meant upside down, and
upright-facing-+X lay outside the search box entirely, so the best it could do was sideways. The
old comment blaming "a torso link frame rotated 90° from the prim frame" was a misdiagnosis.
Fixed the order and added a heading cost (3D shoulder and hip lines along +Y; scoring only their
horizontal part let the solver roll the torso 60° and still pass) and a no-overlap cost.

| solve | final cost | torso − pelvis | notes |
|---|---|---|---|
| heading, flat, wrong quats | 0.246 | 0.18 m | shoulders tilted 60° |
| heading, 3D, wrong quats | 0.93–1.26 | 0.13–0.15 m | folded, rolled to the trail side |
| quats fixed | 0.017 | 0.33 m | real address, 11 limb overlaps |
| + overlap cost | **0.001** | 0.34 m | 0 overlaps, joints mid-range |

**Reset knock.** Root cause was not the joint jitter. The address had **11 limb overlaps** PhysX
resolves on the first step at up to the 10 m/s depenetration cap: trail hand inside lead hand
(5.8 cm) and inside the shaft (3.4 cm), forearms crossed (4.9 cm), arms inside the waist (4.9 cm).
That flung the club into the ball. Fixes: the grip cluster (club, both hands, both forearms)
never self-collides (`FilteredPairsAPI`); the solver penalises overlaps (→ 0); address leaves
2 cm of air at the face; and the ball is held on the tee until the swing arms, with unarmed ball
speed counted as zero.

**CCD.** Sweep CCD is CPU-only, so PhysX GPU drops `enable_ccd` (removed). Centre strikes, face
first, % that PhysX registers (ball leaves at > ⅓ club speed):

| config | ≤5 m/s | 10–15 | 15–20 | 25–30 | 35–40 | near miss, 2 / 5 cm clear |
|---|---|---|---|---|---|---|
| 120 Hz (current) | 100% | 38% | 22% | 0% | 0% | — |
| 120 Hz + speculative CCD on club | identical to the row above (no effect on GPU articulation links) | | | | | |
| 120 Hz + ball contact offset 0.2 m | 100% | 100% | 100% | 100% | 100% | **ghost hits 94–100% / 68–100%** |
| 240 Hz | 100% | 84% | 66% | 26% | 0% | — |
| 480 Hz | 100% | 100% | 100% | 78% | 70% | 0% / — (honest) |

Discrete collision fails once the head moves ~4 cm per substep, about *half* the 8.3 cm overlap
window: past the ball's centre, the minimum-translation push sends the ball out the back. A big
contact offset trades misses for invented strikes, which a policy would farm. The **detector**
is fixed regardless: contact is now the clubhead's swept path (closest approach during the step),
100% at every speed and 0% on near misses, where the old end-of-step sample skipped over the ball.
Physics stays at 120 Hz for now; see the open decision below.

Also fixed: `post_impact` counted double on any step with a reset (the counters ran in a function
`_reset_idx` also calls). `Metrics/min_club_ball_dist` is now the swept closest approach, so it
reads lower than earlier runs for the same motion.

Also found, not changed: the implicit actuators pull every joint toward **joint zero** (the stock
standing pose) with 10–20 N·m/rad, not toward the address. With zero action and gravity off the
clubhead leaves address at 2–7 m/s on the first step. The policy has been fighting this spring.

**Takeaways**
- A quaternion-order mix-up doesn't crash. It quietly made the right answer unreachable, and the
  workaround (a "rotated frame" theory) went into a comment as fact.
- Overlaps at reset are invisible until something gets flung. Check for them after every pose change.
- Measure the failure mode you're trading into: speculative CCD did nothing, and contact offset
  fixed tunneling by inventing strikes.

### Validation run on the fixed env (2026-10-30)

`logs/rsl_rl/golf_swing/2026-10-29_23-27-19_fixes_120hz`: same Stage 0 reward, 1024 envs, 100 iters,
PhysX 120 Hz. **1.19 s/iter** (run 3: 1.11; the tee hold and swept test cost ~4%). Throughput
probe at 480 Hz (decimation 8): **2.94 s/iter**, 2.5× slower.

| | reward first → last 10 | contact (detector) | struck by clubhead | struck by body | touched before backswing | paid ball speed | fell |
|---|---|---|---|---|---|---|---|
| run 3, model_199 (old env) | +0.06 → +6.05 | 2% | 1/64 | 47/64 | 8/64 knocked off | 3.90 m/s | 100% |
| **fixed, model_99** | +1.3 → +4.9 | **78%** | **33/64** | 1/64 | 2/64 (held, harmless) | 2.47 m/s | 89% |

The kicking all but vanished **without touching the reward**: with the ball in front of the face
instead of beside the lead leg, and no reset fling, the club became the easiest thing to hit it
with. Paid ball speed is lower than run 3's because these are real strikes, and at 120 Hz PhysX
registers only some of them at these 10–15 m/s head speeds. Strikes currently send the ball
toward **−Y**, away from the target: arming accepts 25 cm in any direction, so the policy winds up
toward the target and swings back through. Stage 0 doesn't pay direction; Stage 3 will.

Open decisions: physics rate vs analytic impact (above), whether to still gate the payout on the
clubhead (the kick is gone for now, but nothing stops it coming back), and the joint-zero PD targets.

## Analytic launch: smash factor (2026-10-30)

Decided: the ball gets **no collider**, and the env computes the launch the way launch monitors
and golf sims report it. That settles both open questions above: PhysX can't tunnel through or
invent a strike it never sees, and nothing but the clubhead can move the ball, so kicking is
impossible by construction. Throughput: unchanged from 120 Hz.

A strike is the clubhead's swept path passing the ball (after arming), with the ball inside the
head's box grown by its radius and the head moving into it. Then:

- ball speed = **smash** × head speed into the striking side, where
  smash = 1.0 + 0.47 · exp(−(toe/2.2 cm)² − (high/1.1 cm)²): 1.47 dead centre (a driver's max),
  ~1.2 for a 2 cm toe miss, ~1.0 topped. Up-down falls off twice as fast: the face is half as tall.
- the striking side is whichever leads through the ball; the back of the head gets the 1.0 floor.
- start direction = 85% face normal + 15% club path (the launch-monitor rule for a driver).
- spin and launch angle (loft) are Phase 2.

`scripts/check_impact.py` (rigid golfer, set speeds), ball/club = effective smash:

| strike | registered | smash measured | model |
|---|---|---|---|
| centre | 100% at 5–50 m/s | 1.45–1.47 | 1.47 |
| 1 cm toe | 100% | 1.37 | 1.38 |
| 2 cm toe | 100% | 1.20 | 1.20 |
| 9 cm wide of the head | 0% | — | miss |

### Run 4: Stage 0 with the analytic launch (2026-10-30)

`logs/rsl_rl/golf_swing/2026-10-30_00-02-32_smash`: Stage 0 reward unchanged, 1024 envs, 300 iters,
PhysX 120 Hz, **1.09 s/iter**. Training: reward −0.4 → **+50**, `contact_rate` 0.02 → **0.99**,
`Metrics/ball_speed` 0.02 → 12.2 m/s, `Metrics/impact_speed` → 16.8 m/s.

| checkpoint (64 rollouts, mean actions) | contact | clubhead strikes | smash | paid ball speed | clubhead | fell |
|---|---|---|---|---|---|---|
| model_0 | 0% | 0% | — | 0 | 8.3 m/s | 100% |
| model_100 | 45% | 45% | 1.00 | 1.2 m/s | 5.8 m/s | 100% |
| model_200 | 100% | 89% | 1.09 | 16.4 m/s | 17.2 m/s | 100% |
| model_299 | **100%** | **100%** | 1.13 | **19.8 m/s** | 19.5 m/s | 70% |

**The ball goes toward the target now, with no direction reward.** At model_299 every strike leaves
2–10° left of the target line (run "fixes_120hz" sent it backwards). The smash model did it: the
face pays 1.47× head speed and the back only 1.0×, so swinging face-first through the ball pays
47% more, and face-first means down the target line. Early strikes were all back-of-head (smash
exactly 1.00 at model_100); by model_299 the best hit 1.46. Launch elevation is ~0°: the face has
no loft yet. Mean smash 1.13 says most strikes are still off-centre.

First milestone met: a headless run where contact rate climbs. It still lunges and sits down after
the strike (70% fell), which Stage 0 allows.

**Takeaways**
- A launch model with the right physics is also a reward shaper: the smash floor on the back of the
  head bought direction for free.
- When the physics engine can't resolve an event at the timescale you need, compute the event
  and let the engine do the rest. One analytic step replaced CCD, a 4–8× finer step, and a reward gate.

Swings are archived per stage from here on: `scripts/archive_run.py` → `media/` (Kit clip per
checkpoint, side-by-side progression, swing viewer; index in `media/index.md`). Run 3 and the
fixed-env validation run are archived as the first two Stage 0 entries.
