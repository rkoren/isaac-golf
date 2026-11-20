# Experiments log

One row per run. No logged config/result = didn't happen.

Clubhead and ball speeds are in **mph** (the env runs in m/s; TensorBoard logged `Metrics/*_speed` in m/s until 2026-10-30, now `Metrics/*_speed_mph`).

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
| 2 | 150 | 1.11 | ~30k | **+2.6 -> +6.7** | 0.0 -> 0.17 (spiky) | 2.2 -> 5.0 mph | climbs, but by *poking* |
| 3 | 200 | 1.11 | ~30k | **+0.06 -> +6.05** (x100) | max 0.50, mean ~0.04 | 2.6 -> 5.4 mph | ball reward gated on arming |

**Run 1 — the instructive failure.** Stage 0 as originally written in CLAUDE.md (potential-based
clubhead-to-ball distance) *trains the policy to stand still*. The club starts at address ~6 cm
from the ball, so there is no distance to close, and any motion is punished twice: the approach
term goes negative and the energy penalty bites. Measured: `Reward/approach` pinned at -0.99,
`Metrics/clubhead_speed` fell 13.2 -> 7.2 mph, reward fell to -6.0. The optimum of that reward
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
0.12 -> 0.60 mph with an 8.2 mph peak, so genuine armed strikes are happening and getting faster.
Clubhead speed now *rises* (11.6 -> 12.3 mph) where in run 1 it fell to 7.2.

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
| model_0 | 3% | 1/64 | 0/64 | 17/64 | 0.34 mph | 2.3 mph | 100% |
| model_100 | 2% | 1/64 | 27/64 | 12/64 | 2.9 mph | 4.2 mph | 100% |
| model_199 | 2% | 1/64 | **47/64** | 8/64 | 8.7 mph | 10.3 mph | 100% |

At model_199 the armed strikes were left shin ×32, left foot ×13, shaft ×4, clubhead ×1, right leg ×2.
The run 3 write-up read the climbing `Reward/ball_speed` as "genuine armed strikes getting faster":
they are armed, and they are getting faster, but they are **kicks**. The golfer winds up (arming the
contact gate), falls, and sweeps the ball with its lead leg from the ground.

Also found while watching:
- **Reset knock.** In 1 of every 4–8 resets the clubhead starts touching the ball (address leaves ~4 cm,
  ±0.05 rad joint jitter closes it). PhysX pushes them apart and launches the ball, up to 49 mph, on
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

| config | ≤11 mph | 22–34 | 34–45 | 56–67 | 78–89 | near miss, 2 / 5 cm clear |
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
clubhead leaves address at 4–16 mph on the first step. The policy has been fighting this spring.

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
| run 3, model_199 (old env) | +0.06 → +6.05 | 2% | 1/64 | 47/64 | 8/64 knocked off | 8.7 mph | 100% |
| **fixed, model_99** | +1.3 → +4.9 | **78%** | **33/64** | 1/64 | 2/64 (held, harmless) | 5.5 mph | 89% |

The kicking all but vanished **without touching the reward**: with the ball in front of the face
instead of beside the lead leg, and no reset fling, the club became the easiest thing to hit it
with. Paid ball speed is lower than run 3's because these are real strikes, and at 120 Hz PhysX
registers only some of them at these 22–34 mph head speeds. Strikes currently send the ball
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
| centre | 100% at 11–112 mph | 1.45–1.47 | 1.47 |
| 1 cm toe | 100% | 1.37 | 1.38 |
| 2 cm toe | 100% | 1.20 | 1.20 |
| 9 cm wide of the head | 0% | — | miss |

### Run 4: Stage 0 with the analytic launch (2026-10-30)

`logs/rsl_rl/golf_swing/2026-10-30_00-02-32_smash`: Stage 0 reward unchanged, 1024 envs, 300 iters,
PhysX 120 Hz, **1.09 s/iter**. Training: reward −0.4 → **+50**, `contact_rate` 0.02 → **0.99**,
`Metrics/ball_speed` 0.04 → 27.3 mph, `Metrics/impact_speed` → 37.6 mph.

| checkpoint (64 rollouts, mean actions) | contact | clubhead strikes | smash | paid ball speed | clubhead | fell |
|---|---|---|---|---|---|---|
| model_0 | 0% | 0% | — | 0 | 18.6 mph | 100% |
| model_100 | 45% | 45% | 1.00 | 2.7 mph | 13.0 mph | 100% |
| model_200 | 100% | 89% | 1.09 | 36.7 mph | 38.5 mph | 100% |
| model_299 | **100%** | **100%** | 1.13 | **44.3 mph** | 43.6 mph | 70% |

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

## Athletic stance + balance (2026-10-30)

Address re-solved with stance costs: feet **0.40 m** apart (was 0.18; shoulders 0.34), knees
**25°** (was 31–36°), flat feet, centre of mass over their middle, 0 overlaps. Balance terms:
torso below 0.95 m and feet sliding > 3 cm, 10 per m per second; post-impact finish 0.5 → 1 s.

| run | setup | contact | ball speed | clubhead | smash | fell | ball heads |
|---|---|---|---|---|---|---|---|
| 5 `athletic` | balance on **from scratch** | 0% | — | 13.0 mph | — | 6% | — |
| 6 `athletic_strike` | balance off, 300 iters | 100% | 41.6 mph | 48.1 mph | 1.00 | 100% | −174° (backwards) |
| 7 `athletic_balance` | resumed from 6, balance on, 300 iters | 100% | **67.6 mph** | **71.1 mph** | 1.00 | 98% | −164° (backwards) |

(64 rollouts at each run's last checkpoint, mean actions.)

**Run 5 is run 1 again.** From scratch the balance penalty (~30–60 per episode while the
untrained policy falls) dwarfs the rare strike, so the policy learned to stand still. Balance is
off by default now and only switched on when resuming from a striking checkpoint.

**Run 7 swung harder instead of staying up.** At weight 10 a fall costs ~5 per episode and a
67 mph strike pays ~90 (reward is 3 per m/s), so falling is worth it; torso height during training only went 0.48 → 0.55 m.

**Runs 6 and 7 strike backhanded** (smash exactly 1.00, ball backwards). Run 4's face-first swing
was a lucky draw, not something the smash floor reliably produces: the backhand basin is just as
easy to fall into.

**Takeaways**
- Shaping that's easy to satisfy dominates until the task reward shows up. Stage it in by checkpoint.
- A penalty only buys behaviour if it costs more than what breaking it earns.
- One run's emergent behaviour isn't a property of the reward. Check a second seed before claiming it.

## Backwards hits punished (2026-10-30)

Rule: a strike with the back of the head, or a launch heading > 90° off the target line, pays no
ball speed and costs a one-off 20 (a forward strike pays ~90). No golf shot goes backwards.
Strike stage from scratch (balance off), 300 iters, two seeds:

| run | backwards (training, first → last) | contact | smash | ball speed | heading | fell |
|---|---|---|---|---|---|---|
| 8 `nobackwards_s1` | 43% → 0% | 100% | 1.15 | 51.9 mph | +16° (13 to 17) | 98% |
| 9 `nobackwards_s2` | 38% → 0% | 100% | 1.17 | 60.4 mph | +20° (15 to 22) | 100% |

(last checkpoint, 64 rollouts, mean actions; heading + is right of target for this right-hander)

Both seeds abandon the backhand and strike face-first, forward, so this holds where run 4's
direction didn't. Strikes are a consistent ~18° push: Stage 3's direction term is what tightens
that. Balance is the next stage, resumed from run 9.

## Fuller swing vs hold the finish (2026-10-30)

Watching run 9: the strike lands 0.13–0.15 s after address with the torso still at 1.15 m. It is
a jab, not a swing, and the fall comes entirely after impact. So posture-at-impact gating would
change nothing. Two nudges, each resumed from run 9 for 300 iters (balance penalties off):

| run | change | contact | smash | ball speed | clubhead | fell | torso at finish |
|---|---|---|---|---|---|---|---|
| 9 (start) | — | 100% | 1.17 | 60.4 mph | 58.8 mph | 100% | ~0.15 m |
| 10 `fullswing` | strike counts only after a 1.0 m backswing (`impact_arm_dist` 0.25 → 1.0, `backswing_cap` 0.6 → 1.2) | 100% | 1.10 | 82.3 mph | **77.4 mph** | 97% | 0.25 m |
| 11 `holdfinish` | half the ball pay held until the end of the 1 s finish, × posture (`finish_frac=0.5`) | 100% | **1.28** | **83.0 mph** | 69.1 mph | **0%** | 0.74 m |

- Run 10: a longer arc buys head speed (58 → 78 mph), and the swing takes a bit longer (impact at 0.22 s). Still falls.
- Run 11: tying a slice of the *dominant* reward to the finish did what the side penalties
  (run 7) couldn't: no falls, and strikes got more centred (smash 1.17 → 1.28). It finishes crouched.

**Takeaway**
- To change behaviour after the payout, make part of the payout depend on it. A penalty that's
  small next to the reward gets traded away; a deferred share of the reward doesn't.

## Clubs, and both nudges together (2026-10-30)

Clubs: `envs/golf/clubs.py` (driver, 7iron, pw), one asset and one solved address each. Shorter
clubs stand closer: ball 0.82 / 0.62 / 0.57 m from the feet, stance 0.40 / 0.34 / 0.30 m, spine
tilt 29° / 35° / 35°, 0 overlaps. Heads are visual meshes (460cc-style driver, lofted blades);
the strike still uses each club's box collider. Iron heads sit ~2 cm above target height, so
irons get a short tee. Kit clips are now filmed down the line.

| run | setup | contact | smash | ball speed | clubhead | fell | heading |
|---|---|---|---|---|---|---|---|
| 12 `driver_full_finish` | resumed from 11, both nudges | 100% | 1.17 | **86.3 mph** | 80.3 mph | 5% (but kneels: torso 0.73 m at finish) | +31° |
| 13 `7iron_from_driver` | 7-iron warm-started from 11's driver policy | **0%** | — | — | — | — | — |
| 14 → 15 `7iron_full_finish` | 7-iron strike stage from scratch (seed 2), then both nudges | 100% | 1.08 | 93.3 mph | 91.9 mph | 100% | −8° |

- Run 12 is the best driver swing yet: fuller (impact 0.21 s after address), faster, and mostly upright,
  though it finishes kneeling. The push grew (+21° → +31°); Stage 3's direction term owns that.
- Run 13: a driver policy does not transfer to an iron address (different pose, ball 20 cm closer).
  Contact never came back. Train each club's curriculum fresh.
- Run 15: the iron learned to strike hard and roughly straight, but with both nudges applied to a policy
  that already fell, it chose flat-out speed (92 mph head, faster than the driver) over the finish
  share. Applying the finish first, then the full backswing (the order that worked for the driver), is next.

## Speed audit: faster than a human (2026-10-30)

Run 15's 7-iron head speed (92 mph, above the ~90 mph Tour 7-iron average) on a swing that isn't full
was suspicious. Rolled out both final policies (64 episodes, mean actions) with the new at-impact metrics:

| run | clubhead at impact | lead hand at impact | address to impact | club / hand |
|---|---|---|---|---|
| 12 driver | 77.3 mph | 23.4 mph | 0.21 s | 3.3 |
| 15 7-iron | 90.9 mph | **35.5 mph** | 0.27 s | 2.6 |
| real golfers, for scale | 7-iron ~75–90, driver ~90–115 | ~20 mph | ~1 s takeaway to impact | ~4–5 (wrist release) |

- The speed is real at impact, not a post-impact flail: the episode max is the impact frame in most
  swings, with the torso still upright.
- The body producing it isn't human. With no wrists the club can't out-run the hands the way a
  released wrist does, so head speed has to come from hands moving far faster than a person's, and
  the whole swing takes as long as a real downswing alone. The torso also lunges at ~8 mph.
- Why it can: the actuators give full torque (shoulders 67.5 N·m per axis, elbows 45) at any joint speed
  up to 100 rad/s. Muscles lose torque as they speed up, and nothing models that.
- 11 of 12 recorded 7-iron strikes are attributed to the shaft, with smash ~1.0–1.2: heel strikes
  next to the hosel. The detector still counts them as head strikes (the ball overlaps the box).

Now logged per strike: TensorBoard `Metrics/impact_speed_mph` (strikes only; it used to average in
zeros for misses), `Metrics/impact_hand_speed_mph`, `Metrics/impact_time`; also in `record_swings.py`,
the viewer and `media/index.md`. Candidate fix, not yet tried: a torque–speed curve (Isaac Lab
`DCMotorCfg`), or a hand-speed cap in the reward.

## Muscles: Hill torque–speed curve (2026-10-30)

`env.muscle=true` scales each joint's torque by Hill's force–velocity curve, (1 − x)/(1 + x/0.25) with
x = shortening speed / vmax (vmax 20–30 rad/s per joint, `MUSCLE_VMAX`), every physics substep.
Lengthening keeps full torque; past vmax the joint goes slack. Baseline, run 12 with muscles switched on
and no retraining: clubhead at impact 77 → 44 mph, hands 23 → 14 mph, 73% fall. Its timing no longer works.

Run 12 at impact, before muscles: elbows ~30 rad/s, hips and knees 26–29 rad/s, most joints at full
torque for 50–90% of the downswing (bang-bang). Body mass 41 kg; arm segments are human-sized.

All 64 rollouts, mean actions:

| run | setup | clubhead at impact | lead hand | address → impact | smash | fell | torso at finish |
|---|---|---|---|---|---|---|---|
| 12 (reference) | no muscles | 77.3 mph | 23.4 mph | 0.21 s | 1.18 | 6% | 0.72 m |
| 16 `driver_muscle_resume` | muscles on, resumed from 12, both nudges, 300 iters | 48.9 mph | 15.5 mph | 0.28 s | 1.28 | 2% | 0.70 m |
| 17 → 18 `driver_muscle_full_finish` | muscles on from scratch: strike stage, then both nudges | 39.7 mph | 9.1 mph | 0.32 s | **1.31** | 11% | **0.84 m** |

- Speed comes down to human-limited (hands now at or below the ~20 mph a real golfer reaches), and
  strikes get more centred. Run 18's best swings hit 1.42–1.47 smash, the driver's ceiling.
- Run 18 found its own release: hands at 9 mph but clubhead at 40, a 4.5× club-to-hand ratio
  (runs 12/15: 3.3× / 2.6×; a real golfer's wrist release gives ~4–5×). With no wrist joint, it gets
  this by rotating the whole lead arm so the hand stays near the axis while the club sweeps around it.
  It also lunges least (torso ~2 mph at impact vs 5–8).
- The swing did not lengthen: top of the backswing at ~0.2 s, impact ~0.3 s (a human takes ~1 s).
  Fine-tuning (16) kept the old timing, only slower.
- Standing did not start paying for itself. Both still kneel at the finish; 18 stands tallest
  (0.84 m vs address 1.18). The finish credit still pays 22% for a 0.69 m kneel, and dropping
  into the shot is free energy from gravity.
- Run 18's finish reward was still climbing at iteration 598 (2.6 → 14.5 over the last 100).

Archive bug found and fixed: `archive_run.py` replayed only `club` from the run's `env.yaml`, so 16's
first archive was filmed without muscles (20.8 mph hands, 72% "fell"). It now replays `club`, `muscle`,
`muscle_k` and `impact_arm_dist`. `record_swings.py` now reports torso height at the finish, since
"fell" (torso below 0.6 m) misses kneeling.

**Takeaway**
- Human-limited muscles fix the superhuman speed and improve strike quality, but on their own they don't
  make standing pay or the swing longer: the policy keeps the cheapest motion that still works.

## Aim and a stricter finish, with a control (2026-10-31)

Until now the ball pay ignored direction short of backwards: a hit 40° right paid the same as a straight
one. Archived final checkpoints, launch angle off the target line: run 12 +30°, run 16 +40°, run 18 +7°
(all to the right, a push).

New knobs, both off by default:
- `env.aim_sigma` [deg]: ball pay × exp(−(offline / σ)²), offline being the horizontal launch angle off
  the target line. Logged as `Metrics/offline_deg`; `record_swings.py` and the viewer report it
  (+ = right of target).
- `env.finish_low` / `env.finish_high` [m]: the torso heights at the finish that earn zero / full
  finish credit (were fixed at 0.6 / 1.0, which paid 22% for a 0.69 m kneel).

Both runs resume run 18's `model_598.pt` with its settings (`env.muscle=True env.finish_frac=0.5
env.impact_arm_dist=1.0 env.backswing_cap=1.2`), 300 iterations, 1,024 envs, ~8.5 min each, side by side.

All 64 rollouts, mean actions, final checkpoint (`model_897`):

| run | change | clubhead at impact | smash | offline | fell | torso at finish |
|---|---|---|---|---|---|---|
| 18 (start) | — | 39.7 mph | 1.31 | +7° | 11% | 0.84 m |
| 19 `driver_muscle_aim_stand` | `env.finish_low=0.8 env.aim_sigma=15` | 41.1 mph | 1.34 | **+4°** (2.5–4.2° at 700–897) | 2% | 1.00 m (1.00–1.04) |
| 20 `driver_muscle_more` | none (control) | 41.1 mph | 1.32 | +6° (6.4–7.6°) | 2% | 0.98 m (0.96–0.98) |

- Aim works and costs nothing: offline roughly halves with no loss of speed or smash. In training it
  fell from 7.6° to ~3° within the first 50 iterations. Every strike still pushes right; at σ = 15° a 4°
  miss keeps 93% of the pay, so the pressure fades as it gets close.
- Standing up was mostly more training. The control rose from 0.84 to 0.98 m on its own; the stricter
  finish curve added at most a few cm (1.00–1.04 vs 0.96–0.98, within checkpoint-to-checkpoint noise).
  Run 18's finish reward was still climbing when it stopped, and that is what these 300 iterations bought.
- Watching the clips: the kneel became a wide lunge. The trail foot slides far back, the torso bends
  over the lead leg, then rises toward the end of the finish. Upright-ish, not a golf finish; foot slide
  is unpenalised (`feet_slip_weight=0`).
- The swing still doesn't lengthen: impact at 0.32 s, hands ~10 mph, clubhead ~41 mph.

**Takeaway**
- Check a reward change against a control run started from the same checkpoint. Without run 20, run 19's
  taller finish would have been credited to the stricter finish curve.
- Direction is cheap to learn once strikes are reliable; tighten σ (e.g. 8°) to keep pressure on the
  last few degrees.

## Joint damping: the hidden brake (2026-10-31)

Question: why doesn't the swing lengthen? Every run so far tops out ~1.18 m from the ball at 0.17 s,
exactly the 1.2 m `backswing_cap` pays for.

The stock humanoid's implicit actuators carry joint damping: 5 N·m·s/rad on waist, shoulders, pelvis and
hips (1 on elbows and feet, 0.1 on knees), roughly 10–50× a human joint's passive damping. Damping
torque grows with joint speed, so each joint has a terminal speed where muscle torque = damping torque,
reached within a few degrees. Past that, a longer backswing only feeds the damper.

Downswing energy budget (top of backswing → impact, 64 rollouts, scratchpad `damping_audit.py`):

| policy | muscle work | lost to dampers | returned by joint springs | club KE at impact |
|---|---|---|---|---|
| run 19 (stock damping) | 222 J | **163 J (74%)** | 10 J | 53 J |
| run 22 (damping × 0.1) | 161 J | 51 J (32%) | 58 J | 142 J |

In run 19, at impact the damper out-pulls the muscle on waist twist, both shoulders and both hips
(e.g. waist 49 vs 12 N·m): those joints are already slowing down.

New knob `env.damping_scale` (default 1.0) scales `DAMPING` in `golfer.py`. `archive_run.py` replays it.
New metric: backswing, the farthest the head gets from the ball before impact (`Metrics/backswing_m`,
`record_swings.py`, viewer).

Runs, muscles on, from scratch with run 17 → 18's recipe (strike stage 300 iters, then
`env.finish_frac=0.5 env.impact_arm_dist=1.0 env.backswing_cap=1.2` 300 iters):

| run | change | clubhead at impact | lead hand | backswing | address → impact | smash | offline | fell | torso at finish |
|---|---|---|---|---|---|---|---|---|---|
| 17 → 18 (reference) | stock damping | 39.7 mph | 9.1 mph | 1.18 m | 0.32 s | 1.31 | +7° | 11% | 0.84 m |
| 21 → 22 `lowdamp` | `env.damping_scale=0.1` | **67.5 mph** | 22.4 mph | 1.28 m | 0.26 s | 1.22 | +12° | **98%** | 0.30 m |
| 23 → 24 `lowdamp_g995` | also `agent.algorithm.gamma=0.995` | 48.8 mph | 12.7 mph | 1.22 m | 0.26 s | 1.22 | −61° | 98% | 0.21 m |

- Speed: +70% clubhead with hands at human speed (~20 mph). The damper loss is mostly gone and the
  policy now loads the joint springs (hips, arms) on the way back and gets 58 J back on the way down,
  a crude stretch-and-recoil. The swing looks more like one: arms extended, club high in the follow-through.
- The backswing barely grew (1.18 → 1.28 m). At impact the arms now run at 11–22 rad/s against a muscle
  vmax of 25, so they're at their speed limit: a longer arm arc can't add arm speed. In a human the extra
  comes from the chain behind the arms (hips → torso → arms → wrists); here the torso barely turns at
  impact (waist ~6 rad/s, pelvis ~0) and there are no wrists.
- Balance collapsed: 98% end on the ground (sits down after the follow-through, then lies back). The
  dampers had been holding the legs steady for free. And falling roughly breaks even under the held-finish
  reward: 82 mph ball × ½ pay ≈ run 18's 53 mph × 0.8 pay.
- γ 0.995 is a discount-factor test: at 0.99 a reward 1 s later is worth 45% less, which could favour a
  rushed swing. It didn't lengthen anything (impact 0.26 s in both) and this seed trained worse
  (offline −61°, slower). Not the cause; not adopted. Single seeds, so treat the gap loosely.

**Takeaway**
- Measure where the energy goes before reshaping rewards. The short swing was a physics artifact (a speed
  cap from damping), not something to fix with a backswing reward.
- Human-like damping makes balance an active skill. Standing now needs either its own reward
  (`stand_weight`, `feet_slip_weight`) or stiffer legs.

## Getting balance back at low damping (2026-10-31)

Two fixes for run 22's 98% falls, both resumed from its `model_598.pt` for 500 iterations, side by side,
both with `env.aim_sigma=15 env.finish_low=0.8` on top of run 22's settings:
- run 25 `lowdamp_balance`: damping × 0.1 everywhere, plus the balance rewards
  (`env.stand_weight=10 env.feet_slip_weight=10`).
- run 26 `lowdamp_braced`: legs back at stock damping (new knob `env.leg_damping_scale=1.0`; hips,
  knees, ankles), swing joints × 0.1, no balance rewards. The argument: a braced human leg co-contracts
  opposing muscles, which a one-torque-per-joint actuator can't, so leg damping stands in for it.

All 64 rollouts, mean actions, final checkpoint (`model_1097`):

| run | clubhead at impact | lead hand | ball | smash | offline | fell | torso at finish |
|---|---|---|---|---|---|---|---|
| 22 (start) | 67.5 mph | 22.4 mph | 82 mph | 1.22 | +12° | 98% | 0.30 m |
| 25 balance rewards | **70.9 mph** | 23.4 mph | **89 mph** | 1.27 | 1° | 19% | 0.90 m |
| 26 braced legs | 59.7 mph | 20.2 mph | 72 mph | 1.21 | 1° | 16% | 0.93 m |

- Both stand again most of the time. Run 25 was still improving at the end (falls 100 → 83 → 58 → 34 →
  28 → 19% across checkpoints); run 26 flattened from iteration 900.
- Run 25 keeps the speed; braced legs cost ~11 mph, since the leg dampers burn energy again.
- Aim at low damping: ~1° offline in both, from +12°.
- Balance is still marginal on the hardest hits. One-golfer replays: seed 42 falls (ball 100 mph), seed 7
  stands (73 mph). The archived clip happens to be a seed that falls; the 64-rollout numbers are the
  representative ones.
- The backswing stays at ~1.25 m in both: balance doesn't change the swing length.

**Takeaway**
- Balance rewards on a low-damping body beat stiff legs: same standing rate, ~11 mph more. A reward that
  costs nothing while standing doesn't drown the strike once the policy already strikes reliably.

## Lead wrist hinge (2026-10-31)

At low damping the arms reach impact near their muscle vmax, so a longer arm swing can't add speed. A real
golfer's long backswing pays through the wrist: set it at the top, pull the handle down with the body,
and the club releases late (lag). The golfer had no wrist; the club was welded to the lead hand.

`env.wrist=true` swaps the weld for a revolute joint `lead_wrist` (asset `golfer_driver_wrist.usda`,
`make_golfer_asset.py --club driver --wrist`; driver only so far):
- axis: the club's toe-heel line, i.e. the swing plane at address (shaft + face normal). A stand-in for
  the wrist's cock plus forearm roll, since the arms have no forearm rotation.
- limits −25° / +70° from the address angle (+ cocks the head away from the target), gear 10 N·m
  (arms 67.5), muscle vmax 30 rad/s, spring 2 N·m/rad toward address, damping 0.5 × `damping_scale`.
- 22 actions, 84 observations; the address is unchanged (head position identical to the weld's with
  reset jitter off). Reset jitter (±0.05 rad) now also moves the head ±4 cm at the wrist, which can close
  the 2 cm address gap; harmless, since a strike needs arming first.

From scratch, muscles, damping × 0.1: run 27 → 28 is run 21 → 22's two stages; run 29 adds run 25's balance
and aim stage (500 iters). All 64 rollouts, mean actions, final checkpoints:

| run | backswing | address → impact | clubhead at impact | lead hand | club ÷ hand | ball | smash | fell | torso at finish |
|---|---|---|---|---|---|---|---|---|---|
| 22 (no wrist) | 1.28 m at 0.17 s | 0.26 s | 67.5 mph | 22.4 mph | 3.0× | 82 mph | 1.22 | 98% | 0.30 m |
| 28 `wrist_full_finish` | 1.55 m at 0.23 s | 0.33 s | 77.5 mph | 17.0 mph | 4.6× | 76 mph | 1.09 | 100% | 0.13 m |
| 29 `wrist_balance` | **1.68 m** at 0.22 s | 0.33 s | **83.8 mph** | 17.0 mph | **4.9×** | **95 mph** | 1.24 | **100%** | 0.35 m |

- The backswing finally grows, past the 1.2 m the backswing reward pays for, so it's physics doing it. It
  kept growing through both stages.
- More clubhead speed from slower hands; the 4.6–4.9× club-to-hand ratio is a real golfer's 4–5×.
- The release is passive. Downswing energy budget (run 28): the wrist reaches 17 rad/s at impact but its
  muscle does 0.8 J of work. 152 J of muscle work (vs 161 J without the wrist) puts 187 J into the club
  (vs 142 J). At impact the lead arm is braking (−34 N·m), handing its energy on, as a golfer's hands
  slow through impact.
- Balance is not solved with the wrist: run 29 falls on every swing despite the balance rewards. Finish
  torso was creeping up (0.14 → 0.35 m) but nowhere near run 25's 19% falls over the same 500 iterations.
  It drops into a deep squat after impact and tips forward. A harder swing from a body that had never
  learned to stand.
- Heel strikes (labelled "shaft" by `record_swings.py`, counted as head strikes by the env) rose to 27%
  of strikes; smash 1.24.

**Takeaway**
- The wrist is the physics that makes a long backswing pay, as in real golf: a longer, later-released
  swing with slower hands and more clubhead speed, with no reward for swing length.
- Balance has to be learned again for every new body, and a faster swing makes it harder.

### More balance training, loose vs braced legs (2026-11-05)

Two ways to get the wrist golfer standing, each with 1,500 iterations of run 29's balance + aim stage in total:
- run 30 `wrist_balance_more`: run 29 continued for 1,000 more iterations (1097 → 2096).
- run 31 `wrist_braced_balance`: from run 28 (598, where run 29 started) with `env.leg_damping_scale=1.0`
  (hips, knees, ankles at stock damping, as in run 26), 1,500 iterations (598 → 2097).

| run | backswing | clubhead at impact | lead hand | ball | smash | offline | fell | torso at finish |
|---|---|---|---|---|---|---|---|---|
| 29 (start of 30) | 1.68 m | 83.8 mph | 17.0 mph | 95 mph | 1.24 | 2° | 100% | 0.35 m |
| 30 loose legs | 1.69 m | **89.7 mph** | 20.7 mph | **114 mph** | 1.38 | 1° | 98% | 0.42 m |
| 31 braced legs | 1.62 m | 82.0 mph | 17.9 mph | 106 mph | 1.39 | 1° | **83%** | 0.42 m |

- Neither stands. Run 30 spent the extra iterations on speed (clubhead +6 mph, smash 1.24 → 1.38), with
  the finish flat at 0.35–0.44 m the whole way.
- Run 31 holds its height longer after impact (torso 0.91 m 20 steps after impact, vs 0.70 m in run 30,
  over the viewer's 12 recorded swings) before collapsing, and only started to stand in its last 200 iterations (falls 100% → 89% → 83%).
  Bracing costs ~8 mph, as it did without the wrist (run 26).
- Why the balance stage stalls: half the ball pay (`finish_frac=0.5`) is scaled by finish height, and
  with `finish_low=0.8` that credit is exactly zero anywhere below 0.8 m. At a 0.42 m finish, standing a
  little taller earns nothing; only the stand penalty (10 × the drop below 0.95 m) pulls up. Run 25
  (no wrist) never hit this: it was already finishing near 0.8 m.
- "Shaft" labels in the viewer (23–44 of 64) are heel-side strikes; every launch is still a head strike
  in the env, and smash rose to 1.38–1.39.

**Takeaway**
- More iterations alone don't buy balance when the finish reward is flat where the golfer finishes.
  Braced legs help it stay up longer but cost speed.

## A one-off fall penalty (2026-11-05)

Runs 30–31 showed the balance stage gives almost no reason to stay up. Measured from run 31's recorded
swings, the per-step stand penalty (10 × the drop below 0.95 m) costs a faller ~2.3 points and a stander
~1.3, since the fall comes late in the 1 s finish window. The finish credit pays nothing to either (the
standers finish at 0.61–0.74 m, below `finish_low=0.8`). The full ball pay is ~142 points.

New knobs (off by default): `env.fall_penalty`, paid once when the torso first drops below
`env.fall_height` (0.6 m, the same line `record_swings.py` uses for "fell"). New training metric
`Metrics/fell_rate`, over every episode that ends, so falls can be watched in TensorBoard.

Both from run 31 (`model_2097`, wrist, braced legs, run 29's balance + aim stage), 500 iterations:
- run 32 `wrist_fall50`: `env.fall_penalty=50`, about a third of the ball pay.
- run 33 `wrist_fall_control`: unchanged.

Fell rate during training (exploration noise on, 1,024 envs), 100-iteration averages:

| iterations | 2097–2196 | 2197–2296 | 2297–2396 | 2397–2496 | 2497–2596 |
|---|---|---|---|---|---|
| 32 fall penalty | 87% | 82% | 78% | 68% | 66% |
| 33 control | 87% | 89% | 89% | 91% | 94% |

64 rollouts, mean actions, final checkpoints:

| run | clubhead at impact | lead hand | ball | smash | contact | fell | torso at finish |
|---|---|---|---|---|---|---|---|
| 31 (start) | 82.0 mph | 17.9 mph | 106 mph | 1.39 | 100% | 83% | 0.42 m |
| 32 fall penalty | 82.1 mph | 17.9 mph | 98 mph | 1.37 | 95% | **48%** | **0.56 m** |
| 33 control | 82.5 mph | 17.7 mph | 101 mph | 1.38 | 95% | 95% | 0.35 m |

- The penalty works and costs no clubhead speed. The control drifts the other way: without a reason to
  stand, run 31's late gain (100% → 83%) didn't hold.
- Evaluated fall rates are noisy checkpoint to checkpoint (run 32: 86, 81, 61, 72, 66, 48%); the
  training curve, over ~1,000 episodes per iteration, is the steadier read.
- Both final checkpoints whiff 3 of 64 (ball speed counts a miss as 0), so it's not the penalty.
- Most of the swings that stay up are crouched (torso 0.6–0.8 m at the end). The penalty only asks it
  to stay above 0.6 m, so it may learn to hover just above the line.

**Takeaway**
- A reward term has to separate the outcomes you care about. The stand penalty and the finish credit
  barely told a fall from a stand apart; a one-off penalty for crossing a line did.

## Out of the crouch, errant shots, a controlled swing (2026-11-11)

A batch of experiments while the user was away: lift the golfer out of its crouched finish, penalise errant
shots, and push toward a more controlled swing. Mostly on the wrist golfer with braced legs, starting
from run 32 (fall penalty 50); a few on the loose-legs body and one lineage from scratch. Each variant
ran 500 iterations from the same checkpoint as an unchanged control. Evaluation: 64 rollouts with mean actions, final checkpoint. The evaluated fall rate moves
±10 points between neighbouring checkpoints of the same run, so only bigger gaps count.

New tooling:
- `record_swings.py` also reports the spread: 90th-percentile offline angle, 10th-percentile smash, and
  the share of swings finishing with the torso at 0.9 m or higher ("tall finish").
- The env logs `Reward/actions`, `Reward/energy` and `Reward/limits`, so every term's size is visible.
  At the stock scales the energy cost is ~0.4% of the ball pay and does nothing.
- `env.strike_pow` (off by default): ball pay × the sweet-spot factor^p, the same 0–1 factor the smash
  model uses (1 at dead centre, 0.44 for a 2 cm toe miss). Reward-only; the launch is unchanged.

### What the crouch is

Measured from the 12 recorded swings of run 34 (head height, knee bend between the thigh and shin
axes, feet slide):

| | address | impact | end of finish |
|---|---|---|---|
| head height | 1.30 m | 1.02 m | 0.67 m (0.3–0.96) |
| trail knee bend | 28° | **110°** | 144° |
| lead knee bend | 30° | 33° | 140° |
| feet moved from address (both) | 0 | 0.18 m | 0.83 m, ending together |

It drops onto its trail knee in the downswing (head down 28 cm by impact), then slides its feet together
and sits. Every run below keeps the same impact posture (trail knee 104–111°, head 1.00–1.02 m): the
downswing lasts ~0.1 s, so per-second posture penalties barely touch it, and they act on the finish.

### Rounds 1–3: from run 32, then run 34

Run 34 `crouch_lift` moved finish credit to start at 0.6 m (`env.finish_low=0.6`), so the finishes
around 0.55–0.75 m sit on a slope instead of the flat zero below 0.8. Run 36 tried the fall penalty on
the faster loose-legs body (run 30). Runs 37–41 then each changed one thing from run 34's checkpoint.

| run | from | change | clubhead | ball | smash (p10) | offline (p90) | fell | finish torso |
|---|---|---|---|---|---|---|---|---|
| 32 (start) | | | 82.1 mph | 98 mph | 1.37 | 1.9° | 48% | 0.56 m |
| 34 `crouch_lift` | 32 | `finish_low=0.6` | 81.8 mph | 100 mph | 1.32 (1.15) | 2.1° (3.0°) | 48% | 0.54 m |
| 36 `loose_fall50` | 30 | loose legs + fall penalty | **88.9 mph** | **113 mph** | 1.39 | 1° | 72% | 0.52 m |
| 40 `r2_control` | 34 | none | 80.5 mph | 97 mph | 1.31 (1.13) | 2.4° (4.0°) | 20% | 0.64 m |
| 37 `aim7` | 34 | `aim_sigma=7` | 80.1 mph | 99 mph | 1.34 (1.17) | **1.8° (2.6°)** | 28% | 0.61 m |
| 38 `energy50x` | 34 | `energy_cost_scale=0.1` | 80.4 mph | 99 mph | 1.33 (1.17) | 2.3° (3.5°) | 31% | 0.63 m |
| 39 `posture` | 34 | `stand_height=1.05`, `stand_weight=40` | 79.3 mph | 96 mph | 1.30 (1.11) | 2.6° (3.6°) | 14% | 0.66 m |
| 41 `feet50` | 34 | `feet_slip_weight=50` | 80.1 mph | 94 mph | 1.32 (1.10) | 2.7° (4.1°) | 30% | 0.63 m |

- Run 34 itself changed nothing in its own 500 iterations, but the *next* 500 did: the control (run 40)
  went from 48% to 20% falls and 0.54 → 0.64 m. A slope takes a while to climb.
- Against that control, posture, energy and feet are within the checkpoint noise. Without run 40 each
  of them would have looked like a 20–35-point win. Tighter aim is the only clear effect: 90% of
  shots within 2.6° of the line, against 4.0°.
- Feet planted (run 41) cut the feet's slide at the finish from 1.12 m to 0.81 m but didn't raise it.
- The energy cost needed 50× before it was even 5–9% of the ball pay (10× was ~3%, aborted).
- No swing in any run finishes with the torso at 0.9 m or higher.
- Run 35 (run 32 continued unchanged, the control for run 34) drifted from 65% to 81% falls in training;
  its archive was stopped part-way by a low-memory reaper (three clips, no viewer).

### Rounds 4–5: from run 40

Same method, one change each from run 40's checkpoint (`model_3594`), against run 43, run 40 continued
unchanged.

| run | change | clubhead | ball | smash (p10) | offline (p90) | fell | finish torso |
|---|---|---|---|---|---|---|---|
| 43 `r3_control` | none | 78.7 mph | 93 mph | 1.28 (1.14) | 1.6° (2.8°) | 16% | 0.62 m |
| 42 `bar75` | `fall_height=0.75` | 79.1 mph | 92 mph | 1.27 (1.06) | 1.2° (2.1°) | 58% | 0.50 m |
| 45 `bar65` | `fall_height=0.65` | 79.0 mph | 91 mph | 1.25 (1.04) | 1.7° (3.9°) | 28% | 0.62 m |
| 44 `strike1` | `strike_pow=1` | 78.0 mph | 96 mph | **1.34 (1.19)** | 1.6° (2.8°) | 20% | 0.62 m |
| 46 `gamma995` | `agent.algorithm.gamma=0.995` | 79.4 mph | 93 mph | 1.27 (1.08) | 1.7° (2.8°) | 16% | 0.62 m |
| 47 `posture200` | `stand_height=1.05`, `stand_weight=200` | 78.8 mph | 94 mph | 1.28 (1.08) | 3.2° (4.8°) | **12%** | **0.69 m** |

- Raising the fall line to 0.75 m backfired: every swing dips below 0.75 m at some point (none of run 40's
  recorded swings stays above 0.7 m), so the penalty became a flat 50 points on every episode with no
  signal, and the old 0.6 m line lost its pull (falls 22% → 58%). At 0.65 m, which a third to two-fifths
  of swings clear, training falls below the line went 63% → 52%, but the evaluation didn't move.
  A threshold only teaches when a good share of attempts land on each side of it.
- Centred strikes (run 44) is the one errant-shot win here: over the last three checkpoints smash 1.35
  against the control's 1.30, the worst tenth 1.22 against 1.15, ball +3 mph on slightly less head speed.
- γ 0.995 weighs the finish (up to 1 s after impact) ~35% more than γ 0.99 does, and changed nothing.
- `stand_weight=200` made standing the biggest term (−54 against +47 for ball speed) and lifted the
  finish 0.62 → 0.69 m (head at the end 0.69 → 0.83 m) with falls at 11–12% over its last two checkpoints,
  at no speed cost, but it pulls more (−3.1° bias) and the knees still end bent ~155°.

Where the crouch comes from: the no-wrist low-damping golfers (runs 25–26) stayed tall, with the head at
1.22 m at impact and 1.10–1.14 m at the end. The wrist lineage was already at 1.08 m at impact in its
second stage (run 28) and has never stood tall since. The crouched downswing was learned before any
balance term existed, and every run here only fine-tunes on top of it.

### Wrist from scratch with braced legs (runs 49–51, 53–56)

To test whether the crouch was learned in the early stages, the wrist golfer was trained from scratch with
`env.leg_damping_scale=1.0` from iteration 0: run 27 → 28's two stages (runs 49 `scratch_strike`, 50
`scratch_full_finish`), then a balance stage.

- After the two stages (run 50, `model_598`): clubhead 65 mph, ball 68 mph, smash 1.11, offline ~6°,
  falls on every swing, as run 28 did at 78 mph. Bracing costs speed from the start, but the head stays
  higher at impact (1.20 m vs run 28's 1.08 m); the trail knee still bends ~113°.
- Run 51 `scratch_balance` used this session's full recipe straight away (finish credit from 0.6 m,
  `aim_sigma=7`, `strike_pow=1`, fall penalty 50) and **stopped hitting the ball**: contact 100% → 91% →
  8% → 0% by iteration 900, and by the end it stood still (0% falls, no swing). The stand-still trap of
  run 5, rebuilt from different parts: with shots still poor (6–13° offline, smash ~1.1), the
  multiplicative quality terms left ~5% of the ball pay (aim × sweet-spot factor × a zero finish
  credit), and the flat 50-point fall penalty outweighed it.
- Run 53 `scratch_balance2` retries the stage from run 50 with run 29's gentler recipe (finish credit
  from 0.6 m, `aim_sigma=15`, no fall penalty, no strike term).
- Run 53 kept hitting on every swing and kept improving through its 600 iterations: clubhead 65 → 75 mph,
  ball 72 → 89 mph, smash 1.26, offline 1.1° (90% within 2.0°, bias −0.5°, the straightest of the
  session), backswing 1.80 m, lead hand 17 mph. It falls on every swing (no fall penalty), but it
  topples over flat rather than sitting: knees straight at the end. At impact the head is at 1.20 m with
  the trail knee at 86°, against 1.00 m and ~110° in the braced runs that inherited the loose-legs
  swing. Bracing from the start learns a taller downswing.
- Run 54 `scratch_fall30` adds a fall penalty of 30, below the ~59 points of immediate ball pay at its
  89 mph (half the pay waits for the finish), so the stand-still trap stays shut. In 500 iterations:
  falls 100% → 41% (training 99% → 53%, still dropping fast), finish 0.35 → 0.71 m, and **48% of swings
  finish tall** (torso ≥ 0.9 m) with the head at 0.92 m and both knees nearly straight (3°/8°): standing
  up, not crouching, and no launches (torso never above 1.1 m). The outcome is bimodal: it either stands
  tall or falls, which is why the mean finish is 0.71 m. Clubhead 74 mph, ball 82 mph, smash 1.20,
  offline 1.1° (90% within 2.5°), head at impact 1.16 m, trail knee 93°.
- Run 55 `scratch_fall30b`, 500 more unchanged: falls 25% (checkpoints 42, 22, 33, 39, 36, 25%), finish
  0.83 m, **66% tall finishes**, head at the end 1.07 m with the trail leg straight; still no launches.
  Strike quality is what it gave up: smash 1.20 → 1.12, ball 82 → 76 mph at 73 mph clubhead.
- Run 56 `scratch_strike05` adds a gentle `strike_pow=0.5` (pay × √sweet, so a swing stays worth more
  than standing still at this smash): smash over the last three checkpoints 1.15 against run 55's 1.12,
  ball 77.5 mph, but clubhead 73 → 70 mph and offline 1.1° → 1.5° (bias +1.4°); falls 16–38% and tall
  finishes 45–70% across checkpoints, the same as run 55 within noise. No clear win in 500 iterations.

### Long runs on the loose-legs body, and a physics exploit (runs 48, 52)

Run 48 `loose_recipe`: run 36 (loose legs, fall penalty, 72% falls, 89 mph) with finish credit from 0.6 m,
`aim_sigma=7` and `strike_pow=1`, 1,000 iterations. Run 52 `loose_recipe2`: 1,000 more, unchanged.

| run, checkpoint | clubhead | ball | smash (p10) | offline (p90) | fell | finish torso | "tall" finish |
|---|---|---|---|---|---|---|---|
| 48, 2600 | 89.2 mph | 112 mph | 1.37 (1.21) | 1.2° (3.2°) | 67% | 0.53 m | 9% |
| 48, 3594 | 87.9 mph | 104 mph | 1.32 (1.18) | 2.3° (3.5°) | 58% | 0.88 m | 41% |
| 52, 4593 | 85.2 mph | 80 mph | 1.09 (1.00) | 8.1° (9.9°) | 20% | 1.92 m | 81% |

The "tall finishes" are flight. From run 48's checkpoint 2700 on, the golfer launches itself into the air
about half a second after impact: its lower foot goes 0.13 → 0.68 → 1.82 → 2.6 m in steps of 8 frames
(~9 m/s up), the torso reaches 2–4 m. Airborne, it's never below the fall or stand lines and the finish
credit is maxed. The ball pay meanwhile collapsed: once aim and strike quality slipped a little, the two
multiplicative terms shrank the pay further (the training-time paid ball speed fell from 34 to 2 mph), so
flight was worth more than a good shot, and the shots got worse still.
- Cause: PhysX depenetration. The robot's `max_depenetration_velocity` is 10 m/s (`golfer.py`).
  Replaying run 52's last checkpoint with
  `env.scene.robot.spawn.rigid_props.max_depenetration_velocity=1.0`: torso peak 3.7 → 1.4 m, finish
  1.92 → 0.59 m, falls 20% → 58%. The policy drives a body into an overlap and PhysX pushes it out at up to
  10 m/s; a small hop remains.
- Only the loose-legs runs found it (springy legs at damping ×0.1); no braced run's torso ever went above
  1.2 m.
- The archived Kit clip of run 52's last checkpoint happens to be a seed that kneels and falls; the
  viewer's recorded swings show the launches (11 of 12).

### Takeaways

- Controls earned their keep. Round 2's three changes each "cut falls by 20–35 points" until the control
  did the same with no change at all; only tighter aim survived the comparison.
- The crouch is a learned downswing, not a finish problem. Finish and posture terms bolted onto the
  inherited swing moved the finish 0.54 → 0.62–0.69 m at best. Training the wrist golfer with braced
  legs from iteration 0 gave a taller downswing and, with a fall penalty, the first real tall finishes
  (48%, then 66%, knees straight). Fine-tuning reshapes what a policy already does; it rarely reinvents it.
- Errant shots: `aim_sigma=7` and `strike_pow=1` each tighten their own dimension a little (direction
  90th percentile 4.0° → 2.6°; smash 1.30 → 1.35), but both multiply the ball pay, so they belong on a
  policy that already hits well.
- Three ways a reward went wrong: a threshold nobody clears teaches nothing (`fall_height=0.75`);
  multiplicative quality terms plus a fixed penalty on a weak policy make not swinging the best move
  (run 51); and a finish measured only by torso height paid for flying (run 52, PhysX depenetration).

## Depenetration cap, a shot tracer, strike quality without the trap (2026-11-11, runs 57–58)

### Depenetration cap 10 → 1 m/s

`env.max_depenetration_velocity` (new, default 1.0; it was a hard-coded 10 in `golfer.py`) caps how fast
PhysX pushes overlapping bodies apart. The loose-legs golfer had learned to use that push to launch itself
(previous section). Before lowering it for every run:
- **Resets:** across 512 jittered resets, no two limbs overlap at either cap (closest pair 5–8 mm apart).
  The reset jitter (±0.05 rad on every joint) does push a foot or the club's toe up to ~3 cm into the
  ground in some envs. At 10 m/s PhysX pops it out in one step; at 1 m/s it takes ~3 steps (50 ms), and
  the first step's body speeds run a little higher (99th percentile 0.5–1.8 vs 0.3–0.6 m/s). Body speeds
  right after the reset write are clean (≤ 0.13 m/s) at both.
- **Run 55's swing** (braced, trained at 10 m/s), 64 swings per eval:

| | cap 10, seed 42 | cap 1, seed 42 | cap 10, seed 7 | cap 1, seed 7 |
|---|---|---|---|---|
| Clubhead / ball (mph) | 72.8 / 75.6 | 72.6 / 71.9 | 72.5 / 73.2 | 72.5 / 73.9 |
| Smash | 1.12 | 1.07 | 1.08 | 1.09 |
| Fell | 25% | 42% | 33% | 30% |
| Tall finishes | 66% | 47% | 61% | 61% |

  Seed 42 looked like the cap hurt balance; seed 7 shows no difference at all. The seed alone moves a
  64-swing fall rate by 8 points, so comparisons below use 256 swings.
- `archive_run.py` replays each run at the cap it trained with (older runs: 10), so archived videos stay
  faithful.

### Shot tracer

Rendered runs (GUI or `--video`) now draw env 0's ball path as a red line, grown step by step from the
strike to the end of the episode and kept until the next strike (`env.tracer`; Isaac Lab
`VisualizationMarkers`, created only when Kit is rendering, so headless training never sees it). Its
radius grows with distance from the camera, so it reads as one width near and far. Two snags:
- Kit's point instancer ignored a growing marker count (only the first segment ever showed), so the
  tracer always draws 60 segments and shrinks the unused ones to zero.
- The ball flew underground. Every launch points 6–10° down (run 55: vertical speed about −4 m/s at
  32–37 m/s): the launch direction is 85% face normal, and the collider face has no loft (loft is visual
  until Phase 2). The ball has no collider, so it sails through the turf. The tracer therefore draws the
  path flattened to tee height: direction and speed are the sim's, height waits for the flight model.

### Archive: progression videos only

`archive_run.py` keeps `progression.mp4` and `swings.html` in `media/`; the single-checkpoint clips stay
in the run's `videos/play/`. The 156 existing clips (70 MB) were removed from `media/` (they remain in git
history), and run 35's partial archive got a progression from its 3 clips and an index row. `media/` went
193 → 124 MB; the swing viewers (97 MB) are now most of it.

### Strike quality: an additive bonus (runs 57–58)

`env.strike_pow` multiplies the ball pay by sweet^p, so a poor strike pays a fraction of a good one; on a
weak policy that is how run 51 fell into the stand-still trap. `env.strike_bonus` (new, off by default)
only adds: pay × (1 + bonus × sweet), where sweet is 1 dead centre and falls off with distance from the
sweet spot (the factor in smash). An off-centre strike pays what it always did, so swinging can never pay
less than standing still. New logs, in training (`Metrics/strike_heel_cm`, `strike_high_cm` and their
absolute values) and in `record_swings.py`: where on the face the ball was struck, + = toward the heel /
above centre.

Both runs from run 55's model_2195 at the new 1 m/s cap, 500 iterations, run 55's recipe (finish credit
from 0.6 m, aim 15, stand/feet 10, fall 30): run 57 `strike_bonus1` (+ strike_bonus 1), run 58
`scratch_control` (unchanged).

Training log, mean of each 100-iteration block (stochastic actions):

| | smash | heel (abs) | high (abs) | fell | clubhead |
|---|---|---|---|---|---|
| 57 bonus, first block → last | 1.18 → 1.16 | +0.6 (1.3) → +0.7 (1.4) cm | +0.2 (1.0) → +0.1 (1.1) cm | 37% → 42% | 71.9 → 70.7 |
| 58 control, first block → last | 1.14 → 1.14 | +1.3 (1.7) → +1.1 (1.5) cm | −0.1 (1.0) → −0.1 (1.1) cm | 38% → 36% | 71.9 → 70.3 |

Final checkpoints (model_2694), 256 swings each, mean actions:

| | 57 strike bonus | 58 control |
|---|---|---|
| Clubhead / ball (mph) | 71.2 / 77.1 | 70.4 / 73.1 |
| Smash (10th percentile) | **1.16** (1.02) | 1.11 (1.00) |
| Strike on the face: heel (abs), high (abs) | **+1.1 (1.2)**, −0.4 (1.0) cm | +1.8 (1.8), −0.7 (1.1) cm |
| Offline (90th percentile) | 2.0° (3.7°) | 1.5° (3.1°) |
| Fell | 32% | **21%** |
| Finish torso / tall finishes | 0.75 m / 54% | **0.83 m / 62%** |

- The bonus does what it says on strikes: the heel miss shrinks 1.8 → 1.1 cm, smash +0.05, ball +4 mph,
  and nothing like the stand-still trap.
- Balance pays for it (falls +11 points, tall finishes −8), the same trade `strike_pow=0.5` made in
  run 56. Likely cause: the bonus raises the whole ball pay (by about a third at this strike quality), so
  the fixed fall penalty is worth relatively less. Untested fix: scale `fall_penalty` up with it.
- Where the smash goes: in the control the misses sit toward the heel (+1.8 cm, all on that side) and
  scatter about 1 cm up and down. With the sweet spot falling off over 2.2 cm toe-heel but 1.1 cm
  vertically, the two cost about the same. The scatter remains with mean actions: it comes from the reset
  jitter feeding a chaotic swing, not from exploration noise.
- The control at the new cap keeps improving on balance: 21% falls, 62% tall finishes (run 55: 25–33%,
  61–66% at 64 swings).

## Raising the fall penalty (2026-11-16, runs 59–66)

Goal: a balanced finish before the next idea. Everything is the braced-from-scratch wrist golfer at the
1 m/s depenetration cap, run 55's recipe (finish credit from 0.6 m, aim 15, stand/feet 10), 500 iterations
per run, scored on 256 swings with mean actions. Start: run 58's model_2694 (21% falls, 62% tall finishes,
offline 1.5°).

### A dose sweep from one parent

Round 1 chained (run 59: fall 40 from run 58; run 60: the same plus `strike_bonus=1`), then the design
switched to a sweep, so each penalty gets the same parent and the same training:

| From run 58, +500 iters | 61: fall 30 (control) | 59: fall 40 | 62: fall 60 | 63: fall 100 | 60: fall 40 + bonus 1 | 64: fall 40, +1,000 |
|---|---|---|---|---|---|---|
| Clubhead / ball (mph) | 69.1 / 73.2 | 69.3 / 73.7 | 69.3 / 69.9 | 68.6 / 69.5 | 69.1 / 74.0 | 67.9 / 74.3 |
| Smash | 1.12 | 1.13 | 1.07 | 1.07 | 1.14 | 1.15 |
| Offline (90th percentile) | 3.2° (5.0°) | 3.2° (5.0°) | 3.0° (4.7°) | 2.4° (4.1°) | 3.4° (5.1°) | 5.7° (7.6°) |
| Fell | 32% | **25%** | **25%** | 27% | 32% | 29% |
| Finish torso / tall finishes | 0.75 m / 54% | 0.82 m / 58% | 0.82 m / 60% | 0.83 m / 62% | 0.75 m / 51% | 0.78 m / 52% |

- More training alone makes this lineage worse: falls 21% → 32%, offline 1.5° → 3.2° (a push).
- Raising the penalty to 40 holds falls at 25%. Beyond that nothing: 60 and 100 fall just as often and pay
  for it in strike quality (smash 1.07, the heel miss 1.4 → 2.1–2.4 cm). Another 500 iterations at 40
  didn't help either, and the push grew to 5.7°.
- The strike bonus no longer buys strikes once the penalty is up (smash 1.14 vs 1.13).
- In training, with exploration noise, every arm fell ~40% of the time whatever the penalty.
- No stand-still trap even at 100: contact stayed 100%. The ball pay at 73 mph is ~98, half of it paid at
  impact, and three-quarters of swings stand.

### How it falls

From the swing viewers (12 kept swings per eval, runs 58 and 59): the torso crosses 0.75 m 0.6–0.9 s
after impact and the 0.6 m fall line at +0.7 to +1.0 s, just before the episode ends at +1.0 s. Most go
backward, away from the ball (torso 0.8–1.2 m back): it sits back off its heels. The swings that count as
standing drift back the same way, only less. So the penalty lands at the very end of the episode, where it
is discounted most (0.99^60 ≈ 0.55 from impact), and a sit-back that crosses the line after +1.0 s is
never seen.

### Exploration noise

The policy's action std (`distribution.std_param`, the Gaussian noise rsl_rl adds to each action in
training) has hardly moved: 0.49 → 0.47 over 1,500 iterations, on actions clipped to ±1, with
`entropy_coef=0`. That much noise on every joint is a likely reason training falls sit at ~40% while the
mean action falls ~25%: the policy learns to survive its own jitter, not to finish cleanly. Round 4 tests
it: run 59's checkpoint with the std scaled to 0.25 (run 65) and to 0.12 (run 66), each resumed at fall 40
for 500 iterations, against run 64 (same parent, penalty and length, std left at 0.47).

| From run 59, fall 40, +500 iters | 64: std 0.47 (unchanged) | 65: std 0.25 | 66: std 0.12 |
|---|---|---|---|
| Clubhead / ball (mph) | 67.9 / 74.3 | **69.0 / 80.4** | 68.8 / 81.5 |
| Smash (10th percentile) | 1.15 (1.01) | **1.24** (1.04) | 1.25 (1.04) |
| Strike scatter: heel / high (abs) | 1.7 / 1.1 cm | 1.0 / 0.7 cm | 1.0 / 0.7 cm |
| Offline (90th percentile) | 5.7° (7.6°) | 4.1° (5.6°) | 4.3° (5.9°) |
| Fell | 29% | **18%** | 25% |
| Finish torso / tall finishes | 0.78 m / 52% | **0.93 m / 69%** | 0.87 m / 64% |
| Training falls, last 100 iters | 36% | 24% | 28% |

- Halving the noise did what no penalty did: falls 29% → 18%, tall finishes 52% → 69%, finish torso
  0.93 m, the lineage's best on every balance number. It also cleaned up the strike (scatter down by
  a third, smash 1.15 → 1.24, ball +6 mph).
- 0.12 is too little: strikes as clean, but balance stalls (training falls rose again after iteration
  3400). Exploration still has to find the balance fixes.
- The std stays wherever it is set (exactly 0.25 and 0.12 after 500 iterations), so it is effectively a
  hyperparameter. `scripts/set_action_std.py` writes a checkpoint copy with a new std to resume from.
- The push inherited from run 59 (offline +4°) is still there; `aim_sigma` 15 barely holds direction.

### Takeaways

- A penalty only teaches what the policy can reach. Above 40 the fall penalty bought nothing: ~25% of
  swings still fell, at any price. The noise was the ceiling: with std 0.47 on every joint the policy
  learns to survive its own jitter.
- Exploration noise is a knob, and here a big one. rsl_rl's learned std barely moves with
  `entropy_coef=0`, so a policy keeps the noise it started with (`init_std=1.0`) unless it is set by hand.
- Falls are late, slow sit-backs at the edge of the 1 s finish window; a longer window would see the ones
  that cross after it.

## Feet, tempo and curve (2026-11-16, runs 67–72)

From run 65 (fall 40, action std 0.25). Goals: keep forcing balance, plant the lead foot, and get a
longer, more gradual swing. Every run resumes run 65's `model_3692` for 500 iterations on run 65's recipe;
256 swings per eval, mean actions.

### New measurements

- **Curve.** The sim's ball never curves: launch = 85% face + 15% path, no spin, gravity off. To see what
  a real ball would do, the env now logs the face and path angles at impact (`Metrics/face_deg`,
  `path_deg`, `face_to_path_deg`; + = right of target), and `record_swings.py` labels each shot from
  face-to-path for a right-hander (approximate bands: ±2° straight, 2–6° fade/draw, beyond that
  slice/hook). Run 65: face +7°, path −8° to −9° (out-to-in), face-to-path +15° to +16°, so every shot
  is a slice. The start line (+4°) looks nearly straight only because nothing makes it curve.
- **Tempo.** `Metrics/top_time` and `downswing_time` (and the ratio in `record_swings.py`): run 65 tops
  out 0.22 s after address and strikes 0.11 s later (2:1 over a third of a second; a tour golfer is ~3:1
  over ~1.2 s).
- **Lead foot.** `Metrics/lead_foot_moved_m`: run 65's lead foot ends 0.6 m from its address spot.
  It stays planted until ~0.2 s after impact, then the lead leg kicks up (0.1–0.55 m) and out (0.3–1.0 m)
  through the follow-through; ground friction is already 1.0, so it's a move, not a slide. Most likely
  a counterbalance against the club's momentum.

### New knobs (off by default)

- `env.lead_foot_weight`: a per-second penalty on the lead foot sliding beyond `feet_slip_free` or rising
  more than `lead_foot_lift_free` (3 cm). The trail foot keeps only the horizontal slide penalty, so it
  may lift and pivot as in a real finish.
- `env.motor_noise`: signal-dependent motor noise. Each joint's torque gets an error whose standard
  deviation is that fraction of the torque, correlated over `env.motor_noise_tau` (50 ms) so it moves
  the body instead of averaging out within a step. This is the standard model of human motor variability
  (Harris & Wolpert): forceful, abrupt movement is noisier, so the hope was a smoother, slower swing. It
  is part of the world, active in evaluation too. With all knobs off the env is bit-identical to before
  (an A/B eval of run 65 matched line for line, after making the noise draw random numbers only when on).

### Rounds 1–2

| From run 65, +500 iters | 67: control | 68: lead foot 50 | 69: noise 0.1 (on / off) | 70: noise 0.2 (on / off) |
|---|---|---|---|---|
| Clubhead / ball (mph) | 68.2 / 77.2 | 69.0 / 79.4 | 67.5 / 75.3 · 67.6 / 76.8 | 66.9 / 71.6 · 67.5 / 76.1 |
| Smash | 1.20 | 1.24 | 1.17 · 1.19 | 1.14 · 1.19 |
| Fell | 17% | 15% | 18% · 16% | 26% · 18% |
| Finish torso / tall finishes | **0.95 m / 74%** | 0.95 m / 70% | 0.94 m / 71% · 0.95 m / 73% | 0.84 m / 57% · 0.91 m / 67% |
| Lead foot moved | 0.58 m | 0.62 m | 0.60 m | 0.53–0.55 m |
| Top / downswing | 0.22 / 0.11 s | 0.22 / 0.11 s | 0.21–0.22 / 0.10–0.11 s | 0.22 / 0.11 s |
| Face-to-path | +14.9° | +16.3° | +14.3° to +14.5° | +15.1° to +15.7° |

- Run 65 continued (67) keeps improving: 17% falls, 74% tall finishes, finish torso 0.95 m, all bests.
- The lead-foot term at 50 changed nothing, although it was the second-biggest term in training
  (−16 per second against a ball pay of +70, more than double the fall penalty's −7).
- Motor noise didn't touch the tempo: with the noise switched off for scoring, both runs swing exactly
  like the control. With it on they're simply worse. In 500 iterations of fine-tuning the policy
  absorbed the noise instead of slowing down.

### Rounds 3–4

| From run 65, +500 iters | 71: 1.5 s finish window | 72: lead foot 150 | 73: γ 0.999 | 74: γ 0.999 + noise 0.1 |
|---|---|---|---|---|
| Training falls, first → last 100 iters | 78% → 71% (within 1.5 s) | 25% → 24% | 29% → 31% | 29% → 29% |
| Lead foot moved (training) | 0.71–0.81 m | 0.58–0.66 m | 0.53–0.68 m | 0.53–0.63 m |
| Top / downswing | 0.22 / 0.11 s | 0.22 / 0.11 s | 0.22 / 0.11 s | 0.22 / 0.11 s |
| Clubhead / ball (mph), 256 swings | | | 69.3 / 75.5 | 67.8 / 76.2 (noise on) |

- **γ 0.999** (`agent.algorithm.gamma`): PPO's discount of 0.99 per step makes a strike 0.87 s later
  (a human tempo) worth 0.59 of the same strike now, a standing pressure to rush. At 0.999 the same delay
  costs 5%. In 500 iterations the timing didn't move at all, with or without motor noise: the swing's
  timing is deeply ingrained, and removing the pressure gives no reason to change it.
- **Lead foot 150:** still no effect on the lead leg's kick.

### The 1 s finish window hid every fall

Run 71's training falls (71–78% within 1.5 s, against ~25% within 1 s for everything else) prompted a
direct measurement: mean-action rollouts of 256 swings with a 3 s finish window, recording when each
swing first drops below 0.6 m (scratch `fall_timing.py`).

| | fallen by 1.0 s | 1.5 s | 2.0 s | 3.0 s | median fall time after impact |
|---|---|---|---|---|---|
| 67 control | 18% | 67% | 96% | 100% | 1.32 s |
| 68 / 72 lead foot 50 / 150 | 16–17% | 68–73% | 96–97% | 100% | 1.28–1.33 s |
| 69 / 70 motor noise 0.1 / 0.2 | 20–27% | 64–79% | 93–95% | 100% | 1.21–1.33 s |
| 71 1.5 s window | 21% | 66% | 96% | 100% | 1.35 s |
| 73 / 74 γ 0.999 (± noise) | 23–25% | 65–74% | 95–96% | 100% | 1.27–1.34 s |

Every swing of every run falls within 2.5 s of impact. The "17% falls, 74% tall finishes" was the 1 s
window ending first: the golfer is slowly toppling, as the videos showed (it looked wobbly). The window
is now **3 s by default** (`post_impact_steps=180`), so training, `record_swings.py` and the archives all
see a held finish. Every balance number before this section used the 1 s window.

### Takeaways

- Measure over a window long enough to see the failure. A fall penalty and finish credit checked at
  +1.0 s rewarded swings that were still on their way down.
- None of this round's knobs changed the swing: the lead leg kicks through regardless (50 or 150),
  motor noise and γ 0.999 left the timing exactly where it was. Fine-tuning a converged swing for 500
  iterations reshapes small things; timing and the finish strategy are not small.
- The ball would slice: face-to-path +15° on almost every swing (out-to-in path, face open to it).

## Holding the finish over 3 s (2026-11-16, runs 75–78)

From run 67 at the new 3 s window (fall 40, std 0.25), 500 iterations each, 256 swings per eval
(evaluated without termination, so the true fall timing shows).

New knobs (off by default):
- `env.hold_weight`: pays per second after impact, × the finish credit's upright share (torso 0.6 → 1.0 m).
  A swing that falls at 1.8 s earns more than one that falls at 0.9 s, so every swing has a gradient even
  when all of them fall (a fall penalty has no contrast at 100% falls). It starts at impact, so standing
  still at address can't earn it.
- `env.fall_terminate`: ends the episode at the fall (torso below `fall_height`), the spec's "add fall
  termination once standing is rewarded". Without it, ~1.7 s of each 3 s episode was spent lying on the
  ground; training steps now go to swings that are still standing.

| From run 67, +500 iters | 75: 3 s window | 76: + hold 20 | 77: + hold 20, terminate | 78: + hold 50, terminate |
|---|---|---|---|---|
| Fell within 3 s | 100% | 100% | 100% | 100% |
| Median fall after impact | 1.27 s | 1.25 s | 1.32 s | 1.33 s |
| Clubhead / ball (mph) | 69.7 / 75.6 | 69.1 / 71.6 | 67.9 / 72.9 | 68.3 / 73.1 |
| Smash | 1.17 | 1.11 | 1.12 | 1.12 |

- Nothing moved. The hold pay sat flat (~7 per second at 20) and the training episode length with
  termination stayed at 97–98 steps for all 500 iterations: the fall ~1.3 s after impact never shifted.
- Even at 50 per second (as much as the ball pay, over 3 s) the policy didn't trade any speed for
  balance. This lineage's fall seems built into its swing; a 500-iteration fine-tune finds no way out.
- Next: learn the swing and the finish together, from scratch (the tempo chains below).

## Tempo from scratch (2026-11-16, chains `tempo` and `ref`)

Two wrist + braced-legs chains from iteration 0 (strike 300, then the full-finish stage 300; 1 s window,
since there are no balance terms yet): `tempo` with γ 0.999 and motor noise 0.1, `ref` on the defaults.
256 swings at iteration 598:

| | tempo: γ 0.999 + noise 0.1 | ref: defaults |
|---|---|---|
| Top of backswing / impact | 0.20 / 0.30 s | 0.21 / 0.31 s |
| Tempo (backswing : downswing) | 2.0 : 1 | 2.0 : 1 |
| Backswing length | 1.29 m | 1.55 m |
| Clubhead / ball (mph) | 53.4 / 46.3 | 64.7 / 63.7 |
| Offline (bias) | −24° | −18° |

- Same timing. The first random swings were slow (top ~0.57 s, downswing 0.4–0.55 s, in the first 100
  iterations) and both chains sped up identically: by iteration 200–300, top at 0.16–0.18 s.
- So the quick swing isn't the discount's doing, nor something motor noise talks it out of: with this
  body and this reward, a fast backswing is what makes the most clubhead speed. The noise only slowed
  learning (53 vs 65 mph). A tempo would have to be asked for directly.

## Swing and finish learned together: a held 3 s finish (2026-11-16, chains `bal_ref`, `takeaway`)

New knob (off by default): `env.takeaway_weight` charges per (m/s) per second for clubhead speed above
`env.takeaway_speed` (4.5 m/s, ~10 mph, a real backswing's pace) while the club moves away from the ball
before impact. The downswing and follow-through are never charged, so it asks for a slow takeaway without
touching the strike.

Two chains, run together, wrist + braced legs:
- **A `bal_ref`:** the `ref` golfer (defaults from iteration 0, iteration 598) into two balance stages:
  600 iterations of the balance recipe (finish credit from 0.6 m, aim 15, stand/feet 10) with the 3 s
  window, `hold_weight=20` and `fall_terminate`, action std reset to 0.25 (`set_action_std.py`; it was
  0.67); then 500 more with `fall_penalty=40`.
- **B `takeaway`:** a new golfer with `takeaway_weight=10` from iteration 0: strike 300, full finish 300,
  then the same two balance stages (std 0.69 → 0.25).

Training log (stochastic actions; falls within 3 s, episode length rising = holding longer):

| | balance stage 1 start → end of stage 2 | episode length (steps) | clubhead | smash |
|---|---|---|---|---|
| A | 99% → 92% | 91 → 151 | 64.7 → 68.5 mph | 1.13 → 1.31 |
| B | 99% → 38% | 86 → 180 | 51.5 → 52.8 mph (peak 57) | 1.14 → 1.24 |

Final checkpoints (iteration 1696), 256 swings, mean actions, 3 s window:

| | A: reference | B: takeaway rule |
|---|---|---|
| Fell within 3 s | 87% | **25%** |
| Standing tall at 3 s (torso ≥ 0.9 m) | 7% | **74%** |
| Median fall after impact (fallers) | 2.23 s | 1.97 s |
| Clubhead / ball (mph) | **69.0 / 89.6** | 53.5 / 63.4 |
| Smash (10th percentile) | **1.37** (1.23) | 1.23 (1.05) |
| Offline (90th percentile) | **1.1° (2.3°)** | 2.1° (4.3°) |
| Backswing: length, top | 1.82 m, 0.21 s | 1.35 m, 0.25 s |
| Tempo (backswing : downswing) | 1.7 : 1 | 2.2 : 1 |
| Lead foot moved | 0.61 m | **0.29 m** |
| Face-to-path / shape | +8.8° / 94% slice | +5.3° / 18% straight, 42% fade, 40% slice |

- **B is the first golfer to hold a finish:** three quarters of its swings are standing tall 3 s after
  impact, its lead foot stays nearly planted, and it was still improving (training falls 52% → 38% over
  the last 300 iterations). The cost is speed: 15 mph of clubhead against A. The takeaway rule made a
  shorter (1.35 m), gentler swing, and that swing can be balanced.
- **A has the best strike so far** (smash 1.37, 90 mph ball, 1.1° offline) and learned to stay up much
  longer than the old lineage (median fall 2.2 s against 1.3 s), but 87% still fall within 3 s. It was
  also still improving.
- Learning the swing and the finish together worked where 500-iteration fine-tunes of a converged swing
  did not (runs 75–78).
- The takeaway rule moved the top only 0.21 → 0.25 s: the backswing got shorter rather than much slower,
  so this isn't a human tempo yet.

## Continuing both chains (2026-11-17, `bal_ref3`–`4`, `takeaway_15mph`/`20mph`)

Both chains resumed from iteration 1696 for two more 500-iteration stages each, same balance recipe (3 s
window, hold 20, `fall_terminate`, `fall_penalty=40`, std 0.25 carried in the checkpoints), run together.
- **A `bal_ref`:** unchanged.
- **B `takeaway`:** the takeaway limit relaxed in steps to win back speed: `takeaway_speed=6.7` (15 mph),
  then `8.9` (20 mph), weight still 10.

256 swings per final checkpoint, mean actions, 3 s window:

| | A 1696 | A 2195 | A 2694 | B 1696 (10 mph) | B 2195 (15 mph) | B 2694 (20 mph) |
|---|---|---|---|---|---|---|
| Fell within 3 s | 87% | 70% | **60%** | 25% | 25% | 24% |
| Standing tall at 3 s | 7% | 23% | **31%** | 74% | 69% | 70% |
| Median fall after impact (fallers) | 2.23 s | 2.45 s | 2.53 s | 1.97 s | 2.22 s | 1.87 s |
| Clubhead / ball (mph) | 69.0 / 89.6 | 69.4 / 90.2 | 68.4 / 88.7 | 53.5 / 63.4 | 52.8 / 64.1 | 51.6 / 57.7 |
| Smash (10th percentile) | 1.37 (1.23) | 1.38 (1.29) | 1.37 (1.27) | 1.23 (1.05) | 1.26 (1.04) | 1.16 (1.00) |
| Offline (90th percentile) | 1.1° (2.3°) | 1.2° (2.6°) | 1.1° (2.4°) | 2.1° (4.3°) | 2.4° (4.8°) | 2.9° (5.9°) |
| Backswing: length, top | 1.82 m, 0.21 s | 1.82 m, 0.21 s | 1.83 m, 0.21 s | 1.35 m, 0.25 s | 1.40 m, 0.25 s | 1.41 m, 0.26 s |
| Lead foot moved | 0.61 m | 0.50 m | 0.49 m | 0.29 m | 0.31 m | 0.35 m |
| Face-to-path | +8.8° | +8.8° | +8.2° (89% slice) | +5.3° | +6.1° | +9.0° (94% slice) |
| Takeaway charge (training, per s) | – | – | – | −1.0 | −0.25 | ~0 |

- **A keeps learning to stand at full speed:** falls 87% → 60% and tall finishes 7% → 31% over 1,000
  iterations, with clubhead, smash and direction unchanged. Still about 10 points of falls per 500
  iterations, and the falls keep coming later (median 2.5 s after impact).
- **B won back no speed:** at 20 mph the rule charged almost nothing (the backswing no longer reaches the
  limit), yet the clubhead stayed at ~52 mph and the backswing grew only 6 cm. The slow swing is now held
  in place by the balance terms, not by the rule: speeding up risks the hold pay (~17 per second over 3 s)
  that this golfer already collects, a local optimum. Its strike slipped in the last stage (smash 1.16,
  heel miss 1.7 cm, face open to +3°), and its balance didn't move (24–25% falls throughout).
- The takeaway rule bought balance by shrinking the swing, and the swing stays shrunk after the rule is
  gone. The full-speed lineage (A) is the one still improving on both fronts.

Videos: each chain is filmed from iteration 0 to 2694 as one progression (`archive_run.py` now takes several run
dirs), in `media/2026-11-17_stage0o-hold_*_takeaway_20mph/` and `media/2026-11-17_stage0o-hold_*_bal_ref4/`.
Each clip is a single swing: B's last clip happens to show one of its falls (24% of swings), and A's last one
shows a held finish (40% of swings), so judge on the 256-swing table above. The viewer labels 25–37% of chain A's strikes
"shaft" (against 3–7% for B); that is its nearest-shape vote on the recorded frame. The env only launches the ball
from the head's box, and those strikes are centred (smash 1.37, heel offset −0.2 cm).

## Chain A to 4,200 iterations: the plateau (2026-11-17, `bal_ref5`–`7`)

`bal_ref4` model_2694 resumed for three more 500-iteration stages, recipe unchanged. 256 swings, mean
actions, 3 s window (2694 repeated from above):

| | 2694 | 3193 | 3692 | 4191 |
|---|---|---|---|---|
| Fell within 3 s | 60% | 54% | **49%** | 53% |
| Standing tall at 3 s | 31% | **41%** | 40% | 38% |
| Median fall after impact (fallers) | 2.53 s | 2.58 s | 2.47 s | 2.45 s |
| Clubhead / ball (mph) | 68.4 / 88.7 | 69.0 / 87.9 | 68.3 / 87.4 | 67.6 / 85.6 |
| Smash (10th percentile) | 1.37 (1.27) | 1.35 (1.22) | 1.34 (1.21) | 1.31 (1.12) |
| Offline (90th percentile) | 1.1° (2.4°) | 1.5° (3.2°) | 1.2° (2.5°) | 1.7° (3.4°) |
| Lead foot moved | 0.49 m | 0.47 m | 0.51 m | 0.53 m |
| Face-to-path / slice share | +8.2° / 89% | +8.5° / 94% | +7.9° / 84% | +7.2° / 68% |

Training falls (stochastic actions, mean per 250 iterations) went 0.97 → 0.75 over stages 2–4, then 0.69,
0.66, 0.68, 0.64, 0.62, 0.65 over these three. Episode length stayed at 170–175 steps throughout.

- **Balance plateaus near 50% falls and 40% tall** from about iteration 3200. A 256-swing rate is good to
  about ±3 points, so 3193–4191 are one level.
- **The strike slowly gives way:** smash 1.37 → 1.31 (10th percentile 1.27 → 1.12), ball −3 mph. With
  balance stalled, the policy trades strike quality for nothing measurable.
- Best checkpoint of the chain: **`bal_ref6` model_3692** (49% falls, 40% tall, 68 mph, smash 1.34).
  More of the same recipe won't take it further; the next change has to come from the reward or the body.

## A balance reward vs a harsher fall penalty (2026-11-17, `harsh_fall`, `balance`)

The plateaued golfer stands with straight legs and makes no attempt to balance or shift its weight. The
question was whether a much harsher fall penalty fixes that. The reward already makes falling expensive:
a held finish earns the delayed half of the ball pay plus ~60 of hold pay, so a weak strike that holds
(~127) beats a 90 mph ball and a fall 2.5 s after impact (~64). It falls half the time anyway: the penalty
says what not to do, never how to stay up, and it arrives 2.5 s after the swing that set it up. (The old
"saturates above 40" sweep, runs 59–66, ran on the 1 s window, before most falls happened.)

New knobs (off by default):
- `env.balance_weight` [per m per second]: the **extrapolated CoM** (Hof 2005: CoM + velocity/ω0,
  ω0 = √(g/CoM height), where the body is headed) outside the support box of the grounded soles. Each sole
  is a heel and a toe point (`foot_sole`, from the humanoid's foot capsules); a point counts as grounded
  under 2 cm, so a foot rolled onto its toe still supports. This warns before the fall, while torso height
  still looks fine.
- `env.shift_weight` [per second after impact, × the upright share]: pays the CoM's progress from the trail
  foot's address spot toward `shift_target` (0.85 of the way to the lead foot's).
- `Metrics/xcom_out_cm`, `Metrics/finish_shift`; `record_swings.py` prints the weight shift at impact and at
  a held finish, and the share of upright time after impact with the extrapolated CoM off the soles.

Both from `bal_ref6` model_3692, 500 iterations, against `bal_ref7` (same parent, unchanged). 256 swings:

| | `bal_ref7` (fall 40) | `harsh_fall` (fall 150) | `balance` (balance 100, shift 20, fall 40) |
|---|---|---|---|
| Fell within 3 s | 53% | 46% | **40%** |
| Standing tall at 3 s | 38% | 45% | **49%** |
| Weight toward lead foot at a held finish | 0.33 | 0.41 | **0.50** |
| Extrapolated CoM off the soles after impact | 71% | 67% | 67% |
| Clubhead / ball (mph) | 67.6 / 85.6 | 67.4 / 86.3 | 67.7 / 86.2 |
| Smash (10th percentile) | 1.31 (1.12) | 1.33 (1.16) | 1.33 (1.17) |
| Face-to-path / slice share | +7.2° / 68% | +6.9° / 65% | +6.4° / 54% |

(The `balance` run trained with a first version of the support test that read a foot as lifted once its
ankle rose 3 cm, so a heel-up foot didn't count; the sole version replaced it after this round. On this
golfer the two hardly differ: its trail foot really leaves the ground.) Training: weight-shift pay 5.9 →
8.1 per second and the extrapolated CoM 16.5 → 15.2 cm off on average, both still moving at the end.

- **The balance reward beat the harsher penalty:** −13 points of falls against the control (two standard
  errors is ~9), +11 tall, weight half way to the lead foot, no speed lost and a squarer face. The 150
  penalty helped less (−7, about 1.6 standard errors).
- **What the golfer does now** (sole contacts by phase, `balance` model_4191): both feet down at address;
  in the downswing the trail foot leaves the ground (heel 0%, toe 10% grounded); 0.25–0.5 s after impact it
  lands back on the trail foot and kicks the lead foot up (lead heel 28%, toe 20%); from 2 s on it balances
  on the trail leg with the lead foot in the air (12%/8%) while the extrapolated CoM drifts 30 cm off the
  sole: the slow late fall. A real golfer does the reverse, finishing on the lead foot with the trail toe down.

### Round 2: more of the balance reward, with and without the harsh penalty (`balance2`, `balance2_fall150`)

Both from `balance` model_4191, 500 iterations, with the sole-based support test; fall penalty 40 vs 150.

| | `balance` 4191 | `balance2` 4690 (fall 40) | `balance2_fall150` 4690 |
|---|---|---|---|
| Fell within 3 s | 40% | 41% | 42% |
| Standing tall at 3 s | 49% | 53% | 53% |
| Weight toward lead foot at a held finish | 0.50 | 0.44 | 0.46 |
| Clubhead / ball (mph) | 67.7 / 86.2 | 67.7 / 83.0 | 67.1 / 83.1 |
| Smash (10th percentile) | **1.33 (1.17)** | 1.27 (1.09) | 1.28 (1.12) |
| Face-to-path / slice share | +6.4° / 54% | +5.8° / 34% | +5.5° / 31% |

(The share of upright time with the extrapolated CoM off the soles reads 74–77% here; round 1's 67% used the
older support test, so they don't compare.) Training falls sat at 52–54% for both, episode length flat at
177–180 steps; the weight-shift pay crept up (8.4 → 9.4 per second) without the finish moving.

- **The balance reward was a one-time step:** 53% → 40% falls in its first 500 iterations, then flat, with the
  strike slipping again (smash 1.33 → 1.27, ball −3 mph). The harsher penalty adds nothing on top.
- Same lesson as runs 75–78: this swing jumps off the trail foot in the downswing and lands on it, and a
  fine-tune of a converged swing doesn't rebuild that. The next test is the balance reward from the first
  balance stage of a fresh chain, the way the takeaway rule only worked from iteration 0.
- Best checkpoint so far at full speed: `balance` model_4191 (40% falls, 49% tall, 68 mph, smash 1.33).

### A fresh chain with the balance reward (`balfresh`), and two loopholes

Chain A's pre-balance golfer (`ref_full` model_598) through the same two balance stages as `bal_ref1`/`2`
(600 iterations, then 500 with `fall_penalty=40`), with `balance_weight=100 shift_weight=20` from the
first. Same parent, same recipe, one change. 256 swings:

| | `bal_ref1` 1197 | `balfresh1` 1197 | `bal_ref2` 1696 | `balfresh2` 1696 |
|---|---|---|---|---|
| Fell within 3 s | 98% | 100% | 87% | **68%** |
| Standing tall at 3 s | 1% | 0% | 7% | **12%** |
| Median fall after impact (fallers) | – | 1.95 s | 2.23 s | **2.72 s** |
| Extrapolated CoM off the soles after impact | 85% | 51% | 73% | **43%** |
| Clubhead / ball (mph) | 67.8 / 85.5 | 65.2 / 79.0 | 69.0 / 89.6 | 66.3 / 84.5 |
| Smash (10th percentile) | 1.33 (1.19) | 1.28 (1.07) | 1.37 (1.23) | 1.33 (1.13) |

Training falls in stage 2 went 0.98 → 0.90 and episode length 133 → 174 steps (chain A: 0.97 → 0.92,
121 → 151), still moving fast.

Sole contacts (`balfresh2` model_1696): the first second after impact is the best balance yet, both feet
mostly down and the extrapolated CoM off the soles only 13–18% of the time, with the weight moving to 0.79
of the way to the lead foot. After ~1 s it walks toward the target: the held swings end with the trail foot
0.61 m and the lead foot 0.24 m toward the target, and from 2 s both feet are often in the air at once.
Two loopholes in the reward paid for that, both closed after this run:
- the weight shift was measured between the feet's **address** spots, so walking the whole body toward the
  target counted as a shift (finish readings of 1.1–1.6). It is now measured between where the feet are;
- with no sole on the ground the support fell back to all four sole points, so a hop counted as balanced.
  Now no grounded sole counts as at least `balance_airborne` (0.10 m) off.

### The fresh chain with the loopholes closed (`balfix`, `balfix_feet`)

Same start (`ref_full` model_598) and recipe as `balfresh`, with the sole-based support test, the weight
shift measured between where the feet are, and hops counted as unsupported. `balfix_feet` also raises
`feet_slip_weight` 10 → 50. 256 swings at iteration 1696 (`bal_ref2` and `balfresh2` from above):

| | `bal_ref2` (chain A) | `balfresh2` | **`balfix2`** | `balfix_feet2` |
|---|---|---|---|---|
| Fell within 3 s | 87% | 68% | **43%** | 80% |
| Standing tall at 3 s | 7% | 12% | **37%** | 5% |
| Median fall after impact (fallers) | 2.23 s | 2.72 s | 2.75 s | 2.60 s |
| Extrapolated CoM off the soles after impact | 73% | 43% | **30%** | 45% |
| Clubhead / ball (mph) | **69.0 / 89.6** | 66.3 / 84.5 | 67.6 / 81.6 | 67.9 / 86.4 |
| Smash (10th percentile) | **1.37 (1.23)** | 1.33 (1.13) | 1.29 (1.09) | 1.36 (1.22) |
| Face-to-path / slice share | +8.8° / 94% | +9.5° / 92% | +10.1° / 92% | +9.5° / 98% |

(The support test changed between columns, so the CoM row compares only roughly.) After stage 1 (iteration
1197) every chain still falls on every swing; the difference is all in stage 2. `balfix2`'s training falls
went 0.97 → 0.76 over stage 2, 0.95 → 0.76 in its last 100 iterations alone, so it was far from done.

Sole contacts, `balfix2` model_1696: the trail foot still leaves the ground in the downswing, but from
0.25 s to 2 s after impact both feet are down 95–100% of the time with the extrapolated CoM off the soles
0–13% of it, weight 0.58–0.71 of the way to the lead foot: a balanced, weighted-forward finish for two
seconds, the first. After 2 s the feet start to move and the remaining falls happen (median 2.75 s). Held
swings end with the feet 0.24 m (lead) and 0.31 m (trail) from their address spots, stance 0.48 m wide.

- **The balance reward from the first balance stage works:** at iteration 1696, `balfix2` matches what chain A
  reached at 3692 (49% falls, 40% tall) and is still improving steeply. Cost: ~8 mph of ball speed (smash
  1.29 against 1.37) and no change to the slice.
- **Planted feet hurt:** 5× the foot-slide penalty took falls back to 80%. This body balances partly by
  moving its feet; holding them still removes that.

### `balfix` to 3,200 iterations (2026-11-18, `balfix3`–`5`)

`balfix2` model_1696 resumed for three more 500-iteration stages, recipe unchanged (balance 100, shift 20,
fall 40). 256 swings, mean actions, 3 s window (1696 repeated from above):

| | 1696 | **2195** | 2694 | 3193 |
|---|---|---|---|---|
| Fell within 3 s | 43% | **10%** | 12% | 14% |
| Standing tall at 3 s | 37% | **86%** | 84% | 83% |
| Median fall after impact (fallers) | 2.75 s | 2.67 s | 2.37 s | 2.00 s |
| Extrapolated CoM off the soles after impact | 30% | **14%** | 40% | 49% |
| Weight toward lead foot at a held finish | 1.44 | **0.83** | 0.72 | 0.70 |
| Clubhead / ball (mph) | 67.6 / 81.6 | 68.4 / 84.2 | 69.0 / 83.2 | 68.2 / 81.5 |
| Smash (10th percentile) | 1.29 (1.09) | 1.31 (1.10) | 1.30 (1.11) | 1.28 (1.10) |
| Offline (90th percentile) | 1.0° (2.5°) | 1.0° (2.6°) | 1.0° (2.5°) | 1.0° (2.7°) |
| Lead foot moved | 0.42 m | 0.38 m | 0.43 m | 0.42 m |
| Face-to-path / slice share | +10.1° / 92% | +10.0° / 95% | +9.8° / 93% | +10.0° / 93% |

(1696's 1.44 is the projection blowing up when the feet end close together; only 57% of its swings held.)
Training (stochastic actions, mean per 100 iterations): falls 0.68 → 0.19 over `balfix3`, then 0.20–0.29 for
the next 1,000; the extrapolated CoM's mean distance off the soles (every step) 6.0 → 3.2 cm, then back to 6.7;
the finish weight shift 1.28 → 0.84, then 0.68; action std 0.15 → 0.11.

Sole contacts by phase (256 swings each):
- **2195:** the trail foot still leaves the ground in the downswing (heel 0%, toe 14% grounded) and lands within
  0.25 s. From 0.25 s to 2 s after impact both feet stay down (trail toe 70–98%) with the extrapolated CoM off
  the soles 1–3% of the time. In the last 1.3 s the trail toe lifts (19%) and the CoM is off 29%. Held swings
  end with the lead foot 0.34 m from its spot (0.30 m of it toward the ball), the trail foot 0.19 m.
- **3193:** the same through 1 s, then the trail foot comes off the ground (heel 0.55 → 0.08, toe 0.12 → 0.02
  grounded) and it finishes on the lead foot alone, the extrapolated CoM off that one sole 48% of 1–2 s and 90%
  after. That one-legged finish is the loosening in the table.

- **The first full-speed golfer that holds a finish:** at 2195, 90% of swings stay up for 3 s and 86% finish
  standing tall, at 68 mph clubhead and an 84 mph ball, with the weight at 0.83 of the way to the lead foot
  (target 0.85). Chain A's best was 49% falls at 87 mph ball; the slow `takeaway` chain held 75% at 63 mph.
- **Then it plateaus and the finish loosens:** falls and tall finishes hold within noise (±3 points), but the
  trail foot comes off the ground a second after impact, the extrapolated CoM spends 14% → 49% of the finish
  off the soles, the weight drifts back to 0.70 and the fallers go down sooner. The mean-action return falls
  too (110.8 → 107.2): the drift buys nothing the reward can see. A suspect, untested: the action std
  shrinking below 0.12, where run 65's lineage also stalled.
- Video (`media/2026-11-18_stage0p-balance_2026-11-18_01-47-09_balfix5/`, clips at 0, 600, 1200, 1900, 2500, 3193): 1 s after
  impact the last three stand in a high finish, lead leg posted, trail toe down. By the end of the clip 1900
  and 2500 are down and 3193 stands. Each clip is one swing: two falls in three is more than a 10–20% rate
  predicts (a few percent chance), so the single-env Kit replay may differ from the headless evals; unchecked.
- Best checkpoint: **`balfix3` model_2195** (10% falls, 86% tall, 68 mph, ball 84 mph, smash 1.31). The slice
  is untouched (face-to-path +10° as logged; read at impact it is +6.7°, next section).

## Phase 2: ball flight with spin (2026-11-18, `flight/`)

Each strike now gets a launch angle, spin and a flight. Metrics only: no reward uses them yet.

**Data.** TrackMan's 2023 tour averages, PGA and LPGA, every club from driver to PW: club speed, attack
angle, ball speed, smash, launch, spin, apex, landing angle, carry (`flight/data.py`; published as images on
trackman.com/blog/golf/introducing-updated-tour-averages, May 2024).

**Spin.** The axis is perpendicular to the D-plane (the plane of club path and face normal): square to the
path it is pure backspin; a face open to the path tilts it right. The rate is k × club speed × sin(3D spin
loft), the face sliding across the ball, with k per club from both tours (the tours' spin loft recovered from
launch and attack through the env's 85/15 face/path launch blend):

| | Driver | 3-wood | 5-wood | Hybrid | 4i | 5i | 6i | 7i | 8i | 9i | PW |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k [rpm per mph] | 112 | 126 | 168 | 169 | 177 | 181 | 190 | 199 | 209 | 205 | 208 |
| PGA / LPGA spin error | +16 / −14% | −11 / +12% | ±2% | ±4% | ±1% | ±2% | ±2% | ±2% | 0% | ±3% | ±3% |

The driver's low k fits a smooth face with no grooves; its ±15% is the two tours disagreeing (LPGA hits up,
+2.8°, yet spins as much at 19 mph less).

**Flight.** Gravity, drag CD0 + CD1·S, Magnus lift CL_MAX(1 − e^(−S/CL_S)) on the spin ratio S = rω/v, spin
decay τ 25 s (assumed; carry barely depends on it), RK4 at 0.2 s (3 cm from a 0.002 s step). Fitted on the
PGA table, LPGA held out:

| Variant | PGA carry, mean / worst | LPGA carry, mean / worst | LPGA apex RMS | 84 mph test ball |
|---|---|---|---|---|
| **A: 4 coefficients** (kept) | −3.5% / 10.1% | **+0.1% / 4.7%** | 3.5 yd | 94 yd |
| B: + spin decay fitted | −3.3% / 9.9% | +0.2% / 4.7% | 3.3 yd | 94 yd (τ ran to its 100 s bound) |
| C: fitted on both tours | −4.2% / 10.1% | −0.6% / 4.2% | 3.4 yd | 93 yd |
| D: + Reynolds-number drag | −1.4% / 6.1% | −4.1% / 11.4% | 3.0 yd | 78 yd |

CD0 0.261, CD1 0.282, CL_MAX 0.436, CL_S 0.160. D is the textbook overfit: better on the data it saw, worse
on the held-out tour, and the most different at our golfer's speed, which no row covers (the test ball:
84 mph, 12°, 3,000 rpm). A misses the PGA PW (−10%) and driver (−8.5%) and puts wood apexes up to 8 yd high.
Unit tests (`flight/test_flight.py`, stdlib unittest): vacuum projectile, time-step convergence, backspin
lifts, both tour tables at A's accuracy, spin within 20% for every club, open face curves right and
mirrors, more spin loft more spin.

**A bug found on the way: the path was read late.** Near the ball the club turns ~20° per step (30 m/s on a
~1.4 m arc), and the env read the head's velocity at the step's end, about half a step after impact, while
it already interpolated the head's orientation to the moment of impact. `balfix3` model_2195, 256 swings:

| | End of step (until now) | At impact (now) |
|---|---|---|
| Path | −8.0° | −4.8° |
| Attack angle | −0.3° | −9.4° |
| Face-to-path | +10.0° | +6.7° |
| Shot shapes | 95% slice, 5% fade | 70% slice, 30% fade |
| Ball speed (speed into the face) | 84.2 mph | 85.4 mph |

**What the flight says about `balfix3`.** Dynamic loft −5.9° (the face points 16° down from square at impact,
6° down with the driver's loft), attack −9.4°, launch −6.4°, 1,100 rpm with the axis at +65°: every drive
goes into the ground, carry 1 yd. The slice is there but can't show. The same club speed, ball speed, face and
path with a different vertical delivery:

| Delivery | Launch | Spin | Axis | Carry | Lands | Curve |
|---|---|---|---|---|---|---|
| As measured: attack −9.4°, loft −5.9° | −6.4° | 1,000 rpm | +61° | 1 yd | 0 | 0 |
| Attack −1°, loft 13° | 10.9° | 2,050 rpm | +25° | 80 yd | 7 yd right | 6 yd right |
| Same, face square to the path | 10.9° | 1,860 rpm | 0° | 82 yd | 7 yd left (pull) | 0 |
| Attack +3°, loft 16° | 14.1° | 1,930 rpm | +26° | 89 yd | 9 yd right | 8 yd right |

- The flight model reproduces tour carries within 5% on data it never saw, and spin within 20% for every club.
- This golfer chops down on a teed driver with a delofted face. Nothing in its reward has ever cared about
  launch angle, so the first use of the flight is a reward for getting the ball airborne; only then does the
  slice cost anything.
- Training cost: 1.46 s vs 1.38 s per iteration (+6%, while balls land at once; up to ~10% when they fly).

### Paying for distance on the converged swing (`carry2`, `carry2_launch`)

New knobs (off by default): `env.carry_weight` pays per metre of carry along the target line (the computed
flight), split between impact and the finish like the ball pay (half × upright at the end of the 3 s window),
added on top of the ball pay: replacing it would cut nearly all pay at once while every drive carries a yard,
the stand-still trap of run 51. At 2 per metre, an 80 m drive would earn ~160 against ~115 of ball pay.
`env.launch_weight` pays once at impact for the launch angle's share of the way across `launch_range`
(−10° to 12°), a ramp with a slope where carry has almost none (off a 9 cm tee, −6° to 0° adds ~4 m).

Both from `balfix3` model_2195, 500 iterations, recipe unchanged (balance 100, shift 20, fall 40), so
`balfix4` (same parent, same recipe, no carry pay, path read late) is the control. 256 swings:

| | `balfix4` (control) | `carry2` (carry 2/m) | `carry2_launch` (+ launch 50) |
|---|---|---|---|
| Fell within 3 s | 12% | 9% | 15% |
| Standing tall at 3 s | 84% | 84% | 80% |
| Launch / attack / dynamic loft (2195: −6.4° / −9.4° / −5.9°) | – | −7.4° / −10.4° / −6.9° | −4.2° / −8.0° / −3.5° |
| Carry | – | 1 yd | 1 yd |
| Clubhead / ball (mph) | 69.0 / 83.2 | 68.8 / 85.2 | 68.1 / 85.2 |
| Smash | 1.30 | 1.30 | 1.31 |
| Face-to-path / shapes | (read late) | +7.6° / 78% slice | +7.6° / 89% slice |

Training (per 100 iterations): `carry2`'s launch drifted −5.2° → −7.3°; `carry2_launch`'s climbed −5.5° →
−4.2°, about 0.3° per 100 iterations, with carry 0.8 → 1.5 yd. Action std stayed at 0.12 in both.

- **Carry pay alone teaches nothing here:** every ball carries under a yard, so it is a constant.
- **The launch ramp moves it, slowly:** at 0.3° per 100 iterations a 10° launch is ~4,500 iterations away.
  Balance, speed and strike held in both.
- Same lesson as the takeaway rule and the balance reward: the downward chop was set in the first strike
  stages, and a fine-tune of a converged swing with std 0.12 barely explores a different delivery. Next:
  carry (and the launch ramp) from iteration 0 of a fresh chain.

### Carry pay from iteration 0 (`cfresh`, `lfresh`)

`balfix`'s chain from scratch (strike 300 → full 300 → balance 600 + 500 → 500 more, same recipe, std 0.25 from
the first balance stage) with carry pay in every stage: `cfresh` carry 2/m, `lfresh` carry 2/m + launch 50.
256 swings at the end of each balance stage, against `balfix` at the same iterations (its 2195 row re-measured
with the flight model):

| | `balfix` | `cfresh` (carry 2) | `lfresh` (+ launch 50) |
|---|---|---|---|
| Fell within 3 s (1197 / 1696 / 2195) | 100 / 43 / 10% | 100 / 100 / 100% | 100 / 100 / 100% |
| Median fall after impact (2195) | 2.62 s | 0.22 s | 0.18 s |
| Carry (1197 / 1696 / 2195) | – / – / 1 yd | 81 / 86 / 91 yd | 39 / 53 / 57 yd |
| Launch / attack / dynamic loft | −6.4° / −9.4° / −5.9° | +14.2° / +4.8° / +15.9° | +19.3° / +5.8° / +21.6° |
| Clubhead / ball (mph) | 70.6 / 85.4 | 66.1 / 88.1 | 48.2 / 62.5 |
| Smash | 1.31 | 1.35 | 1.34 |
| Weight toward the lead foot at impact | 0.69 | 0.45 | 0.65 |
| Lead hand at impact | 18 mph | 5 mph | 3 mph |
| Offline / side / curve | +1° / 0 / 0 | +6° / +16 yd / +6.5 yd | +2° / +6 yd / +3 yd |
| Face-to-path / shapes | +6.7° / 70% slice | +5.6° / 59% fade, 41% slice | +7.9° / 63% slice |

- **The launch is learned at once.** The first random strikes already launch +12° to +18°; with carry paid,
  the strike stage keeps it and carry grows with speed. Without carry pay, `balfix`'s chain turned the same
  early upward strikes into a −9° chop.
- **How it gets the ball up: the amateur's scoop.** The weight stays back (45% of the way to the lead foot at
  impact, against 69%), the hands stall (lead hand 5 mph against 18) and the club flips past them, adding loft
  (dynamic loft +16° on a 10.5° driver) and hitting up (+5°). Then it falls within 0.2 s of impact, toward the
  trail side, away from the target (12 of 12 viewer swings; `balfix1`'s falls go toward the target).
- **The balance stages never touched it:** episode length stayed at ~32 steps (0.5 s, each episode ends at the
  fall) for 1,600 iterations, where `balfix1`'s grew 62 → 121 as its falls came later. The dive is part of the
  strike, so falling a little later has no gradient. Holding would pay far more (the finish halves of ball and
  carry pay ~140, hold up to 60, no fall penalty), but no small change to this swing gets there: return 114,
  about what `balfix3` earns standing with no carry (111).
- **`lfresh` learned the launch ramp instead of speed:** 19° launch, past the ramp's 12° top, at 48 mph. Its
  full stage, where strikes count only after a 1 m backswing, took ~250 iterations to get back to half contact
  (`cfresh`: ~50), much of it on 8 mph taps launched at ~60°, which the ramp pays as much as a drive.
- Verdict: carry pay teaches the launch, but learned from the strike stage it builds a swing that can't stand.
  The two orderings so far each give half: balance first won't learn to launch, launch first won't learn to
  stand.

### Fine-tunes with more exploration (`ft50`, `ft150`)

From `balfix3` model_2195 with the action std reset 0.12 → 0.25 (`set_action_std.py`), recipe unchanged plus
carry 2/m, launch 50 or 150; 500 + 500 iterations, 256 swings after each. Control: `balfix4` (2694: 12% falls,
84% tall, 69.0 / 83.2 mph, smash 1.30).

| | `ft50a` (2694) | `ft50b` (3193) | `ft150a` (2694) | `ft150b` (3193) |
|---|---|---|---|---|
| Fell within 3 s / standing tall | 16% / 81% | 21% / 77% | 87% / 7% | 42% / 44% |
| Launch / attack / dynamic loft | −3.1° / −6.4° / −2.5° | −3.4° / −6.7° / −2.8° | +13.1° / −0.5° / +15.3° | +13.0° / −1.4° / +15.3° |
| Carry / side | 2 yd / 0 | 1 yd / 0 | 47 yd / +22.5 yd | 51 yd / +25.0 yd |
| Face / face-to-path / offline | – | – | +25.0° / +19.3° / +22° | +24.6° / +19.7° / +22° |
| Clubhead / ball (mph) | 67.5 / 80.8 | 65.6 / 78.6 | 61.1 / 65.2 | 61.7 / 68.6 |
| Smash | 1.28 | 1.27 | 1.15 | 1.19 |

- **std 0.25 at launch 50 bought ~1°:** −3.1° after 500 iterations against `carry2_launch`'s −4.2° at std 0.12,
  then flat (−3.6° to −3.7° for the whole second stage). It plateaus with every ball still carrying 1–2 yd.
- **Launch 150 got the ball up in ~400 iterations, by opening the face 25°.** With the shaft leaning forward at
  impact, turning the face open about the shaft tilts it up: dynamic loft −6° → +15° (a flop shot's trick).
  Every shot starts 22° right and slices (axis +52°, 3,000 rpm); carry along the target line ~44 yd. The launch
  ramp pays the same whichever way the ball starts, and at 150 it outweighs what `aim_sigma` 15 takes off
  the ball pay (logged `Reward/ball_speed` 21 → 3). Balance collapsed (87% falls) and was coming back (42%,
  still falling ~10 points per 100 iterations at the end), clubhead −6 mph.
- `balfix`'s own pre-balance checkpoint (`ref_full` model_598) already launches at −17.5° (face 19° closed,
  attack −10.7°, dynamic loft −18.7°, 80% hooks), so carry pay from the first balance stage would start where
  the fine-tunes did, with no carry to climb. Not run.
- Next: ramps the face can't game: launch pay × the aim factor (`env.launch_aim`), or pay on the attack angle
  (`env.attack_weight`, the club's path into the ball, which turning the face doesn't change).

### Ramps the face can't buy (`fta150`, `att150`)

New knobs (off by default): `env.launch_aim` multiplies the launch pay by the same aim factor as the ball pay
(exp(−(offline/`aim_sigma`)²)); `env.attack_weight` pays once at impact for the attack angle's share of
`attack_range` (−10° to +2°). Same start and recipe as `ft150` (std 0.25 from `balfix3` model_2195, carry 2),
weight 150, 500 + 500 iterations:

| | `fta150a` (2694) | `fta150b` (3193) | `att150a` (2694) | `att150b` (3193) |
|---|---|---|---|---|
| Fell within 3 s / standing tall | 41% / 47% | 24% / 70% | 74% / 12% | 34% / 56% |
| Launch / attack / dynamic loft | +2.0° / −4.1° / +3.1° | +2.4° / −4.4° / +3.6° | +11.4° / +3.3° / +12.7° | +9.8° / +2.8° / +11.0° |
| Carry / side | 13 yd / +1.4 yd | 13 yd / +1.3 yd | 41 yd / +16.8 yd | 40 yd / +13.9 yd |
| Face / face-to-path / offline | +5.3° / +7.7° / +4° | +5.1° / +7.6° / +4° | +20.9° / +15.7° / +19° | +17.8° / +13.4° / +16° |
| Clubhead / ball (mph) | 62.8 / 73.1 | 61.7 / 68.6 | 62.9 / 67.3 | 63.9 / 69.8 |
| Smash | 1.22 | 1.16 | 1.15 | 1.17 |

- **Launch × aim stays on line but stops low:** launch −6.4° → +1.6° in 400 iterations, then flat at ~+2°
  (carry 11–14 yd) for the rest; the face opened only 5° (offline +4°). Balance came back to 24% falls.
- **The attack ramp got what it asked for, and the face opened with it.** Attack −9.4° → +2.2° in ~300
  iterations, the top of the ramp, and dynamic loft and launch came along (+11°, +10°): carry 40 yd. But in
  training the attack and the face moved together (iterations 2200 → 2300: attack −6.6° → −1.5°, face +7° → +16°),
  so on this body the move that swings up also opens the face. 16° right, slicing, side +14 yd; balance
  recovering (74% → 34% falls), the face closing ~3° per 500 iterations.
- Every route up on the converged swing so far costs ~7 mph of clubhead and ~0.15 of smash. Smash is strike
  location only: the misses spread from 0.8 cm off the sweet spot (`balfix3`) to 1.8–1.9 cm (`fta150b`,
  `att150b`). A swing being reshaped hits the centre less often.

### Attack pay × aim (`atta150`, `att150c`–`d`)

`env.attack_aim` multiplies the attack pay by the aim factor (as `launch_aim` does the launch pay). Two arms,
500 + 500 iterations each, attack 150 × aim, carry 2: `atta150` from the same start (`balfix3` at std 0.25), and
`att150c`–`d` continuing `att150b` (open face, 16° right), to see whether the face closes and the launch stays.

| | `atta150a` (2694) | `atta150b` (3193) | `att150c` (3692) | `att150d` (4191) |
|---|---|---|---|---|
| Fell within 3 s / standing tall | 52% / 44% | 25% / 72% | 32% / 64% | 17% / 82% |
| Launch / attack / dynamic loft | +0.3° / −1.7° / +0.6° | +0.5° / −0.8° / +0.7° | +3.2° / +0.6° / +3.7° | +2.3° / +1.1° / +2.5° |
| Carry / side | 5 yd / +0.4 yd | 5 yd / +0.3 yd | 13 yd / +1.5 yd | 9 yd / +0.8 yd |
| Face / face-to-path / offline | +3.7° / +5.6° / +3° | +2.6° / +4.1° / +3° | +5.3° / +5.0° / +5° | +3.9° / +4.6° / +3° |
| Clubhead / ball (mph) | 67.2 / 68.3 | 66.6 / 66.3 | 64.2 / 66.8 | 63.0 / 64.0 |
| Smash / sweet-spot miss | 1.12 / 2.4 cm | 1.10 / 2.5 cm | 1.13 / 2.0 cm | 1.13 / 2.1 cm |

- **A square face and a lofted face, never both.** With the aim factor the attack still climbs to level
  (−9.4° → −0.8°) and the face stays square (offline +3°), but the dynamic loft stays near 0° on a 10.5° driver:
  launch +0.5°, carry 5 yd. Continuing the open-face swing, the face closed 18° → 4° (offline 16° → 3°) with the
  attack held at +1°, and the loft went with it (11° → 2.5°): carry 40 → 9 yd.
- So the missing loft isn't the attack: with a square face and a level attack, the face still points ~10° lower
  than the club's loft, which reads as the shaft leaning toward the target at impact (hands ahead of the head;
  inferred from dynamic loft, not measured). Opening the face was this body's only way found to add it back
  (it has no forearm roll).
- Balance recovered in both (17–25% falls, 72–82% tall); clubhead −5 to −8 mph and smash 1.31 → 1.10–1.13,
  the worst strikes of any balanced golfer here.
- Verdict for the fine-tunes: every pay that moves the launch (launch ramp, attack ramp, with or without aim)
  either opens the face or leaves the loft behind. A real driver swing gets its loft from the ball position:
  teed off the lead heel, the club meets it after the low point (hitting up) with the shaft already released.
  This address has the ball 6 cm ahead of the stance centre, a mid-iron position. Next: `env.ball_forward`.

### Ball off the lead heel (`fwd`, `cfwd`)

New knob: `env.ball_forward` [m] moves the ball (and tee) toward the target from the solved address spot. At
0.1 the ball sits ~16 cm ahead of the stance centre, at the lead heel (the lead foot is at ~+20 cm), a driver
position. Two fresh chains on `balfix`'s schedule with it in every stage: `fwd` (`balfix`'s recipe) and `cfwd`
(+ carry 2/m, `cfresh`'s). 256 swings at the end of each balance stage, at 0.1 (iteration 2195 unless noted):

| | `balfix` | `fwd` | `cfresh` | `cfwd` |
|---|---|---|---|---|
| Fell within 3 s (1197 / 1696 / 2195) | 100 / 43 / 10% | 69 / 39 / 35% | 100 / 100 / 100% | 100 / 100 / 100% |
| Median fall after impact / standing tall | 2.62 s / 88% | 1.27 s / 31% | 0.22 s / 0% | 0.73 s / 0% |
| Carry | 1 yd | 1 yd | 91 yd | 125 yd (110 / 118 / 125) |
| Launch / attack / dynamic loft | −6.4° / −9.4° / −5.9° | −7.5° / +3.7° / −9.4° | +14.2° / +4.8° / +15.9° | +17.4° / +7.7° / +19.1° |
| Clubhead / ball (mph) | 70.6 / 85.4 | 74.1 / 67.1 | 66.1 / 88.1 | 71.0 / 97.8 |
| Smash / sweet-spot miss | 1.31 / 0.8 cm | 1.17 / 1.4 cm | 1.35 / 0.6 cm | 1.40 / 0.4 cm |
| Weight toward the lead foot at impact | 0.69 | 0.55 | 0.45 | 0.36 |
| Lead hand at impact | 18 mph | 16 mph | 5 mph | 5 mph |
| Face / path / face-to-path | +1.9° / −4.8° / +6.7° | −7.1° / +6.9° / −14.0° | +7.1° / +1.5° / +5.6° | +3.3° / +0.5° / +2.8° |
| Shapes / offline / side | 70% slice / +1° / 0 | 100% hook / −5° / 0 | 59% fade / +6° / +16 yd | 83% fade / +3° / +10 yd |
| Torso's move at the fall or the end (12 viewer swings) | toward the target 58% | away from it 82% | backward 83% | backward 92% |

- **The geometry moved the attack as intended:** −9.4° → +3.7° without carry pay, +4.8° → +7.7° with it. Met
  off the lead heel, the club is past its low point and rising.
- **Without carry pay the loft still goes.** `fwd`'s full stage (iteration 598) launched at −0.9° (attack +6°,
  carry 8 yd), nearly level, and the balance stages delofted it to −9.4° with nothing paying for loft. The face
  closed to an in-to-out path: every shot hooks. It balances worse than `balfix` (35% falls against 10%), its
  weight hangs back, its extrapolated CoM is off its soles for its whole finish (training's balance penalty
  −15 against −3), and the strike slipped stage by stage (smash 1.26 → 1.17, ball 77 → 67 mph). On its own the
  ball position only changed the attack.
- **With carry pay: the best flight so far,** 125 yd on a 98 mph ball, smash 1.40 from centred strikes,
  3° offline, a fade. It is still the scoop (weight 0.36 of the way forward, lead hand 5 mph, 19° of dynamic
  loft on a 10.5° driver; a real driver launches 12–15° from +3–5° of attack), so it still falls back on every
  swing.
- **But it falls later, and later each stage:** median 0.47 → 0.65 → 0.73 s after impact, episode length 48 → 60
  → 66 steps (`cfresh` sat at ~32 for 1,600 iterations). With room between strike and fall, the balance stages
  have a gradient to climb. Next: continue `cfwd` 1,500 iterations, as is and with the std reset to 0.25.

### `cfwd` past 2,200, and pay for weight forward at impact (`cfwds`, `ishift`)

From `cfwd3` model_2195, 500-iteration stages on the same recipe (ball 0.1, carry 2, balance 100, shift 20,
fall 40), 256 swings after each. `cfwd4`: as is (stopped after one stage, flat). `cfwds`: action std reset
0.12 → 0.25. `ishift` (from `cfwds4`): plus new knob `env.impact_shift_weight` = 100, paid once at impact for
the weight's progress toward `shift_target` (the shift reward's measure; 0.37 of the way earns ~44, 0.69 ~81).

| Run (iteration) | Fell within 3 s | Median fall after impact | Weight toward the lead foot at impact | Carry | Launch | Ball |
|---|---|---|---|---|---|---|
| `cfwd3` (2195) | 100% | 0.73 s | 0.36 | 125 yd | 17.4° | 97.8 mph |
| `cfwd4` (2694) | 100% | 0.77 s | 0.37 | 127 yd | 17.6° | 98.0 mph |
| `cfwds4` (2694) | 100% | 0.82 s | 0.38 | 125 yd | 18.7° | 96.0 mph |
| `cfwds5` (3193) | 100% | 0.90 s | 0.40 | 130 yd | 18.2° | 98.4 mph |
| `cfwds6` (3692) | 100% | 0.95 s | 0.40 | 128 yd | 18.2° | 97.7 mph |
| `ishift5` (3193) | 100% | 0.80 s | 0.45 | 127 yd | 18.0° | 98.0 mph |
| `ishift6` (3692) | 100% | 0.90 s | 0.47 | 126 yd | 17.5° | 98.3 mph |

- The fall creeps later, ~0.06 s per 500 iterations with std 0.25 and less without; nowhere near a 3 s hold.
  Episode length 66 → 73 steps. Everything else stays put: the scoop (lead hand 4–5 mph), 18° launch, ~127 yd.
- Weight forward at impact paid for itself in weight (0.40 → 0.47 against `cfwds`) at no cost in carry, but the
  fall didn't come later. Not enough at 100; a scooping strike with its weight a little further forward still
  dives.
- Why not a harsher fall penalty: every swing falls, so the penalty is the same on all of them and ranks
  nothing (the threshold rule in CLAUDE.md), and a fall already costs ~240 (penalty 40, the finish halves of
  ball and carry ~140, hold up to 60). `fall_penalty=150` on `bal_ref6` was within noise.

### The balanced golfer with the ball forward (`bfwd`, `bfwd15`)

From `balfix3` model_2195 (10% falls, launch −6.4°) at std 0.25, `balfix`'s balance recipe plus carry 2/m, with
the ball moved forward 0.1 m (`bfwd`) or 0.15 m (`bfwd15`, in line with the lead foot); 500 + 500 iterations.

| | `balfix3` (2195) | `bfwd1` (2694) | `bfwd2` (3193) | `bfwd15a` (2694) | `bfwd15b` (3193) |
|---|---|---|---|---|---|
| Fell within 3 s / standing tall | 10% / 86% | 23% / 68% | 23% / 70% | 33% / 60% | 45% / 48% |
| Launch / attack / dynamic loft | −6.4° / −9.4° / −5.9° | +4.2° / −1.7° / +5.2° | +4.0° / −1.6° / +4.9° | +6.7° / +0.1° / +7.8° | +6.5° / +0.2° / +7.6° |
| Carry / apex | 1 yd | 26 yd / 0.6 yd | 24 yd / 0.6 yd | 37 yd / 1.3 yd | 37 yd / 1.2 yd |
| Clubhead / ball (mph) | 70.6 / 85.4 | 65.3 / 79.5 | 65.6 / 78.8 | 65.8 / 77.2 | 65.4 / 77.9 |
| Weight toward the lead foot at impact / lead hand | 0.69 / 18 mph | 0.63 / 16 mph | 0.62 / 16 mph | 0.63 / 16 mph | 0.62 / 15 mph |
| Face-to-path / shapes | +6.7° / 70% slice | +8.2° / 96% slice | +8.1° / 96% slice | +7.5° / 91% slice | +7.4° / 88% slice |

- **The ball position bought the launch the reward ramps couldn't, without opening the face:** +10° in ~200
  iterations at 0.1, +13° at 0.15, offline 1–3°. It keeps a real weight transfer (0.62 of the way forward at
  impact, hands at 16 mph), the first golfer here that gets the ball in the air and holds its finish.
- **Then it stops:** 4° at 0.1 and 6.5° at 0.15, flat for the second 500 iterations, a low slice (apex ~1 yd).
  The flip is this body's only loft move left, and the flip is what makes `cfwd` fall. Each 5 cm costs balance
  (23% → 45% falls at 0.15, still rising) and ~5 mph of clubhead against `balfix3`.
- Next: a stronger lead wrist (`env.wrist_gear`, 10 N·m by default), so the club can be squared by the wrist
  instead of by stalling the hands.

### A stronger lead wrist (`wgear30`, `cwfwd`)

New knob: `env.wrist_gear` [N·m], the lead wrist's strength (10 by default, `WRIST_GEAR`; arms 67.5), replayed by
`archive_run.py`. `wgear30`: from `bfwd2` model_3193 at 30 N·m, same recipe (ball 0.1, carry 2), 500 + 500
iterations; control `bfwd3`–`4`: the same at 10.

| | `bfwd3` (10 N·m, 3692) | `bfwd4` (10 N·m, 4191) | `wgear30a` (30 N·m, 3692) | `wgear30b` (30 N·m, 4191) |
|---|---|---|---|---|
| Fell within 3 s / standing tall | 25% / 73% | 19% / 78% | 15% / 82% | 27% / 71% |
| Launch / attack / dynamic loft | +3.1° / −3.0° / +4.2° | +1.6° / −4.5° / +2.7° | +6.3° / −1.6° / +7.6° | +6.9° / −1.1° / +8.3° |
| Carry | 20 yd | 11 yd | 39 yd | 41 yd |
| Clubhead / ball (mph) | 64.9 / 77.1 | 65.1 / 72.8 | 66.8 / 77.3 | 65.4 / 77.4 |
| Face-to-path / shapes | +8.1° / 91% slice | +7.9° / 82% slice | +7.1° / 76% slice | +5.3° / 73% fade |

- With the wrist at 30 N·m the balanced golfer adds ~3° of dynamic loft and keeps it, carry 24 → 41 yd, at the same
  balance; at 10 N·m the same training lost it (launch 4.0° → 1.6°, the hands drifting ahead again). The wrist is
  a real lever on loft, a modest one on a swing already learned.
- **From scratch at 30 N·m (`cwfwd`, `fresh_chain.sh` with ball 0.1, carry 2): no falls (5%, 0%, 1% at 1197 /
  1696 / 2195), but no golf swing either.** A short jab (backswing 1.21 m against 1.76, impact 0.20 s after
  address, clubhead 56 mph) with the face 29° open: launch 20°, carry 74 yd, every shot starting 29° right
  (side +38 yd), crouched finish (torso 0.81 m, 1% tall), weight 0.81 of the way forward.
- The loophole: `aim_sigma` scales the ball pay (×0.03 at 29°), but carry pay counts carry along the target line,
  and a 29° push keeps ~88% of it. Every carry-paid chain starts 30–40° right in its strike stage (no aim there);
  with the weak wrist the balance stages' aim pulled `cfwd` back to +3°, with the strong one an open face buys
  more loft than the ball pay loses. New knob `env.carry_aim`: carry pay × the aim factor.
- Next: fresh chains with ball and carry pay × aim from iteration 0 (`aim_sigma=15` in the strike stages too),
  wrist 30 (`cwfwda`) and 10 (`cfwda`).


### Ball and carry pay × aim from iteration 0 (`cwfwda`, `cfwda`)

`fresh_chain.sh` with ball 0.1, carry 2/m, `env.carry_aim=True` in every stage and `aim_sigma=15` in the strike
stages too (the balance stages always had it), lead wrist 30 N·m (`cwfwda`) or 10 (`cfwda`). 256 swings at the
end of each balance stage (iteration 2195 unless noted), against the same chains without aim:

| | `cwfwd` (30 N·m) | `cwfwda` (30 N·m, aim) | `cfwd` (10 N·m) | `cfwda` (10 N·m, aim) |
|---|---|---|---|---|
| Fell within 3 s (1197 / 1696 / 2195) | 5 / 0 / 1% | 100 / 100 / 100% | 100 / 100 / 100% | 100 / 100 / 100% |
| Median fall after impact | – | 0.80 / 1.05 / 1.05 s | 0.47 / 0.65 / 0.73 s | 0.13 / 0.13 / 0.13 s |
| Carry / side | 74 yd / +38 yd | 118 yd / +2 yd | 125 yd / +10 yd | 112 yd / +14 yd |
| Launch / attack / dynamic loft | +20.2° / +12.1° / +21.6° | +18.1° / +8.9° / +19.7° | +17.4° / +7.7° / +19.1° | +15.0° / +7.2° / +16.4° |
| Offline / face-to-path / shapes | +29° / +3.8° / 91% fade | +2° / −1.1° / 78% straight | +3° / +2.8° / 83% fade | +1° / +7.4° / 94% slice |
| Clubhead / ball (mph) | 55.9 / 72.0 | 68.8 / 94.9 | 71.0 / 97.8 | 77.9 / 98.6 |
| Smash | 1.32 | 1.40 | 1.40 | 1.32 |
| Weight toward the lead foot at impact / lead hand | 0.81 / 6 mph | 0.45 / 12 mph | 0.36 / 5 mph | 0.45 / 10 mph |
| Backswing / impact after address | 1.21 m / 0.20 s | 1.97 m / 0.32 s | 1.76 m / 0.32 s | 1.83 m / 0.35 s |

- **Aim closed the push without costing the launch.** In the strike stage the aimed chains launched at 4° and
  −2° (13 mph taps, iteration ~200), but the launch came with the full swing: 14.5° by the end of the full
  stage (iteration 598), 18° by 2195, on line (offline +2°). The 29° push had only been buying loft for the short jab.
- **It is the scoop again, and every swing falls.** Weight 0.45 of the way forward at impact, lead hand 10–12 mph
  (balanced golfers 16–18, `cfwd` 5), 16–20° of dynamic loft on a 10.5° driver. `cwfwda` falls ~1 s after
  impact, backward or away from the target (7 of 12 viewer swings each way), its episode length growing
  54 → 83 steps by iteration ~1,700 and then flat. `cfwda` dives within 0.13 s toward the trail side (12 of 12),
  its feet already 0.6 m from their address spots at impact, episode length ~30 steps throughout (`cfresh`'s pattern).
- The strong wrist separates a late fall from a dive, and a straight shot (78% straight) from a slice (94% at
  10 N·m). The 10 N·m chain swings faster (78 vs 69 mph clubhead), but 29% of its strikes register on the shaft.
- **In total the balanced golfer earns more, but PPO doesn't maximize the total.** Mean episode return in training at
  the end of the last stage: 132 (`cwfwda3`) and 104 (`cfwda3`), against 165 for the balanced `wgear30b` on the
  same balance terms (its carry pay isn't aim-scaled; at 1° offline the aim factor is ~1). That sum is
  undiscounted; the policy maximizes the discounted one (next section): seen from the strike, the scoop's carry
  comes at impact and its fall costs ~half (1 s later), while the balanced golfer's extra pay (finish halves, hold)
  comes 1–3 s later at 0.16–0.55 of face value. A rough tally from the logged numbers puts the scoop ahead at the
  strike, ~165 against ~136. Learned in the strike and full stages, it then sits where every swing falls, so only the
  hold pay ranks the falls (the threshold lesson). (`record_swings.py`'s "return" uses the default reward, not
  the training one, so it can't compare these.)
- Next: `env.carry_frac` (new, the share of carry pay held for the upright finish; below 0 it follows
  `finish_frac`) at 1.0, so a swing that doesn't finish upright earns no carry at all. On the balanced golfer with
  carry doubled (`wcf4`, from `wgear30b` at std 0.25; control `wca4`, the same at the default split, where the
  half paid at impact pays a falling scoop as well as a stand), and from iteration 0
  (`cwfwdf`, with the 3 s finish window from the strike stage, since the finish is where the carry is paid).

### Carry held for the finish (`wcf4`, `wca4`, `wcf14`)

From `wgear30b` model_4191 with the action std reset 0.21 → 0.25, its recipe (ball 0.1, wrist 30, balance 100,
shift 20, fall 40) with carry 4/m × aim, 500 + 500 iterations: `wca4` at the default split (half at impact, half
× upright at the end of the 3 s window), `wcf4` with `env.carry_frac=1.0` (all of it at the finish).

| | `wgear30b` (4191) | `wca4a` (4690) | `wca4b` (5189) | `wcf4a` (4690) | `wcf4b` (5189) |
|---|---|---|---|---|---|
| Fell within 3 s / standing tall | 27% / 71% | 29% / 66% | 27% / 71% | 15% / 82% | 17% / 79% |
| Launch / attack / dynamic loft | +6.9° / −1.1° / +8.3° | +8.4° / −0.7° / +10.0° | +8.6° / −0.8° / +10.2° | +4.7° / −2.5° / +6.0° | +3.7° / −3.4° / +4.9° |
| Carry / apex | 41 yd / 1.5 yd | 48 yd / 2.2 yd | 47 yd / 2.1 yd | 30 yd / 0.8 yd | 23 yd / 0.5 yd |
| Clubhead / ball (mph) | 65.3 / 77.4 | 64.6 / 76.5 | 63.9 / 74.9 | 65.8 / 77.0 | 65.7 / 75.3 |
| Face-to-path / shapes | +5.3° / 73% fade | +6.5° / 63% slice | +6.2° / 57% slice | +4.7° / 84% fade | +4.5° / 81% fade |
| Weight toward the lead foot at impact / held finish | 0.63 / 0.52 | 0.63 / 0.56 | 0.64 / 0.54 | 0.63 / 0.55 | 0.64 / 0.53 |

- **Doubling carry at the default split bought ~1.5° and 6 yd at the same balance.** No slide into the scoop
  (weight 0.63 forward at impact, lead hand 12–13 mph), flat in the second stage.
- **Held for the finish, the balance improved and the loft went:** falls 27% → 15–17%, launch 6.9° → 3.7°, still
  falling in the second stage.
- **First guess, the discount.** PPO maximizes the return discounted by γ = 0.99 per 1/60 s step, an effective
  horizon of ~100 steps (1.7 s): pay 1 s after the strike is worth 0.99^60 = 0.55 of its face value at the strike,
  3 s after it 0.99^180 = 0.16. Held for the end of the 3 s window, carry 4 is worth 4 × 0.16 = 0.66 per metre to
  the strike, against 4 × (0.5 + 0.5 × 0.16) = 2.3 at the default split.
- **Scaling the weight for it changed nothing** (`wcf14`, carry 14 held for the finish, 14 × 0.16 = 2.3, `wca4`'s):
  15% / 12% falls, launch +5.0° / +3.7°, carry 30 / 24 yd at 4690 / 5189, the same as `wcf4`. Finish-time carry pay
  doesn't steer the strike at 4 or at 14; both behave like balance stages with no carry pay at all (balance
  better, the loft drifting down, as `bfwd3`–`4` and `fwd` did).
- **Second guess, credit assignment.** GAE (λ 0.95) passes a reward straight back to the actions before it only over
  ~17 steps ((γλ)^17 ≈ 0.35); anything later reaches the strike only through the critic, if the critic can predict
  it from the state after impact. Here it mostly can't: the observation holds the ball's position, but the sim
  ball flies on the flat collider face, not the computed flight, so nothing in it tells a 41 yd launch from a
  23 yd one. Carry paid at impact lands on the strike's own step and needs no critic. New knob `env.flight_obs`:
  the computed carry, side and launch (zero until impact) as 3 more observations; `scripts/widen_obs.py` adds
  zero-weight inputs to a checkpoint so a run can resume with them (policy and value unchanged at first).
- Next: `wcf14` again with `env.flight_obs` (`wcf14o`, from the same widened start). Running meanwhile: the fresh
  chain `cwfwdf8` (carry 8 held for the finish from iteration 0; the carry-2 version, `cwfwdf`, was stopped at
  iteration 15) and `cwfwda`'s chain with `agent.algorithm.gamma=0.998` (`cwfwdg`; 3 s worth 0.70, the fall a
  second after impact at 0.89 of its cost; the fall and the hold pay depend on posture, which the critic can see).

### Carry held for the finish from scratch, and a longer horizon (`cwfwdf8`, `cwfwdg`)

Both `cwfwda`'s chain (ball 0.1, wrist 30, carry × aim, aim 15 from iteration 0), 256 swings at 2195:

| | `cwfwda` | `cwfwdf8` (carry 8, all at the finish, 3 s window from the strike stage) | `cwfwdg` (γ 0.998) |
|---|---|---|---|
| Contact / fell within 3 s | 100% / 100% | 100% / 100% | 99% / 100% |
| Median fall after impact | 1.05 s | 1.62 s | 1.17 s |
| Backswing / top / impact after address | 1.97 m / 0.18 s / 0.32 s | 1.31 m / 0.15 s / 0.23 s | 1.14 m / 0.20 s / 0.38 s |
| Clubhead at impact / ball (mph) | 68.8 / 94.9 | 60.4 / 80.9 | 5.4 / 3.9 |
| Launch / carry | 18.1° / 118 yd | 6.4° / 31 yd | – / 0 yd |
| Weight toward the lead foot at impact | 0.45 | 0.70 | 0.56 |

- **`cwfwdf8`: no scoop, no launch, no balance.** Weight forward at impact (0.70), on line (96% straight), but a short
  1.3 m jab launched at 6° and every swing still falls, later (1.6 s). Carry paid only at the end of the window
  taught neither the loft nor the stand, as `wcf4`/`wcf14` predicted.
- **`cwfwdg`: γ 0.998 from iteration 0 never learned to strike.** Contact 0% at iterations 100 and 200 of the strike
  stage (γ 0.99: 100% by 214), and the swing that came later is an 11 mph tap. Over a 5 s episode a ~8 s horizon
  makes the per-step costs (energy, actions, trail hand) weigh as much as the strike, and the returns the critic
  must predict are noisier. A longer horizon belongs in a fine-tune of a swing that exists, not in its first stage.

## The 7-iron off the ground (2026-11-19, `i7`, `i7v`, `i7s`, `i7n`)

The driver's launch fight was this body's geometry (the face meets the ball ~16° below its loft, hands ahead), and
`balfix3`'s strike (descending, hands leading, weight forward) is an iron strike. So: the 7-iron, the stock swing,
with the ball on the ground (Stage 4.5). New in the env (CLAUDE.md, Stage 4.5): the ball rests at its radius, the
head never collides with the ground and the turf is analytic (head path sampled 8 times a step; fat-shot ball-speed
loss `env.turf_area`, divot cost `env.dig_weight`, strike classes after Corke et al. 2021). New 7-iron asset (head
collider off, 62% of the mass in the head) and address (sole on the turf, zero overlaps). Lead wrist 10 N·m.

| | `i7` | `i7v` | `i7s` | `i7n` |
|---|---|---|---|---|
| First stage (ball speed + backswing) | + carry × aim from iteration 0, 300 | 300 | 600 | `i7s`'s 600 |
| Contact at its end | 30% | 32% | 91% | 91% |
| Full-swing stage (1 m backswing) | + carry: 0–3% | + carry: 0–7% | + carry: 29% → 94% | no carry: 26% → 0% |
| Balance stages | stopped | 0% contact | 100% contact, 100% falls | 0% contact |

Three false starts before these, all at 0% contact through iteration 200: the dig cost from iteration 0 (the
untrained golfer drags the club through the ground, so it learned to hold the club in the air, 1.45 m from the
ball); the fat-shot loss from iteration 0 (every early strike was fat and paid ~0.2%); and aim 15 from iteration 0
(the 7-iron's random hits fly 14–75° off line, so aim took their pay; it was climbing again by 300).

`i7s`, 256 swings after each balance stage (evals without the fat-shot loss, which training applied: it kept
97–98% of ball speed through the turf on average):

| | 1496 | 1995 | 2494 |
|---|---|---|---|
| Fell within 3 s / median fall after impact | 100% / 0.25 s | 100% / 0.33 s | 100% / 0.35 s |
| Carry / apex | 59 yd / 6.9 yd | 71 yd / 8.9 yd | 74 yd / 9.3 yd |
| Launch / spin / attack / dynamic loft | 20.3° / 4,657 rpm / +1.8° / 23.4° | 20.2° / 5,036 / −0.1° / 23.7° | 19.8° / 5,163 / −1.4° / 23.4° |
| Clubhead / ball (mph) | 54.9 / 61.6 | 57.8 / 67.6 | 58.7 / 68.8 |
| Strikes good / fat / thin / top | 66 / 18 / 16 / 0% | 59 / 38 / 3 / 0% | 65 / 30 / 4 / 1% |
| Offline / face-to-path / shapes | 3° / −13.4° / 100% hook | 2° / −11.0° / 99% hook | 1° / −9.0° / 93% hook |
| Weight toward the lead foot at impact / lead hand | 0.77 / 11 mph | 0.80 / 13 mph | 0.80 / 13 mph |
| Backswing / top / downswing | 1.41 m / 0.25 / 0.10 s | 1.49 m / 0.26 / 0.11 s | 1.50 m / 0.26 / 0.10 s |

- **The ground model works.** The strikes come off the turf like an iron's: 65% good, 30% fat, 4% thin, 1% top
  (Corke's 96 golfers: 63 / 32 / 5), the leading edge 2–7 mm above the ground at impact, the attack turning
  slightly descending, and almost no gouging (dig cost ~0 by the end of the first balance stage).
- **The head-heavy 7-iron learns to strike slower:** 600 iterations for 91% contact (driver ~250). Its moment of
  inertia about the grip is 0.14 kg·m² (driver 0.11, the old uniform-density 7-iron 0.075), so the same muscles
  swing it slower and its strikes pay less. Leaving the first stage at ~30%, both 300-iteration chains collapsed when
  strikes started to need a 1 m backswing.
- **Carry pay is what carries the 7-iron through the full-swing stage.** From the same 91% start, with carry pay
  contact recovered to 94%; without it, it fell to 0 in 50 iterations and never came back. Ball pay alone at
  7-iron speeds (smash ceiling 1.33, ~55 mph) isn't worth the 1 m backswing.
- **With carry: an iron strike, a falling golfer.** Weight 0.8 of the way forward at impact, a level-to-descending
  blow, launch 20° on a 34° club, 74 yd, 1° offline, tempo 2.5:1 (the closest to a golfer's 3:1 yet). But it drops
  onto the trail knee (103° at impact) and falls away from the target ~0.35 s after impact on every swing (11 of
  12 viewer swings), episode length flat at ~43 steps through 1,600 balance iterations: `cfresh`'s pattern, the
  stand the balance stages can't climb to. Path 9° in-to-out with a square face: every shot hooks.

## Balance before carry on the 7-iron, and paying for the backswing (2026-11-20, `i7b`, `i7bk`)

Two questions: was carry pay what built `i7s`'s fall into its swing, as with the driver (`cfresh` against
`balfix`)? And does a longer backswing pay on this body when it's asked for?

Both start from `i7s_strike` model_598 (600 iterations of ball pay, 91% contact) and follow the driver's `balfix`
order with no carry pay anywhere: full-swing stage 300 (no aim), balance stages 600 + 500 (`balance_weight=100
shift_weight=20`, aim 15, fat-shot loss, dig cost), then 500 more with fall 40. Ball pay is doubled
(`ball_speed_weight=6`) throughout in place of carry's (`i7n`, with neither, stopped striking in the full-swing stage).
- **`i7b`:** as above.
- **`i7bk`:** plus `backswing_weight=50 backswing_cap=1.8` and `takeaway_weight=10`. The backswing pay is one-off
  per metre of the farthest clubhead-ball distance, so up to 90 against ~180 of ball pay; at the default 0.6 it is
  worth under a point.

Fixed first: the backswing pay kept paying after impact, because the struck sim ball flies away from the head.
At the default weight that was under a point, at 50 it would have been a strike bonus. It now pays before impact only.

Both kept striking through the full-swing stage (77–100% contact by its 100th iteration): doubled ball pay does
what carry pay did there.

256 swings after the second balance stage (iteration 1995), against `i7s` at the same point:

| | `i7s` (carry) | `i7b` | `i7bk` (+ backswing pay) |
|---|---|---|---|
| Fell within 3 s / median fall after impact | 100% / 0.33 s | 100% / 0.53 s | 100% / 0.40 s |
| Backswing / top / downswing | 1.49 m / 0.26 / 0.11 s | 1.54 m / 0.23 / 0.13 s | **1.82 m / 0.31 / 0.14 s** |
| Clubhead / ball (mph), smash | 57.8 / 67.6, 1.23 | 58.1 / 65.5, 1.24 | **66.5 / 74.4**, 1.19 |
| Carry / launch / attack | 71 yd / 20.2° / −0.1° | 50 yd / 11.0° / −7.3° | 72 yd / 13.2° / −7.3° |
| Strikes good / fat / thin | 59 / 38 / 3% | 65 / 35 / 0% | 75 / 24 / 2% |
| Weight toward the lead foot at impact | 0.80 | 0.87 | 0.85 |
| Lead foot moved | 1.11 m | 0.74 m | 0.38 m |
| Shapes | 99% hook | 99% hook | 100% hook |

- **Carry pay wasn't the 7-iron's problem:** without it both still fall on every swing, a little later (0.4–0.5 s
  after impact). The order that balanced the driver doesn't balance the iron. Without carry the launch drops
  (11–13° against 20°, attack −7°): carry pay was what kept the blow level.
- **A longer backswing does pay on this body when asked for:** `i7bk`'s grew 1.5 → 1.8 m and its top 0.23 →
  0.31 s, for +8 mph of clubhead and +9 mph of ball at no cost in balance (both 100%) or strike (75% good).
  Without the pay neither chain lengthened it: their speed comes from a lunge (next bullet).
- **Why it falls: the trail foot kicks.** Tracked in the viewer swings, `i7s` hits the ball standing on its lead
  leg: the trail foot is 0.16 m up 0.1 s before impact, 0.50 m at impact and 0.70 m 0.1 s after, flung 0.25 m back
  from its address spot; then it comes down and the golfer topples toward the trail side (torso 0.2–0.4 m away
  from the target at the fall). `i7bk` does the same, lower (0.24–0.29 m), and lifts the trail foot before impact
  on 100% of its swings. A leg flung back is the counterweight for a lunge at the ball, and a one-legged strike
  leaves no base to stop on. The driver's `bal_ref` did a milder version (off the trail foot in the downswing,
  landing on it after impact); `balfix` kept both feet down from 0.25 s after impact.

New knob: `env.trail_foot_weight` (off by default) charges per m per second for the trail sole's lowest point
(its toe once the heel lifts) above `env.trail_lift_free` (3 cm), at any time: the heel may lift and the foot
pivot, as a golfer's does, but the toe stays down. `Metrics/trail_lift_cm` logs the trail sole's highest point
before impact, and `record_swings.py` prints the share of swings with the trail foot off the ground before impact.

`i7bk` 500 iterations more (`i7bk3`, model_2494): still 100% falls (median 0.43 s after impact), backswing 1.84 m,
68 mph clubhead, 79 mph ball, 84 yd, 1° offline, 79% good strikes. (Its viewer labels 142 of 256 strikes "shaft":
the label is the nearest shape at the end of the impact step, by when a 68 mph head is ~25 cm past the ball and
the shaft trails it; the strike itself is the head's swept path. A leftover from when the ball had a collider.)

## The trail foot planted, a power cap (2026-11-20, `i7t6`, `i7t0`, `i7c55`)

All three use `i7bk`'s recipe (no carry, ball pay 6, backswing pay 50 to 1.8 m, takeaway 10) plus
`trail_foot_weight=200`:
- **`i7t6`:** the rule from the full-swing stage, from `i7s_strike` model_598.
- **`i7t0`:** a fresh golfer, the rule from iteration 0 (600-iteration strike stage, then the same chain).
- **`i7c55`:** `i7t6` plus `env.ball_speed_cap=24.6` (55 mph): ball pay stops growing above that speed, so swinging
  harder earns nothing and the balance terms decide how hard to swing (new knob, off by default; a first power target).

| | `i7t6` 1995 | `i7t0` 1996 | `i7c55` |
|---|---|---|---|
| Fell within 3 s / median fall after impact | 100% / 0.37 s | 100% / 0.25 s | 100% in training (stopped at 1500) |
| Trail foot off the ground before impact / highest | 72% / 0.04 m | 91% / 0.04 m | |
| Backswing / top / downswing | 1.80 m / 0.30 / 0.15 s | 1.82 m / 0.31 / 0.21 s | |
| Clubhead / ball (mph), smash | 64.2 / 73.4, 1.19 | 58.8 / 61.1, 1.10 | |
| Weight toward the lead foot at impact | 0.74 | 0.29 | |
| Carry / offline | 78 yd / 1° | 59 yd / 1° | |

- **The rule plants the trail foot** (its highest point before impact 4–6 cm, against 20 cm) at no cost in striking
  or backswing, **but the golfer still falls on every swing**, now straight down: both feet stay planted to 0.2 s
  after impact while the torso sinks 1.07 → 0.51 m in 0.4 s. From iteration 0 it is worse: weight left on the
  trail side at impact (0.29) and a fall 0.25 s after impact.
- **The power cap changed nothing in 600 balance iterations** (episode length 44 → 51 steps, every swing falling).
- **Every 7-iron run falls ~0.3 s after impact, and that never stretches.** The driver's balanced lineage started
  its balance stages falling ~1.2 s after impact and stretched it steadily (`balfix` episode length 90 → 165 steps
  over 1,000 iterations); every 7-iron chain sits at 41–50 steps. A late sit-back gives the hold pay something to
  extend; an immediate collapse doesn't. Both clubs fall in 100% of strike-stage episodes from iteration ~25
  (the untrained golfer flails over at low damping), so that isn't the difference.
- **The difference is the lead leg** (12 viewer swings each, knee flexion, 0° straight):

  | | address | −0.1 s | impact | +0.1 | +0.2 | +0.3 | +0.4 s |
  |---|---|---|---|---|---|---|---|
  | driver `bal_ref1`: lead knee | 28° | 3° | 12° | 3° | 3° | 3° | 3° |
  | driver: pelvis height (m) | 0.83 | 0.81 | 0.80 | 0.81 | 0.81 | 0.81 | 0.80 |
  | 7-iron `i7t6`: lead knee | 28° | 39° | 49° | 37° | 81° | 85° | 44° |
  | 7-iron: pelvis height (m) | 0.84 | 0.83 | 0.79 | 0.78 | 0.74 | 0.57 | 0.41 |

  The driver golfer posts up on a straight lead leg through impact, as golfers do, and its hips hold their height.
  The 7-iron golfer bends its lead knee into impact, probably to stay down to a ball on the ground (posting up
  lifts the body, which a teed driver allows and an iron off the turf doesn't), and with the weight on that
  leg it folds. The stand penalty sat 16 cm below the address torso height (0.95 m), so it only charged once the
  collapse was past saving.

## Staying tall, stronger knees, a straight lead arm (2026-11-20, `i7p`, `i7t6p`, `i7k135`, `i7a`)

Three ways at the folding lead knee, each on `i7t6`'s recipe:
- **Stay tall (`stand_height=1.05 stand_weight=100`):** the torso pays for dropping more than 6 cm below its
  address height, so the cost starts as the knee folds. From the full-swing stage (`i7p`) it was the stand-still
  trap: with no fall termination in that stage a fallen golfer pays for seconds, the return fell to −248 and contact
  and clubhead speed collapsed within 125 iterations (stopped). As a fine-tune of `i7t6` model_1995 (`i7t6p`, with
  the balance recipe's fall termination) the torso stayed taller in training (0.94 → 1.01 m) but after 500
  iterations it still fell on every swing (median 0.40 s after impact).
- **Stronger knees (`i7k135`, new knob `env.knee_gear`, default the stock 90 N·m):** 135 N·m, the hip extensor's
  strength. The stock humanoid's knee gives ~2.3 N·m per kg of body against a person's ~3.3, and the muscle model
  gives a joint bent against its torque no more than its static strength (real muscle resists lengthening at
  ~1.5×). Still 100% falls, but **the first 7-iron run whose falls stretch**: median 0.58 s after impact at 1496 and
  0.70 s at 1995 (`i7t6`: 0.30–0.37 s), episode length rising 48 → 67 steps. It still dips into impact (lead knee
  63° 0.1 s before) but then **pushes up onto a straight lead leg** (3° 0.1 s after impact, hips 0.72 → 0.81 m)
  before folding again ~0.4 s after impact. Clubhead 63 mph, ball 70 mph, backswing 1.78 m.
- **A straight lead arm at address (`i7a`):** both clubs' addresses hold the lead elbow bent ~50° (the solver put
  the hands at a fixed spot and nothing asked for a straight arm). Through the downswing the club's pull straightens
  it, lengthening the swing radius:

  | address → impact | lead elbow bend | shoulder to clubhead | lead shoulder height |
  |---|---|---|---|
  | driver `bal_ref1` | 66° → 20° | 1.31 → 1.44 m | 1.21 → 1.28 m |
  | 7-iron `i7t6` | 50° → 41° | 1.20 → 1.24 m | 1.14 → 1.14 m |

  The driver golfer takes the 13 cm by rising onto its lead leg, which a teed ball allows; for the 7-iron 13 cm
  more radius is a club buried in the turf, so it holds the arm bent and stays down on a bent lead knee. A golfer
  addresses with a straight lead arm so impact can return to the address shape. `solve_address.py` has a new
  `--lead_elbow_deg` (lead elbow bend target, off by default) and `--out`. The straightest this arm gets is 20° of
  bend (the elbow joint's −90° limit; the stock humanoid's elbow is bent ~107° at joint zero), against 48°: torso
  1.18 m (was 1.14), spine tilt 30°, ball in the same spot, no overlapping limbs. It trains under a trial club key,
  `7iron_arm` (`clubs.py`, `address/7iron_arm.json`, `golfer_7iron_arm[_wrist].usda`), so the runs still training
  on the 7-iron's files weren't disturbed: a fresh golfer, 600-iteration strike stage, then `i7t6`'s chain.

`i7k135` 500 iterations more (model_2494): median fall 0.78 s after impact, episode length 67 → 72, still 100%;
trail foot down on every swing, 63 mph clubhead, 72 mph ball, 79 yd.

256 swings after the second balance stage (iteration 1996), the straight-arm golfer against the bent-arm one on
the same recipe:

| | `i7t6` (bent arm) | `i7a` (straight arm) |
|---|---|---|
| Fell within 3 s / median fall after impact | 100% / 0.37 s | 100% / **0.63 s** (0.42 s at 1497) |
| Extrapolated CoM off the feet, upright time after impact | 78% | **41%** |
| Clubhead / ball (mph), smash | 64.2 / 73.4, 1.19 | 66.9 / **82.4, 1.30** |
| Carry / offline / strikes good-fat | 78 yd / 1° / 73-20% | **100 yd** / 1° / 74-26% (91-9% at 1497) |
| Backswing / top / downswing | 1.80 m / 0.30 / 0.15 s | 1.85 m / 0.28 / 0.17 s |
| Weight toward the lead foot at impact | 0.74 | 0.74 |

Through impact (12 viewer swings, knee flexion 0° straight):

| `i7a` | address | −0.1 s | impact | +0.1 | +0.2 | +0.3 | +0.4 s |
|---|---|---|---|---|---|---|---|
| lead knee | 28° | 4° | 23° | 2° | 2° | 26° | 52° |
| pelvis height (m) | 0.84 | 0.82 | 0.81 | 0.83 | 0.85 | 0.83 | 0.78 |
| torso height (m) | 1.14 | 1.02 | 1.08 | 1.07 | 1.04 | 1.00 | 0.92 |

- **The straight lead arm removes the collapse.** The arm comes back to its address shape at impact (bend 21° →
  23°, radius 1.25 → 1.26 m), the lead leg posts straight through impact as the driver golfer's does, the hips hold
  their height, and both feet stay planted for 0.4 s after impact. The strike improves with it (smash 1.30, the
  7-iron's best; 100 yd).
- **What's left is the driver's kind of fall:** a lean toward the target (two thirds of the falls), median 0.63 s
  after impact and later each stage (training episode length 50 → 62 steps over the second balance stage), the
  pattern the driver's balance stages climbed from.

### Straight arm: more balance training, stronger knees, more hold pay (2026-11-20, `i7ak`, `i7a3`–`5`, `i7ah`)

- **Stronger knees don't add to the straight arm:** `i7ak` (`i7a`'s strike stage, then `knee_gear=135` from the
  full-swing stage) at 1996: median fall 0.50 s after impact against `i7a`'s 0.63 s, extrapolated CoM off the feet
  88% of upright time against 41%, weight 0.49 toward the lead foot at impact. Same speed (69 mph, 84 mph ball,
  105 yd). The knee only mattered while the bent arm kept it bent; the stock 90 N·m stays.
- **More of the same recipe stretches the fall slowly:** median fall after impact 0.63 s (1996) → 0.80 s (2495) →
  0.88 s (2994) → 0.97 s (3493), training episode length 62 → 72 → 77 → 80 steps, still 100% falls. ~2–4 steps per
  100 iterations, against the driver `balfix` chain's ~7.5 from a later start.
- **Hold pay 60 (from 20), from 2495:** 0.97 s at 2994 and 1.07 s at 3493 against 0.88 / 0.97 s, episode length
  82 → 85 (flattening in its second 500). Speed and strike unchanged (69 mph, 84 mph ball, 101 yd, smash 1.27).
  The balance cost isn't what holds it back: per step it is −4 against +10 of hold pay (the driver at the same
  stage paid −19 to −25 against +13–15).
- **How it falls now** (`i7ah1`, 12 viewer swings): both legs straight, torso ~1.03 m and feet planted for 0.4 s
  after impact; then the trail foot drags toward the target (−0.15 → +0.08 m along the target line by 0.8 s, the
  stance narrowing to ~15 cm), the lead knee folds (71° at 0.8 s) and it goes down toward the target. The
  driver's balanced golfer keeps its trail foot planted 0.5 m behind the lead foot through the finish.
- **No shift pay, or a power cap, from `i7ah2` (500 iterations each, hold 60):** `shift_weight=0` (`i7as`):
  median fall 1.10 s, episode length flat at 86–88; `ball_speed_cap=33.5` (75 mph, `i7ac`): 1.12 s, flat. Neither
  moved the plateau (1.07 s). The cap didn't slow the swing (clubhead 68 mph either way): with no pay above 75 mph
  the strike decayed instead (smash 1.27 → 1.12, ball 76 mph, 81 yd). Paying less for speed doesn't ask for control.
- **The bow:** spine tilt (head over pelvis) and head height around impact, 12 viewer swings:

  | | address | −0.2 s | impact | +0.2 | +0.4 | +0.6 | +1.0 s |
  |---|---|---|---|---|---|---|---|
  | 7-iron `i7ah2` spine tilt | 38° | 62° | 52° | 65° | 70° | 82° | 80° |
  | 7-iron head height (m) | 1.30 | 1.09 | 1.16 | 1.04 | 1.03 | 0.91 | 0.65 |
  | driver `bal_ref1` spine tilt | 37° | 43° | 31° | 32° | 28° | 22° | 17° |
  | driver head height (m) | 1.30 | 1.28 | 1.30 | 1.29 | 1.31 | 1.35 | 1.35 |

  The driver golfer stands up into its finish; the 7-iron golfer bends further over to reach the ball on the
  ground and keeps folding after impact (hips steady to 0.6 s). Holding ~20 kg of upper body, arms and club at
  60–80° of lean needs roughly 50–60 N·m at the waist (an estimate), near the stock 67.5 N·m before the muscle curve
  takes its share; at the driver's 30° about half. New knob `env.waist_gear` (default the stock 67.5) to test it.
- **Strength and credit don't undo the bow (from `i7ah2`, 500 iterations each, hold 60):** a 100 N·m waist
  (`waist_gear=100`, `i7aw`): median fall 1.03 s, spine tilt unchanged to the degree (51° at impact, 83° at 0.6 s).
  Full upright credit only at a 1.2 m torso (`finish_high=1.2`, from 1.0; the bowed golfer's torso link is still at
  1.03 m, so bowing had cost nothing in hold pay or finish credit) (`i7af`): 1.08 s, the same bow. The same with the
  action std reset to 0.25 (`i7afs`, it had shrunk to 0.13): 1.07 s, the same bow. The finish strategy is set
  before these fine-tunes start, as with the driver (runs 75–78). `i7af0` (`finish_high=1.2` from the full-swing
  stage of a straight-arm chain) is the from-the-start version.
- **`i7af0` (`finish_high=1.2` from the full-swing stage, `i7a`'s strike stage) matches `i7a`:** median fall 0.42 s
  (1497) and 0.62 s (1996) against 0.42 / 0.63 s, the same bow (spine 49° at impact, 76° at 0.4 s after), smash 1.31,
  102 yd. Crediting only a tall finish doesn't make one, from the start either.

## A posture pay, and a straight-armed driver (i7sp, drva) — 2026-11-20

- **Posture pay** (`env.posture_weight`, off by default): per second after impact, × upright, for the spine (the
  pelvis-to-head line) standing up out of its bend: full pay at `spine_good_deg` (20°, about where the driver golfer
  finishes), none at `spine_bad_deg` (70°, the bowed iron). It only adds, so falling sooner never pays better (a cost
  after impact would, with `fall_terminate`). Logs `Reward/posture` and `Metrics/spine_deg_after_impact`.
- **`i7sp`: posture 40 from the full-swing stage**, otherwise `i7a`'s recipe from `i7a`'s strike stage. The bow is gone
  (spine tilt and head height, 12 viewer swings):

  | | address | −0.2 s | impact | +0.2 | +0.4 | +0.6 | +0.8 | +1.0 s |
  |---|---|---|---|---|---|---|---|---|
  | `i7a2` spine tilt | 38° | 62° | 48° | 72° | 79° | 81° | 76° | 77° |
  | `i7sp2` spine tilt | 38° | 68° | 40° | 22° | 24° | 20° | 75° | 96° |
  | `i7sp2` head height (m) | 1.30 | 1.02 | 1.22 | 1.35 | 1.28 | 1.17 | 0.58 | 0.15 |

  It still falls on every swing: median fall 0.70 / 0.82 / 0.90 s after impact at 1497 / 1996 / 2495 against `i7a`'s
  0.42 / 0.63 / 0.80 s, so the head start shrinks. Speed and strike unchanged (84 mph ball, smash 1.28, 105 yd, 1°
  offline, 76% good strikes). The fall is new: the lead foot slides 15 cm toward the target in the downswing, the
  weight sits further back at impact (0.51 of the way to the lead foot against 0.69–0.74), and 0.3 s after impact the
  lead leg kicks up (0.30 m by 0.4 s, the knee folding to 93°) while the trail foot slides up under the body (18 cm),
  and it topples toward the target. Extrapolated CoM off the feet 80% of upright time against 39%. It is the
  counterbalancing leg kick of the driver lineage before its balance reward (runs 67–74).
- **`drva`: the driver with a straight lead arm** (`solve_address.py --club driver --lead_elbow_deg 10`, club key
  `driver_arm`), through `balfix`'s from-scratch stage order (`fresh_chain2`), to find out whether the straight arm
  causes the bow. The address stands taller (spine 22° against 37°, torso 1.23 m against 1.18 m: the hands' spot is
  set by the club, so a straighter arm lifts the shoulders), and it learns a different swing from the first stage:
  shots 35° right in the strike stage and 60° right in the full swing at 46 mph (the bent-arm driver: 14° at 62 mph).
  In the balance stages it squats into the ball (pelvis 0.83 → 0.47 m at impact, both feet 16 cm off the ground just
  before it) and holds the crouch through all three balance stages (to 2195): 1% falls, but 39–43 mph balls 61–63° right that the aim factor pays almost nothing
  for, so it lives on the finish pay. It never made a swing comparable to `balfix`, so it says nothing about the bow.
- **Lead foot planted too (`i7spl`: `i7sp` + `lead_foot_weight=100` from the full-swing stage):** the lead foot stays
  down (moves 0.46 m in all against 0.98 m), but it falls sooner (median 0.52 / 0.65 s at 1497 / 1996 against
  0.70 / 0.82 s), 8 mph slower (ball 72–74 mph, 78–83 yd) and sinks: both knees fold to ~85° by 0.4 s after impact and
  it sits down. The penalty takes most of the return (45 against 249 in the same evaluation). Stopped before its third
  stage. The driver's lesson again: this body balances partly by moving its feet.
- **Long run (`i7spx`: `i7sp3` onward with hold 60, 500-iteration stages):** median fall 0.90 s (2495) → 1.05 s (2994)
  → 1.15 s (3493), speed and strike holding (86 mph ball, 108 yd). It now stands tall for 0.8 s after impact (head
  1.43 m, spine 10–15°), then the trail foot slides up under the body (−0.18 → +0.07 m along the target line), the lead
  leg kicks up to ~0.7 m, and it falls off the straight trail leg, backward (83%) and toward the target. The weight
  stays back at impact (0.48 of the way to the lead foot against `i7a`'s 0.69–0.74): standing the spine up came with
  the weight going back, and a finish on the trail leg.
- **`i7sps`: `i7sp` + `impact_shift_weight=150` from the full-swing stage**, pay for the weight forward at the strike
  (as a fine-tune at 100, `ishift`, it moved 0.40 → 0.47). It overshoots: weight 0.90 of the way to the lead foot at
  impact (against 0.52), but it falls sooner (median 0.48 / 0.65 s at 1497 / 1996 against 0.70 / 0.82 s) and 5 mph
  slower (ball 73–78 mph, 75–87 yd): it lunges over the lead leg, the lead foot steps ~0.6 m toward the target, and it
  goes down forward. Stopped before its third stage.
- **Long run, continued:** median fall 1.27 s (3992) → 1.30 s (4491) → 1.37 s (4990), 85 mph ball, 105 yd, offline
  p90 3–4°. Still gaining, ~0.05–0.1 s per 500 iterations.
- **`i7sph`: hold 150 from `i7spx8` (4990)**, against the long run's next two stages at hold 60 (same start, only hold
  differs): hold is the one dense pay for staying up, and the only knob that sped the stretch.
- **Hold 150 changes nothing:** `i7sph` median fall 1.38 s (5489) and 1.42 s (5988) against the long run's 1.38 / 1.40 s
  at hold 60; ball 83–84 mph against 85. The long run has plateaued: 1.37 → 1.38 → 1.40 s over 1,000 iterations,
  action std down to 0.09. Its last stage ends at 1.42 s (6487; ball 84 mph, 102 yd, 76% good strikes, 7% thin).
  Verdict: posture pay fixed the bow and gave the best iron balance yet (falls ~1.4 s after impact against
  `i7ah2`'s 1.07 s), but every swing still falls within 3 s, now off the trail leg.
