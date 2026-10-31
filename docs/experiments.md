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
