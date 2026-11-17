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
