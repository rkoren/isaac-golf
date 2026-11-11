# CLAUDE.md — isaac-golf

Side project: teach a stick-figure agent to swing golf clubs and play a designed course, using RL in NVIDIA Isaac Lab. Primary goal is **learning RL in simulated spaces** — a working, ugly swing beats a pretty stalled project.

# Leave the README plain

## Environment

- **Isaac Lab 3.0.0 + Isaac Sim 6.2** (pip wheels), **uv project, Python 3.12**. Full matrix and install gotchas in `docs/versions.md`; `uv.lock` is the pin.
- **No conda.** Isaac Lab 3.0 ships a uv-based install; `uv.lock` does what a pinned conda env would. Never install into another env; Isaac pins its own torch (2.11.0+cu130).
- The old binary install at `C:\isaacsim` (6.0.0-rc.59) is **unused and untouched** — it only pairs with Isaac Lab's `v3.0.0-beta2` tag. Don't wire anything to it.
- Every command runs through `uv run --env-file omni.env ...`. `omni.env` carries the accepted Omniverse EULA (Kit won't boot without it) and the UTF-8 vars Windows needs.
- **Hardware:** RTX 5070 Ti 16 GB (Blackwell), Ryzen 5700X, 32 GB RAM. 32 GB RAM is at Isaac's floor — never GUI + training simultaneously.
- **Headless is the default in 3.0** — there is no `--headless` flag. GUI is opt-in via `--viz kit`.
- RL stacks: use what Isaac Lab ships (**rsl_rl** PPO; skrl for AMP later). Do not hand-roll PPO.
- Long runs in a detached terminal. Isaac Lab writes checkpoints + TensorBoard to `logs/` (gitignored; `runs/` is ignored too but unused).
- **Physics backend** is per-run: `physics=NAME`. We train on **`isaacsim_physx`** — chosen for Phase 3's USD course and GUI replay, not speed. `newton_mjwarp` measured *faster* (234k vs 160k env-steps/s, 2.0 vs 4.9 GB VRAM) and is a live alternative, not a fallback; revisit at the Stage 2 gate when impact fidelity starts to matter. Keep env configs backend-agnostic (`PresetCfg`) so switching stays cheap. `physics=` defaults to `newton_mjwarp`, so **PhysX must be asked for explicitly every run**.

## Architecture (three phases — do not blend them)

**Phase 1 — Learn the swing (Isaac Lab, massively parallel RL)**
Stick figure + club welded to hands + ball on tee, flat ground. Goal: hit far and straight.

**Phase 2 — Analytic ball flight**
Do NOT simulate ball flight in PhysX (no Magnus effect; sub-ms club impact is numerically unreliable). At impact, read clubhead state → feed an analytic flight model (launch speed/angle, spin, drag + Magnus lift — standard golf-physics equations) → landing position.
The **launch itself is already analytic** (Phase 1 env): ball speed = **smash factor** × clubhead speed into the face, the way launch monitors and sims report it. Smash is 1.47 for a centred driver strike and falls off with distance from the sweet spot (~1.2 for a 2 cm toe miss, ~1.0 topped or skulled); start direction is 85% face normal, 15% club path. Spin and launch angle (loft) come with this phase. Clubs = different welded geometries + loft angles; policy conditions on club ID.
**Clubs** live in `envs/golf/clubs.py` (driver, 7iron, pw: length, shaft angle, head height, stance, smash ceiling, look). Each has its own asset (`assets/golfer_<club>.usda`, built by `make_golfer_asset.py`) and solved address (`envs/golf/address/<club>.json`, written by `solve_address.py --club`). Pick one per run with `env.club=7iron`; one policy per club for now, club-ID conditioning later. Head looks are visual meshes; the strike uses an invisible box collider.

**Phase 3 — Play a course (hierarchical)**
Designed course as USD terrain/heightmap. High-level layer (heuristic or simple policy) picks club + aim + power target per shot; low-level swing policy executes. Never attempt tee-to-hole end-to-end with one policy.

## Phase 1 environment spec

**Starting skeleton:** fork an existing Isaac Lab humanoid task (the Humanoid locomotion env, or the AMP humanoid example if going the motion-imitation route later). Strip to our needs rather than building from zero.

**Body, as measured from the USD (not assumed):** 16 rigid bodies, 15 joints, **21 actuated DoF**, root is a free-floating `torso`. Arms attach to `torso`, so torso joints never reposition the hands — spine tilt at address comes from root orientation set at reset. Coil budget is small: waist twist ±45°, bend −75°/+30°, pelvis side-bend ±35°, shoulders 2 DoF each, elbows 1 DoF, **no wrists** (an optional lead wrist hinge exists, see below). Expect an arm-dominant half-coil swing. Golfer stands ~1.6 m, so scale the club (~0.9 m, not a real 1.14 m driver). Both hands *can* reach one grip point (31% of the lead hand's 0.52 m³ workspace, 1.21 m arc) — reachability was never the issue, the loop was.

**Agent body:** simplified humanoid articulation (USD or MJCF/URDF). Start with the stock Isaac Lab humanoid (~21–28 DoF) even though it's more than a "stick figure" — a working asset beats a custom one. Reduce DoF later only if training struggles. Club: rigid capsule/box welded to both hands as a fixed joint. **Grasping is out of scope — never make the agent learn grip.**

**Observations (per env):**
- Joint positions + velocities (normalized)
- Torso/pelvis orientation (quat or 6D) + linear/angular velocity
- Clubhead position + velocity relative to ball
- Ball position (and velocity once struck, Phase 1 only)
- Target direction vector + commanded power (for conditioning, Phase 2+)
- Phase/time-in-episode scalar

**Actions:** joint torques or position targets (Isaac Lab humanoid examples use torque control — keep their choice); action dim = DoF of the articulation.
Measured 2026-10-30: the implicit actuators also carry PD stiffness (10–20 N·m/rad on arms and waist) toward position targets of **joint zero** — the stock standing pose, not the address. With zero action the golfer springs out of address on the first step (clubhead 4–16 mph, gravity off). Every torque the policy outputs fights that spring. Open question, not yet changed.
**Faster than a human (measured 2026-10-30):** the 7-iron hits 91 mph at impact with the lead hand at 35 mph (real golfers ~20) only 0.27 s after address (real swings ~1 s). With no wrists, head speed has to come from hands moving faster than a person can, and the actuators give full torque at any joint speed. Watch `Metrics/impact_hand_speed_mph` and `Metrics/impact_time` next to clubhead speed; a speed gain with hands far above ~20 mph is the actuators, not technique. Candidate fix: a torque–speed curve, now available as the muscle model below.
**Muscle model (`env.muscle=true`, off by default):** Hill's force–velocity curve on every joint, so torque falls as a joint speeds up: × (1 − x)/(1 + x/0.25), x = shortening speed / vmax, vmax 20–30 rad/s per joint (`MUSCLE_VMAX` in `golf_env_cfg.py`). Applied every physics substep in `_apply_action`. Not `DCMotorCfg`: past its no-load speed a DC motor brakes, which would punish an outer joint being whipped faster by the body behind it (the kinetic chain); muscle just goes slack. Measured 2026-10-30: hands drop to human speed or below (9–16 mph), strikes get cleaner (smash 1.28–1.31), clubhead falls to 40–49 mph, and the swing stays short (impact ~0.3 s). It still kneels at the finish (torso 0.70–0.84 m). Another 300 iterations (2026-10-31) raised the finish to ~1.0 m, but as a wide lunge with the trail foot slid back, not a golf finish.
**Joint damping is a hidden speed cap (measured 2026-10-31):** the stock actuators carry 5 N·m·s/rad on waist, shoulders, pelvis and hips (~10–50× human). In the downswing it burns 74% of the muscle work, and at impact it out-pulls the muscles, so a longer backswing only feeds the damper; that, not the reward, is why every swing stopped at the 1.2 m `backswing_cap`. `env.damping_scale=0.1` (`DAMPING` in `golfer.py`, default 1.0): clubhead 40 → 68 mph with hands at a human 22 mph, the hip and arm springs start storing and returning energy, but the backswing only grows to 1.28 m (the arms now hit their muscle vmax) and it falls 98% of the time, since the leg dampers had been balancing it for free. Discount γ 0.995 changed nothing about timing.
**Balance at low damping (2026-10-31):** resuming that run with `env.stand_weight=10 env.feet_slip_weight=10` (plus `aim_sigma=15`, `finish_low=0.8`) brings falls from 98% to 19% at 71 mph (ball 89 mph, ~1° offline), still improving. Bracing the legs instead (`env.leg_damping_scale=1.0`: hips, knees, ankles at stock damping, no balance rewards) stands as often (16%) but costs ~11 mph. Best driver recipe so far: muscles, `damping_scale=0.1`, balance rewards.
**Lead wrist hinge (`env.wrist=true`, 2026-10-31):** the weld becomes a revolute joint `lead_wrist` about the club's toe-heel axis (the swing plane at address), −25°/+70° from the address angle, weak on purpose (10 N·m vs 67.5 for the arms). Asset `golfer_<club>_wrist.usda` from `make_golfer_asset.py --club <club> --wrist` (driver only so far); 22 actions, 84 observations; address unchanged. It is what makes a long backswing pay: from scratch at low damping the backswing grows to 1.68 m (from 1.28) with no reward for it, clubhead 84 mph from 17 mph hands (4.9×, a real golfer's 4–5×), and the release is passive (the wrist muscle does ~1 J; the body drives it). Balance does not come back yet (2026-11-05): after 1,500 balance-stage iterations it still falls 98% (90 mph, ball 114 mph), or 83% with braced legs (82 mph), both finishing at 0.42 m. The finish credit is zero everywhere below `finish_low` (0.8), so from a 0.42 m finish standing taller earns nothing; only the stand penalty pulls up. Adding `env.fall_penalty=50` to the braced run brings falls to 48% (see the balance notes above).

**Ball + tee:** the ball is a real-spec sphere (r=21.34 mm, 45.9 g) with **gravity disabled**, and the tee is a **visual-only** prim with no collider. Phase 2 owns flight analytically, so PhysX only has to report the impact — a gravity-free ball needs no support, which also means the club can never strike the tee (a static tee would deflect the clubhead like a steel post). Stage 4.5 turns gravity back on and rests the ball on the ground.
**The ball has no collider.** PhysX never touches it: a strike is the clubhead's swept path passing it (after arming, inside the head's box) and the env writes the launch velocity from the smash-factor model. So nothing but the clubhead can move it — no kicks, no tunneling, no ghost hits. Address leaves 2 cm of air between clubface and ball.

**Target line:** the golfer faces **+X** with the ball in front of its chest, and a right-hander sends the ball to its **left, so the target direction is +Y** — not +X. The swing plane is perpendicular to the stance, which is easy to get backwards.
`solve_address.py` enforces this with a heading cost (shoulder and hip lines along +Y) and a no-overlap cost, plus an athletic stance: feet 0.40 m apart (shoulders 0.34), knees 25°, flat feet, weight over their middle. The first solve faced −Y: it built (w,x,y,z) quaternions for a sim that reads (x,y,z,w), which put upright-facing-+X outside its search space (see Known traps). After any address change, `scripts/check_address.py` must report **zero overlapping limbs**.

**Episode:** reset with agent addressing a teed ball; terminate on timeout (~5 s) or N steps post-impact. **No fall termination in Stage 0** (see curriculum); add it only once standing is being rewarded.

**Parallelism:** start at 1,024 envs, scale toward 4,096 as VRAM allows. Headless, fixed small dt with substeps; increase solver iterations around contact. If impact is still noisy, detect clubhead-ball proximity and read velocities at that tick rather than trusting the collision impulse.

## Reward curriculum (the actual RL lesson — stage gates, not one reward)

1. **Stage 0 — swing at all:** reward **ball speed gained** (the real objective; the ball is gravity-free and starts at rest, so any speed is a genuine strike) + a **capped backswing** term for winding up + clubhead speed near the ball. **No alive bonus, no fall penalty, no fall termination** — swinging is the first goal, and if the golfer flings itself over to do it, fine.
   **Do NOT use a clubhead-to-ball distance ("approach") reward here.** The club starts at address ~6 cm from the ball, so distance reduction is optimised by standing perfectly still, and it penalises the backswing — the one thing a swing must do first. Measured: reward fell −1.2 → −7.0 and clubhead speed fell 13.2 → 7.2 mph as the policy learned to freeze. A distance term only makes sense if the club starts away from the ball.
   Contact detection needs **arming**: the clubhead must travel >0.25 m from the ball before a hit counts, or `contact_rate` reads 1.0 from iteration 0 and measures nothing.
   **Backwards hits are never paid:** back of the head, or the ball heading >90° off the target line, earns no ball speed and costs a one-off penalty (20). No golf shot goes backwards.
   **Current recipe (2026-10-30):** strike stage from scratch on defaults, then resume with `env.finish_frac=0.5 env.impact_arm_dist=1.0 env.backswing_cap=1.2` (strike counts only after a 1.0 m backswing; half the ball pay waits for a posture-scaled finish). Fresh per club: a driver policy does not transfer to an iron address.
   **Finish and aim knobs (2026-10-31, off by default):** `env.finish_low`/`env.finish_high` set the finish torso heights worth zero/full credit (0.6/1.0 m by default, which pays 22% for a kneel; 0.8 zeroes it). `env.aim_sigma` [deg] multiplies the ball pay by exp(−(offline/σ)²) on the horizontal launch angle, an early start on Stage 3. Measured from run 18, against a control run: σ = 15° halves the offline angle (+7° → +4°, still a push) at no cost in speed; the stricter finish added at most a few cm over what more training alone gave.
   **Balance, added once contact was met (2026-10-30):** penalties for the torso dropping below 0.95 m and for feet sliding >3 cm, with a 1 s post-impact finish. Off by default: from scratch it drowns the strike and the policy learns to stand still (measured, run 5). Turn it on only when resuming from a striking checkpoint: `--checkpoint <run>/model_N.pt env.stand_weight=10 env.feet_slip_weight=10`. Still no fall termination.
   **That stand penalty barely separates a fall from a stand** (measured 2026-11-05): the fall comes late in the finish window, so a faller pays ~2 points more against a ~140-point ball pay. `env.fall_penalty` (off by default) is a one-off cost when the torso first drops below `env.fall_height` (0.6 m, `record_swings.py`'s "fell" line); `Metrics/fell_rate` logs falls in training. At 50 (wrist, braced legs, against a control): falls 83% → 48% with no clubhead speed lost, while the control drifted to 95%. The swings that stay up are mostly crouched, 0.6–0.8 m.
   **"Any speed is a genuine strike" is false (measured 2026-10-29).** At iter 199, 45 of 52 armed strikes were the left shin/foot — the policy kicks the ball. Fixed by construction: the ball has no collider and only the clubhead's analytic launch moves it. The club knocking it off the tee at reset is gone too (overlap-free address). Details in `docs/experiments.md`.
2. **Stage 1 — contact:** big sparse bonus for club-ball contact; keep stage 0 shaping at reduced weight.
3. **Stage 2 — ball speed:** reward ball speed at/after impact. Expect violent, ugly swings — fine.
4. **Stage 3 — direction:** reward ball-velocity alignment with target direction; penalize the offline component. Started early as `env.aim_sigma` (Stage 0 notes); `Metrics/offline_deg` logs it.
5. **Stage 4 — distance/carry (via Phase 2 flight model):** reward carry toward target; then target-distance accuracy for partial shots (power conditioning).
6. **Stage 4.5 — real lies:** ball off flat ground, not a tee. Penalize **chunking** (clubhead contacts ground before ball) and **thinning** (leading edge strikes the ball's upper half). Needs ground-contact ordering + impact height, so keep both cheap to read from the env.
7. **Stage 5 (optional stretch) — style:** AMP with a golf-swing motion clip so it looks like golf. Only after Stage 4 works.

Promote stages by checkpoint: train, evaluate, freeze, add the next term, continue. Log every stage in `docs/experiments.md` (config, reward curve, verdict, one-line takeaway).

Standard shaping terms kept throughout: action-rate penalty (smoothness), small torque/energy penalty, joint-limit penalty. Tune weights so the task reward dominates.

## Repo conventions

- Layout: `envs/` (env + reward code), `assets/` (USD/URDF — small ones committable), `scripts/` (train/eval/replay entrypoints), `flight/` (Phase 2 analytic model + unit tests — pure Python, testable without Isaac), `docs/` (versions, experiments log), `runs/` (gitignored).
- `.gitignore`: `runs/`, checkpoints (`*.pt`, `*.onnx`), TensorBoard events, `__pycache__/`, large USD/binary assets, Omniverse cache dirs.
- The flight model is ordinary physics code: write it test-first (known inputs → known carry distances from published golf data) before wiring it to Isaac.
- Commit small, push each session. Experiments without a logged config/result didn't happen.
- **Archive every stage's swings:** `uv run python scripts/archive_run.py <run dir> --stage stageN --note "..."` films each checkpoint in Kit (down the line, replaying the run's body settings (`club`, `wrist`, `muscle`, `damping_scale`, `leg_damping_scale`, `impact_arm_dist`) from its `env.yaml`), stacks them into `progression.mp4`, saves the swing viewer, and adds a row to `media/index.md`. `media/` is tracked (logs/ isn't); clips are ~300 KB, a viewer 2–4 MB. Use `--existing` for runs whose env has since changed.
- **Speeds are reported in mph** (clubhead and ball): TensorBoard `Metrics/*_speed_mph`, `record_swings.py`, the swing viewer, `media/index.md`, `docs/experiments.md`. Everything internal (env, cfg, rewards, recorded swing data) stays m/s; convert with `MPH` from `golf_env_cfg.py` only where it is displayed. Direction is the **offline angle** in degrees, + = right of target (a push or slice for this right-hander). Backswing is the farthest the clubhead gets from the ball before impact, in metres.
- **Watch the policy before trusting a reward curve.** `scripts/record_swings.py <run dir> physics=isaacsim_physx` rolls out every checkpoint headless and writes one self-contained HTML viewer to `<run>/swings/` (scrub, slow-mo, camera presets, which body part struck the ball). For a Kit render: `isaaclab play --rl_library rsl_rl --task Isaac-Golf-Swing-Direct --num_envs 1 --checkpoint <path/to/model_N.pt> --video --video_length 300 physics=isaacsim_physx` — needs the `video` extra (in `pyproject.toml`); 3.0's `play` has no `--load_run`.

## Known traps (pre-answered — no session relitigates them)

- **Don't simulate ball aerodynamics in PhysX** — no Magnus lift; the flight model owns everything after impact.
- **Don't learn grasping** — weld the club.
- **Don't train with the GUI open** — headless, or it will crawl and eat RAM.
- **Don't end-to-end the whole golf round** — hierarchy per Phase 3.
- **Don't trust raw contact impulses at impact** — small dt/substeps, or sample clubhead velocity at proximity. Measured (`scripts/check_impact.py`): at 120 Hz PhysX misses most strikes above ~11 mph and all at 56–89 mph; sweep CCD is CPU-only, speculative CCD does nothing on GPU articulation links, and a big ball contact offset invents strikes on near misses. So the ball doesn't collide at all: contact is the clubhead's swept path (reliable at any speed) and the launch is computed (smash factor, Phase 2 above).
- **Don't start with a sparse ball-in-hole reward** — curriculum above.
- **Club welds to `left_hand` only** (lead hand for a right-handed swing). The humanoid *does* have `left_hand`/`right_hand` bodies — 4 cm spheres on fixed joints, zero DoF, ideal anchors. Welding **both** hands closes a kinematic loop (torso → left arm → club → right arm → torso) which PhysX must solve as an extra constraint at only 4 position iterations: sag, jitter, blowup. The trail (right) hand is kept on the grip by a **reward penalty**, not a constraint — tree stays intact and the policy learns to hold on. A soft physical D6 spring is a later logged experiment, not the default.
The **grip cluster never self-collides** (club, both hands, both forearms: `FilteredPairsAPI` in `make_golfer_asset.py`). Fat capsules converging on one grip always overlap, and with no wrists the shaft crosses the lead forearm permanently. PhysX only skips jointed pairs, so before the filter it flung those bodies apart at every reset.
- **Don't fork blind:** env logic lives in `locomotion_direct_env.py`, not the humanoid files.
- **Don't copy Isaac Lab 2.x tutorials verbatim** — 3.0 moved asset data behind `.torch` accessors, dropped `--headless`, and dropped the `-v0` task suffix.
- **Isaac Lab 3.0 quaternions are (x, y, z, w)** — root poses, `InitialStateCfg.rot`, body quats, `isaaclab.utils.math`. USD (`Gf.Quatf`, joint `localRot`) is still (w, x, y, z). Mixing them up doesn't crash; it silently rotates things. That is how the first address solve came out facing the wrong way.
- **Version drift:** Isaac Lab ↔ Isaac Sim ↔ driver versions get pinned once working; upgrades are deliberate, between phases, never mid-training.

## Milestones

- [x] **Phase 0:** Isaac Lab 3.0 + Isaac Sim 6.2 installed; stock humanoid trains headless at 4096 envs, ~160k env-steps/s on PhysX, 4.9 GB VRAM (`docs/experiments.md`)
- [x] Fork humanoid env → golf env skeleton: teed ball + welded club + reset/termination (`envs/golf/`, `assets/golfer_<club>.usda`; address pose + weld solved by `scripts/solve_address.py`)
- [x] Stage 0–1 training: agent reliably contacts the ball (Stage 0 alone, run 4: contact 0.02 → 0.99 in 300 iters, 64/64 clubhead strikes; `media/`)
- [ ] Stage 2–3: fast, directed ball at impact
- [ ] Phase 2 flight model (unit-tested) wired to impact state; carry distances plausible per club
- [ ] Stage 4: power/direction-conditioned swing policy
- [ ] Phase 3: first designed hole in USD; heuristic club/aim layer; complete a hole
- [ ] Stretch: AMP styling; multi-hole course; replay rendering

First milestone that counts: **a headless training run where contact rate climbs.** Everything before that is setup; everything after is iteration.
