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
Do NOT simulate ball flight in PhysX (no Magnus effect; sub-ms club impact is numerically unreliable). At impact, read clubhead state → feed an analytic flight model (launch speed/angle, spin, drag + Magnus lift — standard golf-physics equations) → landing position. Clubs = different welded geometries + loft angles; policy conditions on club ID.

**Phase 3 — Play a course (hierarchical)**
Designed course as USD terrain/heightmap. High-level layer (heuristic or simple policy) picks club + aim + power target per shot; low-level swing policy executes. Never attempt tee-to-hole end-to-end with one policy.

## Phase 1 environment spec

**Starting skeleton:** fork an existing Isaac Lab humanoid task (the Humanoid locomotion env, or the AMP humanoid example if going the motion-imitation route later). Strip to our needs rather than building from zero.

**Body, as measured from the USD (not assumed):** 16 rigid bodies, 15 joints, **21 actuated DoF**, root is a free-floating `torso`. Arms attach to `torso`, so torso joints never reposition the hands — spine tilt at address comes from root orientation set at reset. Coil budget is small: waist twist ±45°, bend −75°/+30°, pelvis side-bend ±35°, shoulders 2 DoF each, elbows 1 DoF, **no wrists**. Expect an arm-dominant half-coil swing. Golfer stands ~1.6 m, so scale the club (~0.9 m, not a real 1.14 m driver). Both hands *can* reach one grip point (31% of the lead hand's 0.52 m³ workspace, 1.21 m arc) — reachability was never the issue, the loop was.

**Agent body:** simplified humanoid articulation (USD or MJCF/URDF). Start with the stock Isaac Lab humanoid (~21–28 DoF) even though it's more than a "stick figure" — a working asset beats a custom one. Reduce DoF later only if training struggles. Club: rigid capsule/box welded to both hands as a fixed joint. **Grasping is out of scope — never make the agent learn grip.**

**Observations (per env):**
- Joint positions + velocities (normalized)
- Torso/pelvis orientation (quat or 6D) + linear/angular velocity
- Clubhead position + velocity relative to ball
- Ball position (and velocity once struck, Phase 1 only)
- Target direction vector + commanded power (for conditioning, Phase 2+)
- Phase/time-in-episode scalar

**Actions:** joint torques or position targets (Isaac Lab humanoid examples use torque control — keep their choice); action dim = DoF of the articulation.

**Ball + tee:** the ball is a real-spec sphere (r=21.34 mm, 45.9 g) with **gravity disabled**, and the tee is a **visual-only** prim with no collider. Phase 2 owns flight analytically, so PhysX only has to report the impact — a gravity-free ball needs no support, which also means the club can never strike the tee (a static tee would deflect the clubhead like a steel post). Stage 4.5 turns gravity back on and rests the ball on the ground.

**Target line:** the golfer faces **+X** with the ball in front of its chest, and a right-hander sends the ball to its **left, so the target direction is +Y** — not +X. The swing plane is perpendicular to the stance, which is easy to get backwards.
**Measured 2026-10-29: the solved address pose does not match this.** It faces **−Y** (head leans −Y of the pelvis, shoulder and hip lines run along X, feet point along Y) with the ball off its *left* side. `solve_address.py` constrains only hand position, foot height and torso/pelvis heights, so heading was left free. Re-solve with a heading term before any direction reward means anything.

**Episode:** reset with agent addressing a teed ball; terminate on timeout (~5 s) or N steps post-impact. **No fall termination in Stage 0** (see curriculum); add it only once standing is being rewarded.

**Parallelism:** start at 1,024 envs, scale toward 4,096 as VRAM allows. Headless, fixed small dt with substeps; increase solver iterations around contact. If impact is still noisy, detect clubhead-ball proximity and read velocities at that tick rather than trusting the collision impulse.

## Reward curriculum (the actual RL lesson — stage gates, not one reward)

1. **Stage 0 — swing at all:** reward **ball speed gained** (the real objective; the ball is gravity-free and starts at rest, so any speed is a genuine strike) + a **capped backswing** term for winding up + clubhead speed near the ball. **No alive bonus, no fall penalty, no fall termination** — swinging is the first goal, and if the golfer flings itself over to do it, fine.
   **Do NOT use a clubhead-to-ball distance ("approach") reward here.** The club starts at address ~6 cm from the ball, so distance reduction is optimised by standing perfectly still, and it penalises the backswing — the one thing a swing must do first. Measured: reward fell −1.2 → −7.0 and clubhead speed fell 5.9 → 3.7 m/s as the policy learned to freeze. A distance term only makes sense if the club starts away from the ball.
   Contact detection needs **arming**: the clubhead must travel >0.25 m from the ball before a hit counts, or `contact_rate` reads 1.0 from iteration 0 and measures nothing.
   **"Any speed is a genuine strike" is false (measured 2026-10-29).** At iter 199, 45 of 52 armed strikes were the left shin/foot — the policy kicks the ball — and 1 in 4–8 resets has the clubhead knocking the ball off the tee. The payout has to check *what* hit the ball, not only when. Details in `docs/experiments.md`.
2. **Stage 1 — contact:** big sparse bonus for club-ball contact; keep stage 0 shaping at reduced weight.
3. **Stage 2 — ball speed:** reward ball speed at/after impact. Expect violent, ugly swings — fine.
4. **Stage 3 — direction:** reward ball-velocity alignment with target direction; penalize the offline component.
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
- **Watch the policy before trusting a reward curve.** `scripts/record_swings.py <run dir> physics=isaacsim_physx` rolls out every checkpoint headless and writes one self-contained HTML viewer to `<run>/swings/` (scrub, slow-mo, camera presets, which body part struck the ball). For a Kit render: `isaaclab play --rl_library rsl_rl --task Isaac-Golf-Swing-Direct --num_envs 1 --checkpoint <path/to/model_N.pt> --video --video_length 300 physics=isaacsim_physx` — needs the `video` extra (in `pyproject.toml`); 3.0's `play` has no `--load_run`.

## Known traps (pre-answered — no session relitigates them)

- **Don't simulate ball aerodynamics in PhysX** — no Magnus lift; the flight model owns everything after impact.
- **Don't learn grasping** — weld the club.
- **Don't train with the GUI open** — headless, or it will crawl and eat RAM.
- **Don't end-to-end the whole golf round** — hierarchy per Phase 3.
- **Don't trust raw contact impulses at impact** — small dt/substeps, or sample clubhead velocity at proximity.
- **Don't start with a sparse ball-in-hole reward** — curriculum above.
- **Club welds to `left_hand` only** (lead hand for a right-handed swing). The humanoid *does* have `left_hand`/`right_hand` bodies — 4 cm spheres on fixed joints, zero DoF, ideal anchors. Welding **both** hands closes a kinematic loop (torso → left arm → club → right arm → torso) which PhysX must solve as an extra constraint at only 4 position iterations: sag, jitter, blowup. The trail (right) hand is kept on the grip by a **reward penalty**, not a constraint — tree stays intact and the policy learns to hold on. A soft physical D6 spring is a later logged experiment, not the default.
- **Don't fork blind:** env logic lives in `locomotion_direct_env.py`, not the humanoid files.
- **Don't copy Isaac Lab 2.x tutorials verbatim** — 3.0 moved asset data behind `.torch` accessors, dropped `--headless`, and dropped the `-v0` task suffix.
- **Version drift:** Isaac Lab ↔ Isaac Sim ↔ driver versions get pinned once working; upgrades are deliberate, between phases, never mid-training.

## Milestones

- [x] **Phase 0:** Isaac Lab 3.0 + Isaac Sim 6.2 installed; stock humanoid trains headless at 4096 envs, ~160k env-steps/s on PhysX, 4.9 GB VRAM (`docs/experiments.md`)
- [x] Fork humanoid env → golf env skeleton: teed ball + welded club + reset/termination (`envs/golf/`, `assets/golfer.usda`; address pose + weld solved by `scripts/solve_address.py`)
- [ ] Stage 0–1 training: agent reliably contacts the ball
- [ ] Stage 2–3: fast, directed ball at impact
- [ ] Phase 2 flight model (unit-tested) wired to impact state; carry distances plausible per club
- [ ] Stage 4: power/direction-conditioned swing policy
- [ ] Phase 3: first designed hole in USD; heuristic club/aim layer; complete a hole
- [ ] Stretch: AMP styling; multi-hole course; replay rendering

First milestone that counts: **a headless training run where contact rate climbs.** Everything before that is setup; everything after is iteration.
