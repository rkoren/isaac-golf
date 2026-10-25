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

**Agent body:** simplified humanoid articulation (USD or MJCF/URDF). Start with the stock Isaac Lab humanoid (~21–28 DoF) even though it's more than a "stick figure" — a working asset beats a custom one. Reduce DoF later only if training struggles. Club: rigid capsule/box welded to both hands as a fixed joint. **Grasping is out of scope — never make the agent learn grip.**

**Observations (per env):**
- Joint positions + velocities (normalized)
- Torso/pelvis orientation (quat or 6D) + linear/angular velocity
- Clubhead position + velocity relative to ball
- Ball position (and velocity once struck, Phase 1 only)
- Target direction vector + commanded power (for conditioning, Phase 2+)
- Phase/time-in-episode scalar

**Actions:** joint torques or position targets (Isaac Lab humanoid examples use torque control — keep their choice); action dim = DoF of the articulation.

**Episode:** reset with agent addressing a teed ball; terminate on fall (pelvis below threshold), N seconds post-impact, or timeout (~5–8 s sim time).

**Parallelism:** start at 1,024 envs, scale toward 4,096 as VRAM allows. Headless, fixed small dt with substeps; increase solver iterations around contact. If impact is still noisy, detect clubhead-ball proximity and read velocities at that tick rather than trusting the collision impulse.

## Reward curriculum (the actual RL lesson — stage gates, not one reward)

1. **Stage 0 — stay up & reach:** alive bonus + fall penalty + reward for decreasing clubhead-to-ball distance. Expect flailing.
2. **Stage 1 — contact:** big sparse bonus for club-ball contact; keep stage 0 shaping at reduced weight.
3. **Stage 2 — ball speed:** reward ball speed at/after impact. Expect violent, ugly swings — fine.
4. **Stage 3 — direction:** reward ball-velocity alignment with target direction; penalize the offline component.
5. **Stage 4 — distance/carry (via Phase 2 flight model):** reward carry toward target; then target-distance accuracy for partial shots (power conditioning).
6. **Stage 5 (optional stretch) — style:** AMP with a golf-swing motion clip so it looks like golf. Only after Stage 4 works.

Promote stages by checkpoint: train, evaluate, freeze, add the next term, continue. Log every stage in `docs/experiments.md` (config, reward curve, verdict, one-line takeaway).

Standard shaping terms kept throughout: action-rate penalty (smoothness), small torque/energy penalty, joint-limit penalty. Tune weights so the task reward dominates.

## Repo conventions

- Layout: `envs/` (env + reward code), `assets/` (USD/URDF — small ones committable), `scripts/` (train/eval/replay entrypoints), `flight/` (Phase 2 analytic model + unit tests — pure Python, testable without Isaac), `docs/` (versions, experiments log), `runs/` (gitignored).
- `.gitignore`: `runs/`, checkpoints (`*.pt`, `*.onnx`), TensorBoard events, `__pycache__/`, large USD/binary assets, Omniverse cache dirs.
- The flight model is ordinary physics code: write it test-first (known inputs → known carry distances from published golf data) before wiring it to Isaac.
- Commit small, push each session. Experiments without a logged config/result didn't happen.

## Known traps (pre-answered — no session relitigates them)

- **Don't simulate ball aerodynamics in PhysX** — no Magnus lift; the flight model owns everything after impact.
- **Don't learn grasping** — weld the club.
- **Don't train with the GUI open** — headless, or it will crawl and eat RAM.
- **Don't end-to-end the whole golf round** — hierarchy per Phase 3.
- **Don't trust raw contact impulses at impact** — small dt/substeps, or sample clubhead velocity at proximity.
- **Don't start with a sparse ball-in-hole reward** — curriculum above.
- **Don't fork blind:** the stock humanoid has **no hands** (arms end at `left_lower_arm`/`right_lower_arm`) — those are the club's weld targets. Env logic lives in `locomotion_direct_env.py`, not the humanoid files.
- **Don't copy Isaac Lab 2.x tutorials verbatim** — 3.0 moved asset data behind `.torch` accessors, dropped `--headless`, and dropped the `-v0` task suffix.
- **Version drift:** Isaac Lab ↔ Isaac Sim ↔ driver versions get pinned once working; upgrades are deliberate, between phases, never mid-training.

## Milestones

- [x] **Phase 0:** Isaac Lab 3.0 + Isaac Sim 6.2 installed; stock humanoid trains headless at 4096 envs, ~160k env-steps/s on PhysX, 4.9 GB VRAM (`docs/experiments.md`)
- [ ] Fork humanoid env → golf env skeleton: teed ball + welded club + reset/termination
- [ ] Stage 0–1 training: agent reliably contacts the ball
- [ ] Stage 2–3: fast, directed ball at impact
- [ ] Phase 2 flight model (unit-tested) wired to impact state; carry distances plausible per club
- [ ] Stage 4: power/direction-conditioned swing policy
- [ ] Phase 3: first designed hole in USD; heuristic club/aim layer; complete a hole
- [ ] Stretch: AMP styling; multi-hole course; replay rendering

First milestone that counts: **a headless training run where contact rate climbs.** Everything before that is setup; everything after is iteration.
