# CLAUDE.md — isaac-golf

Side project: teach a stick-figure agent to swing golf clubs and play a designed course, using RL in NVIDIA Isaac Lab. Primary goal is **learning RL in simulated spaces** — a working, ugly swing beats a pretty stalled project.

# Leave the README plain

## Environment

- **Isaac Sim 6.0** installed at `C:\isaacsim` (workstation binary install). Isaac Lab NOT yet installed — Phase 0 task. Check Isaac Lab's compatibility matrix against Isaac Sim 6.0 and the installed driver before anything else; pin versions in `docs/versions.md` once working.
- **Hardware:** RTX 5070 Ti 16 GB (Blackwell), Ryzen 5700X, 32 GB RAM. Training runs **headless** (`--headless`); GUI only for inspection/replays. 32 GB RAM is at Isaac's floor — avoid GUI + training simultaneously.
- **Dedicated conda env** (`isaac-golf`, or whatever Isaac Lab's installer dictates — likely a pinned Python 3.11/3.12). Never share the `ml` env; Isaac pins its own torch.
- RL stacks: use what Isaac Lab ships (**rsl_rl** or **skrl**, PPO). Do not hand-roll PPO.
- Long runs in a detached terminal; checkpoints + TensorBoard logs to `runs/` (gitignored).

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
- **Version drift:** Isaac Lab ↔ Isaac Sim ↔ driver versions get pinned once working; upgrades are deliberate, between phases, never mid-training.

## Milestones

- [ ] **Phase 0:** Isaac Lab installed against Isaac Sim 6.0; stock humanoid example trains headless on this machine (sanity: steps/sec logged)
- [ ] Fork humanoid env → golf env skeleton: teed ball + welded club + reset/termination
- [ ] Stage 0–1 training: agent reliably contacts the ball
- [ ] Stage 2–3: fast, directed ball at impact
- [ ] Phase 2 flight model (unit-tested) wired to impact state; carry distances plausible per club
- [ ] Stage 4: power/direction-conditioned swing policy
- [ ] Phase 3: first designed hole in USD; heuristic club/aim layer; complete a hole
- [ ] Stretch: AMP styling; multi-hole course; replay rendering

First milestone that counts: **a headless training run where contact rate climbs.** Everything before that is setup; everything after is iteration.
