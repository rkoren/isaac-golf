# Pinned versions

Working stack. Upgrades are deliberate, between phases, never mid-training.

| Thing | Version | Note |
|---|---|---|
| Isaac Lab | 3.0.0 (pip `isaaclab`) | not a repo clone — plain uv dependency |
| Isaac Sim | 6.2 (pip wheels, via `isaacsim` extra) | pulled by uv, lives in `.venv` |
| Python | 3.12 | Isaac Lab 3.0 requires exactly 3.12 |
| torch | 2.11.0+cu130 | cu130 line for Blackwell sm_120 |
| NVIDIA driver | 610.88 (CUDA UMD 13.3) | |
| GPU | RTX 5070 Ti 16 GB | |
| CPU / RAM | Ryzen 5700X / 32 GB | 32 GB is Isaac's floor: never GUI + training together |
| RL stack | rsl_rl (PPO), via `rsl-rl` extra | |

Real pin is `uv.lock`. `pyproject.toml` holds the index config.

## Install gotchas

- **urllib3 conflict (upstream, unavoidable).** `isaacsim-kernel==6.2.0.0` pins `urllib3==2.7.0`;
  `isaaclab==3.0.0` requires `>=2.8.0`. No patch release of either fixes it (3.0.0 is the only 3.x on PyPI).
  Workaround lives in `pyproject.toml`: `[tool.uv] override-dependencies = ["urllib3==2.8.0"]`.
  Honors isaaclab's floor; the kernel's exact pin is a build-env artifact.
- Long-path support must be on (it already was). Isaac's Kit paths blow past 260 chars.
- Install torch *before* isaaclab so the cu130 wheels win.

## Compatibility table (Isaac Lab -> Isaac Sim), for future upgrades

| Isaac Lab | Isaac Sim |
|---|---|
| `main` / `release/3.0.0` | 6.2 |
| `v3.0.0-EA` | 6.1 |
| `v3.0.0-beta2` | 6.0 |
| `v2.3.x` | 4.5 / 5.0 / 5.1 |

## Physics backends

Isaac Lab 3.0 is multi-backend, picked per run with a Hydra override. We installed both.

- **PhysX (Isaac Sim)** — Phase 1 default. Chosen for Phase 3's USD course work and GUI replay, not for speed.
- **Newton / MJWarp** — measured *faster* than PhysX on the stock humanoid (234k vs 160k steps/s,
  2.0 vs 4.9 GB VRAM — see `experiments.md`). Runs without Isaac Sim at all. Treat it as a live
  alternative, not a last resort; revisit at the Stage 2 gate when impact fidelity starts to matter.

Preset names for `physics=` (from `--task=X --help`): `isaacsim_physx`, `ovphysx`, `physx` (auto), `newton_mjwarp`.
**`default` is `newton_mjwarp`** — PhysX must be asked for explicitly every run.

## Phase 1 fork target

Stock humanoid lives in `.venv/Lib/site-packages/isaaclab_tasks/core/locomotion/`:

| File | What it is | Phase 1 plan |
|---|---|---|
| `humanoid/humanoid_direct_env_cfg.py` | `HumanoidEnvCfg` — dims, sim cfg, reward scales, termination | fork, swap reward scales for the curriculum |
| `humanoid/humanoid_direct_env.py` | 20 lines, just binds cfg to base class | fork, trivial |
| `humanoid/humanoid_common.py` | terrain, `JOINT_GEARS`, feet names, `HumanoidPhysicsCfg(PresetCfg)` | reuse gears/terrain, keep the PresetCfg pattern |
| `locomotion_direct_env.py` | `LocomotionDirectEnv` — the actual env logic (259 lines) | **this is the real fork**: obs, rewards, dones, reset |
| `humanoid/agents/rsl_rl_ppo_cfg.py` | PPO hyperparams | copy, tune later |

Stock humanoid numbers: `action_space=21` (torques, scaled per joint by `JOINT_GEARS`), `observation_space=87`,
`dt=1/120`, `decimation=2`, `episode_length_s=16`, default `num_envs=4096`, terminate below `0.8 m`.

Hooks to override (DirectRLEnv): `_pre_physics_step`, `_apply_action`, `_compute_intermediate_values`,
`_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`. Rewards are a module-level
`compute_rewards` function — easy to replace per curriculum stage.

Body/asset facts for the club weld (`isaaclab_assets/robots/humanoid.py`):
- Asset is `humanoid_instanceable.usd` pulled from `ISAAC_NUCLEUS_DIR` (remote Nucleus), not a local file.
  First run downloads it. If Nucleus is ever flaky we vendor a copy into `assets/`.
- **The MuJoCo humanoid has no hands** — arms end at `left_lower_arm` / `right_lower_arm`.
  Those are the fixed-joint weld targets for the club. Confirms CLAUDE.md's "never learn grip".
- Actuators are `ImplicitActuatorCfg` over `.*` with per-joint stiffness/damping, `joint_effort_limit=150`;
  the env drives them as efforts scaled by `JOINT_GEARS`.
- Spawn pose `z=1.34`, all joints 0 — standing. Golf needs an address pose instead: set `init_state` joint_pos.

Two API details that differ from Isaac Lab 2.x tutorials found online:
- Asset data is backend-agnostic now — read tensors via `.torch`, e.g. `self.robot.data.joint_pos.torch`.
- The stock env shapes "get closer to target" as a **potential**: `potential = -dist/step_dt`, reward on
  `potential - prev_potential`. Reuse this shape for Stage 0's clubhead-to-ball distance instead of raw distance.

`PresetCfg` is how one cfg serves both backends: named fields (`isaacsim_physx`, `newton_mjwarp`, ...)
plus a `default`, selected at launch by `physics=NAME`. Keep this in the golf env so the Newton fallback stays free.

## Commands

Everything runs through `uv run --env-file omni.env`. `omni.env` carries the EULA accept + UTF-8 fix.

```powershell
# train (headless is the default; GUI is opt-in via --viz)
uv run --env-file omni.env isaaclab train --rl_library rsl_rl --task Isaac-Humanoid-Direct --num_envs 4096 --max_iterations 50 physics=isaacsim_physx

# inspect / replay (never while training, 32 GB)
uv run --env-file omni.env isaaclab play --rl_library rsl_rl --task Isaac-Humanoid-Direct --num_envs 16 --checkpoint latest --viz kit physics=isaacsim_physx

# tensorboard
uv run python -m tensorboard.main --logdir logs
```

CLI notes (3.0 differs from older docs and from Isaac Lab 2.x):
- **No `--headless`.** Headless is default; pass `--viz kit|newton|rerun|viser` to see anything.
- Task IDs have **no `-v0` suffix**: `Isaac-Humanoid-Direct`, `Isaac-Humanoid` (manager-based), `Isaac-Cartpole`.
- AMP humanoid tasks (Stage 5 stretch) are namespaced `contrib` and are **skrl-only**:
  `IsaacContrib-Humanoid-AMP-{Walk,Run,Dance}-Direct`. rsl_rl can't drive them.
- `isaaclab` subcommands: `train`, `play`, `list_envs`, `zero_agent`, `random_agent`, `benchmark`.
- Programmatic entry exists too: `from isaaclab_rl import TrainingRequest, train`.
- `play` works with `--checkpoint latest` (verified headless; it loads the policy then loops forever).
- Remote assets are cached under `%LOCALAPPDATA%\Temp\https\...`. Temp cleaners will nuke it and the
  next run re-downloads. Not a problem, just don't mistake it for a failure.

Warnings seen on every rsl_rl run, harmless but worth fixing in our own cfg in Phase 1:
`obs_groups` is unset, so rsl_rl guesses `actor`/`critic` from the `policy` group. Set it explicitly —
the fallback is scheduled for removal.
