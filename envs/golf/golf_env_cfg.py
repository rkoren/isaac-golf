"""Config for the Phase 1 golf swing"""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.core.locomotion.humanoid.humanoid_common import JOINT_GEARS, TERRAIN_CFG
from isaaclab_tasks.utils import PresetCfg

from .clubs import CLUBS, WRIST, address, golfer_usd
from .golfer import (
    GOLFER_CFG, LEG_DAMPING, STIFFNESS, SWING_DAMPING, WRIST_DAMPING, WRIST_GEAR, WRIST_STIFFNESS, WRIST_VMAX,
)

# golf ball
BALL_R = 0.02134
BALL_MASS = 0.0459
MPH = 2.23694  # m/s -> mph, for reporting only

BALL_POS = tuple(address("driver")["ball_pos"])  # 2 cm from clubface
TEE_H = BALL_POS[2] - BALL_R   # tee

# limiting movement
MUSCLE_VMAX = {
    ".*_waist.*": 20.0, "pelvis": 20.0, ".*_upper_arm.*": 25.0, ".*_lower_arm": 25.0,
    ".*_thigh.*": 25.0, ".*_shin": 30.0, ".*_foot.*": 25.0,
}

GRASS = (0.30, 0.62, 0.22)
BALL_COLOR = (1.0, 0.1, 0.65)
BALL_GLOW = (1.6, 0.1, 1.0)


@configclass
class GolfPhysicsCfg(PresetCfg):
    """Backend presets"""

    isaacsim_physx: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.2,
        min_position_iteration_count=8,
    )
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=80, nconmax=25, cone="pyramidal", update_data_interval=2,
            integrator="implicitfast", impratio=1,
        ),
        num_substeps=4,
        debug_mode=False,
    )
    default: PhysxCfg = isaacsim_physx


@configclass
class GolfSceneCfg(InteractiveSceneCfg):
    terrain = replace(TERRAIN_CFG, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GRASS))
    robot: ArticulationCfg = replace(GOLFER_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    # gravity off for now
    # no tee collision yet
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_R,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
            # no collider as env computes the launch
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BALL_COLOR, emissive_color=BALL_GLOW),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_POS),
    )
    # visual only
    tee: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tee",
        spawn=sim_utils.CylinderCfg(
            radius=0.004, height=TEE_H, axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(BALL_POS[0], BALL_POS[1], TEE_H / 2)),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class GolfEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 5.0
    decimation = 2
    action_scale = 1.0
    action_space = 21
    observation_space = 81
    state_space = 0

    # down-the-line camera for env 0 (--num_envs 1); base cfg so --video still injects its visualizer
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, physics=GolfPhysicsCfg(),
        default_visualizer_cfg=VisualizerCfg(eye=(0.45, -3.2, 1.4), lookat=(0.6, 1.5, 0.5)),
    )

    scene: GolfSceneCfg = GolfSceneCfg(
        num_envs=4096, env_spacing=5.0, replicate_physics=True, clone_in_fabric=True
    )

    joint_gears: dict[str, float] = JOINT_GEARS
    muscle: bool = False
    muscle_vmax: dict[str, float] = MUSCLE_VMAX
    muscle_k: float = 0.25              # Hill curvature (~0.25 for human muscle)
    damping_scale: float = 1.0          # joint damping
    leg_damping_scale: float | None = None  # hips, knees, ankles; None = damping_scale
    wrist: bool = False                 # lead wrist hinge between hand and club (22 actions)
    club: str = "driver"                # key into clubs.py; apply_club() sets everything club-specific
    ball_pos: tuple[float, float, float] = BALL_POS
    # aim based on golfer (left for a righty)
    target_dir: tuple[float, float, float] = (0.0, 1.0, 0.0)

    # reset jitter
    initial_joint_pos_range: tuple[float, float] = (-0.05, 0.05)
    initial_joint_vel_range: tuple[float, float] = (-0.05, 0.05)

    # stage 0 reward + balance; no alive bonus, no fall termination
    # at first model learned to just stand there
    ball_speed_weight: float = 3.0      # ball speed gained (goal: ball go fast)
    clubhead_speed_weight: float = 0.3  # dense shaping: move head fast near ball
    speed_radius: float = 0.35          # "near the ball" band [m]
    backswing_weight: float = 0.6       # pay for winding up, capped so cannot farm it
    backswing_cap: float = 0.60         # [m] clubhead-ball distance worth paying
    trail_hand_weight: float = 0.2      # encourage right hand on the grip
    # balance: 0 from scratch, 10 when resuming from a striking checkpoint
    stand_weight: float = 0.0           # torso below stand_height [per m]
    stand_height: float = 0.95          # [m] address torso is ~1.18
    feet_slip_weight: float = 0.0       # foot slide beyond feet_slip_free [per m]
    feet_slip_free: float = 0.03        # [m]
    actions_cost_scale: float = 0.005
    energy_cost_scale: float = 0.002
    joint_pos_limits_cost_scale: float = 0.25
    joint_pos_limits_threshold: float = 0.98

    # impact
    impact_radius: float = 0.07         # [m] ball r + half head length
    impact_min_speed: float = 1.0       # [m/s] head speed into the ball
    impact_arm_dist: float = 0.25       # [m] how close to count as hit
    post_impact_steps: int = 60         # end episode this long after first contact (1 s finish)

    # launch = smash x head speed into the face
    smash_max: float = 1.47             # driver, dead centre
    smash_min: float = 1.0              # topped, skulled, or back of head
    smash_sigma_toe: float = 0.022      # [m] toe-heel falloff
    smash_sigma_high: float = 0.011     # [m] up-down falloff (face is half as tall)
    face_launch_weight: float = 0.85    # start direction: face vs path
    backwards_penalty: float = 20.0     # one-off: back of head or ball heading away from target
    finish_frac: float = 0.0            # share of ball pay held until the finish, scaled by posture
    finish_low: float = 0.6             # [m] torso height at the finish that earns nothing
    finish_high: float = 1.0            # [m] torso height that earns full finish credit
    aim_sigma: float = 0.0              # [deg] ball pay falls off with angle off the target line

    # obs scales
    dof_vel_scale: float = 0.1
    angular_velocity_scale: float = 0.25
    clubhead_vel_scale: float = 0.05


def apply_club(cfg: GolfEnvCfg) -> None:
    """Point the scene at cfg.club: golfer asset, address pose, ball, tee, smash ceiling."""
    spec, addr = CLUBS[cfg.club], address(cfg.club)
    robot = cfg.scene.robot
    robot.spawn.usd_path = golfer_usd(cfg.club, cfg.wrist)
    robot.init_state.pos = tuple(addr["root_pos"])
    robot.init_state.rot = tuple(addr["root_rot"])
    robot.init_state.joint_pos = addr["joint_pos"]
    cfg.ball_pos = tuple(addr["ball_pos"])
    cfg.scene.ball.init_state.pos = cfg.ball_pos
    if spec.tee:
        h = cfg.ball_pos[2] - BALL_R
        cfg.scene.tee.spawn.height = h
        cfg.scene.tee.init_state.pos = (cfg.ball_pos[0], cfg.ball_pos[1], h / 2)
    else:
        cfg.scene.tee = None
    cfg.smash_max = spec.smash_max


def apply_body(cfg: GolfEnvCfg) -> None:
    """Joint damping, and the lead wrist's actuator and spaces when cfg.wrist."""
    act = cfg.scene.robot.actuators["body"]
    legs = cfg.damping_scale if cfg.leg_damping_scale is None else cfg.leg_damping_scale
    act.damping = {
        **{k: v * cfg.damping_scale for k, v in SWING_DAMPING.items()},
        **{k: v * legs for k, v in LEG_DAMPING.items()},
    }
    if not cfg.wrist:
        return
    act.damping[WRIST] = WRIST_DAMPING * cfg.damping_scale
    act.stiffness = {**STIFFNESS, WRIST: WRIST_STIFFNESS}
    cfg.joint_gears = {**JOINT_GEARS, WRIST: WRIST_GEAR}
    cfg.muscle_vmax = {**MUSCLE_VMAX, WRIST: WRIST_VMAX}
    cfg.action_space = 22
    cfg.observation_space = 84  # +1 joint pos, vel and action
