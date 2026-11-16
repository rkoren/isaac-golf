"""Golfer articulation: stock mujoco humanoid with a club welded to the left hand."""

from __future__ import annotations

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .clubs import address, golfer_usd

CLUB_BODY = "club"
LEAD_HAND = "left_hand"
TRAIL_HAND = "right_hand"

DRIVER = address("driver")
# [N*m*s/rad] stock humanoid joint damping scaled by GolfEnvCfg.damping_scale / leg_damping_scale
SWING_DAMPING = {".*_waist.*": 5.0, ".*_upper_arm.*": 5.0, "pelvis": 5.0, ".*_lower_arm": 1.0}
LEG_DAMPING = {".*_thigh:0": 5.0, ".*_thigh:1": 5.0, ".*_thigh:2": 5.0, ".*_shin": 0.1, ".*_foot.*": 1.0}
DAMPING = {**SWING_DAMPING, **LEG_DAMPING}
# [N*m/rad] stock springs toward joint zero
STIFFNESS = {
    ".*_waist.*": 20.0, ".*_upper_arm.*": 10.0, "pelvis": 10.0, ".*_lower_arm": 2.0,
    ".*_thigh:0": 10.0, ".*_thigh:1": 20.0, ".*_thigh:2": 10.0,
    ".*_shin": 5.0, ".*_foot.*": 2.0,
}
# lead wrist (env.wrist)
WRIST_GEAR = 10.0       # [N*m]
WRIST_VMAX = 30.0       # [rad/s]
WRIST_STIFFNESS = 2.0   # [N*m/rad] toward the address angle
WRIST_DAMPING = 0.5     # [N*m*s/rad] scaled with the swing joints

GOLFER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=golfer_usd("driver"),
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=None, max_depenetration_velocity=1.0, enable_gyroscopic_forces=True
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=tuple(DRIVER["root_pos"]),
        rot=tuple(DRIVER["root_rot"]),  # (x, y, z, w)
        joint_pos=DRIVER["joint_pos"],
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=STIFFNESS,
            damping=DAMPING,
            joint_velocity_limit={".*": 100.0},
            armature={".*": 0.01},
            joint_effort_limit=150.0,
        ),
    },
)
