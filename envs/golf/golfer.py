"""Golfer articulation: stock mujoco humanoid with a club welded to the left hand."""

from __future__ import annotations

from pathlib import Path

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

GOLFER_USD = str(Path(__file__).resolve().parents[2] / "assets" / "golfer.usda")

CLUB_BODY = "club"
LEAD_HAND = "left_hand"
TRAIL_HAND = "right_hand"

ADDRESS_JOINT_POS = {
    "lower_waist:0": 0.015,
    "lower_waist:1": -0.642,
    "pelvis": 0.041,
    "right_thigh:0": -0.142,
    "left_thigh:0": -0.157,
    "right_thigh:1": -0.531,
    "left_thigh:1": -0.528,
    "right_foot:0": -0.156,
    "right_foot:1": 0.053,
    "left_foot:0": -0.150,
    "left_foot:1": -0.058,
    "right_shin": -0.497,
    "left_shin": -0.492,
    "left_upper_arm:0": 0.750,
    "left_upper_arm:2": 0.022,
    "left_lower_arm": -0.758,
    "right_upper_arm:0": 0.848,
    "right_upper_arm:2": 0.066,
    "right_lower_arm": -0.803,
}

GOLFER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=GOLFER_USD,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=None, max_depenetration_velocity=10.0, enable_gyroscopic_forces=True
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
        pos=(0.123, 0.001, 1.178),
        rot=(-0.0190, 0.4061, -0.0118, 0.9135),  # (x, y, z, w)
        joint_pos=ADDRESS_JOINT_POS,
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                ".*_waist.*": 20.0, ".*_upper_arm.*": 10.0, "pelvis": 10.0, ".*_lower_arm": 2.0,
                ".*_thigh:0": 10.0, ".*_thigh:1": 20.0, ".*_thigh:2": 10.0,
                ".*_shin": 5.0, ".*_foot.*": 2.0,
            },
            damping={
                ".*_waist.*": 5.0, ".*_upper_arm.*": 5.0, "pelvis": 5.0, ".*_lower_arm": 1.0,
                ".*_thigh:0": 5.0, ".*_thigh:1": 5.0, ".*_thigh:2": 5.0,
                ".*_shin": 0.1, ".*_foot.*": 1.0,
            },
            joint_velocity_limit={".*": 100.0},
            armature={".*": 0.01},
            joint_effort_limit=150.0,
        ),
    },
)
