"""Phase 1 golf swing env. Stage 0 goal: swing at all. Falling over is allowed."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import index_fill_
from isaaclab.utils.math import normalize, sample_uniform, scale_transform
from isaaclab.utils.string import resolve_matching_names_values

from .golf_env_cfg import GolfEnvCfg
from .golfer import CLUB_BODY, LEAD_HAND, TRAIL_HAND


class GolfEnv(DirectRLEnv):
    cfg: GolfEnvCfg

    def __init__(self, cfg: GolfEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot, self.ball = self.scene["robot"], self.scene["ball"]
        self.action_scale = self.cfg.action_scale

        # gears by name: joint ordering differs per backend
        self.joint_gears = torch.ones(self.robot.num_joints, device=self.device)
        joint_ids, _, gears = resolve_matching_names_values(self.cfg.joint_gears, self.robot.joint_names)
        self.joint_gears[joint_ids] = torch.tensor(gears, device=self.device)
        self.gear_ratio_scaled = self.joint_gears / torch.max(self.joint_gears)

        body_idx, _ = self.robot.find_bodies([CLUB_BODY, LEAD_HAND, TRAIL_HAND], preserve_order=True)
        self._club_idx, self._lead_idx, self._trail_idx = body_idx

        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.target_dir = normalize(
            torch.tensor(self.cfg.target_dir, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        )
        # per-episode bookkeeping, logged at reset: this is the "contact rate climbs" signal
        self.impacted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.post_impact = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.min_dist = torch.full((self.num_envs,), 1e3, device=self.device)
        self.impact_speed = torch.zeros(self.num_envs, device=self.device)
        self.armed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.backswing = torch.zeros(self.num_envs, device=self.device)
        self.max_ball_speed = torch.zeros(self.num_envs, device=self.device)

        self._compute_intermediate_values()
        width = self._get_observations()["policy"].shape[-1]
        assert width == self.cfg.observation_space, (
            f"observation_space={self.cfg.observation_space} but obs is {width} wide"
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # clamp before scaling: unbounded efforts drive the solver to NaN
        forces = self.action_scale * self.joint_gears * torch.clamp(self.actions, -1.0, 1.0)
        self.robot.actuators.target_command.set_effort_index(value=forces)

    def _apply_action(self) -> None:
        pass

    def _compute_intermediate_values(self):
        self.torso_pos = self.robot.data.root_pos_w.torch
        self.dof_pos_scaled = scale_transform(
            self.robot.data.joint_pos.torch,
            self.robot.data.soft_joint_pos_limits.torch[..., 0],
            self.robot.data.soft_joint_pos_limits.torch[..., 1],
        )
        self.dof_vel = self.robot.data.joint_vel.torch

        # link frame, not com. the club frame origin IS the clubhead, and for a rotating
        # club the com velocity badly understates head speed.
        self.club_pos = self.robot.data.body_link_pos_w.torch[:, self._club_idx]
        self.club_vel = self.robot.data.body_link_lin_vel_w.torch[:, self._club_idx]
        self.lead_pos = self.robot.data.body_link_pos_w.torch[:, self._lead_idx]
        self.trail_pos = self.robot.data.body_link_pos_w.torch[:, self._trail_idx]

        self.ball_pos = self.ball.data.root_pos_w.torch
        self.ball_vel = self.ball.data.root_lin_vel_w.torch

        self.club_ball_dist = torch.linalg.norm(self.ball_pos - self.club_pos, dim=-1)
        self.club_speed = torch.linalg.norm(self.club_vel, dim=-1)

        # impact by proximity, NOT by contact impulse: at dt=1/120 a fast clubhead crosses
        # ~0.4 m per step, so the impulse the solver reports is not trustworthy
        # arm on the backswing: without this, contact_rate is 1.0 from iteration 0 because
        # the head starts inside the impact radius
        self.armed |= self.club_ball_dist > self.cfg.impact_arm_dist
        hit = (
            (self.club_ball_dist < self.cfg.impact_radius)
            & (self.club_speed > self.cfg.impact_min_speed)
            & self.armed
            & ~self.impacted
        )
        self.impact_speed = torch.where(hit, self.club_speed, self.impact_speed)
        self.impacted |= hit
        self.post_impact += self.impacted.long()
        self.min_dist = torch.minimum(self.min_dist, self.club_ball_dist)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = torch.cat(
            (
                self.torso_pos[:, 2].view(-1, 1),
                self.robot.data.root_lin_vel_b.torch,
                self.robot.data.root_ang_vel_b.torch * self.cfg.angular_velocity_scale,
                (-self.robot.data.projected_gravity_b.torch[:, 2]).view(-1, 1),
                self.dof_pos_scaled,
                self.dof_vel * self.cfg.dof_vel_scale,
                self.actions,
                self.club_pos - self.ball_pos,
                self.club_vel * self.cfg.clubhead_vel_scale,
                self.ball_pos - self.torso_pos,
                (self.episode_length_buf / self.max_episode_length).view(-1, 1),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 1. the real objective: ball speed gained. the ball is gravity-free and starts at
        #    rest, so any speed at all means a genuine strike. paid on the increase only, so
        #    it cannot be farmed by coasting after impact.
        # gated on `armed`: ungated, the policy just pokes the ball from address and never
        # swings (measured: contact_rate went 0 -> 0.33 -> 0 while ball speed kept climbing).
        # only a strike that backswung first gets paid.
        ball_speed = torch.linalg.norm(self.ball_vel, dim=-1)
        gained = torch.relu(ball_speed - self.max_ball_speed) * self.armed.float()
        self.max_ball_speed = torch.maximum(self.max_ball_speed, ball_speed)
        ball_reward = self.cfg.ball_speed_weight * gained / self.step_dt

        # 2. wind up. paid on progress toward the cap, so it is dense but bounded.
        capped = torch.clamp(self.club_ball_dist, max=self.cfg.backswing_cap)
        wind = self.cfg.backswing_weight * torch.relu(capped - self.backswing) / self.step_dt
        self.backswing = torch.maximum(self.backswing, capped)

        # 3. dense shaping: swing the head fast while it is near the ball
        near = (self.club_ball_dist < self.cfg.speed_radius).float()
        speed = self.cfg.clubhead_speed_weight * self.club_speed * near

        # trail hand stays on the grip. a reward, not a joint: welding both hands closes a loop
        trail = self.cfg.trail_hand_weight * torch.linalg.norm(self.trail_pos - self.lead_pos, dim=-1)

        actions_cost = torch.sum(self.actions**2, dim=-1)
        energy_cost = torch.sum(torch.abs(self.actions * self.dof_vel * self.gear_ratio_scaled), dim=-1)
        violation = (torch.abs(self.dof_pos_scaled) - self.cfg.joint_pos_limits_threshold) / (
            1.0 - self.cfg.joint_pos_limits_threshold
        )
        limit_cost = torch.sum(
            (torch.abs(self.dof_pos_scaled) > self.cfg.joint_pos_limits_threshold)
            * violation
            * self.gear_ratio_scaled,
            dim=-1,
        )

        # log terms separately. tuning reward weights blind is how you waste a week.
        log = self.extras.setdefault("log", {})
        log["Reward/ball_speed"] = ball_reward.mean()
        log["Reward/backswing"] = wind.mean()
        log["Reward/clubhead_speed"] = speed.mean()
        log["Reward/trail_hand"] = -trail.mean()
        log["Metrics/club_ball_dist"] = self.club_ball_dist.mean()
        log["Metrics/clubhead_speed"] = self.club_speed.mean()
        log["Metrics/ball_speed"] = ball_speed.mean()

        # per-second terms, scaled by step_dt (matches the manager-based convention)
        return self.step_dt * (
            ball_reward
            + wind
            + speed
            - trail
            - self.cfg.actions_cost_scale * actions_cost
            - self.cfg.energy_cost_scale * energy_cost
            - self.cfg.joint_pos_limits_cost_scale * limit_cost
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        # stage 0: no fall termination. flinging itself over to swing is allowed.
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        done = (self.episode_length_buf >= self.max_episode_length) | (
            self.post_impact > self.cfg.post_impact_steps
        )
        return died, done

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        log = self.extras.setdefault("log", {})
        log["Metrics/contact_rate"] = self.impacted[env_ids].float().mean()
        log["Metrics/min_club_ball_dist"] = self.min_dist[env_ids].mean()
        log["Metrics/impact_speed"] = self.impact_speed[env_ids].mean()

        super()._reset_idx(env_ids)
        index_fill_(self.actions, env_ids, 0.0)
        index_fill_(self.impacted, env_ids, False)
        index_fill_(self.post_impact, env_ids, 0)
        index_fill_(self.min_dist, env_ids, 1e3)
        index_fill_(self.impact_speed, env_ids, 0.0)
        index_fill_(self.armed, env_ids, False)
        index_fill_(self.backswing, env_ids, 0.0)
        index_fill_(self.max_ball_speed, env_ids, 0.0)

        # golfer back to the address pose, offset into its env
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=self.robot.data.default_root_vel.torch[env_ids], env_ids=env_ids
        )

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        joint_pos += sample_uniform(*self.cfg.initial_joint_pos_range, joint_pos.shape, joint_pos.device)
        joint_vel += sample_uniform(*self.cfg.initial_joint_vel_range, joint_vel.shape, joint_vel.device)
        limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])
        self.robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids)

        # ball back on the tee
        ball_pose = torch.zeros(len(env_ids), 7, device=self.device)
        ball_pose[:, :3] = (
            torch.tensor(self.cfg.ball_pos, device=self.device) + self.scene.env_origins[env_ids]
        )
        ball_pose[:, 3] = 1.0
        self.ball.write_root_pose_to_sim_index(root_pose=ball_pose, env_ids=env_ids)
        self.ball.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
        )

        self._compute_intermediate_values()
