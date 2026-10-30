"""Phase 1 golf swing env. Stage 0 goal: swing at all. Falling over is allowed."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import index_fill_
from isaaclab.utils.math import normalize, quat_apply, quat_apply_inverse, sample_uniform, scale_transform
from isaaclab.utils.string import resolve_matching_names_values

from .golf_env_cfg import GolfEnvCfg
from .golfer import CLUB_BODY, GOLFER_USD, LEAD_HAND, TRAIL_HAND
from .shapes import body_shapes


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
        self.launch_speed = torch.zeros(self.num_envs, device=self.device)
        self.smash = torch.zeros(self.num_envs, device=self.device)

        box = next(s for s in body_shapes(GOLFER_USD)[CLUB_BODY] if s[0] == "box")
        self.head_half = torch.tensor(
            [abs(box[2 + i][i] - box[1][i]) for i in range(3)], dtype=torch.float32, device=self.device
        )
        self.ball_r = self.scene["ball"].cfg.spawn.radius
        # where the ball rests on the tee, per env. quat is (x, y, z, w) in isaac lab 3.0
        self.tee_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self.tee_pose[:, :3] = torch.tensor(self.cfg.ball_pos, device=self.device) + self.scene.env_origins
        self.tee_pose[:, 6] = 1.0
        self.ball_rest = torch.zeros(self.num_envs, 6, device=self.device)

        self._compute_intermediate_values()
        # start-of-step state for the swept contact test
        self.prev_club_pos = self.club_pos.clone()
        self.prev_club_quat = self.club_quat.clone()
        self.prev_ball_pos = self.ball_pos.clone()
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
        self.club_quat = self.robot.data.body_link_quat_w.torch[:, self._club_idx]
        self.lead_pos = self.robot.data.body_link_pos_w.torch[:, self._lead_idx]
        self.trail_pos = self.robot.data.body_link_pos_w.torch[:, self._trail_idx]

        self.ball_pos = self.ball.data.root_pos_w.torch
        self.ball_vel = self.ball.data.root_lin_vel_w.torch

        self.club_ball_dist = torch.linalg.norm(self.ball_pos - self.club_pos, dim=-1)
        self.club_speed = torch.linalg.norm(self.club_vel, dim=-1)

    def _update_contact(self):
        """Arming, strike and launch, post-impact count. Exactly once per env step, after physics.

        Kept out of _compute_intermediate_values: _reset_idx calls that too, so on any step
        with a reset these counters used to run twice (post_impact counted double).

        The ball has no collider. PhysX misses most strikes above ~5 m/s at 120 Hz and invents
        them with a big contact offset (docs/experiments.md), so a strike is detected here and
        the launch computed the way launch monitors and golf sims report it: smash factor.
        """
        # swept, not sampled: at 15 m/s the head covers ~25 cm per env step, so a distance
        # checked only at step ends skips clean over the ball. use the closest approach of this
        # step's head path to where the ball sat when the step began.
        seg = self.club_pos - self.prev_club_pos
        s = ((self.prev_ball_pos - self.prev_club_pos) * seg).sum(-1) / (seg * seg).sum(-1).clamp_min(1e-9)
        s = s.clamp(0.0, 1.0)
        closest = self.prev_club_pos + s.unsqueeze(-1) * seg
        swept_dist = torch.linalg.norm(self.prev_ball_pos - closest, dim=-1)

        # the head at closest approach, in its own frame: orientation interpolated through the
        # step (nlerp), ball offset and head velocity expressed in it
        q0, q1 = self.prev_club_quat, self.club_quat
        q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
        q = normalize(q0 + s.unsqueeze(-1) * (q1 - q0))
        off = quat_apply_inverse(q, self.prev_ball_pos - closest)
        v_local = quat_apply_inverse(q, self.club_vel)
        # the swept test is a sphere; the head is a long, short box. the ball has to overlap it
        on_head = (off.abs() <= self.head_half + self.ball_r).all(-1)
        # the side that leads through the ball strikes it: +x is the face, -x the back. (at a
        # square strike's closest point the ball's offset along x is ~0, so position can't tell)
        side = torch.where(v_local[:, 0] >= 0, 1.0, -1.0)
        v_in = v_local[:, 0].abs()  # head speed into the ball, along the striking side's normal

        # armed from an EARLIER step: a head that pokes the ball and leaves in one step isn't a swing.
        # arming exists at all because the head starts inside the impact radius, which made
        # contact_rate 1.0 from iteration 0.
        hit = (
            (swept_dist < self.cfg.impact_radius)
            & on_head
            & (v_in > self.cfg.impact_min_speed)
            & self.armed
            & ~self.impacted
        )

        # smash factor = ball speed / clubhead speed. dead centre of a driver face ~1.47, falling
        # off away from the sweet spot (twice as fast up-down: the face is half as tall). the
        # back of the head isn't a face, so it gets the floor.
        sweet = torch.exp(-(off[:, 1] / self.cfg.smash_sigma_toe) ** 2 - (off[:, 2] / self.cfg.smash_sigma_high) ** 2)
        smash = self.cfg.smash_min + (self.cfg.smash_max - self.cfg.smash_min) * sweet
        smash = torch.where(side > 0, smash, torch.full_like(smash, self.cfg.smash_min))
        # start direction: mostly the face, partly the path (the launch-monitor rule for a driver)
        normal = quat_apply(q, torch.stack([side, torch.zeros_like(side), torch.zeros_like(side)], dim=-1))
        w = self.cfg.face_launch_weight
        direction = normalize(w * normal + (1.0 - w) * normalize(self.club_vel))
        speed = smash * v_in

        ids = hit.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            launch = torch.zeros(len(ids), 6, device=self.device)
            launch[:, :3] = speed[ids].unsqueeze(-1) * direction[ids]
            self.ball.write_root_velocity_to_sim_index(root_velocity=launch, env_ids=ids)
        self.launch_speed = torch.where(hit, speed, self.launch_speed)
        self.smash = torch.where(hit, smash, self.smash)

        self.armed |= self.club_ball_dist > self.cfg.impact_arm_dist
        self.impact_speed = torch.where(hit, self.club_speed, self.impact_speed)
        self.impacted |= hit
        self.post_impact += self.impacted.long()
        self.min_dist = torch.minimum(self.min_dist, swept_dist)
        self.prev_club_pos.copy_(self.club_pos)
        self.prev_club_quat.copy_(self.club_quat)
        self.prev_ball_pos.copy_(self.ball_pos)

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
        # the launch the strike computed (0 before it). nothing else can move the ball: it has
        # no collider, so kicks and pre-backswing pokes earn nothing by construction
        ball_speed = self.launch_speed
        gained = torch.relu(ball_speed - self.max_ball_speed)
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
        self._update_contact()
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
        struck = self.impacted[env_ids]
        if struck.any():
            log["Metrics/smash"] = self.smash[env_ids][struck].mean()

        super()._reset_idx(env_ids)
        index_fill_(self.actions, env_ids, 0.0)
        index_fill_(self.impacted, env_ids, False)
        index_fill_(self.post_impact, env_ids, 0)
        index_fill_(self.min_dist, env_ids, 1e3)
        index_fill_(self.impact_speed, env_ids, 0.0)
        index_fill_(self.armed, env_ids, False)
        index_fill_(self.backswing, env_ids, 0.0)
        index_fill_(self.max_ball_speed, env_ids, 0.0)
        index_fill_(self.launch_speed, env_ids, 0.0)
        index_fill_(self.smash, env_ids, 0.0)

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
        self.ball.write_root_pose_to_sim_index(root_pose=self.tee_pose[env_ids], env_ids=env_ids)
        self.ball.write_root_velocity_to_sim_index(root_velocity=self.ball_rest[env_ids], env_ids=env_ids)

        self._compute_intermediate_values()
        self.prev_club_pos[env_ids] = self.club_pos[env_ids]
        self.prev_club_quat[env_ids] = self.club_quat[env_ids]
        self.prev_ball_pos[env_ids] = self.ball_pos[env_ids]
