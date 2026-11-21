"""Phase 1 golf swing env: strike the ball while staying balanced"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from flight import ball as flight_ball
from flight import spin as flight_spin
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import index_fill_
from isaaclab.utils.math import normalize, quat_apply, quat_apply_inverse, sample_uniform, scale_transform
from isaaclab.utils.string import resolve_matching_names_values

from .clubs import CLUBS
from .golf_env_cfg import (
    FLIGHT_KEYS, MPH, STRIKE_CLASSES, TRACER_CFG, TRACER_WIDTH, YD, GolfEnvCfg, apply_body, apply_club,
)
from .golfer import CLUB_BODY, LEAD_HAND, TRAIL_HAND
from .shapes import body_shapes


class GolfEnv(DirectRLEnv):
    cfg: GolfEnvCfg

    def __init__(self, cfg: GolfEnvCfg, render_mode: str | None = None, **kwargs):
        apply_club(cfg)
        apply_body(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        self.robot, self.ball = self.scene["robot"], self.scene["ball"]
        self.action_scale = self.cfg.action_scale

        self.joint_gears = torch.ones(self.robot.num_joints, device=self.device)
        joint_ids, _, gears = resolve_matching_names_values(self.cfg.joint_gears, self.robot.joint_names)
        self.joint_gears[joint_ids] = torch.tensor(gears, device=self.device)
        self.gear_ratio_scaled = self.joint_gears / torch.max(self.joint_gears)
        self.muscle_vmax = torch.ones(self.robot.num_joints, device=self.device)
        joint_ids, _, vmax = resolve_matching_names_values(self.cfg.muscle_vmax, self.robot.joint_names)
        self.muscle_vmax[joint_ids] = torch.tensor(vmax, device=self.device)
        self.muscle_frac = torch.ones(self.num_envs, device=self.device)

        body_idx, _ = self.robot.find_bodies([CLUB_BODY, LEAD_HAND, TRAIL_HAND], preserve_order=True)
        self._club_idx, self._lead_idx, self._trail_idx = body_idx
        self._feet_idx, _ = self.robot.find_bodies(["left_foot", "right_foot"], preserve_order=True)
        (self._head_idx, self._pelvis_idx), _ = self.robot.find_bodies(["head", "pelvis"], preserve_order=True)

        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.target_dir = normalize(
            torch.tensor(self.cfg.target_dir, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        )

        self.impacted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.post_impact = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.min_dist = torch.full((self.num_envs,), 1e3, device=self.device)
        self.impact_speed = torch.zeros(self.num_envs, device=self.device)
        self.impact_hand_speed = torch.zeros(self.num_envs, device=self.device)
        self.impact_time = torch.zeros(self.num_envs, device=self.device)
        self.armed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.backswing = torch.zeros(self.num_envs, device=self.device)
        self.top_dist = torch.zeros(self.num_envs, device=self.device)
        self.max_ball_speed = torch.zeros(self.num_envs, device=self.device)
        self.launch_speed = torch.zeros(self.num_envs, device=self.device)
        self.smash = torch.zeros(self.num_envs, device=self.device)
        self.sweet = torch.zeros(self.num_envs, device=self.device)
        self.strike_off = torch.zeros(self.num_envs, 2, device=self.device)  # ball on the face: heel +, high + [m]
        # ball on the ground: analytic turf (the head never collides with the ground), strike class at impact
        self.ground = not CLUBS[self.cfg.club].tee
        self.head_corners = None  # head's bottom corners in its frame, leading edge first (set at the first reset)
        self.turf_f = torch.linspace(0.0, 1.0, 8, device=self.device)  # samples along each step's head path
        self.turf_mid = 0.5 * (self.turf_f[1:] + self.turf_f[:-1])
        self.turf_pre = torch.zeros(self.num_envs, device=self.device)  # turf cut on the way down [m^2]
        self.turf_factor = torch.ones(self.num_envs, device=self.device)
        self.dig_step = torch.zeros(self.num_envs, device=self.device)  # cut deeper than dig_free this step [m^2]
        self.dig_total = torch.zeros(self.num_envs, device=self.device)
        self.leading_edge = torch.zeros(self.num_envs, device=self.device)  # height at impact [m]
        self.strike_class = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # STRIKE_CLASSES
        self.backwards = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.offline = torch.zeros(self.num_envs, device=self.device)
        self.face_deg = torch.zeros(self.num_envs, device=self.device)
        self.path_deg = torch.zeros(self.num_envs, device=self.device)
        self.top_time = torch.zeros(self.num_envs, device=self.device)
        self.new_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.mass_frac = None
        self.xcom_out = torch.zeros(self.num_envs, device=self.device)
        self.sole_low = torch.zeros(self.num_envs, 2, device=self.device)  # lowest sole point per foot (lead, trail)
        self.trail_lift = torch.zeros(self.num_envs, device=self.device)  # trail sole's highest low point before impact
        self.shift = torch.zeros(self.num_envs, device=self.device)
        heel, toe, sole = self.cfg.foot_sole
        self.sole_pts = torch.tensor([[heel, 0.0, sole], [toe, 0.0, sole]], device=self.device)
        self.grounded = torch.ones(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.motor_n = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        self.motor_rho = math.exp(-self.cfg.sim.dt / self.cfg.motor_noise_tau)
        self.grip_off = None  # trail hand's address spot on the grip, club frame (set at the first reset)
        self.grip_gap = torch.zeros(self.num_envs, device=self.device)

        box = next(s for s in body_shapes(self.cfg.scene.robot.spawn.usd_path)[CLUB_BODY] if s[0] == "box")
        self.head_half = torch.tensor(
            [abs(box[2 + i][i] - box[1][i]) for i in range(3)], dtype=torch.float32, device=self.device
        )
        self.ball_r = self.scene["ball"].cfg.spawn.radius

        self.tee_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self.tee_pose[:, :3] = torch.tensor(self.cfg.ball_pos, device=self.device) + self.scene.env_origins
        self.tee_pose[:, 6] = 1.0
        self.ball_rest = torch.zeros(self.num_envs, 6, device=self.device)

        self._compute_intermediate_values()
        self.feet_home = self.feet_xy.clone()  # address foot positions
        self.feet_home_z = self.feet_z.clone()

        self.prev_club_pos = self.club_pos.clone()
        self.prev_club_quat = self.club_quat.clone()
        self.prev_club_vel = self.club_vel.clone()
        self.prev_ball_pos = self.ball_pos.clone()

        self.loft_normal = None  # set at the first reset, from the address pose (not posed yet here)
        self.trackman = CLUBS[self.cfg.club].trackman
        # flight frame rows: right of target, target, up
        fwd = normalize(self.target_dir[0] * torch.tensor([1.0, 1.0, 0.0], device=self.device))
        self.flight_frame = torch.stack(
            [torch.stack([fwd[1], -fwd[0], fwd[2]]), fwd, torch.tensor([0.0, 0.0, 1.0], device=self.device)]
        )
        self.flight_out = torch.zeros(self.num_envs, len(FLIGHT_KEYS), device=self.device)
        assert self.cfg.flight or not (self.cfg.carry_weight or self.cfg.launch_weight or self.cfg.attack_weight), "flight pay needs env.flight"
        self.flight_path, self.trace_step = None, 0  # env 0's last flight (world points, seconds apart, flight time)

        # env 0's last shot, drawn until its next strike
        self.tracer = None
        if self.cfg.tracer and (self.sim.has_gui or self.sim.has_offscreen_render):
            self.tracer = VisualizationMarkers(TRACER_CFG)
            self.tracer.set_visibility(False)
            eye = getattr(self.cfg.sim.default_visualizer_cfg, "eye", None) or (0.0, -3.0, 1.4)
            self.tracer_eye = torch.tensor(eye, device=self.device) + self.scene.env_origins[0]

        width = self._get_observations()["policy"].shape[-1]
        assert width == self.cfg.observation_space, (
            f"observation_space={self.cfg.observation_space} but obs is {width} wide"
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # every physics substep, so the muscle curve sees current joint speed
        # clamp before scaling: unbounded efforts drive the solver to NaN
        forces = self.action_scale * self.joint_gears * torch.clamp(self.actions, -1.0, 1.0)
        if self.cfg.muscle:
            # shortening speed as a fraction of vmax; lengthening keeps full torque
            x = (self.robot.data.joint_vel.torch * torch.sign(forces) / self.muscle_vmax).clamp(0.0, 1.0)
            capped = forces * (1.0 - x) / (1.0 + x / self.cfg.muscle_k)
            self.muscle_frac = capped.abs().sum(-1) / forces.abs().sum(-1).clamp_min(1e-6)
            forces = capped
        if self.cfg.motor_noise > 0:
            # signal-dependent noise: torque error grows with the torque (Harris & Wolpert)
            rho = self.motor_rho
            self.motor_n.mul_(rho).add_(torch.randn_like(self.motor_n), alpha=math.sqrt(1.0 - rho * rho))
            forces = forces * (1.0 + self.cfg.motor_noise * self.motor_n)
        self.robot.actuators.target_command.set_effort_index(value=forces)
        if self.cfg.grip_spring > 0 and self.grip_off is not None:
            self._grip_spring()

    def _grip_spring(self) -> None:
        # trail hand pulled to its spot on the grip, the club pulled back (a second weld closes a loop PhysX can't hold)
        d, c, h = self.robot.data, self._club_idx, self._trail_idx
        r = quat_apply(d.body_link_quat_w.torch[:, c], self.grip_off.expand(self.num_envs, 3))
        spot = d.body_link_pos_w.torch[:, c] + r
        spot_vel = d.body_link_lin_vel_w.torch[:, c] + torch.linalg.cross(d.body_link_ang_vel_w.torch[:, c], r, dim=-1)
        hand = d.body_link_pos_w.torch[:, h]
        f = self.cfg.grip_spring * (spot - hand) + self.cfg.grip_damping * (spot_vel - d.body_link_lin_vel_w.torch[:, h])
        f = f * (self.cfg.grip_max_force / torch.linalg.norm(f, dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
        self.robot.instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=torch.stack((f, -f), dim=1), positions=torch.stack((hand, spot), dim=1), body_ids=[h, c], is_global=True
        )

    def _compute_intermediate_values(self):
        self.torso_pos = self.robot.data.root_pos_w.torch
        self.dof_pos_scaled = scale_transform(
            self.robot.data.joint_pos.torch,
            self.robot.data.soft_joint_pos_limits.torch[..., 0],
            self.robot.data.soft_joint_pos_limits.torch[..., 1],
        )
        self.dof_vel = self.robot.data.joint_vel.torch

        # link frame: its origin is the clubhead
        self.club_pos = self.robot.data.body_link_pos_w.torch[:, self._club_idx]
        self.club_vel = self.robot.data.body_link_lin_vel_w.torch[:, self._club_idx]
        self.club_quat = self.robot.data.body_link_quat_w.torch[:, self._club_idx]
        self.lead_pos = self.robot.data.body_link_pos_w.torch[:, self._lead_idx]
        self.trail_pos = self.robot.data.body_link_pos_w.torch[:, self._trail_idx]
        if self.grip_off is not None:
            spot = self.club_pos + quat_apply(self.club_quat, self.grip_off.expand(self.num_envs, 3))
            self.grip_gap = torch.linalg.norm(self.trail_pos - spot, dim=-1)
        self.feet_xy = self.robot.data.body_link_pos_w.torch[:, self._feet_idx, :2]
        self.feet_z = self.robot.data.body_link_pos_w.torch[:, self._feet_idx, 2]
        # spine tilt: the pelvis-to-head line off vertical
        spine = self.robot.data.body_link_pos_w.torch[:, self._head_idx] - self.robot.data.body_link_pos_w.torch[:, self._pelvis_idx]
        self.spine_deg = torch.rad2deg(torch.acos((spine[:, 2] / torch.linalg.norm(spine, dim=-1)).clamp(-1.0, 1.0)))

        self.ball_pos = self.ball.data.root_pos_w.torch
        self.ball_vel = self.ball.data.root_lin_vel_w.torch

        self.club_ball_dist = torch.linalg.norm(self.ball_pos - self.club_pos, dim=-1)
        self.club_speed = torch.linalg.norm(self.club_vel, dim=-1)

    def _update_balance(self):
        # extrapolated CoM against the grounded feet, and how far the weight has moved toward the lead foot
        d = self.robot.data
        if self.mass_frac is None:
            m = d.body_mass.torch.to(self.device)
            self.mass_frac = m / m.sum(-1, keepdim=True)
        com = (self.mass_frac[..., None] * d.body_com_pos_w.torch).sum(1)
        com_vel = (self.mass_frac[..., None] * d.body_com_lin_vel_w.torch).sum(1)
        # Hof's extrapolated CoM: inside the feet means the body can stop without stepping
        xcom = com[:, :2] + com_vel[:, :2] / torch.sqrt(9.81 / com[:, 2:3].clamp_min(0.2))
        # heel and toe of each sole: a foot rolled onto its toe still holds the body up
        n = self.num_envs
        quat = d.body_link_quat_w.torch[:, self._feet_idx, None].expand(n, 2, 2, 4)
        pts = d.body_link_pos_w.torch[:, self._feet_idx, None] + quat_apply(
            quat.reshape(-1, 4), self.sole_pts.expand(n, 2, 2, 3).reshape(-1, 3)
        ).reshape(n, 2, 2, 3)
        pts = pts.reshape(n, 4, 3)
        self.grounded = pts[..., 2] < 0.02
        self.sole_low = pts[..., 2].view(n, 2, 2).amin(-1)
        self.trail_lift = torch.where(self.impacted, self.trail_lift, torch.maximum(self.trail_lift, self.sole_low[:, 1]))
        g = self.grounded | ~self.grounded.any(-1, keepdim=True)
        px, py = pts[..., 0], pts[..., 1]
        far = torch.full_like(px, 1e3)
        r, half = -self.cfg.foot_sole[2], 0.05  # sole radius, foot half width
        # support: the box around the grounded heels and toes
        dx = torch.relu(torch.where(g, px, far).amin(-1) - r - xcom[:, 0]) + torch.relu(
            xcom[:, 0] - torch.where(g, px, -far).amax(-1) - r
        )
        dy = torch.relu(torch.where(g, py, far).amin(-1) - half - xcom[:, 1]) + torch.relu(
            xcom[:, 1] - torch.where(g, py, -far).amax(-1) - half
        )
        self.xcom_out = torch.sqrt(dx * dx + dy * dy)
        # a hop has no support at all
        airborne = ~self.grounded.any(-1)
        self.xcom_out = torch.where(airborne, self.xcom_out.clamp_min(self.cfg.balance_airborne), self.xcom_out)
        # between where the feet are now: walking toward the target is not a weight shift
        lead, trail = self.feet_xy[:, 0], self.feet_xy[:, 1]
        axis = lead - trail
        self.shift = ((com[:, :2] - trail) * axis).sum(-1) / (axis * axis).sum(-1).clamp_min(1e-6)

    def _update_contact(self):
        """Arming, strike detection and smash-factor launch. Once per step, after physics."""
        # swept: closest approach of this step's head path to the ball
        seg = self.club_pos - self.prev_club_pos
        s = ((self.prev_ball_pos - self.prev_club_pos) * seg).sum(-1) / (seg * seg).sum(-1).clamp_min(1e-9)
        s = s.clamp(0.0, 1.0)
        closest = self.prev_club_pos + s.unsqueeze(-1) * seg
        swept_dist = torch.linalg.norm(self.prev_ball_pos - closest, dim=-1)

        # head pose at closest approach (nlerp), ball offset and velocity in its frame
        q0, q1 = self.prev_club_quat, self.club_quat
        q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
        q = normalize(q0 + s.unsqueeze(-1) * (q1 - q0))
        # head velocity there too: the club turns ~20 deg a step near the ball, so the step's end reads it late
        v0, v1 = self.prev_club_vel, self.club_vel
        v_imp = normalize(torch.lerp(v0, v1, s.unsqueeze(-1))) * torch.lerp(v0.norm(dim=-1), v1.norm(dim=-1), s).unsqueeze(-1)
        off = quat_apply_inverse(q, self.prev_ball_pos - closest)
        v_local = quat_apply_inverse(q, v_imp)
        # ball must overlap the head box, not just the swept sphere
        on_head = (off.abs() <= self.head_half + self.ball_r).all(-1)
        # leading side strikes: +x face, -x back
        side = torch.where(v_local[:, 0] >= 0, 1.0, -1.0)
        v_in = v_local[:, 0].abs()  # head speed into the ball

        # armed on an earlier step, so one-step poke doesn't count
        hit = (
            (swept_dist < self.cfg.impact_radius)
            & on_head
            & (v_in > self.cfg.impact_min_speed)
            & self.armed
            & ~self.impacted
        )

        # smash: max at the sweet spot, falling off with offset; the back of the head gets the floor
        sweet = torch.exp(-(off[:, 1] / self.cfg.smash_sigma_toe) ** 2 - (off[:, 2] / self.cfg.smash_sigma_high) ** 2)
        smash = self.cfg.smash_min + (self.cfg.smash_max - self.cfg.smash_min) * sweet
        smash = torch.where(side > 0, smash, torch.full_like(smash, self.cfg.smash_min))
        # start direction: mostly face, partly path
        normal = quat_apply(q, torch.stack([side, torch.zeros_like(side), torch.zeros_like(side)], dim=-1))
        w = self.cfg.face_launch_weight
        direction = normalize(w * normal + (1.0 - w) * normalize(v_imp))
        speed = smash * v_in
        if self.ground and self.head_corners is not None:
            speed = speed * self._turf(s, q, closest, hit)

        ids = hit.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            launch = torch.zeros(len(ids), 6, device=self.device)
            launch[:, :3] = speed[ids].unsqueeze(-1) * direction[ids]
            self.ball.write_root_velocity_to_sim_index(root_velocity=launch, env_ids=ids)
        self.launch_speed = torch.where(hit, speed, self.launch_speed)
        self.smash = torch.where(hit, smash, self.smash)
        self.sweet = torch.where(hit, sweet * (side > 0), self.sweet)
        self.strike_off = torch.where(hit.unsqueeze(-1), off[:, 1:], self.strike_off)
        # backwards: back of the head, or the ball heading away from the target
        away = (side < 0) | ((direction * self.target_dir).sum(-1) < 0)
        self.backwards = torch.where(hit, away, self.backwards)
        # launch, face, and path
        self.offline = torch.where(hit, self._off_target_deg(direction), self.offline)
        self.face_deg = torch.where(hit, self._off_target_deg(normal), self.face_deg)
        self.path_deg = torch.where(hit, self._off_target_deg(v_imp), self.path_deg)
        self.new_hit = hit
        if self.cfg.flight and len(ids) > 0:
            self._fly(ids[~away[ids]], q, v_imp, direction, speed)

        self.armed |= self.club_ball_dist > self.cfg.impact_arm_dist
        new_top = ~(self.impacted | hit) & (self.club_ball_dist > self.top_dist)
        self.top_time = torch.where(new_top, self.episode_length_buf * self.step_dt, self.top_time)
        self.top_dist = torch.where(self.impacted | hit, self.top_dist, torch.maximum(self.top_dist, self.club_ball_dist))
        self.impact_speed = torch.where(hit, self.club_speed, self.impact_speed)
        hand_speed = torch.linalg.norm(self.robot.data.body_link_lin_vel_w.torch[:, self._lead_idx], dim=-1)
        self.impact_hand_speed = torch.where(hit, hand_speed, self.impact_hand_speed)
        self.impact_time = torch.where(hit, self.episode_length_buf * self.step_dt, self.impact_time)
        self.impacted |= hit
        self.post_impact += self.impacted.long()
        self.min_dist = torch.minimum(self.min_dist, swept_dist)
        self.prev_club_pos.copy_(self.club_pos)
        self.prev_club_quat.copy_(self.club_quat)
        self.prev_club_vel.copy_(self.club_vel)
        self.prev_ball_pos.copy_(self.ball_pos)

    def _sole_z(self, pos: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Heights of the head's bottom corners, (n, k) poses -> (n, k, 4)."""
        n, k = pos.shape[:2]
        c = self.head_corners.shape[0]
        corner = quat_apply(q.unsqueeze(2).expand(n, k, c, 4).reshape(-1, 4), self.head_corners.expand(n * k, c, 3).reshape(-1, 3))
        return pos[..., 2].unsqueeze(-1) + corner[:, 2].view(n, k, c)

    def _turf(self, s: torch.Tensor, q_hit: torch.Tensor, pos_hit: torch.Tensor, hit: torch.Tensor) -> torch.Tensor:
        """Ball on the ground: turf the head cuts this step, the ball-speed factor, and the strike's class."""
        n, f = self.num_envs, self.turf_f.view(1, -1, 1)
        pos = torch.lerp(self.prev_club_pos.unsqueeze(1), self.club_pos.unsqueeze(1), f)
        q0, q1 = self.prev_club_quat, self.club_quat
        q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
        q = normalize(q0.unsqueeze(1) + f * (q1 - q0).unsqueeze(1))
        depth = torch.relu(-self._sole_z(pos, q).min(-1).values)
        seg = torch.linalg.norm(pos[:, 1:, :2] - pos[:, :-1, :2], dim=-1)
        mid = 0.5 * (depth[:, 1:] + depth[:, :-1])
        # fat: turf cut just before the ball slows the ball (debris and a slowed head)
        before = (self.turf_mid.view(1, -1) <= s.view(-1, 1)).float()
        down = (self.armed & ~self.impacted).float().unsqueeze(-1)
        gap = torch.linalg.norm(0.5 * (pos[:, 1:, :2] + pos[:, :-1, :2]) - self.prev_ball_pos[:, :2].unsqueeze(1), dim=-1)
        near = (gap < self.cfg.turf_window).float()
        self.turf_pre += (seg * torch.relu(mid - self.cfg.turf_grace) * before * down * near).sum(-1)
        self.dig_step = (seg * torch.relu(mid - self.cfg.dig_free)).sum(-1)
        self.dig_total += self.dig_step
        factor = torch.exp(-self.turf_pre / self.cfg.turf_area) if self.cfg.turf_area > 0 else torch.ones_like(self.turf_pre)
        # leading edge at impact: below ground fat, on the ball thin, above its equator top
        edge = self._sole_z(pos_hit.unsqueeze(1), q_hit.unsqueeze(1))[:, 0, :2].min(-1).values
        sin_loft = quat_apply(q_hit, self.loft_normal.expand(n, 3))[:, 2]
        thin = self.ball_r * (1.0 - sin_loft)  # the face meets the ball below this height
        cls = torch.where(edge < 0.0, 1, torch.where(edge <= thin, 0, torch.where(edge <= self.ball_r, 2, 3)))
        self.leading_edge = torch.where(hit, edge, self.leading_edge)
        self.strike_class = torch.where(hit, cls, self.strike_class)
        self.turf_factor = torch.where(hit, factor, self.turf_factor)
        return factor

    def _fly(self, ids: torch.Tensor, q: torch.Tensor, v_imp: torch.Tensor, direction: torch.Tensor,
             speed: torch.Tensor):
        """Phase 2: launch angle, spin and flight from the club at impact (flight/, numpy on the CPU)."""
        if len(ids) == 0:
            return
        frame = self.flight_frame
        normal = quat_apply(q[ids], self.loft_normal.expand(len(ids), 3)) @ frame.T
        start = direction[ids] @ frame.T
        ball_speed = speed[ids].cpu().numpy()
        launch = flight_spin.launch(
            (v_imp[ids] @ frame.T).cpu().numpy(), normal.cpu().numpy(), ball_speed, self.trackman,
            w=self.cfg.face_launch_weight,
        )
        # start line as the env reports it (offline); height and spin from the lofted face
        azimuth = torch.rad2deg(torch.atan2(start[:, 0], start[:, 1])).cpu().numpy()
        vel = flight_ball.launch_vector(ball_speed, launch["launch_deg"], azimuth)
        track = self.tracer is not None and bool((ids == 0).any())
        z0 = self.tee_pose[ids, 2] - self.scene.env_origins[ids, 2]  # ball height above the ground
        out = flight_ball.fly(vel, launch["spin"], z0=z0.cpu().numpy(), path=track)
        cols = [out["carry"], out["side"], out["curve"], out["apex"]] + [launch[k] for k in FLIGHT_KEYS[4:]]
        self.flight_out[ids] = torch.tensor(np.stack(cols, -1), dtype=torch.float32, device=self.device)
        if track:
            i = int((ids == 0).nonzero()[0, 0])
            pts = torch.tensor(out["path"][i], dtype=torch.float32, device=self.device) @ frame
            base = self.tee_pose[0, :3].clone()
            base[2] -= z0[i]
            self.flight_path = (base + pts, out["dt"], float(out["time"][i]))

    def _off_target_deg(self, v: torch.Tensor) -> torch.Tensor:
        flat, tgt = v[:, :2], self.target_dir[:, :2]
        return torch.rad2deg(torch.atan2(flat[:, 0] * tgt[:, 1] - flat[:, 1] * tgt[:, 0], (flat * tgt).sum(-1)))

    def _draw_tracer(self):
        """Draw env 0's computed flight, played back over tracer_flight_s, and keep it until the next strike."""
        if self.new_hit[0]:
            self.trace_step = 0
        if self.flight_path is None:
            self.tracer.set_visibility(False)
            return
        self.trace_step += 1
        path, dt, t_flight = self.flight_path
        shown = min(1.0, self.trace_step * self.step_dt / self.cfg.tracer_flight_s) * t_flight
        # fixed segment count (Kit's instancer misses a changing one), resampled from the flight's time steps
        n = self.cfg.post_impact_steps
        k = (torch.linspace(0.0, shown, n + 1, device=self.device) / dt).clamp(max=len(path) - 1)
        lo = k.floor().long().clamp(max=len(path) - 2)
        pts = torch.lerp(path[lo], path[lo + 1], (k - lo).unsqueeze(-1))
        seg = pts[1:] - pts[:-1]
        length = torch.linalg.norm(seg, dim=-1)
        d = seg / length.clamp_min(1e-6).unsqueeze(-1)
        d[length < 1e-6] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        # rotate the cylinder's +z onto each segment
        quat = normalize(torch.stack([-d[:, 1], d[:, 0], torch.zeros_like(length), 1.0 + d[:, 2]], dim=-1))
        mid = 0.5 * (pts[1:] + pts[:-1])
        r = TRACER_WIDTH * torch.linalg.norm(mid - self.tracer_eye, dim=-1) * (length > 1e-6)
        self.tracer.visualize(translations=mid, orientations=quat, scales=torch.stack([r, r, length], dim=-1))
        self.tracer.set_visibility(True)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        ref = self.tee_pose[:, :3] if self.cfg.ball_ref_obs else self.ball_pos
        obs = torch.cat(
            (
                self.torso_pos[:, 2].view(-1, 1),
                self.robot.data.root_lin_vel_b.torch,
                self.robot.data.root_ang_vel_b.torch * self.cfg.angular_velocity_scale,
                (-self.robot.data.projected_gravity_b.torch[:, 2]).view(-1, 1),
                self.dof_pos_scaled,
                self.dof_vel * self.cfg.dof_vel_scale,
                self.actions,
                self.club_pos - ref,
                self.club_vel * self.cfg.clubhead_vel_scale,
                ref - self.torso_pos,
                (self.episode_length_buf / self.max_episode_length).view(-1, 1),
            ),
            dim=-1,
        )
        if self.cfg.flight_obs:
            # the computed flight (zero until impact), so finish-time carry pay can be traced to the strike
            k, f = FLIGHT_KEYS.index, self.flight_out
            fl = torch.stack((f[:, k("carry")] / 100.0, f[:, k("side")] / 20.0, f[:, k("launch_deg")] / 20.0), dim=-1)
            obs = torch.cat((obs, fl), dim=-1)
        if self.cfg.gravity_obs:
            obs = torch.cat((obs, self.robot.data.projected_gravity_b.torch[:, :2]), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 1. ball speed from the computed launch, paid once; backwards hits pay nothing and cost
        ball_speed = self.launch_speed * (~self.backwards).float()
        if self.cfg.ball_speed_cap > 0:
            ball_speed = ball_speed.clamp(max=self.cfg.ball_speed_cap)
        # direction: a sideways hit pays less than a straight one
        aim = torch.ones_like(ball_speed)
        if self.cfg.aim_sigma > 0:
            aim = torch.exp(-(self.offline / self.cfg.aim_sigma) ** 2)
        ball_speed = ball_speed * aim
        if self.cfg.strike_pow > 0:
            # strike quality: an off-centre hit pays less than smash alone takes off
            ball_speed = ball_speed * self.sweet**self.cfg.strike_pow
        if self.cfg.strike_bonus > 0:
            ball_speed = ball_speed * (1.0 + self.cfg.strike_bonus * self.sweet)
        backwards = self.cfg.backwards_penalty * (self.new_hit & self.backwards).float() / self.step_dt
        gained = torch.relu(ball_speed - self.max_ball_speed)
        self.max_ball_speed = torch.maximum(self.max_ball_speed, ball_speed)
        ball_reward = self.cfg.ball_speed_weight * gained / self.step_dt
        frac = self.cfg.finish_frac
        finishing = (self.post_impact == self.cfg.post_impact_steps + 1).float()
        lo, hi = self.cfg.finish_low, self.cfg.finish_high
        upright = torch.clamp((self.torso_pos[:, 2] - lo) / (hi - lo), 0.0, 1.0)
        finish = self.cfg.ball_speed_weight * frac * ball_speed * upright * finishing / self.step_dt
        ball_reward = (1.0 - frac) * ball_reward + finish
        # distance: carry along the target line from the computed flight, split between impact and finish the same way
        k, f = FLIGHT_KEYS.index, self.flight_out
        flown = (self.new_hit & ~self.backwards).float() / self.step_dt
        toward = torch.sqrt(torch.relu(f[:, k("carry")] ** 2 - f[:, k("side")] ** 2))
        cfrac = frac if self.cfg.carry_frac < 0 else self.cfg.carry_frac
        carry = self.cfg.carry_weight * toward * ((1.0 - cfrac) * flown + cfrac * upright * finishing / self.step_dt)
        if self.cfg.carry_aim:
            carry = carry * aim
        # launch angle
        lo_l, hi_l = self.cfg.launch_range
        launch = self.cfg.launch_weight * torch.clamp((f[:, k("launch_deg")] - lo_l) / (hi_l - lo_l), 0.0, 1.0) * flown
        if self.cfg.launch_aim:
            launch = launch * aim
        # attack angle
        lo_a, hi_a = self.cfg.attack_range
        attack = self.cfg.attack_weight * torch.clamp((f[:, k("attack_deg")] - lo_a) / (hi_a - lo_a), 0.0, 1.0) * flown
        if self.cfg.attack_aim:
            attack = attack * aim
        # rewarding for staying upright after contact
        hold = self.cfg.hold_weight * upright * self.impacted.float()
        # posture: standing up out of the address bend after impact, not bowing over
        good, bad = self.cfg.spine_good_deg, self.cfg.spine_bad_deg
        tall = torch.clamp((bad - self.spine_deg) / (bad - good), 0.0, 1.0)
        posture = self.cfg.posture_weight * tall * upright * self.impacted.float()
        # balance metrics
        balance = self.cfg.balance_weight * self.xcom_out
        on_lead = torch.clamp(self.shift / self.cfg.shift_target, 0.0, 1.0)
        shift = self.cfg.shift_weight * on_lead * upright * self.impacted.float()
        # weight forward at the strike
        impact_shift = self.cfg.impact_shift_weight * on_lead * flown
        # turf cut deeper than a divot, in place of turf resistance
        dig = self.cfg.dig_weight * self.dig_step / self.step_dt

        # 2. wind up, capped; before impact only (the struck ball flies away from the head)
        capped = torch.clamp(self.club_ball_dist, max=self.cfg.backswing_cap) * (~self.impacted).float()
        wind = self.cfg.backswing_weight * torch.relu(capped - self.backswing) / self.step_dt
        self.backswing = torch.maximum(self.backswing, capped)

        # 3. head speed near the ball
        near = (self.club_ball_dist < self.cfg.speed_radius).float()
        speed = self.cfg.clubhead_speed_weight * self.club_speed * near

        # slow takeaway: head speed above takeaway_speed costs while it moves away from the ball before impact
        going_back = (((self.club_pos - self.ball_pos) * self.club_vel).sum(-1) > 0) & ~self.impacted
        takeaway = self.cfg.takeaway_weight * torch.relu(self.club_speed - self.cfg.takeaway_speed) * going_back.float()

        # trail hand on the grip (penalty, not a weld)
        trail = self.cfg.trail_hand_weight * torch.linalg.norm(self.trail_pos - self.lead_pos, dim=-1)

        # balance: stay up
        stand = self.cfg.stand_weight * torch.relu(self.cfg.stand_height - self.torso_pos[:, 2])
        slip = torch.relu(torch.linalg.norm(self.feet_xy - self.feet_home, dim=-1) - self.cfg.feet_slip_free)
        feet = self.cfg.feet_slip_weight * slip.sum(-1)

        lead_lift = self.feet_z[:, 0] - self.feet_home_z[:, 0]
        lead = self.cfg.lead_foot_weight * (slip[:, 0] + torch.relu(lead_lift - self.cfg.lead_foot_lift_free))
        # trail foot: the heel may lift and the foot pivot, the toe stays down
        trail_foot = self.cfg.trail_foot_weight * torch.relu(self.sole_low[:, 1] - self.cfg.trail_lift_free)
        new_fall = (self.torso_pos[:, 2] < self.cfg.fall_height) & ~self.fallen
        self.fallen |= new_fall
        fall = self.cfg.fall_penalty * new_fall.float() / self.step_dt

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

        # log terms separately
        log = self.extras.setdefault("log", {})
        log["Reward/ball_speed"] = ball_reward.mean()
        log["Reward/finish"] = finish.mean()
        log["Reward/carry"] = carry.mean()
        log["Reward/launch"] = launch.mean()
        log["Reward/attack"] = attack.mean()
        log["Reward/hold"] = hold.mean()
        log["Reward/posture"] = posture.mean()
        if self.impacted.any():
            log["Metrics/spine_deg_after_impact"] = self.spine_deg[self.impacted].mean()
        log["Reward/balance"] = -balance.mean()
        log["Reward/weight_shift"] = shift.mean()
        log["Reward/impact_shift"] = impact_shift.mean()
        log["Metrics/xcom_out_cm"] = 100.0 * self.xcom_out.mean()
        log["Reward/backswing"] = wind.mean()
        log["Reward/clubhead_speed"] = speed.mean()
        log["Reward/takeaway"] = -takeaway.mean()
        log["Reward/trail_hand"] = -trail.mean()
        log["Metrics/trail_grip_cm"] = 100.0 * self.grip_gap.mean()
        log["Reward/stand"] = -stand.mean()
        log["Reward/feet"] = -feet.mean()
        log["Reward/lead_foot"] = -lead.mean()
        log["Reward/trail_foot"] = -trail_foot.mean()
        log["Reward/fall"] = -fall.mean()
        log["Reward/dig"] = -dig.mean()
        log["Reward/backwards"] = -backwards.mean()
        log["Reward/actions"] = -self.cfg.actions_cost_scale * actions_cost.mean()
        log["Reward/energy"] = -self.cfg.energy_cost_scale * energy_cost.mean()
        log["Reward/limits"] = -self.cfg.joint_pos_limits_cost_scale * limit_cost.mean()
        log["Metrics/torso_height"] = self.torso_pos[:, 2].mean()
        if self.cfg.motor_noise > 0:
            log["Metrics/motor_noise_frac"] = (self.cfg.motor_noise * self.motor_n).std()
        if self.cfg.muscle:
            log["Metrics/muscle_torque_frac"] = self.muscle_frac.mean()
        log["Metrics/club_ball_dist"] = self.club_ball_dist.mean()
        log["Metrics/clubhead_speed_mph"] = MPH * self.club_speed.mean()
        log["Metrics/ball_speed_mph"] = MPH * ball_speed.mean()

        # per-second terms, scaled by step_dt
        return self.step_dt * (
            ball_reward
            + carry
            + launch
            + attack
            + hold
            + posture
            - balance
            + shift
            + impact_shift
            + wind
            + speed
            - takeaway
            - trail
            - stand
            - feet
            - lead
            - trail_foot
            - fall
            - dig
            - backwards
            - self.cfg.actions_cost_scale * actions_cost
            - self.cfg.energy_cost_scale * energy_cost
            - self.cfg.joint_pos_limits_cost_scale * limit_cost
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        self._update_contact()
        self._update_balance()
        if self.tracer is not None:
            self._draw_tracer()
        # falling is penalized
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.fall_terminate:
            died = self.torso_pos[:, 2] < self.cfg.fall_height
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
        log["Metrics/fell_rate"] = self.fallen[env_ids].float().mean()
        log["Metrics/lead_foot_moved_m"] = torch.linalg.norm(
            self.feet_xy[env_ids, 0] - self.feet_home[env_ids, 0], dim=-1
        ).mean()
        struck = self.impacted[env_ids]
        if struck.any():
            log["Metrics/trail_lift_cm"] = 100.0 * self.trail_lift[env_ids][struck].mean()
            # at the strike: human-plausibility checks
            log["Metrics/impact_speed_mph"] = MPH * self.impact_speed[env_ids][struck].mean()
            log["Metrics/impact_hand_speed_mph"] = MPH * self.impact_hand_speed[env_ids][struck].mean()
            log["Metrics/impact_time"] = self.impact_time[env_ids][struck].mean()
            log["Metrics/smash"] = self.smash[env_ids][struck].mean()
            off = 100.0 * self.strike_off[env_ids][struck]
            log["Metrics/strike_heel_cm"] = off[:, 0].mean()
            log["Metrics/strike_high_cm"] = off[:, 1].mean()
            log["Metrics/strike_heel_abs_cm"] = off[:, 0].abs().mean()
            log["Metrics/strike_high_abs_cm"] = off[:, 1].abs().mean()
            log["Metrics/backwards_rate"] = self.backwards[env_ids][struck].float().mean()
            log["Metrics/offline_deg"] = self.offline[env_ids][struck].abs().mean()
            log["Metrics/backswing_m"] = self.top_dist[env_ids][struck].mean()
            log["Metrics/top_time"] = self.top_time[env_ids][struck].mean()
            log["Metrics/downswing_time"] = (self.impact_time - self.top_time)[env_ids][struck].mean()
            face, path = self.face_deg[env_ids][struck], self.path_deg[env_ids][struck]
            log["Metrics/face_deg"] = face.mean()
            log["Metrics/path_deg"] = path.mean()
            log["Metrics/face_to_path_deg"] = (face - path).mean()
            log["Metrics/face_to_path_abs_deg"] = (face - path).abs().mean()
            held = struck & ~self.fallen[env_ids]
            if held.any():
                log["Metrics/finish_shift"] = self.shift[env_ids][held].mean()
            if self.ground:
                cls = self.strike_class[env_ids][struck]
                for i, name in enumerate(STRIKE_CLASSES):
                    log[f"Metrics/{name}_rate"] = (cls == i).float().mean()
                log["Metrics/leading_edge_mm"] = 1000.0 * self.leading_edge[env_ids][struck].mean()
                log["Metrics/turf_factor"] = self.turf_factor[env_ids][struck].mean()
            flown = struck & ~self.backwards[env_ids]
            if self.cfg.flight and flown.any():
                f = dict(zip(FLIGHT_KEYS, self.flight_out[env_ids][flown].T))
                for k in ("carry", "side", "curve", "apex"):
                    log[f"Flight/{k}_yd"] = YD * f[k].mean()
                log["Flight/side_abs_yd"] = YD * f["side"].abs().mean()
                log["Flight/curve_abs_yd"] = YD * f["curve"].abs().mean()
                for k in FLIGHT_KEYS[4:]:
                    log[f"Flight/{k}"] = f[k].mean()

        if self.ground:
            log["Metrics/dig_cm2"] = 1e4 * self.dig_total[env_ids].mean()

        super()._reset_idx(env_ids)
        index_fill_(self.actions, env_ids, 0.0)
        index_fill_(self.turf_pre, env_ids, 0.0)
        index_fill_(self.turf_factor, env_ids, 1.0)
        index_fill_(self.dig_total, env_ids, 0.0)
        index_fill_(self.leading_edge, env_ids, 0.0)
        index_fill_(self.strike_class, env_ids, 0)
        index_fill_(self.impacted, env_ids, False)
        index_fill_(self.post_impact, env_ids, 0)
        index_fill_(self.min_dist, env_ids, 1e3)
        index_fill_(self.impact_speed, env_ids, 0.0)
        index_fill_(self.impact_hand_speed, env_ids, 0.0)
        index_fill_(self.impact_time, env_ids, 0.0)
        index_fill_(self.armed, env_ids, False)
        index_fill_(self.backswing, env_ids, 0.0)
        index_fill_(self.top_dist, env_ids, 0.0)
        index_fill_(self.max_ball_speed, env_ids, 0.0)
        index_fill_(self.launch_speed, env_ids, 0.0)
        index_fill_(self.smash, env_ids, 0.0)
        index_fill_(self.sweet, env_ids, 0.0)
        index_fill_(self.strike_off, env_ids, 0.0)
        index_fill_(self.backwards, env_ids, False)
        index_fill_(self.offline, env_ids, 0.0)
        index_fill_(self.face_deg, env_ids, 0.0)
        index_fill_(self.path_deg, env_ids, 0.0)
        index_fill_(self.flight_out, env_ids, 0.0)
        index_fill_(self.top_time, env_ids, 0.0)
        index_fill_(self.trail_lift, env_ids, 0.0)
        index_fill_(self.fallen, env_ids, False)

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
        self.feet_home[env_ids] = self.feet_xy[env_ids]
        self.feet_home_z[env_ids] = self.feet_z[env_ids]
        if self.loft_normal is None:
            up = quat_apply_inverse(self.club_quat[env_ids], torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
                                    .expand(len(env_ids), 3)).mean(0)
            up[0] = 0.0
            if self.ground:
                # bottom corners: up at address is the sole's normal (the face normal is level at address)
                u_n = up / up.norm()
                x_n = torch.tensor([1.0, 0.0, 0.0], device=self.device)
                t_n = torch.linalg.cross(u_n, x_n)
                hx, hy, hz = self.head_half.tolist()
                self.head_corners = torch.stack([a * hx * x_n + b * hy * t_n - hz * u_n for a in (1.0, -1.0) for b in (1.0, -1.0)])
            loft = math.radians(CLUBS[self.cfg.club].loft_deg)
            self.loft_normal = math.sin(loft) * up / up.norm()
            self.loft_normal[0] = math.cos(loft)
        if self.grip_off is None:
            # median over the jittered resets: one spot on the grip
            rel = quat_apply_inverse(self.club_quat[env_ids], self.trail_pos[env_ids] - self.club_pos[env_ids])
            self.grip_off = rel.median(0).values
        if self.cfg.motor_noise > 0:
            self.motor_n[env_ids] = torch.randn_like(self.motor_n[env_ids])
        self.prev_club_pos[env_ids] = self.club_pos[env_ids]
        self.prev_club_quat[env_ids] = self.club_quat[env_ids]
        self.prev_club_vel[env_ids] = self.club_vel[env_ids]
        self.prev_ball_pos[env_ids] = self.ball_pos[env_ids]
