"""NumPy/MotrixSim environment for the VBot Section01 course."""

from pathlib import Path

import gymnasium as gym
import motrixsim as mtx
import numpy as np
from PIL import Image

from motrix_envs import registry
from motrix_envs.math.quaternion import Quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import (
    ACTION_DIM,
    FULL_COURSE,
    OBSERVATION_DIM,
    UPHILL_REFERENCE_V2,
    VBotSection01Cfg,
)
from .control import (
    compose_joint_targets,
    forward_velocity_progress,
    route_errors,
    success_mask,
    terrain_blend,
)


DEFAULT_JOINT_ANGLES = np.asarray(
    (0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8),
    dtype=np.float32,
)
FORCE_LIMITS = np.asarray((17.0, 17.0, 34.0) * 4, dtype=np.float32)
LEG_ORDER = (0, 3, 1, 2)  # FR -> RL -> FL -> RR
SCAN_FORWARD = np.asarray((0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.15, 1.30), dtype=np.float32)
SCAN_LATERAL = np.asarray((0.16, 0.0, -0.16), dtype=np.float32)


def _yaw_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion).T
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class VBotSection01Env(NpEnv):
    """Shared implementation used by all curriculum stages."""

    def __init__(self, cfg: VBotSection01Cfg, num_envs: int = 1):
        super().__init__(cfg, num_envs)
        self._body = self.model.get_body("base")
        self._action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_DIM,), dtype=np.float32)
        self._init_dof_pos = self.model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros(self.model.num_dof_vel, dtype=np.float32)
        self._joint_pos_indices = np.asarray(self.model.joint_dof_pos_indices[3:15], dtype=np.int64)
        self._joint_limits = np.asarray(self.model.joint_limits[:, 3:15], dtype=np.float32)
        self._uphill_reference = np.asarray(UPHILL_REFERENCE_V2, dtype=np.float32)
        self._gravity = np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
        heightmap_path = Path(cfg.model_file).with_name("assets") / "preview" / "H_section_01_0122_16.png"
        height_image = np.asarray(Image.open(heightmap_path), dtype=np.float32)
        if height_image.ndim == 3:
            height_image = height_image[..., 0]
        height_range = float(height_image.max() - height_image.min())
        self._heightmap = (height_image - height_image.min()) / max(height_range, 1.0)
        self._foot_sensor_names = tuple(
            f"{leg}_foot_contact_{section}"
            for leg in ("FR", "FL", "RR", "RL")
            for section in (1, 2, 3)
        )
        self._base_sensor_names = tuple(f"base_contact_{section}" for section in (1, 2, 3))

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def get_dof_pos(self, data: mtx.SceneData) -> np.ndarray:
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData) -> np.ndarray:
        return self._body.get_joint_dof_vel(data)

    def _pose_metrics(self, data: mtx.SceneData):
        pose = self._body.get_pose(data)
        quaternion = pose[:, 3:7]
        projected_gravity = Quaternion.rotate_inverse(quaternion, self._gravity)
        yaw = _yaw_from_xyzw(quaternion)
        lateral, heading = route_errors(pose[:, 0], yaw)
        upright = -projected_gravity[:, 2]
        return pose, projected_gravity, lateral, heading, upright

    def _contacts(self, data: mtx.SceneData) -> tuple[np.ndarray, np.ndarray]:
        foot_values = []
        for leg_index in range(4):
            first = leg_index * 3
            section_values = [
                np.asarray(self.model.get_sensor_value(name, data)).reshape(data.shape[0], -1).max(axis=1)
                for name in self._foot_sensor_names[first : first + 3]
            ]
            foot_values.append(np.maximum.reduce(section_values))
        feet = np.stack(foot_values, axis=1) > 1e-5
        base_values = [
            np.asarray(self.model.get_sensor_value(name, data)).reshape(data.shape[0], -1).max(axis=1)
            for name in self._base_sensor_names
        ]
        base = np.maximum.reduce(base_values) > 1e-5
        return feet, base

    def _mode_parameters(self, y: np.ndarray):
        cfg = self.cfg.control
        rough_weight = terrain_blend(y, FULL_COURSE.rough_start_y - 0.15, FULL_COURSE.rough_start_y + 0.15)
        uphill_weight = terrain_blend(y, FULL_COURSE.uphill_blend_start, FULL_COURSE.uphill_blend_end)
        if self.cfg.stage == "s1":
            rough_weight.fill(0.0)
            uphill_weight.fill(0.0)
        elif self.cfg.stage == "s2":
            uphill_weight.fill(0.0)
        speed = cfg.flat_speed + rough_weight * (cfg.rough_speed - cfg.flat_speed)
        speed += uphill_weight * (cfg.uphill_speed - cfg.rough_speed)
        if self.cfg.stable_hold_seconds > 0.0:
            stop_weight = terrain_blend(
                y, FULL_COURSE.target_y - 0.60, FULL_COURSE.target_y - 0.10
            )
            speed *= 1.0 - stop_weight
        frequency = cfg.flat_frequency + rough_weight * (cfg.rough_frequency - cfg.flat_frequency)
        frequency += uphill_weight * (cfg.uphill_frequency - cfg.rough_frequency)
        residual_scale = cfg.flat_residual_scale + rough_weight * (
            cfg.rough_residual_scale - cfg.flat_residual_scale
        )
        residual_scale += uphill_weight * (cfg.uphill_residual_scale - cfg.rough_residual_scale)
        return speed, frequency, residual_scale, rough_weight, uphill_weight

    def _reference_gait(
        self,
        steps: np.ndarray,
        frequency: np.ndarray,
        rough_weight: np.ndarray,
        uphill_weight: np.ndarray,
        gait_phase_offset: np.ndarray,
    ) -> np.ndarray:
        cycle = (
            steps * self.cfg.ctrl_dt * frequency * 4.0
            + gait_phase_offset
            + uphill_weight * self.cfg.control.uphill_phase_offset
        ) % 4.0
        startup = np.clip(steps * self.cfg.ctrl_dt / 1.0, 0.0, 1.0)
        cfg = self.cfg.control
        thigh_stride = cfg.flat_thigh_stride + rough_weight * (
            cfg.rough_thigh_stride - cfg.flat_thigh_stride
        )
        thigh_stride += uphill_weight * (cfg.uphill_thigh_stride - cfg.rough_thigh_stride)
        thigh_lift = cfg.flat_thigh_lift + rough_weight * (cfg.rough_thigh_lift - cfg.flat_thigh_lift)
        thigh_lift += uphill_weight * (cfg.uphill_thigh_lift - cfg.rough_thigh_lift)
        calf_lift_amount = cfg.flat_calf_lift + rough_weight * (
            cfg.rough_calf_lift - cfg.flat_calf_lift
        )
        calf_lift_amount += uphill_weight * (cfg.uphill_calf_lift - cfg.rough_calf_lift)
        targets = np.tile(DEFAULT_JOINT_ANGLES, (steps.shape[0], 1))
        for order_index, leg_index in enumerate(LEG_ORDER):
            leg_phase = (cycle - order_index) % 4.0
            swing = leg_phase < 1.0
            swing_progress = np.clip(leg_phase, 0.0, 1.0)
            stance_progress = np.clip((leg_phase - 1.0) / 3.0, 0.0, 1.0)
            swing_arc = np.sin(np.pi * swing_progress)
            thigh_offset = np.where(
                swing,
                thigh_stride * (1.0 - 2.0 * swing_progress) + thigh_lift * swing_arc,
                thigh_stride * (-1.0 + 2.0 * stance_progress),
            )
            calf_lift = np.where(swing, calf_lift_amount * swing_arc, 0.0)
            targets[:, leg_index * 3 + 1] += startup * thigh_offset
            targets[:, leg_index * 3 + 2] += startup * calf_lift
        return targets

    def _joint_targets(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        pose, _, lateral, heading, _ = self._pose_metrics(state.data)
        mode_speed, frequency, residual_scale, rough_weight, uphill_weight = self._mode_parameters(
            pose[:, 1]
        )
        reference = self._reference_gait(
            state.info["steps"],
            frequency,
            rough_weight,
            uphill_weight,
            state.info["gait_phase_offset"],
        )
        gait_scale = np.ones_like(mode_speed)
        fade_start = self.cfg.control.reference_fade_start_y
        fade_end = self.cfg.control.reference_fade_end_y
        if fade_end > fade_start:
            gait_scale = 1.0 - np.clip(
                (pose[:, 1] - fade_start) / (fade_end - fade_start), 0.0, 1.0
            )
        elif self.cfg.control.scale_reference_with_command:
            command_speed = self._command_speed(pose[:, 1])
            gait_scale = np.clip(
                command_speed / np.maximum(mode_speed, 1e-4), 0.0, 1.0
            )
        if np.any(gait_scale < 1.0):
            reference = DEFAULT_JOINT_ANGLES[None, :] + gait_scale[:, None] * (
                reference - DEFAULT_JOINT_ANGLES[None, :]
            )
        correction = np.clip(
            -self.cfg.control.turn_gain * heading
            + self.cfg.control.lateral_turn_gain * lateral,
            -0.10,
            0.10,
        )
        route_turn = np.zeros_like(reference)
        route_turn[:, (1, 7)] = -correction[:, None] * gait_scale[:, None]
        route_turn[:, (4, 10)] = correction[:, None] * gait_scale[:, None]
        uphill = (
            uphill_weight[:, None]
            * gait_scale[:, None]
            * self._uphill_reference[None, :]
        )
        if self.cfg.control.scale_residual_with_reference:
            residual_floor = np.clip(
                self.cfg.control.residual_reference_floor, 0.0, 1.0
            )
            residual_scale = residual_scale * (
                residual_floor + (1.0 - residual_floor) * gait_scale
            )
        return compose_joint_targets(
            reference,
            route_turn,
            uphill,
            actions,
            residual_scale,
            self._joint_limits[0],
            self._joint_limits[1],
        )

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, ACTION_DIM)
        if actions.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {actions.shape}")
        clipped_actions = np.clip(actions, -1.0, 1.0)
        if self.cfg.control.direct_action_scale > 0.0:
            alpha = self.cfg.control.action_filter_alpha
            applied_actions = (
                alpha * clipped_actions
                + (1.0 - alpha) * state.info["applied_actions"]
            )
            targets = (
                DEFAULT_JOINT_ANGLES[None, :]
                + self.cfg.control.direct_action_scale * applied_actions
            )
            targets = np.clip(targets, self._joint_limits[0], self._joint_limits[1])
        else:
            applied_actions = clipped_actions
            targets = self._joint_targets(applied_actions, state)
        joint_pos = self.get_dof_pos(state.data)
        joint_vel = self.get_dof_vel(state.data)
        torques = self.cfg.control.stiffness * (targets - joint_pos) - self.cfg.control.damping * joint_vel
        state.data.actuator_ctrls = np.clip(torques, -FORCE_LIMITS, FORCE_LIMITS)
        state.info["last_actions"] = state.info["current_actions"].copy()
        state.info["current_actions"] = (
            applied_actions
            if self.cfg.control.direct_action_scale > 0.0
            else actions
        )
        state.info["last_applied_actions"] = state.info["applied_actions"].copy()
        state.info["applied_actions"] = applied_actions
        return state

    def _reset_done_envs(self):
        """Reset done rows while retaining their global curriculum identities."""
        state = self._state
        done = state.done
        assert done.shape == (self._num_envs,)
        if not np.any(done):
            return

        np.putmask(state.info["steps"], done, 0)
        obs, reset_info = self.reset(state.data[done], done=done)
        state.obs[done] = obs

        def replace_dict_values(dst, values, mask):
            for key, value in values.items():
                if key not in dst:
                    dst[key] = value
                elif isinstance(value, np.ndarray):
                    dst[key][mask] = value
                elif isinstance(value, dict):
                    replace_dict_values(dst[key], value, mask)

        replace_dict_values(state.info, reset_info, done)

    def _heightmap_sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        rows, columns = self._heightmap.shape
        image_x = np.clip((x + 5.0) / 10.0 * (columns - 1), 0.0, columns - 1)
        image_y = np.clip((y + 1.5) / 3.0 * (rows - 1), 0.0, rows - 1)
        x0 = np.floor(image_x).astype(np.int64)
        y0 = np.floor(image_y).astype(np.int64)
        x1 = np.minimum(x0 + 1, columns - 1)
        y1 = np.minimum(y0 + 1, rows - 1)
        wx = image_x - x0
        wy = image_y - y0
        upper = self._heightmap[y0, x0] * (1.0 - wx) + self._heightmap[y0, x1] * wx
        lower = self._heightmap[y1, x0] * (1.0 - wx) + self._heightmap[y1, x1] * wx
        return (upper * (1.0 - wy) + lower * wy) * 0.277056

    def _terrain_height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        rough = self._heightmap_sample(x, y)
        slope = np.clip((y - 1.75) * np.tan(np.deg2rad(15.0)), 0.0, 1.294095)
        return np.where(y < -1.5, 0.0, np.where(y <= 1.5, rough, slope))

    def _height_scan(self, data: mtx.SceneData) -> np.ndarray:
        pose = self._body.get_pose(data)
        yaw = _yaw_from_xyzw(pose[:, 3:7])
        forward = np.tile(SCAN_FORWARD, len(SCAN_LATERAL))
        lateral = np.repeat(SCAN_LATERAL, len(SCAN_FORWARD))
        sample_x = pose[:, 0, None] + np.cos(yaw)[:, None] * forward - np.sin(yaw)[:, None] * lateral
        sample_y = pose[:, 1, None] + np.sin(yaw)[:, None] * forward + np.cos(yaw)[:, None] * lateral
        terrain_height = self._terrain_height(sample_x, sample_y)
        return np.clip(terrain_height - pose[:, 2, None], -1.5, 1.5).astype(np.float32)

    def _get_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        pose, projected_gravity, lateral, heading, _ = self._pose_metrics(data)
        quaternion = pose[:, 3:7]
        world_linvel = self.model.get_sensor_value("base_linvel", data)
        local_linvel = Quaternion.rotate_inverse(quaternion, world_linvel)
        gyro = self.model.get_sensor_value("base_gyro", data)
        joint_pos = self.get_dof_pos(data) - DEFAULT_JOINT_ANGLES
        joint_vel = self.get_dof_vel(data)
        speed = self._command_speed(pose[:, 1])
        commands = np.stack((speed, np.zeros_like(speed), -heading), axis=1)
        height_scan = self._height_scan(data) if self.cfg.enable_height_scan else np.zeros((data.shape[0], 24))
        contacts, _ = self._contacts(data)
        obs = np.hstack(
            (
                local_linvel,
                gyro,
                projected_gravity,
                joint_pos,
                joint_vel,
                info["current_actions"],
                commands,
                height_scan,
                contacts.astype(np.float32),
                heading[:, None],
                lateral[:, None],
            )
        ).astype(np.float32)
        if obs.shape != (data.shape[0], OBSERVATION_DIM):
            raise RuntimeError(f"unexpected observation shape {obs.shape}")
        return obs

    def _command_speed(self, y: np.ndarray) -> np.ndarray:
        speed, _, _, _, _ = self._mode_parameters(y)
        if self.cfg.stable_hold_seconds <= 0.0:
            return speed
        brake_span = max(self.cfg.platform_y_min - self.cfg.brake_start_y, 1e-6)
        brake_fraction = np.clip(
            (self.cfg.platform_y_min - y) / brake_span, 0.0, 1.0
        )
        speed = self.cfg.brake_min_speed + brake_fraction * (
            speed - self.cfg.brake_min_speed
        )
        return np.where(y >= self.cfg.platform_y_min, 0.0, speed)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        pose, _, lateral, heading, upright = self._pose_metrics(state.data)
        feet, base_contact = self._contacts(state.data)
        world_velocity = self.model.get_sensor_value("base_linvel", state.data)
        angular_velocity = self.model.get_sensor_value("base_gyro", state.data)
        on_platform = (
            (pose[:, 1] >= self.cfg.platform_y_min)
            & (np.abs(pose[:, 0]) <= self.cfg.platform_x_abs_max)
            & (pose[:, 2] >= self.cfg.platform_base_z_min)
            & (upright >= FULL_COURSE.upright_threshold)
        )
        stable_candidate = (
            on_platform
            & (
                np.linalg.norm(world_velocity[:, :2], axis=1)
                <= self.cfg.stable_linear_speed_max
            )
            & (np.abs(world_velocity[:, 2]) <= self.cfg.stable_vertical_speed_max)
            & (np.linalg.norm(angular_velocity, axis=1) <= self.cfg.stable_angular_speed_max)
        )
        stable_steps = np.where(
            stable_candidate, state.info["stable_steps"] + 1, 0
        ).astype(np.int32)
        max_stable_steps = np.maximum(
            state.info["max_stable_steps"], stable_steps
        ).astype(np.int32)
        ever_on_platform = state.info["ever_on_platform"] | on_platform
        if self.cfg.stable_hold_seconds > 0.0:
            required_steps = max(1, round(self.cfg.stable_hold_seconds / self.cfg.ctrl_dt))
            success = stable_steps >= required_steps
        elif self.cfg.stage == "full":
            success = success_mask(pose[:, 0], pose[:, 1], heading, upright)
        else:
            success = (
                (pose[:, 1] >= self.cfg.course_end_y)
                & (np.abs(lateral) <= FULL_COURSE.target_x_tolerance)
                & (upright >= FULL_COURSE.upright_threshold)
            )
        failed = (upright < 0.5) | (pose[:, 2] < 0.18) | (np.abs(lateral) > 1.25) | base_contact
        terminated = success | failed
        reward = self._reward(
            state,
            pose,
            heading,
            lateral,
            upright,
            success,
            failed,
            on_platform,
            stable_candidate,
        )
        episode_max_y = np.maximum(state.info["max_y"], pose[:, 1])
        reason = np.full(self.num_envs, "running", dtype="<U16")
        reason[success] = "success"
        reason[(upright < 0.5) | (pose[:, 2] < 0.18)] = "fallen"
        reason[np.abs(lateral) > 1.25] = "lateral"
        reason[base_contact & ~success] = "base_contact"
        state.info["contacts"] = feet
        state.info["success"] = success
        state.info["episode_success"] = success.copy()
        state.info["stable_steps"] = stable_steps
        state.info["ever_on_platform"] = ever_on_platform
        state.info["episode_ever_on_platform"] = ever_on_platform.copy()
        state.info["episode_stable_success"] = (
            success.copy()
            if self.cfg.stable_hold_seconds > 0.0
            else np.zeros(self.num_envs, dtype=bool)
        )
        state.info["episode_max_stable_steps"] = max_stable_steps.copy()
        state.info["episode_max_y"] = episode_max_y.copy()
        state.info["final_x"] = pose[:, 0].copy()
        state.info["final_y"] = pose[:, 1].copy()
        state.info["final_heading"] = heading.copy()
        state.info["final_upright"] = upright.copy()
        state.info["termination_reason"] = reason
        state.info["max_y"] = episode_max_y
        state.info["max_stable_steps"] = max_stable_steps
        state.info["last_y"] = pose[:, 1].copy()
        obs = self._get_obs(state.data, state.info)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _reward(
        self,
        state,
        pose,
        heading,
        lateral,
        upright,
        success,
        failed,
        on_platform,
        stable_candidate,
    ):
        cfg = self.cfg.reward
        world_velocity = self.model.get_sensor_value("base_linvel", state.data)
        target_speed = self._command_speed(pose[:, 1])
        tracking = np.exp(-np.square(world_velocity[:, 1] - target_speed) / 0.04)
        progress = forward_velocity_progress(world_velocity[:, 1])
        if self.cfg.stable_hold_seconds > 0.0:
            brake_span = max(self.cfg.platform_y_min - self.cfg.brake_start_y, 1e-6)
            route_scale = np.clip(
                (self.cfg.platform_y_min - pose[:, 1]) / brake_span, 0.0, 1.0
            )
            progress *= route_scale
        action_rate = np.sum(
            np.square(state.info["applied_actions"] - state.info["last_applied_actions"]), axis=1
        )
        action_magnitude = np.sum(np.square(state.info["applied_actions"]), axis=1)
        stall = world_velocity[:, 1] < 0.03
        if self.cfg.stable_hold_seconds > 0.0:
            stall &= ~on_platform
        platform_motion = np.clip(
            np.sum(np.square(world_velocity), axis=1)
            + 0.25 * np.sum(np.square(self.model.get_sensor_value("base_gyro", state.data)), axis=1),
            0.0,
            4.0,
        )
        horizontal_speed = np.linalg.norm(world_velocity[:, :2], axis=1)
        angular_speed = np.linalg.norm(
            self.model.get_sensor_value("base_gyro", state.data), axis=1
        )
        brake_zone = (
            (pose[:, 1] >= self.cfg.brake_start_y)
            & (pose[:, 1] < self.cfg.platform_y_min)
        )
        brake_tracking = np.exp(
            -np.square(
                (horizontal_speed - target_speed) / self.cfg.brake_tracking_sigma
            )
        ) * brake_zone
        stand_quality = np.exp(
            -np.square(horizontal_speed / self.cfg.stable_linear_speed_max)
            -np.square(
                np.abs(world_velocity[:, 2]) / self.cfg.stable_vertical_speed_max
            )
            -np.square(angular_speed / self.cfg.stable_angular_speed_max)
        ) * on_platform
        reward = (
            cfg.route_tracking * tracking
            + cfg.progress * progress
            + cfg.orientation * np.clip(upright, 0.0, 1.0)
            + cfg.heading_penalty * np.square(heading)
            + cfg.lateral_penalty * np.square(lateral)
            + cfg.action_rate * action_rate
            + cfg.action_magnitude * action_magnitude
            + cfg.stall_penalty * stall
            + cfg.platform_step * on_platform
            + cfg.stable_step * stable_candidate
            + cfg.platform_motion_penalty * platform_motion * on_platform
            + cfg.brake_tracking * brake_tracking
            + cfg.platform_stand_reward * stand_quality
            + cfg.success_bonus * success
            + cfg.termination_penalty * failed
        )
        return np.asarray(reward, dtype=np.float32)

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        num_reset = data.shape[0]
        global_indices = np.arange(num_reset) if done is None else np.flatnonzero(done)
        if global_indices.size != num_reset:
            raise ValueError("done mask and reset data must identify the same number of environments")
        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_reset, 1))
        dof_pos[:, 0] = 0.0
        dof_pos[:, 1] = FULL_COURSE.target_y
        dof_pos[:, 2] = 0.0
        if self.cfg.stage == "s3a":
            # Start on the short level approach so the nominal standing pose can
            # settle before the front feet contact the 15-degree ramp.
            start_x = np.random.uniform(
                *self.cfg.s3a_start_x_range, size=num_reset
            ).astype(np.float32)
            start_y = np.random.uniform(
                *self.cfg.s3a_start_y_range, size=num_reset
            ).astype(np.float32)
            start_z = self._terrain_height(start_x, start_y) + 0.462
            dof_pos[:, 3] = start_x
            dof_pos[:, 4] = start_y
            dof_pos[:, 5] = start_z
            half_yaw = np.pi / 4.0
            dof_pos[:, 6:10] = (0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw))
        else:
            start_y = np.random.uniform(
                *self.cfg.start_y_range, size=num_reset
            ).astype(np.float32)
            if self.num_envs > 1 and self.cfg.training_fixed_y_fraction > 0.0:
                fixed_y = np.random.random(num_reset) < self.cfg.training_fixed_y_fraction
                start_y[fixed_y] = FULL_COURSE.start_y
            if (
                self.cfg.stage == "full"
                and self.num_envs > 1
                and self.cfg.use_full_course_local_starts
            ):
                # Parallel training deliberately rehearses the two hard
                # transitions as well as complete starts. Evaluation always
                # uses one environment and therefore always starts at -2.4 m.
                start_y[global_indices % 4 == 1] = -1.0
                start_y[global_indices % 4 == 2] = 1.0
            start_x = np.random.uniform(
                *self.cfg.start_x_range, size=num_reset
            ).astype(np.float32)
            if self.num_envs > 1 and self.cfg.training_narrow_x_fraction > 0.0:
                narrow_x = (
                    np.random.random(num_reset)
                    < self.cfg.training_narrow_x_fraction
                )
                start_x[narrow_x] = np.random.uniform(
                    *self.cfg.training_narrow_x_range,
                    size=int(np.sum(narrow_x)),
                )
            if self.num_envs > 1 and self.cfg.training_fixed_x_fraction > 0.0:
                fixed_x = np.random.random(num_reset) < self.cfg.training_fixed_x_fraction
                start_x[fixed_x] = 0.0
            if self.num_envs > 1 and (
                self.cfg.training_platform_start_fraction > 0.0
                or self.cfg.training_brake_start_fraction > 0.0
            ):
                curriculum_draw = np.random.random(num_reset)
                platform_start = (
                    curriculum_draw < self.cfg.training_platform_start_fraction
                )
                brake_start = (
                    curriculum_draw >= self.cfg.training_platform_start_fraction
                ) & (
                    curriculum_draw
                    < self.cfg.training_platform_start_fraction
                    + self.cfg.training_brake_start_fraction
                )
                platform_count = int(np.sum(platform_start))
                start_x[platform_start] = np.random.uniform(
                    *self.cfg.start_x_range, size=platform_count
                )
                start_y[platform_start] = np.random.uniform(
                    *self.cfg.training_platform_y_range, size=platform_count
                )
                brake_count = int(np.sum(brake_start))
                start_x[brake_start] = np.random.uniform(
                    *self.cfg.start_x_range, size=brake_count
                )
                start_y[brake_start] = np.random.uniform(
                    *self.cfg.training_brake_y_range, size=brake_count
                )
                dof_vel[brake_start, 4] = np.random.uniform(
                    *self.cfg.training_brake_speed_range, size=brake_count
                )
            start_z = self._terrain_height(start_x, start_y) + 0.462
            dof_pos[:, 3] = start_x
            dof_pos[:, 4] = start_y
            dof_pos[:, 5] = start_z
            yaw = np.pi / 2.0 + np.random.uniform(
                -self.cfg.initial_yaw_noise,
                self.cfg.initial_yaw_noise,
                size=num_reset,
            )
            dof_pos[:, 6] = 0.0
            dof_pos[:, 7] = 0.0
            dof_pos[:, 8] = np.sin(yaw / 2.0)
            dof_pos[:, 9] = np.cos(yaw / 2.0)
        dof_pos[:, self._joint_pos_indices] = DEFAULT_JOINT_ANGLES
        data.reset(self.model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self.model)
        self.model.forward_kinematic(data)
        pose = self._body.get_pose(data)
        info = {
            "current_actions": np.zeros((num_reset, ACTION_DIM), dtype=np.float32),
            "last_actions": np.zeros((num_reset, ACTION_DIM), dtype=np.float32),
            "applied_actions": np.zeros((num_reset, ACTION_DIM), dtype=np.float32),
            "last_applied_actions": np.zeros((num_reset, ACTION_DIM), dtype=np.float32),
            "contacts": np.zeros((num_reset, 4), dtype=bool),
            "success": np.zeros(num_reset, dtype=bool),
            "stable_steps": np.zeros(num_reset, dtype=np.int32),
            "max_stable_steps": np.zeros(num_reset, dtype=np.int32),
            "ever_on_platform": np.zeros(num_reset, dtype=bool),
            "gait_phase_offset": (
                np.random.uniform(
                    *self.cfg.training_gait_phase_offset_range,
                    size=num_reset,
                ).astype(np.float32)
                if (
                    self.num_envs > 1
                    and self.cfg.training_gait_phase_offset_range[0]
                    != self.cfg.training_gait_phase_offset_range[1]
                )
                else np.zeros(num_reset, dtype=np.float32)
            ),
            "last_y": pose[:, 1].copy(),
            "max_y": pose[:, 1].copy(),
        }
        return self._get_obs(data, info), info


for _name in (
    "vbot-section01-s1-velocity-course",
    "vbot-section01-s2-terrain-course",
    "vbot-section01-s3a-uphill-course",
    "vbot-section01-full-course-v2-train",
    "vbot-section01-peer-xy-s1-course",
    "vbot-section01-peer-xy-s2-course",
    "vbot-section01-peer-xy-s3a-course",
    "vbot-section01-peer-xy-full-course",
    "vbot-section01-full-random-x-course",
    "vbot-section01-full-random-x10-course",
    "vbot-section01-full-random-x25-course",
    "vbot-section01-full-random-x10-mix50-course",
    "vbot-section01-full-random-x10-mix75-course",
    "vbot-section01-direct-fixed-course",
    "vbot-section01-direct-random-x10-course",
    "vbot-section01-direct-random-x10-y10-mix50-course",
    "vbot-section01-direct-random-x10-y10-mix75-course",
    "vbot-section01-direct-random-x10-y25-course",
    "vbot-section01-direct-random-x10-yfull-course",
    "vbot-section01-direct-random-x25-course",
    "vbot-section01-direct-random-x25-yfull-course",
    "vbot-section01-direct-random-x50-course",
    "vbot-section01-direct-random-xy10-mix50-course",
    "vbot-section01-direct-random-xy10-mix75-course",
    "vbot-section01-direct-random-xy-course",
    "vbot-section01-direct-random-xy-mix50-course",
    "vbot-section01-direct-random-xy-mix75-course",
    "vbot-section01-direct-random-xy-mix65-course",
    "vbot-section01-direct-random-xy-x25mix50-course",
    "vbot-section01-direct-random-xy-x25mix75-course",
    "vbot-section01-direct-random-xy-yaw-course",
    "vbot-section01-direct-random-xy-yaw-neg-x-course",
    "vbot-section01-direct-random-xy-yaw-pos-x-course",
    "vbot-section01-direct-random-xy-yaw-stable-course",
    "vbot-section01-direct-random-xy-yaw-stable-neg-x-course",
    "vbot-section01-direct-random-xy-yaw-stable-pos-x-course",
    "vbot-section01-direct-random-xy-yaw-stable-v2-course",
    "vbot-section01-direct-random-xy-yaw-stable-v2-neg-x-course",
    "vbot-section01-direct-random-xy-yaw-stable-v2-pos-x-course",
    "vbot-section01-direct-random-xy-yaw-stable-v3-course",
    "vbot-section01-direct-random-xy-yaw-stable-v3-neg-x-course",
    "vbot-section01-direct-random-xy-yaw-stable-v3-pos-x-course",
    "vbot-section01-direct-random-xy-yaw-stable-v4-course",
    "vbot-section01-direct-random-xy-yaw-stable-v4-neg-x-course",
    "vbot-section01-direct-random-xy-yaw-stable-v4-pos-x-course",
    "vbot-section01-full-random-xy-course",
    "vbot-section01-full-random-xy-yaw-course",
    "vbot-section01-full-random-xy-yaw-stable-course",
    "vbot-section01-peer-xy-yaw-stable-v2-course",
    "vbot-section01-peer-xy-yaw-stable-v3-course",
    "vbot-section01-peer-xy-yaw-stable-v4-course",
    "vbot-section01-peer-xy-yaw-stable-v4-50-course",
    "vbot-section01-peer-xy-yaw-stable-v5-course",
    "vbot-section01-peer-xy-yaw-stable-v6-course",
    "vbot-section01-full-random-x-route-v2-course",
    "vbot-section01-full-random-xy-route-v2-course",
    "vbot-section01-full-random-xy-yaw-route-v2-course",
    "vbot-section01-full-random-xy-yaw-stable-route-v2-course",
    "vbot-section01-full-random-x-phase-v3-course",
    "vbot-section01-full-random-x-phase-small-v3-course",
    "vbot-section01-full-random-x-phase-medium-v3-course",
):
    registry.register_env(_name, VBotSection01Env, sim_backend="np")
