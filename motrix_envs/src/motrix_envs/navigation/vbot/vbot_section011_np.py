# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection011EnvCfg


def generate_repeating_array(num_period, num_reset, period_counter):
    """
    生成重复数组，用于在固定位置中循环选择
    num_period: 位置总数
    num_reset: 需要重置的环境数
    period_counter: 当前计数器
    """
    idx = []
    for i in range(num_reset):
        idx.append((period_counter + i) % num_period)
    return np.array(idx)


@registry.env("vbot_navigation_section011_curriculum", "np")
@registry.env("vbot_navigation_section011_no_overstay", "np")
@registry.env("vbot_navigation_section011_low_action", "np")
@registry.env("vbot_navigation_section011_safe_progress", "np")
@registry.env("vbot_navigation_section011_go1_transfer", "np")
@registry.env("vbot_navigation_section011_go1_transfer_fast", "np")
@registry.env("vbot_navigation_section011_go1_transfer_medium", "np")
@registry.env("vbot_navigation_section011_go1_transfer_medium_corridor", "np")
@registry.env("vbot_navigation_section011_go1_transfer_fast_corridor", "np")
@registry.env("vbot_navigation_section011_go1_transfer_curriculum", "np")
@registry.env("vbot_navigation_section011_go1_transfer_fast_curriculum", "np")
@registry.env("vbot_navigation_section011_go1_transfer_medium_curriculum", "np")
@registry.env("vbot_navigation_section011_go1_transfer_terrain_skill", "np")
@registry.env("vbot_navigation_section011_go1_transfer_fast_terrain_skill", "np")
@registry.env("vbot_navigation_section011_go1_transfer_fast_terrain_skill_v2", "np")
@registry.env("vbot_navigation_section011_go1_transfer_fast_terrain_skill_v3", "np")
@registry.env("vbot_navigation_section011_rough_skill_v4", "np")
@registry.env("vbot_navigation_section011_rough_skill_v4_safe", "np")
@registry.env("vbot_navigation_section011_rough_skill_v5_stage1", "np")
@registry.env("vbot_navigation_section011_rough_skill_v5_stage1_safe", "np")
@registry.env("vbot_navigation_section011_rough_skill_v6_stage0_scale070", "np")
@registry.env("vbot_navigation_section011_rough_skill_v6_stage0_scale075", "np")
@registry.env("vbot_navigation_section011_rough_skill_v6_stage0_scale080", "np")
@registry.env("vbot_navigation_section011_rough_skill_v6_stage0_scale090", "np")
@registry.env("vbot_navigation_section011_rough_skill_v7_corridor_scale060", "np")
@registry.env("vbot_navigation_section011_rough_skill_v7_corridor_scale070", "np")
@registry.env("vbot_navigation_section011_rough_skill_v7_corridor_scale080", "np")
@registry.env("vbot_navigation_section011_rough_skill_v7_corridor_scale090", "np")
@registry.env("vbot_locomotion_section011_rough_corridor", "np")
@registry.env("vbot_locomotion_section011_rough_corridor_goal_velocity", "np")
@registry.env("vbot_locomotion_section011_rough_corridor_contact", "np")
@registry.env("vbot_locomotion_section011_rough_entry", "np")
@registry.env("vbot_locomotion_section011_rough_entry_stage10", "np")
@registry.env("vbot_locomotion_section011_rough_entry_stage11", "np")
@registry.env("vbot_locomotion_section011_rough_entry_stage112", "np")
@registry.env("vbot_locomotion_section011_rough_entry_stage0", "np")
@registry.env("vbot_locomotion_section011_rough_entry_stage05", "np")
@registry.env("vbot_locomotion_section011_rough_entry_near_edge", "np")
@registry.env("vbot_locomotion_section011_rough_corridor_stage125", "np")
@registry.env("vbot_locomotion_section011_rough_corridor_stage15", "np")
@registry.env("vbot_locomotion_section011_rough_corridor_stage2", "np")
@registry.env("vbot_locomotion_section011_full_route_contact", "np")
@registry.env("vbot_locomotion_section011_full_route_scale16to20", "np")
@registry.env("vbot_locomotion_section011_full_route_scale17to20", "np")
@registry.env("vbot_locomotion_section011_full_route_scale18to20", "np")
@registry.env("vbot_locomotion_section011_full_route_scale19to20", "np")
@registry.env("vbot_locomotion_section011_full_route_scale17to20_late15", "np")
@registry.env("vbot_locomotion_section011_full_route_scale17to20_late14", "np")
@registry.env("vbot_locomotion_section011_full_route_scale17to21", "np")
@registry.env("vbot_locomotion_section011_full_route_scale17to21_late15", "np")
@registry.env("vbot_locomotion_section011_full_route_scale17to1975", "np")
@registry.env("vbot_locomotion_section011_full_route_scale17to2025", "np")
@registry.env("vbot_locomotion_section011_full_route_scale17to205", "np")
@registry.env("vbot_locomotion_section011_full_route_scale17to20_early", "np")
@registry.env("vbot_locomotion_section011_full_route_angular_safe", "np")
@registry.env("vbot_locomotion_section011_full_route_angular_safe_forward06", "np")
@registry.env("vbot_locomotion_section011_approach_stage0", "np")
@registry.env("vbot_locomotion_section011_approach", "np")
@registry.env("vbot_locomotion_section011_integrated_stage0_90", "np")
@registry.env("vbot_locomotion_section011_integrated_stage0_75", "np")
@registry.env("vbot_locomotion_section011_integrated_stage0_50", "np")
@registry.env("vbot_locomotion_section011_integrated_stage1_70", "np")
@registry.env("vbot_locomotion_section011_integrated_stage1_60", "np")
@registry.env("vbot_locomotion_section011_integrated_stage1_50", "np")
@registry.env("vbot_locomotion_section011_integrated_gate105_70", "np")
@registry.env("vbot_locomotion_section011_integrated_gate105_stable_70", "np")
@registry.env("vbot_locomotion_section011_integrated_gate100_stable_70", "np")
@registry.env(
    "vbot_locomotion_section011_integrated_gate100_stable_scale17to20", "np"
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate095_stable_scale17to20", "np"
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate095_hold03_scale17to20", "np"
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate095_dense_safe_scale17to20", "np"
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate095_dense5_safe_scale17to20", "np"
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate095_balanced_safe_scale17to20", "np"
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate095_hold03_dense_safe_scale17to20",
    "np",
)
@registry.env("vbot_locomotion_section011_integrated_forward06_no_skill", "np")
@registry.env(
    "vbot_locomotion_section011_integrated_angular_forward06_no_skill", "np"
)
@registry.env(
    "vbot_locomotion_section011_rough_only_angular_forward06_no_skill", "np"
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate100_stable_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate100_dense_safe_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate100_angular_safe_scale17to20", "np"
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate100_angular_strict_scale17to20",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate100_angular_safe_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate100_angular_strict_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate095_angular_safe_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate095_hold03_angular_safe_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_integrated_gate095_dense_angular_safe_forward06",
    "np",
)
@registry.env("vbot_locomotion_section011_rough080_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_rough075_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_rough080_dense_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_rough080_hold03_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_rough065_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_rough060_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_rough050_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_rough040_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_rough030_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_rough025_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_rough020_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_rough020_hold03_angular_forward06", "np")
@registry.env(
    "vbot_locomotion_section011_rough020_hold03_dense_angular_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_rough020_hold03_dense5_angular_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_rough020_hold03_stop_angular_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_rough020_hold03_stop05_angular_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_rough020_hold03_stop10_angular_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_rough020_hold03_stop15_angular_forward06", "np"
)
@registry.env("vbot_locomotion_section011_rough020_stop15_angular_forward06", "np")
@registry.env(
    "vbot_locomotion_section011_rough020_hold03_dense_stop15_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_rough020_hold03_dense5_stop15_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_rough020_hold03_dense_stop_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_post_second_000_angular_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_post_second_010_angular_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_post_second_030_angular_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_post_second_050_angular_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_post_second_080_angular_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_post_second_100_angular_forward06", "np"
)
@registry.env("vbot_locomotion_section011_post_third_225_angular_forward06", "np")
@registry.env(
    "vbot_locomotion_section011_post_third_225_mixed_handoff_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_post_third_225_mixed_handoff_test_angular_forward06",
    "np",
)
@registry.env("vbot_locomotion_section011_ramp_400_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_ramp_600_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_ramp_top_690_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_platform_780_angular_forward06", "np")
@registry.env(
    "vbot_locomotion_section011_platform_stand_700_angular_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_strict_hold030_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_strict_hold035_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_strict_hold040_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_strict_hold045_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_strict_hold050_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_strict_hold060_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_strict_hold075_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_strict_hold100_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_relaxed_angular_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_relaxed_hold005_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_relaxed_hold010_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_relaxed_hold012_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_relaxed_hold015_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_relaxed_hold018_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_relaxed_hold019_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_relaxed_hold020_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_relaxed_hold025_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_relaxed_hold030_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_relaxed_hold035_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_relaxed_hold040_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_relaxed_hold045_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_platform_stand_relaxed_hold050_angular_forward06",
    "np",
)
@registry.env("vbot_locomotion_section011_mid_bridge_000_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_early_bridge_000_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_early_bridge_m045_angular_forward06", "np")
@registry.env("vbot_locomotion_section011_handoff_m045_angular_forward06", "np")
@registry.env(
    "vbot_locomotion_section011_handoff_healthy_m045_angular_forward06", "np"
)
@registry.env(
    "vbot_locomotion_section011_handoff_healthy_m045_bootstrap_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_healthy_test_m045_bootstrap_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_m045_bootstrap_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_m045_bootstrap_test_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_m025_bootstrap_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_m025_bootstrap_test_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_m010_bootstrap_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_m010_bootstrap_test_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_000_bootstrap_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_000_bootstrap_test_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_p010_bootstrap_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_p010_bootstrap_test_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_p030_bootstrap_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_p030_bootstrap_test_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_p050_bootstrap_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_p050_bootstrap_test_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_p080_bootstrap_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_p080_bootstrap_test_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_p100_bootstrap_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_p100_bootstrap_test_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_p100_recoverable_bootstrap_angular_forward06",
    "np",
)
@registry.env(
    "vbot_locomotion_section011_handoff_seed343_m095_p100_recoverable_bootstrap_test_angular_forward06",
    "np",
)
@registry.env("vbot_locomotion_section011_integrated_gate100_hold10_70", "np")
@registry.env(
    "vbot_locomotion_section011_integrated_gate100_hold10_fall10_70", "np"
)
@registry.env("vbot_locomotion_section011_mixed_route_contact", "np")
@registry.env("vbot_navigation_section011_go1_transfer_fast_corridor_skill", "np")
@registry.env("vbot_navigation_section011", "np")
class VBotSection011Env(NpEnv):
    """
    VBot在Section011地形上的导航任务
    继承自NpEnv，使用VBotSection011EnvCfg配置
    """
    _cfg: VBotSection011EnvCfg
    
    def __init__(self, cfg: VBotSection011EnvCfg, num_envs: int = 1):
        # 调用父类NpEnv初始化
        super().__init__(cfg, num_envs=num_envs)
        
        # 初始化机器人body和接触
        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()
        
        # 获取目标标记的body
        self._target_marker_body = self._model.get_body("target_marker")
        
        # 获取箭头body（用于可视化，不影响物理）
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None
        
        # 动作和观测空间
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # 54 task/proprioceptive features + 8 real terrain-height samples.
        if getattr(cfg, "locomotion_contact_force_observations", False):
            observation_size = 60
        elif getattr(cfg, "locomotion_observations_only", False):
            observation_size = 48
        else:
            observation_size = 62
        self._observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )
        
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators
        
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)

        self._handoff_dof_pos = None
        self._handoff_dof_vel = None
        self._handoff_actions = None
        handoff_state_file = getattr(cfg, "handoff_state_file", None)
        if handoff_state_file:
            with np.load(handoff_state_file, allow_pickle=False) as handoff_states:
                self._handoff_dof_pos = np.asarray(
                    handoff_states["dof_pos"], dtype=np.float32
                )
                self._handoff_dof_vel = np.asarray(
                    handoff_states["dof_vel"], dtype=np.float32
                )
                self._handoff_actions = np.asarray(
                    handoff_states["current_actions"], dtype=np.float32
                )
            sample_count = self._handoff_dof_pos.shape[0]
            expected_shapes = (
                (sample_count, self._model.num_dof_pos),
                (sample_count, self._model.num_dof_vel),
                (sample_count, self._model.num_actuators),
            )
            actual_shapes = (
                self._handoff_dof_pos.shape,
                self._handoff_dof_vel.shape,
                self._handoff_actions.shape,
            )
            if sample_count == 0 or actual_shapes != expected_shapes:
                raise ValueError(
                    "invalid handoff state dataset shapes: "
                    f"expected {expected_shapes}, got {actual_shapes}"
                )
        
        # 查找target_marker的DOF索引
        self._find_target_marker_dof_indices()
        
        # 查找箭头的DOF索引
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()
        
        # 初始化缓存
        self._init_buffer()
        
        # 起跑线出生范围和最终目标均由 section011 配置明确给出。
        self.spawn_x_range = np.asarray(cfg.spawn_x_range, dtype=np.float32)
        self.spawn_y_range = np.asarray(cfg.spawn_y_range, dtype=np.float32)
        self.spawn_height = float(cfg.init_state.pos[2])
        self.final_target_xy = np.asarray(cfg.target_xy, dtype=np.float32)
        self.initial_yaw_noise = float(cfg.initial_yaw_noise)
        self.terrain_scan_distances = np.asarray(
            cfg.terrain_scan_distances, dtype=np.float32
        )
        self._terrain_hfield = self._model.get_hfield("C_hfield_terrain")
        self._terrain_height_matrix = np.asarray(
            self._terrain_hfield.height_matrix, dtype=np.float32
        )
        self._terrain_hfield_bound = np.asarray(
            self._terrain_hfield.bound, dtype=np.float32
        )

        # 平台到达与停稳判定参数。
        self.platform_y_min = float(cfg.platform_y_min)
        self.platform_x_abs_max = float(cfg.platform_x_abs_max)
        self.platform_base_z_min = float(cfg.platform_base_z_min)
        self.stable_linear_speed_max = float(cfg.stable_linear_speed_max)
        self.stable_angular_speed_max = float(cfg.stable_angular_speed_max)
        self.stable_upright_cos_min = float(cfg.stable_upright_cos_min)
        self.stable_hold_steps_required = max(
            1, int(round(float(cfg.stable_hold_seconds) / cfg.ctrl_dt))
        )

        # 跨 reset 累计的评估统计，避免自动 reset 覆盖回合末 info。
        self.completed_episodes = 0
        self.ever_on_platform_episodes = 0
        self.stable_success_episodes = 0
        self.skill_success_episodes = 0
        self.fall_episodes = 0
        self.timeout_episodes = 0
        self.invalid_state_episodes = 0
        self.episode_steps_sum = 0
        self.episode_max_y_sum = 0.0
        self.episode_forward_progress_sum = 0.0
        self.episode_max_y = float("-inf")
        self.all_time_max_y = float("-inf")
        self.all_time_max_waypoints = 0
        self.all_time_max_stable_hold_steps = 0
        self.all_time_max_skill_goal_hold_steps = 0
        self.skill_goal_stable_candidate_steps = 0
        self.fall_y_bin_edges = np.asarray(
            [-np.inf, -1.7, -1.5, -1.3, -1.1, -0.9, -0.6, 1.2, np.inf],
            dtype=np.float32,
        )
        self.fall_position_y_histogram = np.zeros(
            len(self.fall_y_bin_edges) - 1, dtype=np.int64
        )
        self.fall_episode_max_y_histogram = np.zeros_like(
            self.fall_position_y_histogram
        )
        self.fall_upright_cos_sum = 0.0
        self.fall_angular_xy_sum = 0.0
        self.fall_base_clearance_sum = 0.0

        self.waypoint_y = np.asarray(cfg.waypoint_y, dtype=np.float32)
        configured_route_targets = getattr(cfg, "route_waypoint_targets", None)
        self.route_waypoint_targets = (
            None
            if configured_route_targets is None
            else np.asarray(configured_route_targets, dtype=np.float32)
        )
        if self.route_waypoint_targets is not None:
            expected_shape = (len(self.waypoint_y), 2)
            if self.route_waypoint_targets.shape != expected_shape:
                raise ValueError(
                    "route_waypoint_targets must have shape "
                    f"{expected_shape}, got {self.route_waypoint_targets.shape}"
                )
        self.waypoint_episode_histogram = np.zeros(
            len(self.waypoint_y) + 1, dtype=np.int64
        )
        self.waypoint_crossing_counts = np.zeros(
            len(self.waypoint_y), dtype=np.int64
        )
    
        # 导航统计计数器
        self.navigation_stats_step = 0
    
    def _init_buffer(self):
        """初始化缓存和参数"""
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        
        # 归一化系数
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )
        
        # 设置默认关节角度
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle
        
        self._init_dof_pos[-self._num_action:] = self.default_angles
        self.action_filter_alpha = float(
            getattr(self._cfg, "action_filter_alpha", 0.3)
        )
    
    def _find_target_marker_dof_indices(self):
        """查找target_marker在dof_pos中的索引位置"""
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]
        self._base_quat_start = 6
        self._base_quat_end = 10
    
    def _find_arrow_dof_indices(self):
        """查找箭头在dof_pos中的索引位置"""
        self._robot_arrow_dof_start = 22
        self._robot_arrow_dof_end = 29
        self._desired_arrow_dof_start = 29
        self._desired_arrow_dof_end = 36
        
        arrow_init_height = self._cfg.init_state.pos[2] + 0.5 
        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
    
    def _init_contact_geometry(self):
        """初始化接触检测所需的几何体索引"""
        self._init_termination_contact()
        self._init_foot_contact()
    
    def _init_termination_contact(self):
        """初始化终止接触检测：基座geom与地面geom的碰撞"""
        termination_contact_names = self._cfg.asset.terminate_after_contacts_on
        
        # 获取所有地面geom（遍历所有geom，找到包含ground_subtree名称的）
        ground_geoms = []
        ground_prefix = self._cfg.asset.ground_subtree  # "0ground_root"
        for geom_name in self._model.geom_names:
            if geom_name is not None and ground_prefix in geom_name:
                ground_geoms.append(self._model.get_geom_index(geom_name))
        
        # if len(ground_geoms) == 0:
        #     print(f"[Warning] 未找到以 '{ground_prefix}' 开头的地面geom！")
        #     self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
        #     self.num_termination_check = 0
        #     return
        
        # 构建碰撞对：每个基座geom × 每个地面geom
        termination_contact_list = []
        for base_geom_name in termination_contact_names:
            try:
                base_geom_idx = self._model.get_geom_index(base_geom_name)
                for ground_idx in ground_geoms:
                    termination_contact_list.append([base_geom_idx, ground_idx])
            except Exception as e:
                print(f"[Warning] 无法找到基座geom '{base_geom_name}': {e}")
        
        if len(termination_contact_list) > 0:
            self.termination_contact = np.array(termination_contact_list, dtype=np.uint32)
            self.num_termination_check = len(termination_contact_list)
            print(f"[Info] 初始化终止接触检测: {len(termination_contact_names)}个基座geom × {len(ground_geoms)}个地面geom = {self.num_termination_check}个检测对")
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
            print("[Warning] 未找到任何终止接触geom，基座接触检测将被禁用！")
    
    def _init_foot_contact(self):
        self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
        self._foot_geoms = [
            self._model.get_geom(name) for name in self._cfg.asset.foot_names
        ]
        self.num_foot_check = len(self._foot_geoms)
    
    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)
    
    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)
    
    def _extract_root_state(self, data):
        """从self._body中提取根节点状态"""
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]
        root_linvel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_linvel
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def action_space(self):
        return self._action_space
    
    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        # 保存上一步的关节速度（用于计算加速度）
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        
        state.info["last_actions"] = state.info["current_actions"]
        
        if "filtered_actions" not in state.info:
            state.info["filtered_actions"] = actions
        else:
            state.info["filtered_actions"] = (
                self.action_filter_alpha * actions + 
                (1.0 - self.action_filter_alpha) * state.info["filtered_actions"]
            )
        
        state.info["current_actions"] = state.info["filtered_actions"]

        state.data.actuator_ctrls = self._compute_torques(state.info["filtered_actions"], state.data)
        
        return state
    
    def _compute_torques(self, actions, data):
        """计算PD控制力矩（VBot使用motor执行器，需要力矩控制）"""
        action_scale = float(self._cfg.control_config.action_scale)
        terrain_action_scale = getattr(self._cfg, "terrain_action_scale", None)
        if terrain_action_scale is not None:
            root_y = self._body.get_pose(data)[:, 1]
            blend_start, blend_end = self._cfg.terrain_action_scale_blend_y
            blend = np.clip(
                (root_y - blend_start) / (blend_end - blend_start),
                0.0,
                1.0,
            )
            action_scale = action_scale + blend * (
                float(terrain_action_scale) - action_scale
            )
            action_scale = action_scale[:, None]
        action_scaled = actions * action_scale
        target_pos = self.default_angles + action_scaled
        
        # 获取当前关节状态
        current_pos = self.get_dof_pos(data)  # [num_envs, 12]
        current_vel = self.get_dof_vel(data)  # [num_envs, 12]
        
        # PD控制器：tau = kp * (target - current) - kv * vel
        kp = float(getattr(self._cfg.control_config, "stiffness", 80.0))
        kv = float(getattr(self._cfg.control_config, "damping", 6.0))
        
        pos_error = target_pos - current_pos
        torques = kp * pos_error - kv * current_vel
        
        # 限制力矩范围（与XML中的forcerange一致）
        # hip/thigh: ±17 N·m, calf: ±34 N·m
        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)  # FR, FL, RR, RL
        torques = np.clip(torques, -torque_limits, torque_limits)
        
        return torques
    
    def _compute_projected_gravity(self, root_quat: np.ndarray) -> np.ndarray:
        """计算机器人坐标系中的重力向量"""
        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_vec = np.tile(gravity_vec, (root_quat.shape[0], 1))
        return Quaternion.rotate_inverse(root_quat, gravity_vec)

    def _get_foot_contact_force_observations(
        self, data: mtx.SceneData, root_quat: np.ndarray
    ) -> np.ndarray:
        """Return four 3-D contact sensor vectors in the robot frame."""
        local_forces = []
        for sensor_name in self._cfg.sensor.feet:
            force = np.asarray(
                self._model.get_sensor_value(sensor_name, data), dtype=np.float32
            ).reshape(data.shape[0], -1)
            force_xyz = np.zeros((data.shape[0], 3), dtype=np.float32)
            width = min(force.shape[1], 3)
            force_xyz[:, :width] = force[:, :width]
            local_forces.append(Quaternion.rotate_inverse(root_quat, force_xyz))
        contact_observations = np.concatenate(local_forces, axis=1)
        assert contact_observations.shape == (data.shape[0], 12)
        return contact_observations.astype(np.float32)
    
    def _get_heading_from_quat(self, quat: np.ndarray) -> np.ndarray:
        """从四元数计算yaw角（朝向）"""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        return heading

    def _sample_terrain_height(
        self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Sample the actual hfield plus the analytic ramp/platform surfaces."""
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        height = np.zeros_like(x, dtype=np.float32)

        x_min, y_min, _, x_max, y_max, _ = self._terrain_hfield_bound
        on_hfield = np.logical_and.reduce(
            (x >= x_min, x <= x_max, y >= y_min, y <= y_max)
        )
        if np.any(on_hfield):
            matrix = self._terrain_height_matrix
            column = (x[on_hfield] - x_min) / (x_max - x_min) * (
                matrix.shape[1] - 1
            )
            # Image row zero is the positive-y edge of the heightfield.
            row = (y_max - y[on_hfield]) / (y_max - y_min) * (
                matrix.shape[0] - 1
            )
            c0 = np.floor(column).astype(np.int32)
            r0 = np.floor(row).astype(np.int32)
            c1 = np.minimum(c0 + 1, matrix.shape[1] - 1)
            r1 = np.minimum(r0 + 1, matrix.shape[0] - 1)
            dc = column - c0
            dr = row - r0
            height[on_hfield] = (
                matrix[r0, c0] * (1.0 - dc) * (1.0 - dr)
                + matrix[r0, c1] * dc * (1.0 - dr)
                + matrix[r1, c0] * (1.0 - dc) * dr
                + matrix[r1, c1] * dc * dr
            )

        ramp = np.logical_and(y >= 2.0, y < 6.82963)
        height[ramp] = np.tan(np.deg2rad(15.0)) * (y[ramp] - 2.0)
        height[y >= 6.82963] = 1.294
        return height

    def _get_terrain_scan(
        self, root_pos: np.ndarray, root_quat: np.ndarray
    ) -> np.ndarray:
        heading = self._get_heading_from_quat(root_quat)
        distances = self.terrain_scan_distances[None, :]
        sample_x = root_pos[:, 0:1] + np.cos(heading)[:, None] * distances
        sample_y = root_pos[:, 1:2] + np.sin(heading)[:, None] * distances
        ahead_height = self._sample_terrain_height(sample_x, sample_y)
        current_height = self._sample_terrain_height(
            root_pos[:, 0], root_pos[:, 1]
        )
        delta = ahead_height - current_height[:, None]
        return np.clip(delta * self._cfg.terrain_scan_scale, -1.0, 1.0)

    def _get_policy_frame_motion(
        self,
        root_quat: np.ndarray,
        base_lin_vel: np.ndarray,
        velocity_commands: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return locomotion inputs in the frame expected by the policy."""
        if not getattr(self._cfg, "body_frame_locomotion_observations", False):
            return base_lin_vel, velocity_commands

        policy_lin_vel = Quaternion.rotate_inverse(root_quat, base_lin_vel)
        command_world = np.column_stack(
            [velocity_commands[:, :2], np.zeros(len(velocity_commands))]
        ).astype(np.float32)
        command_local = Quaternion.rotate_inverse(root_quat, command_world)
        policy_commands = np.column_stack(
            [command_local[:, :2], velocity_commands[:, 2]]
        ).astype(np.float32)
        return policy_lin_vel, policy_commands

    def _get_navigation_target(
        self,
        robot_position: np.ndarray,
        final_target: np.ndarray,
        waypoint_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the active 2-D route target, falling back to the final goal."""
        if self.route_waypoint_targets is not None and getattr(
            self._cfg, "route_drives_commands", True
        ):
            if waypoint_indices is None:
                waypoint_indices = np.searchsorted(
                    self.waypoint_y, robot_position[:, 1], side="right"
                )
            return self._get_route_target(final_target, waypoint_indices)

        # Legacy corridor experiments remain available as an ablation.
        corridor_x = getattr(self._cfg, "terrain_corridor_x", None)
        if corridor_x is None:
            return final_target

        navigation_target = final_target.copy()
        before_exit = robot_position[:, 1] < self._cfg.terrain_exit_y
        navigation_target[before_exit, 0] = corridor_x
        navigation_target[before_exit, 1] = self._cfg.terrain_exit_y
        return navigation_target

    def _compute_navigation_commands(
        self,
        robot_position: np.ndarray,
        robot_heading: np.ndarray,
        navigation_target: np.ndarray,
        reached_all: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build world-frame velocity commands for the active route target.

        The default preserves the original component-wise position controller.
        Experiments can instead command body-forward motion while using yaw to
        steer toward the waypoint, matching the route controller described in
        the Section 1 teaching material without changing the observation size.
        """
        navigation_error = navigation_target - robot_position
        desired_heading = np.arctan2(
            navigation_error[:, 1], navigation_error[:, 0]
        )
        heading_to_movement = desired_heading - robot_heading
        heading_to_movement = (heading_to_movement + np.pi) % (2 * np.pi) - np.pi

        body_forward_speed = getattr(
            self._cfg, "navigation_body_forward_speed", None
        )
        if body_forward_speed is None:
            speed_limit = float(
                getattr(self._cfg, "navigation_speed_limit", 1.0)
            )
            desired_vel_xy = np.clip(
                navigation_error, -speed_limit, speed_limit
            )
        else:
            distance = np.linalg.norm(navigation_error, axis=1)
            forward_speed = np.minimum(float(body_forward_speed), distance)
            desired_vel_xy = np.column_stack(
                [
                    np.cos(robot_heading) * forward_speed,
                    np.sin(robot_heading) * forward_speed,
                ]
            )

        desired_yaw_rate = np.clip(heading_to_movement, -1.0, 1.0)
        desired_yaw_rate = np.where(
            np.abs(heading_to_movement) < np.deg2rad(8),
            0.0,
            desired_yaw_rate,
        )
        desired_vel_xy = np.where(
            reached_all[:, np.newaxis], 0.0, desired_vel_xy
        )
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )
        return navigation_error, heading_to_movement, velocity_commands

    def _get_route_target(
        self, final_target: np.ndarray, waypoint_indices: np.ndarray
    ) -> np.ndarray:
        """Map each environment's waypoint state to its configured 2-D target."""
        if self.route_waypoint_targets is None:
            return final_target
        waypoint_indices = np.asarray(waypoint_indices, dtype=np.int32)
        route_target = final_target.copy()
        has_route_target = waypoint_indices < len(self.route_waypoint_targets)
        safe_indices = np.minimum(
            waypoint_indices, len(self.route_waypoint_targets) - 1
        )
        route_target[has_route_target] = self.route_waypoint_targets[
            safe_indices[has_route_target]
        ]
        return route_target

    def _update_success_state(
        self,
        root_pos: np.ndarray,
        base_lin_vel: np.ndarray,
        gyro: np.ndarray,
        projected_gravity: np.ndarray,
        info: dict,
    ) -> None:
        """更新“曾踏上平台”和“在平台停稳”两套成功状态。"""
        on_platform = np.logical_and.reduce(
            (
                root_pos[:, 1] >= self.platform_y_min,
                np.abs(root_pos[:, 0]) <= self.platform_x_abs_max,
                root_pos[:, 2] >= self.platform_base_z_min,
            )
        )
        info["on_platform"] = on_platform
        info["first_on_platform"] = np.logical_and(on_platform, ~info["ever_on_platform"])
        info["ever_on_platform"] = np.logical_or(info["ever_on_platform"], on_platform)

        linear_speed = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        angular_speed = np.linalg.norm(gyro, axis=1)
        upright = -projected_gravity[:, 2] >= self.stable_upright_cos_min
        stable_candidate = np.logical_and.reduce(
            (
                on_platform,
                linear_speed <= self.stable_linear_speed_max,
                angular_speed <= self.stable_angular_speed_max,
                upright,
            )
        )

        hold_steps = np.where(stable_candidate, info["stable_hold_steps"] + 1, 0)
        info["stable_hold_steps"] = hold_steps.astype(np.int32)
        self.all_time_max_stable_hold_steps = max(
            self.all_time_max_stable_hold_steps,
            int(np.max(hold_steps)),
        )
        info["stable_candidate"] = stable_candidate
        stable_now = hold_steps >= self.stable_hold_steps_required
        info["stable_success_this_step"] = np.logical_and(
            stable_now, ~info["stable_success"]
        )
        info["stable_success"] = np.logical_or(info["stable_success"], stable_now)

    def get_success_metrics(self) -> dict[str, float | int | list[int]]:
        """返回跨自动 reset 累积的导航与成功指标。"""
        total = self.completed_episodes
        denominator = max(total, 1)
        return {
            "completed_episodes": total,
            "ever_on_platform_episodes": self.ever_on_platform_episodes,
            "stable_success_episodes": self.stable_success_episodes,
            "ever_on_platform_rate": self.ever_on_platform_episodes / denominator,
            "stable_success_rate": self.stable_success_episodes / denominator,
            "skill_success_episodes": self.skill_success_episodes,
            "skill_success_rate": self.skill_success_episodes / denominator,
            "fall_episodes": self.fall_episodes,
            "fall_rate": self.fall_episodes / denominator,
            "timeout_episodes": self.timeout_episodes,
            "timeout_rate": self.timeout_episodes / denominator,
            "invalid_state_episodes": self.invalid_state_episodes,
            "mean_episode_steps": self.episode_steps_sum / denominator,
            "mean_episode_seconds": (
                self.episode_steps_sum * self._cfg.ctrl_dt / denominator
            ),
            "mean_episode_max_y": self.episode_max_y_sum / denominator,
            "max_episode_y": (
                self.episode_max_y if total else float("nan")
            ),
            "all_time_max_y": self.all_time_max_y,
            "all_time_max_waypoints": self.all_time_max_waypoints,
            "all_time_max_stable_hold_steps": (
                self.all_time_max_stable_hold_steps
            ),
            "all_time_max_skill_goal_hold_steps": (
                self.all_time_max_skill_goal_hold_steps
            ),
            "skill_goal_stable_candidate_steps": (
                self.skill_goal_stable_candidate_steps
            ),
            "mean_forward_progress": self.episode_forward_progress_sum / denominator,
            "waypoint_episode_histogram": self.waypoint_episode_histogram.tolist(),
            "waypoint_crossing_counts": self.waypoint_crossing_counts.tolist(),
            "fall_y_bin_edges": self.fall_y_bin_edges.tolist(),
            "fall_position_y_histogram": self.fall_position_y_histogram.tolist(),
            "fall_episode_max_y_histogram": (
                self.fall_episode_max_y_histogram.tolist()
            ),
            "mean_fall_upright_cos": self.fall_upright_cos_sum
            / max(self.fall_episodes, 1),
            "mean_fall_angular_xy": self.fall_angular_xy_sum
            / max(self.fall_episodes, 1),
            "mean_fall_base_clearance": self.fall_base_clearance_sum
            / max(self.fall_episodes, 1),
        }
    
    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """更新目标位置标记的位置和朝向"""
        num_envs = data.shape[0]
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            target_x = float(pose_commands[env_idx, 0])
            target_y = float(pose_commands[env_idx, 1])
            target_yaw = float(pose_commands[env_idx, 2])
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                target_x, target_y, target_yaw
            ]
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        """更新箭头位置（使用DOF控制freejoint，不影响物理）"""
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return
        
        num_envs = data.shape[0]
        arrow_offset = 0.5  # 箭头相对于机器人的高度偏移
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            # 算箭头高度 = 机器人当前高度 + 偏移
            arrow_height = robot_pos[env_idx, 2] + arrow_offset
            
            # 当前运动方向箭头
            cur_v = base_lin_vel_xy[env_idx]
            if np.linalg.norm(cur_v) > 1e-3:
                cur_yaw = np.arctan2(cur_v[1], cur_v[0])
            else:
                cur_yaw = 0.0
            robot_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
            quat_norm = np.linalg.norm(robot_arrow_quat)
            if quat_norm > 1e-6:
                robot_arrow_quat = robot_arrow_quat / quat_norm
            else:
                robot_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                robot_arrow_pos, robot_arrow_quat
            ])
            
            # 期望运动方向箭头
            des_v = desired_vel_xy[env_idx]
            if np.linalg.norm(des_v) > 1e-3:
                des_yaw = np.arctan2(des_v[1], des_v[0])
            else:
                des_yaw = 0.0
            desired_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
            quat_norm = np.linalg.norm(desired_arrow_quat)
            if quat_norm > 1e-6:
                desired_arrow_quat = desired_arrow_quat / quat_norm
            else:
                desired_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                desired_arrow_pos, desired_arrow_quat
            ])
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _euler_to_quat(self, roll, pitch, yaw):
        """欧拉角转四元数 [qx, qy, qz, qw] - Motrix格式"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw], dtype=np.float32)
    
    def update_state(self, state: NpEnvState) -> NpEnvState:
        """
        更新环境状态，计算观测、奖励和终止条件
        """
        data = state.data
        cfg = self._cfg
        
        # 获取基础状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        state.info["episode_max_y"] = np.maximum(
            state.info["episode_max_y"], root_pos[:, 1]
        )
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        
        # 传感器数据
        base_lin_vel = root_vel[:, :3]  # 世界坐标系线速度
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        foot_contacts = np.column_stack(
            [
                np.linalg.norm(
                    self._model.get_sensor_value(sensor_name, data), axis=1
                )
                > 0.01
                for sensor_name in cfg.sensor.feet
            ]
        )
        previous_contacts = state.info["contacts"]
        first_foot_contact = np.logical_and(foot_contacts, ~previous_contacts)
        feet_air_time = state.info["feet_air_time"] + cfg.ctrl_dt
        state.info["feet_air_time_at_contact"] = np.where(
            first_foot_contact, feet_air_time, 0.0
        ).astype(np.float32)
        state.info["feet_air_time"] = np.where(
            foot_contacts, 0.0, feet_air_time
        ).astype(np.float32)
        state.info["feet_contact_time"] = np.where(
            foot_contacts, state.info["feet_contact_time"] + cfg.ctrl_dt, 0.0
        ).astype(np.float32)
        state.info["contacts"] = foot_contacts
        self._update_success_state(
            root_pos, base_lin_vel, gyro, projected_gravity, state.info
        )
        
        # 导航目标
        pose_commands = state.info["pose_commands"]
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]
        
        # 计算位置误差
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        # 计算朝向误差
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # 达到判定（只看位置，与奖励计算保持一致）
        position_threshold = 0.3
        reached_all = distance_to_target < position_threshold  # 楼梯任务：只要到达位置即可
        
        # 计算期望速度命令（可先经过较平缓的高度场通道）。
        navigation_target = self._get_navigation_target(
            robot_position, target_position, state.info["next_waypoint_idx"]
        )
        (
            navigation_error,
            heading_to_movement,
            velocity_commands,
        ) = self._compute_navigation_commands(
            robot_position, robot_heading, navigation_target, reached_all
        )
        skill_goal_y = getattr(cfg, "skill_goal_y", None)
        if (
            getattr(cfg, "stop_command_at_skill_goal", False)
            and skill_goal_y is not None
        ):
            stop_threshold_y = skill_goal_y - float(
                getattr(cfg, "skill_goal_stop_lead_y", 0.0)
            )
            stop_at_skill_goal = root_pos[:, 1] >= stop_threshold_y
            velocity_commands = np.where(
                stop_at_skill_goal[:, None], 0.0, velocity_commands
            )
        desired_vel_xy = velocity_commands[:, :2]
        
        # 归一化观测
        policy_lin_vel, policy_commands = self._get_policy_frame_motion(
            root_quat, base_lin_vel, velocity_commands
        )
        noisy_linvel = policy_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = policy_commands * self.commands_scale
        last_actions = state.info["current_actions"]
        
        # Route variants expose the active look-ahead target in the existing
        # task slots. The first 48 transferred GO1 inputs remain untouched.
        if getattr(cfg, "observe_route_target", False):
            task_position_error = navigation_error
            task_heading_error = heading_to_movement
            task_distance = np.linalg.norm(navigation_error, axis=1)
        else:
            task_position_error = position_error
            task_heading_error = heading_diff
            task_distance = distance_to_target
        position_error_normalized = task_position_error / 5.0
        heading_error_normalized = task_heading_error / np.pi
        distance_normalized = np.clip(task_distance / 5.0, 0, 1)
        reached_flag = state.info["on_platform"].astype(np.float32)
        
        stop_ready = state.info["stable_candidate"]
        stop_ready_flag = stop_ready.astype(np.float32)
        terrain_scan = self._get_terrain_scan(root_pos, root_quat)
        
        obs = np.concatenate(
            [
                noisy_linvel,       # 3
                noisy_gyro,         # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,    # 12
                last_actions,       # 12
                command_normalized, # 3
                position_error_normalized,  # 2
                heading_error_normalized[:, np.newaxis],  # 1 - 最终朝向误差（保留）
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
                terrain_scan,  # 8 real forward terrain-height differences
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 62)
        if getattr(cfg, "locomotion_contact_force_observations", False):
            obs = np.concatenate(
                [obs[:, :48], self._get_foot_contact_force_observations(data, root_quat)],
                axis=1,
            )
        elif getattr(cfg, "locomotion_observations_only", False):
            obs = obs[:, :48]
        
        # 计算奖励
        state.info["navigation_target"] = navigation_target.astype(np.float32)
        reward = self._compute_reward(data, state.info, velocity_commands)
        
        # 计算终止条件
        terminated_state = self._compute_terminated(state)
        terminated = terminated_state.terminated
        
        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        
        return state
    
    def _compute_terminated(self, state: NpEnvState) -> NpEnvState:
        """
        重写终止条件，与locomotion stairs完全一致
        """
        data = state.data
        
        # 基座接触地面终止（使用传感器）
        try:
            base_contact_value = self._model.get_sensor_value("base_contact", data)
            base_contact = (
                np.linalg.norm(
                    np.asarray(base_contact_value).reshape(self._num_envs, -1),
                    axis=1,
                )
                > 0.01
            )
        except Exception as e:
            print(f"[Warning] 无法读取base_contact传感器: {e}")
            base_contact = np.zeros(self._num_envs, dtype=bool)
        
        stable_success = state.info["stable_success"]
        skill_success = state.info["skill_success"]
        skill_termination = (
            skill_success
            if getattr(self._cfg, "terminate_on_skill_goal", False)
            else np.zeros(self._num_envs, dtype=bool)
        )
        base_quat = data.dof_pos[:, self._base_quat_start:self._base_quat_end]
        quat_norm = np.linalg.norm(base_quat, axis=1)
        invalid_quaternion = np.logical_or(
            ~np.isfinite(base_quat).all(axis=1),
            np.logical_or(quat_norm < 0.5, quat_norm > 1.5),
        )
        terminated = np.logical_or.reduce(
            (base_contact, stable_success, skill_termination, invalid_quaternion)
        )

        max_steps = self._cfg.max_episode_steps
        if max_steps is None:
            timeout = np.zeros(self._num_envs, dtype=bool)
        else:
            timeout = state.info["steps"] + 1 >= max_steps
        episode_done = np.logical_or(terminated, timeout)
        if np.any(episode_done):
            fall_mask = np.logical_and(base_contact, episode_done)
            if np.any(fall_mask):
                root_pos, root_quat, _ = self._extract_root_state(data)
                projected_gravity = self._compute_projected_gravity(root_quat)
                gyro = self._model.get_sensor_value(
                    self._cfg.sensor.base_gyro, data
                )
                terrain_height = self._sample_terrain_height(
                    root_pos[:, 0], root_pos[:, 1]
                )
                base_clearance = root_pos[:, 2] - terrain_height
                fall_y = root_pos[fall_mask, 1]
                fall_episode_max_y = state.info["episode_max_y"][fall_mask]
                self.fall_position_y_histogram += np.histogram(
                    fall_y, bins=self.fall_y_bin_edges
                )[0]
                self.fall_episode_max_y_histogram += np.histogram(
                    fall_episode_max_y, bins=self.fall_y_bin_edges
                )[0]
                self.fall_upright_cos_sum += float(
                    np.sum(-projected_gravity[fall_mask, 2])
                )
                self.fall_angular_xy_sum += float(
                    np.sum(np.linalg.norm(gyro[fall_mask, :2], axis=1))
                )
                self.fall_base_clearance_sum += float(
                    np.sum(base_clearance[fall_mask])
                )
            self.completed_episodes += int(np.sum(episode_done))
            self.ever_on_platform_episodes += int(
                np.sum(state.info["ever_on_platform"][episode_done])
            )
            self.stable_success_episodes += int(
                np.sum(stable_success[episode_done])
            )
            self.skill_success_episodes += int(
                np.sum(skill_success[episode_done])
            )
            self.fall_episodes += int(np.sum(base_contact[episode_done]))
            self.timeout_episodes += int(np.sum(timeout[episode_done]))
            self.invalid_state_episodes += int(
                np.sum(invalid_quaternion[episode_done])
            )
            self.episode_steps_sum += int(
                np.sum(state.info["steps"][episode_done] + 1)
            )

            episode_max_y = state.info["episode_max_y"][episode_done]
            episode_start_y = state.info["episode_start_y"][episode_done]
            self.episode_max_y_sum += float(np.sum(episode_max_y))
            self.episode_forward_progress_sum += float(
                np.sum(episode_max_y - episode_start_y)
            )
            self.episode_max_y = max(
                self.episode_max_y, float(np.max(episode_max_y))
            )
            waypoint_counts = state.info["next_waypoint_idx"][episode_done]
            self.waypoint_episode_histogram += np.bincount(
                waypoint_counts, minlength=len(self.waypoint_episode_histogram)
            )
        
        return state.replace(terminated=terminated)
    
    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands: np.ndarray) -> np.ndarray:
        """Section 1 第一版稠密奖励。"""
        cfg = self._cfg
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)

        lin_error = np.sum(
            np.square(velocity_commands[:, :2] - base_lin_vel[:, :2]), axis=1
        )
        yaw_error = np.square(velocity_commands[:, 2] - gyro[:, 2])
        tracking_linear = np.exp(-lin_error / 0.25)
        tracking_yaw = np.exp(-yaw_error / 0.25)

        target_xy = (
            self._get_route_target(
                info["pose_commands"][:, :2], info["next_waypoint_idx"]
            )
            if getattr(cfg, "progress_uses_route_target", False)
            else info["pose_commands"][:, :2]
        )
        distance = np.linalg.norm(target_xy - root_pos[:, :2], axis=1)
        if getattr(cfg, "progress_uses_route_target", False):
            target_changed = np.any(
                np.abs(target_xy - info["previous_navigation_target"]) > 1e-5,
                axis=1,
            )
            progress = np.where(
                target_changed, 0.0, info["previous_route_distance"] - distance
            )
            info["previous_route_distance"] = distance.astype(np.float32)
            info["previous_navigation_target"] = target_xy.astype(np.float32).copy()
        else:
            progress = info["previous_distance"] - distance
            info["previous_distance"] = distance.astype(np.float32)
        progress = np.clip(progress, -0.2, 0.2)

        target_delta = target_xy - root_pos[:, :2]
        target_distance = np.linalg.norm(target_delta, axis=1)
        target_direction = target_delta / np.maximum(target_distance[:, None], 1e-6)
        target_velocity_cap = float(getattr(cfg, "navigation_speed_limit", 1.0))
        target_direction_velocity = np.clip(
            np.sum(base_lin_vel[:, :2] * target_direction, axis=1),
            -target_velocity_cap,
            target_velocity_cap,
        )

        waypoint_idx = info["next_waypoint_idx"]
        has_waypoint = waypoint_idx < len(self.waypoint_y)
        safe_idx = np.minimum(waypoint_idx, len(self.waypoint_y) - 1)
        reached_waypoint = np.logical_and(
            has_waypoint, root_pos[:, 1] >= self.waypoint_y[safe_idx]
        )
        info["waypoint_reached_this_step"] = reached_waypoint
        next_waypoint_idx = np.minimum(
            waypoint_idx + reached_waypoint.astype(np.int32), len(self.waypoint_y)
        )
        info["next_waypoint_idx"] = next_waypoint_idx
        projected_gravity = self._compute_projected_gravity(root_quat)
        terrain_height = self._sample_terrain_height(
            root_pos[:, 0], root_pos[:, 1]
        )
        base_clearance = root_pos[:, 2] - terrain_height
        skill_goal_y = getattr(cfg, "skill_goal_y", None)
        skill_goal_idx = getattr(cfg, "skill_goal_waypoint_idx", None)
        if skill_goal_y is not None:
            reached_skill_goal = root_pos[:, 1] >= skill_goal_y
        elif skill_goal_idx is None:
            reached_skill_goal = np.zeros(self._num_envs, dtype=bool)
        else:
            reached_skill_goal = np.logical_and(
                waypoint_idx < skill_goal_idx,
                next_waypoint_idx >= skill_goal_idx,
            )

        if getattr(cfg, "skill_goal_require_stability", False):
            upright_cos = -projected_gravity[:, 2]
            angular_xy = np.linalg.norm(gyro[:, :2], axis=1)
            stable_crossing = np.logical_and.reduce(
                (
                    reached_skill_goal,
                    upright_cos >= cfg.skill_goal_upright_cos_min,
                    base_clearance >= cfg.skill_goal_base_clearance_min,
                    angular_xy <= cfg.skill_goal_angular_xy_max,
                )
            )
            info["skill_goal_stable_candidate"] = stable_crossing
            info["skill_goal_hold_steps"] = np.where(
                stable_crossing,
                info["skill_goal_hold_steps"] + 1,
                0,
            )
            self.skill_goal_stable_candidate_steps += int(
                np.count_nonzero(stable_crossing)
            )
            self.all_time_max_skill_goal_hold_steps = max(
                self.all_time_max_skill_goal_hold_steps,
                int(np.max(info["skill_goal_hold_steps"])),
            )
            required_hold_steps = max(
                1, int(round(cfg.skill_goal_hold_seconds / cfg.ctrl_dt))
            )
            reached_skill_goal = (
                info["skill_goal_hold_steps"] >= required_hold_steps
            )
        else:
            info["skill_goal_stable_candidate"] = reached_skill_goal
            info["skill_goal_hold_steps"] = np.where(
                reached_skill_goal, info["skill_goal_hold_steps"] + 1, 0
            )

        skill_success_this_step = np.logical_and(
            ~info["skill_success"], reached_skill_goal
        )
        info["skill_success_this_step"] = skill_success_this_step
        info["skill_success"] = np.logical_or(
            info["skill_success"], skill_success_this_step
        )
        if np.any(reached_waypoint):
            self.waypoint_crossing_counts += np.bincount(
                safe_idx[reached_waypoint], minlength=len(self.waypoint_y)
            )
        self.all_time_max_y = max(
            self.all_time_max_y, float(np.max(root_pos[:, 1]))
        )
        self.all_time_max_waypoints = max(
            self.all_time_max_waypoints,
            int(np.max(info["next_waypoint_idx"])),
        )

        orientation_cost = np.sum(np.square(projected_gravity[:, :2]), axis=1)
        vertical_velocity_cost = np.square(base_lin_vel[:, 2])
        base_height_cost = np.square(
            base_clearance - cfg.target_base_clearance
        )
        progress_for_reward = progress
        target_direction_velocity_for_reward = target_direction_velocity
        gate_progress = getattr(cfg, "gate_progress_by_stability", False)
        gate_target_velocity = getattr(
            cfg, "gate_target_direction_velocity_by_stability", False
        )
        gate_angular_stability = getattr(
            cfg, "gate_motion_by_angular_stability", False
        )
        if gate_progress or gate_target_velocity or gate_angular_stability:
            upright_score = np.clip(
                (-projected_gravity[:, 2] - 0.7) / 0.3, 0.0, 1.0
            )
            clearance_score = np.clip(
                (base_clearance - 0.25) / 0.25, 0.0, 1.0
            )
            safety_score = np.minimum(upright_score, clearance_score)
            if gate_angular_stability:
                angular_xy_for_gate = np.linalg.norm(gyro[:, :2], axis=1)
                angular_full = float(cfg.motion_angular_xy_full_reward)
                angular_zero = float(cfg.motion_angular_xy_zero_reward)
                angular_score = np.clip(
                    (angular_zero - angular_xy_for_gate)
                    / max(angular_zero - angular_full, 1e-6),
                    0.0,
                    1.0,
                )
                safety_score = np.minimum(safety_score, angular_score)
        if gate_progress:
            progress_for_reward = np.where(
                progress > 0.0, progress * safety_score, progress
            )
        if gate_target_velocity:
            target_direction_velocity_for_reward = np.where(
                target_direction_velocity > 0.0,
                target_direction_velocity * safety_score,
                target_direction_velocity,
            )
        angular_xy_cost = np.sum(np.square(gyro[:, :2]), axis=1)
        torque_cost = np.sum(np.square(data.actuator_ctrls), axis=1)
        joint_velocity_cost = np.sum(np.square(self.get_dof_vel(data)), axis=1)
        action_rate_cost = np.sum(
            np.square(info["current_actions"] - info["last_actions"]), axis=1
        )
        feet_air_reward = np.sum(
            np.clip(
                info["feet_air_time_at_contact"] - cfg.minimum_swing_seconds,
                0.0,
                0.5,
            ),
            axis=1,
        )
        feet_overstay_cost = np.sum(
            np.clip((info["feet_contact_time"] - 0.4) / 0.4, 0.0, 1.0),
            axis=1,
        ) * ~info["on_platform"]
        foot_clearance_reward = np.zeros(self._num_envs, dtype=np.float32)
        foot_clearance_scale = float(
            getattr(cfg, "reward_foot_clearance", 0.0)
        )
        if foot_clearance_scale:
            foot_positions = np.stack(
                [geom.get_pose(data)[:, :3] for geom in self._foot_geoms], axis=1
            )
            foot_terrain_height = self._sample_terrain_height(
                foot_positions[:, :, 0], foot_positions[:, :, 1]
            )
            foot_clearance = foot_positions[:, :, 2] - foot_terrain_height
            target_clearance = float(cfg.target_foot_clearance)
            clearance_score = np.exp(
                -np.square((foot_clearance - target_clearance) / 0.06)
            )
            swing_feet = ~info["contacts"]
            zone_y_min, zone_y_max = cfg.foot_clearance_zone_y
            in_clearance_zone = np.logical_and(
                root_pos[:, 1] >= zone_y_min, root_pos[:, 1] <= zone_y_max
            )
            foot_clearance_reward = (
                np.sum(clearance_score * swing_feet, axis=1) * in_clearance_zone
            )
        commanded_speed = np.linalg.norm(velocity_commands[:, :2], axis=1)
        planar_speed = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        platform_stop_y = getattr(cfg, "platform_stop_y", None)
        if platform_stop_y is None:
            platform_stop_zone = np.zeros(self._num_envs, dtype=bool)
        else:
            platform_stop_zone = np.logical_and(
                root_pos[:, 1] >= float(platform_stop_y),
                np.abs(root_pos[:, 0]) <= self.platform_x_abs_max,
            )
        motion_reward_mask = ~platform_stop_zone
        stop_upright_score = np.clip(
            (-projected_gravity[:, 2] - 0.7) / 0.3, 0.0, 1.0
        )
        platform_stop_score = (
            np.exp(-np.square(planar_speed / 0.35))
            * np.exp(-np.square(np.linalg.norm(gyro, axis=1) / 0.75))
            * stop_upright_score
            * platform_stop_zone
        )
        stalled = np.logical_and.reduce(
            (commanded_speed > 0.5, planar_speed < 0.1, ~info["on_platform"])
        )

        try:
            base_contact = self._model.get_sensor_value("base_contact", data)
            base_contact = (
                np.linalg.norm(
                    np.asarray(base_contact).reshape(self._num_envs, -1), axis=1
                )
                > 0.01
            )
        except Exception:
            base_contact = np.zeros(self._num_envs, dtype=bool)

        reward = (
            cfg.reward_tracking_linear * tracking_linear * motion_reward_mask
            + cfg.reward_tracking_yaw * tracking_yaw * motion_reward_mask
            + cfg.reward_target_direction_velocity
            * target_direction_velocity_for_reward
            * motion_reward_mask
            + cfg.reward_skill_goal * skill_success_this_step
            + cfg.reward_skill_stable_step
            * info["skill_goal_stable_candidate"]
            + cfg.reward_progress * progress_for_reward * motion_reward_mask
            + cfg.reward_waypoint * reached_waypoint
            + cfg.reward_first_platform * info["first_on_platform"]
            + cfg.reward_stable_step * info["stable_candidate"]
            + cfg.reward_stable_success * info["stable_success_this_step"]
            + cfg.reward_platform_stop * platform_stop_score
            + cfg.reward_feet_air_time * feet_air_reward * motion_reward_mask
            + foot_clearance_scale * foot_clearance_reward
            - cfg.penalty_orientation * orientation_cost
            - cfg.penalty_vertical_velocity * vertical_velocity_cost
            - cfg.penalty_base_height * base_height_cost
            - cfg.penalty_angular_xy * angular_xy_cost
            - cfg.penalty_torque * torque_cost
            - cfg.penalty_joint_velocity * joint_velocity_cost
            - cfg.penalty_action_rate * action_rate_cost
            - cfg.penalty_stall * stalled
            - cfg.penalty_feet_overstay * feet_overstay_cost
            - cfg.penalty_fall * base_contact
        )
        if getattr(cfg, "clip_reward_nonnegative", False):
            reward = np.maximum(reward, 0.0)
        terminal_fall_penalty = float(
            getattr(cfg, "terminal_fall_penalty", 0.0)
        )
        if terminal_fall_penalty:
            reward = np.where(
                base_contact,
                reward - terminal_fall_penalty,
                reward,
            )
        return reward.astype(np.float32)

    def _sample_spawn_points(self, num_envs: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample official, transition, or local-skill curriculum starts.

        Segment 0 always uses the task's regular spawn range. Segment 1 uses
        the heightfield range used by the existing two-way curricula. An
        optional segment 2 covers the flat transition immediately before the
        heightfield, which lets one policy see the states that connect an
        official approach trajectory to a locally spawned terrain trajectory.
        """
        spawn_x = np.random.uniform(*self.spawn_x_range, size=num_envs).astype(
            np.float32
        )
        spawn_y = np.random.uniform(*self.spawn_y_range, size=num_envs).astype(
            np.float32
        )
        spawn_z = np.full(num_envs, self.spawn_height, dtype=np.float32)

        probabilities = getattr(
            self._cfg, "curriculum_spawn_probabilities", None
        )
        if probabilities is None:
            return np.column_stack((spawn_x, spawn_y)), spawn_z

        if len(probabilities) not in (2, 3):
            raise ValueError(
                "curriculum_spawn_probabilities must contain 2 or 3 segments"
            )

        segment = np.random.choice(
            len(probabilities), size=num_envs, p=np.asarray(probabilities)
        )

        hfield = segment == 1
        hfield_x_range = getattr(
            self._cfg, "curriculum_hfield_x_range", None
        )
        if hfield_x_range is not None:
            spawn_x[hfield] = np.random.uniform(
                *hfield_x_range, size=int(np.sum(hfield))
            )
        spawn_y[hfield] = np.random.uniform(
            *self._cfg.curriculum_hfield_y_range, size=int(np.sum(hfield))
        )
        hfield_spawn_clearance = getattr(
            self._cfg, "curriculum_hfield_spawn_clearance", None
        )
        if hfield_spawn_clearance is None:
            spawn_z[hfield] = self._cfg.curriculum_hfield_spawn_z
        elif np.any(hfield):
            spawn_z[hfield] = self._sample_terrain_height(
                spawn_x[hfield], spawn_y[hfield]
            ) + float(hfield_spawn_clearance)

        if len(probabilities) == 3:
            transition = segment == 2
            transition_count = int(np.sum(transition))
            spawn_x[transition] = np.random.uniform(
                *self._cfg.curriculum_transition_x_range,
                size=transition_count,
            )
            spawn_y[transition] = np.random.uniform(
                *self._cfg.curriculum_transition_y_range,
                size=transition_count,
            )
            spawn_z[transition] = self._cfg.curriculum_transition_spawn_z

        return np.column_stack((spawn_x, spawn_y)), spawn_z

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        num_envs = data.shape[0]

        use_handoff_states = self._handoff_dof_pos is not None
        if use_handoff_states:
            sample_indices = np.random.randint(
                0, self._handoff_dof_pos.shape[0], size=num_envs
            )
            dof_pos = self._handoff_dof_pos[sample_indices].copy()
            dof_vel = self._handoff_dof_vel[sample_indices].copy()
            initial_actions = self._handoff_actions[sample_indices].copy()
            robot_init_xyz = dof_pos[:, 3:6].copy()
            robot_init_xy = robot_init_xyz[:, :2]
        else:
            robot_init_xy, terrain_heights = self._sample_spawn_points(num_envs)
            robot_init_xyz = np.column_stack([robot_init_xy, terrain_heights])
            dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
            dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))
            initial_actions = np.zeros(
                (num_envs, self._num_action), dtype=np.float32
            )

            # 设置 base 的 XYZ位置（DOF 3-5）
            dof_pos[:, 3:6] = robot_init_xyz
        
        # 最终目标固定为平台中心，不再叠加出生点偏移。
        target_positions = np.repeat(self.final_target_xy[None, :], num_envs, axis=0)
        target_yaw = np.arctan2(
            target_positions[:, 1] - robot_init_xy[:, 1],
            target_positions[:, 0] - robot_init_xy[:, 0],
        ).astype(np.float32)
        target_headings = target_yaw[:, None]
        
        pose_commands = np.concatenate([target_positions, target_headings], axis=1)

        # 机器人朝向当前导航阶段目标；通道路线会先对准入口。
        initial_navigation_target = self._get_navigation_target(
            robot_init_xy,
            target_positions,
            np.searchsorted(self.waypoint_y, robot_init_xy[:, 1], side="right"),
        )
        initial_route_target = self._get_route_target(
            target_positions,
            np.searchsorted(self.waypoint_y, robot_init_xy[:, 1], side="right"),
        )
        if self.route_waypoint_targets is not None:
            # Keep the transferred seed-73 reset distribution unchanged.
            # A nearby waypoint exaggerates yaw when spawn x is off-center;
            # route targets take over after the initial pose is established.
            initial_heading_target = target_positions
        else:
            initial_heading_target = initial_navigation_target
        robot_yaw_center = np.arctan2(
            initial_heading_target[:, 1] - robot_init_xy[:, 1],
            initial_heading_target[:, 0] - robot_init_xy[:, 0],
        ).astype(np.float32)
        if use_handoff_states:
            robot_yaw = self._get_heading_from_quat(dof_pos[:, 6:10])
        else:
            robot_yaw = robot_yaw_center + np.random.uniform(
                -self.initial_yaw_noise, self.initial_yaw_noise, size=num_envs
            ).astype(np.float32)
            dof_pos[:, 6:10] = np.column_stack([
                np.zeros(num_envs), np.zeros(num_envs), np.sin(0.5 * robot_yaw), np.cos(0.5 * robot_yaw)
            ]).astype(np.float32)
        
        # 归一化base的四元数（DOF 6-9）
        for env_idx in range(num_envs):
            quat = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-6:
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = quat / quat_norm
            else:
                dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            
            # 归一化箭头的四元数（如果箭头body存在）
            if self._robot_arrow_body is not None:
                robot_arrow_quat = dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end]
                quat_norm = np.linalg.norm(robot_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = robot_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                
                desired_arrow_quat = dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end]
                quat_norm = np.linalg.norm(desired_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = desired_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        
        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)
        
        # 更新目标位置标记
        self._update_target_marker(data, pose_commands)
        
        # 获取根节点状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        
        # 关节状态
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        
        # 传感器数据
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        
        # 计算速度命令
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]
        
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        position_threshold = 0.3
        reached_all = distance_to_target < position_threshold  # 楼梯任务：只看位置
        
        # 计算期望速度，可先经过配置的较平缓高度场通道。
        navigation_target = self._get_navigation_target(
            robot_position, target_position
        )
        (
            navigation_error,
            heading_to_movement,
            velocity_commands,
        ) = self._compute_navigation_commands(
            robot_position, robot_heading, navigation_target, reached_all
        )
        desired_vel_xy = velocity_commands[:, :2]

        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)

        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # 归一化观测
        policy_lin_vel, policy_commands = self._get_policy_frame_motion(
            root_quat, base_lin_vel, velocity_commands
        )
        noisy_linvel = policy_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = policy_commands * self.commands_scale
        last_actions = initial_actions.copy()
        
        # 任务相关观测
        if getattr(self._cfg, "observe_route_target", False):
            task_position_error = navigation_error
            task_heading_error = heading_to_movement
            task_distance = np.linalg.norm(navigation_error, axis=1)
        else:
            task_position_error = position_error
            task_heading_error = heading_diff
            task_distance = distance_to_target
        position_error_normalized = task_position_error / 5.0
        heading_error_normalized = task_heading_error / np.pi
        distance_normalized = np.clip(task_distance / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)
        
        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)
        terrain_scan = self._get_terrain_scan(root_pos, root_quat)

        obs = np.concatenate(
            [
                noisy_linvel,       # 3
                noisy_gyro,         # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,    # 12
                last_actions,       # 12
                command_normalized, # 3
                position_error_normalized,  # 2
                heading_error_normalized[:, np.newaxis],  # 1 - 最终朝向误差（保留）
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
                terrain_scan,  # 8 real forward terrain-height differences
            ],
            axis=-1,
        )
        assert obs.shape == (num_envs, 62)
        if getattr(self._cfg, "locomotion_contact_force_observations", False):
            obs = np.concatenate(
                [obs[:, :48], self._get_foot_contact_force_observations(data, root_quat)],
                axis=1,
            )
        elif getattr(self._cfg, "locomotion_observations_only", False):
            obs = obs[:, :48]
        
        info = {
            "pose_commands": pose_commands,
            "last_actions": initial_actions.copy(),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "current_actions": initial_actions.copy(),
            "filtered_actions": initial_actions.copy(),
            "ever_reached": np.zeros(num_envs, dtype=bool),
            "min_distance": distance_to_target.copy(),  # 统一使用min_distance机制
            "previous_distance": distance_to_target.astype(np.float32),
            "navigation_target": initial_navigation_target.astype(np.float32),
            "previous_navigation_target": initial_route_target.astype(
                np.float32
            ).copy(),
            "previous_route_distance": np.linalg.norm(
                initial_route_target - robot_init_xy, axis=1
            ).astype(np.float32),
            "episode_start_y": root_pos[:, 1].astype(np.float32).copy(),
            "episode_max_y": root_pos[:, 1].astype(np.float32).copy(),
            "next_waypoint_idx": np.searchsorted(
                self.waypoint_y, root_pos[:, 1], side="right"
            ).astype(np.int32),
            "waypoint_reached_this_step": np.zeros(num_envs, dtype=bool),
            "on_platform": np.zeros(num_envs, dtype=bool),
            "first_on_platform": np.zeros(num_envs, dtype=bool),
            "ever_on_platform": np.zeros(num_envs, dtype=bool),
            "stable_candidate": np.zeros(num_envs, dtype=bool),
            "stable_hold_steps": np.zeros(num_envs, dtype=np.int32),
            "stable_success": np.zeros(num_envs, dtype=bool),
            "stable_success_this_step": np.zeros(num_envs, dtype=bool),
            "skill_success": np.zeros(num_envs, dtype=bool),
            "skill_success_this_step": np.zeros(num_envs, dtype=bool),
            "skill_goal_hold_steps": np.zeros(num_envs, dtype=np.int32),
            "skill_goal_stable_candidate": np.zeros(num_envs, dtype=bool),
            "feet_air_time": np.zeros(
                (num_envs, self.num_foot_check), dtype=np.float32
            ),
            "feet_air_time_at_contact": np.zeros(
                (num_envs, self.num_foot_check), dtype=np.float32
            ),
            "feet_contact_time": np.zeros(
                (num_envs, self.num_foot_check), dtype=np.float32
            ),
            # 新增：与locomotion一致的字段
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),  # 上一步关节速度
            "contacts": np.zeros((num_envs, self.num_foot_check), dtype=np.bool_),  # 足部接触状态
        }
        
        return obs, info
    
