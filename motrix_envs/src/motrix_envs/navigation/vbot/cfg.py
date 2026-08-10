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

import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/xmls/scene.xml"

@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1

@dataclass
class ControlConfig:
    # stiffness[N*m/rad] 使用XML中kp参数，仅作记录
    # damping[N*m*s/rad] 使用XML中kv参数，仅作记录
    action_scale = 0.25  # 平地navigation使用0.25
    # torque_limit[N*m] 使用XML forcerange参数

@dataclass
class InitState:
    # the initial position of the robot in the world frame
    pos = [0.0, 0.0, 0.5]  
    
    # 位置随机化范围 [x_min, y_min, x_max, y_max]
    pos_randomization_range = [-10.0, -10.0, 10.0, 10.0]  # 在ground上随机分散20m x 20m范围

    # the default angles for all joints. key = joint name, value = target angle [rad]
    # 使用locomotion的关节角度配置
    default_joint_angles = {
        "FR_hip_joint": -0.0,     # 右前髋关节
        "FR_thigh_joint": 0.9,    # 右前大腿
        "FR_calf_joint": -1.8,    # 右前小腿
        "FL_hip_joint": 0.0,      # 左前髋关节
        "FL_thigh_joint": 0.9,    # 左前大腿
        "FL_calf_joint": -1.8,    # 左前小腿
        "RR_hip_joint": -0.0,     # 右后髋关节
        "RR_thigh_joint": 0.9,    # 右后大腿
        "RR_calf_joint": -1.8,    # 右后小腿
        "RL_hip_joint": 0.0,      # 左后髋关节
        "RL_thigh_joint": 0.9,    # 左后大腿
        "RL_calf_joint": -1.8,    # 左后小腿
    }

@dataclass
class Commands:
    # 目标位置相对于机器人初始位置的偏移范围 [dx_min, dy_min, yaw_min, dx_max, dy_max, yaw_max]
    # dx/dy: 相对机器人初始位置的偏移（米）
    # yaw: 目标绝对朝向（弧度），水平方向随机
    pose_command_range = [-5.0, -5.0, -3.14, 5.0, 5.0, 3.14]

@dataclass
class Normalization:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05

@dataclass
class Asset:
    body_name = "base"
    foot_names = ["FR", "FL", "RR", "RL"]
    terminate_after_contacts_on = ["collision_middle_box", "collision_head_box"]
    ground_subtree = "C_"  # 地形根节点，用于subtree接触检测
   
@dataclass
class Sensor:
    base_linvel = "base_linvel"
    base_gyro = "base_gyro"
    feet = [
        "FR_foot_contact",
        "FL_foot_contact",
        "RR_foot_contact",
        "RL_foot_contact",
    ]

@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            # ===== 导航任务核心奖励 =====
            "position_tracking": 2.0,      # 位置误差奖励（提高10倍）
            "fine_position_tracking": 2.0,  # 精细位置奖励（提高10倍）
            "heading_tracking": 1.0,        # 朝向跟踪奖励（新增）
            "forward_velocity": 0.5,        # 前进速度奖励（鼓励朝目标移动）
            
            # ===== Locomotion稳定性奖励（保持但降低权重） =====
            "orientation": -0.05,           # 姿态稳定（降低权重）
            "lin_vel_z": -0.5,              # 垂直速度惩罚
            "ang_vel_xy": -0.05,            # XY轴角速度惩罚
            "torques": -1e-5,               # 扭矩惩罚
            "dof_vel": -5e-5,               # 关节速度惩罚
            "dof_acc": -2.5e-7,             # 关节加速度惩罚
            "action_rate": -0.01,           # 动作变化率惩罚
            
            # ===== 终止惩罚 =====
            "termination": -200.0,          # 终止惩罚
        }
    )

@registry.envcfg("vbot_navigation_flat")
@dataclass
class VBotEnvCfg(EnvCfg):
    model_file: str = model_file
    reset_noise_scale: float = 0.01
    max_episode_seconds: float = 10
    max_episode_steps: int = 1000
    sim_dt: float = 0.01    # 仿真步长 10ms = 100Hz
    ctrl_dt: float = 0.01
    reset_yaw_scale: float = 0.1
    max_dof_vel: float = 100.0  # 最大关节速度阈值，训练初期给予更大容忍度

    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)


@registry.envcfg("vbot_navigation_stairs")
@dataclass
class VBotStairsEnvCfg(VBotEnvCfg):
    """VBot在楼梯地形上的导航配置，继承flat配置"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs.xml"
    max_episode_seconds: float = 20.0  # 增加到20秒，给更多时间学习转向
    max_episode_steps: int = 2000
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25  # 楼梯navigation使用0.2，足够转向但比平地更谨慎
    
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("VBotStairsMultiTarget-v0")
@dataclass
class VBotStairsMultiTargetEnvCfg(VBotStairsEnvCfg):
    """VBot楼梯多目标导航配置，继承单目标配置"""
    max_episode_seconds: float = 60.0  # 多目标需要更长时间
    max_episode_steps: int = 6000


@registry.envcfg("vbot_navigation_stairs_obstacles")
@dataclass
class VBotStairsObstaclesEnvCfg(VBotStairsEnvCfg):
    """VBot楼梯地形带障碍球的导航配置"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs_obstacles.xml"
    max_episode_seconds: float = 20.0
    max_episode_steps: int = 2000

@registry.envcfg("vbot_navigation_long_course")
@dataclass
class VBotLongCourseEnvCfg(VBotStairsEnvCfg):
    """VBot三段地形完整导航配置（比赛任务）- 使用world.xml统一地图"""
    # 使用scene_world.xml作为完整的三段地形地图（集成了world.xml）
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_world.xml"
    max_episode_seconds: float = 60.0  # 优化：减少到60秒，加快训练速度
    max_episode_steps: int = 6000  # 对应60秒 @ 100Hz
    
    @dataclass
    class InitState:
        # 起始位置：section01的中心位置
        pos = [0.0, 0.0, 1.8]  # 高台中心，高度1.8m
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # 小范围随机±0.5m
        
        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    
    @dataclass
    class Commands:
        # 目标范围：覆盖整个30米路线（section01:0-10m, section02:10-20m, section03:20-30m）
        pose_command_range = [-3.0, 20.0, -3.14, 3.0, 32.0, 3.14]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25  # 与stairs保持一致
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)

@registry.envcfg("vbot_navigation_section001")
#通过 @registry.envcfg("vbot_navigation_section001") 注册
@dataclass
class VBotSection001EnvCfg(VBotStairsEnvCfg):
    """VBot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section001.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [0.0, -2.4, 0.5]  # 中心位置
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # X±0.5m, Y±0.5m随机

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与vbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)

@registry.envcfg("vbot_navigation_section01")
#通过 @registry.envcfg("vbot_navigation_section01") 注册
@dataclass
class VBotSection01EnvCfg(VBotStairsEnvCfg):
    """VBot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section01.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [0.0, -2.4, 0.5]  # 中心位置
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # X±0.5m, Y±0.5m随机

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与vbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)

@registry.envcfg("vbot_navigation_section011")
#通过 @registry.envcfg("vbot_navigation_section011") 注册
@dataclass
class VBotSection011EnvCfg(VBotStairsEnvCfg):
    """VBot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section011.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步

    # 从起跑线全宽随机出生，目标固定为 2026 平台中心。
    spawn_x_range: tuple[float, float] = (-0.5, 0.5)
    spawn_y_range: tuple[float, float] = (-2.9, -2.0)
    target_xy: tuple[float, float] = (0.0, 7.8)
    initial_yaw_noise: float = 0.15

    # 成功口径：曾踏上平台（辅助指标）与在平台连续停稳（主指标）。
    platform_y_min: float = 6.9
    platform_x_abs_max: float = 4.5
    platform_base_z_min: float = 1.55
    stable_linear_speed_max: float = 0.25
    stable_angular_speed_max: float = 0.5
    stable_upright_cos_min: float = 0.9
    stable_hold_seconds: float = 1.0
    reach_threshold: float = 0.45

    # 第二版稠密奖励：增加近场路段门，避免第一道奖励离出生点过远。
    waypoint_y: tuple[float, ...] = (-1.5, -0.6, 1.2, 2.25, 4.0, 6.0, 6.9)
    # Optional two-dimensional look-ahead targets. The base task keeps the
    # original final-target navigation; experiment variants can enable a
    # target for every waypoint gate without changing the observation size.
    route_waypoint_targets: tuple[tuple[float, float], ...] | None = None
    route_drives_commands: bool = True
    observe_route_target: bool = False
    route_drives_commands: bool = False
    progress_uses_route_target: bool = False
    reward_tracking_linear: float = 1.5
    reward_tracking_yaw: float = 0.2
    reward_target_direction_velocity: float = 0.0
    reward_skill_goal: float = 0.0
    skill_goal_waypoint_idx: int | None = None
    skill_goal_y: float | None = None
    terminate_on_skill_goal: bool = False
    navigation_speed_limit: float = 1.0
    clip_reward_nonnegative: bool = False
    reward_progress: float = 20.0
    reward_waypoint: float = 10.0
    reward_first_platform: float = 25.0
    reward_stable_step: float = 0.5
    reward_stable_success: float = 300.0
    reward_feet_air_time: float = 1.0
    minimum_swing_seconds: float = 0.15
    terrain_scan_distances: tuple[float, ...] = (
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
        1.6,
    )
    terrain_scan_scale: float = 2.0
    penalty_orientation: float = 0.5
    penalty_vertical_velocity: float = 2.0
    penalty_base_height: float = 2.0
    target_base_clearance: float = 0.5
    penalty_angular_xy: float = 0.05
    penalty_torque: float = 1e-5
    penalty_joint_velocity: float = 5e-5
    penalty_action_rate: float = 0.005
    penalty_stall: float = 0.3
    penalty_feet_overstay: float = 0.02
    penalty_fall: float = 20.0

    @dataclass
    class InitState:
        # 起跑平地表面 z=0，VBot 基座初始高度约 0.5m。
        pos = [0.0, -2.45, 0.5]
        pos_randomization_range = [-0.5, -0.45, 0.5, 0.45]

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 兼容旧观测代码；reset 中直接使用 target_xy，不再把目标当作出生点偏移。
        pose_command_range = [0.0, 7.8, 1.57079632679, 0.0, 7.8, 1.57079632679]
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_curriculum")
@dataclass
class VBotSection011CurriculumEnvCfg(VBotSection011EnvCfg):
    """Section 1 skill curriculum; evaluation still uses the official start."""

    # Keep training aligned with official evaluation. Hard starts are only at
    # the rough-terrain entrance, never near the ramp exit or final platform.
    curriculum_spawn_probabilities: tuple[float, ...] = (0.65, 0.35)
    curriculum_hfield_y_range: tuple[float, float] = (-1.45, -1.25)
    curriculum_hfield_spawn_z: float = 0.65


@registry.envcfg("vbot_navigation_section011_no_overstay")
@dataclass
class VBotSection011NoOverstayEnvCfg(VBotSection011EnvCfg):
    """Ablation used to test whether contact-duration shaping hurts stability."""

    penalty_feet_overstay: float = 0.0


@registry.envcfg("vbot_navigation_section011_low_action")
@dataclass
class VBotSection011LowActionEnvCfg(VBotSection011EnvCfg):
    """Control-amplitude ablation for stabilizing an aggressive gait."""

    @dataclass
    class ControlConfig:
        action_scale = 0.20

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_safe_progress")
@dataclass
class VBotSection011SafeProgressEnvCfg(VBotSection011EnvCfg):
    """Discourage fast lunges that trade a waypoint for an immediate fall."""

    gate_progress_by_stability: bool = True
    penalty_fall: float = 100.0


@registry.envcfg("vbot_navigation_section011_go1_transfer")
@dataclass
class VBotSection011Go1TransferEnvCfg(VBotSection011SafeProgressEnvCfg):
    """Section 1 controls aligned with the pretrained GO1 locomotion policy."""

    body_frame_locomotion_observations: bool = True
    action_filter_alpha: float = 1.0

    @dataclass
    class ControlConfig:
        action_scale = 0.05
        stiffness = 80.0
        damping = 1.0

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_go1_transfer_fast")
@dataclass
class VBotSection011Go1TransferFastEnvCfg(VBotSection011Go1TransferEnvCfg):
    """More aggressive GO1 transfer control found by the action-scale sweep."""

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.06

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_go1_transfer_medium")
@dataclass
class VBotSection011Go1TransferMediumEnvCfg(VBotSection011Go1TransferEnvCfg):
    """Balanced transfer control selected by the 0.04–0.075 scale sweep."""

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.055

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_go1_transfer_medium_corridor")
@dataclass
class VBotSection011Go1TransferMediumCorridorEnvCfg(
    VBotSection011Go1TransferMediumEnvCfg
):
    """Route through the smoother x=0.70 heightfield corridor."""

    terrain_corridor_x: float = 0.90
    terrain_exit_y: float = 1.70


@registry.envcfg("vbot_navigation_section011_go1_transfer_fast_corridor")
@dataclass
class VBotSection011Go1TransferFastCorridorEnvCfg(
    VBotSection011Go1TransferFastEnvCfg
):
    """Aggressive control routed through the x=0.70 heightfield corridor."""

    terrain_corridor_x: float = 0.90
    terrain_exit_y: float = 1.70


@registry.envcfg("vbot_navigation_section011_go1_transfer_curriculum")
@dataclass
class VBotSection011Go1TransferCurriculumEnvCfg(
    VBotSection011Go1TransferEnvCfg
):
    """GO1 transfer with a minority of starts at the rough-terrain entrance."""

    curriculum_spawn_probabilities: tuple[float, ...] = (0.65, 0.35)
    curriculum_hfield_y_range: tuple[float, float] = (-1.45, -1.25)
    curriculum_hfield_spawn_z: float = 0.65


@registry.envcfg("vbot_navigation_section011_go1_transfer_fast_curriculum")
@dataclass
class VBotSection011Go1TransferFastCurriculumEnvCfg(
    VBotSection011Go1TransferCurriculumEnvCfg
):
    """Rough-terrain curriculum using the faster 0.06 action scale."""

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.06

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_go1_transfer_medium_curriculum")
@dataclass
class VBotSection011Go1TransferMediumCurriculumEnvCfg(
    VBotSection011Go1TransferCurriculumEnvCfg
):
    """Rough-terrain curriculum using the balanced 0.055 action scale."""

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.055

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_go1_transfer_terrain_skill")
@dataclass
class VBotSection011Go1TransferTerrainSkillEnvCfg(
    VBotSection011Go1TransferMediumCurriculumEnvCfg
):
    """Focused rough-terrain stepping skill with explicit swing-foot clearance."""

    curriculum_spawn_probabilities: tuple[float, ...] = (0.2, 0.8)
    reward_foot_clearance: float = 0.5
    target_foot_clearance: float = 0.18
    foot_clearance_zone_y: tuple[float, float] = (-1.6, 1.8)


@registry.envcfg("vbot_navigation_section011_go1_transfer_fast_terrain_skill")
@dataclass
class VBotSection011Go1TransferFastTerrainSkillEnvCfg(
    VBotSection011Go1TransferTerrainSkillEnvCfg
):
    """Focused rough-terrain skill using the aggressive 0.06 action scale."""

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.06

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_go1_transfer_fast_terrain_skill_v2")
@dataclass
class VBotSection011Go1TransferFastTerrainSkillV2EnvCfg(
    VBotSection011Go1TransferFastTerrainSkillEnvCfg
):
    """Reinforce rare rough-terrain gate crossings during conservative fine-tuning."""

    reward_waypoint: float = 50.0


@registry.envcfg("vbot_navigation_section011_go1_transfer_fast_terrain_skill_v3")
@dataclass
class VBotSection011Go1TransferFastTerrainSkillV3EnvCfg(
    VBotSection011Go1TransferFastTerrainSkillEnvCfg
):
    """Look-ahead waypoint shaping inspired by the staged reference solution.

    Each target is deliberately beyond the gate that selects it. This keeps
    forward momentum through the heightfield instead of commanding a stop on
    top of a difficult obstacle.
    """

    route_waypoint_targets: tuple[tuple[float, float], ...] = (
        (0.0, -0.6),
        (0.0, 1.2),
        (0.0, 2.25),
        (0.0, 4.0),
        (0.0, 6.0),
        (0.0, 6.9),
        (0.0, 7.8),
    )
    # Preserve the task-observation values learned by seed 73. The route still
    # drives commands and rewards; later experiments can expose it after the
    # locomotion policy has adapted without sacrificing the rare crossing.
    observe_route_target: bool = False
    progress_uses_route_target: bool = True
    reward_target_direction_velocity: float = 1.0


@registry.envcfg("vbot_navigation_section011_rough_skill_v4")
@dataclass
class VBotSection011RoughSkillV4EnvCfg(
    VBotSection011Go1TransferTerrainSkillEnvCfg
):
    """Specialized heightfield locomotion skill with an explicit exit goal."""

    curriculum_spawn_probabilities: tuple[float, ...] = (0.0, 1.0)
    route_waypoint_targets: tuple[tuple[float, float], ...] = (
        (0.0, -0.6),
        (0.0, 1.2),
        (0.0, 2.25),
        (0.0, 4.0),
        (0.0, 6.0),
        (0.0, 6.9),
        (0.0, 7.8),
    )
    route_drives_commands: bool = True
    observe_route_target: bool = True
    progress_uses_route_target: bool = True
    navigation_speed_limit: float = 0.8

    # Goal-direction velocity replaces fixed velocity tracking. Completing the
    # rough segment (waypoint index 3, immediately after y=1.2) ends the
    # episode with a sparse success reward.
    reward_tracking_linear: float = 0.0
    reward_progress: float = 0.0
    reward_waypoint: float = 5.0
    reward_target_direction_velocity: float = 2.0
    reward_skill_goal: float = 200.0
    skill_goal_waypoint_idx: int = 3
    terminate_on_skill_goal: bool = True
    clip_reward_nonnegative: bool = True


@registry.envcfg("vbot_navigation_section011_rough_skill_v4_safe")
@dataclass
class VBotSection011RoughSkillV4SafeEnvCfg(VBotSection011RoughSkillV4EnvCfg):
    """Lower-amplitude rough skill used as the stability-first branch."""

    navigation_speed_limit: float = 0.7

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.05

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_rough_skill_v5_stage1")
@dataclass
class VBotSection011RoughSkillV5Stage1EnvCfg(
    VBotSection011RoughSkillV4EnvCfg
):
    """First rough-terrain curriculum stage: reach the near-side gate.

    V4 asked a transferred flat-ground gait to traverse the complete
    heightfield before receiving its sparse success reward. This stage moves
    that terminal reward to the first learnable subgoal (``y=-0.6``), while
    retaining dense goal-direction velocity and local progress shaping.
    Successful checkpoints can then warm-start the longer V4 objective.
    """

    route_waypoint_targets: tuple[tuple[float, float], ...] = (
        (0.0, -0.6),
        (0.0, -0.4),
        (0.0, 1.2),
        (0.0, 2.25),
        (0.0, 4.0),
        (0.0, 6.0),
        (0.0, 7.8),
    )
    reward_target_direction_velocity: float = 4.0
    reward_progress: float = 10.0
    reward_skill_goal: float = 300.0
    skill_goal_waypoint_idx: int = 2


@registry.envcfg("vbot_navigation_section011_rough_skill_v5_stage1_safe")
@dataclass
class VBotSection011RoughSkillV5Stage1SafeEnvCfg(
    VBotSection011RoughSkillV5Stage1EnvCfg
):
    """Stability-first control amplitude for the first curriculum stage."""

    navigation_speed_limit: float = 0.7

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.05

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_rough_skill_v6_stage0_scale070")
@dataclass
class VBotSection011RoughSkillV6Stage0Scale070EnvCfg(
    VBotSection011RoughSkillV5Stage1EnvCfg
):
    """Bootstrap rough locomotion on a 0.3--0.45 m reachable subgoal."""

    max_episode_seconds: float = 10.0
    max_episode_steps: int = 1000
    route_waypoint_targets: tuple[tuple[float, float], ...] = (
        (0.0, -1.0),
        (0.0, -0.8),
        (0.0, -0.4),
        (0.0, 1.2),
        (0.0, 2.25),
        (0.0, 4.0),
        (0.0, 7.8),
    )
    reward_progress: float = 20.0
    reward_skill_goal: float = 200.0
    skill_goal_waypoint_idx: int | None = None
    skill_goal_y: float = -1.0

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.07

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_rough_skill_v6_stage0_scale075")
@dataclass
class VBotSection011RoughSkillV6Stage0Scale075EnvCfg(
    VBotSection011RoughSkillV6Stage0Scale070EnvCfg
):
    """Stage-0 curriculum with a 0.075 joint-target action scale."""

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.075

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_rough_skill_v6_stage0_scale080")
@dataclass
class VBotSection011RoughSkillV6Stage0Scale080EnvCfg(
    VBotSection011RoughSkillV6Stage0Scale070EnvCfg
):
    """Stage-0 curriculum with a 0.08 joint-target action scale."""

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.08

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_rough_skill_v6_stage0_scale090")
@dataclass
class VBotSection011RoughSkillV6Stage0Scale090EnvCfg(
    VBotSection011RoughSkillV6Stage0Scale070EnvCfg
):
    """High-exploration stage-0 curriculum with a 0.09 action scale."""

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.09

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_rough_skill_v7_corridor_scale060")
@dataclass
class VBotSection011RoughSkillV7CorridorScale060EnvCfg(
    VBotSection011RoughSkillV6Stage0Scale070EnvCfg
):
    """Bootstrap on the low-cross-slope corridor measured near ``x=0.6``."""

    spawn_x_range: tuple[float, float] = (0.5, 0.7)
    route_waypoint_targets: tuple[tuple[float, float], ...] = (
        (0.6, -1.1),
        (0.6, -0.9),
        (0.6, -0.4),
        (0.6, 1.2),
        (0.6, 2.25),
        (0.0, 4.0),
        (0.0, 7.8),
    )
    skill_goal_y: float = -1.1

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.06

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_rough_skill_v7_corridor_scale070")
@dataclass
class VBotSection011RoughSkillV7CorridorScale070EnvCfg(
    VBotSection011RoughSkillV7CorridorScale060EnvCfg
):
    """Corridor bootstrap with a 0.07 action scale."""

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.07

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_rough_skill_v7_corridor_scale080")
@dataclass
class VBotSection011RoughSkillV7CorridorScale080EnvCfg(
    VBotSection011RoughSkillV7CorridorScale060EnvCfg
):
    """Corridor bootstrap with a 0.08 action scale."""

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.08

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section011_rough_skill_v7_corridor_scale090")
@dataclass
class VBotSection011RoughSkillV7CorridorScale090EnvCfg(
    VBotSection011RoughSkillV7CorridorScale060EnvCfg
):
    """Corridor bootstrap with a 0.09 action scale."""

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.09

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_locomotion_section011_rough_corridor")
@dataclass
class VBotSection011RoughCorridorLocomotionEnvCfg(
    VBotSection011RoughSkillV7CorridorScale060EnvCfg
):
    """Train a VBot gait directly on the measured low-roughness corridor.

    The policy sees only the shared 48-dimensional proprioception/command
    prefix. Navigation and terrain task features are deliberately withheld so
    the resulting gait can be transferred into the full 62-dimensional task
    without coupling locomotion to one waypoint layout.
    """

    locomotion_observations_only: bool = True
    action_filter_alpha: float = 1.0
    navigation_speed_limit: float = 0.5
    max_episode_seconds: float = 10.0
    max_episode_steps: int = 1000

    # Dense locomotion objectives only. Route commands still point through the
    # corridor, but no sparse gate reward can dominate balance learning.
    reward_tracking_linear: float = 2.0
    reward_tracking_yaw: float = 0.5
    reward_target_direction_velocity: float = 1.0
    reward_progress: float = 0.0
    reward_waypoint: float = 0.0
    reward_skill_goal: float = 0.0
    skill_goal_y: float | None = None
    skill_goal_waypoint_idx: int | None = None
    terminate_on_skill_goal: bool = False
    penalty_fall: float = 100.0
    clip_reward_nonnegative: bool = True

    @dataclass
    class ControlConfig(VBotSection011Go1TransferEnvCfg.ControlConfig):
        action_scale = 0.20
        stiffness = 80.0
        damping = 6.0

    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_locomotion_section011_rough_corridor_goal_velocity")
@dataclass
class VBotSection011RoughCorridorGoalVelocityEnvCfg(
    VBotSection011RoughCorridorLocomotionEnvCfg
):
    """Remove the non-zero standing reward from fixed velocity tracking."""

    navigation_speed_limit: float = 0.6
    reward_tracking_linear: float = 0.0
    reward_target_direction_velocity: float = 4.0
    reward_progress: float = 5.0
    reward_feet_air_time: float = 0.2
    penalty_stall: float = 1.0


@registry.envcfg("vbot_locomotion_section011_rough_corridor_contact")
@dataclass
class VBotSection011RoughCorridorContactEnvCfg(
    VBotSection011RoughCorridorGoalVelocityEnvCfg
):
    """Add four robot-frame foot contact vectors to the locomotion policy."""

    locomotion_observations_only: bool = False
    locomotion_contact_force_observations: bool = True


@registry.envcfg("vbot_locomotion_section011_full_route_contact")
@dataclass
class VBotSection011FullRouteContactEnvCfg(
    VBotSection011RoughCorridorContactEnvCfg
):
    """Run the same contact-aware gait from the official start to the platform."""

    spawn_x_range: tuple[float, float] = (-0.5, 0.5)
    spawn_y_range: tuple[float, float] = (-2.9, -2.0)
    curriculum_spawn_probabilities: tuple[float, ...] | None = None
    max_episode_seconds: float = 40.0
    max_episode_steps: int = 4000
    route_waypoint_targets: tuple[tuple[float, float], ...] = (
        (0.6, -1.1),
        (0.6, -0.4),
        (0.6, 1.2),
        (0.6, 2.25),
        (0.0, 4.0),
        (0.0, 6.0),
        (0.0, 7.8),
    )


@registry.envcfg("vbot_navigation_section011_go1_transfer_fast_corridor_skill")
@dataclass
class VBotSection011Go1TransferFastCorridorSkillEnvCfg(
    VBotSection011Go1TransferFastTerrainSkillEnvCfg
):
    """Focused terrain skill routed through the smooth x=0.70 corridor."""

    terrain_corridor_x: float = 0.90
    terrain_exit_y: float = 1.70

@registry.envcfg("vbot_navigation_section012")
#通过 @registry.envcfg("vbot_navigation_section012") 注册
@dataclass
class VBotSection012EnvCfg(VBotStairsEnvCfg):
    """VBot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section012.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [-2.5, 15.0, 3.3]  # 中心位置
        pos_randomization_range = [-0., -0., 0., 0.]  # X±0.5m, Y±0.5m随机

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与vbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)

@registry.envcfg("vbot_navigation_section013")
#通过 @registry.envcfg("vbot_navigation_section013") 注册
@dataclass
class VBotSection013EnvCfg(VBotStairsEnvCfg):
    """VBot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section013.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [0.0, 26.0, 3.3]  # 中心位置
        pos_randomization_range = [-0., -0., 0., 0.]  # X±0.5m, Y±0.5m随机

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与vbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)
