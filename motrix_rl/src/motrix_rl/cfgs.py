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

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.skrl.cfg import PPOCfg


class basic:
    @rlcfg("cartpole")
    @dataclass
    class CartPolePPO(PPOCfg):
        max_env_steps: int = 10_000_000
        check_point_interval: int = 500

        # Override PPO configuration
        policy_hidden_layer_sizes: tuple[int, ...] = (32, 32)
        value_hidden_layer_sizes: tuple[int, ...] = (32, 32)
        rollouts: int = 32
        learning_epochs: int = 5
        mini_batches: int = 4

    @rlcfg("bounce_ball")
    @dataclass
    class BounceBallPPO(PPOCfg):
        max_env_steps: int = 50_000_000
        check_point_interval: int = 5000

        # Override PPO configuration for bounce ball task
        policy_hidden_layer_sizes: tuple[int, ...] = (512, 512, 512)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 512, 512)
        rollouts: int = 128
        learning_epochs: int = 15
        mini_batches: int = 16
        learning_rate: float = 2e-4
        num_envs: int = 1024

    @rlcfg("dm-walker", backend="jax")
    @rlcfg("dm-stander", backend="jax")
    @rlcfg("dm-runner", backend="jax")
    @dataclass
    class WalkerPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 4

    @rlcfg("dm-stander", backend="torch")
    @rlcfg("dm-walker", backend="torch")
    @dataclass
    class WalkerPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 32

    @rlcfg("dm-runner", backend="torch")
    @dataclass
    class RunnerPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 2
        mini_batches: int = 32

    @rlcfg("dm-cheetah", backend="jax")
    @dataclass
    class CheetahPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 32
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @rlcfg("dm-cheetah", backend="torch")
    @dataclass
    class CheetahPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 32
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @rlcfg("dm-hopper-stand", backend="jax")
    @rlcfg("dm-hopper-hop", backend="jax")
    @dataclass
    class HopperPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 5
        mini_batches: int = 32
        policy_hidden_layer_sizes: tuple[int, ...] = (32, 32, 32)
        value_hidden_layer_sizes: tuple[int, ...] = (32, 32, 32)

    @rlcfg("dm-hopper-stand", backend="torch")
    @rlcfg("dm-hopper-hop", backend="torch")
    @dataclass
    class HopperPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 5
        mini_batches: int = 32
        policy_hidden_layer_sizes: tuple[int, ...] = (32, 32, 32)
        value_hidden_layer_sizes: tuple[int, ...] = (32, 32, 32)

    @rlcfg("dm-reacher", backend="jax")
    @dataclass
    class ReacherPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 32
        policy_hidden_layer_sizes: tuple[int, ...] = (32, 32, 32)
        value_hidden_layer_sizes: tuple[int, ...] = (32, 32, 32)

    @rlcfg("dm-reacher", backend="torch")
    @dataclass
    class ReacherPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 32
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)


class locomotion:
    @rlcfg("go1-rough-terrain-walk-transfer")
    @rlcfg("go1-flat-terrain-walk")
    @dataclass
    class Go1WalkPPO(PPOCfg):
        """
        Go1 Walk RL config
        """

        seed: int = 42
        share_policy_value_features: bool = False
        max_env_steps: int = 1024 * 60_000
        num_envs: int = 2048

        # Override PPO configuration
        rollouts: int = 24
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        learning_epochs: int = 5
        mini_batches: int = 3
        learning_rate: float = 3e-4

    @rlcfg("go2-flat-terrain-walk")
    @dataclass
    class Go2WalkPPO(PPOCfg):
        """
        Go2 Walk RL config
        """

        seed: int = 42
        share_policy_value_features: bool = False
        max_env_steps: int = 1024 * 60_000
        num_envs: int = 2048

        # Override PPO configuration
        rollouts: int = 24
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        learning_epochs: int = 5
        mini_batches: int = 3
        learning_rate: float = 3e-4

    @rlcfg("go1-rough-terrain-walk")
    @dataclass
    class Go1WalkRoughPPO(Go1WalkPPO):
        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)

    @rlcfg("go1-stairs-terrain-walk")
    @dataclass
    class Go1WalkStairsPPO(Go1WalkRoughPPO): ...


class manipulation:
    @rlcfg("franka-lift-cube")
    @dataclass
    class FrankaLiftPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 4096 * 50000
        check_point_interval: int = 500
        share_policy_value_features: bool = True

        # Override PPO configuration
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        rollouts: int = 24
        learning_epochs: int = 8
        mini_batches: int = 4
        learning_rate: float = 3e-4
        learning_rate_scheduler_kl_threshold: float = 0.01
        entropy_loss_scale: float = 0.001
        rewards_shaper_scale: float = 0.01

    @rlcfg("franka-open-cabinet")
    @dataclass
    class FrankaOpenCabinetPPO(PPOCfg):
        seed: int = 64
        max_env_steps: int = 2048 * 24000
        check_point_interval: int = 500
        share_policy_value_features: bool = False

        # Override PPO configuration
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        rollouts: int = 16
        learning_epochs: int = 5
        mini_batches: int = 32
        learning_rate: float = 3e-4
        rewards_shaper_scale: float = 1e-1
        entropy_loss_scale: float = 0.001


class navigation:
    @rlcfg("anymal_c_navigation_flat")
    @dataclass
    class AnymalCPPOConfig(PPOCfg):
        # ===== Basic Training Parameters =====
        seed: int = 42  # Random seed
        num_envs: int = 2048  # Number of parallel environments during training
        play_num_envs: int = 16  # Number of parallel environments during evaluation
        max_env_steps: int = 100_000_000  # Maximum training steps
        check_point_interval: int = 1000  # Checkpoint save interval (save every 100 iterations)

        # ===== PPO Algorithm Core Parameters =====
        learning_rate: float = 3e-4  # Learning rate
        rollouts: int = 48  # Number of experience replay rollouts
        learning_epochs: int = 6  # Number of training epochs per update
        mini_batches: int = 32  # Number of mini-batches
        discount_factor: float = 0.99  # Discount factor
        lambda_param: float = 0.95  # GAE parameter
        grad_norm_clip: float = 1.0  # Gradient clipping

        # ===== PPO Clipping Parameters =====
        ratio_clip: float = 0.2  # PPO clipping ratio
        value_clip: float = 0.2  # Value clipping
        clip_predicted_values: bool = True  # Clip predicted values

        # Medium-sized network (default configuration, suitable for most tasks)
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @rlcfg("vbot_navigation_flat")
    @dataclass
    class VBotNavigationPPOConfig(PPOCfg):
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 16
        max_env_steps: int = 100_000_000
        check_point_interval: int = 1000

        learning_rate: float = 3e-4
        rollouts: int = 48
        learning_epochs: int = 6
        mini_batches: int = 32
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @rlcfg("vbot_navigation_stairs")
    @dataclass
    class VBotNavigationStairsPPOConfig(PPOCfg):
        """VBot stairs导航配置，与locomotion stairs一致"""
        seed: int = 42
        share_policy_value_features: bool = False
        max_env_steps: int = 1024 * 60_000  # 与locomotion一致
        num_envs: int = 2048

        # 与locomotion stairs一致的PPO配置
        rollouts: int = 24
        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        learning_epochs: int = 5
        mini_batches: int = 3
        learning_rate: float = 3e-4

    @rlcfg("VBotStairsMultiTarget-v0")
    @dataclass
    class VBotNavigationStairsPPOConfig(PPOCfg):
        """VBot stairs导航配置，与locomotion stairs一致"""
        seed: int = 42
        share_policy_value_features: bool = False
        max_env_steps: int = 1024 * 60_000  # 与locomotion一致
        num_envs: int = 2048

        # 与locomotion stairs一致的PPO配置
        rollouts: int = 24
        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        learning_epochs: int = 5
        mini_batches: int = 3
        learning_rate: float = 3e-4

    @rlcfg("vbot_navigation_stairs_obstacles")
    @dataclass
    class VBotNavigationStairsPPOConfig(PPOCfg):
        """VBot stairs导航配置，与locomotion stairs一致"""
        seed: int = 42
        share_policy_value_features: bool = False
        max_env_steps: int = 1024 * 60_000  # 与locomotion一致
        num_envs: int = 2048

        # 与locomotion stairs一致的PPO配置
        rollouts: int = 24
        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        learning_epochs: int = 5
        mini_batches: int = 3
        learning_rate: float = 3e-4

    @rlcfg("vbot_navigation_section011_curriculum")
    @rlcfg("vbot_navigation_section011_no_overstay")
    @rlcfg("vbot_navigation_section011_low_action")
    @rlcfg("vbot_navigation_section011_safe_progress")
    @rlcfg("vbot_locomotion_section011_rough_corridor")
    @rlcfg("vbot_locomotion_section011_rough_corridor_goal_velocity")
    @rlcfg("vbot_locomotion_section011_rough_corridor_contact")
    @rlcfg("vbot_locomotion_section011_rough_entry")
    @rlcfg("vbot_locomotion_section011_rough_entry_stage10")
    @rlcfg("vbot_locomotion_section011_rough_entry_stage11")
    @rlcfg("vbot_locomotion_section011_rough_entry_stage112")
    @rlcfg("vbot_locomotion_section011_rough_entry_stage0")
    @rlcfg("vbot_locomotion_section011_rough_entry_stage05")
    @rlcfg("vbot_locomotion_section011_rough_entry_near_edge")
    @rlcfg("vbot_locomotion_section011_rough_corridor_stage125")
    @rlcfg("vbot_locomotion_section011_rough_corridor_stage15")
    @rlcfg("vbot_locomotion_section011_rough_corridor_stage2")
    @rlcfg("vbot_locomotion_section011_full_route_contact")
    @rlcfg("vbot_locomotion_section011_full_route_scale16to20")
    @rlcfg("vbot_locomotion_section011_full_route_scale17to20")
    @rlcfg("vbot_locomotion_section011_full_route_scale18to20")
    @rlcfg("vbot_locomotion_section011_full_route_scale19to20")
    @rlcfg("vbot_locomotion_section011_full_route_scale17to20_late15")
    @rlcfg("vbot_locomotion_section011_full_route_scale17to20_late14")
    @rlcfg("vbot_locomotion_section011_full_route_scale17to21")
    @rlcfg("vbot_locomotion_section011_full_route_scale17to21_late15")
    @rlcfg("vbot_locomotion_section011_full_route_scale17to1975")
    @rlcfg("vbot_locomotion_section011_full_route_scale17to2025")
    @rlcfg("vbot_locomotion_section011_full_route_scale17to205")
    @rlcfg("vbot_locomotion_section011_full_route_scale17to20_early")
    @rlcfg("vbot_locomotion_section011_full_route_angular_safe")
    @rlcfg("vbot_locomotion_section011_full_route_angular_safe_forward06")
    @rlcfg("vbot_locomotion_section011_approach_stage0")
    @rlcfg("vbot_locomotion_section011_approach")
    @rlcfg("vbot_locomotion_section011_integrated_stage0_90")
    @rlcfg("vbot_locomotion_section011_integrated_stage0_75")
    @rlcfg("vbot_locomotion_section011_integrated_stage0_50")
    @rlcfg("vbot_locomotion_section011_integrated_stage1_70")
    @rlcfg("vbot_locomotion_section011_integrated_stage1_60")
    @rlcfg("vbot_locomotion_section011_integrated_stage1_50")
    @rlcfg("vbot_locomotion_section011_integrated_gate105_70")
    @rlcfg("vbot_locomotion_section011_integrated_gate105_stable_70")
    @rlcfg("vbot_locomotion_section011_integrated_gate100_stable_70")
    @rlcfg("vbot_locomotion_section011_integrated_gate100_stable_scale17to20")
    @rlcfg("vbot_locomotion_section011_integrated_gate095_stable_scale17to20")
    @rlcfg("vbot_locomotion_section011_integrated_gate095_hold03_scale17to20")
    @rlcfg("vbot_locomotion_section011_integrated_gate095_dense_safe_scale17to20")
    @rlcfg("vbot_locomotion_section011_integrated_gate095_dense5_safe_scale17to20")
    @rlcfg("vbot_locomotion_section011_integrated_gate095_balanced_safe_scale17to20")
    @rlcfg("vbot_locomotion_section011_integrated_gate095_hold03_dense_safe_scale17to20")
    @rlcfg("vbot_locomotion_section011_integrated_forward06_no_skill")
    @rlcfg("vbot_locomotion_section011_integrated_angular_forward06_no_skill")
    @rlcfg("vbot_locomotion_section011_rough_only_angular_forward06_no_skill")
    @rlcfg("vbot_locomotion_section011_integrated_gate100_stable_forward06")
    @rlcfg("vbot_locomotion_section011_integrated_gate100_dense_safe_forward06")
    @rlcfg("vbot_locomotion_section011_integrated_gate100_angular_safe_scale17to20")
    @rlcfg("vbot_locomotion_section011_integrated_gate100_angular_strict_scale17to20")
    @rlcfg("vbot_locomotion_section011_integrated_gate100_angular_safe_forward06")
    @rlcfg("vbot_locomotion_section011_integrated_gate100_angular_strict_forward06")
    @rlcfg("vbot_locomotion_section011_integrated_gate095_angular_safe_forward06")
    @rlcfg("vbot_locomotion_section011_integrated_gate095_hold03_angular_safe_forward06")
    @rlcfg("vbot_locomotion_section011_integrated_gate095_dense_angular_safe_forward06")
    @rlcfg("vbot_locomotion_section011_rough080_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough075_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough080_dense_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough080_hold03_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough065_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough060_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough050_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough040_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough030_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough025_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough020_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough020_hold03_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough020_hold03_dense_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough020_hold03_dense5_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough020_hold03_stop_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough020_hold03_stop05_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough020_hold03_stop10_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough020_hold03_stop15_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough020_stop15_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough020_hold03_dense_stop15_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough020_hold03_dense5_stop15_angular_forward06")
    @rlcfg("vbot_locomotion_section011_rough020_hold03_dense_stop_angular_forward06")
    @rlcfg("vbot_locomotion_section011_post_second_000_angular_forward06")
    @rlcfg("vbot_locomotion_section011_post_second_010_angular_forward06")
    @rlcfg("vbot_locomotion_section011_post_second_030_angular_forward06")
    @rlcfg("vbot_locomotion_section011_post_second_050_angular_forward06")
    @rlcfg("vbot_locomotion_section011_post_second_080_angular_forward06")
    @rlcfg("vbot_locomotion_section011_post_second_100_angular_forward06")
    @rlcfg("vbot_locomotion_section011_post_third_225_angular_forward06")
    @rlcfg("vbot_locomotion_section011_ramp_400_angular_forward06")
    @rlcfg("vbot_locomotion_section011_ramp_600_angular_forward06")
    @rlcfg("vbot_locomotion_section011_ramp_top_690_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_780_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_700_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_strict_hold030_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_strict_hold035_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_strict_hold040_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_strict_hold045_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_strict_hold050_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_strict_hold060_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_strict_hold075_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_strict_hold100_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_relaxed_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_relaxed_hold005_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_relaxed_hold010_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_relaxed_hold012_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_relaxed_hold015_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_relaxed_hold018_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_relaxed_hold019_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_relaxed_hold020_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_relaxed_hold025_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_relaxed_hold030_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_relaxed_hold035_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_relaxed_hold040_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_relaxed_hold045_angular_forward06")
    @rlcfg("vbot_locomotion_section011_platform_stand_relaxed_hold050_angular_forward06")
    @rlcfg("vbot_locomotion_section011_mid_bridge_000_angular_forward06")
    @rlcfg("vbot_locomotion_section011_early_bridge_000_angular_forward06")
    @rlcfg("vbot_locomotion_section011_early_bridge_m045_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_m045_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_healthy_m045_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_healthy_m045_bootstrap_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_healthy_test_m045_bootstrap_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_m045_bootstrap_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_m045_bootstrap_test_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_m025_bootstrap_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_m025_bootstrap_test_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_m010_bootstrap_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_m010_bootstrap_test_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_000_bootstrap_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_000_bootstrap_test_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_p010_bootstrap_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_p010_bootstrap_test_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_p030_bootstrap_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_p030_bootstrap_test_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_p050_bootstrap_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_p050_bootstrap_test_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_p080_bootstrap_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_p080_bootstrap_test_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_p100_bootstrap_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_p100_bootstrap_test_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_p100_recoverable_bootstrap_angular_forward06")
    @rlcfg("vbot_locomotion_section011_handoff_seed343_m095_p100_recoverable_bootstrap_test_angular_forward06")
    @rlcfg("vbot_locomotion_section011_integrated_gate100_hold10_70")
    @rlcfg("vbot_locomotion_section011_integrated_gate100_hold10_fall10_70")
    @rlcfg("vbot_locomotion_section011_mixed_route_contact")
    @rlcfg("vbot_navigation_section011")
    @rlcfg("vbot_navigation_section01")
    @dataclass
    class VBotNavigationSection01PPOConfig(PPOCfg):
        """VBot Section01导航配置，与flatnavigation一致"""
        seed: int = 42
        # num_envs: int = 2048
        num_envs: int = 4096
        play_num_envs: int = 1
        max_env_steps: int = 1024 * 60_000
        check_point_interval: int = 1000

        learning_rate: float = 3e-4
        rollouts: int = 48
        learning_epochs: int = 6
        mini_batches: int = 32
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True
        # The framework default (log_std=1, std=2.72) makes the initial joint
        # targets too violent for VBot. std=0.37 still explores while allowing
        # coherent steps to survive long enough to receive navigation rewards.
        initial_log_std: float = -1.0

        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)

    @rlcfg("vbot_navigation_section011_go1_transfer")
    @rlcfg("vbot_navigation_section011_go1_transfer_fast")
    @rlcfg("vbot_navigation_section011_go1_transfer_medium")
    @rlcfg("vbot_navigation_section011_go1_transfer_medium_corridor")
    @rlcfg("vbot_navigation_section011_go1_transfer_fast_corridor")
    @rlcfg("vbot_navigation_section011_go1_transfer_curriculum")
    @rlcfg("vbot_navigation_section011_go1_transfer_fast_curriculum")
    @rlcfg("vbot_navigation_section011_go1_transfer_medium_curriculum")
    @rlcfg("vbot_navigation_section011_go1_transfer_terrain_skill")
    @rlcfg("vbot_navigation_section011_go1_transfer_fast_terrain_skill")
    @rlcfg("vbot_navigation_section011_go1_transfer_fast_terrain_skill_v2")
    @rlcfg("vbot_navigation_section011_go1_transfer_fast_terrain_skill_v3")
    @rlcfg("vbot_navigation_section011_rough_skill_v4")
    @rlcfg("vbot_navigation_section011_rough_skill_v4_safe")
    @rlcfg("vbot_navigation_section011_rough_skill_v5_stage1")
    @rlcfg("vbot_navigation_section011_rough_skill_v5_stage1_safe")
    @rlcfg("vbot_navigation_section011_rough_skill_v6_stage0_scale070")
    @rlcfg("vbot_navigation_section011_rough_skill_v6_stage0_scale075")
    @rlcfg("vbot_navigation_section011_rough_skill_v6_stage0_scale080")
    @rlcfg("vbot_navigation_section011_rough_skill_v6_stage0_scale090")
    @rlcfg("vbot_navigation_section011_rough_skill_v7_corridor_scale060")
    @rlcfg("vbot_navigation_section011_rough_skill_v7_corridor_scale070")
    @rlcfg("vbot_navigation_section011_rough_skill_v7_corridor_scale080")
    @rlcfg("vbot_navigation_section011_rough_skill_v7_corridor_scale090")
    @rlcfg("vbot_navigation_section011_go1_transfer_fast_corridor_skill")
    @dataclass
    class VBotNavigationSection011Go1TransferPPOConfig(
        VBotNavigationSection01PPOConfig
    ):
        """Network shape compatible with the GO1 flat locomotion policy."""

        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @dataclass
    class VBotNavigationSection02PPOConfig(PPOCfg):
        """VBot Section02导航配置，与flatnavigation一致"""
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 16
        max_env_steps: int = 1024 * 60_000
        check_point_interval: int = 1000

        learning_rate: float = 3e-4
        rollouts: int = 48
        learning_epochs: int = 6
        mini_batches: int = 32
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @dataclass
    class VBotNavigationSection03PPOConfig(PPOCfg):
        """VBot Section03导航配置，与flatnavigation一致"""
        seed: int = 42
        num_envs: int = 2048
        play_num_envs: int = 16
        max_env_steps: int = 1024 * 60_000
        check_point_interval: int = 1000

        learning_rate: float = 3e-4
        rollouts: int = 48
        learning_epochs: int = 6
        mini_batches: int = 32
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0

        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True

        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)

    @rlcfg("vbot_navigation_long_course")
    @dataclass
    class VBotNavigationStairsPPOConfig(PPOCfg):
        """VBot stairs导航配置，与locomotion stairs一致"""
        seed: int = 42
        share_policy_value_features: bool = False
        max_env_steps: int = 1024 * 60_000  # 与locomotion一致
        num_envs: int = 2048

        # 与locomotion stairs一致的PPO配置
        rollouts: int = 24
        policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
        learning_epochs: int = 5
        mini_batches: int = 3
        learning_rate: float = 3e-4
