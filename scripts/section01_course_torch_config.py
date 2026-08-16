"""Shared PPO configuration for the peer-structure Section01 reproduction."""

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.skrl.cfg import PPOCfg


COURSE_ENVS = (
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
)

PEER_CURRICULUM_STAGE_STEPS = {
    "vbot-section01-s1-velocity-course": 9600,
    "vbot-section01-s2-terrain-course": 16000,
    "vbot-section01-s3a-uphill-course": 16000,
    "vbot-section01-full-course-v2-train": 6400,
    "vbot-section01-peer-xy-s1-course": 9600,
    "vbot-section01-peer-xy-s2-course": 16000,
    "vbot-section01-peer-xy-s3a-course": 16000,
    "vbot-section01-peer-xy-full-course": 6400,
    "vbot-section01-peer-xy-yaw-stable-v2-course": 6400,
    "vbot-section01-peer-xy-yaw-stable-v3-course": 6400,
    "vbot-section01-peer-xy-yaw-stable-v4-course": 6400,
    "vbot-section01-peer-xy-yaw-stable-v4-50-course": 6400,
    "vbot-section01-peer-xy-yaw-stable-v5-course": 6400,
    "vbot-section01-peer-xy-yaw-stable-v6-course": 6400,
}

DIRECT_ENVS = {
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
}


for _course_env in COURSE_ENVS:

    @rlcfg(_course_env, backend="torch")
    @dataclass
    class Section01CourseTorchPPO(PPOCfg):
        seed: int = 42
        num_envs: int = 1024
        play_num_envs: int = 1
        max_env_steps: int = 1024 * 9600
        check_point_interval: int = 400

        policy_hidden_layer_sizes: tuple[int, ...] = (
            (256, 128, 64) if _course_env in DIRECT_ENVS else (512, 256, 128)
        )
        value_hidden_layer_sizes: tuple[int, ...] = (
            (256, 128, 64) if _course_env in DIRECT_ENVS else (512, 256, 128)
        )
        share_policy_value_features: bool = _course_env not in DIRECT_ENVS
        learning_rate: float = 3e-4
        rollouts: int = 24
        learning_epochs: int = 5
        mini_batches: int = 8
        entropy_loss_scale: float = 0.001 if _course_env in DIRECT_ENVS else 0.0
        discount_factor: float = 0.99
        lambda_param: float = 0.95
        grad_norm_clip: float = 1.0
        ratio_clip: float = 0.2
        value_clip: float = 0.2
        clip_predicted_values: bool = True
