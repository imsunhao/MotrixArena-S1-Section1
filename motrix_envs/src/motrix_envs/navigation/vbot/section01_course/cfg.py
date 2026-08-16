"""Configuration for the VBot Section01 navigation curriculum."""

from dataclasses import dataclass, field
from pathlib import Path

from motrix_envs import registry
from motrix_envs.base import EnvCfg


OBSERVATION_DIM = 78
ACTION_DIM = 12
UPHILL_REFERENCE_V2 = (
    0.0062, 0.0172, -0.0252,
    -0.0064, -0.0174, 0.0,
    0.0065, 0.0, -0.0216,
    0.0047, 0.0, -0.0249,
)


@dataclass(frozen=True)
class Course:
    start_y: float = -2.40
    target_y: float = 7.80
    target_x_tolerance: float = 0.35
    heading_tolerance: float = 0.35
    upright_threshold: float = 0.85
    rough_start_y: float = -1.75
    rough_end_y: float = 1.75
    uphill_blend_start: float = 1.55
    uphill_blend_end: float = 2.15
    uphill_end_y: float = 6.80


FULL_COURSE = Course()


@dataclass
class ControlConfig:
    stiffness: float = 80.0
    damping: float = 1.0
    flat_speed: float = 0.20
    rough_speed: float = 0.14
    uphill_speed: float = 0.14
    flat_frequency: float = 1.25
    rough_frequency: float = 1.20
    uphill_frequency: float = 1.15
    flat_thigh_stride: float = 0.25
    rough_thigh_stride: float = 0.22
    uphill_thigh_stride: float = 0.17
    flat_thigh_lift: float = 0.0
    rough_thigh_lift: float = 0.10
    uphill_thigh_lift: float = 0.24
    flat_calf_lift: float = 0.42
    rough_calf_lift: float = 0.44
    uphill_calf_lift: float = 0.44
    uphill_phase_offset: float = 2.5
    flat_residual_scale: float = 0.10
    rough_residual_scale: float = 0.08
    uphill_residual_scale: float = 0.05
    turn_gain: float = 0.08
    lateral_turn_gain: float = 0.0
    direct_action_scale: float = 0.0
    action_filter_alpha: float = 1.0
    scale_reference_with_command: bool = False
    scale_residual_with_reference: bool = False
    residual_reference_floor: float = 0.0
    reference_fade_start_y: float = 0.0
    reference_fade_end_y: float = 0.0


@dataclass
class RewardConfig:
    route_tracking: float = 4.0
    progress: float = 8.0
    orientation: float = 1.5
    heading_penalty: float = -1.5
    lateral_penalty: float = -2.0
    action_rate: float = -0.02
    action_magnitude: float = -0.005
    stall_penalty: float = -0.5
    success_bonus: float = 100.0
    termination_penalty: float = -10.0
    platform_step: float = 0.0
    stable_step: float = 0.0
    platform_motion_penalty: float = 0.0
    brake_tracking: float = 0.0
    platform_stand_reward: float = 0.0


@dataclass
class VBotSection01Cfg(EnvCfg):
    model_file: str = str(Path(__file__).parent.parent / "xmls" / "scene_section01.xml")
    sim_dt: float = 0.01
    ctrl_dt: float = 0.01
    max_episode_seconds: float = 45.0
    stage: str = "full"
    course_end_y: float = FULL_COURSE.target_y
    enable_height_scan: bool = True
    start_x_range: tuple[float, float] = (0.0, 0.0)
    start_y_range: tuple[float, float] = (FULL_COURSE.start_y, FULL_COURSE.start_y)
    s3a_start_x_range: tuple[float, float] = (0.0, 0.0)
    s3a_start_y_range: tuple[float, float] = (1.0, 1.0)
    initial_yaw_noise: float = 0.0
    use_full_course_local_starts: bool = True
    stable_hold_seconds: float = 0.0
    # The physical platform starts near y=6.83, but the Section1 finish marker
    # and authoritative course target are centered at y=7.80.
    platform_y_min: float = FULL_COURSE.target_y
    platform_x_abs_max: float = 4.5
    platform_base_z_min: float = 1.55
    stable_linear_speed_max: float = 0.25
    stable_vertical_speed_max: float = 0.20
    stable_angular_speed_max: float = 0.5
    brake_start_y: float = 6.9
    brake_min_speed: float = 0.0
    brake_tracking_sigma: float = 0.25
    training_gait_phase_offset_range: tuple[float, float] = (0.0, 0.0)
    training_fixed_x_fraction: float = 0.0
    training_fixed_y_fraction: float = 0.0
    training_narrow_x_fraction: float = 0.0
    training_narrow_x_range: tuple[float, float] = (-0.25, 0.25)
    training_platform_start_fraction: float = 0.0
    training_platform_y_range: tuple[float, float] = (7.8, 8.2)
    training_brake_start_fraction: float = 0.0
    training_brake_y_range: tuple[float, float] = (5.8, 6.5)
    training_brake_speed_range: tuple[float, float] = (0.4, 1.0)
    control: ControlConfig = field(default_factory=ControlConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


@registry.envcfg("vbot-section01-s1-velocity-course")
@dataclass
class VBotSection01S1Cfg(VBotSection01Cfg):
    max_episode_seconds: float = 6.0
    stage: str = "s1"
    course_end_y: float = -1.75
    enable_height_scan: bool = False


@registry.envcfg("vbot-section01-s2-terrain-course")
@dataclass
class VBotSection01S2Cfg(VBotSection01Cfg):
    max_episode_seconds: float = 42.0
    stage: str = "s2"
    course_end_y: float = 1.75
    enable_height_scan: bool = True


@registry.envcfg("vbot-section01-s3a-uphill-course")
@dataclass
class VBotSection01S3aCfg(VBotSection01Cfg):
    max_episode_seconds: float = 55.0
    stage: str = "s3a"
    course_end_y: float = FULL_COURSE.uphill_end_y
    enable_height_scan: bool = True


@registry.envcfg("vbot-section01-full-course-v2-train")
@dataclass
class VBotSection01FullCfg(VBotSection01Cfg):
    max_episode_seconds: float = 90.0
    stage: str = "full"
    course_end_y: float = FULL_COURSE.target_y
    enable_height_scan: bool = True


@registry.envcfg("vbot-section01-xy-s1-course")
@dataclass
class VBotSection01XYS1Cfg(VBotSection01S1Cfg):
    """S1 with only the formal X/Y spawn distribution added."""

    start_x_range: tuple[float, float] = (-0.5, 0.5)
    start_y_range: tuple[float, float] = (-2.9, -2.0)


@registry.envcfg("vbot-section01-xy-s2-course")
@dataclass
class VBotSection01XYS2Cfg(VBotSection01S2Cfg):
    """S2 with the same formal X/Y spawn distribution as evaluation."""

    start_x_range: tuple[float, float] = (-0.5, 0.5)
    start_y_range: tuple[float, float] = (-2.9, -2.0)


@registry.envcfg("vbot-section01-xy-s3a-course")
@dataclass
class VBotSection01XYS3aCfg(VBotSection01S3aCfg):
    """S3a with position jitter on the local ramp-approach reset."""

    s3a_start_x_range: tuple[float, float] = (-0.5, 0.5)
    s3a_start_y_range: tuple[float, float] = (0.9, 1.1)


@registry.envcfg("vbot-section01-xy-full-course")
@dataclass
class VBotSection01XYFullCfg(VBotSection01FullCfg):
    """Full curriculum with random formal X/Y and fixed heading."""

    start_x_range: tuple[float, float] = (-0.5, 0.5)
    start_y_range: tuple[float, float] = (-2.9, -2.0)


@registry.envcfg("vbot-section01-full-random-x-course")
@dataclass
class VBotSection01RandomXCfg(VBotSection01FullCfg):
    start_x_range: tuple[float, float] = (-0.5, 0.5)


@registry.envcfg("vbot-section01-full-random-x10-course")
@dataclass
class VBotSection01RandomX10Cfg(VBotSection01FullCfg):
    start_x_range: tuple[float, float] = (-0.1, 0.1)


@registry.envcfg("vbot-section01-full-random-x25-course")
@dataclass
class VBotSection01RandomX25Cfg(VBotSection01FullCfg):
    start_x_range: tuple[float, float] = (-0.25, 0.25)


@registry.envcfg("vbot-section01-full-random-x10-mix50-course")
@dataclass
class VBotSection01RandomX10Mix50Cfg(VBotSection01RandomX10Cfg):
    training_fixed_x_fraction: float = 0.5


@registry.envcfg("vbot-section01-full-random-x10-mix75-course")
@dataclass
class VBotSection01RandomX10Mix75Cfg(VBotSection01RandomX10Cfg):
    training_fixed_x_fraction: float = 0.75


@dataclass
class VBotSection01DirectCfg(VBotSection01FullCfg):
    """Direct joint-target policy for the random-spawn curriculum."""

    max_episode_seconds: float = 40.0
    use_full_course_local_starts: bool = False
    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            stiffness=80.0,
            damping=6.0,
            flat_speed=0.8,
            rough_speed=0.8,
            uphill_speed=0.8,
            direct_action_scale=0.25,
            action_filter_alpha=0.3,
        )
    )
    reward: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            route_tracking=0.2,
            progress=12.0,
            orientation=1.5,
            heading_penalty=-1.5,
            lateral_penalty=-2.0,
            action_rate=-0.01,
            action_magnitude=0.0,
            stall_penalty=-0.5,
            success_bonus=100.0,
            termination_penalty=-50.0,
        )
    )


@registry.envcfg("vbot-section01-direct-fixed-course")
@dataclass
class VBotSection01DirectFixedCfg(VBotSection01DirectCfg):
    start_x_range: tuple[float, float] = (0.0, 0.0)


@registry.envcfg("vbot-section01-direct-random-x10-course")
@dataclass
class VBotSection01DirectRandomX10Cfg(VBotSection01DirectCfg):
    start_x_range: tuple[float, float] = (-0.1, 0.1)


@registry.envcfg("vbot-section01-direct-random-x10-y10-mix50-course")
@dataclass
class VBotSection01DirectRandomX10Y10Mix50Cfg(VBotSection01DirectRandomX10Cfg):
    start_y_range: tuple[float, float] = (-2.5, -2.3)
    training_fixed_y_fraction: float = 0.5


@registry.envcfg("vbot-section01-direct-random-x10-y10-mix75-course")
@dataclass
class VBotSection01DirectRandomX10Y10Mix75Cfg(VBotSection01DirectRandomX10Cfg):
    start_y_range: tuple[float, float] = (-2.5, -2.3)
    training_fixed_y_fraction: float = 0.75


@registry.envcfg("vbot-section01-direct-random-x10-y25-course")
@dataclass
class VBotSection01DirectRandomX10Y25Cfg(VBotSection01DirectRandomX10Cfg):
    start_y_range: tuple[float, float] = (-2.65, -2.15)


@registry.envcfg("vbot-section01-direct-random-x10-yfull-course")
@dataclass
class VBotSection01DirectRandomX10YFullCfg(VBotSection01DirectRandomX10Cfg):
    start_y_range: tuple[float, float] = (-2.9, -2.0)


@registry.envcfg("vbot-section01-direct-random-x25-course")
@dataclass
class VBotSection01DirectRandomX25Cfg(VBotSection01DirectCfg):
    start_x_range: tuple[float, float] = (-0.25, 0.25)


@registry.envcfg("vbot-section01-direct-random-x25-yfull-course")
@dataclass
class VBotSection01DirectRandomX25YFullCfg(VBotSection01DirectRandomX25Cfg):
    start_y_range: tuple[float, float] = (-2.9, -2.0)


@registry.envcfg("vbot-section01-direct-random-x50-course")
@dataclass
class VBotSection01DirectRandomX50Cfg(VBotSection01DirectCfg):
    start_x_range: tuple[float, float] = (-0.5, 0.5)


@registry.envcfg("vbot-section01-direct-random-xy10-mix50-course")
@dataclass
class VBotSection01DirectRandomXY10Mix50Cfg(VBotSection01DirectRandomX50Cfg):
    start_y_range: tuple[float, float] = (-2.5, -2.3)
    training_fixed_y_fraction: float = 0.5


@registry.envcfg("vbot-section01-direct-random-xy10-mix75-course")
@dataclass
class VBotSection01DirectRandomXY10Mix75Cfg(VBotSection01DirectRandomX50Cfg):
    start_y_range: tuple[float, float] = (-2.5, -2.3)
    training_fixed_y_fraction: float = 0.75


@registry.envcfg("vbot-section01-direct-random-xy-course")
@dataclass
class VBotSection01DirectRandomXYCfg(VBotSection01DirectRandomX50Cfg):
    start_y_range: tuple[float, float] = (-2.9, -2.0)


@registry.envcfg("vbot-section01-direct-random-xy-mix50-course")
@dataclass
class VBotSection01DirectRandomXYMix50Cfg(VBotSection01DirectRandomXYCfg):
    training_fixed_y_fraction: float = 0.5


@registry.envcfg("vbot-section01-direct-random-xy-mix75-course")
@dataclass
class VBotSection01DirectRandomXYMix75Cfg(VBotSection01DirectRandomXYCfg):
    training_fixed_y_fraction: float = 0.75


@registry.envcfg("vbot-section01-direct-random-xy-mix65-course")
@dataclass
class VBotSection01DirectRandomXYMix65Cfg(VBotSection01DirectRandomXYCfg):
    training_fixed_y_fraction: float = 0.65


@registry.envcfg("vbot-section01-direct-random-xy-x25mix50-course")
@dataclass
class VBotSection01DirectRandomXYX25Mix50Cfg(VBotSection01DirectRandomXYCfg):
    training_narrow_x_fraction: float = 0.5


@registry.envcfg("vbot-section01-direct-random-xy-x25mix75-course")
@dataclass
class VBotSection01DirectRandomXYX25Mix75Cfg(VBotSection01DirectRandomXYCfg):
    training_narrow_x_fraction: float = 0.75


@registry.envcfg("vbot-section01-direct-random-xy-yaw-course")
@dataclass
class VBotSection01DirectRandomXYYawCfg(VBotSection01DirectRandomXYCfg):
    initial_yaw_noise: float = 0.15


@registry.envcfg("vbot-section01-direct-random-xy-yaw-neg-x-course")
@dataclass
class VBotSection01DirectRandomXYYawNegativeXCfg(VBotSection01DirectRandomXYYawCfg):
    start_x_range: tuple[float, float] = (-0.5, 0.0)


@registry.envcfg("vbot-section01-direct-random-xy-yaw-pos-x-course")
@dataclass
class VBotSection01DirectRandomXYYawPositiveXCfg(VBotSection01DirectRandomXYYawCfg):
    start_x_range: tuple[float, float] = (0.0, 0.5)


@registry.envcfg("vbot-section01-direct-random-xy-yaw-stable-course")
@dataclass
class VBotSection01DirectRandomXYYawStableCfg(VBotSection01DirectRandomXYYawCfg):
    stable_hold_seconds: float = 1.0
    reward: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            route_tracking=0.2,
            progress=12.0,
            orientation=1.5,
            heading_penalty=-1.5,
            lateral_penalty=-2.0,
            action_rate=-0.01,
            action_magnitude=0.0,
            stall_penalty=-0.5,
            success_bonus=100.0,
            termination_penalty=-50.0,
            platform_step=2.0,
            stable_step=4.0,
        )
    )


@registry.envcfg("vbot-section01-direct-random-xy-yaw-stable-neg-x-course")
@dataclass
class VBotSection01DirectRandomXYYawStableNegativeXCfg(
    VBotSection01DirectRandomXYYawStableCfg
):
    start_x_range: tuple[float, float] = (-0.5, 0.0)


@registry.envcfg("vbot-section01-direct-random-xy-yaw-stable-pos-x-course")
@dataclass
class VBotSection01DirectRandomXYYawStablePositiveXCfg(
    VBotSection01DirectRandomXYYawStableCfg
):
    start_x_range: tuple[float, float] = (0.0, 0.5)


@registry.envcfg("vbot-section01-direct-random-xy-yaw-stable-v2-course")
@dataclass
class VBotSection01DirectRandomXYYawStableV2Cfg(
    VBotSection01DirectRandomXYYawStableCfg
):
    reward: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            route_tracking=0.2,
            progress=12.0,
            orientation=1.5,
            heading_penalty=-1.5,
            lateral_penalty=-2.0,
            action_rate=-0.01,
            action_magnitude=0.0,
            stall_penalty=-0.5,
            success_bonus=100.0,
            termination_penalty=-50.0,
            platform_step=2.0,
            stable_step=8.0,
            platform_motion_penalty=-8.0,
        )
    )


@registry.envcfg("vbot-section01-direct-random-xy-yaw-stable-v2-neg-x-course")
@dataclass
class VBotSection01DirectRandomXYYawStableV2NegativeXCfg(
    VBotSection01DirectRandomXYYawStableV2Cfg
):
    start_x_range: tuple[float, float] = (-0.5, 0.0)


@registry.envcfg("vbot-section01-direct-random-xy-yaw-stable-v2-pos-x-course")
@dataclass
class VBotSection01DirectRandomXYYawStableV2PositiveXCfg(
    VBotSection01DirectRandomXYYawStableV2Cfg
):
    start_x_range: tuple[float, float] = (0.0, 0.5)


@registry.envcfg("vbot-section01-direct-random-xy-yaw-stable-v3-course")
@dataclass
class VBotSection01DirectRandomXYYawStableV3Cfg(
    VBotSection01DirectRandomXYYawStableV2Cfg
):
    training_platform_start_fraction: float = 0.25


@registry.envcfg("vbot-section01-direct-random-xy-yaw-stable-v3-neg-x-course")
@dataclass
class VBotSection01DirectRandomXYYawStableV3NegativeXCfg(
    VBotSection01DirectRandomXYYawStableV3Cfg
):
    start_x_range: tuple[float, float] = (-0.5, 0.0)


@registry.envcfg("vbot-section01-direct-random-xy-yaw-stable-v3-pos-x-course")
@dataclass
class VBotSection01DirectRandomXYYawStableV3PositiveXCfg(
    VBotSection01DirectRandomXYYawStableV3Cfg
):
    start_x_range: tuple[float, float] = (0.0, 0.5)


@registry.envcfg("vbot-section01-direct-random-xy-yaw-stable-v4-course")
@dataclass
class VBotSection01DirectRandomXYYawStableV4Cfg(
    VBotSection01DirectRandomXYYawStableV3Cfg
):
    brake_start_y: float = 5.8
    brake_min_speed: float = 0.05
    training_brake_start_fraction: float = 0.25
    reward: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            route_tracking=0.2,
            progress=12.0,
            orientation=1.5,
            heading_penalty=-1.5,
            lateral_penalty=-2.0,
            action_rate=-0.01,
            action_magnitude=0.0,
            stall_penalty=-0.5,
            success_bonus=100.0,
            termination_penalty=-50.0,
            platform_step=2.0,
            stable_step=8.0,
            platform_motion_penalty=-5.0,
            brake_tracking=6.0,
            platform_stand_reward=8.0,
        )
    )


@registry.envcfg("vbot-section01-direct-random-xy-yaw-stable-v4-neg-x-course")
@dataclass
class VBotSection01DirectRandomXYYawStableV4NegativeXCfg(
    VBotSection01DirectRandomXYYawStableV4Cfg
):
    start_x_range: tuple[float, float] = (-0.5, 0.0)


@registry.envcfg("vbot-section01-direct-random-xy-yaw-stable-v4-pos-x-course")
@dataclass
class VBotSection01DirectRandomXYYawStableV4PositiveXCfg(
    VBotSection01DirectRandomXYYawStableV4Cfg
):
    start_x_range: tuple[float, float] = (0.0, 0.5)


@registry.envcfg("vbot-section01-full-random-xy-course")
@dataclass
class VBotSection01RandomXYCfg(VBotSection01RandomXCfg):
    start_y_range: tuple[float, float] = (-2.9, -2.0)


@registry.envcfg("vbot-section01-full-random-xy-yaw-course")
@dataclass
class VBotSection01RandomXYYawCfg(VBotSection01RandomXYCfg):
    initial_yaw_noise: float = 0.15


@registry.envcfg("vbot-section01-full-random-xy-yaw-stable-course")
@dataclass
class VBotSection01RandomXYYawStableCfg(VBotSection01RandomXYYawCfg):
    stable_hold_seconds: float = 1.0
    reward: RewardConfig = field(
        default_factory=lambda: RewardConfig(platform_step=2.0, stable_step=4.0)
    )


@registry.envcfg("vbot-section01-xy-yaw-stable-v2-course")
@dataclass
class VBotSection01XYYawStableV2Cfg(VBotSection01RandomXYYawStableCfg):
    """Strict one-second stop with a platform braking curriculum."""

    brake_start_y: float = 5.8
    brake_min_speed: float = 0.05
    training_platform_start_fraction: float = 0.25
    training_brake_start_fraction: float = 0.25
    reward: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            platform_step=2.0,
            stable_step=8.0,
            platform_motion_penalty=-5.0,
            brake_tracking=6.0,
            platform_stand_reward=8.0,
        )
    )


@registry.envcfg("vbot-section01-xy-yaw-stable-v3-course")
@dataclass
class VBotSection01XYYawStableV3Cfg(VBotSection01XYYawStableV2Cfg):
    """Fade the nominal gait into the standing pose as the command reaches zero."""

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(scale_reference_with_command=True)
    )


@registry.envcfg("vbot-section01-xy-yaw-stable-v4-course")
@dataclass
class VBotSection01XYYawStableV4Cfg(VBotSection01XYYawStableV2Cfg):
    """Enter the platform with the proven gait, then fade to a standing pose."""

    brake_start_y: float = 7.65
    training_brake_y_range: tuple[float, float] = (7.3, 7.65)
    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            reference_fade_start_y=7.65,
            reference_fade_end_y=FULL_COURSE.target_y,
        )
    )


@registry.envcfg("vbot-section01-xy-yaw-stable-v5-course")
@dataclass
class VBotSection01XYYawStableV5Cfg(VBotSection01XYYawStableV4Cfg):
    """Fade the locomotion residual after the robot enters the platform."""

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            scale_residual_with_reference=True,
            reference_fade_start_y=7.65,
            reference_fade_end_y=FULL_COURSE.target_y,
        )
    )


@registry.envcfg("vbot-section01-xy-yaw-stable-v6-course")
@dataclass
class VBotSection01XYYawStableV6Cfg(VBotSection01XYYawStableV4Cfg):
    """Retain a small residual authority for balance after the gait fades."""

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            scale_residual_with_reference=True,
            residual_reference_floor=0.2,
            reference_fade_start_y=7.65,
            reference_fade_end_y=FULL_COURSE.target_y,
        )
    )


@registry.envcfg("vbot-section01-full-random-x-route-v2-course")
@dataclass
class VBotSection01RandomXRouteV2Cfg(VBotSection01RandomXCfg):
    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(lateral_turn_gain=0.06)
    )


@registry.envcfg("vbot-section01-full-random-xy-route-v2-course")
@dataclass
class VBotSection01RandomXYRouteV2Cfg(VBotSection01RandomXRouteV2Cfg):
    start_y_range: tuple[float, float] = (-2.9, -2.0)


@registry.envcfg("vbot-section01-full-random-xy-yaw-route-v2-course")
@dataclass
class VBotSection01RandomXYYawRouteV2Cfg(VBotSection01RandomXYRouteV2Cfg):
    initial_yaw_noise: float = 0.15


@registry.envcfg("vbot-section01-full-random-xy-yaw-stable-route-v2-course")
@dataclass
class VBotSection01RandomXYYawStableRouteV2Cfg(VBotSection01RandomXYYawRouteV2Cfg):
    stable_hold_seconds: float = 1.0
    reward: RewardConfig = field(
        default_factory=lambda: RewardConfig(platform_step=2.0, stable_step=4.0)
    )


@registry.envcfg("vbot-section01-full-random-x-phase-v3-course")
@dataclass
class VBotSection01RandomXPhaseV3Cfg(VBotSection01RandomXRouteV2Cfg):
    # One complete four-beat gait cycle spans [0, 4). The jitter is enabled
    # only for vectorized training; single-environment evaluation uses zero.
    training_gait_phase_offset_range: tuple[float, float] = (-2.0, 2.0)


@registry.envcfg("vbot-section01-full-random-x-phase-small-v3-course")
@dataclass
class VBotSection01RandomXPhaseSmallV3Cfg(VBotSection01RandomXRouteV2Cfg):
    training_gait_phase_offset_range: tuple[float, float] = (-0.5, 0.5)


@registry.envcfg("vbot-section01-full-random-x-phase-medium-v3-course")
@dataclass
class VBotSection01RandomXPhaseMediumV3Cfg(VBotSection01RandomXRouteV2Cfg):
    training_gait_phase_offset_range: tuple[float, float] = (-1.0, 1.0)
