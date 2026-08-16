"""Pure control and route math shared by training and evaluation."""

import numpy as np

from .cfg import FULL_COURSE


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap radians to the half-open interval [-pi, pi)."""
    angle = np.asarray(angle)
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def route_errors(x: np.ndarray, yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return center-line and heading error for a world +Y route."""
    lateral_error = np.asarray(x)
    heading_error = wrap_angle(np.asarray(yaw) - np.pi / 2.0)
    return lateral_error, heading_error


def terrain_blend(y: np.ndarray, start: float, end: float) -> np.ndarray:
    """Smoothstep interpolation between two longitudinal coordinates."""
    if end <= start:
        raise ValueError("end must be greater than start")
    unit = np.clip((np.asarray(y) - start) / (end - start), 0.0, 1.0)
    return unit * unit * (3.0 - 2.0 * unit)


def success_mask(
    x: np.ndarray,
    y: np.ndarray,
    heading_error: np.ndarray,
    upright: np.ndarray,
) -> np.ndarray:
    """Apply the official-document acceptance criteria to vector state."""
    return (
        (np.asarray(y) >= FULL_COURSE.target_y)
        & (np.abs(np.asarray(x)) <= FULL_COURSE.target_x_tolerance)
        & (np.abs(np.asarray(heading_error)) <= FULL_COURSE.heading_tolerance)
        & (np.asarray(upright) >= FULL_COURSE.upright_threshold)
    )


def forward_velocity_progress(world_velocity_y: np.ndarray) -> np.ndarray:
    """Reward progress in m/s along the course's world +Y direction."""
    return np.asarray(world_velocity_y)


def compose_joint_targets(
    reference: np.ndarray,
    route_turn: np.ndarray,
    uphill_bias: np.ndarray,
    actions: np.ndarray,
    residual_scale: float | np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> np.ndarray:
    """Compose reference and PPO residual targets, then enforce joint limits."""
    scale = np.asarray(residual_scale)
    if scale.ndim == 1:
        scale = scale[:, None]
    targets = (
        np.asarray(reference)
        + np.asarray(route_turn)
        + np.asarray(uphill_bias)
        + scale * np.asarray(actions)
    )
    return np.clip(targets, np.asarray(lower_limits), np.asarray(upper_limits))
