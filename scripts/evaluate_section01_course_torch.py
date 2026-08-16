"""Deterministically evaluate a self-trained Section01 Torch checkpoint."""

from __future__ import annotations

import json

from absl import app, flags
import numpy as np
from skrl import config
from skrl.utils import set_seed
import torch

import motrix_envs.navigation.vbot  # noqa: F401 - register course environments
from motrix_envs import registry as env_registry
from motrix_rl.skrl.torch import wrap_env
from motrix_rl.skrl.torch.train import ppo

from section01_course_torch_config import COURSE_ENVS  # noqa: F401


_ENV = flags.DEFINE_enum("env", COURSE_ENVS[0], COURSE_ENVS, "Curriculum stage")
_POLICY = flags.DEFINE_string("policy", None, "Self-trained Torch checkpoint")
_EVAL_START_X_MIN = flags.DEFINE_float(
    "eval-start-x-min", None, "Optional lower bound for the evaluation start X"
)
_EVAL_START_X_MAX = flags.DEFINE_float(
    "eval-start-x-max", None, "Optional upper bound for the evaluation start X"
)
_NUM_ENVS = flags.DEFINE_integer("num-envs", 32, "Parallel evaluation environments")
_EPISODES = flags.DEFINE_integer("episodes", 32, "Completed episodes to collect")
_MAX_STEPS = flags.DEFINE_integer("max-steps", 10000, "Control-step safety limit")
_SEED = flags.DEFINE_integer("seed", 2026, "Evaluation seed")
_GAIT_PHASE_OFFSET = flags.DEFINE_float(
    "gait-phase-offset", None, "Optional diagnostic reference-gait phase offset"
)
_DIAGNOSTIC_WAYPOINTS = (-1.75, 1.0, 1.55, 1.75, 2.15, 6.8, 7.8)


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1)


def main(argv):
    del argv
    if not _POLICY.value:
        raise app.UsageError("--policy is required")
    if _NUM_ENVS.value <= 0 or _EPISODES.value <= 0:
        raise app.UsageError("--num-envs and --episodes must be positive")
    if (_EVAL_START_X_MIN.value is None) != (_EVAL_START_X_MAX.value is None):
        raise app.UsageError(
            "--eval-start-x-min and --eval-start-x-max must be used together"
        )
    if (
        _EVAL_START_X_MIN.value is not None
        and _EVAL_START_X_MIN.value > _EVAL_START_X_MAX.value
    ):
        raise app.UsageError("evaluation start X minimum must not exceed maximum")

    config.torch.backend = "torch"
    set_seed(_SEED.value)
    trainer = ppo.Trainer(
        _ENV.value,
        "np",
        cfg_override={"play_num_envs": _NUM_ENVS.value, "seed": _SEED.value},
        enable_render=False,
    )
    raw_env = env_registry.make(_ENV.value, sim_backend="np", num_envs=_NUM_ENVS.value)
    # Full-course training deliberately mixes in two local curriculum starts.
    # Evaluation must always sample only the configured formal start range.
    if hasattr(raw_env.cfg, "use_full_course_local_starts"):
        raw_env.cfg.use_full_course_local_starts = False
    if hasattr(raw_env.cfg, "training_platform_start_fraction"):
        raw_env.cfg.training_platform_start_fraction = 0.0
    if hasattr(raw_env.cfg, "training_brake_start_fraction"):
        raw_env.cfg.training_brake_start_fraction = 0.0
    if _EVAL_START_X_MIN.value is not None:
        raw_env.cfg.start_x_range = (
            _EVAL_START_X_MIN.value,
            _EVAL_START_X_MAX.value,
        )
    env = wrap_env(raw_env, enable_render=False)
    models = trainer._make_model(env, trainer._rlcfg)
    agent = trainer._make_agent(models, env, ppo._get_cfg(trainer._rlcfg, env))
    agent.load(_POLICY.value)
    agent.set_running_mode("eval")

    obs, _ = env.reset()
    if _GAIT_PHASE_OFFSET.value is not None:
        raw_env.state.info["gait_phase_offset"].fill(_GAIT_PHASE_OFFSET.value)
    initial_pose = raw_env._body.get_pose(raw_env.state.data)
    episode_start_x = initial_pose[:, 0].copy()
    episode_start_y = initial_pose[:, 1].copy()
    completed_start_x: list[float] = []
    completed_start_y: list[float] = []
    episode_max_y = np.full(_NUM_ENVS.value, -np.inf, dtype=np.float32)
    episode_max_abs_x = np.zeros(_NUM_ENVS.value, dtype=np.float32)
    crossing_steps = np.full(
        (_NUM_ENVS.value, len(_DIAGNOSTIC_WAYPOINTS)), -1, dtype=np.int32
    )
    completed_max_y: list[float] = []
    completed_max_stable_steps: list[int] = []
    completed = successes = falls = timeouts = 0
    ever_on_platform = stable_successes = 0
    episode_details: list[dict] = []
    quota, remainder = divmod(_EPISODES.value, _NUM_ENVS.value)
    target_episodes_per_env = np.full(_NUM_ENVS.value, quota, dtype=np.int32)
    target_episodes_per_env[:remainder] += 1
    completed_episodes_per_env = np.zeros(_NUM_ENVS.value, dtype=np.int32)

    with torch.no_grad():
        for control_step in range(1, _MAX_STEPS.value + 1):
            pose = raw_env._body.get_pose(raw_env.state.data)
            episode_max_y = np.maximum(episode_max_y, pose[:, 1])
            episode_max_abs_x = np.maximum(episode_max_abs_x, np.abs(pose[:, 0]))
            for waypoint_index, waypoint_y in enumerate(_DIAGNOSTIC_WAYPOINTS):
                crossed = (crossing_steps[:, waypoint_index] < 0) & (pose[:, 1] >= waypoint_y)
                crossing_steps[crossed, waypoint_index] = control_step
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, info = env.step(actions)
            done = _numpy(torch.logical_or(terminated, truncated)).astype(bool)
            if not np.any(done):
                continue
            done_indices = np.flatnonzero(done)
            indices = done_indices[
                completed_episodes_per_env[done_indices]
                < target_episodes_per_env[done_indices]
            ]
            if not len(indices):
                episode_max_y[done_indices] = -np.inf
                episode_max_abs_x[done_indices] = 0.0
                crossing_steps[done_indices] = -1
                reset_pose = raw_env._body.get_pose(raw_env.state.data)
                episode_start_x[done_indices] = reset_pose[done_indices, 0]
                episode_start_y[done_indices] = reset_pose[done_indices, 1]
                continue
            completed_start_x.extend(episode_start_x[indices].astype(float).tolist())
            completed_start_y.extend(episode_start_y[indices].astype(float).tolist())
            final_x = _numpy(info.get("final_x"))[indices]
            final_y = _numpy(info.get("final_y"))[indices]
            final_heading = _numpy(info.get("final_heading"))[indices]
            final_upright = _numpy(info.get("final_upright"))[indices]
            termination_reason = np.asarray(info.get("termination_reason")).reshape(-1)[indices]
            # The environment resets completed rows inside ``env.step``. Account
            # for a waypoint reached on that terminal transition using final_y.
            for waypoint_index, waypoint_y in enumerate(_DIAGNOSTIC_WAYPOINTS):
                crossed_on_terminal_step = (
                    (crossing_steps[indices, waypoint_index] < 0)
                    & (final_y >= waypoint_y)
                )
                crossing_steps[
                    indices[crossed_on_terminal_step], waypoint_index
                ] = control_step
            terminal_max = info.get("episode_max_y") if isinstance(info, dict) else None
            if terminal_max is None:
                completed_max_y.extend(episode_max_y[indices].astype(float).tolist())
            else:
                completed_max_y.extend(_numpy(terminal_max)[indices].astype(float).tolist())
            episode_success = info.get("episode_success") if isinstance(info, dict) else None
            if episode_success is not None:
                successes += int(np.sum(_numpy(episode_success)[indices]))
            episode_platform = (
                info.get("episode_ever_on_platform") if isinstance(info, dict) else None
            )
            if episode_platform is not None:
                ever_on_platform += int(np.sum(_numpy(episode_platform)[indices]))
            episode_stable = (
                info.get("episode_stable_success") if isinstance(info, dict) else None
            )
            if episode_stable is not None:
                stable_successes += int(np.sum(_numpy(episode_stable)[indices]))
            episode_max_stable = (
                info.get("episode_max_stable_steps") if isinstance(info, dict) else None
            )
            if episode_max_stable is not None:
                completed_max_stable_steps.extend(
                    _numpy(episode_max_stable)[indices].astype(int).tolist()
                )
            term = _numpy(terminated).astype(bool)
            trunc = _numpy(truncated).astype(bool)
            falls += int(np.sum(term[indices])) - (
                int(np.sum(_numpy(episode_success)[indices]))
                if episode_success is not None
                else 0
            )
            timeouts += int(np.sum(trunc[indices]))
            completed += len(indices)
            completed_episodes_per_env[indices] += 1
            for local_index, env_index in enumerate(indices):
                episode_details.append(
                    {
                        "start_x": float(episode_start_x[env_index]),
                        "start_y": float(episode_start_y[env_index]),
                        "max_y": float(completed_max_y[-len(indices) + local_index]),
                        "max_abs_x": float(episode_max_abs_x[env_index]),
                        "final_x": float(final_x[local_index]),
                        "final_y": float(final_y[local_index]),
                        "final_heading": float(final_heading[local_index]),
                        "final_upright": float(final_upright[local_index]),
                        "termination_reason": str(termination_reason[local_index]),
                        "success": (
                            bool(_numpy(episode_success)[indices][local_index])
                            if episode_success is not None
                            else False
                        ),
                        "ever_on_platform": (
                            bool(_numpy(episode_platform)[indices][local_index])
                            if episode_platform is not None
                            else False
                        ),
                        "stable_success": (
                            bool(_numpy(episode_stable)[indices][local_index])
                            if episode_stable is not None
                            else False
                        ),
                        "max_stable_steps": (
                            int(_numpy(episode_max_stable)[indices][local_index])
                            if episode_max_stable is not None
                            else 0
                        ),
                        "crossing_steps": {
                            str(waypoint_y): int(crossing_steps[env_index, waypoint_index])
                            for waypoint_index, waypoint_y in enumerate(_DIAGNOSTIC_WAYPOINTS)
                        },
                    }
                )
            episode_max_y[done_indices] = -np.inf
            episode_max_abs_x[done_indices] = 0.0
            crossing_steps[done_indices] = -1
            reset_pose = raw_env._body.get_pose(raw_env.state.data)
            episode_start_x[done_indices] = reset_pose[done_indices, 0]
            episode_start_y[done_indices] = reset_pose[done_indices, 1]
            if completed >= _EPISODES.value:
                break

    denominator = max(completed, 1)
    waypoint_crossing_counts = {
        str(waypoint_y): sum(
            int(detail["crossing_steps"][str(waypoint_y)]) >= 0
            for detail in episode_details
        )
        for waypoint_y in _DIAGNOSTIC_WAYPOINTS
    }
    waypoint_count_per_episode = [
        sum(int(step) >= 0 for step in detail["crossing_steps"].values())
        for detail in episode_details
    ]
    waypoint_episode_histogram = np.bincount(
        waypoint_count_per_episode, minlength=len(_DIAGNOSTIC_WAYPOINTS) + 1
    ).tolist()
    print(
        json.dumps(
            {
                "environment": _ENV.value,
                "policy": _POLICY.value,
                "evaluation_start_x_range": (
                    [_EVAL_START_X_MIN.value, _EVAL_START_X_MAX.value]
                    if _EVAL_START_X_MIN.value is not None
                    else list(raw_env.cfg.start_x_range)
                ),
                "seed": _SEED.value,
                "completed_episodes": completed,
                "success_rate": successes / denominator,
                "ever_on_platform_rate": ever_on_platform / denominator,
                "stable_success_rate": stable_successes / denominator,
                "mean_episode_max_stable_steps": (
                    float(np.mean(completed_max_stable_steps))
                    if completed_max_stable_steps
                    else None
                ),
                "all_time_max_stable_steps": (
                    int(np.max(completed_max_stable_steps))
                    if completed_max_stable_steps
                    else None
                ),
                "fall_rate": falls / denominator,
                "timeout_rate": timeouts / denominator,
                "mean_episode_max_y": (
                    float(np.mean(completed_max_y)) if completed_max_y else None
                ),
                "all_time_max_y": (
                    float(np.max(completed_max_y)) if completed_max_y else None
                ),
                "waypoint_crossing_counts": waypoint_crossing_counts,
                "waypoint_episode_histogram": waypoint_episode_histogram,
                "control_steps": control_step,
                "episode_start_x": completed_start_x,
                "episode_start_y": completed_start_y,
                "episode_details": episode_details,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app.run(main)
