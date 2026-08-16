"""Capture one deterministic Section01 Torch episode for renderer-only replay."""

from __future__ import annotations

import json
from pathlib import Path

from absl import app, flags
import numpy as np
from skrl import config
from skrl.utils import set_seed
import torch

import motrix_envs.navigation.vbot  # noqa: F401 - register environments
from motrix_envs import registry as env_registry
from motrix_rl.skrl.torch import wrap_env
from motrix_rl.skrl.torch.train import ppo

from section01_course_torch_config import COURSE_ENVS  # noqa: F401


_ENV = flags.DEFINE_enum("env", COURSE_ENVS[0], COURSE_ENVS, "Environment")
_POLICY = flags.DEFINE_string("policy", None, "Self-trained Torch checkpoint")
_OUTPUT = flags.DEFINE_string("output", None, "Output .npz trajectory")
_SEED = flags.DEFINE_integer("seed", 2026, "Evaluation seed")
_SKIP_EPISODES = flags.DEFINE_integer(
    "skip-episodes", 0, "Completed episodes before the captured episode"
)
_NUM_ENVS = flags.DEFINE_integer(
    "num-envs", 1, "Parallel environments used while searching for a trajectory"
)
_SEARCH_FIRST_SUCCESS = flags.DEFINE_bool(
    "search-first-success",
    False,
    "Keep scanning resets until an episode satisfies the recording gate",
)
_MAX_STEPS = flags.DEFINE_integer(
    "max-steps", 12000, "Safety limit for trajectory-search control steps"
)
_MIN_STABLE_STEPS = flags.DEFINE_integer(
    "min-stable-steps",
    100,
    "Minimum consecutive stable control steps required for recording",
)
_MIN_FINAL_Y = flags.DEFINE_float(
    "min-final-y",
    7.8,
    "Minimum terminal Y required to reach the Section1 finish marker",
)


def _done(value: torch.Tensor) -> bool:
    return bool(value.detach().cpu().numpy().reshape(-1)[0])


def _numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1)


def main(argv):
    del argv
    if not _POLICY.value or not _OUTPUT.value:
        raise app.UsageError("--policy and --output are required")
    if _SKIP_EPISODES.value < 0:
        raise app.UsageError("--skip-episodes must be non-negative")
    if _NUM_ENVS.value <= 0:
        raise app.UsageError("--num-envs must be positive")
    if _NUM_ENVS.value > 1 and _SKIP_EPISODES.value:
        raise app.UsageError("--skip-episodes is only supported with --num-envs=1")
    if _NUM_ENVS.value > 1 and not _SEARCH_FIRST_SUCCESS.value:
        raise app.UsageError("--num-envs>1 requires --search-first-success")
    if _MIN_STABLE_STEPS.value <= 0:
        raise app.UsageError("--min-stable-steps must be positive")

    config.torch.backend = "torch"
    set_seed(_SEED.value)
    trainer = ppo.Trainer(
        _ENV.value,
        "np",
        cfg_override={"play_num_envs": _NUM_ENVS.value, "seed": _SEED.value},
        enable_render=False,
    )
    raw_env = env_registry.make(
        _ENV.value, sim_backend="np", num_envs=_NUM_ENVS.value
    )
    if hasattr(raw_env.cfg, "use_full_course_local_starts"):
        raw_env.cfg.use_full_course_local_starts = False
    if hasattr(raw_env.cfg, "training_platform_start_fraction"):
        raw_env.cfg.training_platform_start_fraction = 0.0
    if hasattr(raw_env.cfg, "training_brake_start_fraction"):
        raw_env.cfg.training_brake_start_fraction = 0.0
    env = wrap_env(raw_env, enable_render=False)
    models = trainer._make_model(env, trainer._rlcfg)
    agent = trainer._make_agent(models, env, ppo._get_cfg(trainer._rlcfg, env))
    agent.load(_POLICY.value)
    agent.set_running_mode("eval")

    obs, _ = env.reset()
    completed = 0
    dof_pos_frames: list[np.ndarray] = []
    dof_vel_frames: list[np.ndarray] = []
    actuator_frames: list[np.ndarray] = []
    result: dict[str, object] | None = None

    with torch.no_grad():
        while completed < _SKIP_EPISODES.value:
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)
            if _done(torch.logical_or(terminated, truncated)):
                completed += 1

        start_pose = raw_env._body.get_pose(raw_env.state.data).copy()
        episode_start_frame = np.zeros(_NUM_ENVS.value, dtype=np.int32)
        selected_env = 0
        selected_start_frame = 0
        for _ in range(_MAX_STEPS.value):
            dof_pos_frames.append(np.asarray(raw_env.state.data.dof_pos).copy())
            dof_vel_frames.append(np.asarray(raw_env.state.data.dof_vel).copy())
            actuator_frames.append(
                np.asarray(raw_env.state.data.actuator_ctrls).copy()
            )
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, info = env.step(actions)
            done = _numpy(torch.logical_or(terminated, truncated)).astype(bool)
            if not np.any(done):
                continue

            episode_success = _numpy(info["episode_success"]).astype(bool)
            episode_platform = _numpy(info["episode_ever_on_platform"]).astype(bool)
            episode_stable = _numpy(info["episode_stable_success"]).astype(bool)
            episode_max_stable = _numpy(info["episode_max_stable_steps"]).astype(int)
            final_y = _numpy(info["final_y"])
            accepted = (
                done
                & episode_stable
                & (episode_max_stable >= _MIN_STABLE_STEPS.value)
                & (final_y >= _MIN_FINAL_Y.value)
            )
            selected = np.flatnonzero(accepted if _SEARCH_FIRST_SUCCESS.value else done)
            if len(selected):
                selected_env = int(selected[0])
                selected_start_frame = int(episode_start_frame[selected_env])
                result = {
                    "environment": _ENV.value,
                    "policy": _POLICY.value,
                    "seed": _SEED.value,
                    "skip_episodes": _SKIP_EPISODES.value,
                    "search_num_envs": _NUM_ENVS.value,
                    "selected_env_index": selected_env,
                    "start_x": float(start_pose[selected_env, 0]),
                    "start_y": float(start_pose[selected_env, 1]),
                    "episode_success": bool(episode_success[selected_env]),
                    "ever_on_platform": bool(episode_platform[selected_env]),
                    "stable_success": bool(episode_stable[selected_env]),
                    "max_stable_steps": int(episode_max_stable[selected_env]),
                    "final_y": float(final_y[selected_env]),
                    "control_steps": len(dof_pos_frames) - selected_start_frame,
                    "control_hz": 100,
                }
                break

            done_indices = np.flatnonzero(done)
            reset_pose = raw_env._body.get_pose(raw_env.state.data)
            start_pose[done_indices] = reset_pose[done_indices]
            episode_start_frame[done_indices] = len(dof_pos_frames)
        else:
            raise RuntimeError("trajectory search exceeded --max-steps")

    if result is None:
        raise RuntimeError("captured episode did not produce a terminal result")
    if (
        not result["stable_success"]
        or result["max_stable_steps"] < _MIN_STABLE_STEPS.value
        or result["final_y"] < _MIN_FINAL_Y.value
    ):
        raise RuntimeError(
            "requested episode does not satisfy the recording acceptance gate: "
            f"min_stable_steps={_MIN_STABLE_STEPS.value}, "
            f"min_final_y={_MIN_FINAL_Y.value}, result={result}"
        )
    result["required_stable_steps"] = _MIN_STABLE_STEPS.value
    result["required_final_y"] = _MIN_FINAL_Y.value

    output = Path(_OUTPUT.value)
    output.parent.mkdir(parents=True, exist_ok=True)
    dof_pos = np.stack(dof_pos_frames)[selected_start_frame:, selected_env]
    dof_vel = np.stack(dof_vel_frames)[selected_start_frame:, selected_env]
    actuator_ctrls = np.stack(actuator_frames)[selected_start_frame:, selected_env]
    np.savez_compressed(
        output,
        dof_pos=dof_pos.astype(np.float32),
        dof_vel=dof_vel.astype(np.float32),
        actuator_ctrls=actuator_ctrls.astype(np.float32),
        metadata=np.asarray(json.dumps(result, ensure_ascii=True)),
    )
    print("captured_episode", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    app.run(main)
