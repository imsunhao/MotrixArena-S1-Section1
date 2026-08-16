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
_MAX_STEPS = flags.DEFINE_integer(
    "max-steps", 12000, "Safety limit for the captured episode"
)
_MIN_STABLE_STEPS = flags.DEFINE_integer(
    "min-stable-steps",
    100,
    "Minimum consecutive stable control steps required for recording",
)
_MIN_FINAL_Y = flags.DEFINE_float(
    "min-final-y",
    7.3,
    "Minimum terminal Y required to show the robot clearly inside the platform",
)


def _done(value: torch.Tensor) -> bool:
    return bool(value.detach().cpu().numpy().reshape(-1)[0])


def main(argv):
    del argv
    if not _POLICY.value or not _OUTPUT.value:
        raise app.UsageError("--policy and --output are required")
    if _SKIP_EPISODES.value < 0:
        raise app.UsageError("--skip-episodes must be non-negative")
    if _MIN_STABLE_STEPS.value <= 0:
        raise app.UsageError("--min-stable-steps must be positive")

    config.torch.backend = "torch"
    set_seed(_SEED.value)
    trainer = ppo.Trainer(
        _ENV.value,
        "np",
        cfg_override={"play_num_envs": 1, "seed": _SEED.value},
        enable_render=False,
    )
    raw_env = env_registry.make(_ENV.value, sim_backend="np", num_envs=1)
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

        start_pose = raw_env._body.get_pose(raw_env.state.data)[0].copy()
        for _ in range(_MAX_STEPS.value):
            dof_pos_frames.append(np.asarray(raw_env.state.data.dof_pos)[0].copy())
            dof_vel_frames.append(np.asarray(raw_env.state.data.dof_vel)[0].copy())
            actuator_frames.append(
                np.asarray(raw_env.state.data.actuator_ctrls)[0].copy()
            )
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, info = env.step(actions)
            if _done(torch.logical_or(terminated, truncated)):
                result = {
                    "environment": _ENV.value,
                    "policy": _POLICY.value,
                    "seed": _SEED.value,
                    "skip_episodes": _SKIP_EPISODES.value,
                    "start_x": float(start_pose[0]),
                    "start_y": float(start_pose[1]),
                    "episode_success": bool(info["episode_success"][0]),
                    "ever_on_platform": bool(info["episode_ever_on_platform"][0]),
                    "stable_success": bool(info["episode_stable_success"][0]),
                    "max_stable_steps": int(info["episode_max_stable_steps"][0]),
                    "final_y": float(info["final_y"][0]),
                    "control_steps": len(dof_pos_frames),
                    "control_hz": 100,
                }
                break
        else:
            raise RuntimeError("captured episode exceeded --max-steps")

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
    np.savez_compressed(
        output,
        dof_pos=np.stack(dof_pos_frames).astype(np.float32),
        dof_vel=np.stack(dof_vel_frames).astype(np.float32),
        actuator_ctrls=np.stack(actuator_frames).astype(np.float32),
        metadata=np.asarray(json.dumps(result, ensure_ascii=True)),
    )
    print("captured_episode", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    app.run(main)
