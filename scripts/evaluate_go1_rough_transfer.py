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

"""Evaluate a checkpoint on the 48-observation GO1 rough transfer task."""

import json

from absl import app, flags
import numpy as np
from skrl import config
from skrl.utils import set_seed

from motrix_envs import registry as env_registry
from motrix_rl.skrl.jax import wrap_env
from motrix_rl.skrl.jax.train import ppo


_ENV = flags.DEFINE_string(
    "env", "go1-rough-terrain-walk-transfer", "GO1 rough evaluation environment"
)
_POLICY = flags.DEFINE_string("policy", None, "JAX PPO checkpoint")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 64, "Parallel environments")
_ROUGH_EPISODES = flags.DEFINE_integer(
    "rough-episodes", 64, "Stop after this many completed rough-terrain episodes"
)
_MAX_CONTROL_STEPS = flags.DEFINE_integer(
    "max-control-steps", 6000, "Vectorized control-step safety limit"
)
_SEED = flags.DEFINE_integer("seed", 2026, "Evaluation random seed")
_SIM_BACKEND = flags.DEFINE_string("sim-backend", None, "Simulation backend")


def main(argv):
    del argv
    if not _POLICY.value:
        raise app.UsageError("--policy is required")
    if _NUM_ENVS.value <= 0 or _ROUGH_EPISODES.value <= 0:
        raise app.UsageError("--num-envs and --rough-episodes must be positive")

    config.jax.backend = "jax"
    trainer = ppo.Trainer(
        _ENV.value,
        _SIM_BACKEND.value,
        cfg_override={"play_num_envs": _NUM_ENVS.value, "seed": _SEED.value},
        enable_render=False,
    )
    rlcfg = trainer._rlcfg
    raw_env = env_registry.make(
        _ENV.value, sim_backend=_SIM_BACKEND.value, num_envs=_NUM_ENVS.value
    )
    set_seed(rlcfg.seed)
    env = wrap_env(raw_env, enable_render=False)
    models = trainer._make_model(env, rlcfg)
    agent_cfg = ppo._get_cfg(rlcfg, env)
    agent = trainer._make_agent(models, env, agent_cfg)
    agent.load(_POLICY.value)
    agent.set_running_mode("eval")

    obs, _ = env.reset()
    episode_steps = np.zeros(_NUM_ENVS.value, dtype=np.int64)
    episode_is_rough = np.zeros(_NUM_ENVS.value, dtype=bool)
    total_completed = 0
    rough_completed = 0
    rough_falls = 0
    rough_timeouts = 0
    rough_episode_steps_sum = 0
    rough_transition_count = 0
    rough_command_error_sum = 0.0
    rough_tracking_reward_sum = 0.0
    reward_sum = 0.0
    transition_count = 0
    control_steps = 0

    while control_steps < _MAX_CONTROL_STEPS.value:
        root_pos = raw_env._body.get_position(raw_env.state.data)
        # The task's flat warm-up plane is at z=-2.5. Curriculum terrain
        # origins use z=0.5 or z=2.0, so this cleanly separates the phases.
        rough_mask = np.asarray(root_pos[:, 2] > 0.0)
        episode_is_rough |= rough_mask

        commands = np.asarray(raw_env.state.info["commands"])
        local_velocity = np.asarray(raw_env.get_local_linvel(raw_env.state.data))
        if np.any(rough_mask):
            velocity_error = np.sum(
                np.square(commands[rough_mask, :2] - local_velocity[rough_mask, :2]),
                axis=1,
            )
            rough_command_error_sum += float(np.sum(velocity_error))
            rough_tracking_reward_sum += float(
                np.sum(np.exp(-velocity_error / raw_env.cfg.reward_config.tracking_sigma))
            )
            rough_transition_count += int(np.sum(rough_mask))

        outputs = agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
        obs, rewards, terminated, truncated, _ = env.step(actions)
        terminated = np.asarray(terminated).reshape(-1).astype(bool)
        truncated = np.asarray(truncated).reshape(-1).astype(bool)
        done = np.logical_or(terminated, truncated)
        episode_steps += 1

        if np.any(done):
            total_completed += int(np.sum(done))
            rough_done = np.logical_and(done, episode_is_rough)
            rough_completed += int(np.sum(rough_done))
            rough_falls += int(np.sum(np.logical_and(terminated, rough_done)))
            rough_timeouts += int(np.sum(np.logical_and(truncated, rough_done)))
            rough_episode_steps_sum += int(np.sum(episode_steps[rough_done]))
            episode_steps[done] = 0
            episode_is_rough[done] = False

        reward_sum += float(np.sum(np.asarray(rewards)))
        transition_count += _NUM_ENVS.value
        control_steps += 1
        if rough_completed >= _ROUGH_EPISODES.value:
            break

    rough_denominator = max(rough_completed, 1)
    metrics = {
        "policy": _POLICY.value,
        "seed": _SEED.value,
        "num_envs": _NUM_ENVS.value,
        "control_steps": control_steps,
        "environment_transitions": transition_count,
        "mean_step_reward": reward_sum / max(transition_count, 1),
        "training_level": int(raw_env.training_level),
        "total_completed_episodes": total_completed,
        "rough_completed_episodes": rough_completed,
        "requested_rough_episodes": _ROUGH_EPISODES.value,
        "reached_rough_episode_target": rough_completed >= _ROUGH_EPISODES.value,
        "rough_fall_episodes": rough_falls,
        "rough_fall_rate": rough_falls / rough_denominator,
        "rough_timeout_episodes": rough_timeouts,
        "rough_timeout_rate": rough_timeouts / rough_denominator,
        "rough_mean_episode_seconds": (
            rough_episode_steps_sum * raw_env.cfg.ctrl_dt / rough_denominator
        ),
        "rough_transition_count": rough_transition_count,
        "rough_mean_velocity_squared_error": (
            rough_command_error_sum / max(rough_transition_count, 1)
        ),
        "rough_mean_tracking_reward": (
            rough_tracking_reward_sum / max(rough_transition_count, 1)
        ),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    app.run(main)
