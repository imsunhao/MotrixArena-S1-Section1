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

"""Evaluate a Section 1 policy without rendering or an infinite play loop."""

import json

import numpy as np
from absl import app, flags
from skrl import config
from skrl.utils import set_seed

from motrix_envs import registry as env_registry
from motrix_rl.skrl.jax import wrap_env
from motrix_rl.skrl.jax.train import ppo


_ENV = flags.DEFINE_string(
    "env", "vbot_navigation_section011", "Environment to evaluate"
)
_POLICY = flags.DEFINE_string("policy", None, "JAX .pickle checkpoint to load")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 64, "Parallel evaluation environments")
_EPISODES = flags.DEFINE_integer(
    "episodes", 64, "Stop after at least this many completed episodes"
)
_MAX_CONTROL_STEPS = flags.DEFINE_integer(
    "max-control-steps", 5000, "Safety limit on vectorized control steps"
)
_SEED = flags.DEFINE_integer("seed", 2026, "Evaluation random seed")
_SIM_BACKEND = flags.DEFINE_string("sim-backend", None, "Simulation backend")
_ACTION_SCALE = flags.DEFINE_float(
    "action-scale", None, "Override the VBot joint-target action scale for evaluation"
)


def main(argv):
    del argv
    if not _POLICY.value:
        raise app.UsageError("--policy is required")
    if _NUM_ENVS.value <= 0 or _EPISODES.value <= 0:
        raise app.UsageError("--num-envs and --episodes must be positive")

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
    if _ACTION_SCALE.present:
        if _ACTION_SCALE.value <= 0:
            raise app.UsageError("--action-scale must be positive")
        raw_env._cfg.control_config.action_scale = _ACTION_SCALE.value

    set_seed(rlcfg.seed)
    env = wrap_env(raw_env, enable_render=False)
    models = trainer._make_model(env, rlcfg)
    agent_cfg = ppo._get_cfg(rlcfg, env)
    agent = trainer._make_agent(models, env, agent_cfg)
    agent.load(_POLICY.value)
    agent.set_running_mode("eval")

    obs, _ = env.reset()
    control_steps = 0
    reward_sum = 0.0
    transition_count = 0
    while control_steps < _MAX_CONTROL_STEPS.value:
        outputs = agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
        obs, rewards, _, _, _ = env.step(actions)
        reward_sum += float(np.sum(np.asarray(rewards)))
        transition_count += _NUM_ENVS.value
        control_steps += 1
        if raw_env.get_success_metrics()["completed_episodes"] >= _EPISODES.value:
            break

    metrics = raw_env.get_success_metrics()
    info = raw_env.state.info
    ongoing_max_y = np.asarray(info["episode_max_y"])
    ongoing_start_y = np.asarray(info["episode_start_y"])
    ongoing_waypoints = np.asarray(info["next_waypoint_idx"])
    metrics.update(
        {
            "policy": _POLICY.value,
            "seed": _SEED.value,
            "num_envs": _NUM_ENVS.value,
            "action_scale": raw_env._cfg.control_config.action_scale,
            "control_steps": control_steps,
            "environment_transitions": transition_count,
            "mean_step_reward": reward_sum / max(transition_count, 1),
            "requested_episodes": _EPISODES.value,
            "reached_episode_target": (
                metrics["completed_episodes"] >= _EPISODES.value
            ),
            "ongoing_mean_episode_max_y": float(np.mean(ongoing_max_y)),
            "ongoing_max_episode_y": float(np.max(ongoing_max_y)),
            "ongoing_mean_forward_progress": float(
                np.mean(ongoing_max_y - ongoing_start_y)
            ),
            "ongoing_max_waypoints": int(np.max(ongoing_waypoints)),
        }
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    app.run(main)
