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

"""Evaluate Section 1 with terrain-aware policy switching.

The stage state is monotonic within an episode. This adds hysteresis around
terrain boundaries: a robot that briefly slips backwards does not oscillate
between two policies on consecutive control steps.
"""

import json

from absl import app, flags
import jax.numpy as jnp
import numpy as np
from skrl import config
from skrl.utils import set_seed

from motrix_envs import registry as env_registry
from motrix_rl.skrl.jax import wrap_env
from motrix_rl.skrl.jax.train import ppo


_ENV = flags.DEFINE_string(
    "env", "vbot_navigation_section011_go1_transfer", "Evaluation environment"
)
_BASE_POLICY = flags.DEFINE_string(
    "base-policy", None, "Policy used before and after specialized terrain"
)
_ROUGH_POLICY = flags.DEFINE_string(
    "rough-policy", None, "Policy used while traversing the heightfield"
)
_SLOPE_POLICY = flags.DEFINE_string(
    "slope-policy", None, "Optional policy used from the slope entrance onward"
)
_ROUGH_START_Y = flags.DEFINE_float(
    "rough-start-y", -1.55, "Switch from base policy to rough policy"
)
_ROUGH_END_Y = flags.DEFINE_float(
    "rough-end-y", 1.35, "Switch back after the heightfield exit"
)
_SLOPE_START_Y = flags.DEFINE_float(
    "slope-start-y", 2.0, "Switch to the optional slope policy"
)
_BASE_ACTION_SCALE = flags.DEFINE_float(
    "base-action-scale", None, "Physical action scale expected by the base policy"
)
_ROUGH_ACTION_SCALE = flags.DEFINE_float(
    "rough-action-scale", None, "Physical action scale expected by the rough policy"
)
_SLOPE_ACTION_SCALE = flags.DEFINE_float(
    "slope-action-scale", None, "Physical action scale expected by the slope policy"
)
_ROUGH_BLEND_STEPS = flags.DEFINE_integer(
    "rough-blend-steps", 0, "Linearly blend base into rough actions over N steps"
)
_SLOPE_BLEND_STEPS = flags.DEFINE_integer(
    "slope-blend-steps", 0, "Linearly blend rough into slope actions over N steps"
)
_NUM_ENVS = flags.DEFINE_integer("num-envs", 64, "Parallel environments")
_EPISODES = flags.DEFINE_integer("episodes", 64, "Completed episode target")
_MAX_CONTROL_STEPS = flags.DEFINE_integer(
    "max-control-steps", 5000, "Vectorized control-step safety limit"
)
_SEED = flags.DEFINE_integer("seed", 2026, "Evaluation random seed")
_SIM_BACKEND = flags.DEFINE_string("sim-backend", None, "Simulation backend")


def _load_agent(trainer, env, rlcfg, policy_path):
    models = trainer._make_model(env, rlcfg)
    agent_cfg = ppo._get_cfg(rlcfg, env)
    agent = trainer._make_agent(models, env, agent_cfg)
    agent.load(policy_path)
    agent.set_running_mode("eval")
    return agent


def _mean_actions(agent, obs):
    outputs = agent.act(obs, timestep=0, timesteps=0)
    return outputs[-1].get("mean_actions", outputs[0])


def main(argv):
    del argv
    if not _BASE_POLICY.value or not _ROUGH_POLICY.value:
        raise app.UsageError("--base-policy and --rough-policy are required")
    if not _ROUGH_START_Y.value < _ROUGH_END_Y.value:
        raise app.UsageError("--rough-start-y must be less than --rough-end-y")
    if _SLOPE_POLICY.value and _SLOPE_START_Y.value <= _ROUGH_END_Y.value:
        raise app.UsageError("--slope-start-y must be after --rough-end-y")
    if _NUM_ENVS.value <= 0 or _EPISODES.value <= 0:
        raise app.UsageError("--num-envs and --episodes must be positive")
    if _ROUGH_BLEND_STEPS.value < 0 or _SLOPE_BLEND_STEPS.value < 0:
        raise app.UsageError("policy blend steps must be non-negative")

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
    env_action_scale = float(raw_env._cfg.control_config.action_scale)
    base_action_scale = (
        _BASE_ACTION_SCALE.value
        if _BASE_ACTION_SCALE.value is not None
        else env_action_scale
    )
    rough_action_scale = (
        _ROUGH_ACTION_SCALE.value
        if _ROUGH_ACTION_SCALE.value is not None
        else env_action_scale
    )
    slope_action_scale = (
        _SLOPE_ACTION_SCALE.value
        if _SLOPE_ACTION_SCALE.value is not None
        else env_action_scale
    )
    for name, value in (
        ("base-action-scale", base_action_scale),
        ("rough-action-scale", rough_action_scale),
        ("slope-action-scale", slope_action_scale),
    ):
        if value <= 0:
            raise app.UsageError(f"--{name} must be positive")
    set_seed(rlcfg.seed)
    env = wrap_env(raw_env, enable_render=False)

    base_agent = _load_agent(trainer, env, rlcfg, _BASE_POLICY.value)
    rough_agent = _load_agent(trainer, env, rlcfg, _ROUGH_POLICY.value)
    slope_agent = (
        _load_agent(trainer, env, rlcfg, _SLOPE_POLICY.value)
        if _SLOPE_POLICY.value
        else None
    )

    obs, _ = env.reset()
    stages = np.zeros(_NUM_ENVS.value, dtype=np.int8)
    stage_ages = np.zeros(_NUM_ENVS.value, dtype=np.int32)
    stage_entry_counts = np.zeros(4, dtype=np.int64)
    control_steps = 0
    reward_sum = 0.0
    transition_count = 0

    while control_steps < _MAX_CONTROL_STEPS.value:
        root_pos, _, _ = raw_env._extract_root_state(raw_env.state.data)
        root_y = root_pos[:, 1]

        enter_rough = np.logical_and(stages == 0, root_y >= _ROUGH_START_Y.value)
        stages[enter_rough] = 1
        stage_ages[enter_rough] = 0
        stage_entry_counts[1] += int(np.sum(enter_rough))

        leave_rough = np.logical_and(stages == 1, root_y >= _ROUGH_END_Y.value)
        stages[leave_rough] = 2
        stage_ages[leave_rough] = 0
        stage_entry_counts[2] += int(np.sum(leave_rough))

        if slope_agent is not None:
            enter_slope = np.logical_and.reduce(
                (stages == 2, root_y >= _SLOPE_START_Y.value)
            )
            stages[enter_slope] = 3
            stage_ages[enter_slope] = 0
            stage_entry_counts[3] += int(np.sum(enter_slope))

        base_actions = _mean_actions(base_agent, obs) * (
            base_action_scale / env_action_scale
        )
        rough_actions = _mean_actions(rough_agent, obs) * (
            rough_action_scale / env_action_scale
        )
        if _ROUGH_BLEND_STEPS.value:
            rough_alpha = np.clip(
                (stage_ages + 1) / _ROUGH_BLEND_STEPS.value, 0.0, 1.0
            )
            blended_rough_actions = base_actions + jnp.asarray(rough_alpha)[:, None] * (
                rough_actions - base_actions
            )
        else:
            blended_rough_actions = rough_actions
        actions = jnp.where(
            jnp.asarray(stages == 1)[:, None], blended_rough_actions, base_actions
        )
        if slope_agent is not None:
            slope_actions = _mean_actions(slope_agent, obs) * (
                slope_action_scale / env_action_scale
            )
            if _SLOPE_BLEND_STEPS.value:
                slope_alpha = np.clip(
                    (stage_ages + 1) / _SLOPE_BLEND_STEPS.value, 0.0, 1.0
                )
                slope_actions = rough_actions + jnp.asarray(slope_alpha)[:, None] * (
                    slope_actions - rough_actions
                )
            actions = jnp.where(
                jnp.asarray(stages == 3)[:, None], slope_actions, actions
            )

        obs, rewards, _, _, _ = env.step(actions)
        reward_sum += float(np.sum(np.asarray(rewards)))
        transition_count += _NUM_ENVS.value
        control_steps += 1
        stage_ages += 1

        reset_envs = np.asarray(raw_env.state.info["steps"]) == 0
        stages[reset_envs] = 0
        stage_ages[reset_envs] = 0
        if raw_env.get_success_metrics()["completed_episodes"] >= _EPISODES.value:
            break

    metrics = raw_env.get_success_metrics()
    info = raw_env.state.info
    ongoing_max_y = np.asarray(info["episode_max_y"])
    ongoing_start_y = np.asarray(info["episode_start_y"])
    ongoing_waypoints = np.asarray(info["next_waypoint_idx"])
    metrics.update(
        {
            "base_policy": _BASE_POLICY.value,
            "rough_policy": _ROUGH_POLICY.value,
            "slope_policy": _SLOPE_POLICY.value,
            "environment_action_scale": env_action_scale,
            "base_action_scale": base_action_scale,
            "rough_action_scale": rough_action_scale,
            "slope_action_scale": slope_action_scale,
            "rough_blend_steps": _ROUGH_BLEND_STEPS.value,
            "slope_blend_steps": _SLOPE_BLEND_STEPS.value,
            "seed": _SEED.value,
            "num_envs": _NUM_ENVS.value,
            "control_steps": control_steps,
            "environment_transitions": transition_count,
            "mean_step_reward": reward_sum / max(transition_count, 1),
            "requested_episodes": _EPISODES.value,
            "reached_episode_target": (
                metrics["completed_episodes"] >= _EPISODES.value
            ),
            "rough_start_y": _ROUGH_START_Y.value,
            "rough_end_y": _ROUGH_END_Y.value,
            "slope_start_y": _SLOPE_START_Y.value,
            "stage_entry_counts": stage_entry_counts.tolist(),
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
