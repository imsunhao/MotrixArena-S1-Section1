#!/usr/bin/env python3
"""Evaluate Section 1 with an arbitrary monotonic chain of policies.

Pass ``--policy`` once per stage and ``--switch-y`` once per boundary. Stage
state only moves forward within an episode, preventing policy chatter when a
robot slips backwards near a terrain boundary.
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
    "env", "vbot_locomotion_section011_full_route_angular_safe_forward06", "Evaluation environment"
)
_POLICIES = flags.DEFINE_multi_string(
    "policy", [], "Checkpoint for one stage; repeat in traversal order"
)
_SWITCH_Y = flags.DEFINE_multi_float(
    "switch-y", [], "Y boundary between adjacent policies; repeat in order"
)
_ACTION_SCALES = flags.DEFINE_multi_float(
    "action-scale",
    [],
    "Physical action scale per policy; omit to use the environment scale",
)
_BLEND_STEPS = flags.DEFINE_integer(
    "blend-steps", 0, "Blend the previous and new policy for this many steps"
)
_NUM_ENVS = flags.DEFINE_integer("num-envs", 64, "Parallel environments")
_EPISODES = flags.DEFINE_integer("episodes", 64, "Completed episode target")
_MAX_CONTROL_STEPS = flags.DEFINE_integer(
    "max-control-steps", 5000, "Vectorized control-step safety limit"
)
_SEED = flags.DEFINE_integer("seed", 2026, "Evaluation random seed")
_SIM_BACKEND = flags.DEFINE_string("sim-backend", None, "Simulation backend")
_BODY_FORWARD_SPEED = flags.DEFINE_float(
    "body-forward-speed", None, "Override commanded body-forward speed"
)


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
    policy_paths = list(_POLICIES.value)
    switch_y = np.asarray(_SWITCH_Y.value, dtype=np.float32)
    if not policy_paths:
        raise app.UsageError("repeat --policy for every policy-chain stage")
    if len(switch_y) != len(policy_paths) - 1:
        raise app.UsageError("the chain requires exactly len(policy)-1 --switch-y values")
    if np.any(np.diff(switch_y) <= 0):
        raise app.UsageError("--switch-y values must be strictly increasing")
    if _NUM_ENVS.value <= 0 or _EPISODES.value <= 0:
        raise app.UsageError("--num-envs and --episodes must be positive")
    if _BLEND_STEPS.value < 0:
        raise app.UsageError("--blend-steps must be non-negative")

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
    if _BODY_FORWARD_SPEED.present:
        if _BODY_FORWARD_SPEED.value <= 0:
            raise app.UsageError("--body-forward-speed must be positive")
        raw_env._cfg.navigation_body_forward_speed = _BODY_FORWARD_SPEED.value

    env_action_scale = float(raw_env._cfg.control_config.action_scale)
    if _ACTION_SCALES.value:
        action_scales = np.asarray(_ACTION_SCALES.value, dtype=np.float32)
        if len(action_scales) != len(policy_paths):
            raise app.UsageError("repeat --action-scale once per policy")
        if np.any(action_scales <= 0):
            raise app.UsageError("--action-scale values must be positive")
    else:
        action_scales = np.full(len(policy_paths), env_action_scale, np.float32)

    set_seed(rlcfg.seed)
    env = wrap_env(raw_env, enable_render=False)
    agents = [
        _load_agent(trainer, env, rlcfg, policy_path)
        for policy_path in policy_paths
    ]
    # Agent construction can touch process-global JAX model state. Match the
    # established hierarchical evaluator by reloading the base policy last.
    agents[0].load(policy_paths[0])
    agents[0].set_running_mode("eval")

    obs, _ = env.reset()
    stages = np.zeros(_NUM_ENVS.value, dtype=np.int16)
    stage_ages = np.zeros(_NUM_ENVS.value, dtype=np.int32)
    stage_entry_counts = np.zeros(len(policy_paths), dtype=np.int64)
    stage_entry_counts[0] = _NUM_ENVS.value
    control_steps = 0
    reward_sum = 0.0
    transition_count = 0

    while control_steps < _MAX_CONTROL_STEPS.value:
        root_pos, _, _ = raw_env._extract_root_state(raw_env.state.data)
        root_y = root_pos[:, 1]
        for next_stage, boundary_y in enumerate(switch_y, start=1):
            enter = np.logical_and(
                stages == next_stage - 1, root_y >= boundary_y
            )
            stages[enter] = next_stage
            stage_ages[enter] = 0
            stage_entry_counts[next_stage] += int(np.sum(enter))

        action_cache = {}

        def stage_actions(stage):
            if stage not in action_cache:
                action_cache[stage] = _mean_actions(agents[stage], obs) * (
                    float(action_scales[stage]) / env_action_scale
                )
            return action_cache[stage]

        actions = stage_actions(0)
        for stage in range(1, len(policy_paths)):
            mask = stages == stage
            if not np.any(mask):
                continue
            selected_actions = stage_actions(stage)
            if _BLEND_STEPS.value:
                alpha = np.clip(
                    (stage_ages + 1) / _BLEND_STEPS.value, 0.0, 1.0
                )
                previous_actions = stage_actions(stage - 1)
                selected_actions = previous_actions + jnp.asarray(alpha)[:, None] * (
                    selected_actions - previous_actions
                )
            actions = jnp.where(
                jnp.asarray(mask)[:, None], selected_actions, actions
            )

        obs, rewards, _, _, _ = env.step(actions)
        reward_sum += float(np.sum(np.asarray(rewards)))
        transition_count += _NUM_ENVS.value
        control_steps += 1
        stage_ages += 1

        reset_envs = np.asarray(raw_env.state.info["steps"]) == 0
        stages[reset_envs] = 0
        stage_ages[reset_envs] = 0
        stage_entry_counts[0] += int(np.sum(reset_envs))
        if raw_env.get_success_metrics()["completed_episodes"] >= _EPISODES.value:
            break

    metrics = raw_env.get_success_metrics()
    info = raw_env.state.info
    ongoing_max_y = np.asarray(info["episode_max_y"])
    ongoing_start_y = np.asarray(info["episode_start_y"])
    ongoing_waypoints = np.asarray(info["next_waypoint_idx"])
    metrics.update(
        {
            "policies": policy_paths,
            "switch_y": switch_y.tolist(),
            "action_scales": action_scales.tolist(),
            "environment_action_scale": env_action_scale,
            "blend_steps": _BLEND_STEPS.value,
            "stage_entry_counts": stage_entry_counts.tolist(),
            "seed": _SEED.value,
            "num_envs": _NUM_ENVS.value,
            "control_steps": control_steps,
            "environment_transitions": transition_count,
            "mean_step_reward": reward_sum / max(transition_count, 1),
            "requested_episodes": _EPISODES.value,
            "reached_episode_target": metrics["completed_episodes"]
            >= _EPISODES.value,
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
