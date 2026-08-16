"""Render one reproducible Section01 Torch evaluation episode."""

from __future__ import annotations

import os
import time

from absl import app, flags
from skrl import config
from skrl.utils import set_seed
import torch

import motrix_envs.navigation.vbot  # noqa: F401 - register environments
from motrix_envs import registry as env_registry
from motrix_envs.np.renderer import NpRenderer
from motrix_rl.skrl.torch import wrap_env
from motrix_rl.skrl.torch.train import ppo

from section01_course_torch_config import COURSE_ENVS  # noqa: F401


_ENV = flags.DEFINE_enum("env", COURSE_ENVS[0], COURSE_ENVS, "Environment")
_POLICY = flags.DEFINE_string("policy", None, "Self-trained Torch checkpoint")
_SEED = flags.DEFINE_integer("seed", 2026, "Evaluation seed")
_SKIP_EPISODES = flags.DEFINE_integer(
    "skip-episodes", 0, "Completed episodes to fast-forward before rendering"
)
_FPS = flags.DEFINE_float("fps", 100.0, "Rendered control steps per wall-clock second")
_MAX_RENDER_STEPS = flags.DEFINE_integer(
    "max-render-steps", 12000, "Safety limit for the rendered episode"
)


def _done(value: torch.Tensor) -> bool:
    return bool(value.detach().cpu().numpy().reshape(-1)[0])


def main(argv):
    del argv
    if not _POLICY.value:
        raise app.UsageError("--policy is required")
    if _SKIP_EPISODES.value < 0:
        raise app.UsageError("--skip-episodes must be non-negative")
    if _FPS.value <= 0:
        raise app.UsageError("--fps must be positive")

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
    with torch.no_grad():
        while completed < _SKIP_EPISODES.value:
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)
            if _done(torch.logical_or(terminated, truncated)):
                completed += 1

        env._renderer = NpRenderer(raw_env)
        env.render()
        time.sleep(max(float(os.environ.get("MOTRIX_PLAY_START_DELAY_SECONDS", "0")), 0.0))

        for _ in range(_MAX_RENDER_STEPS.value):
            started = time.perf_counter()
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, info = env.step(actions)
            if _done(torch.logical_or(terminated, truncated)):
                print(
                    "rendered_episode_done",
                    {
                        "episode_success": bool(info["episode_success"][0]),
                        "ever_on_platform": bool(info["episode_ever_on_platform"][0]),
                        "stable_success": bool(info["episode_stable_success"][0]),
                        "max_stable_steps": int(info["episode_max_stable_steps"][0]),
                        "final_y": float(info["final_y"][0]),
                    },
                    flush=True,
                )
                break
            env.render()
            remaining = 1.0 / _FPS.value - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
        else:
            raise RuntimeError("rendered episode exceeded --max-render-steps")


if __name__ == "__main__":
    app.run(main)
