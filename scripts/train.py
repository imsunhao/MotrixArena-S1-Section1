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


import logging

from absl import app, flags
from skrl import config

from motrix_rl import utils

logger = logging.getLogger(__name__)

_ENV = flags.DEFINE_string("env", "cartpole", "The env to train")
_SIM_BACKEND = flags.DEFINE_string(
    "sim-backend",
    None,
    "The simulation backend to use.(If not specified, it will be choosen automatically)",
)
_NUM_ENVS = flags.DEFINE_integer("num-envs", 2048, "Number of envs to train")
_MAX_ENV_STEPS = flags.DEFINE_integer(
    "max-env-steps", None, "Override maximum environment transitions for short runs"
)
_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    "checkpoint-interval", None, "Override checkpoint interval in batched steps"
)
_INITIAL_POLICY = flags.DEFINE_string(
    "initial-policy", None, "Optional checkpoint used to warm-start training"
)
_LEARNING_RATE = flags.DEFINE_float(
    "learning-rate", None, "Override the PPO learning rate"
)
_LEARNING_EPOCHS = flags.DEFINE_integer(
    "learning-epochs", None, "Override PPO epochs per rollout"
)
_RATIO_CLIP = flags.DEFINE_float(
    "ratio-clip", None, "Override the PPO policy ratio clipping range"
)
_INITIAL_LOG_STD = flags.DEFINE_float(
    "initial-log-std",
    None,
    "Override the initial Gaussian policy log standard deviation",
)
_RESUME_LOG_STD = flags.DEFINE_float(
    "resume-log-std",
    None,
    "JAX only: replace policy log standard deviation after loading a checkpoint",
)
_FREEZE_STATE_PREPROCESSOR = flags.DEFINE_bool(
    "freeze-state-preprocessor",
    False,
    "JAX only: keep the loaded observation-normalization statistics fixed",
)
_RESET_OPTIMIZERS = flags.DEFINE_bool(
    "reset-optimizers",
    False,
    "JAX only: discard optimizer state loaded from the initial policy",
)
_RENDER = flags.DEFINE_bool("render", False, "Render the env")
_TRAIN_BACKEND = flags.DEFINE_string("train-backend", "jax", "The learning backend. (jax/torch)")
_SEED = flags.DEFINE_integer("seed", None, "Random seed for reproducibility")
_RAND_SEED = flags.DEFINE_bool("rand-seed", False, "Generate random seed")


def get_train_backend(supports: utils.DeviceSupports):
    if supports.jax and supports.jax_gpu:
        return "jax"
    elif supports.torch and supports.torch_gpu:
        return "torch"
    elif supports.jax:
        return "jax"
    elif supports.torch:
        return "torch"
    else:
        raise Exception("neither jax nor torch not avaliable on the device.")


def main(argv):
    device_supports = utils.get_device_supports()
    logger.info(device_supports)
    env_name = _ENV.value
    enable_render = _RENDER.value

    rl_override = {}

    if _NUM_ENVS.present:
        rl_override["num_envs"] = _NUM_ENVS.value

    if _MAX_ENV_STEPS.present:
        rl_override["max_env_steps"] = _MAX_ENV_STEPS.value

    if _CHECKPOINT_INTERVAL.present:
        if _CHECKPOINT_INTERVAL.value <= 0:
            raise app.UsageError("--checkpoint-interval must be positive")
        rl_override["check_point_interval"] = _CHECKPOINT_INTERVAL.value

    if _LEARNING_RATE.present:
        if _LEARNING_RATE.value <= 0:
            raise app.UsageError("--learning-rate must be positive")
        rl_override["learning_rate"] = _LEARNING_RATE.value

    if _LEARNING_EPOCHS.present:
        if _LEARNING_EPOCHS.value <= 0:
            raise app.UsageError("--learning-epochs must be positive")
        rl_override["learning_epochs"] = _LEARNING_EPOCHS.value

    if _RATIO_CLIP.present:
        if not 0 < _RATIO_CLIP.value <= 1:
            raise app.UsageError("--ratio-clip must be in (0, 1]")
        rl_override["ratio_clip"] = _RATIO_CLIP.value

    if _INITIAL_LOG_STD.present:
        rl_override["initial_log_std"] = _INITIAL_LOG_STD.value

    if _RESUME_LOG_STD.present and not _INITIAL_POLICY.value:
        raise app.UsageError("--resume-log-std requires --initial-policy")

    if _RAND_SEED.value:
        rl_override["seed"] = None
    elif _SEED.present:
        rl_override["seed"] = _SEED.value

    sim_backend = _SIM_BACKEND.value
    train_backend = "jax"
    if not _TRAIN_BACKEND.present:
        train_backend = get_train_backend(device_supports)
    else:
        train_backend = _TRAIN_BACKEND.value

    trainer = None
    if train_backend == "jax":
        from motrix_rl.skrl.jax.train import ppo

        config.jax.backend = "jax"  # or "numpy"
        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)

    elif train_backend == "torch":
        from motrix_rl.skrl.torch.train import ppo

        config.torch.backend = "torch"
        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)
    else:
        raise Exception(f"Unknown train backend: {train_backend}")

    if train_backend == "jax":
        trainer.train(
            initial_policy=_INITIAL_POLICY.value,
            policy_log_std_override=_RESUME_LOG_STD.value,
            freeze_state_preprocessor=_FREEZE_STATE_PREPROCESSOR.value,
            reset_optimizers=_RESET_OPTIMIZERS.value,
        )
    else:
        if (
            _RESUME_LOG_STD.present
            or _FREEZE_STATE_PREPROCESSOR.value
            or _RESET_OPTIMIZERS.value
        ):
            raise app.UsageError(
                "resume/freeze/reset warm-start options currently require "
                "--train-backend=jax"
            )
        trainer.train(initial_policy=_INITIAL_POLICY.value)


if __name__ == "__main__":
    app.run(main)
