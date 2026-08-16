"""Train the fixed-condition Section01 curriculum with the Torch PPO backend.

The environment names are intentionally isolated from the earlier Section011
experiments.  ``stage-steps`` counts vectorized control steps, so a 1024-env S1
run with 9600 stage steps executes 1024 * 9600 environment transitions.
"""

from __future__ import annotations

import inspect
import os
import re
import tempfile

from absl import app, flags
from skrl import config
import torch

import motrix_envs.navigation.vbot  # noqa: F401 - register course environments
import motrix_rl.skrl as skrl_root
from motrix_rl.skrl.torch.train import ppo

from section01_course_torch_config import COURSE_ENVS, PEER_CURRICULUM_STAGE_STEPS

_ENV = flags.DEFINE_enum("env", COURSE_ENVS[0], COURSE_ENVS, "Curriculum stage")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 1024, "Parallel environments")
_STAGE_STEPS = flags.DEFINE_integer(
    "stage-steps",
    None,
    "Vectorized control steps; peer curriculum environments use their verified default",
)
_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    "checkpoint-interval", 400, "Checkpoint interval in vectorized control steps"
)
_SEED = flags.DEFINE_integer("seed", 42, "Training seed")
_POLICY = flags.DEFINE_string("policy", None, "Warm-start Torch checkpoint")
_RUN_TAG = flags.DEFINE_string("run-tag", None, "Run directory label")
_OUTPUT_ROOT = flags.DEFINE_string("output-root", "runs", "Training output root")
_LEARNING_RATE = flags.DEFINE_float(
    "learning-rate", None, "Optional PPO learning-rate override"
)
_RESET_OPTIMIZER = flags.DEFINE_bool(
    "reset-optimizer", False, "Warm start weights and normalizers without optimizer state"
)
_INITIAL_LOG_STD = flags.DEFINE_float(
    "initial-log-std", None, "Optional initial policy log standard deviation"
)


def _checkpoint_without_optimizer(policy: str) -> str:
    checkpoint = torch.load(policy, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("--reset-optimizer requires a dictionary Torch checkpoint")
    checkpoint = {key: value for key, value in checkpoint.items() if key != "optimizer"}
    fd, path = tempfile.mkstemp(prefix="section01_warm_start_", suffix=".pt")
    os.close(fd)
    torch.save(checkpoint, path)
    return path

def main(argv):
    del argv
    stage_steps = _STAGE_STEPS.value
    if stage_steps is None:
        stage_steps = PEER_CURRICULUM_STAGE_STEPS.get(_ENV.value)
    if stage_steps is None or stage_steps <= 0:
        raise app.UsageError(
            "--stage-steps must be positive for environments without a curriculum default"
        )
    if _NUM_ENVS.value <= 0:
        raise app.UsageError("--num-envs must be positive")
    if _CHECKPOINT_INTERVAL.value <= 0:
        raise app.UsageError("--checkpoint-interval must be positive")
    run_tag = _RUN_TAG.value or f"fixed_torch_seed{_SEED.value}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_tag):
        raise app.UsageError("--run-tag contains unsupported characters")

    config.torch.backend = "torch"
    overrides = {
        "seed": _SEED.value,
        "num_envs": _NUM_ENVS.value,
        "max_env_steps": _NUM_ENVS.value * stage_steps,
        "check_point_interval": _CHECKPOINT_INTERVAL.value,
    }
    if _LEARNING_RATE.value is not None:
        if _LEARNING_RATE.value <= 0:
            raise app.UsageError("--learning-rate must be positive")
        overrides["learning_rate"] = _LEARNING_RATE.value
    if _INITIAL_LOG_STD.value is not None:
        overrides["initial_log_std"] = _INITIAL_LOG_STD.value
    output_prefix = f"{_OUTPUT_ROOT.value.rstrip('/')}/{run_tag}"
    trainer_kwargs = {
        "cfg_override": overrides,
        "enable_render": False,
    }
    if "log_dir" in inspect.signature(ppo.Trainer).parameters:
        trainer_kwargs["log_dir"] = f"{output_prefix}/{_ENV.value}"
    else:
        # The official MotrixArena-S1 Torch trainer uses this module-level
        # prefix instead of accepting a per-run log directory.
        skrl_root.LOG_DIR_PREFIX = output_prefix
    trainer = ppo.Trainer(_ENV.value, "np", **trainer_kwargs)
    warm_start = _POLICY.value
    temporary_checkpoint = None
    if warm_start and _RESET_OPTIMIZER.value:
        temporary_checkpoint = _checkpoint_without_optimizer(warm_start)
        warm_start = temporary_checkpoint
    try:
        trainer.train(initial_policy=warm_start)
    finally:
        if temporary_checkpoint:
            os.unlink(temporary_checkpoint)


if __name__ == "__main__":
    app.run(main)
