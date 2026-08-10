"""Adapt a 48-observation GO1 PPO checkpoint to the 62-observation VBot task."""

import os
import pickle

import numpy as np
from absl import app, flags
from flax import serialization
from skrl import config

from motrix_envs import registry as env_registry
from motrix_rl.skrl.jax import wrap_env
from motrix_rl.skrl.jax.train import ppo


_SOURCE_POLICY = flags.DEFINE_string(
    "source-policy", None, "GO1 JAX PPO checkpoint with 48 observations"
)
_OUTPUT_POLICY = flags.DEFINE_string(
    "output-policy", None, "Path for the adapted VBot checkpoint"
)
_ENV = flags.DEFINE_string(
    "env",
    "vbot_navigation_section011_go1_transfer",
    "Target VBot environment",
)
_SOURCE_OBSERVATIONS = flags.DEFINE_integer(
    "source-observations", 48, "Number of source observation features to copy"
)
_SEED = flags.DEFINE_integer("seed", 2026, "Target checkpoint initialization seed")


def _adapt_model(source_bytes: bytes, target_bytes: bytes, role: str) -> bytes:
    source = serialization.msgpack_restore(source_bytes)
    target = serialization.msgpack_restore(target_bytes)
    source_params = source["params"]
    target_params = target["params"]

    if set(source_params) != set(target_params):
        raise ValueError(f"{role} parameter names do not match")

    for layer_name, source_layer in source_params.items():
        target_layer = target_params[layer_name]
        if layer_name == "Dense_0":
            source_kernel = np.asarray(source_layer["kernel"])
            target_kernel = np.asarray(target_layer["kernel"]).copy()
            if source_kernel.shape[0] != _SOURCE_OBSERVATIONS.value:
                raise ValueError(
                    f"{role} source input has {source_kernel.shape[0]} features, "
                    f"expected {_SOURCE_OBSERVATIONS.value}"
                )
            if source_kernel.shape[1] != target_kernel.shape[1]:
                raise ValueError(f"{role} first hidden-layer width does not match")
            if target_kernel.shape[0] < source_kernel.shape[0]:
                raise ValueError(f"{role} target input is smaller than source input")
            target_kernel.fill(0.0)
            target_kernel[: source_kernel.shape[0]] = source_kernel
            target_layer["kernel"] = target_kernel
            target_layer["bias"] = np.asarray(source_layer["bias"])
            continue

        if isinstance(source_layer, dict):
            for parameter_name, source_value in source_layer.items():
                target_value = np.asarray(target_layer[parameter_name])
                source_value = np.asarray(source_value)
                if source_value.shape != target_value.shape:
                    raise ValueError(
                        f"{role} {layer_name}/{parameter_name} shape mismatch: "
                        f"{source_value.shape} != {target_value.shape}"
                    )
                target_layer[parameter_name] = source_value
        else:
            source_value = np.asarray(source_layer)
            target_value = np.asarray(target_layer)
            if source_value.shape != target_value.shape:
                raise ValueError(
                    f"{role} {layer_name} shape mismatch: "
                    f"{source_value.shape} != {target_value.shape}"
                )
            target_params[layer_name] = source_value

    return serialization.to_bytes(target)


def _adapt_state_preprocessor(source_bytes: bytes, target_bytes: bytes) -> bytes:
    source = serialization.msgpack_restore(source_bytes)
    target = serialization.msgpack_restore(target_bytes)
    source_size = np.asarray(source["running_mean"]).shape[0]
    target_size = np.asarray(target["running_mean"]).shape[0]
    if source_size != _SOURCE_OBSERVATIONS.value or target_size < source_size:
        raise ValueError(
            f"state preprocessor size mismatch: source={source_size}, target={target_size}"
        )

    running_mean = np.zeros(target_size, dtype=np.float32)
    running_variance = np.ones(target_size, dtype=np.float32)
    running_mean[:source_size] = np.asarray(source["running_mean"])
    running_variance[:source_size] = np.asarray(source["running_variance"])
    target["running_mean"] = running_mean
    target["running_variance"] = running_variance
    target["current_count"] = np.asarray(source["current_count"])
    return serialization.to_bytes(target)


def main(argv):
    del argv
    if not _SOURCE_POLICY.value or not _OUTPUT_POLICY.value:
        raise app.UsageError("--source-policy and --output-policy are required")

    config.jax.backend = "jax"
    trainer = ppo.Trainer(
        _ENV.value,
        cfg_override={"play_num_envs": 1, "seed": _SEED.value},
        enable_render=False,
    )
    raw_env = env_registry.make(_ENV.value, num_envs=1)
    env = wrap_env(raw_env, enable_render=False)
    models = trainer._make_model(env, trainer._rlcfg)
    agent_cfg = ppo._get_cfg(trainer._rlcfg, env)
    agent = trainer._make_agent(models, env, agent_cfg)

    output_directory = os.path.dirname(os.path.abspath(_OUTPUT_POLICY.value))
    os.makedirs(output_directory, exist_ok=True)
    agent.save(_OUTPUT_POLICY.value)

    with open(_SOURCE_POLICY.value, "rb") as source_file:
        source_modules = pickle.load(source_file)
    with open(_OUTPUT_POLICY.value, "rb") as target_file:
        target_modules = pickle.load(target_file)

    for role in ("policy", "value"):
        target_modules[role] = _adapt_model(
            source_modules[role], target_modules[role], role
        )
    target_modules["state_preprocessor"] = _adapt_state_preprocessor(
        source_modules["state_preprocessor"], target_modules["state_preprocessor"]
    )
    target_modules["value_preprocessor"] = source_modules["value_preprocessor"]

    with open(_OUTPUT_POLICY.value, "wb") as output_file:
        pickle.dump(target_modules, output_file, protocol=4)
    print(_OUTPUT_POLICY.value)


if __name__ == "__main__":
    app.run(main)
