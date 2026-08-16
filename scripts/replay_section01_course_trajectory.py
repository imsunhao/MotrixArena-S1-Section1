"""Replay a captured Section01 trajectory using the configured renderer."""

from __future__ import annotations

import json
import os
import time

from absl import app, flags
import numpy as np

import motrix_envs.navigation.vbot  # noqa: F401 - register environments
from motrix_envs import registry as env_registry
from motrix_envs.np.renderer import NpRenderer

from section01_course_torch_config import COURSE_ENVS  # noqa: F401


_ENV = flags.DEFINE_enum("env", COURSE_ENVS[0], COURSE_ENVS, "Environment")
_TRAJECTORY = flags.DEFINE_string("trajectory", None, "Captured .npz trajectory")
_FPS = flags.DEFINE_float("fps", 100.0, "Trajectory states per wall-clock second")
_HOLD_SECONDS = flags.DEFINE_float(
    "hold-seconds", 2.0, "Seconds to hold the final stable pose"
)


def main(argv):
    del argv
    if not _TRAJECTORY.value:
        raise app.UsageError("--trajectory is required")
    if _FPS.value <= 0 or _HOLD_SECONDS.value < 0:
        raise app.UsageError("--fps must be positive and --hold-seconds non-negative")

    trajectory = np.load(_TRAJECTORY.value, allow_pickle=False)
    dof_pos = trajectory["dof_pos"]
    dof_vel = trajectory["dof_vel"]
    actuator_ctrls = trajectory["actuator_ctrls"]
    metadata = json.loads(str(trajectory["metadata"]))

    raw_env = env_registry.make(_ENV.value, sim_backend="np", num_envs=1)
    raw_env.init_state()
    raw_env.state.data.set_dof_pos(dof_pos[0:1], raw_env.model)
    raw_env.state.data.set_dof_vel(dof_vel[0:1])
    raw_env.state.data.actuator_ctrls[:] = actuator_ctrls[0:1]
    raw_env.model.forward_kinematic(raw_env.state.data)
    renderer = NpRenderer(raw_env)
    renderer.render()
    time.sleep(max(float(os.environ.get("MOTRIX_PLAY_START_DELAY_SECONDS", "0")), 0.0))

    frame_period = 1.0 / _FPS.value
    for index in range(len(dof_pos)):
        started = time.perf_counter()
        raw_env.state.data.set_dof_pos(dof_pos[index : index + 1], raw_env.model)
        raw_env.state.data.set_dof_vel(dof_vel[index : index + 1])
        raw_env.state.data.actuator_ctrls[:] = actuator_ctrls[index : index + 1]
        raw_env.model.forward_kinematic(raw_env.state.data)
        renderer.render()
        remaining = frame_period - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)

    hold_frames = int(round(_HOLD_SECONDS.value * _FPS.value))
    for _ in range(hold_frames):
        started = time.perf_counter()
        renderer.render()
        remaining = frame_period - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)
    print("replayed_episode", json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    app.run(main)
