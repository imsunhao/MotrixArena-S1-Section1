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

import os
from pathlib import Path

import motrixsim as mtx
import numpy as np
from motrixsim.render import RenderApp, RenderSettings

from motrix_envs.np.env import NpEnv


class NpRenderer:
    """
    The renderer for Np sim environments.
    """

    _env: NpEnv

    def __init__(self, env: NpEnv):
        num_envs = env.num_envs
        num_envs = 1 if num_envs is None else num_envs
        render_env_index = int(os.environ.get("MOTRIX_RENDER_ENV_INDEX", "-1"))
        if 0 <= render_env_index < num_envs:
            self._render_env_index = render_env_index
            render_batch = 1
            self._selected_render_data = mtx.SceneData(env.model, batch=(1,))
        else:
            self._render_env_index = None
            self._selected_render_data = None
            render_batch = num_envs
        spacing = env.render_spacing
        cols = int(np.ceil(np.sqrt(render_batch)))
        offsets = []
        for i in range(render_batch):
            row = i // cols
            col = i % cols
            x = col * spacing
            y = row * spacing
            z = 0.0
            offsets.append([x, y, z])

        self._env = env
        self._render = RenderApp()
        settings = RenderSettings.performance()
        settings.enable_shadow = True  # disable shadow for better performance
        self._render.launch(
            env.model,
            batch=render_batch,
            render_offset=offsets,
            render_settings=settings,
        )
        self._sync_render_data = True
        self._render.system_camera.active = self._sync_render_data
        self._ready_file = os.environ.get("MOTRIX_RENDER_READY_FILE")
        self._ready_signaled = False
        if os.environ.get("MOTRIX_FOLLOW_CAMERA", "0") == "1" and len(env.model.cameras) > 0:
            self._render.set_main_camera(env.model.cameras[0])
            self._render.system_camera.active = False

    def render(self) -> None:
        """
        render the env
        """

        data = self._env.state.data if self._sync_render_data else None
        if data is not None and self._render_env_index is not None:
            selected = slice(
                self._render_env_index, self._render_env_index + 1
            )
            self._selected_render_data.set_dof_pos(
                data.dof_pos[selected], self._env.model
            )
            self._selected_render_data.set_dof_vel(
                data.dof_vel[selected]
            )
            self._selected_render_data.actuator_ctrls[:] = (
                data.actuator_ctrls[selected]
            )
            self._env.model.forward_kinematic(self._selected_render_data)
            data = self._selected_render_data
        self._render.sync(data=data)
        if self._ready_file and not self._ready_signaled:
            Path(self._ready_file).touch()
            self._ready_signaled = True
        if self._render.input.is_key_just_pressed("space"):
            self._sync_render_data = not self._sync_render_data
            self._render.system_camera.active = self._sync_render_data
