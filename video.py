# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import imageio
import os
import numpy as np
import glob

from dmc2gym.natural_imgsource import RandomVideoSource


class VideoRecorder(object):
    def __init__(self, dir_name, resource_files=None, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []
        if resource_files:
            files = glob.glob(os.path.expanduser(resource_files))
            self._bg_source = RandomVideoSource((height, width), files, grayscale=False, total_frames=1000)
        else:
            self._bg_source = None

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            if self._bg_source:
                mask = np.logical_and((frame[:, :, 2] > frame[:, :, 1]), (frame[:, :, 2] > frame[:, :, 0]))  # hardcoded for dmc
                bg = self._bg_source.get_image()
                frame[mask] = bg[mask]
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
