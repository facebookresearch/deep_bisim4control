# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.init_height = 4
        self.init_width = 25
        num_out_channels = 3  # rgb
        kernel = 3

        self.fc = nn.Linear(
            feature_dim, num_filters * self.init_height * self.init_width
        )

        self.deconvs = nn.ModuleList()

        pads = [0, 1, 0]
        for i in range(self.num_layers - 1):
            output_padding = pads[i]
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, kernel, stride=2, output_padding=output_padding)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(
                num_filters, num_out_channels, kernel, stride=2, output_padding=1
            )
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        deconv = h.view(-1, self.num_filters, self.init_height, self.init_width)
        self.outputs['deconv1'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs['obs'] = obs

        return obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param(
                'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
            )
        L.log_param('train_decoder/fc', self.fc, step)


_AVAILABLE_DECODERS = {'pixel': PixelDecoder}


def make_decoder(
    decoder_type, obs_shape, feature_dim, num_layers, num_filters
):
    assert decoder_type in _AVAILABLE_DECODERS
    return _AVAILABLE_DECODERS[decoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )
