# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from envs.n_body_problem import Planets, Electrons, IdealGas


# env1 = Planets(num_bodies=10, num_dimensions=2, dt=0.01, contained_in_a_box=True)
# env1.animate()  # only animates if num_dimensions == 2
#
# env2 = Electrons(num_bodies=10, num_dimensions=2, dt=0.01, contained_in_a_box=True)
# env2.animate()  # only animates if num_dimensions == 2

for i in range(1):
    env3 = IdealGas(num_bodies=10, num_dimensions=2, dt=0.01, contained_in_a_box=True)
    file_name = 'idealgas{}.mp4'.format(i)
    env3.animate(file_name=file_name, pixel_length=64)
