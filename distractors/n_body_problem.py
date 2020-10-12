# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.integrate import odeint


class Planets(object):
    """
    Implements a 2D environments where there are N bodies (planets) that attract each other according to a 1/r law.

    We assume the mass of each body is 1.
    """

    # For each dimension of the hypercube
    MIN_POS = 0.  # if box exists
    MAX_POS = 1.  # if box exists
    INIT_MAX_VEL = 1.
    GRAVITATIONAL_CONSTANT = 1.

    def __init__(self, num_bodies, num_dimensions=2, dt=0.01, contained_in_a_box=True):
        self.num_bodies = num_bodies
        self.num_dimensions = num_dimensions
        self.dt = dt
        self.contained_in_a_box = contained_in_a_box

        # state variables
        self.body_positions = None
        self.body_velocities = None
        self.reset()

    def reset(self):
        self.body_positions = np.random.uniform(self.MIN_POS, self.MAX_POS, size=(self.num_bodies, self.num_dimensions))
        self.body_velocities = self.INIT_MAX_VEL * np.random.uniform(-1, 1, size=(self.num_bodies, self.num_dimensions))

    @property
    def state(self):
        return np.concatenate((self.body_positions, self.body_velocities), axis=1)  # (N, 2D)

    def step(self):

        # Helper functions since ode solver requires flattened inputs
        def flatten(positions, velocities):  # positions shape (N, D); velocities shape (N, D)
            system_state = np.concatenate((positions, velocities), axis=1)  # (N, 2D)
            system_state_flat = system_state.flatten()  # ode solver requires flat, (N*2D,)
            return system_state_flat

        def unflatten(system_state_flat):  # system_state_flat shape (N*2*D,)
            system_state = system_state_flat.reshape(self.num_bodies, 2 * self.num_dimensions)  # (N, 2*D)
            positions = system_state[:, :self.num_dimensions]  # (N, D)
            velocities = system_state[:, self.num_dimensions:]  # (N, D)
            return positions, velocities

        # ODE function
        def system_first_order_ode(system_state_flat, _):

            positions, velocities = unflatten(system_state_flat)
            accelerations = np.zeros_like(velocities)  # init (N, D)

            for i in range(self.num_bodies):
                relative_positions = positions - positions[i]  # (N, D)
                distances = np.linalg.norm(relative_positions, axis=1, keepdims=True)  # (N, 1)
                distances[i] = 1.  # bodies don't affect themselves, and we don't want to divide by zero next

                # forces (see https://en.wikipedia.org/wiki/Numerical_model_of_the_Solar_System)
                force_vectors = self.GRAVITATIONAL_CONSTANT * relative_positions / (distances**self.num_dimensions)  # (N,D)
                force_vector = np.sum(force_vectors, axis=0)  # (D,)
                accelerations[i] = force_vector  # assuming mass 1.

            d_system_state_flat = flatten(velocities, accelerations)
            return d_system_state_flat

        # integrate + update
        current_system_state_flat = flatten(self.body_positions, self.body_velocities)  # (N*2*D,)
        _, next_system_state_flat = odeint(system_first_order_ode, current_system_state_flat, [0., self.dt])  # (N*2*D,)
        self.body_positions, self.body_velocities = unflatten(next_system_state_flat)  # (N, D), (N, D)

        # bounce off boundaries of box
        if self.contained_in_a_box:
            ind_below_min = self.body_positions < self.MIN_POS
            ind_above_max = self.body_positions > self.MAX_POS
            self.body_positions[ind_below_min] += 2. * (self.MIN_POS - self.body_positions[ind_below_min])
            self.body_positions[ind_above_max] += 2. * (self.MAX_POS - self.body_positions[ind_above_max])
            self.body_velocities[ind_below_min] *= -1.
            self.body_velocities[ind_above_max] *= -1.
            self.assert_bodies_in_box()  # check for bugs

    def animate(self, file_name=None, frames=1000, pixel_length=None, tight_format=True):
        """
        Animation function for visual debugging.
        """
        if self.num_dimensions is not 2:
            raise NotImplementedError

        if pixel_length is None:
            fig = plt.figure()
        else:
            # matplotlib can't render if pixel_length is too small, so just run in the background id pixels specified
            import matplotlib
            matplotlib.use('Agg')
            my_dpi = 96  # find your screen's dpi here: https://www.infobyip.com/detectmonitordpi.php
            fig = plt.figure(facecolor='lightslategray', figsize=(pixel_length/my_dpi, pixel_length/my_dpi), dpi=my_dpi)

        ax = fig.add_subplot(1, 1, 1)
        plt.axis('off')
        if tight_format:
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=None, hspace=None)
        body_colors = np.random.uniform(size=self.num_bodies)

        def render(_):
            self.step()
            x = self.body_positions[:, 0]
            y = self.body_positions[:, 1]

            ax.clear()
            # if tight_format:
            #     plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)
            ax.scatter(x, y, marker='o', c=body_colors, cmap='viridis')
            # ax.set_title(self.__class__.__name__ + "\n(temperature inside box: {:.1f})".format(self.temperature))
            ax.set_xlim(self.MIN_POS, self.MAX_POS)
            ax.set_ylim(self.MIN_POS, self.MAX_POS)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            # ax.axis('off')
            ax.set_facecolor('black')
            if tight_format:
                ax.margins(x=0., y=0.)

        interval_milliseconds = 1000 * self.dt
        anim = animation.FuncAnimation(fig, render, frames=frames, interval=interval_milliseconds)

        plt.pause(1)
        if file_name is None:
            file_name = self.__class__.__name__.lower() + '.gif'
        file_name = 'images/' + file_name
        print('Saving file {} ...'.format(file_name))
        anim.save(file_name, writer='imagemagick')
        plt.close(fig)

    def assert_bodies_in_box(self):
        """
        if the sim goes really fast, they can bounce one-step out of box. Let's just check for this for now, fix later
        """
        assert np.all(self.body_positions >= self.MIN_POS) and np.all(self.body_positions <= self.MAX_POS)

    @property
    def temperature(self):
        """
        Temperature is the average kinetic energy of system
        :return: float
        """
        average_kinetic_energy = 0.5 * np.mean(np.linalg.norm(self.body_velocities, axis=1))  # (N, D) --> (1,)
        return average_kinetic_energy


class Electrons(Planets):
    """
    Implements a 2D environments where there are N bodies (electrons) that repel each other according to a 1/r law.
    """

    # override
    GRAVITATIONAL_CONSTANT = -1.  # negative means they repel


class IdealGas(Planets):
    """
    Implements a 2D environments where there are N bodies (gas molecules) that do not interact with each other.
    """

    # override
    GRAVITATIONAL_CONSTANT = 0.  # zero means they don't interact
