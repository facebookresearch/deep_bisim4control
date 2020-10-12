# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
import scipy.interpolate


def read_log_file(file_name, key_name, value_name, smooth=3):
    keys, values = [], []
    try:
        with open(file_name, 'r') as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    key, value = e[key_name], e[value_name]
                    keys.append(int(key))
                    values.append(float(value))
                except:
                    pass
    except:
        print('bad file: %s' % file_name)
        return None, None
    keys, values = np.array(keys), np.array(values)
    if smooth > 1 and values.shape[0] > 0:
        K = np.ones(smooth)
        ones = np.ones(values.shape[0])
        values = np.convolve(values, K, 'same') / np.convolve(ones, K, 'same')

    return keys, values


def parse_log_files(
    file_name_template,
    key_name,
    value_name,
    num_seeds,
    smooth,
    best_k=None,
    max_key=True
):
    all_values = []
    all_keys = []
    actual_keys = None
    for seed in range(1, num_seeds + 1):
        file_name = file_name_template % seed
        keys, values = read_log_file(file_name, key_name, value_name, smooth)
        if keys is None or keys.shape[0] == 0:
            continue
        all_keys.append(keys)
        all_values.append(values)

    if len(all_values) == 0:
        return None, None, None

    all_keys_tmp = sorted(all_keys, key=lambda x: x[-1])
    keys = all_keys_tmp[-1] if max_key else all_keys_tmp[0]
    threshold = keys.shape[0]

    # interpolate    
    for idx, (key, value) in enumerate(zip(all_keys, all_values)):
        f = scipy.interpolate.interp1d(key, value, fill_value='extrapolate')
        all_keys[idx] = keys
        all_values[idx] = f(keys)

    means, half_stds = [], []
    for i in range(threshold):
        vals = []

        for v in all_values:
            if i < v.shape[0]:
                vals.append(v[i])
        if best_k is not None:
            vals = sorted(vals)[-best_k:]
        means.append(np.mean(vals))
        half_stds.append(0.5 * np.std(vals))

    means = np.array(means)
    half_stds = np.array(half_stds)

    keys = all_keys[-1][:threshold]
    assert means.shape[0] == keys.shape[0]

    print(file_name_template, means[-1])
    return keys, means, half_stds
    # return all_keys, all_values


def print_result(
    root,
    title,
    label=None,
    num_seeds=1,
    smooth=3,
    train=False,
    key_name='step',
    value_name='episode_reward',
    max_time=None,
    best_k=None,
    timescale=1,
    max_key=False
):
    file_name = 'train.log' if train else 'eval.log'
    file_name_template = os.path.join(root, 'seed_%d', file_name)
    keys, means, half_stds = parse_log_files(
        file_name_template,
        key_name,
        value_name,
        num_seeds,
        smooth=smooth,
        best_k=best_k,
        max_key=max_key
    )
    label = label or root.split('/')[-1]
    if keys is None:
        return

    if max_time is not None:
        idxs = np.where(keys <= max_time)
        keys = keys[idxs]
        means = means[idxs]
        half_stds = half_stds[idxs]

    keys *= timescale

    plt.plot(keys, means, label=label)
    plt.locator_params(nbins=10, axis='x')
    plt.locator_params(nbins=10, axis='y')
    plt.rcParams['figure.figsize'] = (10, 7)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10
    plt.subplots_adjust(left=0.165, right=0.99, bottom=0.16, top=0.95)
    #plt.ylim(0, 1050)
    plt.tight_layout()

    plt.grid(alpha=0.8)
    plt.title(title)
    plt.fill_between(keys, means - half_stds, means + half_stds, alpha=0.2)
    plt.legend(loc='lower right', prop={
        'size': 6
    }).get_frame().set_edgecolor('0.1')
    plt.xlabel(key_name)
    plt.ylabel(value_name)


def plot_seeds(
    task,
    exp_query,
    root,
    train=True,
    smooth=3,
    key_name='step',
    value_name='episode_reward',
    num_seeds=10
):
    # root = os.path.join(root, task)
    experiment = None
    for exp in os.listdir(root):
        if re.match(exp_query, exp):
            experiment = os.path.join(root, exp)
            break
    if experiment is None:
        return
    file_name = 'train.log' if train else 'eval.log'
    file_name_template = os.path.join(experiment, 'seed_%d', file_name)

    plt.locator_params(nbins=10, axis='x')
    plt.locator_params(nbins=10, axis='y')
    plt.rcParams['figure.figsize'] = (10, 7)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10
    plt.subplots_adjust(left=0.165, right=0.99, bottom=0.16, top=0.95)
    plt.grid(alpha=0.8)
    plt.tight_layout()
    plt.title(task)

    plt.xlabel(key_name)
    plt.ylabel(value_name)

    for seed in range(1, num_seeds + 1):
        file_name = file_name_template % seed
        keys, values = read_log_file(file_name, key_name, value_name, smooth=smooth)
        if keys is None or keys.shape[0] == 0:
            continue

        plt.plot(keys, values, label='seed_%d' % seed, linewidth=0.5)

    plt.legend(loc='lower right', prop={
        'size': 6
    }).get_frame().set_edgecolor('0.1')


def print_baseline(task, baseline, data, color):
    try:
        value = data[task][baseline]
    except:
        return

    plt.axhline(y=value, label=baseline, linestyle='--', color=color)
    plt.legend(loc='lower right', prop={
        'size': 6
    }).get_frame().set_edgecolor('0.1')


def print_planet_baseline(
    task, data, max_time=None, label='planet', color='black', offset=0
):
    try:
        keys, means, half_stds = data[task]
    except:
        return

    if max_time is not None:
        idx = np.searchsorted(keys, max_time)
        keys = keys[:idx]
        means = means[:idx]
        half_stds = half_stds[:idx]

    plt.plot(keys + offset, means, label=label, color=color)
    plt.fill_between(
        keys + offset,
        means - half_stds,
        means + half_stds,
        alpha=0.2,
        color=color
    )
    plt.legend(loc='lower right', prop={
        'size': 6
    }).get_frame().set_edgecolor('0.1')


def plot_experiment(
    task,
    exp_query,
    neg_exp_query=None,
    root='runs',
    exp_ids=None,
    smooth=3,
    train=False,
    key_name='step',
    value_name='eval_episode_reward',
    baselines_data=None,
    num_seeds=10,
    planet_data=None,
    slac_data=None,
    max_time=None,
    best_k=None,
    timescale=1,
    max_key=False
):
    root = os.path.join(root, task)

    experiments = set()
    for exp in os.listdir(root):
        if re.match(exp_query, exp) and (neg_exp_query is None or re.match(neg_exp_query, exp) is None):
            exp = os.path.join(root, exp)
            experiments.add(exp)

    exp_ids = list(range(len(experiments))) if exp_ids is None else exp_ids
    for exp_id, exp in enumerate(sorted(experiments)):
        if exp_id in exp_ids:
            print_result(
                exp,
                task,
                smooth=smooth,
                num_seeds=num_seeds,
                train=train,
                key_name=key_name,
                value_name=value_name,
                max_time=max_time,
                best_k=best_k,
                timescale=timescale,
                max_key=max_key
            )

    if baselines_data is not None:
        print_baseline(task, 'd4pg_pixels', baselines_data, color='gray')
        print_baseline(task, 'd4pg', baselines_data, color='black')

    if planet_data is not None:
        print_planet_baseline(
            task,
            planet_data,
            max_time=max_time,
            label='planet',
            color='peru',
            offset=5
        )

    if slac_data is not None:
        action_repeat = {
            'ball_in_cup_catch': 4,
            'cartpole_swingup': 8,
            'cheetah_run': 4,
            'finger_spin': 2,
            'walker_walk': 2,
            'reacher_easy': 4
        }
        offset = 10 * action_repeat[task]
        print_planet_baseline(
            task,
            slac_data,
            max_time=max_time,
            label='slac',
            color='black',
            offset=offset
        )
