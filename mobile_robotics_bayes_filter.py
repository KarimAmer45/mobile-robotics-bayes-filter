#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_belief(belief):
    
    plt.figure()
    
    ax = plt.subplot(2,1,1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0],1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")
    
    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")


def motion_model(action, belief):
    """Apply the 1D motion model for forward/backward commands."""
    action = str(action).strip().upper()
    if action not in {"F", "B"}:
        raise ValueError(f"Unknown action {action!r}; expected 'F' or 'B'.")

    direction = 1 if action == "F" else -1
    model = ((direction, 0.75), (0, 0.15), (-direction, 0.10))
    prediction = np.zeros_like(belief, dtype=float)

    for cell, probability in enumerate(belief):
        if probability == 0:
            continue
        for step, step_probability in model:
            next_cell = cell + step
            if 0 <= next_cell < belief.shape[0]:
                prediction[next_cell] += probability * step_probability
            else:
                prediction[cell] += probability * step_probability

    return prediction


def sensor_model(observation, belief, world):
    """Correct the belief using the binary floor-color observation."""
    observation = int(observation)
    world = np.asarray(world, dtype=int)
    likelihood = np.where(world == observation, 0.75, 0.25)
    correction = belief * likelihood
    total = correction.sum()
    if total == 0:
        raise ValueError("Sensor correction collapsed to zero probability.")
    return correction / total


def recursive_bayes_filter(actions, observations, belief, world):
    """Run recursive Bayes filtering over a command/observation sequence."""
    actions = list(actions)
    observations = list(observations)
    belief = np.asarray(belief, dtype=float)
    world = np.asarray(world, dtype=int)

    if len(observations) == len(actions) + 1:
        belief = sensor_model(observations[0], belief, world)
        observations = observations[1:]
    elif len(observations) != len(actions):
        raise ValueError("Expected one observation per action, optionally plus one initial observation.")

    for action, observation in zip(actions, observations):
        belief = motion_model(action, belief)
        belief = sensor_model(observation, belief, world)

    return belief
