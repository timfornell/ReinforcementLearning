import gym
import sys
import argparse
import numpy as np

import BaseEnvironment as gym_env
from createParameterDict import createParameterDict

class ExpectedSARSA:

    def __init__(self, env, variables):
        self._alpha = variables["alpha"]
        self._gamma = variables["gamma"]
        self.initialize_environment_dependable_variables(env, variables)

    def run_algorithm(self, reward, action, current_state, previous_state):
        self._visited_states[current_state] += 1

        new_action = np.argmax(Q_matrix[current_state, :])
        learned_value = reward + gamma * Q_matrix[current_state, new_action] - Q_matrix[previous_state, action]
        Q_matrix[previous_state, action] = Q_matrix[previous_state, action] + alpha * learned_value

    def initialize_environment_dependable_variables(self, env, variables):
        if variables["qInit"] is "stochastic":
            self._Q = np.random.uniform(0, 1, (env.observation_space.n, env.action_space.n))
        else:
            self._Q = np.zeros((env.observation_space.n, env.action_space.n))

        self._policy = np.zeros(env.observation_space.n)
        self._visited_states = np.zeros(env.observation_space.n)

    def update_policy(self, env):
        for state in range(env.observation_space.n):
            self._policy[state] = np.argmax(self._Q[state, :])

    def get_action_from_policy(self, state):
        return self._policy[state]
    
    def get_data_to_save(self, env):
        return {"Q": self._Q, "visited_states": self._visited_states, "policy": self._policy}

    def set_variables_from_data(self, data):
        self._Q = data["Q"]
        self._policy = data["policy"]
        self._visited_states = data["visited_states"]
