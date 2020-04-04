import gym
import sys
import argparse
import numpy as np
import BaseEnvironment as gym_env

from discretisizeEnvironments import *

class QLearning:

    def __init__(self, env, function_params, env_params):
        self._alpha = function_params["alpha"]
        self._gamma = function_params["gamma"]
        self.initialize_environment_dependable_variables(env, function_params, env_params)

    def run_algorithm(self, reward, action, current_state, previous_state):
        self._visited_states[current_state] += 1

        learned_value = reward + self._gamma * np.max(self._Q[current_state, :])
        old_value = self._Q[previous_state, action]
        self._Q[previous_state, action] = (1 - self._alpha) * old_value + self._alpha * learned_value

    def initialize_environment_dependable_variables(self, env, function_params, env_params):
        if env.spec.id not in CONTINUOUS_ENVIRONMENTS_TO_DISCRETISIZE:
            if function_params["qInit"] is "stochastic":
                self._Q = np.random.uniform(0, 1, (env.observation_space.n, env.action_space.n))
            else:
                self._Q = np.zeros((env.observation_space.n, env.action_space.n))

            self._policy = np.zeros(env.observation_space.n)
            self._visited_states = np.zeros(env.observation_space.n)
        else:
            self._Q, self._visited_states = discretisize_environment(env, env_params, (function_params["qInit"] is "stochastic"))
            self._policy = self._visited_states

    def update_policy(self, env):
        for state in np.ndenumerate(self._policy):
            self._policy[state[0]] = np.argmax(self._Q[state[0], :])

    def get_action_from_policy(self, state):
        return self._policy[state]
    
    def get_data_to_save(self, env):
        return {"Q": self._Q, "visited_states": self._visited_states, "policy": self._policy}

    def set_variables_from_data(self, data):
        self._Q = data["Q"]
        self._policy = data["policy"]
        self._visited_states = data["visited_states"]
