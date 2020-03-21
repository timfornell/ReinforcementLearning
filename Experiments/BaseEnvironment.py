"""

This file is intended to be used as a base environment for different reinforcemeant learning algorithms. It is meant to
be resposible for performing the training, evaluating and plotting the results.

"""

import sys
import gym
import time
import numpy as np
import matplotlib.pyplot as plt

import MovingAverage as moving_average

AVAILABLE_ENVIRONMENT_PARAMS = ["episodes", "max_steps", "stochastic", "probabilities"]
AVAILABLE_PARAMS = ["env", "env_params", "RL_function", "RL_params", "debug", "params", "action_policy",
                    "action_policy_params", "latest_reward", "Q", "current_state", "previous_state",
                    "reward_per_episode", "visited_states", "latest_action", "policy"]
EVALUATE = "eval"


class GymEnvironment:

    def __init__(self, env, env_params: dict, RL_function, RL_params: dict, specific_params: dict, action_policy: str,
                 action_policy_params: dict, debug: bool):
        for param in env_params:
            if param not in AVAILABLE_ENVIRONMENT_PARAMS:
                sys.exit("Cannot find parameter: %s, exiting..." % param)

        for param in RL_params:
            if param not in AVAILABLE_PARAMS:
                sys.exit("Parameter %s is not an available parameter to read, exiting..." % param)

        self._env = env
        self._env_params = env_params
        self._RL_function = RL_function
        self._RL_params = {**RL_params, **specific_params}
        self._debug = debug
        self._action_policy = action_policy
        self._action_policy_params = action_policy_params
        self._Q = np.zeros((env.observation_space.n, env.action_space.n))
        self._reward_per_episode = np.zeros((env_params["episodes"], 1))
        self._visited_states = np.zeros((env.observation_space.n, 1))
        self._policy = np.zeros((env.observation_space.n, 1))
        self._previous_state = 0
        self._current_state = 0
        self._latest_reward = 0
        self._latest_action = 0

    def train(self):
        self.setup_environment()

        self._Q = np.random.uniform(0, 1, (self._env.observation_space.n, self._env.action_space.n))

        for ep in range(self._env_params["episodes"]):
            print("Training episode %d" % ep)
            self._current_state = self._env.reset()
            self._visited_states[self._current_state] += 1

            for steps in range(self._env_params["max_steps"]):
                # Decide which action to take
                self._latest_action = self.get_action(episode=ep)

                # Perform action and update state and reward variables
                done = self.perform_action(self._latest_action, episode=ep)

                # Run Reinforcement Learning (RL) Algorithm to update Q matrix
                self.run_RL_function(self._latest_action, self._latest_reward)

                self.update_policy()

                if done:
                    break

    def get_action(self, episode):
        action = self._env.action_space.sample()

        if self._action_policy == "epsilon_greedy":
            if np.random.uniform(low=0, high=1) < 1 - self._action_policy_params["epsilon"]:
                action = np.argmax(self._Q[self._current_state, :])
            else:
                action = self._env.action_space.sample()
        elif self._action_policy == "epsilon_greedy_update":
            if np.random.uniform(low=0, high=1) < 1 - self._action_policy_params["epsilon"]:
                action = np.argmax(self._Q[self._current_state, :])
            else:
                action = self._env.action_space.sample()

            self._action_policy_params["epsilon"] = self._action_policy_params["epsilon"] / (episode + 1)
        else:
            sys.exit("Could not find action policy: %s, exiting..." % self._action_policy)

        return action

    def perform_action(self, action, episode):
        new_state, reward, done, info = self._env.step(action)

        self._previous_state = self._current_state
        self._current_state = new_state
        if not done:
            self._reward_per_episode[episode] += reward
        self._visited_states[new_state] += 1
        self._latest_reward = reward

        return done

    def update_policy(self):
        for state in range(self._env.observation_space.n):
            self._policy[state] = np.argmax(self._Q[state, :])

    def run_RL_function(self, action, reward):
        self.update_RL_params()

        update_Q = self._RL_function(self._RL_params)
        self._Q = update_Q

    # This should be implemented better, perhaps all variables should be accessible through a struct like self.
    # available_parameters?
    def update_RL_params(self):
        for key, value in self._RL_params.items():
            if key == "env":
                self._RL_params[key] = self._env
            elif key == "env_params":
                self._RL_params[key] = self._env_params
            elif key == "RL_function":
                self._RL_params[key] = self._RL_function
            elif key == "RL_params":
                self._RL_params[key] = self._RL_params
            elif key == "debug":
                self._RL_params[key] = self._debug
            elif key == "action_policy":
                self._RL_params[key] = self._action_policy
            elif key == "action_policy_params":
                self._RL_params[key] = self._action_policy_params
            elif key == "latest_reward":
                self._RL_params[key] = self._latest_reward
            elif key == "Q":
                self._RL_params[key] = self._Q
            elif key == "current_state":
                self._RL_params[key] = self._current_state
            elif key == "previous_state":
                self._RL_params[key] = self._previous_state
            elif key == "reward_per_episode":
                self._RL_params[key] = self._reward_per_episode
            elif key == "visited_states":
                self._RL_params[key] = self._visited_states
            elif key == "latest_action":
                self._RL_params[key] = self._latest_action
            elif key == "policy":
                self.RL_params[key] = self._policy

    def setup_environment(self):
        for key, value in self._env_params.items():
            if key == "stochastic" and value is True:
                self.set_transition_probabilites(self._env_params["probabilities"])

    def plot_results(self, environment):
        if environment == "CliffWalking-v0":
            Q_max = np.zeros((self._env.observation_space.n, 1))
            Q_plot = np.zeros((self._env.observation_space.n, 1))

            for state in range(48):
                Q_plot[state] = np.max(self._Q[state, :])
                Q_max[state] = np.argmax(self._Q[state, :])

            fig1 = plt.figure(figsize=(6, 1))

            fig1.add_subplot(3, 1, 1)
            plt.imshow(Q_plot.reshape(4, 12))
            plt.title("State Action function")
            plt.colorbar()

            fig1.add_subplot(3, 1, 2)
            plt.imshow(Q_max.reshape(4, 12))
            plt.title("Policy")
            plt.colorbar()

            fig1.add_subplot(3, 1, 3)
            plt.imshow(self._visited_states.reshape(4, 12))
            plt.title("number of visits per state")
            plt.colorbar()

        plt.figure(2)
        plt.title("total expected reward")
        plt.clf()
        reward_per_episode_aver = moving_average.MovingAverage(self._reward_per_episode, 10)
        plt.plot(self._reward_per_episode, alpha=0.5)
        plt.plot(reward_per_episode_aver, alpha=0.5)
        plt.legend(["Reward", "Smoothed reward"])
        plt.xlabel("Iterations")
        plt.ylabel("Reward")
        plt.show()

    def set_transition_probabilites(self, probabilities):
        if len(probabilities) < self._env.action_space.n:
            sys.exit("Number of actions is not the same as number of probabilities specified.")
        elif float(sum(probabilities)) != 1.0:
            sys.exit("Total sum of probabilities is not equal to one.")

        for state in range(self._env.observation_space.n):
            for action in range(self._env.action_space.n):
                items = list(self._env.P[state][action][0])
                items[0] = probabilities[action]
                self._env.P[state][action][0] = tuple(items)

    def evaluate(self):
        step = 1
        self.update_policy()
        state = self._env.reset()

        done = False
        previous_state = None
        while not done:
            print("=========\n Step %d\n=========" % step)
            self._env.render()

            action = int(self._policy[state][0])
            state, reward, done, info = self._env.step(action)

            if previous_state == state:
                print("Something went wrong, state did not change")
                break
            else:
                previous_state = state

            step += 1
            time.sleep(0.1)
